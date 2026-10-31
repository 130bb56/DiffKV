# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DiffKV-v2 attention with chunked prefill and the native GQA decode kernels."""

from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from diffkv_v2.cache import DiffKVCacheController
from diffkv_v2.ops import PARTITION_SIZE, QUANT_ARGS, get_extension
from vllm.config import get_current_vllm_config
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)

SCORE_WINDOW = 64
KV_BUFFER_SIZE = 64
SHORT_PROMPT_LIMIT = 4000


@dataclass
class _BatchState:
    request_ids: list[str]
    computed: dict[str, int]
    scheduled: dict[str, int]
    dummy: bool
    decode_bounds: list[int] = field(default_factory=list)
    compressed_layers: dict[str, set[int]] = field(default_factory=dict)
    executed_layers: set[int] = field(default_factory=set)


_controller: DiffKVCacheController | None = None
_layer_indices: dict[str, int] = {}
_batch_state: _BatchState | None = None


def set_controller(
    controller: DiffKVCacheController | None,
    layer_names: list[str] | None = None,
) -> None:
    global _controller, _layer_indices, _batch_state
    if controller is not None:
        if layer_names is None or len(layer_names) != controller.num_layers:
            raise ValueError("DiffKV requires one ordered name per attention layer")
        if len(set(layer_names)) != len(layer_names):
            raise ValueError("Duplicate DiffKV attention layer names")
    _controller = controller
    _layer_indices = {name: i for i, name in enumerate(layer_names or [])}
    _batch_state = None


def get_controller() -> DiffKVCacheController | None:
    return _controller


def set_batch_state(
    req_ids: list[str],
    computed: dict[str, int],
    scheduled: dict[str, int],
    dummy: bool = False,
) -> None:
    """Prepare one runner batch in its actual query order, before metadata build."""
    global _batch_state
    if len(req_ids) != len(set(req_ids)):
        raise ValueError("Duplicate requests in DiffKV batch")
    state = _BatchState(list(req_ids), dict(computed), dict(scheduled), dummy)
    if dummy:
        _batch_state = state
        return
    controller = _controller
    if controller is None:
        raise RuntimeError("DiffKV worker cache was not initialized")
    decode_ids = []
    for req_id in req_ids:
        request = controller.requests[req_id]
        start, count = computed[req_id], scheduled[req_id]
        if start != request.num_computed_tokens or count <= 0:
            raise ValueError(
                "DiffKV scheduling must advance the original token sequence"
            )
        if start + count > controller.max_model_len:
            raise ValueError("DiffKV scheduled tokens exceed the context limit")
        if request.phase == "prefill":
            if start + count > request.prompt_len:
                raise ValueError(
                    "A DiffKV prefill chunk cannot cross the prompt boundary"
                )
        else:
            if count != 1:
                raise ValueError(
                    "DiffKV-v2 currently supports one decode token per step"
                )
            decode_ids.append(req_id)
    if decode_ids:
        controller.append_decode(decode_ids)
        slots = [controller.requests[req_id].slot_id for req_id in decode_ids]
        # One transfer serves all layers; retained lengths never replace positions.
        state.decode_bounds = (
            controller.kv_len_tables[slots].sum(dim=-1).amax(dim=(0, 2)).cpu().tolist()
        )
    _batch_state = state


def finish_batch() -> dict[str, Any] | None:
    """Commit a completed forward, reclaim prompt storage, and report usage."""
    global _batch_state
    state = _batch_state
    if state is None or state.dummy:
        _batch_state = None
        return None
    controller = _controller
    if controller is None:
        raise RuntimeError("DiffKV worker cache disappeared during a forward")
    expected_layers = set(range(controller.num_layers))
    if state.executed_layers != expected_layers:
        raise RuntimeError("DiffKV forward did not execute every registered layer")
    completed = []
    for req_id in state.request_ids:
        request = controller.requests[req_id]
        end = state.computed[req_id] + state.scheduled[req_id]
        if request.phase == "prefill" and end == request.prompt_len:
            if state.compressed_layers.get(req_id) != expected_layers:
                raise RuntimeError(
                    "DiffKV prompt compression missed an attention layer"
                )
            completed.append(req_id)
    controller.finalize_prompt(completed)
    for req_id in state.request_ids:
        controller.set_num_computed_tokens(
            req_id, state.computed[req_id] + state.scheduled[req_id]
        )
    _batch_state = None
    return controller.report()


@dataclass
class DiffKVV2Metadata(AttentionMetadata):
    num_actual_tokens: int
    query_offsets: list[int]
    batch: _BatchState | None


class DiffKVV2MetadataBuilder(AttentionMetadataBuilder[DiffKVV2Metadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    reorder_batch_threshold = None

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DiffKVV2Metadata:
        offsets = common_attn_metadata.query_start_loc_cpu[
            : common_attn_metadata.num_reqs + 1
        ].tolist()
        state = _batch_state
        if state is not None and not state.dummy:
            if len(offsets) != len(state.request_ids) + 1:
                raise ValueError("DiffKV request order does not match runner metadata")
            for i, req_id in enumerate(state.request_ids):
                if offsets[i + 1] - offsets[i] != state.scheduled[req_id]:
                    raise ValueError(
                        "DiffKV scheduled lengths do not match query offsets"
                    )
        return DiffKVV2Metadata(common_attn_metadata.num_actual_tokens, offsets, state)


class DiffKVV2Backend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16]
    supported_kv_cache_dtypes = ["auto", "float16"]
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_name() -> str:
        return "DIFFKV_V2"

    @staticmethod
    def get_impl_cls():
        return DiffKVV2Impl

    @staticmethod
    def get_builder_cls():
        return DiffKVV2MetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [32]

    @classmethod
    def customize_spec(cls, spec):
        from diffkv_v2.spec import DiffKVSpec

        if spec.head_size != 128 or spec.head_size_v != 128:
            raise ValueError("DiffKV-v2 requires equal K/V head sizes of 128")
        return DiffKVSpec(
            block_size=32,
            num_kv_heads=spec.num_kv_heads,
            head_size=128,
            head_size_v=128,
        )

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return capability.major >= 8

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False


def causal_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Causal attention with queries aligned to the end of the available prefix."""
    num_queries, num_keys = query.shape[0], key.shape[0]
    if num_keys < num_queries:
        raise ValueError("A prefill prefix cannot be shorter than its query chunk")
    if query.device.type == "cuda":
        from vllm.v1.attention.backends.fa_utils import (
            flash_attn_varlen_func,
            get_flash_attn_version,
        )

        fa_version = get_flash_attn_version(head_size=128)
        if fa_version is None:
            raise RuntimeError("DiffKV-v2 prefill requires a supported FlashAttention")
        q_offsets = torch.tensor(
            [0, num_queries], dtype=torch.int32, device=query.device
        )
        k_offsets = torch.tensor([0, num_keys], dtype=torch.int32, device=query.device)
        return flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=q_offsets,
            cu_seqlens_k=k_offsets,
            max_seqlen_q=num_queries,
            max_seqlen_k=num_keys,
            softmax_scale=scale,
            causal=True,
            fa_version=fa_version,
        )
    # A CPU reference permits semantic tests without the native CUDA extension.
    positions = torch.arange(num_keys - num_queries, num_keys, device=query.device)
    mask = torch.arange(num_keys, device=query.device)[None, :] <= positions[:, None]
    result = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        attn_mask=mask,
        scale=scale,
        enable_gqa=True,
    )
    return result.squeeze(0).transpose(0, 1)


def accumulate_prompt_scores(
    scores: torch.Tensor,
    query: torch.Tensor,
    key_prefix: torch.Tensor,
    query_start: int,
    prompt_len: int,
    scale: float,
) -> None:
    """Accumulate final-64 causal probability sums separately for every Q head.

    The maximum over the four heads in a GQA group must happen only after all
    chunks have contributed. Doing a maximum per chunk changes DiffKV's policy.
    """
    first = max(query_start, prompt_len - SCORE_WINDOW)
    query_end = query_start + query.shape[0]
    if first >= query_end:
        return
    selected = query[first - query_start :]
    num_kv_heads = key_prefix.shape[1]
    if query.shape[1] != num_kv_heads * 4:
        raise ValueError("DiffKV-v2 scoring requires four query heads per KV head")
    key_positions = torch.arange(key_prefix.shape[0], device=query.device)
    query_positions = torch.arange(first, query_end, device=query.device)
    mask = key_positions[None, :] > query_positions[:, None]
    # Bound FP32 scratch to one KV head, independently of the layer/head count.
    for head in range(num_kv_heads):
        q = selected[:, head * 4 : (head + 1) * 4].transpose(0, 1).float()
        k = key_prefix[:, head].float().transpose(0, 1)
        logits = torch.matmul(q, k) * scale
        logits.masked_fill_(mask.unsqueeze(0), float("-inf"))
        probabilities = torch.softmax(logits, dim=-1)
        scores[head * 4 : (head + 1) * 4, : key_prefix.shape[0]].add_(
            probabilities.sum(dim=1)
        )
        del logits, probabilities, q, k


class DiffKVV2Impl(AttentionImpl[DiffKVV2Metadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        # The DiffKV hooks exist only in Model Runner V2 (vllm/v1/worker/gpu/).
        # vLLM 0.32 selects Model Runner V2 automatically unless
        # VLLM_USE_V2_MODEL_RUNNER=0, Triton is missing, or a feature V2 does not
        # support is enabled (pipeline parallel with external_launcher, some
        # speculative decoding methods, ubatching):
        # https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/vllm.py#L703-L739
        # https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/vllm.py#L2914-L2953
        if not get_current_vllm_config().use_v2_model_runner:
            raise ValueError(
                "DIFFKV_V2 requires Model Runner V2, but vLLM selected the V1 "
                "model runner"
            )
        if head_size != 128 or num_kv_heads is None or num_heads != num_kv_heads * 4:
            raise ValueError("DiffKV-v2 requires head size 128 and GQA factor 4")
        if (
            alibi_slopes is not None
            or sliding_window is not None
            or logits_soft_cap not in (None, 0)
            or attn_type != AttentionType.DECODER
            or kv_sharing_target_layer_name is not None
            or any(
                value is not None and value is not False for value in kwargs.values()
            )
        ):
            raise ValueError("Unsupported attention feature for DiffKV-v2")
        if kv_cache_dtype not in ("auto", "float16"):
            raise ValueError("DiffKV-v2 owns its K8V4/K4V2 cache quantization")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.kv_cache_dtype = kv_cache_dtype

    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: DiffKVV2Metadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise ValueError("DiffKV-v2 does not support quantized attention output")
        if query.dtype != torch.float16:
            raise ValueError("DiffKV-v2 currently requires FP16 activations")
        q = query.view(-1, self.num_heads, self.head_size)
        k = key.view(-1, self.num_kv_heads, self.head_size)
        v = value.view(-1, self.num_kv_heads, self.head_size)
        out = output.view(-1, self.num_heads, self.head_size)
        state = None if attn_metadata is None else attn_metadata.batch
        if state is None or state.dummy:
            offsets = (
                [0, q.shape[0]]
                if attn_metadata is None
                else attn_metadata.query_offsets
            )
            for start, end in zip(offsets, offsets[1:]):
                if end > start:
                    out[start:end].copy_(
                        causal_prefill(
                            q[start:end], k[start:end], v[start:end], self.scale
                        )
                    )
            return output

        assert attn_metadata is not None
        controller = _controller
        if controller is None:
            raise RuntimeError("DiffKV worker cache is unavailable")
        layer_index = _layer_indices[layer.layer_name]
        if layer_index in state.executed_layers:
            raise RuntimeError("DiffKV attention layer executed twice in one batch")
        decode_rows = []
        decode_ids = []
        for row, req_id in enumerate(state.request_ids):
            start, end = attn_metadata.query_offsets[row : row + 2]
            request = controller.requests[req_id]
            if request.phase == "decode":
                decode_rows.append(start)
                decode_ids.append(req_id)
                continue
            prefix_start = state.computed[req_id]
            prefix_end = prefix_start + state.scheduled[req_id]
            raw = controller.raw_prompt(req_id)[layer_index]
            raw[0, prefix_start:prefix_end].copy_(k[start:end])
            raw[1, prefix_start:prefix_end].copy_(v[start:end])
            out[start:end].copy_(
                causal_prefill(
                    q[start:end], raw[0, :prefix_end], raw[1, :prefix_end], self.scale
                )
            )
            scores = controller.prompt_scores(req_id)[layer_index]
            accumulate_prompt_scores(
                scores,
                q[start:end],
                raw[0, :prefix_end],
                prefix_start,
                request.prompt_len,
                self.scale,
            )
            if prefix_end == request.prompt_len:
                self._compress_prompt(controller, req_id, layer_index, raw, scores)
                state.compressed_layers.setdefault(req_id, set()).add(layer_index)
        if decode_ids:
            row_indices = torch.tensor(decode_rows, device=q.device, dtype=torch.long)
            decode_output = self._decode(
                controller,
                state,
                layer_index,
                decode_ids,
                q[row_indices].contiguous(),
                k[row_indices].contiguous(),
                v[row_indices].contiguous(),
            )
            out.index_copy_(0, row_indices, decode_output)
        state.executed_layers.add(layer_index)
        return output

    def _compress_prompt(self, controller, req_id, layer_index, raw, scores) -> None:
        extension = get_extension()
        request = controller.requests[req_id]
        slots = controller.metadata([req_id]).slot_ids
        offsets = torch.tensor(
            [0, request.prompt_len], dtype=torch.int32, device=raw.device
        )
        common = (
            controller.kv_cache,
            slots,
            controller.block_tables,
            controller.kv_len_tables,
            offsets,
            controller.compress_config_tables,
            KV_BUFFER_SIZE,
            layer_index,
            *QUANT_ARGS,
        )
        if request.prompt_len <= SHORT_PROMPT_LIMIT:
            extension.cache_ops.compress_and_append_cache_prompt_phase(
                raw[0], raw[1], scores.unsqueeze(0), *common
            )
        else:
            maximum = scores.view(self.num_kv_heads, 4, request.prompt_len).amax(dim=1)
            queries_per_key = torch.arange(
                request.prompt_len, 0, -1, dtype=torch.float32, device=raw.device
            ).clamp_max_(SCORE_WINDOW)
            mean_scores = maximum / queries_per_key
            mean_scores[:, -KV_BUFFER_SIZE:] = 1.0
            sorted_indices = mean_scores.argsort(dim=-1).to(torch.int32).unsqueeze(0)
            extension.cache_ops.compress_and_append_cache_long_prompt_phase(
                raw[0], raw[1], maximum.unsqueeze(0), sorted_indices, *common
            )

    def _decode(self, controller, state, layer_index, req_ids, query, key, value):
        extension = get_extension()
        batch_size = len(req_ids)
        slots = controller.metadata(req_ids).slot_ids
        positions = torch.tensor(
            [state.computed[req_id] for req_id in req_ids],
            dtype=torch.int32,
            device=query.device,
        )
        retained_bound = state.decode_bounds[layer_index]
        extension.cache_ops.compress_and_append_cache_decode_phase(
            key,
            value,
            positions,
            controller.kv_cache,
            slots,
            controller.block_tables,
            controller.kv_len_tables,
            controller.compress_config_tables,
            retained_bound,
            KV_BUFFER_SIZE,
            layer_index,
            *QUANT_ARGS,
        )
        bound = retained_bound + 1
        num_partitions = (
            1
            if bound <= PARTITION_SIZE
            else (bound + PARTITION_SIZE - 1) // PARTITION_SIZE + 1
        )
        output = torch.empty_like(query)
        shape = (batch_size, self.num_heads, num_partitions)
        sums = torch.empty(shape, dtype=torch.float32, device=query.device)
        maxima = torch.empty_like(sums)
        partial_output = torch.empty(
            (*shape, self.head_size), dtype=query.dtype, device=query.device
        )
        scores = torch.empty(
            (batch_size, self.num_heads, bound),
            dtype=torch.float32,
            device=query.device,
        )
        extension.ops.sparse_paged_attention_gqa(
            slots,
            positions,
            output,
            sums,
            maxima,
            partial_output,
            scores,
            query,
            controller.kv_cache,
            layer_index,
            self.num_kv_heads,
            self.scale,
            controller.block_tables,
            controller.kv_len_tables,
            bound,
            *QUANT_ARGS,
            None,
        )
        return output
