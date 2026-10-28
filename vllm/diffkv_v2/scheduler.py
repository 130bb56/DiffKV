# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep upstream token scheduling while admitting real DiffKV physical pages."""

from vllm.diffkv_v2.admission import DiffKVAdmission
from vllm.diffkv_v2.spec import DiffKVSpec
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import iter_layer_specs


def uses_diffkv(kv_cache_config) -> bool:
    return any(
        isinstance(spec, DiffKVSpec)
        for group in kv_cache_config.kv_cache_groups
        for spec in iter_layer_specs(group.kv_cache_spec)
    )


class DiffKVBlockManager(FullAttentionManager):
    """One logical request handle, independent of packed physical page count."""

    supports_fine_grained_hash_lookup = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.enable_caching:
            raise ValueError("DiffKV-v2 does not support prefix caching")
        self._record_new_block_ids = False

    def get_num_blocks_to_allocate(
        self,
        request_id,
        num_tokens,
        new_computed_blocks,
        total_computed_tokens,
        num_local_computed_tokens,
        num_tokens_main_model,
        apply_admission_cap=False,
        prefill_end=0,
    ):
        if new_computed_blocks:
            raise ValueError("DiffKV-v2 cannot adopt shared prefix blocks")
        return int(num_tokens > 0 and not self.req_to_blocks.get(request_id))

    def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
        return super().allocate_new_blocks(
            request_id,
            min(num_tokens, self.block_size),
            min(num_tokens_main_model, self.block_size),
        )


class DiffKVKVCacheManager(KVCacheManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.enable_caching or self.use_eagle:
            raise ValueError("DiffKV-v2 requires prefix and speculative caching off")
        config = kwargs["kv_cache_config"]
        if len(config.kv_cache_groups) != 1:
            raise ValueError("DiffKV-v2 requires one homogeneous KV cache group")
        group = config.kv_cache_groups[0]
        specs = tuple(iter_layer_specs(group.kv_cache_spec))
        if not all(isinstance(spec, DiffKVSpec) for spec in specs):
            raise ValueError("DiffKV-v2 cannot mix physical cache formats")
        first_spec = specs[0]
        assert isinstance(first_spec, DiffKVSpec)
        sizes = {tensor.size for tensor in config.kv_cache_tensors}
        if len(sizes) != 1:
            raise ValueError("DiffKV-v2 expects one common backing allocation")
        arena_bytes = sizes.pop()
        if arena_bytes % 6784:
            raise ValueError("DiffKV arena must contain complete packed pages")
        self.physical = DiffKVAdmission(
            arena_bytes // 6784,
            len(group.layer_names),
            first_spec.num_kv_heads,
        )

    @property
    def usage(self):
        return self.physical.usage

    def new_step_starts(self):
        super().new_step_starts()
        self.physical.begin_step()

    def allocate_slots(self, request, num_new_tokens, *args, **kwargs):
        if kwargs.get("num_lookahead_tokens", 0):
            raise ValueError("DiffKV-v2 does not support speculative lookahead")
        if not self.physical.reserve(
            request.request_id,
            request.num_computed_tokens,
            request.num_tokens,
        ):
            return None
        blocks = super().allocate_slots(request, num_new_tokens, *args, **kwargs)
        if blocks is None:
            self.physical.cancel(request.request_id)
        return blocks

    def preempt_physical(self, request_id):
        self.physical.preempt(request_id)

    def update_physical_report(self, report):
        self.physical.update_report(report)
