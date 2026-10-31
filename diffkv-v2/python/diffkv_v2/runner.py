# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DiffKV request lifecycle hooks for the eager V2 model runner."""

from dataclasses import replace
from typing import Any

import torch


def enabled(config) -> bool:
    backend = config.attention_config.backend
    return backend is not None and backend.name == "DIFFKV_V2"


class DiffKVRunnerState:
    def __init__(self, config):
        parallel = config.parallel_config
        unsupported = {
            "non-FP16 weights": config.model_config.dtype != torch.float16,
            "non-eager execution": not config.model_config.enforce_eager,
            "prefix caching": config.cache_config.enable_prefix_caching,
            "async scheduling": config.scheduler_config.async_scheduling,
            "speculative decoding": config.speculative_config is not None,
            "distributed execution": any(
                value != 1
                for value in (
                    parallel.tensor_parallel_size,
                    parallel.pipeline_parallel_size,
                    parallel.data_parallel_size,
                    parallel.decode_context_parallel_size,
                    parallel.prefill_context_parallel_size,
                )
            ),
            "LoRA": config.lora_config is not None,
            "KV transfer": config.kv_transfer_config is not None,
        }
        rejected = [name for name, active in unsupported.items() if active]
        if rejected:
            raise ValueError("DIFFKV_V2 does not support " + ", ".join(rejected))
        self.config = config
        self.controller = None
        self.last_report: dict[str, Any] | None = None

    def profile_workspace(self, device: torch.device) -> torch.Tensor:
        """Include cache metadata and peak attention scratch in memory profiling."""
        config = self.config
        model = config.model_config
        parallel = config.parallel_config
        length = model.max_model_len
        requests = config.scheduler_config.max_num_seqs
        layers = model.get_num_layers(parallel)
        heads = model.get_num_kv_heads(parallel)
        max_pages = torch.cuda.get_device_properties(device).total_memory // 6784
        table_width = (length + 31) // 32 + 2
        metadata = requests * layers * heads * (4 * table_width + 16)
        metadata += 8 * requests + 12 * max_pages
        # Include the measured first-use FP32 GEMM workspace as well as scores.
        score_scratch = 3000 * length + 16 * 1024**2 + 32 * heads * length
        decode_scratch = (
            requests
            * 4
            * heads
            * (4 * length + 264 * ((length + 239) // 240 + 1) + 1024)
        )
        return torch.empty(
            metadata + max(score_scratch, decode_scratch),
            dtype=torch.uint8,
            device=device,
        )

    def initialize(self, kv_caches: dict[str, torch.Tensor]) -> None:
        from diffkv_v2.backend import set_controller
        from diffkv_v2.cache import DiffKVCacheController
        from diffkv_v2.ops import PAGE_BYTES, PAGE_SIZE

        if not kv_caches:
            raise ValueError("DIFFKV_V2 requires decoder attention layers")
        caches = list(kv_caches.values())
        storage = caches[0].untyped_storage()
        if any(
            cache.untyped_storage().data_ptr() != storage.data_ptr() for cache in caches
        ):
            raise ValueError("DIFFKV_V2 requires one shared KV allocation")
        num_pages = storage.nbytes() // PAGE_BYTES
        arena = torch.empty(0, dtype=torch.int16, device=caches[0].device)
        arena.set_(storage, 0, (num_pages, PAGE_SIZE), (PAGE_SIZE, 1))
        config = self.config
        self.controller = DiffKVCacheController(
            arena=arena,
            max_num_requests=config.scheduler_config.max_num_seqs,
            num_layers=len(kv_caches),
            num_kv_heads=config.model_config.get_num_kv_heads(config.parallel_config),
            max_model_len=config.model_config.max_model_len,
        )
        set_controller(self.controller, list(kv_caches))
        self.last_report = self.controller.report()

    def remove_request(self, req_id: str) -> None:
        if self.controller is not None:
            self.controller.free_request(req_id)

    def preempt_requests(self, scheduler_output) -> None:
        if self.controller is not None:
            for req_id in scheduler_output.preempted_req_ids or ():
                self.controller.preempt_request(req_id)

    def add_request(self, request) -> None:
        if self.controller is None:
            raise RuntimeError("DIFFKV_V2 cache was not initialized")
        if request.num_computed_tokens != 0:
            raise ValueError("DIFFKV_V2 resumed requests must recompute from zero")
        self.controller.begin_request(request.req_id, len(request.prefill_token_ids))

    def prepare_batch(self, batch, dummy: bool) -> None:
        from diffkv_v2.backend import set_batch_state

        set_batch_state(
            batch.req_ids,
            dict(zip(batch.req_ids, map(int, batch.num_computed_tokens_np))),
            dict(zip(batch.req_ids, map(int, batch.num_scheduled_tokens))),
            dummy=dummy,
        )

    def finish_batch(self, dummy: bool) -> None:
        if not dummy:
            from diffkv_v2.backend import finish_batch

            self.last_report = finish_batch()

    def attach_report(self, output, *, refresh: bool = False):
        if refresh and self.controller is not None:
            self.last_report = self.controller.report()
        # Empty upstream outputs can be shared singletons.
        return replace(output, diffkv_cache_report=self.last_report)
