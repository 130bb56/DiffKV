# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical storage description for the shared DiffKV page arena."""

from dataclasses import dataclass

import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import FullAttentionSpec


@dataclass(frozen=True, kw_only=True)
class DiffKVSpec(FullAttentionSpec):
    dtype: torch.dtype = torch.int16
    num_head_slots: int | None = 1
    state_content_bytes: int | None = 6784
    tokens_per_state: int = 32
    dcp_sharded: bool = False

    def __post_init__(self):
        super().__post_init__()
        if (
            self.block_size != 32
            or self.dtype != torch.int16
            or self.head_size != 128
            or self.num_head_slots != 1
            or self.state_content_bytes != 6784
            or self.tokens_per_state != 32
        ):
            raise ValueError("DiffKV-v2 requires a 6784-byte packed page")

    @property
    def prefix_cacheable(self) -> bool:
        return False

    @property
    def uses_slot_mapping(self) -> bool:
        return False

    @property
    def has_layer_views(self) -> bool:
        return True

    def max_num_blocks_per_req(self, vllm_config, max_len: int) -> int:
        # This table stores one request handle. Physical pages live in the
        # controller's independent per-layer, per-head tables.
        return 1

    def max_memory_usage_bytes(self, vllm_config) -> int:
        length = vllm_config.model_config.max_model_len
        raw_pages = cdiv(length * self.num_kv_heads * 528, 6784)
        packed_pages = (cdiv(length, 32) + 1) * self.num_kv_heads
        return (raw_pages + packed_pages) * 6784
