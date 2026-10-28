# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-owned DiffKV pages and request state for eager serving.

Packed KV and temporary, uncompressed prefill KV share one physical arena.
The original DiffKV kernels own the packed page tables. Logical token positions
remain independent of the per-head retained lengths written by those kernels.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm.diffkv_v2.ops import (
    HIGH_PAGE_TOKENS,
    LOW_PAGE_TOKENS,
    PAGE_BYTES,
    PAGE_SIZE,
    get_extension,
)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass
class DiffKVRequestState:
    slot_id: int
    prompt_len: int
    num_computed_tokens: int = 0
    phase: str = "prefill"
    raw_page_start: int = 0
    raw_page_count: int = 0


@dataclass(frozen=True)
class DiffKVBatchState:
    request_ids: tuple[str, ...]
    slot_ids: torch.Tensor
    block_tables: torch.Tensor
    kv_len_tables: torch.Tensor
    block_num_tables: torch.Tensor
    compress_config_tables: torch.Tensor


class DiffKVCacheController:
    """Manage the legacy high/low packed layout within a fixed memory budget.

    Raw prefill storage requires contiguous physical pages. The initial eager
    implementation keeps a CPU free-page bitmap, synchronizing only changed
    page IDs. Its fragmentation limit is reported to the scheduler explicitly.
    All calls and attention kernels must use the same CUDA stream.
    """

    def __init__(
        self,
        *,
        max_num_requests: int,
        num_layers: int,
        num_kv_heads: int,
        max_model_len: int,
        num_pages: int | None = None,
        device: torch.device | str = "cuda",
        arena: torch.Tensor | None = None,
        mem_ops: Any | None = None,
    ) -> None:
        dimensions = (max_num_requests, num_layers, num_kv_heads, max_model_len)
        if any(value <= 0 for value in dimensions):
            raise ValueError("DiffKV cache dimensions must be positive")
        if arena is None:
            if num_pages is None or num_pages <= 0:
                raise ValueError("A positive num_pages or an arena is required")
            arena = torch.empty(
                (num_pages, PAGE_SIZE), dtype=torch.int16, device=device
            )
        if (
            arena.dtype != torch.int16
            or arena.ndim != 2
            or arena.shape[1] != PAGE_SIZE
            or not arena.is_contiguous()
            or arena.shape[0] <= 0
        ):
            raise ValueError("DiffKV arena must be contiguous int16[N, PAGE_SIZE]")
        if num_pages is not None and num_pages != arena.shape[0]:
            raise ValueError("num_pages does not match the supplied arena")
        if mem_ops is None and arena.device.type != "cuda":
            raise ValueError("DiffKV kernels require a CUDA arena")

        self.kv_cache = arena
        self.device = arena.device
        self.num_pages = arena.shape[0]
        self.max_num_requests = max_num_requests
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.max_model_len = max_model_len
        self.num_kv_caches = num_layers * num_kv_heads
        self.max_block_table_len = _ceil_div(max_model_len, HIGH_PAGE_TOKENS) + 2
        self.mem_ops = mem_ops if mem_ops is not None else get_extension().mem_mgt_ops

        table_shape = (max_num_requests, num_layers, num_kv_heads)
        self.block_tables = torch.full(
            (*table_shape, self.max_block_table_len),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.kv_len_tables = torch.zeros(
            (*table_shape, 2), dtype=torch.int32, device=self.device
        )
        self.block_num_tables = torch.zeros_like(self.kv_len_tables)
        self.compress_config_tables = torch.ones(
            (max_num_requests, 2), dtype=torch.float32, device=self.device
        )
        self.free_blocks = torch.arange(
            self.num_pages, dtype=torch.int32, device=self.device
        )
        self.block_refs = torch.zeros(
            self.num_pages, dtype=torch.int32, device=self.device
        )
        self._head = 0
        self._tail = 0
        # An explicit count distinguishes a completely full ring from an empty one.
        self._free_page_count = self.num_pages
        self._free_mask = np.ones(self.num_pages, dtype=np.bool_)
        self._largest_free_run = self.num_pages
        self._contiguity_dirty = False
        self._free_slots = deque(range(max_num_requests))
        self.requests: dict[str, DiffKVRequestState] = {}
        self.preemptions = 0

    @property
    def free_pages(self) -> int:
        return self._free_page_count

    def estimate_prompt_pages(self, prompt_len: int) -> int:
        """Packed reservation before prompt-end compression, excluding raw KV."""
        return (_ceil_div(prompt_len, HIGH_PAGE_TOKENS) + 1) * self.num_kv_caches

    def estimate_raw_pages(self, prompt_len: int) -> int:
        # FP16 K/V plus FP32 sums for four query heads per KV head.
        raw_bytes = prompt_len * self.num_kv_caches * (128 * 2 * 2 + 4 * 4)
        return _ceil_div(raw_bytes, PAGE_BYTES)

    def estimate_decode_pages(self, num_requests: int = 1) -> int:
        """Worst case: each head needs one page for each precision."""
        return num_requests * self.num_kv_caches * 2

    def _free_page_ids(self) -> list[int]:
        first_count = min(self.free_pages, self.num_pages - self._head)
        first = self.free_blocks[self._head : self._head + first_count].tolist()
        remaining = self.free_pages - first_count
        return first + self.free_blocks[:remaining].tolist()

    def _find_free_run(self, required: int) -> tuple[int | None, int]:
        edges = np.diff(np.concatenate(([False], self._free_mask, [False])))
        boundaries = np.flatnonzero(edges)
        starts, ends = boundaries[::2], boundaries[1::2]
        lengths = ends - starts
        self._largest_free_run = int(lengths.max(initial=0))
        self._contiguity_dirty = False
        candidates = np.flatnonzero(lengths >= required)
        first = int(starts[candidates[0]]) if candidates.size else None
        return first, self._largest_free_run

    def _ring_ids(self, start: int, count: int) -> list[int]:
        first = min(count, self.num_pages - start)
        return (
            self.free_blocks[start : start + first].tolist()
            + self.free_blocks[: count - first].tolist()
        )

    def _reserve_raw_pages(self, count: int) -> int:
        start, largest = self._find_free_run(count)
        if start is None:
            raise MemoryError(
                f"DiffKV prefill needs {count} contiguous pages; largest run is "
                f"{largest}, total free pages {self.free_pages}"
            )
        self._free_mask[start : start + count] = False
        self._contiguity_dirty = True
        remaining = np.flatnonzero(self._free_mask).astype(np.int32)
        if remaining.size:
            self.free_blocks[: len(remaining)].copy_(
                torch.from_numpy(remaining).to(self.device)
            )
        self._head = 0
        self._tail = len(remaining) % self.num_pages
        self._free_page_count = len(remaining)
        self.block_refs[start : start + count].fill_(1)
        return start

    def _release_raw_pages(self, state: DiffKVRequestState) -> None:
        count = state.raw_page_count
        if count == 0:
            return
        start = state.raw_page_start
        ids = torch.arange(start, start + count, dtype=torch.int32, device=self.device)
        first_count = min(count, self.num_pages - self._tail)
        self.free_blocks[self._tail : self._tail + first_count].copy_(ids[:first_count])
        if count > first_count:
            self.free_blocks[: count - first_count].copy_(ids[first_count:])
        self.block_refs[start : start + count].zero_()
        self._return_pages(count, np.arange(start, start + count))
        state.raw_page_count = 0

    def _take_pages(self, count: int) -> tuple[int, int]:
        if count < 0 or count > self.free_pages:
            raise MemoryError(f"DiffKV needs {count} pages; {self.free_pages} are free")
        start = self._head
        if count:
            ids = self._ring_ids(start, count)
            if not self._free_mask[ids].all():
                raise RuntimeError("DiffKV free ring contains allocated pages")
            self._free_mask[ids] = False
            self._contiguity_dirty = True
        self._head = (self._head + count) % self.num_pages
        self._free_page_count -= count
        return start, self._head

    def _return_pages(self, count: int, page_ids=None) -> None:
        if count < 0 or self.free_pages + count > self.num_pages:
            raise RuntimeError("Invalid DiffKV page release count")
        if count:
            ids = self._ring_ids(self._tail, count) if page_ids is None else page_ids
            if self._free_mask[ids].any():
                raise RuntimeError("DiffKV page was freed twice")
            self._free_mask[ids] = True
            self._contiguity_dirty = True
        self._tail = (self._tail + count) % self.num_pages
        self._free_page_count += count

    def _slot_ids(self, request_ids: list[str] | tuple[str, ...]) -> torch.Tensor:
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("Duplicate DiffKV request IDs")
        return torch.tensor(
            [self.requests[req_id].slot_id for req_id in request_ids],
            dtype=torch.int32,
            device=self.device,
        )

    def begin_request(
        self,
        request_id: str,
        prompt_len: int,
        *,
        prune_ratio: float = 0.02,
        quant_ratio: float = 1.0,
    ) -> int:
        """Reserve raw and packed prompt storage and return a stable request slot."""
        if request_id in self.requests:
            raise ValueError(f"DiffKV request {request_id!r} already exists")
        if not 0 < prompt_len <= self.max_model_len:
            raise ValueError("prompt_len must be within the model context limit")
        if not 0 <= prune_ratio <= quant_ratio <= 1:
            raise ValueError("DiffKV requires 0 <= prune_ratio <= quant_ratio <= 1")
        if not self._free_slots:
            raise MemoryError("No free DiffKV request metadata slots")
        packed_pages = self.estimate_prompt_pages(prompt_len)
        raw_pages = self.estimate_raw_pages(prompt_len)
        if packed_pages + raw_pages > self.free_pages:
            raise MemoryError(
                f"DiffKV prompt needs {packed_pages} packed and {raw_pages} raw "
                f"pages; {self.free_pages} are free"
            )

        raw_start = self._reserve_raw_pages(raw_pages)
        slot = self._free_slots.popleft()
        state = DiffKVRequestState(
            slot, prompt_len, raw_page_start=raw_start, raw_page_count=raw_pages
        )
        self.requests[request_id] = state
        self.block_tables[slot].fill_(-1)
        self.kv_len_tables[slot].zero_()
        self.block_num_tables[slot].zero_()
        self.compress_config_tables[slot] = torch.tensor(
            [prune_ratio, quant_ratio],
            dtype=torch.float32,
            device=self.device,
        )
        start, end = self._take_pages(packed_pages)
        self.mem_ops.allocate_seqs(
            self._slot_ids([request_id]),
            torch.tensor([prompt_len], dtype=torch.int32, device=self.device),
            torch.tensor(
                [packed_pages // self.num_kv_caches],
                dtype=torch.int32,
                device=self.device,
            ),
            self.block_tables,
            self.kv_len_tables,
            self.block_num_tables,
            self.free_blocks,
            self.block_refs,
            start,
            end,
        )
        self.prompt_scores(request_id).zero_()
        return slot

    def raw_prompt(self, request_id: str) -> torch.Tensor:
        """Return arena-backed FP16 [layers, 2, prompt_tokens, kv_heads, 128]."""
        state = self.requests[request_id]
        if state.phase != "prefill" or state.raw_page_count == 0:
            raise ValueError("Raw DiffKV storage is only available during prefill")
        flat = (
            self.kv_cache[
                state.raw_page_start : state.raw_page_start + state.raw_page_count
            ]
            .view(torch.float16)
            .flatten()
        )
        num_elements = self.num_kv_caches * state.prompt_len * 2 * 128
        return flat[:num_elements].view(
            self.num_layers, 2, state.prompt_len, self.num_kv_heads, 128
        )

    def prompt_scores(self, request_id: str) -> torch.Tensor:
        """Return arena-backed FP32 sums [layers, query_heads, prompt_tokens]."""
        state = self.requests[request_id]
        if state.phase != "prefill" or state.raw_page_count == 0:
            raise ValueError("DiffKV prompt scores are only live during prefill")
        flat = (
            self.kv_cache[
                state.raw_page_start : state.raw_page_start + state.raw_page_count
            ]
            .view(torch.float32)
            .flatten()
        )
        kv_float32_elements = self.num_kv_caches * state.prompt_len * 128
        score_elements = self.num_kv_caches * state.prompt_len * 4
        return flat[kv_float32_elements : kv_float32_elements + score_elements].view(
            self.num_layers, self.num_kv_heads * 4, state.prompt_len
        )

    def set_num_computed_tokens(self, request_id: str, num_tokens: int) -> None:
        state = self.requests[request_id]
        if not state.num_computed_tokens <= num_tokens <= self.max_model_len:
            raise ValueError("DiffKV logical token count must advance within context")
        state.num_computed_tokens = num_tokens

    def finalize_prompt(self, request_ids: list[str]) -> int:
        """Release prompt surplus after all layer kernels wrote retained lengths.

        The caller must finish consuming every raw KV view before this call.
        Returns the number of raw and packed pages returned to the common pool.
        """
        if not request_ids:
            return 0
        for req_id in request_ids:
            if self.requests[req_id].phase != "prefill":
                raise ValueError("DiffKV prompt was already finalized")
        slot_ids = self._slot_ids(request_ids)
        head_count = len(request_ids) * self.num_kv_caches
        counts = torch.empty(head_count * 2, dtype=torch.int32, device=self.device)
        offsets = torch.zeros(head_count + 1, dtype=torch.int32, device=self.device)
        offsets[0] = self._tail
        self.mem_ops.prepare_free_prompt(
            slot_ids,
            self.kv_len_tables,
            self.block_num_tables,
            HIGH_PAGE_TOKENS,
            LOW_PAGE_TOKENS,
            offsets,
            counts,
        )
        offsets.cumsum_(dim=0)
        released = int(offsets[-1].item()) - self._tail
        self.mem_ops.free_prompt(
            slot_ids,
            self.block_tables,
            self.block_num_tables,
            self.free_blocks,
            offsets,
            counts,
            self.block_refs,
        )
        self._return_pages(released)
        for req_id in request_ids:
            state = self.requests[req_id]
            released += state.raw_page_count
            self._release_raw_pages(state)
            state.phase = "decode"
        return released

    def append_decode(self, request_ids: list[str]) -> int:
        """Reserve space for one decode token per request before attention."""
        if not request_ids:
            return 0
        for req_id in request_ids:
            state = self.requests[req_id]
            if state.phase != "decode":
                raise ValueError("DiffKV decode requires a finalized prompt")
            if state.num_computed_tokens >= self.max_model_len:
                raise ValueError("DiffKV decode exceeds the model context limit")
        slot_ids = self._slot_ids(request_ids)
        offsets = torch.zeros(
            len(request_ids) * self.num_kv_caches * 2 + 1,
            dtype=torch.int32,
            device=self.device,
        )
        offsets[0] = self._head
        self.mem_ops.prepare_append_seqs(
            slot_ids,
            self.kv_len_tables,
            self.block_num_tables,
            8,
            4,
            4,
            2,
            HIGH_PAGE_TOKENS,
            LOW_PAGE_TOKENS,
            offsets,
        )
        offsets.cumsum_(dim=0)
        required = int(offsets[-1].item()) - self._head
        self._take_pages(required)
        self.mem_ops.append_seqs(
            slot_ids,
            self.block_tables,
            self.kv_len_tables,
            self.block_num_tables,
            self.free_blocks,
            self.block_refs,
            offsets,
        )
        return required

    def free_request(self, request_id: str) -> int:
        """Release packed and raw storage; unknown/already freed IDs are a no-op."""
        state = self.requests.get(request_id)
        if state is None:
            return 0
        slot_ids = self._slot_ids([request_id])
        offsets = torch.zeros(
            self.num_kv_caches * 2 + 1, dtype=torch.int32, device=self.device
        )
        offsets[0] = self._tail
        self.mem_ops.prepare_free_seqs(slot_ids, self.block_num_tables, offsets)
        offsets.cumsum_(dim=0)
        released = int(offsets[-1].item()) - self._tail
        self.mem_ops.free_seqs(
            slot_ids,
            self.block_tables,
            self.block_num_tables,
            self.free_blocks,
            offsets,
            self.block_refs,
        )
        self._return_pages(released)
        released += state.raw_page_count
        self._release_raw_pages(state)
        self.block_tables[state.slot_id].fill_(-1)
        self.kv_len_tables[state.slot_id].zero_()
        self.block_num_tables[state.slot_id].zero_()
        self._free_slots.append(state.slot_id)
        del self.requests[request_id]
        return released

    def preempt_request(self, request_id: str) -> int:
        """Drop state so a resumed request must replay its tokens from position zero."""
        if request_id in self.requests:
            self.preemptions += 1
        return self.free_request(request_id)

    def metadata(self, request_ids: list[str]) -> DiffKVBatchState:
        return DiffKVBatchState(
            tuple(request_ids),
            self._slot_ids(request_ids),
            self.block_tables,
            self.kv_len_tables,
            self.block_num_tables,
            self.compress_config_tables,
        )

    def report(self) -> dict[str, Any]:
        """Return synchronized physical usage and contiguous admission capacity."""
        if self._contiguity_dirty:
            self._find_free_run(1)
        req_ids = list(self.requests)
        slots = [self.requests[req_id].slot_id for req_id in req_ids]
        packed_counts = self.block_num_tables[slots].sum(dim=(1, 2, 3)).tolist()
        usage = {}
        for req_id, packed_pages in zip(req_ids, packed_counts):
            state = self.requests[req_id]
            usage[req_id] = {
                "slot_id": state.slot_id,
                "prompt_len": state.prompt_len,
                "num_computed_tokens": state.num_computed_tokens,
                "phase": state.phase,
                "packed_pages": packed_pages,
                "raw_pages": state.raw_page_count,
            }
        return {
            "total_pages": self.num_pages,
            "free_pages": self.free_pages,
            "largest_free_run": self._largest_free_run,
            "page_bytes": PAGE_BYTES,
            "preemptions": self.preemptions,
            "requests": usage,
        }
