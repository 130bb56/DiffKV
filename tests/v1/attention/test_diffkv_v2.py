# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal chunk boundaries and the final-64 DiffKV compression statistic."""

from types import SimpleNamespace

import pytest
import torch

from vllm.diffkv_v2 import backend
from vllm.diffkv_v2.backend import accumulate_prompt_scores, causal_prefill
from vllm.diffkv_v2.cache import DiffKVCacheController
from vllm.v1.attention.backend import CommonAttentionMetadata


def _inputs(length: int, device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    generator = torch.Generator().manual_seed(20261029)
    tensors = [
        torch.randn(length, heads, 128, generator=generator, dtype=torch.float16)
        for heads in (8, 2, 2)
    ]
    return tuple(tensor.to(device) for tensor in tensors)


def _reference_probabilities(query, key, positions):
    query = query.cpu().double().transpose(0, 1)
    key = key.cpu().double().repeat_interleave(4, dim=1).transpose(0, 1)
    logits = query @ key.transpose(1, 2) * (128**-0.5)
    mask = torch.arange(key.shape[1])[None, :] > positions[:, None]
    return logits.masked_fill(mask, float("-inf")).softmax(dim=-1)


@pytest.mark.parametrize(
    ("device", "length", "ends"),
    [
        ("cpu", 17, (3, 8, 17)),
        ("cuda", 97, (20, 50, 70, 97)),
        ("cuda", 4001, (127, 1023, 3947, 3970, 4001)),
    ],
)
@torch.inference_mode()
def test_chunked_prefill_preserves_original_causal_positions(device, length, ends):
    """A partial prefill must attend to its earlier chunks with offset causality."""
    query, key, value = _inputs(length, device)
    whole = causal_prefill(query, key, value, 128**-0.5)
    chunks = []
    start = 0
    for end in ends:
        chunk = causal_prefill(query[start:end], key[:end], value[:end], 128**-0.5)
        chunks.append(chunk)
        # Check the first and last row against an independent dense calculation.
        rows = torch.tensor(sorted({start, end - 1}))
        probabilities = _reference_probabilities(query[rows], key[:end], rows)
        ref_value = value[:end].cpu().double().repeat_interleave(4, dim=1)
        expected = (probabilities @ ref_value.transpose(0, 1)).transpose(0, 1)
        actual = chunk[(rows - start).to(device)].cpu().double()
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=1e-3)
        start = end
    torch.testing.assert_close(torch.cat(chunks), whole, rtol=3e-3, atol=1e-3)


@pytest.mark.parametrize(
    ("device", "length", "ends"),
    [
        ("cpu", 17, (3, 8, 17)),
        ("cpu", 97, (20, 50, 70, 97)),
        ("cuda", 97, (20, 50, 70, 97)),
        ("cuda", 4001, (127, 1023, 3947, 3970, 4001)),
    ],
)
@torch.inference_mode()
def test_final64_scores_accumulate_across_chunks_before_gqa_max(device, length, ends):
    """Head-wise sums must span all final queries before reducing GQA heads."""
    query, key, _ = _inputs(length, device)
    scores = torch.zeros((8, length), dtype=torch.float32, device=device)
    wrong_chunk_max = torch.zeros((2, length), device=device)
    start = 0
    for end in ends:
        before = scores.clone()
        accumulate_prompt_scores(
            scores, query[start:end], key[:end], start, length, 128**-0.5
        )
        contribution = scores - before
        wrong_chunk_max.add_(contribution.view(2, 4, length).amax(dim=1))
        start = end

    first = max(0, length - 64)
    probabilities = _reference_probabilities(
        query[first:], key, torch.arange(first, length)
    )
    expected = probabilities.sum(dim=1)
    torch.testing.assert_close(scores.cpu().double(), expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        scores.sum(dim=1),
        torch.full((8,), float(min(64, length)), device=device),
        rtol=2e-6,
        atol=2e-5,
    )
    maximum = scores.view(2, 4, length).amax(dim=1)
    expected_maximum = expected.view(2, 4, length).amax(dim=1)
    torch.testing.assert_close(
        maximum.cpu().double(), expected_maximum, rtol=2e-5, atol=2e-6
    )
    assert not torch.allclose(maximum, wrong_chunk_max, rtol=1e-3, atol=1e-4)


@torch.inference_mode()
def test_queries_before_final64_do_not_change_compression_scores():
    query, key, _ = _inputs(129, "cpu")
    scores = torch.zeros((8, 129), dtype=torch.float32)
    accumulate_prompt_scores(scores, query[:65], key[:65], 0, 129, 128**-0.5)
    assert torch.count_nonzero(scores).item() == 0
    accumulate_prompt_scores(scores, query[65:66], key[:66], 65, 129, 128**-0.5)
    torch.testing.assert_close(scores.sum(dim=1), torch.ones(8))
    assert torch.count_nonzero(scores[:, 66:]).item() == 0


def _assert_pool_accounting(controller, report):
    used = sum(
        request["packed_pages"] + request["raw_pages"]
        for request in report["requests"].values()
    )
    assert used + report["free_pages"] == report["total_pages"]
    assert torch.count_nonzero(controller.block_refs).item() == used


def _run_native_request(length, ends):
    controller = DiffKVCacheController(
        num_pages=4096,
        max_num_requests=3,
        num_layers=2,
        num_kv_heads=2,
        max_model_len=4100,
        device="cuda",
    )
    names = ["layer.0.attn", "layer.1.attn"]
    backend.set_controller(controller, names)
    impl = backend.DiffKVV2Impl(8, 128, 128**-0.5, num_kv_heads=2)
    builder = backend.DiffKVV2MetadataBuilder(None, names, None, torch.device("cuda"))
    generator = torch.Generator(device="cuda").manual_seed(131)
    query, key, value = [
        torch.randn(
            (2, length + 4, heads, 128),
            generator=generator,
            device="cuda",
            dtype=torch.float16,
        )
        for heads in (8, 2, 2)
    ]
    try:
        controller.begin_request("a", length)
        arena_pointer = controller.kv_cache.untyped_storage().data_ptr()
        assert controller.raw_prompt("a").untyped_storage().data_ptr() == arena_pointer
        assert (
            controller.prompt_scores("a").untyped_storage().data_ptr() == arena_pointer
        )
        _assert_pool_accounting(controller, controller.report())
        prefill_outputs: list[torch.Tensor] = []
        decode_outputs: list[torch.Tensor] = []
        start = 0
        for end in [*ends, length + 1, length + 2, length + 3, length + 4]:
            count = end - start
            backend.set_batch_state(["a"], {"a": start}, {"a": count})
            offsets = torch.tensor([0, count], dtype=torch.int32)
            common = CommonAttentionMetadata(
                query_start_loc=offsets.cuda(),
                query_start_loc_cpu=offsets,
                seq_lens=torch.tensor([end], device="cuda", dtype=torch.int32),
                num_reqs=1,
                num_actual_tokens=count,
                max_query_len=count,
                max_seq_len=end,
                block_table_tensor=torch.empty(
                    (1, 1), device="cuda", dtype=torch.int32
                ),
                slot_mapping=torch.empty(count, device="cuda", dtype=torch.int64),
            )
            metadata = builder.build(0, common)
            for layer_index, name in enumerate(names):
                output = torch.empty_like(query[layer_index, start:end])
                impl.forward(
                    SimpleNamespace(layer_name=name),
                    query[layer_index, start:end],
                    key[layer_index, start:end],
                    value[layer_index, start:end],
                    controller.kv_cache,
                    metadata,
                    output,
                )
                assert torch.isfinite(output).all()
                if layer_index == 0:
                    outputs = prefill_outputs if end <= length else decode_outputs
                    outputs.append(output.clone())
            report = backend.finish_batch()
            assert report is not None
            _assert_pool_accounting(controller, report)
            if end >= length:
                assert report["requests"]["a"]["raw_pages"] == 0
            start = end

        slot = controller.requests["a"].slot_id
        retained = controller.kv_len_tables[slot].cpu().clone()
        controller.begin_request("partial", 17)
        controller.set_num_computed_tokens("partial", 7)
        controller.preempt_request("partial")
        controller.free_request("a")
        report = controller.report()
        _assert_pool_accounting(controller, report)
        assert report["free_pages"] == report["largest_free_run"] == 4096
        assert report["preemptions"] == 1
        assert controller.free_request("a") == 0
        return (
            torch.cat(prefill_outputs).cpu(),
            torch.cat(decode_outputs).cpu(),
            retained,
        )
    finally:
        backend.set_controller(None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("length", "ends"),
    [
        (1, (1,)),
        (17, (3, 8, 17)),
        (97, (20, 50, 70, 97)),
        (4001, (127, 1023, 3947, 3970, 4001)),
    ],
)
@torch.inference_mode()
def test_native_cache_lifecycle_matches_whole_prefill_after_chunking(length, ends):
    """Compression, GQA decode, preemption and free must preserve pool ownership."""
    whole = _run_native_request(length, [length])
    chunked = _run_native_request(length, ends)
    torch.testing.assert_close(chunked[0], whole[0], atol=1e-3, rtol=3e-3)
    torch.testing.assert_close(chunked[1], whole[1], atol=1e-3, rtol=3e-3)
    torch.testing.assert_close(chunked[2], whole[2], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@torch.inference_mode()
def test_native_mixed_prefill_decode_preserves_request_order_and_positions():
    """Reordering mixed rows must leave each request's causal history unchanged."""
    controller = DiffKVCacheController(
        num_pages=256,
        max_num_requests=2,
        num_layers=1,
        num_kv_heads=2,
        max_model_len=128,
        device="cuda",
    )
    name = "layer.0.attn"
    backend.set_controller(controller, [name])
    impl = backend.DiffKVV2Impl(8, 128, 128**-0.5, num_kv_heads=2)
    builder = backend.DiffKVV2MetadataBuilder(None, [name], None, torch.device("cuda"))
    data = {"a": _inputs(100, "cuda"), "b": _inputs(18, "cuda")}
    controller.begin_request("a", 97)
    plans = [
        (["a"], {"a": 0}, {"a": 97}),
        (["b", "a"], {"b": 0, "a": 97}, {"b": 5, "a": 1}),
        (["a", "b"], {"a": 98, "b": 5}, {"a": 1, "b": 12}),
        (["b", "a"], {"b": 17, "a": 99}, {"b": 1, "a": 1}),
    ]
    try:
        for step, (req_ids, computed, scheduled) in enumerate(plans):
            if step == 1:
                controller.begin_request("b", 17)
            backend.set_batch_state(req_ids, computed, scheduled)
            query, key, value = [
                torch.cat(
                    [
                        data[req_id][index][
                            computed[req_id] : computed[req_id] + scheduled[req_id]
                        ]
                        for req_id in req_ids
                    ]
                )
                for index in range(3)
            ]
            offsets = torch.tensor(
                [0, *[scheduled[req_id] for req_id in req_ids]], dtype=torch.int32
            ).cumsum(0, dtype=torch.int32)
            ends = [computed[req_id] + scheduled[req_id] for req_id in req_ids]
            common = CommonAttentionMetadata(
                query_start_loc=offsets.cuda(),
                query_start_loc_cpu=offsets,
                seq_lens=torch.tensor(ends, device="cuda", dtype=torch.int32),
                num_reqs=len(req_ids),
                num_actual_tokens=query.shape[0],
                max_query_len=max(scheduled.values()),
                max_seq_len=max(ends),
                block_table_tensor=torch.empty(
                    (len(req_ids), 1), device="cuda", dtype=torch.int32
                ),
                slot_mapping=torch.empty(
                    query.shape[0], device="cuda", dtype=torch.int64
                ),
            )
            output = torch.empty_like(query)
            impl.forward(
                SimpleNamespace(layer_name=name),
                query,
                key,
                value,
                controller.kv_cache,
                builder.build(0, common),
                output,
            )
            assert torch.isfinite(output).all()
            for row, req_id in enumerate(req_ids):
                start, end = computed[req_id], ends[row]
                if controller.requests[req_id].phase == "prefill":
                    q, k, v = data[req_id]
                    expected = causal_prefill(q[start:end], k[:end], v[:end], 128**-0.5)
                    torch.testing.assert_close(
                        output[offsets[row] : offsets[row + 1]], expected
                    )
            report = backend.finish_batch()
            assert report is not None
            _assert_pool_accounting(controller, report)
            for req_id, end in zip(req_ids, ends):
                assert report["requests"][req_id]["num_computed_tokens"] == end
        controller.free_request("b")
        controller.free_request("a")
        assert controller.report()["free_pages"] == 256
    finally:
        backend.set_controller(None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@torch.inference_mode()
def test_native_decode_page_growth_then_preemption_recomputes_full_prefix():
    """Reused request slots must replay generated tokens after losing their cache."""
    controller = DiffKVCacheController(
        num_pages=256,
        max_num_requests=1,
        num_layers=1,
        num_kv_heads=2,
        max_model_len=128,
        device="cuda",
    )
    name = "layer.0.attn"
    backend.set_controller(controller, [name])
    impl = backend.DiffKVV2Impl(8, 128, 128**-0.5, num_kv_heads=2)
    builder = backend.DiffKVV2MetadataBuilder(None, [name], None, torch.device("cuda"))
    query, key, value = _inputs(107, "cuda")

    def step(req_id: str, end: int) -> torch.Tensor:
        start = controller.requests[req_id].num_computed_tokens
        count = end - start
        backend.set_batch_state([req_id], {req_id: start}, {req_id: count})
        offsets = torch.tensor([0, count], dtype=torch.int32)
        common = CommonAttentionMetadata(
            query_start_loc=offsets.cuda(),
            query_start_loc_cpu=offsets,
            seq_lens=torch.tensor([end], device="cuda", dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=count,
            max_query_len=count,
            max_seq_len=end,
            block_table_tensor=torch.empty((1, 1), device="cuda", dtype=torch.int32),
            slot_mapping=torch.empty(count, device="cuda", dtype=torch.int64),
        )
        output = torch.empty_like(query[start:end])
        impl.forward(
            SimpleNamespace(layer_name=name),
            query[start:end],
            key[start:end],
            value[start:end],
            controller.kv_cache,
            builder.build(0, common),
            output,
        )
        assert torch.isfinite(output).all()
        report = backend.finish_batch()
        assert report is not None
        assert report["requests"][req_id]["num_computed_tokens"] == end
        _assert_pool_accounting(controller, report)
        return output

    try:
        # Disable pruning/requantization for this phase so 80 decode steps must
        # cross the 32-, 64- and 96-token high-page boundaries deterministically.
        slot = controller.begin_request("active", 17, prune_ratio=0, quant_ratio=0)
        step("active", 17)
        assert controller.block_num_tables[slot, :, :, 0].eq(1).all()
        for end in range(18, 98):
            step("active", end)
        assert controller.kv_len_tables[slot, :, :, 0].eq(97).all()
        assert controller.block_num_tables[slot, :, :, 0].eq(4).all()
        owned = controller.report()["requests"]["active"]["packed_pages"]
        assert controller.preempt_request("active") == owned
        report = controller.report()
        _assert_pool_accounting(controller, report)
        assert report["preemptions"] == 1
        assert report["free_pages"] == report["largest_free_run"] == 256

        # Recompute includes all 80 generated tokens, and uses the normal
        # compression policy. Its final-64 scoring window spans two chunks.
        assert controller.begin_request("active", 97) == slot
        assert controller.requests["active"].num_computed_tokens == 0
        for end in (31, 65, 97):
            step("active", end)
        resumed = torch.cat([step("active", end) for end in range(98, 108)])
        resumed_lengths = controller.kv_len_tables[slot].clone()
        controller.free_request("active")

        # The valid reference is a fresh prefill of the full logical prefix:
        # replay intentionally rebuilds the lossy cache rather than restoring it.
        assert controller.begin_request("fresh", 97) == slot
        step("fresh", 97)
        fresh = torch.cat([step("fresh", end) for end in range(98, 108)])
        torch.testing.assert_close(resumed, fresh, rtol=3e-3, atol=1e-3)
        torch.testing.assert_close(
            resumed_lengths, controller.kv_len_tables[slot], rtol=0, atol=0
        )
        controller.free_request("fresh")
        report = controller.report()
        _assert_pool_accounting(controller, report)
        assert report["free_pages"] == report["largest_free_run"] == 256
        assert not report["requests"]
    finally:
        backend.set_controller(None)
