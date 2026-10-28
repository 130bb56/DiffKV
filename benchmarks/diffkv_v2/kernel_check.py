# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed-cache fixtures and an independent FP64 oracle for DiffKV v2.

Adapted from the frozen c489dc1 decode tests. The oracle performs one full
softmax and does not reproduce the native partition/reduction algorithm.
"""

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

HEAD_SIZE = 128
QUERY_HEADS_PER_KV = 4
PAGE_SIZE = 3392  # uint16 elements
LAYER_IDX = 1
SCALE = HEAD_SIZE**-0.5
QUANT_ARGS = (8, 4, 4, 2, 1, 1, 1, 1, 4, 2, 32, 60)
PARTITION_SIZE = 240
GQA_PARTITION_SIZE = 240


def _align(value, boundary):
    return (value + boundary - 1) // boundary * boundary


def page_layout(k_bits, v_bits, tokens):
    """Describe the public unified-page format, in uint16 elements."""
    k_packs, v_packs = HEAD_SIZE * k_bits // 16, HEAD_SIZE * v_bits // 16
    k_meta = _align(tokens * k_packs * 2, 32) // 2
    value = _align(k_meta * 2 + tokens * 4, 128) // 2
    padded_tokens = _align(tokens, 2)
    v_meta = value + _align(padded_tokens * v_packs * 2, 32) // 2
    score = v_meta + _align(tokens * 4, 32) // 2
    return SimpleNamespace(
        k_meta=k_meta,
        value=value,
        v_meta=v_meta,
        score=score,
        position=score + tokens * 2,
        k_packs=k_packs,
        v_packs=v_packs,
        padded_tokens=padded_tokens,
    )


def _pack_codes(codes, bits):
    width = 16 // bits
    grouped = codes.reshape(codes.shape[0], HEAD_SIZE // width, width).to(torch.int64)
    shifts = torch.arange(width, dtype=torch.int64) * bits
    return torch.sum(grouped << shifts, dim=-1).to(torch.int16)


def _write_region(
    cache,
    pages,
    count,
    bits,
    generator,
    score_mask,
    adversarial_max=False,
    compute_reference=True,
):
    """Pack integer codes independently of the CUDA cache-append kernels."""
    k_bits, v_bits, tokens = bits
    layout = page_layout(k_bits, v_bits, tokens)
    k_codes = torch.randint(0, 1 << k_bits, (count, HEAD_SIZE), generator=generator)
    v_codes = torch.randint(0, 1 << v_bits, (count, HEAD_SIZE), generator=generator)
    # Different region logits and different query heads make incorrect low
    # partition normalization and reading another head before rescaling visible.
    k_codes[:, :8] = 3 if k_bits == 8 else 14
    if adversarial_max and k_bits == 8:
        k_codes[:, 0] = 0
        if count > 1:
            k_codes[1, 0] = 255
    k_scales = torch.full(
        (count,), 1 / 64 if k_bits == 8 else 1 / 4, dtype=torch.float16
    )
    v_scales = torch.full(
        (count,), 1 / 128 if v_bits == 4 else 1 / 2, dtype=torch.float16
    )
    k_zeros = torch.full((count,), -2, dtype=torch.float16)
    v_zeros = torch.full((count,), -0.5, dtype=torch.float16)
    # Dyadic metadata makes dequantization exactly representable in FP32,
    # keeping the reference independent of quantizer rounding differences.
    keys = values = None
    if compute_reference:
        keys = k_codes.double() * k_scales.double()[:, None] + k_zeros.double()[:, None]
        values = (
            v_codes.double() * v_scales.double()[:, None] + v_zeros.double()[:, None]
        )
    k_packed, v_packed = _pack_codes(k_codes, k_bits), _pack_codes(v_codes, v_bits)
    addresses = []
    for page_idx, page in enumerate(pages):
        begin, end = page_idx * tokens, min((page_idx + 1) * tokens, count)
        n = end - begin
        # K storage axes: [pack row, token, thread group, vector element].
        k_page = torch.full((tokens, layout.k_packs), 0x3555, dtype=torch.int16)
        k_page[:n] = k_packed[begin:end]
        k_page = k_page.reshape(tokens, 4, layout.k_packs // 16, 4)
        cache[page, : tokens * layout.k_packs] = k_page.permute(2, 0, 1, 3).reshape(-1)
        # V storage axes: [pack row, token pair, thread group, token in pair].
        v_page = torch.full(
            (layout.padded_tokens, layout.v_packs), 0x3555, dtype=torch.int16
        )
        v_page[:n] = v_packed[begin:end]
        v_page = v_page.reshape(layout.padded_tokens // 2, 2, 8, layout.v_packs // 8)
        cache[page, layout.value : layout.value + v_page.numel()] = v_page.permute(
            3, 0, 2, 1
        ).reshape(-1)
        half_view = cache[page].view(torch.float16)
        half_view[layout.k_meta : layout.k_meta + 2 * n].reshape(n, 2)[:] = torch.stack(
            (k_scales[begin:end], k_zeros[begin:end]), dim=1
        )
        half_view[layout.v_meta : layout.v_meta + 2 * n].reshape(n, 2)[:] = torch.stack(
            (v_scales[begin:end], v_zeros[begin:end]), dim=1
        )
        # Start with nonzero, token-specific scores to distinguish += from =.
        cache[page].view(torch.float32)[layout.score // 2 : layout.score // 2 + n] = (
            torch.arange(begin, end, dtype=torch.float32) % 13 + 1
        ) * 1e-5
        cache[page].view(torch.int32)[
            layout.position // 2 : layout.position // 2 + n
        ] = torch.arange(begin, end)
        score_mask[page, layout.score : layout.score + 2 * n] = True
        addresses.extend(
            page * (PAGE_SIZE // 2) + layout.score // 2 + offset for offset in range(n)
        )
    return keys, values, torch.tensor(addresses, dtype=torch.int64)


def build_fixture(
    lengths_by_seq=((33, 61), (95, 119)),
    kv_heads=2,
    seed=17,
    use_alibi=False,
    adversarial_max=False,
    compute_reference=True,
    uniform_head_lengths=False,
):
    """Return CPU packed inputs and a partition-independent full-softmax oracle.

    Helpers are also imported by the model-free attention microbenchmark.
    score_indices address the flattened FP32 view of kv_cache.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to build packed attention inputs")
    if not lengths_by_seq:
        raise ValueError("At least one sequence is required")
    generator = torch.Generator().manual_seed(seed)
    seqs, layers = len(lengths_by_seq), 2
    slots = max(4, seqs + 2)
    query_heads = kv_heads * QUERY_HEADS_PER_KV
    block_capacity = (
        max(
            (high_length + 31) // 32 + (low_length + 59) // 60
            for high_length, low_length in lengths_by_seq
        )
        + 3
    )
    blocks = slots * layers * kv_heads * block_capacity
    cache = torch.full((blocks, PAGE_SIZE), 0x3555, dtype=torch.int16)
    score_mask = torch.zeros_like(cache, dtype=torch.bool)
    tables = (
        torch.randperm(blocks, generator=generator)
        .reshape(slots, layers, kv_heads, block_capacity)
        .to(torch.int32)
    )
    lengths = torch.zeros((slots, layers, kv_heads, 2), dtype=torch.int32)
    selected_slots = (
        torch.tensor((3, 1)[:seqs], dtype=torch.int32)
        if seqs <= 2
        else torch.arange(1, seqs + 1, dtype=torch.int32).flip(0)
    )
    query = (
        torch.randn((seqs, query_heads, HEAD_SIZE), generator=generator) * 0.12
    ).half()
    for head in range(query_heads):
        query[:, head, :8] = (1.25, -1.0, 0.4, -0.6)[head % 4]
    if adversarial_max:
        query.zero_()
        query[:, :, 0] = 512
    slopes = (
        torch.linspace(0, 0.0003, query_heads, dtype=torch.float32)
        if use_alibi
        else None
    )
    expected_output = (
        torch.empty((seqs, query_heads, HEAD_SIZE), dtype=torch.float64)
        if compute_reference
        else None
    )
    score_updates, score_indices, logical_kv = {}, {}, {}
    max_context_len = 0
    for seq, (high, low) in enumerate(lengths_by_seq):
        slot = int(selected_slots[seq])
        for head in range(kv_heads):
            # Give every KV head a distinct length and partial-page boundary.
            high_length, low_length = (
                (high, low)
                if uniform_head_lengths
                else (max(0, high - head), max(0, low - 2 * head))
            )
            if high_length + low_length == 0:
                raise ValueError("Every KV head needs at least one retained token")
            lengths[slot, LAYER_IDX, head] = torch.tensor(
                (high_length, low_length), dtype=torch.int32
            )
            max_context_len = max(max_context_len, high_length + low_length)
            regions, addresses = [], []
            for count, bits, pages in (
                (
                    high_length,
                    (8, 4, 32),
                    tables[slot, LAYER_IDX, head, : (high_length + 31) // 32],
                ),
                (
                    low_length,
                    (4, 2, 60),
                    tables[slot, LAYER_IDX, head].flip(0)[: (low_length + 59) // 60],
                ),
            ):
                keys, values, indices = _write_region(
                    cache,
                    pages.tolist(),
                    count,
                    bits,
                    generator,
                    score_mask,
                    adversarial_max=adversarial_max,
                    compute_reference=compute_reference,
                )
                regions.append((keys, values))
                addresses.append(indices)
            if not compute_reference:
                continue
            keys = torch.cat([region[0] for region in regions])
            values = torch.cat([region[1] for region in regions])
            q = query[seq, head * 4 : (head + 1) * 4].double()
            logits = q @ keys.T * SCALE
            if slopes is not None:
                logits += slopes[head * 4 : (head + 1) * 4].double()[:, None] * (
                    torch.arange(high_length + low_length).double()
                    - high_length
                    - low_length
                    + 1
                )
            weights = torch.softmax(logits, dim=-1)
            expected_output[seq, head * 4 : (head + 1) * 4] = weights @ values
            score_updates[seq, head] = weights.max(dim=0).values
            score_indices[seq, head] = torch.cat(addresses)
            logical_kv[seq, head] = (keys, values)
    return SimpleNamespace(
        slot_ids=selected_slots,
        positions=torch.tensor(
            [high + low + 137 for high, low in lengths_by_seq], dtype=torch.int32
        ),
        query=query,
        kv_cache=cache,
        block_tables=tables,
        kv_len_tables=lengths,
        alibi_slopes=slopes,
        max_context_len=max_context_len,
        expected_output=expected_output,
        score_updates=score_updates,
        score_indices=score_indices,
        score_mask=score_mask,
        logical_kv=logical_kv,
    )


def device_inputs(fixture, device="cuda"):
    result = {
        name: getattr(fixture, name).to(device)
        for name in (
            "slot_ids",
            "positions",
            "kv_cache",
            "block_tables",
            "kv_len_tables",
        )
    }
    # A wider backing allocation exercises the nontrivial q_stride path.
    seqs, heads, size = fixture.query.shape
    backing = torch.empty(
        (seqs, heads + 2, size), dtype=fixture.query.dtype, device=device
    )
    result["query"] = backing[:, :heads]
    result["query"].copy_(fixture.query)
    result["alibi_slopes"] = (
        fixture.alibi_slopes.to(device) if fixture.alibi_slopes is not None else None
    )
    return result


def allocate_workspaces(fixture, device="cuda", partition_size=None):
    seqs, heads, size = fixture.query.shape
    partition_size = PARTITION_SIZE if partition_size is None else partition_size
    capacity = (
        1
        if fixture.max_context_len <= partition_size
        else (fixture.max_context_len + partition_size - 1) // partition_size + 1
    )
    return dict(
        out=torch.empty((seqs, heads, size), dtype=torch.float16, device=device),
        exp_sums=torch.full((seqs, heads, capacity), float("nan"), device=device),
        max_logits=torch.full((seqs, heads, capacity), float("nan"), device=device),
        tmp_out=torch.full(
            (seqs, heads, capacity, size),
            float("nan"),
            dtype=torch.float16,
            device=device,
        ),
        tmp_scores=torch.full(
            (seqs, heads, fixture.max_context_len), float("nan"), device=device
        ),
    )


def attention_args(fixture, inputs, workspace):
    return (
        inputs["slot_ids"],
        inputs["positions"],
        workspace["out"],
        workspace["exp_sums"],
        workspace["max_logits"],
        workspace["tmp_out"],
        workspace["tmp_scores"],
        inputs["query"],
        inputs["kv_cache"],
        LAYER_IDX,
        fixture.query.shape[1] // 4,
        SCALE,
        inputs["block_tables"],
        inputs["kv_len_tables"],
        fixture.max_context_len,
        *QUANT_ARGS,
        inputs["alibi_slopes"],
    )


def check_fixture(extension, fixture, repeats=3):
    """Check output, score increments, and all non-score cache bytes."""
    inputs = device_inputs(fixture)
    workspace = allocate_workspaces(fixture)
    arguments = attention_args(fixture, inputs, workspace)
    original_scores = fixture.kv_cache.view(torch.float32).reshape(-1)
    maximum_output_error = 0.0
    maximum_score_error = 0.0
    for repeat in range(1, repeats + 1):
        extension.ops.sparse_paged_attention_gqa(*arguments)
        torch.accelerator.synchronize()
        output = workspace["out"].cpu().double()
        torch.testing.assert_close(
            output, fixture.expected_output, atol=1e-3, rtol=3e-4
        )
        maximum_output_error = max(
            maximum_output_error, (output - fixture.expected_output).abs().max().item()
        )
        cache = inputs["kv_cache"].cpu()
        scores = cache.view(torch.float32).reshape(-1)
        for key, indices in fixture.score_indices.items():
            increment = (scores[indices] - original_scores[indices]).double()
            expected = fixture.score_updates[key] * repeat
            torch.testing.assert_close(increment, expected, atol=5e-8, rtol=5e-4)
            maximum_score_error = max(
                maximum_score_error, (increment - expected).abs().max().item()
            )
        if not torch.equal(
            cache[~fixture.score_mask], fixture.kv_cache[~fixture.score_mask]
        ):
            raise AssertionError(
                "Attention changed payload, metadata or unrelated pages"
            )
        for name in ("kv_len_tables", "block_tables"):
            if not torch.equal(inputs[name].cpu(), getattr(fixture, name)):
                raise AssertionError(f"Attention changed {name}")
    return {
        "output_max_abs_error": maximum_output_error,
        "score_increment_max_abs_error": maximum_score_error,
        "repeats": repeats,
    }


def metadata(extension):
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "device_capability": torch.cuda.get_device_capability(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "extension_fingerprint": extension.build_fingerprint,
        "dtype": "float16",
        "head_dim": HEAD_SIZE,
        "queries_per_kv": QUERY_HEADS_PER_KV,
        "quant_args": QUANT_ARGS,
    }


def run_checks(extension):
    cases = {
        "partial_pages": ((33, 61), (95, 119)),
        "high_or_low_only": ((193, 0), (0, 181)),
        "partition_boundary": ((119, 120), (120, 120), (121, 120)),
        "independent_region_boundaries": ((239, 1), (240, 1), (241, 1), (33, 241)),
        "more_than_32_partitions": ((4096, 4096),),
        "long_partial_regions": ((33, 5761), (5761, 1201)),
        "adversarial_softmax": ((32, 0), (31, 0)),
    }
    results = []
    for name, lengths in cases.items():
        fixture = build_fixture(lengths, adversarial_max=name == "adversarial_softmax")
        result = check_fixture(extension, fixture)
        results.append({"name": name, "lengths": lengths, **result})
    fixture = build_fixture(((1, 1),), kv_heads=1)
    results.append({"name": "one_token_regions", **check_fixture(extension, fixture)})
    inputs = device_inputs(fixture)
    workspace = allocate_workspaces(fixture)
    inputs["query"] = inputs["query"].float()
    try:
        extension.ops.sparse_paged_attention_gqa(
            *attention_args(fixture, inputs, workspace)
        )
    except RuntimeError:
        if not torch.equal(inputs["kv_cache"].cpu(), fixture.kv_cache):
            raise AssertionError("Rejected dtype modified cache") from None
    else:
        raise AssertionError("Unsupported FP32 queries were accepted")
    return {"status": "passed", "metadata": metadata(extension), "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from vllm.diffkv_v2.ops import get_extension

    torch.set_num_threads(2)
    report = run_checks(get_extension())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
