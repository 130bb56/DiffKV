# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed-cache fixtures and an independent FP64 oracle for DiffKV.

Adapted from the frozen c489dc1 decode tests. The oracle performs one full
softmax and does not reproduce the native partition/reduction algorithm.
"""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HEAD_SIZE = 128
QUERY_HEADS_PER_KV = 4
PAGE_SIZE = 3392  # uint16 elements
LAYER_IDX = 1
SCALE = HEAD_SIZE**-0.5
QUANT_ARGS = (8, 4, 4, 2, 1, 1, 1, 1, 4, 2, 32, 60)
PARTITION_SIZE = 240
GQA_PARTITION_SIZE = 240
LOG_DIR = Path(__file__).resolve().parents[1] / "logs/attention"


def default_run_dir(name):
    """Return a new timestamped run directory under logs/attention."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return LOG_DIR / f"{stamp}-{name}"


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


# (prompt lengths, qkv_inputs options). The default inputs keep every token at
# high precision; query_scale 1.0 makes the writer quantize some tokens and
# aligned keys (see qkv_inputs) make it prune and quantize many.
PREFILL_CASES = {
    "one_prompt": ((256,), {}),
    "mixed_lengths": ((37, 256, 1024), {}),
    "kv_buffer_boundary": ((63, 64, 65, 129), {}),
    "peaked_scores": ((65, 256, 1024), {"query_scale": 1.0}),
    "aligned_keys": ((65, 256, 1024), {"aligned": 2.0}),
}


def read_region(cache, pages, count, bits):
    """Per-token records of one region, the inverse of _write_region's layout.

    Returns packed K and V codes, FP16 metadata, scores and positions of the
    first `count` tokens, plus `order`, their indices sorted by the bytes of
    position, codes and metadata (positions repeat: the writer stores
    max(position, prompt_len - 64) for the decode score mean).
    """
    k_bits, v_bits, tokens = bits
    layout = page_layout(k_bits, v_bits, tokens)
    fields = {"k_packed": [], "v_packed": [], "k_meta": [], "v_meta": []}
    fields.update(score=[], position=[])
    for page_idx in range(-(-count // tokens)):
        page = cache[pages[page_idx].item()]
        n = min(tokens, count - page_idx * tokens)
        k_page = page[: tokens * layout.k_packs].reshape(
            layout.k_packs // 16, tokens, 4, 4
        )
        fields["k_packed"].append(
            k_page.permute(1, 2, 0, 3).reshape(tokens, layout.k_packs)[:n]
        )
        v_page = page[
            layout.value : layout.value + layout.padded_tokens * layout.v_packs
        ].reshape(layout.v_packs // 8, layout.padded_tokens // 2, 8, 2)
        fields["v_packed"].append(
            v_page.permute(1, 3, 2, 0).reshape(layout.padded_tokens, layout.v_packs)[:n]
        )
        for name, base in (("k_meta", layout.k_meta), ("v_meta", layout.v_meta)):
            fields[name].append(page[base : base + 2 * n].reshape(n, 2))
        fields["score"].append(
            page.view(torch.float32)[layout.score // 2 : layout.score // 2 + n]
        )
        fields["position"].append(
            page.view(torch.int32)[layout.position // 2 : layout.position // 2 + n]
        )
    records = {
        name: torch.cat(values) if values else torch.empty(0)
        for name, values in fields.items()
    }
    if not count:
        records["order"] = torch.empty(0, dtype=torch.long)
        return records
    rows = torch.cat(
        [
            records["position"].view(torch.int16).reshape(-1, 2),
            *(records[name] for name in ("k_packed", "v_packed", "k_meta", "v_meta")),
        ],
        dim=1,
    ).numpy()
    records["order"] = torch.from_numpy(np.lexsort(rows.T[::-1]))
    return records


def _unpack(packed, bits):
    """Codes [tokens, 128] from 16-bit packs holding 16 / bits codes each."""
    width = 16 // bits
    packed = packed.to(torch.int64) & 0xFFFF
    shifts = torch.arange(width, dtype=torch.int64) * bits
    codes = (packed[:, :, None] >> shifts) & ((1 << bits) - 1)
    return codes.reshape(packed.shape[0], HEAD_SIZE).double()


def check_prompt_writer(controller, slots, lengths, starts, key, value, scores):
    """Check the prompt-phase writer against an independent reference.

    The token selection is recomputed in FP32 from the scores the writer
    consumed: mean = score / min(prompt_len - position, 64), pruned below
    prune_ratio / prompt_len, low precision below quant_ratio / prompt_len,
    the last 64 tokens always high precision. Every stored token is
    dequantized and matched to the nearest original token of its precision;
    the match must be one to one, within half a quantization step plus the
    FP16 metadata rounding, with the stored score and position field exact.
    """
    lengths_table = controller.kv_len_tables[slots, LAYER_IDX].cpu()
    tables = controller.block_tables[slots, LAYER_IDX].cpu()
    config = controller.compress_config_tables[slots].cpu()
    cache = controller.kv_cache.cpu()
    key, value, scores = key.float().cpu(), value.float().cpu(), scores.cpu()
    counts = {"high": 0, "low": 0, "pruned": 0}
    maximum_error_ratio = 0.0
    for row, length in enumerate(lengths):
        start = starts[row].item()
        positions = torch.arange(length)
        ideal = length - positions
        base = torch.tensor(1.0, dtype=torch.float32) / length
        prune, quant = base * config[row, 0], base * config[row, 1]
        for head in range(scores.shape[1]):
            score = scores[row, head, :length]
            mean = score / ideal.clamp(max=64).float()
            compressible = (ideal > 64) & (length > 64)
            pruned = compressible & (mean < prune)
            low = compressible & ~pruned & (mean < quant)
            high = ~pruned & ~low
            expected_lengths = [int(high.sum()), int(low.sum())]
            if lengths_table[row, head].tolist() != expected_lengths:
                raise AssertionError(
                    f"Row {row}, KV head {head}: retained lengths "
                    f"{lengths_table[row, head].tolist()}, expected "
                    f"{expected_lengths}"
                )
            counts["high"] += expected_lengths[0]
            counts["low"] += expected_lengths[1]
            counts["pruned"] += int(pruned.sum())
            high_pages = -(-expected_lengths[0] // 32)
            regions = (
                ((8, 4, 32), tables[row, head, :high_pages], high),
                ((4, 2, 60), tables[row, head, high_pages:], low),
            )
            for bits, pages, selected in regions:
                count = int(selected.sum())
                if not count:
                    continue
                stored = read_region(cache, pages, count, bits)
                expected_positions = positions[selected]
                originals = []
                for tensor in (key, value):
                    originals.append(tensor[start + expected_positions, head].double())
                dequantized = []
                for name, meta_name, field_bits in (
                    ("k_packed", "k_meta", bits[0]),
                    ("v_packed", "v_meta", bits[1]),
                ):
                    meta = stored[meta_name].view(torch.float16).double()
                    scale, zero = meta[:, :1], meta[:, 1:]
                    codes = _unpack(stored[name], field_bits)
                    dequantized.append((codes * scale + zero, scale, zero, field_bits))
                deq_key = dequantized[0][0]
                match = torch.cdist(deq_key, originals[0]).argmin(dim=1)
                if match.unique().numel() != count:
                    raise AssertionError(
                        f"Row {row}, KV head {head}: stored {bits[0]}-bit keys do "
                        "not match the selected tokens one to one"
                    )
                for (values, scale, zero, field_bits), original in zip(
                    dequantized, originals
                ):
                    error = (values - original[match]).abs()
                    bound = (
                        scale.abs() * (0.5 + ((1 << field_bits) - 1) * 2**-10)
                        + zero.abs() * 2**-10
                        + 1e-6
                    )
                    if (error > bound).any():
                        raise AssertionError(
                            f"Row {row}, KV head {head}: {bits[0]}-bit token "
                            "exceeds the quantization error bound"
                        )
                    maximum_error_ratio = max(
                        maximum_error_ratio,
                        (error / scale.abs().clamp(min=1e-12)).max().item(),
                    )
                matched_positions = expected_positions[match]
                if not torch.equal(stored["score"], score[matched_positions]):
                    raise AssertionError(
                        f"Row {row}, KV head {head}: stored scores differ"
                    )
                position_field = torch.maximum(
                    matched_positions, torch.full_like(matched_positions, length - 64)
                ).to(torch.int32)
                if not torch.equal(stored["position"], position_field):
                    raise AssertionError(
                        f"Row {row}, KV head {head}: stored positions differ"
                    )
    return {
        "writer_tokens": counts,
        "writer_max_error_over_scale": maximum_error_ratio,
    }


SCORE_WINDOW = 64


def reference_prompt_scores(scores, query, key, scale):
    """Final-64 causal probability sums per query head, as DiffKV defines them.

    Independent of the production path: one explicit matmul and softmax per KV
    head over the whole prompt, summed over the last SCORE_WINDOW queries.
    """
    prompt_len, kv_heads = key.shape[0], key.shape[1]
    first = max(0, prompt_len - SCORE_WINDOW)
    selected = query[first:]
    key_positions = torch.arange(prompt_len, device=query.device)
    query_positions = torch.arange(first, prompt_len, device=query.device)
    mask = key_positions[None, :] > query_positions[:, None]
    for head in range(kv_heads):
        group = slice(head * QUERY_HEADS_PER_KV, (head + 1) * QUERY_HEADS_PER_KV)
        logits = torch.matmul(
            selected[:, group].transpose(0, 1).float(),
            key[:, head].float().transpose(0, 1),
        )
        logits.mul_(scale).masked_fill_(mask.unsqueeze(0), float("-inf"))
        scores[group, :prompt_len].add_(torch.softmax(logits, dim=-1).sum(dim=1))


def reference_prefill(controller, request_ids, starts, query, key, value, output):
    """Prefill one layer per request: FA2, score sums and prompt compression.

    The reference prompt_prefill is compared against. Each request is written,
    attended, scored and compressed on its own, with its own kernel launches.
    """
    from diffkv.backend import KV_BUFFER_SIZE, causal_prefill
    from diffkv.ops import QUANT_ARGS, get_extension

    extension = get_extension()
    for row, req_id in enumerate(request_ids):
        start, end = starts[row], starts[row + 1]
        raw = controller.raw_prompt(req_id)[LAYER_IDX]
        raw[0].copy_(key[start:end])
        raw[1].copy_(value[start:end])
        output[start:end].copy_(causal_prefill(query[start:end], raw[0], raw[1], SCALE))
        scores = controller.prompt_scores(req_id)[LAYER_IDX]
        scores.zero_()
        reference_prompt_scores(scores, query[start:end], raw[0], SCALE)
        extension.cache_ops.compress_and_append_cache_prompt_phase(
            raw[0],
            raw[1],
            scores.unsqueeze(0),
            controller.kv_cache,
            controller.metadata([req_id]).slot_ids,
            controller.block_tables,
            controller.kv_len_tables,
            torch.tensor([0, end - start], dtype=torch.int32, device=raw.device),
            controller.compress_config_tables,
            KV_BUFFER_SIZE,
            LAYER_IDX,
            *QUANT_ARGS,
        )
    return output


def check_prefill(lengths, inputs, kv_heads=8):
    """prompt_prefill against a per-request reference on the same inputs.

    Both start from freshly admitted requests. Attention output, the
    group-maximum scores, retained lengths and every retained token of the
    layer must agree. Both share the prompt-phase writer, so its output is
    also checked against check_prompt_writer's reference.
    """
    from attention.benchmark_prefill import admit_requests, qkv_inputs
    from diffkv.prefill import prompt_prefill

    query, key, value, starts = qkv_inputs(lengths, kv_heads, **inputs)
    reference, request_ids = admit_requests(lengths, kv_heads)
    candidate, _ = admit_requests(lengths, kv_heads)
    expected_output = reference_prefill(
        reference,
        request_ids,
        starts.tolist(),
        query,
        key,
        value,
        torch.empty_like(query),
    )
    slots = candidate.metadata(request_ids).slot_ids
    output, scores = prompt_prefill(
        query,
        key,
        value,
        starts.cuda(),
        max(lengths),
        SCALE,
        candidate.kv_cache,
        slots,
        candidate.block_tables,
        candidate.kv_len_tables,
        candidate.compress_config_tables,
        LAYER_IDX,
    )
    torch.accelerator.synchronize()
    torch.testing.assert_close(output, expected_output, atol=1e-3, rtol=1e-3)
    maximum_score_error = 0.0
    for row, (req_id, length) in enumerate(zip(request_ids, lengths)):
        sums = reference.prompt_scores(req_id)[LAYER_IDX]
        group_max = sums.view(kv_heads, QUERY_HEADS_PER_KV, length).amax(dim=1)
        torch.testing.assert_close(
            scores[row, :, :length], group_max, atol=1e-6, rtol=1e-3
        )
        maximum_score_error = max(
            maximum_score_error,
            (scores[row, :, :length] - group_max).abs().max().item(),
        )
    slots = slots.long()
    for name in ("block_tables", "kv_len_tables"):
        if not torch.equal(
            getattr(reference, name)[slots, LAYER_IDX],
            getattr(candidate, name)[slots, LAYER_IDX],
        ):
            raise AssertionError(f"Prompt compression changed {name}")
    # The writer places tokens in score order, so near-equal scores from
    # differently ordered FP32 sums may swap two tokens' slots. Compare each
    # head's tokens matched by content: payload, metadata and positions bit
    # for bit, the stored score to tolerance.
    lengths_table = reference.kv_len_tables[slots, LAYER_IDX].cpu()
    tables = reference.block_tables[slots, LAYER_IDX].cpu()
    caches = reference.kv_cache.cpu(), candidate.kv_cache.cpu()
    maximum_stored_score_error = 0.0
    tokens_compared = 0
    for row in range(len(lengths)):
        for head in range(kv_heads):
            high, low = lengths_table[row, head].tolist()
            high_pages = -(-high // 32)
            regions = (
                ((8, 4, 32), tables[row, head, :high_pages], high),
                ((4, 2, 60), tables[row, head, high_pages:], low),
            )
            for bits, pages, count in regions:
                expected, actual = (
                    read_region(cache, pages, count, bits) for cache in caches
                )
                expected, actual = (
                    {key: value[records["order"]] for key, value in records.items()}
                    for records in (expected, actual)
                )
                for name in ("position", "k_packed", "v_packed", "k_meta", "v_meta"):
                    if not torch.equal(expected[name], actual[name]):
                        raise AssertionError(
                            f"Prompt compression changed {name} of row {row}, "
                            f"KV head {head}, {bits[0]}-bit keys"
                        )
                torch.testing.assert_close(
                    actual["score"], expected["score"], atol=1e-6, rtol=1e-3
                )
                if count:
                    maximum_stored_score_error = max(
                        maximum_stored_score_error,
                        (actual["score"] - expected["score"]).abs().max().item(),
                    )
                tokens_compared += count
    return {
        "output_max_abs_error": (output - expected_output).abs().max().item(),
        "score_max_abs_error": maximum_score_error,
        "stored_score_max_abs_error": maximum_stored_score_error,
        "tokens_compared": tokens_compared,
        **check_prompt_writer(candidate, slots, lengths, starts, key, value, scores),
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
    prefill = [
        {
            "name": name,
            "lengths": lengths,
            "inputs": inputs,
            **check_prefill(lengths, inputs),
        }
        for name, (lengths, inputs) in PREFILL_CASES.items()
    ]
    return {
        "status": "passed",
        "metadata": metadata(extension),
        "cases": results,
        "prefill_cases": prefill,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="default: logs/attention/<UTC>-correctness/correctness.json",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = default_run_dir("correctness") / "correctness.json"
    from diffkv.ops import get_extension

    torch.set_num_threads(2)
    report = run_checks(get_extension())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
