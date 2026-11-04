# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare DiffKV chunked prefill attention with vLLM FA2 FP16 prefill.

One step holds `batch` prefill requests. Each has `prefix` retained tokens
already in its cache and a `chunk` of new tokens. The DiffKV arm dequantizes
the packed prefix of every (KV head, request) into FP16 rows, runs one FA2
varlen call over prefix plus chunk, sums the attention probabilities per key
token (the score proxy) and adds the prefix token scores into the cache pages.
The FA2 arm writes the chunk K/V into a paged FP16 cache and runs FA2 over
prefix plus chunk with a block table, as the vLLM FLASH_ATTN backend does.
Prefix 0 is a prompt processed in one chunk. Packing, allocation and plan
construction are untimed; no accuracy check is made here.
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch
from attention.benchmark_latency import (
    BLOCK_SIZE,
    Arm,
    compact_device_inputs,
    measure,
    timing_backend,
)
from attention.correctness import (
    HEAD_SIZE,
    LAYER_IDX,
    PAGE_SIZE,
    SCALE,
    build_fixture,
    default_run_dir,
    metadata,
)


def chunk_inputs(batch, chunk, kv_heads, seed=23):
    generator = torch.Generator().manual_seed(seed)
    heads = kv_heads * 4
    shape = (batch * chunk, heads, HEAD_SIZE)
    query = (torch.randn(shape, generator=generator) * 0.12).half().cuda()
    key = (torch.randn((batch * chunk, kv_heads, HEAD_SIZE), generator=generator) * 0.5)
    value = torch.randn((batch * chunk, kv_heads, HEAD_SIZE), generator=generator) * 0.5
    starts = torch.arange(batch + 1, dtype=torch.int32) * chunk
    return query, key.half().cuda(), value.half().cuda(), starts


def packed_prefix(batch, prefix, kv_heads, high_fraction):
    """Packed cache tables for the prefix; empty tables when prefix is 0."""
    if prefix == 0:
        slots = max(4, batch + 2)
        tables = torch.zeros((slots, 2, kv_heads, 4), dtype=torch.int32).cuda()
        lengths = torch.zeros((slots, 2, kv_heads, 2), dtype=torch.int32).cuda()
        cache = torch.zeros((1, PAGE_SIZE), dtype=torch.int16).cuda()
        slot_ids = list(range(batch))
        return cache, tables, lengths, slot_ids, None, 0
    high = round(prefix * high_fraction)
    fixture = build_fixture(
        ((high, prefix - high),) * batch,
        kv_heads=kv_heads,
        uniform_head_lengths=True,
        compute_reference=False,
    )
    inputs = compact_device_inputs(fixture)
    cache_bytes = inputs["kv_cache"].numel() * inputs["kv_cache"].element_size()
    slot_ids = fixture.slot_ids.tolist()
    return (
        inputs["kv_cache"],
        inputs["block_tables"],
        inputs["kv_len_tables"],
        slot_ids,
        fixture.kv_len_tables[slot_ids],
        cache_bytes,
    )


def make_diffkv(batch, prefix, chunk, kv_heads, high_fraction):
    from diffkv.prefill import chunk_prefill_attention, plan_chunk_prefill

    query, key, value, starts = chunk_inputs(batch, chunk, kv_heads)
    cache, tables, lengths, slot_ids, lengths_cpu, cache_bytes = packed_prefix(
        batch, prefix, kv_heads, high_fraction
    )
    plan = plan_chunk_prefill(
        starts.tolist(), slot_ids, lengths_cpu, "cuda", kv_heads=kv_heads
    )
    cu_seqlens = starts.cuda()

    def run(q, k, v):
        return chunk_prefill_attention(
            q, k, v, cu_seqlens, plan, cache, tables, lengths, LAYER_IDX, SCALE
        )

    output, scores, lse = run(query, key, value)
    return Arm(
        run,
        (query, key, value),
        output,
        cache_bytes,
        {
            "backend": "DIFFKV",
            "parts": [
                "query regroup copy and chunk K/V copy into the sequence rows",
                "packed prefix dequantization to FP16 rows (CUDA)",
                "FA2 varlen over (KV head, request) sequences with log-sum-exp",
                "column score kernel (Triton)",
                "prefix score scatter into pages (Triton atomic add)",
                "output regroup copy",
            ]
            if prefix > 0
            else [
                "FA2 over the chunk with log-sum-exp",
                "column score kernel (Triton)",
            ],
            "prefix_score_update_included": prefix > 0,
            "chunk_scores_shape": list(scores.shape),
            "lse_shape": list(lse.shape),
        },
        {},
    )


def make_fa2(batch, prefix, chunk, kv_heads):
    from vllm import _custom_ops
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    query, key, value, starts = chunk_inputs(batch, chunk, kv_heads)
    total = prefix + chunk
    pages_per_request = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
    pages = batch * pages_per_request
    generator = torch.Generator().manual_seed(29)
    key_cache = (
        torch.randn((pages, BLOCK_SIZE, kv_heads, HEAD_SIZE), generator=generator) * 0.5
    ).half().cuda()
    value_cache = (
        torch.randn((pages, BLOCK_SIZE, kv_heads, HEAD_SIZE), generator=generator) * 0.5
    ).half().cuda()
    tables = torch.arange(pages, dtype=torch.int32, device="cuda").reshape(
        batch, pages_per_request
    )
    slots = torch.cat(
        [
            torch.arange(prefix, total, dtype=torch.int64)
            + request * pages_per_request * BLOCK_SIZE
            for request in range(batch)
        ]
    ).cuda()
    cu_seqlens_q = starts.cuda()
    seqused_k = torch.full((batch,), total, dtype=torch.int32, device="cuda")
    output = torch.empty_like(query)
    scales = torch.ones((), dtype=torch.float32, device="cuda")

    def run(q, k, v):
        _custom_ops.reshape_and_cache_flash(
            k, v, key_cache, value_cache, slots, "auto", scales, scales
        )
        flash_attn_varlen_func(
            q=q,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=chunk,
            max_seqlen_k=total,
            seqused_k=seqused_k,
            block_table=tables,
            softmax_scale=SCALE,
            causal=True,
            fa_version=2,
        )

    run(query, key, value)
    return Arm(
        run,
        (query, key, value),
        output,
        key_cache.numel() * key_cache.element_size() * 2,
        {
            "backend": "FLASH_ATTN",
            "fa_version": 2,
            "cache_dtype": "float16",
            "parts": ["reshape_and_cache_flash for the chunk", "FA2 varlen, paged KV"],
            "page_tokens": BLOCK_SIZE,
        },
        {},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prefixes", type=int, nargs="+", default=[0, 2048])
    parser.add_argument("--chunks", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--high-fraction", type=float, default=0.25)
    parser.add_argument("--repeat-ms", type=int, default=500)
    parser.add_argument(
        "--output", type=Path, help="default: logs/<UTC>-prefill-b<batch>/latency.json"
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = default_run_dir(f"prefill-b{args.batch}") / "latency.json"
    if min(args.batch, args.kv_heads, args.repeat_ms, *args.chunks) < 1:
        parser.error("Batch, heads, repeat-ms and chunk lengths must be positive")
    if min(args.prefixes) < 0:
        parser.error("Prefix lengths must be nonnegative")
    if not 0 <= args.high_fraction <= 1:
        parser.error("high-fraction must lie in [0,1]")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    from diffkv.ops import get_extension

    torch.set_num_threads(2)
    extension = get_extension()
    rows = []
    for prefix in args.prefixes:
        for chunk in args.chunks:
            arms = {
                "diffkv": make_diffkv(
                    args.batch, prefix, chunk, args.kv_heads, args.high_fraction
                ),
                "fa2_fp16": make_fa2(args.batch, prefix, chunk, args.kv_heads),
            }
            samples = {name: [] for name in arms}
            order = list(arms)
            for name in [*order, *reversed(order)]:
                arm = arms[name]
                samples[name].extend(
                    measure(arm.function, arm.arguments, args.repeat_ms)
                )
            medians = {
                name: statistics.median(values) for name, values in samples.items()
            }
            row = {
                "batch": args.batch,
                "prefix": prefix,
                "chunk": chunk,
                "kv_heads": args.kv_heads,
                "high_fraction": args.high_fraction,
                "cuda_graph": False,
                "arms": {
                    name: {
                        "details": arm.details,
                        "cache_object_bytes": arm.cache_bytes,
                        "median_us": medians[name],
                        "samples_us": samples[name],
                    }
                    for name, arm in arms.items()
                },
                "fa2_over_diffkv_ratio": medians["fa2_fp16"] / medians["diffkv"],
            }
            rows.append(row)
            print(
                json.dumps({"prefix": prefix, "chunk": chunk, "median_us": medians}),
                flush=True,
            )
            del arms
            torch.cuda.empty_cache()
    report = {
        "status": "completed",
        "metadata": metadata(extension),
        "command": [sys.executable, *sys.argv],
        "timing_backend": timing_backend(),
        "execution_mode": "eager",
        "cuda_graph": False,
        "cold_l2_cache": True,
        "timing_scope": "CUDA events around a direct eager call; L2 flush excluded; "
        "host launch gaps within the event interval are included",
        "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING"),
        "operation": "DiffKV chunked prefill attention (FA2 chunk, packed prefix "
        "rows, merge, chunk scores) versus paged FA2 FP16 prefill with cache write",
        "comparison_scope": "attention of one layer for one step; no compression, "
        "no accuracy check, no serving claim",
        "excluded": [
            "cache compression and append",
            "packing",
            "allocation",
            "plan construction",
            "compilation",
            "input generation",
        ],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"status": report["status"], "output": str(args.output)}, indent=2)
    )


if __name__ == "__main__":
    main()
