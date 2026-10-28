# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare DiffKV v2 and target-vLLM FA2 on identical retained K/V tokens.

This isolates attention-kernel cost. It does not measure cache compression,
serving overhead, pruning accuracy, or an end-to-end full-context speedup.
"""

import argparse
import importlib.metadata
import json
import os
import statistics
import sys
from pathlib import Path

import torch

from benchmarks.diffkv_v2.kernel_check import (
    HEAD_SIZE,
    LAYER_IDX,
    SCALE,
    allocate_workspaces,
    attention_args,
    build_fixture,
    check_fixture,
    device_inputs,
    metadata,
)


def compact_device_inputs(fixture):
    """Exclude unused oracle pages from the timing helper's L2 footprint."""
    pages = set()
    for slot in fixture.slot_ids.tolist():
        for head in range(fixture.kv_len_tables.size(2)):
            high, low = fixture.kv_len_tables[slot, LAYER_IDX, head].tolist()
            table = fixture.block_tables[slot, LAYER_IDX, head]
            pages.update(table[: (high + 31) // 32].tolist())
            if low:
                pages.update(table[-((low + 59) // 60) :].tolist())
    indices = torch.tensor(sorted(pages), dtype=torch.int64)
    remap = torch.full((fixture.kv_cache.size(0),), -1, dtype=torch.int32)
    remap[indices] = torch.arange(indices.numel(), dtype=torch.int32)
    inputs = device_inputs(fixture)
    inputs["kv_cache"] = fixture.kv_cache[indices].cuda()
    inputs["block_tables"] = remap[fixture.block_tables.long()].cuda()
    return inputs


def make_baseline(fixture, block_size=16):
    """Build paged FP16 K/V from the oracle's identical logical retained set."""
    batch, heads, _ = fixture.query.shape
    kv_heads = heads // 4
    length = fixture.max_context_len
    pages_per_request = (length + block_size - 1) // block_size
    shape = (batch, pages_per_request * block_size, kv_heads, HEAD_SIZE)
    key = torch.zeros(shape, dtype=torch.float16)
    value = torch.zeros_like(key)
    for sequence in range(batch):
        for head in range(kv_heads):
            logical_key, logical_value = fixture.logical_kv[sequence, head]
            if logical_key.size(0) != length:
                raise ValueError("Latency fixtures require uniform per-head lengths")
            key[sequence, :length, head] = logical_key.half()
            value[sequence, :length, head] = logical_value.half()
    page_shape = (-1, block_size, kv_heads, HEAD_SIZE)
    key = key.reshape(page_shape).cuda()
    value = value.reshape(page_shape).cuda()
    query = fixture.query.cuda()
    output = torch.empty_like(query)
    starts = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")
    tables = torch.arange(
        batch * pages_per_request, dtype=torch.int32, device="cuda"
    ).reshape(batch, pages_per_request)
    return (query, key, value, output, starts, lengths, tables), length


def timing_backend():
    try:
        from cupti import cupti  # noqa: F401

        version = importlib.metadata.version("cupti-python")
        if int(version.split(".")[0]) >= 13:
            return "flashinfer_cupti_cuda_graph_cold_l2"
    except (ImportError, importlib.metadata.PackageNotFoundError):
        pass
    return "flashinfer_cuda_graph_events_cold_l2"


def measure(fn, arguments, repeat_time_ms, graph_calls):
    from flashinfer.testing import (
        bench_gpu_time_with_cudagraph,
        bench_gpu_time_with_cupti,
    )

    for _ in range(25):
        fn(*arguments)
    torch.accelerator.synchronize()
    options = dict(
        input_args=arguments,
        cold_l2_cache=True,
        dry_run_time_ms=100,
        repeat_time_ms=repeat_time_ms,
    )
    if timing_backend() == "flashinfer_cupti_cuda_graph_cold_l2":
        samples = bench_gpu_time_with_cupti(fn, use_cuda_graph=True, **options)
    else:
        # FlashInfer's CUPTI fallback fixes graph size at ten calls, which can
        # visit fewer copies than its own cold-L2 rotation calculation requires.
        samples = bench_gpu_time_with_cudagraph(
            fn, num_iters_within_graph=graph_calls, **options
        )
    return [float(sample) * 1000 for sample in samples]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--high-fraction", type=float, default=0.25)
    parser.add_argument("--repeat-ms", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch < 1 or args.kv_heads < 1 or not 0 <= args.high_fraction <= 1:
        parser.error("Batch/heads must be positive and high-fraction in [0, 1]")
    if min(args.contexts) < 1 or args.repeat_ms < 1:
        parser.error("Contexts and repeat-ms must be positive")
    from flashinfer.testing.utils import calculate_rotation_count

    from vllm.diffkv_v2.ops import get_extension
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    torch.set_num_threads(2)
    extension = get_extension()
    rows = []
    for context in args.contexts:
        high = round(context * args.high_fraction)
        low = context - high
        fixture = build_fixture(
            ((high, low),) * args.batch,
            kv_heads=args.kv_heads,
            uniform_head_lengths=True,
        )
        correctness = check_fixture(extension, fixture)
        inputs = compact_device_inputs(fixture)
        workspace = allocate_workspaces(fixture)
        candidate_arguments = attention_args(fixture, inputs, workspace)
        baseline_arguments, length = make_baseline(fixture)
        graph_calls = max(
            10,
            *(
                calculate_rotation_count(
                    [value for value in arguments if isinstance(value, torch.Tensor)],
                    torch.device("cuda"),
                )
                for arguments in (candidate_arguments, baseline_arguments)
            ),
        )

        def candidate(*native_arguments):
            extension.ops.sparse_paged_attention_gqa(*native_arguments)

        def baseline(query, key, value, output, starts, lengths, tables, length=length):
            flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                out=output,
                max_seqlen_q=1,
                cu_seqlens_q=starts,
                max_seqlen_k=length,
                seqused_k=lengths,
                block_table=tables,
                softmax_scale=SCALE,
                causal=True,
                fa_version=2,
            )

        candidate(*candidate_arguments)
        torch.testing.assert_close(
            workspace["out"].cpu().double(),
            fixture.expected_output,
            atol=1e-3,
            rtol=3e-4,
        )
        baseline(*baseline_arguments)
        torch.testing.assert_close(
            baseline_arguments[3].cpu().double(),
            fixture.expected_output,
            atol=1e-3,
            rtol=3e-4,
        )
        timings = {"diffkv_v2": [], "vllm_fa2_fp16": []}
        for name in ("diffkv_v2", "vllm_fa2_fp16", "vllm_fa2_fp16", "diffkv_v2"):
            fn, parameters = (
                (candidate, candidate_arguments)
                if name == "diffkv_v2"
                else (baseline, baseline_arguments)
            )
            timings[name].extend(measure(fn, parameters, args.repeat_ms, graph_calls))
        medians = {name: statistics.median(values) for name, values in timings.items()}
        # Minimum bytes: one KV payload/metadata read per head plus score RMW.
        candidate_bytes = (
            args.batch
            * args.kv_heads
            * (high * (128 + 64 + 8 + 8) + low * (64 + 32 + 8 + 8))
        )
        baseline_bytes = args.batch * args.kv_heads * context * HEAD_SIZE * 2 * 2
        rows.append(
            {
                "context": context,
                "batch": args.batch,
                "kv_heads": args.kv_heads,
                "high_tokens": high,
                "low_tokens": low,
                "graph_calls_per_replay": graph_calls,
                "correctness": correctness,
                "latency_us": medians,
                "samples_us": timings,
                "fa2_over_diffkv_ratio": medians["vllm_fa2_fp16"]
                / medians["diffkv_v2"],
                "minimum_kv_bytes": {
                    "diffkv_v2": candidate_bytes,
                    "vllm_fa2_fp16": baseline_bytes,
                },
                "effective_minimum_kv_gbps": {
                    "diffkv_v2": candidate_bytes / (medians["diffkv_v2"] * 1000),
                    "vllm_fa2_fp16": baseline_bytes / (medians["vllm_fa2_fp16"] * 1000),
                },
            }
        )
    report = {
        "status": "completed",
        "metadata": metadata(extension),
        "command": [sys.executable, *sys.argv],
        "timing_backend": timing_backend(),
        "operation": "DiffKV main + reduce + score update versus paged FA2 attention",
        "comparison_scope": (
            "identical retained KV set; no pruning or serving speedup claim"
        ),
        "excluded": [
            "cache writer",
            "allocation",
            "compilation",
            "input generation",
            "host transfer",
        ],
        "tuning": (
            "production v2 partition240 and upstream FA2 defaults, no tuning sweep"
        ),
        "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING"),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"status": report["status"], "output": str(args.output)}, indent=2)
    )


if __name__ == "__main__":
    main()
