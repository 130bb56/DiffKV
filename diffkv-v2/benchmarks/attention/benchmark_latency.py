# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare DiffKV v2, vLLM FA2 FP16 and FlashInfer FP8 decode attention.

All arms read the same retained KV tokens. This isolates attention-kernel cost.
It does not measure cache compression, serving overhead, pruning accuracy, or an
end-to-end full-context speedup. Packing, quantization, planning and the
independent FP64 oracle are untimed. FP8 kernel error uses the actual
dequantized FP8 cache; quantization error uses the common unquantized retained
KV reference and is reported separately.
"""

import argparse
import json
import os
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from attention.correctness import (
    HEAD_SIZE,
    LAYER_IDX,
    SCALE,
    allocate_workspaces,
    attention_args,
    build_fixture,
    check_fixture,
    default_run_dir,
    device_inputs,
    metadata,
)

BLOCK_SIZE = 16


@dataclass
class Arm:
    function: Callable
    arguments: tuple
    output: torch.Tensor
    cache_bytes: int
    details: dict
    correctness: dict


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


def make_baseline(fixture, block_size=BLOCK_SIZE):
    """Build a paged FP16 KV cache from the oracle's retained tokens."""
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
    return "flashinfer_cuda_events_eager_cold_l2"


def measure(fn, arguments, repeat_time_ms):
    """Direct eager launches; event intervals can include host launch gaps."""
    from flashinfer.testing import bench_gpu_time_with_cuda_event

    for _ in range(25):
        fn(*arguments)
    torch.accelerator.synchronize()
    # This helper flushes L2 before the start event on the same stream.
    # It calls fn directly and never captures a CUDA graph.
    samples = bench_gpu_time_with_cuda_event(
        fn,
        input_args=arguments,
        cold_l2_cache=True,
        dry_run_time_ms=100,
        repeat_time_ms=repeat_time_ms,
    )
    return [float(sample) * 1000 for sample in samples]


def _oracle(fixture, key, value, k_scale, v_scale):
    """Full FP64 softmax over native packed values, independent of GPU kernels."""
    key = key[..., :HEAD_SIZE].float().cpu()
    value = value[..., :HEAD_SIZE].float().cpu()
    if isinstance(k_scale, torch.Tensor):
        k_scale = k_scale.cpu()
        v_scale = v_scale.cpu()
    batch, query_heads, _ = fixture.query.shape
    kv_heads = query_heads // 4
    length = fixture.max_context_len
    pages = (length + BLOCK_SIZE - 1) // BLOCK_SIZE
    expected = torch.empty_like(fixture.expected_output)
    for sequence in range(batch):
        begin, end = sequence * pages, (sequence + 1) * pages
        for head in range(kv_heads):
            k = key[begin:end, :, head].reshape(-1, HEAD_SIZE)[:length].double()
            v = value[begin:end, :, head].reshape(-1, HEAD_SIZE)[:length].double()
            if isinstance(k_scale, torch.Tensor):
                ks = k_scale[begin:end, :, head].reshape(-1)[:length].double()[:, None]
                vs = v_scale[begin:end, :, head].reshape(-1)[:length].double()[:, None]
            else:
                ks, vs = k_scale, v_scale
            q = fixture.query[sequence, head * 4 : (head + 1) * 4].double()
            probabilities = torch.softmax(q @ (k * ks).T * SCALE, dim=-1)
            expected[sequence, head * 4 : (head + 1) * 4] = probabilities @ (v * vs)
    if not torch.isfinite(expected).all():
        raise AssertionError("Native quantized-cache oracle produced nonfinite values")
    return expected


def _check(output, expected, common_reference):
    actual = output.cpu().double()
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=3e-4)
    quantization = expected - common_reference
    return {
        "kernel_max_abs_error": (actual - expected).abs().max().item(),
        "kernel_atol": 1e-3,
        "kernel_rtol": 3e-4,
        "quantization_output_max_abs_error": quantization.abs().max().item(),
        "quantization_output_rmse": quantization.square().mean().sqrt().item(),
    }


def _make_fp8(fixture, baseline_arguments, use_tensor_cores=True):
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    from vllm import _custom_ops

    query, source_key, source_value, _, _, _, _ = baseline_arguments
    pages, _, kv_heads, _ = source_key.shape
    batch = query.size(0)
    pages_per_request = pages // batch
    # The local serving baseline resolves LBNHC, exposed to FlashInfer as NHD.
    arena = torch.empty(
        (pages, BLOCK_SIZE, kv_heads, 2 * HEAD_SIZE),
        dtype=torch.float8_e4m3fn,
        device=query.device,
    )
    key, value = arena.split(HEAD_SIZE, dim=-1)
    scales = torch.ones((), dtype=torch.float32, device=query.device)
    slots = torch.arange(pages * BLOCK_SIZE, dtype=torch.int64, device=query.device)
    _custom_ops.reshape_and_cache_flash(
        source_key.reshape(-1, kv_heads, HEAD_SIZE),
        source_value.reshape(-1, kv_heads, HEAD_SIZE),
        key,
        value,
        slots,
        "fp8_e4m3",
        scales,
        scales,
    )
    expected = _oracle(fixture, key, value, 1.0, 1.0)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=query.device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_tensor_cores=use_tensor_cores,
        use_cuda_graph=False,
        backend="auto",
    )
    indptr = torch.arange(batch + 1, dtype=torch.int32) * pages_per_request
    indices = torch.arange(pages, dtype=torch.int32, device=query.device)
    last_page_len = torch.full(
        (batch,), (fixture.max_context_len - 1) % BLOCK_SIZE + 1, dtype=torch.int32
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        query.size(1),
        kv_heads,
        HEAD_SIZE,
        BLOCK_SIZE,
        pos_encoding_mode="NONE",
        sm_scale=SCALE,
        window_left=-1,
        logits_soft_cap=0.0,
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.float16,
    )
    output = torch.empty_like(query)

    def run(q, k, v, out):
        wrapper.run(
            q,
            (k, v),
            q_scale=1.0,
            k_scale=1.0,
            v_scale=1.0,
            out=out,
        )

    arguments = query, key, value, output
    run(*arguments)
    return Arm(
        run,
        arguments,
        output,
        arena.numel() * arena.element_size(),
        {
            "backend": "FLASHINFER",
            "cache_dtype": "fp8_e4m3",
            "query_dtype": "float16",
            "output_dtype": "float16",
            "layout": "NHD (vLLM LBNHC), interleaved key and value content",
            "scale_handling": "per-tensor key and value scale 1.0, no calibration",
            "k_scale": 1.0,
            "v_scale": 1.0,
            "use_tensor_cores": use_tensor_cores,
            "flashinfer_resolved_backend": wrapper._backend,
            "serving_default": use_tensor_cores,
            "page_tokens": BLOCK_SIZE,
            "packing_api": "vllm._custom_ops.reshape_and_cache_flash",
        },
        _check(output, expected, fixture.expected_output),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--high-fraction", type=float, default=0.25)
    parser.add_argument("--repeat-ms", type=int, default=500)
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=["fp8"],
        choices=["fp8", "fp8_cuda"],
        help="FP8 arms to add; pass no value to compare only DiffKV and FA2",
    )
    parser.add_argument(
        "--output", type=Path, help="default: diffkv-v2/logs/latency-b*-<UTC>.json"
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = default_run_dir(f"latency-b{args.batch}") / "latency.json"
    if min(args.batch, args.kv_heads, args.repeat_ms, *args.contexts) < 1:
        parser.error("Batch, heads, repeat-ms and context lengths must be positive")
    if not 0 <= args.high_fraction <= 1:
        parser.error("high-fraction must lie in [0,1]")
    if len(set(args.candidates)) != len(args.candidates):
        parser.error("Duplicate FP8 candidates are not allowed")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    from diffkv_v2.ops import get_extension

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
        diffkv_check = check_fixture(extension, fixture)
        inputs = compact_device_inputs(fixture)
        workspace = allocate_workspaces(fixture)
        packed_args = attention_args(fixture, inputs, workspace)
        dense_args, length = make_baseline(fixture)

        def dense(q, k, v, out, starts, lengths, tables, length=length):
            flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                out=out,
                cu_seqlens_q=starts,
                max_seqlen_q=1,
                max_seqlen_k=length,
                seqused_k=lengths,
                block_table=tables,
                softmax_scale=SCALE,
                causal=True,
                fa_version=2,
            )

        extension.ops.sparse_paged_attention_gqa(*packed_args)
        dense(*dense_args)
        arms = {
            "diffkv_v2": Arm(
                extension.ops.sparse_paged_attention_gqa,
                packed_args,
                workspace["out"],
                inputs["kv_cache"].numel() * inputs["kv_cache"].element_size(),
                {"backend": "DIFFKV_V2", "importance_update_included": True},
                {
                    **diffkv_check,
                    **_check(
                        workspace["out"],
                        fixture.expected_output,
                        fixture.expected_output,
                    ),
                },
            ),
            "fa2_fp16": Arm(
                dense,
                dense_args,
                dense_args[3],
                dense_args[1].numel() * dense_args[1].element_size() * 2,
                {"backend": "FLASH_ATTN", "fa_version": 2, "cache_dtype": "float16"},
                _check(dense_args[3], fixture.expected_output, fixture.expected_output),
            ),
        }
        for candidate in args.candidates:
            arms[candidate] = _make_fp8(fixture, dense_args, candidate == "fp8")
        samples = {name: [] for name in arms}
        order = list(arms)
        for name in [*order, *reversed(order)]:
            arm = arms[name]
            samples[name].extend(measure(arm.function, arm.arguments, args.repeat_ms))
        medians = {name: statistics.median(values) for name, values in samples.items()}
        # Minimum bytes: one KV payload and metadata read per head plus score RMW.
        minimum_kv_bytes = {
            "diffkv_v2": args.batch
            * args.kv_heads
            * (high * (128 + 64 + 8 + 8) + low * (64 + 32 + 8 + 8)),
            "fa2_fp16": args.batch * args.kv_heads * context * HEAD_SIZE * 2 * 2,
        }
        row = {
            "batch": args.batch,
            "context": context,
            "kv_heads": args.kv_heads,
            "high_fraction": args.high_fraction,
            "high_tokens": high,
            "low_tokens": low,
            "cuda_graph": False,
            "arms": {
                name: {
                    "details": arm.details,
                    "correctness": arm.correctness,
                    "cache_object_bytes": arm.cache_bytes,
                    "cache_object_gbps": arm.cache_bytes / (medians[name] * 1000),
                    "median_us": medians[name],
                    "samples_us": samples[name],
                }
                for name, arm in arms.items()
            },
            "fa2_over_diffkv_ratio": medians["fa2_fp16"] / medians["diffkv_v2"],
            "minimum_kv_bytes": minimum_kv_bytes,
            "effective_minimum_kv_gbps": {
                name: value / (medians[name] * 1000)
                for name, value in minimum_kv_bytes.items()
            },
        }
        summary = {"context": context, "median_us": medians}
        if args.candidates:
            winner = min(args.candidates, key=medians.__getitem__)
            row["fastest_tested_fp8"] = winner
            row["fastest_fp8_over_diffkv_ratio"] = (
                medians[winner] / medians["diffkv_v2"]
            )
            summary["fastest_tested_fp8"] = winner
        rows.append(row)
        print(json.dumps(summary), flush=True)
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
        "flashinfer_extra_cudaflags": os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS"),
        "operation": "DiffKV main + reduce + score update versus paged FA2 "
        "and FlashInfer FP8 decode attention",
        "comparison_scope": (
            "identical retained KV before FP8 quantization, kernel only; "
            "no pruning or serving speedup claim"
        ),
        "oracle": "full FP64 softmax of actual native dequantized cache",
        "excluded": [
            "cache writer",
            "packing",
            "quantization",
            "cache view preparation",
            "planning",
            "allocation",
            "compilation",
            "input generation",
            "host transfer",
        ],
        "tuning": (
            "production v2 partition240, upstream FA2 and FlashInfer defaults, "
            "no tuning sweep"
        ),
        "selection_scope": "fastest measured FP8 candidate, no global SOTA claim",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"status": report["status"], "output": str(args.output)}, indent=2)
    )


if __name__ == "__main__":
    main()
