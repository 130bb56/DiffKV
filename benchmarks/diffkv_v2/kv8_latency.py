# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare native FP8 decode paths on the same retained KV as DiffKV v2.

Packing, quantization, planning and the independent FP64 oracle are untimed.
Kernel error uses each candidate's actual dequantized cache. Quantization error
uses the common unquantized retained KV reference and is reported separately.
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

from benchmarks.diffkv_v2.kernel_check import (
    HEAD_SIZE,
    SCALE,
    allocate_workspaces,
    attention_args,
    build_fixture,
    check_fixture,
    metadata,
)
from benchmarks.diffkv_v2.latency import (
    compact_device_inputs,
    make_baseline,
    measure,
    timing_backend,
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


def _view_spec(tensor):
    return tensor.dtype, tensor.shape, tensor.stride(), tensor.storage_offset()


def _restore_view(arena, spec):
    """Recreate native views on each rotated copy, preserving scale aliasing."""
    dtype, shape, strides, offset = spec
    return torch.as_strided(arena.view(dtype), shape, strides, offset)


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
        workspace, "NHD", use_tensor_cores=use_tensor_cores, backend="auto"
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
    key_spec = _view_spec(key)
    value_spec = _view_spec(value)
    output = torch.empty_like(query)

    def run(q, packed, out):
        wrapper.run(
            q,
            (_restore_view(packed, key_spec), _restore_view(packed, value_spec)),
            q_scale=1.0,
            k_scale=1.0,
            v_scale=1.0,
            out=out,
        )

    arguments = query, arena, output
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
            "layout": "NHD (vLLM LBNHC), interleaved K/V content",
            "scale_handling": "per-tensor K/V scale 1.0, no calibration",
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
        nargs="+",
        default=["fp8"],
        choices=["fp8", "fp8_cuda"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch, args.kv_heads, args.repeat_ms, *args.contexts) < 1:
        parser.error("Batch, heads, repeat-ms and context lengths must be positive")
    if not 0 <= args.high_fraction <= 1:
        parser.error("high-fraction must lie in [0,1]")
    if len(set(args.candidates)) != len(args.candidates):
        parser.error("Duplicate candidates are not allowed")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    from flashinfer.testing.utils import calculate_rotation_count

    from vllm.diffkv_v2.ops import get_extension
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    torch.set_num_threads(2)
    extension = get_extension()
    rows = []
    for context in args.contexts:
        high = round(context * args.high_fraction)
        fixture = build_fixture(
            ((high, context - high),) * args.batch,
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
        graph_calls = max(
            10,
            *(
                calculate_rotation_count(
                    [p for p in arm.arguments if isinstance(p, torch.Tensor)],
                    torch.device("cuda"),
                )
                for arm in arms.values()
            ),
        )
        samples = {name: [] for name in arms}
        order = list(arms)
        for name in [*order, *reversed(order)]:
            arm = arms[name]
            samples[name].extend(
                measure(arm.function, arm.arguments, args.repeat_ms, graph_calls)
            )
        medians = {name: statistics.median(values) for name, values in samples.items()}
        winner = min(args.candidates, key=medians.__getitem__)
        rows.append(
            {
                "batch": args.batch,
                "context": context,
                "kv_heads": args.kv_heads,
                "high_fraction": args.high_fraction,
                "graph_calls_per_replay": graph_calls,
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
                "fastest_tested_fp8": winner,
                "fastest_fp8_over_diffkv_ratio": medians[winner] / medians["diffkv_v2"],
            }
        )
        print(
            json.dumps(
                {
                    "context": context,
                    "median_us": medians,
                    "fastest_tested_fp8": winner,
                }
            ),
            flush=True,
        )
    report = {
        "status": "completed",
        "metadata": metadata(extension),
        "command": [sys.executable, *sys.argv],
        "timing_backend": timing_backend(),
        "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING"),
        "flashinfer_extra_cudaflags": os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS"),
        "comparison_scope": (
            "identical retained KV before FP8 quantization, kernel only"
        ),
        "oracle": "full FP64 softmax of actual native dequantized cache",
        "excluded": [
            "packing",
            "quantization",
            "planning",
            "allocation",
            "compilation",
            "input generation",
            "host transfer",
        ],
        "selection_scope": "fastest measured candidate, no global SOTA claim",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
