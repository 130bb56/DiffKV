# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trace eager DiffKV, FP16 and FP8 attention with Nsight Systems.

``profile`` runs the attention calls under ``nsys``. ``analyze`` attributes the
kernels in the exported SQLite trace to the CPU NVTX ranges.
"""

import argparse
import hashlib
import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import regex as re

PID_MASK = ~((1 << 24) - 1)
LABEL = re.compile(r"attention/(diffkv_v2|fa2_fp16|fp8)/(\d+)$")


def profile(args):
    import torch
    from attention.benchmark_latency import (
        _check,
        _make_fp8,
        compact_device_inputs,
        make_baseline,
    )
    from attention.correctness import (
        SCALE,
        allocate_workspaces,
        attention_args,
        build_fixture,
        check_fixture,
        metadata,
    )
    from diffkv_v2.ops import get_extension
    from flashinfer.testing.utils import get_l2_cache_size

    from vllm.vllm_flash_attn import flash_attn_varlen_func

    torch.set_num_threads(2)
    extension = get_extension()
    high = round(args.context * 0.25)
    fixture = build_fixture(
        ((high, args.context - high),) * args.batch,
        kv_heads=8,
        uniform_head_lengths=True,
    )
    correctness = {"diffkv_v2": check_fixture(extension, fixture)}
    inputs = compact_device_inputs(fixture)
    workspace = allocate_workspaces(fixture)
    packed_args = attention_args(fixture, inputs, workspace)
    dense_args, length = make_baseline(fixture)

    def dense(q, k, v, out, starts, lengths, tables):
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

    dense(*dense_args)
    correctness["fa2_fp16"] = _check(
        dense_args[3], fixture.expected_output, fixture.expected_output
    )
    fp8 = _make_fp8(fixture, dense_args)
    correctness["fp8"] = fp8.correctness
    arms = {
        "diffkv_v2": (extension.ops.sparse_paged_attention_gqa, packed_args),
        "fa2_fp16": (dense, dense_args),
        "fp8": (fp8.function, fp8.arguments),
    }
    flush = torch.empty(
        get_l2_cache_size(torch.device("cuda")) * 2,
        device="cuda",
        dtype=torch.uint8,
    )
    order = [*arms, *reversed(arms)]
    events = {
        arm: [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(args.iterations * 2)
        ]
        for arm in arms
    }
    for function, arguments in arms.values():
        for _ in range(25):
            flush.zero_()
            function(*arguments)
    for pairs in events.values():
        for start, end in pairs:
            start.record()
            end.record()
    torch.accelerator.synchronize()

    print("PROFILE START eager, CUDA Graph OFF", flush=True)
    counters = dict.fromkeys(arms, 0)
    torch.cuda.cudart().cudaProfilerStart()
    try:
        for arm in order:
            function, arguments = arms[arm]
            for _ in range(args.iterations):
                iteration = counters[arm]
                counters[arm] += 1
                start, end = events[arm][iteration]
                with torch.cuda.nvtx.range(f"flush/{arm}/{iteration}"):
                    flush.zero_()
                start.record()
                with torch.cuda.nvtx.range(f"attention/{arm}/{iteration}"):
                    function(*arguments)
                end.record()
            torch.accelerator.synchronize()
    finally:
        torch.accelerator.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

    samples = {
        arm: [start.elapsed_time(end) * 1000 for start, end in pairs]
        for arm, pairs in events.items()
    }
    report = {
        "status": "completed",
        "batch": args.batch,
        "context": args.context,
        "kv_heads": 8,
        "high_fraction": 0.25,
        "execution_mode": "eager",
        "cuda_graph": False,
        "cold_l2_cache": True,
        "l2_flush_bytes": flush.numel(),
        "iterations_per_arm": args.iterations * 2,
        "metadata": metadata(extension),
        "correctness": correctness,
        "event_interval_us": samples,
        "event_interval_p50_us": {
            arm: statistics.median(values) for arm, values in samples.items()
        },
        "scope": "Profiled direct attention calls. Preparation, warmup and L2 "
        "flush excluded from event intervals. Includes tracing and NVTX overhead; "
        "does not replace uninstrumented latency measurements.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "completed", "output": str(args.output)}), flush=True)


def union_ns(intervals):
    total, right = 0, None
    for start, end in sorted(intervals):
        total += max(0, end - max(start, right if right is not None else start))
        right = max(end, right if right is not None else end)
    return total


def analyze(path, event_path=None):
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row

    def rows(table, required):
        columns = {r[1] for r in connection.execute(f'PRAGMA table_info("{table}")')}
        if missing := set(required.split()) - columns:
            raise ValueError(f"{table}: missing columns {sorted(missing)}")
        return connection.execute(f'SELECT rowid AS _rowid, * FROM "{table}"')

    strings = {r["id"]: r["value"] for r in rows("StringIds", "id value")}
    ranges, flush_ranges = [], 0
    for row in rows("NVTX_EVENTS", "start end text textId globalTid"):
        name = row["text"] or strings.get(row["textId"], "")
        flush_ranges += int(name.startswith("flush/"))
        if match := LABEL.fullmatch(name):
            if row["end"] is None or row["end"] <= row["start"]:
                raise ValueError(f"Incomplete NVTX range: {name}")
            ranges.append((match[1], int(match[2]), row))
    if not ranges:
        raise ValueError("No attention/{arm}/{iteration} NVTX ranges")
    if len({(a, i) for a, i, _ in ranges}) != len(ranges):
        raise ValueError("Duplicate attention invocation labels")

    kernels = defaultdict(list)
    for row in rows(
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "start end correlationId globalPid demangledName deviceId streamId",
    ):
        kernels[row["globalPid"], row["correlationId"]].append(row)
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master")}
    launches = defaultdict(list)
    for kind in ("RUNTIME", "DRIVER"):
        table = f"CUPTI_ACTIVITY_KIND_{kind}"
        if table not in tables:
            continue
        for row in rows(table, "start end correlationId globalTid nameId"):
            name = strings[row["nameId"]].lower()
            if "launch" in name and ("kernel" in name or "graph" in name):
                launches[row["globalTid"]].append((kind, row))

    invocations, attributed_ids = [], set()
    for arm, iteration, scope in sorted(ranges):
        selected = [
            (kind, api)
            for kind, api in launches[scope["globalTid"]]
            if scope["start"] <= api["start"] and api["end"] <= scope["end"]
        ]
        if not selected:
            raise ValueError(f"No launch APIs: {arm}/{iteration}")
        selected.sort(key=lambda pair: pair[0] != "RUNTIME")
        attributed, matched_apis = {}, {}
        for kind, api in selected:
            if "graph" in strings[api["nameId"]].lower():
                raise ValueError("CUDA Graph launch found in eager attention range")
            key = (api["globalTid"] & PID_MASK, api["correlationId"])
            for kernel in kernels.get(key, []):
                if kernel["_rowid"] not in attributed:
                    attributed[kernel["_rowid"]] = (kernel, api)
                    matched_apis[kind, api["_rowid"]] = api
        if not attributed:
            raise ValueError(f"No correlated GPU kernels: {arm}/{iteration}")
        missing = [
            strings[api["nameId"]]
            for _, api in selected
            if not kernels.get((api["globalTid"] & PID_MASK, api["correlationId"]))
        ]
        if missing:
            raise ValueError(f"Unmatched launch APIs: {arm}/{iteration}: {missing}")
        attributed_ids.update(attributed)

        details = []
        for kernel, api in attributed.values():
            if kernel["end"] <= kernel["start"]:
                raise ValueError("Invalid GPU kernel duration")
            details.append(
                {
                    "name": strings[kernel["demangledName"]],
                    "start_ns": kernel["start"],
                    "end_ns": kernel["end"],
                    "duration_us": (kernel["end"] - kernel["start"]) / 1000,
                    "device": kernel["deviceId"],
                    "stream": kernel["streamId"],
                    "launch_api": strings[api["nameId"]],
                    "launch_api_duration_us": (api["end"] - api["start"]) / 1000,
                    "api_start_to_kernel_start_us": (kernel["start"] - api["start"])
                    / 1000,
                    "api_return_to_kernel_start_us": (kernel["start"] - api["end"])
                    / 1000,
                }
            )
        intervals = [(r["start_ns"], r["end_ns"]) for r in details]
        busy = union_ns(intervals)
        span = max(b for _, b in intervals) - min(a for a, _ in intervals)
        apis = list(matched_apis.values())
        invocations.append(
            {
                "arm": arm,
                "iteration": iteration,
                "kernel_count": len(details),
                "kernel_sum_us": sum(r["duration_us"] for r in details),
                "kernel_union_us": busy / 1000,
                "kernel_span_us": span / 1000,
                "internal_gpu_gap_us": (span - busy) / 1000,
                "host_dispatch_us": (scope["end"] - scope["start"]) / 1000,
                "host_before_first_launch_us": (
                    min(a["start"] for a in apis) - scope["start"]
                )
                / 1000,
                "launch_api_count": len(apis),
                "launch_api_union_us": union_ns([(a["start"], a["end"]) for a in apis])
                / 1000,
                "api_start_to_kernel_start_mean_us": statistics.mean(
                    r["api_start_to_kernel_start_us"] for r in details
                ),
                "api_return_to_kernel_start_mean_us": statistics.mean(
                    r["api_return_to_kernel_start_us"] for r in details
                ),
                "kernels": sorted(details, key=lambda r: r["start_ns"]),
            }
        )
    warnings = (
        [
            row["text"]
            for row in rows("DIAGNOSTIC_EVENT", "severity text")
            if row["severity"] > 1
        ]
        if "DIAGNOSTIC_EVENT" in tables
        else []
    )
    other_kernels = Counter(
        strings[kernel["demangledName"]]
        for group in kernels.values()
        for kernel in group
        if kernel["_rowid"] not in attributed_ids
    )
    connection.close()
    if event_path is not None:
        events = json.loads(event_path.read_text())["event_interval_us"]
        for row in invocations:
            values = events[row["arm"]]
            index = (
                row["iteration"] if isinstance(values, list) else str(row["iteration"])
            )
            row["cuda_event_interval_us"] = values[index]
            row["event_minus_kernel_union_us"] = values[index] - row["kernel_union_us"]

    arms = {}
    for arm in sorted({r["arm"] for r in invocations}):
        samples = [r for r in invocations if r["arm"] == arm]
        grouped = defaultdict(list)
        for row in samples:
            for kernel in row["kernels"]:
                grouped[kernel["name"]].append(kernel["duration_us"])
        arms[arm] = {
            "invocations": len(samples),
            "metrics": {
                key: {
                    "p50": statistics.median(row[key] for row in samples),
                    "mean": statistics.mean(row[key] for row in samples),
                }
                for key in samples[0]
                if key not in ("arm", "iteration", "kernels")
            },
            "kernels": [
                {
                    "name": name,
                    "count": len(values),
                    "total_us": sum(values),
                    "p50_us": statistics.median(values),
                    "mean_us_per_invocation": sum(values) / len(samples),
                }
                for name, values in sorted(grouped.items())
            ],
        }
    return {
        "input": str(path.resolve()),
        "input_sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
        "execution_mode": "eager",
        "events_input": str(event_path.resolve()) if event_path is not None else None,
        "validation": {
            "attention_ranges": len(ranges),
            "flush_ranges": flush_ranges,
            "attention_kernels": len(attributed_ids),
            "unmatched_attention_launches": 0,
            "other_kernels": dict(other_kernels),
            "profiler_warnings": warnings,
        },
        "limitations": [
            "Profiler instrumentation perturbs timing; diagnostic measurements only.",
            "CPU dispatch and GPU execution overlap; never add their durations.",
            "Launch-to-start includes API execution and GPU queue delay.",
            "API-return-to-kernel-start is signed; kernels can start before return.",
            "Internal GPU gap excludes idle time outside the kernel span.",
            "Event minus kernel union is residual time, not isolated CPU overhead.",
        ],
        "arms": arms,
        "invocations": invocations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    trace = commands.add_parser("profile", help="run attention under nsys")
    trace.add_argument("--batch", type=int, default=1)
    trace.add_argument("--context", type=int, default=512)
    trace.add_argument("--iterations", type=int, default=50)
    trace.add_argument(
        "--output",
        type=Path,
        help="default: logs/<UTC>-nsys-b<batch>-kv<context>/cuda_events.json",
    )
    attribute = commands.add_parser(
        "analyze", help="attribute kernels in an exported SQLite trace"
    )
    attribute.add_argument("--input", required=True, type=Path)
    attribute.add_argument(
        "--output",
        type=Path,
        help="default: kernel_attribution.json beside --input",
    )
    attribute.add_argument("--events", type=Path)
    args = parser.parse_args()
    if args.command == "analyze":
        if args.output is None:
            args.output = args.input.parent / "kernel_attribution.json"
        report = analyze(args.input, args.events)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Saved {args.output}")
        return
    if args.output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        name = f"{stamp}-nsys-b{args.batch}-kv{args.context}"
        args.output = (
            Path(__file__).resolve().parent / "logs" / name / "cuda_events.json"
        )
    if min(args.batch, args.context, args.iterations) < 1:
        trace.error("Batch, context and iterations must be positive")
    if args.output.exists():
        trace.error(f"Output already exists: {args.output}")
    profile(args)


if __name__ == "__main__":
    main()
