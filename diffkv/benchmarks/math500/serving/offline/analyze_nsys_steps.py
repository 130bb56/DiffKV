# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split nsys-traced engine steps into GPU kernel time, GPU idle time and syncs.

Reads the trace.sqlite and outputs.json of a run_offline_serving.sh run with
DIFFKV_NSYS_STEPS. Each NVTX range step/<n> covers one LLMEngine.step() on the
CPU; the GPU kernels, copies and memsets launched inside the range belong to
that step.
"""

import argparse
import bisect
import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import regex as re

PID_MASK = ~((1 << 24) - 1)
STEP = re.compile(r"step/(\d+)")
COPY_KINDS = {1: "htod", 2: "dtoh", 8: "dtod"}
# Matched against the kernel's short name; the first match wins.
CATEGORIES = (
    ("diffkv_attention_main", ("register_gqa_attention_kernel",)),
    ("diffkv_attention_reduce", ("register_gqa_reduce_kernel",)),
    ("diffkv_cache_update_decode", ("compress_and_append_cache_decode_phase",)),
    ("diffkv_cache_update_prompt", ("compress_and_append_cache",)),
    (
        "diffkv_page_management",
        ("allocate_seqs", "append_seqs", "free_prompt", "free_seqs"),
    ),
    (
        "flashinfer_attention",
        ("BatchPrefillWithPagedKVCache", "BatchDecodeWithPagedKVCache", "MergeStates"),
    ),
    ("flash_attention", ("flash_fwd",)),
    ("kv_cache_write", ("reshape_and_cache",)),
    ("rms_norm", ("rms_norm",)),
    ("rotary_embedding", ("rotary_embedding",)),
    ("activation", ("act_and_mul",)),
    ("index_gather_scatter", ("index", "gather", "scatter")),
)


def category(short_name, demangled):
    lowered = short_name.lower()
    if "gemm" in lowered or (
        short_name in ("Kernel", "Kernel2", "device_kernel")
        and "gemm" in demangled.lower()
    ):
        return "gemm"
    for name, patterns in CATEGORIES:
        if any(pattern.lower() in lowered for pattern in patterns):
            return name
    return "other"


def union(intervals):
    total, right = 0, None
    for start, end in sorted(intervals):
        if right is None or start > right:
            total += end - start
            right = end
        elif end > right:
            total += end - right
            right = end
    return total


def summary(values):
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def analyze(trace, outputs_path):
    connection = sqlite3.connect(trace.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master")}

    def rows(table):
        if table not in tables:
            return []
        return connection.execute(
            f'SELECT rowid AS _rowid, * FROM "{table}"'
        ).fetchall()

    def columns(table):
        return {r[1] for r in connection.execute(f'PRAGMA table_info("{table}")')}

    has_short_name = "shortName" in columns("CUPTI_ACTIVITY_KIND_KERNEL")

    strings = {r["id"]: r["value"] for r in rows("StringIds")}
    ranges = []
    for row in rows("NVTX_EVENTS"):
        text = row["text"] or strings.get(row["textId"], "")
        if match := STEP.fullmatch(text or ""):
            if row["end"] is None or row["end"] <= row["start"]:
                raise ValueError(f"Incomplete NVTX range: {text}")
            ranges.append((int(match[1]), row["start"], row["end"], row["globalTid"]))
    if not ranges:
        raise ValueError("No step/<n> NVTX ranges in the trace")
    ranges.sort()

    activities = defaultdict(list)
    for table, kind in (
        ("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
        ("CUPTI_ACTIVITY_KIND_MEMSET", "memset"),
    ):
        by_process = "globalPid" in columns(table)
        for row in rows(table):
            pid = row["globalPid"] if by_process else row["globalTid"] & PID_MASK
            activities[pid, row["correlationId"]].append((kind, row))

    apis = []
    for table, layer in (
        ("CUPTI_ACTIVITY_KIND_RUNTIME", "runtime"),
        ("CUPTI_ACTIVITY_KIND_DRIVER", "driver"),
    ):
        for row in rows(table):
            apis.append(
                (
                    row["start"],
                    row["end"],
                    row["globalTid"],
                    row["correlationId"],
                    strings[row["nameId"]],
                    layer,
                )
            )
    apis.sort()
    api_starts = [api[0] for api in apis]

    per_step_outputs = {}
    if outputs_path is not None:
        profile = json.loads(outputs_path.read_text()).get("profile") or {}
        per_step_outputs = {row["step"]: row for row in profile.get("per_step", [])}

    steps, kernel_totals = [], defaultdict(lambda: [0, 0.0, ""])
    for step, start, end, tid in ranges:
        pid = tid & PID_MASK
        first = bisect.bisect_left(api_starts, start)
        last = bisect.bisect_right(api_starts, end)
        categories = defaultdict(float)
        copies = Counter()
        copy_ms = defaultdict(float)
        syncs = Counter()
        sync_ms = defaultdict(float)
        kernels = 0
        intervals, seen = [], set()
        for api_start, api_end, api_tid, correlation, name, layer in apis[first:last]:
            if api_tid & PID_MASK != pid:
                continue
            if "Synchronize" in name:
                syncs[layer] += 1
                sync_ms[layer] += (api_end - api_start) / 1e6
            for kind, activity in activities.get((pid, correlation), ()):
                if (kind, activity["_rowid"]) in seen:
                    continue
                seen.add((kind, activity["_rowid"]))
                duration = (activity["end"] - activity["start"]) / 1e6
                intervals.append((activity["start"], activity["end"]))
                if kind == "kernel":
                    kernels += 1
                    demangled = strings.get(activity["demangledName"], "")
                    short = (
                        strings.get(activity["shortName"], "")
                        if has_short_name
                        else demangled.split("(")[0].split("<")[0].split()[-1]
                    )
                    group = category(short, demangled)
                    categories[group] += duration
                    total = kernel_totals[short]
                    total[0] += 1
                    total[1] += duration
                    total[2] = group
                elif kind == "memcpy":
                    direction = COPY_KINDS.get(activity["copyKind"], "other")
                    copies[direction] += 1
                    copy_ms[direction] += duration
                else:
                    copies["memset"] += 1
                    copy_ms["memset"] += duration
        if not intervals:
            raise ValueError(f"No GPU activity launched in step/{step}")
        clipped = [(max(a, start), min(b, end)) for a, b in intervals if b > start]
        busy = union([(a, b) for a, b in clipped if b > a])
        wall = end - start
        gpu_first = min(a for a, _ in intervals)
        gpu_last = max(b for _, b in intervals)
        steps.append(
            {
                "step": step,
                **{
                    key: value
                    for key, value in per_step_outputs.get(step, {}).items()
                    if key != "step"
                },
                "wall_ms": wall / 1e6,
                "gpu_busy_ms": busy / 1e6,
                "gpu_idle_ms": (wall - busy) / 1e6,
                "gpu_idle_before_first_activity_ms": max(0, gpu_first - start) / 1e6,
                "gpu_idle_after_last_activity_ms": max(0, end - gpu_last) / 1e6,
                "gpu_work_after_range_ms": max(0, gpu_last - end) / 1e6,
                "kernels": kernels,
                "sync_api_calls": sum(syncs.values()),
                "sync_api_ms": sum(sync_ms.values()),
                "sync_api_calls_by_layer": dict(syncs),
                "copies": dict(copies),
                "copy_ms": dict(copy_ms),
                "kernel_ms": dict(sorted(categories.items())),
            }
        )

    names = sorted({name for row in steps for name in row["kernel_ms"]})
    numeric = (
        "wall_ms",
        "gpu_busy_ms",
        "gpu_idle_ms",
        "gpu_idle_before_first_activity_ms",
        "gpu_idle_after_last_activity_ms",
        "kernels",
        "sync_api_calls",
        "sync_api_ms",
    )
    aggregate = {key: summary([row[key] for row in steps]) for key in numeric}
    aggregate["kernel_ms"] = {
        name: summary([row["kernel_ms"].get(name, 0.0) for row in steps])
        for name in names
    }
    top = sorted(kernel_totals.items(), key=lambda item: -item[1][1])[:30]
    return {
        "trace": str(trace),
        "steps": len(steps),
        "first_step": steps[0]["step"],
        "definitions": {
            "wall_ms": "NVTX step/<n> range on the CPU, one LLMEngine.step()",
            "gpu_busy_ms": "Union of kernel, copy and memset intervals launched in "
            "the step, clipped to the range",
            "gpu_idle_ms": "wall_ms - gpu_busy_ms",
            "sync_api_calls": "CUDA runtime or driver calls with Synchronize in "
            "the name, launched in the step",
            "kernel_ms": "Sum of kernel durations per category (kernel short name)",
        },
        "per_step_mean": aggregate,
        "top_kernels": [
            {"name": name, "category": group, "calls": calls, "total_ms": total}
            for name, (calls, total, group) in top
        ],
        "per_step": steps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--outputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    report = analyze(args.trace, args.outputs)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    mean = report["per_step_mean"]
    print(
        json.dumps(
            {
                "steps": report["steps"],
                "wall_ms": mean["wall_ms"]["mean"],
                "gpu_busy_ms": mean["gpu_busy_ms"]["mean"],
                "gpu_idle_ms": mean["gpu_idle_ms"]["mean"],
                "sync_api_calls": mean["sync_api_calls"]["mean"],
                "kernel_ms": {k: v["mean"] for k, v in mean["kernel_ms"].items()},
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
