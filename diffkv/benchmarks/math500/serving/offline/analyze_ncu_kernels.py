# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Average Nsight Compute metrics per kernel over one profiled engine step.

Reads kernels_raw.csv (`ncu --import ... --csv --page raw --print-units base`)
from a run_offline_serving.sh DIFFKV_NCU_STEP run. Each kernel runs once per
layer in the step; the summary averages its instances.
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import regex as re

METRICS = (
    "gpu__time_duration.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sector_hit_rate.pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__maximum_warps_per_active_cycle_pct",
    "launch__registers_per_thread",
    "launch__grid_size",
    "launch__block_size",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__inst_executed.sum",
)
STALL = re.compile(r"smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio")
PIPE = re.compile(r"sm__inst_executed_pipe_(\w+)\.avg\.pct_of_peak_sustained_active")
NAME = re.compile(r"(?:^|\s)([\w:]+)\s*(?:<|\()")


def number(text):
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def short_name(name):
    match = NAME.search(name)
    return match[1].split("::")[-1] if match else name


def summarize(raw_path, outputs_path):
    lines = raw_path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if '"Kernel Name"' in line)
    rows = list(csv.reader(lines[start:]))
    header, units, data = rows[0], rows[1], rows[2:]
    columns = {name: index for index, name in enumerate(header)}
    selected = [m for m in METRICS if m in columns]
    selected += [m for m in header if STALL.fullmatch(m) or PIPE.fullmatch(m)]
    kernels = defaultdict(list)
    for row in data:
        if len(row) != len(header):
            continue
        kernels[short_name(row[columns["Kernel Name"]])].append(row)
    if not kernels:
        raise ValueError(f"No kernel rows in {raw_path}")
    summary = {}
    for name, instances in sorted(kernels.items()):
        means = {}
        for metric in selected:
            values = [number(row[columns[metric]]) for row in instances]
            values = [value for value in values if value is not None]
            if values:
                means[metric] = statistics.fmean(values)
        summary[name] = {
            "instances": len(instances),
            "units": {m: units[columns[m]] for m in selected if m in means},
            "mean": means,
        }
    profile = None
    if outputs_path is not None:
        profile = json.loads(outputs_path.read_text()).get("profile")
    return {
        "raw": str(raw_path),
        "definitions": {
            "mean": "Mean over the kernel's instances in the profiled step "
            "(one per attention layer)",
            "stall ratios": "smsp__average_warps_issue_stalled_<reason>_per_issue_"
            "active.ratio: warps stalled on the reason per issued instruction",
        },
        "profile": profile,
        "kernels": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--outputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    report = summarize(args.raw, args.outputs)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {
                    "instances": entry["instances"],
                    "duration_ns": entry["mean"].get("gpu__time_duration.sum"),
                    "dram_pct": entry["mean"].get(
                        "dram__throughput.avg.pct_of_peak_sustained_elapsed"
                    ),
                }
                for name, entry in report["kernels"].items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
