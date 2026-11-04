# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check GPU telemetry from record_telemetry.py.

wait: block until the GPU has cooled down, as origin/main
benchmarks/run_kernel_e2e_grid.py cool_before_measurement does: at least
--min-seconds have passed, the SM clock is at --clock MHz, the temperature is at
most --max-temp, and no thermal or power limit is active. Exit 1 after
--timeout seconds.
check: append the verdict for one measured interval to --output. The interval
is valid when every sample in it runs at --clock MHz, no thermal, power,
hardware slowdown or sync boost limit is active, and no gap between samples
(including the nearest sample outside each end) exceeds MAX_GAP_S. Exit 0 if
valid, 1 if not.
watch: follow the telemetry from --start while the command runs. Write
--throttle-marker and exit as soon as a sample shows one of those limits or an
SM clock other than --clock; write --stuck-marker and exit once GPU utilization
has been 0% for --idle-seconds.
"""

import argparse
import contextlib
import json
import signal
import sys
import time
from pathlib import Path

# NVML clock event reason bits; the same set origin/main
# benchmarks/monitor_kernel_e2e_gpu.py treats as invalid.
LIMITS = {
    "sw_power_cap": 0x4,
    "hw_slowdown": 0x8,
    "sync_boost": 0x10,
    "sw_thermal_slowdown": 0x20,
    "hw_thermal_slowdown": 0x40,
    "hw_power_brake_slowdown": 0x80,
}
LIMIT_BITS = sum(LIMITS.values())
COOLING_LIMITS = 0x4 | 0x20 | 0x40 | 0x80
MAX_GAP_S = 1.0


def read_samples(path):
    samples = []
    for line in Path(path).read_text().splitlines():
        with contextlib.suppress(json.JSONDecodeError):
            samples.append(json.loads(line))
    return samples


def wait(args):
    started = time.time()
    last_report = 0.0
    while True:
        samples = read_samples(args.telemetry)
        latest = samples[-1] if samples else {}
        now = time.time()
        ready = (
            now - started >= args.min_seconds
            and now - latest.get("epoch", 0) < 3
            and "error" not in latest
            and latest.get("sm_mhz") == args.clock
            and latest.get("temperature_c", 999) <= args.max_temp
            and not latest.get("reasons", 0) & COOLING_LIMITS
        )
        if ready:
            print(
                f"cooled: {latest['temperature_c']} C after {now - started:.0f} s",
                flush=True,
            )
            return 0
        if now - started > args.timeout:
            print(
                f"not cooled to {args.max_temp} C within {args.timeout} s: {latest}",
                flush=True,
            )
            return 1
        if now - last_report >= 60:
            print(f"cooling: {latest.get('temperature_c')} C", flush=True)
            last_report = now
        time.sleep(2)


def check(args):
    samples = sorted(read_samples(args.telemetry), key=lambda s: s["epoch"])
    inside = [s for s in samples if args.start <= s["epoch"] <= args.end]
    before = [s for s in samples if s["epoch"] < args.start][-1:]
    after = [s for s in samples if s["epoch"] > args.end][:1]
    span = before + inside + after
    issues = []
    if not inside:
        issues.append("no sample in the interval")
    if not before or not after:
        issues.append("no sample before or after the interval")
    gaps = [b["epoch"] - a["epoch"] for a, b in zip(span, span[1:])]
    if gaps and max(gaps) > MAX_GAP_S:
        issues.append(f"sample gap {max(gaps):.2f} s > {MAX_GAP_S} s")
    errors = [s for s in inside if "error" in s]
    if errors:
        issues.append(f"{len(errors)} failed queries")
    ok = [s for s in inside if "error" not in s]
    if len({s["uuid"] for s in ok}) > 1:
        issues.append("GPU changed during the interval")
    limits = {
        name: sum(bool(s["reasons"] & bit) for s in ok) for name, bit in LIMITS.items()
    }
    for name, count in limits.items():
        if count:
            issues.append(f"{name} in {count} samples")
    off_clock = [s["sm_mhz"] for s in ok if s["sm_mhz"] != args.clock]
    if off_clock:
        issues.append(
            f"SM clock not {args.clock} MHz in {len(off_clock)} "
            f"samples (min {min(off_clock)}, max {max(off_clock)})"
        )
    verdict = {
        "valid": not issues,
        "issues": issues,
        "start_epoch": args.start,
        "end_epoch": args.end,
        "duration_s": args.end - args.start,
        "clock_mhz": args.clock,
        "samples": len(inside),
        "max_gap_s": max(gaps, default=None),
        "temperature_max_c": max((s["temperature_c"] for s in ok), default=None),
        "power_max_w": max((s["power_w"] for s in ok), default=None),
        "limit_samples": limits,
    }
    output = Path(args.output)
    attempts = json.loads(output.read_text())["attempts"] if output.exists() else []
    attempts.append(verdict)
    output.write_text(
        json.dumps({"valid": verdict["valid"], "attempts": attempts}, indent=2) + "\n"
    )
    print(json.dumps(verdict), flush=True)
    return 0 if verdict["valid"] else 1


def watch(args):
    last_busy = args.start
    pending = ""
    with open(args.telemetry) as telemetry:
        while True:
            pending += telemetry.read()
            *lines, pending = pending.split("\n")
            for line in lines:
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if sample["epoch"] < args.start or "error" in sample:
                    continue
                if sample["reasons"] & LIMIT_BITS or sample["sm_mhz"] != args.clock:
                    Path(args.throttle_marker).write_text(json.dumps(sample) + "\n")
                    print(f"throttled or off clock: {json.dumps(sample)}", flush=True)
                    return 0
                if sample["utilization_pct"] > 0:
                    last_busy = sample["epoch"]
                elif sample["epoch"] - last_busy >= args.idle_seconds:
                    note = {"idle_since_epoch": last_busy, "seconds": args.idle_seconds}
                    Path(args.stuck_marker).write_text(json.dumps(note) + "\n")
                    print(f"GPU idle for {args.idle_seconds:.0f} s", flush=True)
                    return 0
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    waiting = commands.add_parser("wait")
    waiting.add_argument("--telemetry", required=True)
    waiting.add_argument("--clock", type=int, required=True)
    waiting.add_argument("--min-seconds", type=float, required=True)
    waiting.add_argument("--max-temp", type=float, required=True)
    waiting.add_argument("--timeout", type=float, required=True)
    checking = commands.add_parser("check")
    checking.add_argument("--telemetry", required=True)
    checking.add_argument("--start", type=float, required=True)
    checking.add_argument("--end", type=float, required=True)
    checking.add_argument("--clock", type=int, required=True)
    checking.add_argument("--output", required=True)
    watching = commands.add_parser("watch")
    watching.add_argument("--telemetry", required=True)
    watching.add_argument("--start", type=float, required=True)
    watching.add_argument("--clock", type=int, required=True)
    watching.add_argument("--idle-seconds", type=float, required=True)
    watching.add_argument("--throttle-marker", required=True)
    watching.add_argument("--stuck-marker", required=True)
    args = parser.parse_args()
    # In a container this runs as PID 1, which ignores SIGTERM without a handler.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    sys.exit({"wait": wait, "check": check, "watch": watch}[args.command](args))


if __name__ == "__main__":
    main()
