#!/usr/bin/env python3
"""Read-only GPU1 telemetry, written as flushed JSONL at a one-second cadence.

Uses nvidia-smi only. It does not create a CUDA context or change GPU settings.
Stop with SIGTERM, SIGINT, or the presence of --stop-file.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import signal
import statistics
import subprocess
import threading
import time
import uuid


SCHEMA_VERSION = 1
CORE_FIELDS = {
    "gpu_index": ("index",),
    "gpu_uuid": ("uuid",),
    "gpu_name": ("name",),
    "sm_clock_mhz": ("clocks.current.sm", "clocks.sm"),
    "memory_clock_mhz": ("clocks.current.memory", "clocks.mem"),
    "temperature_c": ("temperature.gpu",),
    "power_draw_w": ("power.draw",),
    "power_limit_w": ("power.limit",),
    "gpu_utilization_pct": ("utilization.gpu",),
    "pstate": ("pstate",),
}
REASON_SUFFIXES = {
    "active_mask": "active",
    "supported_mask": "supported",
    "gpu_idle": "gpu_idle",
    "apps_clocks": "applications_clocks_setting",
    "sw_power_cap": "sw_power_cap",
    "hw_slowdown": "hw_slowdown",
    "hw_thermal": "hw_thermal_slowdown",
    "hw_power_brake": "hw_power_brake_slowdown",
    "sw_thermal": "sw_thermal_slowdown",
    "sync_boost": "sync_boost",
}
FIELD_CANDIDATES = dict(CORE_FIELDS)
FIELD_CANDIDATES.update({
    "throttle_" + key: tuple(prefix + "." + suffix for prefix in
        ("clocks_throttle_reasons", "clocks_event_reasons"))
    for key, suffix in REASON_SUFFIXES.items()
})


def utc(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="microseconds")


def query_gpu(fields, timeout=5.0):
    command = ["nvidia-smi", "-i", "1", "--query-gpu=" + ",".join(fields),
               "--format=csv,noheader,nounits"]
    result = subprocess.run(command, text=True, capture_output=True,
                            timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"nvidia-smi exit {result.returncode}: "
                           + (result.stderr or result.stdout).strip()[:1500])
    rows = list(csv.reader(line for line in result.stdout.splitlines() if line.strip()))
    if len(rows) != 1 or len(rows[0]) != len(fields):
        raise RuntimeError("Expected one GPU1 CSV row with all requested fields: "
                           + result.stdout[:1500])
    return dict(zip(fields, (value.strip() for value in rows[0])))


def discover_fields(timeout=5.0):
    """Avoid letting a driver-unsupported property break the complete query."""
    errors = []
    try:
        result = subprocess.run(["nvidia-smi", "--help-query-gpu"], text=True,
                                capture_output=True, timeout=timeout, check=True)
        advertised = set(re.findall(r'"([a-z][a-z0-9._]*)"', result.stdout))
    except (OSError, subprocess.SubprocessError) as error:
        advertised = set()
        errors.append(str(error))
    selected, unavailable = {}, {}
    for key, candidates in FIELD_CANDIDATES.items():
        name = next((name for name in candidates if name in advertised), None)
        if name is None and not advertised:
            name = candidates[0]
        if name:
            selected[key] = name
        else:
            unavailable[key] = "Not advertised by nvidia-smi --help-query-gpu"
    try:
        query_gpu(list(selected.values()), timeout)
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        errors.append(str(error))
        # This path is startup-only. Normal sampling uses one subprocess.
        for key, name in list(selected.items()):
            try:
                query_gpu([name], timeout)
            except (OSError, subprocess.SubprocessError, RuntimeError) as field_error:
                unavailable[key] = str(field_error)
                del selected[key]
    return selected, unavailable, errors


def parse_value(key, value):
    if value is None or value.strip().lower() in (
            "n/a", "[n/a]", "not supported", "[not supported]", "unknown", "-"):
        return None
    if key in ("gpu_uuid", "gpu_name", "pstate"):
        return value
    if key.startswith("throttle_") and not key.endswith("_mask"):
        return {"active": True, "not active": False}.get(value.lower())
    try:
        if key.endswith("_mask"):
            return int(value, 0)
        number = float(value)
        if not math.isfinite(number):
            return None
        return int(number) if number.is_integer() else number
    except ValueError:
        return None


def sample_gpu(fields, unavailable=None, timeout=5.0):
    start, perf_start = time.time(), time.perf_counter()
    error, raw = None, {}
    try:
        if not fields:
            raise RuntimeError("No queryable nvidia-smi fields")
        raw = query_gpu(list(fields.values()), timeout)
    except (OSError, subprocess.SubprocessError, RuntimeError) as query_error:
        error = str(query_error)
    end, perf_end = time.time(), time.perf_counter()
    epoch = (start + end) / 2
    data = {key: parse_value(key, raw.get(fields.get(key))) for key in FIELD_CANDIDATES}
    missing = {key: (unavailable or {}).get(key, "Missing or unsupported query value")
               for key, value in data.items() if value is None}
    if data["gpu_index"] is not None and data["gpu_index"] != 1:
        error = f"Wrong physical GPU index: {data['gpu_index']}"
    return {
        "schema_version": SCHEMA_VERSION, "record_type": "sample",
        "time_epoch": epoch, "time_utc": utc(epoch),
        "perf_counter_s": (perf_start + perf_end) / 2,
        "query_epoch_start": start, "query_epoch_end": end,
        "query_duration_s": perf_end - perf_start,
        "status": "error" if error else "partial" if missing else "ok",
        "data": data, "raw_values": raw, "missing_fields": missing,
        "error": error,
    }


def read_records(path):
    records = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                records.append({"record_type": "parse_error", "line": number,
                                "error": str(error)})
    return records


def append_sample_index(path):
    """Validate an existing JSONL before appending; never repair or truncate it."""
    path = Path(path)
    with path.open("rb") as stream:
        stream.seek(0, 2)
        if not stream.tell():
            raise ValueError("Cannot append to an empty telemetry file")
        stream.seek(-1, 2)
        if stream.read(1) != b"\n":
            raise ValueError("Existing telemetry must end with a newline")
    next_index = 0
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Empty telemetry record at line {number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid telemetry JSON at line {number}: {error}") from error
            if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"Invalid telemetry schema at line {number}")
            if record.get("record_type") not in ("metadata", "sample", "stopped"):
                raise ValueError(f"Invalid telemetry record type at line {number}")
            if record["record_type"] == "sample":
                index = record.get("sample_index")
                if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                    raise ValueError(f"Invalid sample_index at line {number}")
                next_index = max(next_index, index + 1)
    return next_index


# NVML clock-event bit definitions, verified against the local CUDA nvml.h.
REASON_BITS = {
    "throttle_gpu_idle": 0x001,
    "throttle_apps_clocks": 0x002,
    "throttle_sw_power_cap": 0x004,
    "throttle_hw_slowdown": 0x008,
    "throttle_sync_boost": 0x010,
    "throttle_sw_thermal": 0x020,
    "throttle_hw_thermal": 0x040,
    "throttle_hw_power_brake": 0x080,
    "throttle_display_clocks": 0x100,
}


def numeric_summary(values):
    values = [value for value in values if isinstance(value, (int, float))
              and not isinstance(value, bool) and math.isfinite(value)]
    return ({"min": min(values), "mean": statistics.mean(values), "max": max(values)}
            if values else {"min": None, "mean": None, "max": None})


def analyze_interval(records, measured_epoch_start, measured_epoch_end,
                     expected_sm_mhz=1395, interval_s=1.0,
                     boundary_nearest=True, clock_tolerance_mhz=0.0):
    """Audit point samples inside one measured_generation wall-clock range.

    Adjacent outside samples support the cadence check and are reported
    separately. Their clocks/flags are never attributed to measured work.
    ``valid`` means complete telemetry with no unexpected limiter or SM-clock
    deviation observed. It is independent of workload correctness. A clear
    result describes sampled instants and cannot exclude sub-second events.
    """
    start, end = float(measured_epoch_start), float(measured_epoch_end)
    if not all(math.isfinite(value) for value in
               (start, end, interval_s, expected_sm_mhz, clock_tolerance_mhz)):
        raise ValueError("Measurement timestamps and interval must be finite")
    if end <= start or interval_s <= 0 or clock_tolerance_mhz < 0:
        raise ValueError("Expected a positive measured interval and sample cadence")
    records = list(records)
    samples = [row for row in records if row.get("record_type") == "sample"
               and isinstance(row.get("time_epoch"), (int, float))
               and math.isfinite(row["time_epoch"])]
    samples.sort(key=lambda row: row["time_epoch"])
    inside = [row for row in samples if start <= row["time_epoch"] <= end]
    boundary = []
    if boundary_nearest:
        before = [row for row in samples if row["time_epoch"] < start]
        after = [row for row in samples if row["time_epoch"] > end]
        if before and start - before[-1]["time_epoch"] <= interval_s * 1.5:
            boundary.append(before[-1])
        if after and after[0]["time_epoch"] - end <= interval_s * 1.5:
            boundary.append(after[0])
    supported_rows = sorted(inside + boundary, key=lambda row: row["time_epoch"])
    issues = []
    if not inside:
        issues.append("No sample midpoint inside the measured interval")
    parse_errors = [row for row in records if row.get("record_type") == "parse_error"]
    if parse_errors:
        issues.append(f"JSONL contains {len(parse_errors)} malformed record(s)")
    error_rows = [row for row in supported_rows if row.get("status") == "error"]
    partial_rows = [row for row in supported_rows if row.get("status") == "partial"]
    if error_rows:
        issues.append(f"{len(error_rows)} query error sample(s) in the interval or boundary support")
    if partial_rows:
        issues.append(f"{len(partial_rows)} partial sample(s) in the interval or boundary support")
    missing = {}
    for row in supported_rows:
        missing_names = set(row.get("missing_fields", {}))
        missing_names.update(key for key in FIELD_CANDIDATES
                             if row.get("data", {}).get(key) is None)
        for field in missing_names:
            missing[field] = missing.get(field, 0) + 1
        if row.get("schema_version") != SCHEMA_VERSION:
            issues.append("Unexpected sample schema version")
        if not isinstance(row.get("query_duration_s"), (int, float)):
            issues.append("Sample query duration is missing")
        elif row["query_duration_s"] > interval_s * 1.5:
            issues.append("GPU query duration exceeds 1.5 sampling intervals")
    if missing:
        issues.append("Missing or unsupported telemetry fields: " + ", ".join(sorted(missing)))
    gaps = [b["time_epoch"] - a["time_epoch"]
            for a, b in zip(supported_rows, supported_rows[1:])]
    if gaps and max(gaps) > interval_s * 1.5:
        issues.append(f"Sample gap {max(gaps):.6f}s exceeds 1.5 sampling intervals")
    for label, epoch in (("start", start), ("end", end)):
        if not supported_rows or min(abs(row["time_epoch"] - epoch)
                                     for row in supported_rows) > interval_s * 1.5:
            issues.append(f"No nearby sample supports the measurement {label}")
    if any(row.get("data", {}).get("gpu_index") != 1 for row in supported_rows):
        issues.append("Missing or wrong physical GPU index")
    uuids = {row.get("data", {}).get("gpu_uuid") for row in supported_rows}
    if None in uuids or len(uuids) > 1:
        issues.append("GPU UUID missing or changed across the measured interval")
    for a, b in zip(supported_rows, supported_rows[1:]):
        if all(isinstance(row.get("perf_counter_s"), (int, float)) for row in (a, b)):
            epoch_delta = b["time_epoch"] - a["time_epoch"]
            perf_delta = b["perf_counter_s"] - a["perf_counter_s"]
            if perf_delta <= 0 or abs(epoch_delta - perf_delta) > max(.2, interval_s * .5):
                issues.append("Wall-clock timestamps disagree with monotonic sample timing")
                break

    # Missing individual flag fields remain a visible quality issue even if
    # the active/supported masks allow deriving their sampled state.
    reason_counts = {}
    for name, bit in REASON_BITS.items():
        active = absent = 0
        for row in inside:
            data = row.get("data", {})
            value = data.get(name)
            if value is None:
                mask, supported = data.get("throttle_active_mask"), data.get("throttle_supported_mask")
                if isinstance(mask, int) and isinstance(supported, int):
                    value = bool(mask & bit) if supported & bit else False
            active += int(value is True)
            absent += int(value is None)
        reason_counts[name] = {"active_samples": active, "missing_samples": absent}
    unknown_active = sum(bool(row.get("data", {}).get("throttle_active_mask", 0)
                              and row["data"]["throttle_active_mask"] & ~0x1ff)
                         for row in inside)
    categories = {
        "thermal_flags": ("throttle_hw_thermal", "throttle_sw_thermal"),
        "power_flags": ("throttle_sw_power_cap", "throttle_hw_power_brake"),
        "other_flags": ("throttle_hw_slowdown", "throttle_sync_boost"),
        "configuration_flags": ("throttle_apps_clocks", "throttle_display_clocks"),
    }
    thermal_seen = any(reason_counts[key]["active_samples"] for key in categories["thermal_flags"])
    power_seen = any(reason_counts[key]["active_samples"] for key in categories["power_flags"])
    other_seen = any(reason_counts[key]["active_samples"] for key in categories["other_flags"])
    deviations = [{"time_epoch": row["time_epoch"], "time_utc": row.get("time_utc"),
                   "sample_index": row.get("sample_index"),
                   "sm_clock_mhz": row["data"]["sm_clock_mhz"]}
                  for row in inside if isinstance(row.get("data", {}).get("sm_clock_mhz"), (int, float))
                  and abs(row["data"]["sm_clock_mhz"] - expected_sm_mhz) > clock_tolerance_mhz]
    complete = not issues
    observed = bool(thermal_seen or power_seen or other_seen or unknown_active or deviations)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "throttling_or_clock_deviation_observed" if observed else "ok" if complete else "incomplete",
        "valid": complete and not observed, "telemetry_complete": complete,
        "measured_epoch_start": start, "measured_epoch_end": end,
        "duration_s": end - start, "interval_s": interval_s,
        "sample_count": len(inside), "boundary_sample_count": len(boundary),
        "boundary_samples": [{"time_epoch": row["time_epoch"],
                              "sample_index": row.get("sample_index"),
                              "status": row.get("status"),
                              "relation": "before" if row["time_epoch"] < start else "after"}
                             for row in boundary],
        "max_sampling_gap_s": max(gaps) if gaps else None,
        "max_query_duration_s": max((row.get("query_duration_s", 0) for row in supported_rows), default=None),
        "query_error_samples": len(error_rows), "partial_samples": len(partial_rows),
        "missing_fields": missing,
        "errors": issues,
        "query_errors": [row.get("error") for row in error_rows],
        "gpu_uuid": next(iter(uuids)) if len(uuids) == 1 else None,
        "clock": {**numeric_summary([row.get("data", {}).get("sm_clock_mhz") for row in inside]),
                  "expected_sm_mhz": expected_sm_mhz, "tolerance_mhz": clock_tolerance_mhz,
                  "deviation_count": len(deviations), "deviations": deviations},
        "gpu_idle_samples": reason_counts["throttle_gpu_idle"]["active_samples"],
        "unknown_active_mask_samples": unknown_active,
        "no_thermal_throttling_observed": False if thermal_seen else True if complete else None,
        "no_power_limiting_observed": False if power_seen else True if complete else None,
        "no_unexpected_throttling_observed": False if observed else True if complete else None,
        "sampling_limits": "Point samples cannot exclude unsampled transient events. Outside boundary samples are not treated as measured work. GPU idle and applications/display clock settings are reported separately.",
    }
    for category, names in categories.items():
        result[category] = {name: reason_counts[name] for name in names}
    result["data_stats"] = {key: numeric_summary([row.get("data", {}).get(key) for row in inside])
                            for key in ("memory_clock_mhz", "temperature_c", "power_draw_w",
                                        "power_limit_w", "gpu_utilization_pct")}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--query-timeout", type=float, default=5.0)
    parser.add_argument("--append", action="store_true",
                        help="Validate and append a new session to an existing telemetry JSONL")
    args = parser.parse_args(argv)
    if not all(math.isfinite(value) and value > 0 for value in
               (args.interval, args.query_timeout)):
        parser.error("interval and query timeout must be positive")
    try:
        first_sample_index = append_sample_index(args.output) if args.append else 0
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    stop_reason = ["stop_file"]

    def on_signal(signum, _frame):
        stop_reason[0] = signal.Signals(signum).name
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def stopped():
        return stop.is_set() or bool(args.stop_file and args.stop_file.exists())

    session_id = str(uuid.uuid4())
    with args.output.open("a" if args.append else "x", buffering=1) as stream:
        def emit(record):
            record["session_id"] = session_id
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()

        fields, unavailable, discovery_errors = discover_fields(args.query_timeout)
        epoch = time.time()
        emit({"schema_version": SCHEMA_VERSION, "record_type": "metadata",
              "time_epoch": epoch, "time_utc": utc(epoch),
              "perf_counter_s": time.perf_counter(), "physical_gpu": 1,
              "interval_s": args.interval, "query_fields": fields,
              "unavailable_fields": unavailable, "discovery_errors": discovery_errors,
              "append": args.append, "sample_index_start": first_sample_index,
              "read_only": True, "cuda_context_created": False})
        count = errors = 0
        deadline = time.perf_counter()
        while not stopped():
            sample = sample_gpu(fields, unavailable, args.query_timeout)
            sample["sample_index"] = first_sample_index + count
            sample["schedule_lag_s"] = max(0.0,
                sample["perf_counter_s"] - sample["query_duration_s"] / 2 - deadline)
            emit(sample)
            count += 1
            errors += int(sample["status"] != "ok")
            deadline += args.interval
            if deadline < time.perf_counter():
                deadline += math.floor((time.perf_counter() - deadline) / args.interval + 1) * args.interval
            while not stopped() and time.perf_counter() < deadline:
                stop.wait(min(0.2, max(0.0, deadline - time.perf_counter())))
        epoch = time.time()
        emit({"schema_version": SCHEMA_VERSION, "record_type": "stopped",
              "time_epoch": epoch, "time_utc": utc(epoch),
              "perf_counter_s": time.perf_counter(), "reason": stop_reason[0],
              "samples": count, "non_ok_samples": errors})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
