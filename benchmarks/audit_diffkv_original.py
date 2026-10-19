#!/usr/bin/env python3
"""Audit existing DiffKV original captures using CPU-only NVTX-aligned telemetry.

The default permits a partial collection. --require-complete requires all 14
single-generation captures, SM 1395 MHz, and clear thermal/power-cap flags.
No GPU queries, profiling, model execution, or hardware changes are performed.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics


KST = timezone(timedelta(hours=9))
UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
FLAGS = ("sw_thermal_slowdown", "hw_thermal_slowdown", "sw_power_cap")
FIELDS = {
    "SM_MHz": "clocks.current.sm [MHz]",
    "memory_MHz": "clocks.current.memory [MHz]",
    "temperature_C": "temperature.gpu",
    "power_W": "power.draw [W]",
    "power_limit_W": "power.limit [W]",
    "GPU_utilization_pct": "utilization.gpu [%]",
    "memory_utilization_pct": "utilization.memory [%]",
    "memory_used_MiB": "memory.used [MiB]",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_summary(values):
    return {"min": min(values), "mean": statistics.mean(values), "max": max(values)}


def timestamp_ns(value):
    instant = datetime.strptime(value, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=KST)
    delta = instant - UNIX_EPOCH
    return (delta.days * 86400 + delta.seconds) * 10**9 + delta.microseconds * 1000


def load_csv(path, epoch):
    rows = []
    with path.open(newline="") as stream:
        for line, raw in enumerate(csv.DictReader(stream), 2):
            require(None not in raw and all(v is not None for v in raw.values()),
                    f"{path}:{line}: incomplete telemetry row")
            row = {key.strip(): value.strip() for key, value in raw.items()}
            item = {
                "source_line": line,
                "timestamp": row["timestamp"],
                "relative_ns": timestamp_ns(row["timestamp"]) - epoch,
                "physical_GPU": int(row["index"]),
                "pstate": row["pstate"],
                "all_clock_event_reasons": row["clocks_throttle_reasons.active"],
            }
            for name, column in FIELDS.items():
                item[name] = float(row[column].split()[0])
                require(math.isfinite(item[name]), f"{path}:{line}: nonfinite {name}")
            for flag in FLAGS:
                value = row["clocks_throttle_reasons." + flag]
                require(value in ("Active", "Not Active"),
                        f"{path}:{line}: unknown {flag}: {value}")
                item[flag] = value == "Active"
            rows.append(item)
    require(rows, f"{path}: no telemetry rows")
    require(all(a["relative_ns"] < b["relative_ns"] for a, b in zip(rows, rows[1:])),
            f"{path}: timestamps are not strictly increasing")
    return rows


def sql_range(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as con:
        epochs = con.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchall()
        require(len(epochs) == 1, f"{path}: expected one session epoch")
        ranges = con.execute("""SELECT e.start,e.end,
            COALESCE(NULLIF(e.text,''),s.value),e.globalTid
            FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId=s.id
            WHERE e.end>e.start AND COALESCE(NULLIF(e.text,''),s.value)
            LIKE 'measured_generation%'""").fetchall()
    require(len(ranges) == 1 and ranges[0][2] == "measured_generation",
            f"{path}: expected exactly one measured_generation NVTX range")
    start, end, _, tid = ranges[0]
    return epochs[0][0], (start, end, tid)


def audit_capture(case, folder, requested):
    summary_path = folder / "summary.json"
    sqlite_path = folder / "trace.sqlite"
    csv_path = folder / "gpu.csv"
    summary = read_json(summary_path)
    epoch, measured_range = sql_range(sqlite_path)
    measurement = summary["measurement"]
    start, end = measurement["start_ns"], measurement["end_ns"]
    require(measurement["range_name"] == "measured_generation", "wrong measured NVTX name")
    require(measured_range == (start, end, measurement["global_tid"]),
            "SQLite NVTX range does not match summary")
    require(measurement["device_id"] == 0, "summary device is not GPU0")
    require(Path(summary["sqlite_path"]).resolve() == sqlite_path.resolve(),
            "summary SQLite path does not match capture")
    require(summary["backend"] == "diffkv", "summary backend is not diffkv")
    require(summary["case"]["id"] == case["id"], "summary case mismatch")
    metadata = summary["capture_metadata"]
    require(metadata["physical_gpu"] == 0, "metadata GPU is not GPU0")
    require(metadata["status"] == "completed", "capture not completed")
    require(metadata["diffkv_variant"] == "original", "capture variant is not original")
    require(metadata["backend"] == "diffkv", "metadata backend is not diffkv")
    require(metadata["case"]["id"] == case["id"], "metadata case mismatch")
    require(abs(summary["wall_ms"] - (end - start) / 1e6) <= 1e-6, "wall/NVTX mismatch")
    for flag in ("wall_conserved", "kernel_calls_conserved", "kernel_duration_conserved",
                 "metadata_steps_matched"):
        require(summary["checks"][flag] is True, "summary check failed: " + flag)
    rows = load_csv(csv_path, epoch)
    selected = [row for row in rows if start <= row["relative_ns"] < end]
    require(selected, "no telemetry samples inside measured NVTX range")
    require(all(row["physical_GPU"] == 0 for row in selected),
            "nonzero GPU index in selected rows")
    mismatches = [row for row in selected if row["SM_MHz"] != requested]
    signals = {
        flag: {
            "active_samples": sum(row[flag] for row in selected),
            "total_samples": len(selected),
            "sample_fraction": statistics.mean(row[flag] for row in selected),
            "first_active_sample": next((row for row in selected if row[flag]), None),
        } for flag in FLAGS
    }
    gaps = [(b["relative_ns"] - a["relative_ns"]) / 1e6
            for a, b in zip(selected, selected[1:])]
    return {
        "id": folder.name, "case": case, "backend": "diffkv", "variant": "original",
        "summary": str(summary_path), "summary_sha256": sha256(summary_path),
        "GPU_csv": str(csv_path), "GPU_csv_sha256": sha256(csv_path),
        "sqlite": str(sqlite_path), "sqlite_sha256_from_summary": summary["sqlite_sha256"],
        "utc_epoch_ns": epoch, "measurement": measurement,
        "CSV_samples_all_ranges": len(rows), "sample_count": len(selected),
        "numeric": {name: numeric_summary([row[name] for row in selected]) for name in FIELDS},
        "pstate_sample_counts": dict(Counter(row["pstate"] for row in selected)),
        "all_clock_event_reasons_sample_counts": dict(Counter(
            row["all_clock_event_reasons"] for row in selected)),
        "clock_verification": {"requested_SM_MHz": requested,
                               "all_samples_match": not mismatches,
                               "mismatch_sample_count": len(mismatches),
                               "mismatch_samples": mismatches},
        "event_signals": signals, "first_sample": selected[0], "last_sample": selected[-1],
        "boundary_coverage_ms": {
            "start_to_first_sample": (selected[0]["relative_ns"] - start) / 1e6,
            "last_sample_to_end": (end - selected[-1]["relative_ns"]) / 1e6,
            "max_between_sample_gap": max(gaps) if gaps else None,
        },
        "checks": {"SQLite_NVTX_matches_summary": True, "physical_GPU0": True,
                   "summary_integrity_checks_passed": True, "original_variant": True},
    }


def audit(run):
    plan = read_json(run / "plan.json")
    require(plan["physical_gpu"] == 0 and plan["requested_SM_clock_MHz"] == 1395,
            "plan must request GPU0 and SM 1395 MHz")
    require(plan["backends"] == ["diffkv"] and plan["diffkv_variant"] == "original",
            "plan must contain only DiffKV original")
    require(plan["repeats_per_case_backend"] == 1 and plan["total_measurements"] == 14,
            "plan must request 14 single measurements")
    cases = plan["cases"]
    require(len(cases) == 14 and len({case["id"] for case in cases}) == 14,
            "plan must contain 14 unique cases")
    require(plan["telemetry_interval_ms"] == 200, "plan telemetry interval must be 200 ms")
    captures, pending, errors = [], [], []
    for case in cases:
        folder = run / (case["id"] + "-diffkv")
        missing = [str(folder / name) for name in ("summary.json", "trace.sqlite", "gpu.csv")
                   if not (folder / name).is_file()]
        if missing:
            pending.append({"job_id": folder.name, "missing_artifacts": missing})
            continue
        try:
            captures.append(audit_capture(case, folder, plan["requested_SM_clock_MHz"]))
        except (ValueError, KeyError, TypeError, IndexError, sqlite3.Error, OSError) as error:
            errors.append({"job_id": folder.name, "error": str(error)})
    samples = sum(capture["sample_count"] for capture in captures)
    mismatches = [c["id"] for c in captures if not c["clock_verification"]["all_samples_match"]]
    thermal_affected = [c["id"] for c in captures
                        if any(c["event_signals"][flag]["active_samples"] for flag in FLAGS[:2])]
    totals = {flag: sum(c["event_signals"][flag]["active_samples"] for c in captures)
              for flag in FLAGS}
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256(Path(__file__)),
        "physical_GPU": 0, "GPU_work_executed": False,
        "complete": len(captures) == 14 and not pending and not errors,
        "expected_measurements": 14, "audited_measurements": len(captures),
        "audited_samples": samples, "pending_jobs": pending, "errors": errors,
        "clock_verification": {
            "requested_SM_MHz": 1395, "all_audited_samples_match": bool(samples) and not mismatches,
            "mismatch_measurements": mismatches,
            "mismatch_samples": sum(c["clock_verification"]["mismatch_sample_count"] for c in captures),
        },
        "thermal_verification": {
            "all_audited_samples_thermal_clear": bool(samples) and not thermal_affected,
            "affected_measurements": thermal_affected,
            "sw_thermal_active_samples": totals["sw_thermal_slowdown"],
            "hw_thermal_active_samples": totals["hw_thermal_slowdown"],
        },
        "powercap": {"active_samples": totals["sw_power_cap"], "total_samples": samples,
                     "sample_fraction": totals["sw_power_cap"] / samples if samples else None},
        "numeric_ranges": {
            field: {"min": min(c["numeric"][field]["min"] for c in captures),
                    "max": max(c["numeric"][field]["max"] for c in captures)}
            for field in FIELDS
        } if captures else {},
        "captures": captures,
        "method": {
            "CSV_timezone": "Asia/Seoul (UTC+09:00)",
            "alignment": "Integer CSV Unix ns minus SQLite utcEpochNs; measured_generation start<=sample<end",
            "scope": "One original DiffKV generation per case; model loading and warmup excluded by NVTX",
            "sampling_interval_ms": 200,
            "signals": "SW thermal, HW thermal and SW power cap counted independently",
            "limitation": "Sampled telemetry cannot exclude changes or events between samples",
            "require_complete": "14 valid captures, every SM sample 1395 MHz, no thermal or SW power-cap samples",
        },
    }
    result["complete_with_verified_base_clock"] = (
        result["complete"] and result["clock_verification"]["all_audited_samples_match"])
    result["complete_with_verified_telemetry"] = (
        result["complete_with_verified_base_clock"]
        and result["thermal_verification"]["all_audited_samples_thermal_clear"]
        and not result["powercap"]["active_samples"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    try:
        result = audit(run)
    except (ValueError, KeyError, TypeError, IndexError, sqlite3.Error, OSError) as error:
        parser.exit(1, f"Audit failed: {error}\n")
    output = run / "telemetry-audit.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({key: result[key] for key in (
        "complete", "audited_measurements", "audited_samples", "clock_verification",
        "thermal_verification", "powercap", "numeric_ranges", "errors",
        "complete_with_verified_telemetry")}, ensure_ascii=False, indent=2))
    print("OUTPUT " + str(output))
    if result["errors"] or (args.require_complete and not result["complete_with_verified_telemetry"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
