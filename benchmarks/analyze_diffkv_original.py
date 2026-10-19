#!/usr/bin/env python3
"""Analyze completed original-DiffKV captures without executing GPU work.

Wait for progress.json capture acknowledgements, export each finished trace,
and update the single-sample report artifacts. Publication is opt-in and needs
all 14 captures plus a successful clock and thermal telemetry audit.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PLAN = ROOT / "logs/performance/nsys-base1395-repeat3-20261019-001155/plan.json"
ANALYZER = ROOT / "benchmarks/analyze_offline_nsys.py"
PYTHON = ROOT / ".cache/baselines/fp16-env/bin/python"
NSYS = Path("/opt/nvidia/nsight-systems-cli/2024.4.1/target-linux-x64/nsys")
REPORT = ROOT / "profiling-DIffKV-table.md"
CASE_KEYS = ("id", "batch", "input_length", "output_length")
GROUPS = (
    ("gemm", ("kernel:gemm_including_split_k_reduce",)),
    ("attention", (
        "kernel:attention_main", "kernel:attention_reduce",
        "kernel:attention_other_or_fp16",
    )),
    ("kv_cache", ("kernel:kv_cache",)),
    ("other_gpu", ("kernel:other",)),
    ("kernel_overlap", ("kernel:overlap",)),
    ("copy_memset_only", (
        "memcpy_only", "memset_only", "memcpy_memset_overlap_only",
    )),
    ("gpu_idle", ("gpu_idle",)),
)
DISPLAY_METRICS = (
    ("wall", "E2E latency (s)"),
    ("gemm", "GEMM"),
    ("attention", "Attention"),
    ("kv_cache", "KV Update"),
    ("other_gpu", "non GEMM GPU op"),
    ("gpu_idle", "CPU path"),
)


def read_json(path):
    return json.loads(path.read_text())


def write_text(path, value):
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value)
    temporary.replace(path)


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def case_identity(case):
    return {key: case.get(key) for key in CASE_KEYS}


def validate_plan(plan):
    cases = plan.get("cases", [])
    expected = read_json(REFERENCE_PLAN)["cases"]
    require(len(cases) == len(expected) == 14, "The run must contain the 14 reference cases")
    require(
        [case_identity(case) for case in cases] == [case_identity(case) for case in expected],
        "Run cases must match the reference workload and order",
    )
    require(plan.get("backends") == ["diffkv"], "Only the diffkv backend is allowed")
    require(plan.get("diffkv_variant") == "original", "The plan must select the original variant")
    require(plan.get("repeats_per_case_backend") == 1, "Exactly one measurement is allowed per case")
    require(plan.get("physical_gpu") == 0, "The plan must select physical GPU 0")
    require(plan.get("requested_SM_clock_MHz") == 1395, "The plan must request 1395 MHz")
    return cases


def validate_worker(folder, case):
    metadata = read_json(folder / "worker/profile.json")
    worker = read_json(folder / f"worker/{case['id']}-diffkv.json")
    for label, data in (("capture metadata", metadata), ("worker", worker)):
        require(data.get("status") == "completed", f"{label}: status must be completed")
        require(data.get("backend") == "diffkv", f"{label}: backend mismatch")
        require(data.get("physical_gpu") == 0, f"{label}: physical GPU mismatch")
        require(data.get("diffkv_variant") == "original", f"{label}: variant must be original")
        recorded_case = data.get("case", {}) if label == "capture metadata" else data
        require(case_identity(recorded_case) == case_identity(case), f"{label}: case mismatch")
    require(metadata.get("capture_started") is True, "The capture was not started")
    require(metadata.get("capture_stopped") is True, "The capture was not stopped")
    require("python_variant" not in worker, "Original DiffKV must not contain Python variant patches")
    require(worker.get("measurement", {}).get("measured_batches") == 1, "Expected one measured batch")
    samples = worker.get("samples", [])
    require(isinstance(samples, list) and len(samples) == 1, "Expected exactly one worker sample")
    require(samples[0].get("repeat") == 1, "The single worker sample must be repeat 1")
    return metadata


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_summary(summary, case, metadata, sqlite_path):
    require(summary.get("backend") == "diffkv", "Summary backend mismatch")
    require(case_identity(summary.get("case", {})) == case_identity(case), "Summary case mismatch")
    require(summary.get("capture_metadata") == metadata, "Summary capture metadata mismatch")
    require(summary.get("sqlite_path") == str(sqlite_path), "Summary SQLite path mismatch")
    measurement = summary.get("measurement", {})
    require(measurement.get("range_name") == "measured_generation", "Summary NVTX range mismatch")
    require(measurement.get("device_id") == 0, "Summary CUDA device mismatch")
    checks = summary.get("checks", {})
    for key in (
        "wall_conserved", "kernel_calls_conserved", "kernel_duration_conserved",
        "metadata_steps_matched", "outer_scopes_used", "inner_scopes_excluded_when_outer_present",
    ):
        require(checks.get(key) is True, f"checks.{key} must be true")
    for key in ("phase_attribution_pct", "phase_duration_attribution_pct"):
        value = checks.get(key)
        require(finite_number(value) and math.isclose(value, 100, abs_tol=1e-7), f"checks.{key} must be 100")
    for key in (
        "gpu_kernels_clipped_at_generation_boundary",
        "gpu_memory_operations_clipped_at_generation_boundary",
    ):
        require(checks.get(key) == 0, f"checks.{key} must be zero")
    wall_ms = summary.get("wall_ms")
    require(finite_number(wall_ms) and wall_ms > 0, "wall_ms must be finite and positive")
    buckets = summary.get("wall_breakdown", [])
    names = [row.get("name") for row in buckets]
    expected_names = {name for _, group in GROUPS for name in group}
    require(len(names) == len(set(names)), "Duplicate wall buckets")
    require(set(names) == expected_names, "Missing or unknown wall buckets")
    for bucket in buckets:
        value = bucket.get("time_ms")
        require(finite_number(value) and value >= 0, "Wall bucket time must be finite and nonnegative")
        percentage = bucket.get("whole_wall_pct")
        require(
            finite_number(percentage) and math.isclose(percentage, value / wall_ms * 100, abs_tol=1e-7),
            "Wall bucket percentage does not match its duration",
        )
    require(
        math.isclose(sum(row["time_ms"] for row in buckets), wall_ms, rel_tol=1e-9, abs_tol=1e-6),
        "Wall buckets do not conserve measured generation time",
    )


def run_logged(command, log_path):
    with log_path.open("w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def analyze_capture(run_dir, case):
    job_id = f"{case['id']}-diffkv"
    folder = run_dir / job_id
    trace = folder / "trace.nsys-rep"
    require(trace.is_file() and trace.stat().st_size > 0, f"Missing completed trace: {trace}")
    metadata = validate_worker(folder, case)
    sqlite_path = folder / "trace.sqlite"
    summary_path = folder / "summary.json"
    if not sqlite_path.exists():
        temporary = folder / f".trace-export-{os.getpid()}.sqlite"
        run_logged([
            str(NSYS), "export", "--type", "sqlite", "--output", str(temporary), str(trace),
        ], folder / "export.log")
        require(temporary.is_file() and temporary.stat().st_size > 0, "SQLite export is empty")
        temporary.replace(sqlite_path)
    if not summary_path.exists():
        temporary = folder / f".summary-{os.getpid()}.json"
        run_logged([
            str(PYTHON), str(ANALYZER), str(sqlite_path),
            "--capture-metadata", str(folder / "worker/profile.json"),
            "--range-name", "measured_generation", "--output", str(temporary),
        ], folder / "analysis.log")
        summary = read_json(temporary)
        validate_summary(summary, case, metadata, sqlite_path)
        temporary.replace(summary_path)
    summary = read_json(summary_path)
    validate_summary(summary, case, metadata, sqlite_path)
    buckets = {row["name"]: row["time_ms"] for row in summary["wall_breakdown"]}
    wall_ms = summary["wall_ms"]
    metrics = {"wall": {"value_s": wall_ms / 1000}}
    for key, names in GROUPS:
        value_ms = sum(buckets[name] for name in names)
        metrics[key] = {"value_s": value_ms / 1000, "whole_wall_pct": value_ms / wall_ms * 100}
    return {
        "case": case_identity(case), "job_id": job_id, "backend": "diffkv",
        "diffkv_variant": "original", "n": 1, "metrics": metrics,
        "summary_path": str(summary_path), "trace_path": str(trace),
        "checks": summary["checks"],
    }


def format_cell(metric):
    value = f"{metric['value_s']:.5f}"
    if "whole_wall_pct" in metric:
        value += f" ({metric['whole_wall_pct']:.2f}%)"
    return value


def markdown(run_dir, plan, records):
    lines = [
        "# DiffKV 원본, base clock 1395MHz 고정, 조건별 1회 측정",
        "",
        "DiffKV 원본(`original`)을 각 조건에서 1회 측정한 값입니다. 단위는 초이며, 시간은 소수점 5자리, 괄호 안 전체 생성 시간 대비 비중은 소수점 2자리로 표시했습니다. 표본 수는 조건별 n=1입니다.",
        "",
        "| B | ISL | OSL | " + " | ".join(label for _, label in DISPLAY_METRICS) + " |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        case = record["case"]
        cells = [str(case[key]) for key in ("batch", "input_length", "output_length")]
        cells.extend(format_cell(record["metrics"][key]) for key, _ in DISPLAY_METRICS)
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "",
        f"완료: {len(records)}/{len(plan['cases'])}개 조건, 조건별 측정 1회입니다.",
        "",
        f"각 조건에서 새 프로세스를 시작하고 {plan['warmup_tokens']}토큰 warmup 후 전체 생성을 1회 수행했습니다. DiffKV v1의 Python 최적화 패치를 적용하지 않았으며, worker 결과의 `diffkv_variant=original`, `python_variant` 필드 부재, 측정 표본 1개를 확인했습니다.",
        "기존 v1 측정과 동일한 DiffKV CUDA 빌드에서 original 경로를 실행했으며, no-sync-bulk-lengths Python 변경은 적용하지 않았습니다.",
        f"Warmup 후 최소 {plan['cooling_minimum_s']}초 동안 냉각하고 GPU 온도 {plan['pre_measurement_temperature_max_C']}°C 이하, thermal slowdown 비활성을 확인한 뒤 측정했습니다. 냉각 시간은 측정 범위에서 제외했습니다.",
        "전체 생성 시간은 `measured_generation` NVTX 범위의 wall time입니다. 모델 로딩과 warmup은 제외했으며, Nsight Systems 계측 overhead는 포함했습니다.",
        f"GPU0의 SM clock은 1395MHz, 전력 제한은 {plan['power_limit_W']}W로 유지했습니다. 최종 문서는 측정 구간의 클럭 일치, SW, HW thermal slowdown 비활성, power cap 비활성에 관한 telemetry audit을 통과한 뒤 발행합니다.",
        "",
        "GEMM은 split-K reduce를 포함합니다. Attention은 attention main, reduce, 기타 attention kernel을 합산했습니다. KV Update는 KV cache 관련 kernel입니다.",
        "`non GEMM GPU op`은 `kernel:other`, `CPU path`는 `gpu_idle`입니다. CPU path는 해당 프로세스와 GPU에서 kernel, copy, memset이 기록되지 않은 시간이며, CPU 실행 시간 자체를 별도로 계측한 값은 아닙니다.",
        "Copy, memset만 실행한 시간과 서로 다른 GPU kernel 분류의 중첩 시간은 표에서 생략하고 원자료에 유지했습니다. 모든 wall 분류는 중복 없이 전체 생성 시간을 보존합니다.",
        "",
        f"측정 조건과 실행 목록: [plan.json]({run_dir / 'plan.json'})",
        "",
        f"조건별 단일 측정값과 비중: [stats.json]({run_dir / 'stats.json'})",
        "",
        f"개별 summary 경로와 검증 결과: [all-results.json]({run_dir / 'all-results.json'})",
        "",
        f"클럭과 thermal 검증: [telemetry-audit.json]({run_dir / 'telemetry-audit.json'})",
    ])
    audit_path = run_dir / "telemetry-audit.json"
    if audit_path.exists():
        audit = read_json(audit_path)
        if audit.get("complete") is True and audit.get("audited_measurements") == 14:
            clocks = audit["numeric_ranges"]["SM_MHz"]
            thermal = audit["thermal_verification"]
            lines.extend([
                "",
                f"측정 구간의 {audit['audited_samples']}개 로그 샘플에서 SM clock은 최소 {clocks['min']:g}MHz, 최대 {clocks['max']:g}MHz였고, SW thermal 제한 활성 {thermal['sw_thermal_active_samples']}개, HW thermal 제한 활성 {thermal['hw_thermal_active_samples']}개, power cap 활성 {audit['powercap']['active_samples']}개였습니다.",
            ])
    return "\n".join(lines) + "\n"


def update_artifacts(run_dir, plan, results):
    records = [results[case["id"]] for case in plan["cases"] if case["id"] in results]
    stats = {
        "schema_version": 1, "run": str(run_dir),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "diffkv", "diffkv_variant": "original", "physical_gpu": 0,
        "n": 1, "time_unit": "seconds", "requested_SM_clock_MHz": 1395,
        "completed_measurements": len(records), "total_measurements": 14,
        "complete": len(records) == 14,
        "latency_source": "measured_generation NVTX wall time, including profiling overhead",
        "percentage": "category duration / the same measured generation wall time * 100",
        "display_metric_keys": [key for key, _ in DISPLAY_METRICS],
        "cases": records,
    }
    write_json(run_dir / "stats.json", stats)
    write_json(run_dir / "all-results.json", {
        "run": str(run_dir), "GPU_work_executed": False,
        "completed_measurements": len(records), "total_measurements": 14,
        "captures": [{key: record[key] for key in (
            "case", "job_id", "backend", "diffkv_variant", "n",
            "summary_path", "trace_path", "checks",
        )} for record in records],
    })
    content = markdown(run_dir, plan, records)
    write_text(run_dir / "preview.md.txt", content)
    return content


def validate_publication(run_dir, progress):
    require(progress.get("status") == "completed", "Publication requires completed collection")
    audit = read_json(run_dir / "telemetry-audit.json")
    require(audit.get("complete") is True, "Telemetry audit must be complete")
    require(audit.get("expected_measurements") == 14, "Telemetry audit must expect 14 measurements")
    require(audit.get("audited_measurements") == 14, "Telemetry audit must cover 14 measurements")
    require(audit.get("errors") == [], "Telemetry audit contains errors")
    clocks = audit.get("clock_verification", {})
    require(clocks.get("requested_SM_MHz") == 1395, "Telemetry audit must verify 1395 MHz")
    require(clocks.get("all_audited_samples_match") is True, "Measured SM clocks must match")
    require(clocks.get("mismatch_samples") == 0, "Telemetry audit contains clock mismatches")
    thermal = audit.get("thermal_verification", {})
    require(thermal.get("all_audited_samples_thermal_clear") is True, "Thermal slowdown must be clear")
    for key in ("sw_thermal_active_samples", "hw_thermal_active_samples"):
        require(thermal.get(key) == 0, f"Telemetry audit {key} must be zero")
    require(audit.get("powercap", {}).get("active_samples") == 0, "Power cap slowdown must be clear")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    plan = read_json(run_dir / "plan.json")
    cases = validate_plan(plan)
    expected_jobs = {f"{case['id']}-diffkv" for case in cases}
    results = {}
    content = update_artifacts(run_dir, plan, results)
    while len(results) < len(cases):
        progress_path = run_dir / "progress.json"
        if not progress_path.exists():
            time.sleep(5)
            continue
        progress = read_json(progress_path)
        captured = progress.get("captured", [])
        require(isinstance(captured, list), "progress.captured must be a list")
        require(len(captured) == len(set(captured)), "Duplicate captured job acknowledgements")
        require(set(captured) <= expected_jobs, "Unexpected jobs in progress.captured")
        require(progress.get("status") in ("running", "completed", "stopped"), "Unknown collector status")
        for case in cases:
            if case["id"] in results or f"{case['id']}-diffkv" not in captured:
                continue
            print(f"Analyzing {case['id']} ({len(results) + 1}/14)", flush=True)
            results[case["id"]] = analyze_capture(run_dir, case)
            content = update_artifacts(run_dir, plan, results)
        if len(results) == len(cases):
            break
        require(progress["status"] == "running", f"Collection {progress['status']} with only {len(results)}/14 analyzed")
        time.sleep(5)
    if args.publish:
        validate_publication(run_dir, read_json(run_dir / "progress.json"))
        write_text(REPORT, content)
        print(f"Published: {REPORT}", flush=True)
    else:
        print(f"Analyzed 14/14 captures. Preview: {run_dir / 'preview.md.txt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
