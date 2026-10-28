#!/usr/bin/env python3
"""Audit the fixed-workload E2E grid and report case-wise speedups, using CPU only.

The default outputs are RUN_DIR/summary.json and RUN_DIR/results.md. A custom
--output-prefix writes PREFIX.json and PREFIX.md. Partial runs are reportable;
missing, failed, or incomparable measurements never contribute a speedup.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re


BACKENDS = ("fp16_vllm_0_2_6", "diffkv_v1_original", "diffkv_v1_optimized")
CASE_KEYS = ("id", "batch", "input_length", "output_length")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def digest(value):
    # This serialization is also used by run_kernel_e2e_grid.py.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return json.loads(path.read_text(), parse_constant=reject_constant)


def write_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value)
    temporary.replace(path)


def gpu_identity(value):
    if not isinstance(value, str):
        return None
    return (value[4:] if value.startswith("GPU-") else value).lower()


def case_identity(value):
    return {key: value.get(key) for key in CASE_KEYS} if isinstance(value, dict) else None


def validate_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("plan.json must contain an object")
    cases = plan.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("plan.cases must be a nonempty list")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each plan case must be an object")
        name = case.get("id")
        if not isinstance(name, str) or not name or name in seen or Path(name).name != name:
            raise ValueError(f"Duplicate or unsafe case ID: {name!r}")
        seen.add(name)
        for key in CASE_KEYS[1:]:
            if not integer(case.get(key)) or case[key] <= 0:
                raise ValueError(f"{name}: {key} must be a positive integer")
    if plan.get("backends") != list(BACKENDS):
        raise ValueError(f"plan.backends must be {list(BACKENDS)!r}")
    if plan.get("measured_generations_per_case_backend", 1) != 1:
        raise ValueError("This report requires one measured generation per case/backend")
    return cases


def snapshot_audit(value, expected_gpu, expected_sm):
    result = {"status": "unverified", "errors": []}
    if not isinstance(value, dict) or value.get("returncode") != 0:
        result["errors"].append("Missing or unsuccessful nvidia-smi snapshot")
        return result
    try:
        rows = list(csv.reader(value.get("stdout", "").splitlines(), skipinitialspace=True))
        if len(rows) != 1 or len(rows[0]) != 8:
            raise ValueError("Expected one GPU row with eight fields")
        fields = [entry.strip() for entry in rows[0]]
        result.update(gpu_uuid=fields[0], sm_clock_mhz=float(fields[1]),
                      memory_clock_mhz=float(fields[2]), temperature_c=float(fields[3]),
                      power_draw_w=float(fields[4]), power_limit_w=float(fields[5]),
                      gpu_utilization_pct=float(fields[6]), pstate=fields[7])
        if gpu_identity(fields[0]) != gpu_identity(expected_gpu):
            result["errors"].append("Snapshot GPU UUID differs from measured GPU")
        if expected_sm is not None and result["sm_clock_mhz"] != expected_sm:
            result["errors"].append(f"Snapshot SM clock is not {expected_sm} MHz")
        result["status"] = "valid" if not result["errors"] else "invalid"
    except (TypeError, ValueError) as error:
        result["errors"].append(str(error))
    return result


def load_telemetry(run_dir, override):
    path = override
    if path is None:
        for name in ("gpu-telemetry.jsonl", "telemetry.jsonl", "gpu_telemetry.jsonl"):
            candidate = run_dir / name
            if candidate.is_file():
                path = candidate
                break
    state = {"path": str(path) if path else None, "status": "unavailable"}
    if path is None or not path.is_file():
        state["reason"] = "Continuous GPU telemetry file is not available"
        return state, None, None
    module_path = Path(__file__).with_name("monitor_kernel_e2e_gpu.py")
    try:
        spec = importlib.util.spec_from_file_location("kernel_e2e_telemetry", module_path)
        monitor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monitor)
        records = monitor.read_records(path)
        state.update(status="loaded", sha256=sha_file(path), helper_sha256=sha_file(module_path))
        return state, monitor, records
    except Exception as error:
        state.update(status="unavailable", reason=f"Cannot read telemetry: {error}")
        return state, None, None


def audit_result(path, backend, case, plan, plan_sha, monitor, records):
    audit = {"path": str(path), "status": "missing", "valid": False,
             "timing_qualified": False, "errors": []}
    if not path.is_file():
        audit["errors"].append("Result file is missing")
        return audit, None
    try:
        data = read_json(path)
        if not isinstance(data, dict):
            raise ValueError("Result must contain an object")
    except (OSError, ValueError) as error:
        audit.update(status="unreadable", errors=[str(error)])
        return audit, None
    audit.update(sha256=sha_file(path), status=data.get("status", "unknown"),
                 worker_error=data.get("error"))
    errors = audit["errors"]

    def check(condition, message):
        if not condition:
            errors.append(message)

    check(data.get("status") == "completed", f"Worker status is {data.get('status')!r}")
    if data.get("status") != "completed":
        return audit, data
    batch, isl, osl = (case[key] for key in CASE_KEYS[1:])
    check(data.get("backend") == backend, "Backend differs from result filename")
    check(case_identity(data.get("case")) == case_identity(case), "Case differs from plan")
    if "plan_sha256" in data:
        check(data["plan_sha256"] == plan_sha, "Worker used a different plan digest")
    if "samples" in data:
        check(integer(data["samples"]) and data["samples"] == 1, "Expected one measured generation")
    e2e = data.get("e2e_s")
    check(finite(e2e) and e2e > 0, "e2e_s must be finite and positive")
    audit["e2e_s"] = e2e if finite(e2e) else None
    metrics = data.get("request_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    for name in ("mean_ttft_s", "mean_tpot_s", "mean_latency_s"):
        check(finite(metrics.get(name)) and metrics[name] >= 0, f"Invalid request_metrics.{name}")
    audit["request_metrics"] = metrics
    for name in ("mean_ttft_s", "mean_latency_s"):
        if finite(metrics.get(name)) and finite(e2e):
            check(metrics[name] <= e2e + 0.1, f"request_metrics.{name} exceeds measured E2E")
    start, end = data.get("measured_epoch_start"), data.get("measured_epoch_end")
    check(finite(start) and finite(end) and end > start, "Invalid measured epoch interval")
    audit.update(measured_epoch_start=start, measured_epoch_end=end)
    if finite(start) and finite(end) and finite(e2e):
        check(math.isclose(end - start, e2e, rel_tol=0.001, abs_tol=0.05),
              "Measured epoch interval differs from e2e_s")
    actual_gpu = data.get("gpu_uuid")
    check(isinstance(actual_gpu, str) and bool(actual_gpu.strip()), "Missing GPU UUID")
    if plan.get("gpu_uuid"):
        check(gpu_identity(actual_gpu) == gpu_identity(plan["gpu_uuid"]), "GPU UUID differs from plan")
    audit["gpu_uuid"] = actual_gpu

    inputs = data.get("input", {})
    if not isinstance(inputs, dict):
        inputs = {}
    for name in ("prompt_sha256", "batch_sha256"):
        check(isinstance(inputs.get(name), str) and SHA256.fullmatch(inputs[name]) is not None,
              f"Invalid input.{name}")
    prompt = inputs.get("prompt_token_ids")
    if prompt is not None:
        valid_prompt = (isinstance(prompt, list) and len(prompt) == isl
                        and all(integer(token) and token >= 0 for token in prompt))
        check(valid_prompt, "Prompt token IDs do not match ISL")
        if valid_prompt:
            check(digest(prompt) == inputs.get("prompt_sha256"), "Prompt digest does not match token IDs")
            check(digest([prompt] * batch) == inputs.get("batch_sha256"),
                  "Batch digest does not match B identical prompts")
    if "manifest_sha256" in inputs and plan.get("inputs", {}).get("sha256"):
        check(inputs["manifest_sha256"] == plan["inputs"]["sha256"], "Input manifest differs from plan")
    audit["input"] = {key: value for key, value in inputs.items() if key != "prompt_token_ids"}

    sampling = data.get("sampling", {})
    if not isinstance(sampling, dict):
        sampling = {}
    check(finite(sampling.get("temperature")) and sampling["temperature"] == 0,
          "Sampling temperature must be zero")
    check(sampling.get("ignore_eos") is True, "Sampling must ignore EOS")
    check(integer(sampling.get("max_tokens")) and sampling["max_tokens"] == osl,
          "Sampling max_tokens differs from OSL")
    check(integer(sampling.get("seed")) and sampling["seed"] == 42, "Sampling seed must be 42")
    if "n" in sampling:
        check(integer(sampling["n"]) and sampling["n"] == 1, "Sampling n must be one")
    audit["sampling"] = sampling
    engine = data.get("engine_args")
    check(isinstance(engine, dict) and bool(engine), "Missing engine_args")
    if isinstance(engine, dict):
        for key, expected in plan.get("engine", {}).items():
            check(engine.get(key) == expected, f"engine_args.{key} differs from plan")
        for key in ("dtype", "load_format"):
            if key in plan.get("model", {}):
                check(engine.get(key) == plan["model"][key], f"engine_args.{key} differs from model plan")
    audit["engine_args"] = engine

    guard = data.get("fixed_batch_guard", {})
    if not isinstance(guard, dict):
        guard = {}
    check(guard.get("passed") is True, "Fixed-batch guard did not pass")
    for key in ("expected_batch", "observed_decode_batch_min", "observed_decode_batch_max"):
        check(integer(guard.get(key)) and guard[key] == batch, f"fixed_batch_guard.{key} differs from B")
    check(integer(guard.get("preemptions")) and guard["preemptions"] == 0,
          "Fixed-batch run must have zero preemptions")
    if "preemption_attempts" in guard:
        check(guard["preemption_attempts"] == 0, "Fixed-batch run attempted preemption")
    if "decode_steps" in guard:
        check(guard["decode_steps"] == osl - 1, "Decode step count differs from OSL - 1")
    histogram = guard.get("decode_batch_histogram")
    if histogram is not None:
        check(isinstance(histogram, dict) and histogram == {str(batch): osl - 1},
              "Decode batch histogram differs from the fixed B and OSL")
    audit["fixed_batch_guard"] = guard

    tokens = data.get("output_token_ids")
    shape_valid = (isinstance(tokens, list) and len(tokens) == batch
                   and all(isinstance(row, list) and len(row) == osl
                           and all(integer(token) and token >= 0 for token in row) for row in tokens))
    check(shape_valid, "Output token IDs must contain exactly B requests of OSL tokens each")
    audit["output"] = {"shape_valid": shape_valid, "expected_requests": batch,
                       "expected_tokens_per_request": osl,
                       "actual_requests": len(tokens) if isinstance(tokens, list) else None,
                       "actual_lengths": [len(row) if isinstance(row, list) else None for row in tokens]
                       if isinstance(tokens, list) else None}
    if shape_valid:
        audit["output"]["sha256"] = digest(tokens)
        if "output_sha256" in data:
            check(data["output_sha256"] == audit["output"]["sha256"], "Output digest does not match token IDs")
        if "generated_tokens" in data:
            check(data["generated_tokens"] == batch * osl, "generated_tokens differs from B * OSL")
    extension = data.get("extension_sha256")
    check(isinstance(extension, str) and SHA256.fullmatch(extension) is not None, "Invalid extension SHA256")
    plan_key = {BACKENDS[1]: "original", BACKENDS[2]: "optimized"}.get(backend)
    if plan_key and plan.get(plan_key, {}).get("extension_sha256"):
        check(extension == plan[plan_key]["extension_sha256"], "Extension digest differs from plan")
    audit["extension_sha256"] = extension
    expected_sm = plan.get("clock", {}).get("sm_mhz")
    audit["telemetry_snapshots"] = {
        name: snapshot_audit(data.get(f"telemetry_{name}"), actual_gpu, expected_sm)
        for name in ("before", "after")
    }
    if monitor is not None and finite(start) and finite(end) and end > start:
        try:
            audit["continuous_telemetry"] = monitor.analyze_interval(
                records, start, end, expected_sm_mhz=expected_sm or 1395,
                interval_s=1.0, boundary_nearest=True)
            interval_audit = audit["continuous_telemetry"]
            if interval_audit.get("gpu_uuid") and gpu_identity(interval_audit["gpu_uuid"]) != gpu_identity(actual_gpu):
                interval_audit.update(status="invalid", valid=False, telemetry_complete=False)
                interval_audit.setdefault("errors", []).append("Continuous telemetry GPU differs from measured GPU")
        except Exception as error:
            audit["continuous_telemetry"] = {"status": "unavailable", "reason": str(error)}
    else:
        audit["continuous_telemetry"] = {"status": "unavailable", "reason": "No usable telemetry or epoch interval"}
    audit["valid"] = not errors
    audit["timing_qualified"] = audit["valid"] and audit["continuous_telemetry"].get("valid") is True
    if errors:
        audit["status"] = "invalid"
    return audit, data


def compare_pair(numerator, denominator, audits, data):
    errors = []
    for name in (numerator, denominator):
        if not audits[name]["valid"]:
            errors.append(f"{name}: {audits[name]['status']}")
    if not errors:
        left, right = data[numerator], data[denominator]
        for key in ("prompt_sha256", "batch_sha256"):
            if left["input"][key] != right["input"][key]:
                errors.append(f"Input {key} differs")
        for key in ("sampling", "engine_args"):
            if left[key] != right[key]:
                errors.append(f"{key} differs")
        if gpu_identity(left["gpu_uuid"]) != gpu_identity(right["gpu_uuid"]):
            errors.append("Measured GPU differs")
        if left.get("model_metadata_sha256") != right.get("model_metadata_sha256"):
            errors.append("Model metadata digests differ")
    workload_comparable = not errors
    telemetry_valid = all(audits[name].get("continuous_telemetry", {}).get("valid") is True
                          for name in (numerator, denominator))
    if workload_comparable:
        for name in (numerator, denominator):
            interval = audits[name].get("continuous_telemetry", {})
            if interval.get("valid") is not True:
                errors.append(f"{name}: timing unqualified, telemetry={interval.get('status', 'unavailable')}")
    return {"numerator": numerator, "denominator": denominator, "valid": not errors,
            "workload_comparable": workload_comparable, "telemetry_valid": telemetry_valid,
            "errors": errors,
            "speedup": data[numerator]["e2e_s"] / data[denominator]["e2e_s"] if not errors else None}


def token_agreement(original, optimized):
    exact_requests = equal_tokens = total_tokens = 0
    divergences = []
    first_positions = Counter()
    for request, (old, new) in enumerate(zip(original, optimized)):
        equal = sum(left == right for left, right in zip(old, new))
        equal_tokens += equal
        total_tokens += len(old)
        if equal == len(old):
            exact_requests += 1
            continue
        first = next(index for index, (left, right) in enumerate(zip(old, new)) if left != right)
        first_positions[first] += 1
        divergences.append({"request_index": request, "first_token_index": first,
                            "first_token_position": first + 1, "original_token_id": old[first],
                            "optimized_token_id": new[first], "differing_tokens": len(old) - equal})
    return {"available": True, "all_requests_exact": exact_requests == len(original),
            "requests": len(original), "exact_requests": exact_requests,
            "divergent_requests": len(divergences), "equal_tokens": equal_tokens,
            "total_tokens": total_tokens, "differing_tokens": total_tokens - equal_tokens,
            "token_agreement_fraction": equal_tokens / total_tokens,
            "first_divergence_token_index_counts": {str(key): count for key, count in sorted(first_positions.items())},
            "divergences": divergences,
            "index_convention": "request_index and first_token_index are zero-based; first_token_position is one-based"}


def geometric_mean(cases, key, require_telemetry=False):
    values = [(row["case"]["id"], row["comparisons"][key]["speedup"]) for row in cases
              if row["comparisons"][key]["valid"]
              and (not require_telemetry or row["comparisons"][key]["telemetry_valid"])]
    return {"value": math.exp(math.fsum(math.log(value) for _, value in values) / len(values)) if values else None,
            "case_count": len(values), "planned_case_count": len(cases),
            "case_ids": [name for name, _ in values], "complete": len(values) == len(cases),
            "requires_valid_continuous_telemetry": True,
            "definition": "exp(mean(log(per-case numerator E2E / optimized E2E))); equal weight per case, not aggregate runtime"}


def observed_clear(intervals, key, planned_count):
    values = [value.get(key) for value in intervals]
    if any(value is False for value in values):
        return False
    return True if len(values) == planned_count and all(value is True for value in values) else None


def cell(value):
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def markdown(summary):
    lines = ["# Fixed-workload model generation E2E", "",
             f"Status: **{summary['status']}**. Workload-valid measurements: {summary['valid_measurements']}/{summary['planned_measurements']}; "
             f"qualified timings: {summary['qualified_timings']}/{summary['planned_measurements']}; "
             f"comparable cases: {summary['comparable_cases']}/{summary['planned_cases']}.", "",
             "Each cell is one synchronized generation including prefill and decode, in seconds. "
             "Loading and warmup are excluded. Speedups divide the named baseline E2E by optimized E2E.", "",
             "| Case | B | ISL | OSL | FP16 (s) | Original (s) | Optimized (s) | Original / optimized | FP16 / optimized | Exact old/new requests | Audit |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in summary["cases"]:
        case = row["case"]
        values = [case["id"], case["batch"], case["input_length"], case["output_length"]]
        for name in BACKENDS:
            audit = row["results"][name]
            value = f"{audit['e2e_s']:.5f}" if audit["valid"] else audit["status"]
            if audit["valid"] and not audit["timing_qualified"]:
                value += " [unqualified]"
            values.append(value)
        for key in ("original_over_optimized", "fp16_over_optimized"):
            ratio = row["comparisons"][key]
            values.append(f"{ratio['speedup']:.3f}x" if ratio["valid"] else "unavailable")
        agreement = row["original_optimized_token_agreement"]
        values.append(f"{agreement['exact_requests']}/{agreement['requests']}" if agreement["available"] else "unavailable")
        values.append(row["status"] + ("; telemetry verified" if row["telemetry_valid"] else "; telemetry unverified"))
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    lines.append("")
    for key, label in (("original_over_optimized", "Original / optimized"),
                       ("fp16_over_optimized", "FP16 / optimized")):
        mean = summary["geometric_mean_speedups"][key]
        value = f"{mean['value']:.3f}x" if mean["value"] is not None else "unavailable"
        lines.append(f"{label} geometric mean **over cases**: **{value}**, "
                     f"{mean['case_count']}/{mean['planned_case_count']} cases. "
                     "This is not a ratio of aggregate runtimes.")
    agreement = summary["original_optimized_token_agreement"]
    lines.extend(["", f"Old/new generated-token agreement: {agreement['exact_requests']}/{agreement['requests']} exact requests, "
                  f"{agreement['divergent_requests']} requests with a first divergence, "
                  f"{agreement['differing_tokens']} differing token positions in comparable pairs. "
                  "Per-request first divergence positions are in summary.json.", "",
                  "Telemetry: " + summary["telemetry_note"], "",
                  "One measured generation per backend/case, without repeat averaging. "
                  "Missing, failed, invalid, incomparable, or telemetry-unqualified timings are excluded from speedups. "
                  "Raw elapsed times remain visible for unqualified measurements."])
    issues = []
    for row in summary["cases"]:
        for name, audit in row["results"].items():
            if not audit["valid"]:
                detail = "; ".join(audit["errors"])
                if audit.get("worker_error"):
                    detail += "; " + str(audit["worker_error"])
                issues.append(f"{row['case']['id']} / {name}: {detail}")
        for key, comparison in row["comparisons"].items():
            if not comparison["valid"] and all(audit["valid"] for audit in row["results"].values()):
                issues.append(f"{row['case']['id']} / {key}: " + "; ".join(comparison["errors"]))
    if issues:
        lines.extend(["", "Unusable measurements and comparisons:", ""])
        lines.extend("- " + cell(issue) for issue in issues)
    return "\n".join(lines) + "\n"


def analyze(run_dir, telemetry_path=None):
    plan_path = run_dir / "plan.json"
    plan = read_json(plan_path)
    cases = validate_plan(plan)
    plan_sha = sha_file(plan_path)
    telemetry, monitor, records = load_telemetry(run_dir, telemetry_path)
    rows = []
    for case in cases:
        audits, data = {}, {}
        for name in BACKENDS:
            path = run_dir / f"{case['id']}-{name}.json"
            audits[name], data[name] = audit_result(path, name, case, plan, plan_sha, monitor, records)
        comparisons = {
            "original_over_optimized": compare_pair(BACKENDS[1], BACKENDS[2], audits, data),
            "fp16_over_optimized": compare_pair(BACKENDS[0], BACKENDS[2], audits, data),
        }
        agreement = (token_agreement(data[BACKENDS[1]]["output_token_ids"], data[BACKENDS[2]]["output_token_ids"])
                     if comparisons["original_over_optimized"]["workload_comparable"] else {"available": False})
        valid = all(value["valid"] for value in comparisons.values())
        workload_comparable = all(value["workload_comparable"] for value in comparisons.values())
        telemetry_valid = all(audits[name].get("continuous_telemetry", {}).get("valid") is True
                              for name in BACKENDS)
        rows.append({"case": case_identity(case),
                     "status": "comparable" if valid else "timing_unqualified" if workload_comparable else "incomplete_or_incomparable",
                     "workload_comparable": workload_comparable,
                     "telemetry_valid": telemetry_valid,
                     "results": audits, "comparisons": comparisons,
                     "original_optimized_token_agreement": agreement})
    valid_count = sum(audit["valid"] for row in rows for audit in row["results"].values())
    qualified_count = sum(audit["timing_qualified"] for row in rows for audit in row["results"].values())
    comparable_count = sum(row["status"] == "comparable" for row in rows)
    agreements = [row["original_optimized_token_agreement"] for row in rows
                  if row["original_optimized_token_agreement"]["available"]]
    aggregate_agreement = {key: sum(value[key] for value in agreements)
                           for key in ("requests", "exact_requests", "divergent_requests", "equal_tokens", "total_tokens", "differing_tokens")}
    first_positions = Counter()
    for agreement in agreements:
        first_positions.update(agreement["first_divergence_token_index_counts"])
    aggregate_agreement.update(
        available_case_count=len(agreements), planned_case_count=len(cases), complete=len(agreements) == len(cases),
        all_requests_exact=all(value["all_requests_exact"] for value in agreements) if len(agreements) == len(cases) else None,
        all_available_requests_exact=all(value["all_requests_exact"] for value in agreements) if agreements else None,
        first_divergence_token_index_counts={key: value for key, value in sorted(first_positions.items(), key=lambda item: int(item[0]))})
    telemetry_states = Counter(str(audit.get("continuous_telemetry", {}).get("status", "unknown"))
                               for row in rows for audit in row["results"].values() if audit["valid"])
    interval_audits = [audit["continuous_telemetry"] for row in rows for audit in row["results"].values()
                      if audit["valid"]]
    telemetry_valid_count = sum(value.get("valid") is True for value in interval_audits)
    summary = {
        "schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir), "plan_path": str(plan_path), "plan_sha256": plan_sha,
        "analyzer_sha256": sha_file(Path(__file__)),
        "status": "completed" if comparable_count == len(cases) else "incomplete_or_incomparable",
        "planned_cases": len(cases), "planned_measurements": len(cases) * len(BACKENDS),
        "valid_measurements": valid_count, "comparable_cases": comparable_count,
        "qualified_timings": qualified_count,
        "workload_comparable_cases": sum(row["workload_comparable"] for row in rows),
        "backends": list(BACKENDS), "cases": rows,
        "geometric_mean_speedups": {key: geometric_mean(rows, key)
                                   for key in ("original_over_optimized", "fp16_over_optimized")},
        "telemetry_verified_geometric_mean_speedups": {
            key: geometric_mean(rows, key, require_telemetry=True)
            for key in ("original_over_optimized", "fp16_over_optimized")},
        "original_optimized_token_agreement": aggregate_agreement,
        "telemetry": dict(telemetry, per_measurement_status_counts=dict(telemetry_states),
                          valid_measurements=telemetry_valid_count,
                          evaluated_measurements=len(interval_audits),
                          planned_measurements=len(cases) * len(BACKENDS),
                          complete_and_valid=telemetry_valid_count == len(cases) * len(BACKENDS),
                          no_thermal_throttling_observed=observed_clear(
                              interval_audits, "no_thermal_throttling_observed", len(cases) * len(BACKENDS)),
                          no_power_limiting_observed=observed_clear(
                              interval_audits, "no_power_limiting_observed", len(cases) * len(BACKENDS))),
        "telemetry_note": (f"Continuous monitor: {telemetry['status']}; per-measurement states: {dict(telemetry_states)}. "
                           f"Verified intervals: {telemetry_valid_count}/{len(cases) * len(BACKENDS)} planned measurements. "
                           "Full interval audits and before/after snapshots are retained in summary.json. "
                           "Speedups require comparable workloads and valid continuous telemetry on both measurements."),
        "limits": ["One measured generation per case/backend; no repeat averaging or uncertainty estimate.",
                   "Geometric means give equal weight to cases, not requests or total runtime.",
                   "Token agreement is exact ID equality between original and optimized DiffKV; it is not an accuracy score.",
                   "Elapsed-time ratios exclude telemetry-unqualified measurements; their raw times and token IDs remain preserved."],
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--telemetry", type=Path, help="Continuous GPU monitor JSONL; auto-detected by default")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    try:
        summary = analyze(run_dir, args.telemetry.resolve() if args.telemetry else None)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.output_prefix:
        prefix = args.output_prefix.resolve()
        summary_path, report_path = Path(str(prefix) + ".json"), Path(str(prefix) + ".md")
    else:
        summary_path, report_path = run_dir / "summary.json", run_dir / "results.md"
    write_text(summary_path, json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write_text(report_path, markdown(summary))
    print(json.dumps({"status": summary["status"], "valid_measurements": summary["valid_measurements"],
                      "planned_measurements": summary["planned_measurements"],
                      "summary": str(summary_path), "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
