"""CPU-only audit and decomposition of completed and incomplete online PoC runs.

Reads immutable/raw run artifacts, never starts GPU work, and never rewrites a
measurement. Repeated runs are paired chronologically within an identical trace,
physical GPU and engine configuration. Request samples within a shared online
queue are not treated as independent replicate experiments.
"""

import argparse
import bisect
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from run_online_poc import compare, digest, distribution, save


ROOT = Path(__file__).resolve().parents[1]


def numeric(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def scalar_close(actual, expected, tolerance=1e-6):
    return numeric(actual) and numeric(expected) and abs(actual - expected) <= tolerance * max(1., abs(expected))


def decompose_request(row, server_record=None):
    timing = row.get("server_timing") or server_record or {}
    result = {key: row.get(key) for key in (
        "index", "request_id", "status", "scheduled_offset_s", "send_offset_s", "send_lag_ms",
        "first_token_offset_s", "last_token_offset_s", "response_end_offset_s", "e2e_s",
        "scheduled_e2e_s", "ttft_ms", "scheduled_ttft_ms", "tpot_ms", "http_completion_s",
        "prompt_tokens", "output_tokens", "output_token_ids_sha256", "error")}
    result["initial_admission_wait_s"] = timing.get("first_scheduled_after_received_s")
    result["preemption_wait_s"] = timing.get("recompute_wait_s")
    result["server_residence_s"] = timing.get("finished_after_received_s")
    result["preemptions"] = timing.get("preemptions")
    result["prefill_steps"] = timing.get("prefill_steps")
    result["decode_steps"] = timing.get("decode_steps")
    parts = [result[key] for key in ("initial_admission_wait_s", "preemption_wait_s", "server_residence_s")]
    if all(numeric(value) for value in parts) and numeric(row.get("e2e_s")):
        admission, preemption, residence = parts
        result["server_other_residence_s"] = residence - admission - preemption
        result["http_and_event_loop_residual_s"] = row["e2e_s"] - residence
        result["decomposition_sum_s"] = (admission + preemption + result["server_other_residence_s"]
                                          + result["http_and_event_loop_residual_s"])
        result["decomposition_conserved"] = scalar_close(result["decomposition_sum_s"], row["e2e_s"])
        result["decomposition_nonnegative"] = all(result[key] >= -1e-5 for key in (
            "initial_admission_wait_s", "preemption_wait_s", "server_other_residence_s",
            "http_and_event_loop_residual_s"))
    else:
        result.update(server_other_residence_s=None, http_and_event_loop_residual_s=None,
                      decomposition_sum_s=None, decomposition_conserved=None, decomposition_nonnegative=None)
    for key in ("received_offset_s", "first_scheduled_offset_s", "finished_offset_s"):
        result["server_" + key] = timing.get(key)
    return result


def rate_slope(points):
    if len(points) < 2:
        return None
    mx = statistics.mean(x for x, _ in points)
    my = statistics.mean(y for _, y in points)
    variance = sum((x - mx) ** 2 for x, _ in points)
    return sum((x - mx) * (y - my) for x, y in points) / variance if variance else None


def count_at(ordered, value):
    return bisect.bisect_right(ordered, value)


def occupancy_samples(arrivals, departures, end, bins=64):
    arrivals, departures = sorted(arrivals), sorted(departures)
    if not numeric(end) or end <= 0:
        return []
    return [{"offset_s": end * index / bins,
             "arrived": count_at(arrivals, end * index / bins),
             "completed": count_at(departures, end * index / bins),
             "outstanding": count_at(arrivals, end * index / bins) - count_at(departures, end * index / bins)}
            for index in range(bins + 1)]


def cohort_analysis(rows, mode, cohort_count=4):
    complete = [row for row in rows if row.get("status") == "completed"]
    arrival_key = "send_offset_s" if mode == "closed-loop" else "scheduled_offset_s"
    ordered = sorted((row for row in rows if numeric(row.get(arrival_key))),
                     key=lambda row: (row[arrival_key], row["index"]))
    cohorts = []
    actual_cohorts = min(cohort_count, len(ordered))
    for index in range(actual_cohorts):
        group = ordered[index * len(ordered) // actual_cohorts:(index + 1) * len(ordered) // actual_cohorts]
        if not group:
            continue
        cohorts.append({"cohort": index + 1, "request_count": len(group),
                        "completed": sum(row["status"] == "completed" for row in group),
                        "failed": sum(row["status"] != "completed" for row in group),
                        "arrival_min_s": group[0][arrival_key], "arrival_max_s": group[-1][arrival_key],
                        "request_index_min": min(row["index"] for row in group),
                        "request_index_max": max(row["index"] for row in group),
                        **{key: distribution(row.get(key) for row in group if row["status"] == "completed") for key in (
                            "e2e_s", "scheduled_e2e_s", "ttft_ms", "tpot_ms",
                            "send_lag_ms", "initial_admission_wait_s", "preemption_wait_s", "server_other_residence_s",
                            "http_and_event_loop_residual_s")}})
    arrivals = sorted(row[arrival_key] for row in ordered)
    departures = sorted(row["last_token_offset_s"] for row in complete if numeric(row.get("last_token_offset_s")))
    final = max(departures, default=0.)
    end_arrivals = max(arrivals, default=0.)
    timeline = occupancy_samples(arrivals, departures, max(end_arrivals, final))
    arrival_timeline = occupancy_samples(arrivals, departures, end_arrivals)
    late_points = [(row["offset_s"], row["outstanding"]) for row in arrival_timeline
                   if row["offset_s"] >= .5 * end_arrivals]
    drift = ((cohorts[-1]["e2e_s"]["mean"] / cohorts[0]["e2e_s"]["mean"])
             if len(cohorts) > 1 and cohorts[0]["e2e_s"]["mean"] and cohorts[-1]["e2e_s"]["mean"] else None)
    slope = rate_slope(late_points)
    if len(complete) != len(rows):
        observation = "incomplete_requests_preclude_capacity_interpretation"
    elif mode == "burst" or end_arrivals == 0:
        observation = "simultaneous_burst_drain_only"
    elif mode == "closed-loop":
        observation = "completion_coupled_concurrency_experiment"
    elif not departures or departures[0] >= end_arrivals:
        observation = "offered_arrivals_ended_before_first_completion"
    elif slope is not None and slope > 0 and drift is not None and drift > 1.1:
        observation = "backlog_and_later_cohort_latency_increased_in_observed_window"
    else:
        observation = "finite_arrival_window_with_mixed_or_no_detected_latency_growth"
    return {"arrival_axis": arrival_key, "cohorts": cohorts,
            "last_arrival_s": end_arrivals, "first_completion_s": min(departures, default=None),
            "last_completion_s": max(departures, default=None),
            "drain_after_last_arrival_s": max(0., final - end_arrivals),
            "completed_by_last_arrival": count_at(departures, end_arrivals),
            "outstanding_at_last_arrival": len(arrivals) - count_at(departures, end_arrivals),
            "late_arrival_window_backlog_slope_requests_per_s": slope,
            "last_over_first_cohort_mean_e2e": drift,
            "window_observation": observation,
            "steady_state_established": False,
            "interpretation": "Finite-window backlog and latency drift can be startup effects. These observations do not establish queue stability or sustainable throughput. Outstanding requests include executing, queued and transport/frontend work.",
            "timeline": timeline, "arrival_window_timeline": arrival_timeline}


def audit_summary(path):
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes)
    workload = raw["workload"]
    metrics = raw.get("server_metrics") or {}
    metadata = metrics.get("metadata") or {}
    server_rows = metrics.get("requests") or {}
    rows = [decompose_request(row, server_rows.get(row["request_id"])) for row in raw.get("requests", [])]
    expected_n, expected_i, expected_o = [workload[key] for key in ("num_requests", "input_length", "output_length")]
    actual = metadata.get("backend")
    completed = [row for row in rows if row["status"] == "completed"]
    state = metrics.get("state") or {}
    checks = {
        "status_completed": raw.get("status") == "completed",
        "expected_number_of_requests": len(rows) == expected_n,
        "unique_request_ids": len({row["request_id"] for row in rows}) == len(rows),
        "every_request_completed": len(completed) == expected_n,
        "exact_lengths": all(row["prompt_tokens"] == expected_i and row["output_tokens"] == expected_o for row in rows),
        "generated_token_total": raw.get("generated_tokens") == expected_n * expected_o,
        "server_completed_total": metrics.get("completed") == expected_n,
        "server_submitted_total": metrics.get("submitted") == expected_n,
        "actual_backend_matches_label": actual == raw.get("backend") and actual in ("fp16", "diffkv"),
        "diffkv_v1_if_applicable": actual != "diffkv" or metadata.get("diffkv_variant") == "no-sync-bulk-lengths",
        "server_no_error": metrics.get("engine_error") is None and metrics.get("failed") == 0,
        "server_fully_drained": metrics.get("active") == 0 and all(state.get(key) == 0 for key in (
            "running", "waiting", "swapped", "pending_frontend")),
        "batch_limit_matches_engine": metadata.get("common_engine_args", {}).get("max_num_seqs") == workload["batch_limit"],
        "decomposition_available": all(row["decomposition_conserved"] is not None for row in completed),
        "decomposition_conserved": all(row["decomposition_conserved"] for row in completed),
        "decomposition_nonnegative": all(row["decomposition_nonnegative"] for row in completed),
        "request_timing_identity": all(
            scalar_close(row["ttft_ms"] / 1000 + (expected_o - 1) * (row["tpot_ms"] or 0) / 1000, row["e2e_s"])
            for row in completed),
        "scheduled_latency_identity": all(
            row["scheduled_offset_s"] is None or scalar_close(
                row["scheduled_e2e_s"], row["e2e_s"] + row["send_lag_ms"] / 1000)
            for row in completed),
        "prefill_token_conservation": metrics.get("initial_prefill_tokens", 0) + metrics.get("recompute_prefill_tokens", 0)
            == metrics.get("phase_tokens", {}).get("prefill"),
        "generated_phase_token_conservation": sum(metrics.get("emitted_tokens", {}).values()) == expected_n * expected_o,
    }
    decomposition_available = checks["decomposition_available"]
    if decomposition_available:
        # A transport failure before server receipt has no server timing record.
        # Keep that request in all-request failure checks; do not invent events.
        checks["request_preemptions_reconcile"] = sum(
            row["preemptions"] for row in rows if numeric(row["preemptions"])) == metrics.get("preemption_count")
        for phase in ("prefill", "decode"):
            histogram = metrics.get(phase + "_batch_histogram", {})
            checks[phase + "_request_steps_reconcile"] = sum(row.get(phase + "_steps") or 0 for row in rows) == sum(int(b) * n for b, n in histogram.items())
    for key, stats_key in (("e2e_s", "e2e_s"), ("scheduled_e2e_s", "scheduled_e2e_s"),
                           ("ttft_ms", "ttft_ms"), ("tpot_ms", "tpot_ms")):
        calculated = distribution(row.get(key) for row in completed)
        saved = raw.get(stats_key) or {}
        checks[key + "_statistics_reconcile"] = all(
            calculated[stat] is None and saved.get(stat) is None or scalar_close(calculated[stat], saved.get(stat))
            for stat in ("mean", "p50", "p95", "p99"))
    trace_path = raw.get("run_metadata", {}).get("trace_path")
    if trace_path and Path(trace_path).exists():
        checks["source_trace_hash"] = hashlib.sha256(Path(trace_path).read_bytes()).hexdigest() == raw["trace_sha256"]
        trace = json.loads(Path(trace_path).read_text())
        checks["source_request_ids_match"] = [row["request_id"] for row in rows] == [row["request_id"] for row in trace["requests"]]
        checks["source_arrivals_match"] = [row["scheduled_offset_s"] for row in rows] == [row["scheduled_offset_s"] for row in trace["requests"]]
    else:
        checks["source_trace_hash"] = False
    decomposition = {key: distribution(row.get(key) for row in completed) for key in (
        "initial_admission_wait_s", "preemption_wait_s", "server_other_residence_s",
        "http_and_event_loop_residual_s", "server_residence_s")}
    if completed:
        decomposition["sum_of_component_means_s"] = sum(decomposition[key]["mean"] or 0 for key in (
            "initial_admission_wait_s", "preemption_wait_s", "server_other_residence_s", "http_and_event_loop_residual_s"))
    phase_keys = ("phase_seconds", "phase_tokens", "phase_steps", "phase_batch_seconds", "prefill_batch_histogram",
                  "decode_batch_histogram", "decode_batch_mean", "decode_batch_max", "initial_prefill_tokens",
                  "recompute_prefill_tokens", "prefill_padding_tokens", "preemption_count", "recompute_count",
                  "recompute_requests", "max_running", "max_waiting", "max_kv_usage_percent",
                  "running_request_seconds", "waiting_request_seconds", "kv_usage_percent_seconds")
    phases = {key: metrics.get(key) for key in phase_keys}
    decode_seconds = metrics.get("phase_seconds", {}).get("decode", 0)
    batch_seconds = metrics.get("phase_batch_seconds", {}).get("decode", {})
    phases["time_weighted_decode_batch"] = (sum(int(batch) * seconds for batch, seconds in batch_seconds.items()) / decode_seconds
                                             if decode_seconds and batch_seconds else None)
    phase_sum = sum(metrics.get("phase_seconds", {}).values())
    phases["mean_running_during_engine_steps"] = metrics.get("running_request_seconds", 0) / phase_sum if phase_sum else None
    phases["mean_waiting_during_engine_steps"] = metrics.get("waiting_request_seconds", 0) / phase_sum if phase_sum else None
    phases["phase_total_s"] = phase_sum
    phases["outside_observed_engine_steps_s"] = raw["duration_s"] - phase_sum
    if expected_n * expected_i:
        phases["recompute_prefill_tokens_over_original_input"] = metrics.get("recompute_prefill_tokens", 0) / (expected_n * expected_i)
    return {"label": path.parent.name, "status": raw.get("status"), "audit_passed": all(checks.values()),
            "checks": checks, "summary_path": str(path.resolve()), "summary_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "trace_sha256": raw["trace_sha256"], "backend": raw.get("backend"), "workload": workload,
            "started_utc": raw.get("run_metadata", {}).get("started_utc"),
            "common_engine_args": metadata.get("common_engine_args"), "gpu_uuid": metadata.get("gpu_uuid"),
            "client_source_sha256": raw.get("run_metadata", {}).get("client_source_sha256"),
            "client_transport": {key: raw.get("run_metadata", {}).get(key, default) for key, default in (
                ("client_protocol_version", 1), ("transport_policy", "pooled-keepalive"),
                ("client_force_close", False), ("client_connection_limit", 0),
                ("client_limit_per_host", 0), ("retries", 0))},
            "server_variant": metadata.get("diffkv_variant"), "duration_s": raw["duration_s"],
            "completed": raw.get("completed"), "failed": raw.get("failed"),
            "latency": {key: raw.get(key) for key in ("e2e_s", "scheduled_e2e_s", "ttft_ms", "tpot_ms", "send_lag_ms")},
            "request_throughput": raw["request_throughput"], "output_token_throughput": raw["output_token_throughput"],
            "decomposition": decomposition, "phases": phases,
            "temporal": cohort_analysis(rows, workload["arrival_mode"]), "requests": rows,
            "warnings": raw.get("warnings", []),
            "decomposition_definitions": {
                "initial_admission_wait_s": "Server adapter receipt through first scheduler admission, including async frontend waiting",
                "preemption_wait_s": "Sum of server preemption-to-next-scheduling intervals",
                "server_other_residence_s": "Remaining server residence: model execution, other-request interleaving, scheduler and frontend time outside measured waits; not pure GPU compute",
                "http_and_event_loop_residual_s": "Client E2E minus server receipt-to-final-output duration; includes HTTP, payload parsing, event-loop dispatch and final serialization; not pure network time",
                "percentiles": "Component percentiles are not additive. Only per-request durations and component means are summed."}}


def read_journal(path):
    if not path.exists():
        return []
    values = []
    for line in path.read_text().splitlines():
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return values


def inventory(root):
    planned = {}
    for path in sorted(root.glob("*plan*.json")):
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for task in value.get("tasks", []):
            planned[task["label"]] = {"task": task, "plan_path": str(path.resolve())}
    folders = {path.name: path for path in (root / "runs").glob("*") if path.is_dir()}
    entries = []
    for label in sorted(set(planned) | set(folders)):
        folder = folders.get(label, root / "runs" / label)
        summary = folder / "summary.json"
        if summary.exists():
            try:
                entry = audit_summary(summary)
            except Exception as error:
                entry = {"label": label, "status": "analysis_error", "audit_passed": False,
                         "summary_path": str(summary.resolve()), "error": f"{type(error).__name__}: {error}"}
        else:
            interrupted = folder / "interrupted.json"
            start = folder / "run.json"
            server_path = root / "servers" / label / "server-metadata.json"
            server = json.loads(server_path.read_text()) if server_path.exists() else {}
            failure = json.loads(interrupted.read_text()) if interrupted.exists() else {}
            metadata = json.loads(start.read_text()) if start.exists() else {}
            requests = read_journal(folder / "requests.jsonl")
            entry = {"label": label, "status": "interrupted" if interrupted.exists() else
                     "server_failed" if server.get("status") in ("failed", "OOM") else
                     "in_progress_or_interrupted" if start.exists() else "planned",
                     "audit_passed": False, "backend": metadata.get("backend_label", planned.get(label, {}).get("task", {}).get("backend")),
                     "recorded_requests": len(requests), "completed": sum(row.get("status") == "completed" for row in requests),
                     "failed": sum(row.get("status") != "completed" for row in requests),
                     "error": failure.get("error") or server.get("error"), "run_path": str(folder.resolve())}
        entry.update(planned.get(label, {}))
        entries.append(entry)
    return entries


def pair_runs(entries):
    groups = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        if not entry.get("summary_path") or not entry.get("workload"):
            continue
        key = (entry["trace_sha256"], digest(entry["common_engine_args"]), entry["gpu_uuid"] or "",
               digest(entry.get("client_transport", {})))
        groups[key][entry["backend"]].append(entry)
    pairs = []
    for group_key, backends in sorted(groups.items()):
        fp16 = sorted(backends.get("fp16", []), key=lambda row: (row.get("started_utc") or "", row["label"]))
        diffkv = sorted(backends.get("diffkv", []), key=lambda row: (row.get("started_utc") or "", row["label"]))
        for ordinal, (left, right) in enumerate(zip(fp16, diffkv), start=1):
            try:
                value = compare(left["summary_path"], right["summary_path"])
            except Exception as error:
                value = {"comparison_valid": False, "comparison_error": f"{type(error).__name__}: {error}",
                         "speedup_fp16_over_diffkv": {
                             **{key: {stat: None for stat in ("mean", "p50", "p95", "p99")}
                                for key in ("e2e_s", "scheduled_e2e_s", "ttft_ms", "tpot_ms", "http_completion_s")},
                             "makespan": None, "output_token_throughput": None}}
            value.update(trace_sha256=group_key[0], replicate_pair=ordinal,
                         comparison_group_sha256=digest(group_key),
                         fp16_label=left["label"], diffkv_label=right["label"], workload=left["workload"],
                         independent_audits_passed=left["audit_passed"] and right["audit_passed"],
                         earlier_backend="fp16" if left["started_utc"] < right["started_utc"] else "diffkv",
                         pairing="chronological ordinal within identical trace, GPU, common engine args and client transport protocol")
            value["valid"] = value["comparison_valid"] and value["independent_audits_passed"]
            value["decomposition_mean_s_fp16_minus_diffkv"] = {
                key: left["decomposition"][key]["mean"] - right["decomposition"][key]["mean"]
                if left["decomposition"][key]["mean"] is not None and right["decomposition"][key]["mean"] is not None else None
                for key in ("initial_admission_wait_s", "preemption_wait_s", "server_other_residence_s", "http_and_event_loop_residual_s")}
            pairs.append(value)
    return pairs


def aggregate_pairs(pairs):
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair["comparison_group_sha256"]].append(pair)
    result = []
    for group, attempts in groups.items():
        valid = [pair for pair in attempts if pair["valid"]]
        measures = {name: distribution(pair["speedup_fp16_over_diffkv"][name]["mean"] for pair in valid)
                    for name in ("e2e_s", "scheduled_e2e_s", "ttft_ms", "tpot_ms")}
        result.append({"comparison_group_sha256": group, "workload": attempts[0]["workload"],
                       "paired_attempts": len(attempts), "valid_pairs": len(valid),
                       "mean_latency_speedup_across_replicate_pairs": measures,
                       "all_valid_mean_e2e_speedups_at_least_1_4": bool(valid) and all(
                           pair["speedup_fp16_over_diffkv"]["e2e_s"]["mean"] >= 1.4 for pair in valid),
                       "interpretation": "Replicates are whole experimental runs. This summarizes ratios of request means within each matched run pair; request samples sharing a queue are not independent replicates."})
    return result


def attach_thermal(entries, pairs, root, explicit_path=None):
    paths = [explicit_path] if explicit_path else sorted(
        root.glob("thermal-audit*.json"),
        key=lambda path: (path.name == "thermal-audit-current.json", path.stat().st_mtime))
    audits = {}
    sources = []
    for path in paths:
        if not path.exists():
            continue
        source = json.loads(path.read_text())
        sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for row in source.get("runs", []):
            audits[row["run"]] = {**row, "source_artifact": str(path.resolve())}
    by_name = {entry["label"]: entry for entry in entries}
    for label, entry in by_name.items():
        entry["thermal_audit"] = audits.get(label)
    for pair in pairs:
        left, right = [audits.get(pair[key]) for key in ("fp16_label", "diffkv_label")]
        available = (left is not None and right is not None
                     and left.get("telemetry_available", left.get("n", 0) > 0)
                     and right.get("telemetry_available", right.get("n", 0) > 0))
        any_thermal = (left["thermal_n"] > 0 or right["thermal_n"] > 0) if available else None
        asymmetric = ((left["thermal_n"] > 0) != (right["thermal_n"] > 0)) if available else None
        pair["thermal_review"] = {"both_audits_available": available, "thermal_observed_any": any_thermal,
                                  "thermal_presence_differs": asymmetric, "thermal_review_required": any_thermal,
                                  "thermal_control_needed": any_thermal,
                                  "inherited_state_confound_established": None,
                                  "cache_only_quantitative_attribution_identified": False,
                                  "fp16": left, "diffkv": right,
                                  "interpretation": "Thermal observations require review, not automatic E2E invalidation. Inherited heat from prior work can confound a pair; thermal/DVFS response caused by a backend after a matched start is part of deployed automatic-clock E2E. Inspect cooldown, start samples, execution order and repetition before attributing carryover. Structural validity remains separate. These telemetry observations do not quantify a cache-only gain or thermal latency penalty. A thermal sample fraction is not exact thermal wall time; no observed flag does not prove absence between samples."}
    return sources


def cell(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown(entries, pairs):
    lines = ["# Online PoC search results", "", "All trials, including failed and incomplete trials, are retained. Latency is request E2E including queueing. Ratios above 1 favor DiffKV-v1.", "",
             "| Trial | Backend | State | Audit | B | ISL | OSL | N | Arrival | Mean E2E, s | p95 E2E, s | Mean admission, s | Mean preempt wait, s | Mean decode B | Preemptions | Makespan, s | Thermal samples |",
             "|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for run in entries:
        workload, latency, decomp, phases = [run.get(key, {}) for key in ("workload", "latency", "decomposition", "phases")]
        arrival = workload.get("arrival_mode", "-")
        if workload.get("request_rate"):
            arrival += f" {workload['request_rate']} req/s"
        values = [run["label"], run.get("backend"), run["status"], run["audit_passed"], workload.get("batch_limit"),
                  workload.get("input_length"), workload.get("output_length"), workload.get("num_requests"), arrival,
                  latency.get("e2e_s", {}).get("mean"), latency.get("e2e_s", {}).get("p95"),
                  decomp.get("initial_admission_wait_s", {}).get("mean"), decomp.get("preemption_wait_s", {}).get("mean"),
                  phases.get("decode_batch_mean"), phases.get("preemption_count"), run.get("duration_s"),
                  f"{run['thermal_audit']['thermal_n']}/{run['thermal_audit']['n']}" if run.get("thermal_audit") else None]
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    lines += ["", "| FP16 trial | DiffKV-v1 trial | Structural validity | Mean E2E speedup | p95 speedup | Scheduled mean speedup | Makespan speedup | Thermal review needed |",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for pair in pairs:
        speed = pair["speedup_fp16_over_diffkv"]
        values = [pair["fp16_label"], pair["diffkv_label"], pair["valid"], speed["e2e_s"]["mean"],
                  speed["e2e_s"]["p95"], speed["scheduled_e2e_s"]["mean"], speed["makespan"],
                  pair.get("thermal_review", {}).get("thermal_control_needed")]
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    lines += ["", "Decomposition: initial admission + preemption wait + other server residence + HTTP/event-loop residual = client request E2E. Other residence is not pure GPU execution. Only means and per-request durations are additive; component percentiles are not.",
              "", "Finite arrivals and their drain do not establish steady-state capacity. The JSON includes chronological cohorts, outstanding-request timelines, arrival-window backlog slope and late/early cohort latency for assessing transient accumulation.", ""]
    for run in entries:
        if run.get("error"):
            lines.append(f"- {run['label']}: {cell(run['error'])}")
        failed_checks = [key for key, value in run.get("checks", {}).items() if not value]
        if failed_checks:
            lines.append(f"- {run['label']} audit failures: {', '.join(failed_checks)}")
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "logs/performance/online-poc-20261022")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--thermal-audit", type=Path, help="Optional external thermal-window audit; default discovers thermal-audit*.json")
    args = parser.parse_args()
    output = args.output_dir or args.root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    entries = inventory(args.root)
    pairs = pair_runs(entries)
    thermal_sources = attach_thermal(entries, pairs, args.root, args.thermal_audit)
    repeated = aggregate_pairs(pairs)
    request_rows = [{"run": entry["label"], "backend": entry["backend"], **row}
                    for entry in entries for row in entry.get("requests", [])]
    cohort_rows = []
    for entry in entries:
        for group in entry.get("temporal", {}).get("cohorts", []):
            cohort_rows.append({"run": entry["label"], "backend": entry["backend"],
                                **{key: value for key, value in group.items() if not isinstance(value, dict)},
                                **{key + "_" + stat: value[stat] for key, value in group.items() if isinstance(value, dict)
                                   for stat in ("mean", "p50", "p95")}})
    save(output / "analysis.json", {"generated_utc": datetime.now(timezone.utc).isoformat(),
                                   "analyzer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                   "root": str(args.root.resolve()), "runs": entries, "pairs": pairs,
                                   "repeat_groups": repeated, "thermal_audit_sources": thermal_sources})
    save(output / "comparisons.json", pairs)
    write_csv(output / "requests.csv", request_rows)
    write_csv(output / "cohorts.csv", cohort_rows)
    (output / "search-table.md").write_text(markdown(entries, pairs))
    print(json.dumps({"runs": len(entries), "completed": sum(row["status"] == "completed" for row in entries),
                      "audit_passed": sum(row["audit_passed"] for row in entries), "pairs": len(pairs),
                      "valid_pairs": sum(pair["valid"] for pair in pairs), "output": str(output.resolve())}))


if __name__ == "__main__":
    main()
