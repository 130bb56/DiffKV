"""CPU checks for online latency decomposition and finite-window analysis."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analyze_online_poc import attach_thermal, audit_summary, cohort_analysis, decompose_request, inventory, pair_runs
from run_online_poc import summarize


def row(index, arrival, latency, completed=True):
    return {"index": index, "request_id": str(index), "status": "completed" if completed else "failed",
            "scheduled_offset_s": arrival, "send_offset_s": arrival + .01,
            "last_token_offset_s": arrival + .01 + latency if completed else None,
            "e2e_s": latency if completed else None, "scheduled_e2e_s": latency + .01 if completed else None,
            "ttft_ms": 1000, "tpot_ms": 10}


class AnalysisTests(unittest.TestCase):
    def test_transport_failure_without_server_timing_keeps_failed_status(self):
        successful = row(0, 0, 1)
        successful.update(prompt_tokens=2, output_tokens=2, ttft_ms=500, tpot_ms=500, send_lag_ms=10,
                          server_timing={"first_scheduled_after_received_s": .1,
                                         "recompute_wait_s": 0, "finished_after_received_s": .9,
                                         "preemptions": 0, "prefill_steps": 1, "decode_steps": 1})
        failed = row(1, 2, 1, False)
        failed.update(prompt_tokens=2, output_tokens=0, send_lag_ms=10,
                      error="connection reset before server receipt")
        shared = {"num_requests": 2, "input_length": 2, "output_length": 2,
                  "batch_limit": 2, "arrival_mode": "deterministic",
                  "requests": [{"request_id": "0"}, {"request_id": "1"}]}
        metrics = {"completed": 1, "submitted": 1, "failed": 0, "engine_error": None,
                   "preemption_count": 0, "prefill_batch_histogram": {"1": 1},
                   "decode_batch_histogram": {"1": 1}, "phase_tokens": {"prefill": 2},
                   "initial_prefill_tokens": 2, "phase_seconds": {"prefill": .2, "decode": .6}}
        raw = summarize(shared, [successful, failed], 4, "trace", metrics, {})
        raw["backend"] = "fp16"
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "summary.json"
            source.write_text(json.dumps(raw))
            result = audit_summary(source)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["failed"], 1)
        self.assertFalse(result["audit_passed"])
        self.assertTrue(result["checks"]["request_preemptions_reconcile"])
        self.assertEqual(len(result["requests"]), 2)

    def test_thermal_flag_triggers_review_not_proven_confound(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "thermal-audit-current.json").write_text(json.dumps({"runs": [
                {"run": "fp16", "n": 10, "thermal_n": 2, "telemetry_available": True},
                {"run": "diffkv", "n": 10, "thermal_n": 0, "telemetry_available": True}]}))
            pair = {"fp16_label": "fp16", "diffkv_label": "diffkv", "valid": True}
            attach_thermal([], [pair], root, None)
            self.assertTrue(pair["valid"])
            self.assertTrue(pair["thermal_review"]["thermal_review_required"])
            self.assertTrue(pair["thermal_review"]["thermal_presence_differs"])
            self.assertIsNone(pair["thermal_review"]["inherited_state_confound_established"])
            self.assertFalse(pair["thermal_review"]["cache_only_quantitative_attribution_identified"])
            self.assertNotIn("thermal_confounded", pair["thermal_review"])

    def test_request_residence_decomposition_conserves_latency(self):
        original = row(0, 0, 10)
        original["server_timing"] = {"first_scheduled_after_received_s": 2.,
                                     "recompute_wait_s": 3., "finished_after_received_s": 9.,
                                     "preemptions": 1}
        value = decompose_request(original)
        self.assertEqual(value["server_other_residence_s"], 4)
        self.assertEqual(value["http_and_event_loop_residual_s"], 1)
        self.assertEqual(value["decomposition_sum_s"], 10)
        self.assertTrue(value["decomposition_conserved"])
        self.assertTrue(value["decomposition_nonnegative"])
        original["server_timing"]["recompute_wait_s"] = 10
        self.assertFalse(decompose_request(original)["decomposition_nonnegative"])

    def test_small_trace_cohorts_keep_every_request(self):
        value = cohort_analysis([row(0, 0, 1), row(1, .1, 1)], "deterministic")
        self.assertEqual(sum(group["request_count"] for group in value["cohorts"]), 2)
        self.assertEqual(value["window_observation"], "offered_arrivals_ended_before_first_completion")
        self.assertFalse(value["steady_state_established"])

    def test_finite_window_growth_is_observation_not_stability_claim(self):
        value = cohort_analysis([row(i, float(i), 10 + 2 * i) for i in range(20)], "poisson")
        self.assertEqual(value["window_observation"], "backlog_and_later_cohort_latency_increased_in_observed_window")
        self.assertGreater(value["late_arrival_window_backlog_slope_requests_per_s"], 0)
        self.assertGreater(value["last_over_first_cohort_mean_e2e"], 1.1)
        self.assertFalse(value["steady_state_established"])
        flat = cohort_analysis([row(i, 2. * i, 1) for i in range(20)], "deterministic")
        self.assertEqual(flat["last_over_first_cohort_mean_e2e"], 1)
        self.assertFalse(flat["steady_state_established"])

    def test_failed_requests_remain_in_arrival_and_cohort_counts(self):
        value = cohort_analysis([row(0, 0, 1), row(1, 2, 1, False)], "deterministic")
        self.assertEqual(sum(group["request_count"] for group in value["cohorts"]), 2)
        self.assertEqual(sum(group["failed"] for group in value["cohorts"]), 1)
        self.assertEqual(value["outstanding_at_last_arrival"], 1)
        self.assertEqual(value["window_observation"], "incomplete_requests_preclude_capacity_interpretation")

    def test_inventory_retains_startup_failure_interruption_and_planned_work(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "screen-plan.json").write_text(json.dumps({"tasks": [
                {"label": label, "backend": "fp16"} for label in ("failed", "interrupted", "planned")]}))
            (root / "servers/failed").mkdir(parents=True)
            (root / "servers/failed/server-metadata.json").write_text(json.dumps({"status": "OOM", "error": "capacity"}))
            run = root / "runs/interrupted"
            run.mkdir(parents=True)
            (run / "run.json").write_text(json.dumps({"backend_label": "fp16"}))
            (run / "interrupted.json").write_text(json.dumps({"error": "cancelled"}))
            (run / "requests.jsonl").write_text(json.dumps({"status": "completed"}) + "\n{unfinished")
            values = {value["label"]: value for value in inventory(root)}
            self.assertEqual(values["failed"]["status"], "server_failed")
            self.assertEqual(values["interrupted"]["status"], "interrupted")
            self.assertEqual(values["interrupted"]["completed"], 1)
            self.assertEqual(values["planned"]["status"], "planned")

    def test_comparison_arithmetic_failure_is_retained_as_invalid_pair(self):
        common = {"trace_sha256": "trace", "common_engine_args": {"max_num_seqs": 32}, "gpu_uuid": "gpu",
                  "audit_passed": False, "workload": {"batch_limit": 32}, "started_utc": "2026-10-22T00:00:00",
                  "decomposition": {key: {"mean": None} for key in (
                      "initial_admission_wait_s", "preemption_wait_s", "server_other_residence_s", "http_and_event_loop_residual_s")}}
        entries = [{**common, "backend": backend, "label": backend, "summary_path": backend + ".json"}
                   for backend in ("fp16", "diffkv")]
        with patch("analyze_online_poc.compare", side_effect=ZeroDivisionError("no completed tokens")):
            pairs = pair_runs(entries)
        self.assertEqual(len(pairs), 1)
        self.assertFalse(pairs[0]["valid"])
        self.assertIn("ZeroDivisionError", pairs[0]["comparison_error"])

    def test_different_transport_protocols_are_not_paired(self):
        common = {"trace_sha256": "trace", "common_engine_args": {"max_num_seqs": 32}, "gpu_uuid": "gpu",
                  "audit_passed": True, "workload": {"batch_limit": 32}, "started_utc": "2026-10-22T00:00:00"}
        entries = [{**common, "backend": backend, "label": backend, "summary_path": backend + ".json",
                    "client_transport": {"client_protocol_version": version}}
                   for backend, version in (("fp16", 2), ("diffkv", 1))]
        with patch("analyze_online_poc.compare") as compare_mock:
            self.assertEqual(pair_runs(entries), [])
            compare_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
