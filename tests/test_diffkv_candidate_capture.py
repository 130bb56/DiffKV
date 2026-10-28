"""CPU checks for candidate snapshot extraction and exact-comparison gates."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = load("candidate_capture_test_module", ROOT / "benchmarks/validate_diffkv_kernel_candidate.py")
fixtures = load("candidate_capture_fixture_module", ROOT / "tests/kernels/test_diffkv_decode_attention.py")


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.build_fixture(((33, 61),), kv_heads=1)
        self.inputs = {key: getattr(self.fixture, key).clone()
                       for key in ("kv_cache", "kv_len_tables", "block_tables")}

    def test_snapshot_has_every_retained_score_in_logical_order(self):
        result = validator.cache_snapshot(torch, fixtures, self.fixture, self.inputs)
        expected = self.fixture.kv_cache.view(torch.float32).reshape(-1)[self.fixture.score_indices[0, 0]]
        self.assertTrue(torch.equal(result["scores"], expected))
        self.assertEqual(result["score_layout"], [
            {"sequence": 0, "kv_head": 0, "precision": "high", "tokens": 33},
            {"sequence": 0, "kv_head": 0, "precision": "low", "tokens": 61}])
        self.assertEqual(result["non_score_cache_sha256"], validator.tensor_hash(
            self.fixture.kv_cache[~self.fixture.score_mask]))

    def test_score_changes_do_not_hide_non_score_corruption(self):
        original = validator.cache_snapshot(torch, fixtures, self.fixture, self.inputs)
        address = self.fixture.score_indices[0, 0][0]
        self.inputs["kv_cache"].view(torch.float32).reshape(-1)[address] += 1
        changed_score = validator.cache_snapshot(torch, fixtures, self.fixture, self.inputs)
        self.assertEqual(original["non_score_cache_sha256"], changed_score["non_score_cache_sha256"])
        self.assertFalse(torch.equal(original["scores"], changed_score["scores"]))
        page = int(self.fixture.block_tables[int(self.fixture.slot_ids[0]), fixtures.LAYER_IDX, 0, 0])
        self.inputs["kv_cache"][page, 0] ^= 1
        changed_payload = validator.cache_snapshot(torch, fixtures, self.fixture, self.inputs)
        self.assertNotEqual(original["non_score_cache_sha256"], changed_payload["non_score_cache_sha256"])

    def test_exact_compare_rejects_changed_scores_and_positions(self):
        metadata = {"schema_version": 1, "fixture_sha256": "same", "operation": "same",
                    "contexts": [94], "batch": 1, "kv_heads": 1, "steps": 1,
                    "seed": 17, "policies": [(0.02, 1.0)]}
        state = validator.cache_snapshot(torch, fixtures, self.fixture, self.inputs)
        state["step"] = 1
        selection = copy.deepcopy(state)
        selection["policy"] = (0.02, 1.0)
        capture = {"metadata": metadata, "cases": [{
            "context": 94, "initial_cache_sha256": "same", "outputs": torch.ones((1, 1, 4, 128)),
            "snapshots": [state], "selections": [selection]}]}
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            args = SimpleNamespace(baseline=folder / "before.pt", candidate=folder / "after.pt",
                                   output=folder / "compare.json", allow_float_roundoff=False,
                                   output_atol=1e-3, score_atol=5e-8, rtol=5e-4)
            torch.save(capture, args.baseline)
            torch.save(capture, args.candidate)
            self.assertEqual(validator.compare(args), 0)
            changed = copy.deepcopy(capture)
            changed["cases"][0]["selections"][0]["retained_positions"][0] += 1
            torch.save(changed, args.candidate)
            self.assertEqual(validator.compare(args), 1)
            report = json.loads(args.output.read_text())
            self.assertFalse(report["cases"][0]["selections"][0]["retained_positions_identical"])


class WarmupTests(unittest.TestCase):
    def test_minimum_calls_and_wall_time_both_apply_outside_timing(self):
        clock = [0.0]
        counts = {"calls": 0, "syncs": 0}

        def operation():
            counts["calls"] += 1
            clock[0] += .01

        def synchronize():
            counts["syncs"] += 1

        fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize))
        with patch.object(validator.time, "perf_counter", side_effect=lambda: clock[0]):
            result = validator.warmup_operation(fake_torch, operation, (), 150, 2.0)
        self.assertEqual(counts["calls"], 250)
        self.assertEqual(counts["syncs"], 4)
        self.assertEqual(result["actual_iterations"], 250)
        self.assertGreaterEqual(result["elapsed_s"], 2.0)
        self.assertEqual(result["minimum_iterations"], 150)
        self.assertEqual(result["minimum_seconds"], 2.0)

    def test_count_only_warmup_stops_at_exact_minimum(self):
        counts = {"calls": 0}

        def operation():
            counts["calls"] += 1

        fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
        result = validator.warmup_operation(fake_torch, operation, (), 205, 0.0)
        self.assertEqual(counts["calls"], 205)
        self.assertEqual(result["actual_iterations"], 205)


if __name__ == "__main__":
    unittest.main()
