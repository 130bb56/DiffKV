"""CPU checks for length replay integrity and observer attribution."""

from collections import deque
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import run_r1_math as runner
import validate_r1_greedy as parity
import analyze_r1_math as analysis


class ReplayTests(unittest.TestCase):
    def test_replay_uses_token_ids_including_eos_and_requires_same_workload(self):
        requests = [{"request_id": "a"}, {"request_id": "b"}]
        reference = {"status": "completed",
                     "metadata": {"backend": "fp16", "mode": "natural", "workload_sha256": "same"},
                     "requests": [{"request_id": "a", "output_tokens": 2, "output_token_ids": [10, 128001]},
                                  {"request_id": "b", "output_tokens": 1, "output_token_ids": [128001]}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reference.json"
            path.write_text(json.dumps(reference))
            self.assertEqual(runner.length_limits(requests, "length_replay", 8192, path, "same"),
                             {"a": 2, "b": 1})
            with self.assertRaisesRegex(ValueError, "different workload"):
                runner.length_limits(requests, "length_replay", 8192, path, "other")
            reference["requests"][0]["output_tokens"] = 3
            path.write_text(json.dumps(reference))
            with self.assertRaisesRegex(ValueError, "count differs"):
                runner.length_limits(requests, "length_replay", 8192, path, "same")

    def test_diffkv_add_preserves_arrival_and_compression(self):
        captured = {}
        engine = SimpleNamespace(add_request=lambda **kw: captured.update(kw))
        runner.add_request(engine, "diffkv", "r", [1, 2], "sampling", 123.0)
        self.assertEqual(captured["arrival_time"], 123.0)
        self.assertEqual(captured["quant_configs"], [8, 4, 4, 2])
        self.assertEqual(captured["compress_configs"], [0.02, 1.0])


class ObserverTests(unittest.TestCase):
    def test_step_records_phase_batch_kv_and_preemption(self):
        clock = [10.0]
        manager = SimpleNamespace(num_total_gpu_blocks=100, get_num_free_gpu_blocks=lambda: 25)
        scheduler = SimpleNamespace(
            block_manager=manager, running=["a", "b"], waiting=["c"], swapped=[],
            _preempt=lambda *args, **kwargs: None,
            _preempt_by_recompute=lambda *args, **kwargs: None)
        metadata = [SimpleNamespace(request_id="a", seq_data={0: None}),
                    SimpleNamespace(request_id="b", seq_data={1: None})]
        engine = SimpleNamespace(scheduler=scheduler)
        engine._schedule = lambda: (metadata, SimpleNamespace(prompt_run=False, num_batched_tokens=2), [])

        def step():
            engine._schedule()
            scheduler._preempt(SimpleNamespace(request_id="b"))
            scheduler._preempt_by_recompute(SimpleNamespace(request_id="b"))
            clock[0] += 0.25
            return ["output"]

        engine.step = step
        with patch.object(runner.time, "perf_counter", side_effect=lambda: clock[0]):
            observer = runner.StepObserver(engine, None, "fp16")
            outputs, ended, row = observer.step(0, 9.0)
        self.assertEqual(outputs, ["output"])
        self.assertEqual(ended, 10.25)
        self.assertEqual(row["duration_s"], 0.25)
        self.assertEqual(row["batch"], 2)
        self.assertEqual(row["preemptions"], 1)
        self.assertEqual(row["state_after_step"]["kv_usage_percent"], 75.0)
        self.assertEqual(observer.summary()["mean_decode_batch"], 2)
        self.assertEqual(observer.summary()["recompute_count"], 1)
        self.assertEqual(observer.request_preemptions["b"], 1)


class ParityTests(unittest.TestCase):
    def reference(self):
        return {"status": "completed", "workload_sha256": "x", "max_new_tokens": 4,
                "requests": [{"request_id": "a", "prompt_token_ids": [1, 2],
                              "output_token_ids": [3, 4, 5]}]}

    def test_exact_and_first_mismatch(self):
        first, second = self.reference(), self.reference()
        self.assertTrue(parity.compare(first, second)["exact_match"])
        second["requests"][0]["output_token_ids"][1] = 9
        value = parity.compare(first, second)
        self.assertFalse(value["exact_match"])
        self.assertEqual(value["checks"][0]["first_mismatch_index"], 1)

    def test_missing_final_token_fails(self):
        first, second = self.reference(), self.reference()
        second["requests"][0]["output_token_ids"].pop()
        value = parity.compare(first, second)
        self.assertFalse(value["exact_match"])
        self.assertEqual(value["checks"][0]["first_mismatch_index"], 2)


class AnalysisSchemaTests(unittest.TestCase):
    def test_runner_schema_separates_accuracy_from_length_replay(self):
        workload = {"engine": {"max_model_len": 9216, "seed": 42},
                    "generation": {"max_tokens": 8192},
                    "requests": [{"request_id": "a", "unique_id": "a", "prompt_tokens": 2,
                                  "answer": "4"}]}

        def result(backend, mode, e2e, text):
            return {"status": "completed", "metadata": {
                "backend": backend, "mode": mode, "gpu": 0,
                "workload_sha256": "same", "engine": workload["engine"],
                "generation": dict(workload["generation"], ignore_eos=mode == "length_replay"),
                "ignore_eos": mode == "length_replay", "length_reference_sha256": "reference"},
                "requests": [{"request_id": "a", "prompt_tokens": 2, "max_tokens": 2,
                              "output_token_ids": [1, 128001], "output_tokens": 2,
                              "output_text": text, "finish_reason": "length",
                              "ttft_s": e2e / 2, "tpot_s": e2e / 2, "e2e_s": e2e}]}

        parser = SimpleNamespace(extract_answer=lambda text, **kwargs: text,
                                 strip_string=lambda text, **kwargs: text)
        grader = SimpleNamespace(math_equal=lambda predicted, truth, **kwargs: predicted == truth)
        fp16 = result("fp16", "natural", 3.0, "4")
        diffkv = result("diffkv", "natural", 2.0, "4")
        replay = result("diffkv", "length_replay", 2.0, "wrong answer")
        value = analysis.analyze(workload, "same", fp16, diffkv, replay,
                                 fp16_hash="reference", parser=parser, grader=grader)
        self.assertTrue(value["all_audits_passed"])
        self.assertTrue(value["correctness"]["passed"])
        self.assertTrue(value["performance"]["passed"])
        self.assertEqual(value["performance"]["mean_e2e_speedup"], 1.5)
        diffkv["metadata"]["generation"]["max_tokens"] = 4096
        bad_generation = analysis.analyze(workload, "same", fp16, diffkv, replay,
                                         fp16_hash="reference", parser=parser, grader=grader)
        self.assertFalse(bad_generation["all_audits_passed"])
        self.assertIsNone(bad_generation["correctness"])
        diffkv["metadata"]["generation"]["max_tokens"] = 8192
        replay["requests"][0]["output_tokens"] = 1
        value = analysis.analyze(workload, "same", fp16, diffkv, replay,
                                 fp16_hash="reference", parser=parser, grader=grader)
        self.assertFalse(value["all_audits_passed"])
        self.assertIsNone(value["performance"])


if __name__ == "__main__":
    unittest.main()
