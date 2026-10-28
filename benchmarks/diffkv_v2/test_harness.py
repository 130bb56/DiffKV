# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests guarding workload drift, unequal work, and incorrect scoring."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from analyze import compare
from common import (
    OutputRecorder,
    backend_options,
    load_workload,
    output_limits,
    request_signature,
    summarize,
)
from grading import grade_one

HERE = Path(__file__).parent


class HarnessTest(unittest.TestCase):
    def setUp(self):
        self.workload_path = HERE / "fixtures/math50.json"
        self.workload = load_workload(self.workload_path, 50)
        self.lengths_path = HERE / "fixtures/lengths50.json"

    def test_frozen_50_replay_preserves_174055_tokens(self):
        limits = output_limits(
            self.workload, self.workload_path, "length_replay", self.lengths_path
        )
        self.assertEqual(sum(limits.values()), 174055)
        self.assertEqual((min(limits.values()), max(limits.values())), (934, 8192))

    def test_500_cannot_accidentally_run_the_50_subset(self):
        with self.assertRaisesRegex(ValueError, "Expected 500"):
            load_workload(self.workload_path, 500)

    def test_duplicate_requests_and_wrong_length_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"
            duplicate = copy.deepcopy(self.workload)
            duplicate["requests"][1]["request_id"] = duplicate["requests"][0][
                "request_id"
            ]
            path.write_text(json.dumps(duplicate))
            with self.assertRaisesRegex(ValueError, "unique"):
                load_workload(path, 50)
            changed = copy.deepcopy(self.workload)
            changed["requests"][0]["prompt_token_ids"][0] += 1
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "another workload"):
                output_limits(changed, path, "length_replay", self.lengths_path)
            self.assertNotEqual(
                request_signature(changed),
                request_signature(self.workload),
            )

    @staticmethod
    def output(tokens, finished):
        return SimpleNamespace(
            request_id="r",
            finished=finished,
            metrics=None,
            outputs=[
                SimpleNamespace(
                    token_ids=tokens,
                    text="answer",
                    finish_reason="length" if finished else None,
                    stop_reason=None,
                )
            ],
        )

    def test_final_notification_does_not_move_last_token_time(self):
        recorder = OutputRecorder(
            [{"request_id": "r", "prompt_token_ids": [1]}], {"r": 2}, "length_replay"
        )
        recorder.observe(self.output([2], False), 1.0)
        recorder.observe(self.output([2, 3], False), 3.0)
        recorder.observe(self.output([2, 3], True), 4.0)
        row = recorder.rows()[0]
        self.assertEqual((row["ttft_s"], row["tpot_s"], row["e2e_s"]), (1.0, 2.0, 4.0))
        self.assertEqual(summarize([row], 4.0)["output_tokens_per_s"], 0.5)

    def test_early_eos_invalidates_fixed_work(self):
        recorder = OutputRecorder(
            [{"request_id": "r", "prompt_token_ids": [1]}], {"r": 2}, "length_replay"
        )
        with self.assertRaisesRegex(ValueError, "stopped early"):
            recorder.observe(self.output([2], True), 1.0)

    def test_equal_total_tokens_cannot_hide_different_per_request_work(self):
        workload = {
            "requests": [
                {"request_id": name, "prompt_token_ids": [1], "answer": "1"}
                for name in ("a", "b")
            ]
        }

        def result(backend, lengths):
            return {
                "status": "completed",
                "metadata": {
                    "backend": backend,
                    "mode": "length_replay",
                    "request_signature": request_signature(workload),
                    "engine": backend_options(backend),
                    "generation": {},
                    "warmup_tokens": 32,
                    "length_reference_sha256": "same",
                    "commit": "same",
                },
                "requests": [
                    {
                        "request_id": name,
                        "prompt_tokens": 1,
                        "output_tokens": count,
                        "output_token_ids": [2] * count,
                        "max_tokens": count,
                        "ttft_s": 1.0,
                        "last_token_s": 3.0,
                        "e2e_s": 4.0,
                        "tpot_s": 2.0 / (count - 1),
                    }
                    for name, count in zip(("a", "b"), lengths)
                ],
                "summary": {"elapsed_s": 4.0},
            }

        fp16 = result("fp16", (2, 3))
        self.assertEqual(
            compare(workload, fp16, result("diffkv", (2, 3)))["throughput_ratio"], 1.0
        )
        with self.assertRaisesRegex(ValueError, "equal per-request"):
            compare(workload, fp16, result("diffkv", (3, 2)))

    def test_quantized_baselines_change_only_cache_and_attention_options(self):
        metadata = {}
        for backend in ("fp16", "diffkv", "fp8"):
            output = subprocess.check_output(
                [
                    sys.executable,
                    str(HERE / "run.py"),
                    "--backend",
                    backend,
                    "--model",
                    ".cache/models/DeepSeek-R1-Distill-Llama-8B",
                    "--mode",
                    "length_replay",
                    "--lengths-from",
                    str(self.lengths_path),
                    "--output",
                    "/tmp/diffkv-validation-unused.json",
                    "--validate-only",
                ],
                text=True,
            )
            result = json.loads(output)
            self.assertEqual(result["output_token_budget"], 174055)
            metadata[backend] = result["metadata"]
        baseline = metadata["fp16"]["engine"]
        self.assertNotIn("kv_cache_dtype", baseline)
        self.assertNotIn("attention_backend", baseline)
        for backend in ("diffkv", "fp8"):
            self.assertEqual(
                {
                    key: value
                    for key, value in metadata[backend]["engine"].items()
                    if key not in ("kv_cache_dtype", "attention_backend")
                },
                baseline,
            )
            self.assertEqual(
                metadata[backend]["generation"], metadata["fp16"]["generation"]
            )
        self.assertEqual(metadata["fp8"]["engine"]["kv_cache_dtype"], "fp8_e4m3")
        self.assertIn("default 1.0", metadata["fp8"]["kv_cache"]["scale_handling"])

    def test_pairwise_comparison_allows_only_named_quantization_changes(self):
        workload = {
            "requests": [{"request_id": "r", "prompt_token_ids": [1], "answer": "1"}]
        }
        recorder = OutputRecorder(workload["requests"], {"r": 2}, "length_replay")
        recorder.observe(self.output([2], False), 1.0)
        recorder.observe(self.output([2, 3], True), 3.0)

        def result(backend):
            return {
                "status": "completed",
                "metadata": {
                    "backend": backend,
                    "mode": "length_replay",
                    "request_signature": request_signature(workload),
                    "engine": {"dtype": "float16", **backend_options(backend)},
                    "generation": {"temperature": 0.0, "top_p": 1.0},
                    "warmup_tokens": 32,
                    "length_reference_sha256": "same",
                },
                "requests": recorder.rows(),
                "summary": {"elapsed_s": 3.0},
            }

        for reference in ("fp16", "diffkv"):
            compared = compare(
                workload, result(reference), result("fp8"), reference, "fp8"
            )
            self.assertEqual(set(compared["summaries"]), {reference, "fp8"})
            self.assertEqual(compared["throughput_ratio"], 1.0)
        wrong = result("fp8")
        wrong["metadata"]["engine"]["kv_cache_dtype"] = "auto"
        with self.assertRaisesRegex(ValueError, "unexpected kv_cache_dtype"):
            compare(workload, result("fp16"), wrong, "fp16", "fp8")
        for group, key, value in (
            ("engine", "dtype", "bfloat16"),
            ("generation", "temperature", 0.5),
        ):
            wrong = result("fp8")
            wrong["metadata"][group][key] = value
            with self.assertRaisesRegex(
                ValueError, "settings differ|differ in generation"
            ):
                compare(workload, result("fp16"), wrong, "fp16", "fp8")

    def test_grading_uses_final_answer_and_never_last_number_fallback(self):
        self.assertTrue(grade_one(r"<think>\boxed{2}</think>\boxed{3}", "3")["correct"])
        self.assertFalse(grade_one("I considered 3 possibilities", "3")["correct"])

    def test_sync_corrections_remain_explicit(self):
        result = grade_one(r"\boxed{1}, \boxed{2}", "1, 2")
        self.assertFalse(result["raw_correct"])
        self.assertTrue(result["correct"])
        self.assertEqual(result["correction"], "multiple_boxed_set")


if __name__ == "__main__":
    unittest.main()
