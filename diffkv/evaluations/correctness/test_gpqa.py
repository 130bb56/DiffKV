# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the GPQA-diamond harness: scoring, shared generation settings."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

from correctness.score_outputs import grade_gpqa
from generation.generation_utils import GPQA_DIAMOND, GPQA_DIAMOND_197

GENERATE = Path(__file__).resolve().parents[1] / "generation/generate.py"


class GPQATest(unittest.TestCase):
    def test_answer_is_first_match_after_reasoning(self):
        result = grade_gpqa("<think>Answer: A</think>Answer: B\nAnswer: C", "B")
        self.assertEqual(result["prediction"], "B")
        self.assertTrue(result["correct"])
        self.assertTrue(grade_gpqa("<think>A</think>no letter", "A")["answer_missing"])

    @unittest.skipUnless(
        GPQA_DIAMOND.exists() and GPQA_DIAMOND_197.exists(), "no workloads"
    )
    def test_exclusion_keeps_other_prompts_and_choice_orders(self):
        full = json.loads(GPQA_DIAMOND.read_text())["requests"]
        reduced = json.loads(GPQA_DIAMOND_197.read_text())
        self.assertEqual(
            [row for row in full if row["request_id"] != "gpqa-diamond-127"],
            reduced["requests"],
        )
        self.assertEqual(reduced["excluded"][0]["request_id"], "gpqa-diamond-127")

    @unittest.skipUnless(GPQA_DIAMOND_197.exists(), "no 197-question workload")
    def test_only_output_limits_differ_from_math500(self):
        def validate(*arguments):
            command = [
                sys.executable,
                str(GENERATE),
                "--backend",
                "diffkv",
                "--model",
                "models/DeepSeek-R1-Distill-Llama-8B",
                "--mode",
                "natural",
                "--output",
                "/tmp/diffkv-validation-unused.json",
                "--validate-only",
                *arguments,
            ]
            return json.loads(subprocess.check_output(command, text=True))

        gpqa = validate(
            "--workload",
            str(GPQA_DIAMOND_197),
            "--expected-count",
            "197",
            "--max-tokens",
            "16384",
        )
        math500 = validate("--expected-count", "500")
        self.assertEqual(gpqa["requests"], 197)
        for result in (gpqa, math500):
            metadata = result["metadata"]
            metadata["limits"] = (
                metadata["engine"].pop("max_model_len"),
                metadata["generation"].pop("max_tokens"),
            )
        self.assertEqual(gpqa["metadata"]["limits"], (17408, 16384))
        self.assertEqual(math500["metadata"]["limits"], (9216, 8192))
        for key in ("engine", "kv_cache", "generation", "warmup_tokens"):
            self.assertEqual(gpqa["metadata"][key], math500["metadata"][key])


if __name__ == "__main__":
    unittest.main()
