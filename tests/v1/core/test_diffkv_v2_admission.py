# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission must reserve measured pages before mutating engine scheduling."""

import importlib.util
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[3] / "vllm/diffkv_v2/admission.py"
spec = importlib.util.spec_from_file_location("diffkv_admission_cpu", MODULE)
assert spec is not None and spec.loader is not None
admission_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(admission_module)
DiffKVAdmission = admission_module.DiffKVAdmission


class DiffKVAdmissionTest(unittest.TestCase):
    @staticmethod
    def state(packed=10, raw=0, phase="decode"):
        return {"packed_pages": packed, "raw_pages": raw, "phase": phase}

    @staticmethod
    def report(requests, total=100, largest=None):
        free = total - sum(
            row["packed_pages"] + row["raw_pages"] for row in requests.values()
        )
        return {
            "total_pages": total,
            "free_pages": free,
            "largest_free_run": free if largest is None else largest,
            "page_bytes": 6784,
            "requests": requests,
        }

    def test_reserves_raw_and_packed_without_assuming_compression(self):
        guard = DiffKVAdmission(100, 2, 2)
        raw, packed = guard.prompt_pages(64)
        self.assertEqual((raw, packed), (20, 12))
        self.assertTrue(guard.reserve("first", 0, 64))
        self.assertEqual(guard.available_pages, 68)
        self.assertFalse(guard.reserve("second", 0, 64))

    def test_contiguous_raw_capacity_and_decode_reservations_both_apply(self):
        guard = DiffKVAdmission(100, 2, 2)
        guard.update_report(self.report({"decode": self.state(70)}, largest=18))
        guard.begin_step()
        self.assertTrue(guard.reserve("decode", 64, 65))
        self.assertEqual(guard.available_pages, 22)
        self.assertFalse(guard.reserve("new", 0, 64))

    def test_prefill_continuation_does_not_reserve_prompt_twice(self):
        guard = DiffKVAdmission(100, 2, 2)
        guard.update_report(self.report({"p": self.state(12, 20, "prefill")}))
        guard.begin_step()
        self.assertTrue(guard.reserve("p", 32, 64))
        self.assertEqual(guard.available_pages, 68)

    def test_preemption_credits_actual_victim_once_and_cancels_reservation(self):
        guard = DiffKVAdmission(100, 2, 2)
        guard.update_report(self.report({"a": self.state(50), "b": self.state(45)}))
        guard.begin_step()
        self.assertFalse(guard.reserve("a", 64, 65))
        guard.preempt("b")
        guard.preempt("b")
        self.assertEqual(guard.available_pages, 50)
        self.assertTrue(guard.reserve("a", 64, 65))
        self.assertEqual(guard.available_pages, 42)
        # Observed occupancy stays unchanged until the worker confirms release.
        self.assertAlmostEqual(guard.usage, 0.95)

    def test_worker_report_must_account_for_every_physical_page(self):
        guard = DiffKVAdmission(100, 2, 2)
        report = self.report({"a": self.state(50)})
        report["free_pages"] = 90
        with self.assertRaisesRegex(RuntimeError, "accounting"):
            guard.update_report(report)

    def test_impossible_prompt_fails_instead_of_waiting_forever(self):
        guard = DiffKVAdmission(10, 2, 2)
        with self.assertRaisesRegex(ValueError, "requires 32"):
            guard.reserve("new", 0, 64)


if __name__ == "__main__":
    unittest.main()
