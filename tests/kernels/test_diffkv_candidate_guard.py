"""Isolated candidate attention alignment and short-tail tests on GPU 1.

Example: PYTHONPATH=. .cache/baselines/fp16-env/bin/python \
    tests/kernels/test_diffkv_candidate_guard.py --extension /path/to/_C.so
"""

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(__name__ == "__main__", "Run directly with --extension to select an isolated candidate")
class CandidateTests(unittest.TestCase):
    def run_and_check(self, fixture, inputs=None):
        inputs = fixtures.device_inputs(fixture) if inputs is None else inputs
        workspace = fixtures.allocate_workspaces(fixture)
        operation(*fixtures.attention_args(fixture, inputs, workspace))
        torch.cuda.synchronize()
        torch.testing.assert_close(workspace["out"].cpu().double(), fixture.expected_output,
                                   atol=1e-3, rtol=3e-4)
        actual_cache = inputs["kv_cache"].cpu()
        original_scores = fixture.kv_cache.view(torch.float32).reshape(-1)
        actual_scores = actual_cache.view(torch.float32).reshape(-1)
        for key, indices in fixture.score_indices.items():
            torch.testing.assert_close((actual_scores[indices] - original_scores[indices]).double(),
                                       fixture.score_updates[key], atol=5e-8, rtol=5e-4)
        self.assertTrue(torch.equal(actual_cache[~fixture.score_mask], fixture.kv_cache[~fixture.score_mask]))
        self.assertTrue(torch.equal(inputs["kv_len_tables"].cpu(), fixture.kv_len_tables))
        self.assertTrue(torch.equal(inputs["block_tables"].cpu(), fixture.block_tables))
        return inputs

    def test_aligned_storage_offset_is_accepted(self):
        fixture = fixtures.build_fixture()
        inputs = fixtures.device_inputs(fixture)
        backing = torch.empty(inputs["kv_cache"].numel() + 4, dtype=torch.int16, device="cuda")
        inputs["kv_cache"] = backing[4:].reshape(inputs["kv_cache"].shape)
        self.assertEqual(inputs["kv_cache"].storage_offset(), 4)
        self.assertEqual(inputs["kv_cache"].data_ptr() % 8, 0)
        inputs["kv_cache"].copy_(fixture.kv_cache)
        self.run_and_check(fixture, inputs)

    def test_misaligned_contiguous_storage_offset_is_rejected(self):
        fixture = fixtures.build_fixture()
        inputs = fixtures.device_inputs(fixture)
        backing = torch.empty(inputs["kv_cache"].numel() + 1, dtype=torch.int16, device="cuda")
        inputs["kv_cache"] = backing[1:].reshape(inputs["kv_cache"].shape)
        inputs["kv_cache"].copy_(fixture.kv_cache)
        self.assertTrue(inputs["kv_cache"].is_contiguous())
        self.assertEqual(inputs["kv_cache"].data_ptr() % 8, 2)
        workspace = fixtures.allocate_workspaces(fixture)
        with self.assertRaises(RuntimeError):
            operation(*fixtures.attention_args(fixture, inputs, workspace))
        self.assertTrue(torch.equal(inputs["kv_cache"].cpu(), fixture.kv_cache), "Rejected call changed cache")

    def test_misaligned_page_stride_is_rejected(self):
        fixture = fixtures.build_fixture()
        inputs = fixtures.device_inputs(fixture)
        shape = (inputs["kv_cache"].shape[0], fixtures.PAGE_SIZE + 1)
        inputs["kv_cache"] = torch.full(shape, 0x3555, dtype=torch.int16, device="cuda")
        self.assertTrue(inputs["kv_cache"].is_contiguous())
        self.assertEqual(inputs["kv_cache"].data_ptr() % 8, 0)
        self.assertEqual(inputs["kv_cache"].stride(0) * inputs["kv_cache"].element_size() % 8, 2)
        workspace = fixtures.allocate_workspaces(fixture)
        with self.assertRaises(RuntimeError):
            operation(*fixtures.attention_args(fixture, inputs, workspace))
        self.assertTrue(torch.all(inputs["kv_cache"] == 0x3555), "Rejected call changed cache")

    def test_short_vector_and_page_tails(self):
        for count in (1, 2, 7, 8, 9, 31, 32, 33, 59, 60, 61):
            for high, low in ((count, 0), (0, count), (count, count)):
                with self.subTest(high=high, low=low):
                    self.run_and_check(fixtures.build_fixture(((high, low),), kv_heads=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--op", choices=["legacy", "gqa"], default="legacy")
    options, unittest_arguments = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["OMP_NUM_THREADS"] = "2"
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    extension = load_module("vllm._C", options.extension.resolve())
    fixtures = load_module("candidate_short_fixture", ROOT / "tests/kernels/test_diffkv_decode_attention.py")
    if options.op == "gqa":
        fixtures.PARTITION_SIZE = fixtures.GQA_PARTITION_SIZE
    operation = getattr(extension.ops, "sparse_paged_attention" if options.op == "legacy" else "sparse_paged_attention_gqa")
    unittest.main(argv=[sys.argv[0], *unittest_arguments], verbosity=2)
