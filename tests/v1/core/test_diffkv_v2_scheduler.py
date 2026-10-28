# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise logical request handles against real upstream cache managers."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm.diffkv_v2 import backend
from vllm.diffkv_v2.runner import DiffKVRunnerState
from vllm.diffkv_v2.scheduler import DiffKVKVCacheManager
from vllm.diffkv_v2.spec import DiffKVSpec
from vllm.sampling_params import SamplingParams
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.request import Request


class DiffKVManagerTest(unittest.TestCase):
    def setUp(self):
        spec = DiffKVSpec(block_size=32, num_kv_heads=2, head_size=128)
        config = KVCacheConfig(
            num_blocks=50,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=100 * 6784,
                    layers=[name],
                    layer_stride=50 * 6784,
                    block_stride=6784,
                    offset=index * 50 * 6784,
                )
                for index, name in enumerate(("layer0", "layer1"))
            ],
            kv_cache_groups=[KVCacheGroupSpec(["layer0", "layer1"], spec)],
        )
        self.manager = DiffKVKVCacheManager(
            kv_cache_config=config,
            max_model_len=9216,
            scheduler_block_size=32,
            hash_block_size=32,
            enable_caching=False,
        )

    def report(self, packed=0, raw=0, phase="decode"):
        return {
            "total_pages": 100,
            "free_pages": 100 - packed - raw,
            "largest_free_run": 100 - packed - raw,
            "page_bytes": 6784,
            "requests": {
                "r": {"packed_pages": packed, "raw_pages": raw, "phase": phase}
            }
            if packed + raw
            else {},
        }

    def test_chunked_prefill_decode_and_free_use_reported_physical_capacity(self):
        manager = self.manager
        self.assertEqual(manager.physical.total_pages, 100)
        request = Request("r", [1] * 64, SamplingParams(max_tokens=8), None)
        manager.new_step_starts()
        self.assertEqual(len(manager.allocate_slots(request, 32).blocks[0]), 1)
        self.assertEqual(manager.usage, 0)
        manager.update_physical_report(self.report(12, 20, "prefill"))
        self.assertAlmostEqual(manager.usage, 0.32)
        manager.new_step_starts()
        request.num_computed_tokens = 32
        self.assertEqual(len(manager.allocate_slots(request, 32).blocks[0]), 0)
        manager.update_physical_report(self.report(12))
        manager.new_step_starts()
        request.num_computed_tokens = 64
        request.append_output_token_ids(1)
        self.assertEqual(len(manager.allocate_slots(request, 1).blocks[0]), 0)
        manager.free(request)
        self.assertAlmostEqual(manager.usage, 0.12)
        manager.update_physical_report(self.report())
        self.assertEqual(manager.usage, 0)
        self.assertEqual(manager.block_pool.get_num_free_blocks(), 49)

    def test_long_sequences_hold_only_one_logical_handle(self):
        single = self.manager.coordinator.single_type_managers[0]
        self.assertEqual(len(single.allocate_new_blocks("long", 9000, 9000)), 1)
        self.assertEqual(single.allocate_new_blocks("long", 9216, 9216), [])
        single.free("long")
        self.assertEqual(self.manager.block_pool.get_num_free_blocks(), 49)


class DiffKVRunnerLifecycleTest(unittest.TestCase):
    def setUp(self):
        backend.set_controller(None)
        self.runner = DiffKVRunnerState.__new__(DiffKVRunnerState)
        self.runner.controller = None
        self.runner.last_report = None

    def tearDown(self):
        backend.set_controller(None)

    def test_dummy_before_cache_initialization_never_calls_native_operators(self):
        batch = SimpleNamespace(
            req_ids=["dummy"],
            num_computed_tokens_np=[0],
            num_scheduled_tokens=[3],
        )
        self.runner.prepare_batch(batch, dummy=True)
        attention = backend.DiffKVV2Impl(4, 128, 128**-0.5, num_kv_heads=1)
        query = torch.zeros((3, 4, 128), dtype=torch.float16)
        key = torch.zeros((3, 1, 128), dtype=torch.float16)
        value = torch.ones_like(key)
        output = torch.empty_like(query)
        with patch.object(backend, "get_extension", side_effect=AssertionError):
            attention.forward(None, query, key, value, None, None, output)
            self.runner.finish_batch(dummy=True)
        torch.testing.assert_close(output, torch.ones_like(output))

    def test_zero_token_report_does_not_mutate_shared_empty_output(self):
        report = {"requests": {}, "free_pages": 100}
        self.runner.controller = SimpleNamespace(report=lambda: report)
        result = self.runner.attach_report(EMPTY_MODEL_RUNNER_OUTPUT, refresh=True)
        self.assertIs(result.diffkv_cache_report, report)
        self.assertIsNone(EMPTY_MODEL_RUNNER_OUTPUT.diffkv_cache_report)
        self.assertIsNot(result, EMPTY_MODEL_RUNNER_OUTPUT)

    def test_mixed_batch_appends_only_decode_and_finalizes_completed_prefill(self):
        requests = {
            "prefill": SimpleNamespace(
                slot_id=0,
                phase="prefill",
                num_computed_tokens=3,
                prompt_len=5,
            ),
            "decode": SimpleNamespace(
                slot_id=1,
                phase="decode",
                num_computed_tokens=8,
                prompt_len=4,
            ),
        }
        controller = SimpleNamespace(
            num_layers=1,
            max_model_len=32,
            requests=requests,
            append_decode=Mock(),
            finalize_prompt=Mock(),
            kv_len_tables=torch.tensor([[[[3, 0]]], [[[4, 6]]]]),
            set_num_computed_tokens=lambda req_id, value: setattr(
                requests[req_id],
                "num_computed_tokens",
                value,
            ),
            report=lambda: {
                req_id: req.num_computed_tokens for req_id, req in requests.items()
            },
        )
        backend.set_controller(controller, ["layer0"])
        backend.set_batch_state(
            ["decode", "prefill"],
            {"decode": 8, "prefill": 3},
            {"decode": 1, "prefill": 2},
        )
        controller.append_decode.assert_called_once_with(["decode"])
        self.assertEqual(backend._batch_state.decode_bounds, [10])
        backend._batch_state.executed_layers.add(0)
        backend._batch_state.compressed_layers["prefill"] = {0}
        report = backend.finish_batch()
        controller.finalize_prompt.assert_called_once_with(["prefill"])
        self.assertEqual(report, {"prefill": 5, "decode": 9})


if __name__ == "__main__":
    unittest.main()
