"""CPU-only checks for online measurement attribution and response integrity."""

import asyncio
from collections import deque
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "online_benchmark_server", ROOT / "benchmarks/online_benchmark_server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class FakeGroup:
    request_id = "r0"

    def get_max_num_running_seqs(self):
        return 1


class FakeSequence:
    def __init__(self, previous, prompt=4):
        self.previous, self.prompt = previous, prompt

    def get_output_len(self):
        return self.previous

    def get_len(self):
        return self.prompt + self.previous


class ObserverTests(unittest.TestCase):
    def make_engine(self, clock):
        manager = SimpleNamespace(get_num_free_gpu_blocks=lambda: 4, num_total_gpu_blocks=10)
        scheduler = SimpleNamespace(
            block_manager=manager, running=[FakeGroup()], waiting=[], swapped=[],
            _preempt=lambda *args, **kwargs: None,
            _preempt_by_recompute=lambda *args, **kwargs: None)
        steps = deque()
        engine = SimpleNamespace(scheduler=scheduler)

        def schedule():
            phase, previous = steps[0]
            metadata = [SimpleNamespace(request_id="r0", seq_data={0: FakeSequence(previous)})]
            output = SimpleNamespace(prompt_run=phase == "prefill", num_batched_tokens=4 + previous)
            return metadata, output, []

        async def step():
            engine._schedule()
            phase, previous = steps.popleft()
            clock[0] += 0.25
            return [SimpleNamespace(request_id="r0", outputs=[SimpleNamespace(
                token_ids=list(range(previous + 1)))])]

        engine._schedule, engine.step_async = schedule, step
        tracker = SimpleNamespace(_new_requests=SimpleNamespace(qsize=lambda: 0))
        wrapper = SimpleNamespace(engine=engine, _request_tracker=tracker)
        return wrapper, steps

    def test_initial_wait_and_recompute_are_separate(self):
        clock = [10.0]
        with patch.object(server.time, "perf_counter", side_effect=lambda: clock[0]):
            wrapper, steps = self.make_engine(clock)
            observer = server.PhaseObserver(wrapper, "fp16", None)
            observer.request_received("r0", 4, 3)
            clock[0] = 11.0
            steps.append(("prefill", 0))
            asyncio.run(wrapper.engine.step_async())
            observer.request_output("r0", 1, False)
            clock[0] = 11.5
            wrapper.engine.scheduler._preempt(FakeGroup())
            wrapper.engine.scheduler._preempt_by_recompute(FakeGroup())
            clock[0] = 13.0
            steps.append(("prefill", 1))
            asyncio.run(wrapper.engine.step_async())
            clock[0] = 13.5
            steps.append(("decode", 2))
            asyncio.run(wrapper.engine.step_async())
            observer.request_output("r0", 3, True)
            snapshot = observer.snapshot()
        request = snapshot["requests"]["r0"]
        self.assertEqual(request["first_scheduled_after_received_s"], 1.0)
        self.assertEqual(request["first_token_after_received_s"], 1.25)
        self.assertEqual(request["finished_after_received_s"], 3.75)
        self.assertEqual(request["recompute_wait_s"], 1.5)
        self.assertEqual(request["preemptions"], 1)
        self.assertEqual(snapshot["initial_prefill_tokens"], 4)
        self.assertEqual(snapshot["recompute_prefill_tokens"], 5)
        self.assertEqual(snapshot["phase_tokens"], {"prefill": 9, "decode": 1})
        self.assertEqual(snapshot["emitted_tokens"], {"prefill": 2, "decode": 1})
        self.assertEqual(snapshot["phase_seconds"], {"prefill": 0.5, "decode": 0.25})
        self.assertEqual(snapshot["phase_batch_seconds"], {"prefill": {1: 0.5}, "decode": {1: 0.25}})
        self.assertEqual(snapshot["running_request_seconds"], 0.75)
        self.assertEqual(snapshot["kv_usage_percent_seconds"], 45.0)
        self.assertNotIn("requests", observer.snapshot(include_requests=False))

    def test_reset_discards_warmup_request_metrics(self):
        wrapper, _ = self.make_engine([0.0])
        observer = server.PhaseObserver(wrapper, "fp16", None)
        observer.request_received("warmup", 4, 1)
        observer.request_output("warmup", 1, True)
        observer.reset()
        self.assertEqual(observer.snapshot()["requests"], {})

    def test_default_is_diffkv_v1_and_fixed_length_is_enforced(self):
        args = server.parse_args(["--backend", "diffkv", "--model", "/tmp/model",
                                  "--output-dir", "/tmp/results"])
        self.assertEqual(args.diffkv_variant, "no-sync-bulk-lengths")
        with self.assertRaises(RuntimeError):
            server.validate_output_length(1, 2, True, [])

    def test_final_http_frame_has_hash_and_server_timing(self):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            self.skipTest("FastAPI not installed in this CPU interpreter")
        wrapper, _ = self.make_engine([0.0])
        observer = server.PhaseObserver(wrapper, "fp16", None)

        async def generate(*args, **kwargs):
            for count in (1, 2):
                yield SimpleNamespace(
                    request_id="r0", prompt_token_ids=[1, 2, 3, 4], finished=count == 2,
                    outputs=[SimpleNamespace(token_ids=[10, 11][:count], text="result",
                                             finish_reason="length")])

        async def abort(request_id):
            pass

        wrapper.generate, wrapper.abort = generate, abort
        payload = {"request_id": "r0", "prompt_token_ids": [1, 2, 3, 4],
                   "max_tokens": 2, "stream": True, "ignore_eos": True}

        async def request_json():
            return payload

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(backend="fp16", max_model_len=8,
                                   output_dir=Path(directory), monitor_output=False)
            app = server.build_app(wrapper, observer, lambda **kwargs: kwargs, args, {})
            endpoint = next(route.endpoint for route in app.routes if route.path == "/generate")

            async def collect():
                response = await endpoint(SimpleNamespace(json=request_json))
                return [json.loads(chunk.rstrip(b"\0")) async for chunk in response.body_iterator]

            frames = asyncio.run(collect())
        expected = hashlib.sha256(b"[10,11]").hexdigest()
        self.assertEqual(frames[-1]["output_token_ids_sha256"], expected)
        self.assertEqual(frames[-1]["output_tokens"], 2)
        self.assertIn("finished_after_received_s", frames[-1]["server_timing"])


if __name__ == "__main__":
    unittest.main()
