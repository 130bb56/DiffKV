"""CPU-only checks for fair online workload timing and comparison validity."""

import asyncio
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from run_online_poc import arrival_offsets, compare, digest, distribution, make_connector, measure, request, summarize, validate_trace


def trace(mode="burst"):
    return {"schema_version": 1, "batch_limit": 2, "input_length": 2, "output_length": 2,
            "num_requests": 2, "arrival_mode": mode, "prompt_sha256": digest([11, 12]),
            "requests": [{"request_id": f"r{i}", "prompt_length": 2,
                          "prompt_token_ids": [11, 12], "max_tokens": 2,
                          "scheduled_offset_s": None if mode == "closed-loop" else i * 0.5}
                         for i in range(2)]}


def records():
    return [{"index": i, "request_id": f"r{i}", "status": "completed", "output_tokens": 2,
             "prompt_tokens": 2, "e2e_s": 2.0, "scheduled_e2e_s": 2.25,
             "ttft_ms": 1000, "scheduled_ttft_ms": 1250, "tpot_ms": 1000,
             "http_completion_s": 2.01, "send_lag_ms": 250.0}
            for i in range(2)]


def server_metadata():
    return {"completed": 2, "submitted": 2, "failed": 0, "engine_error": None, "active": 0,
            "state": {"running": 0, "waiting": 0, "swapped": 0, "pending_frontend": 0},
            "metadata": {"common_engine_args": {"max_num_seqs": 2, "max_model_len": 4},
                         "gpu_uuid": "same-device"}}


class WorkloadTests(unittest.TestCase):
    def test_poisson_schedule_is_reusable_and_independent_of_completion(self):
        first = arrival_offsets(128, "poisson", .2, 42)
        self.assertEqual(first, arrival_offsets(128, "poisson", .2, 42))
        self.assertNotEqual(first, arrival_offsets(128, "poisson", .2, 43))
        self.assertTrue(all(right > left for left, right in zip(first, first[1:])))
        self.assertEqual(arrival_offsets(4, "deterministic", .5), [0., 2., 4., 6.])
        self.assertEqual(arrival_offsets(5, "bursts", burst_size=2, burst_interval=3), [0, 0, 3, 3, 6])
        self.assertEqual(arrival_offsets(3, "closed-loop"), [None] * 3)
        for bad in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                arrival_offsets(3, "poisson", bad)

    def test_trace_rejects_shape_changes_and_reordered_arrivals(self):
        self.assertEqual(validate_trace(trace()), trace())
        changed = trace()
        changed["requests"][1]["prompt_token_ids"] = [11, 13]
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_trace(changed)
        changed = trace()
        changed["requests"][1]["scheduled_offset_s"] = -1
        with self.assertRaisesRegex(ValueError, "arrival"):
            validate_trace(changed)
        changed = trace("closed-loop")
        changed["requests"][0]["scheduled_offset_s"] = 0
        with self.assertRaisesRegex(ValueError, "Closed loop"):
            validate_trace(changed)

    def test_failed_requests_cannot_disappear_from_denominator(self):
        result = summarize(trace(), records()[:1], 5, "trace", server_metadata(), {})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["total_requests"], 2)
        self.assertEqual(result["failed"], 1)
        result = summarize(trace(), records(), 5, "trace", server_metadata(), {})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["e2e_s"]["mean"], 2)
        self.assertEqual(result["scheduled_e2e_s"]["mean"], 2.25)
        self.assertTrue(result["warnings"])

    def test_percentile_definition_is_nearest_rank(self):
        value = distribution([1, 2, 3, 4])
        self.assertEqual(value["p50"], 2)
        self.assertEqual(value["p95"], 4)
        self.assertEqual(value["mean"], 2.5)

    def test_comparison_rejects_different_trace_device_engine_or_failure(self):
        base = summarize(trace(), records(), 5, "trace", server_metadata(), {})
        fp16, diffkv = deepcopy(base), deepcopy(base)
        fp16["backend"] = "fp16"
        diffkv["backend"] = "diffkv"
        fp16["server_metrics"]["metadata"]["backend"] = "fp16"
        diffkv["server_metrics"]["metadata"].update(backend="diffkv", diffkv_variant="no-sync-bulk-lengths")
        fp16["e2e_s"]["mean"] = 2.8
        with tempfile.TemporaryDirectory() as directory:
            f, d = Path(directory) / "fp16.json", Path(directory) / "diffkv.json"
            f.write_text(json.dumps(fp16))
            d.write_text(json.dumps(diffkv))
            result = compare(f, d)
            self.assertTrue(result["comparison_valid"])
            self.assertAlmostEqual(result["speedup_fp16_over_diffkv"]["e2e_s"]["mean"], 1.4)
            for key, bad in (("trace_sha256", "other"), ("status", "failed")):
                changed = deepcopy(diffkv)
                changed[key] = bad
                d.write_text(json.dumps(changed))
                self.assertFalse(compare(f, d)["comparison_valid"])
            for key, bad in (("gpu_uuid", "other-device"), ("common_engine_args", {"max_num_seqs": 1}),
                             ("backend", "fp16"), ("diffkv_variant", "original")):
                changed = deepcopy(diffkv)
                changed["server_metrics"]["metadata"][key] = bad
                d.write_text(json.dumps(changed))
                self.assertFalse(compare(f, d)["comparison_valid"])
            changed = deepcopy(diffkv)
            changed["server_metrics"]["state"]["waiting"] = 1
            d.write_text(json.dumps(changed))
            self.assertFalse(compare(f, d)["comparison_valid"])
            changed = deepcopy(diffkv)
            changed["run_metadata"]["client_protocol_version"] = 2
            changed["run_metadata"]["transport_policy"] = "fresh-tcp-connection-per-request"
            changed["run_metadata"]["client_force_close"] = True
            d.write_text(json.dumps(changed))
            self.assertFalse(compare(f, d)["checks"]["same_transport_protocol"])


class FakeContent:
    def __init__(self, chunks):
        self.chunks = chunks

    async def iter_any(self):
        for chunk in self.chunks:
            await asyncio.sleep(0)
            yield chunk


class FakeResponse:
    status = 200

    def __init__(self, chunks):
        self.content = FakeContent(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class FakeSession:
    def __init__(self, chunks):
        self.chunks = chunks
        self.payload = None

    def post(self, url, json):
        self.payload = json
        return FakeResponse(self.chunks)


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(importlib.util.find_spec("aiohttp"), "aiohttp available in benchmark Python environment")
    async def test_sequential_requests_use_new_connections_and_failure_is_not_retried(self):
        import aiohttp
        from aiohttp import web

        transports, received_ids, connection_headers = [], [], []
        async def generate(incoming):
            payload = await incoming.json()
            transports.append(incoming.transport)
            received_ids.append(payload["request_id"])
            connection_headers.append(incoming.headers.get("Connection"))
            if payload["request_id"] == "failure":
                incoming.transport.abort()
                return web.Response()
            frames = [{"request_id": payload["request_id"], "prompt_tokens": 2,
                       "output_tokens": count, "finished": count == 2,
                       **({"finish_reason": "length", "text": "ok"} if count == 2 else {})}
                      for count in (1, 2)]
            return web.Response(body=b"".join(json.dumps(frame).encode() + b"\0" for frame in frames))
        app = web.Application()
        app.router.add_post("/generate", generate)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            connector = make_connector()
            self.assertTrue(connector.force_close)
            self.assertEqual(connector.limit, 0)
            self.assertEqual(connector.limit_per_host, 0)
            async with aiohttp.ClientSession(connector=connector,
                                             timeout=aiohttp.ClientTimeout(total=2)) as session:
                for identity in ("first", "second"):
                    value, error = await request(session, f"http://127.0.0.1:{port}",
                                                 trace()["requests"][0], 0, identity)
                    self.assertIsNone(error)
                    self.assertEqual(value["output_tokens"], 2)
                partial, error = await request(session, f"http://127.0.0.1:{port}",
                                               trace()["requests"][0], 0, "failure")
                self.assertIsNotNone(error)
                self.assertEqual(partial["output_tokens"], 0)
                self.assertNotIn("e2e_s", partial)
            self.assertEqual(received_ids, ["first", "second", "failure"])
            self.assertEqual(len({id(transport) for transport in transports}), 3)
            self.assertEqual(connection_headers, ["close", "close", "close"])
        finally:
            await runner.cleanup()

    async def test_fragmented_frames_and_exact_token_length(self):
        row = trace()["requests"][0]
        frames = [
            {"request_id": "r0", "prompt_tokens": 2, "output_tokens": 1, "finished": False},
            {"request_id": "r0", "prompt_tokens": 2, "output_tokens": 2, "finished": True,
             "text": "ok", "finish_reason": "length", "output_token_ids_sha256": "hash",
             "server_timing": {"first_scheduled_after_received_s": .1}},
        ]
        encoded = b"".join(json.dumps(frame).encode() + b"\0" for frame in frames)
        session = FakeSession([encoded[:17], encoded[17:93], encoded[93:]])
        result, error = await request(session, "http://unused", row, 0)
        self.assertIsNone(error)
        self.assertEqual(result["output_tokens"], 2)
        self.assertEqual(result["output_token_ids_sha256"], "hash")
        self.assertEqual(result["server_timing"]["first_scheduled_after_received_s"], .1)
        self.assertTrue(session.payload["ignore_eos"])
        self.assertEqual(session.payload["prompt_token_ids"], [11, 12])
        frames[-1]["output_tokens"] = 1
        truncated = b"".join(json.dumps(frame).encode() + b"\0" for frame in frames)
        partial, error = await request(FakeSession([truncated]), "http://unused", row, 0)
        self.assertIn("Expected 2 output tokens", error)
        self.assertNotIn("e2e_s", partial)

    @unittest.skipUnless(importlib.util.find_spec("aiohttp"), "aiohttp available in benchmark Python environment")
    async def test_real_http_open_loop_dispatch_does_not_wait_for_prior_completion(self):
        from aiohttp import web

        state = {"submitted": 0, "completed": 0, "active": 0, "peak_active": 0}
        async def health(_):
            return web.json_response({"status": "ready", "backend": "fp16"})
        async def metrics(_):
            return web.json_response({**server_metadata(), **state})
        async def reset(_):
            return web.json_response({"status": "reset"})
        async def generate(incoming):
            payload = await incoming.json()
            state["submitted"] += 1
            state["active"] += 1
            state["peak_active"] = max(state["active"], state["peak_active"])
            response = web.StreamResponse()
            await response.prepare(incoming)
            for count in (1, 2):
                await asyncio.sleep(.05)
                frame = {"request_id": payload["request_id"], "prompt_tokens": 2,
                         "output_tokens": count, "finished": count == 2}
                if count == 2:
                    frame.update(text="ok", finish_reason="length")
                await response.write(json.dumps(frame).encode() + b"\0")
            state["active"] -= 1
            state["completed"] += 1
            await response.write_eof()
            return response
        app = web.Application()
        app.add_routes([web.get("/health", health), web.get("/metrics", metrics),
                        web.post("/reset", reset), web.post("/generate", generate)])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                shared = trace("deterministic")
                shared["requests"][1]["scheduled_offset_s"] = .005
                source = folder / "trace.json"
                source.write_text(json.dumps(shared))
                args = argparse.Namespace(trace=source, output_dir=folder / "run", resume=False,
                                          backend="fp16", url=f"http://127.0.0.1:{port}",
                                          timeout_seconds=5, warmup_tokens=0)
                result = await measure(args, shared)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["run_metadata"]["client_protocol_version"], 2)
                self.assertTrue(result["run_metadata"]["client_force_close"])
                self.assertEqual(result["run_metadata"]["retries"], 0)
                self.assertEqual(state["peak_active"], 2)
                first, second = result["requests"]
                self.assertLess(second["send_offset_s"], first["response_end_offset_s"])
                self.assertGreaterEqual(second["scheduled_e2e_s"], second["e2e_s"])
                self.assertEqual(len((folder / "run/requests.jsonl").read_text().splitlines()), 2)
        finally:
            await runner.cleanup()


if __name__ == "__main__":
    unittest.main()
