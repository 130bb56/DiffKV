"""Reproducible fixed-length online HTTP workload preparation and measurement.

This CPU client does not start a model, change clocks, or require a fixed actual
decode batch. B denotes the common server max_num_seqs, while actual batching,
queueing and recomputation remain observable engine behavior. An open-loop trace
is immutable and dispatched independently of response completion. Closed-loop
concurrency is a separate experiment, never described as equal arrival times.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

from run_online_e2e import NullJSONFrames, StreamObservation


ROOT = Path(__file__).resolve().parents[1]
CLIENT_PROTOCOL_VERSION = 2
TRANSPORT_POLICY = "fresh-tcp-connection-per-request"


def make_connector():
    import aiohttp

    # Avoid racing a server's idle keepalive close when reusing a pooled socket.
    # Connection establishment remains inside the recorded request E2E interval.
    return aiohttp.TCPConnector(force_close=True, limit=0, limit_per_host=0)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def distribution(values):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None,
                "p99": None, "min": None, "max": None}
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite statistic")
    return {"n": len(values), "mean": statistics.mean(values),
            **{f"p{percent}": values[math.ceil(percent / 100 * len(values)) - 1]
               for percent in (50, 95, 99)},
            "min": values[0], "max": values[-1], "percentile_method": "nearest rank"}


def arrival_offsets(count, mode, rate=None, seed=42, burst_size=1, burst_interval=1):
    if count < 1:
        raise ValueError("num_requests must be positive")
    if mode == "closed-loop":
        return [None] * count
    if mode == "burst":
        return [0.0] * count
    if mode == "bursts":
        if burst_size < 1 or not math.isfinite(burst_interval) or burst_interval <= 0:
            raise ValueError("bursts requires a positive size and finite positive interval")
        return [(index // burst_size) * burst_interval for index in range(count)]
    if rate is None or not math.isfinite(rate) or rate <= 0:
        raise ValueError("poisson/deterministic requires a finite positive request_rate")
    if mode == "deterministic":
        return [index / rate for index in range(count)]
    if mode != "poisson":
        raise ValueError(f"Unknown arrival mode: {mode}")
    rng = random.Random(seed)
    offsets = [0.0]
    for _ in range(1, count):
        offsets.append(offsets[-1] + rng.expovariate(rate))
    return offsets


def prepare(args):
    source = json.loads(args.inputs.read_text())
    seed_ids = source["seed_token_ids"]
    if not seed_ids or any(type(token) is not int or token < 0 for token in seed_ids):
        raise ValueError("Input manifest needs valid seed_token_ids")
    if args.isl < 1 or args.osl < 1 or args.batch_limit < 1:
        raise ValueError("ISL, OSL, and batch_limit must be positive")
    concurrency = args.concurrency if args.concurrency is not None else args.batch_limit
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    tokens = (seed_ids * math.ceil(args.isl / len(seed_ids)))[:args.isl]
    offsets = arrival_offsets(args.num_requests, args.arrival, args.request_rate,
                              args.seed, args.burst_size, args.burst_interval)
    trace = {
        "schema_version": 1, "purpose": "fixed-length online serving PoC",
        "batch_limit": args.batch_limit, "batch_definition": "common server max_num_seqs; actual batch is observed",
        "input_length": args.isl, "output_length": args.osl,
        "num_requests": args.num_requests, "arrival_mode": args.arrival,
        "request_rate": args.request_rate if args.arrival in ("poisson", "deterministic") else None,
        "burst_size": args.burst_size if args.arrival == "bursts" else None,
        "burst_interval_s": args.burst_interval if args.arrival == "bursts" else None,
        "closed_loop_concurrency": concurrency if args.arrival == "closed-loop" else None,
        "arrival_seed": args.seed, "prompt_source": str(args.inputs.resolve()),
        "prompt_source_sha256": file_digest(args.inputs), "prompt_sha256": digest(tokens),
        "sampling": {"temperature": 0.0, "ignore_eos": True, "stop": [], "n": 1},
        "requests": [{"request_id": f"r{index:06d}", "prompt_token_ids": tokens,
                      "prompt_length": args.isl, "max_tokens": args.osl,
                      "scheduled_offset_s": offset}
                     for index, offset in enumerate(offsets)],
    }
    validate_trace(trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if json.loads(args.output.read_text()) != trace:
            raise FileExistsError("Refusing to replace a different shared trace")
    else:
        save(args.output, trace)
    print(json.dumps({"trace": str(args.output.resolve()), "trace_sha256": file_digest(args.output),
                      "num_requests": len(offsets), "last_arrival_s": offsets[-1]}), flush=True)


def validate_trace(trace):
    requests = trace["requests"]
    if any(type(trace.get(key)) is not int or trace[key] < 1
           for key in ("num_requests", "batch_limit", "input_length", "output_length")):
        raise ValueError("Workload counts and lengths must be positive integers")
    if len(requests) != trace["num_requests"] or not requests:
        raise ValueError("Request count mismatch")
    if trace["arrival_mode"] not in ("burst", "bursts", "deterministic", "poisson", "closed-loop"):
        raise ValueError("Unsupported arrival mode")
    ids = set()
    previous = 0.0
    for row in requests:
        if row["request_id"] in ids:
            raise ValueError("Duplicate request ID")
        ids.add(row["request_id"])
        tokens = row["prompt_token_ids"]
        if (len(tokens) != trace["input_length"] or row["prompt_length"] != trace["input_length"]
                or row["max_tokens"] != trace["output_length"]
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError("Fixed request shape mismatch")
        if digest(tokens) != trace["prompt_sha256"]:
            raise ValueError("Prompt token hash mismatch")
        offset = row["scheduled_offset_s"]
        if trace["arrival_mode"] == "closed-loop":
            if offset is not None:
                raise ValueError("Closed loop does not have predetermined arrivals")
        else:
            if (type(offset) not in (float, int) or not math.isfinite(offset)
                    or offset < previous):
                raise ValueError("Invalid or unordered arrival offset")
            previous = offset
    return trace


async def endpoint(session, url, method="GET"):
    async with session.request(method, url) as response:
        value = await response.json()
        if response.status != 200:
            raise RuntimeError(f"{method} {url}: HTTP {response.status}: {str(value)[:1000]}")
        return value


async def request(session, url, row, sent, request_id=None):
    request_id = request_id or row["request_id"]
    observation = StreamObservation(request_id, row["prompt_length"], row["max_tokens"], sent)
    parser = NullJSONFrames()
    evidence = {}
    payload = {"request_id": request_id, "prompt": None,
               "prompt_token_ids": row["prompt_token_ids"], "max_tokens": row["max_tokens"],
               "temperature": 0.0, "ignore_eos": True, "stop": [], "n": 1, "stream": True}
    try:
        async with session.post(url + "/generate", json=payload) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:2000]}")
            async for chunk in response.content.iter_any():
                arrived = time.perf_counter()
                for frame in parser.feed(chunk):
                    observation.observe(frame, arrived)
                    for key, value in frame.items():
                        if key.startswith("server_") or key in ("output_token_ids", "output_sha256", "output_token_ids_sha256"):
                            evidence[key] = value
            parser.finish()
        return {**observation.metrics(), **evidence}, None
    except Exception as error:
        return {**observation.partial_metrics(), **evidence}, f"{type(error).__name__}: {error}"


def summarize(trace, records, duration, trace_sha, server_metrics, metadata):
    records = sorted(records, key=lambda row: row["index"])
    completed = [row for row in records if row["status"] == "completed"]
    expected_ids = [row["request_id"] for row in trace["requests"]]
    checks = {
        "all_requests_recorded": len(records) == trace["num_requests"],
        "request_ids_match": [row["request_id"] for row in records] == expected_ids,
        "all_requests_completed": len(completed) == trace["num_requests"],
        "fixed_lengths": all(row.get("prompt_tokens") == trace["input_length"]
                             and row.get("output_tokens") == trace["output_length"] for row in records),
        "server_metrics_available": isinstance(server_metrics, dict) and "metrics_error" not in server_metrics,
    }
    if checks["server_metrics_available"]:
        checks["server_completed_matches"] = server_metrics.get("completed") == len(completed)
        checks["server_submitted_matches"] = server_metrics.get("submitted") == trace["num_requests"]
        checks["server_no_failed_requests"] = server_metrics.get("failed") == 0
        checks["server_no_engine_error"] = server_metrics.get("engine_error") is None
    token_count = sum(row.get("output_tokens", 0) for row in completed)
    result = {"schema_version": 1, "status": "completed" if all(checks.values()) else "failed",
              "trace_sha256": trace_sha, "workload": {key: value for key, value in trace.items() if key != "requests"},
              "total_requests": trace["num_requests"], "completed": len(completed),
              "failed": trace["num_requests"] - len(completed), "generated_tokens": token_count,
              "duration_s": duration, "request_throughput": len(completed) / duration,
              "output_token_throughput": token_count / duration, "checks": checks,
              "server_metrics": server_metrics, "run_metadata": metadata, "requests": records,
              "definitions": {
                  "e2e_s": "Last generated-token HTTP arrival minus actual client send; includes server queueing and transport",
                  "scheduled_e2e_s": "Last generated-token HTTP arrival minus predetermined arrival; includes client send lag; open-loop only",
                  "http_completion_s": "HTTP stream end minus actual client send",
                  "duration_s": "First scheduled arrival (or closed-loop start) through all HTTP responses drained",
                  "request_throughput": "Successfully completed requests divided by complete arrival-and-drain duration",
                  "server_timing": "Server-local durations, used to decompose queueing; client/server clock subtraction is not used",
              }}
    for key in ("e2e_s", "scheduled_e2e_s", "ttft_ms", "scheduled_ttft_ms", "tpot_ms", "http_completion_s"):
        result[key] = distribution(row.get(key) for row in completed)
    result["send_lag_ms"] = distribution(row.get("send_lag_ms") for row in records)
    result["warnings"] = []
    if result["send_lag_ms"]["max"] is not None and result["send_lag_ms"]["max"] > 100:
        result["warnings"].append("Client send lag exceeded 100 ms; inspect scheduled-origin latency and arrival adherence")
    return result


async def measure(args, trace):
    import aiohttp

    trace_sha = file_digest(args.trace)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if args.resume and summary_path.exists():
        old = json.loads(summary_path.read_text())
        if old["trace_sha256"] == trace_sha and old["status"] == "completed":
            print(json.dumps({"resumed": str(summary_path), "status": "completed"}), flush=True)
            return old
        raise RuntimeError("Existing result is incomplete or belongs to another trace; use a fresh output directory")
    if any(output.iterdir()):
        raise FileExistsError("Measurement output directory must be empty; use --resume only for a completed run")
    metadata = {"backend_label": args.backend, "url": args.url, "started_utc": datetime.now(timezone.utc).isoformat(),
                "client_source_sha256": file_digest(__file__),
                "stream_parser_source_sha256": file_digest(Path(__file__).with_name("run_online_e2e.py")),
                "trace_path": str(args.trace.resolve()), "trace_sha256": trace_sha,
                "timeout_seconds": args.timeout_seconds, "warmup_output_tokens": args.warmup_tokens,
                "client_protocol_version": CLIENT_PROTOCOL_VERSION, "transport_policy": TRANSPORT_POLICY,
                "client_force_close": True, "client_connection_limit": 0,
                "client_limit_per_host": 0, "retries": 0, "command": sys.argv}
    save(output / "run.json", metadata)
    timeout = aiohttp.ClientTimeout(total=args.timeout_seconds)
    records = []
    event_file = (output / "events.jsonl").open("w")
    completed_file = (output / "requests.jsonl").open("w")
    def journal(value):
        event_file.write(json.dumps(value, allow_nan=False) + "\n")
        event_file.flush()
    try:
        async with aiohttp.ClientSession(timeout=timeout, connector=make_connector()) as session:
            health = await endpoint(session, args.url + "/health")
            if health.get("status") != "ready":
                raise RuntimeError(f"Server not ready: {health}")
            if health.get("backend") != args.backend:
                raise RuntimeError("Server backend differs from the requested measurement label")
            before = await endpoint(session, args.url + "/metrics")
            common = before.get("metadata", {}).get("common_engine_args", {})
            if common.get("max_num_seqs") != trace["batch_limit"]:
                raise ValueError(f"Server max_num_seqs {common.get('max_num_seqs')} != trace batch_limit {trace['batch_limit']}")
            if common.get("max_model_len", 0) < trace["input_length"] + trace["output_length"]:
                raise ValueError("Server max_model_len is smaller than ISL+OSL")
            if args.warmup_tokens < 0 or trace["input_length"] + args.warmup_tokens > common["max_model_len"]:
                raise ValueError("Warmup tokens exceed the available context")
            if before.get("active", 0) or before.get("state", {}).get("running", 0):
                raise RuntimeError("Server has existing active requests")
            if args.warmup_tokens:
                warm = {**trace["requests"][0], "max_tokens": args.warmup_tokens}
                warm_result, error = await request(session, args.url, warm, time.perf_counter(), "poc-warmup")
                if error:
                    raise RuntimeError("Warmup failed: " + error)
                save(output / "warmup.json", warm_result)
            reset = await endpoint(session, args.url + "/reset", "POST")
            if reset.get("status") != "reset":
                raise RuntimeError("Server did not reset")
            save(output / "server-before.json", reset)
            start = time.perf_counter()
            stopped = asyncio.Event()
            submitted = 0
            async def send_one(index, row):
                nonlocal submitted
                offset = row["scheduled_offset_s"]
                if offset is not None:
                    await asyncio.sleep(max(0, start + offset - time.perf_counter()))
                sent = time.perf_counter()
                record = {"index": index, "request_id": row["request_id"], "scheduled_offset_s": offset,
                          "send_offset_s": sent - start,
                          "send_lag_ms": (sent - start - offset) * 1000 if offset is not None else None}
                submitted += 1
                journal({"event": "send", **record})
                values, error = await request(session, args.url, row, sent)
                end = time.perf_counter()
                record.update(values)
                record.update(status="failed" if error else "completed", error=error,
                              response_end_offset_s=end - start, http_completion_s=end - sent)
                if not error:
                    record["first_token_offset_s"] = sent - start + record["ttft_ms"] / 1000
                    record["last_token_offset_s"] = sent - start + record["e2e_s"]
                    if offset is not None:
                        record["scheduled_e2e_s"] = record["last_token_offset_s"] - offset
                        record["scheduled_ttft_ms"] = (record["first_token_offset_s"] - offset) * 1000
                records.append(record)
                completed_file.write(json.dumps(record, allow_nan=False) + "\n")
                completed_file.flush()
                journal({"event": "complete", "request_id": row["request_id"], "status": record["status"],
                         "response_end_offset_s": end - start, "e2e_s": record.get("e2e_s"), "error": error})
                print(f"REQUEST {args.backend} {row['request_id']} {record['status']} "
                      f"output={record.get('output_tokens')} e2e={record.get('e2e_s')}", flush=True)
            async def progress():
                while not stopped.is_set():
                    try:
                        await asyncio.wait_for(stopped.wait(), timeout=30)
                    except asyncio.TimeoutError:
                        value = {"event": "progress", "elapsed_s": time.perf_counter() - start,
                                 "submitted": submitted, "completed": sum(r["status"] == "completed" for r in records),
                                 "failed": sum(r["status"] != "completed" for r in records)}
                        journal(value)
                        save(output / "progress.json", value)
                        print("PROGRESS " + json.dumps(value), flush=True)
            progress_task = asyncio.create_task(progress())
            try:
                if trace["arrival_mode"] == "closed-loop":
                    queue = asyncio.Queue()
                    for index, row in enumerate(trace["requests"]):
                        queue.put_nowait((index, row))
                    async def worker():
                        while not queue.empty():
                            index, row = queue.get_nowait()
                            await send_one(index, row)
                    await asyncio.gather(*(worker() for _ in range(trace["closed_loop_concurrency"])))
                else:
                    await asyncio.gather(*(send_one(index, row) for index, row in enumerate(trace["requests"])))
            finally:
                stopped.set()
                await progress_task
            duration = time.perf_counter() - start
            await asyncio.sleep(0)
            try:
                server_metrics = await endpoint(session, args.url + "/metrics")
            except Exception as error:
                server_metrics = {"metrics_error": str(error)}
            result = summarize(trace, records, duration, trace_sha, server_metrics, metadata)
            result["backend"] = args.backend
            save(summary_path, result)
            print("RESULT " + json.dumps({key: result[key] for key in (
                "backend", "status", "completed", "failed", "duration_s", "e2e_s", "scheduled_e2e_s", "request_throughput")}), flush=True)
            return result
    except BaseException as error:
        save(output / "interrupted.json", {"error": f"{type(error).__name__}: {error}",
                                           "recorded_requests": len(records), "trace_sha256": trace_sha})
        raise
    finally:
        event_file.close()
        completed_file.close()


def compare(fp16_path, diffkv_path):
    fp16, diffkv = [json.loads(Path(path).read_text()) for path in (fp16_path, diffkv_path)]
    checks = {"both_completed": fp16["status"] == diffkv["status"] == "completed",
              "same_trace": fp16["trace_sha256"] == diffkv["trace_sha256"],
              "same_output_token_total": fp16["generated_tokens"] == diffkv["generated_tokens"],
              "expected_backends": fp16.get("backend") == "fp16" and diffkv.get("backend") == "diffkv"}
    transport = [value.get("run_metadata", {}) for value in (fp16, diffkv)]
    checks["same_transport_protocol"] = all(
        transport[0].get(key, default) == transport[1].get(key, default)
        for key, default in (("client_protocol_version", 1),
                             ("transport_policy", "pooled-keepalive"),
                             ("client_force_close", False), ("client_connection_limit", 0),
                             ("client_limit_per_host", 0), ("retries", 0)))
    metadata = [value.get("server_metrics", {}).get("metadata", {}) for value in (fp16, diffkv)]
    checks["actual_server_backends"] = metadata[0].get("backend") == "fp16" and metadata[1].get("backend") == "diffkv"
    checks["diffkv_v1_variant"] = metadata[1].get("diffkv_variant") == "no-sync-bulk-lengths"
    for backend, value in (("fp16", fp16), ("diffkv", diffkv)):
        observed = value.get("server_metrics", {})
        state = observed.get("state", {})
        checks[f"{backend}_fully_drained"] = observed.get("active") == 0 and all(
            state.get(key) == 0 for key in ("running", "waiting", "swapped", "pending_frontend"))
    for key in ("common_engine_args", "gpu_uuid"):
        checks[f"same_{key}"] = bool(metadata[0].get(key)) and metadata[0].get(key) == metadata[1].get(key)
    ratios = {}
    for key in ("e2e_s", "scheduled_e2e_s", "ttft_ms", "tpot_ms", "http_completion_s"):
        ratios[key] = {stat: fp16[key][stat] / diffkv[key][stat]
                       if fp16[key][stat] is not None and diffkv[key][stat] else None
                       for stat in ("mean", "p50", "p95", "p99")}
    ratios["makespan"] = fp16["duration_s"] / diffkv["duration_s"]
    ratios["output_token_throughput"] = diffkv["output_token_throughput"] / fp16["output_token_throughput"]
    output_hash_pairs = [(left.get("output_token_ids_sha256"), right.get("output_token_ids_sha256"))
                         for left, right in zip(fp16.get("requests", []), diffkv.get("requests", []))]
    output_hashes = {"available_pairs": sum(bool(a) and bool(b) for a, b in output_hash_pairs),
                     "equal_pairs": sum(bool(a) and a == b for a, b in output_hash_pairs),
                     "meaning": "Token identity is recorded, not assumed: quantized KV may alter outputs. Fixed lengths alone are not an accuracy validation."}
    return {"comparison_valid": all(checks.values()), "checks": checks,
            "speedup_fp16_over_diffkv": ratios,
            "output_hash_comparison": output_hashes,
            "same_client_source": fp16.get("run_metadata", {}).get("client_source_sha256") == diffkv.get("run_metadata", {}).get("client_source_sha256"),
            "fp16": str(Path(fp16_path).resolve()), "diffkv": str(Path(diffkv_path).resolve()),
            "interpretation": "Latency ratios above 1 favor DiffKV. Open-loop scheduled latency includes client dispatch lag. Burst and finite trace makespan include drain; this is not steady-state throughput."}


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare")
    create.add_argument("--inputs", type=Path, default=ROOT / "logs/performance/llama-matched-inputs.json")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--isl", type=int, required=True)
    create.add_argument("--osl", type=int, required=True)
    create.add_argument("--batch-limit", type=int, required=True)
    create.add_argument("--num-requests", type=int, required=True)
    create.add_argument("--arrival", choices=("burst", "bursts", "poisson", "deterministic", "closed-loop"), required=True)
    create.add_argument("--request-rate", type=float)
    create.add_argument("--seed", type=int, default=42)
    create.add_argument("--burst-size", type=int, default=8)
    create.add_argument("--burst-interval", type=float, default=10)
    create.add_argument("--concurrency", type=int)
    run = sub.add_parser("measure")
    run.add_argument("--trace", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--url", required=True)
    run.add_argument("--backend", choices=("fp16", "diffkv"), required=True)
    run.add_argument("--timeout-seconds", type=float, default=3600)
    run.add_argument("--warmup-tokens", type=int, default=16)
    run.add_argument("--resume", action="store_true")
    pair = sub.add_parser("compare")
    pair.add_argument("--fp16", type=Path, required=True)
    pair.add_argument("--diffkv", type=Path, required=True)
    pair.add_argument("--output", type=Path)
    return root


def main():
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "measure":
        trace = validate_trace(json.loads(args.trace.read_text()))
        result = asyncio.run(measure(args, trace))
        if result["status"] != "completed":
            sys.exit(2)
    else:
        result = compare(args.fp16, args.diffkv)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            save(args.output, result)
        print(json.dumps(result, indent=2))
        if not result["comparison_valid"]:
            sys.exit(2)


if __name__ == "__main__":
    main()
