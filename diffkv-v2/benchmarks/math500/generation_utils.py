# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only input validation and measurements for the MATH-500 runs."""

import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"
PROBLEMS = {
    50: HERE / "problems/math500_sample50.json",
    500: HERE / "problems/math500_all.json",
}
PROBLEM_SET_NAMES = {50: "sample50", 500: "all500"}
OUTPUT_LENGTHS = HERE / "serving/offline/output_lengths_sample50.json"
BACKEND_OPTIONS = {
    "fp16": {},
    "diffkv": {"attention_backend": "DIFFKV_V2"},
    "fp8": {"kv_cache_dtype": "fp8_e4m3", "attention_backend": "FLASHINFER"},
}


def backend_options(backend):
    return dict(BACKEND_OPTIONS[backend])


def cache_metadata(backend):
    options = backend_options(backend)
    return {
        "kv_cache_dtype": options.get("kv_cache_dtype", "auto"),
        "storage_format": {
            "fp16": "float16",
            "diffkv": "K8V4/K4V2",
            "fp8": "fp8_e4m3",
        }[backend],
        "attention_backend": options.get("attention_backend", "auto"),
        "scale_handling": {
            "fp16": "unquantized",
            "diffkv": "DiffKV per-token asymmetric quantization",
            "fp8": "checkpoint per-tensor K/V scales, default 1.0 when absent",
        }[backend],
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_run_dir(category, name):
    """Return a new timestamped run directory under logs/<category>."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return LOG_DIR / category / f"{stamp}-{name}"


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_workload(path, expected_count):
    workload = read_json(path)
    rows = workload["requests"]
    if expected_count not in (50, 500) or len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} requests, found {len(rows)}")
    seen = set()
    for row in rows:
        request_id = row["request_id"]
        if not isinstance(request_id, str) or not request_id or request_id in seen:
            raise ValueError("Request IDs must be unique nonempty strings")
        seen.add(request_id)
        tokens = row["prompt_token_ids"]
        if not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError(f"Invalid prompt token IDs: {request_id}")
        if row.get("prompt_tokens", len(tokens)) != len(tokens):
            raise ValueError(f"Prompt token count differs: {request_id}")
        if not isinstance(row["answer"], str) or not row["answer"]:
            raise ValueError(f"Missing reference answer: {request_id}")
    return workload


def request_signature(workload):
    rows = [
        [row["request_id"], row["prompt_token_ids"], row["answer"]]
        for row in workload["requests"]
    ]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def output_limits(workload, workload_path, mode, reference_path, max_tokens=8192):
    requests = workload["requests"]
    if mode == "natural":
        if reference_path is not None:
            raise ValueError("Natural generation must not use fixed output lengths")
        return {row["request_id"]: max_tokens for row in requests}
    if reference_path is None:
        raise ValueError("length_replay requires --lengths-from")
    reference = read_json(reference_path)
    metadata = reference.get("metadata", reference)
    if metadata.get("backend") != "fp16" or metadata.get("mode") != "natural":
        raise ValueError("Lengths must come from a natural FP16 reference")
    if reference.get("status") != "completed":
        raise ValueError("Length reference is incomplete")
    hashes = {sha256(workload_path), workload.get("source_workload_sha256")}
    if metadata.get("workload_sha256") not in hashes - {None}:
        raise ValueError("Length reference uses another workload")
    if metadata.get("request_signature", request_signature(workload)) != (
        request_signature(workload)
    ):
        raise ValueError("Length reference uses another workload")
    rows = reference["requests"]
    limits = {row["request_id"]: row["output_tokens"] for row in rows}
    if len(limits) != len(rows) or set(limits) != {
        row["request_id"] for row in requests
    }:
        raise ValueError("Length reference request IDs differ")
    if any(
        type(count) is not int or not 0 < count <= max_tokens
        for count in limits.values()
    ):
        raise ValueError("Invalid fixed output length")
    return limits


def distribution(values):
    values = sorted(values)
    if not values:
        return {key: None for key in ("mean", "p50", "p95", "p99", "min", "max")}
    return {
        "mean": statistics.mean(values),
        **{
            f"p{p}": values[max(0, math.ceil(len(values) * p / 100) - 1)]
            for p in (50, 95, 99)
        },
        "min": values[0],
        "max": values[-1],
    }


class OutputRecorder:
    """Observe cumulative engine outputs without changing token generation."""

    def __init__(self, requests, limits, mode):
        self.requests = {row["request_id"]: row for row in requests}
        self.limits = limits
        self.mode = mode
        self.counts = {}
        self.first = {}
        self.last = {}
        self.completed = {}

    def observe(self, output, elapsed_s):
        request_id = output.request_id
        if request_id not in self.requests or request_id in self.completed:
            raise ValueError(f"Unknown or already completed request: {request_id}")
        if len(output.outputs) != 1:
            raise ValueError("Expected exactly one completion per request")
        completion = output.outputs[0]
        tokens = list(completion.token_ids)
        count = len(tokens)
        previous = self.counts.get(request_id, 0)
        if count < previous or count > self.limits[request_id]:
            raise ValueError(f"Invalid cumulative output length: {request_id}")
        self.counts[request_id] = count
        if count > previous:
            self.first.setdefault(request_id, elapsed_s)
            self.last[request_id] = elapsed_s
        if not output.finished:
            return
        if count == 0:
            raise ValueError(f"Empty final output: {request_id}")
        if self.mode == "length_replay" and count != self.limits[request_id]:
            raise ValueError(f"Fixed-length request stopped early: {request_id}")
        self.completed[request_id] = {
            "request_id": request_id,
            "prompt_tokens": len(self.requests[request_id]["prompt_token_ids"]),
            "output_tokens": count,
            "output_token_ids": tokens,
            "output_text": completion.text,
            "finish_reason": completion.finish_reason,
            "stop_reason": completion.stop_reason,
            "max_tokens": self.limits[request_id],
            "ttft_s": self.first[request_id],
            "tpot_s": (
                (self.last[request_id] - self.first[request_id]) / (count - 1)
                if count > 1
                else None
            ),
            "last_token_s": self.last[request_id],
            "e2e_s": elapsed_s,
            "num_preemptions": getattr(output.metrics, "num_preemptions", None),
        }

    def rows(self):
        if set(self.completed) != set(self.requests):
            raise ValueError("Engine stopped before every request completed")
        return [self.completed[request_id] for request_id in self.requests]


def summarize(rows, elapsed_s):
    if not rows or not math.isfinite(elapsed_s) or elapsed_s <= 0:
        raise ValueError("Expected completed requests and positive elapsed time")
    tokens = sum(row["output_tokens"] for row in rows)
    return {
        "completed": len(rows),
        "generated_tokens": tokens,
        "elapsed_s": elapsed_s,
        "output_tokens_per_s": tokens / elapsed_s,
        "requests_per_s": len(rows) / elapsed_s,
        **{
            key: distribution([row[key] for row in rows if row[key] is not None])
            for key in ("ttft_s", "tpot_s", "e2e_s", "output_tokens")
        },
    }
