"""Offline synthetic FP16/DiffKV comparison with fixed ISL, OSL, and decode B.

Default grid: ISL/OSL in [512, 1024, 2048], B in [8, 16, 32].
Each backend/case runs in a fresh process. Capacity failures are reported without
reducing B or allowing KV preemption. No serving arrival process is simulated.
"""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import fcntl
import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import signal
import shlex
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = [512, 1024, 2048]
BATCHES = [8, 16, 32]
DEFAULT_PYTHON = ROOT / ".cache/baselines/fp16-env/bin/python"

FP16_REPO = ROOT / ".cache/baselines/vllm-1db83e3"
FP16_COMMIT = "1db83e31a2468cae37f326a642c0a4c4edbb5e4f"
BITS = [8, 4, 4, 2]
GROUPS = [1, 1, 1, 1]
THRESHOLDS = [0.02, 1.0]


TERMINAL_STATUSES = {"completed", "capacity_limited", "oom", "failed", "interrupted", "skipped"}
METRICS = (
    ("TTFT (ms)", "ttft_ms"),
    ("TPOT (ms/token)", "tpot_ms"),
    ("Prefill Throughput (tok/s)", "prefill_tokens_s"),
    ("Decode Throughput (tok/s)", "decode_tokens_s"),
    ("E2E Latency (s)", "elapsed_s"),
)


def _cell(value):
    """Keep arbitrary metadata and failure messages inside a Markdown cell."""
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _metric(result, key, selected):
    if not selected:
        return "not selected"
    if result is None:
        return "not run"
    status = result.get("status", "unknown")
    if status in ("capacity_limited", "oom", "OOM"):
        return "OOM"
    if status != "completed":
        return _cell(status if status in TERMINAL_STATUSES else "interrupted")
    value = result.get(key)
    if isinstance(value, dict):
        value = value.get("mean")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f"{value:.3f}"
    return "N/A"


def _numeric_metric(result, key, selected):
    if not selected or not result or result.get("status") != "completed":
        return None
    value = result.get(key)
    if isinstance(value, dict):
        value = value.get("mean")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def _comparison_cell(results_by_backend, key, backends):
    names = ("fp16", "diffkv")
    cells = [_metric(results_by_backend.get(name), key, name in backends) for name in names]
    values = [_numeric_metric(results_by_backend.get(name), key, name in backends) for name in names]
    if all(value is not None for value in values):
        best = max(values) if key in ("prefill_tokens_s", "decode_tokens_s") else min(values)
        cells = [f"**{cell}**" if value == best else cell
                 for cell, value in zip(cells, values)]
    return " / ".join(cells)


class EvaluationReport:
    """Append one run section and one comparison row per completed case.

    ``metadata['backends']`` specifies the selected workers. Results passed to
    ``record`` are keyed by backend name. ``force=True`` also records unfinished
    or missing workers, for example when the run is interrupted.
    """

    def __init__(self, path, metadata, run_id):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)
        self.run_id = str(run_id)
        self.backends = tuple(dict.fromkeys(metadata.get("backends", ("fp16", "diffkv"))))
        self.reported = set()
        self.errors = []
        self.finished = False
        command = metadata.get("command", "")
        if isinstance(command, (list, tuple)):
            command = shlex.join(map(str, command))
        cases = metadata.get("cases", ())
        section_title = metadata.get("section_title")
        title = f"{_cell(section_title)}, Run {_cell(self.run_id)}" if section_title else f"Run {_cell(self.run_id)}"
        extra_metadata = ""
        for key, label in (("diffkv_variant", "DiffKV variant"),
                           ("physical_gpus", "Physical GPUs"),
                           ("gpu_mapping", "GPU mapping")):
            if key in metadata:
                value = metadata[key]
                if isinstance(value, dict):
                    value = ", ".join(f"{name}: {gpu}" for name, gpu in value.items())
                extra_metadata += f"- {label}: {_cell(value)}\n"

        def values(name, case_key):
            explicit = metadata.get(name)
            if explicit is not None:
                return list(explicit)
            return sorted({case[case_key] for case in cases}, reverse=True)

        header = (
            f"\n\n## {title}\n\n"
            f"- Started: {_cell(metadata.get('started_at', self.run_id))}\n"
            f"- Model: {_cell(metadata.get('model', 'unknown'))}\n"
            f"- GPU: {_cell(metadata.get('gpu', 'unknown'))}\n"
            f"{extra_metadata}"
            f"- ISL: {_cell(values('isls', 'input_length'))}, "
            f"OSL: {_cell(values('osls', 'output_length'))}, "
            f"B: {_cell(values('batches', 'batch'))}\n"
            f"- Measured batches per case: {_cell(metadata.get('measured_batches_per_case', metadata.get('repeats', 1)))}\n"
            f"- Warmup output length: {_cell(metadata.get('warmup_output_length', metadata.get('warmup_tokens', 0)))}\n"
            f"- Logs: {_cell(metadata.get('log_directory', metadata.get('output_dir', 'unknown')))}\n"
            f"- Command: {_cell(command)}\n\n"
            "Each cell is **FP16 / DiffKV**. TTFT and TPOT are request means, "
            "averaged across measured batches. Among pairs of completed numeric results, "
            "lower latency and higher throughput are **bold**; exact ties are both bold.\n\n"
            "Prefill Throughput = input tokens / summed prefill engine.step time. "
            "Decode Throughput = B * (OSL - 1) / summed decode engine.step time; "
            "the first token is generated during prefill. "
            "Phase times include scheduling, model execution, sampling and detokenization. "
            "E2E Latency measures the complete generation call, including request "
            "registration and cleanup, excluding model loading and warmup. "
            "OOM means insufficient memory for the requested fixed workload.\n\n"
            "| Case (ISL, OSL, B) | " + " | ".join(label for label, _ in METRICS) + " |\n"
            "|---|" + "---:|" * len(METRICS) + "\n"
        )
        self._append(header)

    def _append(self, text):
        with self.path.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        print(text, end="", flush=True)

    def record(self, case, results_by_backend, force=False):
        if self.finished:
            raise RuntimeError("Cannot append a case after finishing the report")
        case_id = case.get("id", (case["input_length"], case["output_length"], case["batch"]))
        if case_id in self.reported:
            return None
        if not force and any(
            results_by_backend.get(backend, {}).get("status") not in TERMINAL_STATUSES
            for backend in self.backends
        ):
            return None
        label = f"({case['input_length']}, {case['output_length']}, {case['batch']})"
        cells = [
            _comparison_cell(results_by_backend, key, self.backends)
            for _, key in METRICS
        ]
        row = "| " + " | ".join([label, *cells]) + " |\n"
        self._append(row)
        self.reported.add(case_id)
        for backend in self.backends:
            result = results_by_backend.get(backend)
            if result and result.get("status") in ("failed", "interrupted") and result.get("error"):
                self.errors.append(f"- {label}, {backend}: {_cell(result['error'])}\n")
        return row

    def finish(self, status, counts=None):
        if self.finished:
            return
        footer = f"\nRun status: {_cell(status)}. Recorded cases: {len(self.reported)}.\n"
        if counts:
            footer += "\nWorker results: " + ", ".join(
                f"{_cell(key)}={_cell(value)}" for key, value in counts.items()
            ) + ".\n"
        if self.errors:
            footer += "\nFailure details:\n\n" + "".join(self.errors)
        self._append(footer)
        self.finished = True

def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def runtime_metadata():
    versions, paths = {}, {}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "xformers",
                 "flash-attn", "triton", "vllm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    for name in ("torch", "transformers", "triton", "xformers"):
        module = importlib.import_module(name)
        paths[name] = str(Path(module.__file__).resolve())
    return {"package_versions": versions, "runtime_import_paths": paths}


def get_worker(engine, backend):
    workers = engine.workers if backend == "fp16" else engine.orchestrator.workers
    assert len(workers) == 1
    return workers[0]


def verify_weights(torch, worker, model_path):
    from safetensors import safe_open

    model = worker.model_runner.model
    parameters = dict(model.named_parameters())
    dtype_counts = Counter(str(p.dtype) for p in parameters.values())
    assert parameters and all(p.dtype == torch.float16 for p in parameters.values()
                              if p.is_floating_point()), dict(dtype_counts)
    name = "model.embed_tokens.weight"
    index = model_path / "model.safetensors.index.json"
    shard = (model_path / json.loads(index.read_text())["weight_map"][name]
             if index.exists() else model_path / "model.safetensors")
    with safe_open(str(shard), framework="pt", device="cpu") as checkpoint:
        saved = checkpoint.get_slice(name)[1:3, :8]
    loaded = parameters[name].detach()[1:3, :8].cpu()
    expected = saved.to(torch.float16)
    assert torch.equal(loaded, expected), "Checkpoint value mismatch"
    attention = model.model.layers[0].self_attn.attn
    return {
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "model_source": inspect.getfile(type(model)),
        "attention_class": f"{type(attention).__module__}.{type(attention).__name__}",
        "attention_source": inspect.getfile(type(attention)),
        "parameter_dtype_counts": dict(dtype_counts),
        "parameter_count": sum(p.numel() for p in parameters.values()),
        "checkpoint_value_check": {
            "parameter": name, "shard": str(shard), "slice": "[1:3, :8]",
            "checkpoint_dtype": str(saved.dtype), "loaded_dtype": str(loaded.dtype),
            "exact_match_after_fp16_cast": True,
            "checkpoint_sample_fp16": expected.tolist(), "loaded_sample": loaded.tolist(),
        },
    }


def verify_cache(torch, worker, backend, engine):
    if backend == "fp16":
        rows = []
        for layer, (key, value) in enumerate(worker.gpu_cache):
            assert key.dtype == value.dtype == torch.float16
            assert key.is_cuda and value.is_cuda
            rows.append({"layer": layer, "key_dtype": str(key.dtype),
                         "value_dtype": str(value.dtype), "key_shape": list(key.shape),
                         "value_shape": list(value.shape), "device": str(key.device)})
        assert rows
        return {"kv_precision": "K16V16", "block_tokens": engine.cache_config.block_size,
                "kv_tensors": rows}
    cache = worker.gpu_cache
    assert cache.dtype == torch.int16 and cache.is_cuda
    config = engine.cache_config
    assert config.kv_buffer_size == 64
    page_configs = [
        {"bits_and_groups": list(key), "tokens_per_block": value}
        for key, value in sorted(config.quantized_block_num_tokens.items())
    ]
    return {
        "kv_precision": "K8V4 + K4V2", "packed_storage_dtype": str(cache.dtype),
        "packed_storage_shape": list(cache.shape), "device": str(cache.device),
        "packed_storage_note": "int16 container holds quantized KV and metadata; not K16V16",
        "block_bytes": config.block_bytes, "block_int16_elements": config.block_size,
        "page_configs": page_configs, "kv_buffer_size": config.kv_buffer_size,
    }


def verify_outputs(outputs, prompt_ids, batch, output_length):
    assert len(outputs) == batch, len(outputs)
    assert all(out.finished for out in outputs)
    assert all(out.prompt_token_ids == prompt_ids for out in outputs)
    counts = [len(out.outputs[0].token_ids) for out in outputs]
    assert counts == [output_length] * batch, counts
    assert all(len(out.outputs) == 1 for out in outputs)
    return counts


def generate(llm, sampling, prompts, backend):
    kwargs = {"sampling_params": sampling, "prompt_token_ids": prompts, "use_tqdm": False}
    if backend == "diffkv":
        kwargs.update(quant_configs=BITS, quant_groups=GROUPS, compress_configs=THRESHOLDS)
    return llm.generate(**kwargs)


def summarize_ms(values):
    ordered = sorted(values)
    return {"n": len(values), "mean": statistics.mean(values),
            "p50": statistics.median(values),
            "p95": ordered[math.ceil(0.95 * len(values)) - 1],
            "min": min(values), "max": max(values),
            "p95_method": "nearest rank"}


class RequestTimings:
    """Time request outputs and prefill/decode engine steps without extra CUDA sync."""

    def __init__(self, engine, guard):
        self.enabled = False
        self.requests = {}
        self.guard = guard
        self.phase_seconds = Counter()
        self.phase_tokens = Counter()
        original_add = engine.add_request
        original_step = engine.step

        @functools.wraps(original_add)
        def add(*args, **kwargs):
            if self.enabled:
                submitted = time.perf_counter()
                request_id = str(args[0] if args else kwargs["request_id"])
                assert request_id not in self.requests
                self.requests[request_id] = {
                    "request_id": request_id, "submit_perf_s": submitted,
                    "token_arrivals_perf_s": [], "output_count": 0,
                }
            return original_add(*args, **kwargs)

        @functools.wraps(original_step)
        def step(*args, **kwargs):
            if self.enabled:
                self.guard.last_step = None
                started = time.perf_counter()
            outputs = original_step(*args, **kwargs)
            if self.enabled:
                returned = time.perf_counter()
                scheduled = self.guard.last_step
                if scheduled is not None:
                    phase, tokens = scheduled
                    self.phase_seconds[phase] += returned - started
                    self.phase_tokens[phase] += tokens
                for output in outputs:
                    request_id = str(output.request_id)
                    row = self.requests[request_id]
                    count = len(output.outputs[0].token_ids)
                    assert count == row["output_count"] + 1, (request_id, count, row["output_count"])
                    row["output_count"] = count
                    row["token_arrivals_perf_s"].append(returned)
                    if count == 1:
                        row["first_token_perf_s"] = returned
                    if output.finished:
                        row["last_token_perf_s"] = returned
            return outputs

        engine.add_request = add
        engine.step = step

    def begin(self):
        self.requests = {}
        self.phase_seconds = Counter()
        self.phase_tokens = Counter()
        self.enabled = True

    def end(self, started, batch, input_length, output_length):
        self.enabled = False
        assert len(self.requests) == batch
        assert self.phase_tokens["prefill"] == batch * input_length
        assert self.phase_tokens["decode"] == batch * (output_length - 1)
        assert self.phase_seconds["prefill"] > 0 and self.phase_seconds["decode"] > 0
        rows = []
        for request_id in sorted(self.requests, key=int):
            row = self.requests[request_id]
            assert row["output_count"] == output_length
            assert len(row["token_arrivals_perf_s"]) == output_length
            first, last, submit = (
                row["first_token_perf_s"], row["last_token_perf_s"], row["submit_perf_s"]
            )
            assert submit <= first <= last
            row["submit_ms_from_generate_start"] = (submit - started) * 1000
            row["first_token_ms_from_generate_start"] = (first - started) * 1000
            row["last_token_ms_from_generate_start"] = (last - started) * 1000
            row["token_arrivals_ms_from_submit"] = [
                (stamp - submit) * 1000 for stamp in row["token_arrivals_perf_s"]
            ]
            row["ttft_ms"] = (first - submit) * 1000
            row["tpot_ms"] = (last - first) * 1000 / (output_length - 1)
            row["request_latency_ms"] = (last - submit) * 1000
            rows.append(row)
        return {
            "requests": rows,
            "ttft_ms": summarize_ms([row["ttft_ms"] for row in rows]),
            "tpot_ms": summarize_ms([row["tpot_ms"] for row in rows]),
            "request_latency_ms": summarize_ms([row["request_latency_ms"] for row in rows]),
            "prefill_s": self.phase_seconds["prefill"],
            "decode_s": self.phase_seconds["decode"],
            "prefill_tokens": self.phase_tokens["prefill"],
            "decode_tokens": self.phase_tokens["decode"],
            "prefill_tokens_s": self.phase_tokens["prefill"] / self.phase_seconds["prefill"],
            "decode_tokens_s": self.phase_tokens["decode"] / self.phase_seconds["decode"],
        }

class CapacityLimited(RuntimeError):
    """The requested fixed batch cannot run without KV preemption."""


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def cases_for(args):
    # Preserve the requested large-to-small experiment order.
    return [
        {"id": f"b{batch}-isl{isl}-osl{osl}", "batch": batch,
         "input_length": isl, "output_length": osl}
        for batch, isl, osl in itertools.product(
            sorted(set(args.batches), reverse=True),
            sorted(set(args.isls), reverse=True),
            sorted(set(args.osls), reverse=True))
    ]


def load_inputs(path, lengths):
    source = json.loads(path.read_text())
    seed = source["seed_token_ids"]
    if not seed or not all(isinstance(token, int) and token >= 0 for token in seed):
        raise ValueError("Input manifest must contain valid seed_token_ids")
    prompts = {length: (seed * ((length + len(seed) - 1) // len(seed)))[:length]
               for length in lengths}
    for case in source.get("cases", []):
        length = case["prompt_length"]
        if length in prompts and case["prompt_token_ids"] != prompts[length]:
            raise ValueError("Shared input manifest is not the repeated synthetic seed")
    return source, prompts


class FixedBatchGuard:
    """Allow split prefill, require every decode step to contain exactly B."""

    def __init__(self, engine, batch):
        self.batch = batch
        self.reset()
        scheduler = engine.scheduler
        original = scheduler.schedule

        @functools.wraps(original)
        def schedule(*args, **kwargs):
            metadata, output = original(*args, **kwargs)
            if output.ignored_seq_groups:
                raise CapacityLimited("The scheduler could not admit all requested prompts")
            if getattr(output, "blocks_to_swap_in", {}) or getattr(output, "blocks_to_swap_out", {}):
                raise CapacityLimited("KV swapping is incompatible with this fixed batch")
            if not output.is_empty():
                actual = sum(len(row.seq_data) for row in metadata)
                self.last_step = ("prefill" if output.prompt_run else "decode",
                                  output.num_batched_tokens)
                if output.prompt_run:
                    self.prefill_batches[actual] += 1
                else:
                    self.decode_batches[actual] += 1
                    if actual != self.batch or output.num_batched_tokens != self.batch:
                        raise CapacityLimited(
                            f"Decode B={actual}, scheduled tokens={output.num_batched_tokens}; "
                            f"required B={self.batch}")
            return metadata, output

        def preempt(*args, **kwargs):
            self.preemption_attempts += 1
            raise CapacityLimited("KV cache cannot accommodate the fixed workload without preemption")

        scheduler.schedule = schedule
        scheduler._preempt = preempt

    def reset(self):
        self.prefill_batches = Counter()
        self.decode_batches = Counter()
        self.preemption_attempts = 0
        self.last_step = None

    def summary(self):
        return {"prefill_batch_histogram": dict(self.prefill_batches),
                "decode_batch_histogram": dict(self.decode_batches),
                "decode_steps": sum(self.decode_batches.values()),
                "preemption_attempts": self.preemption_attempts}


def worker(args, case, source, prompts_by_length):
    backend = args.worker_backend
    variant = getattr(args, "diffkv_variant", "original") if backend == "diffkv" else None
    repo = FP16_REPO if backend == "fp16" else ROOT
    path = args.output_dir / f"{case['id']}-{backend}.json"
    common = {
        "model": str(Path(source["model_path"]).resolve()),
        "tokenizer": str(Path(source["tokenizer_path"]).resolve()),
        "dtype": "float16", "load_format": "safetensors",
        "tensor_parallel_size": 1, "max_model_len": max(args.isls) + max(args.osls),
        "max_num_batched_tokens": 8192, "max_num_seqs": max(args.batches),
        "max_paddings": 4096, "swap_space": 0, "gpu_memory_utilization": 0.9,
        "enforce_eager": True, "disable_log_stats": False, "seed": 42,
    }
    specific = {"block_size": 16} if backend == "fp16" else {"kv_buffer_size": 64}
    result = {"status": "initializing", "backend": backend, **case,
              "diffkv_variant": variant,
              "physical_gpu": args.gpu, "input_sha256": digest(prompts_by_length[case["input_length"]]),
              "input_manifest_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
              "common_engine_args": common, "backend_engine_args": specific,
              "sampling": {"temperature": 0, "ignore_eos": True, "n": 1},
              "diffkv_generate_args": ({"quant_configs": BITS, "quant_groups": GROUPS,
                                         "compress_configs": THRESHOLDS} if backend == "diffkv" else None),
              "measurement": {"mode": "offline synthetic fixed workload",
                              "included": "LLM.generate: registration, prefill, decode, sampling, detokenization, free",
                              "excluded": "loading, tokenization, warmup, verification, result serialization",
                              "warmup_batches": int(args.warmup_tokens > 0),
                              "warmup_output_length": args.warmup_tokens,
                              "measured_batches": args.repeats,
                              "phase_timing": "sum of engine.step wall times by scheduler prompt_run; includes scheduling, model, sampling, detokenization; no added CUDA synchronization",
                              "prefill_throughput": "input tokens / summed prefill step seconds",
                              "decode_throughput": "post-first output tokens / summed decode step seconds; excludes first tokens produced during prefill",
                              "fixed_batch": "all decode steps must have B; split prefill is allowed"},
              "samples": []}
    save(path, result)
    torch = guard = None
    phase = "initialization"

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    try:
        os.environ.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "OMP_NUM_THREADS": "2",
                           "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
                           "TRANSFORMERS_OFFLINE": "1", "HF_HOME": str(ROOT / ".cache/huggingface"),
                           "XDG_CACHE_HOME": str(ROOT / ".cache"),
                           "TRITON_CACHE_DIR": str(ROOT / ".cache/triton")})
        if backend == "fp16" and git(repo, "rev-parse", "HEAD") != FP16_COMMIT:
            raise RuntimeError("Unexpected original FP16 vLLM revision")
        source_diff = git(repo, "diff", "HEAD", "--", "vllm", "csrc")
        if source_diff and backend == "fp16":
            raise RuntimeError("Engine source has uncommitted changes")
        sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != ROOT]
        sys.path.insert(0, str(repo))
        import torch
        import vllm
        import vllm._C
        from vllm import LLM, SamplingParams

        for module in (vllm, vllm._C):
            if not Path(module.__file__).resolve().is_relative_to(repo):
                raise RuntimeError(f"Wrong backend import: {module.__file__}")
        if source_diff:
            from online_benchmark_server import verify_patched_build
            result["local_build"] = verify_patched_build(repo, Path(vllm._C.__file__), source_diff)
            path.with_suffix(".source.patch").write_text(source_diff)
        torch.set_num_threads(2)
        torch.manual_seed(42)
        torch.cuda.set_device(0)
        result.update(runtime_metadata())
        result.update({"gpu_name": torch.cuda.get_device_name(0),
                       "gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "")),
                       "gpu_memory_before_load": list(torch.cuda.mem_get_info()),
                       "source_commit": git(repo, "rev-parse", "HEAD"),
                       "vllm_import": vllm.__file__, "extension_import": vllm._C.__file__})
        print(f"LOAD {backend} {case['id']}", flush=True)
        llm = LLM(**common, **specific)
        engine = llm.llm_engine
        w = get_worker(engine, backend)
        if backend == "diffkv" and variant != "original":
            if variant.startswith("gqa-"):
                from diffkv_gqa import apply_variant
            else:
                from diffkv_v1 import apply_variant
            result["python_variant"] = apply_variant(
                engine, w, variant, args.output_dir / f"{case['id']}-diffkv.ablation")
            result["measurement"]["diffkv_variant"] = variant
        result["weights_verification"] = verify_weights(torch, w, Path(common["model"]))
        result["cache_verification"] = verify_cache(torch, w, backend, engine)
        result["gpu_cache_blocks"] = engine.cache_config.num_gpu_blocks
        guard = FixedBatchGuard(engine, case["batch"])
        timings = RequestTimings(engine, guard)
        prompt = prompts_by_length[case["input_length"]]
        prompts = [list(prompt) for _ in range(case["batch"])]
        phase = "warmup"
        if args.warmup_tokens:
            warm = generate(llm, SamplingParams(max_tokens=args.warmup_tokens, **result["sampling"]),
                            prompts, backend)
            torch.cuda.synchronize()
            verify_outputs(warm, prompt, case["batch"], args.warmup_tokens)
        phase = "measurement"
        result["status"] = "running"
        save(path, result)
        for repeat in range(args.repeats):
            guard.reset()
            engine.num_prompt_tokens, engine.num_generation_tokens = [], []
            engine.last_logging_time = time.monotonic()
            sampling = SamplingParams(max_tokens=case["output_length"], **result["sampling"])
            timings.begin()
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = generate(llm, sampling, prompts, backend)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            phase = "verification"
            observed = timings.end(start, case["batch"], case["input_length"], case["output_length"])
            counts = verify_outputs(outputs, prompt, case["batch"], case["output_length"])
            if engine.has_unfinished_requests():
                raise RuntimeError("Unfinished requests after generation")
            if guard.summary()["decode_steps"] != case["output_length"] - 1:
                raise RuntimeError("Unexpected number of fixed-batch decode steps")
            tokens_in, tokens_out = case["batch"] * case["input_length"], sum(counts)
            sample = {"repeat": repeat + 1, "elapsed_s": elapsed,
                      "completed_requests": len(outputs), "input_tokens": tokens_in,
                      "output_tokens": tokens_out, "output_tokens_s": tokens_out / elapsed,
                      "total_tokens_s": (tokens_in + tokens_out) / elapsed,
                      "generated_output_counts": counts,
                      "output_sha256": digest([o.outputs[0].token_ids for o in outputs]),
                      **guard.summary(),
                      **{k: observed[k] for k in ("ttft_ms", "tpot_ms", "request_latency_ms",
                                                 "prefill_s", "decode_s", "prefill_tokens", "decode_tokens",
                                                 "prefill_tokens_s", "decode_tokens_s")}}
            result["samples"].append(sample)
            save(path, result)
            print(f"RESULT {backend} {case['id']} repeat={repeat+1} "
                  f"TTFT={sample['ttft_ms']['mean']:.3f}ms TPOT={sample['tpot_ms']['mean']:.3f}ms/token "
                  f"prefill_tokens_s={sample['prefill_tokens_s']:.3f} "
                  f"decode_tokens_s={sample['decode_tokens_s']:.3f} "
                  f"E2E={elapsed:.3f}s", flush=True)
            phase = "measurement"
        result["status"] = "completed"
        for metric in ("elapsed_s", "output_tokens_s", "total_tokens_s", "prefill_tokens_s", "decode_tokens_s"):
            result[metric] = statistics.mean(sample[metric] for sample in result["samples"])
        for metric in ("ttft_ms", "tpot_ms"):
            result[metric] = statistics.mean(sample[metric]["mean"] for sample in result["samples"])
        return_code = 0
    except BaseException as exc:
        oom = torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError)
        cache_capacity = isinstance(exc, ValueError) and (
            "No available memory for the cache blocks" in str(exc) or
            ("model's max seq len" in str(exc) and "KV cache" in str(exc)))
        result.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else
                       "capacity_limited" if isinstance(exc, CapacityLimited) or oom or cache_capacity else "failed",
                       "failed_phase": phase, "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc(),
                       "guard_at_failure": guard.summary() if guard else None})
        # A guard failure leaves scheduler state partially modified. This process
        # must exit, never reuse the failed engine or report partial throughput.
        return_code = 0 if result["status"] == "capacity_limited" else 1
        print(f"{result['status'].upper()} {backend} {case['id']}: {result['error']}", flush=True)
    finally:
        save(path, result)
        if torch is not None and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return return_code


def comparison(folder, cases, backends):
    rows = []
    for case in cases:
        row = dict(case)
        completed = {}
        for backend in backends:
            path = folder / f"{case['id']}-{backend}.json"
            if path.exists():
                result = json.loads(path.read_text())
                row[backend] = {key: result.get(key) for key in
                                ("status", "ttft_ms", "tpot_ms", "prefill_tokens_s", "decode_tokens_s",
                                 "elapsed_s", "output_tokens_s", "total_tokens_s", "error",
                                 "physical_gpu", "gpu_uuid", "diffkv_variant")}
                if result["status"] == "completed":
                    completed[backend] = result
        if set(completed) == {"fp16", "diffkv"}:
            fp, diff = completed["fp16"], completed["diffkv"]
            for key in ("input_sha256", "input_manifest_sha256", "common_engine_args", "sampling"):
                if fp[key] != diff[key]:
                    raise RuntimeError(f"Comparison mismatch for {case['id']}: {key}")
            if fp.get("physical_gpu") != diff.get("physical_gpu"):
                raise RuntimeError(f"Comparison uses different GPUs for {case['id']}")
            if fp.get("gpu_uuid") and diff.get("gpu_uuid") and fp["gpu_uuid"] != diff["gpu_uuid"]:
                raise RuntimeError(f"Comparison uses different GPU UUIDs for {case['id']}")
            row["diffkv_output_throughput_over_fp16"] = diff["output_tokens_s"] / fp["output_tokens_s"]
        rows.append(row)
    save(folder / "comparison.json", {"cases": rows})
    return rows


def update_report(folder, cases, backends, report, force=False):
    for row in comparison(folder, cases, backends):
        results = {backend: row[backend] for backend in backends if backend in row}
        # On interruption, record the case that was started, rather than adding
        # rows for every unattempted case in the remaining grid.
        if results:
            report.record(row, results, force=force)


def stop_worker(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isls", type=int, nargs="+", choices=LENGTHS, default=LENGTHS)
    parser.add_argument("--osls", type=int, nargs="+", choices=[*LENGTHS, 4096], default=LENGTHS)
    parser.add_argument("--batches", type=int, nargs="+", choices=[4, *BATCHES], default=BATCHES)
    parser.add_argument("--backends", nargs="+", choices=("fp16", "diffkv"), default=["fp16", "diffkv"])
    parser.add_argument("--diffkv-variant",
                        choices=("original", "no-sync", "no-sync-bulk-lengths",
                                 "gqa-reference", "gqa-reuse"),
                        default="original", help="DiffKV execution variant; GQA reuse requires a rebuilt CUDA extension")
    parser.add_argument("--skip-diffkv-on-fp16-oom", action="store_true",
                        help="Skip the paired DiffKV worker if FP16 cannot fit the fixed workload")
    parser.add_argument("--case-indices", type=int, nargs="+",
                        help="Select indices from the full grid while keeping its engine limits")
    parser.add_argument("--gpu", type=int, default=0, help="Both backends use this GPU sequentially")
    parser.add_argument("--inputs", type=Path, default=ROOT / "logs/performance/llama-matched-inputs.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "evaluation.md",
                        help="Append the readable FP16 / DiffKV comparison table here")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON, help="Worker interpreter")
    parser.add_argument("--repeats", type=int, default=1, help="Measured batches per case")
    parser.add_argument("--warmup-tokens", type=int, default=16, help="One warmup batch; 0 disables it")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without GPU work or file writes")
    parser.add_argument("--worker-backend", choices=("fp16", "diffkv"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-case", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup_tokens < 0 or args.gpu < 0:
        parser.error("repeats must be positive; warmup-tokens and gpu must be nonnegative")
    if args.warmup_tokens > min(args.osls):
        parser.error("warmup-tokens must not exceed the shortest measured OSL")
    if args.skip_diffkv_on_fp16_oom and args.backends != ["fp16", "diffkv"]:
        parser.error("skip-diffkv-on-fp16-oom requires --backends fp16 diffkv in that order")
    args.inputs = args.inputs.resolve()
    cases = cases_for(args)
    if args.case_indices is not None:
        if len(set(args.case_indices)) != len(args.case_indices) or any(
                index < 0 or index >= len(cases) for index in args.case_indices):
            parser.error("case-indices must be unique valid indices of the full grid")
        cases = [cases[index] for index in args.case_indices]
    source, prompts = load_inputs(args.inputs, set(args.isls))
    plan = {"mode": "offline synthetic fixed workload", "case_count_per_backend": len(cases),
            "backends": list(dict.fromkeys(args.backends)), "gpu": args.gpu,
            "measured_batches_per_case": args.repeats,
            "warmup_output_length": args.warmup_tokens,
            "diffkv_variant": args.diffkv_variant, "case_indices": args.case_indices,
            "skip_diffkv_on_fp16_oom": args.skip_diffkv_on_fp16_oom,
            "model": source["model_path"], "cases": cases,
            "input_sha256_by_isl": {length: digest(prompt) for length, prompt in prompts.items()}}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if args.worker_backend:
        if args.output_dir is None or args.worker_case is None:
            parser.error("Worker requires an output directory and case index")
        args.output_dir = args.output_dir.resolve()
        return worker(args, cases[args.worker_case], source, prompts)
    args.python = args.python.expanduser().absolute()
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        parser.error(f"Worker Python is not executable: {args.python}")
    started_at = datetime.now(timezone(timedelta(hours=9)))
    stamp = started_at.strftime("%Y%m%d-%H%M%S-%f")
    folder = (args.output_dir or ROOT / "logs/performance" / f"fixed-grid-{stamp}").resolve()
    folder.mkdir(parents=True, exist_ok=False)
    save(folder / "plan.json", plan)
    script = Path(__file__).resolve()
    report = EvaluationReport(args.report.resolve(), {
        **plan, "started_at": started_at.isoformat(timespec="seconds"),
        "isls": args.isls, "osls": args.osls, "batches": args.batches,
        "log_directory": str(folder),
        "physical_gpus": [args.gpu],
        "gpu_mapping": {case["id"]: args.gpu for case in cases},
        "command": shlex.join([sys.executable, str(script), *sys.argv[1:]]),
    }, stamp)
    failures = 0
    process = None
    statuses = Counter()

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    try:
        for index, case in enumerate(cases):
            for backend in plan["backends"]:
                if backend == "diffkv" and args.skip_diffkv_on_fp16_oom:
                    fp16 = json.loads((folder / f"{case['id']}-fp16.json").read_text())
                    if fp16["status"] == "capacity_limited":
                        save(folder / f"{case['id']}-diffkv.json", {
                            **case, "backend": "diffkv", "status": "skipped",
                            "physical_gpu": args.gpu, "diffkv_variant": args.diffkv_variant,
                            "reason": "FP16 could not fit the fixed workload; case skipped as requested",
                        })
                        statuses["skipped"] += 1
                        update_report(folder, cases, plan["backends"], report)
                        print(f"SKIP {case['id']} diffkv: FP16 capacity limited", flush=True)
                        continue
                command = [str(args.python), str(script), "--worker-backend", backend,
                           "--worker-case", str(index), "--output-dir", str(folder),
                           "--inputs", str(args.inputs), "--gpu", str(args.gpu),
                           "--repeats", str(args.repeats), "--warmup-tokens", str(args.warmup_tokens),
                           "--isls", *map(str, args.isls), "--osls", *map(str, args.osls),
                           "--batches", *map(str, args.batches)]
                command.extend(["--diffkv-variant", args.diffkv_variant])
                if args.case_indices is not None:
                    command.extend(["--case-indices", *map(str, args.case_indices)])
                log_path = folder / f"{case['id']}-{backend}.log"
                print(f"RUN {case['id']} {backend}; log={log_path}", flush=True)
                with log_path.open("w") as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    code = process.wait()
                path = folder / f"{case['id']}-{backend}.json"
                if not path.exists():
                    save(path, {"status": "failed", "backend": backend, **case,
                                "error": f"Worker exited with code {code} before writing a result"})
                result = json.loads(path.read_text())
                if result["status"] not in ("completed", "capacity_limited", "failed", "interrupted"):
                    result.update({"last_worker_status": result["status"], "status": "failed",
                                   "error": f"Worker exited with code {code} without a terminal result"})
                result.update({"process_returncode": code, "command": command})
                save(path, result)
                failures += int(code != 0 or result["status"] not in ("completed", "capacity_limited"))
                displayed_status = "OOM" if result["status"] == "capacity_limited" else result["status"]
                statuses[displayed_status] += 1
                update_report(folder, cases, plan["backends"], report)
                print(f"DONE {case['id']} {backend}: {displayed_status}", flush=True)
    except KeyboardInterrupt:
        stop_worker(process)
        update_report(folder, cases, plan["backends"], report, force=True)
        report.finish("interrupted", dict(statuses))
        return 130
    except BaseException:
        stop_worker(process)
        update_report(folder, cases, plan["backends"], report, force=True)
        report.finish("failed", dict(statuses))
        raise
    finally:
        stop_worker(process)
    report.finish("completed" if failures == 0 else "completed with failures", dict(statuses))
    print(f"Evaluation: {args.report.resolve()}", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
