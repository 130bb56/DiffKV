"""Common HTTP benchmark adapter for the original FP16 and DiffKV async engines.

The adapter observes the existing scheduler and async step at runtime. It does
not change scheduling, add CUDA synchronization, or edit either engine source.
"""

import argparse
import asyncio
from collections import Counter
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
FP16_REPO = ROOT / ".cache/baselines/vllm-1db83e3"
FP16_COMMIT = "1db83e31a2468cae37f326a642c0a4c4edbb5e4f"
BITS = [8, 4, 4, 2]
GROUPS = [1, 1, 1, 1]
THRESHOLDS = [0.02, 1.0]


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def failure_status(error):
    message = str(error).lower()
    return "OOM" if any(text in message for text in (
        "out of memory", "cuda error: memory allocation", "lack of cpu swap",
        "no free blocks", "blocks required but only")) else "failed"


def configured_eos_token_ids(model_path, tokenizer_eos_token_id):
    """Honor the checkpoint's complete EOS list and the loaded tokenizer EOS."""
    path = Path(model_path) / "generation_config.json"
    configured = json.loads(path.read_text()).get("eos_token_id") if path.exists() else None
    if configured is None:
        configured = []
    elif isinstance(configured, int) and not isinstance(configured, bool):
        configured = [configured]
    if not isinstance(configured, list) or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in configured):
        raise ValueError("Model eos_token_id must be a nonnegative integer or list of integers")
    if tokenizer_eos_token_id is not None:
        if (isinstance(tokenizer_eos_token_id, bool)
                or not isinstance(tokenizer_eos_token_id, int) or tokenizer_eos_token_id < 0):
            raise ValueError("Tokenizer eos_token_id must be a nonnegative integer")
        configured = [*configured, tokenizer_eos_token_id]
    return list(dict.fromkeys(configured))


def verify_patched_build(repo, extension_path, source_diff):
    """Reject stale CUDA binaries when benchmarking a locally patched engine."""
    manifest_path = repo / ".cache/diffkv-build.json"
    if not manifest_path.exists():
        raise RuntimeError("Patched DiffKV source requires a verified local CUDA build")
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(source_diff.encode()).hexdigest() != manifest["source_diff_sha256"]:
        raise RuntimeError("DiffKV source patch differs from the recorded CUDA build")
    for relative, expected in manifest["source_sha256"].items():
        if hashlib.sha256((repo / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"DiffKV build source changed: {relative}")
    if hashlib.sha256(extension_path.read_bytes()).hexdigest() != manifest["extension_sha256"]:
        raise RuntimeError("DiffKV CUDA extension differs from the verified local build")
    return {"manifest": str(manifest_path), "source_diff_sha256": manifest["source_diff_sha256"],
            "extension_sha256": manifest["extension_sha256"]}


def sampling_options(payload, eos_token_ids=()):
    max_tokens = payload.get("max_tokens", 4096)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    ignore_eos = payload.get("ignore_eos", True)
    if not isinstance(ignore_eos, bool):
        raise ValueError("ignore_eos must be a boolean")
    stop = payload.get("stop", [])
    if stop is None:
        stop = []
    elif isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or any(not isinstance(item, str) or not item for item in stop):
        raise ValueError("stop must be a list of nonempty strings")
    stop_token_ids = payload.get("stop_token_ids")
    if stop_token_ids is None:
        stop_token_ids = []
    if not isinstance(stop_token_ids, list) or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in stop_token_ids):
        raise ValueError("stop_token_ids must be a list of nonnegative integers")
    if not ignore_eos:
        stop_token_ids = [*stop_token_ids, *eos_token_ids]
    temperature = payload.get("temperature", 0.0)
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature < 0):
        raise ValueError("temperature must be finite and nonnegative")
    options = {"n": 1, "temperature": float(temperature),
               "ignore_eos": ignore_eos, "max_tokens": max_tokens, "stop": list(stop),
               "stop_token_ids": list(dict.fromkeys(stop_token_ids))}
    if "repetition_penalty" in payload:
        penalty = payload["repetition_penalty"]
        if (isinstance(penalty, bool) or not isinstance(penalty, (int, float))
                or not math.isfinite(penalty) or penalty <= 0):
            raise ValueError("repetition_penalty must be finite and positive")
        options["repetition_penalty"] = float(penalty)
    if "frequency_penalty" in payload:
        penalty = payload["frequency_penalty"]
        if (isinstance(penalty, bool) or not isinstance(penalty, (int, float))
                or not math.isfinite(penalty) or not -2 <= penalty <= 2):
            raise ValueError("frequency_penalty must be finite and within [-2, 2]")
        options["frequency_penalty"] = float(penalty)
    if "top_p" in payload:
        top_p = payload["top_p"]
        if (isinstance(top_p, bool) or not isinstance(top_p, (int, float))
                or not math.isfinite(top_p) or not 0 < top_p <= 1):
            raise ValueError("top_p must be finite and within (0, 1]")
        options["top_p"] = float(top_p)
    return options


def validate_output_length(count, max_tokens, ignore_eos, stop, stop_token_ids=()):
    if count <= 0 or count > max_tokens:
        raise RuntimeError(f"Expected 1 to {max_tokens} output tokens, got {count}")
    if ignore_eos and not stop and not stop_token_ids and count != max_tokens:
        raise RuntimeError(f"Expected {max_tokens} output tokens, got {count}")


class PhaseObserver:
    """Count actual processed tokens and observe unmodified async engine steps."""

    def __init__(self, async_engine, backend, worker):
        self.async_engine = async_engine
        self.engine = async_engine.engine
        self.backend = backend
        self.worker = worker
        self.reset()
        original_schedule = self.engine._schedule
        original_step = self.engine.step_async
        scheduler = self.engine.scheduler

        @functools.wraps(original_schedule)
        def schedule(*args, **kwargs):
            metadata, output, ignored = original_schedule(*args, **kwargs)
            self.current_step = None
            if metadata:
                phase = "prefill" if output.prompt_run else "decode"
                before = {}
                actual_tokens = initial_tokens = recompute_tokens = 0
                batch = 0
                for item in metadata:
                    counts = [data.get_output_len() for data in item.seq_data.values()]
                    if len(counts) != 1:
                        raise RuntimeError("Online benchmark requires n=1, one sequence per request")
                    before[item.request_id] = counts[0]
                    batch += len(item.seq_data)
                    if phase == "prefill":
                        for data in item.seq_data.values():
                            tokens = data.get_len()
                            actual_tokens += tokens
                            if data.get_output_len() == 0:
                                initial_tokens += tokens
                            else:
                                recompute_tokens += tokens
                self.current_step = {
                    "phase": phase, "before": before, "batch": batch,
                    "tokens": actual_tokens, "initial_tokens": initial_tokens,
                    "recompute_tokens": recompute_tokens,
                    "padded_tokens": int(output.num_batched_tokens),
                }
            self.sample_state()
            return metadata, output, ignored

        @functools.wraps(original_step)
        async def step(*args, **kwargs):
            self.current_step = None
            started = time.perf_counter()
            try:
                outputs = await original_step(*args, **kwargs)
            except BaseException as error:
                self.engine_error = {"status": failure_status(error), "error": str(error)}
                raise
            elapsed = time.perf_counter() - started
            observed = self.current_step
            if observed is not None:
                phase = observed["phase"]
                self.phase_seconds[phase] += elapsed
                self.phase_steps[phase] += 1
                emitted = 0
                for output in outputs:
                    previous = observed["before"].get(output.request_id)
                    if previous is not None:
                        count = len(output.outputs[0].token_ids)
                        increment = count - previous
                        if increment != 1:
                            raise RuntimeError(
                                f"Unexpected output increment for {output.request_id}: {increment}")
                        emitted += increment
                self.emitted_tokens[phase] += emitted
                if phase == "prefill":
                    self.phase_tokens[phase] += observed["tokens"]
                    self.initial_prefill_tokens += observed["initial_tokens"]
                    self.recompute_prefill_tokens += observed["recompute_tokens"]
                    self.prefill_padding_tokens += observed["padded_tokens"] - observed["tokens"]
                    self.prefill_batch_histogram[observed["batch"]] += 1
                else:
                    self.phase_tokens[phase] += emitted
                    self.decode_batch_histogram[observed["batch"]] += 1
            self.sample_state()
            return outputs

        self.engine._schedule = schedule
        self.engine.step_async = step
        original_preempt = scheduler._preempt

        @functools.wraps(original_preempt)
        def preempt(*args, **kwargs):
            self.preemption_count += 1
            return original_preempt(*args, **kwargs)

        scheduler._preempt = preempt
        original_recompute = scheduler._preempt_by_recompute

        @functools.wraps(original_recompute)
        def recompute(seq_group, *args, **kwargs):
            self.recompute_count += 1
            self.recomputed_request_ids.add(str(seq_group.request_id))
            return original_recompute(seq_group, *args, **kwargs)

        scheduler._preempt_by_recompute = recompute

    def reset(self):
        self.current_step = None
        self.phase_seconds = Counter()
        self.phase_tokens = Counter()
        self.phase_steps = Counter()
        self.emitted_tokens = Counter()
        self.decode_batch_histogram = Counter()
        self.prefill_batch_histogram = Counter()
        self.initial_prefill_tokens = 0
        self.recompute_prefill_tokens = 0
        self.prefill_padding_tokens = 0
        self.preemption_count = self.recompute_count = 0
        self.recomputed_request_ids = set()
        self.max_running = self.max_waiting = self.max_swapped = 0
        self.max_pending_frontend = 0
        self.max_kv_usage_percent = 0.0
        self.current_state = {}
        self.engine_error = None
        self.reset_time = time.time()

    def sample_state(self):
        scheduler = self.engine.scheduler
        manager = (scheduler.block_manager if self.backend == "fp16"
                   else self.worker.block_manager)
        free = int(manager.get_num_free_gpu_blocks())
        total = int(manager.num_total_gpu_blocks)
        usage = 100.0 * (total - free) / total if total else 0.0
        running = sum(group.get_max_num_running_seqs() for group in scheduler.running)
        waiting = len(scheduler.waiting)
        swapped = len(scheduler.swapped)
        pending = self.async_engine._request_tracker._new_requests.qsize()
        self.max_running = max(self.max_running, running)
        self.max_waiting = max(self.max_waiting, waiting)
        self.max_swapped = max(self.max_swapped, swapped)
        self.max_pending_frontend = max(self.max_pending_frontend, pending)
        self.max_kv_usage_percent = max(self.max_kv_usage_percent, usage)
        self.current_state = {
            "running": running, "waiting": waiting, "swapped": swapped,
            "pending_frontend": pending, "free_gpu_blocks": free,
            "total_gpu_blocks": total, "kv_usage_percent": usage,
        }

    def snapshot(self):
        self.sample_state()
        histogram = self.decode_batch_histogram
        steps = sum(histogram.values())
        return {
            "reset_unix_time": self.reset_time,
            "phase_seconds": {name: self.phase_seconds[name] for name in ("prefill", "decode")},
            "phase_tokens": {name: self.phase_tokens[name] for name in ("prefill", "decode")},
            "phase_steps": {name: self.phase_steps[name] for name in ("prefill", "decode")},
            "emitted_tokens": {name: self.emitted_tokens[name] for name in ("prefill", "decode")},
            "prefill_tokens_s": (self.phase_tokens["prefill"] / self.phase_seconds["prefill"]
                                 if self.phase_seconds["prefill"] else None),
            "decode_tokens_s": (self.phase_tokens["decode"] / self.phase_seconds["decode"]
                                if self.phase_seconds["decode"] else None),
            "initial_prefill_tokens": self.initial_prefill_tokens,
            "recompute_prefill_tokens": self.recompute_prefill_tokens,
            "prefill_padding_tokens": self.prefill_padding_tokens,
            "prefill_batch_histogram": dict(self.prefill_batch_histogram),
            "decode_batch_histogram": dict(histogram),
            "mean_decode_batch": (sum(batch * count for batch, count in histogram.items()) / steps
                                  if steps else None),
            "max_decode_batch": max(histogram, default=0),
            "decode_batch_mean": (sum(batch * count for batch, count in histogram.items()) / steps
                                  if steps else None),
            "decode_batch_max": max(histogram, default=0),
            "preemption_count": self.preemption_count,
            "recompute_count": self.recompute_count,
            "recompute_requests": len(self.recomputed_request_ids),
            "max_running": self.max_running, "max_waiting": self.max_waiting,
            "max_swapped": self.max_swapped,
            "max_pending_frontend": self.max_pending_frontend,
            "max_kv_usage_percent": self.max_kv_usage_percent,
            "state": self.current_state, "engine_error": self.engine_error,
            "definitions": {
                "phase_seconds": "Sum of async engine step wall times; scheduling, model, sampling, detokenization; no added CUDA synchronization; excludes HTTP transport",
                "prefill_tokens_s": "Actual prefill tokens, including recomputed context and excluding padding, divided by prefill step seconds",
                "decode_tokens_s": "New output tokens emitted in decode steps divided by decode step seconds; excludes output tokens emitted in prefill steps",
                "max_kv_usage_percent": "Peak observed physical GPU KV allocator block usage at schedule and async step boundaries",
            },
        }


def build_app(async_engine, observer, sampling_class, args, metadata):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI()
    active = set()
    seen = set()
    counters = Counter()
    errors = []

    def metrics():
        return {"backend": args.backend, **observer.snapshot(),
                "submitted": counters["submitted"], "completed": counters["completed"],
                "failed": counters["failed"], "active": len(active),
                "errors": list(errors), "metadata": metadata}

    async def progress():
        while True:
            await asyncio.sleep(30)
            value = metrics()
            save_json(args.output_dir / "server-metrics.json", value)
            state = value["state"]
            print(f"PROGRESS {args.backend} completed={value['completed']} "
                  f"active={value['active']} running={state['running']} "
                  f"waiting={state['waiting']} kv_usage={state['kv_usage_percent']:.1f}% "
                  f"recompute={value['recompute_count']}", flush=True)

    @app.on_event("startup")
    async def startup():
        app.state.progress_task = asyncio.create_task(progress())

    @app.on_event("shutdown")
    async def shutdown():
        app.state.progress_task.cancel()
        save_json(args.output_dir / "server-metrics.json", metrics())

    @app.get("/health")
    async def health():
        if observer.engine_error is not None:
            return JSONResponse({"status": "failed", "backend": args.backend,
                                 **observer.engine_error}, status_code=503)
        return {"status": "ready", "backend": args.backend}

    @app.get("/metrics")
    async def get_metrics():
        value = metrics()
        save_json(args.output_dir / "server-metrics.json", value)
        return value

    @app.post("/reset")
    async def reset():
        # A client can receive its final frame just before the stream's finally
        # block runs. Permit this short drain, and ignore finished-tracker cleanup.
        deadline = time.monotonic() + 2.0
        while (active or async_engine.engine.has_unfinished_requests()
               or not async_engine._request_tracker._new_requests.empty()):
            if time.monotonic() >= deadline:
                raise HTTPException(status_code=409, detail="Requests are still in progress")
            await asyncio.sleep(0.01)
        if observer.engine_error is not None:
            raise HTTPException(status_code=503, detail=observer.engine_error)
        observer.reset()
        counters.clear()
        errors.clear()
        seen.clear()
        value = metrics()
        save_json(args.output_dir / "server-metrics.json", value)
        return {"status": "reset", "metrics": value}

    @app.post("/generate")
    async def generate(request: Request):
        if observer.engine_error is not None:
            raise HTTPException(status_code=503, detail=observer.engine_error)
        payload = await request.json()
        request_id = str(payload["request_id"])
        if request_id in seen:
            raise HTTPException(status_code=409, detail="Duplicate request_id")
        prompt = payload.get("prompt")
        prompt_ids = payload.get("prompt_token_ids")
        if prompt_ids is not None:
            if not isinstance(prompt_ids, list) or not prompt_ids or any(
                    not isinstance(token, int) or isinstance(token, bool) or token < 0
                    for token in prompt_ids):
                raise HTTPException(status_code=400, detail="Invalid prompt_token_ids")
        elif not isinstance(prompt, str):
            raise HTTPException(status_code=400, detail="prompt or prompt_token_ids is required")
        try:
            options = sampling_options(payload, metadata.get("effective_eos_token_ids", ()))
            sampling = sampling_class(**options)
        except (ValueError, TypeError) as error:
            raise HTTPException(status_code=400, detail=str(error))
        max_tokens = options["max_tokens"]
        if prompt_ids is not None and len(prompt_ids) + max_tokens > args.max_model_len:
            raise HTTPException(status_code=400, detail="Input plus output exceeds max_model_len")
        seen.add(request_id)
        active.add(request_id)
        counters["submitted"] += 1
        if args.backend == "diffkv":
            iterator = async_engine.generate(
                prompt, sampling, BITS, GROUPS, THRESHOLDS, request_id,
                prompt_token_ids=prompt_ids)
        else:
            iterator = async_engine.generate(
                prompt, sampling, request_id, prompt_token_ids=prompt_ids)

        async def frames():
            completed = False
            last_count = 0
            try:
                async for output in iterator:
                    if len(output.outputs) != 1:
                        raise RuntimeError("Expected exactly one completion")
                    completion = output.outputs[0]
                    count = len(completion.token_ids)
                    if count < last_count:
                        raise RuntimeError("Generated token count decreased")
                    last_count = count
                    frame = {"request_id": request_id, "output_tokens": count,
                             "prompt_tokens": len(output.prompt_token_ids),
                             "finished": bool(output.finished)}
                    if getattr(args, "monitor_output", False):
                        frame["last_token_id"] = completion.token_ids[-1] if count else None
                        if count % 32 == 0 or output.finished:
                            frame["preview_text"] = completion.text[-2048:]
                        if output.finished:
                            frame["output_token_ids"] = list(completion.token_ids)
                            frame["stop_token_ids"] = options["stop_token_ids"]
                            frame["effective_eos_token_ids"] = metadata.get("effective_eos_token_ids", [])
                    if output.finished:
                        frame["text"] = completion.text
                        frame["finish_reason"] = completion.finish_reason
                        validate_output_length(count, max_tokens, options["ignore_eos"],
                                               options["stop"], options["stop_token_ids"])
                        counters["completed"] += 1
                        completed = True
                    yield frame
            except asyncio.CancelledError:
                counters["failed"] += 1
                errors.append({"request_id": request_id, "status": "cancelled",
                               "error": "Client disconnected"})
                await async_engine.abort(request_id)
                raise
            except Exception as error:
                counters["failed"] += 1
                status = (observer.engine_error or {}).get("status", failure_status(error))
                detail = {"request_id": request_id, "status": status,
                          "error": str(error), "finished": True}
                errors.append(detail)
                print(f"REQUEST_ERROR {args.backend} {json.dumps(detail)}", flush=True)
                yield detail
            finally:
                active.discard(request_id)
                if not completed:
                    await async_engine.abort(request_id)

        if payload.get("stream", True):
            async def stream():
                async for frame in frames():
                    yield (json.dumps(frame, ensure_ascii=False) + "\0").encode("utf-8")
            return StreamingResponse(stream(), media_type="application/octet-stream")
        final = None
        async for frame in frames():
            final = frame
        return JSONResponse(final or {"request_id": request_id, "error": "No output"})

    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["fp16", "diffkv"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--monitor-output", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=100)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-paddings", type=int, default=4096)
    parser.add_argument("--swap-space", type=float, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo = FP16_REPO if args.backend == "fp16" else ROOT
    metadata = {"status": "initializing", "backend": args.backend,
                "model": str(args.model.resolve()), "pid": os.getpid(),
                "monitor_output": args.monitor_output,
                "monitoring_interval_tokens": 32 if args.monitor_output else None}
    path = args.output_dir / "server-metadata.json"
    save_json(path, metadata)
    try:
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.update({"OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
                           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                           "HF_HOME": str(ROOT / ".cache/huggingface"),
                           "XDG_CACHE_HOME": str(ROOT / ".cache"),
                           "TRITON_CACHE_DIR": str(ROOT / ".cache/triton")})
        sys.path[:] = [item for item in sys.path
                       if Path(item or os.getcwd()).resolve() != ROOT]
        sys.path.insert(0, str(repo))
        from run_offline_e2e import get_worker, git, runtime_metadata, verify_cache, verify_weights
        if args.backend == "fp16" and git(repo, "rev-parse", "HEAD") != FP16_COMMIT:
            raise RuntimeError("Unexpected original FP16 source revision")
        source_diff = git(repo, "diff", "HEAD", "--", "vllm", "csrc")
        if source_diff and args.backend == "fp16":
            raise RuntimeError("Engine source has uncommitted changes")
        import torch
        import uvicorn
        import vllm
        import vllm._C
        from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
        for module in (vllm, vllm._C):
            if not Path(module.__file__).resolve().is_relative_to(repo):
                raise RuntimeError(f"Wrong backend import: {module.__file__}")
        if source_diff:
            metadata["local_build"] = verify_patched_build(repo, Path(vllm._C.__file__), source_diff)
            (args.output_dir / "source.patch").write_text(source_diff)
        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        torch.cuda.set_device(0)
        common = {
            "model": str(args.model.resolve()),
            "tokenizer": str((args.tokenizer or args.model).resolve()),
            "dtype": "float16", "load_format": "safetensors", "tensor_parallel_size": 1,
            "max_model_len": args.max_model_len, "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_paddings": args.max_paddings, "swap_space": args.swap_space,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager, "seed": args.seed,
            "disable_log_requests": True, "disable_log_stats": False,
        }
        specific = {"block_size": 16} if args.backend == "fp16" else {"kv_buffer_size": 64}
        metadata.update({"common_engine_args": common, "backend_engine_args": specific,
                         "source_commit": git(repo, "rev-parse", "HEAD"),
                         "vllm_import": vllm.__file__, "extension_import": vllm._C.__file__,
                         "async_engine_source": inspect.getfile(AsyncLLMEngine),
                         "gpu_name": torch.cuda.get_device_name(0),
                         "gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "")),
                         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "diffkv_generate_args": ({"quant_configs": BITS, "quant_groups": GROUPS,
                                                  "compress_configs": THRESHOLDS}
                                                 if args.backend == "diffkv" else None),
                         **runtime_metadata()})
        print(f"LOAD {args.backend} online-serving", flush=True)
        async_engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**common, **specific))
        engine = async_engine.engine
        metadata["effective_eos_token_ids"] = configured_eos_token_ids(
            args.model.resolve(), engine.tokenizer.eos_token_id)
        metadata["tokenizer_eos_token_id"] = engine.tokenizer.eos_token_id
        metadata["eos_configuration_source"] = str(args.model.resolve() / "generation_config.json")
        print(f"EOS {args.backend}: {metadata['effective_eos_token_ids']}", flush=True)
        worker = get_worker(engine, args.backend)
        metadata["weights_verification"] = verify_weights(torch, worker, args.model.resolve())
        metadata["cache_verification"] = verify_cache(torch, worker, args.backend, engine)
        metadata["gpu_cache_blocks"] = engine.cache_config.num_gpu_blocks
        observer = PhaseObserver(async_engine, args.backend, worker)
        metadata["status"] = "ready"
        save_json(path, metadata)
        app = build_app(async_engine, observer, SamplingParams, args, metadata)
        print(f"READY {args.backend} http://{args.host}:{args.port}", flush=True)
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)
    except BaseException as error:
        metadata.update(status=failure_status(error), error=str(error), traceback=traceback.format_exc())
        save_json(path, metadata)
        print(f"SERVER_ERROR {args.backend} {metadata['status']}: {error}", flush=True)
        raise


if __name__ == "__main__":
    main()
