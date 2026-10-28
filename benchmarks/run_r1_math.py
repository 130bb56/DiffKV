"""Run the frozen R1 MATH workload through one FP16 or DiffKV engine.

All requests enter the scheduler before the first measured engine step and
share one arrival time. Timing excludes model loading and warmup. Run each
backend in a fresh process with its own Python environment and physical GPU.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import functools
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
FP16_REPO = ROOT / ".cache/baselines/vllm-1db83e3"
BITS = [8, 4, 4, 2]
GROUPS = [1, 1, 1, 1]
THRESHOLDS = [0.02, 1.0]
COMMON_ENGINE = {
    "dtype": "float16", "load_format": "safetensors", "tensor_parallel_size": 1,
    "max_model_len": 9216, "max_num_batched_tokens": 9216,
    "max_num_seqs": 64, "max_paddings": 4096, "swap_space": 0,
    "gpu_memory_utilization": 0.9, "enforce_eager": True,
    "disable_log_stats": True,
}


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def read_workload(path):
    value = json.loads(Path(path).read_text())
    requests = value["requests"]
    if not requests:
        raise ValueError("Workload has no requests")
    seen = set()
    for row in requests:
        request_id = row["request_id"]
        if not isinstance(request_id, str) or not request_id or request_id in seen:
            raise ValueError("Request IDs must be unique nonempty strings")
        seen.add(request_id)
        tokens = row["prompt_token_ids"]
        if (not isinstance(tokens, list) or not tokens or
                any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError(f"Invalid prompt_token_ids: {request_id}")
        if row.get("prompt_tokens", len(tokens)) != len(tokens):
            raise ValueError(f"Prompt count mismatch: {request_id}")
    return value


def length_limits(requests, mode, max_tokens, reference_path=None, workload_sha256=None):
    if mode == "natural":
        return {row["request_id"]: max_tokens for row in requests}
    if reference_path is None:
        raise ValueError("length_replay requires --lengths-from")
    reference = json.loads(Path(reference_path).read_text())
    metadata = reference.get("metadata", reference)
    if metadata.get("backend") != "fp16" or metadata.get("mode") != "natural":
        raise ValueError("Length reference must be the natural FP16 result")
    if reference.get("status") != "completed":
        raise ValueError("Length reference is incomplete")
    if workload_sha256 and metadata.get("workload_sha256") != workload_sha256:
        raise ValueError("Length reference uses a different workload")
    rows = reference["requests"]
    limits = {row["request_id"]: row["output_tokens"] for row in rows}
    if len(limits) != len(rows) or set(limits) != {row["request_id"] for row in requests}:
        raise ValueError("Length reference request IDs differ from workload")
    for row in rows:
        count = row["output_tokens"]
        if type(count) is not int or not 0 < count <= max_tokens:
            raise ValueError("Invalid reference output length")
        if len(row["output_token_ids"]) != count:
            raise ValueError("Reference output count differs from token IDs")
    return limits


def configure_environment(gpu, backend):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.update({
        "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(ROOT / ".cache/huggingface"),
        "XDG_CACHE_HOME": str(ROOT / ".cache"),
        "TRITON_CACHE_DIR": str(ROOT / ".cache/triton"),
    })
    repo = FP16_REPO if backend == "fp16" else ROOT
    sys.path[:] = [item for item in sys.path
                   if Path(item or os.getcwd()).resolve() != ROOT]
    sys.path.insert(0, str(repo))
    return repo


def configured_eos(model_path, tokenizer_eos):
    path = Path(model_path) / "generation_config.json"
    values = json.loads(path.read_text()).get("eos_token_id", []) if path.exists() else []
    if type(values) is int:
        values = [values]
    if values is None:
        values = []
    if not isinstance(values, list) or any(type(token) is not int or token < 0 for token in values):
        raise ValueError("Invalid checkpoint EOS token IDs")
    if tokenizer_eos is not None:
        if type(tokenizer_eos) is not int or tokenizer_eos < 0:
            raise ValueError("Invalid tokenizer EOS token ID")
        values.append(tokenizer_eos)
    return list(dict.fromkeys(values))


def make_engine(args):
    repo = configure_environment(args.gpu, args.backend)
    import torch
    import vllm
    import vllm._C
    from vllm import EngineArgs, LLMEngine
    from run_offline_e2e import get_worker, runtime_metadata, verify_cache, verify_weights

    for module in (vllm, vllm._C):
        if not Path(module.__file__).resolve().is_relative_to(repo):
            raise RuntimeError(f"Wrong backend import: {module.__file__}")
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    engine_options = {**COMMON_ENGINE, **getattr(args, "engine_options", {}), "seed": args.seed}
    common = {**engine_options, "model": str(args.model.resolve()),
              "tokenizer": str(args.model.resolve()), "seed": args.seed}
    specific = {"block_size": 16} if args.backend == "fp16" else {"kv_buffer_size": 64}
    print(f"LOAD {args.backend} GPU{args.gpu}", flush=True)
    engine = LLMEngine.from_engine_args(EngineArgs(**common, **specific))
    worker = get_worker(engine, args.backend)
    source = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "HEAD", "--", "vllm", "csrc"], text=True)
    source_path = args.output.with_suffix(".source.patch")
    source_path.write_text(source)
    metadata = {
        "engine": engine_options,
        "backend_engine": specific,
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "source_patch": str(source_path.resolve()),
        "source_patch_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "vllm_import": vllm.__file__, "extension_import": vllm._C.__file__,
        "extension_sha256": file_digest(vllm._C.__file__),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "")),
        "gpu_cache_blocks": engine.cache_config.num_gpu_blocks,
        "effective_eos_token_ids": configured_eos(args.model, engine.tokenizer.eos_token_id),
        "diffkv_variant": args.diffkv_variant if args.backend == "diffkv" else None,
        **runtime_metadata(),
    }
    if args.backend == "diffkv":
        from diffkv_v1 import apply_variant
        metadata["python_variant"] = apply_variant(
            engine, worker, args.diffkv_variant,
            args.output.parent / (args.output.stem + "-variant"))
        metadata["quant_configs"] = BITS
        metadata["quant_groups"] = GROUPS
        metadata["compress_configs"] = THRESHOLDS
    metadata["weights_verification"] = verify_weights(torch, worker, args.model.resolve())
    metadata["cache_verification"] = verify_cache(torch, worker, args.backend, engine)
    return engine, worker, metadata


def add_request(engine, backend, request_id, tokens, sampling, arrival_time):
    kwargs = {"request_id": request_id, "prompt": None, "sampling_params": sampling,
              "prompt_token_ids": list(tokens), "arrival_time": arrival_time}
    if backend == "diffkv":
        kwargs.update(quant_configs=list(BITS), quant_groups=list(GROUPS),
                      compress_configs=list(THRESHOLDS))
    engine.add_request(**kwargs)


class StepObserver:
    """Observe scheduler results without introducing CUDA synchronization."""

    def __init__(self, engine, worker, backend):
        self.engine, self.worker, self.backend = engine, worker, backend
        self.preemptions = 0
        self.recomputations = 0
        self.request_preemptions = Counter()
        self.current = None
        self.decode_batches = Counter()
        self.phase_seconds = Counter()
        self.phase_steps = Counter()
        self.max_kv_usage_percent = 0.0
        original_schedule = engine._schedule

        @functools.wraps(original_schedule)
        def schedule(*args, **kwargs):
            metadata, output, ignored = original_schedule(*args, **kwargs)
            self.current = {
                "phase": "prefill" if output.prompt_run else "decode",
                "batch": sum(len(row.seq_data) for row in metadata),
                "request_ids": [str(row.request_id) for row in metadata],
                "padded_tokens": int(output.num_batched_tokens),
                "state_at_schedule": self.state(),
            }
            return metadata, output, ignored

        engine._schedule = schedule
        scheduler = engine.scheduler
        original_preempt = scheduler._preempt

        @functools.wraps(original_preempt)
        def preempt(*args, **kwargs):
            self.preemptions += 1
            group = args[0] if args else kwargs.get("seq_group")
            if group is not None:
                self.request_preemptions[str(group.request_id)] += 1
            return original_preempt(*args, **kwargs)

        scheduler._preempt = preempt
        original_recompute = scheduler._preempt_by_recompute

        @functools.wraps(original_recompute)
        def recompute(*args, **kwargs):
            self.recomputations += 1
            return original_recompute(*args, **kwargs)

        scheduler._preempt_by_recompute = recompute

    def state(self):
        scheduler = self.engine.scheduler
        manager = scheduler.block_manager if self.backend == "fp16" else self.worker.block_manager
        total = int(manager.num_total_gpu_blocks)
        free = int(manager.get_num_free_gpu_blocks())
        usage = 100.0 * (total - free) / total if total else 0.0
        self.max_kv_usage_percent = max(self.max_kv_usage_percent, usage)
        return {"running": len(scheduler.running), "waiting": len(scheduler.waiting),
                "swapped": len(scheduler.swapped), "free_gpu_blocks": free,
                "total_gpu_blocks": total, "kv_usage_percent": usage}

    def step(self, index, origin):
        self.current = None
        preemptions = self.preemptions
        started = time.perf_counter()
        outputs = self.engine.step()
        ended = time.perf_counter()
        observed = self.current or {"phase": "empty", "batch": 0, "request_ids": []}
        elapsed = ended - started
        phase = observed["phase"]
        self.phase_seconds[phase] += elapsed
        self.phase_steps[phase] += 1
        if phase == "decode" and observed["batch"]:
            self.decode_batches[observed["batch"]] += 1
        row = {"step": index, "start_s": started - origin, "end_s": ended - origin,
               "duration_s": elapsed, **observed,
               "preemptions": self.preemptions - preemptions,
               "state_after_step": self.state()}
        return outputs, ended, row

    def summary(self):
        count = sum(self.decode_batches.values())
        return {"phase_seconds": dict(self.phase_seconds), "phase_steps": dict(self.phase_steps),
                "decode_batch_histogram": dict(self.decode_batches),
                "mean_decode_batch": (sum(batch * n for batch, n in self.decode_batches.items()) / count
                                      if count else None),
                "preemption_count": self.preemptions, "recompute_count": self.recomputations,
                "max_kv_usage_percent": self.max_kv_usage_percent}


def warmup(engine, args, requests):
    from vllm import SamplingParams
    if args.warmup_tokens <= 0:
        return
    arrival = time.monotonic()
    for index, row in enumerate(requests):
        add_request(engine, args.backend, f"warmup-{index}", row["prompt_token_ids"],
                    SamplingParams(temperature=0.0, ignore_eos=True,
                                   max_tokens=args.warmup_tokens), arrival)
    while engine.has_unfinished_requests():
        engine.step()
    import torch
    from vllm.model_executor.utils import set_random_seed
    torch.cuda.synchronize()
    set_random_seed(args.seed)
    print(f"WARMUP {args.backend} requests={len(requests)} tokens={args.warmup_tokens}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["fp16", "diffkv"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--mode", choices=["natural", "length_replay"], default="natural")
    parser.add_argument("--lengths-from", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--diffkv-variant", choices=["original", "no-sync", "no-sync-bulk-lengths"],
                        default="no-sync-bulk-lengths")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace result: {args.output}")
    if not 0 < args.max_tokens or args.warmup_tokens < 0:
        raise ValueError("Invalid output length")
    workload = read_workload(args.workload)
    args.engine_options = workload["engine"]
    if args.engine_options.get("seed", args.seed) != args.seed:
        raise ValueError("Engine seed differs from frozen workload")
    requests = workload["requests"]
    digest = file_digest(args.workload)
    limits = length_limits(requests, args.mode, args.max_tokens, args.lengths_from, digest)
    # length_replay is allowed for every backend so that all of them generate exactly
    # the per-request token counts of one FP16 natural run (ignore_eos).
    for row in requests:
        if len(row["prompt_token_ids"]) + limits[row["request_id"]] > args.engine_options["max_model_len"]:
            raise ValueError(f"Input plus output exceeds max_model_len: {row['request_id']}")
    metadata = {"backend": args.backend, "mode": args.mode, "gpu": args.gpu,
                "model": str(args.model.resolve()), "workload": str(args.workload.resolve()),
                "workload_sha256": digest, "seed": args.seed,
                "ignore_eos": args.mode == "length_replay",
                "temperature": args.temperature, "top_p": args.top_p,
                "max_tokens": args.max_tokens, "warmup_tokens": args.warmup_tokens,
                "runner_sha256": file_digest(__file__), "pid": os.getpid(),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "engine": {**COMMON_ENGINE, **args.engine_options, "seed": args.seed},
                "generation": {"temperature": args.temperature, "top_p": args.top_p,
                               "max_tokens": args.max_tokens,
                               "ignore_eos": args.mode == "length_replay", "seed": args.seed},
                "definitions": {
                    "arrival": "All requests registered before first engine step, with one common arrival time",
                    "ttft_s": "Common arrival to first returned output token, includes scheduler queueing",
                    "tpot_s": "(Last output token time - first output token time) / (output_tokens - 1)",
                    "e2e_s": "Common arrival to completed request output, excludes loading and warmup",
                    "step_duration_s": "engine.step wall time, no observer CUDA synchronization",
                    "randomness": "Global engine seed is fixed; different scheduler batches consume different random draws, so per-request sampling draws are not paired",
                }}
    if args.lengths_from:
        metadata["length_reference_sha256"] = file_digest(args.lengths_from)
        metadata["length_reference"] = str(args.lengths_from.resolve())
    result = {"status": "initializing", "metadata": metadata, "requests": []}
    save_json(args.output, result)
    completed = {}
    observer = None
    try:
        engine, worker, engine_metadata = make_engine(args)
        metadata.update(engine_metadata)
        warmup(engine, args, requests)
        from vllm import SamplingParams
        from vllm.model_executor.utils import set_random_seed
        set_random_seed(args.seed)
        observer = StepObserver(engine, worker, args.backend)
        ignore_eos = args.mode == "length_replay"
        params = {row["request_id"]: SamplingParams(
            n=1, temperature=args.temperature, top_p=args.top_p,
            max_tokens=limits[row["request_id"]], ignore_eos=ignore_eos,
            stop_token_ids=[] if ignore_eos else metadata["effective_eos_token_ids"])
                  for row in requests}
        result["status"] = "running"
        save_json(args.output, result)
        origin = time.perf_counter()
        arrival = time.monotonic()
        for row in requests:
            add_request(engine, args.backend, row["request_id"], row["prompt_token_ids"],
                        params[row["request_id"]], arrival)
        metadata["registration_s"] = time.perf_counter() - origin
        first_token = {}
        previous_counts = Counter()
        source_rows = {row["request_id"]: row for row in requests}
        steps_path = args.output.with_suffix(".steps.jsonl")
        result["steps_file"] = str(steps_path.resolve())
        last_progress = origin
        index = 0
        with steps_path.open("x") as steps_stream:
            while engine.has_unfinished_requests():
                outputs, ended, step = observer.step(index, origin)
                index += 1
                emitted = 0
                for output in outputs:
                    request_id = output.request_id
                    if len(output.outputs) != 1:
                        raise RuntimeError("Expected one completion per request")
                    completion = output.outputs[0]
                    count = len(completion.token_ids)
                    increment = count - previous_counts[request_id]
                    if increment < 0 or increment > 1:
                        raise RuntimeError("Unexpected token increment")
                    emitted += increment
                    previous_counts[request_id] = count
                    if count:
                        first_token.setdefault(request_id, ended)
                    if output.finished:
                        if not 0 < count <= limits[request_id]:
                            raise RuntimeError(f"Invalid final output length: {request_id}")
                        if ignore_eos and count != limits[request_id]:
                            raise RuntimeError(f"Length replay stopped early: {request_id}")
                        first = first_token[request_id]
                        completed[request_id] = {
                            "request_id": request_id,
                            "prompt_tokens": len(source_rows[request_id]["prompt_token_ids"]),
                            "max_tokens": limits[request_id], "output_tokens": count,
                            "output_token_ids": list(completion.token_ids),
                            "output_text": completion.text, "finish_reason": completion.finish_reason,
                            "ttft_s": first - origin, "e2e_s": ended - origin,
                            "tpot_s": (ended - first) / (count - 1) if count > 1 else None,
                            "preemptions": observer.request_preemptions[request_id],
                        }
                step["emitted_tokens"] = emitted
                steps_stream.write(json.dumps(step, separators=(",", ":")) + "\n")
                if ended - last_progress >= 30:
                    steps_stream.flush()
                    result["requests"] = [completed[row["request_id"]] for row in requests
                                          if row["request_id"] in completed]
                    result["summary"] = {**observer.summary(), "elapsed_s": ended - origin,
                                         "completed": len(completed),
                                         "generated_tokens": sum(previous_counts.values()),
                                         "state": observer.state()}
                    save_json(args.output, result)
                    print(f"PROGRESS {args.backend} {args.mode} elapsed={ended-origin:.1f}s "
                          f"completed={len(completed)}/{len(requests)} "
                          f"tokens={sum(previous_counts.values())} "
                          f"kv={observer.state()['kv_usage_percent']:.1f}% "
                          f"preemptions={observer.preemptions}", flush=True)
                    last_progress = ended
        if set(completed) != set(source_rows):
            raise RuntimeError("Engine stopped with missing request outputs")
        result["requests"] = [completed[row["request_id"]] for row in requests]
        rows = result["requests"]
        result["summary"] = {
            **observer.summary(), "completed": len(rows),
            "elapsed_s": time.perf_counter() - origin,
            "generated_tokens": sum(row["output_tokens"] for row in rows),
            "mean_output_tokens": statistics.mean(row["output_tokens"] for row in rows),
            "mean_ttft_s": statistics.mean(row["ttft_s"] for row in rows),
            "mean_tpot_s": (statistics.mean(row["tpot_s"] for row in rows if row["tpot_s"] is not None)
                            if any(row["tpot_s"] is not None for row in rows) else None),
            "mean_e2e_s": statistics.mean(row["e2e_s"] for row in rows),
        }
        result["status"] = "completed"
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(args.output, result)
        print(json.dumps({"status": "completed", "backend": args.backend,
                          "mode": args.mode, "output": str(args.output.resolve()),
                          "summary": result["summary"]}), flush=True)
    except BaseException as error:
        result["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
        result["requests"] = list(completed.values())
        if observer is not None:
            result["observer"] = observer.summary()
        save_json(args.output, result)
        raise


if __name__ == "__main__":
    main()
