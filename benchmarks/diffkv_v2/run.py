# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the frozen MATH burst through vLLM 0.32's LLMEngine.

One backend per process. Tokenization, model loading, warmup and grading are
outside the measured interval. Importing this module does not initialize CUDA.
"""

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from common import (
    BACKEND_OPTIONS,
    OutputRecorder,
    backend_options,
    cache_metadata,
    load_workload,
    output_limits,
    request_signature,
    save_json,
    sha256,
    summarize,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def engine_options(model):
    return {
        "model": str(model.resolve()),
        "tokenizer": str(model.resolve()),
        "dtype": "float16",
        "load_format": "safetensors",
        "tensor_parallel_size": 1,
        "seed": 42,
        "max_model_len": 9216,
        "max_num_batched_tokens": 9216,
        "max_num_seqs": 64,
        "gpu_memory_utilization": 0.9,
        "enforce_eager": True,
        "enable_chunked_prefill": True,
        "async_scheduling": False,
        "enable_prefix_caching": False,
        "speculative_config": None,
        "generation_config": "vllm",
        "disable_log_stats": False,
        "stream_interval": 1,
    }


class SchedulerRecorder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.preemptions = 0
        self.peak_kv_usage = 0.0
        self.peak_running = 0
        self.samples = 0

    def record(self, scheduler_stats, iteration_stats):
        if scheduler_stats is not None:
            self.peak_kv_usage = max(self.peak_kv_usage, scheduler_stats.kv_cache_usage)
            self.peak_running = max(self.peak_running, scheduler_stats.num_running_reqs)
            self.samples += 1
        if iteration_stats is not None:
            self.preemptions += iteration_stats.num_preempted_reqs

    def summary(self):
        return {
            "preemption_count": self.preemptions,
            "peak_scheduler_kv_usage_percent": 100 * self.peak_kv_usage,
            "peak_running_requests": self.peak_running,
            "scheduler_samples": self.samples,
        }


def create_engine(options, recorder):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.metrics.loggers import StatLoggerBase

    class Observer(StatLoggerBase):
        def __init__(self, vllm_config, engine_index=0):
            pass

        def record(
            self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0
        ):
            recorder.record(scheduler_stats, iteration_stats)

        def log_engine_initialized(self):
            pass

    return LLMEngine.from_engine_args(
        EngineArgs(**options),
        stat_loggers=[Observer],
        enable_multiprocessing=False,
    )


def eos_ids(model, tokenizer):
    path = model / "generation_config.json"
    configured = json.loads(path.read_text()).get("eos_token_id", [])
    if configured is None:
        configured = []
    if isinstance(configured, int):
        configured = [configured]
    result = [*configured, tokenizer.eos_token_id]
    result = [token for token in result if token is not None]
    if any(type(token) is not int or token < 0 for token in result):
        raise ValueError("Invalid checkpoint EOS token IDs")
    return sorted(set(result))


def provenance():
    def git(*arguments):
        return subprocess.check_output(
            ["git", *arguments],
            cwd=ROOT,
            text=True,
        ).strip()

    import torch

    import vllm

    return {
        "commit": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git("status", "--porcelain", "-uno")),
        "vllm_version": vllm.__version__,
        "vllm_file": str(Path(vllm.__file__).resolve()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_properties": str(torch.cuda.get_device_properties(0)),
        "gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "")),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING"),
        "runtime_environment": {
            name: os.environ.get(name)
            for name in (
                "VLLM_ENABLE_V1_MULTIPROCESSING",
                "XDG_CACHE_HOME",
                "FLASHINFER_WORKSPACE_BASE",
                "FLASHINFER_EXTRA_CUDAFLAGS",
                "VLLM_CONFIG_ROOT",
                "VLLM_NO_USAGE_STATS",
            )
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=tuple(BACKEND_OPTIONS), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workload", type=Path, default=HERE / "fixtures/math50.json")
    parser.add_argument("--expected-count", type=int, choices=(50, 500), default=50)
    parser.add_argument("--mode", choices=("natural", "length_replay"), required=True)
    parser.add_argument("--lengths-from", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    workload = load_workload(args.workload, args.expected_count)
    limits = output_limits(workload, args.workload, args.mode, args.lengths_from)
    requests = workload["requests"]
    if args.warmup_tokens < 0:
        raise ValueError("Warmup length must be nonnegative")
    for row in requests:
        if (
            len(row["prompt_token_ids"])
            + max(
                limits[row["request_id"]],
                args.warmup_tokens,
            )
            > 9216
        ):
            raise ValueError(f"Input plus output exceeds 9216: {row['request_id']}")
    options = engine_options(args.model)
    options.update(backend_options(args.backend))
    metadata = {
        "backend": args.backend,
        "mode": args.mode,
        "workload_sha256": sha256(args.workload),
        "request_signature": request_signature(workload),
        "expected_count": args.expected_count,
        "engine": options,
        "kv_cache": cache_metadata(args.backend),
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "max_tokens": 8192,
            "ignore_eos": args.mode == "length_replay",
        },
        "warmup_tokens": args.warmup_tokens,
        "length_reference_sha256": sha256(args.lengths_from)
        if args.lengths_from
        else None,
        "runner_sha256": sha256(__file__),
        "implementation_limits": {
            "new_request_admissions_per_step": 1 if args.backend == "diffkv" else None,
            "note": "DiffKV reserves one contiguous raw prefill allocation per "
            "step; existing decode requests remain batched. This serving cost "
            "is included in the comparison with the native scheduler.",
        },
        "definitions": {
            "arrival": "All requests registered before the first engine step",
            "elapsed_s": "Common arrival before registration to final completed output",
            "output_tokens_per_s": "Completed generated tokens divided by elapsed_s",
            "ttft_s": "Common arrival to first observed generated token",
            "tpot_s": "(Last token observation - first) / (generated tokens - 1)",
            "e2e_s": "Common arrival to completed request output",
            "peak_scheduler_kv_usage_percent": "Upstream scheduler metric; a custom "
            "backend's physical pool occupancy may differ",
        },
    }
    if args.validate_only:
        print(
            json.dumps(
                {
                    "metadata": metadata,
                    "requests": len(requests),
                    "output_token_budget": sum(limits.values()),
                },
                indent=2,
            )
        )
        return
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # In-process execution guarantees registration completes before scheduling.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    os.environ.setdefault(
        "FLASHINFER_WORKSPACE_BASE",
        str(ROOT / ".cache/bringup/flashinfer"),
    )
    os.environ.setdefault(
        "VLLM_CONFIG_ROOT",
        str(ROOT / ".cache/bringup/vllm-config"),
    )
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    result = {
        "schema_version": 1,
        "status": "initializing",
        "metadata": metadata,
        "requests": [],
    }
    save_json(args.output, result)
    engine = None
    try:
        from tokenizers import Tokenizer

        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        tokenizer_path = args.model / "tokenizer.json"
        decoder_hash = sha256(tokenizer_path)
        expected_hash = workload["tokenizer"]["files_sha256"]["tokenizer.json"]
        if decoder_hash != expected_hash:
            raise ValueError("Checkpoint tokenizer.json differs from frozen workload")
        decoder = Tokenizer.from_file(str(tokenizer_path))
        metadata["text_decoder"] = {
            "implementation": "tokenizers.Tokenizer.from_file",
            "tokenizer_json_sha256": decoder_hash,
            "skip_special_tokens": True,
            "outside_measured_interval": True,
        }
        stats = SchedulerRecorder()
        engine = create_engine(options, stats)
        metadata.update(provenance())
        forward_context = engine.vllm_config.compilation_config.static_forward_context
        attention_layers = [
            layer
            for layer in forward_context.values()
            if hasattr(layer, "attn_backend")
        ]
        metadata["kv_cache"]["resolved_cache_dtype"] = (
            engine.vllm_config.cache_config.cache_dtype
        )
        metadata["kv_cache"]["resolved_attention_backends"] = sorted(
            {layer.attn_backend.get_name() for layer in attention_layers}
        )
        if args.backend == "fp8":
            metadata["kv_cache"]["loaded_per_tensor_scales"] = {
                kind: sorted(
                    {
                        float(getattr(layer, f"_{kind}_scale_float"))
                        for layer in attention_layers
                    }
                )
                for kind in ("k", "v")
            }
        metadata["effective_eos_token_ids"] = eos_ids(
            args.model,
            engine.get_tokenizer(),
        )
        metadata["resolved_engine"] = {
            "async_scheduling": engine.vllm_config.scheduler_config.async_scheduling,
            "enable_chunked_prefill": (
                engine.vllm_config.scheduler_config.enable_chunked_prefill
            ),
            "enable_prefix_caching": (
                engine.vllm_config.cache_config.enable_prefix_caching
            ),
        }
        metadata["started_utc"] = datetime.now(UTC).isoformat()
        if args.warmup_tokens:
            for index, row in enumerate(requests):
                engine.add_request(
                    f"warmup-{index}",
                    {"prompt_token_ids": row["prompt_token_ids"]},
                    SamplingParams(
                        temperature=0.0,
                        top_p=1.0,
                        seed=42,
                        ignore_eos=True,
                        max_tokens=args.warmup_tokens,
                    ),
                )
            while engine.has_unfinished_requests():
                engine.step()
        stats.reset()
        recorder = OutputRecorder(requests, limits, args.mode)
        internal_to_external = {}
        params = {
            row["request_id"]: SamplingParams(
                n=1,
                temperature=0.0,
                top_p=1.0,
                seed=42,
                max_tokens=limits[row["request_id"]],
                ignore_eos=args.mode == "length_replay",
                stop_token_ids=[]
                if args.mode == "length_replay"
                else metadata["effective_eos_token_ids"],
                output_kind=RequestOutputKind.CUMULATIVE,
            )
            for row in requests
        }
        result["status"] = "running"
        save_json(args.output, result)
        started = time.perf_counter()
        arrival = time.time()
        for row in requests:
            request_id = row["request_id"]
            internal_id = engine.add_request(
                request_id,
                {"prompt_token_ids": row["prompt_token_ids"]},
                params[request_id],
                arrival_time=arrival,
            )
            internal_to_external[internal_id] = request_id
        metadata["registration_s"] = time.perf_counter() - started
        last_progress = started
        elapsed = 0.0
        while engine.has_unfinished_requests():
            outputs = engine.step()
            now = time.perf_counter()
            elapsed = now - started
            for output in outputs:
                output.request_id = internal_to_external.get(
                    output.request_id,
                    output.request_id,
                )
                recorder.observe(output, elapsed)
            if now - last_progress >= 30:
                result["requests"] = list(recorder.completed.values())
                result["progress"] = {
                    "elapsed_s": elapsed,
                    "completed": len(recorder.completed),
                    "generated_tokens": sum(recorder.counts.values()),
                    **stats.summary(),
                }
                save_json(args.output, result)
                print(json.dumps(result["progress"]), flush=True)
                last_progress = now
        result["requests"] = recorder.rows()
        mismatches = 0
        for row in result["requests"]:
            row["engine_output_text"] = row["output_text"]
            row["output_text"] = decoder.decode(
                row["output_token_ids"],
                skip_special_tokens=True,
            )
            mismatches += row["output_text"] != row["engine_output_text"]
        metadata["engine_text_mismatch_count"] = mismatches
        result["summary"] = {
            **summarize(result["requests"], elapsed),
            **stats.summary(),
        }
        result["status"] = "completed"
        metadata["finished_utc"] = datetime.now(UTC).isoformat()
        save_json(args.output, result)
        print(json.dumps(result["summary"], indent=2), flush=True)
    except BaseException as error:
        result["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        result["error"] = f"{type(error).__name__}: {error}"
        save_json(args.output, result)
        raise
    finally:
        if engine is not None:
            engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
