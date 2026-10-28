"""Compare isolated DiffKV CUDA extensions with a real cached Llama-3 model.

Capture each binary in a fresh process on physical GPU 1, then compare the
saved reports. The in-place extension and shared build manifest are untouched.
This measures full model generation, including prefill, decode, sampling and
detokenization. It does not isolate CUDA attention time.
"""

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / ".cache/models/Meta-Llama-3-8B"
PROMPT_SEED = "The capital of France is Paris. The sky is blue. Solve the next question carefully. "
PYTHON_SOURCES = (
    "benchmarks/diffkv_v1.py", "benchmarks/run_offline_e2e.py",
    "benchmarks/benchmark_diffkv_kernel_model.py", "vllm/config.py",
    "vllm/engine/llm_engine.py", "vllm/worker/model_runner.py",
    "vllm/model_executor/models/sparse_llama.py",
    "vllm/model_executor/layers/sparse_attention_big_kernel.py",
    "vllm/model_executor/layers/sampler.py",
)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def binary_provenance(extension, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    actual = file_hash(extension)
    if manifest.get("extension_sha256") != actual:
        raise ValueError("The isolated build manifest does not identify the selected extension")
    built_sources = manifest.get("source_sha256")
    if not isinstance(built_sources, dict) or not built_sources:
        raise ValueError("The isolated build manifest must record source_sha256 hashes")
    current_sources = {name: file_hash(ROOT / name) if (ROOT / name).is_file() else None
                       for name in built_sources}
    return {"extension": str(extension), "extension_sha256": actual,
            "build_manifest": str(manifest_path), "build_manifest_sha256": file_hash(manifest_path),
            "built_source_sha256": built_sources, "current_source_sha256": current_sources,
            "sources_differ_from_build": [name for name in built_sources
                                           if built_sources[name] != current_sources[name]],
            "source_note": "Build-time hashes describe this isolated binary. Current CUDA source may describe another candidate.",
            "build_command": manifest.get("command"), "built_utc": manifest.get("built_utc")}


def load_extension(path):
    if "vllm" in sys.modules or "vllm._C" in sys.modules:
        raise RuntimeError("Capture needs a fresh process without vllm already imported")
    spec = importlib.util.spec_from_file_location("vllm._C", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load isolated CUDA extension: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["vllm._C"] = module
    spec.loader.exec_module(module)
    return module


def run(args):
    if args.gpu != 1:
        raise ValueError("This helper is restricted to physical GPU 1")
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    extension = args.extension.resolve(strict=True)
    manifest_path = (args.build_manifest or extension.parent / "build-manifest.json").resolve(strict=True)
    model = args.model.resolve(strict=True)
    for required in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
        if not (model / required).is_file():
            raise ValueError(f"Cached Llama-3 model is missing {required}: {model}")
    model_index = json.loads((model / "model.safetensors.index.json").read_text())
    if any(not (model / name).is_file() for name in set(model_index["weight_map"].values())):
        raise ValueError("Cached model weights are incomplete")
    provenance = binary_provenance(extension, manifest_path)
    python_hashes = {name: file_hash(ROOT / name) for name in PYTHON_SOURCES}
    common = dict(model=str(model), tokenizer=str(model), dtype="float16",
                  load_format="safetensors", tensor_parallel_size=1,
                  max_model_len=args.isl + args.osl, max_num_batched_tokens=8192,
                  max_num_seqs=64, max_paddings=4096, kv_buffer_size=64,
                  swap_space=0, gpu_memory_utilization=0.9, enforce_eager=True,
                  disable_log_stats=True, seed=args.seed)
    sampling_args = dict(temperature=0.0, ignore_eos=True, n=1,
                         stop=[], stop_token_ids=[])
    result = {"schema_version": 1, "status": "initializing",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "physical_gpu": 1, "label": args.label, "binary": provenance,
              "python_source_sha256": python_hashes,
              "model_metadata_sha256": {name: file_hash(model / name) for name in
                                         ("config.json", "model.safetensors.index.json", "tokenizer_config.json")},
              "engine_args": common, "sampling": sampling_args,
              "diffkv_variant": "no-sync-bulk-lengths",
              "attention_path": args.attention_path,
              "record_warmup_cuda_kernels": args.record_kernels,
              "quant_configs": [8, 4, 4, 2], "quant_groups": [1, 1, 1, 1],
              "compress_configs": [0.02, 1.0],
              "workload": {"batch": args.batch, "input_length": args.isl,
                           "output_length": args.osl, "repeats": args.repeats,
                           "warmup_batch": args.warmup_batch, "warmup_tokens": args.warmup_tokens},
              "measurement": "Synchronized wall time around LLM.generate, excludes model loading, warmup and verification. Phase times sum engine.step wall time without extra synchronization.",
              "samples": []}
    save(args.output, result)
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES="1", OMP_NUM_THREADS="2",
                          TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
                          TRANSFORMERS_OFFLINE="1", HF_HOME=str(ROOT / ".cache/huggingface"),
                          XDG_CACHE_HOME=str(ROOT / ".cache"),
                          TRITON_CACHE_DIR=str(ROOT / ".cache/triton"))
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "benchmarks"))
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one visible GPU, physical GPU 1, is required")
        torch.cuda.set_device(0)
        selected_extension = load_extension(extension)
        import vllm
        from vllm import LLM, SamplingParams
        from diffkv_v1 import apply_variant
        from run_offline_e2e import (FixedBatchGuard, RequestTimings, generate,
                                     get_worker, verify_cache, verify_outputs, verify_weights)
        if Path(vllm.__file__).resolve().parent != ROOT / "vllm":
            raise RuntimeError(f"Wrong Python engine package: {vllm.__file__}")
        if sys.modules.get("vllm._C") is not selected_extension:
            raise RuntimeError("The selected isolated CUDA extension was replaced during import")
        vllm._C = selected_extension
        props = torch.cuda.get_device_properties(0)
        result.update(gpu_name=props.name, gpu_uuid=str(getattr(props, "uuid", "")),
                      torch=torch.__version__, cuda=torch.version.cuda,
                      extension_import=str(Path(selected_extension.__file__).resolve()),
                      gpu_memory_before_load=list(torch.cuda.mem_get_info()))
        print(f"LOAD {args.label}: {extension}", flush=True)
        llm = LLM(**common)
        engine = llm.llm_engine
        worker = get_worker(engine, "diffkv")
        result["python_variant"] = apply_variant(
            engine, worker, "no-sync-bulk-lengths",
            args.output.parent / (args.output.stem + ".ablation"))
        attentions = [layer.self_attn.attn for layer in worker.model_runner.model.model.layers]
        if args.attention_path == "legacy":
            for attention in attentions:
                attention.use_gqa_reuse = False
        result["attention_routing"] = {
            "requested": args.attention_path,
            "layers": [{"layer": index, "use_gqa_reuse": bool(attention.use_gqa_reuse),
                        "head_size": attention.head_size,
                        "queries_per_kv": attention.num_queries_per_kv}
                       for index, attention in enumerate(attentions)],
            "note": "default preserves the installed production routing; legacy disables GQA before the existing warmup",
        }
        if args.record_kernels and args.attention_path == "default" and any(
                attention.head_size == 128 and attention.num_queries_per_kv == 4
                and not attention.use_gqa_reuse for attention in attentions):
            raise RuntimeError("Supported default GQA routing is disabled; apply the production routing patch before this capture")
        result["weights_verification"] = verify_weights(torch, worker, model)
        result["cache_verification"] = verify_cache(torch, worker, "diffkv", engine)
        tokenizer = llm.get_tokenizer()
        seed_tokens = tokenizer.encode(PROMPT_SEED, add_special_tokens=False)
        if not seed_tokens:
            raise RuntimeError("Synthetic prompt seed encoded to no tokens")
        prompt = (seed_tokens * ((args.isl + len(seed_tokens) - 1) // len(seed_tokens)))[:args.isl]
        prompts = [list(prompt) for _ in range(args.batch)]
        result["input"] = {"seed_text": PROMPT_SEED, "seed_token_ids": seed_tokens,
                           "prompt_token_ids": prompt, "prompt_sha256": digest(prompt),
                           "batch_sha256": digest(prompts)}
        result["status"] = "warming_up"
        save(args.output, result)
        profiler = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
                    if args.record_kernels else nullcontext())
        with profiler as captured:
            warm = generate(llm, SamplingParams(max_tokens=args.warmup_tokens, **sampling_args),
                            [list(prompt) for _ in range(args.warmup_batch)], "diffkv")
            torch.cuda.synchronize()
        if args.record_kernels:
            histogram = Counter(event.name for event in captured.events()
                                if event.device_type == torch.autograd.DeviceType.CUDA)
            result["warmup_cuda_kernels"] = dict(sorted(histogram.items()))
            gqa_names = ("register_gqa_attention_kernel", "sparse_gqa_attention_kernel")
            observed = set()
            if any("sparse_paged_attention_wrapper" in name for name in histogram):
                observed.add("legacy")
            if any(any(token in name for token in gqa_names) for name in histogram):
                observed.add("gqa")
            # This benchmark fixes FP16, K8V4/K4V2, chunks1 and the public
            # page layout. Geometry and the production enable flag complete
            # the dispatch conditions for its cached Llama-3 model.
            expected = {"gqa" if attention.use_gqa_reuse and attention.head_size == 128
                        and attention.num_queries_per_kv == 4 else "legacy"
                        for attention in attentions}
            result["warmup_kernel_verification"] = {
                "expected_attention_families": sorted(expected),
                "observed_attention_families": sorted(observed),
                "passed": expected == observed,
                "scope": "CUDA kernel names from the existing warmup only; no added generation call",
            }
            if observed != expected:
                raise RuntimeError(f"Warmup attention kernels differ from routing: expected {expected}, observed {observed}")
            print(json.dumps({"warmup_kernel_verification": result["warmup_kernel_verification"],
                              "attention_cuda_kernels": {name: count for name, count in histogram.items()
                                                         if "sparse_paged_attention_wrapper" in name
                                                         or any(token in name for token in gqa_names)}}), flush=True)
        verify_outputs(warm, prompt, args.warmup_batch, args.warmup_tokens)
        if engine.has_unfinished_requests():
            raise RuntimeError("Unfinished requests after warmup")
        guard = FixedBatchGuard(engine, args.batch)
        timings = RequestTimings(engine, guard)
        result["status"] = "running"
        save(args.output, result)
        for repeat in range(args.repeats):
            guard.reset()
            torch.manual_seed(args.seed)
            timings.begin()
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = generate(llm, SamplingParams(max_tokens=args.osl, **sampling_args),
                               prompts, "diffkv")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError("Invalid synchronized generation duration")
            phases = timings.end(started, args.batch, args.isl, args.osl)
            counts = verify_outputs(outputs, prompt, args.batch, args.osl)
            if engine.has_unfinished_requests() or guard.summary()["decode_steps"] != args.osl - 1:
                raise RuntimeError("Measured generation did not preserve the fixed workload")
            token_ids = [out.outputs[0].token_ids for out in outputs]
            sample = {"repeat": repeat + 1, "elapsed_s": elapsed,
                      "generated_tokens": sum(counts), "output_tokens_s": sum(counts) / elapsed,
                      "output_sha256": digest(token_ids), "output_token_ids": token_ids,
                      **guard.summary(),
                      **{name: phases[name] for name in ("prefill_s", "decode_s", "prefill_tokens_s",
                                                         "decode_tokens_s", "ttft_ms", "tpot_ms")}}
            result["samples"].append(sample)
            save(args.output, result)
            print(json.dumps({key: value for key, value in sample.items()
                              if key not in ("output_token_ids",)}), flush=True)
        if file_hash(extension) != provenance["extension_sha256"]:
            raise RuntimeError("The selected extension changed during capture")
        result["status"] = "completed"
        result["summary"] = {"elapsed_median_s": statistics.median(s["elapsed_s"] for s in result["samples"]),
                             "elapsed_mean_s": statistics.mean(s["elapsed_s"] for s in result["samples"]),
                             "decode_tokens_s_median": statistics.median(s["decode_tokens_s"] for s in result["samples"]),
                             "output_hashes": [s["output_sha256"] for s in result["samples"]],
                             "repeat_outputs_identical": len({s["output_sha256"] for s in result["samples"]}) == 1}
        save(args.output, result)
        print(json.dumps(result["summary"], indent=2), flush=True)
        return 0
    except BaseException as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        save(args.output, result)
        raise


def compare(args):
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace comparison: {args.output}")
    before, after = (json.loads(path.read_text()) for path in (args.baseline, args.candidate))
    keys = ("schema_version", "physical_gpu", "gpu_name", "gpu_uuid", "torch", "cuda",
            "engine_args", "sampling", "diffkv_variant", "quant_configs", "quant_groups",
            "compress_configs", "workload", "model_metadata_sha256", "python_source_sha256", "input")
    checks = {key: key in before and key in after and before[key] == after[key] for key in keys}
    checks["both_completed"] = before.get("status") == after.get("status") == "completed"
    checks["repeat_outputs_stable"] = all(report.get("summary", {}).get("repeat_outputs_identical", False)
                                           for report in (before, after))
    hashes_before = [row["output_sha256"] for row in before.get("samples", [])]
    hashes_after = [row["output_sha256"] for row in after.get("samples", [])]
    checks["output_tokens_identical"] = bool(hashes_before) and hashes_before == hashes_after
    report = {"status": "passed" if all(checks.values()) else "failed",
              "checks": checks, "baseline": str(args.baseline.resolve()),
              "candidate": str(args.candidate.resolve()),
              "baseline_binary": before["binary"], "candidate_binary": after["binary"]}
    report["attention_routing"] = {"baseline": before.get("attention_routing"),
                                   "candidate": after.get("attention_routing")}
    report["warmup_kernel_verification"] = {"baseline": before.get("warmup_kernel_verification"),
                                            "candidate": after.get("warmup_kernel_verification")}
    if checks["both_completed"]:
        report.update(baseline_summary=before["summary"], candidate_summary=after["summary"],
                      e2e_speedup=before["summary"]["elapsed_median_s"] / after["summary"]["elapsed_median_s"],
                      decode_throughput_speedup=after["summary"]["decode_tokens_s_median"] / before["summary"]["decode_tokens_s_median"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if all(checks.values()) else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("capture", "compare"), default="capture")
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--build-manifest", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--label", default="isolated-extension")
    parser.add_argument("--attention-path", choices=("default", "legacy"), default="default")
    parser.add_argument("--record-kernels", action="store_true",
                        help="Record and verify CUDA attention kernel names inside the existing warmup, without extra generation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(1,), default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--isl", type=int, default=2048)
    parser.add_argument("--osl", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-batch", type=int, default=2)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "capture" and args.extension is None:
        parser.error("capture requires --extension")
    if args.mode == "compare" and (args.baseline is None or args.candidate is None):
        parser.error("compare requires --baseline and --candidate")
    if (min(args.batch, args.isl, args.repeats, args.warmup_batch, args.warmup_tokens) < 1
            or args.osl < 2 or args.batch > 64 or args.warmup_batch > 64
            or args.isl > 8192 or args.warmup_tokens > args.osl):
        parser.error("Invalid fixed workload or warmup dimensions")
    if args.record_kernels and args.warmup_tokens < 2:
        parser.error("--record-kernels needs at least two warmup tokens to observe decode attention")
    return args


if __name__ == "__main__":
    options = parse_args()
    raise SystemExit(run(options) if options.mode == "capture" else compare(options))
