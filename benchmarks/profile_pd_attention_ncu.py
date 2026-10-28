#!/usr/bin/env python3
"""Capture selected real-model prefill/decode Attention forwards for Nsight Compute.

Run each backend in a fresh process. The fixed B8/ISL2048/OSL2048 default
reproduces the existing nsys workload and engine limits. Nsight Compute should
filter both the push/pop NVTX range ``attention_ncu/`` and the desired attention
kernel names: the forward range also contains cache writers and initialization.
Metadata readbacks are outside that range and outside the CUDA-event interval.
Use --mode event without ncu for separately labelled forward event timings.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import functools
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import run_offline_e2e as worker


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("diffkv", "fp16"), required=True)
    parser.add_argument("--mode", choices=("ncu", "event"), default="ncu")
    parser.add_argument("--capture-api", choices=("generation", "roi"), default="roi",
                        help="CUDA profiler API scope; roi avoids profiling unselected steps")
    parser.add_argument("--gpu", type=int, choices=(0,), default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--input-length", type=int, default=2048)
    parser.add_argument("--output-length", type=int, default=2048)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 15, 31])
    parser.add_argument("--prefill-batches", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--decode-only", action="store_true",
                        help="Profile decode kernels only; do not capture any prefill forward")
    parser.add_argument("--decode-steps", type=int, nargs="+", default=[1, 1024, 2047])
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--inputs", type=Path,
                        default=ROOT / "logs/performance/llama-matched-inputs.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.batch <= 32:
        parser.error("batch must be between 1 and 32")
    if not 1 <= args.input_length <= 2048 or not 2 <= args.output_length <= 2048:
        parser.error("input-length must be 1..2048 and output-length 2..2048")
    if not 0 <= args.warmup_tokens <= 2048 or min(args.layers) < 0:
        parser.error("warmup-tokens must be 0..2048 and layers must be nonnegative")
    if min(args.prefill_batches) < 1 or min(args.decode_steps) < 1:
        parser.error("prefill-batches and decode-steps are one-based positive ordinals")
    if max(args.decode_steps) >= args.output_length:
        parser.error("decode-steps must be less than output-length")
    args.layers = sorted(set(args.layers))
    args.prefill_batches = sorted(set(args.prefill_batches))
    if args.decode_only:
        args.prefill_batches = []
    args.decode_steps = sorted(set(args.decode_steps))
    return args


def tensor_info(value):
    if value is None:
        return None
    return {"shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "device": str(value.device)}


def host_value(value):
    """Read only outside the selected forward ROI and event interval."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [host_value(item) for item in value]
    return value.detach().cpu().tolist()


def metadata_snapshot(metadata, layer, attention, backend, arguments):
    result = {"is_prompt": bool(metadata.is_prompt),
              "prompt_lens": host_value(metadata.prompt_lens),
              "max_context_len": host_value(metadata.max_context_len),
              "use_cuda_graph": bool(metadata.use_cuda_graph)}
    for name in ("num_bits_k_high", "num_bits_v_high", "num_bits_k_low", "num_bits_v_low",
                 "num_chunks_k_high", "num_chunks_v_high", "num_chunks_k_low", "num_chunks_v_low",
                 "key_vec_size", "val_vec_size", "num_tokens_per_block_high",
                 "num_tokens_per_block_low", "block_size"):
        if hasattr(metadata, name):
            result[name] = host_value(getattr(metadata, name))
    if backend == "diffkv":
        slots = host_value(metadata.slot_ids)
        result["slot_ids"] = slots
        # Only selected active slots and this layer, not the entire cache.
        retained = metadata.kv_len_tables[slots, layer].detach().cpu().tolist()
        result["retained_high_low_by_sequence_head"] = retained
        result["retained_total_by_sequence_head"] = [
            [sum(pair) for pair in sequence] for sequence in retained]
        result["retained_length_source"] = "active-slot kv_len_tables at this layer"
        result["compress_config_by_sequence"] = (
            metadata.compress_config_tables[slots].detach().cpu().tolist())
        result["input_positions"] = host_value(arguments.get("input_positions"))
        if not metadata.is_prompt:
            result["attention_context_bound_after_append"] = (
                result["max_context_len"][layer] + 1)
    else:
        lengths = (host_value(metadata.prompt_lens) if metadata.is_prompt
                   else host_value(metadata.context_lens))
        result["logical_context_lens_by_sequence"] = lengths
        result["retained_total_by_sequence_head"] = [
            [length] * attention.num_kv_heads for length in lengths]
        result["retained_length_source"] = (
            "dense logical prompt_lens/context_lens, identical for each KV head; "
            "includes current token and is not a physical cache length table")
    return result


def main():
    args = parse_args()
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    case = {"id": f"b{args.batch}-isl{args.input_length}-osl{args.output_length}",
            "batch": args.batch, "input_length": args.input_length,
            "output_length": args.output_length}
    result_path = folder / f"{case['id']}-{args.backend}.json"
    state = {
        "status": "initializing", "started_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend, "physical_gpu": args.gpu, "mode": args.mode,
        "capture_api": args.capture_api,
        "case": case, "layers": args.layers, "prefill_batches": args.prefill_batches,
        "decode_steps": args.decode_steps,
        "diffkv_variant": "no-sync-bulk-lengths" if args.backend == "diffkv" else None,
        "runner_path": str(Path(__file__).resolve()), "runner_sha256": sha256(__file__),
        "worker_path": str(Path(worker.__file__).resolve()),
        "worker_sha256": sha256(worker.__file__), "worker_result_path": str(result_path),
        "input_manifest_path": str(args.inputs.resolve()),
        "input_manifest_sha256": sha256(args.inputs),
        "engine_limits": {"max_model_len": 4096, "max_num_seqs": 32,
                          "max_num_batched_tokens": 8192, "max_paddings": 4096},
        "roi": {"nvtx_name": "attention_ncu", "nvtx_push_pop_filter": "attention_ncu/",
                "scope": "one selected Attention.forward, including cache writer and workspace",
                "kernel_selection": "external ncu kernel-name filter must select attention kernels",
                "labels": "attention_prefill_bN_lL or attention_decode_sN_lL",
                "readbacks": "before range push and after range pop, outside CUDA event interval",
                "perturbation": "selected metadata readbacks synchronize and can affect cache state"},
        "event_measurement": {
            "scope": "selected entire Attention.forward, including cache writer and initialization",
            "label": "ncu replay diagnostic" if args.mode == "ncu" else "without ncu, selected-call diagnostic",
            "resolved": "after measured generation completion and synchronization",
            "limits": "not isolated main/reduce kernel timing, not an uninstrumented E2E benchmark"},
        "capture_started": False, "capture_stopped": False,
        "profiler_api": {"scope": args.capture_api, "active": False,
                         "start_calls": 0, "stop_calls": 0},
        "selected_calls": [], "engine_steps": [],
    }
    worker.save(folder / "profile.json", state)
    active = {"timings": None, "phase": None, "ordinal": None}
    pending_events = []

    def profiler_start(torch, record=None):
        result = torch.cuda.cudart().cudaProfilerStart()
        state["capture_started"] = True
        state["capture_stopped"] = False
        state["profiler_api"]["active"] = True
        state["profiler_api"]["start_calls"] += 1
        state["profiler_api"]["last_start_result"] = str(result)
        if record is not None:
            record["profiler_api_started"] = True
            record["profiler_api_start_result"] = str(result)

    def profiler_stop(torch, record=None):
        result = torch.cuda.cudart().cudaProfilerStop()
        state["capture_stopped"] = True
        state["profiler_api"]["active"] = False
        state["profiler_api"]["stop_calls"] += 1
        state["profiler_api"]["last_stop_result"] = str(result)
        if record is not None:
            record["profiler_api_stopped"] = True
            record["profiler_api_stop_result"] = str(result)

    original_timings = worker.RequestTimings

    class AttentionTimings(original_timings):
        def __init__(self, engine, guard):
            super().__init__(engine, guard)
            self.phase_ordinals = Counter()
            active["timings"] = self
            original_schedule = engine.scheduler.schedule

            @functools.wraps(original_schedule)
            def schedule(*schedule_args, **schedule_kwargs):
                metadata, output = original_schedule(*schedule_args, **schedule_kwargs)
                if self.enabled and not output.is_empty():
                    phase = "prefill" if output.prompt_run else "decode"
                    self.phase_ordinals[phase] += 1
                    active.update(phase=phase, ordinal=self.phase_ordinals[phase])
                    state["engine_steps"].append({
                        "phase": phase, "phase_ordinal": self.phase_ordinals[phase],
                        "scheduled_sequence_groups": len(metadata)})
                return metadata, output

            engine.scheduler.schedule = schedule

        def begin(self):
            self.phase_ordinals.clear()
            active.update(phase=None, ordinal=None)
            state["engine_steps"] = []
            return super().begin()

    def install_attention_capture(torch, model_worker, model_path):
        # worker.worker applies the v1 patch immediately before this hook.
        layers = model_worker.model_runner.model.model.layers
        if max(args.layers) >= len(layers):
            raise ValueError(f"Requested layer does not exist in {len(layers)}-layer model")
        import vllm._C as extension
        extension_path = Path(extension.__file__).resolve()
        state["provenance"] = {
            "extension_path": str(extension_path), "extension_sha256": sha256(extension_path),
            "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
            "model_path": str(model_path), "layer_count": len(layers),
            "instrumented_attention": [],
        }
        for layer_index in args.layers:
            attention = layers[layer_index].self_attn.attn
            if args.backend == "diffkv" and getattr(attention, "use_gqa_reuse", False):
                raise RuntimeError("Expected the legacy DiffKV-v1 attention path")
            if args.backend == "diffkv" and attention.layer != layer_index:
                raise RuntimeError("Attention cache-layer ID differs from model layer index")
            original_forward = attention.forward
            signature = inspect.signature(original_forward)
            source = inspect.getsource(original_forward)
            state["provenance"]["instrumented_attention"].append({
                "layer": layer_index,
                "class": f"{type(attention).__module__}.{type(attention).__name__}",
                "forward_source_file": original_forward.__code__.co_filename,
                "forward_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "signature": str(signature), "num_heads": attention.num_heads,
                "num_kv_heads": attention.num_kv_heads, "head_size": attention.head_size,
                "scale": attention.scale,
                "cache_layer_id": getattr(attention, "layer", None),
            })

            def make_forward(original, call_signature, selected_layer):
                @functools.wraps(original)
                def forward(self, *forward_args, **forward_kwargs):
                    timings = active["timings"]
                    if timings is None or not timings.enabled:
                        return original(*forward_args, **forward_kwargs)
                    phase, ordinal = active["phase"], active["ordinal"]
                    chosen = args.prefill_batches if phase == "prefill" else args.decode_steps
                    if phase not in ("prefill", "decode") or ordinal not in chosen:
                        return original(*forward_args, **forward_kwargs)
                    bound = call_signature.bind(*forward_args, **forward_kwargs)
                    bound.apply_defaults()
                    metadata = bound.arguments["input_metadata"]
                    if bool(metadata.is_prompt) != (phase == "prefill"):
                        raise RuntimeError("Scheduler phase and Attention metadata disagree")
                    label = (f"attention_prefill_b{ordinal}_l{selected_layer}" if phase == "prefill"
                             else f"attention_decode_s{ordinal}_l{selected_layer}")
                    record = {"label": label, "phase": phase, "phase_ordinal": ordinal,
                              "layer": selected_layer, "status": "entered",
                              "tensor_inputs": {name: tensor_info(value)
                                                for name, value in bound.arguments.items()
                                                if torch.is_tensor(value)},
                              "before": metadata_snapshot(metadata, selected_layer, self,
                                                          args.backend, bound.arguments)}
                    state["selected_calls"].append(record)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    record["profiler_api_started"] = False
                    record["profiler_api_stopped"] = False
                    roi_api = args.mode == "ncu" and args.capture_api == "roi"
                    if roi_api:
                        profiler_start(torch, record)
                    ranges_open = 0
                    event_started = False
                    try:
                        start.record()
                        event_started = True
                        torch.cuda.nvtx.range_push(label)
                        ranges_open += 1
                        torch.cuda.nvtx.range_push("attention_ncu")
                        ranges_open += 1
                        output = original(*forward_args, **forward_kwargs)
                    finally:
                        try:
                            for _ in range(ranges_open):
                                torch.cuda.nvtx.range_pop()
                            if event_started:
                                end.record()
                        finally:
                            if roi_api and state["profiler_api"]["active"]:
                                profiler_stop(torch, record)
                    record["output"] = tensor_info(output)
                    record["after"] = metadata_snapshot(metadata, selected_layer, self,
                                                        args.backend, bound.arguments)
                    record["status"] = "returned"
                    pending_events.append((record, start, end))
                    worker.save(folder / "profile.json", state)
                    print(f"ATTENTION CAPTURED {label}", flush=True)
                    return output

                return forward

            attention.forward = MethodType(make_forward(original_forward, signature, layer_index),
                                           attention)
        state["status"] = "ready"
        worker.save(folder / "profile.json", state)
        return {"status": "not_run", "reason": "performance profiling, no checkpoint value checks",
                "attention_instrumentation": state["provenance"]}

    original_generate = worker.generate

    @functools.wraps(original_generate)
    def generate(*generate_args, **generate_kwargs):
        timings = active["timings"]
        if timings is None or not timings.enabled:
            return original_generate(*generate_args, **generate_kwargs)
        import torch
        state["status"] = "capturing"
        generation_api = args.mode == "ncu" and args.capture_api == "generation"
        if generation_api:
            profiler_start(torch)
        try:
            outputs = original_generate(*generate_args, **generate_kwargs)
            torch.cuda.synchronize()
        finally:
            if generation_api and state["profiler_api"]["active"]:
                profiler_stop(torch)
            worker.save(folder / "profile.json", state)
        for record, start, end in pending_events:
            record["cuda_event_forward_ms"] = start.elapsed_time(end)
        state["generation"] = {
            "completed_requests": len(outputs), "finished": [bool(item.finished) for item in outputs],
            "output_counts": [len(item.outputs[0].token_ids) for item in outputs],
            "output_sha256": worker.digest([item.outputs[0].token_ids for item in outputs]),
            "observed_phase_counts": dict(timings.phase_ordinals),
        }
        worker.save(folder / "profile.json", state)
        return outputs

    worker.RequestTimings = AttentionTimings
    worker.verify_weights = install_attention_capture
    worker.generate = generate
    worker_args = argparse.Namespace(
        worker_backend=args.backend, gpu=args.gpu, output_dir=folder,
        diffkv_variant="no-sync-bulk-lengths", inputs=args.inputs.resolve(),
        # Engine limits stay equal to the old whole-grid measurements.
        isls=list(worker.LENGTHS), osls=list(worker.LENGTHS), batches=list(worker.BATCHES),
        warmup_tokens=args.warmup_tokens, repeats=1)
    source, prompts = worker.load_inputs(worker_args.inputs, {args.input_length})
    code = worker.worker(worker_args, case, source, prompts)
    result = json.loads(result_path.read_text())
    expected = {(phase, ordinal, layer) for phase, ordinals in
                (("prefill", args.prefill_batches), ("decode", args.decode_steps))
                for ordinal in ordinals for layer in args.layers}
    actual = [(row["phase"], row["phase_ordinal"], row["layer"])
              for row in state["selected_calls"] if row["status"] == "returned"]
    state["selection_validation"] = {
        "expected_calls": len(expected), "observed_calls": len(actual),
        "missing": [list(item) for item in sorted(expected - set(actual))],
        "unexpected": [list(item) for item in sorted(set(actual) - expected)],
        "duplicates": [list(item) for item, count in Counter(actual).items() if count != 1],
    }
    valid = len(actual) == len(expected) and set(actual) == expected
    state.update(worker_returncode=code, worker_status=result.get("status"),
                 worker_samples=result.get("samples", []),
                 finished_utc=datetime.now(timezone.utc).isoformat())
    state["status"] = ("completed" if result.get("status") == "completed" and valid
                       else result.get("status") if result.get("status") != "completed"
                       else "incomplete_selection")
    worker.save(folder / "profile.json", state)
    return code if state["status"] == "completed" else (code or 1)


if __name__ == "__main__":
    raise SystemExit(main())
