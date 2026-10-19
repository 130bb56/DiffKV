#!/usr/bin/env python3
"""Capture one fresh FP16/DiffKV workload on physical GPU 0 with Nsight Systems.

Run a separate process for each backend and case. CUDA profiler API brackets
the complete measured generation, excluding model loading and warmup. NVTX
engine_step_N ranges include scheduling; prefill and decode_step_N ranges
start after scheduling identifies the phase and end when engine.step returns.
"""

import argparse
from datetime import datetime, timezone
import functools
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import run_offline_e2e as worker


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("fp16", "diffkv"), required=True)
    parser.add_argument("--gpu", type=int, choices=(0,), default=0,
                        help="Physical GPU index; this collection is restricted to GPU 0")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--input-length", type=int, required=True)
    parser.add_argument("--output-length", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--inputs", type=Path,
                        default=ROOT / "logs/performance/llama-matched-inputs.json")
    args = parser.parse_args()
    if not 1 <= args.batch <= max(worker.BATCHES):
        parser.error("batch must be between 1 and the fixed max_num_seqs of 32")
    if not 1 <= args.input_length <= max(worker.LENGTHS):
        parser.error("input-length must be between 1 and 2048")
    if not 2 <= args.output_length <= max(worker.LENGTHS):
        parser.error("output-length must be between 2 and 2048")
    if not 0 <= args.warmup_tokens <= max(worker.LENGTHS):
        parser.error("warmup-tokens must be between 0 and 2048")
    return args


def main():
    args = parse_args()
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    case = {
        "id": f"b{args.batch}-isl{args.input_length}-osl{args.output_length}",
        "batch": args.batch,
        "input_length": args.input_length,
        "output_length": args.output_length,
    }
    result_path = folder / f"{case['id']}-{args.backend}.json"
    state = {
        "status": "initializing",
        "backend": args.backend,
        "physical_gpu": args.gpu,
        "case": case,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256(__file__),
        "worker_path": str(Path(worker.__file__).resolve()),
        "worker_sha256": sha256(worker.__file__),
        "worker_result_path": str(result_path),
        "capture": "whole_generation",
        "measurement": "Profiled generation; model loading and warmup excluded.",
        "correctness_validation": "Numerical correctness checks skipped; workload and capacity guards retained.",
        "engine_limits": {
            "max_model_len": 4096,
            "max_num_seqs": 32,
            "max_num_batched_tokens": 8192,
            "max_paddings": 4096,
        },
        "diffkv_variant": "no-sync-bulk-lengths" if args.backend == "diffkv" else None,
        "capture_started": False,
        "capture_stopped": False,
        "engine_steps": [],
        "nvtx_scope_definitions": {
            "measured_generation": "Complete measured generate call including final CUDA synchronization.",
            "engine_step_N": "Complete engine.step call, including scheduling and output processing.",
            "prefill": "Scheduled prefill, from scheduler return to engine.step return; may occur more than once.",
            "decode_step_N": "Scheduled decode, from scheduler return to engine.step return; N starts at 1.",
        },
    }
    worker.save(folder / "profile.json", state)
    current_timings = []

    original_timings = worker.RequestTimings

    class ProfileTimings(original_timings):
        def __init__(self, engine, guard):
            super().__init__(engine, guard)
            current_timings.append(self)
            self.step_ordinal = 0
            self.prefill_ordinal = 0
            self.decode_ordinal = 0
            self.in_step = False
            self.phase_range_open = False
            self.current_step = None
            original_step = engine.step
            original_schedule = engine.scheduler.schedule

            @functools.wraps(original_schedule)
            def schedule(*schedule_args, **schedule_kwargs):
                metadata, output = original_schedule(*schedule_args, **schedule_kwargs)
                if self.enabled and self.in_step and not output.is_empty():
                    import torch
                    if self.phase_range_open:
                        raise RuntimeError("More than one nonempty scheduler result in one engine.step")
                    if output.prompt_run:
                        self.prefill_ordinal += 1
                        label = "prefill"
                        phase, phase_ordinal = "prefill", self.prefill_ordinal
                    else:
                        self.decode_ordinal += 1
                        label = f"decode_step_{self.decode_ordinal}"
                        phase, phase_ordinal = "decode", self.decode_ordinal
                    self.current_step.update({
                        "phase": phase,
                        "phase_ordinal": phase_ordinal,
                        "inner_nvtx_name": label,
                    })
                    torch.cuda.nvtx.range_push(label)
                    self.phase_range_open = True
                return metadata, output

            @functools.wraps(original_step)
            def step(*step_args, **step_kwargs):
                if not self.enabled:
                    return original_step(*step_args, **step_kwargs)
                import torch
                ordinal = self.step_ordinal
                self.step_ordinal += 1
                self.in_step = True
                self.phase_range_open = False
                self.current_step = {
                    "nvtx_name": f"engine_step_{ordinal}",
                    "ordinal": ordinal,
                    "engine_step_ordinal": ordinal,
                    "phase": None,
                    "phase_ordinal": None,
                }
                torch.cuda.nvtx.range_push(self.current_step["nvtx_name"])
                try:
                    return original_step(*step_args, **step_kwargs)
                finally:
                    if self.phase_range_open:
                        torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_pop()
                    self.in_step = False
                    self.phase_range_open = False
                    state["engine_steps"].append(self.current_step)
                    self.current_step = None

            engine.scheduler.schedule = schedule
            engine.step = step

        def begin(self):
            self.step_ordinal = 0
            self.prefill_ordinal = 0
            self.decode_ordinal = 0
            state["engine_steps"] = []
            return super().begin()

    original_generate = worker.generate

    @functools.wraps(original_generate)
    def generate(*generate_args, **generate_kwargs):
        measured = bool(current_timings and current_timings[-1].enabled)
        if not measured:
            return original_generate(*generate_args, **generate_kwargs)
        import torch
        torch.cuda.cudart().cudaProfilerStart()
        state["capture_started"] = True
        print(f"PROFILE START {args.backend} {case['id']} whole_generation", flush=True)
        torch.cuda.nvtx.range_push("measured_generation")
        try:
            output = original_generate(*generate_args, **generate_kwargs)
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            state["capture_stopped"] = True
            worker.save(folder / "profile.json", state)
            print(f"PROFILE CAPTURED {args.backend} {case['id']} whole_generation", flush=True)
        return output

    worker.RequestTimings = ProfileTimings
    worker.generate = generate
    worker.verify_weights = lambda *a, **kw: {
        "status": "skipped", "reason": "performance profiling only"}
    worker.verify_cache = lambda *a, **kw: {
        "status": "skipped", "reason": "performance profiling only"}
    worker.verify_outputs = lambda outputs, *a, **kw: [len(o.outputs[0].token_ids) for o in outputs]
    worker_args = argparse.Namespace(
        worker_backend=args.backend,
        gpu=args.gpu,
        output_dir=folder,
        diffkv_variant="no-sync-bulk-lengths",
        inputs=args.inputs.resolve(),
        # Keep the existing engine limits independent of the selected case.
        isls=list(worker.LENGTHS),
        osls=list(worker.LENGTHS),
        batches=list(worker.BATCHES),
        warmup_tokens=args.warmup_tokens,
        repeats=1,
    )
    source, prompts = worker.load_inputs(worker_args.inputs, {args.input_length})
    code = worker.worker(worker_args, case, source, prompts)
    result = json.loads(result_path.read_text())
    state["worker_returncode"] = code
    state["worker_status"] = result.get("status")
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    if result.get("status") == "completed" and state["capture_stopped"]:
        state["status"] = "completed"
    elif result.get("status") == "capacity_limited":
        state["status"] = "capacity_limited"
    else:
        state["status"] = "incomplete"
    worker.save(folder / "profile.json", state)
    return code if state["status"] in ("completed", "capacity_limited") else (code or 1)


if __name__ == "__main__":
    raise SystemExit(main())
