"""Profile existing and GQA attention kernels without numerical checks."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import benchmark_diffkv_gqa as benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 1536, 6000])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.contexts) < 2 or min(args.batch, args.iterations) < 1:
        parser.error("Lengths must be >= 2 and counts positive")
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(benchmark.ROOT))
    import torch
    import vllm._C as extension
    from vllm._C import ops

    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    reference = benchmark.load_fixture_module()
    report = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "numerical_checks": "not_run: performance profiling only",
        "provenance": benchmark.provenance(torch, extension),
        "limits": "CUPTI kernel timings and launch resources, no hardware stall/cache counters. Profiler timings are diagnostic; use unprofiled CUDA-event timings for speed claims.",
        "cases": [],
    }
    for context in args.contexts:
        fixture = reference.build_fixture(
            lengths_by_seq=benchmark.mixed_lengths(context, args.batch),
            kv_heads=8, seed=17 + context, compute_reference=False)
        for variant in benchmark.VARIANTS:
            runner = benchmark.make_runner(torch, ops, reference, fixture, variant)
            with torch.inference_mode():
                for _ in range(30):
                    runner["run"]()
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):
                    runner["run"]()
                end.record()
                end.synchronize()
                unprofiled_ms = start.elapsed_time(end) / args.iterations
                with torch.profiler.profile(activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    for _ in range(args.iterations):
                        runner["run"]()
                    torch.cuda.synchronize()
            trace_path = folder / f"b{args.batch}-ctx{context}-{variant}.trace.json"
            profiler.export_chrome_trace(str(trace_path))
            trace = json.loads(trace_path.read_text())
            groups = defaultdict(list)
            launches = []
            for event in trace["traceEvents"]:
                if event.get("ph") != "X":
                    continue
                if event.get("cat") == "kernel":
                    groups[event["name"]].append(event)
                elif event.get("cat") == "cuda_runtime" and "LaunchKernel" in event.get("name", ""):
                    launches.append(event)
            kernels = []
            for name, events in groups.items():
                total_us = sum(event["dur"] for event in events)
                kernels.append({
                    "name": name, "calls": len(events),
                    "total_us": total_us, "us_per_attention_call": total_us / args.iterations,
                    "mean_us_per_kernel": total_us / len(events),
                    "launch_resources": events[0].get("args", {}),
                })
            case = {
                "batch": args.batch, "context": context, "variant": variant,
                "iterations": args.iterations, "trace": str(trace_path),
                "unprofiled_ms_per_attention_call": unprofiled_ms,
                "profiled_kernel_us_per_attention_call": sum(k["us_per_attention_call"] for k in kernels),
                "cuda_launch_api_us_per_attention_call": sum(e["dur"] for e in launches) / args.iterations,
                "kernels": sorted(kernels, key=lambda k: -k["total_us"]),
            }
            report["cases"].append(case)
            benchmark.save(folder / "summary.json", report)
            print(json.dumps({k: case[k] for k in (
                "batch", "context", "variant", "unprofiled_ms_per_attention_call",
                "profiled_kernel_us_per_attention_call")}), flush=True)
            del runner, profiler, trace
        del fixture
    report.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat())
    benchmark.save(folder / "summary.json", report)


if __name__ == "__main__":
    main()
