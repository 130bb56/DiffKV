"""Check and time KV reuse across the four query heads in a DiffKV GQA group.

Example, after building the extension with the grouped-query entry point::

    CUDA_VISIBLE_DEVICES=0 .cache/baselines/fp16-env/bin/python \
        benchmarks/benchmark_diffkv_gqa.py --output /tmp/diffkv-gqa.json

This measures one attention layer, not model throughput. Packed mixed-precision
fixtures and the independent CPU FP64 reference come from the CUDA regression
test. By default correctness must pass before timing. --timing-only skips the
reference and correctness checks to screen speed candidates first.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests/kernels/test_diffkv_decode_attention.py"
VARIANTS = ("reference", "gqa-reuse")
PARTITION_SIZE = 5760


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command):
    try:
        completed = subprocess.run(command, cwd=ROOT, text=True,
                                   capture_output=True, timeout=10, check=False)
        return {"returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def telemetry():
    # Query every visible physical GPU; UUID in the report identifies the target
    # even if CUDA_VISIBLE_DEVICES remaps its ordinal.
    return command_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,temperature.gpu,"
        "clocks.sm,clocks.mem,power.draw,power.limit,pstate",
        "--format=csv,noheader,nounits"])


def load_fixture_module():
    if not FIXTURE_PATH.is_file():
        raise RuntimeError(f"Required packed-fixture helper is missing: {FIXTURE_PATH}")
    name = "diffkv_decode_attention_reference"
    spec = importlib.util.spec_from_file_location(name, FIXTURE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stats(samples):
    ordered = sorted(samples)
    return {"median_ms": statistics.median(samples),
            "mean_ms": statistics.mean(samples), "min_ms": min(samples),
            "max_ms": max(samples),
            "p95_ms": ordered[math.ceil(len(ordered) * 0.95) - 1],
            "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "samples_ms": samples}


def mixed_lengths(context, batch):
    # The longest request has exactly `context` retained tokens. Vary the
    # high/low split and request lengths to exercise stride and partition logic.
    result = []
    for seq in range(batch):
        total = max(2, context - (seq % 4) * 7)
        high = max(1, min(total - 1, int(total * (0.40 + 0.03 * (seq % 3)))))
        result.append((high, total - high))
    return tuple(result)


def make_runner(torch, ops, reference, fixture, variant):
    device_fields = reference.device_inputs(fixture)
    workspace = reference.allocate_workspaces(fixture)
    partitions = workspace["exp_sums"].shape[2]
    arguments = reference.attention_args(fixture, device_fields, workspace)
    operation = (ops.sparse_paged_attention if variant == "reference"
                 else ops.sparse_paged_attention_gqa)

    def run():
        operation(*arguments)

    initial_cache = device_fields["kv_cache"].clone()

    def reset():
        # No restoration is included in the measured region. A full cache copy
        # also makes an accidental write outside score storage detectable.
        device_fields["kv_cache"].copy_(initial_cache)

    return {"run": run, "reset": reset, "output": workspace["out"],
            "cache": device_fields["kv_cache"], "partitions": partitions,
            "workspace_bytes": sum(t.numel() * t.element_size() for t in
                                   workspace.values())}


def verify(torch, fixture, runner, args):
    runner["reset"]()
    runner["run"]()
    torch.cuda.synchronize()
    actual = runner["output"].cpu().double()
    expected = fixture.expected_output.double()
    if not torch.isfinite(actual).all():
        raise AssertionError("Attention output contains non-finite values")
    torch.testing.assert_close(actual, expected, atol=args.output_atol, rtol=args.rtol)
    cache = runner["cache"].cpu()
    # Only retained-token score fields may change. Padding, K/V, metadata and
    # positions must remain byte-identical.
    if not torch.equal(cache[~fixture.score_mask], fixture.kv_cache[~fixture.score_mask]):
        raise AssertionError("Attention changed cache bytes outside retained-token scores")
    cache_fp32 = cache.view(torch.float32).reshape(-1)
    initial_fp32 = fixture.kv_cache.view(torch.float32).reshape(-1)
    score_abs_error = 0.0
    score_count = 0
    for key, indices in fixture.score_indices.items():
        indices = torch.as_tensor(indices, dtype=torch.long)
        expected_update = fixture.score_updates[key].double()
        actual_update = (cache_fp32[indices] - initial_fp32[indices]).double()
        if not torch.isfinite(actual_update).all():
            raise AssertionError("Attention score cache contains non-finite values")
        torch.testing.assert_close(actual_update, expected_update,
                                   atol=args.score_atol, rtol=args.rtol)
        score_abs_error = max(score_abs_error,
                              (actual_update - expected_update).abs().max().item())
        score_count += len(indices)
    return {"passed": True, "output_max_abs_error": (actual - expected).abs().max().item(),
            "score_max_abs_error": score_abs_error, "checked_scores": score_count,
            "non_score_cache_bytes_unchanged": True}


def provenance(torch, extension):
    props = torch.cuda.get_device_properties(0)
    source_paths = sorted(path for path in (ROOT / "csrc").rglob("*")
                          if path.suffix in (".cu", ".cuh", ".h", ".cpp"))
    source_paths += [ROOT / "setup.py", Path(__file__).resolve(), FIXTURE_PATH]
    manifest = ROOT / ".cache/diffkv-build.json"
    extension_path = Path(extension.__file__).resolve()
    if not extension_path.is_relative_to(ROOT):
        raise RuntimeError(f"Imported extension is outside this checkout: {extension_path}")
    source_diff = subprocess.check_output(
        ["git", "-C", str(ROOT), "diff", "HEAD", "--", "vllm", "csrc"],
        text=True).strip()
    verified_build = None
    if source_diff:
        from online_benchmark_server import verify_patched_build
        verified_build = verify_patched_build(ROOT, extension_path, source_diff)
    return {
        "python": sys.version, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "gpu": {"name": props.name, "uuid": str(getattr(props, "uuid", "")),
                "compute_capability": [props.major, props.minor],
                "total_memory_bytes": props.total_memory},
        "extension_path": str(extension_path), "extension_sha256": sha256(extension_path),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "git_head": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "build_manifest_sha256": sha256(manifest) if manifest.is_file() else None,
        "verified_local_build": verified_build,
        "build_provenance_note": "Locally modified engine source must match the recorded CUDA build and loaded binary.",
        "telemetry_before": telemetry(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 1536, 2560, 6000])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timing-only", action="store_true",
                        help="Skip reference generation and correctness checks; report unvalidated timings")
    parser.add_argument("--output-atol", type=float, default=1e-3)
    parser.add_argument("--score-atol", type=float, default=5e-8)
    parser.add_argument("--rtol", type=float, default=5e-4)
    parser.add_argument("--output", type=Path, default=Path("/tmp/diffkv-gqa.json"))
    args = parser.parse_args()
    if (min(args.contexts) < 2 or min(args.batch, args.kv_heads, args.iterations, args.repeats) < 1
            or args.warmup < 0 or min(args.output_atol, args.score_atol, args.rtol) < 0):
        parser.error("Lengths must be >= 2, counts positive, warmup and tolerances nonnegative")
    args.output = args.output.resolve()
    return args


def main():
    args = parse_args()
    report = {"status": "initializing", "started_utc": datetime.now(timezone.utc).isoformat(),
              "benchmark": "single-layer packed mixed high/low GQA KV reuse, FP16, head_size=128, four queries per KV head",
              "arguments": {**vars(args), "output": str(args.output)},
              "variants": VARIANTS,
              "workspace_partition_bound": PARTITION_SIZE,
              "measurement": {"timer": "CUDA events, current stream, per-call milliseconds",
                              "baseline": "Existing sparse_paged_attention with correctness fixes; this is not the historical CUDA binary.",
                              "included": "attention main, reduce, score-cache update",
                              "excluded": "fixture, reference, allocation, resets, correctness, telemetry",
                              "score_state": "Restored before warmup and every trial; scores accumulate within timed iterations and do not feed attention computation.",
                              "ordering": "Shuffled order for each pair of rounds, reverse order in its second round",
                              "limits": "Microbenchmark, not E2E; repeated fixed KV, no append/prune or model execution."},
              "cases": []}
    save(args.output, report)
    try:
        sys.path.insert(0, str(ROOT))
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU/driver is unavailable; no CUDA correctness or timing was performed")
        import vllm._C as extension
        from vllm._C import ops
        if not hasattr(ops, "sparse_paged_attention_gqa"):
            raise RuntimeError("Loaded CUDA extension has no sparse_paged_attention_gqa; rebuild this checkout first")
        reference = load_fixture_module()
        torch.set_num_threads(2)
        torch.cuda.set_device(0)
        report["provenance"] = provenance(torch, extension)
        report["cache_layout"] = {"head_size": reference.HEAD_SIZE,
                                  "queries_per_kv_head": reference.QUERY_HEADS_PER_KV,
                                  "unified_page_uint16_elements": reference.PAGE_SIZE,
                                  "layer_idx": reference.LAYER_IDX,
                                  "quant_args": reference.QUANT_ARGS,
                                  "high_kv_bits": [8, 4], "low_kv_bits": [4, 2],
                                  "chunks": [1, 1, 1, 1], "high_low_tokens_per_page": [32, 60]}
        report["status"] = "timing" if args.timing_only else "correctness"
        randomizer = random.Random(args.seed)
        for context in args.contexts:
            fixture = reference.build_fixture(lengths_by_seq=mixed_lengths(context, args.batch),
                                              kv_heads=args.kv_heads, seed=args.seed + context,
                                              compute_reference=not args.timing_only)
            # The scalar layout self-test uses these dequantized arrays; the
            # benchmark needs only its final output/score oracle.
            del fixture.logical_kv
            case = {"context_bound": context, "lengths_high_low": mixed_lengths(context, args.batch),
                    "batch": args.batch, "kv_heads": args.kv_heads,
                    "query_heads": args.kv_heads * reference.QUERY_HEADS_PER_KV,
                    "correctness": {}, "trials": [], "timing": {}}
            report["cases"].append(case)
            runners = {variant: make_runner(torch, ops, reference, fixture, variant)
                       for variant in VARIANTS}
            with torch.inference_mode():
                if args.timing_only:
                    case["correctness"] = {
                        "status": "not_run",
                        "reason": "Timing-only screening requested; FP64 reference and correctness checks were skipped."}
                else:
                    for name, runner in runners.items():
                        try:
                            case["correctness"][name] = verify(torch, fixture, runner, args)
                        except Exception as error:
                            case["correctness"][name] = {"passed": False, "error": str(error)}
                            raise RuntimeError(f"Correctness failed, context={context}, variant={name}: {error}") from error
                save(args.output, report)
                for runner in runners.values():
                    runner["reset"]()
                    for _ in range(args.warmup):
                        runner["run"]()
                torch.cuda.synchronize()
                measurements = {name: [] for name in runners}
                for repeat in range(args.repeats):
                    if repeat % 2 == 0:
                        order = list(runners)
                        randomizer.shuffle(order)
                    else:
                        order = list(reversed(order))
                    for name in order:
                        runner = runners[name]
                        runner["reset"]()
                        torch.cuda.synchronize()
                        before = telemetry()
                        start, end = (torch.cuda.Event(enable_timing=True),
                                      torch.cuda.Event(enable_timing=True))
                        start.record()
                        for _ in range(args.iterations):
                            runner["run"]()
                        end.record()
                        end.synchronize()
                        elapsed_ms = start.elapsed_time(end) / args.iterations
                        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                            raise RuntimeError(f"Invalid CUDA event timing for {name}: {elapsed_ms}")
                        measurements[name].append(elapsed_ms)
                        case["trials"].append({"repeat": repeat, "name": name,
                                               "ms_per_call": elapsed_ms,
                                               "telemetry_before": before, "telemetry_after": telemetry()})
                    save(args.output, report)
                for name, samples in measurements.items():
                    case["timing"][name] = {
                        **stats(samples), "workspace_bytes": runners[name]["workspace_bytes"],
                        "max_num_partitions": runners[name]["partitions"],
                        "paired_speedup_over_reference": [baseline / value for baseline, value in
                                                           zip(measurements["reference"], samples)]}
            correctness_status = "not_run" if args.timing_only else "passed"
            print(f"context={context} correctness={correctness_status} " + ", ".join(
                f"{name}={values['median_ms']:.4f}ms" for name, values in case["timing"].items()), flush=True)
            del runners, fixture
        report["status"] = "timing_only_completed" if args.timing_only else "completed"
        report["provenance"]["telemetry_after"] = telemetry()
        exit_code = 0
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        print(report["error"], file=sys.stderr, flush=True)
        exit_code = 1
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(args.output, report)
    print(f"Report: {args.output}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
