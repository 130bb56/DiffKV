"""Capture and compare isolated DiffKV CUDA binaries on physical GPU 1.

Capture before and after a build in separate processes using --extension. No
vllm package import is needed, so an existing in-place binary stays untouched.
Snapshots include outputs, accumulated scores, cache bytes outside scores, and
the real decode cache writer's retained positions under several policies.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests/kernels/test_diffkv_decode_attention.py"
FIXTURE_CACHE_BUDGET_BYTES = 950_000_000


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def tensor_hash(value):
    return hashlib.sha256(memoryview(value.detach().cpu().contiguous().numpy())).hexdigest()


def telemetry(gpu_uuid):
    target = (gpu_uuid if gpu_uuid.startswith("GPU-") else "GPU-" + gpu_uuid) if gpu_uuid else "1"
    command = ["nvidia-smi", "--id", target, "--query-gpu=index,uuid,name,temperature.gpu,"
               "clocks.sm,clocks.mem,power.draw,power.limit,pstate,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
        return {"returncode": completed.returncode, "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(), "target_gpu_uuid": gpu_uuid}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error), "target_gpu_uuid": gpu_uuid}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def mixed_lengths(context, batch):
    result = []
    for seq in range(batch):
        total = max(2, context - seq % 4 * 7)
        high = max(1, min(total - 1, int(total * (0.40 + 0.03 * (seq % 3)))))
        result.append((high, total - high))
    return tuple(result)


def tensor_storage_bytes(value, seen=None):
    """Estimate serialized tensor storage without counting shared views twice."""
    if seen is None:
        seen = set()
    if isinstance(value, dict):
        return sum(tensor_storage_bytes(item, seen) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_storage_bytes(item, seen) for item in value)
    if hasattr(value, "untyped_storage"):
        storage = value.untyped_storage()
        identity = (str(value.device), storage.data_ptr(), storage.nbytes())
        if identity not in seen:
            seen.add(identity)
            return storage.nbytes()
    return 0


def load_or_build_fixture(torch, fixtures, context, args):
    key = {"schema_version": 1, "context": context, "batch": args.batch,
           "kv_heads": args.kv_heads, "seed": args.seed + context,
           "oracle": bool(args.oracle), "fixture_sha256": file_hash(FIXTURE_PATH),
           "torch": str(torch.__version__)}
    cache_dir = getattr(args, "fixture_cache", None)
    entry = None
    cache_info = {"enabled": cache_dir is not None, "hit": False, "key": key}
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        identity = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        entry = cache_dir / f"{identity}.pt"
        cache_info["path"] = str(entry.resolve())
        if entry.is_file():
            stored = torch.load(entry, map_location="cpu", weights_only=True)
            if stored.get("key") != key or not isinstance(stored.get("fixture"), dict):
                raise ValueError(f"Fixture cache key or payload mismatch: {entry}")
            cache_info.update(hit=True, bytes=entry.stat().st_size)
            return SimpleNamespace(**stored["fixture"]), cache_info
    fixture = fixtures.build_fixture(mixed_lengths(context, args.batch), kv_heads=args.kv_heads,
                                     seed=args.seed + context, compute_reference=args.oracle)
    # Logical dequantized K/V are needed only while constructing the FP64
    # oracle. They can dwarf the packed fixture and are unused by this harness.
    fixture.logical_kv = {}
    if entry is not None:
        payload = vars(fixture)
        estimated = tensor_storage_bytes(payload) + 1_000_000
        existing = sum(path.stat().st_size for path in cache_dir.glob("*.pt"))
        if existing + estimated <= FIXTURE_CACHE_BUDGET_BYTES:
            temporary = entry.with_suffix(".pt.tmp")
            torch.save({"key": key, "fixture": payload}, temporary)
            size = temporary.stat().st_size
            if existing + size <= FIXTURE_CACHE_BUDGET_BYTES:
                temporary.replace(entry)
                cache_info.update(written=True, bytes=size)
            else:
                temporary.unlink()
                cache_info["skipped"] = "Fixture cache budget would exceed 950 MB"
        else:
            cache_info["skipped"] = "Fixture cache budget would exceed 950 MB"
    return fixture, cache_info


def cache_snapshot(torch, fixture_module, fixture, inputs):
    cache = inputs["kv_cache"].detach().cpu()
    lengths = inputs["kv_len_tables"].detach().cpu()
    tables = inputs["block_tables"].detach().cpu()
    mask = fixture.score_mask.clone()
    scores, positions, layout = [], [], []
    for seq, slot in enumerate(fixture.slot_ids.tolist()):
        for head in range(fixture.query.shape[1] // 4):
            high, low = lengths[slot, fixture_module.LAYER_IDX, head].tolist()
            for precision, count, bits in (("high", high, (8, 4, 32)),
                                            ("low", low, (4, 2, 60))):
                page_layout = fixture_module.page_layout(*bits)
                region_scores, region_positions = [], []
                page_capacity = bits[2]
                for page_index in range((count + page_capacity - 1) // page_capacity):
                    page = int(tables[slot, fixture_module.LAYER_IDX, head,
                                      page_index if precision == "high" else -1 - page_index])
                    n = min(page_capacity, count - page_index * page_capacity)
                    score_begin = page_layout.score
                    position_begin = page_layout.position
                    region_scores.append(cache[page].view(torch.float32)[score_begin // 2:score_begin // 2 + n])
                    region_positions.append(cache[page].view(torch.int32)[position_begin // 2:position_begin // 2 + n])
                    mask[page, score_begin:score_begin + 2 * n] = True
                scores.append(torch.cat(region_scores) if region_scores else torch.empty(0))
                positions.append(torch.cat(region_positions) if region_positions else torch.empty(0, dtype=torch.int32))
                layout.append({"sequence": seq, "kv_head": head, "precision": precision, "tokens": count})
    result = {"scores": torch.cat(scores), "retained_positions": torch.cat(positions),
              "score_layout": layout, "lengths": lengths[fixture.slot_ids.long(), fixture_module.LAYER_IDX],
              "block_tables_sha256": tensor_hash(tables),
              "non_score_cache_sha256": tensor_hash(cache[~mask])}
    if not torch.isfinite(result["scores"]).all():
        raise AssertionError("Nonfinite retained score")
    return result


def append_snapshot(torch, extension, fixture_module, fixture, inputs, policy, seed):
    # Each policy starts from exactly the same accumulated score state.
    candidate = dict(inputs)
    for name in ("kv_cache", "kv_len_tables", "block_tables"):
        candidate[name] = inputs[name].clone()
    generator = torch.Generator().manual_seed(seed)
    shape = (fixture.query.shape[0], fixture.query.shape[1] // 4, fixture_module.HEAD_SIZE)
    key = torch.randn(shape, generator=generator).half().cuda()
    value = torch.randn(shape, generator=generator).half().cuda()
    positions = inputs["positions"] + 1
    configs = torch.tensor(policy, dtype=torch.float32, device="cuda").expand(
        fixture.kv_len_tables.shape[0], 2).contiguous()
    extension.cache_ops.compress_and_append_cache_decode_phase(
        key, value, positions, candidate["kv_cache"], candidate["slot_ids"],
        candidate["block_tables"], candidate["kv_len_tables"], configs,
        fixture.max_context_len, 64, fixture_module.LAYER_IDX, *fixture_module.QUANT_ARGS)
    torch.cuda.synchronize()
    return cache_snapshot(torch, fixture_module, fixture, candidate)


def warmup_operation(torch, operation, arguments, minimum_iterations, minimum_seconds):
    """Meet both warmup minima, synchronizing batches outside timed trials."""
    started = time.perf_counter()
    count = 0
    elapsed = 0.0
    while count < minimum_iterations or elapsed < minimum_seconds:
        # Complete an exact remaining iteration minimum when it is shorter
        # than a chunk. Further time-based warmup uses full 100-call chunks.
        chunk = min(100, minimum_iterations - count) if count < minimum_iterations else 100
        for _ in range(chunk):
            operation(*arguments)
        count += chunk
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    torch.cuda.synchronize()
    return {"minimum_iterations": minimum_iterations, "minimum_seconds": minimum_seconds,
            "actual_iterations": count, "elapsed_s": time.perf_counter() - started,
            "synchronize_chunk_calls": 100,
            "scope": "Outside CUDA-event trials; no clock or power settings changed"}


def time_operation(torch, operation, arguments, inputs, initial_cache, args):
    inputs["kv_cache"].copy_(initial_cache)
    warmup = warmup_operation(torch, operation, arguments, args.warmup, args.warmup_seconds)
    samples = []
    for _ in range(args.repeats):
        inputs["kv_cache"].copy_(initial_cache)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            operation(*arguments)
        end.record()
        end.synchronize()
        sample = start.elapsed_time(end) / args.iterations
        if not math.isfinite(sample) or sample <= 0:
            raise RuntimeError("Invalid CUDA event duration")
        samples.append(sample)
    return {"median_ms": statistics.median(samples), "mean_ms": statistics.mean(samples),
            "samples_ms": samples, "iterations": args.iterations, "repeats": args.repeats,
            "warmup": warmup,
            "scope": "attention main, reduce, score update; excludes cache writer, reset, allocation, readback"}


def capture(args):
    # Explicitly reject GPU 0, which may be occupied by another experiment.
    if args.gpu != 1:
        raise ValueError("This isolated candidate harness is restricted to physical GPU 1")
    if args.extension is None or not args.extension.is_file():
        raise ValueError("capture requires --extension PATH to the intended binary")
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace capture: {args.output}")
    if args.cuda_device_ordinal == 0 and not args.expected_gpu_uuid:
        raise ValueError("Container ordinal 0 requires --expected-gpu-uuid for physical GPU 1")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device_ordinal)
    os.environ["OMP_NUM_THREADS"] = "2"
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU 1 is unavailable")
    # Load the binary directly. Importing vllm would load the shared in-place
    # extension before we could select this isolated candidate.
    extension = load_module("vllm._C", args.extension.resolve())
    fixtures = load_module("candidate_decode_fixture", FIXTURE_PATH)
    fixtures.PARTITION_SIZE = args.workspace_partition_size
    operation_name = "sparse_paged_attention" if args.op == "legacy" else "sparse_paged_attention_gqa"
    operation = getattr(extension.ops, operation_name)
    props = torch.cuda.get_device_properties(0)
    actual_uuid = str(getattr(props, "uuid", ""))
    if args.expected_gpu_uuid and actual_uuid.removeprefix("GPU-").lower() != args.expected_gpu_uuid.removeprefix("GPU-").lower():
        raise RuntimeError(f"GPU UUID mismatch: expected {args.expected_gpu_uuid}, observed {actual_uuid}")
    report = {"schema_version": 1, "status": "capturing", "created_utc": datetime.now(timezone.utc).isoformat(),
              "physical_gpu": 1, "gpu_name": props.name, "gpu_uuid": actual_uuid,
              "cuda_device_ordinal": args.cuda_device_ordinal,
              "expected_gpu_uuid": args.expected_gpu_uuid,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "extension": str(args.extension.resolve()), "extension_sha256": file_hash(args.extension),
              "fixture_sha256": file_hash(FIXTURE_PATH), "harness_sha256": file_hash(__file__),
              "operation": operation_name, "contexts": args.contexts, "batch": args.batch,
              "workspace_partition_size": args.workspace_partition_size,
              "fixture_cache": str(args.fixture_cache.resolve()) if args.fixture_cache else None,
              "kv_heads": args.kv_heads, "steps": args.steps, "seed": args.seed,
              "policies": args.policies, "capture_mode": "timing_only" if args.timing_only else "full",
              "timing_warmup": args.warmup,
              "timing_warmup_seconds": args.warmup_seconds,
              "limits": "Synthetic fixed packed KV pages in randomized physical order, one attention layer, no model or E2E claim",
              "cases": []}
    snapshots = {"metadata": report, "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = args.output.with_suffix(args.output.suffix + ".json")
    save_json(sidecar, report)
    for context in args.contexts:
        before_telemetry = telemetry(actual_uuid)
        fixture, cache_info = load_or_build_fixture(torch, fixtures, context, args)
        inputs = fixtures.device_inputs(fixture)
        workspace = fixtures.allocate_workspaces(fixture)
        arguments = fixtures.attention_args(fixture, inputs, workspace)
        initial_cache = inputs["kv_cache"].clone()
        state = {"context": context, "lengths_high_low": mixed_lengths(context, args.batch),
                 "initial_cache_sha256": tensor_hash(fixture.kv_cache), "snapshots": [], "selections": []}
        outputs = []
        checkpoints = sorted({1, min(4, args.steps), min(16, args.steps), args.steps})
        for step in range(1, args.steps + 1):
            if step > 1:
                generator = torch.Generator().manual_seed(args.seed + context * 1000 + step)
                query = (fixture.query.float() + torch.randn(fixture.query.shape, generator=generator) * .005).half()
                inputs["query"].copy_(query)
            profile = args.profile and step == 1
            if profile:
                torch.cuda.cudart().cudaProfilerStart()
                torch.cuda.nvtx.range_push("diffkv_candidate")
            operation(*arguments)
            if profile:
                torch.cuda.nvtx.range_pop()
                torch.cuda.cudart().cudaProfilerStop()
            torch.cuda.synchronize()
            output = workspace["out"].detach().cpu().clone()
            if not torch.isfinite(output).all():
                raise AssertionError("Nonfinite attention output")
            outputs.append(output)
            if step == 1 and args.oracle:
                torch.testing.assert_close(output.double(), fixture.expected_output,
                                           atol=1e-3, rtol=3e-4)
                actual_scores = inputs["kv_cache"].cpu().view(torch.float32).reshape(-1)
                initial_scores = fixture.kv_cache.view(torch.float32).reshape(-1)
                for key, indices in fixture.score_indices.items():
                    torch.testing.assert_close(
                        (actual_scores[indices] - initial_scores[indices]).double(),
                        fixture.score_updates[key], atol=5e-8, rtol=5e-4)
            if step in checkpoints and not args.timing_only:
                snapshot = cache_snapshot(torch, fixtures, fixture, inputs)
                snapshot["step"] = step
                if snapshot["non_score_cache_sha256"] != tensor_hash(fixture.kv_cache[~fixture.score_mask]):
                    raise AssertionError("Attention changed non-score cache bytes")
                if not torch.equal(inputs["kv_len_tables"].cpu(), fixture.kv_len_tables):
                    raise AssertionError("Attention changed retained lengths")
                if not torch.equal(inputs["block_tables"].cpu(), fixture.block_tables):
                    raise AssertionError("Attention changed block tables")
                state["snapshots"].append(snapshot)
        state["outputs"] = torch.stack(outputs)
        if not args.timing_only:
            for policy in args.policies:
                state["selections"].append({"policy": policy, **append_snapshot(
                    torch, extension, fixtures, fixture, inputs, policy, args.seed + context + 881)})
        if args.timing:
            inputs["query"].copy_(fixture.query)
            state["timing"] = time_operation(torch, operation, arguments, inputs, initial_cache, args)
        case_report = {"context": context, "oracle_first_step_passed": bool(args.oracle),
                       "fixture_cache": cache_info,
                       "workspace_max_partitions": workspace["exp_sums"].shape[2],
                       "checkpoints": [] if args.timing_only else checkpoints, "captured_outputs": args.steps,
                       "real_cache_selection_policies": [] if args.timing_only else args.policies,
                       "accuracy_status": "not_validated" if args.timing_only else "captured_for_comparison",
                       "timing": state.get("timing"), "telemetry_before": before_telemetry,
                       "telemetry_after": telemetry(actual_uuid)}
        report["cases"].append(case_report)
        snapshots["cases"].append(state)
        save_json(sidecar, report)
        print(json.dumps(case_report), flush=True)
        del inputs, workspace, fixture, initial_cache, arguments, outputs
    report["status"] = "completed"
    torch.save(snapshots, args.output)
    save_json(sidecar, report)
    print(f"CAPTURE {args.output}", flush=True)
    return 0


def compare_tensor(torch, left, right, atol, rtol):
    same_shape = left.shape == right.shape and left.dtype == right.dtype
    exact = same_shape and torch.equal(left, right)
    close = same_shape and torch.allclose(left, right, atol=atol, rtol=rtol)
    max_abs = (left.double() - right.double()).abs().max().item() if same_shape and left.numel() else None
    return {"exact": bool(exact), "close": bool(close), "max_abs_error": max_abs}


def compare(args):
    import torch
    left = torch.load(args.baseline, map_location="cpu", weights_only=False)
    right = torch.load(args.candidate, map_location="cpu", weights_only=False)
    identity = ("schema_version", "fixture_sha256", "operation", "contexts", "batch", "kv_heads", "steps", "seed", "policies")
    checks = {key: left["metadata"][key] == right["metadata"][key] for key in identity}
    checks["capture_mode"] = left["metadata"].get("capture_mode", "full") == right["metadata"].get("capture_mode", "full")
    allow_operation_change = getattr(args, "allow_operation_change", False)
    operation_changed = not checks["operation"]
    identity_valid = all(value for key, value in checks.items()
                         if not (key == "operation" and allow_operation_change))
    report = {"status": "comparing", "baseline": str(args.baseline.resolve()),
              "candidate": str(args.candidate.resolve()), "identity_checks": checks,
              "allow_operation_change": allow_operation_change,
              "operations": {"baseline": left["metadata"]["operation"],
                             "candidate": right["metadata"]["operation"],
                             "changed": operation_changed},
              "warnings": (["Different attention operations were explicitly allowed; all input identities, output/score tolerances and exact cache selection checks remain enforced."]
                           if operation_changed and allow_operation_change else []),
              "require_bitwise": not args.allow_float_roundoff, "cases": []}
    if not identity_valid or len(left["cases"]) != len(right["cases"]):
        raise ValueError("Capture inputs or operation differ")
    all_passed = True
    for before, after in zip(left["cases"], right["cases"]):
        case = {"context": before["context"], "fixture_identical": before["initial_cache_sha256"] == after["initial_cache_sha256"],
                "outputs": compare_tensor(torch, before["outputs"], after["outputs"], args.output_atol, args.rtol),
                "snapshots": [], "selections": []}
        passed = case["fixture_identical"] and case["outputs"]["close" if args.allow_float_roundoff else "exact"]
        for stage in ("snapshots", "selections"):
            if len(before[stage]) != len(after[stage]):
                raise ValueError("Capture stages differ")
            for first, second in zip(before[stage], after[stage]):
                row = {"step" if stage == "snapshots" else "policy": first["step" if stage == "snapshots" else "policy"],
                       "scores": compare_tensor(torch, first["scores"], second["scores"], args.score_atol, args.rtol),
                       "retained_positions_identical": torch.equal(first["retained_positions"], second["retained_positions"]),
                       "lengths_identical": torch.equal(first["lengths"], second["lengths"]),
                       "score_layout_identical": first["score_layout"] == second["score_layout"],
                       "non_score_cache_identical": first["non_score_cache_sha256"] == second["non_score_cache_sha256"],
                       "block_tables_identical": first["block_tables_sha256"] == second["block_tables_sha256"]}
                row["passed"] = row["scores"]["close" if args.allow_float_roundoff else "exact"] and all(
                    value for key, value in row.items() if key.endswith("identical"))
                passed = passed and row["passed"]
                case[stage].append(row)
        if before.get("timing") and after.get("timing"):
            case["baseline_timing"] = before["timing"]
            case["candidate_timing"] = after["timing"]
            case["median_speedup"] = before["timing"]["median_ms"] / after["timing"]["median_ms"]
        case["passed"] = bool(passed)
        all_passed = all_passed and passed
        report["cases"].append(case)
    report["status"] = "passed" if all_passed else "failed"
    report["passed"] = bool(all_passed)
    save_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if all_passed else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["capture", "compare"])
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--gpu", type=int, choices=[1], default=1)
    parser.add_argument("--cuda-device-ordinal", type=int, choices=[0, 1], default=1,
                        help="Use 0 only inside a container restricted to the physical GPU 1 UUID")
    parser.add_argument("--expected-gpu-uuid", help="Required when a GPU 1 container exposes CUDA ordinal 0")
    parser.add_argument("--op", choices=["legacy", "gqa"], default="legacy")
    parser.add_argument("--workspace-partition-size", type=int, default=5760,
                        help="Partition bound used only to size temporary workspaces; must cover the selected binary's partition count")
    parser.add_argument("--fixture-cache", type=Path,
                        help="Directory for CPU packed fixtures and optional FP64 oracles, capped at 950 MB")
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 1536, 2560, 6000])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--policies", nargs="+", default=["0:0", "0.02:1", "1:1"])
    parser.add_argument("--oracle", action="store_true", help="Also check the first output against independent CPU FP64")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--timing-only", action="store_true",
                        help="Use one output call, skip oracle/score/selection capture, and run CUDA-event timing")
    parser.add_argument("--profile", action="store_true", help="Profile only first call per context in diffkv_candidate NVTX range")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--warmup-seconds", type=float, default=0.0,
                        help="Warm up for at least this wall time and --warmup calls, synchronized every 100 calls")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--allow-float-roundoff", action="store_true")
    parser.add_argument("--allow-operation-change", action="store_true",
                        help="Allow legacy versus GQA comparison while preserving every other identity and cache selection check")
    parser.add_argument("--output-atol", type=float, default=1e-3)
    parser.add_argument("--score-atol", type=float, default=5e-8)
    parser.add_argument("--rtol", type=float, default=5e-4)
    args = parser.parse_args(argv)
    if args.timing_only:
        args.timing = True
        args.steps = 1
        args.oracle = False
    if (min(args.contexts) < 2 or min(args.batch, args.kv_heads, args.steps, args.iterations, args.repeats) < 1
            or args.workspace_partition_size < 1 or args.warmup < 0
            or not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 0):
        parser.error("Invalid fixture or timing dimensions")
    try:
        args.policies = [tuple(float(item) for item in policy.split(":")) for policy in args.policies]
    except ValueError:
        parser.error("Policies must be prune:quant float pairs")
    if any(len(policy) != 2 or not 0 <= policy[0] <= policy[1] <= 1 for policy in args.policies):
        parser.error("Policies must satisfy 0 <= prune <= quant <= 1")
    if args.mode == "compare" and (args.baseline is None or args.candidate is None):
        parser.error("compare requires --baseline and --candidate")
    return args


if __name__ == "__main__":
    options = parse_args()
    raise SystemExit(capture(options) if options.mode == "capture" else compare(options))
