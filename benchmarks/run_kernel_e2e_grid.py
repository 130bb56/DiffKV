"""One synchronized generation per case/backend, using isolated CUDA binaries.

Physical GPU1 only. The saved plan fixes the engine, input IDs, batch and output
length. Continuous telemetry is collected by a separate monitor, outside the
timed generation. No source files or installed extensions are swapped.
"""

import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
GPU = "GPU-feccd811-0bc3-0d68-066b-9bf5bc7cdbb7"
PYTHON = ROOT / ".cache/baselines/fp16-env/bin/python"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def telemetry():
    command = ["nvidia-smi", "-i", GPU,
               "--query-gpu=uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,utilization.gpu,pstate",
               "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    return {"epoch_s": time.time(), "returncode": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def cool_before_measurement(folder, target, torch, case_id=None):
    """Restore the historical warmup-to-measurement cooling gate."""
    from monitor_kernel_e2e_gpu import read_records
    policy = json.loads((folder / "cooling-policy.json").read_text())
    minimum_seconds = policy.get("minimum_seconds_by_case", {}).get(
        case_id, policy["minimum_seconds"])
    started = time.monotonic()
    observations = []
    torch.cuda.synchronize()
    while True:
        samples = [r for r in read_records(folder / "telemetry.jsonl")
                   if r.get("record_type") == "sample"]
        row = samples[-1] if samples else {}
        data = row.get("data", {})
        elapsed = time.monotonic() - started
        fresh = time.time() - row.get("time_epoch", 0) < 3
        ready = (fresh and row.get("status") == "ok"
                 and data.get("gpu_uuid") == GPU
                 and data.get("sm_clock_mhz") == 1395
                 and data.get("temperature_c", 999) <= policy["temperature_max_c"]
                 and data.get("throttle_sw_thermal") is False
                 and data.get("throttle_hw_thermal") is False
                 and data.get("throttle_sw_power_cap") is False
                 and data.get("throttle_hw_power_brake") is False
                 and elapsed >= minimum_seconds)
        observations.append({"elapsed_s": elapsed, "time_epoch": row.get("time_epoch"),
                             "temperature_c": data.get("temperature_c"),
                             "sm_clock_mhz": data.get("sm_clock_mhz"),
                             "active_mask": data.get("throttle_active_mask"), "ready": ready})
        save(target.with_suffix(".cooldown.json"), observations)
        if ready:
            return {**policy, "case_id": case_id,
                    "effective_minimum_seconds": minimum_seconds,
                    "actual_seconds": elapsed, "last_sample": row}
        if not fresh:
            raise RuntimeError("Continuous telemetry stopped during cooldown")
        if elapsed > policy["timeout_seconds"]:
            raise RuntimeError("Cooling criterion not reached before generation")
        time.sleep(2)


def worker(folder, backend, index):
    plan = json.loads((folder / "plan.json").read_text())
    case = plan["cases"][index]
    target = folder / f"{case['id']}-{backend}.json"
    if target.exists():
        raise FileExistsError(target)
    result = {"status": "initializing", "backend": backend, "case": case,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "worker_sha256": sha(__file__), "plan_sha256": sha(folder / "plan.json"),
              "physical_gpu": 1, "samples": 1}
    save(target, result)
    try:
        if plan["gpu_uuid"] != GPU or plan["physical_gpu"] != 1:
            raise ValueError("The plan must select physical GPU1")
        os.environ.update(CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS="2",
                          TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
                          TRANSFORMERS_OFFLINE="1", HF_HOME=str(ROOT / ".cache/huggingface"),
                          XDG_CACHE_HOME=str(ROOT / ".cache"),
                          TRITON_CACHE_DIR=str(ROOT / ".cache/triton"))
        fp16 = backend == "fp16_vllm_0_2_6"
        repo = Path(plan["fp16"]["repo"]) if fp16 else ROOT
        sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != ROOT]
        sys.path.insert(0, str(repo))
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(42)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Physical GPU1 must be the only visible CUDA device")
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        actual_uuid = str(getattr(props, "uuid", ""))
        if actual_uuid.removeprefix("GPU-").lower() != GPU.removeprefix("GPU-").lower():
            raise RuntimeError(f"GPU UUID mismatch: {actual_uuid}")
        extension = None
        if not fp16:
            spec_key = "original" if backend == "diffkv_v1_original" else "optimized"
            spec_data = plan[spec_key]
            extension = Path(spec_data["extension"])
            if sha(extension) != spec_data["extension_sha256"]:
                raise RuntimeError("Isolated extension digest differs from plan")
            spec = importlib.util.spec_from_file_location("vllm._C", extension)
            selected = importlib.util.module_from_spec(spec)
            sys.modules["vllm._C"] = selected
            spec.loader.exec_module(selected)
        import vllm
        import vllm._C
        from vllm import LLM, SamplingParams
        if Path(vllm.__file__).resolve().parent != repo / "vllm":
            raise RuntimeError(f"Wrong engine import: {vllm.__file__}")
        if not fp16:
            if sys.modules["vllm._C"] is not selected:
                raise RuntimeError("Isolated extension was replaced")
            vllm._C = selected
        else:
            extension = Path(vllm._C.__file__).resolve()
            if not extension.is_relative_to(repo):
                raise RuntimeError("Wrong FP16 extension import")
            revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            if revision != "1db83e31a2468cae37f326a642c0a4c4edbb5e4f":
                raise RuntimeError("Unexpected FP16 engine revision")
        from run_offline_e2e import (load_inputs, generate, get_worker, verify_weights,
                                     verify_cache, verify_outputs, FixedBatchGuard, RequestTimings)
        source_path = Path(plan["inputs"]["source"])
        if sha(source_path) != plan["inputs"]["sha256"]:
            raise RuntimeError("Input manifest differs from plan")
        source, prompts_by_length = load_inputs(source_path, [case["input_length"]])
        if source["model_revision"] != plan["model"]["revision"]:
            raise RuntimeError("Model revision differs from plan")
        prompt = prompts_by_length[case["input_length"]]
        prompts = [list(prompt) for _ in range(case["batch"])]
        common = dict(plan["engine"], model=source["model_path"], tokenizer=source["tokenizer_path"],
                      dtype=plan["model"]["dtype"], load_format=plan["model"]["load_format"])
        specific = {"block_size": 16} if fp16 else {"kv_buffer_size": 64}
        sampling = dict(temperature=0.0, ignore_eos=True, n=1,
                        max_tokens=case["output_length"])
        if fp16:
            patch = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--", "vllm", "csrc"], text=True)
            (folder / (target.stem + ".source.patch")).write_text(patch)
            result["source_patch_sha256"] = hashlib.sha256(patch.encode()).hexdigest()
            result["model_rope_scaling"] = json.loads((Path(source["model_path"])/"config.json").read_text()).get("rope_scaling")
        result["helper_sha256"] = {name: sha(ROOT/"benchmarks"/name) for name in
                                   ("run_offline_e2e.py", "diffkv_v1.py")}
        result.update(gpu_uuid=GPU, gpu_name=props.name, torch=torch.__version__, cuda=torch.version.cuda,
                      engine_args=common, backend_engine_args=specific,
                      engine_repo=str(repo), engine_import=vllm.__file__, extension=str(extension),
                      extension_sha256=sha(extension), sampling=dict(sampling, seed=42),
                      input={"prompt_sha256": digest(prompt), "batch_sha256": digest(prompts),
                             "manifest_sha256": sha(source_path), "prompt_token_ids": prompt},
                      model_metadata_sha256={n:sha(Path(source["model_path"])/n) for n in
                                             ("config.json", "model.safetensors.index.json", "tokenizer_config.json")},
                      python_source_sha256={str(p.relative_to(repo)):sha(p) for p in
                                             sorted((repo/"vllm").rglob("*.py"))},
                      measurement="Synchronized wall time around one generate call; loading, warmup, profiler, telemetry queries and file writes excluded. Includes prefill, decode, sampling, detokenization, cleanup.")
        llm = LLM(**common, **specific)
        engine = llm.llm_engine
        helper_backend = "fp16" if fp16 else "diffkv"
        model_worker = get_worker(engine, helper_backend)
        if not fp16:
            from diffkv_v1 import apply_variant
            result["python_variant"] = apply_variant(engine, model_worker, "no-sync-bulk-lengths",
                                                     folder / f"{case['id']}-{backend}.ablation")
            attentions = [layer.self_attn.attn for layer in model_worker.model_runner.model.model.layers]
            enabled = backend == "diffkv_v1_optimized"
            for attention in attentions:
                if not enabled:
                    attention.use_gqa_reuse = False
                if bool(attention.use_gqa_reuse) != enabled:
                    raise RuntimeError("Incorrect attention routing")
            result["attention_routing"] = {"gqa_enabled": enabled, "layers": len(attentions)}
        result["weights_verification"] = verify_weights(torch, model_worker, Path(source["model_path"]))
        result["cache_verification"] = verify_cache(torch, model_worker, helper_backend, engine)
        result["status"] = "warming_up"
        save(target, result)
        # The first existing warmup for each DiffKV binary verifies CUDA routing.
        # The measured generation never has a profiler attached.
        record = index == 0 and not fp16
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) if record else nullcontext()
        warm_params = SamplingParams(temperature=0.0, ignore_eos=True, n=1,
                                     max_tokens=plan["warmup"]["tokens"])
        with profiler as captured:
            warm = generate(llm, warm_params, prompts, helper_backend)
            torch.cuda.synchronize()
        verify_outputs(warm, prompt, case["batch"], plan["warmup"]["tokens"])
        if record:
            names = Counter(e.name for e in captured.events() if e.device_type == torch.autograd.DeviceType.CUDA)
            expected = "register_gqa_attention_kernel" if enabled else "sparse_paged_attention_wrapper"
            if not any(expected in name for name in names):
                raise RuntimeError("Expected attention CUDA kernel not observed in warmup")
            result["warmup_cuda_attention_kernels"] = {n:c for n,c in names.items() if "attention" in n or "reduce" in n}
        if engine.has_unfinished_requests():
            raise RuntimeError("Warmup left unfinished requests")
        guard = FixedBatchGuard(engine, case["batch"])
        timings = RequestTimings(engine, guard)
        params = SamplingParams(**sampling)
        if (folder / "cooling-policy.json").exists():
            result["status"] = "cooling"
            save(target, result)
            result["cooldown"] = cool_before_measurement(folder, target, torch, case["id"])
        result["telemetry_before"] = telemetry()
        result["status"] = "running"
        save(target, result)
        torch.manual_seed(42)
        timings.begin()
        torch.cuda.synchronize()
        measured_epoch_start = time.time()
        started = time.perf_counter()
        outputs = generate(llm, params, prompts, helper_backend)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        measured_epoch_end = time.time()
        result["telemetry_after"] = telemetry()
        phases = timings.end(started, case["batch"], case["input_length"], case["output_length"])
        counts = verify_outputs(outputs, prompt, case["batch"], case["output_length"])
        fixed = guard.summary()
        if engine.has_unfinished_requests() or fixed["decode_steps"] != case["output_length"]-1:
            raise RuntimeError("Measured generation violated the fixed workload")
        hist = fixed["decode_batch_histogram"]
        fixed.update(passed=True, expected_batch=case["batch"],
                     observed_decode_batch_min=min(hist), observed_decode_batch_max=max(hist),
                     preemptions=fixed["preemption_attempts"])
        tokens = [list(out.outputs[0].token_ids) for out in outputs]
        if sha(extension) != result["extension_sha256"]:
            raise RuntimeError("Extension changed during generation")
        result.update(status="completed", e2e_s=elapsed,
                      measured_epoch_start=measured_epoch_start, measured_epoch_end=measured_epoch_end,
                      fixed_batch_guard=fixed, output_token_ids=tokens,
                      output_sha256=digest(tokens), generated_tokens=sum(counts),
                      request_metrics={"mean_ttft_s":phases["ttft_ms"]["mean"]/1000,
                                       "mean_tpot_s":phases["tpot_ms"]["mean"]/1000,
                                       "mean_latency_s":phases["request_latency_ms"]["mean"]/1000},
                      phase_metrics={k:phases[k] for k in ("prefill_s","decode_s","prefill_tokens_s","decode_tokens_s")})
        save(target, result)
        print(json.dumps({"case":case["id"],"backend":backend,"e2e_s":elapsed,"status":"completed"}),flush=True)
        return 0
    except BaseException as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        save(target, result)
        raise


def orchestrate(folder, continue_invalid_telemetry=False):
    from monitor_kernel_e2e_gpu import read_records, analyze_interval
    plan = json.loads((folder/"plan.json").read_text())
    completed = []
    invalid_telemetry = []
    finished_attempts = []

    def progress(status, **details):
        save(folder/"progress.json", {"status": status, "completed": completed,
                                     "invalid_telemetry": invalid_telemetry,
                                     "finished_attempts": finished_attempts,
                                     "continue_invalid_telemetry": continue_invalid_telemetry,
                                     **details})

    for index, case in enumerate(plan["cases"]):
        names = plan["backends"]
        shift = index % len(names)
        for backend in names[shift:]+names[:shift]:
            output = folder/f"{case['id']}-{backend}.json"
            if output.exists():
                saved = json.loads(output.read_text())
                if saved.get("status") != "completed" or saved.get("plan_sha256") != sha(folder/"plan.json"):
                    raise RuntimeError(f"Existing incomplete or different result: {output}")
                audit = analyze_interval(read_records(folder/"telemetry.jsonl"),
                                         saved["measured_epoch_start"], saved["measured_epoch_end"])
                save(folder/(output.stem+".telemetry-audit.json"),audit)
                if not audit["valid"]:
                    if not continue_invalid_telemetry:
                        raise RuntimeError(f"Existing timing did not pass telemetry: {output}")
                    invalid_telemetry.append(output.name)
                else:
                    completed.append(output.name)
                finished_attempts.append(output.name)
                progress("running", active=None, last_skipped=output.name)
                continue
            name = f"{case['id']}-{backend}"
            progress("running", active=name)
            print(f"START {len(finished_attempts)+1}/42 {name}",flush=True)
            command=[str(PYTHON),str(Path(__file__).resolve()),"--run-dir",str(folder),
                     "--worker-backend",backend,"--case-index",str(index)]
            with (folder/(name+".log")).open("w") as log:
                process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            if process.returncode:
                progress("failed", active=name, returncode=process.returncode)
                raise RuntimeError(f"Worker failed: {name}; see its log")
            value=json.loads(output.read_text())
            # Wait for the next telemetry sample to bracket the end boundary.
            time.sleep(1.2)
            audit=analyze_interval(read_records(folder/"telemetry.jsonl"),
                                   value["measured_epoch_start"],value["measured_epoch_end"])
            save(folder/(name+".telemetry-audit.json"),audit)
            if not audit["valid"]:
                if not continue_invalid_telemetry:
                    progress("telemetry_invalid", active=name, audit=audit)
                    raise RuntimeError(f"Measured interval failed telemetry: {name}")
                invalid_telemetry.append(output.name)
            else:
                completed.append(output.name)
            finished_attempts.append(output.name)
            progress("running", active=None, last_finished=output.name)
            qualification = "DONE" if audit["valid"] else "UNQUALIFIED"
            print(f"{qualification} {len(finished_attempts)}/42 {name} {value['e2e_s']:.3f}s",flush=True)
    progress("completed_with_invalid_telemetry" if invalid_telemetry else "completed")
    return 0


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--worker-backend",choices=("fp16_vllm_0_2_6","diffkv_v1_original","diffkv_v1_optimized"))
    parser.add_argument("--case-index",type=int)
    parser.add_argument("--continue-invalid-telemetry", action="store_true",
                        help="Orchestrator only: retain completed workloads with invalid telemetry, mark them unqualified, and continue")
    options=parser.parse_args()
    folder=options.run_dir.resolve(strict=True)
    if options.worker_backend and options.case_index is None:
        parser.error("Worker needs --case-index")
    raise SystemExit(worker(folder,options.worker_backend,options.case_index) if options.worker_backend else orchestrate(folder, options.continue_invalid_telemetry))
