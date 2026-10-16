"""Run the full GSM8K test split and persist each answer as it finishes."""

import argparse
import ast
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".cache/datasets/gsm8k/test.jsonl"
DATA_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
DATA_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
MODELS = {"llama": ROOT / ".cache/models/Meta-Llama-3-8B",
          "qwen": ROOT / ".cache/models/Qwen3-8B"}
BASELINE_COMMIT = "1db83e31a2468cae37f326a642c0a4c4edbb5e4f"
FP16_REPO = ROOT / ".cache/baselines/vllm-1db83e3"
FP16_PYTHON = ROOT / ".cache/baselines/fp16-env/bin/python"


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_data():
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == DATA_SHA256
    rows = [json.loads(line) for line in DATA.read_text().splitlines()]
    assert len(rows) == 1319
    assert all(re.search(r"#### (-?[0-9.,]+)", row["answer"]) for row in rows)
    return rows


def grade(text, solution):
    truth = re.search(r"#### (-?[0-9.,]+)", solution).group(1).replace(",", "")
    match = re.search(r"The answer is \$?(-?[0-9.,]*[0-9]+)", text)
    prediction = match.group(1).replace(",", "") if match else "[invalid]"
    return prediction, truth, prediction == truth


def launch(args):
    load_data()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    jobs = []
    for model, gpu in (("llama", 0), ("qwen", 1)):
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--model", model, "--gpu", str(gpu),
                   "--output-dir", str(args.output_dir),
                   "--batch-size", str(args.batch_size),
                   "--max-tokens", str(args.max_tokens)]
        log = args.output_dir / f"{model}.log"
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        jobs.append({"model": model, "physical_gpu": gpu, "pid": process.pid,
                     "command": command, "log": str(log)})
    manifest = {"started_utc": now(), "dataset": "GSM8K test", "total": 1319,
                "source": "https://github.com/openai/grade-school-math",
                "dataset_revision": DATA_REVISION, "dataset_sha256": DATA_SHA256,
                "jobs": jobs}
    save(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / f"{args.model}-status.json"
    state = {"status": "initializing", "pid": os.getpid(), "started_utc": now(),
             "model": str(MODELS[args.model]), "physical_gpu": args.gpu,
             "dataset": "GSM8K test", "dataset_sha256": DATA_SHA256,
             "total": 1319, "methods": {}}

    def update():
        state["updated_utc"] = now()
        save(status_path, state)

    update()
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu), "OMP_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "HF_HOME": str(ROOT / ".cache/huggingface"),
        "TRITON_CACHE_DIR": str(ROOT / ".cache/triton"),
        "XDG_CACHE_HOME": str(ROOT / ".cache"),
    })
    sys.path.insert(0, str(ROOT))
    torch = None
    try:
        import torch
        from vllm import LLM, SamplingParams
        from vllm.dataset.gsm import GSM8K_PROMPT

        rows = load_data()
        torch.set_num_threads(2)
        torch.manual_seed(42)
        torch.cuda.set_device(0)
        llm = LLM(model=str(MODELS[args.model]), dtype="float16",
                  load_format="safetensors", seed=42, tensor_parallel_size=1,
                  max_model_len=4096, max_num_batched_tokens=8192,
                  max_num_seqs=args.batch_size, kv_buffer_size=64, swap_space=0,
                  enforce_eager=True, gpu_memory_utilization=0.92)
        tokenizer = llm.get_tokenizer()
        prompts = [GSM8K_PROMPT + f"Q: {row['question']}\nA:" for row in rows]
        if args.model == "qwen":
            prompts = [tokenizer.apply_chat_template(
                [{"role": "system", "content":
                  "Solve the final question using the examples. Reason step by step "
                  "and end with 'The answer is <number>.'"},
                 {"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
                for prompt in prompts]
        encoded = [tokenizer.encode(prompt, add_special_tokens=True)
                   for prompt in prompts]
        assert max(map(len, encoded)) + args.max_tokens <= 4096
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                  stop=["Q:", "</s>", "<|im_end|>",
                                        "<|start_header_id|>user<|end_header_id|>",
                                        "<|eot_id|>"])
        state.update({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                      "cuda": torch.version.cuda, "weights_dtype": "float16",
                      "seed": 42, "temperature": 0.0, "max_tokens": args.max_tokens,
                      "batch_size": args.batch_size, "kv_buffer_size": 64,
                      "prompt_format": "8-shot chat, thinking disabled" if args.model == "qwen"
                      else "8-shot completion", "stop": sampling.stop,
                      "prompt_sha256": hashlib.sha256(
                          json.dumps(prompts, ensure_ascii=False).encode()).hexdigest(),
                      "prompt_tokens_min": min(map(len, encoded)),
                      "prompt_tokens_max": max(map(len, encoded)),
                      "grading": "Exact answer match after removing commas from both "
                      "prediction and ground truth. Extract 'The answer is <number>'."})
        thresholds = [0.02, 1.0] if args.model == "llama" else [0.0, 3.0]
        methods = [("diffkv_k8v4_k4v2", [8, 4, 4, 2], [1, 1, 1, 1], thresholds),
                   ("uniform_k8v8", [8, 8], [1, 1], [0.0, 0.0])]
        engine = llm.llm_engine
        for name, bits, groups, compression in methods:
            state["status"] = "running"
            state["current_method"] = name
            result = {"status": "running", "bits": bits, "groups": groups,
                      "thresholds": compression, "completed": 0, "correct": 0,
                      "invalid_answers": 0, "length_truncated": 0,
                      "generated_tokens": 0, "accuracy": None}
            state["methods"][name] = result
            update()
            started = time.monotonic()
            last_progress = started
            output_path = args.output_dir / f"{args.model}-{name}.jsonl"
            with output_path.open("x") as output_file:
                for start in range(0, len(rows), args.batch_size):
                    pending = {}
                    for index in range(start, min(start + args.batch_size, len(rows))):
                        request_id = f"{name}-{index}"
                        engine.add_request(request_id, prompts[index], sampling,
                                           quant_configs=bits, quant_groups=groups,
                                           compress_configs=compression,
                                           prompt_token_ids=encoded[index])
                        pending[request_id] = index
                    while engine.has_unfinished_requests():
                        for out in engine.step():
                            if not out.finished:
                                continue
                            index = pending.pop(out.request_id)
                            assert len(out.outputs) == 1
                            completion = out.outputs[0]
                            assert completion.finish_reason in ("stop", "length")
                            prediction, truth, correct = grade(completion.text, rows[index]["answer"])
                            record = {"index": index, "question": rows[index]["question"],
                                      "reference_solution": rows[index]["answer"],
                                      "prediction": prediction, "ground_truth": truth,
                                      "correct": correct,
                                      "text": completion.text,
                                      "generated_tokens": len(completion.token_ids),
                                      "finish_reason": completion.finish_reason}
                            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            output_file.flush()
                            result["completed"] += 1
                            result["correct"] += int(correct)
                            result["invalid_answers"] += int(prediction == "[invalid]")
                            result["length_truncated"] += int(completion.finish_reason == "length")
                            result["generated_tokens"] += len(completion.token_ids)
                            result["accuracy"] = result["correct"] / result["completed"]
                        current = time.monotonic()
                        if current - last_progress >= 30:
                            result["elapsed_seconds"] = current - started
                            update()
                            print(f"{name}: {result['completed']}/1319, "
                                  f"correct={result['correct']}, "
                                  f"elapsed={result['elapsed_seconds']:.1f}s", flush=True)
                            last_progress = current
                    assert not pending, pending
                    result["elapsed_seconds"] = time.monotonic() - started
                    update()
                    print(f"{name}: {result['completed']}/1319, "
                          f"accuracy={result['accuracy']:.4%}", flush=True)
                os.fsync(output_file.fileno())
            assert result["completed"] == 1319
            result["status"] = "complete"
            update()
        state["accuracy_difference_percentage_points"] = 100 * (
            state["methods"]["diffkv_k8v4_k4v2"]["accuracy"] -
            state["methods"]["uniform_k8v8"]["accuracy"])
        state["status"] = "complete"
        state["finished_utc"] = now()
        update()
        print("COMPLETE", json.dumps(state["methods"]), flush=True)
    except BaseException as error:
        state.update({"status": "failed", "error": repr(error),
                      "traceback": traceback.format_exc()})
        update()
        raise
    finally:
        if torch is not None and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def package_versions():
    versions = {}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "xformers",
                 "flash-attn", "triton", "vllm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def verify_tensors(torch, engine, model_path):
    from safetensors import safe_open

    run_fp16 = engine.workers[0]
    model = run_fp16.model_runner.model
    parameters = dict(model.named_parameters())
    dtype_counts = Counter(str(p.dtype) for p in parameters.values())
    assert parameters and all(
        p.dtype == torch.float16 for p in parameters.values()
        if p.is_floating_point()
    ), f"Unexpected model parameter dtypes: {dtype_counts}"
    cache_rows = []
    for layer, (key, value) in enumerate(run_fp16.gpu_cache):
        assert key.dtype == value.dtype == torch.float16
        assert key.is_cuda and value.is_cuda
        cache_rows.append({
            "layer": layer, "key_dtype": str(key.dtype),
            "value_dtype": str(value.dtype), "key_shape": list(key.shape),
            "value_shape": list(value.shape), "device": str(key.device),
        })
    assert cache_rows, "No actual GPU KV tensors were allocated"

    name = "model.embed_tokens.weight"
    embedding = parameters[name]
    index = model_path / "model.safetensors.index.json"
    if index.exists():
        shard = model_path / json.loads(index.read_text())["weight_map"][name]
    else:
        shard = model_path / "model.safetensors"
    with safe_open(str(shard), framework="pt", device="cpu") as checkpoint:
        saved = checkpoint.get_slice(name)[1:3, :8]
    loaded = embedding.detach()[1:3, :8].cpu()
    expected = saved.to(torch.float16)
    assert torch.equal(loaded, expected), "Loaded weights differ from checkpoint"
    attention = model.model.layers[0].self_attn.attn
    return {
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "attention_class": (
            f"{type(attention).__module__}.{type(attention).__name__}"
        ),
        "configured_model_dtype": str(engine.model_config.dtype),
        "parameter_dtype_counts": dict(dtype_counts),
        "parameter_count": sum(p.numel() for p in parameters.values()),
        "checkpoint_value_check": {
            "parameter": name, "shard": str(shard),
            "slice": "[1:3, :8]", "checkpoint_dtype": str(saved.dtype),
            "loaded_dtype": str(loaded.dtype), "exact_match_after_fp16_cast": True,
            "checkpoint_sample_fp16": expected.tolist(),
            "loaded_sample": loaded.tolist(),
        },
        "kv_tensors": cache_rows,
    }


def source_state(directory):
    return json.loads((directory / "llama-status.json").read_text())


def make_prompts(rows):
    tree = ast.parse((ROOT / "vllm/dataset/gsm.py").read_text())
    prefix = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and any(
                      isinstance(target, ast.Name) and target.id == "GSM8K_PROMPT"
                      for target in node.targets))
    return [prefix + f"Q: {row['question']}\nA:" for row in rows]


def digest(prompts):
    return hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest()


def run_fp16(directory):
    source = source_state(directory)
    assert source["status"] == "complete"
    state = {"status": "initializing", "pid": os.getpid(), "started_utc": now(),
             "backend": "original_vllm_fp16", "kv_precision": "K16V16",
             "weights_dtype": "float16", "model": source["model"],
             "physical_gpu": source["physical_gpu"], "completed": 0, "correct": 0,
             "total": 1319, "invalid_answers": 0, "length_truncated": 0,
             "generated_tokens": 0}
    path = directory / "llama-fp16-status.json"

    def update():
        state["updated_utc"] = now()
        save(path, state)

    update()
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": str(source["physical_gpu"]), "OMP_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "HF_HOME": str(ROOT / ".cache/huggingface"),
        "TRITON_CACHE_DIR": str(ROOT / ".cache/triton-fp16"),
        "XDG_CACHE_HOME": str(ROOT / ".cache"),
    })
    sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != ROOT]
    sys.path.insert(0, str(FP16_REPO))
    torch = None
    try:
        import torch
        import vllm
        import vllm._C
        from vllm import LLM, SamplingParams

        commit = subprocess.check_output(
            ["git", "-C", str(FP16_REPO), "rev-parse", "HEAD"], text=True).strip()
        assert commit == BASELINE_COMMIT
        for module in (vllm, vllm._C):
            assert Path(module.__file__).resolve().is_relative_to(FP16_REPO)
        state.update({"source_root": str(FP16_REPO), "source_commit": commit,
                      "vllm_import": vllm.__file__, "extension_import": vllm._C.__file__,
                      "source_tracked_diff": subprocess.check_output(
                          ["git", "-C", str(FP16_REPO), "diff", "HEAD", "--"], text=True),
                      "package_versions": package_versions()})
        rows = load_data()
        prompts = make_prompts(rows)
        assert digest(prompts) == source["prompt_sha256"]
        assert Path(source["model"]).resolve() == MODELS["llama"]
        torch.set_num_threads(2)
        torch.manual_seed(source["seed"])
        torch.cuda.set_device(0)
        llm = LLM(model=source["model"], dtype="float16", load_format="safetensors",
                  seed=source["seed"], tensor_parallel_size=1, max_model_len=4096,
                  max_num_batched_tokens=8192, max_num_seqs=source["batch_size"],
                  block_size=16, swap_space=0, enforce_eager=True,
                  gpu_memory_utilization=0.92)
        engine = llm.llm_engine
        verification = verify_tensors(torch, engine, MODELS["llama"])
        assert verification["attention_class"] == (
            "vllm.model_executor.layers.attention.PagedAttention")
        state["tensor_verification"] = verification
        encoded = [llm.get_tokenizer().encode(prompt, add_special_tokens=True)
                   for prompt in prompts]
        assert (min(map(len, encoded)), max(map(len, encoded))) == (
            source["prompt_tokens_min"], source["prompt_tokens_max"])
        assert max(map(len, encoded)) + source["max_tokens"] <= 4096
        sampling = SamplingParams(temperature=source["temperature"],
                                  max_tokens=source["max_tokens"], stop=source["stop"])
        for key in ("seed", "temperature", "max_tokens", "batch_size", "stop",
                    "prompt_sha256", "dataset_sha256", "prompt_format", "grading"):
            state[key] = source[key]
        state.update({"status": "running", "gpu": torch.cuda.get_device_name(0),
                      "torch": torch.__version__, "cuda": torch.version.cuda})
        started = last_progress = time.monotonic()
        update()
        output = directory / "llama-fp16.jsonl"
        with output.open("x") as stream:
            for start in range(0, len(rows), source["batch_size"]):
                pending = {}
                for index in range(start, min(start + source["batch_size"], len(rows))):
                    request_id = str(index)
                    engine.add_request(request_id, prompts[index], sampling,
                                       prompt_token_ids=encoded[index])
                    pending[request_id] = index
                while engine.has_unfinished_requests():
                    for out in engine.step():
                        if not out.finished:
                            continue
                        index = pending.pop(out.request_id)
                        assert len(out.outputs) == 1
                        completion = out.outputs[0]
                        assert completion.finish_reason in ("stop", "length")
                        prediction, truth, correct = grade(completion.text, rows[index]["answer"])
                        record = {"index": index, "question": rows[index]["question"],
                                  "reference_solution": rows[index]["answer"],
                                  "prediction": prediction, "ground_truth": truth,
                                  "correct": correct, "text": completion.text,
                                  "generated_tokens": len(completion.token_ids),
                                  "finish_reason": completion.finish_reason}
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        state["completed"] += 1
                        state["correct"] += int(correct)
                        state["invalid_answers"] += int(prediction == "[invalid]")
                        state["length_truncated"] += int(completion.finish_reason == "length")
                        state["generated_tokens"] += len(completion.token_ids)
                        state["accuracy"] = state["correct"] / state["completed"]
                    current = time.monotonic()
                    if current - last_progress >= 30:
                        state["elapsed_seconds"] = current - started
                        update()
                        print(f"FP16: {state['completed']}/1319, "
                              f"correct={state['correct']}", flush=True)
                        last_progress = current
                assert not pending
                state["elapsed_seconds"] = time.monotonic() - started
                update()
            os.fsync(stream.fileno())
        assert state["completed"] == 1319
        state.update({"status": "complete", "finished_utc": now()})
        update()
        print(f"FP16 COMPLETE: {state['correct']}/1319, "
              f"accuracy={state['accuracy']:.4%}", flush=True)
    except BaseException as error:
        state.update({"status": "failed", "error": repr(error),
                      "traceback": traceback.format_exc()})
        update()
        raise
    finally:
        if torch is not None and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def read_answers(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    indexed = {row["index"]: row for row in rows}
    assert len(indexed) == len(rows) == 1319
    assert set(indexed) == set(range(1319))
    for row in rows:
        assert grade(row["text"], row["reference_solution"]) == (
            row["prediction"], row["ground_truth"], row["correct"])
    return indexed


def paired_metrics(baseline, candidate):
    assert set(baseline) == set(candidate)
    pairs = []
    for index in sorted(baseline):
        a, b = baseline[index], candidate[index]
        assert (a["question"], a["reference_solution"], a["ground_truth"]) == (
            b["question"], b["reference_solution"], b["ground_truth"])
        pairs.append((index, a, b))
    count = len(pairs)
    correct_a = sum(a["correct"] for _, a, _ in pairs)
    correct_b = sum(b["correct"] for _, _, b in pairs)
    return {"total": count, "fp16_correct": correct_a, "candidate_correct": correct_b,
            "fp16_accuracy_percent": 100 * correct_a / count,
            "candidate_accuracy_percent": 100 * correct_b / count,
            "accuracy_difference_percentage_points": 100 * (correct_b - correct_a) / count,
            "both_correct": sum(a["correct"] and b["correct"] for _, a, b in pairs),
            "both_wrong": sum(not a["correct"] and not b["correct"] for _, a, b in pairs),
            "fp16_correct_candidate_wrong": [i for i, a, b in pairs if a["correct"] and not b["correct"]],
            "fp16_wrong_candidate_correct": [i for i, a, b in pairs if not a["correct"] and b["correct"]],
            "different_predictions": sum(a["prediction"] != b["prediction"] for _, a, b in pairs),
            "fp16_length_truncated": sum(a["finish_reason"] == "length" for _, a, _ in pairs),
            "candidate_length_truncated": sum(b["finish_reason"] == "length" for _, _, b in pairs),
            "fp16_invalid_answers": sum(a["prediction"] == "[invalid]" for _, a, _ in pairs),
            "candidate_invalid_answers": sum(b["prediction"] == "[invalid]" for _, _, b in pairs)}


def compare_fp16(directory):
    source = source_state(directory)
    fp16 = json.loads((directory / "llama-fp16-status.json").read_text())
    assert source["status"] == fp16["status"] == "complete"
    for key in ("model", "seed", "temperature", "max_tokens", "batch_size", "stop",
                "prompt_sha256", "dataset_sha256", "prompt_format", "grading"):
        assert source[key] == fp16[key], key
    baseline = read_answers(directory / "llama-fp16.jsonl")
    comparisons = {}
    for name in ("diffkv_k8v4_k4v2", "uniform_k8v8"):
        candidate = read_answers(directory / f"llama-{name}.jsonl")
        result = paired_metrics(baseline, candidate)
        comparisons[name] = result
        print(f"{name} vs FP16: {result['candidate_accuracy_percent']:.4f}% vs "
              f"{result['fp16_accuracy_percent']:.4f}%, "
              f"delta={result['accuracy_difference_percentage_points']:+.4f} percentage points, "
              f"lost={len(result['fp16_correct_candidate_wrong'])}, "
              f"gained={len(result['fp16_wrong_candidate_correct'])}", flush=True)
    result = {"status": "complete", "compared_utc": now(), "model": source["model"],
              "dataset": "GSM8K test", "total": 1319,
              "reference_backend": "original vLLM, FP16 weights, K16V16",
              "baseline_commit": fp16["source_commit"], "comparisons": comparisons}
    save(directory / "llama-fp16-comparison.json", result)
    return result


def gpu_idle(gpu):
    used = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip()
    return int(used) < 1024


def predecessor_exited(pid):
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        if b"run_gsm8k_correctness.py" not in cmdline:
            return True
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].startswith("Z ")
    except FileNotFoundError:
        return True


def wait_and_run_fp16(directory):
    queue_path = directory / "llama-fp16-queue.json"
    state = {"status": "waiting_for_llama_evaluation", "pid": os.getpid(),
             "queued_utc": now(), "scope": "Llama3-8B only",
             "baseline_repo": str(FP16_REPO), "baseline_python": str(FP16_PYTHON),
             "dependency": str(directory / "llama-status.json")}

    def update():
        state["updated_utc"] = now()
        save(queue_path, state)

    update()
    try:
        while True:
            source = source_state(directory)
            if source["status"] == "failed":
                raise RuntimeError(f"Preceding Llama evaluation failed: {source.get('error')}")
            if source["status"] == "complete":
                state["status"] = "waiting_for_gpu_release"
                if predecessor_exited(source["pid"]) and gpu_idle(source["physical_gpu"]):
                    break
            update()
            time.sleep(5)
        command = [str(FP16_PYTHON), "-u", str(Path(__file__).resolve()), "--fp16-mode", "worker",
                   "--output-dir", str(directory)]
        state.update({"status": "running_fp16_baseline", "command": command})
        with (directory / "llama-fp16.log").open("x") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT)
            state["baseline_pid"] = process.pid
            update()
            returncode = process.wait()
        assert returncode == 0, f"FP16 worker failed with exit {returncode}"
        state["status"] = "comparing"
        update()
        result = compare_fp16(directory)
        state.update({"status": "complete", "finished_utc": now(),
                      "comparisons": result["comparisons"]})
        update()
    except BaseException as error:
        state.update({"status": "failed", "error": repr(error),
                      "traceback": traceback.format_exc()})
        update()
        raise


def enqueue_fp16(directory):
    path = directory / "llama-fp16-queue.json"
    assert not path.exists(), "FP16 follow-up has already been queued"
    source = source_state(directory)
    assert source["status"] in ("initializing", "running", "complete")
    assert digest(make_prompts(load_data())) == source["prompt_sha256"]
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--fp16-mode", "wait",
               "--output-dir", str(directory)]
    with (directory / "llama-fp16-followup.log").open("x") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    print(json.dumps({"queued_pid": process.pid, "scope": "Llama3-8B only",
                      "baseline": "original vLLM, FP16 weights, K16V16",
                      "dependency_pid": source["pid"], "command": command}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--queue-fp16", action="store_true")
    parser.add_argument("--fp16-mode", choices=("worker", "wait", "compare"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=list(MODELS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    assert args.batch_size > 0 and 0 < args.max_tokens < 4096
    if args.fp16_mode == "worker":
        run_fp16(args.output_dir)
    elif args.fp16_mode == "wait":
        wait_and_run_fp16(args.output_dir)
    elif args.fp16_mode == "compare":
        compare_fp16(args.output_dir)
    elif args.queue_fp16:
        enqueue_fp16(args.output_dir)
    elif args.launch:
        launch(args)
    elif args.model:
        run(args)
    else:
        parser.error("--model is required without --launch")


if __name__ == "__main__":
    main()
