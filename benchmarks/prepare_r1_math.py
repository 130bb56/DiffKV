"""Freeze a seeded MATH-500 subset and R1 chat-template token IDs.

Run with the DiffKV Python environment. This is a CPU-only preparation step.
The selected indices are sampled before tokenization, with no length filtering.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "HuggingFaceH4/MATH-500"
DATASET_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
DATASET_SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
ENGINE = {"max_model_len": 9216, "max_num_batched_tokens": 9216,
          "max_num_seqs": 64, "gpu_memory_utilization": 0.9,
          "enforce_eager": True, "swap_space": 0, "dtype": "float16",
          "tensor_parallel_size": 1, "seed": 42}
GENERATION = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192,
              "ignore_eos": False, "seed": 42}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset_file(path):
    if sha256(path) != DATASET_SHA256:
        raise ValueError("MATH-500 source SHA256 does not match the pinned revision")
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    if len(rows) != 500 or len({row["unique_id"] for row in rows}) != 500:
        raise ValueError("Expected 500 distinct MATH-500 test examples")
    return rows


def build_requests(rows, tokenizer, count=50, seed=42, max_model_len=9216,
                   max_tokens=8192):
    if not 1 <= count <= len(rows):
        raise ValueError("count must be within the dataset size")
    indices = random.Random(seed).sample(range(len(rows)), count)
    requests = []
    overlength = []
    for index, dataset_index in enumerate(indices):
        row = rows[dataset_index]
        messages = [{"role": "user", "content": row["problem"] + SUFFIX}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
        tokens = tokenizer.apply_chat_template(messages, tokenize=True,
                                               add_generation_prompt=True)
        # Tokenizing rendered chat text must not insert a second BOS token.
        if tokens != tokenizer.encode(prompt, add_special_tokens=False):
            raise ValueError("Chat template text and direct tokenization differ")
        if not tokens or not all(isinstance(token, int) and token >= 0 for token in tokens):
            raise ValueError("Invalid chat-template token IDs")
        request = {"index": index, "request_id": f"math500-{dataset_index:03d}",
                   "dataset_index": dataset_index, "unique_id": row["unique_id"],
                   "problem": row["problem"], "answer": row["answer"],
                   "solution": row["solution"], "subject": row.get("subject"),
                   "level": row.get("level"), "prompt": prompt,
                   "prompt_token_ids": tokens, "prompt_tokens": len(tokens)}
        requests.append(request)
        if len(tokens) + max_tokens > max_model_len:
            overlength.append({"request_id": request["request_id"],
                               "prompt_tokens": len(tokens)})
    if overlength:
        raise ValueError("Selected prompts exceed the model length; no truncation or "
                         f"resampling was performed: {overlength}")
    return requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=ROOT / ".cache/models/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--dataset-file", type=Path,
                        default=ROOT / ".cache/datasets/math500/test.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=9216)
    parser.add_argument("--max-num-batched-tokens", type=int, default=9216)
    parser.add_argument("--download-dataset", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace frozen workload: {args.output}")
    if args.download_dataset:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(DATASET_ID, "test.jsonl", repo_type="dataset",
                                     revision=DATASET_REVISION,
                                     local_dir=str(args.dataset_file.parent))
        if Path(downloaded).resolve() != args.dataset_file.resolve():
            raise ValueError("--dataset-file must end in test.jsonl when downloading")
    from transformers import AutoTokenizer
    rows = load_dataset_file(args.dataset_file)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    requests = build_requests(rows, tokenizer, args.count, args.seed,
                              max_model_len=args.max_model_len)
    generation = dict(GENERATION, seed=args.seed)
    engine = dict(ENGINE, seed=args.seed, max_model_len=args.max_model_len,
                  max_num_batched_tokens=args.max_num_batched_tokens)
    if engine["max_num_batched_tokens"] < engine["max_model_len"]:
        raise ValueError("Unchunked prefill requires max_num_batched_tokens >= max_model_len")
    model_files = {path.name: sha256(path) for path in args.model.iterdir()
                   if path.is_file() and path.suffix in (".json", ".jinja", ".model")
                   and path.stat().st_size < 100_000_000}
    generation_config = args.model / "generation_config.json"
    eos = json.loads(generation_config.read_text()).get("eos_token_id") if generation_config.exists() else None
    payload = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
               "dataset": {"id": DATASET_ID, "split": "test", "size": len(rows),
                           "revision": DATASET_REVISION, "sha256": DATASET_SHA256,
                           "source": f"https://huggingface.co/datasets/{DATASET_ID}"},
               "seed": args.seed, "selection": "random.Random(seed).sample(range(500), count)",
               "model": str(args.model.resolve()),
               "tokenizer": {"class": type(tokenizer).__name__,
                             "transformers_version": importlib.metadata.version("transformers"),
                             "files_sha256": model_files,
                             "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                             "eos_token_id": tokenizer.eos_token_id,
                             "generation_config_eos_token_id": eos,
                             "add_generation_prompt": True, "add_special_tokens_after_template": False},
               "generation": generation, "engine": engine,
               "diffkv": {"quant_configs": [8, 4, 4, 2], "quant_groups": [1, 1, 1, 1],
                          "compress_configs": [0.02, 1.0]},
               "requests": requests}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    lengths = [row["prompt_tokens"] for row in requests]
    print(json.dumps({"output": str(args.output.resolve()), "sha256": sha256(args.output),
                      "requests": len(requests), "prompt_tokens_min": min(lengths),
                      "prompt_tokens_max": max(lengths), "prompt_tokens_mean": sum(lengths) / len(lengths)}, indent=2))


if __name__ == "__main__":
    main()
