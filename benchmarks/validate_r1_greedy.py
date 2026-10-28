"""Compare frozen prompt greedy token IDs from HF Transformers and FP16 vLLM."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from run_r1_math import (COMMON_ENGINE, add_request, configure_environment,
                         configured_eos, file_digest, make_engine, read_workload,
                         save_json)


def compare(reference, candidate):
    if reference.get("status") != "completed" or candidate.get("status") != "completed":
        raise ValueError("Both parity runs must complete")
    if reference["workload_sha256"] != candidate["workload_sha256"]:
        raise ValueError("Parity runs use different workloads")
    if reference["max_new_tokens"] != candidate["max_new_tokens"]:
        raise ValueError("Parity runs use different output limits")
    right = {row["request_id"]: row for row in candidate["requests"]}
    if set(right) != {row["request_id"] for row in reference["requests"]}:
        raise ValueError("Parity runs use different prompts")
    checks = []
    for left in reference["requests"]:
        row = right[left["request_id"]]
        if left["prompt_token_ids"] != row["prompt_token_ids"]:
            raise ValueError("Parity prompt tokens differ")
        expected, actual = left["output_token_ids"], row["output_token_ids"]
        mismatch = next((index for index, (a, b) in enumerate(zip(expected, actual)) if a != b), None)
        if mismatch is None and len(expected) != len(actual):
            mismatch = min(len(expected), len(actual))
        checks.append({"request_id": left["request_id"], "exact_match": expected == actual,
                       "reference_tokens": len(expected), "candidate_tokens": len(actual),
                       "first_mismatch_index": mismatch,
                       "reference_token_at_mismatch": (expected[mismatch] if mismatch is not None and mismatch < len(expected) else None),
                       "candidate_token_at_mismatch": (actual[mismatch] if mismatch is not None and mismatch < len(actual) else None)})
    return {"status": "passed" if all(row["exact_match"] for row in checks) else "failed",
            "exact_match": all(row["exact_match"] for row in checks), "checks": checks,
            "definition": "Exact greedy output token IDs on identical frozen chat-template prompt IDs"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["hf", "fp16", "compare"], required=True)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-attn", choices=["eager", "sdpa"], default="eager")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace parity result: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.backend == "compare":
        if args.reference is None or args.candidate is None:
            parser.error("compare requires --reference and --candidate")
        result = compare(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()))
        result.update(reference=str(args.reference.resolve()), candidate=str(args.candidate.resolve()),
                      reference_sha256=file_digest(args.reference), candidate_sha256=file_digest(args.candidate))
        save_json(args.output, result)
        print(json.dumps(result), flush=True)
        return 0 if result["exact_match"] else 1
    if args.workload is None or args.model is None:
        parser.error("generation requires --workload and --model")
    workload = read_workload(args.workload)
    args.engine_options = workload["engine"]
    rows = workload["requests"][:args.count]
    if len(rows) != args.count or args.count < 1 or args.max_new_tokens < 1:
        raise ValueError("Invalid validation prompt count or output length")
    result = {"status": "running", "backend": args.backend,
              "workload_sha256": file_digest(args.workload),
              "model": str(args.model.resolve()), "max_new_tokens": args.max_new_tokens,
              "seed": args.seed, "requests": [], "validator_sha256": file_digest(__file__)}
    save_json(args.output, result)
    if args.backend == "hf":
        configure_environment(args.gpu, "fp16")
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, local_files_only=True,
            attn_implementation=args.hf_attn).cuda().eval()
        eos = configured_eos(args.model, tokenizer.eos_token_id)
        result.update(transformers_version=transformers.__version__, torch_version=torch.__version__,
                      dtype="float16", hf_attention=args.hf_attn, effective_eos_token_ids=eos)
        for row in rows:
            tokens = row["prompt_token_ids"]
            inputs = torch.tensor([tokens], dtype=torch.long, device="cuda")
            with torch.inference_mode():
                output = model.generate(
                    inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                    max_new_tokens=args.max_new_tokens, eos_token_id=eos,
                    pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                                  else tokenizer.eos_token_id),
                    use_cache=True, repetition_penalty=1.0)
            generated = output[0, len(tokens):].tolist()
            result["requests"].append({"request_id": row["request_id"],
                                       "prompt_token_ids": tokens,
                                       "output_token_ids": generated,
                                       "output_text": tokenizer.decode(generated, skip_special_tokens=True)})
            save_json(args.output, result)
    else:
        # A fresh FP16 engine processes one prompt at a time for parity with HF.
        args.diffkv_variant = "original"
        engine, worker, metadata = make_engine(args)
        result["engine_metadata"] = metadata
        from vllm import SamplingParams
        for row in rows:
            add_request(engine, "fp16", row["request_id"], row["prompt_token_ids"],
                        SamplingParams(temperature=0.0, top_p=1.0,
                                       max_tokens=args.max_new_tokens,
                                       stop_token_ids=metadata["effective_eos_token_ids"]),
                        time.monotonic())
            final = None
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if output.finished:
                        final = output
            if final is None:
                raise RuntimeError("No completed parity output")
            result["requests"].append({"request_id": row["request_id"],
                                       "prompt_token_ids": row["prompt_token_ids"],
                                       "output_token_ids": list(final.outputs[0].token_ids),
                                       "output_text": final.outputs[0].text})
            save_json(args.output, result)
    result["status"] = "completed"
    save_json(args.output, result)
    print(json.dumps({"status": "completed", "backend": args.backend,
                      "prompts": len(rows), "output": str(args.output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
