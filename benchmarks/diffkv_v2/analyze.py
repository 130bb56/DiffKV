# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit and compare matching completed MATH runs for two named KV backends."""

import argparse
import math
from pathlib import Path

from common import (
    BACKEND_OPTIONS,
    backend_options,
    load_workload,
    read_json,
    request_signature,
    save_json,
    sha256,
    summarize,
)


def audit(run, workload, backend):
    if run["status"] != "completed":
        raise ValueError(f"{backend} run did not complete")
    metadata = run["metadata"]
    if metadata["backend"] != backend:
        raise ValueError(f"Expected {backend} result")
    expected_options = backend_options(backend)
    for key, default in (("kv_cache_dtype", "auto"), ("attention_backend", None)):
        if metadata["engine"].get(key, default) != expected_options.get(key, default):
            raise ValueError(f"{backend} uses an unexpected {key}")
    if metadata["request_signature"] != request_signature(workload):
        raise ValueError(f"{backend} used different prompts or reference answers")
    rows = run["requests"]
    requests = {row["request_id"]: row for row in workload["requests"]}
    if len(rows) != len(requests) or {row["request_id"] for row in rows} != set(
        requests
    ):
        raise ValueError(f"{backend} request IDs differ")
    for row in rows:
        if row["prompt_tokens"] != len(requests[row["request_id"]]["prompt_token_ids"]):
            raise ValueError("Prompt token count differs")
        if not (
            0
            < row["output_tokens"]
            == len(row["output_token_ids"])
            <= row["max_tokens"]
        ):
            raise ValueError("Output token count differs")
        if (
            metadata["mode"] == "length_replay"
            and row["output_tokens"] != row["max_tokens"]
        ):
            raise ValueError("Fixed-length output stopped early")
        if not 0 <= row["ttft_s"] <= row["last_token_s"] <= row["e2e_s"]:
            raise ValueError("Request timings are inconsistent")
        if row["e2e_s"] > run["summary"]["elapsed_s"]:
            raise ValueError("Request completed after the reported benchmark end")
        if row["output_tokens"] > 1:
            expected = (row["last_token_s"] - row["ttft_s"]) / (
                row["output_tokens"] - 1
            )
            if row["tpot_s"] is None or not math.isclose(row["tpot_s"], expected):
                raise ValueError("TPOT is inconsistent with token timestamps")
    return summarize(rows, run["summary"]["elapsed_s"])


def compare(
    workload, reference, candidate, reference_backend="fp16", candidate_backend="diffkv"
):
    if reference_backend == candidate_backend:
        raise ValueError("Reference and candidate backend labels must differ")
    summaries = {
        reference_backend: audit(reference, workload, reference_backend),
        candidate_backend: audit(candidate, workload, candidate_backend),
    }
    fm, dm = reference["metadata"], candidate["metadata"]
    for key in ("mode", "generation", "warmup_tokens", "length_reference_sha256"):
        if fm[key] != dm[key]:
            raise ValueError(f"Compared runs differ in {key}")
    fe = {
        key: value
        for key, value in fm["engine"].items()
        if key not in ("attention_backend", "kv_cache_dtype")
    }
    de = {
        key: value
        for key, value in dm["engine"].items()
        if key not in ("attention_backend", "kv_cache_dtype")
    }
    if fe != de:
        raise ValueError("Compared engine settings differ")
    if fm.get("commit") != dm.get("commit"):
        raise ValueError("Compared vLLM commits differ")
    if fm.get("gpu_uuid") != dm.get("gpu_uuid"):
        raise ValueError("Compared physical GPUs differ")
    if fm.get("resolved_engine") != dm.get("resolved_engine"):
        raise ValueError("Resolved engine settings differ")
    if fm.get("text_decoder") != dm.get("text_decoder"):
        raise ValueError("Canonical output text decoders differ")
    if fm["mode"] == "length_replay":
        fixed = {
            row["request_id"]: row["output_tokens"] for row in reference["requests"]
        }
        if fixed != {
            row["request_id"]: row["output_tokens"] for row in candidate["requests"]
        }:
            raise ValueError(
                "Throughput comparison requires equal per-request output lengths"
            )
    result = {
        "request_signature": request_signature(workload),
        "mode": fm["mode"],
        "count": len(workload["requests"]),
        "reference_backend": reference_backend,
        "candidate_backend": candidate_backend,
        "summaries": summaries,
        "throughput_ratio": summaries[candidate_backend]["output_tokens_per_s"]
        / summaries[reference_backend]["output_tokens_per_s"],
        "mean_e2e_speedup": summaries[reference_backend]["e2e_s"]["mean"]
        / summaries[candidate_backend]["e2e_s"]["mean"],
        "scheduler": {
            name: {
                key: run["summary"].get(key)
                for key in ("preemption_count", "peak_scheduler_kv_usage_percent")
            }
            for name, run in (
                (reference_backend, reference),
                (candidate_backend, candidate),
            )
        },
        "interpretation": "A finite burst comparison. Natural EOS runs may produce "
        "different token counts; length replay compares identical output counts.",
    }
    if fm["mode"] == "natural":
        from grading import grade_run

        fgrade, dgrade = grade_run(workload, reference), grade_run(workload, candidate)
        result["correctness"] = {reference_backend: fgrade, candidate_backend: dgrade}
        result["accuracy_difference_percentage_points"] = (
            100 * (dgrade["accuracy"] - fgrade["accuracy"])
            if fgrade["accuracy"] is not None and dgrade["accuracy"] is not None
            else None
        )
        vendor = Path(__file__).parent / "vendor"
        result["grading"] = {
            "method": "Archived math_parser/math_equal after final </think>, "
            "no last-number fallback, include_percentage=False. Raw and sync-corrected "
            "grades are both retained, including each boxed-set/unit/ordinal "
            "correction.",
            "source_sha256": {
                name: sha256(vendor / name)
                for name in ("math_parser.py", "math_grader.py")
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, choices=(50, 500), default=50)
    parser.add_argument(
        "--fp16", "--reference", dest="reference", type=Path, required=True
    )
    parser.add_argument(
        "--diffkv", "--candidate", dest="candidate", type=Path, required=True
    )
    parser.add_argument(
        "--reference-backend", choices=tuple(BACKEND_OPTIONS), default="fp16"
    )
    parser.add_argument(
        "--candidate-backend", choices=tuple(BACKEND_OPTIONS), default="diffkv"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    workload = load_workload(args.workload, args.expected_count)
    result = compare(
        workload,
        read_json(args.reference),
        read_json(args.candidate),
        args.reference_backend,
        args.candidate_backend,
    )
    result["artifacts"] = {
        name: {"path": str(path.resolve()), "sha256": sha256(path)}
        for name, path in (
            ("workload", args.workload),
            (args.reference_backend, args.reference),
            (args.candidate_backend, args.candidate),
        )
    }
    save_json(args.output, result)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
