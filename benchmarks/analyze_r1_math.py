"""Audit natural R1 MATH accuracy and a separate FP16-length replay comparison.

Length replay outputs are never used as natural accuracy samples. The thresholds
are a 50-example screening rule, not a statistical noninferiority conclusion.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_grader():
    modules = []
    for filename in ("math_parser", "math_grader"):
        spec = importlib.util.spec_from_file_location(
            "r1_" + filename, ROOT / "vllm/dataset" / f"{filename}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def grade(text, reference, parser, grader):
    # Use the final answer after the thinking section if it is present.
    final = text.rsplit("</think>", 1)[-1]
    prediction = parser.extract_answer(final, data_name="math", use_last_number=False)
    truth = parser.strip_string(reference, skip_unit=False)
    correct = bool(prediction) and bool(grader.math_equal(
        prediction, truth, include_percentage=False, timeout=True))
    return {"prediction": prediction, "reference": truth,
            "correct": correct, "answer_missing": not bool(prediction),
            "has_think_end": "</think>" in text, "has_boxed_answer": "\\boxed" in final}


def distribution(values):
    values = sorted(values)
    if not values:
        return {key: None for key in ("mean", "min", "max", "p50", "p95")}
    def percentile(q):
        index = (len(values) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return {"mean": statistics.mean(values), "min": values[0], "max": values[-1],
            "p50": percentile(.5), "p95": percentile(.95)}


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def run_metadata(run):
    # Metadata is the canonical runner format. Top-level keys are supported for
    # manually exported runs, but contradictory identities are rejected.
    metadata = dict(run.get("metadata") or {})
    for key in ("backend", "mode", "gpu", "workload_sha256", "engine"):
        if key in run:
            if key in metadata and metadata[key] != run[key]:
                raise ValueError(f"Conflicting run identity: {key}")
            metadata[key] = run[key]
    return metadata


def audit_run(run, workload, workload_hash, backend, mode):
    metadata = run_metadata(run)
    rows = run.get("requests", [])
    expected = {row["request_id"]: row for row in workload["requests"]}
    ids = [row.get("request_id") for row in rows]
    checks = {"backend": metadata.get("backend") == backend,
              "mode": metadata.get("mode") == mode,
              "workload_sha256": metadata.get("workload_sha256") == workload_hash,
              "request_ids": len(ids) == len(expected) and len(set(ids)) == len(ids) and set(ids) == set(expected),
              "status": run.get("status", "complete") in ("complete", "completed"),
              "common_engine": all((metadata.get("engine") or {}).get(key) == value
                                   for key, value in workload["engine"].items())}
    expected_generation = dict(workload["generation"], ignore_eos=mode == "length_replay")
    checks["generation"] = all((metadata.get("generation") or {}).get(key) == value
                               for key, value in expected_generation.items())
    checks["complete_outputs"] = bool(rows) and all(
        row.get("finish_reason") in ("stop", "length")
        and isinstance(row.get("output_text"), str)
        and isinstance(row.get("output_token_ids"), list)
        and len(row["output_token_ids"]) == row.get("output_tokens")
        and 0 < row["output_tokens"] <= workload["generation"]["max_tokens"]
        and row.get("prompt_tokens") == expected.get(row.get("request_id"), {}).get("prompt_tokens")
        for row in rows)
    checks["finite_positive_e2e"] = bool(rows) and all(numeric(row.get("e2e_s")) and row["e2e_s"] > 0 for row in rows)
    checks["latency_consistency"] = bool(rows) and all(
        numeric(row.get("ttft_s")) and row["ttft_s"] >= 0
        and (row.get("output_tokens", 0) == 1 or numeric(row.get("tpot_s")) and row["tpot_s"] >= 0)
        and numeric(row.get("e2e_s"))
        and math.isclose(row["ttft_s"] + max(0, row.get("output_tokens", 0) - 1) * (row.get("tpot_s") or 0),
                         row["e2e_s"], rel_tol=1e-5, abs_tol=1e-5) for row in rows)
    if mode == "natural":
        checks["eos_allowed"] = metadata.get("ignore_eos", False) is False
    else:
        checks["eos_ignored"] = metadata.get("ignore_eos") is True
    return {"passed": all(checks.values()), "checks": checks, "metadata": metadata}


def natural_metrics(run, workload, parser, grader):
    rows = {row["request_id"]: row for row in run["requests"]}
    graded = []
    for request in workload["requests"]:
        row = rows[request["request_id"]]
        result = grade(row["output_text"], request["answer"], parser, grader)
        result.update(request_id=request["request_id"], unique_id=request["unique_id"],
                      output_tokens=row["output_tokens"], finish_reason=row["finish_reason"])
        graded.append(result)
    return {"correct": sum(row["correct"] for row in graded), "count": len(graded),
            "accuracy": statistics.mean(row["correct"] for row in graded),
            "answer_missing": sum(row["answer_missing"] for row in graded),
            "length_capped": sum(row["finish_reason"] == "length" for row in graded),
            "finish_reasons": dict(Counter(row["finish_reason"] for row in graded)),
            "output_tokens": distribution([row["output_tokens"] for row in graded]),
            "requests": graded}


def latency_metrics(run):
    return {key: distribution([row[key] for row in run["requests"] if numeric(row.get(key))])
            for key in ("ttft_s", "tpot_s", "e2e_s", "output_tokens")}


def analyze(workload, workload_hash, fp16, diffkv, replay=None,
            fp16_hash=None, parser=None, grader=None):
    audits = {"fp16_natural": audit_run(fp16, workload, workload_hash, "fp16", "natural"),
              "diffkv_natural": audit_run(diffkv, workload, workload_hash, "diffkv", "natural")}
    if replay is not None:
        audits["diffkv_replay"] = audit_run(replay, workload, workload_hash, "diffkv", "length_replay")
    result = {"schema_version": 1, "workload_sha256": workload_hash, "audits": audits,
              "screening_only": True,
              "interpretation": "One seeded 50-question burst, one sampling seed, no uncertainty or sustained-load claim. Replay is excluded from accuracy.",
              "correctness": None, "performance": None}
    if audits["fp16_natural"]["passed"] and audits["diffkv_natural"]["passed"]:
        if parser is None or grader is None:
            parser, grader = load_grader()
        baseline = natural_metrics(fp16, workload, parser, grader)
        compressed = natural_metrics(diffkv, workload, parser, grader)
        difference = 100 * (compressed["accuracy"] - baseline["accuracy"])
        ratio = compressed["output_tokens"]["mean"] / baseline["output_tokens"]["mean"]
        paired = [{"request_id": f["request_id"], "fp16_correct": f["correct"],
                   "diffkv_correct": d["correct"],
                   "length_ratio": d["output_tokens"] / f["output_tokens"]}
                  for f, d in zip(baseline["requests"], compressed["requests"])]
        result["correctness"] = {"fp16": baseline, "diffkv": compressed,
                                  "accuracy_difference_percentage_points": difference,
                                  "mean_output_length_ratio": ratio,
                                  "accuracy_rule_passed": difference >= -5 - 1e-9,
                                  "length_rule_passed": .9 <= ratio <= 1.1,
                                  "passed": difference >= -5 - 1e-9 and .9 <= ratio <= 1.1,
                                  "paired": paired}
    if replay is not None and audits["fp16_natural"]["passed"] and audits["diffkv_replay"]["passed"]:
        fm, rm = run_metadata(fp16), run_metadata(replay)
        original = {row["request_id"]: row for row in fp16["requests"]}
        checks = {"same_physical_gpu": fm.get("gpu") is not None and fm.get("gpu") == rm.get("gpu"),
                  "reference_sha256": fp16_hash is not None and rm.get("length_reference_sha256") == fp16_hash,
                  "per_request_lengths": all(row["output_tokens"] == original[row["request_id"]]["output_tokens"]
                                             and row.get("max_tokens") == original[row["request_id"]]["output_tokens"]
                                             for row in replay["requests"])}
        if fm.get("gpu_uuid") is not None or rm.get("gpu_uuid") is not None:
            checks["same_gpu_uuid"] = fm.get("gpu_uuid") is not None and fm.get("gpu_uuid") == rm.get("gpu_uuid")
        baseline, compressed = latency_metrics(fp16), latency_metrics(replay)
        speedup = baseline["e2e_s"]["mean"] / compressed["e2e_s"]["mean"]
        result["performance"] = {"checks": checks, "valid": all(checks.values()),
                                  "fp16_natural": baseline, "diffkv_length_replay": compressed,
                                  "mean_e2e_speedup": speedup,
                                  "threshold": 1.4, "passed": all(checks.values()) and speedup >= 1.4}
    result["all_audits_passed"] = all(value["passed"] for value in audits.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--fp16", type=Path, required=True)
    parser.add_argument("--diffkv", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text())
    result = analyze(load(args.workload), sha256(args.workload), load(args.fp16),
                     load(args.diffkv), load(args.replay) if args.replay else None,
                     fp16_hash=sha256(args.fp16))
    result["artifacts"] = {key: {"path": str(path.resolve()), "sha256": sha256(path)}
                           for key, path in (("workload", args.workload), ("fp16", args.fp16),
                                             ("diffkv", args.diffkv), ("replay", args.replay)) if path}
    result["grading"] = {"extractor": "repository math_parser.extract_answer, no last-number fallback, text after final </think>",
                         "equivalence": "repository math_grader.math_equal, include_percentage=False, timeout=True",
                         "sources_sha256": {name: sha256(ROOT / "vllm/dataset" / name)
                                            for name in ("math_parser.py", "math_grader.py")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "all_audits_passed": result["all_audits_passed"],
                      "correctness_passed": (result["correctness"] or {}).get("passed"),
                      "performance_passed": (result["performance"] or {}).get("passed")}, indent=2))


if __name__ == "__main__":
    main()
