#!/usr/bin/env python3
"""Compare FP16, original DiffKV, and optimized DiffKV on frozen natural MATH outputs.

Accuracy only. Reuses analyze_r1_math's extractor, equivalence grader, and natural
metrics. No replay, latency, throughput, or performance acceptance is reported.
Formatting corrections require evidence in each output, never a request-ID rule.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[1]
LABELS = ("fp16", "original", "optimized")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_helpers():
    spec = importlib.util.spec_from_file_location("r1_accuracy_helpers", ROOT / "benchmarks/analyze_r1_math.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def boxes(text):
    """Return balanced boxed expressions and their positions, preserving content."""
    found = []
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        start = match.end()
        depth, end = 1, start
        while end < len(text) and depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        found.append({"content": text[start:end - 1] if depth == 0 else text[start:],
                      "start": match.start(), "end": end, "closed": depth == 0})
    return found


def format_audit(text, raw, parser, grader):
    final = text.rsplit("</think>", 1)[-1]
    found = boxes(final)
    result = {"correct": raw["correct"], "prediction": raw["prediction"],
              "correction": None, "issues": []}
    if raw["answer_missing"]:
        result["issues"].append("No answer extracted under the fixed final-section protocol")
    if any(not box["closed"] for box in found):
        result["issues"].append("Unclosed boxed expression in the final section")
    if len(raw["prediction"]) > 256:
        result["issues"].append("Extracted prediction contains unusually long content; review parser behavior")
    if raw["correct"]:
        return result
    # Ordinal formatting changes a label such as 12^{\mathrm{th}} into 12.
    # A mathematical exponent without an explicit ordinal text macro is retained.
    if found and found[-1]["closed"]:
        ordinal = re.fullmatch(
            r"\s*([+-]?\d+)\s*\^\s*\{\s*\\(?:mathrm|text)\s*\{\s*(?:st|nd|rd|th)\s*\}\s*\}\s*",
            found[-1]["content"])
        if ordinal and grader.math_equal(ordinal.group(1), raw["reference"], include_percentage=False, timeout=True):
            result.update(correct=True, prediction=ordinal.group(1), correction={
                "rule": "explicit_ordinal_text_suffix", "boxed_content": found[-1]["content"],
                "evidence": final[max(0, found[-1]["start"] - 120):found[-1]["end"] + 80],
                "reason": "The final boxed integer has an explicit ordinal text suffix; removing the suffix matches the reference."})
            return result
    # Conservatively repair only a terminal list of boxed integer answers. This
    # does not collect intermediate derivation boxes or arbitrary answer guesses.
    paragraph_start = final.rstrip().rfind("\n\n") + 2
    terminal = [box for box in found if box["start"] >= paragraph_start and box["closed"]]
    truth_parts = raw["reference"].split(",")
    if 1 < len(truth_parts) == len(terminal) <= 8:
        predictions = [parser.strip_string(box["content"], skip_unit=False) for box in terminal]
        integer_pattern = r"[+-]?\d+"
        simple_integers = all(re.fullmatch(integer_pattern, value) for value in predictions + truth_parts)
        connectors = []
        for left, right in zip(terminal, terminal[1:]):
            connector = final[left["end"]:right["start"]]
            for markup in (r"\(", r"\)", r"\[", r"\]", "$", "**"):
                connector = connector.replace(markup, "")
            connectors.append(re.sub(r"\s+", "", connector).lower())
        if simple_integers and all(value in ("and", ",", ",and") for value in connectors):
            if sorted(map(int, predictions)) == sorted(map(int, truth_parts)):
                result.update(correct=True, prediction=",".join(predictions), correction={
                    "rule": "terminal_split_boxed_integer_list", "boxed_contents": [box["content"] for box in terminal],
                    "evidence": final[paragraph_start:].strip(),
                    "reason": "The terminal answer joins separately boxed integers with list connectors; the complete integer multiset matches the reference."})
                return result
    if len(found) > 1:
        result["issues"].append("Multiple final-section boxes; no evidence-backed formatting correction applied")
    return result


def audit_run(run, workload, workload_sha, label, helper):
    try:
        original = helper.audit_run(run, workload, workload_sha,
                                    "fp16" if label == "fp16" else "diffkv", "natural")
        # Timing checks belong to the old performance analysis and do not gate
        # reuse of an otherwise complete natural accuracy sample.
        checks = {key: value for key, value in original["checks"].items()
                  if key not in ("finite_positive_e2e", "latency_consistency")}
        metadata = original["metadata"]
        checks["model"] = metadata.get("model") == workload.get("model")
        checks["seed"] = metadata.get("seed") == workload["generation"]["seed"]
        expected_eos = workload.get("tokenizer", {}).get("generation_config_eos_token_id")
        expected_eos = expected_eos if isinstance(expected_eos, list) else [expected_eos]
        checks["eos_token_ids"] = metadata.get("effective_eos_token_ids") == expected_eos
        checks["integer_output_token_ids"] = all(
            all(type(token) is int and token >= 0 for token in row["output_token_ids"])
            for row in run.get("requests", []))
        checks["per_request_generation_limit"] = all(
            row.get("max_tokens") == workload["generation"]["max_tokens"]
            for row in run.get("requests", []))
        if label != "fp16":
            checks["diffkv_configuration"] = all(metadata.get(key) == value for key, value in workload["diffkv"].items())
        return {"passed": all(checks.values()), "checks": checks,
                "identity": {key: metadata.get(key) for key in
                             ("backend", "mode", "model", "workload_sha256", "seed", "generation", "engine",
                              "effective_eos_token_ids", "gpu", "gpu_uuid", "extension_sha256", "runner_sha256",
                              "diffkv_variant", "quant_configs", "quant_groups", "compress_configs", "started_utc", "finished_utc")}}
    except (KeyError, TypeError, ValueError) as error:
        return {"passed": False, "checks": {}, "error": str(error)}


def compare_tokens(old, new):
    prefix = 0
    for left, right in zip(old, new):
        if left != right:
            break
        prefix += 1
    exact = old == new
    return {"exact": exact, "original_tokens": len(old), "optimized_tokens": len(new),
            "common_prefix_tokens": prefix, "first_divergence_token_index": None if exact else prefix,
            "original_token_at_divergence": old[prefix] if prefix < len(old) else None,
            "optimized_token_at_divergence": new[prefix] if prefix < len(new) else None,
            "aligned_equal_tokens": sum(a == b for a, b in zip(old, new)),
            "aligned_compared_tokens": min(len(old), len(new)),
            "unmatched_length_tokens": abs(len(old) - len(new))}


def binary_provenance(path, optimized, helper):
    if path is None:
        return {"verified": False, "reason": "No installation manifest supplied"}
    try:
        manifest = json.loads(path.read_text())
        meta = helper.run_metadata(optimized)
        installed = datetime.fromisoformat(manifest["installed_utc"])
        started = datetime.fromisoformat(meta["started_utc"])
        patch = Path(meta["source_patch"])
        checks = {
            "installed_before_runner_start": installed < started,
            "extension_digest_matches_manifest": meta["extension_sha256"] == manifest["extension_sha256"],
            "source_patch_matches_manifest": meta["source_patch_sha256"] == manifest["source_diff_sha256"],
            "saved_source_patch_digest": patch.is_file() and sha(patch) == meta["source_patch_sha256"],
            "runner_digest": meta["runner_sha256"] == sha(ROOT / "benchmarks/run_r1_math.py"),
        }
        return {"verified": all(checks.values()), "checks": checks, "manifest_path": str(path.resolve()),
                "manifest_sha256": sha(path), "installed_utc": manifest["installed_utc"],
                "runner_started_utc": meta["started_utc"], "extension_sha256": meta["extension_sha256"],
                "basis": "Installation receipt predates the runner start, which precedes its extension import; runtime digest and saved source patch match the receipt."}
    except (OSError, KeyError, TypeError, ValueError) as error:
        return {"verified": False, "reason": str(error)}


def analyze(workload, workload_sha, runs, helper=None, binary_manifest=None):
    helper = helper or load_helpers()
    parser, grader = helper.load_grader()
    audits = {label: audit_run(runs[label], workload, workload_sha, label, helper) for label in LABELS}
    metadata = {label: helper.run_metadata(runs[label]) for label in LABELS}
    shared_checks = {
        "all_natural_run_audits": all(value["passed"] for value in audits.values()),
        "identical_common_engine": metadata["fp16"].get("engine") == metadata["original"].get("engine") == metadata["optimized"].get("engine"),
        "identical_generation": metadata["fp16"].get("generation") == metadata["original"].get("generation") == metadata["optimized"].get("generation"),
        "same_model": len({value.get("model") for value in metadata.values()}) == 1,
        "same_diffkv_variant": metadata["original"].get("diffkv_variant") == metadata["optimized"].get("diffkv_variant"),
    }
    result = {"schema_version": 1, "scope": "accuracy_only", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "workload_sha256": workload_sha, "count": len(workload["requests"]), "audits": audits,
              "shared_settings_audit": {"passed": all(shared_checks.values()), "checks": shared_checks},
              "binary_provenance": binary_provenance(binary_manifest, runs["optimized"], helper),
              "results": {}, "requests": [], "comparisons": None}
    graded, outputs = {}, {}
    for label in LABELS:
        if not audits[label]["passed"]:
            result["results"][label] = {"valid": False, "status": runs[label].get("status")}
            continue
        metrics = helper.natural_metrics(runs[label], workload, parser, grader)
        graded[label] = {row["request_id"]: row for row in metrics.pop("requests")}
        outputs[label] = {row["request_id"]: row for row in runs[label]["requests"]}
        audited = []
        for request in workload["requests"]:
            request_id = request["request_id"]
            raw = graded[label][request_id]
            row = outputs[label][request_id]
            corrected = format_audit(row["output_text"], raw, parser, grader)
            raw["format_audit"] = corrected
            audited.append(corrected)
        result["results"][label] = {
            "valid": True, "raw": metrics,
            "format_audited": {"correct": sum(row["correct"] for row in audited), "count": len(audited),
                               "accuracy": statistics.mean(row["correct"] for row in audited),
                               "correction_count": sum(row["correction"] is not None for row in audited),
                               "method": "Raw grading plus conservative, per-output evidence for explicit ordinal labels and terminal split-boxed integer lists."},
            "eos_stopped": sum(row["finish_reason"] == "stop" for row in outputs[label].values()),
            "ends_with_eos_token": sum(row["output_token_ids"][-1] in metadata[label]["effective_eos_token_ids"] for row in outputs[label].values()),
            "at_generation_limit": sum(row["output_tokens"] == workload["generation"]["max_tokens"] for row in outputs[label].values()),
        }
    for request in workload["requests"]:
        request_id = request["request_id"]
        row = {"request_id": request_id, "unique_id": request["unique_id"], "reference": request["answer"], "runs": {}}
        for label in LABELS:
            if label not in graded:
                row["runs"][label] = None
                continue
            raw = graded[label][request_id]
            output = outputs[label][request_id]
            row["runs"][label] = {
                "raw": {key: raw[key] for key in ("prediction", "reference", "correct", "answer_missing", "has_think_end", "has_boxed_answer")},
                "format_audited": raw["format_audit"], "output_tokens": output["output_tokens"],
                "finish_reason": output["finish_reason"], "at_generation_limit": output["output_tokens"] == workload["generation"]["max_tokens"],
                "ends_with_eos_token": output["output_token_ids"][-1] in metadata[label]["effective_eos_token_ids"],
                "output_text_sha256": hashlib.sha256(output["output_text"].encode()).hexdigest(),
                "output_token_ids_sha256": object_sha(output["output_token_ids"])}
        if "original" in outputs and "optimized" in outputs:
            row["original_optimized_tokens"] = compare_tokens(outputs["original"][request_id]["output_token_ids"],
                                                              outputs["optimized"][request_id]["output_token_ids"])
        result["requests"].append(row)
    if result["shared_settings_audit"]["passed"]:
        comparisons = {}
        for baseline in ("fp16", "original"):
            paired = {}
            for method in ("raw", "format_audited"):
                lost, gained, both, neither = [], [], [], []
                for row in result["requests"]:
                    left, right = row["runs"][baseline][method]["correct"], row["runs"]["optimized"][method]["correct"]
                    (both if left and right else lost if left else gained if right else neither).append(row["request_id"])
                paired[method] = {"accuracy_difference_percentage_points": 100 * (
                    result["results"]["optimized"][method]["accuracy"] - result["results"][baseline][method]["accuracy"]),
                    "baseline_correct_optimized_wrong": lost, "baseline_wrong_optimized_correct": gained,
                    "both_correct": both, "both_wrong": neither}
            comparisons["optimized_vs_" + baseline] = paired
        token_rows = [row["original_optimized_tokens"] for row in result["requests"]]
        first = [row["first_divergence_token_index"] for row in token_rows if not row["exact"]]
        comparisons["original_optimized_token_agreement"] = {
            "exact_requests": sum(row["exact"] for row in token_rows), "requests": len(token_rows),
            "first_divergence_token_index": helper.distribution(first),
            "first_divergence_token_index_counts": dict(Counter(first)),
            "index_convention": "Zero-based generated-token index; an ended sequence diverges at its first missing token.",
            "interpretation": "Equal global seed does not pair per-request sampling draws across different scheduler batches; token differences alone do not identify kernel numerical error."}
        result["comparisons"] = comparisons
        delta = comparisons["optimized_vs_fp16"]["format_audited"]["accuracy_difference_percentage_points"]
        ratio = (result["results"]["optimized"]["raw"]["output_tokens"]["mean"] /
                 result["results"]["fp16"]["raw"]["output_tokens"]["mean"])
        result["screening"] = {
            "accuracy_difference_percentage_points": delta,
            "accuracy_threshold_percentage_points": -5,
            "accuracy_passed": delta >= -5 - 1e-9,
            "mean_output_length_ratio": ratio,
            "output_length_ratio_bounds": [.9, 1.1],
            "length_passed": .9 <= ratio <= 1.1,
            "passed": delta >= -5 - 1e-9 and .9 <= ratio <= 1.1,
            "scope": "Pre-existing 50-question screening rule, not statistical equivalence."}
    result["grading"] = {"extractor": "Existing grade(): text after final </think>, repository extract_answer, no last-number fallback",
                         "equivalence": "Repository math_equal(include_percentage=False, timeout=True)",
                         "sources_sha256": {name: sha(ROOT / "vllm/dataset" / name) for name in ("math_parser.py", "math_grader.py")},
                         "historical_manual_overrides_used": False,
                         "format_audit_scope": "Only the two documented formatting classes are automatically corrected with local output evidence. This is not an exhaustive human review of all derivations."}
    result["limitations"] = ["One frozen 50-question sample and one global sampling seed; no statistical equivalence conclusion.",
                             "Sampling and scheduler differences can change output tokens and scores even with the same seed.",
                             "A boxed answer inside thinking is not substituted for a missing final answer after </think>.",
                             "No performance result or performance acceptance is calculated."]
    return result


def markdown(result):
    lines = ["# R1 MATH50 kernel correctness", "", "Accuracy only, natural EOS, same frozen prompt token IDs and sampling settings.", "",
             "| Path | Raw score | Format-audited score | Mean output tokens | EOS stop | Length cap |",
             "|---|---:|---:|---:|---:|---:|"]
    for label in LABELS:
        data = result["results"][label]
        if not data["valid"]:
            lines.append(f"| {label} | invalid/incomplete | unavailable | unavailable | unavailable | unavailable |")
            continue
        raw, audited = data["raw"], data["format_audited"]
        lines.append(f"| {label} | {raw['correct']}/{raw['count']} ({100*raw['accuracy']:.1f}%) | "
                     f"{audited['correct']}/{audited['count']} ({100*audited['accuracy']:.1f}%) | "
                     f"{raw['output_tokens']['mean']:.2f} | {data['eos_stopped']} | {raw['length_capped']} |")
    lines.extend(["", f"Shared input/settings audit: {result['shared_settings_audit']['passed']}. "
                  f"Optimized installation provenance verified: {result['binary_provenance']['verified']}."])
    if result["comparisons"]:
        screen = result["screening"]
        lines.extend(["", f"Screening passed: **{screen['passed']}**. "
                      f"Accuracy difference {screen['accuracy_difference_percentage_points']:+.1f} percentage points "
                      f"must be at least -5.0. Mean output-length ratio {screen['mean_output_length_ratio']:.4f} "
                      "must be in [0.9, 1.1]."])
        for baseline in ("fp16", "original"):
            comp = result["comparisons"]["optimized_vs_" + baseline]
            lines.append(f"Optimized minus {baseline}: raw {comp['raw']['accuracy_difference_percentage_points']:+.1f} percentage points, "
                         f"format-audited {comp['format_audited']['accuracy_difference_percentage_points']:+.1f} percentage points.")
        tokens = result["comparisons"]["original_optimized_token_agreement"]
        lines.extend(["", f"Original/optimized generated-token exact agreement: {tokens['exact_requests']}/{tokens['requests']} requests. "
                      f"Median first divergence index: {tokens['first_divergence_token_index']['p50']} (zero-based)."])
    corrections = [(row["request_id"], label, row["runs"][label]["format_audited"]["correction"])
                   for row in result["requests"] for label in LABELS
                   if row["runs"][label] and row["runs"][label]["format_audited"]["correction"]]
    lines.extend(["", "Formatting corrections, checked against each output:", ""])
    lines.extend(f"- {label}, {request_id}: {correction['rule']}. Evidence is retained in analysis.json."
                 for request_id, label, correction in corrections)
    lines.extend(["", "Per-request three-way grades, output lengths, EOS/limit flags, grading issues, and token divergence are in analysis.json. "
                  "Historical request-ID corrections are not imported. Answers inside thinking cannot repair a missing final answer.", "",
                  "One sampled 50-question run with temperature 0.6, top_p 0.95, seed 42, max_tokens 8192. "
                  "Equal seeds do not pair per-request RNG draws across different schedules. This comparison does not establish statistical equivalence."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workload", "fp16", "original", "optimized", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--binary-manifest", type=Path, help="Installation receipt for optimized runtime provenance")
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text())
    result = analyze(read(args.workload), sha(args.workload), {label: read(getattr(args, label)) for label in LABELS},
                     binary_manifest=args.binary_manifest)
    result["artifacts"] = {name: {"path": str(getattr(args, name).resolve()), "sha256": sha(getattr(args, name))}
                           for name in ("workload",) + LABELS}
    result["analyzer_sha256"] = sha(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for target, text in ((args.output, json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"),
                         (args.output.with_suffix(".md"), markdown(result))):
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        temporary.write_text(text)
        temporary.replace(target)
    print(json.dumps({"output": str(args.output.resolve()), "shared_settings_audit": result["shared_settings_audit"]["passed"],
                      "scores": {label: {"raw": value.get("raw", {}).get("correct"),
                                         "format_audited": value.get("format_audited", {}).get("correct")}
                                 for label, value in result["results"].items()}}))


if __name__ == "__main__":
    main()
