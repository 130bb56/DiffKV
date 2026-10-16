"""Compare FP16 and DiffKV HTTP serving, including trace preparation and MATH grading.

The runner sends and measures requests in this process. Each backend's HTTP
server runs in a separate process, using the same shared request trace.
"""

import argparse
import ast
import asyncio
import contextlib
from datetime import datetime, timedelta, timezone
import fcntl
from functools import lru_cache
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Dict, List
import urllib.error
import urllib.request

from online_benchmark_server import configured_eos_token_ids


ROOT = Path(__file__).resolve().parents[1]
FP16_REPO = ROOT / ".cache/baselines/vllm-1db83e3"
DEFAULT_PYTHON = ROOT / ".cache/baselines/fp16-env/bin/python"
DEFAULT_MODEL = ROOT / ".cache/models/Meta-Llama-3-8B"


DEFAULT_DATASET_DIR = ROOT / ".cache/performance-math"
METHOD = (
    "MATH grading prefers the last explicit Final Answer: or The final answer is "
    "section, removes its demonstrated 'I hope it is correct.' closing phrase "
    "for grading only, and extracts a boxed answer within that section when present. "
    "Without an explicit section, use the repository minerva_math extract_answer. "
    "Apply repository symmetric normalization and comparison: "
    "prediction = strip_string(extract_prediction(completion_text), "
    "skip_unit=True); reference = strip_string(ground_truth, skip_unit=True); "
    "math_equal(reference, prediction, timeout=True). "
    "Ground truth comes from parse_ground_truth(..., 'minerva_math'). "
    "Legacy minerva_math.normalize_final_answer is omitted because substring removal "
    "corrupts expressions such as infinity; this is not identical to legacy process_results."
)
ERROR_POLICY = (
    "Raised extraction, normalization, or comparison exceptions and the original "
    "grader's caught symbolic timeout warning are explicit grading errors. "
    "Any grading or saved-response error makes accuracy unavailable. "
    "Failed/OOM generation counts as incorrect in the full request denominator."
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pure_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load scorer module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def scoring_components():
    directory = ROOT / "vllm/dataset"
    parser = _load_pure_module("_diffkv_online_math_parser", directory / "math_parser.py")
    grader = _load_pure_module("_diffkv_online_math_grader", directory / "math_grader.py")
    return parser, grader


@lru_cache(maxsize=1)
def _prompt_functions():
    """Load only the original pure prompt functions, avoiding GPU dataset imports."""
    source = ROOT / "vllm/dataset/minerva_math.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    names = {"doc_to_text", "doc_to_fewshot", "list_fewshot_samples", "get_fewshot_prompt"}
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise RuntimeError("Original Minerva MATH prompt definitions are missing")
    namespace = {"Dict": Dict, "List": List}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def _native_chat_template(tokenizer):
    if tokenizer is None:
        raise ValueError("minerva-4shot-chat requires the checkpoint tokenizer")
    template = tokenizer.get_chat_template()
    if not isinstance(template, str) or not template:
        raise ValueError("The checkpoint tokenizer has no native chat template")
    return template


def resolve_math_prompt_style(requested_style, tokenizer):
    """Choose native chat from tokenizer metadata, retaining explicit selections."""
    if requested_style not in {"auto", "minerva-4shot", "minerva-4shot-chat", "raw-fixed"}:
        raise ValueError(f"Unsupported MATH prompt style: {requested_style}")
    if requested_style != "auto":
        return requested_style, "Explicit prompt style requested"
    if getattr(tokenizer, "chat_template", None):
        return "minerva-4shot-chat", "Checkpoint tokenizer.chat_template is present; use native four-shot chat"
    return "minerva-4shot", "Checkpoint tokenizer.chat_template is absent; use original four-shot completion"


def make_math_chat_messages(problem):
    """Keep all original example text in four user/assistant demonstration turns."""
    if not isinstance(problem, str):
        raise TypeError("MATH problem must be a string")
    functions = _prompt_functions()
    messages = []
    for sample in functions["list_fewshot_samples"]():
        messages.extend([
            {"role": "user", "content": functions["doc_to_text"](sample)},
            {"role": "assistant", "content": sample["solution"]},
        ])
    messages.append({"role": "user", "content": functions["doc_to_text"]({"problem": problem})})
    return messages


def make_math_prompt(problem, prompt_style="minerva-4shot", tokenizer=None):
    """Preserve original examples and problems in raw or native four-shot chat form.

    Chat uses four original problem/solution exchanges and one final user message.
    Rendering adds the checkpoint's assistant generation header. Callers must
    encode that rendered text with add_special_tokens=False; the native template
    already emits BOS. Both four-shot styles retain the original Problem: stops.
    """
    if not isinstance(problem, str):
        raise TypeError("MATH problem must be a string")
    if prompt_style not in {"minerva-4shot", "minerva-4shot-chat"}:
        raise ValueError(f"Unsupported MATH few-shot prompt style: {prompt_style}")
    functions = _prompt_functions()
    prompt = functions["get_fewshot_prompt"]() + functions["doc_to_text"]({"problem": problem})
    if prompt_style == "minerva-4shot-chat":
        _native_chat_template(tokenizer)
        return tokenizer.apply_chat_template(
            make_math_chat_messages(problem),
            tokenize=False, add_generation_prompt=True,
        )
    return prompt


def prompt_metadata(prompt_style="minerva-4shot", tokenizer=None, model_path=None):
    source = ROOT / "vllm/dataset/minerva_math.py"
    functions = _prompt_functions()
    prefix = functions["get_fewshot_prompt"]()
    metadata = {
        "style": "minerva-4shot", "shots": len(functions["list_fewshot_samples"]()),
        "format": "Original four worked examples followed by Problem:\\n<problem>\\n\\nSolution:",
        "model_type": "Base causal language model, completion prompt without chat template",
        "source_file": str(source), "source_sha256": sha256(source),
        "source_functions": ["doc_to_text", "doc_to_fewshot", "list_fewshot_samples", "get_fewshot_prompt"],
        "fewshot_prefix_sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
        "fewshot_prefix_characters": len(prefix),
        "sampling": {"n": 1, "temperature": 0.0, "ignore_eos": False,
                     "stop": ["Problem:", "Problem:\n"]},
        "max_tokens": "Configured by the run; natural EOS/stop termination is enabled",
        "source_tuning_llama_max_tokens": 1024,
    }
    if prompt_style == "minerva-4shot":
        return metadata
    if prompt_style != "minerva-4shot-chat":
        raise ValueError(f"Unsupported MATH few-shot prompt style: {prompt_style}")
    template = _native_chat_template(tokenizer)
    configuration = Path(model_path or tokenizer.name_or_path) / "tokenizer_config.json"
    saved_template = json.loads(configuration.read_text())["chat_template"]
    source_templates = saved_template.values() if isinstance(saved_template, dict) else [saved_template]
    if template not in source_templates:
        raise ValueError("Active chat template does not match the checkpoint tokenizer_config.json")
    metadata.update({
        "style": prompt_style,
        "format": "Four original Minerva problem/solution user/assistant exchanges and one final "
                  "user problem, rendered with the checkpoint native chat template and assistant generation header",
        "model_type": "Instruct causal language model, native chat template",
        "wrapped_prompt_style": "minerva-4shot",
        "chat_template": {
            "source_file": str(configuration.resolve()), "source_sha256": sha256(configuration),
            "content": template, "sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "message_roles": ["user", "assistant"] * 4 + ["user"], "message_count": 9,
            "message_content": "Each example user message is original doc_to_text(sample); each "
                               "assistant message is the exact original sample solution; the final "
                               "user message is doc_to_text(selected problem), with no added instruction",
            "message_source": "Exact nine message dictionaries are recorded in each request's prompt_messages",
            "turn_boundaries": "Native template message endings, including each assistant demonstration, "
                               "and the native assistant generation header",
            "tokenize": False, "add_generation_prompt": True,
        },
        "encoding": {
            "add_special_tokens": False, "bos_token": tokenizer.bos_token,
            "bos_token_id": tokenizer.bos_token_id,
            "required_bos_count": 1, "required_bos_position": 0,
        },
    })
    return metadata


def scorer_metadata():
    directory = ROOT / "vllm/dataset"
    source_paths = [Path(__file__).resolve(), directory / "math_parser.py",
                    directory / "math_grader.py"]
    return {"method": METHOD, "source_files": [
        {"path": str(path), "sha256": sha256(path)} for path in source_paths
    ], "prediction_extraction": {
        "priority": "Last explicit Final Answer: or The final answer is section; "
                    "prefer it over earlier intermediate boxed values",
        "closing_phrase": "Remove I hope it is correct. from the captured answer for grading only",
        "boxed_section": "Use repository extract_answer on the explicit section when it contains a boxed answer",
        "fallback": "Repository extract_answer(text, 'minerva_math') when no explicit section exists",
        "generation_changed": False,
    }, "prediction_normalization": "strip_string(extract_prediction(text), skip_unit=True)",
        "reference_normalization": "strip_string(parse_ground_truth reference, skip_unit=True)",
        "legacy_minerva_normalizer_used": False,
        "legacy_difference": "Prefer explicit final-answer sections over intermediate boxes; "
                             "omit the demonstrated closing phrase from grading; symmetric reference "
                             "normalization; omit destructive substring-removal normalizer",
        "timeout_error_policy": ERROR_POLICY}


def attach_ground_truth(trace, dataset_dir=DEFAULT_DATASET_DIR):
    """Verify every selected problem, enrich trace in-place, and return trace."""
    import pyarrow.parquet as parquet

    parser, _ = scoring_components()
    dataset_dir = Path(dataset_dir).resolve()
    requests = trace["requests"]
    expected = int(trace.get("num_requests", len(requests)))
    if len(requests) != expected:
        raise ValueError(f"Trace has {len(requests)} requests; expected {expected}")
    if len({str(request["request_id"]) for request in requests}) != len(requests):
        raise ValueError("Trace contains duplicate request IDs")
    if len({request["source_id"] for request in requests}) != len(requests):
        raise ValueError("Selected MATH requests must be sampled without replacement")
    tables = {}
    provenance = {}
    for request in requests:
        source_id = request["source_id"]
        match = re.fullmatch(r"([a-z_]+)/test/(\d+)\.parquet:(\d+)", source_id)
        if match is None:
            raise ValueError(f"Unexpected MATH source_id: {source_id!r}")
        subject, part, index = match.groups()
        path = dataset_dir / f"{subject}_test_{part}.parquet"
        if path not in tables:
            table = parquet.read_table(path)
            required = {"problem", "solution", "type", "level"}
            if not required.issubset(table.column_names):
                raise ValueError(f"MATH schema missing fields: {path}")
            tables[path] = table.select(sorted(required)).to_pylist()
            provenance[path] = {"path": str(path), "sha256": sha256(path),
                                "rows": table.num_rows, "subject": subject, "split": "test"}
        rows = tables[path]
        index = int(index)
        if not 0 <= index < len(rows):
            raise ValueError(f"MATH source row out of bounds: {source_id}")
        row = rows[index]
        if request.get("problem", request["prompt"]) != row["problem"]:
            raise ValueError(f"MATH problem does not match local ground truth: {source_id}")
        if request.get("subject", subject) != subject:
            raise ValueError(f"MATH subject mismatch: {source_id}")
        solution, reference = parser.parse_ground_truth(row, "minerva_math")
        extracted = parser.extract_answer(row["solution"], data_name="minerva_math")
        normalized = parser.strip_string(reference, skip_unit=True)
        request.update({
            "solution": row["solution"], "ground_truth": reference,
            "extracted_ground_truth": extracted, "normalized_ground_truth": normalized,
            "ground_truth_type": row["type"], "ground_truth_level": row["level"],
            "ground_truth_source_sha256": provenance[path]["sha256"],
        })
        if solution != row["solution"].strip():
            raise ValueError(f"Unexpected original solution normalization: {source_id}")
    trace["ground_truth_dataset"] = {
        "name": "EleutherAI/hendrycks_math", "split": "test",
        "verified_selected_requests": len(requests),
        "files": [provenance[path] for path in sorted(provenance)],
        "verification": "Exact source_id row lookup and exact problem text equality",
    }
    trace["correctness_scorer"] = scorer_metadata()
    return trace


def extract_prediction(completion_text):
    """Read the stated final answer before considering earlier boxed work.

    Capture through the demonstrated closing phrase or the end of the response,
    so decimal points and arbitrary LaTeX expressions remain intact. An empty
    explicit answer stays empty instead of falling back to an intermediate value.
    """
    if not isinstance(completion_text, str):
        raise TypeError("Saved completion text must be a string")
    parser, _ = scoring_components()
    markers = list(re.finditer(
        r"\b(?:Final\s+Answer\s*:\s*(?:The\s+final\s+answer\s+is\s*)?"
        r"|The\s+final\s+answer\s+is\s*)",
        completion_text, flags=re.IGNORECASE,
    ))
    if not markers:
        return parser.extract_answer(completion_text, data_name="minerva_math")
    answer = completion_text[markers[-1].end():].strip()
    closing = re.search(r"\bI\s+hope\s+it\s+is\s+correct\b[.!]?", answer,
                        flags=re.IGNORECASE)
    if closing:
        answer = answer[:closing.start()].strip()
    answer = answer.rstrip(".").strip()
    if re.search(r"\\boxed\b", answer):
        return parser.extract_answer(answer, data_name="minerva_math")
    return answer


def grade_completion(ground_truth, completion_text):
    """Grade one saved completion and make raised or swallowed timeout errors explicit."""
    parser, grader = scoring_components()
    result = {"extracted_prediction": None, "normalized_prediction": None,
              "normalized_reference": None,
              "correct": None, "status": "grading_error", "error": None,
              "grader_output": ""}
    try:
        extracted = extract_prediction(completion_text)
        normalized = parser.strip_string(extracted, skip_unit=True)
        reference = parser.strip_string(ground_truth, skip_unit=True)
        result.update(extracted_prediction=extracted, normalized_prediction=normalized,
                      normalized_reference=reference)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            correct = grader.math_equal(reference, normalized, timeout=True)
        result["grader_output"] = captured.getvalue()
        # Original math_equal catches symbolic timeout exceptions and prints this
        # warning before returning False. Preserve the comparator, but expose the
        # evaluation error rather than silently asserting an incorrect answer.
        if "Timeout error catched in math grader" in result["grader_output"]:
            raise RuntimeError("Original math grader caught a symbolic comparison timeout/error")
        result.update(correct=bool(correct), status="correct" if correct else "incorrect")
    except Exception as error:
        result.update(status="grading_error", correct=None,
                      error=f"{type(error).__name__}: {error}")
    return result


def _write_text(result, trace, output_dir):
    lines = [
        f"MATH correctness, backend: {result['backend']}",
        f"Status: {result['status']}",
        f"Correct: {result['correct']}/{result['total']}",
        (f"Accuracy: {result['accuracy_percent']:.2f}%" if result['accuracy_percent'] is not None
         else "Accuracy: unavailable because grading or saved-response errors occurred"),
        f"Missing responses: {result['missing_responses']}",
        f"Grading errors: {result['grading_errors']}",
        f"Method: {METHOD}",
        "Full model completions: responses.txt in this directory",
        "Ground truth files: " + json.dumps(trace.get("ground_truth_dataset", {}), ensure_ascii=False),
        "",
    ]
    for row in result["requests"]:
        lines.extend([
            "=" * 80,
            f"Request: {row['request_id']}, source: {row.get('source_id', '')}",
            f"Result: {row['status']}, correct: {row['correct']}",
            f"Ground truth: {row.get('ground_truth', '')}",
            f"Normalized reference: {row.get('normalized_reference')}",
            f"Extracted prediction: {row.get('extracted_prediction')}",
            f"Normalized prediction: {row.get('normalized_prediction')}",
            f"Error: {row.get('error') or ''}",
            f"Grader output: {row.get('grader_output') or ''}",
            "Problem:", row.get("problem", ""),
            "Ground truth solution:", row.get("solution", ""), "",
        ])
    if result["errors"]:
        lines.extend(["Run errors:", json.dumps(result["errors"], indent=2, ensure_ascii=False)])
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "correctness.txt.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "correctness.txt")


def grade_run(trace, summary, output_dir):
    """Grade after timing completes; missing/failed generation stays in denominator."""
    output_dir = Path(output_dir)
    requests = trace["requests"]
    total = len(requests)
    responses = {}
    errors = []
    expected_ids = {str(request["request_id"]) for request in requests}
    for response in summary.get("requests", []):
        request_id = str(response.get("request_id"))
        if request_id not in expected_ids:
            errors.append(f"Unexpected saved response ID: {request_id}")
        elif request_id in responses:
            errors.append(f"Duplicate saved response ID: {request_id}")
        else:
            responses[request_id] = response
    correct = missing = grading_errors = 0
    records = []
    for request in requests:
        request_id = str(request["request_id"])
        response = responses.get(request_id)
        row = {"request_id": request_id, "source_id": request.get("source_id"),
               "problem": request.get("problem", request.get("prompt", "")),
               "solution": request.get("solution", ""),
               "ground_truth": request.get("ground_truth"), "correct": False,
               "extracted_prediction": None, "normalized_prediction": None,
               "normalized_reference": None,
               "status": "failed", "error": None, "grader_output": ""}
        if response is None:
            missing += 1
            grading_errors += 1
            row.update(correct=None, status="missing_response", error="No saved response record")
        elif response.get("status") != "completed":
            row.update(status=str(response.get("status", "failed")),
                       error=response.get("error"))
            if not isinstance(response.get("completion_text"), str):
                missing += 1
            # The client distinguishes a failed generation from a finished
            # response whose completion text could not be retained for scoring.
            # The latter is missing evaluation evidence, not an incorrect answer.
            if "correctness cannot be evaluated" in str(response.get("error", "")).lower():
                grading_errors += 1
                row.update(correct=None, status="grading_error")
        elif not isinstance(response.get("completion_text"), str):
            missing += 1
            grading_errors += 1
            row.update(correct=None, status="missing_response",
                       error="Completed request has no saved completion_text")
        elif not isinstance(request.get("ground_truth"), str):
            grading_errors += 1
            row.update(correct=None, status="grading_error", error="Trace lacks attached ground_truth")
        else:
            row.update(grade_completion(request["ground_truth"], response["completion_text"]))
            if row["status"] == "grading_error":
                grading_errors += 1
            elif row["correct"]:
                correct += 1
        records.append(row)
    grading_errors += len(errors)
    result = {
        "status": "failed" if grading_errors else "completed",
        "backend": summary.get("backend", "unknown"), "correct": correct, "total": total,
        "accuracy_percent": (100.0 * correct / total if total and not grading_errors else None),
        "missing_responses": missing, "grading_errors": grading_errors,
        "failed_generation": sum(response.get("status") != "completed" for response in responses.values()),
        "method": METHOD, "scorer": scorer_metadata(),
        "timeout_error_policy": ERROR_POLICY,
        "ground_truth_dataset": trace.get("ground_truth_dataset"),
        "requests": records, "errors": errors,
        "text_log": str((output_dir / "correctness.txt").resolve()),
        "timing": "Scoring runs after HTTP performance measurement; excluded from serving duration",
    }
    _write_text(result, trace, output_dir)
    return result


def math_self_test(trace=None):
    parser, grader = scoring_components()
    prompt_functions = _prompt_functions()
    prefix = "".join(prompt_functions["doc_to_fewshot"](example)
                     for example in prompt_functions["list_fewshot_samples"]())
    sample_problem = "Find the value of x.\nPreserve this exact problem text."
    prompt = make_math_prompt(sample_problem)
    assert prompt == prefix + "Problem:\n" + sample_problem + "\n\nSolution:"
    assert prefix.count("Problem:\n") == 4 and prompt.count("Problem:\n") == 5
    assert prompt_metadata()["shots"] == 4
    assert prompt_metadata()["sampling"] == {
        "n": 1, "temperature": 0.0, "ignore_eos": False, "stop": ["Problem:", "Problem:\n"]}
    cases = [
        ("numeric", r"The solution is \boxed{2}.", r"Therefore \boxed{2}.", True),
        ("symbolic", r"\boxed{x^2+2x+1}", r"\boxed{(x+1)^2}", True),
        ("wrong", r"\boxed{7}", r"\boxed{8}", False),
        ("empty extraction", r"\boxed{7}", "No answer is available.", False),
        ("infinity regression", r"\boxed{[0,\infty)}", r"\boxed{[0,\infty)}", True),
        ("domain prefix regression", r"\boxed{x\in[-2,7]}", r"\boxed{[-2,7]}", True),
        ("wrong domain endpoint", r"\boxed{x\in[-2,7]}", r"\boxed{[-2,8]}", False),
        ("wrong numeric answer", r"\boxed{2}", r"\boxed{23}", False),
        ("fraction numeric equality", r"\boxed{\frac{1}{2}}", r"\boxed{0.5}", True),
        ("bare explicit final answer", r"\boxed{2}",
         "Final Answer: The final answer is 2. I hope it is correct.", True),
        ("explicit final after intermediate box", r"\boxed{6}",
         r"An intermediate value is \boxed{5.625}. "
         "Final Answer: The final answer is 6. I hope it is correct.", True),
        ("boxed explicit final answer", r"\boxed{6}",
         r"\boxed{5.625}. Final Answer: \boxed{6}. I hope it is correct.", True),
        ("dollar explicit final answer", r"\boxed{2}",
         "Final Answer: The final answer is $2$. I hope it is correct.", True),
        ("explicit signed decimal", r"\boxed{-5.625}",
         "Final Answer: The final answer is -5.625. I hope it is correct.", True),
        ("explicit domain and infinity", r"\boxed{[0,\infty)}",
         r"Final Answer: The final answer is $[0,\infty)$. I hope it is correct.", True),
        ("explicit latex fraction", r"\boxed{\frac{1}{2}}",
         r"Final Answer: The final answer is $\frac{1}{2}$. I hope it is correct.", True),
        ("last explicit final answer", r"\boxed{6}",
         "Final Answer: 5.625. I hope it is correct.\n"
         "Final Answer: 6. I hope it is correct.", True),
        ("empty explicit answer does not use earlier box", r"\boxed{2}",
         r"\boxed{2}. Final Answer: . I hope it is correct.", False),
        ("explicit nonanswer", r"\boxed{2}",
         "Final Answer: I do not know. I hope it is correct.", False),
    ]
    for name, solution, completion, expected in cases:
        reference = parser.parse_ground_truth({"solution": solution}, "minerva_math")[1]
        result = grade_completion(reference, completion)
        if result["correct"] != expected or result["status"] == "grading_error":
            raise AssertionError((name, result))
        prediction = parser.strip_string(
            extract_prediction(completion), skip_unit=True)
        normalized_reference = parser.strip_string(reference, skip_unit=True)
        if bool(grader.math_equal(normalized_reference, prediction, timeout=True)) != expected:
            raise AssertionError(f"Repository comparator mismatch: {name}")
        if result["normalized_prediction"] != prediction:
            raise AssertionError(f"Prediction normalization mismatch: {name}")
        if result["normalized_reference"] != normalized_reference:
            raise AssertionError(f"Reference normalization mismatch: {name}")
    for completion in (r"Therefore \boxed{2}.", "The result is 2.", "No answer is available."):
        assert extract_prediction(completion) == parser.extract_answer(completion, "minerva_math")
    assert grade_completion("2", None)["status"] == "grading_error"
    tiny_trace = {"requests": [{"request_id": name, "ground_truth": "2", "prompt": name}
                               for name in ("good", "failed", "missing", "empty")]}
    tiny_summary = {"backend": "self-test", "requests": [
        {"request_id": "good", "status": "completed", "completion_text": r"\boxed{2}"},
        {"request_id": "failed", "status": "OOM", "error": "CUDA out of memory"},
        {"request_id": "missing", "status": "completed"},
        {"request_id": "empty", "status": "completed", "completion_text": ""},
    ]}
    with tempfile.TemporaryDirectory(prefix="online-math-correctness-test-") as directory:
        result = grade_run(tiny_trace, tiny_summary, directory)
        assert result["total"] == 4 and result["correct"] == 1
        assert result["accuracy_percent"] is None and result["grading_errors"] == 1
        assert result["requests"][1]["correct"] is False
        assert result["requests"][3]["correct"] is False
        assert (Path(directory) / "correctness.txt").is_file()
        tiny_summary["requests"][2]["completion_text"] = r"\boxed{9}"
        result = grade_run(tiny_trace, tiny_summary, directory)
        assert result["status"] == "completed" and result["accuracy_percent"] == 25.0
        protocol_trace = {"requests": [tiny_trace["requests"][0]]}
        protocol_summary = {"requests": [{
            "request_id": "good", "status": "failed", "completion_text": None,
            "error": "ValueError: Finished response has no completion text; correctness cannot be evaluated",
        }]}
        result = grade_run(protocol_trace, protocol_summary, directory)
        assert result["status"] == "failed" and result["accuracy_percent"] is None
        assert result["grading_errors"] == result["missing_responses"] == 1
        assert result["requests"][0]["correct"] is None
        assert result["requests"][0]["status"] == "grading_error"
        original = grader.math_equal
        def failing_comparator(*args, **kwargs):
            raise RuntimeError("injected grading error")
        try:
            grader.math_equal = failing_comparator
            assert grade_completion("2", r"\boxed{2}")["status"] == "grading_error"
        finally:
            grader.math_equal = original
        original_timeout = grader.symbolic_equal_timeout
        def failing_timeout(*args, **kwargs):
            raise TimeoutError("injected original symbolic timeout")
        try:
            grader.symbolic_equal_timeout = failing_timeout
            timeout_result = grade_completion("x", r"\boxed{y}")
            assert timeout_result["status"] == "grading_error"
            assert "Timeout error catched" in timeout_result["grader_output"]
        finally:
            grader.symbolic_equal_timeout = original_timeout
    verified = roundtrip_correct = None
    if trace is not None:
        attached = attach_ground_truth(json.loads(json.dumps(trace)))
        verified = attached["ground_truth_dataset"]["verified_selected_requests"]
        assert all(request["ground_truth"] for request in attached["requests"])
        roundtrip_correct = 0
        for request in attached["requests"]:
            result = grade_completion(request["ground_truth"], request["solution"])
            if result["correct"] is not True or result["status"] != "correct":
                raise AssertionError(("Ground-truth roundtrip failed", request["source_id"], result))
            if result["normalized_reference"] != request["normalized_ground_truth"]:
                raise AssertionError(("Attached reference normalization mismatch", request["source_id"]))
            roundtrip_correct += 1
        prompted = json.loads(json.dumps(attached))
        for request in prompted["requests"]:
            problem = request.get("problem", request["prompt"])
            request["problem"] = problem
            request["prompt"] = make_math_prompt(problem)
        attach_ground_truth(prompted)
        assert [row["ground_truth"] for row in prompted["requests"]] == [
            row["ground_truth"] for row in attached["requests"]]
        assert all(row["prompt"].endswith("\n\nSolution:") for row in prompted["requests"])
    if "vllm" in sys.modules:
        raise AssertionError("CPU grader imported the vllm GPU package")
    return {"status": "passed", "tests": [case[0] for case in cases] + [
        "exact original four-shot prompt", "natural EOS/stop sampling metadata",
        "problem field ground-truth mapping", "repository comparator parity",
        "original extraction fallback", "non-string completion grading error",
        "missing saved output", "failed/OOM denominator", "empty output",
        "finished response text preservation error", "explicit grader exception",
        "original swallowed timeout visibility", "CPU-only imports"],
        "verified_selected_requests": verified, "ground_truth_roundtrip_correct": roundtrip_correct}

class NullJSONFrames:
    """Decode NUL-delimited JSON irrespective of HTTP chunk boundaries."""

    def __init__(self):
        self.pending = bytearray()

    def feed(self, chunk):
        self.pending.extend(chunk)
        frames = []
        while True:
            position = self.pending.find(b"\0")
            if position < 0:
                break
            raw = bytes(self.pending[:position])
            del self.pending[:position + 1]
            if not raw:
                continue
            frame = json.loads(raw)
            if not isinstance(frame, dict):
                raise ValueError("Stream frame is not a JSON object")
            frames.append(frame)
        return frames

    def finish(self):
        if self.pending.strip():
            raise ValueError("Stream ended with an incomplete NUL-delimited frame")


class StreamObservation:
    """Observe CPU arrival times of cumulative output token counts."""

    def __init__(self, request_id, prompt_tokens, expected_output_tokens, sent,
                 require_exact_output=True):
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.expected_output_tokens = expected_output_tokens
        self.require_exact_output = require_exact_output
        self.sent = sent
        self.output_tokens = 0
        self.first = None
        self.last = None
        self.finished = False
        self.first_observed_output_tokens = None
        self.max_tokens_per_update = 0
        self.output_updates = 0
        self.arrivals = []
        self.completion_text = None
        self.finish_reason = None
        self.preview_text = None
        self.preview_output_tokens = None
        self.token_id_observations = []
        self.token_metadata = {}

    def observe(self, frame, arrival):
        if str(frame.get("request_id")) != self.request_id:
            raise ValueError("Response request_id differs from submitted request")
        if self.finished:
            raise ValueError("Received a frame after the finished frame")
        if frame.get("error"):
            raise RuntimeError(str(frame["error"]))
        count = frame.get("output_tokens")
        if isinstance(count, bool) or not isinstance(count, int) or count < self.output_tokens:
            raise ValueError("Invalid or decreasing cumulative output token count")
        if count > self.expected_output_tokens:
            raise ValueError("Response exceeded the requested output length")
        if frame.get("prompt_tokens") != self.prompt_tokens:
            raise ValueError("Response prompt token count differs from shared trace")
        if count > self.output_tokens:
            added = count - self.output_tokens
            self.max_tokens_per_update = max(self.max_tokens_per_update, added)
            self.output_updates += 1
            if self.first is None:
                self.first = arrival
                self.first_observed_output_tokens = count
            self.last = arrival
            self.arrivals.append({"output_tokens": count,
                                  "arrival_s_from_send": arrival - self.sent})
            self.output_tokens = count
        if "last_token_id" in frame:
            token = frame["last_token_id"]
            if count and (isinstance(token, bool) or not isinstance(token, int) or token < 0):
                raise ValueError("Invalid last generated token ID")
            if count and (not self.token_id_observations or self.token_id_observations[-1]["output_tokens"] != count):
                self.token_id_observations.append({"output_tokens": count, "token_id": token})
        if "preview_text" in frame:
            if not isinstance(frame["preview_text"], str):
                raise ValueError("Output preview must be a string")
            self.preview_text = frame["preview_text"]
            self.preview_output_tokens = count
        for name in ("output_token_ids", "stop_token_ids", "effective_eos_token_ids"):
            if name in frame:
                ids = frame[name]
                if not isinstance(ids, list) or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in ids):
                    raise ValueError(f"Invalid {name}")
                if name == "output_token_ids" and len(ids) != count:
                    raise ValueError("Output token IDs differ from cumulative token count")
                self.token_metadata[name] = list(ids)
        if frame.get("finished"):
            if "text" not in frame:
                raise ValueError("Finished response is missing completion text; correctness cannot be evaluated")
            if not isinstance(frame["text"], str):
                raise ValueError("Finished response completion text must be a string; correctness cannot be evaluated")
            reason = frame.get("finish_reason")
            if reason not in ("length", "stop"):
                raise ValueError(f"Finished response has invalid finish_reason: {reason!r}")
            self.completion_text = frame["text"]
            self.finish_reason = reason
            self.finished = True

    def metrics(self):
        if not self.finished:
            raise ValueError("Response ended without a finished frame")
        if self.require_exact_output and self.output_tokens != self.expected_output_tokens:
            raise ValueError(f"Expected {self.expected_output_tokens} output tokens, "
                             f"received {self.output_tokens}")
        if self.first is None or self.last is None:
            raise ValueError("Response did not contain generated tokens")
        result = {
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "ttft_ms": (self.first - self.sent) * 1000,
            "tpot_ms": ((self.last - self.first) * 1000 / (self.output_tokens - 1)
                        if self.output_tokens > 1 else None),
            "e2e_s": self.last - self.sent,
            "first_observed_output_tokens": self.first_observed_output_tokens,
            "max_tokens_per_update": self.max_tokens_per_update,
            "output_updates": self.output_updates,
            "token_arrivals": self.arrivals,
            "completion_text": self.completion_text,
            "finish_reason": self.finish_reason,
            "max_output_tokens": self.expected_output_tokens,
            "require_exact_output": self.require_exact_output,
        }
        result.update(self.output_evidence())
        return result

    def output_evidence(self):
        evidence = dict(self.token_metadata)
        if self.token_id_observations:
            evidence["token_id_observations"] = list(self.token_id_observations)
            if [row["output_tokens"] for row in self.token_id_observations] == list(range(1, self.output_tokens + 1)):
                evidence.setdefault("output_token_ids", [row["token_id"] for row in self.token_id_observations])
            evidence["output_token_ids_complete"] = self.finished and "output_token_ids" in self.token_metadata
        if self.preview_text is not None:
            evidence.update(partial_text=self.preview_text,
                            partial_text_output_tokens=self.preview_output_tokens)
        return evidence

    def partial_metrics(self):
        return {"prompt_tokens": self.prompt_tokens, "output_tokens": self.output_tokens,
                "token_arrivals": self.arrivals, "completion_text": self.completion_text,
                "finish_reason": self.finish_reason, **self.output_evidence()}


def repeated_output(text):
    """Flag eight consecutive identical lines or substantial repeated text units."""
    tail = text[-2048:]
    previous = None
    count = 0
    for line in (line.strip() for line in tail.splitlines() if line.strip()):
        if len(line) < 8:
            previous, count = None, 0
            continue
        count = count + 1 if line == previous else 1
        previous = line
        if count >= 8:
            return {"kind": "repeated_line", "unit": line, "repeats": count}
    compact = " ".join(tail.split()) + (" " if tail[-1:].isspace() else "")
    # A suffix of a periodic stream is periodic even when the preview cuts
    # through a word. Bounded suffix comparisons avoid an expensive global
    # backreference search on every otherwise ordinary math preview.
    for size in range(12, min(256, len(compact) // 8) + 1):
        unit = compact[-size:]
        if (compact.endswith(unit * 8) and sum(character.isalnum() for character in unit) >= 4
                and len({character for character in unit if character.isalnum()}) >= 3):
            repeats = 8
            while compact.endswith(unit * (repeats + 1)):
                repeats += 1
            return {"kind": "periodic_text", "unit": unit, "repeats": repeats}
    return None


def detect_output_corruption(text):
    """Detect severe LaTeX structure collapse in a completed output.

    An unfinished brace alone is common while streaming and is not enough to
    invalidate a response. Require at least four unclosed literal braces plus
    an unclosed environment or an empty terminal display-math pair.
    """
    unclosed_braces = 0
    unmatched_closing_braces = 0
    backslashes = 0
    for character in text:
        if character == "\\":
            backslashes += 1
            continue
        escaped = backslashes % 2 == 1
        backslashes = 0
        if not escaped and character == "{":
            unclosed_braces += 1
        elif not escaped and character == "}":
            if unclosed_braces:
                unclosed_braces -= 1
            else:
                unmatched_closing_braces += 1
    if unclosed_braces < 4:
        return None
    environments = []
    for match in re.finditer(r"(?<!\\)\\(begin|end)\{([^{}\s]+)\}", text):
        command, name = match.groups()
        if command == "begin":
            environments.append(name)
        elif environments and environments[-1] == name:
            environments.pop()
    terminal_empty_display = re.search(r"(?<!\\)\$\$\s*(?<!\\)\$\$\s*\Z", text) is not None
    if not environments and not terminal_empty_display:
        return None
    return {"kind": "malformed_latex", "unclosed_braces": unclosed_braces,
            "unmatched_closing_braces": unmatched_closing_braces,
            "unclosed_environments": environments,
            "terminal_empty_display": terminal_empty_display, "unit": text[-256:],
            "reason": f"{unclosed_braces} unclosed braces, unclosed environments={environments}, "
                      f"terminal empty display={terminal_empty_display}"}


class InvalidOutput(RuntimeError):
    def __init__(self, evidence):
        self.evidence = evidence
        reason = evidence.get("reason")
        super().__init__(reason if reason is not None else
                         f"{evidence['kind']}: {evidence['repeats']} consecutive repeats")


class OutputMonitor:
    def __init__(self, path):
        self.path = path
        self.stream = path.open("x", encoding="utf-8")
        self.abort_evidence = None
        self.checks = 0
        self.detector_seconds = 0.0

    def observe(self, observation, frame):
        preview = frame.get("preview_text", frame.get("text") if frame.get("finished") else None)
        if preview is None:
            return
        self.stream.write(f"REQUEST id={observation.request_id} output_tokens={observation.output_tokens} "
                          f"last_token_id={frame.get('last_token_id')} finished={observation.finished}\n")
        self.stream.write(preview + "\n\n")
        self.stream.flush()
        if self.abort_evidence is not None:
            return
        started = time.perf_counter()
        detected = repeated_output(preview)
        if detected is None and observation.finished:
            detected = detect_output_corruption(observation.completion_text)
        self.detector_seconds += time.perf_counter() - started
        self.checks += 1
        if detected:
            self.abort_evidence = {"request_id": observation.request_id,
                                   "output_tokens": observation.output_tokens,
                                   "preview_text": preview, **detected}
            self.stream.write("OUTPUT_ABORT " + json.dumps(self.abort_evidence, ensure_ascii=False) + "\n")
            self.stream.flush()
            raise InvalidOutput(self.abort_evidence)

    def close(self):
        self.stream.close()


def stats(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None,
                "p99": None, "min": None, "max": None}
    return {"n": len(values), "mean": statistics.mean(values),
            "p50": statistics.median(values),
            "p95": values[math.ceil(0.95 * len(values)) - 1],
            "p99": values[math.ceil(0.99 * len(values)) - 1],
            "min": values[0], "max": values[-1], "percentile_method": "nearest rank"}


def is_memory_oom(error):
    message = str(error).lower()
    return ("out of memory" in message or "outofmemoryerror" in message or
            "cuda_error_out_of_memory" in message)


def number(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}"


def parse_request_rate(value):
    try:
        rate = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("request rate must be a positive number or inf") from error
    if math.isnan(rate) or rate <= 0:
        raise argparse.ArgumentTypeError("request rate must be positive and finite, or positive inf")
    return rate


class Reporter:
    def __init__(self, path, prefix="", stdout_stream=None):
        self.stream = path.open("x", encoding="utf-8")
        self.prefix = prefix
        self.stdout_stream = stdout_stream

    def write(self, text):
        print(self.prefix + text, flush=True)
        self.stream.write(text + "\n")
        self.stream.flush()
        if self.stdout_stream is not None:
            self.stdout_stream.write(text + "\n")
            self.stdout_stream.flush()

    def close(self):
        self.stream.close()


def sampling_settings(row, defaults=None):
    defaults = defaults or {}
    ignore_eos = row.get("ignore_eos", defaults.get("ignore_eos", True))
    if not isinstance(ignore_eos, bool):
        raise ValueError("ignore_eos must be a boolean")
    stop = row.get("stop", defaults.get("stop", []))
    if stop is None:
        stop = []
    elif isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or any(not isinstance(item, str) or not item for item in stop):
        raise ValueError("stop must be a list of nonempty strings")
    settings = {"ignore_eos": ignore_eos, "stop": list(stop)}
    penalty = row.get("repetition_penalty", defaults.get("repetition_penalty", 1.0))
    if (isinstance(penalty, bool) or not isinstance(penalty, (int, float))
            or not math.isfinite(penalty) or penalty <= 0):
        raise ValueError("repetition_penalty must be finite and positive")
    if penalty != 1.0:
        settings["repetition_penalty"] = float(penalty)
    frequency = row.get("frequency_penalty", defaults.get("frequency_penalty", 0.0))
    if (isinstance(frequency, bool) or not isinstance(frequency, (int, float))
            or not math.isfinite(frequency) or not -2 <= frequency <= 2):
        raise ValueError("frequency_penalty must be finite and within [-2, 2]")
    if frequency != 0.0:
        settings["frequency_penalty"] = float(frequency)
    temperature = row.get("temperature", defaults.get("temperature", 0.0))
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature < 0):
        raise ValueError("temperature must be finite and nonnegative")
    top_p = row.get("top_p", defaults.get("top_p", 1.0))
    if (isinstance(top_p, bool) or not isinstance(top_p, (int, float))
            or not math.isfinite(top_p) or not 0 < top_p <= 1):
        raise ValueError("top_p must be finite and within (0, 1]")
    if temperature != 0.0:
        settings["temperature"] = float(temperature)
    if top_p != 1.0:
        settings["top_p"] = float(top_p)
    ngram_size = row.get("no_repeat_ngram_size", defaults.get("no_repeat_ngram_size", 0))
    max_repeats = row.get("max_ngram_repeats", defaults.get("max_ngram_repeats", 2))
    if isinstance(ngram_size, bool) or not isinstance(ngram_size, int) or ngram_size < 0:
        raise ValueError("no_repeat_ngram_size must be a nonnegative integer")
    if isinstance(max_repeats, bool) or not isinstance(max_repeats, int) or max_repeats < 1:
        raise ValueError("max_ngram_repeats must be a positive integer")
    if ngram_size:
        settings.update(no_repeat_ngram_size=ngram_size, max_ngram_repeats=max_repeats)
    return settings


def request_payload(row, max_tokens, request_id=None, force_fixed=False):
    settings = sampling_settings(row)
    if force_fixed:
        settings.update(ignore_eos=True, stop=[])
    return {"request_id": str(row["request_id"]) if request_id is None else request_id,
            "prompt": row["prompt"], "prompt_token_ids": row["prompt_token_ids"],
            "max_tokens": max_tokens, "temperature": 0.0, "n": 1,
            "stream": True, **settings}


def load_trace(path, request_rate):
    request_rate = parse_request_rate(request_rate)
    source = json.loads(path.read_text(encoding="utf-8"))
    requests = source["requests"]
    if not isinstance(requests, list) or not requests:
        raise ValueError("Trace must contain a nonempty requests list")
    if "request_rate" in source and float(source["request_rate"]) != request_rate:
        raise ValueError("CLI request rate differs from the shared trace")
    seen = set()
    previous = -1.0
    for row in requests:
        request_id = str(row["request_id"])
        if request_id in seen:
            raise ValueError("Trace request IDs must be unique")
        seen.add(request_id)
        offset = row["scheduled_offset_s"]
        if not isinstance(offset, (int, float)) or not math.isfinite(offset) or offset < previous:
            raise ValueError("Trace offsets must be finite, nonnegative and ordered")
        if offset < 0:
            raise ValueError("Trace offsets must be nonnegative")
        if math.isinf(request_rate) and offset != 0:
            raise ValueError("An inf request-rate trace must schedule every request at offset 0")
        previous = offset
        token_ids = row["prompt_token_ids"]
        if not token_ids or any(isinstance(token, bool) or not isinstance(token, int) or token < 0
                                for token in token_ids):
            raise ValueError("Trace prompt_token_ids must contain nonnegative integers")
        if len(token_ids) != row["prompt_length"]:
            raise ValueError("Trace prompt_length differs from its token IDs")
        limit = row["max_tokens"]
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("Trace max_tokens must be a positive integer")
        sampling_settings(row, source)
        if not isinstance(row["prompt"], str):
            raise ValueError("Trace prompt must be a string")
    return source


async def endpoint_json(session, url, method="GET"):
    async with session.request(method, url, json={} if method == "POST" else None) as response:
        text = await response.text()
        if response.status != 200:
            raise RuntimeError(f"{method} {url}: HTTP {response.status}: {text[:2000]}")
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError(f"{url} did not return a JSON object")
        return value


async def stream_request(session, url, row, expected_output_tokens, sent, request_id=None,
                         force_fixed=False, monitor=None):
    payload = request_payload(row, expected_output_tokens, request_id, force_fixed)
    request_id = payload["request_id"]
    observation = StreamObservation(request_id, row["prompt_length"], expected_output_tokens, sent,
                                    require_exact_output=payload["ignore_eos"] and not payload["stop"])
    parser = NullJSONFrames()
    try:
        async with session.post(url + "/generate", json=payload) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {body[:2000]}")
            async for chunk in response.content.iter_any():
                arrival = time.perf_counter()
                for frame in parser.feed(chunk):
                    observation.observe(frame, arrival)
                    if monitor is not None:
                        monitor.observe(observation, frame)
            parser.finish()
        return observation.metrics(), None
    except InvalidOutput as error:
        return {**observation.partial_metrics(), "output_abort": error.evidence}, str(error)
    except asyncio.CancelledError:
        if monitor is None or monitor.abort_evidence is None:
            raise
        return {**observation.partial_metrics(), "cancelled_due_to_output_abort": True}, "Cancelled after invalid output"
    except Exception as error:
        # Preserve observed token counts on a failed stream, but do not report
        # incomplete-request latency values as successful measurements.
        return observation.partial_metrics(), f"{type(error).__name__}: {error}"


async def run_http_client(args, trace, reporter):
    monitor = OutputMonitor(args.output_dir / "output-monitor.txt") if getattr(args, "monitor_output", False) else None
    try:
        return await _run_http_client(args, trace, reporter, monitor)
    finally:
        if monitor is not None:
            monitor.close()


async def _run_http_client(args, trace, reporter, monitor=None):
    import aiohttp

    url = args.url.rstrip("/")
    requests = [{**row, **sampling_settings(row, trace)} for row in trace["requests"]]
    timeout = aiohttp.ClientTimeout(total=args.timeout_seconds)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    started_utc = datetime.now(timezone.utc).isoformat()
    reporter.write(f"Online HTTP Benchmark, started={started_utc}")
    limits = sorted({row["max_tokens"] for row in requests})
    reporter.write(f"URL={url}, requests={len(requests)}, offered_rate={args.request_rate} req/s, "
                   f"max_output_tokens={limits}, timeout={args.timeout_seconds}s, retries=0")
    settings = sorted({json.dumps({"max_tokens": row["max_tokens"],
                                   "repetition_penalty": row.get("repetition_penalty", 1.0),
                                   "frequency_penalty": row.get("frequency_penalty", 0.0),
                                   "temperature": row.get("temperature", 0.0),
                                   "top_p": row.get("top_p", 1.0),
                                   "no_repeat_ngram_size": row.get("no_repeat_ngram_size", 0),
                                   "max_ngram_repeats": row.get("max_ngram_repeats", 2),
                                   "ignore_eos": row["ignore_eos"], "stop": row["stop"]}, sort_keys=True)
                       for row in requests})
    reporter.write("Generation settings: " + "; ".join(settings))
    if math.isinf(args.request_rate):
        reporter.write("Arrival mode: all requests scheduled at trace time 0, with no client concurrency limit. "
                       "Actual HTTP send times and send lag are recorded.")
    else:
        reporter.write("Arrival mode: shared Poisson arrival trace, with no client concurrency limit.")
    reporter.write(f"Shared trace={args.inputs.resolve()}")
    reporter.write("TTFT, TPOT and E2E Latency use observed HTTP token arrival times. "
                   "Throughput includes the complete offered trace and its drain time.")
    records = []
    counters = {"submitted": 0, "completed": 0, "failed": 0}
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        health = await endpoint_json(session, url + "/health")
        if health.get("status") != "ready":
            raise RuntimeError(f"Server is not ready: {health}")
        warmup, error = await stream_request(session, url, requests[0], 16,
                                             time.perf_counter(), "warmup", force_fixed=True)
        if error:
            raise RuntimeError("Warmup failed: " + error)
        reset = await endpoint_json(session, url + "/reset", "POST")
        if reset.get("status") != "reset":
            raise RuntimeError(f"Server metrics were not reset after warmup: {reset}")
        reporter.write("Warmup completed, server metrics reset. Measured trace starts now.")
        started = time.perf_counter()
        stop_progress = asyncio.Event()

        async def send_one(index, row):
            target = started + row["scheduled_offset_s"]
            remaining = target - time.perf_counter()
            if remaining > 0:
                await asyncio.sleep(remaining)
            sent = time.perf_counter()
            counters["submitted"] += 1
            record = {"index": index, "request_id": str(row["request_id"]),
                      "scheduled_offset_s": row["scheduled_offset_s"],
                      "send_offset_s": sent - started,
                      "send_lag_ms": (sent - target) * 1000,
                      "max_output_tokens": row["max_tokens"],
                      "ignore_eos": row["ignore_eos"], "stop": row["stop"]}
            measurements, error = await stream_request(session, url, row, row["max_tokens"], sent, monitor=monitor)
            record.update(measurements)
            record["response_end_offset_s"] = time.perf_counter() - started
            record["status"] = ("invalid_output" if measurements.get("output_abort") else
                                "cancelled" if measurements.get("cancelled_due_to_output_abort") else
                                "OOM" if error and is_memory_oom(error) else "failed" if error else "completed")
            record["error"] = error
            counters["failed" if error else "completed"] += 1
            records.append(record)
            reporter.write(f"REQUEST id={record['request_id']} status={record['status']} "
                           f"input_tokens={record['prompt_tokens']} output_tokens={record['output_tokens']} "
                           f"TTFT={number(record.get('ttft_ms'))} ms "
                           f"TPOT={number(record.get('tpot_ms'))} ms/token "
                           f"E2E_Latency={number(record.get('e2e_s'))} s "
                           f"send_lag={number(record['send_lag_ms'])} ms" +
                           (f" error={error}" if error else ""))
            if measurements.get("output_abort"):
                evidence = measurements["output_abort"]
                reporter.write(f"OUTPUT_ABORT request_id={record['request_id']} "
                               f"output_tokens={record['output_tokens']} reason={evidence['kind']} "
                               + (f"repeats={evidence['repeats']} " if "repeats" in evidence else
                                  f"unclosed_braces={evidence['unclosed_braces']} "
                                  f"unclosed_environments={evidence['unclosed_environments']} ")
                               + f"unit={evidence['unit'][:160]!r} "
                               f"evidence={monitor.path.resolve()}")
                current = asyncio.current_task()
                for task in request_tasks:
                    if task is not current and not task.done():
                        task.cancel()

        async def progress():
            while not stop_progress.is_set():
                try:
                    await asyncio.wait_for(stop_progress.wait(), timeout=30)
                except asyncio.TimeoutError:
                    reporter.write(f"PROGRESS elapsed={time.perf_counter() - started:.1f}s "
                                   f"submitted={counters['submitted']}/{len(requests)} "
                                   f"completed={counters['completed']} failed={counters['failed']} "
                                   f"active={counters['submitted'] - counters['completed'] - counters['failed']}")

        progress_task = asyncio.create_task(progress())
        request_tasks = [asyncio.create_task(send_one(index, row)) for index, row in enumerate(requests)]
        try:
            # Creating all tasks immediately preserves the shared trace even
            # when earlier requests remain active. Connections are unlimited.
            await asyncio.gather(*request_tasks)
        except asyncio.CancelledError:
            if monitor is None or monitor.abort_evidence is None:
                raise
            await asyncio.gather(*request_tasks, return_exceptions=True)
        finally:
            stop_progress.set()
            await progress_task
        duration = time.perf_counter() - started
        metrics_error = None
        try:
            server_metrics = await endpoint_json(session, url + "/metrics")
        except Exception as error:
            server_metrics = None
            metrics_error = f"{type(error).__name__}: {error}"
    records.sort(key=lambda row: row["index"])
    completed = [row for row in records if row["status"] == "completed"]
    sends = [row["send_offset_s"] for row in records]
    send_span = max(sends) - min(sends) if len(sends) > 1 else 0
    generated_tokens = sum(row["output_tokens"] for row in completed)
    result = {
        "status": ("invalid_output" if monitor is not None and monitor.abort_evidence is not None else
                   "completed" if not counters["failed"] and not metrics_error else "failed"),
        "started_utc": started_utc,
        "url": url, "trace_path": str(args.inputs.resolve()),
        "offered_request_rate": "inf" if math.isinf(args.request_rate) else args.request_rate,
        "total_requests": len(requests), **counters,
        "generated_tokens": generated_tokens,
        "partial_failed_output_tokens": sum(row["output_tokens"] for row in records
                                             if row["status"] != "completed"),
        "duration_s": duration,
        "request_throughput": len(completed) / duration,
        "output_token_throughput": generated_tokens / duration,
        "avg_latency_per_token_s": (statistics.mean(row["e2e_s"] /
                                                       (row["prompt_tokens"] + row["output_tokens"])
                                                       for row in completed) if completed else None),
        "ttft_ms": stats(row["ttft_ms"] for row in completed),
        "tpot_ms": stats(row["tpot_ms"] for row in completed),
        "e2e_s": stats(row["e2e_s"] for row in completed),
        "actual_send_rate": ((len(sends) - 1) / send_span if send_span > 0 else None),
        "actual_send_rate_definition": "(submitted requests - 1) / (last send - first send)",
        "first_send_offset_s": min(sends) if sends else None, "last_send_offset_s": max(sends) if sends else None,
        "send_lag_ms": stats(row["send_lag_ms"] for row in records),
        "timeout_seconds": args.timeout_seconds, "warmup_output_tokens": warmup["output_tokens"],
        "server_metrics": server_metrics, "metrics_error": metrics_error, "requests": records,
    }
    if monitor is not None:
        result.update(output_abort=monitor.abort_evidence,
                      output_monitor={"path": str(monitor.path.resolve()), "checks": monitor.checks,
                                      "detector_seconds": monitor.detector_seconds})
    reporter.write(f"RESULT status={result['status']} completed={len(completed)}/{len(requests)} "
                   f"failed={counters['failed']} time={duration:.3f}s "
                   f"request_throughput={result['request_throughput']:.6f} req/s "
                   f"output_token_throughput={result['output_token_throughput']:.3f} tok/s")
    reporter.write("Metric                       Mean          P95          P99          Min")
    for label, metric in (("TTFT (ms)", "ttft_ms"), ("TPOT (ms/token)", "tpot_ms"),
                          ("E2E Latency (s)", "e2e_s")):
        row = result[metric]
        reporter.write(f"{label:<25} {number(row['mean']):>12} {number(row['p95']):>12} "
                       f"{number(row['p99']):>12} {number(row['min']):>12}")
    reporter.write(f"Average Latency per Token={number(result['avg_latency_per_token_s'], 6)} s/token")
    reporter.write(f"Actual send rate={number(result['actual_send_rate'], 6)} req/s, "
                   f"mean send lag={number(result['send_lag_ms']['mean'])} ms, "
                   f"max send lag={number(result['send_lag_ms']['max'])} ms")
    if metrics_error:
        reporter.write("SERVER METRICS ERROR " + metrics_error)
    return result


def write_responses(path, trace, result):
    """Write original prompts and completions after timed HTTP work has ended."""
    with path.open("x", encoding="utf-8") as stream:
        stream.write("Online HTTP Responses\n")
        stream.write("Prompts and completions are saved after performance measurement.\n")
        for record in result["requests"]:
            row = trace["requests"][record["index"]]
            stream.write("\n" + "=" * 80 + "\n")
            stream.write(f"Request ID: {record['request_id']}\n")
            stream.write(f"Status: {record['status']}\n")
            stream.write(f"Finish Reason: {record.get('finish_reason')}\n")
            stream.write(f"Output Tokens: {record['output_tokens']}\n")
            if record.get("error"):
                stream.write(f"Error: {record['error']}\n")
            stream.write("\nPrompt:\n" + row["prompt"] + "\n")
            stream.write("\nCompletion:\n")
            if record.get("completion_text") is None:
                stream.write("[MISSING COMPLETION TEXT, correctness cannot be evaluated]\n")
                if record.get("partial_text") is not None:
                    stream.write("\nObserved partial output tail (last 2048 characters):\n" + record["partial_text"] + "\n")
            else:
                stream.write(record["completion_text"] + "\n")
            if "output_token_ids" in record:
                stream.write("\nObserved output token IDs (complete=" + str(record.get("output_token_ids_complete", False)) + "):\n")
                stream.write(json.dumps(record["output_token_ids"]) + "\n")
            if "token_id_observations" in record:
                stream.write("\nObserved token positions and IDs:\n" + json.dumps(record["token_id_observations"]) + "\n")
            for name in ("stop_token_ids", "effective_eos_token_ids"):
                if name in record:
                    stream.write(name + ": " + json.dumps(record[name]) + "\n")


def protocol_self_test():
    import unittest

    # Exact malformed suffix from DiffKV math-086, naturally stopped at 404
    # tokens in online-math-instruct-frequency03-20261011. Keep a local fixture
    # so this regression does not depend on generated log files being present.
    malformed_math_086 = r"""$$\begin{split}
& \quad \left(\dfrac{\dfrac{(3+2\sqrt5)^2}{4}}{\dfrac{\dfrac{(3+2\sqrt5)^2}{4}}{\dfrac{\dfrac{\dfrac{(3+2\sqrt5)^2}{4}\dfrac{\dfrac{(3+2\sqrt5)^2}{4}\dfrac{(3+2\sqrt5)^2}{4}\dfrac{(3+2\sqrt5)^2}{4}\dfrac{(3+2\sqrt5)^2}{4}\dfrac{(3+2\\
Let's simplify the expression:
$$
$$"""

    class ProtocolTests(unittest.TestCase):
        def test_fragmented_and_coalesced_frames(self):
            frames = [{"request_id": "a", "output_tokens": count,
                       "prompt_tokens": 3, "finished": count == 4}
                      for count in (0, 1, 4)]
            frames[-1].update(text="The answer is \\boxed{7}.", finish_reason="length")
            wire = b"\0".join(json.dumps(frame).encode() for frame in frames) + b"\0"
            parser = NullJSONFrames()
            decoded = []
            for chunk in (wire[:1], wire[1:14], wire[14:40], wire[40:]):
                decoded.extend(parser.feed(chunk))
            parser.finish()
            self.assertEqual(decoded, frames)
            parser = NullJSONFrames()
            self.assertEqual(parser.feed(wire), frames)
            parser.finish()

        def test_first_last_and_separate_finished_frame(self):
            observed = StreamObservation("a", 3, 4, sent=10)
            def frame(count, finished=False):
                result = {"request_id": "a", "prompt_tokens": 3,
                          "output_tokens": count, "finished": finished}
                if finished:
                    result.update(text="The answer is \\boxed{7}.", finish_reason="length")
                return result
            observed.observe(frame(0), 10.5)
            observed.observe(frame(1), 12)
            observed.observe(frame(4), 18)
            observed.observe(frame(4, True), 20)
            metrics = observed.metrics()
            self.assertEqual(metrics["ttft_ms"], 2000)
            self.assertEqual(metrics["tpot_ms"], 2000)
            self.assertEqual(metrics["e2e_s"], 8)
            self.assertEqual(metrics["max_tokens_per_update"], 3)
            self.assertEqual(metrics["completion_text"], "The answer is \\boxed{7}.")
            self.assertEqual(metrics["finish_reason"], "length")

        def test_first_update_contains_multiple_tokens(self):
            observed = StreamObservation("a", 3, 4, sent=10)
            observed.observe({"request_id": "a", "prompt_tokens": 3,
                              "output_tokens": 2, "finished": False}, 12)
            observed.observe({"request_id": "a", "prompt_tokens": 3,
                              "output_tokens": 4, "finished": True,
                              "text": "The answer is \\boxed{7}.", "finish_reason": "length"}, 18)
            self.assertEqual(observed.metrics()["tpot_ms"], 2000)
            self.assertEqual(observed.metrics()["first_observed_output_tokens"], 2)

        def test_incomplete_and_short_stream_rejected(self):
            parser = NullJSONFrames()
            parser.feed(b'{"request_id":')
            with self.assertRaises(ValueError):
                parser.finish()
            observed = StreamObservation("a", 3, 4, sent=10)
            observed.observe({"request_id": "a", "prompt_tokens": 3,
                              "output_tokens": 3, "finished": True,
                              "text": "The answer is \\boxed{7}.", "finish_reason": "length"}, 12)
            with self.assertRaises(ValueError):
                observed.metrics()

        def test_short_natural_stop_and_single_token_eos(self):
            observed = StreamObservation("a", 3, 1024, sent=10, require_exact_output=False)
            observed.observe({"request_id": "a", "prompt_tokens": 3,
                              "output_tokens": 1, "finished": False}, 11)
            observed.observe({"request_id": "a", "prompt_tokens": 3,
                              "output_tokens": 2, "finished": True,
                              "text": "\\boxed{7}", "finish_reason": "stop"}, 12)
            metrics = observed.metrics()
            self.assertEqual(metrics["output_tokens"], 2)
            self.assertEqual(metrics["tpot_ms"], 1000)
            self.assertEqual(metrics["e2e_s"], 2)
            self.assertFalse(metrics["require_exact_output"])
            one = StreamObservation("a", 3, 1024, sent=10, require_exact_output=False)
            one.observe({"request_id": "a", "prompt_tokens": 3,
                         "output_tokens": 1, "finished": True,
                         "text": "", "finish_reason": "stop"}, 11)
            self.assertEqual(one.metrics()["output_tokens"], 1)
            self.assertIsNone(one.metrics()["tpot_ms"])

        def test_natural_output_cannot_be_empty_or_exceed_limit(self):
            overrun = StreamObservation("a", 3, 4, sent=10, require_exact_output=False)
            with self.assertRaisesRegex(ValueError, "exceeded"):
                overrun.observe({"request_id": "a", "prompt_tokens": 3,
                                 "output_tokens": 5, "finished": False}, 11)
            empty = StreamObservation("a", 3, 4, sent=10, require_exact_output=False)
            empty.observe({"request_id": "a", "prompt_tokens": 3,
                           "output_tokens": 0, "finished": True,
                           "text": "", "finish_reason": "stop"}, 11)
            with self.assertRaisesRegex(ValueError, "generated tokens"):
                empty.metrics()

        def test_stop_forwarding_and_forced_warmup(self):
            from online_benchmark_server import sampling_options, validate_output_length

            row = {"request_id": "a", "prompt": "problem", "prompt_token_ids": [1, 2, 3],
                   "ignore_eos": False, "stop": ["Problem:", "Problem:\n"]}
            payload = request_payload(row, 1024)
            forwarded = sampling_options(payload)
            self.assertFalse(forwarded["ignore_eos"])
            self.assertEqual(forwarded["stop"], row["stop"])
            self.assertEqual(forwarded["max_tokens"], 1024)
            self.assertEqual(forwarded["temperature"], 0)
            self.assertEqual(forwarded["n"], 1)
            validate_output_length(3, 1024, forwarded["ignore_eos"], forwarded["stop"])
            warmup = request_payload(row, 16, "warmup", force_fixed=True)
            self.assertTrue(warmup["ignore_eos"])
            self.assertEqual(warmup["stop"], [])
            self.assertEqual(warmup["max_tokens"], 16)
            with self.assertRaisesRegex(RuntimeError, "Expected 16"):
                validate_output_length(3, 16, True, [])
            with self.assertRaises(RuntimeError):
                validate_output_length(1025, 1024, False, [])

        def test_generated_ngram_limits_forwarded_without_changing_lengths(self):
            row = {"request_id": "a", "prompt": "problem", "prompt_token_ids": [1, 2, 3],
                   "ignore_eos": False, "stop": ["Problem:"], "no_repeat_ngram_size": 32,
                   "max_ngram_repeats": 2, "temperature": 0.6, "top_p": 0.9}
            payload = request_payload(row, 4096)
            self.assertEqual(payload["no_repeat_ngram_size"], 32)
            self.assertEqual(payload["max_ngram_repeats"], 2)
            self.assertEqual(payload["max_tokens"], 4096)
            self.assertFalse(payload["ignore_eos"])
            self.assertEqual(payload["temperature"], 0.6)
            self.assertEqual(payload["top_p"], 0.9)
            warmup = request_payload(row, 16, "warmup", force_fixed=True)
            self.assertEqual(warmup["max_tokens"], 16)
            self.assertEqual(warmup["no_repeat_ngram_size"], 32)
            self.assertTrue(warmup["ignore_eos"])
            disabled = {**row, "no_repeat_ngram_size": 0}
            self.assertNotIn("no_repeat_ngram_size", request_payload(disabled, 4096))
            self.assertNotIn("max_ngram_repeats", request_payload(disabled, 4096))
            for name, values in (("no_repeat_ngram_size", (True, -1, 3.5)),
                                 ("max_ngram_repeats", (True, 0, -1, 2.5))):
                for value in values:
                    with self.subTest(name=name, value=value):
                        with self.assertRaises(ValueError):
                            request_payload({**row, name: value}, 4096)

        def test_real_stream_path_accepts_natural_stop_and_rejects_fixed_short(self):
            class Content:
                async def iter_any(self):
                    frames = [{"request_id": "a", "prompt_tokens": 3,
                               "output_tokens": 1, "finished": False},
                              {"request_id": "a", "prompt_tokens": 3,
                               "output_tokens": 3, "finished": True,
                               "text": "\\boxed{7}", "finish_reason": "stop"}]
                    for frame in frames:
                        await asyncio.sleep(0)
                        yield json.dumps(frame).encode() + b"\0"

            class Response:
                status = 200
                content = Content()

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    return False

            class Session:
                payload = None

                def post(self, url, json):
                    self.payload = json
                    return Response()

            row = {"request_id": "a", "prompt": "problem", "prompt_token_ids": [1, 2, 3],
                   "prompt_length": 3, "ignore_eos": False, "stop": ["Problem:"]}
            session = Session()
            result, error = asyncio.run(stream_request(session, "http://fixture", row, 1024,
                                                       time.perf_counter()))
            self.assertIsNone(error)
            self.assertEqual(result["output_tokens"], 3)
            self.assertEqual(result["completion_text"], "\\boxed{7}")
            self.assertFalse(session.payload["ignore_eos"])
            self.assertEqual(session.payload["stop"], ["Problem:"])
            fixed = {key: value for key, value in row.items() if key not in ("ignore_eos", "stop")}
            result, error = asyncio.run(stream_request(session, "http://fixture", fixed, 1024,
                                                       time.perf_counter()))
            self.assertIn("Expected 1024 output tokens", error)

        def test_completion_text_preserved_from_fragmented_final_frame(self):
            text = "Solution:\nThe answer is \\boxed{\\frac{1}{2}}.\n끝"
            frames = [{"request_id": "a", "prompt_tokens": 3,
                       "output_tokens": 1, "finished": False},
                      {"request_id": "a", "prompt_tokens": 3,
                       "output_tokens": 4, "finished": True,
                       "text": text, "finish_reason": "length"}]
            wire = b"\0".join(json.dumps(frame, ensure_ascii=False).encode() for frame in frames) + b"\0"
            parser = NullJSONFrames()
            decoded = parser.feed(wire[:len(wire) - 4])
            decoded.extend(parser.feed(wire[len(wire) - 4:]))
            parser.finish()
            observed = StreamObservation("a", 3, 4, sent=10)
            for frame, arrival in zip(decoded, (12, 18)):
                observed.observe(frame, arrival)
            metrics = observed.metrics()
            self.assertEqual(metrics["completion_text"], text)
            self.assertEqual(metrics["finish_reason"], "length")
            self.assertEqual(metrics["ttft_ms"], 2000)
            self.assertEqual(metrics["tpot_ms"], 2000)
            self.assertEqual(metrics["e2e_s"], 8)

        def test_missing_or_malformed_final_text_is_explicit_failure(self):
            base = {"request_id": "a", "prompt_tokens": 3,
                    "output_tokens": 4, "finished": True, "finish_reason": "length"}
            for text_fields in ({}, {"text": None}, {"text": ["not a string"]}):
                with self.subTest(text_fields=text_fields):
                    parser = NullJSONFrames()
                    frames = parser.feed(json.dumps({**base, **text_fields}).encode() + b"\0")
                    observed = StreamObservation("a", 3, 4, sent=10)
                    with self.assertRaisesRegex(ValueError, "correctness cannot be evaluated"):
                        observed.observe(frames[0], 12)

        def test_invalid_finish_reason_rejected(self):
            observed = StreamObservation("a", 3, 4, sent=10)
            with self.assertRaisesRegex(ValueError, "invalid finish_reason"):
                observed.observe({"request_id": "a", "prompt_tokens": 3,
                                  "output_tokens": 4, "finished": True,
                                  "text": "answer", "finish_reason": None}, 12)

        def test_readable_responses_preserve_prompt_and_completion(self):
            import tempfile

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "responses.txt"
                trace = {"requests": [{"prompt": "Problem:\n1 + 1 = ?"}]}
                result = {"requests": [{"index": 0, "request_id": "a", "status": "completed",
                                         "finish_reason": "length", "output_tokens": 4,
                                         "completion_text": "The answer is \\boxed{2}.\n", "error": None}]}
                write_responses(path, trace, result)
                saved = path.read_text(encoding="utf-8")
                self.assertIn("Prompt:\nProblem:\n1 + 1 = ?", saved)
                self.assertIn("Completion:\nThe answer is \\boxed{2}.\n", saved)

        def test_oom_requires_literal_memory_error(self):
            self.assertTrue(is_memory_oom("CUDA out of memory"))
            self.assertFalse(is_memory_oom("HTTP 500: request failed"))

        def test_cli_request_rate_finite_and_positive_inf(self):
            parser = argparse.ArgumentParser()
            parser.add_argument("--request-rate", type=parse_request_rate)
            self.assertEqual(parser.parse_args(["--request-rate", "1"]).request_rate, 1)
            self.assertEqual(parser.parse_args(["--request-rate", "inf"]).request_rate, math.inf)
            for invalid in ("nan", "-inf", "0", "-1"):
                with self.subTest(value=invalid), self.assertRaises(argparse.ArgumentTypeError):
                    parse_request_rate(invalid)

        def test_inf_trace_requires_zero_offsets_and_preserves_finite_trace(self):
            import tempfile

            trace = {"request_rate": "inf", "requests": [
                {"request_id": str(index), "prompt": "problem", "prompt_token_ids": [1, 2, 3],
                 "prompt_length": 3, "scheduled_offset_s": 0, "max_tokens": 4096}
                for index in range(2)]}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "trace.json"
                path.write_text(json.dumps(trace), encoding="utf-8")
                self.assertEqual(load_trace(path, math.inf), trace)
                trace["requests"][1]["scheduled_offset_s"] = 1
                path.write_text(json.dumps(trace), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "offset 0"):
                    load_trace(path, math.inf)
                trace["request_rate"] = 1
                path.write_text(json.dumps(trace), encoding="utf-8")
                self.assertEqual(load_trace(path, 1), trace)
                with self.assertRaisesRegex(ValueError, "differs"):
                    load_trace(path, math.inf)

        def test_variable_output_limits_and_sampling_defaults(self):
            import tempfile

            trace = {"request_rate": "inf", "ignore_eos": False,
                     "stop": ["Problem:", "Problem:\n"], "requests": [
                         {"request_id": str(index), "prompt": "problem", "prompt_token_ids": [1, 2, 3],
                          "prompt_length": 3, "scheduled_offset_s": 0, "max_tokens": limit}
                         for index, limit in enumerate((512, 1024, 4096))]}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "trace.json"
                path.write_text(json.dumps(trace), encoding="utf-8")
                loaded = load_trace(path, math.inf)
                self.assertEqual(loaded, trace)
                self.assertEqual(sampling_settings(loaded["requests"][0], loaded),
                                 {"ignore_eos": False, "stop": ["Problem:", "Problem:\n"]})
                override = {**loaded["requests"][0], "ignore_eos": True, "stop": []}
                self.assertEqual(sampling_settings(override, loaded), {"ignore_eos": True, "stop": []})
                for invalid in (0, -1, True, 1.5):
                    trace["requests"][0]["max_tokens"] = invalid
                    path.write_text(json.dumps(trace), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        load_trace(path, math.inf)

        def test_repetition_guard_threshold_and_normal_math(self):
            for unit in ("I hope it is correct. ", "The same answer follows. "):
                self.assertIsNone(repeated_output(unit * 7))
                self.assertGreaterEqual(repeated_output(unit * 8)["repeats"], 8)
                self.assertIsNotNone(repeated_output(unit * 9 + unit[:7]))
            formula = "x = x + 1\n"
            self.assertEqual(repeated_output(formula * 8)["kind"], "repeated_line")
            self.assertIsNone(repeated_output(formula * 7))
            self.assertIsNone(repeated_output("\n".join(f"x_{n} = {n} + 1 = {n + 1}" for n in range(100))))
            self.assertIsNone(repeated_output("0 " * 500))

        def test_saved_math_086_severe_latex_collapse(self):
            detected = detect_output_corruption(malformed_math_086)
            self.assertIsNone(repeated_output(malformed_math_086))
            self.assertEqual(detected["kind"], "malformed_latex")
            self.assertEqual(detected["unclosed_braces"], 6)
            self.assertEqual(detected["unclosed_environments"], ["split"])
            self.assertTrue(detected["terminal_empty_display"])
            # Neither a missing brace nor an empty display alone is a reason
            # to classify a finished response as severely corrupted.
            for text in (r"\frac{1}{2", "$$\n$$", "{{{{ ordinary prose",
                         r"\begin{split} x = 3", r"\{1,2,3,4\}"):
                self.assertIsNone(detect_output_corruption(text))

        def test_latex_guard_accepts_balanced_nested_and_multiple_displays(self):
            nested = "x"
            for denominator in range(1, 41):
                nested = "\\frac{" + nested + "}{" + str(denominator) + "}"
            valid = ("$$\\begin{split}\n" + nested + "\n\\end{split}$$\n"
                     "$$x^2 + 2x + 1 = (x + 1)^2$$\n"
                     "$$\\begin{aligned}a &= \\{1,2,3\\} \\\\ b &= 4\\end{aligned}$$")
            self.assertIsNone(detect_output_corruption(valid))
            self.assertIsNone(detect_output_corruption("{{{{\n\\begin{split}\n" + "}" * 4 + "\\end{split}"))

        def test_latex_guard_only_checks_finished_and_preserves_final_ids(self):
            import tempfile

            final_ids = list(range(403)) + [128009]
            class Content:
                async def iter_any(self):
                    # The same incomplete LaTeX in an interim preview must not
                    # trigger the final-output structural guard.
                    yield json.dumps({"request_id": "a", "prompt_tokens": 3,
                                      "output_tokens": 32, "last_token_id": 31,
                                      "preview_text": malformed_math_086,
                                      "finished": False}).encode() + b"\0"
                    yield json.dumps({"request_id": "a", "prompt_tokens": 3,
                                      "output_tokens": 404, "last_token_id": 128009,
                                      "finished": True, "text": malformed_math_086,
                                      "finish_reason": "stop", "output_token_ids": final_ids,
                                      "stop_token_ids": [128001, 128009],
                                      "effective_eos_token_ids": [128001, 128009]}).encode() + b"\0"
            class Response:
                status = 200
                content = Content()
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *args):
                    return False
            class Session:
                def post(self, url, json):
                    return Response()
            row = {"request_id": "a", "prompt": "problem", "prompt_token_ids": [1, 2, 3],
                   "prompt_length": 3, "ignore_eos": False, "stop": []}
            with tempfile.TemporaryDirectory() as temporary:
                monitor = OutputMonitor(Path(temporary) / "output-monitor.txt")
                try:
                    record, error = asyncio.run(stream_request(Session(), "http://unused", row,
                                                               4096, 0, monitor=monitor))
                    self.assertEqual(monitor.checks, 2)
                    self.assertEqual(record["output_tokens"], 404)
                    self.assertEqual(record["output_abort"]["kind"], "malformed_latex")
                    self.assertEqual(record["completion_text"], malformed_math_086)
                    self.assertEqual(record["output_token_ids"], final_ids)
                    self.assertTrue(record["output_token_ids_complete"])
                    self.assertIn("6 unclosed braces", error)
                finally:
                    monitor.close()

        def test_token_evidence_retains_aborted_prefix_and_final_eos(self):
            observed = StreamObservation("a", 3, 100, sent=10, require_exact_output=False)
            for count, token in enumerate((42, 128001, 19), 1):
                observed.observe({"request_id": "a", "prompt_tokens": 3,
                                  "output_tokens": count, "last_token_id": token,
                                  "finished": False}, 10 + count)
            partial = observed.partial_metrics()
            self.assertEqual(partial["output_token_ids"], [42, 128001, 19])
            self.assertFalse(partial["output_token_ids_complete"])
            self.assertNotIn("ttft_ms", partial)
            observed.observe({"request_id": "a", "prompt_tokens": 3,
                              "output_tokens": 3, "last_token_id": 19, "finished": True,
                              "text": "answer", "finish_reason": "stop",
                              "output_token_ids": [42, 128001, 19],
                              "stop_token_ids": [128001, 128009],
                              "effective_eos_token_ids": [128001, 128009]}, 14)
            final = observed.metrics()
            self.assertTrue(final["output_token_ids_complete"])
            self.assertEqual(len(final["token_id_observations"]), 3)
            self.assertEqual(final["effective_eos_token_ids"], [128001, 128009])

        def test_guard_off_does_not_abort_repetitive_completion(self):
            class Content:
                async def iter_any(self):
                    yield json.dumps({"request_id": "a", "prompt_tokens": 3,
                                      "output_tokens": 32, "finished": True,
                                      "text": "I hope it is correct. " * 8,
                                      "finish_reason": "length"}).encode() + b"\0"
            class Response:
                status = 200
                content = Content()
                async def __aenter__(self):
                    return self
                async def __aexit__(self, *args):
                    return False
            class Session:
                def post(self, url, json):
                    return Response()
            row = {"request_id": "a", "prompt": "problem", "prompt_token_ids": [1, 2, 3],
                   "prompt_length": 3, "ignore_eos": False, "stop": []}
            record, error = asyncio.run(stream_request(Session(), "http://unused", row, 32, 0))
            self.assertIsNone(error)
            self.assertEqual(record["output_tokens"], 32)
            self.assertNotIn("output_abort", record)
            self.assertNotIn("output_token_ids", record)

        def test_guard_cancels_other_streams_and_preserves_readable_partials(self):
            import tempfile
            from types import SimpleNamespace
            from unittest.mock import patch

            async def exercise(directory, malformed_final=False):
                active_started = asyncio.Event()
                closed = []
                repetition = "I hope it is correct. " * 8

                class Response:
                    status = 200
                    def __init__(self, payload):
                        self.payload = payload
                        self.content = self
                    async def __aenter__(self):
                        return self
                    async def __aexit__(self, *args):
                        closed.append(self.payload["request_id"])
                        return False
                    async def iter_any(self):
                        request_id = self.payload["request_id"]
                        if request_id == "warmup":
                            # Warmup intentionally bypasses the output guard.
                            yield json.dumps({"request_id": "warmup", "prompt_tokens": 3,
                                              "output_tokens": 16, "finished": True,
                                              "preview_text": repetition, "text": repetition,
                                              "finish_reason": "length"}).encode() + b"\0"
                            return
                        if request_id == "active":
                            active_started.set()
                            yield json.dumps({"request_id": request_id, "prompt_tokens": 3,
                                              "output_tokens": 1, "last_token_id": 99,
                                              "preview_text": "Computing the answer.",
                                              "finished": False}).encode() + b"\0"
                            await asyncio.Event().wait()
                        else:
                            await active_started.wait()
                            limit = 404 if malformed_final else 32
                            for count in range(1, limit + 1):
                                frame = {"request_id": request_id, "prompt_tokens": 3,
                                         "output_tokens": count,
                                         "last_token_id": 128001 if count == 2 else count,
                                         "finished": False}
                                if malformed_final and count % 32 == 0:
                                    frame["preview_text"] = "Working through the expression."
                                elif not malformed_final and count == 32:
                                    frame["preview_text"] = repetition
                                if malformed_final and count == limit:
                                    frame.update(finished=True, text=malformed_math_086,
                                                 last_token_id=128009, finish_reason="stop",
                                                 preview_text=malformed_math_086,
                                                 output_token_ids=[128001 if index == 2 else index
                                                                   for index in range(1, limit)] + [128009],
                                                 stop_token_ids=[128001, 128009],
                                                 effective_eos_token_ids=[128001, 128009])
                                await asyncio.sleep(0)
                                yield json.dumps(frame).encode() + b"\0"

                class Session:
                    def __init__(self, **kwargs):
                        pass
                    async def __aenter__(self):
                        return self
                    async def __aexit__(self, *args):
                        return False
                    def post(self, url, json):
                        return Response(json)

                async def endpoint(session, url, method="GET"):
                    return {"status": "ready" if url.endswith("/health") else "reset"}

                class Log:
                    lines = []
                    def write(self, line):
                        self.lines.append(line)

                trace = {"ignore_eos": False, "stop": [], "requests": [
                    {"request_id": request_id, "prompt": "problem", "prompt_token_ids": [1, 2, 3],
                     "prompt_length": 3, "max_tokens": 4096,
                     "scheduled_offset_s": 60 if request_id == "deferred" else 0}
                    for request_id in ("bad", "active", "deferred")]}
                args = SimpleNamespace(url="http://unused", inputs=directory / "trace.json",
                                       output_dir=directory, request_rate=1,
                                       timeout_seconds=10, monitor_output=True)
                log = Log()
                fake_aiohttp = SimpleNamespace(ClientSession=Session,
                                               TCPConnector=lambda **kwargs: None,
                                               ClientTimeout=lambda **kwargs: None)
                with patch.dict(sys.modules, aiohttp=fake_aiohttp),\
                        patch.dict(globals(), endpoint_json=endpoint):
                    result = await asyncio.wait_for(run_http_client(args, trace, log), timeout=2)
                return result, trace, log.lines, closed

            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                result, trace, lines, closed = asyncio.run(exercise(directory))
                self.assertEqual(result["status"], "invalid_output")
                self.assertEqual(result["submitted"], 2)
                records = {row["request_id"]: row for row in result["requests"]}
                self.assertEqual(records["bad"]["status"], "invalid_output")
                self.assertEqual(records["bad"]["output_token_ids"][1], 128001)
                self.assertEqual(records["bad"]["output_tokens"], 32)
                self.assertEqual(records["active"]["status"], "cancelled")
                self.assertEqual(records["active"]["output_token_ids"], [99])
                self.assertEqual(records["active"]["partial_text"], "Computing the answer.")
                self.assertCountEqual(closed, ["warmup", "bad", "active"])
                self.assertTrue(any(line.startswith("OUTPUT_ABORT") for line in lines))
                monitor_text = (directory / "output-monitor.txt").read_text(encoding="utf-8")
                self.assertIn("OUTPUT_ABORT", monitor_text)
                self.assertNotIn("id=warmup", monitor_text)
                responses = directory / "responses.txt"
                write_responses(responses, trace, result)
                response_text = responses.read_text(encoding="utf-8")
                self.assertIn("Observed partial output tail", response_text)
                self.assertIn("128001", response_text)
                self.assertIn("Computing the answer.", response_text)

            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                result, trace, lines, closed = asyncio.run(exercise(directory, malformed_final=True))
                self.assertEqual(result["status"], "invalid_output")
                records = {row["request_id"]: row for row in result["requests"]}
                self.assertEqual(records["bad"]["status"], "invalid_output")
                self.assertEqual(records["bad"]["output_abort"]["kind"], "malformed_latex")
                self.assertEqual(records["bad"]["output_tokens"], 404)
                self.assertEqual(records["bad"]["completion_text"], malformed_math_086)
                self.assertEqual(records["bad"]["output_token_ids"][-1], 128009)
                self.assertTrue(records["bad"]["output_token_ids_complete"])
                self.assertEqual(records["active"]["status"], "cancelled")
                self.assertCountEqual(closed, ["warmup", "bad", "active"])
                self.assertTrue(any("reason=malformed_latex" in line for line in lines))
                responses = directory / "responses.txt"
                write_responses(responses, trace, result)
                self.assertIn(malformed_math_086, responses.read_text(encoding="utf-8"))

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests))
    return 0 if result.wasSuccessful() else 1

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def resolve_sampling_policy(model, requested_policy, prompt_style=None):
    resolved_policy = requested_policy
    reason = "Explicit sampling policy requested"
    if requested_policy == "auto":
        if prompt_style == "minerva-4shot-chat":
            resolved_policy = "checkpoint"
            reason = "Resolved native four-shot chat uses the checkpoint generation policy"
        elif prompt_style in {"minerva-4shot", "raw-fixed"}:
            resolved_policy = "greedy"
            reason = "Resolved raw prompt style preserves the original greedy policy"
        else:
            raise ValueError("auto sampling policy requires a resolved MATH prompt style")
    if resolved_policy == "greedy":
        temperature, top_p = 0.0, 1.0
        metadata = {"requested": requested_policy, "resolved": "greedy",
                    "source_file": None, "source_sha256": None, "raw_fields": None}
    elif resolved_policy == "checkpoint":
        source = Path(model) / "generation_config.json"
        contents = source.read_bytes()
        configuration = json.loads(contents)
        if not isinstance(configuration, dict) or configuration.get("do_sample") is not True:
            raise ValueError("checkpoint sampling policy requires generation_config.json do_sample=true")
        for name in ("temperature", "top_p"):
            value = configuration.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"checkpoint generation_config.json must specify numeric {name}")
        temperature, top_p = float(configuration["temperature"]), float(configuration["top_p"])
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("checkpoint temperature must be finite and positive")
        if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
            raise ValueError("checkpoint top_p must be finite and in (0, 1]")
        metadata = {"requested": requested_policy, "resolved": "checkpoint",
                    "source_file": str(source.resolve()),
                    "source_sha256": hashlib.sha256(contents).hexdigest(),
                    "raw_fields": configuration}
    else:
        raise ValueError(f"Unsupported sampling policy: {requested_policy}")
    metadata["reason"] = reason
    metadata["applied_fields"] = {"do_sample": temperature > 0.0,
                                  "temperature": temperature, "top_p": top_p}
    return temperature, top_p, metadata


def resolve_frequency_penalty(requested_value, prompt_style):
    if requested_value is None:
        penalty = 0.0
        reason = "Neutral frequency penalty; no automatic anti-repetition value selected"
    else:
        if isinstance(requested_value, bool):
            raise ValueError("frequency penalty must be numeric, finite and in [-2, 2]")
        penalty = float(requested_value)
        reason = "Explicit frequency penalty requested"
    if not math.isfinite(penalty) or not -2.0 <= penalty <= 2.0:
        raise ValueError("frequency penalty must be finite and in [-2, 2]")
    return penalty, {"requested": requested_value, "resolved": penalty, "reason": reason}


def resolve_ngram_control(requested_size, requested_repeats):
    if (isinstance(requested_size, bool) or not isinstance(requested_size, int)
            or requested_size < 0):
        raise ValueError("no-repeat ngram size must be a nonnegative integer")
    if (isinstance(requested_repeats, bool) or not isinstance(requested_repeats, int)
            or requested_repeats < 1):
        raise ValueError("max ngram repeats must be a positive integer")
    source = ROOT / "benchmarks/repetition_control.py"
    implementation = None
    if requested_size:
        implementation = {"source_file": str(source.resolve()),
                          "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    return {
        "no_repeat_ngram_size": requested_size,
        "max_ngram_repeats": requested_repeats,
        "selection": {"requested": {"no_repeat_ngram_size": requested_size,
                                     "max_ngram_repeats": requested_repeats},
                      "resolved": {"no_repeat_ngram_size": requested_size,
                                    "max_ngram_repeats": requested_repeats},
                      "reason": "Explicit ngram control" if requested_size else
                                "Disabled; no automatic ngram control selected"},
        "scope": "generated token IDs only",
        "eos_exempt": True,
        "implementation": implementation,
    }


def prepare_trace(args):
    repetition_penalty = float(getattr(args, "repetition_penalty", 1.0))
    if not math.isfinite(repetition_penalty) or repetition_penalty <= 0:
        raise ValueError("repetition penalty must be finite and positive")
    source = json.loads(args.inputs.read_text())
    rows = source["requests"]
    if not 1 <= args.num_requests <= len(rows):
        raise ValueError("Request count must fit the existing MATH input set")
    source_model = Path(source["model_path"]).resolve()
    if source_model != DEFAULT_MODEL.resolve():
        raise ValueError("The historical source trace has an unexpected model checkpoint")
    rng = random.Random(args.seed)
    selected = rng.sample(rows, args.num_requests)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    prompt_style, selection_reason = resolve_math_prompt_style(args.prompt_style, tokenizer)
    frequency_penalty, frequency_selection = resolve_frequency_penalty(
        getattr(args, "frequency_penalty", None), prompt_style)
    generation_options = resolve_ngram_control(
        getattr(args, "no_repeat_ngram_size", 0), getattr(args, "max_ngram_repeats", 2))
    temperature, top_p, sampling_policy = resolve_sampling_policy(
        args.model, getattr(args, "sampling_policy", "auto"), prompt_style)
    requested_monitor = getattr(args, "monitor_output", None)
    if requested_monitor is not None and not isinstance(requested_monitor, bool):
        raise ValueError("monitor_output must be True, False or None")
    args.monitor_output = prompt_style != "raw-fixed" if requested_monitor is None else requested_monitor
    monitor_reason = ("Enabled for natural MATH generation" if args.monitor_output else
                      "Disabled for fixed-length raw workload") if requested_monitor is None else "Explicit monitor choice"
    eos_token_ids = configured_eos_token_ids(args.model, tokenizer.eos_token_id)
    requests = []
    offset = 0.0
    for index, row in enumerate(selected):
        if index and not math.isinf(args.request_rate):
            offset += rng.expovariate(args.request_rate)
        ids = row["prompt_token_ids"]
        if row["prompt_length"] != len(ids):
            raise ValueError("Source prompt token count mismatch")
        if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
            raise ValueError("Source prompt hash mismatch")
        problem = row["prompt"]
        is_chat = prompt_style == "minerva-4shot-chat"
        prompt = (problem if prompt_style == "raw-fixed" else
                  make_math_prompt(problem, prompt_style=prompt_style, tokenizer=tokenizer))
        ids = (tokenizer.encode(prompt, add_special_tokens=False) if is_chat else
               tokenizer.encode(prompt))
        if is_chat and (not ids or ids[0] != tokenizer.bos_token_id or
                        ids.count(tokenizer.bos_token_id) != 1):
            raise ValueError("Native chat prompt must encode exactly one BOS token at the start")
        if len(ids) + args.max_output_len > args.max_model_len:
            raise ValueError("Evaluation prompt plus maximum output exceeds the configured context")
        requests.append({
            "request_id": f"math-{index:03d}", "source_id": row["source_id"],
            "subject": row["subject"], "problem": problem, "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "source_prompt_sha256": row["prompt_sha256"],
            "prompt_token_ids": ids, "prompt_length": len(ids),
            "scheduled_offset_s": offset, "max_tokens": args.max_output_len,
            "ignore_eos": prompt_style == "raw-fixed",
            "stop": [] if prompt_style == "raw-fixed" else ["Problem:", "Problem:\n"],
        })
        if is_chat:
            requests[-1]["prompt_messages"] = make_math_chat_messages(problem)
        if repetition_penalty != 1.0:
            requests[-1]["repetition_penalty"] = repetition_penalty
        if frequency_penalty != 0.0:
            requests[-1]["frequency_penalty"] = frequency_penalty
        if generation_options["no_repeat_ngram_size"]:
            requests[-1]["no_repeat_ngram_size"] = generation_options["no_repeat_ngram_size"]
            requests[-1]["max_ngram_repeats"] = generation_options["max_ngram_repeats"]
        if temperature != 0.0:
            requests[-1]["temperature"] = temperature
        if top_p != 1.0:
            requests[-1]["top_p"] = top_p
    lengths = [len(row["prompt_token_ids"]) for row in requests]
    protocol = ({"style": "raw-fixed", "shots": 0} if prompt_style == "raw-fixed" else
                prompt_metadata(prompt_style=prompt_style, tokenizer=tokenizer, model_path=args.model))
    protocol["selection"] = {"requested": args.prompt_style, "resolved": prompt_style,
                             "reason": selection_reason}
    if sampling_policy["resolved"] == "checkpoint" and "sampling" in protocol:
        protocol["sampling"].update(temperature=temperature, top_p=top_p)
    if prompt_style == "minerva-4shot":
        protocol["model_type"] = "Causal language model, completion prompt without chat template"
    elif prompt_style == "minerva-4shot-chat":
        protocol["encoding"]["verified_requests"] = len(requests)
        protocol["encoding"]["verified_bos_count_per_request"] = 1
        protocol["encoding"]["encoded_special_token_counts"] = {
            token: {"token_id": token_id,
                    "minimum": min(request["prompt_token_ids"].count(token_id) for request in requests),
                    "maximum": max(request["prompt_token_ids"].count(token_id) for request in requests)}
            for token, token_id in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)
            if any(token_id in request["prompt_token_ids"] for request in requests)
        }
    download_source = args.model / "download_source.json"
    model_verification = args.model / "provenance-verification.json"
    trace = {
        "model": str(args.model),
        "model_download_source": json.loads(download_source.read_text()) if download_source.exists() else None,
        "model_provenance_verification": json.loads(model_verification.read_text()) if model_verification.exists() else None,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "effective_eos_token_ids": eos_token_ids,
        "output_monitoring": {"enabled": args.monitor_output,
                              "requested": requested_monitor, "reason": monitor_reason,
                              "interval_tokens": 32, "repeat_threshold": 8,
                              "on_repetition": "abort the entire run",
                              "on_corrupted_final_output": "abort the entire run"},
        "dataset": source["dataset"]["repo"],
        "split": "test", "selection": "seeded sample without replacement from existing 1000 MATH requests",
        "subjects": sorted({row["subject"] for row in requests}),
        "source_file": str(args.inputs),
        "source_model": str(source_model),
        "source_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        "seed": args.seed, "request_rate": "inf" if math.isinf(args.request_rate) else args.request_rate,
        "arrival_process": "simultaneous" if math.isinf(args.request_rate) else "poisson",
        "num_requests": len(requests), "max_output_len": args.max_output_len,
        "prompt_protocol": protocol,
        "sampling_policy": sampling_policy,
        "frequency_penalty_selection": frequency_selection,
        "generation_options": generation_options,
        "sampling": {"policy": sampling_policy["resolved"], "temperature": temperature,
                     "top_p": top_p, "n": 1,
                     "repetition_penalty": repetition_penalty,
                     "frequency_penalty": frequency_penalty,
                     "max_tokens": args.max_output_len,
                     "ignore_eos": prompt_style == "raw-fixed",
                     "stop_token_ids": [] if prompt_style == "raw-fixed" else eos_token_ids,
                     "stop": [] if prompt_style == "raw-fixed" else ["Problem:", "Problem:\n"]},
        "scheduled_submission_seconds": offset,
        "input_length": {"min": min(lengths), "max": max(lengths), "mean": statistics.mean(lengths)},
        "engine": {
            "max_model_len": args.max_model_len, "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": 8192, "max_paddings": 4096,
            "gpu_memory_utilization": 0.9, "swap_space": 0, "enforce_eager": True,
            "weights_dtype": "float16", "tensor_parallel_size": 1,
        },
        "requests": requests,
    }
    return attach_ground_truth(trace)


def stop(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_ready(process, url, timeout=240):
    deadline = time.monotonic() + timeout
    last_notice = time.monotonic()
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited during model loading, code {process.returncode}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        if time.monotonic() - last_notice >= 30:
            print("LOADING server, waiting for model initialization", flush=True)
            last_notice = time.monotonic()
        time.sleep(0.5)
    raise RuntimeError("Server readiness timeout")


def failure_status(message):
    lowered = message.lower()
    if any(fragment in lowered for fragment in (
        "out of memory", "outofmemoryerror", "no available memory for the cache",
        "larger than the maximum number of tokens that can be stored in kv cache",
    )):
        return "oom"
    return "failed"


def measure_http_trace(args, trace, backend, stdout_stream):
    """Run the HTTP client here; only the model server needs a subprocess."""
    reporter = Reporter(args.output_dir / "client.txt", f"{backend}: ", stdout_stream)
    try:
        result = asyncio.run(run_http_client(args, trace, reporter))
        write_responses(args.output_dir / "responses.txt", trace, result)
        temporary = args.output_dir / "summary.json.tmp"
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output_dir / "summary.json")
        reporter.write(f"Readable results: {args.output_dir.resolve() / 'client.txt'}")
        reporter.write(f"Readable responses: {args.output_dir.resolve() / 'responses.txt'}")
        return result
    except Exception as error:
        reporter.write(f"CLIENT FAILED {type(error).__name__}: {error}")
        raise
    finally:
        reporter.close()


def run_backend(args, backend, folder, trace_path):
    folder.mkdir()
    repo = FP16_REPO if backend == "fp16" else ROOT
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONPATH": str(repo),
                "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    command = [str(args.python), "-u", str(ROOT / "benchmarks/online_benchmark_server.py"),
               "--backend", backend, "--model", str(args.model), "--port", str(port),
               "--output-dir", str(folder), "--max-model-len", str(args.max_model_len),
               "--max-num-seqs", str(args.max_num_seqs), "--max-num-batched-tokens", "8192",
               "--max-paddings", "4096", "--gpu-memory-utilization", "0.9",
               "--seed", str(args.seed)]
    if args.monitor_output:
        command.append("--monitor-output")
    save(folder / "command.json", {"server": command,
                                  "client": {"execution": "runner process",
                                             "source_file": str(Path(__file__).resolve()),
                                             "trace": str(trace_path)}})
    server = None
    print(f"START {backend}, GPU {args.gpu}, HTTP {url}", flush=True)
    with (folder / "server.txt").open("w") as server_log, (folder / "client-stdout.txt").open("w") as client_log:
        try:
            server = subprocess.Popen(command, cwd=repo, env=env, stdout=server_log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            wait_ready(server, url)
            client_args = argparse.Namespace(
                url=url, inputs=trace_path, output_dir=folder,
                request_rate=args.request_rate,
                timeout_seconds=getattr(args, "timeout_seconds", 3 * 3600),
                monitor_output=args.monitor_output)
            trace = load_trace(trace_path, args.request_rate)
            summary = measure_http_trace(client_args, trace, backend, client_log)
            summary_path = folder / "summary.json"
            if summary.get("status") == "completed":
                measured = summary.get("requests", [])
                token_total = sum(row.get("output_tokens", 0) for row in measured)
                valid_lengths = all(0 < row.get("output_tokens", 0) <= args.max_output_len for row in measured)
                if args.prompt_style == "raw-fixed":
                    valid_lengths &= token_total == args.num_requests * args.max_output_len
                if (summary["completed"] != args.num_requests or len(measured) != args.num_requests
                        or summary["generated_tokens"] != token_total or not valid_lengths):
                    raise RuntimeError("Measured requests or generated token total mismatch")
            else:
                engine_error = (summary.get("server_metrics") or {}).get("engine_error") or {}
                memory_failure = str(engine_error.get("status", "")).lower() == "oom"
                memory_failure |= any(str(request.get("status", "")).lower() == "oom"
                                      for request in summary.get("requests", []))
                if not memory_failure:
                    server_log.flush()
                    memory_failure = failure_status((folder / "server.txt").read_text(errors="replace")) == "oom"
                if memory_failure:
                    summary["status"] = "oom"
            summary["backend"] = backend
            save(summary_path, summary)
            return summary
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            server_log.flush()
            logs = (folder / "server.txt").read_text(errors="replace")
            message = f"{exc}\n{logs[-6000:]}"
            summary = {"backend": backend, "status": failure_status(message), "error": str(exc),
                       "completed": 0, "failed": args.num_requests}
            save(folder / "summary.json", summary)
            print(f"FAILED {backend}: {summary['status']}, {exc}", flush=True)
            return summary
        finally:
            stop(server)


def value(result, key, stat=None):
    if result.get("status") != "completed":
        return "OOM" if result.get("status") == "oom" else result.get("status", "not run")
    found = result
    for part in key.split("."):
        found = found.get(part) if isinstance(found, dict) else None
    if stat and isinstance(found, dict):
        found = found.get(stat)
    if isinstance(found, (int, float)) and math.isfinite(found):
        return f"{found:.3f}"
    return "N/A"


def append_report(args, trace, folder, results):
    def pairs(key, stat=None):
        return " / ".join(value(results.get(b, {}), key, stat) for b in ("fp16", "diffkv"))
    text = f"\n\n## Online MATH, {folder.name}\n\n"
    text += "Measured with a custom HTTP client and streaming server adapter around each backend's AsyncLLMEngine. This run does not invoke the official vllm bench serve command.\n\n"
    arrival = "all requests dispatched concurrently with scheduled_offset_s=0" if math.isinf(args.request_rate) else "Poisson arrivals"
    prompt_style = trace["prompt_protocol"]["style"]
    output_mode = "fixed OSL" if prompt_style == "raw-fixed" else "maximum output length"
    text += f"{args.model.name}, MATH test {args.num_requests} questions, {args.request_rate:g} req/s, {arrival}, {output_mode} {args.max_output_len}, GPU {args.gpu}, FP16 / DiffKV.\n\n"
    text += f"Same prompt token IDs and scheduled arrival times in both paths. Input lengths: min {trace['input_length']['min']}, mean {trace['input_length']['mean']:.3f}, max {trace['input_length']['max']}. Selected subjects: {', '.join(trace['subjects'])}. The sample is drawn without replacement from the previously used 1000 MATH inputs.\n\n"
    text += f"Engine: FP16 weights, max_model_len={args.max_model_len}, max_num_seqs={args.max_num_seqs}, max_num_batched_tokens=8192, max_paddings=4096, GPU memory utilization=0.9, swap=0, eager. FP16 KV: K16V16. DiffKV: K8V4/K4V2, thresholds [0.02, 1.0], KV buffer 64. Sampling policy={trace['sampling'].get('policy', 'greedy')}, temperature={trace['sampling']['temperature']}, top_p={trace['sampling'].get('top_p', 1.0)}, repetition_penalty={trace['sampling'].get('repetition_penalty', 1.0)}, frequency_penalty={trace['sampling'].get('frequency_penalty', 0.0)}, ignore_eos={trace['sampling']['ignore_eos']}. One 16-token warmup request excluded, one measured run, all submitted requests awaited.\n\n"
    text += f"Checkpoint EOS tokens are honored in both paths: stop_token_ids={trace['sampling']['stop_token_ids']}.\n\n"
    if (folder / "diffkv" / "source.patch").exists():
        text += "DiffKV includes the prompt-cache importance-score indexing fix; the patched source and loaded CUDA binary were verified.\n\n"
    if args.monitor_output:
        text += "Partial output tails were checked every 32 generated tokens in both paths. Eight consecutive repeats or severe final LaTeX corruption trigger an abort of the whole run; aborted runs are not added as completed comparisons.\n\n"
    generation_options = trace.get("generation_options", {})
    ngram_size = generation_options.get("no_repeat_ngram_size", 0)
    max_repeats = generation_options.get("max_ngram_repeats", 2)
    text += f"Generation control: no_repeat_ngram_size={ngram_size}, max_ngram_repeats={max_repeats}.\n\n"
    if ngram_size:
        text += (f"The same runtime logits processor in both paths blocks a token that would complete "
                 f"more than {max_repeats} occurrences of the same {ngram_size}-token ngram. "
                 "It examines generated token IDs only and exempts checkpoint EOS tokens. "
                 "Both original engines are used without source changes for this control.\n\n")
    text += "TTFT and E2E Latency include HTTP and queuing. TPOT = (last token arrival - first token arrival) / (output tokens - 1). Prefill/Decode Throughput use measured engine phase time, exclude padding and count recompute separately. Each cell is **FP16 / DiffKV**.\n\n"
    text += "| Case (ISL, OSL, Rate) | TTFT (ms) | TPOT (ms/token) | Prefill Throughput (tok/s) | Decode Throughput (tok/s) | E2E Latency (s) | Request Throughput (req/s) |\n|---|---:|---:|---:|---:|---:|---:|\n"
    text += f"| (MATH, {args.max_output_len}, {args.request_rate:g}) | {pairs('ttft_ms','mean')} | {pairs('tpot_ms','mean')} | {pairs('server_metrics.prefill_tokens_s')} | {pairs('server_metrics.decode_tokens_s')} | {pairs('e2e_s','mean')} | {pairs('request_throughput')} |\n\n"
    text += "| Percentile | TTFT (ms) | TPOT (ms/token) | E2E Latency (s) |\n|---|---:|---:|---:|\n"
    for stat in ("p95", "p99"):
        text += f"| {stat.upper()} | {pairs('ttft_ms',stat)} | {pairs('tpot_ms',stat)} | {pairs('e2e_s',stat)} |\n"
    text += "\n| Path | Status | Completed | Failed | Generated Tokens | Time (s) |\n|---|---|---:|---:|---:|---:|\n"
    for backend in ("fp16", "diffkv"):
        row = results.get(backend, {})
        status = "OOM" if row.get("status") == "oom" else row.get("status", "not run")
        text += f"| {backend} | {status} | {row.get('completed',0)} | {row.get('failed',0)} | {row.get('generated_tokens',0)} | {value(row,'duration_s')} |\n"
    text += "\n| Path | Correct | Accuracy (%) | Correctness Status |\n|---|---:|---:|---|\n"
    for backend in ("fp16", "diffkv"):
        correctness = results.get(backend, {}).get("correctness", {})
        accuracy = correctness.get("accuracy_percent")
        status = correctness.get("status", "not measured")
        score = f"{accuracy:.2f}" if isinstance(accuracy, (int, float)) else "N/A"
        correct = f"{correctness['correct']} / {correctness['total']}" if correctness else "N/A"
        text += f"| {backend} | {correct} | {score} | {status} |\n"
    text += "\nAccuracy prioritizes explicit final answers over intermediate boxed values, with repository parser fallback. Reference and prediction use the same strip_string(skip_unit=True) normalization and the repository numeric/symbolic equivalence comparator. Grading and answer logging run after the measured requests finish and are excluded from performance time.\n\n"
    if prompt_style == "minerva-4shot":
        text += f"Prompts use the exact repository Minerva MATH 4-shot examples and Problem:/Solution: format. Generation stops at EOS or the repository stop strings Problem: and Problem:\\n; {args.max_output_len} tokens is a maximum, so actual output lengths vary. The repository's separate accuracy entrypoint uses a 1024-token cap; this serving run retains the requested {args.max_output_len}-token cap.\n"
    elif prompt_style == "minerva-4shot-chat":
        text += f"Prompts use the four unchanged original Minerva examples as user/assistant exchanges, followed by the selected problem: nine messages total. The checkpoint native chat template supplies EOS turn boundaries and the assistant generation header; no instructions are added. Generation honors checkpoint EOS and the original Problem: and Problem:\\n stop strings, with the same {args.max_output_len}-token maximum. Full message and template metadata are recorded in requests.json.\n"
    else:
        text += f"Prompts contain only the original problem text. Generation is forced to {args.max_output_len} tokens with EOS ignored.\n"
    text += "\n| Engine Metric | FP16 / DiffKV |\n|---|---:|\n"
    for label, key in (("Mean Decode B", "decode_batch_mean"), ("Max Decode B", "decode_batch_max"),
                       ("Max Waiting Requests", "max_waiting"), ("Max GPU KV Usage (%)", "max_kv_usage_percent"),
                       ("Recompute Count", "recompute_count"), ("Recompute Prefill Tokens", "recompute_prefill_tokens")):
        text += f"| {label} | {pairs('server_metrics.'+key)} |\n"
    text += "\nAverage Latency (s/token), following the public client formula mean(E2E / (input + output tokens)): " + pairs("avg_latency_per_token_s") + ".\n\n"
    for backend in ("fp16", "diffkv"):
        for name in ("server.txt", "client.txt", "responses.txt", "correctness.txt", "source.patch"):
            log = folder / backend / name
            if log.exists():
                target = Path(os.path.relpath(log, args.report.parent)).as_posix()
                text += f"- [{backend} {name}]({target})\n"
    control_source = (generation_options.get("implementation") or {}).get("source_file")
    if control_source:
        target = Path(os.path.relpath(control_source, args.report.parent)).as_posix()
        text += f"- [Shared repetition-control source]({target})\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream, fcntl.LOCK_UN)
    comparison_text = text
    for backend in ("fp16", "diffkv"):
        for name in ("server.txt", "client.txt", "responses.txt", "correctness.txt", "source.patch"):
            log = folder / backend / name
            report_target = Path(os.path.relpath(log, args.report.parent)).as_posix()
            comparison_target = Path(os.path.relpath(log, folder)).as_posix()
            comparison_text = comparison_text.replace(f"]({report_target})", f"]({comparison_target})")
    if control_source:
        report_target = Path(os.path.relpath(control_source, args.report.parent)).as_posix()
        comparison_target = Path(os.path.relpath(control_source, folder)).as_posix()
        comparison_text = comparison_text.replace(f"]({report_target})", f"]({comparison_target})")
    (folder / "comparison.md").write_text(comparison_text)
    print(text, flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Local checkpoint used by both FP16 and DiffKV")
    parser.add_argument("--inputs", type=Path, default=ROOT / "logs/performance/figure17-inputs.json")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--request-rate", type=float, default=math.inf)
    parser.add_argument("--timeout-seconds", type=float, default=3 * 3600,
                        help="HTTP request timeout, including queueing and generation")
    parser.add_argument("--prompt-style", choices=("auto", "minerva-4shot", "minerva-4shot-chat", "raw-fixed"),
                        default="auto", help="auto uses native chat when the checkpoint tokenizer has a chat template")
    parser.add_argument("--sampling-policy", choices=("auto", "greedy", "checkpoint"), default="auto",
                        help="auto uses checkpoint generation policy for native chat and greedy for raw prompt styles")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Standard repetition penalty applied identically to both paths; 1.0 disables it")
    parser.add_argument("--frequency-penalty", type=float, default=None,
                        help="Defaults to neutral 0.0; a nonzero frequency penalty must be selected explicitly")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0,
                        help="Generated-token ngram length to control identically in both paths; 0 disables it")
    parser.add_argument("--max-ngram-repeats", type=int, default=2,
                        help="Maximum occurrences of the same generated-token ngram when control is enabled")
    parser.add_argument("--max-output-len", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "evaluation.md")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="Run CPU protocol, prompt, and MATH scoring checks without loading a model")
    parser.add_argument("--monitor-output", action=argparse.BooleanOptionalAction, default=None,
                        help="Abort on sustained repetition or severe final LaTeX corruption; defaults on for natural MATH, off for raw-fixed")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        if protocol_self_test():
            return 1
        print(json.dumps(math_self_test(), indent=2))
        return 0
    if math.isnan(args.request_rate) or args.request_rate <= 0:
        parser.error("request rate must be positive or inf")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("timeout must be positive and finite")
    if not math.isfinite(args.repetition_penalty) or args.repetition_penalty <= 0:
        parser.error("repetition penalty must be finite and positive")
    if args.frequency_penalty is not None and (
            not math.isfinite(args.frequency_penalty) or not -2.0 <= args.frequency_penalty <= 2.0):
        parser.error("frequency penalty must be finite and in [-2, 2]")
    if args.no_repeat_ngram_size < 0 or args.max_ngram_repeats < 1:
        parser.error("no-repeat ngram size must be nonnegative and max ngram repeats must be positive")
    if args.max_output_len <= 0 or args.max_model_len <= 0:
        parser.error("output and context lengths must be positive")
    args.inputs = args.inputs.resolve()
    args.model = args.model.resolve()
    if not args.model.is_dir():
        parser.error("model must be an existing local checkpoint directory")
    args.report = args.report.resolve()
    args.python = args.python.absolute()
    trace = prepare_trace(args)
    if args.dry_run:
        print(json.dumps({k:v for k,v in trace.items() if k != "requests"}, indent=2))
        return 0
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d-%H%M%S-%f")
    folder = (args.output_dir or ROOT / "evaluation-logs" / f"online-math-{stamp}").resolve()
    folder.mkdir(parents=True, exist_ok=False)
    trace_path = folder / "requests.json"
    save(trace_path, trace)
    save(folder / "run.json", {"status": "running", "trace": str(trace_path),
                             "gpu": args.gpu, "command": shlex.join(sys.argv)})
    results = {}
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    try:
        for backend in ("fp16", "diffkv"):
            results[backend] = run_backend(args, backend, folder / backend, trace_path)
            if results[backend].get("status") == "invalid_output":
                save(folder / "comparison.json", results)
                save(folder / "run.json", {"status": "invalid_output", "backend": backend,
                                         "results": results})
                print(f"STOPPED {backend}: repeated or corrupted output detected; server and client stopped. "
                      f"See {folder / backend / 'output-monitor.txt'}", flush=True)
                return 2
            results[backend]["correctness"] = grade_run(trace, results[backend], folder / backend)
            save(folder / backend / "summary.json", results[backend])
            correctness = results[backend]["correctness"]
            print(f"CORRECTNESS {backend}: {correctness['correct']} / {correctness['total']}, "
                  f"accuracy={correctness['accuracy_percent']}, status={correctness['status']}", flush=True)
            save(folder / "comparison.json", results)
    except KeyboardInterrupt:
        print("INTERRUPTED, stopped owned server and client processes", flush=True)
        save(folder / "run.json", {"status": "interrupted", "results": results})
        append_report(args, trace, folder, results)
        return 130
    append_report(args, trace, folder, results)
    status = "completed" if all(row.get("status") == "completed" and
                                row.get("correctness", {}).get("status") == "completed"
                                for row in results.values()) else "failed"
    save(folder / "run.json", {"status": status, "results": results})
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
