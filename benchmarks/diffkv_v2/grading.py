# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Use the archived MATH grader and report the sync's explicit corrections."""

import contextlib
import io

import regex as re


def boxed_answers(text):
    answers = []
    start = 0
    while True:
        index = text.find("\\boxed{", start)
        if index < 0:
            return answers
        end, depth = index + 7, 1
        while end < len(text) and depth:
            depth += text[end] == "{"
            depth -= text[end] == "}"
            end += 1
        answers.append(text[index + 7 : end - 1])
        start = end


def normalize_decoration(text):
    """Match the corrections archived in grade500.py, without new heuristics."""
    text = re.sub(r"\\text\{[^}]*\}", "", text.strip())
    text = re.sub(r"\^\{?\\(mathrm|text)\{(st|nd|rd|th)\}\}?", "", text)
    for old, new in (
        ("\\dfrac", "\\frac"),
        ("\\tfrac", "\\frac"),
        ("\\left", ""),
        ("\\right", ""),
        ("$", ""),
        ("\\,", ""),
        ("\\!", ""),
        (" ", ""),
    ):
        text = text.replace(old, new)
    return text.rstrip(".")


def grade_one(text, reference):
    from vendor import math_grader, math_parser

    final = text.rsplit("</think>", 1)[-1]
    captured = io.StringIO()
    error = None
    prediction = ""
    raw_correct = False
    try:
        with contextlib.redirect_stdout(captured):
            prediction = math_parser.extract_answer(
                final,
                data_name="math",
                use_last_number=False,
            )
            truth = math_parser.strip_string(reference, skip_unit=False)
            raw_correct = bool(prediction) and bool(
                math_grader.math_equal(
                    prediction,
                    truth,
                    include_percentage=False,
                    timeout=True,
                )
            )
        if "Timeout error" in captured.getvalue():
            error = captured.getvalue().strip()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    correction = None
    boxes = boxed_answers(final)
    if not raw_correct and error is None and boxes:
        truth = normalize_decoration(reference)
        if normalize_decoration(boxes[-1]) == truth:
            correction = "unit_or_ordinal_decoration"
        elif (
            "," in truth
            and len(boxes) > 1
            and {normalize_decoration(box) for box in boxes}
            == {part.strip() for part in truth.split(",")}
        ):
            correction = "multiple_boxed_set"
    return {
        "prediction": prediction,
        "reference": reference,
        "raw_correct": raw_correct,
        "correct": raw_correct or correction is not None,
        "correction": correction,
        "boxed_answers": boxes,
        "answer_missing": not bool(prediction),
        "grading_error": error,
    }


def grade_run(workload, run):
    if run["metadata"]["mode"] != "natural":
        raise ValueError("Fixed-length replay is excluded from correctness")
    outputs = {row["request_id"]: row for row in run["requests"]}
    rows = []
    for request in workload["requests"]:
        output = outputs[request["request_id"]]
        rows.append(
            {
                "request_id": request["request_id"],
                **grade_one(output["output_text"], request["answer"]),
                "finish_reason": output["finish_reason"],
                "output_tokens": output["output_tokens"],
            }
        )
    errors = sum(row["grading_error"] is not None for row in rows)
    raw_correct = sum(row["raw_correct"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    return {
        "count": len(rows),
        "raw_correct": raw_correct,
        "correct": correct,
        "raw_accuracy": raw_correct / len(rows) if not errors else None,
        "accuracy": correct / len(rows) if not errors else None,
        "grading_errors": errors,
        "answer_missing": sum(row["answer_missing"] for row in rows),
        "length_capped": sum(row["finish_reason"] == "length" for row in rows),
        "corrections": [row for row in rows if row["correction"]],
        "requests": rows,
    }
