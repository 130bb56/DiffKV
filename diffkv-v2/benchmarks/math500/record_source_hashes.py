# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture code identity before a run, including untracked bringup sources."""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from math500.generation_utils import save_json, sha256

ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIRS = ("diffkv-v2",)
UPSTREAM_FILES = (
    "vllm/config/vllm.py",
    "vllm/v1/attention/backends/registry.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/core/single_type_kv_cache_manager.py",
    "vllm/v1/outputs.py",
    "vllm/v1/worker/gpu/model_runner.py",
)
SOURCE_SUFFIXES = {".py", ".cpp", ".cu", ".cuh", ".h", ".json", ".txt", ".sh"}


def capture():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT)

    paths = {ROOT / path for path in UPSTREAM_FILES}
    for directory in SOURCE_DIRS:
        paths.update(
            path
            for path in (ROOT / directory).rglob("*")
            if path.is_file()
            and path.suffix in SOURCE_SUFFIXES
            and "logs" not in path.relative_to(ROOT / directory).parts
        )
    # Include additional tracked edits if the bringup grows another hook.
    paths.update(
        ROOT / path
        for path in git("diff", "--name-only", "HEAD", "--", "vllm", "csrc")
        .decode()
        .splitlines()
        if (ROOT / path).is_file()
    )
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    patch = git("diff", "HEAD", "--", "vllm", "csrc")
    return {
        "schema_version": 1,
        "captured_utc": datetime.now(UTC).isoformat(),
        "git_commit": git("rev-parse", "HEAD").decode().strip(),
        "git_status": git(
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *SOURCE_DIRS,
            *UPSTREAM_FILES,
        )
        .decode()
        .splitlines(),
        "tracked_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "source_tree_sha256": hashlib.sha256(encoded).hexdigest(),
        "files_sha256": hashes,
        "scope": "Custom Python/CUDA sources, benchmark inputs and code, "
        "upstream hooks, plus other modified tracked vllm/csrc files. "
        "Capture before launching the run; a later snapshot cannot establish "
        "the source bytes imported by an already running process.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    result = capture()
    save_json(args.output, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "git_commit",
                    "source_tree_sha256",
                    "captured_utc",
                )
            },
            indent=2,
        )
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
