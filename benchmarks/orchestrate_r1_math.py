"""Execute the approved R1 screening, stopping at a failed greedy parity gate."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from run_r1_math import ROOT, file_digest, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--wait-model", action="store_true")
    args = parser.parse_args()
    folder, model = args.run_dir.resolve(), args.model.resolve()
    workload = folder / "workload.json"
    fp_python = ROOT / ".cache/baselines/fp16-env/bin/python"
    dq_python = ROOT / ".venv/bin/python"
    state = {"status": "running", "stage": "parity", "commands": [],
             "started_utc": datetime.now(timezone.utc).isoformat(),
             "workload_sha256": file_digest(workload),
             "orchestrator_sha256": file_digest(__file__)}
    state_path = folder / "run-status.json"
    if state_path.exists():
        raise FileExistsError(f"Refusing to replace run status: {state_path}")

    def update(stage, status="running"):
        state.update(stage=stage, status=status,
                     updated_utc=datetime.now(timezone.utc).isoformat())
        save_json(state_path, state)
        print(f"STAGE {stage} {status}", flush=True)

    def launch(name, python, script, arguments):
        command = [str(python), "-u", str(ROOT / "benchmarks" / script), *map(str, arguments)]
        log = (folder / f"{name}.log").open("x")
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record = {"name": name, "argv": command, "pid": process.pid,
                  "log": str(folder / f"{name}.log")}
        state["commands"].append(record)
        save_json(state_path, state)
        return process, log, record

    def finish(jobs):
        failed = []
        for process, log, record in jobs:
            record["exit_code"] = process.wait()
            log.close()
            save_json(state_path, state)
            if record["exit_code"]:
                failed.append(record["name"])
        if failed:
            raise RuntimeError(f"Stages failed: {', '.join(failed)}")

    common = ["--workload", workload, "--model", model]
    try:
        update("model-integrity")
        provenance = json.loads((folder / "provenance.json").read_text())
        expected = provenance["files"]
        while any(not (model / row["rfilename"]).exists()
                  or (model / row["rfilename"]).stat().st_size != row["size"]
                  for row in expected):
            if not args.wait_model:
                raise RuntimeError("Model weight download is incomplete")
            time.sleep(30)
        state["verified_model_files"] = []
        for row in expected:
            digest = hashlib.sha256()
            with (model / row["rfilename"]).open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != row["lfs"]["sha256"]:
                raise RuntimeError(f"Model SHA256 mismatch: {row['rfilename']}")
            state["verified_model_files"].append({"name": row["rfilename"], "sha256": digest.hexdigest()})
        save_json(state_path, state)
        update("parity")
        finish([
            launch("parity-hf", dq_python, "validate_r1_greedy.py",
                   ["--backend", "hf", "--gpu", 1, "--output", folder / "parity-hf.json", *common]),
            launch("parity-fp16", fp_python, "validate_r1_greedy.py",
                   ["--backend", "fp16", "--gpu", 0, "--output", folder / "parity-fp16.json", *common]),
        ])
        update("parity-compare")
        comparison = launch("parity-check", dq_python, "validate_r1_greedy.py",
                            ["--backend", "compare", "--reference", folder / "parity-hf.json",
                             "--candidate", folder / "parity-fp16.json", "--output", folder / "parity-check.json"])
        process, log, record = comparison
        record["exit_code"] = process.wait()
        log.close()
        gate = json.loads((folder / "parity-check.json").read_text())
        if record["exit_code"] or gate.get("exact_match") is not True:
            state["reason"] = "Exact greedy output parity failed. Accuracy/performance runs were not started."
            update("parity-compare", "parity_failed")
            return 2
        update("natural")
        finish([
            launch("fp16-natural", fp_python, "run_r1_math.py",
                   ["--backend", "fp16", "--gpu", 0, "--output", folder / "fp16-natural.json", *common]),
            launch("diffkv-natural", fp_python, "run_r1_math.py",
                   ["--backend", "diffkv", "--gpu", 1, "--output", folder / "diffkv-natural.json", *common]),
        ])
        update("length-replay")
        finish([launch("diffkv-replay", fp_python, "run_r1_math.py",
                       ["--backend", "diffkv", "--gpu", 0, "--mode", "length_replay",
                        "--lengths-from", folder / "fp16-natural.json",
                        "--output", folder / "diffkv-replay.json", *common])])
        update("analysis")
        finish([launch("analysis", dq_python, "analyze_r1_math.py",
                       ["--workload", workload, "--fp16", folder / "fp16-natural.json",
                        "--diffkv", folder / "diffkv-natural.json", "--replay", folder / "diffkv-replay.json",
                        "--output", folder / "analysis.json"])])
        analysis = json.loads((folder / "analysis.json").read_text())
        if not analysis["all_audits_passed"]:
            raise RuntimeError("Result audit failed; see analysis.json")
        state["correctness_passed"] = analysis["correctness"]["passed"]
        state["performance_passed"] = analysis["performance"]["passed"]
        update("complete", "completed")
        return 0
    except BaseException as error:
        state["error"] = str(error)
        update(state["stage"], "failed")
        raise


if __name__ == "__main__":
    sys.exit(main())
