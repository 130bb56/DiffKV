"""Collect one isolated decode main-kernel report on physical GPU 1.

Docker exposes only GPU-feccd811-0bc3-0d68-066b-9bf5bc7cdbb7. The worker sees
that GPU as CUDA ordinal zero. No clocks, driver settings, or installed binaries
are modified. Use --plan-only to save commands without starting GPU work.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = "GPU-feccd811-0bc3-0d68-066b-9bf5bc7cdbb7"
IMAGE = "nvidia/cuda:12.4.1-base-ubuntu22.04"
NCU_ROOT = Path("/usr/local/cuda-12.4/nsight-compute-2024.1.0")
NCU = "/host-cuda/nsight-compute-2024.1.0/target/linux-desktop-glibc_2_11_3-x64/ncu"
PYTHON = ROOT / ".cache/baselines/fp16-env/bin/python"
WORKER = ROOT / "benchmarks/validate_diffkv_kernel_candidate.py"
EXTRACTOR = ROOT / "logs/performance/attention-pd-20261022/extract.py"
SECTIONS = ("SpeedOfLight", "LaunchStats", "Occupancy", "MemoryWorkloadAnalysis",
            "MemoryWorkloadAnalysis_Tables", "SchedulerStats", "WarpStateStats",
            "SourceCounters", "InstructionStats")
KERNEL = "regex:.*sparse_paged_attention_wrapper.*"
STATIC_PREFIX = "void vllm::sparse_paged_attention_wrapper<unsigned short, 128, 4, 8, 4, 4, 2, 1, 1, 1, 1,"
KERNEL_NAMES = {
    "legacy": ("sparse_paged_attention_wrapper",),
    "gqa": ("register_gqa_attention_kernel", "sparse_gqa_attention_kernel"),
}


def kernel_filter(op):
    return KERNEL if op == "legacy" else "regex:.*(" + "|".join(KERNEL_NAMES[op]) + ").*"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def container(gpu, root=ROOT):
    command = ["docker", "run", "--rm", "--network", "none"]
    if gpu:
        command += ["--gpus", f"device={GPU_UUID}", "--cap-add", "SYS_ADMIN"]
    command += ["--entrypoint", "/bin/bash", "--workdir", str(root)]
    mounts = [(str(root), str(root), "rw"),
              ("/usr/local/bin/python3.11", "/usr/local/bin/python3.11", "ro"),
              ("/usr/local/lib/python3.11", "/usr/local/lib/python3.11", "ro"),
              ("/usr/lib/x86_64-linux-gnu/libffi.so.6", "/usr/lib/x86_64-linux-gnu/libffi.so.6", "ro"),
              ("/usr/lib/x86_64-linux-gnu/libssl.so.1.1", "/usr/lib/x86_64-linux-gnu/libssl.so.1.1", "ro"),
              ("/usr/lib/x86_64-linux-gnu/libcrypto.so.1.1", "/usr/lib/x86_64-linux-gnu/libcrypto.so.1.1", "ro"),
              ("/usr/bin/git", "/usr/bin/git", "ro"),
              ("/lib/x86_64-linux-gnu/libpcre.so.3", "/lib/x86_64-linux-gnu/libpcre.so.3", "ro"),
              ("/usr/local/cuda-12.4", "/host-cuda", "ro"),
              ("/usr/lib/x86_64-linux-gnu/libcuda.so.525.147.05", "/host-driver/libcuda.so.1", "ro"),
              ("/usr/lib/x86_64-linux-gnu/libcuda.so.525.147.05", "/host-driver/libcuda.so", "ro"),
              ("/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.525.147.05", "/host-driver/libnvidia-ptxjitcompiler.so.1", "ro"),
              ("/usr/lib/x86_64-linux-gnu/libnvidia-nvvm.so.525.147.05", "/host-driver/libnvidia-nvvm.so.4", "ro")]
    for source, target, mode in mounts:
        if not Path(source).exists():
            raise FileNotFoundError(f"Required container mount: {source}")
        command += ["--volume", f"{source}:{target}:{mode}"]
    for item in ("PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=" + str(root),
                 "LD_LIBRARY_PATH=/host-driver:/usr/local/nvidia/lib64:/usr/local/nvidia/lib",
                 "NVIDIA_DRIVER_CAPABILITIES=compute,utility", "HF_HUB_OFFLINE=1",
                 "TRANSFORMERS_OFFLINE=1", "GIT_CONFIG_COUNT=1", "GIT_CONFIG_KEY_0=safe.directory",
                 "GIT_CONFIG_VALUE_0=" + str(root), "OMP_NUM_THREADS=2"):
        command += ["--env", item]
    return command + [IMAGE, "-c", 'umask 000; exec "$@"', "candidate-ncu"]


def execute(command, log):
    print(f"RUN {log}", flush=True)
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Exit {result.returncode}; see {log}")


def plan(args):
    folder = args.output_dir.resolve()
    worker_output = folder / "worker.pt"
    worker = [str(PYTHON), str(WORKER), "capture", "--extension", str(args.extension.resolve()),
              "--gpu", "1", "--cuda-device-ordinal", "0", "--expected-gpu-uuid", GPU_UUID,
              "--op", args.op, "--workspace-partition-size", str(args.workspace_partition_size),
              "--contexts", "3072", "--batch", "8", "--kv-heads", "8",
              "--steps", "1", "--profile", "--output", str(worker_output)]
    report = folder / "main"
    capture = container(True) + [NCU, "--config-file", "off", "--profile-from-start", "off",
               "--target-processes", "all", "--devices", "0", "--clock-control", "none",
               "--cache-control", args.cache_control, "--replay-mode", "kernel", "--nvtx",
               "--nvtx-include", "diffkv_candidate/", "--kernel-name-base", "demangled",
               "--kernel-name", kernel_filter(args.op), "--launch-count", "1", "--import-source", "yes",
               "--export", str(report)]
    for section in SECTIONS:
        capture += ["--section", section]
    capture += worker
    exports = {}
    for filename, options in (
        ("main.raw.csv", ["--page", "raw", "--csv", "--print-units", "base"]),
        ("main.details.txt", ["--page", "details", "--print-details", "all", "--print-units", "base"]),
        ("main.sass.txt", ["--page", "source", "--print-source", "sass"]),
        ("main.session.txt", ["--page", "session"])):
        exports[filename] = container(False) + [NCU, "--import", str(report) + ".ncu-rep"] + options
    exports["extract.log"] = container(False) + [str(PYTHON), str(Path(__file__).resolve()),
                                                  "--extract-report", str(report) + ".ncu-rep",
                                                  "--op", args.op,
                                                  "--output-dir", str(folder)]
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "physical_gpu": 1, "gpu_uuid": GPU_UUID, "container_cuda_ordinal": 0,
            "extension": str(args.extension.resolve()), "extension_sha256": sha256(args.extension),
            "worker_sha256": sha256(WORKER), "launcher_sha256": sha256(__file__),
            "case": {"batch": 8, "kv_heads": 8, "gqa": 4, "head_size": 128,
                     "context": 3072, "precision": "K8V4/K4V2", "main_launches": 1,
                     "operation": args.op, "workspace_partition_size": args.workspace_partition_size},
            "kernel_name_filter": kernel_filter(args.op),
            "cache_control": args.cache_control, "clock_control": "none",
            "sections": list(SECTIONS), "capture": capture, "exports": exports,
            "measurement_scope": "NCU kernel-replay diagnostic, one main kernel; use separate CUDA-event measurements for speed"}


def load_extractor():
    spec = importlib.util.spec_from_file_location("candidate_existing_ncu_extractor", EXTRACTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_report(report_path, output_dir, op="legacy"):
    extractor = load_extractor()
    ncu = extractor.import_ncu_report()
    report = ncu.load_report(str(report_path.resolve()))
    actions = [report.range_by_idx(i).action_by_idx(j) for i in range(report.num_ranges())
               for j in range(report.range_by_idx(i).num_actions())]
    if len(actions) != 1:
        raise RuntimeError(f"Expected one main action, found {len(actions)}")
    action = actions[0]
    name = action.name(ncu.IAction.NameBase_DEMANGLED)
    if not any(token in name for token in KERNEL_NAMES[op]):
        raise RuntimeError(f"Unexpected kernel: {name}")
    data = extractor.extract_action_data(action)
    data.update(name=name, nvtx_ranges=extractor.nvtx_ranges(action))
    widths = defaultdict(lambda: {"static_pcs": 0, "warp_instruction_executions": 0,
                                  "theoretical_sectors": 0, "ideal_sectors": 0,
                                  "long_scoreboard_samples": 0, "pcs": []})
    for row in data["sass_by_pc"]:
        if not re.search(r"\bLDG(?:\.|\s)", row["sass"]):
            continue
        metrics = row["metrics"]
        # memory_access_size_type is an enum, not a count of bits. For example,
        # NCU 2024.1 reports 8 for LDG.E.U16. Classify the actual SASS opcode.
        # Ignore unrelated per-instance launch metrics which resolve to a SASS
        # offset but carry no source-correlated instruction/access data.
        if not any(key in metrics for key in ("inst_executed", "memory_access_type")):
            continue
        opcode = re.search(r"\b(LDG(?:\.[A-Z0-9]+)*)", row["sass"]).group(1)
        size = re.search(r"\.(?:U|S)?(8|16|32|64|128)(?:\.|$)", opcode)
        width = size.group(1) if size else "32"
        bucket = widths[width]
        bucket["static_pcs"] += 1
        for target, key in (("warp_instruction_executions", "inst_executed"),
                            ("theoretical_sectors", "memory_l2_theoretical_sectors_global"),
                            ("ideal_sectors", "memory_l2_theoretical_sectors_global_ideal"),
                            ("long_scoreboard_samples", "smsp__pcsamp_warps_issue_stalled_long_scoreboard")):
            bucket[target] += metrics.get(key, 0) or 0
        bucket["pcs"].append({"pc": row["pc"], "sass": row["sass"], "source": row["source"],
                              "raw_memory_access_size_type": metrics.get("memory_access_size_type")})
    metrics = data["metrics"]
    selected = {key: metrics.get(key) for key in (
        "gpu__time_duration.sum", "launch__registers_per_thread",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "memory_l2_theoretical_sectors_global", "memory_l2_theoretical_sectors_global_ideal")}
    total = selected["memory_l2_theoretical_sectors_global"]
    ideal = selected["memory_l2_theoretical_sectors_global_ideal"]
    selected["software_global_excess_sectors"] = total - ideal if total is not None and ideal is not None else None
    selected["software_global_actual_over_ideal"] = total / ideal if total is not None and ideal else None
    result = {"schema_version": 1, "status": "completed", "report": str(report_path.resolve()),
              "report_sha256": sha256(report_path), "extractor": str(EXTRACTOR),
              "extractor_sha256": sha256(EXTRACTOR), "gpu_uuid_by_container_binding": GPU_UUID,
              "extraction_driver_sha256": sha256(__file__),
              "operation": op, "kernel_name_filter": kernel_filter(op),
              "summary": selected, "global_load_widths_bits": dict(widths), "action": data,
              "limits": ["Software theoretical/ideal sectors model each warp access footprint; hardware L1TEX sectors are a distinct counter.",
                         "PC stall samples are samples, not elapsed-time fractions.",
                         "Only one NCU main-kernel action is measured, with kernel replay and explicit cache-control.",
                         "Load widths are decoded from SASS opcodes; raw memory_access_size_type is an enum, not bits.",
                         "Static instruction counts are not dynamic traffic."]}
    save(output_dir / "metrics.json", result)
    print(json.dumps({"summary": selected, "load_widths_bits": {key: {k: v for k, v in row.items() if k != "pcs"}
                                                                   for key, row in widths.items()}}, indent=2))


def static_sass(extension, output_dir, op="legacy"):
    output_dir.mkdir(parents=True, exist_ok=True)
    tool = "/usr/local/cuda-12.4/bin/cuobjdump"
    symbols_raw = subprocess.run([tool, "--dump-elf-symbols", str(extension)], capture_output=True,
                                 text=True, check=True).stdout
    symbols = [line.split()[-1] for line in symbols_raw.splitlines()
               if "STT_FUNC" in line and "STO_ENTRY" in line
               and any(token in line for token in KERNEL_NAMES[op])]
    demangled = subprocess.run(["c++filt"], input="\n".join(symbols) + "\n",
                               capture_output=True, text=True, check=True).stdout.splitlines()
    selected = [(symbol, name) for symbol, name in zip(symbols, demangled)
                if (name.startswith(STATIC_PREFIX) if op == "legacy" else
                    "<unsigned short" in name and any(token in name for token in KERNEL_NAMES[op]))]
    if op == "gqa" and any("register_gqa_attention_kernel" in name for _, name in selected):
        selected = [(symbol, name) for symbol, name in selected if "register_gqa_attention_kernel" in name]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one fp16 K8V4/K4V2 GQA4 group1 kernel; found {len(selected)}")
    symbol, name = selected[0]
    command = [tool, "--dump-sass", "--function", symbol, str(extension)]
    raw = subprocess.run(command, capture_output=True, text=True, check=True)
    (output_dir / "selected.sass").write_text(raw.stdout)
    (output_dir / "cuobjdump.stderr.txt").write_text(raw.stderr)
    instructions = []
    for line in raw.stdout.splitlines():
        match = re.search(r"/\*([0-9a-f]+)\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)\s+(.*?);", line)
        if match:
            instructions.append({"pc": "0x" + match.group(1), "opcode": match.group(2), "sass": line.strip()})
    if not instructions:
        raise RuntimeError("cuobjdump returned no selected instructions")
    global_loads = [row for row in instructions if row["opcode"].split(".")[0] == "LDG"]
    by_opcode = Counter(row["opcode"] for row in global_loads)
    result = {"extension": str(extension.resolve()), "extension_sha256": sha256(extension),
              "operation": op,
              "command": command, "function": name, "symbol": symbol,
              "static_instruction_count": len(instructions), "static_global_load_count": len(global_loads),
              "static_global_loads_by_opcode": dict(by_opcode), "global_loads": global_loads,
              "sass_sha256": sha256(output_dir / "selected.sass"),
              "limits": "Static code sites only; dynamic counts and coalescing require NCU capture."}
    save(output_dir / "static-sass.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "global_loads"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--op", choices=("legacy", "gqa"), default="legacy")
    parser.add_argument("--workspace-partition-size", type=int, default=5760)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-control", choices=("all", "none"), default="all")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--extract-report", type=Path)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    if args.workspace_partition_size < 1:
        parser.error("--workspace-partition-size must be positive")
    args.output_dir = args.output_dir.resolve()
    if args.extract_report:
        extract_report(args.extract_report, args.output_dir, args.op)
        return
    if not args.extension or not args.extension.is_file():
        parser.error("--extension must identify an existing isolated _C binary")
    if args.static_only:
        static_sass(args.extension.resolve(), args.output_dir, args.op)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "command.json"
    if args.export_only:
        manifest = json.loads(manifest_path.read_text())
        if manifest["extension_sha256"] != sha256(args.extension):
            raise RuntimeError("Extension no longer matches recorded capture")
    else:
        if manifest_path.exists():
            raise FileExistsError(f"Preserve existing run: {manifest_path}")
        manifest = plan(args)
        save(manifest_path, manifest)
    if args.plan_only:
        print(json.dumps({"manifest": str(manifest_path), "gpu_started": False}, indent=2))
        return
    if not args.export_only:
        execute(manifest["capture"], args.output_dir / "capture.log")
    if not (args.output_dir / "main.ncu-rep").is_file():
        raise RuntimeError("Missing NCU report")
    for name, command in manifest["exports"].items():
        execute(command, args.output_dir / name)
    print(json.dumps({"status": "completed", "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
