# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated native operators for the 128-dimensional DiffKV v2 cache format.

Builds against the active interpreter and PyTorch, independently of vLLM's _C.
Only the FP16, GQA4, K8V4/K4V2 configuration is supported by this bringup.
"""

import hashlib
import json
import os
import platform
import subprocess
import sysconfig
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import regex as re
import torch

PAGE_SIZE = 3392  # int16 elements
PAGE_BYTES = 6784
HIGH_PAGE_TOKENS = 32
LOW_PAGE_TOKENS = 60
PARTITION_SIZE = 240
QUANT_ARGS = (8, 4, 4, 2, 1, 1, 1, 1, 4, 2, 32, 60)


def _source_root() -> Path:
    root = Path(__file__).resolve().parents[2] / "csrc" / "diffkv_v2"
    if not (root / "bindings.cpp").is_file():
        raise RuntimeError(
            "DiffKV v2 requires an editable source checkout with csrc/diffkv_v2"
        )
    return root


def _build_identity(root: Path, cuda_home: str) -> tuple[str, dict]:
    compiler_version = subprocess.run(
        [str(Path(cuda_home) / "bin" / "nvcc"), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(r"release (\d+)\.(\d+)", compiler_version)
    if match is None or int(match[1]) != int(torch.version.cuda.split(".")[0]):
        raise RuntimeError(
            f"CUDA toolkit at {cuda_home} must match PyTorch CUDA major version "
            f"{torch.version.cuda}; nvcc reports: {compiler_version.strip()}"
        )
    identity = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python_abi": sysconfig.get_config_var("SOABI"),
        "machine": platform.machine(),
        "cxx11_abi": torch.compiled_with_cxx11_abi(),
        "cuda_home": cuda_home,
        "nvcc": compiler_version.strip(),
        "arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "visible_device_capabilities": (
            None
            if os.environ.get("TORCH_CUDA_ARCH_LIST")
            else [
                torch.cuda.get_device_capability(i)
                for i in range(torch.accelerator.device_count())
            ]
        ),
        "cxx": os.environ.get("CXX", "c++"),
        "loader_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.suffix in {".cpp", ".cu", ".cuh", ".h"}
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    return fingerprint, identity


@lru_cache(maxsize=1)
def get_extension(*, verbose: bool = False) -> ModuleType:
    """Build or reuse the extension for this source, Python, Torch and CUDA ABI.

    Build failures propagate; this function never falls back to dense attention.
    CUDA kernels use PyTorch's current stream on the input tensor's device.
    """
    from torch.utils.cpp_extension import CUDA_HOME, load

    if torch.version.cuda is None or torch.version.hip is not None:
        raise RuntimeError("DiffKV v2 currently requires a CUDA build of PyTorch")
    if CUDA_HOME is None:
        raise RuntimeError("DiffKV v2 requires the CUDA toolkit to build its operators")
    root = _source_root()
    fingerprint, identity = _build_identity(root, CUDA_HOME)
    build_base = Path(
        os.environ.get(
            "VLLM_DIFFKV_V2_BUILD_DIR", str(root.parents[1] / ".cache" / "diffkv-v2")
        )
    )
    build_dir = build_base / fingerprint[:20]
    build_dir.mkdir(parents=True, exist_ok=True)
    extension = load(
        name="_diffkv_v2_C",
        sources=[
            str(root / name)
            for name in (
                "bindings.cpp",
                "attention/sparse_gqa_attention_kernels.cu",
                "cache_kernels.cu",
                "long_prompt_cache_kernels.cu",
                "mem_mgt_kernels.cu",
            )
        ],
        extra_cflags=[
            "-O2",
            "-fvisibility=hidden",
            f'-DDIFFKV_V2_BUILD_FINGERPRINT=\\"{fingerprint}\\"',
        ],
        extra_cuda_cflags=["-O2", "-lineinfo"],
        build_directory=str(build_dir),
        verbose=verbose,
        with_cuda=True,
    )
    if extension.build_fingerprint != fingerprint:
        raise RuntimeError("DiffKV v2 extension fingerprint mismatch; rebuild required")
    (build_dir / "manifest.json").write_text(json.dumps(identity, indent=2) + "\n")
    return extension


def __getattr__(name: str):
    if name in {"ops", "cache_ops", "mem_mgt_ops", "mem_mgt"}:
        return getattr(get_extension(), name)
    raise AttributeError(name)
