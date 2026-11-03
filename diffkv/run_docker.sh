#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Run one command inside the bringup container. The host (Ubuntu 18.04,
# glibc 2.27) cannot load torch 2.13, so every DiffKV command goes through here.
# Usage: run_docker.sh <command> [args...]; DIFFKV_GPU picks the GPU (default 0).
set -euo pipefail

diffkv_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
diffkv_image=${DIFFKV_CONTAINER_IMAGE:-sha256:ca18e2f9134073dc2cc5d9899d7d9974e321008ecc4ab7745477eeb5508befe8}
# Host Nsight Systems, mounted read-only so `nsys` works inside the container.
nsys_dir=${DIFFKV_NSYS_DIR:-/opt/nvidia/nsight-systems-cli/2024.4.1}
profiler_args=()
if [ -d "$nsys_dir" ]; then
  profiler_args=(-v "$nsys_dir:$nsys_dir:ro" -e "NSYS_BIN=$nsys_dir/target-linux-x64")
fi
# Host Nsight Compute, mounted read-only so `ncu` works inside the container.
# The driver restricts GPU performance counters to admins
# (RmProfilingAdminOnly=1), so ncu also needs DIFFKV_SYS_ADMIN=1. The driver
# checks the profiled process itself, and a non-root user in the container
# does not hold CAP_SYS_ADMIN even with --cap-add, so that mode runs as root.
ncu_dir=${DIFFKV_NCU_DIR:-/opt/nvidia/nsight-compute/2023.3.0}
if [ -d "$ncu_dir" ]; then
  profiler_args+=(-v "$ncu_dir:$ncu_dir:ro" -e "NCU_BIN=$ncu_dir")
fi
user=(--user "$(id -u):$(id -g)")
if [ "${DIFFKV_SYS_ADMIN:-0}" = 1 ]; then
  # git refuses a repository owned by another user unless safe.directory allows it.
  profiler_args+=(--cap-add=SYS_ADMIN -e GIT_CONFIG_COUNT=1
    -e GIT_CONFIG_KEY_0=safe.directory -e "GIT_CONFIG_VALUE_0=*")
  user=(--user 0:0)
fi
# A linked git worktree symlinks .venv, .toolchains, models, .cache and the
# precompiled vllm binaries to the primary checkout. Mount that checkout
# read-only (its .cache writable) and import diffkv and vllm from this tree.
primary_repo=$(dirname -- "$(git -C "$diffkv_repo" rev-parse \
  --path-format=absolute --git-common-dir)")
worktree_mount=()
if [ "$primary_repo" != "$diffkv_repo" ]; then
  worktree_mount=(-v "$primary_repo:$primary_repo:ro"
    -v "$primary_repo/.cache:$primary_repo/.cache"
    -e "PYTHONPATH=$diffkv_repo/diffkv/python:$diffkv_repo/diffkv/benchmarks:$diffkv_repo")
fi

exec docker run --rm --gpus "\"device=${DIFFKV_GPU:-0}\"" --ipc=host \
  "${user[@]}" "${profiler_args[@]}" "${worktree_mount[@]}" \
  -e CUDA_MODULE_LOADING=LAZY \
  -e CUDA_HOME="$diffkv_repo/.toolchains/cuda-12.9" \
  -e TORCH_CUDA_ARCH_LIST=8.6 -e MAX_JOBS=2 \
  -e HF_HUB_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false -e OMP_NUM_THREADS=2 \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_CONFIG_ROOT="$diffkv_repo/.cache/bringup/vllm-config" \
  -e FLASHINFER_WORKSPACE_BASE="$diffkv_repo" \
  -e XDG_CACHE_HOME="$diffkv_repo/.cache" \
  -e TRITON_CACHE_DIR="$diffkv_repo/.cache/bringup/triton" \
  -e UV_CACHE_DIR="$diffkv_repo/.cache/bringup/uv" \
  -v "$diffkv_repo:$diffkv_repo" -w "$diffkv_repo" "$diffkv_image" \
  bash -lc 'export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:${NSYS_BIN:+$NSYS_BIN:}${NCU_BIN:+$NCU_BIN:}$PATH"; exec "$@"' \
  bash "$@"
