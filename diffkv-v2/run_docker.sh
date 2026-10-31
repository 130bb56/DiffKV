#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Run one command inside the bringup container. The host (Ubuntu 18.04,
# glibc 2.27) cannot load torch 2.13, so every DiffKV command goes through here.
# Usage: run_docker.sh <command> [args...]; DIFFKV_GPU picks the GPU (default 0).
set -euo pipefail

diffkv_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
diffkv_image=${DIFFKV_CONTAINER_IMAGE:-sha256:ca18e2f9134073dc2cc5d9899d7d9974e321008ecc4ab7745477eeb5508befe8}

exec docker run --rm --gpus "\"device=${DIFFKV_GPU:-0}\"" --ipc=host \
  --user "$(id -u):$(id -g)" \
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
  bash -lc 'export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"; exec "$@"' \
  bash "$@"
