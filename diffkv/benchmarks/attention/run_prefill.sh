#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Prefill latency of one layer for whole prompts (chunked prefill off): current
# DiffKV backend path, DiffKV prompt_prefill, vLLM FA2 FP16 and FlashInfer FP8,
# one run per batch size. Prompt lengths come from benchmark_prefill.py
# defaults (256 1024) unless DIFFKV_PROMPTS is set. The GPU SM clock
# is locked to 1395 MHz and GPU telemetry is recorded.
# Usage: run_prefill.sh [batch ...]   (default: 1 8 32)
# Results: attention/logs/<UTC>-prefill-b<batch>/
#   source_hashes.json, gpu_telemetry.csv, latency.json, console.log
# DIFFKV_GPU picks the GPU (default 0).
set -euo pipefail

batches=("$@")
if [ ${#batches[@]} -eq 0 ]; then
  batches=(1 8 32)
fi
gpu=${DIFFKV_GPU:-0}
image=${DIFFKV_CONTAINER_IMAGE:-sha256:ca18e2f9134073dc2cc5d9899d7d9974e321008ecc4ab7745477eeb5508befe8}
extra=()
if [ -n "${DIFFKV_PROMPTS:-}" ]; then
  # shellcheck disable=SC2206
  extra+=(--prompts ${DIFFKV_PROMPTS})
fi

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
diffkv=$(cd -- "$here/../.." && pwd)
run_docker=("bash" "$diffkv/run_docker.sh")
telemetry=
trap 'if [ -n "$telemetry" ]; then kill "$telemetry" 2>/dev/null || true; fi' EXIT

docker run --rm --gpus "\"device=$gpu\"" --cap-add=SYS_ADMIN "$image" \
  nvidia-smi -i 0 -lgc 1395,1395
for batch in "${batches[@]}"; do
  run_dir="$here/logs/$(date -u +%Y%m%dT%H%M%SZ)-prefill-b$batch"
  mkdir -p "$run_dir"
  nvidia-smi -i "$gpu" --format=csv --loop-ms=200 \
    --query-gpu=timestamp,index,uuid,pstate,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,clocks_throttle_reasons.active,memory.used,utilization.gpu \
    > "$run_dir/gpu_telemetry.csv" 2>&1 &
  telemetry=$!
  "${run_docker[@]}" python "$diffkv/benchmarks/math500/record_source_hashes.py" \
    --output "$run_dir/source_hashes.json"
  "${run_docker[@]}" env FLASHINFER_EXTRA_CUDAFLAGS=--compress-mode=none \
    python "$here/benchmark_prefill.py" --batch "$batch" "${extra[@]}" \
    --output "$run_dir/latency.json" > "$run_dir/console.log" 2>&1
  kill "$telemetry" 2>/dev/null || true
  telemetry=
  echo "$run_dir"
done
