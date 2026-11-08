#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Decode attention latency for DiffKV, vLLM FA2 FP16 and FlashInfer FP8 at KV
# lengths 1024, 2048, 4096 and 8192, one run per batch size. Each run goes
# through utils/lock_clock.sh: SM clock locked to the base clock, cooldown
# before the run, telemetry, and a rerun when throttled.
# Usage: run_latency.sh [batch ...]   (default: 1 8 32)
# Results: logs/attention/<UTC>-latency-b<batch>/
#   source_hashes.json, gpu_telemetry.jsonl, clock_check.json, cpu_time.json,
#   latency.json, console.log
# DIFFKV_GPU picks the GPU (default 0).
# DIFFKV_CONFIGS="<name> ..." times only those configs (diffkv, fa2_fp16, fp8,
# fp8_cuda) and adds their names to the directory. Splitting the configs over
# GPUs this way compares separate processes on separate devices, not one
# process on one device.
set -euo pipefail

batches=("$@")
if [ ${#batches[@]} -eq 0 ]; then
  batches=(1 8 32)
fi

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
evaluations=$(dirname -- "$here")
diffkv=$(dirname -- "$evaluations")
run_docker=("bash" "$diffkv/run_docker.sh")
extra=()
suffix=
if [ -n "${DIFFKV_CONFIGS:-}" ]; then
  # shellcheck disable=SC2206
  extra+=(--configs ${DIFFKV_CONFIGS})
  suffix="-${DIFFKV_CONFIGS// /-}"
fi

for batch in "${batches[@]}"; do
  run_dir="$evaluations/logs/attention/$(date -u +%Y%m%dT%H%M%SZ)-latency-b$batch$suffix"
  mkdir -p "$run_dir"
  "${run_docker[@]}" python "$evaluations/utils/record_source_hashes.py" \
    --output "$run_dir/source_hashes.json"
  bash "$evaluations/utils/lock_clock.sh" "$run_dir" \
    "${run_docker[@]}" env FLASHINFER_EXTRA_CUDAFLAGS=--compress-mode=none \
    python "$here/benchmark_latency.py" --batch "$batch" "${extra[@]}" \
    --output "$run_dir/latency.json" > "$run_dir/console.log" 2>&1
  echo "$run_dir"
done
