#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Decode attention latency for DiffKV, vLLM FA2 FP16 and FlashInfer FP8 at KV
# lengths 1024, 2048, 4096 and 8192, one run per batch size. Each run goes
# through diffkv/benchmarks/evaluation/lock_clock.sh: SM clock locked to the base clock,
# cooldown before the run, telemetry, and a rerun when throttled.
# Usage: run_latency.sh [batch ...]   (default: 1 8 32)
# Results: attention/logs/<UTC>-latency-b<batch>/
#   source_hashes.json, gpu_telemetry.jsonl, clock_check.json, latency.json,
#   console.log
# DIFFKV_GPU picks the GPU (default 0).
set -euo pipefail

batches=("$@")
if [ ${#batches[@]} -eq 0 ]; then
  batches=(1 8 32)
fi

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
diffkv=$(cd -- "$here/../.." && pwd)
run_docker=("bash" "$diffkv/run_docker.sh")

for batch in "${batches[@]}"; do
  run_dir="$here/logs/$(date -u +%Y%m%dT%H%M%SZ)-latency-b$batch"
  mkdir -p "$run_dir"
  "${run_docker[@]}" python "$diffkv/benchmarks/math500/record_source_hashes.py" \
    --output "$run_dir/source_hashes.json"
  bash "$diffkv/benchmarks/evaluation/lock_clock.sh" "$run_dir" \
    "${run_docker[@]}" env FLASHINFER_EXTRA_CUDAFLAGS=--compress-mode=none \
    python "$here/benchmark_latency.py" --batch "$batch" \
    --output "$run_dir/latency.json" > "$run_dir/console.log" 2>&1
  echo "$run_dir"
done
