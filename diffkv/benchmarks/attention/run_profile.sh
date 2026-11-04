#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Nsight Systems traces of eager decode attention for DiffKV, vLLM FA2 FP16 and
# FlashInfer FP8, one trace per batch size and KV length (512, 2048, 8192). Each
# trace goes through diffkv/benchmarks/evaluation/lock_clock.sh: SM clock locked to the
# base clock, cooldown before the run, telemetry, and a rerun when throttled.
# Usage: run_profile.sh [batch ...]   (default: 1 8 50)
# Results: attention/logs/<UTC>-nsys-b<batch>-kv<length>/
#   source_hashes.json, gpu_telemetry.jsonl, clock_check.json, cuda_events.json,
#   trace.nsys-rep, trace.sqlite, kernel_attribution.json, console.log
# DIFFKV_GPU picks the GPU (default 0).
set -euo pipefail

batches=("$@")
if [ ${#batches[@]} -eq 0 ]; then
  batches=(1 8 50)
fi
contexts=(512 2048 8192)

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
diffkv=$(cd -- "$here/../.." && pwd)
run_docker=("bash" "$diffkv/run_docker.sh")

for batch in "${batches[@]}"; do
  for context in "${contexts[@]}"; do
    run_dir="$here/logs/$(date -u +%Y%m%dT%H%M%SZ)-nsys-b$batch-kv$context"
    mkdir -p "$run_dir"
    "${run_docker[@]}" python "$diffkv/benchmarks/math500/record_source_hashes.py" \
      --output "$run_dir/source_hashes.json"
    bash "$diffkv/benchmarks/evaluation/lock_clock.sh" "$run_dir" \
      "${run_docker[@]}" env FLASHINFER_EXTRA_CUDAFLAGS=--compress-mode=none \
      nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi \
      --capture-range-end=stop --sample=none --cpuctxsw=none --force-overwrite=true \
      --output "$run_dir/trace" \
      python "$here/profile_nsys.py" profile --batch "$batch" --context "$context" \
      --output "$run_dir/cuda_events.json" > "$run_dir/console.log" 2>&1
    "${run_docker[@]}" nsys export --type sqlite --output "$run_dir/trace.sqlite" \
      "$run_dir/trace.nsys-rep" >> "$run_dir/console.log" 2>&1
    "${run_docker[@]}" python "$here/profile_nsys.py" analyze \
      --input "$run_dir/trace.sqlite" --events "$run_dir/cuda_events.json" \
      >> "$run_dir/console.log" 2>&1
    echo "$run_dir"
  done
done
