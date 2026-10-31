#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Nsight Systems traces of eager decode attention for DiffKV v2, vLLM FA2 FP16 and
# FlashInfer FP8, one trace per batch size and KV length (512, 2048, 8192). The GPU
# SM clock is locked to 1395 MHz and GPU telemetry is recorded.
# Usage: run_profile.sh [batch ...]   (default: 1 8 50)
# Results: attention/logs/<UTC>-nsys-b<batch>-kv<length>/
#   source_hashes.json, gpu_telemetry.csv, cuda_events.json, trace.nsys-rep,
#   trace.sqlite, kernel_attribution.json, console.log
# DIFFKV_GPU picks the GPU (default 0).
set -euo pipefail

batches=("$@")
if [ ${#batches[@]} -eq 0 ]; then
  batches=(1 8 50)
fi
contexts=(512 2048 8192)
gpu=${DIFFKV_GPU:-0}
image=${DIFFKV_CONTAINER_IMAGE:-sha256:ca18e2f9134073dc2cc5d9899d7d9974e321008ecc4ab7745477eeb5508befe8}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
diffkv=$(cd -- "$here/../.." && pwd)
run_docker=("bash" "$diffkv/run_docker.sh")
telemetry=
trap 'if [ -n "$telemetry" ]; then kill "$telemetry" 2>/dev/null || true; fi' EXIT

docker run --rm --gpus "\"device=$gpu\"" --cap-add=SYS_ADMIN "$image" \
  nvidia-smi -i 0 -lgc 1395,1395
for batch in "${batches[@]}"; do
  for context in "${contexts[@]}"; do
    run_dir="$here/logs/$(date -u +%Y%m%dT%H%M%SZ)-nsys-b$batch-kv$context"
    mkdir -p "$run_dir"
    nvidia-smi -i "$gpu" --format=csv --loop-ms=200 \
      --query-gpu=timestamp,index,uuid,pstate,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,clocks_throttle_reasons.active,memory.used,utilization.gpu \
      > "$run_dir/gpu_telemetry.csv" 2>&1 &
    telemetry=$!
    "${run_docker[@]}" python "$diffkv/benchmarks/math500/record_source_hashes.py" \
      --output "$run_dir/source_hashes.json"
    "${run_docker[@]}" env FLASHINFER_EXTRA_CUDAFLAGS=--compress-mode=none \
      nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi \
      --capture-range-end=stop --sample=none --cpuctxsw=none \
      --output "$run_dir/trace" \
      python "$here/profile_nsys.py" profile --batch "$batch" --context "$context" \
      --output "$run_dir/cuda_events.json" > "$run_dir/console.log" 2>&1
    kill "$telemetry" 2>/dev/null || true
    telemetry=
    "${run_docker[@]}" nsys export --type sqlite --output "$run_dir/trace.sqlite" \
      "$run_dir/trace.nsys-rep" >> "$run_dir/console.log" 2>&1
    "${run_docker[@]}" python "$here/profile_nsys.py" analyze \
      --input "$run_dir/trace.sqlite" --events "$run_dir/cuda_events.json" \
      >> "$run_dir/console.log" 2>&1
    echo "$run_dir"
  done
done
