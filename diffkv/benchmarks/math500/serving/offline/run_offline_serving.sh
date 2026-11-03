#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# MATH-500 offline serving on the 50-problem sample with one KV backend: all
# requests start at once in one LLMEngine process and each request is fixed to
# its bringup FP16 output length (output_lengths_sample50.json, ignore_eos).
# The GPU SM clock is locked to 1395 MHz and GPU telemetry is recorded.
# Usage: run_offline_serving.sh <fp16|diffkv|fp8>
# Results: math500/logs/serving/offline/<UTC>-sample50-<backend>/
#   source_hashes.json, gpu_telemetry.csv, outputs.json, console.log
# DIFFKV_GPU picks the GPU (default 0).
# DIFFKV_NSYS_STEPS="FIRST COUNT" runs under Nsight Systems, traces measured
# engine steps FIRST..FIRST+COUNT-1 and stops the run there. The directory gets
# the suffix -nsys-s<FIRST>-n<COUNT> and also trace.nsys-rep, trace.sqlite and
# step_breakdown.json (analyze_nsys_steps.py).
# DIFFKV_NCU_STEPS="FIRST COUNT" runs under Nsight Compute (--set full) and
# profiles the attention, reduce and KV cache write kernels of measured engine
# steps FIRST..FIRST+COUNT-1, then stops the run. The directory gets the suffix
# -ncu-s<FIRST>-n<COUNT> and also kernels.ncu-rep, kernels_raw.csv and
# kernel_summary.json (analyze_ncu_kernels.py).
set -euo pipefail

backend=${1:?backend: fp16, diffkv or fp8}
gpu=${DIFFKV_GPU:-0}
image=${DIFFKV_CONTAINER_IMAGE:-sha256:ca18e2f9134073dc2cc5d9899d7d9974e321008ecc4ab7745477eeb5508befe8}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
math500=$(cd -- "$here/../.." && pwd)
diffkv=$(cd -- "$math500/../.." && pwd)
repo=$(dirname -- "$diffkv")
run_docker=("bash" "$diffkv/run_docker.sh")
suffix=
profiler=()
profile=()
if [ -n "${DIFFKV_NSYS_STEPS:-}" ] && [ -n "${DIFFKV_NCU_STEPS:-}" ]; then
  echo "Set DIFFKV_NSYS_STEPS or DIFFKV_NCU_STEPS, not both" >&2
  exit 2
fi
if [ -n "${DIFFKV_NSYS_STEPS:-}" ]; then
  read -r first count <<< "$DIFFKV_NSYS_STEPS"
  suffix="-nsys-s$first-n$count"
  profile=(--profile-steps "$first" "$count")
elif [ -n "${DIFFKV_NCU_STEPS:-}" ]; then
  read -r first count <<< "$DIFFKV_NCU_STEPS"
  suffix="-ncu-s$first-n$count"
  profile=(--profile-steps "$first" "$count")
  export DIFFKV_SYS_ADMIN=1
fi
run_dir="$math500/logs/serving/offline/$(date -u +%Y%m%dT%H%M%SZ)-sample50-$backend$suffix"
mkdir -p "$run_dir"
if [ -n "${DIFFKV_NSYS_STEPS:-}" ]; then
  profiler=(nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi
    --capture-range-end=stop --sample=none --cpuctxsw=none
    --output "$run_dir/trace")
elif [ -n "${DIFFKV_NCU_STEPS:-}" ]; then
  kernels="register_gqa_attention_kernel|register_gqa_reduce_kernel"
  kernels+="|compress_and_append_cache_decode_phase|BatchPrefillWithPagedKVCache"
  kernels+="|BatchDecodeWithPagedKVCache|MergeStates|reshape_and_cache"
  profiler=(ncu --profile-from-start off --set full --kernel-name "regex:$kernels"
    --export "$run_dir/kernels")
fi

docker run --rm --gpus "\"device=$gpu\"" --cap-add=SYS_ADMIN "$image" \
  nvidia-smi -i 0 -lgc 1395,1395
nvidia-smi -i "$gpu" --format=csv --loop-ms=200 \
  --query-gpu=timestamp,index,uuid,pstate,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,clocks_throttle_reasons.active,memory.used,utilization.gpu \
  > "$run_dir/gpu_telemetry.csv" 2>&1 &
telemetry=$!
trap 'kill "$telemetry" 2>/dev/null || true' EXIT

"${run_docker[@]}" python "$math500/record_source_hashes.py" \
  --output "$run_dir/source_hashes.json"
"${run_docker[@]}" env FLASHINFER_EXTRA_CUDAFLAGS=--compress-mode=none \
  "${profiler[@]}" python "$math500/generate.py" --backend "$backend" \
  --model "$repo/models/DeepSeek-R1-Distill-Llama-8B" \
  --expected-count 50 --mode length_replay \
  --lengths-from "$here/output_lengths_sample50.json" \
  --output "$run_dir/outputs.json" "${profile[@]}" > "$run_dir/console.log" 2>&1
if [ -n "$suffix" ]; then
  kill "$telemetry" 2>/dev/null || true
fi
if [ -n "${DIFFKV_NSYS_STEPS:-}" ]; then
  "${run_docker[@]}" nsys export --type sqlite --output "$run_dir/trace.sqlite" \
    "$run_dir/trace.nsys-rep" >> "$run_dir/console.log" 2>&1
  "${run_docker[@]}" python "$here/analyze_nsys_steps.py" \
    --trace "$run_dir/trace.sqlite" --outputs "$run_dir/outputs.json" \
    --output "$run_dir/step_breakdown.json" >> "$run_dir/console.log" 2>&1
elif [ -n "${DIFFKV_NCU_STEPS:-}" ]; then
  "${run_docker[@]}" ncu --import "$run_dir/kernels.ncu-rep" --csv --page raw \
    --print-units base > "$run_dir/kernels_raw.csv" 2>> "$run_dir/console.log"
  "${run_docker[@]}" python "$here/analyze_ncu_kernels.py" \
    --raw "$run_dir/kernels_raw.csv" --outputs "$run_dir/outputs.json" \
    --output "$run_dir/kernel_summary.json" >> "$run_dir/console.log" 2>&1
fi
echo "$run_dir"
