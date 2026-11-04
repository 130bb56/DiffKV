#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# MATH-500 offline serving on the 50-problem sample with one KV backend: all
# requests start at once in one LLMEngine process and each request is fixed to
# its bringup FP16 output length (output_lengths_sample50.json, ignore_eos).
# The run goes through diffkv/benchmarks/evaluation/lock_clock.sh: SM clock locked to the
# base clock, cooldown before the run, telemetry, and a rerun when throttled.
# Usage: run_offline_serving.sh <fp16|diffkv|fp8>
# Results: math500/logs/serving/offline/<UTC>-sample50-<backend>/
#   source_hashes.json, gpu_telemetry.jsonl, clock_check.json, cpu_time.json,
#   outputs.json, console.log
# DIFFKV_GPU picks the GPU (default 0).
# DIFFKV_NSYS_STEPS="FIRST COUNT" runs under Nsight Systems, traces measured
# engine steps FIRST..FIRST+COUNT-1 and stops the run there. The directory gets
# the suffix -nsys-s<FIRST>-n<COUNT> and also trace.nsys-rep, trace.sqlite and
# step_breakdown.json (analyze_nsys_steps.py).
# DIFFKV_NCU_STEPS="FIRST COUNT" runs under Nsight Compute (--set full) and
# profiles the attention, reduce and KV cache write kernels that decoder layer
# 0's attention module launches in measured engine steps FIRST..FIRST+COUNT-1,
# then stops the run. The directory gets the suffix -ncu-s<FIRST>-n<COUNT> and
# also kernels.ncu-rep, kernels_raw.csv and kernel_summary.json
# (analyze_ncu_kernels.py).
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
sys_admin=0
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
  profile=(--profile-steps "$first" "$count" --profile-first-layer)
  sys_admin=1
fi
run_dir="$math500/logs/serving/offline/$(date -u +%Y%m%dT%H%M%SZ)-sample50-$backend$suffix"
mkdir -p "$run_dir"
if [ -n "${DIFFKV_NSYS_STEPS:-}" ]; then
  profiler=(nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi
    --capture-range-end=stop --sample=none --cpuctxsw=none --force-overwrite=true
    --output "$run_dir/trace")
elif [ -n "${DIFFKV_NCU_STEPS:-}" ]; then
  kernels="register_gqa_attention_kernel|register_gqa_reduce_kernel"
  kernels+="|compress_and_append_cache_decode_phase|BatchPrefillWithPagedKVCache"
  kernels+="|BatchDecodeWithPagedKVCache|MergeStates|reshape_and_cache"
  profiler=(ncu --profile-from-start off --set full --kernel-name "regex:$kernels"
    --force-overwrite --export "$run_dir/kernels")
fi

"${run_docker[@]}" python "$math500/record_source_hashes.py" \
  --output "$run_dir/source_hashes.json"
status=0
bash "$diffkv/benchmarks/evaluation/lock_clock.sh" "$run_dir" \
  env DIFFKV_SYS_ADMIN=$sys_admin "${run_docker[@]}" \
  env FLASHINFER_EXTRA_CUDAFLAGS=--compress-mode=none \
  "${profiler[@]}" python "$math500/generate.py" --backend "$backend" \
  --model "$repo/models/DeepSeek-R1-Distill-Llama-8B" \
  --expected-count 50 --mode length_replay \
  --lengths-from "$here/output_lengths_sample50.json" \
  --output "$run_dir/outputs.json" "${profile[@]}" > "$run_dir/console.log" 2>&1 ||
  status=$?
if [ "$sys_admin" = 1 ]; then
  # The profiled run executed as root; return its files to the caller.
  docker run --rm -v "$run_dir:$run_dir" "$image" \
    chown -R "$(id -u):$(id -g)" "$run_dir"
fi
if [ "$status" -ne 0 ]; then
  echo "$run_dir: lock_clock.sh exit $status (see console.log)" >&2
  exit "$status"
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
