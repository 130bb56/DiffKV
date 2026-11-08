#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Prefill latency of one layer for whole prompts (chunked prefill off): DiffKV
# prompt_prefill, vLLM FA2 FP16 and FlashInfer FP8, one run per batch size.
# Prompt lengths come from benchmark_prefill.py defaults (256 512 1024) unless
# DIFFKV_PROMPTS is set. DIFFKV_PREFILL_PARTS=1 also
# times prompt_prefill's attention, score and compression launches.
# DIFFKV_CONFIGS="<name> ..." times only those configs (diffkv, fa2_fp16,
# flashinfer_fp8) and adds their names to the directory. Splitting the configs
# over GPUs this way compares separate processes on separate devices, not one
# process on one device.
# Each run goes through utils/lock_clock.sh: SM clock locked to the base clock,
# cooldown before the run, telemetry, and a rerun when throttled.
# Usage: run_prefill.sh [batch ...]   (default: 1 8 32)
# Results: logs/attention/<UTC>-prefill-b<batch>/
#   source_hashes.json, gpu_telemetry.jsonl, clock_check.json, cpu_time.json,
#   latency.json, console.log
# DIFFKV_GPU picks the GPU (default 0).
# DIFFKV_NCU_KERNEL=<regex> runs under Nsight Compute (--set full, as root, see
# run_docker.sh) and profiles DIFFKV_NCU_COUNT (default 1) launches of the
# matching kernels after DIFFKV_NCU_SKIP (default 0) launches; the directory
# gets the suffix -ncu and also kernels.ncu-rep and kernels_details.txt. Its
# latency.json is not a timing result.
set -euo pipefail

batches=("$@")
if [ ${#batches[@]} -eq 0 ]; then
  batches=(1 8 32)
fi
image=${DIFFKV_CONTAINER_IMAGE:-sha256:ca18e2f9134073dc2cc5d9899d7d9974e321008ecc4ab7745477eeb5508befe8}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
evaluations=$(dirname -- "$here")
diffkv=$(dirname -- "$evaluations")
run_docker=("bash" "$diffkv/run_docker.sh")
extra=()
if [ -n "${DIFFKV_PROMPTS:-}" ]; then
  # shellcheck disable=SC2206
  extra+=(--prompts ${DIFFKV_PROMPTS})
fi
if [ "${DIFFKV_PREFILL_PARTS:-0}" = 1 ]; then
  extra+=(--parts)
fi
suffix=
if [ -n "${DIFFKV_CONFIGS:-}" ]; then
  # shellcheck disable=SC2206
  extra+=(--configs ${DIFFKV_CONFIGS})
  suffix="-${DIFFKV_CONFIGS// /-}"
fi
sys_admin=0
if [ -n "${DIFFKV_NCU_KERNEL:-}" ]; then
  suffix="$suffix-ncu"
  sys_admin=1
fi

for batch in "${batches[@]}"; do
  run_dir="$evaluations/logs/attention/$(date -u +%Y%m%dT%H%M%SZ)-prefill-b$batch$suffix"
  mkdir -p "$run_dir"
  profiler=()
  if [ -n "${DIFFKV_NCU_KERNEL:-}" ]; then
    profiler=(ncu --set full --kernel-name "regex:$DIFFKV_NCU_KERNEL"
      --launch-skip "${DIFFKV_NCU_SKIP:-0}" --launch-count "${DIFFKV_NCU_COUNT:-1}"
      --force-overwrite --export "$run_dir/kernels")
  fi
  "${run_docker[@]}" python "$evaluations/utils/record_source_hashes.py" \
    --output "$run_dir/source_hashes.json"
  status=0
  bash "$evaluations/utils/lock_clock.sh" "$run_dir" \
    env DIFFKV_SYS_ADMIN=$sys_admin "${run_docker[@]}" \
    env FLASHINFER_EXTRA_CUDAFLAGS=--compress-mode=none \
    "${profiler[@]}" python "$here/benchmark_prefill.py" --batch "$batch" \
    "${extra[@]}" --output "$run_dir/latency.json" \
    > "$run_dir/console.log" 2>&1 || status=$?
  if [ "$sys_admin" = 1 ]; then
    # The profiled run executed as root; return its files to the caller.
    docker run --rm -v "$run_dir:$run_dir" "$image" \
      chown -R "$(id -u):$(id -g)" "$run_dir"
  fi
  if [ "$status" -ne 0 ]; then
    echo "$run_dir: lock_clock.sh exit $status (see console.log)" >&2
    exit "$status"
  fi
  if [ -n "${DIFFKV_NCU_KERNEL:-}" ]; then
    "${run_docker[@]}" ncu --import "$run_dir/kernels.ncu-rep" --page details \
      > "$run_dir/kernels_details.txt" 2>> "$run_dir/console.log"
  fi
  echo "$run_dir"
done
