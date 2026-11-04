#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Run one measurement command with the GPU SM clock locked to its base clock and
# no thermal throttling.
#   1. stop if another process uses the GPU
#   2. record GPU telemetry every 200 ms (record_telemetry.py)
#   3. lock the SM clock to the base clock from `nvidia-smi base-clocks`
#   4. cool down as origin/main benchmarks/run_kernel_e2e_grid.py does: wait at
#      least DIFFKV_COOL_MIN_S (default 30) seconds and until the SM clock is
#      locked, the GPU is at most DIFFKV_COOL_MAX_C (default 70) C and no thermal
#      or power limit is active; give up after DIFFKV_COOL_TIMEOUT_S (default
#      1800) seconds
#   5. run the command while check_throttle.py watch follows the telemetry,
#      and kill it (the container named by DIFFKV_CIDFILE when the command
#      starts one through run_docker.sh) as soon as a sample is throttled or off
#      the locked clock, or once GPU utilization stays 0% for DIFFKV_STUCK_S
#      (default 600) seconds
#   6. check the run's interval (check_throttle.py check); if it was killed for
#      throttling or is not valid, go back to 3, at most DIFFKV_LOCK_ATTEMPTS
#      (default 3) runs in total
# The clock is unlocked on exit.
# Usage: lock_clock.sh <run_dir> <command> [args...]
# Results in run_dir: gpu_telemetry.jsonl, clock_check.json (one verdict per
# run), throttled.json (the sample that killed the last run) or stuck.json
# Exit status: the command's status if it fails, 3 if the GPU is in use or does
# not cool down, 4 if every run was throttled, 5 if the command was stuck.
# DIFFKV_GPU picks the GPU (default 0).
set -euo pipefail

run_dir=${1:?run directory}
shift
gpu=${DIFFKV_GPU:-0}
min_cool=${DIFFKV_COOL_MIN_S:-30}
max_temp=${DIFFKV_COOL_MAX_C:-70}
cool_timeout=${DIFFKV_COOL_TIMEOUT_S:-1800}
attempts=${DIFFKV_LOCK_ATTEMPTS:-3}
stuck_s=${DIFFKV_STUCK_S:-600}
image=${DIFFKV_CONTAINER_IMAGE:-sha256:ca18e2f9134073dc2cc5d9899d7d9974e321008ecc4ab7745477eeb5508befe8}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
diffkv=$(cd -- "$here/../.." && pwd)
run_docker=(env DIFFKV_GPU="$gpu" DIFFKV_SYS_ADMIN=0 bash "$diffkv/run_docker.sh")
telemetry="$run_dir/gpu_telemetry.jsonl"

admin_smi() {
  docker run --rm --gpus "\"device=$gpu\"" --cap-add=SYS_ADMIN "$image" \
    nvidia-smi -i 0 "$@" > /dev/null
}

used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
apps=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
if [ -n "$apps" ] || [ "$used" -gt 100 ]; then
  echo "GPU $gpu is in use ($used MiB); set DIFFKV_GPU to a free GPU" >&2
  exit 3
fi
clock=$(nvidia-smi base-clocks -i "$gpu" | awk '/Graphics/ {print $2 + 0}')

mkdir -p "$run_dir"
"${run_docker[@]}" python "$here/record_telemetry.py" --output "$telemetry" &
recorder=$!
cleanup() {
  kill "$recorder" 2>/dev/null || true
  wait "$recorder" 2>/dev/null || true
  admin_smi -rgc || true
}
trap cleanup EXIT
until [ -s "$telemetry" ]; do
  kill -0 "$recorder" 2>/dev/null || { echo "telemetry did not start" >&2; exit 3; }
  sleep 1
done

for attempt in $(seq 1 "$attempts"); do
  admin_smi -lgc "$clock,$clock"
  "${run_docker[@]}" python "$here/check_throttle.py" wait \
    --telemetry "$telemetry" --clock "$clock" --min-seconds "$min_cool" \
    --max-temp "$max_temp" --timeout "$cool_timeout" || exit 3
  echo "lock_clock: run $attempt of $attempts at $clock MHz"
  cidfile="$run_dir/command.cid"
  rm -f "$cidfile" "$run_dir/throttled.json" "$run_dir/stuck.json"
  start=$(date +%s.%N)
  DIFFKV_CIDFILE="$cidfile" "$@" &
  command=$!
  "${run_docker[@]}" python "$here/check_throttle.py" watch \
    --telemetry "$telemetry" --start "$start" --clock "$clock" \
    --idle-seconds "$stuck_s" --throttle-marker "$run_dir/throttled.json" \
    --stuck-marker "$run_dir/stuck.json" &
  watcher=$!
  killed=
  while kill -0 "$command" 2>/dev/null; do
    if [ -e "$run_dir/throttled.json" ]; then
      killed="throttled or off clock"
    elif [ -e "$run_dir/stuck.json" ]; then
      killed="GPU idle for $stuck_s s"
    fi
    if [ -n "$killed" ]; then
      echo "lock_clock: $killed, killing the command"
      if [ -s "$cidfile" ]; then
        docker kill "$(cat "$cidfile")" > /dev/null || true
      else
        kill "$command" || true
      fi
      break
    fi
    sleep 1
  done
  status=0
  wait "$command" || status=$?
  end=$(date +%s.%N)
  kill "$watcher" 2>/dev/null || true
  wait "$watcher" 2>/dev/null || true
  rm -f "$cidfile"
  [ ! -e "$run_dir/stuck.json" ] || exit 5
  [ -n "$killed" ] || [ "$status" -eq 0 ] || exit "$status"
  sleep 1
  if "${run_docker[@]}" python "$here/check_throttle.py" check \
      --telemetry "$telemetry" --start "$start" --end "$end" --clock "$clock" \
      --output "$run_dir/clock_check.json" && [ -z "$killed" ]; then
    exit 0
  fi
  echo "lock_clock: run $attempt was throttled or off clock"
done
exit 4
