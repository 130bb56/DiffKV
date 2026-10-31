#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# MATH-500 correctness: generate until EOS with one KV backend, then score.
# Usage: run_correctness.sh <fp16|diffkv|fp8> <sample50|all500>
# Results: math500/logs/correctness/<UTC>-<problem set>-<backend>/
#   source_hashes.json, outputs.json, console.log, grades.json
# DIFFKV_GPU picks the GPU (default 0).
set -euo pipefail

backend=${1:?backend: fp16, diffkv or fp8}
problem_set=${2:?problem set: sample50 or all500}
case $problem_set in
  sample50) count=50 ;;
  all500) count=500 ;;
  *) echo "unknown problem set: $problem_set" >&2; exit 2 ;;
esac

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
math500=$(dirname -- "$here")
diffkv=$(cd -- "$math500/../.." && pwd)
repo=$(dirname -- "$diffkv")
run_docker=("bash" "$diffkv/run_docker.sh")
run_dir="$math500/logs/correctness/$(date -u +%Y%m%dT%H%M%SZ)-$problem_set-$backend"
mkdir -p "$run_dir"

"${run_docker[@]}" python "$math500/record_source_hashes.py" \
  --output "$run_dir/source_hashes.json"
"${run_docker[@]}" env FLASHINFER_EXTRA_CUDAFLAGS=--compress-mode=none \
  python "$math500/generate.py" --backend "$backend" \
  --model "$repo/models/DeepSeek-R1-Distill-Llama-8B" \
  --expected-count "$count" --mode natural --output "$run_dir/outputs.json" \
  > "$run_dir/console.log" 2>&1
"${run_docker[@]}" python "$here/score_outputs.py" \
  --outputs "$run_dir/outputs.json" --output "$run_dir/grades.json"
echo "$run_dir"
