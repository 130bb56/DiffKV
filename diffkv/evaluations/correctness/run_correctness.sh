#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Correctness: generate until EOS with one KV backend, then score.
# math500: the 50-problem sample or all 500 problems, output limit 8192 by
# default (at 8192, 86 of 500 fp16 outputs stopped at the limit).
# gpqa: the 197 questions of benchmarks/gpqa/gpqa_diamond_197.json
# (gpqa-diamond-127 is left out: its prompt is longer than 1024 tokens), output
# limit 16384 by default (at 8192, 130 of 197 fp8 outputs stopped at the limit).
# Engine and generation settings are the same for both benchmarks except the
# output limit and max_model_len (output limit + 1024). An output limit above
# 8192 needs chunked prefill on, since max_num_batched_tokens stays 9216.
# Usage: run_correctness.sh math500 <fp16|diffkv|fp8> <sample50|all500> \
#            [output limit, default 8192]
#        run_correctness.sh gpqa <fp16|diffkv|fp8> [output limit, default 16384]
# Results: logs/correctness/<UTC>-<sample50|all500>-<backend>/ (math500, 8192)
#          logs/correctness/<UTC>-<sample50|all500>-osl<output limit>-<backend>/
#          (math500, other limits)
#          logs/correctness/<UTC>-diamond197-osl<output limit>-<backend>/ (gpqa)
#   source_hashes.json, outputs.json, console.log, grades.json
# DIFFKV_GPU picks the GPU (default 0).
# DIFFKV_CHUNKED_PREFILL=1 splits prompts over steps and adds -chunked to the
# directory name. DiffKV rejects chunked prompts, so this is for fp16 and fp8.
set -euo pipefail

benchmark=${1:?benchmark: math500 or gpqa}
backend=${2:?backend: fp16, diffkv or fp8}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
evaluations=$(dirname -- "$here")
diffkv=$(dirname -- "$evaluations")
repo=$(dirname -- "$diffkv")
run_docker=("bash" "$diffkv/run_docker.sh")
case $benchmark in
  math500)
    problem_set=${3:?problem set: sample50 or all500}
    case $problem_set in
      sample50) count=50 ;;
      all500) count=500 ;;
      *) echo "unknown problem set: $problem_set" >&2; exit 2 ;;
    esac
    max_tokens=${4:-8192}
    workload=(--expected-count "$count" --max-tokens "$max_tokens")
    if [ "$max_tokens" = 8192 ]; then
      name=$problem_set-$backend
    else
      name=$problem_set-osl$max_tokens-$backend
    fi
    ;;
  gpqa)
    max_tokens=${3:-16384}
    workload=(--workload "$evaluations/benchmarks/gpqa/gpqa_diamond_197.json"
      --expected-count 197 --max-tokens "$max_tokens")
    name=diamond197-osl$max_tokens-$backend
    ;;
  *) echo "unknown benchmark: $benchmark" >&2; exit 2 ;;
esac
if [ "${DIFFKV_CHUNKED_PREFILL:-0}" = 1 ]; then
  workload+=(--chunked-prefill)
  name=$name-chunked
fi
run_dir="$evaluations/logs/correctness/$(date -u +%Y%m%dT%H%M%SZ)-$name"
mkdir -p "$run_dir"

"${run_docker[@]}" python "$evaluations/utils/record_source_hashes.py" \
  --output "$run_dir/source_hashes.json"
"${run_docker[@]}" env FLASHINFER_EXTRA_CUDAFLAGS=--compress-mode=none \
  python "$evaluations/generation/generate.py" --backend "$backend" \
  --model "$repo/models/DeepSeek-R1-Distill-Llama-8B" \
  "${workload[@]}" --mode natural --output "$run_dir/outputs.json" \
  > "$run_dir/console.log" 2>&1
"${run_docker[@]}" python "$here/score_outputs.py" --benchmark "$benchmark" \
  --outputs "$run_dir/outputs.json" --output "$run_dir/grades.json"
echo "$run_dir"
