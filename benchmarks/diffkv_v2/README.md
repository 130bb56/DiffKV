# DiffKV-v2 MATH benchmark on vLLM 0.32.0

This harness compares native FP16 attention, `DIFFKV_V2`, and FP8 using the frozen
DeepSeek-R1-Distill-Llama-8B workload from the October week 5 sync. Each backend
runs in a fresh process. The harness uses vLLM's current `LLMEngine.add_request`
and `step` APIs without monkeypatching or adding CUDA synchronizations to timing.
The engine integration lives in `vllm/diffkv_v2`, with explicit scheduler and
model-runner hooks in the upstream files.

The bundled `fixtures/math50.json` preserves the exact 50 prompt token ID lists,
request order, references, and seed-42 selection from the archived greedy run.
`fixtures/lengths50.json` preserves the corresponding per-request output lengths:
174,055 tokens total, 934 minimum, 8192 maximum. Both fixtures record their source
hashes. No prompt tokenization or random sampling occurs during a run.

## Runs

Run from the repository root in the environment containing this vLLM checkout.
The model checkpoint is already local in the example. Replace output paths for
each new run, since the scripts refuse to overwrite existing results.

Freeze the implementation before final timing runs. Record a source manifest
before each process starts, since the commit ID alone omits uncommitted and
untracked bringup files:

```bash
.venv/bin/python benchmarks/diffkv_v2/source_manifest.py \
  --output evaluation-logs/vllm032-math/source-before-fp16.json
```

The manifest hashes custom Python/CUDA code, the frozen benchmark fixtures, and
modified upstream hooks. Both final timing runs should have the same
`source_tree_sha256`. A manifest captured after a process starts cannot establish
the exact source bytes that process imported.

Validate inputs without importing torch or vLLM or starting a GPU job:

```bash
.venv/bin/python benchmarks/diffkv_v2/run.py \
  --backend fp16 --model .cache/models/DeepSeek-R1-Distill-Llama-8B \
  --mode length_replay --lengths-from benchmarks/diffkv_v2/fixtures/lengths50.json \
  --output /tmp/math50-validation.json --validate-only
```

Run identical-work throughput, sequentially on the same GPU:

```bash
.venv/bin/python benchmarks/diffkv_v2/run.py \
  --backend fp16 --gpu 0 --model .cache/models/DeepSeek-R1-Distill-Llama-8B \
  --mode length_replay --lengths-from benchmarks/diffkv_v2/fixtures/lengths50.json \
  --output evaluation-logs/vllm032-math/fp16-fixed.json
.venv/bin/python benchmarks/diffkv_v2/run.py \
  --backend diffkv --gpu 0 --model .cache/models/DeepSeek-R1-Distill-Llama-8B \
  --mode length_replay --lengths-from benchmarks/diffkv_v2/fixtures/lengths50.json \
  --output evaluation-logs/vllm032-math/diffkv-fixed.json
.venv/bin/python benchmarks/diffkv_v2/analyze.py \
  --workload benchmarks/diffkv_v2/fixtures/math50.json \
  --fp16 evaluation-logs/vllm032-math/fp16-fixed.json \
  --diffkv evaluation-logs/vllm032-math/diffkv-fixed.json \
  --output evaluation-logs/vllm032-math/throughput.json
```

Correctness uses separate `--mode natural` runs, with no `--lengths-from`.
Both modes use greedy generation, `temperature=0`, `top_p=1`, seed 42. Natural
runs stop at the checkpoint's EOS IDs, capped at 8192 generated tokens. Replay
ignores EOS and generates precisely each request's frozen length. Never grade
replay outputs as correctness samples.

Run the same 50-request subset for the initial correctness comparison:

```bash
.venv/bin/python benchmarks/diffkv_v2/run.py \
  --backend fp16 --gpu 0 --model .cache/models/DeepSeek-R1-Distill-Llama-8B \
  --mode natural --output evaluation-logs/vllm032-math/fp16-natural50.json
.venv/bin/python benchmarks/diffkv_v2/run.py \
  --backend diffkv --gpu 0 --model .cache/models/DeepSeek-R1-Distill-Llama-8B \
  --mode natural --output evaluation-logs/vllm032-math/diffkv-natural50.json
.venv/bin/python benchmarks/diffkv_v2/analyze.py \
  --workload benchmarks/diffkv_v2/fixtures/math50.json \
  --fp16 evaluation-logs/vllm032-math/fp16-natural50.json \
  --diffkv evaluation-logs/vllm032-math/diffkv-natural50.json \
  --output evaluation-logs/vllm032-math/correctness50.json
```

For optional full MATH-500 correctness, pass the existing frozen 500-request workload and
`--expected-count 500` to both `run.py` and `analyze.py`, for example:

```bash
.venv/bin/python benchmarks/diffkv_v2/run.py \
  --backend fp16 --gpu 0 --model .cache/models/DeepSeek-R1-Distill-Llama-8B \
  --workload evaluation-logs/r1-math500-greedy-20261024/workload.json \
  --expected-count 500 --mode natural \
  --output evaluation-logs/vllm032-math/fp16-natural500.json
```

Repeat with `--backend diffkv` and another output file, then analyze the two
natural results with the same workload and expected count. A 50-request input
cannot silently pass as the 500-question evaluation. The sync's 80.2% and 77.2%
accuracy figures refer to 500 questions.

## FP8 baseline

`--backend fp8` selects `kv_cache_dtype=fp8_e4m3`, `attention_backend=FLASHINFER`.
It uses checkpoint per-tensor K/V scales, or 1.0 when absent, without calibration.
FP8 retains FP16 weights and every other engine and generation setting above.
Runtime metadata records the requested format and scale
policy, resolved cache dtype, available attention-layer backend names, and FP8
loaded tensor scales. An empty backend-name list means runtime layer introspection
was unavailable, not that backend selection was independently verified.

Run FP8 sequentially on the same GPU, using the same `--mode natural`
or `--mode length_replay --lengths-from benchmarks/diffkv_v2/fixtures/lengths50.json`
arguments. Hardware support must be verified on the target GPU before timing.
Pairwise analysis accepts explicit labels, for example:

```bash
.venv/bin/python benchmarks/diffkv_v2/analyze.py \
  --workload benchmarks/diffkv_v2/fixtures/math50.json \
  --reference evaluation-logs/vllm032-math/fp16-fixed.json \
  --candidate evaluation-logs/vllm032-math/fp8-fixed.json \
  --candidate-backend fp8 --output evaluation-logs/vllm032-math/fp8-vs-fp16.json
.venv/bin/python benchmarks/diffkv_v2/analyze.py \
  --workload benchmarks/diffkv_v2/fixtures/math50.json \
  --reference evaluation-logs/vllm032-math/diffkv-fixed.json \
  --reference-backend diffkv \
  --candidate evaluation-logs/vllm032-math/fp8-fixed.json \
  --candidate-backend fp8 --output evaluation-logs/vllm032-math/fp8-vs-diffkv.json
```

`--fp16` and `--diffkv` remain aliases for reference and candidate paths, with
default labels `fp16` and `diffkv`. Throughput ratio always means candidate/reference.
Only each label's intended KV dtype and attention backend may differ; all other
engine settings, generation settings, prompts, and fixed output lengths must match.

## Measurements and grading

All requests enter before the first measured step. The engine runs in-process
to preserve this burst boundary. Model loading and 32-token warmup are excluded.
Throughput is total generated output tokens divided by time from the common
arrival, before registration, to the final completed output. Request timings
are observed at `engine.step()` returns, and include scheduler waiting. TTFT,
TPOT, E2E are reported with means and nearest-rank percentiles. Scheduler
preemptions and peak KV usage are recorded through the stat-logger interface.
Scheduler KV usage is explicitly labeled and is not assumed to equal a custom
backend's physical packed-pool occupancy.

Common settings are FP16, TP=1, max model length 9216, batched-token budget 9216,
max sequences 64, memory utilization 0.9, eager execution, chunked prefill on,
async scheduling off, prefix caching off, speculative decoding off. The model's
generation defaults
are disabled, while its EOS IDs are preserved explicitly. The FP16 backend is
selected by vLLM, and the compressed run selects `DIFFKV_V2`. Keep GPU clocks and
temperature comparable to the archived 1395 MHz measurements when comparing
against those historical numbers.

The initial DiffKV allocator admits at most one new or recomputed request per
step to reserve contiguous temporary prefill storage safely. Already running
decode requests remain batched. This limit applies only to DiffKV and is recorded
in result metadata, so its actual serving overhead remains in the comparison.
`CUDA_MODULE_LOADING=LAZY` is set before importing torch unless already supplied,
and its effective value is recorded for the local driver/runtime combination.
Cache and configuration directories default to writable paths under the checkout;
usage reporting is disabled. `VLLM_ENABLE_V1_MULTIPROCESSING=0` is set explicitly.

Output token IDs are decoded after timing with the checkpoint's original
`tokenizer.json`, whose hash must match the frozen workload. This preserves the
ByteLevel decoder used by the archived tokenizer: the current Transformers
LlamaTokenizer replaces it with a SentencePiece decoder and emits literal `Ġ`
markers. The result retains `engine_output_text` beside canonical `output_text`
and records the number of text mismatches. This changes neither the frozen input
IDs nor generated IDs. Canonical output text is used for grading.

`analyze.py` rejects failed or partial runs, different prompts, engine settings,
commits, and unequal per-request output lengths for replay. Natural comparisons
grade text after the final `</think>` with the archived parser and symbolic
grader, without a last-number fallback. Both raw grades and the sync's explicit
boxed-set, unit, ordinal corrections are recorded. Grading errors are reported
separately and invalidate aggregate accuracy. The optional CPU grading
dependencies are listed in `grading-requirements.txt`; generation does not
import them. Vendor source hashes are recorded in each accuracy report.

Run the CPU contract checks without pytest or a GPU:

```bash
.venv/bin/python -m unittest discover -s benchmarks/diffkv_v2 -p test_harness.py -v
```
