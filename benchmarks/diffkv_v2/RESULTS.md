# DiffKV-v2 bringup results on vLLM 0.32.0

These measurements use DeepSeek-R1-Distill-Llama-8B and the frozen 50-question
MATH-500 subset. FP8 reaches 549.834 output tokens/s, versus DiffKV's 437.733 and
FP16's 301.529. FP8 delivers 1.256x DiffKV's throughput and lower mean TTFT in this
fixed-work burst. DiffKV's attention kernel beats FP16 in eight of nine measured
cases; FP8 is faster than both in all nine cases.
After the documented grading corrections, FP8 scores 40/50, DiffKV 39/50, and
FP16 38/50 in natural-EOS generation. These 50-question scores do not establish
a general accuracy advantage or reproduce the sync's full 500-question result.

## Setup

Source baseline: `v0.32.0`, commit `5df63a7203cdb9328d824a27b682e77c7b9928be`.
Serving uses one RTX 3090, GPU 0, with a 1395 MHz graphics clock, FP16 weights,
TP=1, eager execution, chunked prefill, synchronous scheduling, no prefix cache,
and no speculative decoding. Native attention resolves to FlashAttention 2
with FP16 KV; DiffKV uses K8V4/K4V2, thresholds `[0.02, 1.0]`, and a 64-token buffer.
FP8 uses FlashInfer, E4M3 KV, and verified K/V scales of 1.0. All three use model
length 9216, batched-token budget 9216, maximum 64 sequences,
memory utilization 0.9, and 32-token warmup. See [environment](ENVIRONMENT.md)
and [commands and metric definitions](README.md).

## Natural-EOS correctness

Greedy generation uses temperature 0, top_p 1, EOS 128001, and an 8192-token cap.
Grading uses canonical checkpoint-tokenizer text after the final `</think>`,
the archived boxed-answer/SymPy grader, and explicit decoration/set corrections.

| Metric | FP16 | DiffKV-v2 | FP8 FlashInfer |
| --- | ---: | ---: | ---: |
| Raw correct | 36/50 (72%) | 38/50 (76%) | 38/50 (76%) |
| Correct after documented corrections | 38/50 (76%) | 39/50 (78%) | 40/50 (80%) |
| Grading errors | 0 | 0 | 0 |
| Missing extracted answers | 10 | 10 | 9 |
| Outputs reaching the length cap | 11 | 10 | 9 |
| Generated tokens | 185,093 | 180,644 | 162,419 |
| Preemptions | 47 | 0 | 0 |

FP8 answers one more question correctly than DiffKV, +2 percentage points, and
two more than FP16, +4 percentage points, on this fixed sample. This small sample
does not establish general accuracy superiority. Different natural output lengths
make these runs unsuitable for an identical-work throughput comparison.
Local artifacts: [FP16/DiffKV grading and per-question outcomes](../../evaluation-logs/vllm032-bringup/correctness50.json), [FP8 grading](../../evaluation-logs/vllm032-bringup/fp8-correctness-vs-fp16.json), [FP16 outputs](../../evaluation-logs/vllm032-bringup/fp16-natural50.json), [DiffKV outputs](../../evaluation-logs/vllm032-bringup/diffkv-natural50.json), [FP8 outputs](../../evaluation-logs/vllm032-bringup/fp8-natural50.json).

## Fixed-work serving

All 50 requests arrive together. All three backends replay the archived per-request
output lengths, totaling 174,055 tokens, with `ignore_eos=True`. This is an
in-process LLMEngine burst measurement. It does not measure Poisson arrivals
or sustained online-serving SLOs. Model loading and warmup are excluded.

| Metric | FP16 | DiffKV-v2 | FP8 FlashInfer |
| --- | ---: | ---: | ---: |
| Generated tokens | 174,055 | 174,055 | 174,055 |
| Total completion time, s | 577.241 | 397.628 | 316.559 |
| Output throughput, token/s | 301.529 | 437.733 | 549.834 |
| Throughput relative to FP16 | 1.000x | 1.452x | 1.823x |
| Mean E2E, s | 233.431 | 179.314 | 143.966 |
| Mean TTFT, s | 1.466 | 3.987 | 1.460 |
| Mean TPOT, ms | 77.320 | 51.579 | 41.902 |
| Preemptions | 68 | 0 | 0 |
| Maximum observed KV-pool occupancy, % | 100.000 | 63.640 | 93.582 |

FP8 is faster than DiffKV in this serving workload: throughput is 25.6% higher,
with lower mean E2E, TTFT, and TPOT. Runs used the same physical GPU 0 at 1395 MHz,
frozen prompts, per-request output lengths, warmup, and common engine settings;
the intended changes are KV representation and attention backend. Both FP8 and
DiffKV completed without preemption. This establishes the result for this finite
burst and these implementations, not a sustained online-serving comparison.

DiffKV admits one new/recomputed prompt per step while batching existing decodes.
This admission cost is included in the measured TTFT. KV occupancy is sampled
after each step, excluding temporary prefill
storage already released before reporting; it is not peak total GPU memory.
The first FP16 fixed-work attempt failed with disk exhaustion and is excluded.
The table uses the completed retry. Local artifacts: [FP16/DiffKV comparison](../../evaluation-logs/vllm032-bringup/throughput.json), [FP16 retry](../../evaluation-logs/vllm032-bringup/fp16-fixed50.json), [DiffKV run](../../evaluation-logs/vllm032-bringup/diffkv-fixed50.json), [hardware telemetry](../../evaluation-logs/vllm032-bringup/hardware.csv).
FP8 artifacts: [raw run](../../evaluation-logs/vllm032-bringup/fp8-fixed50.json), [comparison with FP16](../../evaluation-logs/vllm032-bringup/fp8-throughput-vs-fp16.json), [comparison with DiffKV](../../evaluation-logs/vllm032-bringup/fp8-throughput-vs-diffkv.json).

## 8-bit baseline selection

FP8 E4M3 is the additional baseline because the [vLLM study](https://vllm.ai/blog/2026-05-11-turboquant)
recommends FP8 as its default KV quantization choice. Its H100 results do not imply
native FP8 Tensor Core gains on RTX 3090. The [FP8 validation study](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html)
uses uncalibrated scales of 1.0, also the [v0.32 default when checkpoint scales are absent](https://docs.vllm.ai/en/v0.32.0/features/quantization/quantized_kvcache/).
The harness selects FlashInfer for FP8. No global SOTA claim is made.

## Attention kernel latency

All three arms start from the same retained, dequantized DiffKV cache, at head
dimension 128, eight KV heads, GQA factor 4, and 25% high-precision retained tokens.
FP8 additionally quantizes those values to E4M3 with K/V scales 1.0. Its FlashInfer
FA2 decode path uses `use_tensor_cores=True`, FP16 queries and output, and the
serving cache layout. This does not imply native FP8 Tensor Core arithmetic on RTX 3090.
Timing uses CUDA graph events, rotating cold-L2 buffers, forward then reverse
candidate order, and 500 ms per arm/order at 1395 MHz. All values below come from
the new paired runs. Latencies are median microseconds; ratio is DiffKV/FP8.

| Batch | Retained tokens/head | FP16 FA2, us | DiffKV-v2, us | FP8 FlashInfer, us | DiffKV/FP8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 18.887 | 29.514 | 12.356 | 2.389x |
| 1 | 2048 | 34.737 | 32.768 | 20.165 | 1.625x |
| 1 | 8192 | 80.691 | 51.814 | 38.707 | 1.339x |
| 8 | 512 | 51.712 | 36.250 | 26.931 | 1.346x |
| 8 | 2048 | 140.288 | 80.794 | 58.061 | 1.392x |
| 8 | 8192 | 409.702 | 272.896 | 181.555 | 1.503x |
| 50 | 512 | 153.702 | 132.915 | 81.715 | 1.627x |
| 50 | 2048 | 554.394 | 446.362 | 287.744 | 1.551x |
| 50 | 8192 | 2160.742 | 1612.595 | 1098.854 | 1.468x |

FP8 is fastest in all nine measured cases. The batch-50 CUDA-core alternative,
`use_tensor_cores=False`, takes 793.088, 3131.802, and 12333.210 us at lengths
512, 2048, and 8192, respectively, so it is slower than the serving default.
This is a comparison of the measured paths, not an exhaustive tuning search.

Each kernel passes an independent FP64 softmax oracle using its actual
dequantized cache, with `atol=1e-3`, `rtol=3e-4`. FP8's maximum kernel error across
these cases is 0.000259. Separately, its added quantization changes the FP64
reference output by maximum absolute errors of 0.004301 to 0.006808 per case.
Those two errors have different references. The common input already contains
DiffKV's retained, dequantized values, so this test does not measure DiffKV's
compression/eviction loss against the original full KV or end-to-end model accuracy.

DiffKV timing includes main attention, reduction, and score update. Cache writing,
quantization, planning, allocation, compilation, and host transfers are excluded.
These retained-KV timings alone do not establish a full-context or serving speedup.
Local [paired summary](../../logs/bringup/fp8-kernel-latency-summary.json) and raw samples: [batch 1](../../logs/bringup/fp8-kernel-latency-b1.json), [batch 8](../../logs/bringup/fp8-kernel-latency-b8.json), [batch 50](../../logs/bringup/fp8-kernel-latency-b50.json).
The original two-arm measurements remain available as history: [summary](../../logs/bringup/kernel-latency-summary.json), [batch 1](../../logs/bringup/kernel-latency-b1.json), [batch 8](../../logs/bringup/kernel-latency-b8.json), [batch 50](../../logs/bringup/kernel-latency-b50.json).

## Provenance

Validation passed for the native FP64 oracle (eight cases, three repeats),
14 GPU backend tests, 11 CPU admission/scheduler tests, and 10 benchmark-harness
tests. GPU coverage includes chunked prefill, mixed prefill/decode ordering,
page-boundary crossings, preemption/recomputation, and complete cache release.
The staged implementation also passed the repository's applicable pre-commit
hooks. Local evidence: [native checks](../../logs/bringup/native-kernel-check-final-loader.json),
[GPU tests](../../logs/bringup/native-final-loader-pytest.txt),
[CPU scheduler tests](../../.cache/bringup/scheduler-tests-final.log),
[pre-commit](../../.cache/bringup/all-staged-precommit-final.log).

The [FP16 retry manifest](../../evaluation-logs/vllm032-bringup/source-before-fp16-fixed50-retry.json)
and [DiffKV manifest](../../evaluation-logs/vllm032-bringup/source-before-diffkv-fixed50.json)
share source hash `554306b7a7fcd41dbdf03df9c137b1aef36a79724bab06e04c534310b8ca626f`.
The [FP8 fixed-work manifest](../../evaluation-logs/vllm032-bringup/source-before-fp8-fixed50.json)
and [FP8 natural-EOS manifest](../../evaluation-logs/vllm032-bringup/source-before-fp8-natural50.json)
share `7d0a4a4a218a36191e5446f8c9d413e0354fdaeed558703e3e059812462d2905`.
The [source audit](../../evaluation-logs/vllm032-bringup/source-audit-fp8-phase.json)
confirms unchanged engine integration and native kernel bytes; changes add FP8
benchmark support and include the native README in the manifest's coverage.
The native fingerprint changed from `074fbf5fa562426d0a06...` to
`0ba8f325678b5e7eb3e0...` after a loader-only change; CUDA source and object hashes
are identical. [Rebuild evidence](../../logs/bringup/native-loader-rebuild.json)
records the comparison and repeated correctness checks. Raw artifacts linked
here are local files retained outside Git.
