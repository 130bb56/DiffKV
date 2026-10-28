# DiffKV full decode attention optimization

Model-level follow-up: the same installed binary (`b28c5e92...`) failed the
existing MATH-50 accuracy screen. After reviewing answer-format extraction
errors, FP16 scored 42/50, original DiffKV-v1 40/50, and this optimized build
38/50. Its -8 percentage-point difference from FP16 misses the -5-point rule;
the mean output-length ratio, 0.966, is within the required range. Kernel tests
below remain valid within their scope, but do not establish model-level
accuracy preservation. The user subsequently authorized resuming the 14-case GPU1 E2E run; the accuracy failure remains unresolved.
See the [accuracy audit](/home/130bb56/DiffKV/evaluation-logs/r1-math50-kernel-correctness-20261024/analysis.md)
and [binary provenance](/home/130bb56/DiffKV/evaluation-logs/r1-math50-kernel-correctness-20261024/provenance-audit.json).

2026-10-24. The selected `pvpair + split4float` implementation accelerates the complete decode attention CUDA operation, including main, reduction, and importance-score updates. Against the original DiffKV-v1 binary before the load-alignment and register-GQA changes, the six-context comparison measures 2.36-2.58x at contexts 2048-8192 and 1.74x at context 512. Cache writing, allocation, reset, and readback are outside the timed operation. These are synthetic single-layer results, with no model E2E speedup claim.

## Measurement

Both binaries used physical GPU1, RTX 3090, an SM clock locked at 1395 MHz, batch 8, KV heads 8, GQA4, D128, FP16, K8V4/K4V2, and one quantization chunk. Memory clock was observed at 9501 MHz under load and was not locked. Each row is one CUDA-event trial averaging 200 calls after at least 100 warmup calls and 0.3 seconds of warmup. A single trial does not establish run-to-run variation. Earlier measurements with unlocked clocks are not mixed into this table. [Clock record](/home/130bb56/DiffKV/logs/performance/kernel-full-2x-20261024/clock.json), [comparison](/home/130bb56/DiffKV/logs/performance/kernel-full-2x-20261024/selected-wide-comparison.json).

| Context | Original, us | Selected, us | Speedup |
|---:|---:|---:|---:|
| 512 | 65.106 | 37.427 | 1.740x |
| 2048 | 204.774 | 82.601 | 2.479x |
| 3072 | 282.004 | 117.432 | 2.401x |
| 4096 | 361.697 | 153.057 | 2.363x |
| 6000 | 532.045 | 206.054 | 2.582x |
| 8192 | 691.523 | 280.991 | 2.461x |

## Changes after the first register-GQA implementation

The existing main kernel keeps partition 240, 128 threads, `__launch_bounds__(128, 4)`, and four-query reuse of decoded K/V. This follow-up changes three areas:

- **PV token-pair hoisting:** Load each pair's packed V payload once, then load its token metadata and four query weights once per token before visiting the dimension packs. Per-output token accumulation order and the cache format remain unchanged.
- **Split score reduction:** Four CTAs share each sequence/KV-head's score range, increasing the B8/KV8 reduction grid from 64 to 256 CTAs. Each repeats the small partition-statistics calculation. Only tile zero writes the output, with two query accumulators interleaved for independent work. Each token has one owner that rescales four probabilities in registers and adds their maximum to its cached importance score. The reducer no longer writes globally rescaled `tmp_scores` or rereads them for the maximum. This buffer is scratch with no downstream consumer; every active entry is overwritten by main before use. Four warps compute query statistics, followed by one CTA barrier. More than 32 partitions use a strided path.
- **FP32 epsilon:** Change `(sum + 1e-6) * factor` to `(sum + 1e-6f) * factor`. The former generated FP64 addition and multiplication in the rejected fused reducer. Selected main and reduce SASS contain zero FP64 instructions.

The [main header](/home/130bb56/DiffKV/csrc/attention/sparse_gqa_register_kernels.cuh), [reducer header](/home/130bb56/DiffKV/csrc/attention/sparse_gqa_reduce_kernels.cuh), and [launcher](/home/130bb56/DiffKV/csrc/attention/sparse_gqa_attention_kernels.cu) contain the changes. Main, reducer, and Python workspace use P240. Unsupported configurations retain the legacy path, and `use_gqa_reuse=False` selects it explicitly.

## Diagnostic breakdown

A separate profiler capture recorded 20 complete calls at B8/C3072 under the same clock setting. It compares the previously installed register-GQA version with the selected implementation. These are profiler kernel means, separate from the unprofiled CUDA-event table above. [Previous diagnostic](/home/130bb56/DiffKV/logs/performance/kernel-full-2x-20261024/current-diagnostic.json), [selected diagnostic](/home/130bb56/DiffKV/logs/performance/kernel-full-2x-20261024/final-diagnostic.json).

| Kernel | Previous register GQA, us | Selected, us |
|---|---:|---:|
| Main | 108.617 | 106.179 |
| Reduce and score update | 29.891 | 10.451 |
| Kernel sum | 138.508 | 116.630 |

## Correctness and build evidence

The six contexts above ran 32 calls with changing queries and seed 17. Each first output passed an independent CPU FP64 full-softmax oracle. Baseline comparison passed output and accumulated-score tolerances: maximum output difference was 0.000244140625, and maximum accumulated-score difference was 5.960464477539063e-08. Outputs and floating scores are not bitwise identical.

At steps 1, 4, 16, and 32, retained KV positions, high/low lengths, score layout, block tables, and all non-score cache bytes matched exactly. Applying the actual cache writer with policies `[0, 0]`, `[0.02, 1]`, and `[1, 1]` also produced exactly matching retained positions and cache metadata. The [comparison artifact](/home/130bb56/DiffKV/logs/performance/kernel-full-2x-20261024/selected-wide-comparison.json) records these checks separately from floating-point tolerances.

The full-dispatch build's FP16/BF16 main and reducer SASS match the selected restricted build exactly, with zero FP64 instructions in all four functions. The full build passed 15 regression tests and four alignment/tail guard tests. [SASS comparison](/home/130bb56/DiffKV/logs/performance/kernel-full-2x-20261024/full-sass-comparison.json), [regression log](/home/130bb56/DiffKV/logs/performance/kernel-full-2x-20261024/final-regression.log), [guard log](/home/130bb56/DiffKV/logs/performance/kernel-full-2x-20261024/final-guard.log).

The full production binary is installed, SHA256 `b28c5e9251f14deb8041c4a538597522467b7463ee42e0fea483f6a66ba2dc24`. The [build manifest](/home/130bb56/DiffKV/logs/performance/kernel-full-2x-20261024/final/build-manifest.json) and [result summary](/home/130bb56/DiffKV/logs/performance/kernel-full-2x-20261024/result-summary.json) record provenance and validation. The 14-case E2E evaluation is pending and is not represented by these kernel measurements. The [earlier report](/home/130bb56/DiffKV/docs/attention/register_gqa_optimization.md) preserves the first register-GQA experiment and its separate model measurement.

## Reproduce the recorded comparison

The captures contain binary and fixture hashes. Recheck their outputs and cache decisions without rerunning GPU measurements:

```bash
.cache/baselines/fp16-env/bin/python benchmarks/validate_diffkv_kernel_candidate.py compare \
  --baseline logs/performance/kernel-full-2x-20261024/original-wide.pt \
  --candidate logs/performance/kernel-full-2x-20261024/selected-wide.pt \
  --allow-operation-change --allow-float-roundoff \
  --output /tmp/full-decode-comparison.json
```
