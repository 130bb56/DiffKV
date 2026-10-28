# DiffKV decode attention의 packed K/V load 정렬 최적화

이 문서는 앞선 load 정렬 실험의 기록입니다. 현재 기본 경로에 적용한 후속 변경과 최종 성능은 [GQA register 재사용 최적화](/home/130bb56/DiffKV/docs/attention/register_gqa_optimization.md)를 참고하십시오.

작성일은 2026-10-24, 시간대는 Asia/Seoul입니다. 이 문서는 K64, K64+V32 후보의 구현과 검증 근거를 정리한 작업 기록입니다. 반복 CUDA-event 측정에서는 K64+V32가 1.175~1.187배 빠르고, 검증한 상태와 출력은 baseline과 bitwise 일치했습니다. 검증을 마친 K64+V32 binary를 기본 extension에 적용했습니다. 이후 최적화에서는 커널 시간과 정확성 검증에 집중하고, 모델 전체 성능 측정은 최적화 완료 후 마지막에 수행합니다.

## 목적과 비교 범위

기존 decode main은 8B K vector와 4B V vector를 소스에서 구조체로 읽지만, 구조체의 정렬이 2B여서 컴파일 결과는 여러 개의 16-bit load였습니다. 이 실험에서는 packed payload의 주소와 저장 형식에 맞는 정렬을 명시하여 실제 load 폭을 넓혔습니다.

대상은 DiffKV-v1의 `sparse_paged_attention_wrapper`와 기존 reduce를 합친 decode attention 경로입니다. K8V4/K4V2, quantization chunk 각각 1개, D128, query/KV head 비율 4, B8, KV head 8개를 사용했습니다. 모든 새 GPU 측정은 물리 GPU1, RTX 3090에서 수행했습니다. UUID는 `GPU-feccd811-0bc3-0d68-066b-9bf5bc7cdbb7`입니다. NCU 컨테이너 안에서는 이 GPU만 노출되며 CUDA ordinal은 0입니다.

원자료는 [kernel-wide-load-20261024](/home/130bb56/DiffKV/logs/performance/kernel-wide-load-20261024)에 있습니다. 기존 병목 분석은 [profiling_result_gpt.md](/home/130bb56/DiffKV/docs/attention/profiling_result_gpt.md), [deep SASS 분석](/home/130bb56/DiffKV/logs/performance/attention-pd-20261022/deep-sass-analysis.md)에 있습니다.

## 변경한 코드와 정렬 안전성

변경한 CUDA 소스는 두 파일입니다.

| 위치 | 변경 내용 |
|---|---|
| [cache_utils.h](/home/130bb56/DiffKV/csrc/attention/cache_utils.h:10) | `k_vec_type`에 `alignas(8)`, `v_vec_type`에 `alignas(4)`를 적용했습니다. K64 중간 후보는 K만 변경했습니다. |
| [sparse_attention_kernels.cu](/home/130bb56/DiffKV/csrc/attention/sparse_attention_kernels.cu:1185) | cache base와 page stride가 8B 정렬인지 `TORCH_CHECK`로 확인합니다. Contiguous tensor도 storage offset 때문에 정렬이 어긋날 수 있어 두 조건을 각각 검사합니다. |

CPU C++17 `static_assert`로 실제 헤더 선언의 크기와 정렬을 확인했습니다.

| 타입 | 변경 전 크기, 정렬 | K64+V32 크기, 정렬 |
|---|---|---|
| `k_vec_type` | 8B, 2B | 8B, 8B |
| `v_vec_type` | 4B, 2B | 4B, 4B |
| `quant_meta_type` | 4B, 2B | 4B, 2B |

각 payload의 `data` 시작 offset은 0입니다. 저장 byte 수와 pack 순서는 유지됩니다. 정상 cache page는 6,784B, 즉 int16 원소 3,392개이며, K 주소의 page 내 offset은 int16 4개 단위입니다. V segment는 128B 정렬이고 V offset은 int16 2개 단위입니다. 따라서 base와 page stride의 8B 검사는 high, low K64와 V32 접근을 모두 충족합니다.

이 변경은 load의 정렬 요구를 명시합니다. 실제 instruction 폭은 SASS로 별도 확인했습니다. Dequantization 식, FP32 누산과 reduction 순서, softmax, partition 5,760, 중요도 score 갱신, cache writer는 변경하지 않았습니다. GQA 재사용 경로도 새로 활성화하거나 수정하지 않았습니다.

## NCU에서 확인한 변화

같은 B8, KV8, GQA4, context 3,072 fixture로 main kernel 1개씩 수집했습니다. NCU는 kernel replay, `--cache-control all`, `--clock-control none` 조건입니다. GPU clock을 고정하지 않았으며, 아래 duration은 단일 진단 capture의 값입니다. 반복 측정의 성능 판정과 구분합니다.

| Main 지표 | 기존 baseline | K64 | K64+V32 |
|---|---:|---:|---:|
| NCU duration, us | 227.584 | 224.832 | 189.024 |
| Register/thread | 108 | 108 | 110 |
| Dynamic global-load warp instructions | 2,020,432 | 1,160,896 | 874,384 |
| 전체 warp instructions | 45,203,436 | 46,561,644 | 46,531,448 |
| Global theoretical sectors | 11,772,872 | 5,088,584 | 3,974,216 |
| 실제 L1 global-load sectors | 11,595,751 | 4,951,094 | 3,767,203 |
| DRAM read bytes | 28,653,824 | 28,653,696 | 28,654,848 |
| Long-scoreboard, cycles/issued instruction | 2.9213 | 2.8189 | 1.3068 |
| Eligible warps/scheduler | 0.6345 | 0.6583 | 0.7999 |
| Issue activity, % | 41.80 | 43.45 | 49.46 |
| DRAM peak 대비 throughput, % | 15.80 | 15.98 | 19.01 |

Source counter의 theoretical sector와 실제 DRAM byte는 서로 다른 지표입니다. K64+V32에서 theoretical sector는 66.24%, 실제 L1 global-load sector는 67.51% 감소했지만, DRAM read byte는 약 28.65MB로 같습니다. 따라서 이 변경의 효과를 외부 메모리에서 읽는 byte 수의 감소로 설명하지 않습니다.

SASS에서 K의 `LDG.E.64.CONSTANT`, V의 32-bit `LDG.E.CONSTANT`를 확인했습니다. K64만 적용하면 sector 요청은 크게 줄어도 main duration 감소는 이 단일 capture에서 1.21%에 그쳤습니다. Wider load 이후 bit extraction과 주소 관련 명령이 증가하여 전체 warp instruction 수는 약 3% 늘었습니다. K64+V32에서는 long-scoreboard ratio가 더 낮아지고 eligible warp와 issue activity가 높아졌습니다. 이 관측은 packed load의 요청과 의존성이 개선되었다는 해석을 지지하지만, 각 요인의 독립 시간이나 커널 wall time 비율을 계산한 결과는 아닙니다.

기존 실행에서도 DRAM bandwidth 포화는 관측되지 않았습니다. 개선 후에도 DRAM throughput은 peak의 약 19%입니다. Long-scoreboard sample이 변환이나 unpack consumer에서 관측된다는 사실을 dequantization 산술의 독립 실행 시간으로 해석하지 않습니다.

원자료는 [세 후보 NCU 비교](/home/130bb56/DiffKV/logs/performance/kernel-wide-load-20261024/ncu-comparison-all.json), [K64 상세 해석](/home/130bb56/DiffKV/logs/performance/kernel-wide-load-20261024/ncu-analysis-k64.json), [최종 후보 SASS](/home/130bb56/DiffKV/logs/performance/kernel-wide-load-20261024/sass-k64-v32/selected.sass)에 있습니다. Duration은 NCU details의 `nsecond` 값을 us로 환산했습니다. JSON metric의 단위 문자열만 보고 원래 수치를 초 단위로 해석하지 않도록 주의해야 합니다.

## 정확성 검증

Baseline, K64, K64+V32는 모델을 로드하지 않는 packed-cache fixture에 동일 입력을 사용했습니다. 물리 page 순서는 무작위로 배치했고, high, low 길이와 partial page를 포함했습니다.

| 항목 | 범위와 결과 |
|---|---|
| Context | 512, 2,048, 3,072, 4,096, 6,000, 8,192의 6조건을 확인했습니다. |
| 반복 attention 호출 | 조건마다 query를 바꾸며 32번 호출했습니다. 모든 output tensor가 baseline과 bitwise 일치하고 최대 절대 오차는 0입니다. |
| 누적 중요도 score | 호출 1, 4, 16, 32의 score snapshot이 bitwise 일치했습니다. |
| 실제 cache writer | 누적 score에서 정책 `[0, 0]`, `[0.02, 1]`, `[1, 1]`을 각각 적용했습니다. 선택한 위치, high/low 길이, block table, score, score 외 cache byte가 모두 일치했습니다. |
| 독립 기준값 | 각 조건의 첫 호출에서 full FP64 softmax 기준의 output, token별 최대 확률에 대한 검증을 통과했습니다. |
| 불변 상태 | Attention 호출이 payload, metadata, position, block table, retained length를 변경하지 않았음을 확인했습니다. |

32번의 attention 호출은 고정 packed KV를 소비하는 검증입니다. 매 호출마다 실제 모델 생성과 pruning을 수행한 32 decode step이라는 뜻은 아닙니다. 실제 cache writer 정책은 누적된 score 상태에서 별도로 실행했습니다. 모델 전체 출력과 성능은 아래 모델 측정에서 확인합니다.

두 후보의 [K64 비교](/home/130bb56/DiffKV/logs/performance/kernel-wide-load-20261024/k64/correctness-comparison.json), [K64+V32 비교](/home/130bb56/DiffKV/logs/performance/kernel-wide-load-20261024/k64-v32/correctness-comparison.json)는 모두 `passed`이며 `require_bitwise`가 적용되었습니다.

기존 [decode attention 테스트](/home/130bb56/DiffKV/tests/kernels/test_diffkv_decode_attention.py)의 12개 테스트도 최종 후보 binary에서 통과했습니다. Partial high/low page, high-only/low-only, 5,760 경계 전후와 다중 partition, partial high 뒤의 low partition, ALiBi, softmax 최대값의 thread-group 경계, 지원하지 않는 GQA 비율 거부, CPU packed layout 검증을 포함합니다. [실행 기록](/home/130bb56/DiffKV/logs/performance/kernel-wide-load-20261024/k64-v32/regression-tests.log)을 보존했습니다.

추가 guard 테스트 4개는 정렬된 storage offset 허용, 정렬되지 않은 contiguous storage offset 거부, 정렬되지 않은 page stride 거부, 짧은 vector와 page tail 검증입니다. 모두 통과했으며 [guard 실행 기록](/home/130bb56/DiffKV/logs/performance/kernel-wide-load-20261024/k64-v32/guards.log)에 있습니다.

CPU 테스트 5개도 통과했으며 [실행 기록](/home/130bb56/DiffKV/logs/performance/kernel-wide-load-20261024/cpu-tests.log)을 보존했습니다.

## 성능 결과

### 반복 CUDA-event 측정

최종 측정은 B8, KV8, context 512, 3,072, 8,192에서 baseline, K64+V32, K64+V32, baseline의 ABBA 순서로 실행했습니다. 각 실행은 최소 2초와 2,000회 warmup, trial당 500호출, 10회 trial을 사용했습니다. 한 후보당 두 실행의 20개 trial 중앙값으로 비교했습니다. 시간 범위에는 main, reduce, 중요도 score update가 포함되고 cache writer, reset, allocation, readback은 제외됩니다.

| Context | Baseline median, us | K64+V32 median, us | Speedup | Latency 감소 |
|---:|---:|---:|---:|---:|
| 512 | 48.696 | 41.456 | 1.175배 | 14.87% |
| 3,072 | 221.172 | 187.877 | 1.177배 | 15.05% |
| 8,192 | 552.335 | 465.402 | 1.187배 | 15.74% |

[final-timing.json](/home/130bb56/DiffKV/logs/performance/kernel-wide-load-20261024/final-timing.json)의 상태는 `completed`이며 네 실행의 return code는 모두 0입니다. 각 context에서 candidate의 가장 느린 trial도 baseline의 가장 빠른 trial보다 빨랐습니다. 다만 한 GPU의 같은 날 실행한 synthetic attention workload이며, 이 개선율을 다른 workload나 모델 E2E에 일반화하지 않습니다. 위 단일 NCU capture의 duration은 이 표의 값으로 사용하지 않았습니다.

### 실제 모델 sanity와 E2E

모델은 로컬 Meta-Llama-3-8B, FP16 weight, B8, ISL2,048, OSL256, greedy, EOS 무시입니다. DiffKV-v1 `no-sync-bulk-lengths`, K8V4/K4V2, threshold `[0.02, 1]`, eager, GPU memory utilization 0.9, swap 0, max sequences 64를 사용합니다. 동일 B2, ISL2,048, OSL32 warmup 후 3회 측정합니다. E2E는 request 등록, prefill, decode, sampling, detokenization을 포함하며 모델 로드와 warmup을 제외합니다.

| 항목 | Baseline | K64+V32 | 비교 |
|---|---:|---:|---|
| E2E median, s | 10.797267 | 10.519347 | 1.0264배 |
| Decode throughput median, token/s | 283.839 | 295.293 | 1.0404배 |
| 생성 output token hash | 3회 모두 동일 | baseline과 동일 | 토큰 ID 완전 일치 |

Baseline 원자료는 [model-baseline.json](/home/130bb56/DiffKV/logs/performance/kernel-wide-load-20261024/model-baseline.json)에 있습니다. [모델 비교 결과](/home/130bb56/DiffKV/logs/performance/kernel-wide-load-20261024/model-comparison.json)는 모든 입력과 출력 일치 검사를 통과했습니다. 이 모델 반복 측정은 커널 최적화 단계의 필요 범위를 넘었으며, 후속 커널 변경마다 반복하지 않습니다. Microbenchmark 개선율을 모델 E2E 개선율로 사용하지 않습니다. 이 dummy 입력의 출력 일치는 태스크 정확도 평가를 대신하지 않습니다.

## Binary, source provenance와 rollback

후보는 `setup.py build_ext --build-lib <격리 폴더>`로 빌드하고, fresh process에서 `vllm._C`를 해당 binary로 직접 로드했습니다. 비교 과정에서는 저장소의 in-place extension과 공유 build manifest를 교체하지 않았습니다. 각 binary의 build 당시 source hash와 현재 작업 소스 hash를 분리해 기록합니다.

| Binary | SHA256 |
|---|---|
| Baseline | `36e8386cf4628a77957e225f7204bfa769816d1eb9c0b2df15a5412be9267d01` |
| K64 | `6e28c6d476181cfd85507d61d7f1b9b1b9f4f1bc30f306cd1729bdefbb59a925` |
| K64+V32 | `73a3dcff71fc9d1a8a1ccae542d16a31b2460913a317f78231743802a4b37f44` |

Manifest는 각각 [baseline](/home/130bb56/DiffKV/logs/performance/kernel-wide-load-20261024/baseline/build-manifest.json), [K64](/home/130bb56/DiffKV/logs/performance/kernel-wide-load-20261024/k64/build-manifest.json), [K64+V32](/home/130bb56/DiffKV/logs/performance/kernel-wide-load-20261024/k64-v32/build-manifest.json)에 있습니다. Manifest의 빌드 직후 `verification` 문구에 남은 pending 상태는 이후 수행한 correctness, regression, NCU 결과를 대신하지 않습니다.

실험을 baseline으로 되돌리려면 fresh process의 `--extension`을 보존된 baseline binary로 지정하면 됩니다. Source rollback이 필요하면 [최종 후보 source.diff](/home/130bb56/DiffKV/logs/performance/kernel-wide-load-20261024/k64-v32/source.diff)를 확인하고, 현재 변경과 일치하는지 reverse-apply check 후 이 패치만 되돌리면 됩니다. 최종 binary는 2026-10-24 16:07 KST에 기본 `vllm/_C.cpython-311-x86_64-linux-gnu.so`에 atomic replace로 적용했고, 공유 `.cache/diffkv-build.json`도 갱신했습니다. 적용 기록은 [installed-build-manifest.json](/home/130bb56/DiffKV/logs/performance/kernel-wide-load-20261024/installed-build-manifest.json)에 있습니다. 기존 프로세스가 이미 매핑한 binary는 유지되고 새 프로세스가 최적화 binary를 읽습니다.

## 짧은 재현 예시

아래 명령은 저장소 루트 `/home/130bb56/DiffKV`에서 실행하는 예시입니다. 기존 산출물을 덮어쓰지 않도록 새 output 경로를 사용해야 합니다.

```bash
.cache/baselines/fp16-env/bin/python benchmarks/validate_diffkv_kernel_candidate.py capture \
  --extension logs/performance/kernel-wide-load-20261024/k64-v32/vllm/_C.cpython-311-x86_64-linux-gnu.so \
  --contexts 512 2048 3072 4096 6000 8192 --batch 8 --kv-heads 8 --steps 32 \
  --oracle --output /tmp/k64-v32-repro.pt

.cache/baselines/fp16-env/bin/python benchmarks/validate_diffkv_kernel_candidate.py compare \
  --baseline logs/performance/kernel-wide-load-20261024/baseline/capture.pt \
  --candidate /tmp/k64-v32-repro.pt --output /tmp/k64-v32-repro-comparison.json

.cache/baselines/fp16-env/bin/python benchmarks/collect_kernel_candidate_ncu.py \
  --extension logs/performance/kernel-wide-load-20261024/k64-v32/vllm/_C.cpython-311-x86_64-linux-gnu.so \
  --output-dir logs/performance/kernel-wide-load-repro-20261024/ncu

.cache/baselines/fp16-env/bin/python benchmarks/benchmark_diffkv_kernel_model.py \
  --extension logs/performance/kernel-wide-load-20261024/k64-v32/vllm/_C.cpython-311-x86_64-linux-gnu.so \
  --build-manifest logs/performance/kernel-wide-load-20261024/k64-v32/build-manifest.json \
  --label k64-v32-repro --output /tmp/model-k64-v32-repro.json
```

NCU 수집기는 물리 GPU1 UUID만 노출한 컨테이너를 사용하며, driver 설정과 clock을 변경하지 않습니다. 반복 성능 비교는 [ABBA runner](/home/130bb56/DiffKV/logs/performance/kernel-wide-load-20261024/run_final_timing.py)의 조건을 따르되 새 산출물 폴더에서 수행해야 합니다. 이 runner는 원래 폴더의 고정 파일명을 쓰므로 기존 완료 결과 위에서 그대로 다시 실행하면 안 됩니다.

## 후속 검토 범위

GQA 재사용의 작은 partition과 shared-memory carveout 조합은 별도 후보로 검토할 수 있습니다. 기존 GQA 재사용 구현은 측정한 조건에서 성능이 나빴고, partition을 바꾸면 normalization과 partial-output 반올림도 달라질 수 있습니다. 이번 K64+V32의 bitwise 검증과 구분해 검증해야 합니다. 이번 패치에는 포함하지 않았습니다.
