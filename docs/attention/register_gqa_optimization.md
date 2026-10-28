# DiffKV-v1 decode attention의 GQA 재사용 최적화

이 문서는 첫 register-GQA 버전의 실험 기록입니다. 이후 PV token-pair hoisting과 split-score reducer를 적용한 전체 CUDA 호출 결과는 [후속 보고서](/home/130bb56/DiffKV/docs/attention/full_decode_optimization.md)를 확인해 주세요. 두 보고서는 clock 조건과 측정 시점이 다릅니다.

2026-10-24, 물리 GPU1의 RTX 3090에서 K8V4/K4V2 decode attention을 최적화했습니다. D128, query/KV head 비율 4, quantization chunk 각각 1개인 FP16 경로에 적용합니다. 중요도 기반 KV 선택 정책과 cache 저장 형식은 유지합니다.

## 결과의 범위

Batch 8, KV head 8, context 3072의 동일 packed-cache 입력을 사용했습니다. 비교 대상은 이번 최적화와 앞선 load 정렬 최적화를 적용하기 전 DiffKV-v1 binary입니다.

| 측정 범위 | 기존 | 선택한 커널 | 속도 향상 |
|---|---:|---:|---:|
| NCU attention main | 227.584 us | 92.832 us | 2.452배 |
| CUDA event, main + reduce + score 갱신 | 217.083 us | 112.764 us | 1.925배 |

NCU는 kernel replay, `--cache-control all`, `--clock-control none` 조건의 진단 capture입니다. CUDA event는 warmup 후 커널 호출 시간을 측정합니다. 두 측정의 실행 조건과 범위가 다르므로, NCU main 시간을 CUDA event 전체 시간에서 빼서 reduce 시간을 계산할 수 없습니다. GPU clock과 power limit은 변경하지 않았습니다. 모델 전체 속도는 위 숫자에서 추정하지 않습니다.

원자료는 [기존 NCU](/home/130bb56/DiffKV/logs/performance/kernel-wide-load-20261024/ncu-baseline/metrics.json), [선택한 커널 NCU](/home/130bb56/DiffKV/logs/performance/kernel-2x-20261024/register-gqa-p240-t128-temp4/ncu/metrics.json), [결과 요약](/home/130bb56/DiffKV/logs/performance/kernel-2x-20261024/result-summary.json)에 있습니다.

## 최종 모델 측정

커널 선택과 검증을 마친 뒤 Meta-Llama-3-8B, 입력 2048토큰, 출력 256토큰, batch 8을 기존 binary와 정식 새 binary에서 각각 한 번 측정했습니다. 두 경로 모두 DiffKV-v1의 Python 최적화, 동일 eager 엔진, greedy, `ignore_eos=True`, 정책 `[0.02, 1]`을 사용했습니다. FP16 KV baseline과의 비교는 아닙니다.

| 모델 지표 | 기존 DiffKV-v1 | 새 기본 경로 | 속도 향상 |
|---|---:|---:|---:|
| Decode 처리량 | 282.84 token/s | 308.66 token/s | 1.091배 |
| 전체 생성 시간 | 10.795 s | 10.192 s | 1.059배 |

이 수치는 backend당 한 번의 최종 측정입니다. 기존 warmup에서 CUDA kernel 이름을 기록하여 기존 경로와 새 GQA 경로가 실제로 선택됐음을 확인했고, 측정 generation에는 profiler를 사용하지 않았습니다. 두 경로의 생성 token ID 2048개가 모두 같았습니다. 고정 decode batch 8, decode step 255, preemption 0도 일치했습니다. Kernel main의 2.45배 향상을 모델 전체의 2.45배 향상으로 해석할 수 없습니다. [모델 비교 결과](/home/130bb56/DiffKV/logs/performance/kernel-2x-20261024/model-comparison.json)에 입력, Python source, binary, 실제 kernel 이름과 출력 hash를 기록했습니다.

## 병목과 변경

기존 커널은 같은 KV를 사용하는 query head마다 K/V load와 dequantization을 반복했습니다. 새 커널은 query head 4개를 하나의 CTA에서 처리하고, FP32로 복원한 K/V 값을 register에서 공유합니다. K/V 중간값을 FP16으로 추가 반올림하지 않습니다.

128 threads, partition 240, `__launch_bounds__(128, 4)`를 사용합니다. QK의 네 query shuffle을 교차 배치하고, softmax의 query별 통계를 함께 처리해 barrier를 줄였습니다. PV partial output의 shared-memory 주소는 XOR 변환으로 bank 충돌을 줄였습니다. Register는 thread당 127개, static shared memory는 9280B이며 spill은 없습니다. 작은 partition이 physical page 중간에서 시작하는 경우는 token 유효성 조건으로 처리합니다.

NCU의 실제 L1 global-load sector는 11,595,751에서 2,804,115로 감소했습니다. 이는 L1 요청 지표이며 DRAM에서 읽은 byte 수와 같은 뜻은 아닙니다. Load 대기와 명령 의존성을 줄이는 방향으로 변경했고, 별도 pipeline, query staging, fused reducer 후보는 전체 호출 시간이 더 나아지지 않아 제외했습니다.

[register kernel](/home/130bb56/DiffKV/csrc/attention/sparse_gqa_register_kernels.cuh), [launcher](/home/130bb56/DiffKV/csrc/attention/sparse_gqa_attention_kernels.cu), [Python dispatch](/home/130bb56/DiffKV/vllm/model_executor/layers/sparse_attention_big_kernel.py)가 구현을 담고 있습니다. Main과 reducer의 partition 크기, Python workspace 크기를 모두 240에 맞춥니다. 지원되는 dtype, head 구성, quantization, page 크기, query stride 조건을 만족하는 경우 이 경로를 선택합니다. 그 외에는 기존 attention을 사용합니다. `use_gqa_reuse=False`로 기존 경로를 명시적으로 선택할 수 있습니다.

## 정확성

Context 512, 2048, 3072, 4096, 6000, 8192에서 query를 바꿔 가며 32회 호출했습니다. 각 조건의 첫 호출은 독립적인 FP64 full softmax 기준값과 비교했습니다. 기존 커널과 output 최대 절대 차이는 0.000244140625였고, output과 누적 중요도 score는 허용 오차를 통과했습니다. Partition과 합산 순서가 바뀌므로 bitwise 동일한 출력은 아닙니다.

누적 score에 실제 cache writer의 정책 `[0, 0]`, `[0.02, 1]`, `[1, 1]`을 적용했습니다. 유지한 KV 위치, high/low 길이, block table, score 외 cache byte는 모두 기존 결과와 정확히 같았습니다. 검증 결과는 [전체 비교](/home/130bb56/DiffKV/logs/performance/kernel-2x-20261024/selected-comparison.json)에 있습니다.

검증에 사용한 fixture 원본은 [보존한 소스](/home/130bb56/DiffKV/logs/performance/kernel-2x-20261024/validated-fixture-source.py)에 있습니다. 이후 regression test에 P240 workspace와 경계 조건을 추가했으므로 현재 fixture 파일 hash는 이 capture의 hash와 다릅니다. 새로 재현할 때는 동일한 현재 fixture로 baseline과 candidate를 각각 capture해야 합니다.

실험 binary는 legacy dispatch를 축소하여 빠르게 빌드한 파일입니다. 설치에는 전체 dispatch를 포함한 정식 build를 사용합니다. 정식 build object와 선택한 실험 object의 GQA kernel SASS instruction은 [비교 결과](/home/130bb56/DiffKV/logs/performance/kernel-2x-20261024/final-sass-comparison.json)에서 동일함을 확인했습니다.

정식 binary는 기존 및 P240 경계, 36개 partition, ALiBi, adversarial softmax를 포함한 15개 regression test와 4개 alignment/tail guard test를 통과했습니다. [Regression log](/home/130bb56/DiffKV/logs/performance/kernel-2x-20261024/final-regression.log), [guard log](/home/130bb56/DiffKV/logs/performance/kernel-2x-20261024/final-guard.log)에 결과가 있습니다.

기본 extension과 Python dispatch에 적용한 binary SHA256은 `7edd20b3e7e6bcdb87d49dd7f68365cbec567ce78e543d2f87d0cff04c5974d2`입니다. 빌드 명령, source hash와 전체 dispatch 포함 여부는 [build manifest](/home/130bb56/DiffKV/logs/performance/kernel-2x-20261024/final/build-manifest.json)에 기록했습니다.
