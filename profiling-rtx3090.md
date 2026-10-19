# RTX 3090, DiffKV v1 전체 워크로드의 Nsight Systems 측정

갱신 시각: 2026-10-18T15:40:04+09:00

## 1. 측정 범위

기존 GPU0 평가에서 DiffKV v1과 FP16이 모두 완료된 14개 조건을 대상으로, **DiffKV v1 실행만 GPU0에서 측정합니다.** 모델은 Meta-Llama-3-8B, weight는 FP16이며 v1 옵션은 `no-sync-bulk-lengths`입니다.

| Batch | 입력 길이 | 출력 길이 | 조합 수 |
|---:|---|---|---:|
| 8 | 512, 1024, 2048 | 512, 1024, 2048 | 9 |
| 16 | 512, 1024 | 512, 1024 | 4 |
| 32 | 512 | 512 | 1 |

각 행의 입력과 출력 길이를 교차 조합합니다. 입력은 기존 평가의 고정 token prompt이며, 전체 batch를 한꺼번에 제출하고 지정한 길이까지 생성합니다. 조건마다 새 프로세스를 사용합니다.

Nsight Systems 2024.4.1의 CUDA, NVTX, OS runtime trace를 사용합니다. 모델 로딩과 16-token warmup을 제외하고 실제 `generate` 전체와 마지막 GPU 완료까지 캡처합니다. 엔진 한계는 max_model_len 4096, max_num_seqs 32, max_num_batched_tokens 8192, eager, TP 1, swap 0입니다.

이번 단계에서는 실행 시간과 실행 구성을 기록합니다. 병목 진단 기준은 사용자와 별도로 정하며, 이 표의 시간 비중으로 병목을 판정하지 않습니다.

선정 근거는 [기존 평가](/home/130bb56/DiffKV/evaluation.md:294), 실행 목록은 [plan.json](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/plan.json), 캡처 코드는 [profile_offline_nsys.py](/home/130bb56/DiffKV/benchmarks/profile_offline_nsys.py), 집계 코드는 [analyze_offline_nsys.py](/home/130bb56/DiffKV/benchmarks/analyze_offline_nsys.py)입니다.

## 2. 전체 생성 시간의 분해

분석 완료: **14 / 14개 조건**. 아래 표에는 분석이 완료된 조건을 기록합니다.

전체 시간은 `measured_generation` NVTX wall time입니다. 각 셀은 **초 (전체 생성 시간 대비 %)**입니다. GPU kernel 구간은 합집합을 사용하고, 서로 다른 분류의 동시 실행은 중첩 열로 분리합니다. Copy와 memset은 kernel 실행과 겹치지 않는 부분만 표시합니다. 원자료 기준 각 행의 분해 항목 합계는 전체 시간, 비중 합계는 100%이며 표시값에는 반올림 차이가 있을 수 있습니다.

| B | 입력 | 출력 | 전체, 초 | GEMM | Attention | KV 갱신 | 기타 GPU | GPU kernel 중첩 | Copy, memset만 실행 | GPU 실행 없음 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 512 | 512 | 14.086 | 10.655 (75.65%) | 1.263 (8.96%) | 0.408 (2.90%) | 0.449 (3.18%) | 0.000 (0.00%) | 0.015 (0.11%) | 1.296 (9.20%) |
| 8 | 512 | 1024 | 28.248 | 20.556 (72.77%) | 3.097 (10.97%) | 0.845 (2.99%) | 0.867 (3.07%) | 0.000 (0.00%) | 0.031 (0.11%) | 2.852 (10.10%) |
| 8 | 512 | 2048 | 57.796 | 40.459 (70.00%) | 8.537 (14.77%) | 1.850 (3.20%) | 1.738 (3.01%) | 0.000 (0.00%) | 0.067 (0.12%) | 5.144 (8.90%) |
| 8 | 1024 | 512 | 15.567 | 11.452 (73.57%) | 1.854 (11.91%) | 0.451 (2.90%) | 0.489 (3.14%) | 0.000 (0.00%) | 0.016 (0.10%) | 1.304 (8.38%) |
| 8 | 1024 | 1024 | 30.154 | 21.384 (70.92%) | 4.253 (14.10%) | 0.928 (3.08%) | 0.918 (3.04%) | 0.000 (0.00%) | 0.034 (0.11%) | 2.637 (8.75%) |
| 8 | 1024 | 2048 | 63.220 | 42.371 (67.02%) | 11.612 (18.37%) | 2.146 (3.40%) | 1.905 (3.01%) | 0.000 (0.00%) | 0.075 (0.12%) | 5.111 (8.09%) |
| 8 | 2048 | 512 | 18.638 | 13.055 (70.04%) | 3.052 (16.38%) | 0.531 (2.85%) | 0.559 (3.00%) | 0.000 (0.00%) | 0.019 (0.10%) | 1.422 (7.63%) |
| 8 | 2048 | 1024 | 35.128 | 23.231 (66.13%) | 6.703 (19.08%) | 1.101 (3.13%) | 1.014 (2.89%) | 0.000 (0.00%) | 0.040 (0.11%) | 3.039 (8.65%) |
| 8 | 2048 | 2048 | 71.900 | 44.989 (62.57%) | 16.911 (23.52%) | 2.520 (3.50%) | 2.053 (2.85%) | 0.000 (0.00%) | 0.086 (0.12%) | 5.342 (7.43%) |
| 16 | 512 | 512 | 16.890 | 12.034 (71.25%) | 2.327 (13.78%) | 0.431 (2.55%) | 0.514 (3.04%) | 0.000 (0.00%) | 0.017 (0.10%) | 1.567 (9.28%) |
| 16 | 512 | 1024 | 33.711 | 22.615 (67.08%) | 5.811 (17.24%) | 0.896 (2.66%) | 0.974 (2.89%) | 0.000 (0.00%) | 0.037 (0.11%) | 3.380 (10.03%) |
| 16 | 1024 | 512 | 19.923 | 13.646 (68.49%) | 3.405 (17.09%) | 0.474 (2.38%) | 0.583 (2.93%) | 0.000 (0.00%) | 0.020 (0.10%) | 1.795 (9.01%) |
| 16 | 1024 | 1024 | 38.372 | 24.687 (64.34%) | 8.310 (21.66%) | 1.021 (2.66%) | 1.085 (2.83%) | 0.000 (0.00%) | 0.043 (0.11%) | 3.226 (8.41%) |
| 32 | 512 | 512 | 21.030 | 13.402 (63.73%) | 4.205 (20.00%) | 0.454 (2.16%) | 0.655 (3.11%) | 0.000 (0.00%) | 0.036 (0.17%) | 2.278 (10.83%) |

## 3. Prefill과 decode

CPU 범위는 scheduling과 output 처리를 포함한 전체 `engine_step_N`입니다. 실제 scheduler phase를 metadata에서 읽어 분할 prefill도 구분합니다. Engine step 밖의 구간은 prefill, decode 범위 합에 포함되지 않지만 E2E에는 포함됩니다. GPU kernel은 CUDA launch correlation으로 해당 step에 귀속합니다. CPU 범위와 귀속 GPU 시간은 겹칠 수 있으므로 서로 더하지 않습니다. 아래 GPU 시간은 호출별 실행 시간 합이며, 전체 생성 대비 비중은 해당 합을 전체 wall time으로 나눈 값입니다.

| B | 입력 | 출력 | Prefill 횟수 | Prefill latency, 초 (%) | Prefill GPU kernel, 초 (%) | Decode 횟수 | Decode latency, 초 (%) | Decode GPU kernel, 초 (%) | Decode latency, ms/step | Decode GPU, ms/step |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 512 | 512 | 1 | 0.876 (6.22%) | 0.840 (5.97%) | 511 | 13.203 (93.73%) | 11.934 (84.73%) | 25.837 | 23.355 |
| 8 | 512 | 1024 | 1 | 0.879 (3.11%) | 0.841 (2.98%) | 1023 | 27.355 (96.84%) | 24.524 (86.82%) | 26.740 | 23.973 |
| 8 | 512 | 2048 | 1 | 0.884 (1.53%) | 0.848 (1.47%) | 2047 | 56.885 (98.42%) | 51.737 (89.52%) | 27.789 | 25.274 |
| 8 | 1024 | 512 | 1 | 1.760 (11.30%) | 1.693 (10.87%) | 511 | 13.800 (88.65%) | 12.554 (80.65%) | 27.007 | 24.568 |
| 8 | 1024 | 1024 | 1 | 1.767 (5.86%) | 1.700 (5.64%) | 1023 | 28.373 (94.09%) | 25.783 (85.50%) | 27.735 | 25.203 |
| 8 | 1024 | 2048 | 1 | 1.767 (2.79%) | 1.701 (2.69%) | 2047 | 61.427 (97.16%) | 56.333 (89.11%) | 30.008 | 27.520 |
| 8 | 2048 | 512 | 2 | 3.642 (19.54%) | 3.513 (18.85%) | 511 | 14.988 (80.42%) | 13.685 (73.42%) | 29.332 | 26.780 |
| 8 | 2048 | 1024 | 2 | 3.660 (10.42%) | 3.527 (10.04%) | 1023 | 31.454 (89.54%) | 28.522 (81.19%) | 30.747 | 27.881 |
| 8 | 2048 | 2048 | 2 | 3.649 (5.08%) | 3.524 (4.90%) | 2047 | 68.224 (94.89%) | 62.948 (87.55%) | 33.329 | 30.751 |
| 16 | 512 | 512 | 1 | 1.749 (10.36%) | 1.682 (9.96%) | 511 | 15.132 (89.59%) | 13.624 (80.66%) | 29.613 | 26.661 |
| 16 | 512 | 1024 | 1 | 1.761 (5.22%) | 1.688 (5.01%) | 1023 | 31.934 (94.73%) | 28.607 (84.86%) | 31.216 | 27.964 |
| 16 | 1024 | 512 | 2 | 3.573 (17.93%) | 3.433 (17.23%) | 511 | 16.342 (82.03%) | 14.675 (73.66%) | 31.981 | 28.718 |
| 16 | 1024 | 1024 | 2 | 3.564 (9.29%) | 3.435 (8.95%) | 1023 | 34.791 (90.67%) | 31.668 (82.53%) | 34.009 | 30.956 |
| 32 | 512 | 512 | 2 | 3.515 (16.71%) | 3.389 (16.12%) | 511 | 17.504 (83.23%) | 15.327 (72.88%) | 34.254 | 29.994 |

## 4. Decode의 연산 분류

각 셀은 **ms/decode step (전체 생성 시간 대비 %)**입니다. 비중의 분자는 decode에 귀속된 해당 분류의 전체 kernel 실행 시간이며, 분모는 전체 생성 wall time입니다. 이 표에는 prefill과 GPU 실행이 없는 구간이 없으므로 행의 비중 합계는 100%가 아닙니다.

| B | 입력 | 출력 | GEMM | Attention | KV 갱신 | 기타 GPU |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 512 | 512 | 19.314 (70.07%) | 2.440 (8.85%) | 0.790 (2.87%) | 0.810 (2.94%) |
| 8 | 512 | 1024 | 19.325 (69.98%) | 3.013 (10.91%) | 0.822 (2.98%) | 0.814 (2.95%) |
| 8 | 512 | 2048 | 19.378 (68.63%) | 4.163 (14.74%) | 0.902 (3.19%) | 0.832 (2.95%) |
| 8 | 1024 | 512 | 19.354 (63.53%) | 3.527 (11.58%) | 0.861 (2.83%) | 0.825 (2.71%) |
| 8 | 1024 | 1024 | 19.369 (65.71%) | 4.107 (13.93%) | 0.896 (3.04%) | 0.831 (2.82%) |
| 8 | 1024 | 2048 | 19.932 (64.54%) | 5.647 (18.28%) | 1.043 (3.38%) | 0.898 (2.91%) |
| 8 | 2048 | 512 | 19.378 (53.13%) | 5.599 (15.35%) | 0.975 (2.67%) | 0.828 (2.27%) |
| 8 | 2048 | 1024 | 19.614 (57.12%) | 6.365 (18.54%) | 1.043 (3.04%) | 0.858 (2.50%) |
| 8 | 2048 | 2048 | 20.433 (58.17%) | 8.168 (23.25%) | 1.215 (3.46%) | 0.936 (2.67%) |
| 16 | 512 | 512 | 20.475 (61.95%) | 4.493 (13.59%) | 0.821 (2.48%) | 0.873 (2.64%) |
| 16 | 512 | 1024 | 20.564 (62.40%) | 5.650 (17.15%) | 0.865 (2.62%) | 0.885 (2.69%) |
| 16 | 1024 | 512 | 20.501 (52.58%) | 6.458 (16.56%) | 0.883 (2.26%) | 0.876 (2.25%) |
| 16 | 1024 | 1024 | 21.032 (56.07%) | 8.021 (21.38%) | 0.976 (2.60%) | 0.928 (2.47%) |
| 32 | 512 | 512 | 20.027 (48.66%) | 8.109 (19.70%) | 0.844 (2.05%) | 1.015 (2.47%) |

## 5. E2E 전체, prefill, decode 각각의 커널 실행 시간

각 조건에서 E2E 전체, prefill, decode의 상위 5개 커널을 각각 정렬합니다. 해당 범위 GPU 합 대비 비중은 해당 범위에 귀속된 모든 kernel 실행 시간의 합을 분모로 사용합니다. E2E 대비 비중은 전체 생성 wall time을 분모로 사용합니다. 순위는 시간 사용 내역이며 병목 판정이 아닙니다. 범위별 전체 커널 목록은 조건별 CSV와 summary JSON에 있습니다.

CSV는 scope별로 필터한 뒤 합산합니다. E2E 행과 Prefill, Decode 행을 함께 더하면 같은 커널 실행을 중복 계산합니다. CSV의 phase scope_wall_pct는 귀속 GPU kernel 시간 합을 해당 phase의 CPU engine-step 범위 합으로 나눈 값이며, 가산적인 wall time 분해가 아닙니다.

### B8, 입력 512, 출력 512

**E2E 전체, GPU kernel 합 12.774546초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 16,352 | 5.020275 | 39.30% | 35.64% | 307.013 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 32,704 | 3.132606 | 24.52% | 22.24% | 95.787 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 16,352 | 1.168509 | 9.15% | 8.30% | 71.460 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(T1::Params)` | 16,352 | 1.004147 | 7.86% | 7.13% | 61.408 |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 512 | 0.622047 | 4.87% | 4.42% | 1214.935 |

**Prefill, GPU kernel 합 0.840270초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` | 64 | 0.511325 | 60.85% | 3.63% | 7989.446 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 64 | 0.273384 | 32.54% | 1.94% | 4271.622 |
| `attn_fwd` | 32 | 0.015538 | 1.85% | 0.11% | 485.575 |
| `void vllm::silu_and_mul_kernel<c10::Half>(T1 *, const T1 *, int)` | 32 | 0.014723 | 1.75% | 0.10% | 460.096 |
| `void vllm::fused_add_rms_norm_kernel<c10::Half>(T1 *, T1 *, const T1 *, float, int, int)` | 64 | 0.012597 | 1.50% | 0.09% | 196.821 |

**Decode, GPU kernel 합 11.934276초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 16,352 | 5.020275 | 42.07% | 35.64% | 307.013 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 32,704 | 3.132606 | 26.25% | 22.24% | 95.787 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 16,352 | 1.168509 | 9.79% | 8.30% | 71.460 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(T1::Params)` | 16,352 | 1.004147 | 8.41% | 7.13% | 61.408 |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 511 | 0.620827 | 5.20% | 4.41% | 1214.925 |

원자료: [nsys report](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl512.nsys-rep), [범위별 전체 커널 CSV](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl512.kernels.csv), [집계 JSON](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl512.summary.json), [캡처 metadata](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl512-worker/profile.json).

### B8, 입력 512, 출력 1024

**E2E 전체, GPU kernel 합 25.365470초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 32,736 | 10.058347 | 39.65% | 35.61% | 307.256 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 65,472 | 6.272309 | 24.73% | 22.20% | 95.801 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 32,736 | 2.897545 | 11.42% | 10.26% | 88.513 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(T1::Params)` | 32,736 | 2.010920 | 7.93% | 7.12% | 61.428 |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 1,024 | 1.244463 | 4.91% | 4.41% | 1215.295 |

**Prefill, GPU kernel 합 0.841274초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` | 64 | 0.511846 | 60.84% | 1.81% | 7997.589 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 64 | 0.273801 | 32.55% | 0.97% | 4278.141 |
| `attn_fwd` | 32 | 0.015570 | 1.85% | 0.06% | 486.575 |
| `void vllm::silu_and_mul_kernel<c10::Half>(T1 *, const T1 *, int)` | 32 | 0.014728 | 1.75% | 0.05% | 460.264 |
| `void vllm::fused_add_rms_norm_kernel<c10::Half>(T1 *, T1 *, const T1 *, float, int, int)` | 64 | 0.012610 | 1.50% | 0.04% | 197.024 |

**Decode, GPU kernel 합 24.524196초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 32,736 | 10.058347 | 41.01% | 35.61% | 307.256 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 65,472 | 6.272309 | 25.58% | 22.20% | 95.801 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 32,736 | 2.897545 | 11.82% | 10.26% | 88.513 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(T1::Params)` | 32,736 | 2.010920 | 8.20% | 7.12% | 61.428 |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 1,023 | 1.243242 | 5.07% | 4.40% | 1215.290 |

원자료: [nsys report](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl1024.nsys-rep), [범위별 전체 커널 CSV](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl1024.kernels.csv), [집계 JSON](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl1024.summary.json), [캡처 metadata](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl1024-worker/profile.json).

### B8, 입력 512, 출력 2048

**E2E 전체, GPU kernel 합 52.584784초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 65,504 | 20.207148 | 38.43% | 34.96% | 308.487 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 131,008 | 12.560324 | 23.89% | 21.73% | 95.874 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 65,504 | 8.045152 | 15.30% | 13.92% | 122.819 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(T1::Params)` | 65,504 | 4.031177 | 7.67% | 6.97% | 61.541 |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 2,048 | 2.492337 | 4.74% | 4.31% | 1216.961 |

**Prefill, GPU kernel 합 0.848280초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` | 64 | 0.516452 | 60.88% | 0.89% | 8069.557 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 64 | 0.275972 | 32.53% | 0.48% | 4312.066 |
| `attn_fwd` | 32 | 0.015692 | 1.85% | 0.03% | 490.367 |
| `void vllm::silu_and_mul_kernel<c10::Half>(T1 *, const T1 *, int)` | 32 | 0.014758 | 1.74% | 0.03% | 461.189 |
| `void vllm::fused_add_rms_norm_kernel<c10::Half>(T1 *, T1 *, const T1 *, float, int, int)` | 64 | 0.012648 | 1.49% | 0.02% | 197.621 |

**Decode, GPU kernel 합 51.736504초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 65,504 | 20.207148 | 39.06% | 34.96% | 308.487 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 131,008 | 12.560324 | 24.28% | 21.73% | 95.874 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 65,504 | 8.045152 | 15.55% | 13.92% | 122.819 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(T1::Params)` | 65,504 | 4.031177 | 7.79% | 6.97% | 61.541 |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 2,047 | 2.491116 | 4.82% | 4.31% | 1216.960 |

원자료: [nsys report](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl2048.nsys-rep), [범위별 전체 커널 CSV](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl2048.kernels.csv), [집계 JSON](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl2048.summary.json), [캡처 metadata](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl2048-worker/profile.json).

### B8, 입력 1024, 출력 512

**E2E 전체, GPU kernel 합 14.246665초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 16,352 | 5.036337 | 35.35% | 32.35% | 307.995 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 32,704 | 3.133600 | 22.00% | 20.13% | 95.817 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 16,352 | 1.695620 | 11.90% | 10.89% | 103.695 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(T1::Params)` | 16,352 | 1.005566 | 7.06% | 6.46% | 61.495 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 32 | 0.835314 | 5.86% | 5.37% | 26103.558 |

**Prefill, GPU kernel 합 1.692575초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 32 | 0.835314 | 49.35% | 5.37% | 26103.558 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` | 96 | 0.725260 | 42.85% | 4.66% | 7554.795 |
| `attn_fwd` | 32 | 0.051832 | 3.06% | 0.33% | 1619.737 |
| `void vllm::silu_and_mul_kernel<c10::Half>(T1 *, const T1 *, int)` | 32 | 0.029495 | 1.74% | 0.19% | 921.716 |
| `void vllm::fused_add_rms_norm_kernel<c10::Half>(T1 *, T1 *, const T1 *, float, int, int)` | 64 | 0.025139 | 1.49% | 0.16% | 392.795 |

**Decode, GPU kernel 합 12.554090초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 16,352 | 5.036337 | 40.12% | 32.35% | 307.995 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 32,704 | 3.133600 | 24.96% | 20.13% | 95.817 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 16,352 | 1.695620 | 13.51% | 10.89% | 103.695 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(T1::Params)` | 16,352 | 1.005566 | 8.01% | 6.46% | 61.495 |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 511 | 0.621515 | 4.95% | 3.99% | 1216.273 |

원자료: [nsys report](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl1024-osl512.nsys-rep), [범위별 전체 커널 CSV](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl1024-osl512.kernels.csv), [집계 JSON](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl1024-osl512.summary.json), [캡처 metadata](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl1024-osl512-worker/profile.json).

### B8, 입력 1024, 출력 1024

**E2E 전체, GPU kernel 합 27.482847초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 32,736 | 10.094816 | 36.73% | 33.48% | 308.370 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 65,472 | 6.273389 | 22.83% | 20.80% | 95.818 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 32,736 | 3.961386 | 14.41% | 13.14% | 121.010 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(T1::Params)` | 32,736 | 2.014171 | 7.33% | 6.68% | 61.528 |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 1,024 | 1.246126 | 4.53% | 4.13% | 1216.920 |

**Prefill, GPU kernel 합 1.700234초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 32 | 0.839871 | 49.40% | 2.79% | 26245.961 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` | 96 | 0.728104 | 42.82% | 2.41% | 7584.413 |
| `attn_fwd` | 32 | 0.052006 | 3.06% | 0.17% | 1625.195 |
| `void vllm::silu_and_mul_kernel<c10::Half>(T1 *, const T1 *, int)` | 32 | 0.029521 | 1.74% | 0.10% | 922.540 |
| `void vllm::fused_add_rms_norm_kernel<c10::Half>(T1 *, T1 *, const T1 *, float, int, int)` | 64 | 0.025166 | 1.48% | 0.08% | 393.221 |

**Decode, GPU kernel 합 25.782614초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 32,736 | 10.094816 | 39.15% | 33.48% | 308.370 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 65,472 | 6.273389 | 24.33% | 20.80% | 95.818 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 32,736 | 3.961386 | 15.36% | 13.14% | 121.010 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(T1::Params)` | 32,736 | 2.014171 | 7.81% | 6.68% | 61.528 |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 1,023 | 1.244903 | 4.83% | 4.13% | 1216.914 |

원자료: [nsys report](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl1024-osl1024.nsys-rep), [범위별 전체 커널 CSV](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl1024-osl1024.kernels.csv), [집계 JSON](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl1024-osl1024.summary.json), [캡처 metadata](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl1024-osl1024-worker/profile.json).

### B8, 입력 1024, 출력 2048

**E2E 전체, GPU kernel 합 58.034250초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 65,504 | 20.653906 | 35.59% | 32.67% | 315.308 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 131,008 | 13.167535 | 22.69% | 20.83% | 100.509 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 65,504 | 10.951192 | 18.87% | 17.32% | 167.184 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(T1::Params)` | 65,504 | 4.060851 | 7.00% | 6.42% | 61.994 |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 2,048 | 2.516871 | 4.34% | 3.98% | 1228.941 |

**Prefill, GPU kernel 합 1.701371초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 32 | 0.840450 | 49.40% | 1.33% | 26264.050 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` | 96 | 0.728596 | 42.82% | 1.15% | 7589.540 |
| `attn_fwd` | 32 | 0.052090 | 3.06% | 0.08% | 1627.826 |
| `void vllm::silu_and_mul_kernel<c10::Half>(T1 *, const T1 *, int)` | 32 | 0.029511 | 1.73% | 0.05% | 922.229 |
| `void vllm::fused_add_rms_norm_kernel<c10::Half>(T1 *, T1 *, const T1 *, float, int, int)` | 64 | 0.025180 | 1.48% | 0.04% | 393.433 |

**Decode, GPU kernel 합 56.332880초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 65,504 | 20.653906 | 36.66% | 32.67% | 315.308 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 131,008 | 13.167535 | 23.37% | 20.83% | 100.509 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 65,504 | 10.951192 | 19.44% | 17.32% | 167.184 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(T1::Params)` | 65,504 | 4.060851 | 7.21% | 6.42% | 61.994 |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 2,047 | 2.515652 | 4.47% | 3.98% | 1228.946 |

원자료: [nsys report](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl1024-osl2048.nsys-rep), [범위별 전체 커널 CSV](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl1024-osl2048.kernels.csv), [집계 JSON](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl1024-osl2048.summary.json), [캡처 metadata](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl1024-osl2048-worker/profile.json).

### B8, 입력 2048, 출력 512

**E2E 전체, GPU kernel 합 17.197316초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 16,352 | 5.039202 | 29.30% | 27.04% | 308.170 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 32,704 | 3.142115 | 18.27% | 16.86% | 96.077 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 16,352 | 2.705303 | 15.73% | 14.51% | 165.442 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 64 | 1.686688 | 9.81% | 9.05% | 26354.492 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` | 192 | 1.463319 | 8.51% | 7.85% | 7621.453 |

**Prefill, GPU kernel 합 3.512702초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 64 | 1.686688 | 48.02% | 9.05% | 26354.492 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` | 192 | 1.463319 | 41.66% | 7.85% | 7621.453 |
| `attn_fwd` | 64 | 0.191343 | 5.45% | 1.03% | 2989.741 |
| `void vllm::silu_and_mul_kernel<c10::Half>(T1 *, const T1 *, int)` | 64 | 0.059101 | 1.68% | 0.32% | 923.450 |
| `void vllm::fused_add_rms_norm_kernel<c10::Half>(T1 *, T1 *, const T1 *, float, int, int)` | 128 | 0.050409 | 1.44% | 0.27% | 393.820 |

**Decode, GPU kernel 합 13.684614초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 16,352 | 5.039202 | 36.82% | 27.04% | 308.170 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 32,704 | 3.142115 | 22.96% | 16.86% | 96.077 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 16,352 | 2.705303 | 19.77% | 14.51% | 165.442 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(T1::Params)` | 16,352 | 1.005831 | 7.35% | 5.40% | 61.511 |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 511 | 0.621703 | 4.54% | 3.34% | 1216.639 |

원자료: [nsys report](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl2048-osl512.nsys-rep), [범위별 전체 커널 CSV](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl2048-osl512.kernels.csv), [집계 JSON](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl2048-osl512.summary.json), [캡처 metadata](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl2048-osl512-worker/profile.json).

### B8, 입력 2048, 출력 1024

**E2E 전체, GPU kernel 합 32.049064초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 32,736 | 10.191002 | 31.80% | 29.01% | 311.309 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 65,472 | 6.409527 | 20.00% | 18.25% | 97.897 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 32,736 | 6.167661 | 19.24% | 17.56% | 188.406 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(T1::Params)` | 32,736 | 2.020469 | 6.30% | 5.75% | 61.720 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 64 | 1.693880 | 5.29% | 4.82% | 26466.872 |

**Prefill, GPU kernel 합 3.527053초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 64 | 1.693880 | 48.03% | 4.82% | 26466.872 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` | 192 | 1.469446 | 41.66% | 4.18% | 7653.367 |
| `attn_fwd` | 64 | 0.192140 | 5.45% | 0.55% | 3002.181 |
| `void vllm::silu_and_mul_kernel<c10::Half>(T1 *, const T1 *, int)` | 64 | 0.059139 | 1.68% | 0.17% | 924.043 |
| `void vllm::fused_add_rms_norm_kernel<c10::Half>(T1 *, T1 *, const T1 *, float, int, int)` | 128 | 0.050462 | 1.43% | 0.14% | 394.237 |

**Decode, GPU kernel 합 28.522011초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 32,736 | 10.191002 | 35.73% | 29.01% | 311.309 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 65,472 | 6.409527 | 22.47% | 18.25% | 97.897 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 32,736 | 6.167661 | 21.62% | 17.56% | 188.406 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(T1::Params)` | 32,736 | 2.020469 | 7.08% | 5.75% | 61.720 |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 1,023 | 1.251443 | 4.39% | 3.56% | 1223.307 |

원자료: [nsys report](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl2048-osl1024.nsys-rep), [범위별 전체 커널 CSV](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl2048-osl1024.kernels.csv), [집계 JSON](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl2048-osl1024.summary.json), [캡처 metadata](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl2048-osl1024-worker/profile.json).

### B8, 입력 2048, 출력 2048

**E2E 전체, GPU kernel 합 66.472210초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 65,504 | 21.031245 | 31.64% | 29.25% | 321.068 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 65,504 | 15.885119 | 23.90% | 22.09% | 242.506 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 131,008 | 13.744169 | 20.68% | 19.12% | 104.911 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(T1::Params)` | 65,504 | 4.079267 | 6.14% | 5.67% | 62.275 |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 2,049 | 2.555818 | 3.84% | 3.55% | 1247.349 |

**Prefill, GPU kernel 합 3.524171초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 64 | 1.692105 | 48.01% | 2.35% | 26439.144 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` | 192 | 1.468627 | 41.67% | 2.04% | 7649.098 |
| `attn_fwd` | 64 | 0.191879 | 5.44% | 0.27% | 2998.112 |
| `void vllm::silu_and_mul_kernel<c10::Half>(T1 *, const T1 *, int)` | 64 | 0.059137 | 1.68% | 0.08% | 924.011 |
| `void vllm::fused_add_rms_norm_kernel<c10::Half>(T1 *, T1 *, const T1 *, float, int, int)` | 128 | 0.050438 | 1.43% | 0.07% | 394.045 |

**Decode, GPU kernel 합 62.948039초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 65,504 | 21.031245 | 33.41% | 29.25% | 321.068 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 65,504 | 15.885119 | 25.24% | 22.09% | 242.506 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 131,008 | 13.744169 | 21.83% | 19.12% | 104.911 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x2_tn_align8>(T1::Params)` | 65,504 | 4.079267 | 6.48% | 5.67% | 62.275 |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 2,047 | 2.553386 | 4.06% | 3.55% | 1247.379 |

원자료: [nsys report](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl2048-osl2048.nsys-rep), [범위별 전체 커널 CSV](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl2048-osl2048.kernels.csv), [집계 JSON](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl2048-osl2048.summary.json), [캡처 metadata](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl2048-osl2048-worker/profile.json).

### B16, 입력 512, 출력 512

**E2E 전체, GPU kernel 합 15.305619초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 16,352 | 5.077784 | 33.18% | 30.06% | 310.530 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 49,056 | 4.645506 | 30.35% | 27.50% | 94.698 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 16,352 | 2.192621 | 14.33% | 12.98% | 134.089 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 32 | 0.840767 | 5.49% | 4.98% | 26273.954 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` | 96 | 0.729808 | 4.77% | 4.32% | 7602.166 |

**Prefill, GPU kernel 합 1.681699초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 32 | 0.840767 | 50.00% | 4.98% | 26273.954 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` | 96 | 0.729808 | 43.40% | 4.32% | 7602.166 |
| `attn_fwd` | 32 | 0.030760 | 1.83% | 0.18% | 961.237 |
| `void vllm::silu_and_mul_kernel<c10::Half>(T1 *, const T1 *, int)` | 32 | 0.029517 | 1.76% | 0.17% | 922.414 |
| `void vllm::fused_add_rms_norm_kernel<c10::Half>(T1 *, T1 *, const T1 *, float, int, int)` | 64 | 0.025177 | 1.50% | 0.15% | 393.391 |

**Decode, GPU kernel 합 13.623919초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 16,352 | 5.077784 | 37.27% | 30.06% | 310.530 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 49,056 | 4.645506 | 34.10% | 27.50% | 94.698 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 16,352 | 2.192621 | 16.09% | 12.98% | 134.089 |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 511 | 0.626505 | 4.60% | 3.71% | 1226.038 |
| `void vllm::compress_and_append_cache_decode_phase_kernel<unsigned short, (int)128, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8>(const T1 *, const T1 *, const int *, unsigned short *, const int *, const int *, int *, const float *, int, int, int, int, int, int, int, int)` | 16,352 | 0.419565 | 3.08% | 2.48% | 25.658 |

원자료: [nsys report](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b16-isl512-osl512.nsys-rep), [범위별 전체 커널 CSV](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b16-isl512-osl512.kernels.csv), [집계 JSON](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b16-isl512-osl512.summary.json), [캡처 metadata](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b16-isl512-osl512-worker/profile.json).

### B16, 입력 512, 출력 1024

**E2E 전체, GPU kernel 합 30.294811초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 32,736 | 10.199399 | 33.67% | 30.26% | 311.565 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 98,208 | 9.354250 | 30.88% | 27.75% | 95.249 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 32,736 | 5.537425 | 18.28% | 16.43% | 169.154 |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 1,024 | 1.256807 | 4.15% | 3.73% | 1227.351 |
| `void vllm::compress_and_append_cache_decode_phase_kernel<unsigned short, (int)128, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8>(const T1 *, const T1 *, const int *, unsigned short *, const int *, const int *, int *, const float *, int, int, int, int, int, int, int, int)` | 32,736 | 0.884463 | 2.92% | 2.62% | 27.018 |

**Prefill, GPU kernel 합 1.687572초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 32 | 0.844018 | 50.01% | 2.50% | 26375.562 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` | 96 | 0.732265 | 43.39% | 2.17% | 7627.756 |
| `attn_fwd` | 32 | 0.030821 | 1.83% | 0.09% | 963.160 |
| `void vllm::silu_and_mul_kernel<c10::Half>(T1 *, const T1 *, int)` | 32 | 0.029547 | 1.75% | 0.09% | 923.341 |
| `void vllm::fused_add_rms_norm_kernel<c10::Half>(T1 *, T1 *, const T1 *, float, int, int)` | 64 | 0.025206 | 1.49% | 0.07% | 393.839 |

**Decode, GPU kernel 합 28.607239초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 32,736 | 10.199399 | 35.65% | 30.26% | 311.565 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 98,208 | 9.354250 | 32.70% | 27.75% | 95.249 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 32,736 | 5.537425 | 19.36% | 16.43% | 169.154 |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 1,023 | 1.255577 | 4.39% | 3.72% | 1227.348 |
| `void vllm::compress_and_append_cache_decode_phase_kernel<unsigned short, (int)128, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8>(const T1 *, const T1 *, const int *, unsigned short *, const int *, const int *, int *, const float *, int, int, int, int, int, int, int, int)` | 32,736 | 0.884463 | 3.09% | 2.62% | 27.018 |

원자료: [nsys report](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b16-isl512-osl1024.nsys-rep), [범위별 전체 커널 CSV](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b16-isl512-osl1024.kernels.csv), [집계 JSON](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b16-isl512-osl1024.summary.json), [캡처 metadata](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b16-isl512-osl1024-worker/profile.json).

### B16, 입력 1024, 출력 512

**E2E 전체, GPU kernel 합 18.107896초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 16,352 | 5.080505 | 28.06% | 25.50% | 310.696 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 49,056 | 4.656041 | 25.71% | 23.37% | 94.913 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 16,352 | 3.164242 | 17.47% | 15.88% | 193.508 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 64 | 1.696121 | 9.37% | 8.51% | 26501.884 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` | 192 | 1.471018 | 8.12% | 7.38% | 7661.551 |

**Prefill, GPU kernel 합 3.433153초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 64 | 1.696121 | 49.40% | 8.51% | 26501.884 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` | 192 | 1.471018 | 42.85% | 7.38% | 7661.551 |
| `attn_fwd` | 64 | 0.105181 | 3.06% | 0.53% | 1643.455 |
| `void vllm::silu_and_mul_kernel<c10::Half>(T1 *, const T1 *, int)` | 64 | 0.059151 | 1.72% | 0.30% | 924.236 |
| `void vllm::fused_add_rms_norm_kernel<c10::Half>(T1 *, T1 *, const T1 *, float, int, int)` | 128 | 0.050465 | 1.47% | 0.25% | 394.257 |

**Decode, GPU kernel 합 14.674743초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 16,352 | 5.080505 | 34.62% | 25.50% | 310.696 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 49,056 | 4.656041 | 31.73% | 23.37% | 94.913 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 16,352 | 3.164242 | 21.56% | 15.88% | 193.508 |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 511 | 0.626641 | 4.27% | 3.15% | 1226.303 |
| `void vllm::compress_and_append_cache_decode_phase_kernel<unsigned short, (int)128, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8>(const T1 *, const T1 *, const int *, unsigned short *, const int *, const int *, int *, const float *, int, int, int, int, int, int, int, int)` | 16,352 | 0.451169 | 3.07% | 2.26% | 27.591 |

원자료: [nsys report](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b16-isl1024-osl512.nsys-rep), [범위별 전체 커널 CSV](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b16-isl1024-osl512.kernels.csv), [집계 JSON](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b16-isl1024-osl512.summary.json), [캡처 metadata](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b16-isl1024-osl512-worker/profile.json).

### B16, 입력 1024, 출력 1024

**E2E 전체, GPU kernel 합 35.103004초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 32,736 | 10.349547 | 29.48% | 26.97% | 316.152 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 98,208 | 9.665876 | 27.54% | 25.19% | 98.422 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 32,736 | 7.885227 | 22.46% | 20.55% | 240.873 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 64 | 1.696489 | 4.83% | 4.42% | 26507.646 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` | 192 | 1.472331 | 4.19% | 3.84% | 7668.393 |

**Prefill, GPU kernel 합 3.434891초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 64 | 1.696489 | 49.39% | 4.42% | 26507.646 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` | 192 | 1.472331 | 42.86% | 3.84% | 7668.393 |
| `attn_fwd` | 64 | 0.105160 | 3.06% | 0.27% | 1643.121 |
| `void vllm::silu_and_mul_kernel<c10::Half>(T1 *, const T1 *, int)` | 64 | 0.059167 | 1.72% | 0.15% | 924.482 |
| `void vllm::fused_add_rms_norm_kernel<c10::Half>(T1 *, T1 *, const T1 *, float, int, int)` | 128 | 0.050505 | 1.47% | 0.13% | 394.568 |

**Decode, GPU kernel 합 31.668113초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8>(T1::Params)` | 32,736 | 10.349547 | 32.68% | 26.97% | 316.152 |
| `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn` | 98,208 | 9.665876 | 30.52% | 25.19% | 98.422 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 32,736 | 7.885227 | 24.90% | 20.55% | 240.873 |
| `ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn` | 1,023 | 1.265296 | 4.00% | 3.30% | 1236.849 |
| `void vllm::compress_and_append_cache_decode_phase_kernel<unsigned short, (int)128, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8>(const T1 *, const T1 *, const int *, unsigned short *, const int *, const int *, int *, const float *, int, int, int, int, int, int, int, int)` | 32,736 | 0.998135 | 3.15% | 2.60% | 30.490 |

원자료: [nsys report](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b16-isl1024-osl1024.nsys-rep), [범위별 전체 커널 CSV](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b16-isl1024-osl1024.kernels.csv), [집계 JSON](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b16-isl1024-osl1024.summary.json), [캡처 metadata](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b16-isl1024-osl1024-worker/profile.json).

### B32, 입력 512, 출력 512

**E2E 전체, GPU kernel 합 18.715835초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_32x6_tn_align8>(T1::Params)` | 16,352 | 5.105441 | 27.28% | 24.28% | 312.221 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 16,352 | 4.005399 | 21.40% | 19.05% | 244.949 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_64x6_tn_align8>(T1::Params)` | 16,352 | 2.373528 | 12.68% | 11.29% | 145.152 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 64 | 1.695525 | 9.06% | 8.06% | 26492.584 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` | 192 | 1.470410 | 7.86% | 6.99% | 7658.384 |

**Prefill, GPU kernel 합 3.389047초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256_32x3_tn_align8>(T1::Params)` | 64 | 1.695525 | 50.03% | 8.06% | 26492.584 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_tn_align8>(T1::Params)` | 192 | 1.470410 | 43.39% | 6.99% | 7658.384 |
| `attn_fwd` | 64 | 0.061870 | 1.83% | 0.29% | 966.718 |
| `void vllm::silu_and_mul_kernel<c10::Half>(T1 *, const T1 *, int)` | 64 | 0.059166 | 1.75% | 0.28% | 924.476 |
| `void vllm::fused_add_rms_norm_kernel<c10::Half>(T1 *, T1 *, const T1 *, float, int, int)` | 128 | 0.050459 | 1.49% | 0.24% | 394.212 |

**Decode, GPU kernel 합 15.326788초**

| Kernel | 호출 수 | 합계, 초 | 해당 범위 GPU 합 대비 % | E2E 대비 % | 호출 평균, μs |
|---|---:|---:|---:|---:|---:|
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x64_32x6_tn_align8>(T1::Params)` | 16,352 | 5.105441 | 33.31% | 24.28% | 312.221 |
| `void vllm::sparse_paged_attention_wrapper<unsigned short, (int)128, (int)4, (int)8, (int)4, (int)4, (int)2, (int)1, (int)1, (int)1, (int)1, (int)32, (int)60, (int)8, (int)5760>(const int *, const int *, float *, float *, T1 *, float *, const T1 *, unsigned short *, int, int, int, float, const int *, const int *, int, int, const float *, int, int)` | 16,352 | 4.005399 | 26.13% | 19.05% | 244.949 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_64x6_tn_align8>(T1::Params)` | 16,352 | 2.373528 | 15.49% | 11.29% | 145.152 |
| `ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_32x6_tn` | 16,352 | 1.312155 | 8.56% | 6.24% | 80.244 |
| `void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_32x6_tn_align8>(T1::Params)` | 16,352 | 0.809249 | 5.28% | 3.85% | 49.489 |

원자료: [nsys report](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b32-isl512-osl512.nsys-rep), [범위별 전체 커널 CSV](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b32-isl512-osl512.kernels.csv), [집계 JSON](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b32-isl512-osl512.summary.json), [캡처 metadata](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b32-isl512-osl512-worker/profile.json).

## 6. 해석 범위

시간 비중은 전체 실행에서 해당 작업이 사용한 시간을 나타냅니다. 이 측정만으로 연산 처리량, 대역폭, occupancy의 적정 수준이나 제거 가능한 비용을 확정하지 않습니다. GPU 실행이 없는 구간 역시 곧바로 Python 비용이나 CPU 병목으로 해석하지 않습니다. Nsight Systems를 붙인 시간은 계측 오버헤드를 포함하며 프로파일러 없는 E2E 성능 수치와 구분합니다.

커널 시간과 CUDA API의 CPU 대기는 겹칠 수 있습니다. CUDA API 대기 시간을 GPU 시간에 추가해 전체 시간을 계산하지 않습니다. Profiler 종료 후 report 저장 시간도 전체 생성 시간에서 제외합니다.

## 7. 참고
### FP16, DiffKV 비교

각 셀은 **위 FP16, 아래 DiffKV**입니다. 시간은 초, 괄호는 각 경로의 전체 생성 시간 대비 비율입니다.

| B | 입력 | 출력 | 전체, 초 | GEMM | Attention | KV 갱신 | 기타 GPU | GPU kernel 중첩 | Copy, memset만 실행 | GPU 실행 없음 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 512 | 512 | 13.107<br>14.086 | 10.634 (81.14%)<br>10.655 (75.65%) | 0.683 (5.21%)<br>1.263 (8.96%) | 0.054 (0.41%)<br>0.408 (2.90%) | 0.418 (3.19%)<br>0.449 (3.18%) | 0.000 (0.00%)<br>0.000 (0.00%) | 0.014 (0.11%)<br>0.015 (0.11%) | 1.304 (9.95%)<br>1.296 (9.20%) |
| 8 | 512 | 1024 | 26.058<br>28.248 | 20.544 (78.84%)<br>20.556 (72.77%) | 1.789 (6.86%)<br>3.097 (10.97%) | 0.107 (0.41%)<br>0.845 (2.99%) | 0.818 (3.14%)<br>0.867 (3.07%) | 0.000 (0.00%)<br>0.000 (0.00%) | 0.030 (0.12%)<br>0.031 (0.11%) | 2.770 (10.63%)<br>2.852 (10.10%) |
| 8 | 512 | 2048 | 56.081<br>57.796 | 41.317 (73.67%)<br>40.459 (70.00%) | 5.743 (10.24%)<br>8.537 (14.77%) | 0.221 (0.39%)<br>1.850 (3.20%) | 1.698 (3.03%)<br>1.738 (3.01%) | 0.000 (0.00%)<br>0.000 (0.00%) | 0.070 (0.12%)<br>0.067 (0.12%) | 7.031 (12.54%)<br>5.144 (8.90%) |
| 8 | 1024 | 512 | 14.705<br>15.567 | 11.434 (77.76%)<br>11.452 (73.57%) | 1.131 (7.69%)<br>1.854 (11.91%) | 0.057 (0.39%)<br>0.451 (2.90%) | 0.456 (3.10%)<br>0.489 (3.14%) | 0.000 (0.00%)<br>0.000 (0.00%) | 0.016 (0.11%)<br>0.016 (0.10%) | 1.611 (10.96%)<br>1.304 (8.38%) |
| 8 | 1024 | 1024 | 28.733<br>30.154 | 21.672 (75.43%)<br>21.384 (70.92%) | 2.778 (9.67%)<br>4.253 (14.10%) | 0.112 (0.39%)<br>0.928 (3.08%) | 0.882 (3.07%)<br>0.918 (3.04%) | 0.000 (0.00%)<br>0.000 (0.00%) | 0.034 (0.12%)<br>0.034 (0.11%) | 3.254 (11.33%)<br>2.637 (8.75%) |
| 8 | 1024 | 2048 | 64.069<br>63.220 | 45.796 (71.48%)<br>42.371 (67.02%) | 8.599 (13.42%)<br>11.612 (18.37%) | 0.252 (0.39%)<br>2.146 (3.40%) | 1.928 (3.01%)<br>1.905 (3.01%) | 0.000 (0.00%)<br>0.000 (0.00%) | 0.079 (0.12%)<br>0.075 (0.12%) | 7.416 (11.58%)<br>5.111 (8.09%) |
| 8 | 2048 | 512 | 17.823<br>18.638 | 13.019 (73.05%)<br>13.055 (70.04%) | 2.153 (12.08%)<br>3.052 (16.38%) | 0.064 (0.36%)<br>0.531 (2.85%) | 0.523 (2.93%)<br>0.559 (3.00%) | 0.000 (0.00%)<br>0.000 (0.00%) | 0.021 (0.12%)<br>0.019 (0.10%) | 2.043 (11.46%)<br>1.422 (7.63%) |
| 8 | 2048 | 1024 | 35.335<br>35.128 | 24.558 (69.50%)<br>23.231 (66.13%) | 5.184 (14.67%)<br>6.703 (19.08%) | 0.130 (0.37%)<br>1.101 (3.13%) | 1.021 (2.89%)<br>1.014 (2.89%) | 0.000 (0.00%)<br>0.000 (0.00%) | 0.044 (0.13%)<br>0.040 (0.11%) | 4.398 (12.45%)<br>3.039 (8.65%) |
| 8 | 2048 | 2048 | 75.805<br>71.900 | 50.018 (65.98%)<br>44.989 (62.57%) | 13.742 (18.13%)<br>16.911 (23.52%) | 0.275 (0.36%)<br>2.520 (3.50%) | 2.108 (2.78%)<br>2.053 (2.85%) | 0.000 (0.00%)<br>0.000 (0.00%) | 0.098 (0.13%)<br>0.086 (0.12%) | 9.563 (12.62%)<br>5.342 (7.43%) |
| 16 | 512 | 512 | 15.878<br>16.890 | 12.021 (75.71%)<br>12.034 (71.25%) | 1.373 (8.65%)<br>2.327 (13.78%) | 0.057 (0.36%)<br>0.431 (2.55%) | 0.482 (3.04%)<br>0.514 (3.04%) | 0.000 (0.00%)<br>0.000 (0.00%) | 0.017 (0.11%)<br>0.017 (0.10%) | 1.927 (12.14%)<br>1.567 (9.28%) |
| 16 | 512 | 1024 | 33.827<br>33.711 | 24.202 (71.55%)<br>22.615 (67.08%) | 4.107 (12.14%)<br>5.811 (17.24%) | 0.120 (0.35%)<br>0.896 (2.66%) | 0.983 (2.90%)<br>0.974 (2.89%) | 0.000 (0.00%)<br>0.000 (0.00%) | 0.039 (0.12%)<br>0.037 (0.11%) | 4.377 (12.94%)<br>3.380 (10.03%) |
| 16 | 1024 | 512 | 19.061<br>19.923 | 13.632 (71.52%)<br>13.646 (68.49%) | 2.311 (12.12%)<br>3.405 (17.09%) | 0.064 (0.34%)<br>0.474 (2.38%) | 0.548 (2.88%)<br>0.583 (2.93%) | 0.000 (0.00%)<br>0.000 (0.00%) | 0.022 (0.12%)<br>0.020 (0.10%) | 2.484 (13.03%)<br>1.795 (9.01%) |
| 16 | 1024 | 1024 | 39.204<br>38.372 | 26.473 (67.53%)<br>24.687 (64.34%) | 6.298 (16.07%)<br>8.310 (21.66%) | 0.132 (0.34%)<br>1.021 (2.66%) | 1.091 (2.78%)<br>1.085 (2.83%) | 0.000 (0.00%)<br>0.000 (0.00%) | 0.049 (0.12%)<br>0.043 (0.11%) | 5.161 (13.16%)<br>3.226 (8.41%) |
| 32 | 512 | 512 | 20.144<br>21.030 | 13.499 (67.01%)<br>13.402 (63.73%) | 2.722 (13.51%)<br>4.205 (20.00%) | 0.067 (0.33%)<br>0.454 (2.16%) | 0.620 (3.08%)<br>0.655 (3.11%) | 0.000 (0.00%)<br>0.000 (0.00%) | 0.039 (0.20%)<br>0.036 (0.17%) | 3.197 (15.87%)<br>2.278 (10.83%) |
