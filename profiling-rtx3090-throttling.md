# RTX 3090 GEMM 성능 저하와 스로틀링 근거

GPU0의 **B=8, ISL=2048, OSL=2048**에서, 기존 3회 측정은 FP16과 DiffKV v1의 클럭과 thermal 제한 상태가 달랐습니다. FP16 실행 안에서도 같은 구성의 GEMM이 느려지는 동안 SM 클럭이 하락했고, 드라이버의 `SW Thermal Slowdown`이 활성화됐습니다. 두 경로를 1500MHz로 고정한 측정에서는 GEMM 시간 차이가 **0.30%**로 줄어, 기존의 큰 차이가 재현되지 않았습니다.

GPU0에서 직접 조회한 **base clock은 1395MHz**입니다. 1500MHz는 사용자가 별도로 고정한 동작 클럭입니다.

## 동일 워크로드의 GEMM 시간 비교

모델은 Meta Llama 3 8B, weight dtype은 FP16, tensor parallel은 1입니다. DiffKV는 v1의 `no-sync-bulk-lengths` 경로이며, 비교 대상은 FP16 vLLM 경로입니다. 아래 시간은 모델 로딩과 16토큰 warmup을 제외한 Nsight Systems 측정값이며, 계측 overhead를 포함합니다. E2E는 전체 생성 NVTX 범위의 wall time, GEMM은 해당 범위에 포함된 GEMM과 split K reduce GPU 커널 시간의 합입니다.

| 측정 조건 | FP16 E2E, s | v1 E2E, s | FP16 GEMM, s | v1 GEMM, s | FP16 대비 v1 GEMM 시간 변화 |
|---|---:|---:|---:|---:|---:|
| 클럭 미고정, 이전 단일 측정 | 64.692 | 67.022 | 42.637 | 42.763 | +0.29% |
| 클럭 미고정, 연속 3회 평균 ± 표본표준편차 | 81.477 ± 3.703 | 73.775 ± 3.442 | 54.123 ± 2.692 | 46.148 ± 2.011 | −14.74% |
| 냉각 후 클럭 미고정, 각 1회 | 64.841 | 67.466 | 42.639 | 42.720 | +0.19% |
| SM 1500MHz 고정, 각 1회 | 70.203 | 74.500 | 45.932 | 46.070 | +0.30% |

시간 변화는 `100 × (v1 GEMM 시간 / FP16 GEMM 시간 − 1)`입니다. 음수는 v1의 시간이 짧다는 뜻입니다. **81초와 73초는 E2E이며, 같은 행의 GEMM 시간은 54.123초와 46.148초입니다.** 기존 3회 평균에서 보였던 GEMM 차이는 7.975초였습니다.

반복 측정과 고정 측정의 두 경로는 입력 token SHA256이 모두 `e8792bab78fe6be4856eb01d2c8b346788ce244be3f2494ec8288d8f327b1a63`으로 같습니다. 여기서 동일 입력은 동일 prompt와 B, ISL, OSL을 뜻합니다. 중간 activation의 수치까지 bit 단위로 같다고 가정하지 않습니다. 고정 측정에서는 GEMM의 커널 이름, 실행 구성, 총 호출 수 329825회도 양쪽이 같았습니다.

시간과 실행 구성의 근거는 [case-analysis.json](/home/130bb56/DiffKV/logs/performance/gemm-roofline-cold-20261018-204308/case-analysis.json), [fixed-analysis.json](/home/130bb56/DiffKV/logs/performance/gemm-fixed1500-20261018-211743/fixed-analysis.json), 입력 일치 확인은 [evidence.json](/home/130bb56/DiffKV/logs/performance/gemm-throttling-evidence-20261019/evidence.json)에 있습니다.

## 반복별 클럭과 제한 신호

각 경로는 새 프로세스에서 warmup 후 전체 생성을 같은 프로세스 안에서 연속 3회 실행했습니다. 아래는 각 `measured_generation_rN` 범위 안에 들어온 GPU 로그만 집계한 결과입니다. **활성 비율은 전체 수집 샘플 중 해당 flag가 Active인 샘플의 비율**이며, 커널 점유율이나 E2E 시간 비중이 아닙니다.

| 경로 | 반복 | E2E, s | GEMM, s | SM 평균, MHz | SM 최소~최대, MHz | SW Thermal Active | SW Power Cap Active |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 1 | 77.209 | 51.017 | 1312.1 | 840~1935 | 285/380, 75.00% | 95/380, 25.00% |
| FP16 | 2 | 83.391 | 55.568 | 1151.9 | 930~1800 | 379/411, 92.21% | 32/411, 7.79% |
| FP16 | 3 | 83.832 | 55.785 | 1146.6 | 825~1860 | 380/413, 92.01% | 33/413, 7.99% |
| v1 | 1 | 69.934 | 43.934 | 1682.0 | 1470~1950 | 163/343, 47.52% | 180/343, 52.48% |
| v1 | 2 | 76.579 | 47.862 | 1415.1 | 1140~1860 | 338/377, 89.66% | 39/377, 10.34% |
| v1 | 3 | 74.813 | 46.647 | 1460.0 | 1140~1830 | 317/369, 85.91% | 52/369, 14.09% |

이 여섯 범위의 memory clock은 모든 샘플에서 9501MHz였고, `HW Thermal Slowdown`은 모두 Not Active였습니다. **HW thermal이 비활성이어도 SW thermal은 활성일 수 있습니다.** 드라이버가 보고하는 서로 다른 제한 신호입니다.

FP16은 v1보다 낮은 평균 SM 클럭, 더 높은 thermal 활성 비율에서 측정됐습니다. 따라서 이 행의 GEMM 차이를 DiffKV 구현의 성능 우위로 해석할 수 없습니다. 반복 평균과 표준편차를 계산해도 측정 조건의 차이는 제거되지 않습니다.

원본은 [FP16 gpu.csv](/home/130bb56/DiffKV/logs/performance/nsys-repeat3-20261018-173725/b8-isl2048-osl2048-fp16/gpu.csv), [v1 gpu.csv](/home/130bb56/DiffKV/logs/performance/nsys-repeat3-20261018-173725/b8-isl2048-osl2048-diffkv/gpu.csv), 각 디렉터리의 `r1.summary.json`, `r2.summary.json`, `r3.summary.json`, `trace.sqlite`입니다.

## FP16 실행 안에서 성능이 떨어진 위치

첫 FP16 실행의 decode 2047개 step을 네 구간으로 나눴습니다. 각 step의 선형 계층 GEMM은 M=8이며, K와 N은 모델 차원으로 고정됩니다. Attention의 context 길이가 늘어나는 것과 달리, 이 GEMM들의 행렬 크기는 decode 위치에 따라 커지지 않습니다.

| Decode step | Step 수 | GEMM 합계, s | GEMM, ms/step | SM 평균, MHz | SW Thermal Active | SW Power Cap Active |
|---|---:|---:|---:|---:|---:|---:|
| 1~512 | 512 | 9.904 | 19.344 | 1777.8 | 0/71, 0.00% | 71/71, 100.00% |
| 513~1024 | 512 | 12.707 | 24.819 | 1188.4 | 86/93, 92.47% | 7/93, 7.53% |
| 1025~1536 | 512 | 12.670 | 24.747 | 1169.2 | 98/98, 100.00% | 0/98, 0.00% |
| 1537~2047 | 511 | 12.576 | 24.610 | 1158.0 | 101/101, 100.00% | 0/101, 0.00% |

마지막 구간은 511개 step이므로, 구간별 합계 대신 GEMM 시간을 step 수로 나눠 비교했습니다.

첫 구간 대비 마지막 구간의 **GEMM 시간은 step당 27.22% 증가**, 같은 연산량을 시간으로 나눈 **처리율은 21.40% 감소**했습니다. 평균 SM 클럭은 34.87% 하락했습니다. 아래는 실제 역할별 변화입니다.

| 역할 | M, K, N | 호출 수/step | 처음 구간, μs/call | 마지막 구간, μs/call | 호출당 시간 증가 |
|---|---|---:|---:|---:|---:|
| QKV projection | 8, 4096, 6144 | 32 | 61.574 | 63.980 | +3.91% |
| O projection | 8, 4096, 4096 | 32 | 45.661 | 62.953 | +37.87% |
| MLP gate/up | 8, 4096, 28672 | 32 | 309.766 | 381.001 | +23.00% |
| Split K reduce | gate/up의 부분합 병합 | 32 | 5.824 | 7.534 | +29.35% |
| MLP down | 8, 14336, 4096 | 32 | 143.573 | 204.508 | +42.44% |
| lm_head | 8, 4096, 128256 | 1 | 1219.441 | 1570.597 | +28.80% |

가장 큰 절대 증가분은 gate/up과 down입니다. 각각 step당 2.280ms, 1.950ms 늘었으며, 합쳐서 전체 GEMM 증가분 5.266ms/step의 약 80.3%입니다.

2047개 step 모두 `[QKV, O, gate_up, splitKreduce, down] × 32 + lm_head`의 **161개 호출**을 확인했습니다. 역할별 커널 이름, grid, block, register, shared memory 구성은 전 구간 동일했습니다. O와 down은 같은 커널 이름과 실행 구성을 사용하므로 각 layer 안의 호출 위치로 나눴습니다. GPU 커널의 `correlationId`를 CUDA runtime launch와 연결하고, 해당 launch를 포함하는 decode NVTX에 배정했습니다.

역할별 원자료와 전수 검증 결과는 [decode-role-evidence.json](/home/130bb56/DiffKV/logs/performance/gemm-throttling-evidence-20261019/decode-role-evidence.json)에 있습니다.

## 클럭 하락과 함께 기록된 실제 로그

아래 시각은 2026년 10월 18일 KST입니다. `경과`는 FP16 첫 측정의 전체 생성 시작부터의 시간입니다. 최초 SW thermal 활성 샘플은 **18:28:02.252, 경과 19.534초, decode step 566**에서 관측됐습니다.

| 시각 | 경과, s | SM, MHz | Memory, MHz | GPU 온도, °C | SW Thermal | SW Power Cap |
|---|---:|---:|---:|---:|---|---|
| 18:27:42.904 | 0.186 | 1785 | 9501 | 70 | Not Active | Active |
| 18:28:02.045 | 19.327 | 1770 | 9501 | 75 | Not Active | Active |
| 18:28:02.252 | 19.534 | 1695 | 9501 | 75 | Active | Not Active |
| 18:28:12.818 | 30.100 | 960 | 9501 | 70 | Active | Not Active |
| 18:28:37.733 | 55.015 | 1065 | 9501 | 68 | Active | Not Active |

해당 원본 CSV의 행 번호는 143, 237, 238, 290, 413입니다. 최초 thermal 활성은 [gpu.csv의 238행](/home/130bb56/DiffKV/logs/performance/nsys-repeat3-20261018-173725/b8-isl2048-osl2048-fp16/gpu.csv:238)에 있습니다.

최초 thermal 활성 당시 SM은 1695MHz로 base 1395MHz보다 높았습니다. 즉, **base 아래로 내려갔는지가 throttling의 판단 기준은 아닙니다.** 여기서는 실제 제한 flag와 클럭 하락을 함께 확인했습니다.

`SW Thermal Slowdown`은 온도 제한에 의한 클럭 감소, `SW Power Cap`은 전력 제한에 의한 클럭 감소를 뜻합니다. [NVIDIA nvidia-smi 문서](https://docs.nvidia.com/deploy/nvidia-smi/index.html#clocks-event-reasons)

이 로그는 드라이버의 thermal 제한 활성화를 직접 보여줍니다. 다만 기록된 GPU 온도만으로 어느 온도 센서가 제한을 유발했는지는 특정할 수 없습니다. Memory junction이나 hotspot 온도는 수집하지 않았으므로 특정 부품의 과열로 단정하지 않습니다.

## 냉각과 클럭 고정 후 비교

냉각 후 클럭을 고정하지 않은 측정에서는 thermal 활성 샘플이 양쪽 모두 0이었고, GEMM은 FP16 42.639초, v1 42.720초로 0.19% 차이였습니다. 이때 `SW Power Cap`은 양쪽 모두 100%였으므로 thermal 제한이 없는 조건과 전력 제한이 없는 조건을 구분해야 합니다.

1500MHz 고정 측정은 각 경로를 새 프로세스에서 실행하고, 시작 전 GPU 온도가 50°C 이하가 될 때까지 기다렸습니다. 아래는 실제 측정 범위 안의 로그입니다.

| 경로 | GEMM, s | SM 최소/평균/최대, MHz | Memory 최소/평균/최대, MHz | GPU 온도, °C | SW Thermal Active | HW Thermal Active | SW Power Cap Active |
|---|---:|---|---|---|---:|---:|---:|
| FP16 | 45.932 | 1500/1500/1500 | 9501/9501/9501 | 56~65 | 0/321 | 0/321 | 0/321 |
| v1 | 46.070 | 1500/1500/1500 | 9501/9501/9501 | 55~67 | 0/341 | 0/341 | 0/341 |

고정 후 GEMM 차이는 v1이 0.138초, 0.30% 느린 수준입니다. **기존 3회 평균에서 보였던 7.975초의 v1 GEMM 우위는 재현되지 않았습니다.**

고정 측정의 원본은 [FP16 gpu.csv](/home/130bb56/DiffKV/logs/performance/gemm-fixed1500-20261018-211743/fp16.gpu.csv), [v1 gpu.csv](/home/130bb56/DiffKV/logs/performance/gemm-fixed1500-20261018-211743/diffkv.gpu.csv), [측정 조건](/home/130bb56/DiffKV/logs/performance/gemm-fixed1500-20261018-211743/plan.json)에 있습니다. 각 경로 1회 측정이므로 반복 편차 추정은 제공하지 않습니다. 약 200ms 간격의 로그에서 확인한 값이며, 모든 순간의 클럭이나 제한 신호를 측정한 것은 아닙니다.

## 스로틀링을 의심하고 확인한 근거

동일 B, ISL, OSL에서 GEMM 차이가 크게 나타났고, FP16 실행 안에서도 행렬 크기와 커널 구성이 유지된 채 GEMM 시간이 증가했습니다. 그 구간에 평균 SM 클럭 하락과 SW thermal 활성화가 함께 나타났습니다. 냉각 후 비교와 1500MHz 고정 비교에서는 두 경로의 큰 GEMM 차이가 재현되지 않았습니다.

따라서 **기존 비교에는 열 제한과 서로 다른 클럭 상태가 섞였고, 해당 GEMM 차이를 DiffKV 최적화 효과로 볼 근거가 없습니다.** Thermal 제한 자체는 로그로 확인됐습니다. 다만 기존 시간 차이의 정확히 몇 초를 thermal, power, 다른 실행 상태가 각각 설명하는지까지 분리한 결과는 아닙니다.

## Base clock과 현재 클럭 확인

GPU0의 base clock은 다음 명령으로 직접 조회합니다.

```bash
nvidia-smi base-clocks -i 0
```

실제 출력은 다음과 같습니다.

```text
GPU 0:
    Graphics: 1395MHz
```

2026년 10월 19일에 다시 조회한 출력은 [base-clock-query.txt](/home/130bb56/DiffKV/logs/performance/gemm-throttling-evidence-20261019/base-clock-query.txt)에 저장했습니다.

현재 동작 클럭과 제한 신호는 별도로 조회합니다.

```bash
nvidia-smi -i 0 -q -d CLOCK,PERFORMANCE
```

`Clocks > Graphics`, `Clocks > SM`은 현재 동작 주파수입니다. 사용자가 적용한 `--lock-gpu-clocks=1500,1500`에 따라 측정 중 1500MHz를 유지했으며, base clock 1395MHz와 구분됩니다.

[첨부 PDF](</home/130bb56/.codex/attachments/3c835ada-c13b-479a-8d53-3e4bcf95664a/s9956-best-practices-when-benchmarking-cuda-applications_V2.pdf>) 35~39쪽은 `Default Applications Clocks` 값으로 고정하는 절차입니다. 예시의 Graphics는 1215MHz입니다. 현재 GPU0에서는 해당 default 항목이 N/A이며, GPU0의 base는 위 `base-clocks` 명령으로 별도 확인했습니다.

## 로그 수집과 시간 정렬 방법

측정에 사용한 주요 로그 필드는 다음 명령으로 수집할 수 있습니다.

```bash
nvidia-smi -i 0 \
  --query-gpu=timestamp,index,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,power.limit,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_power_cap \
  --format=csv --loop-ms=200
```

CSV의 KST timestamp를 Unix ns로 바꾼 뒤 SQLite의 `TARGET_INFO_SESSION_START_TIME.utcEpochNs`를 빼서 Nsight trace의 상대 시각으로 맞췄습니다. 그중 `measured_generation_rN.start <= sample < measured_generation_rN.end`인 샘플만 반복별로 집계했습니다. Decode 구간별 집계도 같은 방식으로 해당 step 범위 안의 샘플을 골랐습니다. 표의 SM 평균은 샘플 산술평균이며 개별 GEMM 커널에 대한 시간 가중 평균은 아닙니다.

문서 근거를 재집계하는 아래 명령은 저장된 JSON, CSV, SQLite를 읽으며 GPU 워크로드를 실행하지 않습니다.

```bash
cd /home/130bb56/DiffKV

python logs/performance/gemm-throttling-evidence-20261019/build_evidence.py
python logs/performance/gemm-throttling-evidence-20261019/analyze_decode_roles.py
```

집계 결과와 출처 경로, SHA256은 [evidence.json](/home/130bb56/DiffKV/logs/performance/gemm-throttling-evidence-20261019/evidence.json), 역할별 커널 집계와 검증은 [decode-role-evidence.json](/home/130bb56/DiffKV/logs/performance/gemm-throttling-evidence-20261019/decode-role-evidence.json)에 있습니다.


setting
```bash
(.venv) 130bb56@CASSLAB-Server15:~/DiffKV$ sudo nvidia-smi -i 0 --lock-gpu-clocks=1395,1395
```