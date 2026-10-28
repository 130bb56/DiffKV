# DiffKV-v1 Decode Attention profiling

측정일: 2026-10-22, KST. 대상: B=8, ISL=2048, OSL=2048, Meta-Llama-3-8B, RTX 3090 GPU0.

사용자가 확정한 최종 범위는 **decode attention main, reduce**입니다. Prefill 성능 분석과 KV compression, append는 제외합니다. FP16은 이미 존재하는 trace와 counter 산출물만 비교 참고합니다.

**판정:** DiffKV-v1 decode main은 DRAM bandwidth 포화 상태가 아닙니다. DRAM은 peak의 14.8~19.2%, FP32는 9.2~10.1%에 그칩니다. Packed payload의 16-bit scalar load가 K에서 ideal 대비 약 4배 sector 요청을 만들고, 그 load를 기다리는 unpack, 변환 consumer에서 의존성이 드러납니다. 먼저 K payload load 폭과 load-to-use 구조를 개선할 근거가 가장 강합니다. Dequant의 변환과 affine만 전체 warp 명령의 28.75%이지만, 이를 실행 시간 비율로 해석할 수는 없습니다.

## 1. 기존 표에서 확인한 decode 비용

아래는 첨부 표의 원자료인 기존 nsys 3회 측정의 평균과 표본표준편차입니다. 새 ncu replay의 실행 시간을 전체 생성 시간으로 외삽한 값이 아닙니다.

| DiffKV-v1 구간 | 전체 생성 1회 중 합계, s | 표본표준편차, s | 호출 수 | 호출당 평균, us |
|---|---:|---:|---:|---:|
| Decode main | 17.311691 | 0.003273 | 65,504 | 264.284 |
| Decode reduce | 0.880359 | 0.000101 | 65,504 | 13.440 |
| Decode attention 합계 | 18.192049 | 0.003332 | 65,504 쌍 | 277.724 |

Decode attention 중 main 비중은 95.16%, reduce는 4.84%입니다. Reduce를 완전히 없앤다는 비현실적인 상한을 적용해도 decode attention 개선은 약 1.051배입니다. 따라서 main을 우선 분석합니다. 단일 partition에서도 reduce가 중요도 score를 갱신하므로 실제로 삭제할 수 있다는 뜻은 아닙니다.

표의 Attention과 KV Update는 같은 전체 생성 nsys capture 안에서 커널 이름으로 구분한 서로 다른 항목입니다. KV compression, append는 이미 KV Update이며 Attention에서 다시 차감하지 않습니다. 이번 main/reduce 분석에도 포함하지 않습니다. QKV, output projection은 GEMM입니다.

- [기존 표](/home/130bb56/DiffKV/profiling-final-result.md), [3회 측정표](/home/130bb56/DiffKV/profiling-TOBE-table.md)
- [원자료 검증](/home/130bb56/DiffKV/logs/performance/attention-pd-20261022/baseline-audit.txt), [재계산 JSON](/home/130bb56/DiffKV/logs/performance/attention-pd-20261022/baseline.json)
- [분류 코드](/home/130bb56/DiffKV/benchmarks/analyze_offline_nsys.py:29)

## 2. 실제 decode 구현

| 항목 | FP16, 기존 비교 경로 | DiffKV-v1, 이번 분석 대상 |
|---|---|---|
| 구현 언어, backend | vLLM CUDA C++, SIMT | 자체 CUDA C++, SIMT |
| Main kernel | `paged_attention_v2_kernel` | `sparse_paged_attention_wrapper` |
| Reduce kernel | `paged_attention_v2_reduce_kernel` | `vllm::reduce_kernel` |
| 읽는 KV | 전체 context의 FP16 K/V | 남은 token의 high K8/V4, low K4/V2 |
| Main 계산 | partition별 QK, softmax, PV, 부분 출력 | packed KV 읽기, unpack, FP32 역양자화, QK, softmax, PV, token별 확률 저장 |
| Reduce 계산 | partition별 softmax 통계로 부분 출력 결합 | 최종 출력 기록, 공유 query head 4개의 token별 확률 최댓값을 cache importance score에 누적 |
| 이번 조건의 partition | 512-token 단위, 실제 5~8개 | 실제 CUDA partition 5760, 실제 partition 수 1 |
| Tensor Core | 사용하지 않음 | 사용하지 않음 |

**Decode의 두 경로는 Triton이나 CUTLASS GEMM 커널이 아닙니다.** 양자화 정밀도는 캐시 저장 형식입니다. DiffKV main은 이를 FP32로 복원해 계산하며, INT4/INT2 Tensor Core 연산을 하는 구조가 아닙니다.

DiffKV main은 query head당 CTA 하나, grid=(32,8,1), block=256입니다. 같은 KV head를 공유하는 query head 4개가 각각 같은 packed KV를 읽고 역양자화합니다. Reduce는 KV head당 CTA 하나, grid=(8,8,1), block=256입니다. 실제 단일 partition에서는 partition 결합 없이 output 복사와 importance 누적을 수행합니다.

- [Python 호출](/home/130bb56/DiffKV/vllm/model_executor/layers/sparse_attention_big_kernel.py:395)
- [K 읽기와 역양자화](/home/130bb56/DiffKV/csrc/attention/sparse_attention_kernels.cu:352), [V 계산](/home/130bb56/DiffKV/csrc/attention/sparse_attention_kernels.cu:611)
- [단일 partition reduce](/home/130bb56/DiffKV/csrc/attention/sparse_attention_kernels.cu:867), [launch와 partition 설정](/home/130bb56/DiffKV/csrc/attention/sparse_attention_kernels.cu:1214)

## 3. 측정 방법과 해석 범위

실제 8개 요청을 끝까지 생성하면서 decode step 1, 1024, 2047의 layer 0, 15, 31을 선택했습니다. 각 forward에서 main, reduce를 ncu kernel-name filter로 분리했습니다. 초기 논리 context는 2049, 중간 3072, 마지막 4095입니다. 실제 retained KV 길이는 별도로 기록했습니다. 첫 출력 token은 prompt 처리에서 나오므로 decode step은 총 2047개입니다.

NCU 2024.1.0, `--clock-control none`, `--cache-control all`, `--replay-mode kernel`을 사용했습니다. GPU는 자동 클럭 상태이며 action별 실제 SM clock을 기록합니다. 이번 계측 명령은 GPU 설정을 변경하지 않습니다. 과거 nsys 표는 1395 MHz 조건이므로 새 ncu latency와 과거 표의 latency를 직접 나눠 speedup으로 제시하지 않습니다.

`--cache-control all`은 replay 전 cache를 flush하는 진단 조건입니다. NCU는 원래 실행의 cache 재사용, 커널 간 overlap, E2E 시간을 보존하는 도구가 아닙니다. 여기의 DRAM 바이트는 이 replay 조건의 실제 전체 커널 traffic이며 순수 KV payload만을 뜻하지 않습니다. 바이트에는 metadata, 임시 확률, output 등이 포함됩니다. RTX 3090의 외부 메모리는 GDDR6X이며, 이하 DRAM은 이 외부 메모리를 가리킵니다.

Metadata readback은 선택한 forward의 NVTX 범위 밖에서 수행합니다. CUDA event는 cache writer까지 포함하는 전체 Python forward를 감싸므로 Attention main/reduce 시간으로 사용하지 않습니다. 커널 시간은 ncu `gpu__time_duration.sum`을 사용합니다. Kernel replay와 profiling API 때문에 늘어난 전체 generation wall time도 성능 결과로 사용하지 않습니다.

분석에 쓰는 counter와 의미는 다음과 같습니다.

| 질문 | counter, 분석 방법 |
|---|---|
| 어느 커널이 오래 걸리는가 | `gpu__time_duration.sum`, 기존 nsys decode main/reduce 합계 |
| DRAM 대역폭 한계인가 | `dram__bytes_read.sum`, `dram__bytes_write.sum`, DRAM cycles와 peak bytes/cycle, bytes/time |
| L2 또는 접근 지연인가 | `lts__t_sectors.sum`, L1/L2 hit rate, `long_scoreboard`, `short_scoreboard` |
| 접근이 비효율적인가 | SourceCounters의 global theoretical/ideal sector 수, SASS load 폭, predication, 실제 retained 길이 |
| 연산 처리율 한계인가 | FP32 `2*FFMA+FADD+FMUL`, 실제 clock 기준 FP32 peak, math-pipe throttle, integer/convert/shuffle 명령 구성 |
| 병렬성이 부족한가 | grid, registers/thread, shared memory, achieved occupancy, eligible warps, issue activity |
| 대기가 어느 위치에 몰리는가 | `smsp__pcsamp_*`의 SASS PC별 sample, 레지스터 의존성 확인 |

FP32 FLOP counter는 integer unpack, 형변환, 주소 계산, shuffle, 지수함수를 포함하지 않습니다. 따라서 낮은 FP32 roofline 점만으로 memory-bound라고 단정하지 않습니다. Warp stall ratio와 PC sample 비율은 wall-time 비율이 아닙니다.

## 4. Decode main은 DRAM bandwidth 포화가 아닙니다

세 layer의 산술평균입니다. 서로 다른 layer의 분포이며 반복 측정의 신뢰구간이 아닙니다. 모든 원래 값은 [decode-metrics.csv](/home/130bb56/DiffKV/logs/performance/attention-pd-20261022/decode-metrics.csv)에 있습니다.

| Decode step | Main, us | Reduce, us | Main DRAM, GB/s | DRAM peak 대비 | L2 request peak 대비 | FP32 peak 대비 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 141.835 | 10.101 | 137.13 | 15.05% | 13.39% | 9.53% |
| 1024 | 202.827 | 13.099 | 151.36 | 16.61% | 14.66% | 9.76% |
| 2047 | 263.392 | 16.725 | 161.45 | 17.72% | 15.87% | 9.99% |

9개 main action 전체의 DRAM peak 비율은 14.79~19.22%, L2 request 비율은 13.06~17.66%입니다. FP32 peak 비율도 9.18~10.10%입니다. **외부 메모리 대역폭 포화도, scalar FP32 산술 처리율 포화도 관측되지 않았습니다.** 낮은 bandwidth를 보고 compute-bound라고 바꿔 부르는 것도 근거가 없습니다.

### 4.1 중간 시점 layer15의 roofline 계산

Core capture의 step1024, layer15, main을 예로 계산합니다.

| 값 | 측정, 계산 결과 |
|---|---:|
| 실행 시간 | 204.960 us |
| 실제 SM clock | 1963.340 MHz |
| 실행된 FP32 work, `2*FFMA+FADD+FMUL` | 830,163,904 FLOPs |
| 유효 QK, PV work, 보존 token 기준 | 401,203,200 FLOPs |
| FP32 처리율 | 4.050 TFLOP/s |
| 실제 clock 기준 FP32 FMA peak | 41.214 TFLOP/s |
| DRAM read / write | 29,401,088 / 4,013,440 bytes |
| DRAM 전체 / 처리율 | 33,414,528 bytes / 163.030 GB/s |
| 실제 memory clock 기준 DRAM peak | 911.088 GB/s |
| L2 request bytes | 94,868,192 bytes |
| FP32 arithmetic intensity, 전체 DRAM byte 기준 | 24.844 FLOP/byte |

FP32 peak는 `82 SM * 128 FP32 lanes/SM * 2 FLOP/FMA * actual_SM_clock`입니다. DRAM peak는 ncu의 `dram__bytes.sum.peak_sustained * dram__cycles_elapsed.avg.per_second`를 사용했습니다. 광고 boost clock이나 hardware structured-sparsity peak를 사용하지 않습니다. GA102의 FP32 구조와 GDDR6X 사양은 [NVIDIA GA102 whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf#page=12)에서도 확인할 수 있습니다.

```text
T_FP32_floor = 830,163,904 / 41.214e12 = 20.143 us
T_DRAM_floor = 33,414,528 / 911.088e9 = 36.675 us
T_L2_request_floor = ncu L2 request peak 비율 * duration = 32.992 us
max(위 floor) = 36.675 us, 실제는 204.960 us
```

FP32/DRAM roofline에서 이 점의 bandwidth ceiling은 약 22.636 TFLOP/s이며 실제는 4.050 TFLOP/s, 약 17.89%입니다. 따라서 실행점이 roofline의 기울어진 영역에 있다는 이유만으로 bandwidth 포화라고 말할 수 없습니다. 현재 커널은 그 ceiling에 크게 못 미칩니다.

이 floor들은 해당 자원을 이상적으로 처리하는 데 필요한 최소 시간입니다. **실제 실행 시간에서 36.675 us를 뺀 값을 dequant 시간이나 memory inefficiency 시간으로 부르면 안 됩니다.** 이 계산은 integer, conversion, shuffle, 제어, 동기화와 의존성 지연을 포함하지 않으며, 각 자원 작업은 서로 겹칠 수 있습니다.

### 4.2 실행 의존성과 latency hiding의 제약

같은 action에서 achieved occupancy는 31.05%, scheduler당 active warp는 3.736개이나 eligible warp는 0.643개, issue activity는 42.66%입니다. 즉 resident warp 상당수가 그 cycle에 issue할 수 없습니다.

주요 warp stall ratio는 long scoreboard 3.187, fixed-latency dependency의 wait 1.842, barrier 0.789, MIO throttle 0.550, short scoreboard 0.427 cycles/issued instruction입니다. Math-pipe throttle은 0.119에 그칩니다. Ratio는 warp 관점의 평균이고 커널 wall time의 additive 분해가 아닙니다.

Register 108개/thread와 shared memory 약 26.8 KB/CTA로 resident CTA는 SM당 최대 2개입니다. 256-thread CTA 두 개는 16 warp, GA102의 48 warp/SM 대비 이론상 33.33% occupancy이고 실제 31.05%와 가깝습니다. Grid는 256 CTA, ncu waves/SM은 1.561입니다. 따라서 더 많은 대역폭을 쓰려면 단순히 바이트를 줄이는 것 외에 **load 의존성, 실행 명령 수, 적은 resident warp와 낮은 eligible warp**를 개선해야 합니다.

이는 현재 관측된 제한에 대한 판단입니다. 아래의 상세 SASS 근거로 어떤 load와 변환에서 의존성이 드러나는지를 좁힙니다. 어떤 한 최적화가 몇 배 빨라질지는 아직 측정하지 않았습니다.

## 5. 압축으로 줄인 traffic이 왜 속도 개선으로 이어지지 않는가

기존에 확보된 FP16 `auto-core-fp16` artifact의 같은 step1024/layer15를 참고하면 다음과 같습니다. FP16은 재실행하지 않았습니다. 두 action은 자동 클럭 상태에서 수집됐으므로 이 latency 비교는 진단 참고값이며, 클럭을 통제한 speedup 검증은 아닙니다.

| Main counter | FP16 기존 reference | DiffKV-v1 |
|---|---:|---:|
| 전체 DRAM traffic | 112.559 MB | 33.415 MB |
| DRAM 처리율 | 748.716 GB/s | 163.030 GB/s |
| DRAM peak 대비 | 82.20% | 17.89% |
| L2 request peak 대비 | 77.86% | 16.10% |
| Grid CTA 수 | 1536, 6 partitions | 256, 1 partition |
| SM clock | 1875.359 MHz | 1963.340 MHz |
| Main 실행 시간 | 150.336 us | 204.960 us |

FP16은 DRAM, L2 처리율이 높고, DRAM floor 123.572 us가 실제 150.336 us에 가깝습니다. 이 reference에서는 메모리 경로 bandwidth 제한이 강합니다. 반면 DiffKV-v1은 **전체 DRAM traffic을 3.369배 줄였지만, 유효 처리율도 FP16의 21.77%로 낮습니다.**

```text
r = bytes_FP16 / bytes_v1 = 3.369
BW_v1 / BW_FP16 = 0.21775
T_FP16 / T_v1 = r * (BW_v1 / BW_FP16) = 0.7335
```

이 식은 `BW=bytes/time`에서 나온 관측값의 항등식이지, 원인별 시간을 분리한 모델이 아닙니다. **FP16이 bandwidth-bound라는 사실은 현재 DiffKV 구현도 높은 유효 대역폭을 낸다는 뜻이 아닙니다.** 압축 traffic 감소의 이점을 살리려면 아래의 scalar load와 unpack 의존성 비용을 먼저 낮춰야 합니다.

같은 KV head를 query head 4개가 읽는다는 사실도 DRAM에서 무조건 4번 전송한다는 뜻은 아닙니다. 이 지점의 unique packed payload는 27,495,168 bytes, unique quant metadata는 1,567,200 bytes, 합계 29,062,368 bytes입니다. 실제 main DRAM read는 29,401,088 bytes로 이 합계의 1.012배입니다. 이 비교가 모든 read를 완전히 귀속한 것은 아니지만, **논리적 4회 읽기를 실제 DRAM traffic 4배로 해석할 근거는 없습니다.** DRAM traffic과 L1/L2 hit counter를 보면 반복 읽기의 상당 부분이 cache에서 재사용된 것으로 해석됩니다. 반면 CTA별 unpack과 역양자화 명령의 중복은 그대로 남습니다.

## 6. Dequantization overhead, 실제로 확인한 비중

상세 capture는 **step1024, layer15의 decode main/reduce만** 선택했습니다. Main은 207.872 us, reduce는 11.808 us입니다. Core와 독립된 실행이며 상세 실행의 retained 합계는 high 90,615개, low 105,445개, 총 196,060개입니다. Core의 같은 선택점은 총 195,900개입니다. 최종 출력 token hash는 같지만 내부 KV 선택은 완전히 같지 않으므로, 상세 명령어 수와 core의 token 수를 섞어 계산하지 않았습니다. 이 차이의 원인은 이번 profiling만으로 확정하지 않았습니다.

### 6.1 역양자화가 차지하는 명령어와 산술 작업량

Deep SASS의 전체 warp instruction은 45,280,140개입니다. 레지스터 dataflow로 integer code를 float로 바꾸는 `I2F`와 그 결과에 scale, offset을 적용하는 affine `FFMA`를 구분하고, 실행 횟수를 이 실행의 retained 길이와 대조했습니다.

| 확인한 dequant 작업 | 동적 실행량 | 비중의 정확한 분모 |
|---|---:|---|
| Integer-to-float `I2F` | 6,508,736 warp instructions | 전체 warp instructions의 14.3744% |
| Affine dequant `FFMA` | 6,508,736 warp instructions | 전체 warp instructions의 14.3744% |
| 위 두 종류 합계 | 13,017,472 warp instructions | **전체 warp instructions의 28.7487%** |
| Affine dequant의 pred-on thread `FFMA` | 200,765,440 | `4 query/KV * 196,060 tokens * 256 K/V elements`와 일치 |
| Affine dequant FP32 FLOPs | 401,530,880 | **전체 실행 scalar FP32 FLOPs 831,145,408의 48.3105%** |

448개 `I2F` PC와 448개 affine `FFMA` PC의 동적 실행 수가 각각 일치합니다. 이 비중에는 별도의 bit unpack 명령, 양자화 metadata 읽기, 주소 계산은 포함하지 않았습니다. `LOP3`, `SHF`, `PRMT`에도 unpack 작업이 있지만 주소 계산 등 다른 목적도 있어 opcode 합계를 그대로 순수 dequant 비용으로 부르지 않습니다.

**판정:** 역양자화는 무시할 수 있는 작업이 아닙니다. 변환과 affine만으로 전체 warp 명령의 약 28.75%이며 bit unpack은 추가입니다. 그러나 **시간의 28.75%라는 뜻은 아닙니다.** 명령마다 latency와 throughput이 다르고, 다른 warp와 메모리 작업이 겹칩니다. Affine FP32 work가 전체 FP32 work의 약 절반이라는 사실도 compute peak 포화를 뜻하지 않습니다. 실제 main FP32 처리율은 peak의 약 10%입니다.

### 6.2 역양자화와 load가 충분히 overlap되는가

Deep main의 scheduler PC sample 26,199개 중 long scoreboard는 9,722개, 37.11%, wait는 20.91%, barrier는 9.39%입니다. 이 값들은 **활성 warp에서 샘플링한 scheduler 상태의 비중**이며 커널 wall time의 비중이 아닙니다.

Load 의존성이 노출되는 구체적인 예는 다음과 같습니다. PC는 이 report에서의 실제 주소이며, 다른 실행에서는 주소가 바뀝니다.

| 기다리는 consumer PC, SASS | 앞선 입력 load | long-scoreboard samples | 해당 reason 9,722개 중 |
|---|---|---:|---:|
| `0x7f7c99782900`, `I2F.U8 R48, R57` | K8 payload의 `LDG.E.U16.CONSTANT`, R57 생산 | 1,031 | 10.60% |
| `0x7f7c99783570`, `PRMT R67, R68, 0x7770, RZ` | K4 payload의 `LDG.E.U16.CONSTANT`, R68 생산 | 1,455 | 14.97% |

두 consumer만으로 long-scoreboard sample의 25.57%입니다. 즉 payload가 도착하기 전에 unpack, 변환을 진행하지 못하는 load-to-use 의존성이 드러납니다. **이 sample을 역양자화 산술 자체의 지연으로 귀속하면 틀립니다.** 반대로 이를 보면 모든 load 지연이 dequant 연산에 완전히 가려졌다고 주장할 수도 없습니다.

Affine FFMA PC의 long-scoreboard sample 합은0입니다. 이는 affine 비용이0이라는 뜻이 아니라 load 대기가 앞선 unpack, conversion consumer에서 관찰됐다는 뜻입니다. 현재 확인할 수 있는 것은 dequant 명령 작업량과 노출된 load 의존성입니다. Dequant 단독 elapsed time, memory와 겹친 정확한 시간, 겹치지 않은 시간은 이 fused kernel의 counter만으로 분리되지 않습니다.

## 7. Memory access efficiency, 주요 문제는 scalar payload load입니다

### 7.1 실제 compiler 출력과 sector 수

CUDA 소스의 K vector는 4개의 uint16, V vector는 2개의 uint16을 담지만, 실제 SASS는 이 payload를 `LDG.E.U16.CONSTANT`로 나눠 읽습니다. 따라서 소스에서 struct/vector처럼 보인다는 사실만으로 64-bit K load나 32-bit V load가 실행된다고 판단하면 안 됩니다.

| Deep main의 source-correlated counter | 값 |
|---|---:|
| `memory_l2_theoretical_sectors_global` | 12,096,684 |
| `memory_l2_theoretical_sectors_global_ideal` | 3,918,252 |
| Excessive sectors | 8,178,432 |
| Theoretical / ideal | **3.0873배** |
| Excessive / theoretical | **67.6089%** |
| 위 excessive 중 K payload에서 발생한 수 | 6,878,176, **84.10%** |

이 counter는 소프트웨어가 계산한 global instruction별 요청 sector 수입니다. **DRAM에서 실제 전송한 sector나 시간 비율이 아닙니다.** L1/L2 hit를 거친 hardware `lts__t_sectors` 또는 `dram__bytes`와 구분합니다. 위 67.61%를 memory access inefficiency의 시간 비중이라고 적을 수 없습니다.

주요 K8, K4 payload load의 sector 효율은 약 25%, 즉 theoretical/ideal 약 4배입니다. V payload는 약 50%, 약 2배입니다. K는 한 warp가 접근하는 주소 집합이 8-byte 간격으로 분포하는 vector layout인데 한 명령이 각 주소에서 2 byte씩 읽습니다. 이는 인접 lane ID의 주소가 항상 8 byte씩 증가한다는 뜻은 아닙니다. V의 주소 집합은 4-byte 간격이고 각 주소에서 2 byte씩 읽습니다. 여러 scalar 명령이 각 vector field를 다시 방문하는 형태가 request 수와 명령 수를 늘립니다.

**다음에 가장 먼저 볼 위치는 K payload load입니다.** K8 load PC 영역 `0x7f7c99782770`~`0x7f7c997828d0`, K4 영역 `0x7f7c99783450`~`0x7f7c99783530`의 instruction 폭, 정렬, lane 주소를 확인해야 합니다. Source의 [K load](/home/130bb56/DiffKV/csrc/attention/sparse_attention_kernels.cu:367), [low K load](/home/130bb56/DiffKV/csrc/attention/sparse_attention_kernels.cu:432)가 대응하는 작업입니다.

정렬을 보장한 vector load로 바꾸면 load instruction과 요청 sector를 줄일 가능성이 큽니다. 다만 실제 성능 개선은 아직 구현, 측정하지 않았습니다. Scalar load들이 같은 cache line을 재사용하므로 theoretical sector가 4분의 1로 줄어도 DRAM byte나 시간이 4분의 1로 줄어든다고 예측하면 안 됩니다.

### 7.2 Variable length, page indirection은 얼마나 설명하는가

현재 각 CTA는 한 query head의 KV 전체를 순회하므로 head별 길이가 달라도 그 자체가 warp 내 arbitrary gather를 뜻하지 않습니다. 한 page 안의 vector layout은 정해져 있고, page table은 다음 payload 주소를 만드는 의존성을 추가합니다.

상세 K payload load의 active-lane 비율은 high 약 99.74%, low 약 93.67%인데, sector 효율은 둘 다 약 25%입니다. **이 4배 sector 문제를 inactive lane이나 head별 길이 차이만으로 설명할 수 없습니다.** 실제 load 폭과 주소 stride가 더 직접적인 원인입니다. V payload의 요청 lane bytes가 유효 payload를 초과하는 정도는 high 약 0.50%, low 약 4.27%입니다. 이 값도 시간 비중은 아닙니다.

Core step1/layer15는 모든 head의 retained 길이가 정확히 2049, 길이 CV=0인데도 main은 146.944 us, DRAM 처리율은 15.42%입니다. 따라서 per-head variable total length가 이 커널의 낮은 처리율을 설명하는 유일한 원인이 아닙니다.

다만 high/low 구성 차이는 남습니다. Core step1024/layer15의 total-length CV는 0.55%이나 high-length CV는 26.11%, low-length CV는 21.86%입니다. Head마다 high/low loop의 작업 비율이 달라집니다. CTA 간 실행 시간 편차와 page-tail 비용은 이 구성 차이도 고려해야 하며, total length CV만으로 load balance를 확정하지 않습니다.

Page indirection만의 elapsed time이나 cache miss penalty 합계는 분리 측정하지 않았습니다. Page-table load, quant metadata load, payload load, 그 consumer를 연결해 확인할 수 있으나, 각각을 제거한 semantically equivalent 실행을 비교하지 않았으므로 독립 시간으로 제시하지 않습니다.

## 8. Reduce는 무엇이 제한하는가

Core reduce는 9.952~17.248 us, DRAM peak 대비 30.50~43.84%, eligible warp/scheduler는 약 0.15개입니다. Grid가 64 CTA여서 82 SM보다 적고, main보다 resident 작업도 작습니다. Deep에서는 global theoretical/ideal sector가 168,041/159,447, 약 1.054배로, main의 3.087배보다 훨씬 효율적입니다.

Deep reduce의 long scoreboard는 1,157 PC sample 중 768개, 약 66.38%입니다. 많은 sample이 `tmp_scores`의 확률을 읽은 뒤 비교하는 `FSETP` consumer에 있습니다. 이는 작은 grid와 load 의존성이 제한하는 커널이라는 판단을 지지합니다. Sample 수가 main보다 작으므로 세부 PC 간 우선순위는 더 신중히 해석합니다.

이 경로는 partition 결합보다 **4개 query head 확률의 max, cache score read/modify/write**가 주된 추가 의미입니다. FP16 reduce와 연산 의미가 다르므로 두 reduce의 시간 차이를 동일한 partition reduction 알고리즘의 효율 차이라고 부르면 안 됩니다.

기존 전체 decode attention에서 reduce는 4.84%입니다. Main의 scalar load, dequant 문제를 먼저 다룬 뒤, score staging과 갱신의 접근 방법, main과의 결합 가능성을 검토하는 순서가 타당합니다. 중요도 score의 정확한 갱신은 다음 step의 압축 선택에 영향을 주므로 출력 tensor만 맞는 검증으로는 부족합니다.

## 9. 다음에 볼 커널, 코드와 판단 기준

아래는 이번에 구현, 실행한 최적화가 아니라 측정으로 정한 검토 우선순위입니다.

| 우선순위 | 위치, 검토할 변경 | 근거 | 성능, 정확성 확인 기준 |
|---|---|---|---|
| 1 | Main의 high/low K payload load, 정렬을 보장한 64-bit load와 기존 uint16 struct load의 SASS 비교, 이후 V의 32-bit load 검토 | K payload가 excessive global sectors의 84.10%, K sector 효율 약25%, V 약50%. Load를 기다리는 unpack/convert PC가 확인됨 | 실제 SASS load 폭, theoretical/ideal sector, LDG 수, long-scoreboard sample, main latency를 함께 확인. Wider load만으로4배 speedup을 예측하지 않음 |
| 2 | Main의 unpack, `I2F`, affine, metadata load의 scheduling과 재사용, 다음 page load를 앞당길 수 있는지 검토 | 변환+affine만28.75% warp instructions, affine만48.31% scalarFP32 work. Long scoreboard와 wait가 함께 큼 | Dequant 명령 수, eligible warp, wait/longSB, registers/shared memory와 occupancy 동시 확인. 명령을 겹치려다 register가 늘어 residency가 나빠지는지 확인 |
| 3 | Main의 query-head 간 KV load/dequant 재사용과 partition, CTA 배치 함께 검토 | 같은 KV head의4 query CTA가 dequant 반복, 현재256CTA와2CTA/SM residency. GQA 재사용만 적용하면64CTA가 되어82SM도 채우지 못함 | 중복 변환 수와 CTA 병렬성을 함께 평가. Partition을 늘릴 때 softmax 결합, tmp_scores, importance 누적 의미까지 보존 |
| 4 | Reduce의 score load/갱신, main과의 score staging 방식 검토 | 작은64CTA grid, eligible warp 약0.15, longSB sample 약66%. 전체 decode attention 기여는 기존4.84% | Main+reduce 합계로 개선 확인. 최종 output뿐 아니라 cache score와 후속 retained KV 선택도 확인 |

**우선순위 1의 근거가 가장 직접적입니다.** 실제 instruction 폭, sector amplification, 그 payload를 기다리는 consumer를 연결해 확인했습니다. 단순히 page indirection을 없애거나 FP32 FLOP peak만 높이는 방향보다 먼저 확인할 지점입니다. 다만 최적화 뒤의 speedup은 아직 모릅니다.

## 10. 요청하신 두 가지의 시간 비중에 대한 답

| 항목 | 이번 측정으로 확정한 비중 | 아직 모르는 값 |
|---|---|---|
| Dequantization overhead | Convert+affine가 warp instruction의28.7487%, affine가 scalarFP32 FLOP의48.3105%. Bit unpack은 추가. K payload consumer 두 곳이 longSB sample의25.57% | Dequant만의 elapsed us, 전체 wall time 중 비율, 메모리와 겹치는 정확한 시간 |
| Memory access inefficiency | Main global theoretical sector의67.6089%가 ideal 대비 excessive, 그 초과분의84.10%가 K payload. K 약4배, V 약2배 sector amplification | Scalar load, page indirection, variable length 각각의 독립 elapsed us와 wall-time 비율 |

이 둘은 서로 배타적인 시간 항목이 아닙니다. 예를 들어 `I2F`가 load를 기다리는 상태는 dequant consumer에서 관측되지만 원인은 앞선 payload load 의존성입니다. 같은 sample을 dequant 시간과 memory 시간 양쪽에 더하면 중복 집계합니다.

따라서 **“dequant X%, memory inefficiency Y%”라는 실행 시간 분해는 현재 알 수 없습니다.** 알려진 값은 위의 명령, 산술, 요청 sector, scheduler sample 비중입니다. Dequant 제거 실험도 pre-dequantized cache를 쓰면 byte 수와 접근 방식이 달라지고, dummy arithmetic이나 load 제거는 의존성까지 바뀌므로 그 시간 차이를 기존 fused kernel의 정확한 구성 시간으로 부를 수 없습니다. 이번에는 이런 별도 kernel 변형 실험을 수행하지 않았습니다.

## 11. 논문의 bandwidth 기대와 현재 구현을 연결하면

사용자께서 제시하신 “압축으로 줄인 traffic의 이점이 dequant와 접근 비용보다 크면 빨라진다”는 조건은 타당합니다. 현재 결과는 그 전제인 높은 유효 처리율이 달성되지 않았다는 쪽을 지지합니다. Nominal cache 압축률을 전체 실제 DRAM traffic 절감률이나 speedup으로 바꾸면 안 됩니다.

로컬 [DiffKV.pdf](/home/130bb56/DiffKV/DiffKV.pdf)의 출판 판본에서는 custom attention speedup이 Figure15a, PDF12쪽에 있습니다. Figure13은 KV compaction 그래프입니다. Figure15a는 K8V8의 payload 절반에 대해 이론2배 대비 실측1.7배를 보고하며 metadata와 dequant 비용을 설명합니다. 이 판본의 실험은 대부분 L40이며, 해당 그래프는 지금의 RTX3090, adaptive high/low, B8/ISL2048/OSL2048 조건을 그대로 재현한 결과가 아닙니다. 현재 커널에서 확인된 load 폭과 낮은 처리율을 별도로 다뤄야 합니다.

## 12. 산출물, 재현과 검증

### Decode 측정 산출물

| 산출물 | 경로 |
|---|---|
| 여러 step/layer의 decode counter | [core metrics](/home/130bb56/DiffKV/logs/performance/attention-pd-20261022/auto-core-diffkv/metrics.json), [NCU report](/home/130bb56/DiffKV/logs/performance/attention-pd-20261022/auto-core-diffkv/attention.ncu-rep) |
| Decode-only 상세 instruction, PC, sector | [deep metrics](/home/130bb56/DiffKV/logs/performance/attention-pd-20261022/auto-deep-diffkv/metrics.json), [NCU report](/home/130bb56/DiffKV/logs/performance/attention-pd-20261022/auto-deep-diffkv/attention.ncu-rep) |
| Decode만 추린 비교, 계산 | [analysis.json](/home/130bb56/DiffKV/logs/performance/attention-pd-20261022/analysis.json), [CSV](/home/130bb56/DiffKV/logs/performance/attention-pd-20261022/decode-metrics.csv) |
| SASS dataflow와 명령, sector 귀속 | [상세 분석](/home/130bb56/DiffKV/logs/performance/attention-pd-20261022/deep-sass-analysis.md), [PC별 JSON](/home/130bb56/DiffKV/logs/performance/attention-pd-20261022/deep-sass-analysis.json) |
| 소스와 데이터 layout 검증 | [source audit](/home/130bb56/DiffKV/logs/performance/attention-pd-20261022/dequant-source-audit.md) |
| 실제 실행 명령 전체 | [core command](/home/130bb56/DiffKV/logs/performance/attention-pd-20261022/auto-core-diffkv/command.json), [deep command](/home/130bb56/DiffKV/logs/performance/attention-pd-20261022/auto-deep-diffkv/command.json) |
| Shape, length, 결과 token, binary 기록 | [core worker](/home/130bb56/DiffKV/logs/performance/attention-pd-20261022/auto-core-diffkv/worker/profile.json), [deep worker](/home/130bb56/DiffKV/logs/performance/attention-pd-20261022/auto-deep-diffkv/worker/profile.json) |
| Profiler runner, CPU 추출 | [runner](/home/130bb56/DiffKV/benchmarks/profile_pd_attention_ncu.py), [collector](/home/130bb56/DiffKV/logs/performance/attention-pd-20261022/collect.py), [extractor](/home/130bb56/DiffKV/logs/performance/attention-pd-20261022/extract.py), [analysis script](/home/130bb56/DiffKV/logs/performance/attention-pd-20261022/analyze_decode.py) |

Core report에는 범위를 decode-only로 좁히기 전에 수집된 prompt action6개가 함께 남아 있습니다. 이 보고서와 analysis/CSV는 그6개를 제외하고 decode main9개, reduce9개만 사용합니다. Deep report는 decode forward1개, main/reduce2개뿐입니다. FP16 수치는 새 실행 없이 이미 있던 `auto-core-fp16` 산출물을 읽었습니다.

동일 환경에서 decode-only로 재현하는 명령은 다음과 같습니다. 기존 결과를 덮어쓰지 않도록 새로운 label을 써야 합니다. 이 명령은 재현 예시이며 추가 실행하지 않았습니다.

```bash
cd /home/130bb56/DiffKV
.cache/baselines/fp16-env/bin/python logs/performance/attention-pd-20261022/collect.py \
  --backend diffkv --label decode-core-repro --decode-only \
  --layers 0 15 31 --decode-steps 1 1024 2047 --detail core
.cache/baselines/fp16-env/bin/python logs/performance/attention-pd-20261022/collect.py \
  --backend diffkv --label decode-deep-repro --decode-only \
  --layers 15 --decode-steps 1024 --detail full
```

실제 detailed capture는 SpeedOfLight, LaunchStats, Occupancy, ComputeWorkloadAnalysis, MemoryWorkloadAnalysis, MemoryWorkloadAnalysis_Tables, SchedulerStats, WarpStateStats, SourceCounters, InstructionStats, Single/Tensor roofline section과 FP32 raw instruction sum을 수집했습니다. Decode Tensor instruction count는0입니다. Counter definition과 sampling 한계는 설치된 [NCU2024.1 Profiling Guide](/usr/local/cuda-12.4/nsight-compute-2024.1.0/docs/ProfilingGuide/index.html)에 따릅니다.

검증 결과:

- Core, deep 모두 요청8개가 끝까지2048 output token을 생성했습니다. Decode step2047개와 scheduler batch8을 확인했습니다.
- Core의 선택15forward/24action, deep의1forward/2action에 대해 NVTX label, 커널 종류, 순서, 중복과 누락 검증이 통과했습니다. Core의 최종 분석 대상은 decode18action입니다.
- Deep main의 opcode 동적 합계45,280,140이 `inst_executed` 합계와 일치합니다. Dequant 계수, K/V byte와 sector 귀속에 대한12개 검증이 모두 통과했습니다. Metric 읽기 오류는 main/reduce 모두0개입니다.
- Extension SHA256은 `36e8386cf4628a77957e225f7204bfa769816d1eb9c0b2df15a5412be9267d01`, 과거 표의 DiffKV binary와 일치합니다. Build manifest의52개 소스 hash와 현재 소스도 확인했습니다.
- Core/deep의 출력 token SHA256은 모두 `60c2c3c1aadb1b3bab997fd6d77ad839d6869ac32d0c4daa7324d260a38efb51`입니다. 다만 내부 retained KV는 위에서 밝힌 대로 조금 다릅니다. Token hash 일치는 attention 수치 오차나 cache score 정확성을 검증한 것과 다릅니다. 이번에는 새로운 수치 정확성 검증을 수행하지 않았습니다.
- Kernel별 CUDA lineinfo가 없어 source line mapping은 없습니다. Instruction별 위치는 SASS PC, register dataflow와 소스 구조, 동적 count 일치를 사용해 판별했습니다.
- GPU 측정은 끝났고 측정 프로세스, 이 작업의 telemetry 수집도 종료했습니다. GPU clock 설정을 추가로 변경하지 않았습니다.
