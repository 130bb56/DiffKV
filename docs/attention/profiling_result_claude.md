# DiffKV-v1 Decode Attention: FP16보다 왜 느린가

[profiling_result_gpt.md](profiling_result_gpt.md)와 같은 측정 데이터를 다시 정리한 문서입니다. GPU를 새로 돌리지 않았고, 기존 ncu/nsys 결과와 커널 소스만 읽었습니다.

**측정 조건:** Llama-3-8B, RTX 3090, B=8, ISL=2048, OSL=2048, decode 단계의 attention 커널만 봅니다. KV 압축/append 커널(KV Update)과 prefill은 범위 밖입니다.

## 요약

1. DiffKV-v1의 decode attention은 FP16보다 **약 1.4배 느립니다.** 이 차이는 거의 전부 main 커널(`sparse_paged_attention_wrapper`)에서 생깁니다.
2. DiffKV는 메모리에서 **3.4~3.7배 적은 바이트**를 읽습니다. 하지만 읽는 속도가 FP16의 **약 1/5**입니다(GPU 메모리 대역폭의 18% 대 82%). 그래서 시간은 더 걸립니다.
3. 메모리 대역폭도 18%, FP32 연산기도 10%만 씁니다. 대부분의 시간 동안 **메모리에서 값이 오기를 기다리는 커널**입니다.
4. 기다리는 이유는 세 가지입니다.
   - K를 2바이트씩 쪼개 읽습니다.
   - 읽은 값을 바로 다음 명령에서 풀어서(dequant) 쓰기 때문에, 값이 올 때까지 그 warp가 멈춥니다.
   - 기다리는 동안 대신 일할 warp가 적습니다.
5. 가장 먼저 고칠 곳은 **load 명령 하나가 읽는 바이트 수**입니다. 소스의 `k_vec_type`이 2바이트 정렬 struct라서, 컴파일러가 8바이트 읽기를 2바이트 읽기 4번으로 쪼갭니다.

## 1. 얼마나 느린가

전체 생성 1회(nsys, 3회 평균) 기준 attention 시간은 다음과 같습니다.

| | FP16 | DiffKV-v1 |
|---|---:|---:|
| Attention 합계 (prefill+decode) | 13.216 s | 18.435 s |
| 그중 decode main | | 17.312 s (95.2%) |
| 그중 decode reduce | | 0.880 s (4.8%) |

DiffKV decode attention은 커널 두 개로 이루어집니다. 실제 계산은 **main**이 하고, **reduce**는 결과를 정리하고 중요도 점수를 갱신합니다. 시간의 95%가 main이므로, 이 문서도 main을 중심으로 봅니다.

ncu로 decode step 1, 1024, 2047에서 layer 0/15/31을 잘라 본 결과도 같습니다. 아래 값은 layer 3개의 평균이며, 단위는 커널 1회 호출당 us입니다.

| Decode step | 문맥 길이 | FP16 main+reduce | DiffKV main+reduce | DiffKV / FP16 |
|---:|---:|---:|---:|---:|
| 1 | 2049 | 108.2 | 151.9 | 1.40배 |
| 1024 | 3072 | 154.8 | 215.9 | 1.40배 |
| 2047 | 4095 | 200.5 | 280.1 | 1.40배 |

문맥 길이와 관계없이 항상 1.4배 느립니다.

## 2. 덜 읽는데 왜 느린가: 읽는 양과 읽는 속도

Decode attention은 원래 **메모리를 읽는 일이 대부분**인 작업입니다. 토큰 하나를 만들 때마다 지금까지의 K/V 전체를 메모리에서 읽어야 하기 때문입니다. 그래서 커널 시간은 대략 `읽는 바이트 / 읽는 속도`로 정해집니다.

Step 1024, layer 15의 main 커널을 비교하면 다음과 같습니다.

| | FP16 | DiffKV-v1 | 비율 |
|---|---:|---:|---:|
| 메모리(DRAM)에서 읽고 쓴 양 | 112.6 MB | 33.4 MB | DiffKV가 **3.37배 적음** |
| 읽는 속도 | 748.7 GB/s | 163.0 GB/s | DiffKV가 **4.59배 느림** |
| 최대 대역폭(911 GB/s) 대비 | 82.2% | 17.9% | |
| 실행 시간 | 150.3 us | 205.0 us | DiffKV가 **1.36배 느림** |

FP16 커널은 GPU 메모리 대역폭의 82%를 씁니다. DiffKV 커널은 18%를 씁니다.

**압축은 거의 전부 양자화에서 나옵니다.** 이 시점에 남아 있는 토큰 수는 195,900개로, FP16의 196,608개와 거의 같습니다. pruning으로 빠진 토큰은 0.4%뿐입니다. 바이트가 줄어든 것은 토큰당 크기가 작아졌기 때문입니다.

| KV head 하나, 토큰 하나 | K | V | 합계 |
|---|---:|---:|---:|
| FP16 | 128 × 2B | 128 × 2B | 512 B |
| DiffKV high (K8V4) | 128 B | 64 B | 192 B |
| DiffKV low (K4V2) | 64 B | 32 B | 96 B |

여기서는 high 90,508개, low 105,392개가 섞여 있어, 토큰당 평균 크기가 FP16의 약 27%입니다.

**논문은 읽는 양이 1/3이면 시간도 1/3이 된다고 가정합니다.** 이 가정이 맞으려면 DiffKV도 FP16만큼 빠르게 읽어야 합니다. FP16처럼 대역폭의 82%를 쓴다면 이 커널은 이론상 약 45us입니다(33.4MB ÷ 749GB/s). 지금은 205us입니다. 45us는 이론상 하한이며, 최적화 후 예상 시간은 아닙니다.

## 3. 느리게 읽는 원인

### 3.1 배경: 메모리를 기다리는 동안 GPU가 하는 일

GPU는 32개 thread를 한 묶음(warp)으로 실행합니다. 어떤 warp가 메모리 값을 기다리면, 스케줄러는 그동안 **다른 warp로 갈아타서** 일을 시킵니다. 메모리 지연은 수백 사이클이라, 메모리 버스를 꽉 채우려면 다음 두 가지가 필요합니다.

- **한 번에 많이 요청하기:** load 명령 하나가 많은 바이트를 가져와야 합니다.
- **갈아탈 warp가 충분하기:** 한 SM(warp를 실행하는 GPU 안의 처리 단위, RTX 3090에 82개)에 실행 가능한 warp가 많아야 합니다.

FP16 커널은 첫 번째를 잘합니다. DiffKV 커널은 두 가지 모두 불리합니다.

### 3.2 원인 1: K를 2바이트씩 쪼개 읽는다 (가장 직접적인 원인)

소스만 보면 DiffKV도 K를 8바이트(uint16 4개), V를 4바이트(uint16 2개) 묶음으로 한 번에 읽는 것처럼 보입니다.

```cpp
// csrc/attention/cache_utils.h
struct k_vec_type { uint16_t data[4]; };   // 8바이트지만 정렬은 2바이트
struct v_vec_type { uint16_t data[2]; };   // 4바이트지만 정렬은 2바이트
```

하지만 이 struct에는 `alignas`가 없어서 정렬이 uint16 기준인 2바이트입니다. 컴파일러는 8바이트 주소 정렬을 보장받지 못하므로, 실제 기계어(SASS)에서 **2바이트 load(`LDG.E.U16`)를 여러 번** 내보냅니다. ncu로 확인한 실제 SASS가 그렇습니다.

반면 FP16 vLLM 커널은 K를 `uint2`(8바이트 정렬), V를 `uint4`(16바이트 정렬) 타입으로 읽습니다([dtype_float16.cuh:42](../../.cache/baselines/vllm-1db83e3/csrc/attention/dtype_float16.cuh)). load 명령 하나가 가져오는 양이 DiffKV는 2바이트, FP16은 8~16바이트입니다.

ncu의 source counter를 보면 이 차이가 메모리 요청 수로 나타납니다. 메모리 요청은 32바이트 sector 단위로 셉니다.

| | 실제 요청 sector | 이상적인 sector | 낭비 |
|---|---:|---:|---:|
| Main 전체 | 12,096,684 | 3,918,252 | **3.09배**, 요청의 67.6%가 불필요 |
| K 값 읽기 | | | 필요한 양의 4배를 요청 (효율 25%) |
| V 값 읽기 | | | 필요한 양의 2배를 요청 (효율 50%) |

불필요한 요청의 **84%가 K**에서 나옵니다.

**주의할 점:** 이것은 요청 수입니다. DRAM에서 실제로 가져온 바이트가 3배인 것은 아닙니다. 같은 cache line을 여러 번 요청하므로 대부분 L1/L2 cache에서 걸러집니다. DRAM read는 29.4MB로, 꼭 필요한 양 29.06MB(양자화된 K/V 값 + scale·zero point)의 1.01배입니다. DRAM에서 읽은 바이트는 늘지 않았습니다. 늘어난 것은 load 명령 수입니다. 명령 하나가 2바이트만 읽으므로 대역폭을 채우지 못합니다.

### 3.3 원인 2: 도착하자마자 풀어 써야 하는 구조

K 값 하나를 쓰려면 이 순서를 차례로 거쳐야 합니다.

```text
2바이트 load → bit 풀기(PRMT/LOP3/SHF) → 정수를 float로(I2F) → scale·zero 적용(FFMA) → QK 내적
```

각 단계가 앞 단계의 결과를 기다립니다. 그래서 load가 늦게 오면 그 뒤 전체가 멈춥니다.

ncu의 PC sampling은 warp가 무엇을 하고 있는지 주기적으로 찍어 보는 기능입니다. 그 결과는 다음과 같습니다.

- 샘플의 **37%가 "메모리 값 기다림"**(long scoreboard) 상태였습니다.
- 그중 **1/4(25.6%)이 단 두 명령**에 몰렸습니다. 둘 다 메모리에서 읽은 K 값을 처음 사용하는 명령이고, 그 값이 도착하기를 기다리고 있었습니다.
  - K8 경로의 `I2F` (정수를 float로 바꾸기)
  - K4 경로의 `PRMT` (bit 풀기)

Dequant 명령 수 자체도 많습니다.

| 명령 | 전체 warp 명령 중 비중 |
|---|---:|
| `I2F` (정수를 float로) | 14.4% |
| Dequant용 `FFMA` (scale·zero 적용) | 14.4% |
| **합계** | **28.7%** |
| 참고: `LOP3` / `SHF` / `PRMT` (bit 풀기 + 주소 계산이 섞여 있음) | 10.7% / 7.7% / 3.7% |
| 참고: 실제 메모리 load `LDG` | 4.6% |

**이 28.7%는 명령 개수의 비중이지 시간 비중이 아닙니다.** 명령마다 걸리는 시간이 다르고 서로 겹쳐 실행되므로, 한 커널 안에 섞여 있는 dequant 시간을 따로 떼어 잴 수는 없습니다.

**GQA 때문에 dequant가 4번 반복됩니다.** Llama-3-8B는 query head 4개가 KV head 하나를 공유합니다. 현재 커널은 query head마다 CTA(thread block)를 하나씩 띄웁니다. 그래서 같은 KV를 4개 CTA가 각자 읽고 각자 dequant합니다. DRAM 읽기는 cache 덕분에 약 1번이지만(위의 1.01배), **dequant 명령은 4번** 실행됩니다.

### 3.4 원인 3: 기다리는 동안 대신 일할 warp가 적다

| Main 커널 (step 1024 평균) | FP16 | DiffKV-v1 |
|---|---:|---:|
| Thread당 register | | 108개 |
| CTA당 shared memory | | 26.8 KB |
| SM당 동시에 올라가는 CTA | | 2개 (warp 16개 / 최대 48) |
| Occupancy (실제 상주 warp 비율) | 38.9% | 30.9% |
| 사이클마다 바로 실행 가능한 warp (scheduler당) | 0.45 | 0.65 |
| 명령을 하나라도 낸 사이클 비율 | 31.0% | 42.8% |
| Grid 크기 | 1536 CTA (512토큰 단위로 6분할) | 256 CTA (분할 없음) |

DiffKV는 register와 shared memory를 많이 써서 SM당 CTA 2개만 올라갑니다. CTA는 모두 256개(query head 32 × 요청 8)입니다. SM 82개에 2개씩 한 번에 164개를 올릴 수 있으므로, 전체 CTA를 처리하려면 1.56회에 나눠 올려야 합니다.

**FP16도 occupancy와 실행 가능 warp 수는 비슷하게 낮습니다.** 그런데도 FP16이 대역폭을 82%까지 쓰는 이유는 원인 1 때문입니다. FP16은 load 한 번에 8~16바이트를 가져오므로, 적은 warp로도 메모리 버스가 가득 찹니다. DiffKV는 load 한 번에 2바이트를 읽고, 그 뒤에 풀기·변환 명령이 이어집니다. 그래서 warp 수가 같아도 메모리 버스를 채우지 못합니다. 원인 1~3은 모두 GPU 전체에서 동시에 처리 중인 메모리 요청량을 줄입니다. 그 결과 메모리 대역폭을 다 쓰지 못합니다.

### 3.5 원인이 아닌 것

- **Pruning으로 인한 head별 길이 차이:** Step 1, layer 15에서는 모든 head의 길이가 정확히 2049로 같습니다(pruning 없음). 그런데도 main+reduce가 FP16의 108.1us 대비 157.0us로 1.45배 느립니다.
- **Page table 간접 참조:** 영향이 있을 수는 있지만 따로 잰 적은 없습니다. K load에서 warp의 thread 94~100%가 실제로 값을 읽는데도 sector 효율은 25%입니다. 그래서 주된 원인은 명령 하나가 2바이트만 읽는 것입니다.
- **연산기 부족:** FP32 사용률이 10%이고, 연산기가 바빠서 기다린 경우(math-pipe throttle)도 거의 없습니다. 메모리 대역폭(18%)도 연산기(10%)도 다 쓰지 않습니다. ncu도 이 커널에 "compute throughput과 memory bandwidth가 둘 다 peak의 60% 미만이면 보통 latency 문제"라고 표시합니다. 이 커널에서는 warp가 load를 보낸 뒤 값이 올 때까지 멈춰 있고, 그동안 대신 실행할 warp가 없어서 SM이 명령을 내지 못합니다. 명령을 낸 사이클은 42.8%이고, 나머지 57%는 아무 명령도 내지 못했습니다.
- **Tensor Core 미사용:** FP16 커널도 Tensor Core를 쓰지 않으므로 차이의 원인이 아닙니다.

## 4. Reduce 커널

| | FP16 | DiffKV-v1 |
|---|---:|---:|
| 호출당 시간 | 4.6~5.0 us | 10.0~17.2 us |
| 하는 일 | 분할된 결과 합치기 | 출력 복사, 그리고 query head 4개의 확률 최댓값을 **KV 중요도 점수에 누적** |
| Grid | | 64 CTA (SM 82개보다 적음) |

DiffKV reduce는 FP16 reduce와 하는 일이 다릅니다. 중요도 점수는 다음 step에서 어떤 토큰을 pruning하거나 low precision으로 내릴지 정하는 데 쓰이는, DiffKV 고유의 작업입니다.

- CTA 수가 적고, 실행 가능 warp가 0.15개뿐입니다.
- PC 샘플의 66%가 메모리 대기입니다.

다만 decode attention 전체에서 5%뿐이므로 우선순위는 낮습니다.

## 5. 고칠 순서

모두 아직 구현하거나 측정하지 않은 후보입니다. 뒤로 갈수록 근거가 약합니다.

| 순서 | 무엇을 | 근거 | 확인할 것 |
|---:|---|---|---|
| 1 | `k_vec_type`/`v_vec_type`에 정렬 보장(예: `alignas(8)`, `alignas(4)`)이나 `uint2` 같은 내장 vector 타입을 써서 **K는 64-bit, V는 32-bit load**로 바꾸기. 실제 page 내 offset이 정렬돼 있는지 먼저 확인해야 함 | 불필요한 sector의 84%가 K. 메모리 대기 샘플이 K load 직후 지점에 몰림 | SASS가 `LDG.E.64`로 바뀌는지, sector 비율, main 시간 |
| 2 | 다음 page의 K/V를 prefetch(double buffering)해서, 풀기·변환이 값을 기다리지 않게 하기 | 메모리 대기 37%, 실행 가능 warp 0.64 | Register가 늘어 occupancy가 더 떨어지지 않는지 |
| 3 | 같은 KV head를 쓰는 query head 4개가 **dequant를 한 번만** 하도록 CTA 구조 바꾸기. CTA 수가 64로 줄지 않게 문맥 분할(partition)을 함께 늘려야 함. **10/18에 한 번 구현했으나 모든 조건에서 1.33~2.62배 느려져 꺼 둔 상태**([summary.json](../../logs/performance/gqa-reuse-20261018/summary.json)). 당시 ncu 카운터가 없어 원인은 모름 | dequant 명령이 4번 중복 | 기존 구현이 왜 느렸는지부터 ncu로 확인. 중복 명령 수와 CTA 병렬도, 중요도 점수 누적 결과가 기존과 같은지 |
| 4 | Reduce의 점수 갱신 방식 개선, 또는 main과 합치기 | Decode attention의 5% | Main+reduce 합계, 중요도 점수와 이후 토큰 선택이 기존과 같은지 |

1번만으로 몇 배 빨라질지는 모릅니다. Sector가 1/4로 줄어도 DRAM 바이트는 이미 거의 최소이므로, 시간이 1/4로 줄지는 않습니다. 기대하는 효과는 load 명령 수와 대기의 감소입니다.

## 6. E2E에서 이 문제의 크기

B8/2048/2048 offline 전체 생성(nsys 3회 평균, 1395 MHz 고정)에서 attention이 차지하는 몫은 다음과 같습니다.

| | FP16 | DiffKV-v1 |
|---|---:|---:|
| 전체 생성 시간 | 73.57 s | 77.06 s |
| Attention | 13.22 s (18.0%) | 18.43 s (23.9%) |

Attention만 바꿨을 때를 단순 계산하면 다음과 같습니다. 다른 항목은 그대로라고 가정한 산술입니다.

| 가정 | DiffKV-v1 전체 시간 | FP16 대비 |
|---|---:|---:|
| 지금 | 77.06 s | 0.95배 (느림) |
| Attention이 FP16과 같은 속도 | 71.84 s | 1.02배 |
| Attention이 이상적(FP16 ÷ 3.4) | 62.51 s | 1.18배 |

이 workload(B8, 메모리 여유)에서는 GEMM이 시간의 60~70%를 차지합니다. 그래서 attention을 이상적으로 고쳐도 offline 고정 batch 이득은 약 1.2배가 한계입니다. Online serving의 큰 이득은 attention 속도보다 **KV 용량 → 더 큰 batch**에서 나옵니다. 다만 batch가 커지고 문맥이 길어질수록 attention 비중이 커지므로, attention 최적화의 가치도 그만큼 커집니다.

## 7. 측정 방법과 주의할 점

- **ncu 측정 범위:** 8개 요청을 실제로 끝까지 생성하면서, decode step 1/1024/2047의 layer 0/15/31에서 main/reduce를 잘랐습니다("core" 측정). Step 1024, layer 15는 명령·PC·sector까지 한 번 더 자세히 잡았습니다("deep" 측정).
- **ncu 설정:** NCU 2024.1.0, `--clock-control none`(자동 클럭), `--cache-control all`(커널마다 cache 비움), `--replay-mode kernel`입니다. 그래서 ncu 시간은 진단용 값입니다. 1395 MHz 고정 클럭의 nsys 표와 직접 나눠 speedup을 계산하면 안 됩니다. 이 문서의 FP16/DiffKV 비교는 같은 ncu 조건끼리입니다.
- **DRAM 바이트:** K/V 값 외에 scale·zero point, 임시 확률, 출력까지 포함한 값입니다.
- **Warp 상태 비율:** "메모리 대기 37%" 같은 값은 PC 샘플의 비율이지 실행 시간의 비율이 아닙니다.
- **Deep과 core의 토큰 수 차이:** 같은 지점에서 deep은 196,060개, core는 195,900개를 유지했습니다. 최종 출력 토큰 hash는 같지만, 내부적으로 유지된 KV가 약간 다릅니다. 그래서 두 측정의 수치를 섞어 계산하지 않았습니다.
- **SASS와 소스 대응:** CUDA lineinfo가 없어 소스 줄 번호와 직접 이어지지 않습니다. 각 레지스터 값을 어느 명령이 만들고 어느 명령이 쓰는지, 그리고 명령별 실행 횟수를 따라가서 소스와 맞췄습니다. 12개 검증 항목을 모두 통과했습니다.

## 8. 원자료

| 내용 | 경로 |
|---|---|
| Step/layer별 main/reduce 수치 (FP16, DiffKV) | [decode-metrics.csv](../../logs/performance/attention-pd-20261022/decode-metrics.csv), [analysis.json](../../logs/performance/attention-pd-20261022/analysis.json) |
| 명령·sector·PC 상세 분석 | [deep-sass-analysis.md](../../logs/performance/attention-pd-20261022/deep-sass-analysis.md), [deep-sass-analysis.json](../../logs/performance/attention-pd-20261022/deep-sass-analysis.json) |
| Dequant 소스 분석 | [dequant-source-audit.md](../../logs/performance/attention-pd-20261022/dequant-source-audit.md) |
| nsys 전체 생성 합계 | [baseline-audit.txt](../../logs/performance/attention-pd-20261022/baseline-audit.txt), [profiling-final-result.md](../../profiling-final-result.md) |
| NCU report | [core](../../logs/performance/attention-pd-20261022/auto-core-diffkv/attention.ncu-rep), [deep](../../logs/performance/attention-pd-20261022/auto-deep-diffkv/attention.ncu-rep), [FP16 core](../../logs/performance/attention-pd-20261022/auto-core-fp16/attention.ncu-rep) |
| 커널 소스 | [K load와 dequant](../../csrc/attention/sparse_attention_kernels.cu:352), [vector 타입 정의](../../csrc/attention/cache_utils.h:10), [reduce 점수 갱신](../../csrc/attention/sparse_attention_kernels.cu:867) |
| 측정 스크립트 | [profile_pd_attention_ncu.py](../../benchmarks/profile_pd_attention_ncu.py), [collect.py](../../logs/performance/attention-pd-20261022/collect.py) |
