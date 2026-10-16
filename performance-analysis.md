# Llama3-8B, FP16 vLLM과 DiffKV의 성능 차이

2026-10-16 조사입니다. 주 대상은 [evaluation.md의 (ISL 512, OSL 2048, B8)](/home/130bb56/DiffKV/evaluation.md:76)입니다. 기존 OSL 512 분석과 이번 대상의 결과를 구분합니다.

**현재 확인한 가장 큰 불필요한 비용은 decode에 남은 명시적 GPU 동기화입니다.** 동일 workload와 CUDA binary에서 동기화만 제거하자 TPOT가 37.177에서 30.935ms/token으로 약 16.8% 감소했습니다. 이번 FP16 재측정은 30.979ms/token이므로 거의 같은 수준입니다. 메모리 절감이 논문의 큰 속도 향상으로 이어지지 않는 이유는 attention과 cache 갱신 커널의 추가 비용까지 함께 봐야 합니다.

## 1. 질문에 나온 수치의 단계 분석

원자료는 [FP16](/home/130bb56/DiffKV/logs/performance/fixed-grid-20261010-171112-437414/b8-isl512-osl2048-fp16.json), [DiffKV](/home/130bb56/DiffKV/logs/performance/fixed-grid-20261010-171112-437414/b8-isl512-osl2048-diffkv.json)입니다.

| 항목 | FP16 | DiffKV | DiffKV 증가분 |
|---|---:|---:|---:|
| TTFT, ms | 893.911 | 908.161 | 14.250 |
| TPOT, ms/token | 34.283 | 35.600 | 1.317 |
| Prefill engine.step 합, s | 0.893678 | 0.908138 | 0.014460 |
| Decode engine.step 합, s | 70.153141 | 72.847513 | 2.694372 |
| E2E, s | 71.072540 | 73.782405 | 2.709865 |

**전체 추가 시간의 99.43%가 decode에서 발생했습니다.** TTFT와 prefill throughput, TPOT와 decode throughput은 각각 같은 단계의 시간을 다르게 표현한 지표입니다. 다섯 지표가 열등하다는 사실이 독립된 다섯 병목을 뜻하지는 않습니다.

두 경로의 GPU UUID, 입력 hash, 출력 hash가 같습니다. 요청 8개가 각각 2048 token을 생성했고, decode2047회 모두 실제 B8, preemption0회였습니다. 생성량이나 batch 감소 때문이 아닙니다. Grid 전체의 max_num_seqs 설정은 양쪽 모두 32이며, 실제 B8은 [FixedBatchGuard](/home/130bb56/DiffKV/benchmarks/run_offline_e2e.py:461)와 histogram으로 확인했습니다.

이 과거 평가는 반복 1회, FP16 먼저 실행한 결과입니다. 당시 GPU 온도와 클럭 기록이 없어 3.81% 차이를 통계적으로 안정된 상수로 보지는 않습니다.

## 2. 동일 workload의 인과적 실험

당시 `diagnose_decode_overhead.py`는 기존 worker를 재사용했습니다. 현재 최적화 적용 함수는 [diffkv_v1.py](/home/130bb56/DiffKV/benchmarks/diffkv_v1.py:32)에 있으며, 실행은 [offline runner](/home/130bb56/DiffKV/benchmarks/run_offline_e2e.py:732)로 통합했습니다. 조건은 Llama3-8B Base, FP16 weights, GPU 0, ISL 512, OSL 2048, B8, warmup 16, greedy, ignore_eos, max_model_len=4096, max_num_seqs=32, eager, swap 0입니다.

엔진 파일을 고치지 않고 프로세스 안에서 Python method만 교체했습니다. CUDA binary, 양자화, pruning 설정은 같습니다.

| 현재 빌드의 경로 | TTFT, ms | TPOT, ms/token | Decode, tok/s | E2E, s |
|---|---:|---:|---:|---:|
| DiffKV, Python 원형 유지, 첫 실행 | 887.317 | 37.177 | 215.267 | 76.988 |
| DiffKV, 명시적 sync 제거 | 893.010 | 30.935 | 258.716 | 64.218 |
| DiffKV, sync 제거와 lengths 일괄 읽기 | 896.071 | 30.366 | 263.563 | 63.056 |
| FP16 | 884.244 | 30.979 | 258.352 | 64.298 |
| DiffKV, Python 원형 유지, 마지막 재실행 | 907.441 | 37.295 | 214.584 | 77.251 |

원자료는 [실행 요약](/home/130bb56/DiffKV/logs/performance/target-ablation-20261016-130112/summary.json), [검증과 개선율 계산](/home/130bb56/DiffKV/logs/performance/target-ablation-20261016-130112/analysis.json), [GPU 온도와 클럭](/home/130bb56/DiffKV/logs/performance/target-ablation-20261016-130112/gpu-telemetry.csv)입니다. 각 하위 폴더의 ablation.json과 Python 파일에 적용한 변경과 hash를 기록했습니다.

no-sync는 attention과 engine의 명시적 torch.cuda.synchronize()만 제거합니다. no-sync-bulk-lengths는 여기에 layer별 .item() 32회를 한 번의 .cpu().tolist()로 바꿉니다. _prepare_decode가 이미 계산한 동일한 32-layer 길이 벡터를 읽으므로 kernel에 넘기는 값은 같습니다. Sampling의 CPU 결과 읽기와 benchmark 경계 동기화는 유지합니다.

모든 출력 SHA256는 131e46db07c9e311bf470bd4cb339af9874786eb6094217354406c3d453f9e92로 같습니다. 첫 원형 대비 sync 제거 후 TPOT는 16.79%, E2E는 16.59% 감소했습니다. 마지막 원형의 TPOT도 37.295ms로 복귀해 단순한 실행 순서에 따른 환경 개선 설명을 약화합니다. Lengths 일괄 읽기의 추가 TPOT 감소는 1.84%입니다. 작은 추가 차이와 FP16 대비 약 2% 차이는 반복과 클럭 통제가 충분하지 않아 안정된 우위로 단정하지 않습니다.

현재 CUDA binary는 10/11 01:23 KST에 prompt score indexing 두 곳을 수정해 빌드한 버전입니다. [빌드 manifest](/home/130bb56/DiffKV/.cache/diffkv-build.json)의 source, patch, extension hash가 실제 파일과 일치합니다. 반면 10/10 평가에는 extension hash가 없습니다. **이번 6.24ms 감소를 과거 표의 1.317ms 차이에 대입하거나 기여율을 계산하면 안 됩니다.** 현재 빌드 내부의 비교와 과거 결과의 단계 분석을 분리합니다.

## 3. 동기화가 추가되는 위치

Llama는 [SparsePagedAttention](/home/130bb56/DiffKV/vllm/model_executor/models/sparse_llama.py:158)을 사용합니다.

| 위치 | Decode당 호출 수 | 기다리는 시점 |
|---|---:|---|
| [Attention 시작](/home/130bb56/DiffKV/vllm/model_executor/layers/sparse_attention_big_kernel.py:341) | 32 | QKV projection과 RoPE 이후 |
| [Cache 갱신 이후](/home/130bb56/DiffKV/vllm/model_executor/layers/sparse_attention_big_kernel.py:385) | 32 | Compression 종료 |
| [Attention 종료](/home/130bb56/DiffKV/vllm/model_executor/layers/sparse_attention_big_kernel.py:450) | 32 | Attention 종료 |
| [Engine step](/home/130bb56/DiffKV/vllm/engine/llm_engine.py:542) | 4 | Scheduling, model, metadata 처리 사이 |
| 합계 | **100** | |

시간 출력 print는 주석 처리됐지만 동기화는 남아 CPU의 선행 enqueue를 끊습니다. FP16의 [attention](/home/130bb56/DiffKV/.cache/baselines/vllm-1db83e3/vllm/model_executor/layers/attention.py:158), [engine](/home/130bb56/DiffKV/.cache/baselines/vllm-1db83e3/vllm/engine/llm_engine.py:580)에는 대응하는 명시적 동기화가 없습니다.

또한 [GPU에서 계산한 context vector](/home/130bb56/DiffKV/vllm/worker/model_runner.py:236)를 layer마다 [item()](/home/130bb56/DiffKV/vllm/model_executor/layers/sparse_attention_big_kernel.py:355)으로 읽어 32회의 scalar D2H 전송과 대기를 추가합니다. Sync 제거 실험에서도 이 32회는 유지했고, bulk 실험에서만 바꿨습니다.

## 4. GPU profiling의 kernel 비용

### 4.1 이번 대상, ISL 512, OSL 2048, B8

현재 빌드에서 Python 동기화 원형을 유지하고, 동일 GPU 0에서 FP16과 DiffKV를 순서대로 수집했습니다. Capture는 decode ordinal 1024~1055의 32 steps, 논리 context 1536~1567입니다. [Profiler 실행 요약](/home/130bb56/DiffKV/logs/performance/target-profile-20261016-131053/summary.json), [FP16 kernel 집계](/home/130bb56/DiffKV/logs/performance/target-profile-20261016-131053/fp16/fp16.kernels.json), [DiffKV kernel 집계](/home/130bb56/DiffKV/logs/performance/target-profile-20261016-131053/diffkv/diffkv.kernels.json)에 raw trace 경로와 hash도 있습니다. CUDA launch correlation으로 귀속되지 않은 GPU event는 양쪽 모두 0개입니다.

| 항목, decode당 ms | FP16 | DiffKV |
|---|---:|---:|
| Attention main | 2.412 | 5.202 |
| Attention reduce | 0.108 | 0.223 |
| Attention 합 | 2.520 | 5.425 |
| Cache 갱신 | 0.100 | 0.840 |
| GEMM 계열 합 | 19.298 | 19.282 |
| 전체 kernel 합 | 22.686 | 26.335 |
| GPU busy interval union | 22.720 | 26.398 |
| Wall time | 26.308 | 41.649 |
| Wall minus GPU busy | 3.588 | 15.252 |
| 명시적 device sync, 호출 수 | 0 | 100 |
| D2H memcpy, 호출 수 | 2 | 36 |

**이번 조건에서도 attention은 약 2.15배, cache 갱신은 약 8.42배 오래 걸렸습니다.** GPU kernel 시간 합의 증가 3.649ms 중 attention 증가가 2.905ms, cache 증가가 0.740ms입니다. GEMM 시간은 거의 같습니다. 이 값은 capture 구간 내부의 kernel 차이이며, 전체 E2E 차이의 기여율로 환산하지 않습니다.

### 4.2 실제 pruning은 이 snapshot에서 약 0.35%였습니다

[Cache metadata snapshot](/home/130bb56/DiffKV/logs/performance/target-profile-20261016-131053/diffkv/cache-metadata.json)은 capture 직전 decode ordinal 1023의 cache append 전 상태입니다. 현재 token append 전 KV 기준 길이는 1534이며, append 후 논리 attention context는 1535입니다. Snapshot의 D2H 복사는 capture 범위 밖에서 한 번 수행했습니다.

| 전체 request, layer, KV head에 걸친 token entry | 개수 |
|---|---:|
| Pruning이 없을 때 | 3,141,632 |
| High K8V4 | 1,393,687 |
| Low K4V2 | 1,736,846 |
| 실제 보관 합 | 3,130,533 |
| Pruned | 11,099, 0.3533% |

보관 entry 중 high 44.52%, low 55.48%입니다. **양자화로 byte 수는 줄지만, 처리해야 하는 token entry 수는 99.65% 남아 있습니다.** 이 상태에서 기존 token의 score 검사, 역양자화, score 쓰기를 수행하므로, pruning으로 연산량을 크게 줄이는 효과는 이 시점에 거의 없습니다. 한 시점의 전체 head 집계이며 모든 decode 구간의 평균이나 과거 10/10 빌드의 pruning 비율은 아닙니다.

[입력 생성 코드](/home/130bb56/DiffKV/benchmarks/run_offline_e2e.py:452)는 BOS를 포함한 23-token seed를 반복합니다. ISL 512 입력에는 BOS 128000이 23번 들어갑니다. 논문 threshold는 [MATH training split에서 calibration](/home/130bb56/DiffKV/DiffKV.pdf#page=10)한 값이므로, 설정이 같아도 이 반복 synthetic 입력에서 같은 pruning 비율을 기대할 근거는 없습니다. 입력 반복이 낮은 pruning 비율을 일으킨 정도는 입력을 교체한 별도 비교 없이 정량적으로 단정하지 않습니다.

### 4.3 기존 ISL 1024, OSL 512 trace와의 대조

이 절은 **과거 ISL 1024, OSL 512, B8, decode 17~80의 64-step trace**입니다. 질문의 ISL 512, OSL 2048 수치에 직접 대입하지 않습니다. [집계](/home/130bb56/DiffKV/logs/performance/llama-profile-torch-comparison.json), [FP16 trace](/home/130bb56/DiffKV/logs/performance/llama-profile-torch-fp16.trace.json), [DiffKV trace](/home/130bb56/DiffKV/logs/performance/llama-profile-torch-diffkv.trace.json)의 hash를 검증했고, raw trace에서 kernel 시간을 재합산해 일치를 확인했습니다.

| 항목, decode당 ms | FP16 | DiffKV |
|---|---:|---:|
| Wall time | 35.101 | 47.835 |
| GPU busy interval union | 21.833 | 24.759 |
| Wall minus GPU busy | 13.268 | 23.076 |
| GEMM 계열 합 | 19.190 | 19.235 |
| Attention main | 1.696 | 3.721 |
| Attention reduce | 0.113 | 0.180 |
| Cache copy 또는 compression | 0.094 | 0.790 |
| 명시적 device sync, 호출 수 | 0 | 100 |
| D2H memcpy, 호출 수 | 2 | 36 |

Attention 합은 1.809에서 3.901ms, cache 갱신은 0.094에서 0.790ms로 증가했습니다. GEMM 시간은 거의 같습니다. **이 trace에서는 KV 저장량이 줄어도 attention과 cache 커널 자체는 더 오래 실행됐습니다.**

Profiler와 annotation은 wall time을 교란합니다. Sync API 시간 7.513ms에는 GPU 실행 대기도 포함되므로 순수 overhead로 간주하거나 kernel 시간에 더하면 안 됩니다. Wall minus busy 역시 CPU 작업, launch gap, profiler overhead가 섞인 값입니다. 개선폭에는 2절의 profiler 없는 전체 생성 측정을 사용합니다.

## 5. CUDA 구현에서 확인한 비용

### 5.1 Partition 5760과 register 사용량

실제 설정 [8,4,4,2,1,1,1,1]은 [high page 32, low page 60인 template](/home/130bb56/DiffKV/csrc/attention/sparse_attention_kernels.cu:1358)을 선택합니다. [Partition 식](/home/130bb56/DiffKV/csrc/attention/sparse_attention_kernels.cu:1210)은 ceil(4800/(32*60))*(32*60)=5760입니다.

논리적 context가 513~2559인 이번 workload에서는 retained KV 길이도 2559 이하이므로 main grid는 항상 (32 query heads, 8 requests, 1 partition), block 256입니다. FP16 V2는 partition 512, block 128이므로 같은 범위에서 partition 2~5개입니다.

DiffKV는 짧은 context에서도 [dynamic shared 23,312 bytes](/home/130bb56/DiffKV/csrc/attention/sparse_attention_kernels.cu:1218)를 예약합니다. Static 2,496 bytes를 더한 25,808 bytes는 이번 trace와 일치합니다. 이번 main trace의 register는 thread당 140개입니다. 140*256=35,840이므로 registers/SM 65,536에서 CTA 두 개가 동시에 상주할 수 없습니다. FP16 main은 93 registers/thread, block 128, shared 2,336 bytes였습니다.

기존 trace의 occupancy estimate는 DiffKV 17%, FP16 42%입니다. Launch 자원 기반 추정이며 실제 hardware counter 측정은 아닙니다. 큰 partition과 register 사용량은 구체적인 튜닝 대상이지만, 이를 바꾼 CUDA 빌드 비교 없이 각각의 기여도를 확정하지는 않습니다.

### 5.2 Cache 갱신의 score scan과 직렬 양자화

FP16은 [새 K와 V를 복사](/home/130bb56/DiffKV/.cache/baselines/vllm-1db83e3/csrc/cache_kernels.cu:185)합니다. DiffKV는 매 decode, 매 layer에서 기존 [high token](/home/130bb56/DiffKV/csrc/cache_kernels.cu:861), [low token](/home/130bb56/DiffKV/csrc/cache_kernels.cu:903)의 score와 position을 읽고, minimum reduction으로 pruning과 high-to-low 전환 대상을 정합니다. 필요하면 기존 token을 requantize합니다.

이후 새 [K](/home/130bb56/DiffKV/csrc/cache_kernels.cu:1234), [V](/home/130bb56/DiffKV/csrc/cache_kernels.cu:1271)를 양자화합니다. 현재 CHUNKS=1이면 thread 0 하나가 K와 V 각각 128차원의 min/max와 packing을 처리합니다. Requantization의 min/max와 packing도 같은 구조이며, unpack은 여러 thread가 수행합니다.

Context가 길어지면 attention뿐 아니라 score 검사량도 증가합니다. KV buffer64는 최근 token을 보호하는 조건이며, 64 token마다 한 번만 compression을 실행한다는 의미가 아닙니다.

### 5.3 Attention의 역양자화와 score 갱신

K와 V를 읽을 때 scale, zero point, unpack, dequantization이 필요합니다. 또 [FP32 tmp_scores](/home/130bb56/DiffKV/vllm/model_executor/layers/sparse_attention_big_kernel.py:399)를 매 layer 만들고 main이 score를 저장한 뒤, reduce가 GQA query head 4개의 score를 다시 읽어 [KV token score를 갱신](/home/130bb56/DiffKV/csrc/attention/sparse_attention_kernels.cu:876)합니다.

두 경로 모두 cache kernel 1개와 attention main, reduce 2개를 실행합니다. 단순히 kernel 개수가 많아서라는 설명은 부정확합니다. 각 kernel의 작업, launch 자원, host 대기가 다릅니다.

### 5.4 Prefill은 별도의 추가 작업이 있습니다

DiffKV prefill은 [Triton attention에서 qk_products를 추출](/home/130bb56/DiffKV/vllm/model_executor/layers/sparse_attention_big_kernel.py:202)하고, softmax score sum 계산과 prompt cache 분류, 양자화를 추가합니다. Decode 동기화를 제거해도 이 추가 작업은 남습니다. 과거 대상의 prefill 차이는 14.460ms이며, 그 내부 원인별 시간은 이번 조사에서 별도 계측하지 않았습니다.

## 6. 논문과 비교할 때 수정할 해석

**논문도 fixed batch8에서 latency 향상을 보고합니다.** [PDF p12 Figure15b](/home/130bb56/DiffKV/DiffKV.pdf#page=12)는 sequence length4096에서 1.4~1.6배를 제시합니다. “큰 batch를 사용하지 않아서”만으로 이번 차이를 설명할 수 없습니다.

다만 Figure15 범례는 uniform K8V8, K8V4, K4V2입니다. 현재 adaptive K8V4+K4V2의 정확한 대조 실험은 아닙니다. 해당 그림에는 모델명, ISL과 OSL의 분리, pruning 조건이 명시되지 않았습니다. Sequence length를 prompt length라고 단정해서도 안 됩니다.

[Figure17](/home/130bb56/DiffKV/DiffKV.pdf#page=13)은 MATH 1000 requests, Llama max output 4K이며, 더 큰 batch가 throughput을 높이는 효과를 설명합니다. FP16도 모든 요청을 처리하는 이번 fixed B8에는 그 capacity 이점이 직접 반영되지 않습니다. 논문 GPU는 [L40 48GB](/home/130bb56/DiffKV/DiffKV.pdf#page=9), 현재는 RTX3090 24GB입니다. 하드웨어 차이가 있다는 사실과 그것이 느려진 원인이라는 주장은 구분해야 합니다.

현재 K8V4/K4V2, thresholds [0.02,1.0], W64는 논문 Llama 정책과 일치합니다. 최근 64 tokens도 FP16 residual window에 남기지 않고 high K8V4로 즉시 양자화하므로, 이 설정을 잘못 맞춘 문제는 아닙니다.

Weights는 양쪽 모두 FP16, 8,030,261,248 parameters로 약 16.06GB입니다. 기존 trace에서 FP16의 attention과 cache 합은 전체 kernel 시간의 약 8.7%였고 GEMM이 대부분이었습니다. KV 저장량 감소가 전체 모델 시간의 같은 비율 감소를 보장하지 않습니다. 이 비율도 다른 길이나 GPU에 일반화하지 않습니다.

## 7. 검증 자료와 재실행

당시 혼합 attention 조사에 사용한 `diagnose_sparse_low_value.py`는 정리 과정에서 삭제했습니다. 이 진단은 high-only, low-only, mixed 10개 case를 기존 binary로 실행해 FP64 CPU reference와 비교했습니다. 모두 통과했고 최대 output 오차 9.45e-4, score 오차 3.73e-9였습니다. [결과와 hash](/home/130bb56/DiffKV/logs/performance/sparse-low-value-repro-20261016.json)를 저장했습니다. 제한된 입력 검증이며 모델 전체 correctness를 보장하는 검사는 아닙니다.

현재 runner에서는 새 output directory를 지정해 다음처럼 재실행할 수 있습니다. 이 명령은 현재 인터페이스용 예시이며, 위 표를 측정할 당시의 명령을 변경한 기록이 아닙니다. 기본 27개 grid에서 `--case-indices 24`가 ISL 512, OSL 2048, B8을 선택하므로 기존 `max_model_len=4096`, `max_num_seqs=32` 설정을 유지합니다.

```bash
/home/130bb56/DiffKV/.cache/baselines/fp16-env/bin/python -u \
  /home/130bb56/DiffKV/benchmarks/run_offline_e2e.py \
  --backends diffkv --diffkv-variant no-sync --case-indices 24 --gpu 0 \
  --output-dir /tmp/diffkv-no-sync-new \
  --report /tmp/diffkv-no-sync-new.md
```

`--diffkv-variant` 값은 `original`, `no-sync`, `no-sync-bulk-lengths`입니다. FP16은 `--backends fp16`을 사용하며 DiffKV variant를 적용하지 않습니다. 당시 조사에 사용한 `profile_target_decode.py`는 같은 grid 설정에서 decode 1024~1055의 32 steps를 수집하고 결과에 diagnostic_only를 표시했습니다. 이 진단 파일은 정리 과정에서 삭제했으며, 기존 trace와 JSON 측정 기록은 보존합니다.

과거 OSL 512 원자료 [llama-performance-gap.json](/home/130bb56/DiffKV/logs/performance/llama-performance-gap.json)도 유지합니다. 그 실험에서 ISL 512,1024,2048의 E2E 차이 중 decode 비중은 각각 99.46%,98.68%,97.54%였습니다. 이번 대상의 99.43%와 별개의 측정입니다.

## 8. 최적화 경로의 offline 전체 grid, GPU 0 측정

2026-10-16 14:57 KST에 ISL [512, 1024, 2048], OSL [512, 1024, 2048], B [8, 16, 32]의 27개 조건을 완료했습니다. DiffKV는 no-sync-bulk-lengths 경로이며 CUDA binary는 2절과 같습니다. [evaluation.md의 기존 optimized 표](/home/130bb56/DiffKV/evaluation.md:294)에서 GPU 1로 측정했던 14개 조건의 수치를 삭제하고 GPU 0 재측정값으로 교체했습니다. 기존 GPU 0의 13개 조건은 보존했습니다. 최종 54개 worker 결과의 physical GPU는 모두 0, UUID는 f2139c13-2e60-9ae2-f007-9965bca6b295입니다.

FP16은 14개 조건에서 완료했고 13개 조건은 고정 batch를 유지할 KV cache 용량이 부족했습니다. DiffKV는 27개 모두 완료했습니다. 양쪽 완료 14개 조건에서 E2E는 수치상 DiffKV가 10개에서 짧고 4개에서 길었습니다. 그중 (512, 1024, 8)은 29.382206초와 29.384506초로 차이가 2.30ms입니다. DiffKV가 더 길었던 나머지 조건은 (512, 512, 16), (1024, 512, 8), (512, 512, 8)이며 각각 6.19%, 3.33%, 7.27% 증가했습니다. TTFT와 prefill throughput은 양쪽 완료 14개 모두 FP16이 우세했습니다. 각 조건 1회 측정이므로 작은 차이를 안정된 우위로 해석하지 않습니다.

질문의 (512, 2048, 8)은 FP16 E2E 66.095016초, 최적화 DiffKV 63.166996초로 **4.43% 단축, 1.04635배**였습니다. TPOT는 31.853200에서 30.421310ms/token으로 4.50% 감소했습니다. 이 값은 이번 전체 grid의 재측정이며, 2절의 단일 조건 진단과 별도의 실행입니다. [최종 검증과 조건별 계산](/home/130bb56/DiffKV/logs/performance/offline-optimized-grid-20261016-134916-315827/gpu0-validation.json)에 원본 GPU 0 수치 보존, GPU 0 재측정 provenance, 입력과 설정 일치, 출력 길이, 고정 decode batch, preemption 0회, source와 extension hash 검증을 기록했습니다. 양쪽 완료 14쌍의 출력 hash도 관측상 모두 같았습니다.

## 9. ISL 512, OSL 4096, B8과 B4 추가 측정

2026-10-16 GPU 0에서 각 조건 1회, DiffKV no-sync-bulk-lengths 경로로 측정했습니다. B8의 E2E는 FP16 121.821초, DiffKV 134.198초로 DiffKV가 10.16% 길었고, B4는 FP16 125.424초, DiffKV 121.971초로 DiffKV가 2.75% 짧았습니다. TPOT는 B8 29.538 / 32.555ms/token, B4 30.517 / 29.672ms/token입니다. 양쪽 모두 OOM 없이 출력 4096토큰, decode 4095회, 고정 batch, preemption 0회를 확인했습니다. 이번 두 조건의 공통 engine 설정은 max_model_len=4608, max_num_seqs=8, max_num_batched_tokens=8192입니다. 세 Python 최적화 함수의 hash와 CUDA binary hash는 8절 실험과 같습니다. [수치, 검증, 실행 설정](/home/130bb56/DiffKV/logs/performance/long-context-20261016-osl4096-gpu0/analysis.json)에 기록했습니다.
