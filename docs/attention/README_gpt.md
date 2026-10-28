# FP16과 DiffKV-v1 Attention 구현 요약

기준은 첨부한 성능표의 실제 실행입니다. FP16은 `.cache/baselines/vllm-1db83e3`, DiffKV-v1은 현재 저장소에 `no-sync-bulk-lengths`를 적용한 경로입니다. 표의 14개 조건은 B=8,16,32, ISL/OSL=512~2048 범위이며, 두 경로 모두 모델 가중치는 FP16입니다.

여기서 Attention 본체는 Q, K, V를 받아 다음 출력을 계산하는 부분입니다. QKV projection, RoPE는 그 전에 실행하고 output projection은 그 뒤에 실행합니다.

```text
P = softmax(scale * QKᵀ + causal mask)
O = PV
```

## 1. 실행되는 구현

| 구간 | FP16 baseline | DiffKV-v1 |
|---|---|---|
| Prefill | xFormers 호출, 이 측정에서는 FlashAttention CUDA 커널 실행 | 저장소의 Triton `attn_fwd` 실행 |
| Decode | FP16 캐시를 읽는 vLLM PagedAttention V1 또는 V2 | 양자화 캐시를 역양자화하며 읽는 `sparse_paged_attention` main과 reduce |
| Prefill의 K/V | 압축 전 FP16 K/V | 압축 전 FP16 K/V |
| Decode의 K/V | FP16 캐시 | high K8/V4, low K4/V2 캐시 |

FP16의 소스는 xFormers 구현을 자동 선택합니다. **이 표의 측정에서 Flash 커널이 실행됐다는 의미**이며, 모든 환경에서 같은 커널을 고정 호출한다는 뜻은 아닙니다. [FP16 호출](../../.cache/baselines/vllm-1db83e3/vllm/model_executor/layers/attention.py#L146), [실제 Flash 실행 로그](../../logs/performance/nsys-base1395-repeat3-20261019-001155/b8-isl512-osl512-fp16/r1.summary.json#L5000), [v1 prefill 호출](../../vllm/model_executor/layers/sparse_attention_big_kernel.py#L200)

### 파일 경로와 함수

아래 경로는 `/home/130bb56/DiffKV` 기준입니다. 파일명 링크를 누르면 해당 구현으로 이동합니다.

| 경로 | 소스 파일 | 함수와 연결 지점 |
|---|---|---|
| FP16 prefill, decode 진입 | [.cache/baselines/vllm-1db83e3/vllm/model_executor/layers/attention.py](../../.cache/baselines/vllm-1db83e3/vllm/model_executor/layers/attention.py#L59) | `PagedAttention.forward()` 59행, prefill의 xFormers 호출 146행, decode 선택 함수 `_paged_attention()` 212행 |
| FP16 prefill xFormers 진입 | [.cache/baselines/fp16-env/lib/python3.11/site-packages/xformers/ops/fmha/__init__.py](../../.cache/baselines/fp16-env/lib/python3.11/site-packages/xformers/ops/fmha/__init__.py#L362) | `memory_efficient_attention_forward()` 362행 → 내부 함수 480행 → dispatch 486행 → 선택한 `op.apply()` 490행 |
| FP16 prefill 구현 선택 | [.cache/baselines/fp16-env/lib/python3.11/site-packages/xformers/ops/fmha/dispatch.py](../../.cache/baselines/fp16-env/lib/python3.11/site-packages/xformers/ops/fmha/dispatch.py#L126) | `_dispatch_fw()` 126행, 지원 조건을 만족하는 op를 고르는 `_run_priority_list()` 55행 |
| FP16 prefill Flash 위임 | [.cache/baselines/fp16-env/lib/python3.11/site-packages/xformers/ops/fmha/flash.py](../../.cache/baselines/fp16-env/lib/python3.11/site-packages/xformers/ops/fmha/flash.py#L632) | `FwOp.apply()` 632행 → `_flash_fwd()` 91행 → 이 환경의 `torch.ops.aten._flash_attention_forward()` 호출 116행 |
| FP16 decode CUDA | [.cache/baselines/vllm-1db83e3/csrc/attention/attention_kernels.cu](../../.cache/baselines/vllm-1db83e3/csrc/attention/attention_kernels.cu#L85) | 공통 계산 `paged_attention_kernel()` 85행, V1 진입 400행, V2 진입 427행, V2 reduce 455행 |
| DiffKV-v1 prefill, decode 진입 | [vllm/model_executor/layers/sparse_attention_big_kernel.py](../../vllm/model_executor/layers/sparse_attention_big_kernel.py#L111) | `SparsePagedAttention.forward()` 111행, prefill 호출 200행, decode 호출 423행 |
| DiffKV-v1 prefill Triton | [vllm/model_executor/layers/triton_flash_attention.py](../../vllm/model_executor/layers/triton_flash_attention.py#L191) | Python wrapper `_attention.forward()` 582행 → `attn_fwd()` 191행 → QK, softmax, PV를 계산하는 `_attn_fwd_inner()` 63행 |
| DiffKV-v1 decode Python/CUDA 연결 | [csrc/pybind.cpp](../../csrc/pybind.cpp#L25) | `ops.sparse_paged_attention`을 C++ `sparse_paged_attention()`에 연결하는 25행 |
| DiffKV-v1 decode CUDA | [csrc/attention/sparse_attention_kernels.cu](../../csrc/attention/sparse_attention_kernels.cu#L1407) | C++ 진입 `sparse_paged_attention()` 1407행 → main `sparse_paged_attention_kernel()` 125행, reduce `reduce_kernel()` 776행 |
| DiffKV-v1 실행 패치 | [benchmarks/diffkv_v1.py](../../benchmarks/diffkv_v1.py#L32) | `apply_variant()` 32행, 동기화 제거와 길이 읽기 변경 57행 |

FP16 prefill의 Flash CUDA 구현 원문은 해당 로컬 설치 패키지에 없습니다. 위 표는 확인 가능한 Python 위임 경로까지 표시한 것이며, CUDA 실행 자체는 측정 로그의 `pytorch_flash::flash_fwd_kernel`로 확인했습니다.

## 2. Prefill의 차이

**FP16:** prompt 전체의 Q/K/V로 causal attention을 계산합니다. K/V를 양자화하지 않으며, xFormers가 선택한 Flash 커널이 QK, softmax, PV를 처리합니다.

**DiffKV-v1:** 같은 FP16 Q/K/V를 사용하지만 자체 Triton 구현으로 계산합니다. Query 64개, key/value 32개 크기의 tile을 순회하며, FP32의 최대값, 지수합, 출력 누산값을 갱신하는 online softmax를 사용합니다. 전체 확률 행렬을 저장하지 않고 출력을 계산합니다. [tile 설정](../../vllm/model_executor/layers/triton_flash_attention.py#L168), [online softmax와 PV](../../vllm/model_executor/layers/triton_flash_attention.py#L141)

v1의 Attention 커널은 출력과 함께 **마지막 최대 64개 query의 QK logits를 추가로 기록**합니다. 이 버퍼는 `[B, query_heads, 64, 최대 prompt 길이]`인 FP32 tensor입니다. 예를 들어 B8, query heads32, 길이2048이면 128 MiB입니다. 따라서 FP16과 prefill의 수학적 출력 목표는 같아도, 실행 커널과 추가 메모리 작업은 다릅니다. [버퍼 할당](../../vllm/model_executor/layers/triton_flash_attention.py#L622), [logits 저장](../../vllm/model_executor/layers/triton_flash_attention.py#L131)

**v1도 prefill 출력을 압축 전 K/V로 계산합니다.** 이후의 중요도 계산, pruning, 양자화는 미래 decode용 캐시를 만드는 별도 처리입니다. 이 캐시 관리 처리를 Attention 출력 연산과 혼동하면 안 됩니다.

## 3. FP16 decode

새 token의 K/V를 FP16 캐시에 저장한 뒤, query 하나가 보존된 전체 context를 참조합니다. 캐시의 K/V를 그대로 읽어 QK, softmax, PV를 계산하며 역양자화는 없습니다.

표의 모델은 query heads가 32개입니다. 실제 선택은 다음과 같고, 14개 조건의 로그와 일치합니다.

| Decode batch | 실제 커널 | 동작 |
|---|---|---|
| B8, B16 | PagedAttention V2 main, reduce | Context를 512-token partition으로 나눠 부분 출력을 계산한 뒤, softmax 통계로 보정하여 결합합니다. |
| B32 | PagedAttention V1 | CUDA thread block 하나가 query head 하나의 전체 context를 처리합니다. 별도 partition reduce가 없습니다. |

소스의 V1 선택 조건은 `context_len <= 8192`이면서 `ceil(context_len/512)==1 또는 B*query_heads>512`입니다. 표에서는 decode 길이가 512를 넘고, B16의 곱은 정확히 512라서 V2입니다. 이 **PagedAttention V1이라는 이름은 DiffKV-v1과 별개**입니다. [선택 조건과 호출](../../.cache/baselines/vllm-1db83e3/vllm/model_executor/layers/attention.py#L235)

## 4. DiffKV-v1 decode

현재 token의 K/V를 캐시에 반영한 뒤 `ops.sparse_paged_attention`을 호출합니다. 이 호출은 다음 두 CUDA 커널을 실행합니다. [호출](../../vllm/model_executor/layers/sparse_attention_big_kernel.py#L423), [main과 reduce launch](../../csrc/attention/sparse_attention_kernels.cu#L1087)

```text
main:   packed K 읽기, 역양자화 → QK → softmax
        packed V 읽기, 역양자화 → PV → 임시 출력과 확률 저장
reduce: 최종 출력 기록, KV token 중요도 점수 누적
```

캐시는 `int16` tensor를 저장 용기로 사용하며, 내부에는 unsigned packed K/V, FP16 scale와 실수 offset, FP32 누적 점수, INT32 위치 기준값이 있습니다. 대표 설정에서 한 페이지는 3392개의 16-bit 원소이며 high는 32 tokens, low는 60 tokens를 담습니다. K8/V4와 K4/V2는 **캐시 저장 정밀도**입니다. Query와 output까지 8-bit 또는 4-bit가 된다는 뜻은 아닙니다.

Main의 thread block은 query head 하나를 담당합니다. Packed 정수 code를 `code*scale+offset`으로 FP32 복원한 뒤 QK와 PV에 사용합니다. Softmax와 출력 누산도 FP32이며 임시 출력은 query dtype으로 저장합니다. KV head 하나를 여러 query head가 공유하면, 각 query head의 block이 같은 캐시를 각각 읽습니다. [K 읽기와 역양자화](../../csrc/attention/sparse_attention_kernels.cu#L352), [softmax](../../csrc/attention/sparse_attention_kernels.cu#L483), [PV](../../csrc/attention/sparse_attention_kernels.cu#L611)

이 설정의 실제 partition 크기는 `ceil(4800/(32*60))*(32*60)=5760`입니다. 표의 context 길이는 4096 이하이므로 **단일 partition**으로 계산합니다. Python의 4800은 임시 버퍼 용량 계산 기준이며, 실제 CUDA partition과 구분해야 합니다. [partition 계산](../../csrc/attention/sparse_attention_kernels.cu#L1214)

**단일 partition이어도 reduce는 실행됩니다.** 임시 출력을 최종 출력으로 복사하고, 같은 KV head를 공유하는 query head들의 token별 attention 확률 중 최댓값을 캐시 점수에 더합니다. 이 점수가 다음 decode의 캐시 압축 판단에 쓰이므로 reduce를 단순히 제거할 수 없습니다. [단일 partition reduce](../../csrc/attention/sparse_attention_kernels.cu#L867)

Pruning은 캐시에 남길 token을 고르는 별도 단계입니다. Attention main이 보존된 token 중 다시 top-k를 고르는 구조는 아니며, main은 현재 남은 high/low token 전체를 참조합니다.

## 5. v1에 이미 적용된 최적화

`no-sync-bulk-lengths`는 Attention CUDA 커널을 바꾸는 옵션이 아닙니다. 다음 Python 비용을 줄입니다. [benchmarks/diffkv_v1.py:57](../../benchmarks/diffkv_v1.py#L57)

| 변경 | 실제 의미 |
|---|---|
| Attention forward와 engine step의 명시적 `torch.cuda.synchronize()` 제거 | CPU가 각 지점에서 GPU 완료를 기다리던 호출을 제거합니다. 같은 CUDA stream의 연산 순서는 유지됩니다. |
| Layer별 길이 읽기를 `.cpu().tolist()` 한 번으로 묶음 | 32-layer 모델에서 32번의 scalar `.item()` 대신 step당 한 번 길이 벡터를 읽습니다. CPU 복사 자체는 여전히 동기식입니다. |

따라서 이 두 변경을 다시 v1 이후의 신규 최적화로 계산하면 안 됩니다.

## 6. 표의 의미와 추가 최적화 검토

표의 Attention은 **전체 생성의 prefill과 모든 decode step에서 Attention으로 분류한 커널 시간**입니다. FP16의 main/reduce, v1의 main/reduce를 포함합니다. QKV/output projection은 GEMM, 캐시 갱신은 KV Update로 따로 집계합니다. [benchmarks/analyze_offline_nsys.py:29](../../benchmarks/analyze_offline_nsys.py#L29)

**이 표만으로 prefill이나 decode가 compute-bound인지 memory-bound인지 모릅니다.** 기존 표는 실행 시간의 분해이며, 해당 조사에는 Nsight Compute 카운터나 연산량, 전송량에 따른 병목 판정이 포함되지 않았습니다. [profiling-rtx3090-explain.md:400](../../profiling-rtx3090-explain.md#L400)

다음은 현재 v1에서 검토할 수 있는 후보입니다. 성능 개선이 검증됐다는 뜻은 아닙니다.

| 후보 | 구현에서 확인한 비용 | 적용 전에 확인할 것 |
|---|---|---|
| Decode 임시 버퍼 재사용 | Layer마다 output, 확률, 부분 출력 버퍼를 생성합니다. | 실제 allocator 비용을 측정하고, 연속 메모리와 커널의 partition stride, stream별 수명을 보존해야 합니다. |
| Decode metadata 읽기 개선 | 같은 token의 scale와 offset을 여러 thread가 읽습니다. | Broadcast나 staging의 추가 명령, 동기화 비용까지 포함해 기존 커널과 비교해야 합니다. |
| Prefill tile 설정 비교 | 현재 Triton은 64×32의 한 설정을 사용합니다. | 길이, GPU별 register와 shared memory 사용량, 마지막 64 query logits 기록을 함께 검증해야 합니다. |

최적화 검증은 prefill과 decode를 나눠 수행해야 합니다. v1 decode는 출력뿐 아니라 **누적 점수와 다음 step의 캐시 선택까지** 같아야 하며, 커널 시간 개선과 전체 생성 시간 개선을 따로 확인해야 합니다. 이 요약을 위해 새로운 GPU 성능 측정은 수행하지 않았습니다.
