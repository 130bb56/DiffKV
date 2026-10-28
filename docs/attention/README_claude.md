# Attention 구현 안내: FP16 vs DiffKV-v1

[README_gpt.md](README_gpt.md)의 내용을 다시 정리한 문서입니다. 이 문서는 **두 경로에서 어떤 코드가 실제로 실행되는지**를 다룹니다. 성능 분석은 [profiling_result_claude.md](profiling_result_claude.md)를 보세요.

- **FP16:** `.cache/baselines/vllm-1db83e3`에 있는 원래 vLLM입니다.
- **DiffKV-v1:** 이 저장소의 DiffKV에 `no-sync-bulk-lengths` 패치를 적용한 경로입니다.
- **공통 조건:** 모델 가중치는 둘 다 FP16이고, 성능표의 14개 조건(B 8/16/32, ISL/OSL 512~2048)을 기준으로 합니다.

## 요약

| | FP16 | DiffKV-v1 |
|---|---|---|
| **Prefill 커널** | FlashAttention (xFormers 경유) | 저장소의 Triton `attn_fwd` |
| **Prefill이 읽는 K/V** | 압축 전 FP16 | 압축 전 FP16 (같음) |
| **Decode 커널** | vLLM PagedAttention V1 또는 V2 | `sparse_paged_attention` main + reduce |
| **Decode가 읽는 K/V** | FP16 캐시 | 양자화 캐시: high는 K8V4, low는 K4V2 |
| **Tensor Core** | Decode에서 미사용 | Decode에서 미사용 |

1. **Prefill 출력은 두 경로 모두 압축 전 K/V로 계산합니다.** 다른 것은 커널 구현과, DiffKV가 중요도 계산용 값을 추가로 저장하는 것입니다.
2. **양자화는 캐시에 저장할 때만 적용됩니다.** DiffKV decode는 8/4/2-bit 캐시를 읽어 FP32로 복원한 뒤 계산합니다. INT4/INT2로 연산하지 않고, query와 output도 FP16 그대로입니다.
3. **압축과 pruning은 attention 커널 밖에서 일어납니다.** 어떤 토큰을 low로 내리거나 버릴지는 KV Update 단계에서 정합니다. Attention main은 그 시점에 남은 토큰을 전부 읽습니다.

## 1. Attention이 하는 일

Transformer layer 하나에서 attention의 위치는 다음과 같습니다.

```text
QKV projection (GEMM) → RoPE → [Attention] → output projection (GEMM)
```

Attention 자체가 계산하는 것은 다음 두 줄입니다.

```text
P = softmax(scale · Q Kᵀ + causal mask)
O = P V
```

- **Prefill:** prompt 전체를 한 번에 처리합니다. Query가 수천 개라서 연산량이 많습니다.
- **Decode:** 새 토큰 하나의 query가 지금까지 쌓인 K/V 전체를 읽습니다. 연산은 적고 메모리 읽기가 대부분입니다.

## 2. Prefill

### FP16
vLLM이 xFormers의 `memory_efficient_attention_forward()`를 부르고, xFormers가 조건에 맞는 구현을 고릅니다. 이번 측정에서는 PyTorch에 포함된 FlashAttention(`pytorch_flash::flash_fwd_kernel`)이 실행됐습니다. 이는 nsys 로그로 확인한 것이고, 다른 환경에서도 항상 같은 커널이 선택된다는 뜻은 아닙니다.

### DiffKV-v1
저장소에 있는 Triton 커널 `attn_fwd`를 씁니다. FlashAttention과 같은 방식입니다.

- Query 64개 × key 32개 크기의 tile 단위로 계산합니다.
- softmax는 최댓값과 지수합을 FP32로 갱신해 가며 계산합니다(online softmax).
- 전체 확률 행렬은 메모리에 만들지 않습니다.

**FP16과 다른 점은 QK logits를 추가로 저장한다는 것입니다.**
- 저장 대상: 마지막 최대 64개 query의 QK logits입니다.
- 크기: `[B, query head 32, 64, prompt 길이]` 크기의 FP32 버퍼입니다. B8, 길이 2048이면 128 MiB입니다.
- 용도: 이 값으로 prompt 토큰의 중요도를 계산하고, 그 결과로 어떤 토큰을 high/low로 저장하거나 버릴지 정합니다.

[버퍼 할당](../../vllm/model_executor/layers/triton_flash_attention.py:622), [logits 저장](../../vllm/model_executor/layers/triton_flash_attention.py:131)

## 3. Decode: FP16

새 토큰의 K/V를 FP16 캐시에 쓴 뒤, query가 캐시의 K/V를 그대로 읽어 계산합니다. 역양자화는 없습니다. 읽기 단위는 K가 `uint2`(8바이트), V가 `uint4`(16바이트)입니다.

vLLM은 조건에 따라 커널 두 종류 중 하나를 고릅니다. 조건은 `문맥 길이 ≤ 8192` 이면서, `512토큰 분할이 1개뿐`이거나 `요청 수 × query head 수 > 512`이면 V1입니다.

| 측정 조건 | 실행 커널 | 동작 |
|---|---|---|
| B8, B16 | **V2** main + reduce | 문맥을 512토큰씩 나눠 따로 계산한 뒤 reduce에서 합칩니다 |
| B32 | **V1** | CTA 하나가 query head 하나의 문맥 전체를 처리하고, reduce가 없습니다 |

B16은 16 × 32 = 512로, 512를 넘지 않으므로 V2입니다. [선택 코드](../../.cache/baselines/vllm-1db83e3/vllm/model_executor/layers/attention.py:235)

> vLLM의 "PagedAttention V1"과 "DiffKV-v1"은 이름만 같고 서로 무관합니다.

## 4. Decode: DiffKV-v1

### 4.1 한 layer의 decode 순서

```text
QKV projection → RoPE
→ KV Update: 새 토큰의 K/V를 캐시에 쓰고, 필요하면 압축(high→low 강등, pruning)
→ Attention main: 남은 토큰 전부 읽기 → 복원 → QK → softmax → PV
→ Attention reduce: 출력 기록 + 토큰별 중요도 점수 누적
→ output projection
```

이 문서와 성능표의 "Attention"은 main과 reduce만 가리킵니다. KV Update는 별도 항목입니다.

### 4.2 KV 캐시 구조

DiffKV 캐시는 dtype이 int16인 tensor 하나입니다. 실제 값은 int16과 상관없이 아래 형식의 바이트로 들어 있습니다.

**페이지:** 한 페이지는 int16 3,392개, 즉 6,784바이트입니다.

| 페이지 종류 | 페이지당 토큰 수 | 토큰 하나에 들어가는 것 (KV head 하나 기준) |
|---|---:|---|
| high (K8V4) | 32 | K 128B, V 64B, K/V의 scale·zero point(FP16, 8B), 중요도 점수(FP32)와 토큰 인덱스(INT32) 8B |
| low (K4V2) | 60 | K 64B, V 32B, K/V의 scale·zero point(FP16, 8B), 중요도 점수(FP32)와 토큰 인덱스(INT32) 8B |

scale과 zero point는 토큰마다 K와 V에 하나씩 있습니다(quant group 1). 토큰당 크기는 high가 208B, low가 112B입니다. [바이트 정의](../../vllm/config.py:390)

**페이지 목록(block table):** 페이지 목록은 **layer × 요청 × KV head마다 따로** 있습니다. head마다 남은 토큰 수가 다르기 때문입니다. 같은 목록을 양쪽 끝에서 씁니다.

```text
block table: [ high 페이지 → → →        ← ← ← low 페이지 ]
               왼쪽 끝부터 채움            오른쪽 끝부터 채움
```

[block table 위치 계산](../../csrc/attention/sparse_attention_kernels.cu:328)

### 4.3 Main 커널

`sparse_paged_attention_wrapper`가 내부 함수 `sparse_paged_attention_kernel`을 부릅니다.

- **구성:** CTA(256 thread) 하나가 요청 하나의 query head 하나를 맡습니다. B8이면 query head 32 × 요청 8 = 256 CTA입니다.
- **K 처리:** high 페이지를 돈 다음 low 페이지를 돌면서 다음 순서로 진행합니다.
  ```text
  packed K 읽기 → bit 풀기 → 정수를 float로 → code × scale + zero → q와 내적
  ```
- **softmax:** FP32로 계산합니다.
- **V 처리:** 같은 방식으로 복원한 뒤 확률을 곱해 출력에 누적합니다(PV).
- **결과 저장:** 임시 출력과 **토큰별 확률**을 저장합니다. 이 확률은 reduce가 중요도 점수를 갱신하는 데 씁니다.

Llama-3-8B는 query head 4개가 KV head 하나를 공유합니다(GQA). 현재 구조에서는 그 4개 CTA가 같은 KV를 **각자 읽고 각자 복원**합니다.

[K 읽기와 복원](../../csrc/attention/sparse_attention_kernels.cu:352), [softmax](../../csrc/attention/sparse_attention_kernels.cu:483), [PV](../../csrc/attention/sparse_attention_kernels.cu:611)

### 4.4 문맥 분할(partition)

- **실제 분할 크기:** 5,760토큰입니다. 소스에 적힌 4800을 페이지 크기(32 × 60 = 1920)의 배수로 올린 값입니다. [계산](../../csrc/attention/sparse_attention_kernels.cu:1214)
- **이번 측정에서는 분할이 1개입니다.** 측정 조건의 문맥 길이가 최대 4,096이기 때문입니다. 그래서 FP16 V2처럼 문맥을 나눠 병렬화하지 않습니다.
- **Python의 4800은 다른 값입니다.** 임시 버퍼 크기를 정할 때 쓰는 값이고, 실제 CUDA 분할 크기가 아닙니다.

### 4.5 Reduce 커널

`reduce_kernel`은 CTA 하나가 요청 하나의 KV head 하나를 맡습니다. B8이면 KV head 8 × 요청 8 = 64 CTA입니다. 하는 일은 두 가지입니다.

1. 임시 출력을 최종 출력에 기록합니다.
2. 그 KV head를 공유하는 query head 4개의 토큰별 확률 중 **최댓값을 캐시의 중요도 점수에 더합니다.**

분할이 1개라 합칠 것이 없어도 reduce는 반드시 실행됩니다. 중요도 점수가 다음 step의 압축 판단에 쓰이기 때문입니다. FP16 V2 reduce와는 하는 일이 다릅니다. [단일 분할 reduce](../../csrc/attention/sparse_attention_kernels.cu:867)

## 5. v1이 원본 DiffKV에서 바꾼 것

`no-sync-bulk-lengths`는 **CUDA 커널을 바꾸지 않습니다.** Python 쪽 대기 비용만 줄입니다. [diffkv_v1.py](../../benchmarks/diffkv_v1.py:32)

| 변경 | 의미 |
|---|---|
| Attention forward와 engine step의 `torch.cuda.synchronize()` 제거 | CPU가 매번 GPU 완료를 기다리던 호출을 없앴습니다. 같은 CUDA stream이라 실행 순서는 그대로입니다 |
| Layer별 길이 읽기 32번(`.item()`)을 step당 1번(`.cpu().tolist()`)으로 묶음 | GPU→CPU 복사 횟수가 줄어듭니다. 복사 자체는 여전히 동기식입니다 |

이 패치로 B8/ISL512/OSL2048의 TPOT가 37.2ms에서 30.4ms로 줄었습니다([performance-analysis.md](../../performance-analysis.md)). 이후 최적화를 평가할 때 이 이득을 다시 세면 안 됩니다.

## 6. 이미 시도했지만 채택하지 않은 커널 변경 (2026-10-18)

| 시도 | 내용 | 결과 | 원자료 |
|---|---|---|---|
| Launch 설정 변경 | CTA당 thread 수를 256에서 128로, partition 크기를 5760에서 960/1920으로 줄임 | 128 thread만 바꾸면 34~57% 느려졌습니다. 분할을 줄이는 효과는 문맥 길이에 따라 -10%~+9%로 갈렸습니다. B8/512/512 E2E 차이는 -0.15%로 잡음 수준이어서 기본값을 유지했습니다 | [validation.json](../../logs/performance/v2-implementation-20261018/validation.json) |
| GQA 재사용 (`sparse_paged_attention_gqa`) | KV head 하나를 한 번만 복원해 query head 4개가 공유 | 측정한 6개 조건 모두에서 1.33~2.62배 **느려졌습니다** | [summary.json](../../logs/performance/gqa-reuse-20261018/summary.json) |

GQA 재사용 커널은 코드에 남아 있지만 꺼져 있습니다(`use_gqa_reuse = False`, [호출부](../../vllm/model_executor/layers/sparse_attention_big_kernel.py:421)). 이 구조는 CTA 수가 1/4로 줄어듭니다. 당시에는 ncu 카운터 권한이 없어서 느려진 원인을 확인하지 못했습니다. 원인을 모르므로, 다른 방식으로 구현하면 빨라질 가능성은 남아 있습니다.

## 7. 성능표의 "Attention" 열

[profiling-final-result.md](../../profiling-final-result.md)의 Attention 열은 **전체 생성 동안 prefill과 모든 decode step의 attention 커널 시간 합계**입니다.

- FP16은 Flash와 PagedAttention main/reduce를 포함합니다.
- DiffKV는 Triton `attn_fwd`와 main/reduce를 포함합니다.
- 캐시 쓰기와 압축은 KV Update 열, QKV/output projection은 GEMM 열입니다. [분류 코드](../../benchmarks/analyze_offline_nsys.py:29)

예를 들어 DiffKV-v1 B8/2048/2048의 Attention 18.43초는 다음처럼 나뉩니다.

| 구간 | 시간 | 비중 |
|---|---:|---:|
| Prefill | 0.24 s | 1.3% |
| Decode main | 17.31 s | 93.9% |
| Decode reduce | 0.88 s | 4.8% |

## 8. 파일 지도

경로는 저장소 루트 기준입니다.

| 무엇 | 파일 | 함수와 위치 |
|---|---|---|
| FP16 attention 진입 | [attention.py](../../.cache/baselines/vllm-1db83e3/vllm/model_executor/layers/attention.py:59) | `PagedAttention.forward()` 59행, xFormers 호출 146행, decode `_paged_attention()` 212행, V1/V2 선택 235행 |
| FP16 prefill (xFormers) | [fmha/\_\_init\_\_.py](../../.cache/baselines/fp16-env/lib/python3.11/site-packages/xformers/ops/fmha/__init__.py:362), [dispatch.py](../../.cache/baselines/fp16-env/lib/python3.11/site-packages/xformers/ops/fmha/dispatch.py:126), [flash.py](../../.cache/baselines/fp16-env/lib/python3.11/site-packages/xformers/ops/fmha/flash.py:632) | 진입 → 구현 선택 → `torch.ops.aten._flash_attention_forward()`. Flash CUDA 소스는 설치 패키지에 없습니다 |
| FP16 decode CUDA | [attention_kernels.cu](../../.cache/baselines/vllm-1db83e3/csrc/attention/attention_kernels.cu:85) | 공통 `paged_attention_kernel()` 85행, V1 400행, V2 427행, V2 reduce 455행 |
| DiffKV attention 진입 | [sparse_attention_big_kernel.py](../../vllm/model_executor/layers/sparse_attention_big_kernel.py:111) | `SparsePagedAttention.forward()` 111행, prefill 호출 200행, decode 호출 421행 |
| DiffKV prefill (Triton) | [triton_flash_attention.py](../../vllm/model_executor/layers/triton_flash_attention.py:191) | `attn_fwd()` 191행, 내부 루프 `_attn_fwd_inner()` 63행, Python 진입 `forward()` 582행 |
| Python↔CUDA 연결 | [pybind.cpp](../../csrc/pybind.cpp:25) | `ops.sparse_paged_attention`, `ops.sparse_paged_attention_gqa` |
| DiffKV decode CUDA | [sparse_attention_kernels.cu](../../csrc/attention/sparse_attention_kernels.cu:1407) | C++ 진입 1407행 → launcher 1148행 → main `sparse_paged_attention_kernel()` 125행, reduce `reduce_kernel()` 776행 |
| K/V 읽기 단위 정의 | [cache_utils.h](../../csrc/attention/cache_utils.h:10) | `k_vec_type`(uint16×4), `v_vec_type`(uint16×2) |
| v1 패치 | [diffkv_v1.py](../../benchmarks/diffkv_v1.py:32) | `apply_variant()` |

## 9. 최적화를 검증할 때 주의할 점

- **Prefill과 decode는 따로 검증합니다.** 커널이 완전히 다르기 때문입니다.
- **DiffKV decode는 출력만 맞아서는 부족합니다.** reduce가 누적한 중요도 점수와, 그 점수로 결정되는 다음 step의 토큰 선택까지 같아야 합니다.
- **커널 시간과 E2E 시간은 따로 확인합니다.** 커널이 빨라져도 E2E 이득이 잡음 수준일 수 있습니다(6절의 분할 크기 사례).
- **DiffKV-v1 decode attention이 왜 느린지와 고칠 순서**는 [profiling_result_claude.md](profiling_result_claude.md)에 있습니다.
