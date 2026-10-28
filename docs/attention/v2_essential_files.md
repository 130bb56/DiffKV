# DiffKV v2 decode attention: 필수 파일 5개의 AS-IS / TO-BE

commit `40d7f17` (2026-10-25, 부모 `bdefda1`). v2는 decode attention 커널만 바꾼다. prefill(Triton flash attention + score), cache writer, 메모리 관리, 압축 정책은 v1 그대로다.

## 필수 파일이 이 5개인 근거

`bdefda1`을 새 worktree에 받고 아래 5개 파일만 복사해 extension을 빌드한 뒤(`_C.so` 68c58e12…), 설치된 v2 binary(b28c5e92…, 작업 트리 전체로 빌드)와 비교했다.

- 커널 단위: `benchmarks/validate_diffkv_kernel_candidate.py`로 두 binary의 `sparse_paged_attention_gqa`를 context 512/1536/2560/6000, batch 8, KV head 8, 32 step, 정책 3개에서 capture하고 비교. 출력, 누적 score, cache byte 모두 bitwise 동일. (`require_bitwise` 통과)
- 모델 단위: R1-Distill-Llama-8B, MATH-500 50문제, greedy, EOS 종료. 설치된 v2 실행과 50/50 요청의 출력 token 열이 동일, 생성 token 167,076개 동일.

따라서 작업 트리에 남아 있던 다른 변경(`sparse_attention_kernels.cu` 9줄, 테스트, 벤치마크 도구, 문서)은 v2 동작에 필요하지 않다. `sparse_attention_kernels.cu`의 9줄은 legacy 경로의 정렬 `TORCH_CHECK`와, v2가 더 이상 호출하지 않는 `sparse_paged_attention_gqa_reduce`의 partition 상수 변경이다.

## 측정된 효과 (참고)

- 모델: R1 MATH 50문제, 동일 작업량(OSL 고정), 1395 MHz. v1 → v2 처리량 435 → 555 token/s(1.28배), 생성 token당 decode 시간 2.28 → 1.79 ms. batch 구성과 KV 사용률은 동일. (`evaluation-logs/r1-math50-20261023/throughput-fixed-20261024/report.md`)
- 커널: 작성자 측정으로 B8/KV8 단일 layer decode attention 전체(main + reduce + score 갱신)가 v1 binary 대비 context 2048~8192에서 2.36~2.58배, context 512에서 1.74배. NCU L1 global-load sector 11.6M → 2.8M. (`docs/attention/full_decode_optimization.md`)
- 정확도: 500문제 greedy에서 FP16 80.2%, v2 77.2%. v1과의 차이는 50문제 greedy에서 41 대 38로, logit margin 분석상 커널 왜곡이 아니라 근소한 token flip. (`evaluation-logs/r1-math500-greedy-20261024/report.md`, `greedy-20261024/margins.json`)

---

## 1. `csrc/attention/sparse_gqa_register_kernels.cuh` (새 파일, 385줄)

**AS-IS (v1에서 실제로 돌던 decode 커널, `sparse_attention_kernels.cu`의 `sparse_paged_attention`)**
- CTA 하나가 query head 하나를 맡는다. GQA 4이므로 같은 KV head의 packed K/V를 CTA 4개가 각자 global memory에서 읽고 각자 dequant한다. L1/L2가 DRAM 중복은 걸러 주지만 load 명령과 dequant 명령은 4번 실행된다.
- 256 threads, partition 4800(reduce 쪽은 5760). dequant한 K/V를 FP16으로 반올림해 shared memory에 올렸다가 다시 읽는다.
- `k_vec_type`이 2B 정렬이라 K payload를 `LDG.E.U16` 2B 단위로 읽는다(프로파일링에서 확인). 메모리 대역폭 사용률 18%, PC 샘플의 37%가 load 대기.
- (참고) `bdefda1`에는 GQA 공유 커널의 첫 시도(`sparse_gqa_attention_kernel`, 437줄, 256 threads, tile 32 token, partition 5760, shared-memory staging + swizzle)도 있었지만 모든 조건에서 더 느려 `use_gqa_reuse=False`로 꺼 둔 상태였다.

**TO-BE (`register_gqa_attention_kernel`)**
- CTA 하나가 KV head 하나와 그 query head 4개를 함께 맡는다. grid = (KV head 8, batch, partition 수). dequant한 K/V를 FP32 register에 두고 4개 query에 재사용한다. FP16 중간 반올림이 없다.
- 128 threads, `__launch_bounds__(128, 4)`, partition 240. thread당 register 127개, static shared 9,280 B, spill 없음.
- K: warp 안에서 8개 dimension thread가 16 dim씩 맡고 token group 4개를 병렬로 처리한다(`compute_keys<PageTokens, KeyBits>`가 high page 32 token/8-bit와 low page 60 token/4-bit를 템플릿으로 나눔). 4개 query의 QK는 shuffle을 교차 배치해 의존성을 줄였다.
- softmax: query 4개의 max와 sum을 warp shuffle로 함께 처리해 barrier를 줄였다. 확률은 shared `probabilities[4][240]`에 둔다.
- PV: token 두 개의 packed V payload를 한 번 load하고, token별 metadata와 4개 query weight를 한 번만 읽은 뒤 dimension pack을 순회한다(token-pair hoisting). partial output은 XOR 변환한 column(`output_column`)으로 저장해 bank 충돌을 없앴다. `SharedWorkspace`는 확률 버퍼와 output partial을 union으로 공유한다.
- partition이 physical page 중간에서 시작할 수 있어 token 유효성 검사로 처리한다. ALiBi는 선택.

## 2. `csrc/attention/sparse_gqa_reduce_kernels.cuh` (새 파일, 249줄)

**AS-IS (`sparse_paged_attention_gqa_reduce` / legacy reduce)**
- grid = (KV head, batch) = B8/KV8에서 CTA 64개. partition별 exp_sum과 max_logit을 합쳐 output을 만들고, `tmp_scores`를 전역으로 rescale해 다시 읽어 query head 4개의 최댓값을 cache의 importance score에 더한다. 프로파일링에서 CTA 수가 SM 82개보다 적고 샘플의 66%가 메모리 대기였다.

**TO-BE (`register_gqa_reduce_kernel`)**
- grid = (KV head, batch, score split 4), block 256, dynamic shared = 4 × partition 수 × 4 B. B8/KV8에서 CTA 256개. 4개 CTA가 한 (sequence, KV head)의 token 범위를 나눠 맡고, 작은 partition 통계는 각자 다시 계산한다.
- output은 tile 0만 쓴다. token마다 owner thread 하나가 4개 query의 확률을 register에서 rescale하고 그 최댓값을 cache의 score에 더한다. `tmp_scores`를 전역으로 rescale해 쓰거나 다시 읽지 않는다(이 버퍼는 scratch로만 남고 다음 main이 덮어쓴다).
- partition이 32개를 넘으면 strided 경로. `(sum + 1e-6f) * factor`로 FP64 명령을 없앴다(SASS에 FP64 0개).
- 작성자 측정: reduce + score 갱신 29.9 → 10.5 us (B8/C3072).

## 3. `csrc/attention/sparse_gqa_attention_kernels.cu` (변경, -351/+25)

**AS-IS**: 437줄. 위에서 말한 첫 GQA 시도 커널 본체(`sparse_gqa_attention_kernel`, 상수 kThreads 256, kTileTokens 32, kPartitionSize 5760, shared staging용 `value_column` swizzle과 `store_value_quad`)와 launcher. reduce는 `sparse_attention_kernels.cu`의 `sparse_paged_attention_gqa_reduce`를 호출.

**TO-BE**: 161줄. 커널 본체를 제거하고 launcher만 남긴다.
```cpp
namespace vllm { namespace gqa_reuse {
constexpr int kHeadSize = 128;
constexpr int kQueriesPerKv = 4;
constexpr int kThreads = 128;
// The kernel masks partial pages, so partitions can split a physical page.
constexpr int kPartitionSize = 240;
#include "sparse_gqa_register_kernels.cuh"
#include "sparse_gqa_reduce_kernels.cuh"
void launch_reduce(...)   // grid (num_kv_heads, batch, 4), block 256, dynamic shared 4*partitions*sizeof(float)
template<typename scalar_t> void launch(...)  // grid (num_kv_heads, batch, max_partitions), block kThreads
}}
void sparse_paged_attention_gqa(...) {
  TORCH_CHECK(K8V4/K4V2, chunk 1, vec 4/2, page 32/60, head 128, 4 query/KV, query stride, contiguous,
              8B-aligned cache base and page stride, nonempty batch, workspace sizes >= partitions)
  launch<uint16_t | __nv_bfloat16>(...);
  launch_reduce(...);
}
```
op 이름과 pybind 등록(`sparse_paged_attention_gqa`)은 `bdefda1` 그대로이고, 본체만 바뀐다. FP16과 BF16을 지원한다.

## 4. `csrc/attention/cache_utils.h` (변경, 8줄)

```diff
-struct k_vec_type {
+// Each key vector occupies one naturally aligned 64-bit cache payload load.
+struct alignas(K_VEC_SIZE * sizeof(uint16_t)) k_vec_type {
   uint16_t data[K_VEC_SIZE];
 };
-struct v_vec_type {
+// Two adjacent token payloads share one naturally aligned 32-bit load.
+struct alignas(V_VEC_SIZE * sizeof(uint16_t)) v_vec_type {
   uint16_t data[V_VEC_SIZE];
 };
```

**AS-IS**: 두 struct의 정렬이 uint16 기준 2 B. 컴파일러가 8 B 정렬을 보장받지 못해 K payload 하나를 2 B load 4번으로 읽었다(SASS `LDG.E.U16`). ncu source counter에서 K 읽기의 sector 효율 25%.

**TO-BE**: `k_vec_type` 8 B 정렬, `v_vec_type` 4 B 정렬. 크기와 pack 순서는 그대로(8 B, 4 B). page 크기 6,784 B와 page 내 K/V offset이 이 정렬을 만족하므로 cache 형식 변경은 없다. legacy 커널과 v2 커널 모두 이 타입으로 load한다.

## 5. `vllm/model_executor/layers/sparse_attention_big_kernel.py` (변경, 33줄)

```diff
 _PARTITION_SIZE = 4800
+# Must match the specialized GQA main and reduction kernels.
+_GQA_PARTITION_SIZE = 240
...
-        # Selected before warmup by the GQA-reuse benchmark variant.
-        self.use_gqa_reuse = False
+        # Allow the GQA kernel only for supported decode configurations.
+        self.use_gqa_reuse = True
...
             max_context_len += 1
+            use_gqa_reuse = (
+                self.use_gqa_reuse
+                and query.dtype == torch.float16
+                and self.head_size == 128
+                and self.num_queries_per_kv == 4
+                and (num_bits_k_high, num_bits_v_high, num_bits_k_low, num_bits_v_low,
+                     num_chunks_k_high, num_chunks_v_high, num_chunks_k_low, num_chunks_v_low) == (8, 4, 4, 2, 1, 1, 1, 1)
+                and (key_vec_size, val_vec_size) == (4, 2)
+                and (num_tokens_per_block_high, num_tokens_per_block_low) == (32, 60)
+                and query.stride(1) == 128 and query.stride(2) == 1
+            )
+            partition_size = _GQA_PARTITION_SIZE if use_gqa_reuse else _PARTITION_SIZE
...
-            if max_context_len <= _PARTITION_SIZE:
+            if max_context_len <= partition_size:
                 num_partitions = 1
             else:
-                num_partitions = ((max_context_len + _PARTITION_SIZE - 1) // _PARTITION_SIZE) + 1
+                num_partitions = ((max_context_len + partition_size - 1) // partition_size) + 1
...
-            attention_op = (ops.sparse_paged_attention_gqa if self.use_gqa_reuse
+            attention_op = (ops.sparse_paged_attention_gqa if use_gqa_reuse
                             else ops.sparse_paged_attention)
```

**AS-IS**: `use_gqa_reuse=False`가 고정값이라 decode는 항상 legacy `sparse_paged_attention`으로 갔고, workspace(`tmp_output`, `exp_sums`, `max_logits`)는 partition 4800 기준으로 잡았다.

**TO-BE**: forward마다 지원 조건(FP16, head 128, GQA 4, K8V4/K4V2 chunk 1, vec 4/2, page 32/60, query stride)을 검사해 맞으면 `sparse_paged_attention_gqa`를, 아니면 legacy를 부른다. workspace의 partition 수는 선택한 커널에 맞춰 240 또는 4800 기준으로 계산한다. `use_gqa_reuse=False`로 두면 legacy를 강제할 수 있다. v1의 Python 최적화(no-sync-bulk-lengths 변형)는 이 파일 위에 그대로 적용된다.

---

## 함께 바뀌지 않은 것

- prefill: `triton_flash_attention.py`, `triton_fused_softmax_sum.py`. prompt 길이가 64의 배수일 때의 score race는 그대로 남아 있다.
- cache writer(`compress_and_append_cache_*`), 메모리 관리(`mem_mgt_kernels.cu`), block manager, scheduler, 압축 정책(threshold 0.02/1.0, KV buffer 64).
- op 등록(`csrc/pybind.cpp`, `csrc/ops.h`).
