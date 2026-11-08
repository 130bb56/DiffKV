// SPDX-License-Identifier: Apache-2.0
// Whole-prompt DiffKV token scores, and scores plus prompt-phase compression
// in one host call.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "cache.h"
#include "checks.h"

namespace diffkv {
namespace {

constexpr int kHeadSize = 128;
constexpr int kQueriesPerKv = 4;
// Last queries of a prompt that score its tokens (NUM_TOKENS_SCORE).
constexpr int kScoreWindow = 64;
constexpr int kKeyTile = 64;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kVector = 8;  // halves per 16-byte load
// Row pitches padded by 16 bytes so consecutive rows start on different
// shared-memory banks.
constexpr int kHalfPitch = kHeadSize + 8;
constexpr int kFloatPitch = kKeyTile + 4;
static_assert(kScoreWindow * kFloatPitch * sizeof(float) <=
              kScoreWindow * kHalfPitch * sizeof(half));

// Copy 64 [128] FP16 rows to shared memory, zero-filling the rows for which
// row_offset returns a negative offset.
template <typename RowOffset>
__device__ void load_rows(half* dst, const half* src, RowOffset row_offset) {
  for (int index = threadIdx.x; index < kKeyTile * kHeadSize / kVector;
       index += kThreads) {
    const int row = index / (kHeadSize / kVector);
    const int column = index % (kHeadSize / kVector) * kVector;
    const int64_t offset = row_offset(row);
    uint4 data = make_uint4(0, 0, 0, 0);
    if (offset >= 0) {
      data = *reinterpret_cast<const uint4*>(src + offset + column);
    }
    *reinterpret_cast<uint4*>(dst + row * kHalfPitch + column) = data;
  }
}

// One block per (64-key tile, KV head, request). For each of the group's four
// query heads, S = Q K^T over the last kScoreWindow queries runs on FP16
// tensor cores with FP32 accumulation; each key sums exp(S * scale - lse)
// over the scored queries that see it, and its score is the maximum over the
// heads. Output: scores[request, KV head, key position].
__global__ void __launch_bounds__(kThreads) prompt_scores_kernel(
    const half* __restrict__ query, const half* __restrict__ key,
    const float* __restrict__ lse, const int* __restrict__ cu_seqlens,
    float* __restrict__ scores, const int64_t q_stride, const int64_t k_stride,
    const int64_t lse_head_stride, const int64_t lse_token_stride,
    const int num_kv_heads, const int max_prompt_len, const float scale) {
  using namespace nvcuda;
  __shared__ __align__(32) half s_keys[kKeyTile * kHalfPitch];
  // Query rows of one head; the logits overwrite them once S is formed.
  __shared__ __align__(32) half s_queries[kScoreWindow * kHalfPitch];
  __shared__ float s_partial[kThreads / kKeyTile][kKeyTile];
  __shared__ float s_lse[kScoreWindow];
  float* s_logits = reinterpret_cast<float*>(s_queries);

  const int key_begin = blockIdx.x * kKeyTile;
  const int kv_head = blockIdx.y;
  const int sequence = blockIdx.z;
  const int start = cu_seqlens[sequence];
  const int length = cu_seqlens[sequence + 1] - start;
  if (key_begin >= length) {
    return;
  }
  const int scored = min(length, kScoreWindow);
  const int first_query = length - scored;

  load_rows(s_keys, key + kv_head * kHeadSize, [&](int row) -> int64_t {
    const int position = key_begin + row;
    return position < length ? (start + position) * k_stride : -1;
  });

  const int warp = threadIdx.x / 32;
  // Each warp forms a 16-query strip against two 16-key tiles.
  const int strip = warp % 4;
  const int first_tile = warp / 4 * 2;
  const int column = threadIdx.x % kKeyTile;
  const int quarter = threadIdx.x / kKeyTile;
  const int key_position = key_begin + column;
  float score = 0.0f;
  for (int head = 0; head < kQueriesPerKv; head++) {
    const int query_head = kv_head * kQueriesPerKv + head;
    __syncthreads();
    load_rows(
        s_queries, query + query_head * kHeadSize, [&](int row) -> int64_t {
          return row < scored ? (start + first_query + row) * q_stride : -1;
        });
    if (threadIdx.x < scored) {
      s_lse[threadIdx.x] =
          lse[query_head * lse_head_stride +
              (start + first_query + threadIdx.x) * lse_token_stride];
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> logits[2];
    wmma::fill_fragment(logits[0], 0.0f);
    wmma::fill_fragment(logits[1], 0.0f);
    for (int depth = 0; depth < kHeadSize; depth += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
      wmma::load_matrix_sync(a, s_queries + strip * 16 * kHalfPitch + depth,
                             kHalfPitch);
      for (int tile = 0; tile < 2; tile++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
        wmma::load_matrix_sync(
            b, s_keys + (first_tile + tile) * 16 * kHalfPitch + depth,
            kHalfPitch);
        wmma::mma_sync(logits[tile], a, b, logits[tile]);
      }
    }
    __syncthreads();
    for (int tile = 0; tile < 2; tile++) {
      wmma::store_matrix_sync(
          s_logits + strip * 16 * kFloatPitch + (first_tile + tile) * 16,
          logits[tile], kFloatPitch, wmma::mem_row_major);
    }
    __syncthreads();

    // Four threads per key column, each summing a quarter of the queries.
    float sum = 0.0f;
    if (key_position < length) {
      const int rows = kScoreWindow / (kThreads / kKeyTile);
      for (int row = quarter * rows; row < (quarter + 1) * rows && row < scored;
           row++) {
        if (key_position <= first_query + row) {
          sum +=
              __expf(s_logits[row * kFloatPitch + column] * scale - s_lse[row]);
        }
      }
    }
    s_partial[quarter][column] = sum;
    __syncthreads();
    if (quarter == 0) {
      for (int other = 1; other < kThreads / kKeyTile; other++) {
        sum += s_partial[other][column];
      }
      score = fmaxf(score, sum);
    }
  }
  if (quarter == 0 && key_position < length) {
    scores[(static_cast<int64_t>(sequence) * num_kv_heads + kv_head) *
               max_prompt_len +
           key_position] = score;
  }
}

}  // namespace
}  // namespace diffkv

torch::Tensor prompt_scores(torch::Tensor& query, torch::Tensor& key,
                            torch::Tensor& lse, torch::Tensor& cu_seqlens,
                            const int64_t max_prompt_len, const double scale) {
  const c10::cuda::CUDAGuard device_guard(query.device());
  for (const auto& [tensor, name] :
       {std::pair{&query, "query"}, std::pair{&key, "key"}}) {
    TORCH_CHECK(tensor->is_cuda() && tensor->device() == query.device() &&
                    tensor->scalar_type() == at::kHalf,
                name, " must be FP16 on the query device");
    TORCH_CHECK(tensor->dim() == 3 && tensor->size(2) == 128 &&
                    tensor->stride(2) == 1 && tensor->stride(1) == 128 &&
                    tensor->stride(0) % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(tensor->data_ptr()) % 16 == 0,
                name,
                " must be 16-byte aligned [tokens, heads, 128] rows with a "
                "token stride divisible by 8");
  }
  const int num_kv_heads = key.size(1);
  TORCH_CHECK(query.size(1) == num_kv_heads * 4 && query.size(0) == key.size(0),
              "query must have four heads per KV head and the keys' tokens");
  diffkv::check_tensor(cu_seqlens, query, at::kInt, "sequence starts");
  TORCH_CHECK(lse.is_cuda() && lse.device() == query.device() &&
                  lse.scalar_type() == at::kFloat && lse.dim() == 2 &&
                  lse.size(0) == query.size(1) && lse.size(1) == query.size(0),
              "lse must be FP32 [query heads, tokens]");
  const int num_sequences = cu_seqlens.numel() - 1;
  TORCH_CHECK(num_sequences > 0 && max_prompt_len > 0,
              "Prompt scores need at least one non-empty prompt");
  auto scores = torch::empty({num_sequences, num_kv_heads, max_prompt_len},
                             query.options().dtype(at::kFloat));
  const dim3 grid((max_prompt_len + diffkv::kKeyTile - 1) / diffkv::kKeyTile,
                  num_kv_heads, num_sequences);
  diffkv::prompt_scores_kernel<<<grid, diffkv::kThreads, 0,
                                 at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(query.data_ptr()),
      reinterpret_cast<const half*>(key.data_ptr()), lse.data_ptr<float>(),
      cu_seqlens.data_ptr<int>(), scores.data_ptr<float>(), query.stride(0),
      key.stride(0), lse.stride(0), lse.stride(1), num_kv_heads,
      static_cast<int>(max_prompt_len), static_cast<float>(scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return scores;
}

torch::Tensor score_and_compress_prompts(
    torch::Tensor& query, torch::Tensor& key, torch::Tensor& value,
    torch::Tensor& lse, torch::Tensor& cu_seqlens, const int64_t max_prompt_len,
    const double scale, torch::Tensor& kv_cache, torch::Tensor& slot_ids,
    torch::Tensor& block_tables, torch::Tensor& kv_len_tables,
    torch::Tensor& compress_config_tables, const int kv_buffer_size,
    const int layer_idx, const int num_bits_k_high, const int num_bits_v_high,
    const int num_bits_k_low, const int num_bits_v_low,
    const int num_chunks_k_high, const int num_chunks_v_high,
    const int num_chunks_k_low, const int num_chunks_v_low,
    const int k_vec_size, const int v_vec_size,
    const int num_tokens_per_page_high, const int num_tokens_per_page_low) {
  torch::Tensor scores =
      prompt_scores(query, key, lse, cu_seqlens, max_prompt_len, scale);
  compress_and_append_cache_prompt_phase(
      key, value, scores, kv_cache, slot_ids, block_tables, kv_len_tables,
      cu_seqlens, compress_config_tables, kv_buffer_size, layer_idx,
      num_bits_k_high, num_bits_v_high, num_bits_k_low, num_bits_v_low,
      num_chunks_k_high, num_chunks_v_high, num_chunks_k_low, num_chunks_v_low,
      k_vec_size, v_vec_size, num_tokens_per_page_high,
      num_tokens_per_page_low);
  return scores;
}
