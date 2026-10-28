#include <torch/extension.h>
#include "checks.h"
#include <ATen/cuda/CUDAContext.h>

#include "cuda_compat.h"
#include "reduction_utils.cuh"
#include "./quantization/kv_cache_quant/kv_cache_quant.h"
#include "./attention/cache_utils.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#define DEBUG false
#define WARP_SIZE 32
#define NUM_THREADS 128
#define NUM_THREADS_PROMPT 512
#define NUM_THREADS_DECODE 128

// Use the qk products computed between the last few tokens and all tokens
// in the prompt to compute the compression metric.
// # Note: this parameter should be consistent with the one in
// triton_flash_attention.py
#define NUM_TOKENS_SCORE 64

namespace diffkv_v2 {
template <typename scalar_t, int HEAD_SIZE, int BITS_K_HIGH, int BITS_V_HIGH,
          int BITS_K_LOW, int BITS_V_LOW, int CHUNKS_K_HIGH, int CHUNKS_V_HIGH,
          int CHUNKS_K_LOW, int CHUNKS_V_LOW, int NUM_TOKENS_PER_PAGE_HIGH,
          int NUM_TOKENS_PER_PAGE_LOW,
          int THREAD_GROUP_SIZE_V>
__global__ void compress_and_append_cache_prompt_phase_kernel(
    const scalar_t* __restrict__ key,    // [total_num_tokens, num_kv_heads,
                                         // head_size]
    const scalar_t* __restrict__ value,  // [total_num_tokens, num_kv_heads,
                                         // head_size]
    const float* __restrict__ score,   // [num_seqs, num_heads, max_prompt_len]
    uint16_t* __restrict__ kv_cache,   // [num_blocks, unified_page_size]
    const int* __restrict__ slot_ids,  // [num_seqs]
    const int* __restrict__ block_tables,  // [num_slots, num_layers,
                                           // num_kv_heads,
                                           // max_num_blocks_per_seq]
    int* __restrict__ kv_len_tables,  // [num_slots, num_layers, num_kv_heads,
                                      // 2]
    const int* __restrict__ seq_start_loc,             // [num_seqs + 1]
    const float* __restrict__ compress_config_tables,  // [num_slots, 2]
    const int max_prompt_len, const int kv_buffer_size, const int layer_idx,
    const int kv_stride, const int num_layers, const int num_heads,
    const int num_kv_heads, const int unified_page_size,
    const int max_num_blocks_per_seq) {
  const int seq_idx = blockIdx.y;
  const int slot_idx = slot_ids[seq_idx];
  const int kv_head_idx = blockIdx.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int thread_idx = threadIdx.x;

  const int prompt_start = seq_start_loc[seq_idx];
  const int prompt_end = seq_start_loc[seq_idx + 1];
  const int prompt_len = prompt_end - prompt_start;
  const int prompt_len_power2 = POWER2_ROUND_UP(prompt_len);
  const int max_prompt_len_power2 = POWER2_ROUND_UP(max_prompt_len);

  extern __shared__ char shared_memory[];
  // s_scores and s_keep are of size max_prompt_len (no padding)
  // s_mean_scores and s_positions are of size max_prompt_len_power2 (with
  // padding) Padding is due to the requirement of bitonic sort.
  // TODO(yuwei): can we get rid of padding?
  float* s_mean_scores = reinterpret_cast<float*>(shared_memory);

  size_t s_positions_offset = max_prompt_len_power2 * sizeof(float);
  int* s_positions = reinterpret_cast<int*>(shared_memory + s_positions_offset);

  size_t s_scores_offset =
      s_positions_offset + max_prompt_len_power2 * sizeof(int);
  float* s_scores = reinterpret_cast<float*>(shared_memory + s_scores_offset);

  // s_keep tracks whether tokens are kept at high precision or quantized to low
  // precision or pruned. 1: kept at high precision, 2: quantized, 3: pruned.
  // Using char type is as memory efficient as using bool type, since each bool
  // is one byte.
  // TODO(yuwei): We can actually get rid of s_keep for performance
  // optimization; keep it now for better code readability. size_t s_keep_offset
  // = s_scores_offset + max_prompt_len * sizeof(float); char* s_keep =
  // reinterpret_cast<char*>(shared_memory + s_keep_offset);

  const float range_k_high = get_quant_range(BITS_K_HIGH);
  const float range_v_high = get_quant_range(BITS_V_HIGH);
  const float range_k_low = get_quant_range(BITS_K_LOW);
  const float range_v_low = get_quant_range(BITS_V_LOW);

  constexpr int K_PACK_SIZE_HIGH = 16 / BITS_K_HIGH;
  constexpr int V_PACK_SIZE_HIGH = 16 / BITS_V_HIGH;
  constexpr int K_PACK_SIZE_LOW = 16 / BITS_K_LOW;
  constexpr int V_PACK_SIZE_LOW = 16 / BITS_V_LOW;

  constexpr int NUM_K_PACKS_HIGH = HEAD_SIZE / K_PACK_SIZE_HIGH;
  constexpr int NUM_V_PACKS_HIGH = HEAD_SIZE / V_PACK_SIZE_HIGH;
  constexpr int NUM_K_PACKS_LOW = HEAD_SIZE / K_PACK_SIZE_LOW;
  constexpr int NUM_V_PACKS_LOW = HEAD_SIZE / V_PACK_SIZE_LOW;

  constexpr int NUM_K_VECS_HIGH = NUM_K_PACKS_HIGH / K_VEC_SIZE;
  constexpr int NUM_K_VECS_LOW = NUM_K_PACKS_LOW / K_VEC_SIZE;
  constexpr int NUM_VECS_PER_THREAD_K_HIGH =
      NUM_K_VECS_HIGH / THREAD_GROUP_SIZE_K;
  constexpr int NUM_VECS_PER_THREAD_K_LOW =
      NUM_K_VECS_LOW / THREAD_GROUP_SIZE_K;

  constexpr int CHUNK_SIZE_K_HIGH = HEAD_SIZE / CHUNKS_K_HIGH;
  constexpr int CHUNK_SIZE_V_HIGH = HEAD_SIZE / CHUNKS_V_HIGH;
  constexpr int CHUNK_SIZE_K_LOW = HEAD_SIZE / CHUNKS_K_LOW;
  constexpr int CHUNK_SIZE_V_LOW = HEAD_SIZE / CHUNKS_V_LOW;

  constexpr int NUM_K_PACKS_HIGH_PER_GROUP = NUM_K_PACKS_HIGH / CHUNKS_K_HIGH;
  constexpr int NUM_V_PACKS_HIGH_PER_GROUP = NUM_V_PACKS_HIGH / CHUNKS_V_HIGH;
  constexpr int NUM_K_PACKS_LOW_PER_GROUP = NUM_K_PACKS_LOW / CHUNKS_K_LOW;
  constexpr int NUM_V_PACKS_LOW_PER_GROUP = NUM_V_PACKS_LOW / CHUNKS_V_LOW;

  // for mod operations
  constexpr int LOG2_NUM_VECS_PER_THREAD_K_HIGH =
      log2_of_pow2(NUM_VECS_PER_THREAD_K_HIGH);
  constexpr int LOG2_NUM_VECS_PER_THREAD_K_LOW =
      log2_of_pow2(NUM_VECS_PER_THREAD_K_LOW);
  static_assert(LOG2_NUM_VECS_PER_THREAD_K_HIGH > 0);
  static_assert(LOG2_NUM_VECS_PER_THREAD_K_LOW > 0);

  static_assert(NUM_K_PACKS_HIGH % K_VEC_SIZE == 0);
  static_assert(NUM_K_PACKS_LOW % K_VEC_SIZE == 0);
  static_assert(NUM_K_VECS_HIGH % THREAD_GROUP_SIZE_K == 0);
  static_assert(NUM_K_VECS_LOW % THREAD_GROUP_SIZE_K == 0);

  constexpr int NUM_PACKS_PER_THREAD_V_HIGH =
      NUM_V_PACKS_HIGH / THREAD_GROUP_SIZE_V;
  constexpr int NUM_PACKS_PER_THREAD_V_LOW =
      NUM_V_PACKS_LOW / THREAD_GROUP_SIZE_V;

  // for mod operations
  constexpr int LOG2_NUM_PACKS_PER_THREAD_V_HIGH =
      log2_of_pow2(NUM_PACKS_PER_THREAD_V_HIGH);
  constexpr int LOG2_NUM_PACKS_PER_THREAD_V_LOW =
      log2_of_pow2(NUM_PACKS_PER_THREAD_V_LOW);
  static_assert(LOG2_NUM_PACKS_PER_THREAD_V_HIGH >= 0);
  static_assert(LOG2_NUM_PACKS_PER_THREAD_V_LOW >= 0);

  static_assert(NUM_V_PACKS_HIGH % THREAD_GROUP_SIZE_V == 0);
  static_assert(NUM_V_PACKS_LOW % THREAD_GROUP_SIZE_V == 0);

  // Align the starting address of each segment (key, key meta, val, val meta,
  // score, pos) to 32 bytes.
  constexpr int KEY_BASE_HIGH = 0;
  constexpr int KEY_META_BASE_HIGH =
      DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_HIGH * NUM_K_PACKS_HIGH * 2, 32) *
          32 / sizeof(uint16_t) +
      KEY_BASE_HIGH;
  constexpr int VAL_BASE_HIGH =
      DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_HIGH * 4 * CHUNKS_K_HIGH +
                          KEY_META_BASE_HIGH * sizeof(uint16_t),
                      128) *
      128 / sizeof(uint16_t);
  constexpr int PADDED_NUM_TOKENS_PER_PAGE_HIGH =
      DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_HIGH, V_VEC_SIZE) * V_VEC_SIZE;
  constexpr int VAL_META_BASE_HIGH =
      DIVIDE_ROUND_UP(PADDED_NUM_TOKENS_PER_PAGE_HIGH * NUM_V_PACKS_HIGH * 2,
                      32) *
          32 / sizeof(uint16_t) +
      VAL_BASE_HIGH;
  constexpr int SCORE_BASE_HIGH =
      DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_HIGH * 4 * CHUNKS_V_HIGH, 32) * 32 /
          sizeof(uint16_t) +
      VAL_META_BASE_HIGH;
  constexpr int POSITION_BASE_HIGH =
      NUM_TOKENS_PER_PAGE_HIGH * 4 / sizeof(uint16_t) + SCORE_BASE_HIGH;

  constexpr int KEY_BASE_LOW = 0;
  constexpr int KEY_META_BASE_LOW =
      DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_LOW * NUM_K_PACKS_LOW * 2, 32) * 32 /
          sizeof(uint16_t) +
      KEY_BASE_LOW;
  constexpr int VAL_BASE_LOW =
      DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_LOW * 4 * CHUNKS_K_LOW +
                          KEY_META_BASE_LOW * sizeof(uint16_t),
                      128) *
      128 / sizeof(uint16_t);
  constexpr int PADDED_NUM_TOKENS_PER_PAGE_LOW =
      DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_LOW, V_VEC_SIZE) * V_VEC_SIZE;
  constexpr int VAL_META_BASE_LOW =
      DIVIDE_ROUND_UP(PADDED_NUM_TOKENS_PER_PAGE_LOW * NUM_V_PACKS_LOW * 2,
                      32) *
          32 / sizeof(uint16_t) +
      VAL_BASE_LOW;
  constexpr int SCORE_BASE_LOW =
      DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_LOW * 4 * CHUNKS_V_LOW, 32) * 32 /
          sizeof(uint16_t) +
      VAL_META_BASE_LOW;
  constexpr int POSITION_BASE_LOW =
      NUM_TOKENS_PER_PAGE_LOW * 4 / sizeof(uint16_t) + SCORE_BASE_LOW;

  // make sure that each key vec only corresponds to one group of metadata
  static_assert(NUM_K_PACKS_HIGH % (K_VEC_SIZE * CHUNKS_K_HIGH) == 0);
  static_assert(NUM_K_PACKS_LOW % (K_VEC_SIZE * CHUNKS_K_LOW) == 0);

  // Compute actual threshold
  const float base_threshold = 1.0f / prompt_len;
  const float prune_threshold =
      base_threshold * compress_config_tables[slot_idx * 2 + 0];
  const float quant_threshold =
      base_threshold * compress_config_tables[slot_idx * 2 + 1];

  assert(prune_threshold >= 0 && prune_threshold <= 1);
  assert(quant_threshold >= 0 && quant_threshold <= 1);
  assert(quant_threshold >= prune_threshold);

  // Iterate over prompt_len dim
  // float score_sum = 0.0f;
  int count_quantized = 0;
  int count_pruned = 0;

  for (int token_idx = thread_idx; token_idx < prompt_len;
       token_idx += NUM_THREADS_PROMPT) {
    // Initialize s_keep
    // s_keep[token_idx] = 1;
    // Calculate the score
    float score_ = 0.0f;
    for (int j = 0; j < num_queries_per_kv; j++) {
      score_ =
          MAX(score_,
              score[seq_idx * num_heads * max_prompt_len +
                    (kv_head_idx * num_queries_per_kv + j) * max_prompt_len +
                    token_idx]);
    }
    s_scores[token_idx] = score_;
    // Calculate the mean score
    int ideal_num_queries = prompt_len - token_idx;
    int real_num_queries = MIN(ideal_num_queries, NUM_TOKENS_SCORE);
    float mean_score = score_ / real_num_queries;
    // score_sum += mean_score;

    if (ideal_num_queries > kv_buffer_size) {
      // Prune the tokens whose score is smaller than prune_threshold
      // Quantize the tokens score is smaller than quant_threshold otherwise
      if (mean_score < prune_threshold) {
        count_pruned += 1;
      } else if (mean_score < quant_threshold) {
        count_quantized += 1;
      }
    } else {
      // For the last kv_buffer_size tokens, we set the mean score to 1.0 (the
      // largest possible value after normalization) Those tokens are guaranteed
      // not to be evicted.
      mean_score = 1.0f;
    }

    // Update s_mean_scores, s_positions
    s_mean_scores[token_idx] = mean_score;
    s_positions[token_idx] = token_idx;
  }
  __syncthreads();

  // __shared__ int count_pruned;
  // __shared__ int count_quantized;

  count_pruned = blockReduceSum<int>(count_pruned);
  __syncthreads();
  count_quantized = blockReduceSum<int>(count_quantized);

  assert(count_pruned + count_quantized <= prompt_len);

  if (prompt_len <= kv_buffer_size) {
    // Keep all tokens at high precision
    if (thread_idx == 0) {
      count_pruned = 0;
      count_quantized = 0;
    }
    __syncthreads();
  } else {
    // // Normalize the scores
    // score_sum = blockReduceSum<float>(score_sum);
    // for (int token_idx = thread_idx; token_idx < prompt_len; token_idx +=
    // NUM_THREADS_PROMPT) {
    //   if (token_idx < prompt_len - kv_buffer_size) {
    //     s_mean_scores[token_idx] /= score_sum;
    //   } else {
    //     // For the last kv_buffer_size tokens, we set the mean score to 1.0
    //     (the largest possible value after normalization)
    //     // Those tokens are guaranteed not to be evicted.
    //     s_mean_scores[token_idx] = 1.0f;
    //   }
    // }
    __syncthreads();

    // Sort the mean scores in ascending order
    bitonic_sort_ascend(s_mean_scores, s_positions, prompt_len);
    __syncthreads();

    if (DEBUG && layer_idx == 0 && slot_idx == 0 && kv_head_idx == 0 &&
        thread_idx == 0) {
      printf(
          "\n[Debug info from compress_and_append_cache_prompt_phase_kernel], "
          "seq_idx: %d, slot_idx: %d, layer_idx: %d\n",
          seq_idx, slot_idx, layer_idx);
      printf("Sorted mean scores:\n");
      for (int token_idx = 0; token_idx < prompt_len_power2; token_idx++) {
        printf("%f ", s_mean_scores[token_idx]);
      }
      printf("\n");
      printf("Sorted positions:\n");
      for (int token_idx = 0; token_idx < prompt_len_power2; token_idx++) {
        printf("%d ", s_positions[token_idx]);
      }
      printf("\n");
      // printf("s_keep:\n");
      // for (int token_idx = 0; token_idx < prompt_len; token_idx++) {
      //   printf("%d ", s_keep[token_idx]);
      // }
      // printf("\n");
      printf("count_pruned = %d\n", count_pruned);
      printf("count_quantized = %d\n", count_quantized);
    }
  }

  // Update kv_len_tables
  if (thread_idx == 0) {
    const int kv_len_tables_offset = slot_idx * num_layers * num_kv_heads * 2 +
                                     layer_idx * num_kv_heads * 2 +
                                     kv_head_idx * 2;
    kv_len_tables[kv_len_tables_offset] =
        prompt_len - count_pruned - count_quantized;
    kv_len_tables[kv_len_tables_offset + 1] = count_quantized;
  }
  __syncthreads();

  // Quantize kv and write back to cache
  const int* head_block_table_left =
      block_tables + (slot_idx * num_layers * num_kv_heads +
                      layer_idx * num_kv_heads + kv_head_idx) *
                         max_num_blocks_per_seq;
  // const int* head_block_table_right = head_block_table_left +
  // max_num_blocks_per_seq - 1;
  const int* head_block_table_right =
      head_block_table_left +
      DIVIDE_ROUND_UP(prompt_len - count_pruned - count_quantized,
                      NUM_TOKENS_PER_PAGE_HIGH);

  // TODO: what if DIVIDE_ROUND_UP(prompt_len - count_pruned - count_quantized,
  // NUM_TOKENS_PER_PAGE_HIGH) = 0

  // Handle low-precision pages
  for (int token_idx = thread_idx + count_pruned;
       token_idx < count_pruned + count_quantized;
       token_idx += NUM_THREADS_PROMPT) {
    // Access block table from the end of high precision blocks
    int64_t dst_physical_block_number = static_cast<int64_t>(
        *(head_block_table_right +
          (token_idx - count_pruned) / NUM_TOKENS_PER_PAGE_LOW));
    int token_idx_within_the_page =
        (token_idx - count_pruned) % NUM_TOKENS_PER_PAGE_LOW;
    const int64_t src_kv_idx_base =
        (prompt_start + s_positions[token_idx]) * kv_stride +
        kv_head_idx * HEAD_SIZE;

    // Loop over key groups
    const int64_t k_offset =
        dst_physical_block_number * unified_page_size + KEY_BASE_LOW;
    const int64_t k_meta_offset =
        dst_physical_block_number * unified_page_size + KEY_META_BASE_LOW;
    for (int key_chunk_idx = 0; key_chunk_idx < CHUNKS_K_LOW; key_chunk_idx++) {
      // Calculate the start and end indices of the key group
      int key_group_start = key_chunk_idx * CHUNK_SIZE_K_LOW;
      int key_group_end = (key_chunk_idx + 1) * CHUNK_SIZE_K_LOW;

      // Calculate quantization scale and zero point
      float key_min = FLT_MAX;
      float key_max = -FLT_MAX;
#pragma unroll
      for (int offset = key_group_start; offset < key_group_end; offset += 1) {
        const int64_t src_kv_idx = src_kv_idx_base + offset;
        key_min = MIN(key_min, to_float(key[src_kv_idx]));
        key_max = MAX(key_max, to_float(key[src_kv_idx]));
      }

      float key_scale = __fdividef(key_max - key_min, range_k_low);
      float key_zero_point = key_min;
      float inv_key_scale = __fdividef(1.0f, key_scale);

      // Write key to cache
      // layout of keys within a unified page:
      // [NUM_PACKS/K_VEC_SIZE/THREAD_GROUP_SIZE_K,
      //                                        NUM_TOKENS_PER_PAGE,
      //                                        THREAD_GROUP_SIZE_K,
      //                                        K_VEC_SIZE]
#pragma unroll
      for (int i = key_chunk_idx * NUM_K_PACKS_LOW_PER_GROUP;
           i < (key_chunk_idx + 1) * NUM_K_PACKS_LOW_PER_GROUP; i++) {
        // const int vec_idx = i / K_VEC_SIZE;
        // const int idx_0 = vec_idx % NUM_VECS_PER_THREAD_K_LOW;
        // const int idx_1 = token_idx_within_the_page;
        // const int idx_2 = vec_idx / NUM_VECS_PER_THREAD_K_LOW;  // thread
        // group offset const int idx_3 = i % K_VEC_SIZE;
        const int vec_idx = i >> LOG2_K_VEC_SIZE;
        const int idx_0 = mod_pow2(vec_idx, LOG2_NUM_VECS_PER_THREAD_K_LOW);
        const int idx_1 = token_idx_within_the_page;
        const int idx_2 =
            vec_idx >> LOG2_NUM_VECS_PER_THREAD_K_LOW;  // thread group offset
        const int idx_3 = mod_pow2(i, LOG2_K_VEC_SIZE);
        const int idx =
            idx_0 * NUM_TOKENS_PER_PAGE_LOW * THREAD_GROUP_SIZE_K * K_VEC_SIZE +
            idx_1 * THREAD_GROUP_SIZE_K * K_VEC_SIZE + idx_2 * K_VEC_SIZE +
            idx_3;
        kv_cache[k_offset + idx] = quant_and_pack<scalar_t>(
            key + src_kv_idx_base + i * K_PACK_SIZE_LOW, BITS_K_LOW,
            inv_key_scale, key_zero_point);
      }
      from_float(kv_cache[k_meta_offset +
                          token_idx_within_the_page * 2 * CHUNKS_K_LOW +
                          key_chunk_idx * 2],
                 key_scale);
      from_float(kv_cache[k_meta_offset +
                          token_idx_within_the_page * 2 * CHUNKS_K_LOW +
                          key_chunk_idx * 2 + 1],
                 key_zero_point);
    }

    // Loop over value groups
    const int64_t v_offset =
        dst_physical_block_number * unified_page_size + VAL_BASE_LOW;
    const int64_t v_meta_offset =
        dst_physical_block_number * unified_page_size + VAL_META_BASE_LOW;
    for (int value_chunk_idx = 0; value_chunk_idx < CHUNKS_V_LOW;
         value_chunk_idx++) {
      // Calculate the start and end indices of the value group
      int value_group_start = value_chunk_idx * CHUNK_SIZE_V_LOW;
      int value_group_end = (value_chunk_idx + 1) * CHUNK_SIZE_V_LOW;

      // Calculate quantization scale and zero point
      float value_min = FLT_MAX;
      float value_max = -FLT_MAX;

#pragma unroll
      for (int offset = value_group_start; offset < value_group_end;
           offset += 1) {
        const int64_t src_kv_idx = src_kv_idx_base + offset;
        value_min = MIN(value_min, to_float(value[src_kv_idx]));
        value_max = MAX(value_max, to_float(value[src_kv_idx]));
      }

      float value_scale = __fdividef(value_max - value_min, range_v_low);
      float value_zero_point = value_min;
      float inv_value_scale = __fdividef(1.0f, value_scale);

      // Write value to cache
      // layout of values within a unified page: [NUM_PACKS /
      // THREAD_GROUP_SIZE_V,
      //                                          NUM_TOKENS_PER_PAGE /
      //                                          V_VEC_SIZE,
      //                                          THREAD_GROUP_SIZE_V,
      //                                          V_VEC_SIZE]
#pragma unroll
      for (int i = value_chunk_idx * NUM_V_PACKS_LOW_PER_GROUP;
           i < (value_chunk_idx + 1) * NUM_V_PACKS_LOW_PER_GROUP; i++) {
        // const int idx_0 = i % NUM_PACKS_PER_THREAD_V_LOW;
        // const int idx_1 = token_idx_within_the_page / V_VEC_SIZE;
        // const int idx_2 = i / NUM_PACKS_PER_THREAD_V_LOW;
        // const int idx_3 = token_idx_within_the_page % V_VEC_SIZE;
        const int idx_0 = mod_pow2(i, LOG2_NUM_PACKS_PER_THREAD_V_LOW);
        const int idx_1 = token_idx_within_the_page >> LOG2_V_VEC_SIZE;
        const int idx_2 = i >> LOG2_NUM_PACKS_PER_THREAD_V_LOW;
        const int idx_3 = mod_pow2(token_idx_within_the_page, LOG2_V_VEC_SIZE);
        const int idx =
            idx_0 * PADDED_NUM_TOKENS_PER_PAGE_LOW * THREAD_GROUP_SIZE_V +
            idx_1 * THREAD_GROUP_SIZE_V * V_VEC_SIZE + idx_2 * V_VEC_SIZE +
            idx_3;
        kv_cache[v_offset + idx] = quant_and_pack<scalar_t>(
            value + src_kv_idx_base + i * V_PACK_SIZE_LOW, BITS_V_LOW,
            inv_value_scale, value_zero_point);
      }
      from_float(kv_cache[v_meta_offset +
                          token_idx_within_the_page * 2 * CHUNKS_V_LOW +
                          value_chunk_idx * 2],
                 value_scale);
      from_float(kv_cache[v_meta_offset +
                          token_idx_within_the_page * 2 * CHUNKS_V_LOW +
                          value_chunk_idx * 2 + 1],
                 value_zero_point);
    }

    // Write score to cache.
    const int64_t score_offset =
        dst_physical_block_number * unified_page_size + SCORE_BASE_LOW;
    *reinterpret_cast<float*>(
        &kv_cache[score_offset + token_idx_within_the_page * 2]) =
        s_scores[s_positions[token_idx]];

    // Write position to cache.
    int ideal_num_queries = prompt_len - s_positions[token_idx];
    int real_num_queries = MIN(ideal_num_queries, NUM_TOKENS_SCORE);
    const int64_t position_offset =
        dst_physical_block_number * unified_page_size + POSITION_BASE_LOW;
    // *reinterpret_cast<int*>(&kv_cache[position_offset +
    // token_idx_within_the_page * 2]) = s_positions[token_idx];

    // A smart trick: by writing s_positions[token_idx] + (ideal_num_queries -
    // real_num_queries) to the position cache,
    //                we effectively ensure that in decode phase, num_queries is
    //                calculated correctly.
    *reinterpret_cast<int*>(
        &kv_cache[position_offset + token_idx_within_the_page * 2]) =
        s_positions[token_idx] + (ideal_num_queries - real_num_queries);
  }

  // Handle high-precision pages
  for (int token_idx = thread_idx + count_pruned + count_quantized;
       token_idx < prompt_len; token_idx += NUM_THREADS_PROMPT) {
    // Access block table from left
    int64_t dst_physical_block_number = static_cast<int64_t>(
        *(head_block_table_left + (token_idx - count_pruned - count_quantized) /
                                      NUM_TOKENS_PER_PAGE_HIGH));
    int token_idx_within_the_page =
        (token_idx - count_pruned - count_quantized) % NUM_TOKENS_PER_PAGE_HIGH;
    const int64_t src_kv_idx_base =
        (prompt_start + s_positions[token_idx]) * kv_stride +
        kv_head_idx * HEAD_SIZE;

    // if ((token_idx == prompt_len - 1 || token_idx == count_pruned +
    // count_quantized) && seq_idx == 0) {
    //   printf("[Debug info from compress_and_append_cache_prompt_phase_kernel]
    //   high-prec writes to KV, seq_idx: %d, slot_idx: %d, layer_idx: %d,
    //   kv_head_idx: %d, token_idx: %d, prompt_len: %d, count_pruned: %d,
    //   count_quantized: %d, dst_phy_blk_num: %lld\n",
    //     seq_idx, slot_idx, layer_idx, kv_head_idx, token_idx, prompt_len,
    //     count_pruned, count_quantized, (long long)dst_physical_block_number);
    // }

    // Loop over key groups
    const int64_t k_offset =
        dst_physical_block_number * unified_page_size + KEY_BASE_HIGH;
    const int64_t k_meta_offset =
        dst_physical_block_number * unified_page_size + KEY_META_BASE_HIGH;
    for (int key_chunk_idx = 0; key_chunk_idx < CHUNKS_K_HIGH;
         key_chunk_idx++) {
      // Calculate the start and end indices of the key group
      int key_group_start = key_chunk_idx * CHUNK_SIZE_K_HIGH;
      int key_group_end = (key_chunk_idx + 1) * CHUNK_SIZE_K_HIGH;

      // Calculate quantization scale and zero point
      float key_min = FLT_MAX;
      float key_max = -FLT_MAX;
#pragma unroll
      for (int offset = key_group_start; offset < key_group_end; offset += 1) {
        const int64_t src_kv_idx = src_kv_idx_base + offset;
        key_min = MIN(key_min, to_float(key[src_kv_idx]));
        key_max = MAX(key_max, to_float(key[src_kv_idx]));
      }

      float key_scale = __fdividef(key_max - key_min, range_k_high);
      float key_zero_point = key_min;
      float inv_key_scale = __fdividef(1.0f, key_scale);

      // Write key to cache
      // layout of keys within a unified page:
      // [NUM_PACKS/K_VEC_SIZE/THREAD_GROUP_SIZE_K,
      //                                        NUM_TOKENS_PER_PAGE,
      //                                        THREAD_GROUP_SIZE_K,
      //                                        K_VEC_SIZE]
#pragma unroll
      for (int i = key_chunk_idx * NUM_K_PACKS_HIGH_PER_GROUP;
           i < (key_chunk_idx + 1) * NUM_K_PACKS_HIGH_PER_GROUP; i++) {
        // const int vec_idx = i / K_VEC_SIZE;
        // const int idx_0 = vec_idx % NUM_VECS_PER_THREAD_K_HIGH;
        // const int idx_1 = token_idx_within_the_page;
        // const int idx_2 = vec_idx / NUM_VECS_PER_THREAD_K_HIGH;  // thread
        // group offset const int idx_3 = i % K_VEC_SIZE;
        const int vec_idx = i >> LOG2_K_VEC_SIZE;
        const int idx_0 = mod_pow2(vec_idx, LOG2_NUM_VECS_PER_THREAD_K_HIGH);
        const int idx_1 = token_idx_within_the_page;
        const int idx_2 =
            vec_idx >> LOG2_NUM_VECS_PER_THREAD_K_HIGH;  // thread group offset
        const int idx_3 = mod_pow2(i, LOG2_K_VEC_SIZE);
        const int idx = idx_0 * NUM_TOKENS_PER_PAGE_HIGH * THREAD_GROUP_SIZE_K *
                            K_VEC_SIZE +
                        idx_1 * THREAD_GROUP_SIZE_K * K_VEC_SIZE +
                        idx_2 * K_VEC_SIZE + idx_3;
        kv_cache[k_offset + idx] = quant_and_pack<scalar_t>(
            key + src_kv_idx_base + i * K_PACK_SIZE_HIGH, BITS_K_HIGH,
            inv_key_scale, key_zero_point);
      }
      from_float(kv_cache[k_meta_offset +
                          token_idx_within_the_page * 2 * CHUNKS_K_HIGH +
                          key_chunk_idx * 2],
                 key_scale);
      from_float(kv_cache[k_meta_offset +
                          token_idx_within_the_page * 2 * CHUNKS_K_HIGH +
                          key_chunk_idx * 2 + 1],
                 key_zero_point);
    }

    // Loop over value groups
    const int64_t v_offset =
        dst_physical_block_number * unified_page_size + VAL_BASE_HIGH;
    const int64_t v_meta_offset =
        dst_physical_block_number * unified_page_size + VAL_META_BASE_HIGH;
    for (int value_chunk_idx = 0; value_chunk_idx < CHUNKS_V_HIGH;
         value_chunk_idx++) {
      // Calculate the start and end indices of the value group
      int value_group_start = value_chunk_idx * CHUNK_SIZE_V_HIGH;
      int value_group_end = (value_chunk_idx + 1) * CHUNK_SIZE_V_HIGH;

      // Calculate quantization scale and zero point
      float value_min = FLT_MAX;
      float value_max = -FLT_MAX;
#pragma unroll
      for (int offset = value_group_start; offset < value_group_end;
           offset += 1) {
        const int64_t src_kv_idx = src_kv_idx_base + offset;
        value_min = MIN(value_min, to_float(value[src_kv_idx]));
        value_max = MAX(value_max, to_float(value[src_kv_idx]));
      }

      float value_scale = __fdividef(value_max - value_min, range_v_high);
      float value_zero_point = value_min;
      float inv_value_scale = __fdividef(1.0f, value_scale);

      // Write value to cache
      // layout of values within a unified page: [NUM_PACKS /
      // THREAD_GROUP_SIZE_V,
      //                                          NUM_TOKENS_PER_PAGE /
      //                                          V_VEC_SIZE,
      //                                          THREAD_GROUP_SIZE_V,
      //                                          V_VEC_SIZE]
#pragma unroll
      for (int i = value_chunk_idx * NUM_V_PACKS_HIGH_PER_GROUP;
           i < (value_chunk_idx + 1) * NUM_V_PACKS_HIGH_PER_GROUP; i++) {
        // const int idx_0 = i % NUM_PACKS_PER_THREAD_V_HIGH;
        // const int idx_1 = token_idx_within_the_page / V_VEC_SIZE;
        // const int idx_2 = i / NUM_PACKS_PER_THREAD_V_HIGH;
        // const int idx_3 = token_idx_within_the_page % V_VEC_SIZE;
        const int idx_0 = mod_pow2(i, LOG2_NUM_PACKS_PER_THREAD_V_HIGH);
        const int idx_1 = token_idx_within_the_page >> LOG2_V_VEC_SIZE;
        const int idx_2 = i >> LOG2_NUM_PACKS_PER_THREAD_V_HIGH;
        const int idx_3 = mod_pow2(token_idx_within_the_page, LOG2_V_VEC_SIZE);
        const int idx =
            idx_0 * PADDED_NUM_TOKENS_PER_PAGE_HIGH * THREAD_GROUP_SIZE_V +
            idx_1 * THREAD_GROUP_SIZE_V * V_VEC_SIZE + idx_2 * V_VEC_SIZE +
            idx_3;
        kv_cache[v_offset + idx] = quant_and_pack<scalar_t>(
            value + src_kv_idx_base + i * V_PACK_SIZE_HIGH, BITS_V_HIGH,
            inv_value_scale, value_zero_point);
      }
      from_float(kv_cache[v_meta_offset +
                          token_idx_within_the_page * 2 * CHUNKS_V_HIGH +
                          value_chunk_idx * 2],
                 value_scale);
      from_float(kv_cache[v_meta_offset +
                          token_idx_within_the_page * 2 * CHUNKS_V_HIGH +
                          value_chunk_idx * 2 + 1],
                 value_zero_point);
    }

    // Write score to cache.
    const int64_t score_offset =
        dst_physical_block_number * unified_page_size + SCORE_BASE_HIGH;
    *reinterpret_cast<float*>(
        &kv_cache[score_offset + token_idx_within_the_page * 2]) =
        s_scores[s_positions[token_idx]];

    // Write position to cache.
    int ideal_num_queries = prompt_len - s_positions[token_idx];
    int real_num_queries = MIN(ideal_num_queries, NUM_TOKENS_SCORE);
    const int64_t position_offset =
        dst_physical_block_number * unified_page_size + POSITION_BASE_HIGH;
    // A smart trick: by writing s_positions[token_idx] + (ideal_num_queries -
    // real_num_queries) to the position cache,
    //                we effectively ensure that in decode phase, num_queries is
    //                calculated correctly.
    *reinterpret_cast<int*>(
        &kv_cache[position_offset + token_idx_within_the_page * 2]) =
        s_positions[token_idx] + (ideal_num_queries - real_num_queries);
  }
}

template <typename scalar_t, int HEAD_SIZE, int BITS_K_HIGH, int BITS_V_HIGH,
          int BITS_K_LOW, int BITS_V_LOW, int CHUNKS_K_HIGH, int CHUNKS_V_HIGH,
          int CHUNKS_K_LOW, int CHUNKS_V_LOW, int NUM_TOKENS_PER_PAGE_HIGH,
          int NUM_TOKENS_PER_PAGE_LOW,
          int THREAD_GROUP_SIZE_V>
__global__ void compress_and_append_cache_decode_phase_kernel(
    const scalar_t* __restrict__ key,    // [num_seqs, num_kv_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_seqs, num_kv_heads, head_size]
    const int* __restrict__ positions,   // [num_seqs]
    uint16_t* __restrict__ kv_cache,     // [num_blocks, unified_page_size]
    const int* __restrict__ slot_ids,    // [num_seqs]
    const int* __restrict__ block_tables,  // [num_slots, num_layers,
                                           // num_kv_heads,
                                           // max_num_blocks_per_seq]
    int* __restrict__ kv_len_tables,  // [num_slots, num_layers, num_kv_heads,
                                      // 2]
    const float* __restrict__ compress_config_tables,  // [num_slots, 2]
    const int max_context_len, const int kv_buffer_size, const int layer_idx,
    const int kv_stride, const int num_layers, const int num_kv_heads,
    const int unified_page_size, const int max_num_blocks_per_seq) {
  const int seq_idx = blockIdx.y;
  const int slot_idx = slot_ids[seq_idx];
  const int kv_head_idx = blockIdx.x;
  const int thread_idx = threadIdx.x;

  const int kv_len_tables_offset = slot_idx * num_layers * num_kv_heads * 2 +
                                   layer_idx * num_kv_heads * 2 +
                                   kv_head_idx * 2;
  const int context_len_left = kv_len_tables[kv_len_tables_offset];
  const int context_len_right = kv_len_tables[kv_len_tables_offset + 1];
  const int context_len = context_len_left + context_len_right;
  const int position = positions[seq_idx];

  // extern __shared__ char shared_memory[];
  // float* s_scores = reinterpret_cast<float*>(shared_memory);
  // size_t s_positions_offset = max_context_len * sizeof(float);
  // int* s_positions = reinterpret_cast<int*>(shared_memory +
  // s_positions_offset);

  const float range_k_high = get_quant_range(BITS_K_HIGH);
  const float range_v_high = get_quant_range(BITS_V_HIGH);
  const float range_k_low = get_quant_range(BITS_K_LOW);
  const float range_v_low = get_quant_range(BITS_V_LOW);

  constexpr int K_PACK_SIZE_HIGH = 16 / BITS_K_HIGH;
  constexpr int V_PACK_SIZE_HIGH = 16 / BITS_V_HIGH;
  constexpr int K_PACK_SIZE_LOW = 16 / BITS_K_LOW;
  constexpr int V_PACK_SIZE_LOW = 16 / BITS_V_LOW;

  constexpr int NUM_K_PACKS_HIGH = HEAD_SIZE / K_PACK_SIZE_HIGH;
  constexpr int NUM_V_PACKS_HIGH = HEAD_SIZE / V_PACK_SIZE_HIGH;
  constexpr int NUM_K_PACKS_LOW = HEAD_SIZE / K_PACK_SIZE_LOW;
  constexpr int NUM_V_PACKS_LOW = HEAD_SIZE / V_PACK_SIZE_LOW;

  constexpr int NUM_K_VECS_HIGH = NUM_K_PACKS_HIGH / K_VEC_SIZE;
  constexpr int NUM_K_VECS_LOW = NUM_K_PACKS_LOW / K_VEC_SIZE;
  constexpr int NUM_VECS_PER_THREAD_K_HIGH =
      NUM_K_VECS_HIGH / THREAD_GROUP_SIZE_K;
  constexpr int NUM_VECS_PER_THREAD_K_LOW =
      NUM_K_VECS_LOW / THREAD_GROUP_SIZE_K;

  constexpr int CHUNK_SIZE_K_HIGH = HEAD_SIZE / CHUNKS_K_HIGH;
  constexpr int CHUNK_SIZE_V_HIGH = HEAD_SIZE / CHUNKS_V_HIGH;
  constexpr int CHUNK_SIZE_K_LOW = HEAD_SIZE / CHUNKS_K_LOW;
  constexpr int CHUNK_SIZE_V_LOW = HEAD_SIZE / CHUNKS_V_LOW;

  constexpr int NUM_K_PACKS_HIGH_PER_GROUP = NUM_K_PACKS_HIGH / CHUNKS_K_HIGH;
  constexpr int NUM_V_PACKS_HIGH_PER_GROUP = NUM_V_PACKS_HIGH / CHUNKS_V_HIGH;
  constexpr int NUM_K_PACKS_LOW_PER_GROUP = NUM_K_PACKS_LOW / CHUNKS_K_LOW;
  constexpr int NUM_V_PACKS_LOW_PER_GROUP = NUM_V_PACKS_LOW / CHUNKS_V_LOW;

  constexpr int LOG2_NUM_VECS_PER_THREAD_K_HIGH =
      log2_of_pow2(NUM_VECS_PER_THREAD_K_HIGH);
  constexpr int LOG2_NUM_VECS_PER_THREAD_K_LOW =
      log2_of_pow2(NUM_VECS_PER_THREAD_K_LOW);
  static_assert(LOG2_NUM_VECS_PER_THREAD_K_HIGH >= 0);
  static_assert(LOG2_NUM_VECS_PER_THREAD_K_LOW >= 0);

  constexpr int NUM_PACKS_PER_THREAD_V_HIGH =
      NUM_V_PACKS_HIGH / THREAD_GROUP_SIZE_V;
  constexpr int NUM_PACKS_PER_THREAD_V_LOW =
      NUM_V_PACKS_LOW / THREAD_GROUP_SIZE_V;
  constexpr int LOG2_NUM_PACKS_PER_THREAD_V_HIGH =
      log2_of_pow2(NUM_PACKS_PER_THREAD_V_HIGH);
  constexpr int LOG2_NUM_PACKS_PER_THREAD_V_LOW =
      log2_of_pow2(NUM_PACKS_PER_THREAD_V_LOW);
  static_assert(LOG2_NUM_PACKS_PER_THREAD_V_HIGH >= 0);
  static_assert(LOG2_NUM_PACKS_PER_THREAD_V_LOW >= 0);

  // Align the starting address of each segment (key, key meta, val, val meta,
  // score, pos) to 32 bytes.
  constexpr int KEY_BASE_HIGH = 0;
  constexpr int KEY_META_BASE_HIGH =
      DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_HIGH * NUM_K_PACKS_HIGH * 2, 32) *
          32 / sizeof(uint16_t) +
      KEY_BASE_HIGH;
  constexpr int VAL_BASE_HIGH =
      DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_HIGH * 4 * CHUNKS_K_HIGH +
                          KEY_META_BASE_HIGH * sizeof(uint16_t),
                      128) *
      128 / sizeof(uint16_t);
  constexpr int PADDED_NUM_TOKENS_PER_PAGE_HIGH =
      DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_HIGH, V_VEC_SIZE) * V_VEC_SIZE;
  constexpr int VAL_META_BASE_HIGH =
      DIVIDE_ROUND_UP(PADDED_NUM_TOKENS_PER_PAGE_HIGH * NUM_V_PACKS_HIGH * 2,
                      32) *
          32 / sizeof(uint16_t) +
      VAL_BASE_HIGH;
  constexpr int SCORE_BASE_HIGH =
      DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_HIGH * 4 * CHUNKS_V_HIGH, 32) * 32 /
          sizeof(uint16_t) +
      VAL_META_BASE_HIGH;
  constexpr int POSITION_BASE_HIGH =
      NUM_TOKENS_PER_PAGE_HIGH * 4 / sizeof(uint16_t) + SCORE_BASE_HIGH;

  constexpr int KEY_BASE_LOW = 0;
  constexpr int KEY_META_BASE_LOW =
      DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_LOW * NUM_K_PACKS_LOW * 2, 32) * 32 /
          sizeof(uint16_t) +
      KEY_BASE_LOW;
  constexpr int VAL_BASE_LOW =
      DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_LOW * 4 * CHUNKS_K_LOW +
                          KEY_META_BASE_LOW * sizeof(uint16_t),
                      128) *
      128 / sizeof(uint16_t);
  constexpr int PADDED_NUM_TOKENS_PER_PAGE_LOW =
      DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_LOW, V_VEC_SIZE) * V_VEC_SIZE;
  constexpr int VAL_META_BASE_LOW =
      DIVIDE_ROUND_UP(PADDED_NUM_TOKENS_PER_PAGE_LOW * NUM_V_PACKS_LOW * 2,
                      32) *
          32 / sizeof(uint16_t) +
      VAL_BASE_LOW;
  constexpr int SCORE_BASE_LOW =
      DIVIDE_ROUND_UP(NUM_TOKENS_PER_PAGE_LOW * 4 * CHUNKS_V_LOW, 32) * 32 /
          sizeof(uint16_t) +
      VAL_META_BASE_LOW;
  constexpr int POSITION_BASE_LOW =
      NUM_TOKENS_PER_PAGE_LOW * 4 / sizeof(uint16_t) + SCORE_BASE_LOW;

  // Iterate over scores and positions in kv_cache
  // Each thread handles one entry in a block.
  const int* head_block_table_left =
      block_tables + (slot_idx * num_layers * num_kv_heads +
                      layer_idx * num_kv_heads + kv_head_idx) *
                         max_num_blocks_per_seq;
  const int* head_block_table_right =
      head_block_table_left + max_num_blocks_per_seq - 1;
  float score_sum = 0.0f;

  // thread local least critical token (high precision)
  float local_min_score_left = FLT_MAX;
  int local_min_idx_left = -1;

  for (int token_idx = thread_idx; token_idx < context_len_left;
       token_idx += NUM_THREADS_DECODE) {
    // Process high-precision blocks
    // NOTE: access block table from left
    const int64_t physical_block_number = static_cast<int64_t>(
        *(head_block_table_left + (token_idx / NUM_TOKENS_PER_PAGE_HIGH)));
    const int token_idx_within_the_page = token_idx % NUM_TOKENS_PER_PAGE_HIGH;

    const int64_t score_offset = physical_block_number * unified_page_size +
                                 SCORE_BASE_HIGH +
                                 token_idx_within_the_page * 2;
    const int64_t position_offset = physical_block_number * unified_page_size +
                                    POSITION_BASE_HIGH +
                                    token_idx_within_the_page * 2;
    float score = *reinterpret_cast<float*>(&kv_cache[score_offset]);
    int cached_position = *reinterpret_cast<int*>(&kv_cache[position_offset]);
    int num_queries = position - cached_position;
    // if (num_queries <= 0) {
    //   printf("[Debug info from compress_and_append_cache_decode_phase_kernel]
    //   seq_idx: %d, kv_head_idx: %d, token_idx: %d, num_queries: %d, pos: %d,
    //   cached_pos: %d, idx_in_page: %d, phy_blk_num: %lld\n",
    //         seq_idx, kv_head_idx, token_idx, num_queries, position,
    //         cached_position, token_idx_within_the_page, (long
    //         long)physical_block_number);
    // }
    assert(num_queries > 0);

    score = score / num_queries;
    if (num_queries >= kv_buffer_size && score < local_min_score_left) {
      local_min_score_left = score;
      local_min_idx_left = token_idx;
    }

    // if (seq_idx == 0) {
    //   printf("[Debug info from compress_and_append_cache_decode_phase_kernel]
    //   high-prec seq_idx: %d, slot_idx: %d, layer_idx: %d, kv_head_idx: %d,
    //   token_idx: %d, num_queries: %d, mean_score: %f\n",
    //         seq_idx, slot_idx, layer_idx, kv_head_idx, token_idx,
    //         num_queries, score);
    // }

    // score_sum += score;
    // s_scores[token_idx] = score;
    // s_positions[token_idx] = cached_position;
  }

  // thread local least critical token (low precision)
  float local_min_score_right = FLT_MAX;
  int local_min_idx_right = -1;

  for (int token_idx = context_len_left + thread_idx; token_idx < context_len;
       token_idx += NUM_THREADS_DECODE) {
    // process low-precision blocks
    // NOTE: access block table from right
    const int64_t physical_block_number = static_cast<int64_t>(
        *(head_block_table_right -
          (token_idx - context_len_left) / NUM_TOKENS_PER_PAGE_LOW));
    const int token_idx_within_the_page =
        (token_idx - context_len_left) % NUM_TOKENS_PER_PAGE_LOW;
    const int64_t score_offset = physical_block_number * unified_page_size +
                                 SCORE_BASE_LOW + token_idx_within_the_page * 2;
    const int64_t position_offset = physical_block_number * unified_page_size +
                                    POSITION_BASE_LOW +
                                    token_idx_within_the_page * 2;
    float score = *reinterpret_cast<float*>(&kv_cache[score_offset]);
    int cached_position = *reinterpret_cast<int*>(&kv_cache[position_offset]);
    int num_queries = position - cached_position;

    // if (num_queries <= 0) {
    //   printf("[Debug info from compress_and_append_cache_decode_phase_kernel]
    //   low-prec seq_idx: %d, slot_idx: %d, layer_idx: %d, kv_head_idx: %d,
    //   token_idx: %d, num_queries: %d, pos: %d, context_len_left: %d,
    //   context_len_right: %d, cached_pos: %d, idx_in_page: %d, phy_blk_num:
    //   %lld\n",
    //         seq_idx, slot_idx, layer_idx, kv_head_idx, token_idx,
    //         num_queries, position, context_len_left, context_len_right,
    //         cached_position, token_idx_within_the_page, (long
    //         long)physical_block_number);
    // }

    assert(num_queries > 0);

    // if (seq_idx == 0 && kv_head_idx == 0) {
    //   printf("[Debug info from compress_and_append_cache_decode_phase_kernel]
    //   seq_idx: %d, kv_head_idx: %d, token_idx: %d, num_queries: %d\n",
    //         seq_idx, kv_head_idx, token_idx, num_queries);
    // }

    score = score / num_queries;
    if (num_queries >= kv_buffer_size && score < local_min_score_right) {
      local_min_score_right = score;
      local_min_idx_right = token_idx;
    }

    // score_sum += score;
    // s_scores[token_idx] = score;
    // s_positions[token_idx] = cached_position;
  }
  __syncthreads();

  // Calculate the victim token index
  __shared__ int left_victim_token_idx;
  __shared__ int right_victim_token_idx;

  if (context_len + 1 <= kv_buffer_size) {
    if (thread_idx == 0) {
      // The new token is appended to a high precision block
      left_victim_token_idx = context_len_left;
      kv_len_tables[kv_len_tables_offset] += 1;
      right_victim_token_idx = -1;
    }
    __syncthreads();
  } else {
    // NOTE: we no more normalize token score over the entire sequence
    // but just use its own mean score for reference

    // Step 1: Reduction across threads in the block
    __shared__ float s_min_scores[NUM_THREADS_DECODE];
    __shared__ int s_min_idxs[NUM_THREADS_DECODE];
    s_min_scores[thread_idx] = local_min_score_left;
    s_min_idxs[thread_idx] = local_min_idx_left;
    __syncthreads();

#pragma unroll
    for (unsigned int s = NUM_THREADS_DECODE / 2; s > 0; s >>= 1) {
      if (thread_idx < s) {
        if (s_min_scores[thread_idx + s] < s_min_scores[thread_idx]) {
          s_min_scores[thread_idx] = s_min_scores[thread_idx + s];
          s_min_idxs[thread_idx] = s_min_idxs[thread_idx + s];
        }
      }
      __syncthreads();
    }

    float min_score_left = s_min_scores[0];
    int min_idx_left = s_min_idxs[0];

    /* Sort tokens from low precision blocks */
    // Step 1: Reduction across threads in the block
    s_min_scores[thread_idx] = local_min_score_right;
    s_min_idxs[thread_idx] = local_min_idx_right;
    __syncthreads();

#pragma unroll
    for (unsigned int s = NUM_THREADS_DECODE / 2; s > 0; s >>= 1) {
      if (thread_idx < s) {
        if (s_min_scores[thread_idx + s] < s_min_scores[thread_idx]) {
          s_min_scores[thread_idx] = s_min_scores[thread_idx + s];
          s_min_idxs[thread_idx] = s_min_idxs[thread_idx + s];
        }
      }
      __syncthreads();
    }

    float min_score_right = s_min_scores[0];
    int min_idx_right = s_min_idxs[0];

    // Compute actual threshold
    // TODO: investigate using 1.0f / context_len (KV length) here
    // const float base_threshold = 1.0f / context_len;
    const float base_threshold = 1.0f / position;
    const float prune_threshold =
        base_threshold * compress_config_tables[slot_idx * 2 + 0];
    const float quant_threshold =
        base_threshold * compress_config_tables[slot_idx * 2 + 1];

    assert(prune_threshold >= 0 && prune_threshold <= 1);
    assert(quant_threshold >= 0 && quant_threshold <= 1);
    assert(quant_threshold >= prune_threshold);

    // Pruning and quantization policy
    if (thread_idx == 0) {
      if (min_score_left < prune_threshold &&
          min_score_left < min_score_right) {
        // Prune the left victim token
        left_victim_token_idx = min_idx_left;
        right_victim_token_idx = -1;
      } else if (min_score_left < quant_threshold) {
        if (min_score_right < prune_threshold) {
          // Requantize the left victim token to low precision, occupying the
          // space of the right victim token
          left_victim_token_idx = min_idx_left;
          right_victim_token_idx = min_idx_right;
        } else {
          // Requantize the left victim token to low precision, occupying a new
          // slot
          left_victim_token_idx = min_idx_left;
          right_victim_token_idx = context_len_right + context_len_left;
          kv_len_tables[kv_len_tables_offset + 1] += 1;
        }
      } else {
        // Append the new token to a high precision block
        left_victim_token_idx = context_len_left;
        right_victim_token_idx = -1;
        kv_len_tables[kv_len_tables_offset] += 1;
      }
    }
    __syncthreads();

    // if (layer_idx == 0 && seq_idx == 0 && kv_head_idx == 0 && thread_idx ==
    // 0) {
    //   printf("[Debug info from compress_and_append_cache_decode_phase_kernel]
    //   min_score_left: %f, min_idx_left: %d, min_score_right: %f,
    //   min_idx_right: %d, prune_threshold: %f, quant_threshold: %f,
    //   left_victim_token_idx: %d, right_victim_token_idx: %d\n",
    //     min_score_left, min_idx_left, min_score_right, min_idx_right,
    //     prune_threshold, quant_threshold, left_victim_token_idx,
    //     right_victim_token_idx);
    // }
  }

  // The code above shows that we have four cases:
  // 1. Prune the left victim token
  // 2. Requantize the left victim token, occupying the space of the right
  // victim token
  // 3. Requantize the left victim token, occupying a new slot
  // 4. Append the new token to a high precision block (left)
  //
  // What we need do to:
  // In case 2 and 3, requantize the left victim token to low precision
  // and store it in right_victim_token_idx (either min_idx_right or
  // context_len_right) In all four cases, quantize the new token to high
  // precision and store it in the left_victim_token_idx (either min_idx_left or
  // context_len_left)

  if (right_victim_token_idx >= 0) {
    // Requantize the high precision victim token (left) to low precision
    // (right)
    int64_t left_physical_block_number = static_cast<int64_t>(
        *(head_block_table_left +
          left_victim_token_idx / NUM_TOKENS_PER_PAGE_HIGH));
    int64_t left_token_idx_within_the_page =
        left_victim_token_idx % NUM_TOKENS_PER_PAGE_HIGH;
    int64_t left_k_offset =
        left_physical_block_number * unified_page_size + KEY_BASE_HIGH;
    int64_t left_k_meta_offset =
        left_physical_block_number * unified_page_size + KEY_META_BASE_HIGH;
    int64_t left_v_offset =
        left_physical_block_number * unified_page_size + VAL_BASE_HIGH;
    int64_t left_v_meta_offset =
        left_physical_block_number * unified_page_size + VAL_META_BASE_HIGH;
    int64_t left_score_offset =
        left_physical_block_number * unified_page_size + SCORE_BASE_HIGH;
    int64_t left_position_offset =
        left_physical_block_number * unified_page_size + POSITION_BASE_HIGH;

    int64_t right_physical_block_number = static_cast<int64_t>(
        *(head_block_table_right - (right_victim_token_idx - context_len_left) /
                                       NUM_TOKENS_PER_PAGE_LOW));
    int64_t right_token_idx_within_the_page =
        (right_victim_token_idx - context_len_left) % NUM_TOKENS_PER_PAGE_LOW;
    int64_t right_k_offset =
        right_physical_block_number * unified_page_size + KEY_BASE_LOW;
    int64_t right_k_meta_offset =
        right_physical_block_number * unified_page_size + KEY_META_BASE_LOW;
    int64_t right_v_offset =
        right_physical_block_number * unified_page_size + VAL_BASE_LOW;
    int64_t right_v_meta_offset =
        right_physical_block_number * unified_page_size + VAL_META_BASE_LOW;
    int64_t right_score_offset =
        right_physical_block_number * unified_page_size + SCORE_BASE_LOW;
    int64_t right_position_offset =
        right_physical_block_number * unified_page_size + POSITION_BASE_LOW;

    // Dequant key to float
    __shared__ float key_float[HEAD_SIZE];
    // layout of keys within a unified page:
    // [NUM_PACKS/K_VEC_SIZE/THREAD_GROUP_SIZE_K,
    //                                        NUM_TOKENS_PER_PAGE,
    //                                        THREAD_GROUP_SIZE_K,
    //                                        K_VEC_SIZE]
    for (int i = thread_idx; i < NUM_K_PACKS_HIGH; i += NUM_THREADS_DECODE) {
      int key_chunk_idx = i / NUM_K_PACKS_HIGH_PER_GROUP;
      const float key_scale_high =
          to_float(kv_cache[left_k_meta_offset +
                            left_token_idx_within_the_page * 2 * CHUNKS_K_HIGH +
                            key_chunk_idx * 2]);
      const float key_zero_point_high =
          to_float(kv_cache[left_k_meta_offset +
                            left_token_idx_within_the_page * 2 * CHUNKS_K_HIGH +
                            key_chunk_idx * 2 + 1]);

      const int vec_idx = i / K_VEC_SIZE;
      const int idx_0 = vec_idx % NUM_VECS_PER_THREAD_K_HIGH;
      const int idx_1 = left_token_idx_within_the_page;
      const int idx_2 =
          vec_idx / NUM_VECS_PER_THREAD_K_HIGH;  // thread group offset
      const int idx_3 = i % K_VEC_SIZE;
      const int idx =
          idx_0 * NUM_TOKENS_PER_PAGE_HIGH * THREAD_GROUP_SIZE_K * K_VEC_SIZE +
          idx_1 * THREAD_GROUP_SIZE_K * K_VEC_SIZE + idx_2 * K_VEC_SIZE + idx_3;
      unpack_and_dequant(kv_cache[left_k_offset + idx], BITS_K_HIGH,
                         key_scale_high, key_zero_point_high,
                         &key_float[i * K_PACK_SIZE_HIGH]);
    }

    __syncthreads();

    // Requant key to low precision, each thread handles one group
    for (int key_chunk_idx = thread_idx; key_chunk_idx < CHUNKS_K_LOW;
         key_chunk_idx += NUM_THREADS_DECODE) {
      int key_group_start = key_chunk_idx * CHUNK_SIZE_K_LOW;
      int key_group_end = (key_chunk_idx + 1) * CHUNK_SIZE_K_LOW;

      // Calculate min/max of key
      float key_min = FLT_MAX;
      float key_max = -FLT_MAX;
      for (int offset = key_group_start; offset < key_group_end; offset += 1) {
        key_min = MIN(key_min, key_float[offset]);
        key_max = MAX(key_max, key_float[offset]);
      }

      float key_scale = __fdividef(key_max - key_min, range_k_low);
      float key_zero_point = key_min;
      float inv_key_scale = __fdividef(1.0f, key_scale);

#pragma unroll
      for (int i = key_chunk_idx * NUM_K_PACKS_LOW_PER_GROUP;
           i < (key_chunk_idx + 1) * NUM_K_PACKS_LOW_PER_GROUP; i++) {
        const int vec_idx = i / K_VEC_SIZE;
        const int idx_0 = vec_idx % NUM_VECS_PER_THREAD_K_LOW;
        const int idx_1 = right_token_idx_within_the_page;
        const int idx_2 =
            vec_idx / NUM_VECS_PER_THREAD_K_LOW;  // thread group offset
        const int idx_3 = i % K_VEC_SIZE;
        const int idx =
            idx_0 * NUM_TOKENS_PER_PAGE_LOW * THREAD_GROUP_SIZE_K * K_VEC_SIZE +
            idx_1 * THREAD_GROUP_SIZE_K * K_VEC_SIZE + idx_2 * K_VEC_SIZE +
            idx_3;
        kv_cache[right_k_offset + idx] =
            quant_and_pack<float>(key_float + i * K_PACK_SIZE_LOW, BITS_K_LOW,
                                  inv_key_scale, key_zero_point);
      }
      from_float(kv_cache[right_k_meta_offset +
                          right_token_idx_within_the_page * 2 * CHUNKS_K_LOW +
                          key_chunk_idx * 2],
                 key_scale);
      from_float(kv_cache[right_k_meta_offset +
                          right_token_idx_within_the_page * 2 * CHUNKS_K_LOW +
                          key_chunk_idx * 2 + 1],
                 key_zero_point);
    }

    // Dequant value to float
    __shared__ float value_float[HEAD_SIZE];
    // layout of values within a unified page: [NUM_PACKS / THREAD_GROUP_SIZE_V,
    //                                          NUM_TOKENS_PER_PAGE /
    //                                          V_VEC_SIZE, THREAD_GROUP_SIZE_V,
    //                                          V_VEC_SIZE]
    for (int i = thread_idx; i < NUM_V_PACKS_HIGH; i += NUM_THREADS_DECODE) {
      int value_chunk_idx = i / NUM_V_PACKS_HIGH_PER_GROUP;
      const float value_scale_high =
          to_float(kv_cache[left_v_meta_offset +
                            left_token_idx_within_the_page * 2 * CHUNKS_V_HIGH +
                            value_chunk_idx * 2]);
      const float value_zero_point_high =
          to_float(kv_cache[left_v_meta_offset +
                            left_token_idx_within_the_page * 2 * CHUNKS_V_HIGH +
                            value_chunk_idx * 2 + 1]);

      const int idx_0 = i % NUM_PACKS_PER_THREAD_V_HIGH;
      const int idx_1 = left_token_idx_within_the_page / V_VEC_SIZE;
      const int idx_2 = i / NUM_PACKS_PER_THREAD_V_HIGH;
      const int idx_3 = left_token_idx_within_the_page % V_VEC_SIZE;
      const int idx =
          idx_0 * PADDED_NUM_TOKENS_PER_PAGE_HIGH * THREAD_GROUP_SIZE_V +
          idx_1 * THREAD_GROUP_SIZE_V * V_VEC_SIZE + idx_2 * V_VEC_SIZE + idx_3;
      unpack_and_dequant(kv_cache[left_v_offset + idx], BITS_V_HIGH,
                         value_scale_high, value_zero_point_high,
                         &value_float[i * V_PACK_SIZE_HIGH]);
    }

    __syncthreads();

    // Requant value to low precision, each thread handles one group
    for (int value_chunk_idx = thread_idx; value_chunk_idx < CHUNKS_V_LOW;
         value_chunk_idx += NUM_THREADS_DECODE) {
      int value_group_start = value_chunk_idx * CHUNK_SIZE_V_LOW;
      int value_group_end = (value_chunk_idx + 1) * CHUNK_SIZE_V_LOW;

      // Calculate min/max of value
      float value_min = FLT_MAX;
      float value_max = -FLT_MAX;
      for (int offset = value_group_start; offset < value_group_end;
           offset += 1) {
        value_min = MIN(value_min, value_float[offset]);
        value_max = MAX(value_max, value_float[offset]);
      }

      float value_scale = __fdividef(value_max - value_min, range_v_low);
      float value_zero_point = value_min;
      float inv_value_scale = __fdividef(1.0f, value_scale);
#pragma unroll
      for (int i = value_chunk_idx * NUM_V_PACKS_LOW_PER_GROUP;
           i < (value_chunk_idx + 1) * NUM_V_PACKS_LOW_PER_GROUP; i++) {
        const int idx_0 = i % NUM_PACKS_PER_THREAD_V_LOW;
        const int idx_1 = right_token_idx_within_the_page / V_VEC_SIZE;
        const int idx_2 = i / NUM_PACKS_PER_THREAD_V_LOW;
        const int idx_3 = right_token_idx_within_the_page % V_VEC_SIZE;
        const int idx =
            idx_0 * PADDED_NUM_TOKENS_PER_PAGE_LOW * THREAD_GROUP_SIZE_V +
            idx_1 * THREAD_GROUP_SIZE_V * V_VEC_SIZE + idx_2 * V_VEC_SIZE +
            idx_3;
        kv_cache[right_v_offset + idx] =
            quant_and_pack<float>(value_float + i * V_PACK_SIZE_LOW, BITS_V_LOW,
                                  inv_value_scale, value_zero_point);
      }
      from_float(kv_cache[right_v_meta_offset +
                          right_token_idx_within_the_page * 2 * CHUNKS_V_LOW +
                          value_chunk_idx * 2],
                 value_scale);
      from_float(kv_cache[right_v_meta_offset +
                          right_token_idx_within_the_page * 2 * CHUNKS_V_LOW +
                          value_chunk_idx * 2 + 1],
                 value_zero_point);
    }

    if (thread_idx == 0) {
      // Write score to cache
      *reinterpret_cast<float*>(
          &kv_cache[right_score_offset + right_token_idx_within_the_page * 2]) =
          *reinterpret_cast<float*>(
              &kv_cache[left_score_offset +
                        left_token_idx_within_the_page * 2]);

      // Write position to cache
      // assert(*reinterpret_cast<int*>(&kv_cache[left_position_offset +
      // left_token_idx_within_the_page * 2]) ==
      // s_positions[left_victim_token_idx]);
      *reinterpret_cast<int*>(&kv_cache[right_position_offset +
                                        right_token_idx_within_the_page * 2]) =
          *reinterpret_cast<int*>(
              &kv_cache[left_position_offset +
                        left_token_idx_within_the_page * 2]);
    }
  }
  __syncthreads();

  /* Write the new token to a high precision block */
  int64_t dst_physical_block_number =
      static_cast<int64_t>(*(head_block_table_left +
                             left_victim_token_idx / NUM_TOKENS_PER_PAGE_HIGH));
  int64_t token_idx_within_the_page =
      left_victim_token_idx % NUM_TOKENS_PER_PAGE_HIGH;

  int64_t k_offset =
      dst_physical_block_number * unified_page_size + KEY_BASE_HIGH;
  int64_t k_meta_offset =
      dst_physical_block_number * unified_page_size + KEY_META_BASE_HIGH;

  int64_t v_offset =
      dst_physical_block_number * unified_page_size + VAL_BASE_HIGH;
  int64_t v_meta_offset =
      dst_physical_block_number * unified_page_size + VAL_META_BASE_HIGH;

  const int64_t src_kv_idx_base = seq_idx * kv_stride + kv_head_idx * HEAD_SIZE;

  // Loop over key groups
  for (int key_chunk_idx = thread_idx; key_chunk_idx < CHUNKS_K_HIGH;
       key_chunk_idx += NUM_THREADS_DECODE) {
    int key_group_start = key_chunk_idx * CHUNK_SIZE_K_HIGH;
    int key_group_end = (key_chunk_idx + 1) * CHUNK_SIZE_K_HIGH;

    // Calculate min/max of key
    float key_min = FLT_MAX;
    float key_max = -FLT_MAX;
    for (int offset = key_group_start; offset < key_group_end; offset += 1) {
      key_min = MIN(key_min, to_float(key[src_kv_idx_base + offset]));
      key_max = MAX(key_max, to_float(key[src_kv_idx_base + offset]));
    }

    float key_scale = __fdividef(key_max - key_min, range_k_high);
    float key_zero_point = key_min;
    float inv_key_scale = __fdividef(1.0f, key_scale);

    for (int i = key_chunk_idx * NUM_K_PACKS_HIGH_PER_GROUP;
         i < (key_chunk_idx + 1) * NUM_K_PACKS_HIGH_PER_GROUP; i++) {
      const int vec_idx = i / K_VEC_SIZE;
      const int idx_0 = vec_idx % NUM_VECS_PER_THREAD_K_HIGH;
      const int idx_1 = token_idx_within_the_page;
      const int idx_2 = vec_idx / NUM_VECS_PER_THREAD_K_HIGH;
      const int idx_3 = i % K_VEC_SIZE;
      const int idx =
          idx_0 * NUM_TOKENS_PER_PAGE_HIGH * THREAD_GROUP_SIZE_K * K_VEC_SIZE +
          idx_1 * THREAD_GROUP_SIZE_K * K_VEC_SIZE + idx_2 * K_VEC_SIZE + idx_3;
      kv_cache[k_offset + idx] =
          quant_and_pack<scalar_t>(key + src_kv_idx_base + i * K_PACK_SIZE_HIGH,
                                   BITS_K_HIGH, inv_key_scale, key_zero_point);
    }
    // Write key quant metadata
    from_float(
        kv_cache[k_meta_offset + token_idx_within_the_page * 2 * CHUNKS_K_HIGH +
                 key_chunk_idx * 2],
        key_scale);
    from_float(
        kv_cache[k_meta_offset + token_idx_within_the_page * 2 * CHUNKS_K_HIGH +
                 key_chunk_idx * 2 + 1],
        key_zero_point);
  }

  // Loop over value groups
  for (int value_chunk_idx = thread_idx; value_chunk_idx < CHUNKS_V_HIGH;
       value_chunk_idx += NUM_THREADS_DECODE) {
    int value_group_start = value_chunk_idx * CHUNK_SIZE_V_HIGH;
    int value_group_end = (value_chunk_idx + 1) * CHUNK_SIZE_V_HIGH;

    // Calculate min/max of value
    float value_min = FLT_MAX;
    float value_max = -FLT_MAX;
    for (int offset = value_group_start; offset < value_group_end;
         offset += 1) {
      value_min = MIN(value_min, to_float(value[src_kv_idx_base + offset]));
      value_max = MAX(value_max, to_float(value[src_kv_idx_base + offset]));
    }

    float value_scale = __fdividef(value_max - value_min, range_v_high);
    float value_zero_point = value_min;
    float inv_value_scale = __fdividef(1.0f, value_scale);
    for (int i = value_chunk_idx * NUM_V_PACKS_HIGH_PER_GROUP;
         i < (value_chunk_idx + 1) * NUM_V_PACKS_HIGH_PER_GROUP; i++) {
      const int idx_0 = i % NUM_PACKS_PER_THREAD_V_HIGH;
      const int idx_1 = token_idx_within_the_page / V_VEC_SIZE;
      const int idx_2 = i / NUM_PACKS_PER_THREAD_V_HIGH;
      const int idx_3 = token_idx_within_the_page % V_VEC_SIZE;
      const int idx =
          idx_0 * PADDED_NUM_TOKENS_PER_PAGE_HIGH * THREAD_GROUP_SIZE_V +
          idx_1 * THREAD_GROUP_SIZE_V * V_VEC_SIZE + idx_2 * V_VEC_SIZE + idx_3;
      kv_cache[v_offset + idx] = quant_and_pack<scalar_t>(
          value + src_kv_idx_base + i * V_PACK_SIZE_HIGH, BITS_V_HIGH,
          inv_value_scale, value_zero_point);
    }
    // Write value quant metadata
    from_float(
        kv_cache[v_meta_offset + token_idx_within_the_page * 2 * CHUNKS_V_HIGH +
                 value_chunk_idx * 2],
        value_scale);
    from_float(
        kv_cache[v_meta_offset + token_idx_within_the_page * 2 * CHUNKS_V_HIGH +
                 value_chunk_idx * 2 + 1],
        value_zero_point);
  }

  if (thread_idx == 0) {
    // Write score to cache
    const int64_t score_offset =
        dst_physical_block_number * unified_page_size + SCORE_BASE_HIGH;
    *reinterpret_cast<float*>(
        &kv_cache[score_offset + token_idx_within_the_page * 2]) =
        0.0f;  // it will be updated in the next iteration

    // Write position to cache
    const int64_t position_offset =
        dst_physical_block_number * unified_page_size + POSITION_BASE_HIGH;
    *reinterpret_cast<int*>(
        &kv_cache[position_offset + token_idx_within_the_page * 2]) = position;
  }
}

}  // namespace diffkv_v2
#define LAUNCH_CACHE_KERNEL_PROMPT_PHASE(                                     \
    T, HEAD_SIZE, BITS_K_HIGH, BITS_V_HIGH, BITS_K_LOW, BITS_V_LOW,           \
    CHUNKS_K_HIGH, CHUNKS_V_HIGH, CHUNKS_K_LOW, CHUNKS_V_LOW,                 \
    NUM_TOKENS_PER_PAGE_HIGH, NUM_TOKENS_PER_PAGE_LOW, THREAD_GROUP_SIZE_V)   \
  diffkv_v2::compress_and_append_cache_prompt_phase_kernel<                   \
      T, HEAD_SIZE, BITS_K_HIGH, BITS_V_HIGH, BITS_K_LOW, BITS_V_LOW,         \
      CHUNKS_K_HIGH, CHUNKS_V_HIGH, CHUNKS_K_LOW, CHUNKS_V_LOW,               \
      NUM_TOKENS_PER_PAGE_HIGH, NUM_TOKENS_PER_PAGE_LOW, THREAD_GROUP_SIZE_V> \
      <<<grid, block, shared_mem_size, stream>>>(                             \
          reinterpret_cast<T*>(key.data_ptr()),                               \
          reinterpret_cast<T*>(value.data_ptr()), score.data_ptr<float>(),    \
          reinterpret_cast<uint16_t*>(kv_cache.data_ptr()),                   \
          slot_ids.data_ptr<int>(), block_tables.data_ptr<int>(),             \
          kv_len_tables.data_ptr<int>(), seq_start_loc.data_ptr<int>(),       \
          compress_config_tables.data_ptr<float>(), max_prompt_len,           \
          kv_buffer_size, layer_idx, kv_stride, num_layers, num_heads,        \
          num_kv_heads, unified_page_size, max_num_blocks_per_seq);

#define CALL_PROMPT_PHASE_CACHE_LAUNCHER_QUANT_CONFIG(T, HEAD_SIZE)            \
  TORCH_CHECK((quant_config == std::vector<int>{8, 4, 4, 2, 1, 1, 1, 1}),      \
              "DiffKV v2 requires K8V4/K4V2, one chunk");                      \
  TORCH_CHECK(num_tokens_per_page_high == 32 && num_tokens_per_page_low == 60, \
              "DiffKV v2 requires pages 32/60");                               \
  LAUNCH_CACHE_KERNEL_PROMPT_PHASE(T, HEAD_SIZE, 8, 4, 4, 2, 1, 1, 1, 1, 32,   \
                                   60, 8)

void compress_and_append_cache_prompt_phase(
    torch::Tensor& key,           // [total_num_tokens, num_kv_heads, head_size]
    torch::Tensor& value,         // [total_num_tokens, num_kv_heads, head_size]
    torch::Tensor& score,         // [num_seqs, num_heads, max_prompt_len]
    torch::Tensor& kv_cache,      // [num_blocks, unified_page_size]
    torch::Tensor& slot_ids,      // [num_seqs]
    torch::Tensor& block_tables,  // [num_slots, num_layers, num_kv_heads,
                                  // max_num_blocks_per_seq]
    torch::Tensor& kv_len_tables,  // [num_slots, num_layers, num_kv_heads, 2]
    torch::Tensor& seq_start_loc,  // [num_seqs + 1]
    torch::Tensor&
        compress_config_tables,  // [num_slots, 2] (prune_ratio, quant_ratio)
    const int kv_buffer_size, const int layer_idx, const int num_bits_k_high,
    const int num_bits_v_high, const int num_bits_k_low,
    const int num_bits_v_low, const int num_chunks_k_high,
    const int num_chunks_v_high, const int num_chunks_k_low,
    const int num_chunks_v_low, const int k_vec_size, const int v_vec_size,
    const int num_tokens_per_page_high, const int num_tokens_per_page_low) {
  diffkv_v2::check_cache(key, value, kv_cache, slot_ids, block_tables,
                         kv_len_tables, compress_config_tables, layer_idx);
  const c10::cuda::CUDAGuard device_guard(key.device());
  TORCH_CHECK(kv_buffer_size > 0, "KV buffer size must be positive");
  diffkv_v2::check_tensor(score, key, at::kFloat, "prompt scores");
  diffkv_v2::check_tensor(seq_start_loc, key, at::kInt, "sequence starts");
  TORCH_CHECK(score.dim() == 3 && score.size(0) == slot_ids.numel() &&
                  score.size(2) > 0 && seq_start_loc.dim() == 1 &&
                  seq_start_loc.numel() == slot_ids.numel() + 1,
              "Invalid prompt score or sequence-start shape");
  TORCH_CHECK(score.size(1) == key.size(1) * 4,
              "Prompt scores must have four query heads per KV head");

  TORCH_CHECK(k_vec_size == K_VEC_SIZE, "k_vec_size should be ", K_VEC_SIZE);
  TORCH_CHECK(v_vec_size == V_VEC_SIZE, "v_vec_size should be ", V_VEC_SIZE);

  std::vector<int> quant_config = {
      num_bits_k_high,   num_bits_v_high,   num_bits_k_low,   num_bits_v_low,
      num_chunks_k_high, num_chunks_v_high, num_chunks_k_low, num_chunks_v_low,
  };
  int lowest_bits = *std::min_element(quant_config.begin(), quant_config.end());
  int thread_group_size_v = get_thread_group_size_v(lowest_bits);

  TORCH_CHECK(key.dim() == 3, "Invalid DiffKV v2 cache input");
  TORCH_CHECK(value.dim() == 3, "Invalid DiffKV v2 cache input");
  TORCH_CHECK(score.dim() == 3, "Invalid DiffKV v2 cache input");
  TORCH_CHECK(kv_cache.dim() == 2, "Invalid DiffKV v2 cache input");
  TORCH_CHECK(slot_ids.dim() == 1, "Invalid DiffKV v2 cache input");
  TORCH_CHECK(block_tables.dim() == 4, "Invalid DiffKV v2 cache input");
  TORCH_CHECK(kv_len_tables.dim() == 4, "Invalid DiffKV v2 cache input");
  TORCH_CHECK(kv_len_tables.size(3) == 2, "Invalid DiffKV v2 cache input");
  TORCH_CHECK(seq_start_loc.dim() == 1, "Invalid DiffKV v2 cache input");
  TORCH_CHECK(compress_config_tables.dim() == 2,
              "Invalid DiffKV v2 cache input");
  TORCH_CHECK(compress_config_tables.size(1) == 2,
              "Invalid DiffKV v2 cache input");

  TORCH_CHECK(key.stride(0) == value.stride(0),
              "Invalid DiffKV v2 cache input");
  const int kv_stride = key.stride(0);
  const int num_kv_heads = key.size(1);
  const int head_size = key.size(2);

  const int num_seqs = score.size(0);
  const int num_heads = score.size(1);
  const int max_prompt_len = score.size(2);
  const int num_layers = block_tables.size(1);
  const int max_num_blocks_per_seq = block_tables.size(3);
  const int unified_page_size = kv_cache.size(1);
  TORCH_CHECK(unified_page_size == 3392, "Invalid DiffKV v2 cache input");

  const int max_prompt_len_power2 = POWER2_ROUND_UP_HOST(max_prompt_len);
  const int shared_mem_size =
      max_prompt_len_power2 * sizeof(float)  // s_mean_scores
      + max_prompt_len_power2 * sizeof(int)  // s_positions
      + max_prompt_len * sizeof(float);      // s_scores
  // + max_prompt_len * sizeof(char);         // s_keep

  size_t max_shared_mem_size =
      at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;
  TORCH_CHECK(shared_mem_size < max_shared_mem_size,
              "Shared memory usage exceeds the limit: ", shared_mem_size, " > ",
              max_shared_mem_size);

  dim3 grid(num_kv_heads, num_seqs);
  dim3 block(NUM_THREADS_PROMPT);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (key.dtype() == at::ScalarType::Half) {
    switch (head_size) {
      // LAUNCH_CACHE_KERNEL_PROMPT_PHASE(uint16_t);
      case 128:
        CALL_PROMPT_PHASE_CACHE_LAUNCHER_QUANT_CONFIG(uint16_t, 128);
        break;
      default:
        TORCH_CHECK(false, "Unsupported head_size: ", head_size);
        break;
    }
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", key.dtype());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define LAUNCH_CACHE_KERNEL_DECODE_PHASE(                                     \
    T, HEAD_SIZE, BITS_K_HIGH, BITS_V_HIGH, BITS_K_LOW, BITS_V_LOW,           \
    CHUNKS_K_HIGH, CHUNKS_V_HIGH, CHUNKS_K_LOW, CHUNKS_V_LOW,                 \
    NUM_TOKENS_PER_PAGE_HIGH, NUM_TOKENS_PER_PAGE_LOW, THREAD_GROUP_SIZE_V)   \
  diffkv_v2::compress_and_append_cache_decode_phase_kernel<                   \
      T, HEAD_SIZE, BITS_K_HIGH, BITS_V_HIGH, BITS_K_LOW, BITS_V_LOW,         \
      CHUNKS_K_HIGH, CHUNKS_V_HIGH, CHUNKS_K_LOW, CHUNKS_V_LOW,               \
      NUM_TOKENS_PER_PAGE_HIGH, NUM_TOKENS_PER_PAGE_LOW, THREAD_GROUP_SIZE_V> \
      <<<grid, block, shared_mem_size, stream>>>(                             \
          reinterpret_cast<T*>(key.data_ptr()),                               \
          reinterpret_cast<T*>(value.data_ptr()), positions.data_ptr<int>(),  \
          reinterpret_cast<uint16_t*>(kv_cache.data_ptr()),                   \
          slot_ids.data_ptr<int>(), block_tables.data_ptr<int>(),             \
          kv_len_tables.data_ptr<int>(),                                      \
          compress_config_tables.data_ptr<float>(), max_context_len,          \
          kv_buffer_size, layer_idx, kv_stride, num_layers, num_kv_heads,     \
          unified_page_size, max_num_blocks_per_seq);

#define CALL_DECODE_PHASE_CACHE_LAUNCHER_QUANT_CONFIG(T, HEAD_SIZE)            \
  TORCH_CHECK((quant_config == std::vector<int>{8, 4, 4, 2, 1, 1, 1, 1}),      \
              "DiffKV v2 requires K8V4/K4V2, one chunk");                      \
  TORCH_CHECK(num_tokens_per_page_high == 32 && num_tokens_per_page_low == 60, \
              "DiffKV v2 requires pages 32/60");                               \
  LAUNCH_CACHE_KERNEL_DECODE_PHASE(T, HEAD_SIZE, 8, 4, 4, 2, 1, 1, 1, 1, 32,   \
                                   60, 8)

void compress_and_append_cache_decode_phase(
    torch::Tensor& key,            // [num_seqs, num_kv_heads, head_size]
    torch::Tensor& value,          // [num_seqs, num_kv_heads, head_size]
    torch::Tensor& positions,      // [num_seqs]
    torch::Tensor& kv_cache,       // [num_blocks, unified_page_size]
    torch::Tensor& slot_ids,       // [num_seqs]
    torch::Tensor& block_tables,   // [num_slots, num_layers, num_kv_heads,
                                   // max_num_blocks_per_seq]
    torch::Tensor& kv_len_tables,  // [num_slots, num_layers, num_kv_heads, 2]
    torch::Tensor&
        compress_config_tables,  // [num_slots, 2] (prune_ratio, quant_ratio)
    const int max_context_len, const int kv_buffer_size, const int layer_idx,
    const int num_bits_k_high, const int num_bits_v_high,
    const int num_bits_k_low, const int num_bits_v_low,
    const int num_chunks_k_high, const int num_chunks_v_high,
    const int num_chunks_k_low, const int num_chunks_v_low,
    const int k_vec_size, const int v_vec_size,
    const int num_tokens_per_page_high, const int num_tokens_per_page_low) {
  diffkv_v2::check_cache(key, value, kv_cache, slot_ids, block_tables,
                         kv_len_tables, compress_config_tables, layer_idx);
  const c10::cuda::CUDAGuard device_guard(key.device());
  TORCH_CHECK(kv_buffer_size > 0, "KV buffer size must be positive");
  diffkv_v2::check_tensor(positions, key, at::kInt, "positions");
  TORCH_CHECK(positions.dim() == 1 && positions.numel() == slot_ids.numel() &&
                  key.size(0) == slot_ids.numel(),
              "Decode expects one token per request");
  TORCH_CHECK(max_context_len > 0, "max_context_len must be positive");

  TORCH_CHECK(k_vec_size == K_VEC_SIZE, "k_vec_size should be ", K_VEC_SIZE);
  TORCH_CHECK(v_vec_size == V_VEC_SIZE, "v_vec_size should be ", V_VEC_SIZE);

  std::vector<int> quant_config = {
      num_bits_k_high,   num_bits_v_high,   num_bits_k_low,   num_bits_v_low,
      num_chunks_k_high, num_chunks_v_high, num_chunks_k_low, num_chunks_v_low};
  int lowest_bits = *std::min_element(quant_config.begin(), quant_config.end());
  int thread_group_size_v = get_thread_group_size_v(lowest_bits);

  TORCH_CHECK(key.dim() == 3, "Invalid DiffKV v2 cache input");
  TORCH_CHECK(value.dim() == 3, "Invalid DiffKV v2 cache input");
  TORCH_CHECK(positions.dim() == 1, "Invalid DiffKV v2 cache input");
  TORCH_CHECK(kv_cache.dim() == 2, "Invalid DiffKV v2 cache input");
  TORCH_CHECK(slot_ids.dim() == 1, "Invalid DiffKV v2 cache input");
  TORCH_CHECK(block_tables.dim() == 4, "Invalid DiffKV v2 cache input");
  TORCH_CHECK(kv_len_tables.dim() == 4, "Invalid DiffKV v2 cache input");
  TORCH_CHECK(kv_len_tables.size(3) == 2, "Invalid DiffKV v2 cache input");
  TORCH_CHECK(compress_config_tables.dim() == 2,
              "Invalid DiffKV v2 cache input");
  TORCH_CHECK(compress_config_tables.size(1) == 2,
              "Invalid DiffKV v2 cache input");

  TORCH_CHECK(key.stride(0) == value.stride(0),
              "Invalid DiffKV v2 cache input");
  const int kv_stride = key.stride(0);

  const int num_seqs = key.size(0);
  const int num_kv_heads = key.size(1);
  const int head_size = key.size(2);
  const int num_layers = block_tables.size(1);
  const int max_num_blocks_per_seq = block_tables.size(3);
  const int unified_page_size = kv_cache.size(1);
  TORCH_CHECK(unified_page_size == 3392, "Invalid DiffKV v2 cache input");

  // const int shared_mem_size = max_context_len * sizeof(float) +
  // max_context_len * sizeof(int);
  const int shared_mem_size = 0;

  dim3 grid(num_kv_heads, num_seqs);
  dim3 block(NUM_THREADS_DECODE);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (key.dtype() == at::ScalarType::Half) {
    // LAUNCH_CACHE_KERNEL_DECODE_PHASE(uint16_t);
    switch (head_size) {
      // LAUNCH_CACHE_KERNEL_PROMPT_PHASE(uint16_t);
      case 128:
        CALL_DECODE_PHASE_CACHE_LAUNCHER_QUANT_CONFIG(uint16_t, 128);
        break;
      default:
        TORCH_CHECK(false, "Unsupported head_size: ", head_size);
        break;
    }

    // // Synchronize device and check for errors
    // cudaDeviceSynchronize();
    // cudaError_t cudaStatus = cudaGetLastError();
    // if (cudaStatus != cudaSuccess) {
    //   fprintf(stderr, "compress_and_append_cache_decode_phase_kernel error:
    //   %s\n", cudaGetErrorString(cudaStatus)); fprintf(stderr, "key shape:
    //   (%d, %d, %d)\n", num_seqs, num_kv_heads, head_size); fprintf(stderr,
    //   "value shape: (%d, %d, %d)\n", num_seqs, num_kv_heads, head_size);
    //   abort();
    // }
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", key.dtype());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
