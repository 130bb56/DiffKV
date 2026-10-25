/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// Include inside vllm::gqa_reuse after kPartitionSize.
// Launch grid=(num_kv_heads, num_seqs, score_splits), block=256, dynamic shared
// bytes=4*max_num_partitions*sizeof(float). All CTAs repeat the small partition
// statistics, but own disjoint token ranges. Only score tile zero writes out.
// tmp_scores remains partition-normalized scratch; the next main overwrites
// active entries before use. Only out and retained-token cache scores are final.

namespace split_score_reduce_detail {

constexpr int kThreads = 256;
constexpr int kHeadSize = 128;
constexpr int kQueries = 4;

__device__ __forceinline__ float warp_maximum(float value) {
  #pragma unroll
  for (int mask = 16; mask; mask /= 2) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, mask));
  }
  return value;
}

__device__ __forceinline__ float warp_total(float value) {
  #pragma unroll
  for (int mask = 16; mask; mask /= 2) {
    value += __shfl_xor_sync(0xffffffff, value, mask);
  }
  return value;
}

template<int PageTokens, int KeyBits, int ValueBits>
struct ScoreLayout {
  static constexpr int kKeyPacks = kHeadSize * KeyBits / 16;
  static constexpr int kValuePacks = kHeadSize * ValueBits / 16;
  static constexpr int kKeyMetaBase =
      DIVIDE_ROUND_UP(PageTokens * kKeyPacks * 2, 32) * 16;
  static constexpr int kValueBase =
      DIVIDE_ROUND_UP(kKeyMetaBase * 2 + PageTokens * 4, 128) * 64;
  static constexpr int kPaddedTokens =
      DIVIDE_ROUND_UP(PageTokens, V_VEC_SIZE) * V_VEC_SIZE;
  static constexpr int kValueMetaBase = kValueBase
      + DIVIDE_ROUND_UP(kPaddedTokens * kValuePacks * 2, 32) * 16;
  static constexpr int kScoreBase = kValueMetaBase
      + DIVIDE_ROUND_UP(PageTokens * 4, 32) * 16;
};

__device__ __forceinline__ void update_score(
    uint16_t* kv_cache, const int* high_table, const int* low_table,
    int page_stride, int high_length, int token, float score) {
  int64_t offset;
  if (token < high_length) {
    const int64_t physical = high_table[token / 32];
    offset = physical * page_stride + ScoreLayout<32, 8, 4>::kScoreBase
        + (token % 32) * 2;
  } else {
    const int low_token = token - high_length;
    const int64_t physical = low_table[-(low_token / 60)];
    offset = physical * page_stride + ScoreLayout<60, 4, 2>::kScoreBase
        + (low_token % 60) * 2;
  }
  *reinterpret_cast<float*>(kv_cache + offset) += score;
}

}  // namespace split_score_reduce_detail

template<typename scalar_t>
__global__ __launch_bounds__(256) void register_gqa_reduce_kernel(
    scalar_t* __restrict__ out,
    const float* __restrict__ exp_sums,
    const float* __restrict__ max_logits,
    const scalar_t* __restrict__ tmp_out,
    float* __restrict__ tmp_scores,
    const int* __restrict__ block_tables,
    const int* __restrict__ kv_len_tables,
    const int* __restrict__ slot_ids,
    uint16_t* __restrict__ kv_cache,
    const int unified_page_size,
    const int layer_idx,
    const int num_layers,
    const int num_kv_heads,
    const int max_context_len,
    const int max_num_blocks_per_seq,
    const int max_num_partitions) {
  static_assert(kPartitionSize > 0);
  constexpr int kReduceThreads = split_score_reduce_detail::kThreads;
  const int sequence = blockIdx.y;
  const int kv_head = blockIdx.x;
  const int tile = blockIdx.z;
  const int num_heads = num_kv_heads * 4;
  const int first_query = sequence * num_heads + kv_head * 4;
  const int table_index =
      (slot_ids[sequence] * num_layers + layer_idx) * num_kv_heads + kv_head;
  const int* high_table = block_tables + table_index * max_num_blocks_per_seq;
  const int* low_table = high_table + max_num_blocks_per_seq - 1;
  const int high_length = kv_len_tables[table_index * 2];
  const int low_length = kv_len_tables[table_index * 2 + 1];
  const int context_length = high_length + low_length;

  // Warp-aligned contiguous ranges keep every probability load coalesced.
  // An idle tile can return before any CTA barrier. Tile zero still copies
  // the output for an empty context, matching the original scratch contract.
  const int score_tokens_per_tile =
      DIVIDE_ROUND_UP(context_length, static_cast<int>(gridDim.z) * 32) * 32;
  const int score_begin = tile * score_tokens_per_tile;
  const int score_end = MIN(context_length, score_begin + score_tokens_per_tile);
  if (tile != 0 && score_begin >= context_length) {
    return;
  }

  if (context_length <= kPartitionSize) {
    if (tile == 0) {
      #pragma unroll
      for (int element = threadIdx.x; element < 4 * 128;
           element += kReduceThreads) {
        const int query_index = first_query + element / 128;
        const int dimension = element % 128;
        out[query_index * 128 + dimension] =
            tmp_out[query_index * max_num_partitions * 128 + dimension];
      }
    }
    for (int token = score_begin + threadIdx.x; token < score_end;
         token += kReduceThreads) {
      // Issue independent query loads before consuming any probability.
      const float p0 = tmp_scores[(first_query + 0) * max_context_len + token];
      const float p1 = tmp_scores[(first_query + 1) * max_context_len + token];
      const float p2 = tmp_scores[(first_query + 2) * max_context_len + token];
      const float p3 = tmp_scores[(first_query + 3) * max_context_len + token];
      float score = MAX(0.f, p0);
      score = MAX(score, p1);
      score = MAX(score, p2);
      score = MAX(score, p3);
      split_score_reduce_detail::update_score(kv_cache, high_table, low_table,
          unified_page_size, high_length, token, score);
    }
    return;
  }

  const int high_partitions = DIVIDE_ROUND_UP(high_length, kPartitionSize);
  const int num_partitions = high_partitions
      + DIVIDE_ROUND_UP(low_length, kPartitionSize);
  assert(num_partitions > 1 && num_partitions <= max_num_partitions);
  extern __shared__ float split_score_factors[];
  __shared__ float inverse_sums[4];
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (warp < 4) {
    const int query_index = first_query + warp;
    const float* query_maxima = max_logits + query_index * max_num_partitions;
    const float* query_sums = exp_sums + query_index * max_num_partitions;
    float* factors = split_score_factors + warp * max_num_partitions;
    float total = 0.f;
    if (num_partitions <= 32) {
      // The common P240/C3072 case has 14 partitions. Hold its maximum and
      // sum in registers across the warp reduction instead of loading twice.
      const float local_maximum = lane < num_partitions
          ? query_maxima[lane] : -FLT_MAX;
      const float local_sum = lane < num_partitions ? query_sums[lane] : 0.f;
      const float maximum = split_score_reduce_detail::warp_maximum(local_maximum);
      const float factor = expf(local_maximum - maximum);
      total = local_sum * factor;
      if (lane < num_partitions) {
        factors[lane] = (local_sum + 1e-6f) * factor;
      }
    } else {
      float maximum = -FLT_MAX;
      for (int partition = lane; partition < num_partitions; partition += 32) {
        maximum = fmaxf(maximum, query_maxima[partition]);
      }
      maximum = split_score_reduce_detail::warp_maximum(maximum);
      for (int partition = lane; partition < num_partitions; partition += 32) {
        const float factor = expf(query_maxima[partition] - maximum);
        const float sum = query_sums[partition];
        total += sum * factor;
        factors[partition] = (sum + 1e-6f) * factor;
      }
    }
    total = split_score_reduce_detail::warp_total(total);
    if (lane == 0) {
      inverse_sums[warp] = __fdividef(1.f, total + 1e-6f);
    }
  }
  __syncthreads();

  if (tile == 0) {
    // Accumulate two query outputs together. Their independent loads and
    // FMAs provide ILP while preserving ascending partitions and v*r*inv.
    const int query0 = threadIdx.x / 128;
    const int query1 = query0 + 2;
    const int dimension = threadIdx.x % 128;
    const scalar_t* source0 = tmp_out
        + (first_query + query0) * max_num_partitions * 128 + dimension;
    const scalar_t* source1 = tmp_out
        + (first_query + query1) * max_num_partitions * 128 + dimension;
    const float* factors0 = split_score_factors + query0 * max_num_partitions;
    const float* factors1 = split_score_factors + query1 * max_num_partitions;
    const float inverse0 = inverse_sums[query0];
    const float inverse1 = inverse_sums[query1];
    float output0 = 0.f;
    float output1 = 0.f;
    for (int partition = 0; partition < num_partitions; ++partition) {
      const float value0 = to_float(source0[partition * 128]);
      const float value1 = to_float(source1[partition * 128]);
      output0 += value0 * factors0[partition] * inverse0;
      output1 += value1 * factors1[partition] * inverse1;
    }
    from_float(out[(first_query + query0) * 128 + dimension], output0);
    from_float(out[(first_query + query1) * 128 + dimension], output1);
  }

  for (int token = score_begin + threadIdx.x; token < score_end;
       token += kReduceThreads) {
    const int partition = token < high_length
        ? token / kPartitionSize
        : high_partitions + (token - high_length) / kPartitionSize;
    const float p0 = tmp_scores[(first_query + 0) * max_context_len + token];
    const float p1 = tmp_scores[(first_query + 1) * max_context_len + token];
    const float p2 = tmp_scores[(first_query + 2) * max_context_len + token];
    const float p3 = tmp_scores[(first_query + 3) * max_context_len + token];
    const float s0 = p0 * (split_score_factors[0 * max_num_partitions + partition]
                           * inverse_sums[0]);
    const float s1 = p1 * (split_score_factors[1 * max_num_partitions + partition]
                           * inverse_sums[1]);
    const float s2 = p2 * (split_score_factors[2 * max_num_partitions + partition]
                           * inverse_sums[2]);
    const float s3 = p3 * (split_score_factors[3 * max_num_partitions + partition]
                           * inverse_sums[3]);
    float score = MAX(0.f, s0);
    score = MAX(score, s1);
    score = MAX(score, s2);
    score = MAX(score, s3);
    split_score_reduce_detail::update_score(kv_cache, high_table, low_table,
        unified_page_size, high_length, token, score);
  }
}
