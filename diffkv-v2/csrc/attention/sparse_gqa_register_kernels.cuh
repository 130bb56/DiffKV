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

// Include inside diffkv_v2::gqa_reuse, after the common constants are declared.
// kPartitionSize must also match sparse_paged_attention_gqa_reduce.
// Stage probabilities while reusing decoded K/V directly in registers.

namespace register_gqa_detail {

constexpr int kDimensionThreads = 8;
constexpr int kDimensionsPerThread = 128 / kDimensionThreads;
constexpr int kGroupsPerWarp = 32 / kDimensionThreads;
constexpr int kWarps = kThreads / 32;
constexpr int kValueGroups = kWarps * kGroupsPerWarp;

union SharedWorkspace {
  float probabilities[4][kPartitionSize];
  float output_partials[kValueGroups][128];
};

// Both dimension-sliced stores and contiguous final-output reads use all
// 32 banks. The XOR transform is bijective within each 128-element row.
__device__ __forceinline__ int output_column(int group, int dimension) {
  return dimension ^ (dimension >> 4) ^ ((group & 3) << 3);
}

__device__ __forceinline__ float maximum(float value) {
#pragma unroll
  for (int delta = 16; delta; delta /= 2) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, delta));
  }
  return value;
}

__device__ __forceinline__ float sum(float value) {
#pragma unroll
  for (int delta = 16; delta; delta /= 2) {
    value += __shfl_xor_sync(0xffffffff, value, delta);
  }
  return value;
}

template <int PageTokens, int KeyBits>
__device__ __forceinline__ void compute_keys(
    const uint16_t* kv_cache, const int* block_table, bool reverse_pages,
    int page_stride, int region_begin, int region_end, int logical_offset,
    int first_partition_token, int context_length, float scale,
    const float* alibi_slopes, int first_head,
    const float (&query_values)[4][kDimensionsPerThread],
    SharedWorkspace& workspace, float (&local_maxima)[4]) {
  constexpr int kPack = 16 / KeyBits;
  constexpr int kPacks = 128 / kPack;
  constexpr int kMetaBase = DIVIDE_ROUND_UP(PageTokens * kPacks * 2, 32) * 16;
  constexpr int kVectors = kDimensionsPerThread / (kPack * K_VEC_SIZE);
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  // Four token groups, each with eight strided dimension threads.
  const int token_group = lane % kGroupsPerWarp;
  const int dimension_thread = lane / kGroupsPerWarp;
  const int original_key_group = dimension_thread / 2;
  const int first_vector = (dimension_thread % 2) * kVectors;

  for (int page = region_begin / PageTokens + warp;
       page < DIVIDE_ROUND_UP(region_end, PageTokens); page += kWarps) {
    const int64_t physical =
        reverse_pages ? block_table[-page] : block_table[page];
    const int64_t page_base = physical * page_stride;
    for (int token = token_group; token < PageTokens; token += kGroupsPerWarp) {
      const int region_token = page * PageTokens + token;
      const bool valid =
          region_token >= region_begin && region_token < region_end;
      float products[4] = {0.f, 0.f, 0.f, 0.f};
      if (valid) {
        const quant_meta_type meta = *reinterpret_cast<const quant_meta_type*>(
            kv_cache + page_base + kMetaBase + token * 2);
        const float scale_key = to_float(meta.scale);
        const float zero_key = to_float(meta.zero_point);
#pragma unroll
        for (int vector = 0; vector < kVectors; ++vector) {
          const int offset = (((first_vector + vector) * PageTokens + token) *
                                  THREAD_GROUP_SIZE_K +
                              original_key_group) *
                             K_VEC_SIZE;
          const k_vec_type packed = *reinterpret_cast<const k_vec_type*>(
              kv_cache + page_base + offset);
#pragma unroll
          for (int packed_element = 0; packed_element < K_VEC_SIZE;
               ++packed_element) {
            float decoded[kPack];
            unpack_and_dequant(packed.data[packed_element], KeyBits, scale_key,
                               zero_key, decoded);
#pragma unroll
            for (int element = 0; element < kPack; ++element) {
              const int dimension =
                  (vector * K_VEC_SIZE + packed_element) * kPack + element;
#pragma unroll
              for (int query = 0; query < 4; ++query) {
                products[query] = fmaf(query_values[query][dimension],
                                       decoded[element], products[query]);
              }
            }
          }
        }
      }
// Every lane participates, including groups beyond a partial page.
#pragma unroll
      for (int delta = kGroupsPerWarp; delta < 32; delta *= 2) {
        float shuffled[4];
#pragma unroll
        for (int query = 0; query < 4; ++query) {
          shuffled[query] =
              __shfl_down_sync(0xffffffff, products[query], delta);
        }
#pragma unroll
        for (int query = 0; query < 4; ++query) {
          products[query] += shuffled[query];
        }
      }
#pragma unroll
      for (int query = 0; query < 4; ++query) {
        if (dimension_thread == 0 && valid) {
          const int logical_token = logical_offset + region_token;
          const float slope =
              alibi_slopes == nullptr ? 0.f : alibi_slopes[first_head + query];
          const float logit =
              products[query] * scale +
              (slope != 0.f ? slope * (logical_token - context_length + 1)
                            : 0.f);
          workspace
              .probabilities[query][logical_token - first_partition_token] =
              logit;
          local_maxima[query] = fmaxf(local_maxima[query], logit);
        }
      }
    }
  }
}

template <int PageTokens, int KeyBits, int ValueBits>
__device__ __forceinline__ void compute_values(
    const uint16_t* kv_cache, const int* block_table, bool reverse_pages,
    int page_stride, int region_begin, int region_end, int logical_offset,
    int first_partition_token, const SharedWorkspace& workspace,
    quant_meta_type (&value_metadata)[kWarps][60],
    float (&accumulators)[4][kDimensionsPerThread]) {
  constexpr int kKeyPacks = 128 * KeyBits / 16;
  constexpr int kValuePack = 16 / ValueBits;
  constexpr int kValuePacks = 128 / kValuePack;
  constexpr int kPaddedTokens =
      DIVIDE_ROUND_UP(PageTokens, V_VEC_SIZE) * V_VEC_SIZE;
  constexpr int kKeyMetaBase =
      DIVIDE_ROUND_UP(PageTokens * kKeyPacks * 2, 32) * 16;
  constexpr int kValueBase =
      DIVIDE_ROUND_UP(kKeyMetaBase * 2 + PageTokens * 4, 128) * 64;
  constexpr int kValueMetaBase =
      kValueBase + DIVIDE_ROUND_UP(kPaddedTokens * kValuePacks * 2, 32) * 16;
  constexpr int kPacksPerThread = kDimensionsPerThread / kValuePack;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int token_group = lane / kDimensionThreads;
  const int dimension_thread = lane % kDimensionThreads;

  for (int page = region_begin / PageTokens + warp;
       page < DIVIDE_ROUND_UP(region_end, PageTokens); page += kWarps) {
    const int64_t physical =
        reverse_pages ? block_table[-page] : block_table[page];
    const int64_t page_base = physical * page_stride;
    // Each warp owns its metadata row. No block-wide tile barrier is needed.
    for (int token = lane; token < PageTokens; token += 32) {
      value_metadata[warp][token] = *reinterpret_cast<const quant_meta_type*>(
          kv_cache + page_base + kValueMetaBase + token * 2);
    }
    __syncwarp();
    for (int vector = token_group;
         vector < DIVIDE_ROUND_UP(PageTokens, V_VEC_SIZE);
         vector += kGroupsPerWarp) {
      const int first_token = vector * V_VEC_SIZE;
      // Keep each token pair's payload packed: four 32-bit words for V4,
      // two for V2. Each physical payload word is still fetched once.
      v_vec_type packed_values[kPacksPerThread];
#pragma unroll
      for (int pack = 0; pack < kPacksPerThread; ++pack) {
        const int offset =
            pack * kPaddedTokens * kDimensionThreads +
            (vector * kDimensionThreads + dimension_thread) * V_VEC_SIZE;
        packed_values[pack] = *reinterpret_cast<const v_vec_type*>(
            kv_cache + page_base + kValueBase + offset);
      }
#pragma unroll
      for (int element = 0; element < V_VEC_SIZE; ++element) {
        const int token = first_token + element;
        const int region_token = page * PageTokens + token;
        if (token < PageTokens && region_token >= region_begin &&
            region_token < region_end) {
          // These values depend on the token, not on its output pack. Load and
          // convert them once before visiting the four/two packed dim slices.
          const quant_meta_type meta = value_metadata[warp][token];
          const float value_scale = to_float(meta.scale);
          const float value_zero = to_float(meta.zero_point);
          const int weight_offset =
              logical_offset + region_token - first_partition_token;
          float weights[4];
#pragma unroll
          for (int query = 0; query < 4; ++query) {
            weights[query] = workspace.probabilities[query][weight_offset];
          }
#pragma unroll
          for (int pack = 0; pack < kPacksPerThread; ++pack) {
            float decoded[kValuePack];
            unpack_and_dequant(packed_values[pack].data[element], ValueBits,
                               value_scale, value_zero, decoded);
#pragma unroll
            for (int query = 0; query < 4; ++query) {
              const float weight = weights[query];
#pragma unroll
              for (int dimension = 0; dimension < kValuePack; ++dimension) {
                const int output_dimension = pack * kValuePack + dimension;
                accumulators[query][output_dimension] =
                    fmaf(weight, decoded[dimension],
                         accumulators[query][output_dimension]);
              }
            }
          }
        }
      }
    }
    // Keep a faster subgroup from overwriting metadata for a slower subgroup.
    __syncwarp();
  }
}

}  // namespace register_gqa_detail

template <typename scalar_t>
__global__ __launch_bounds__(128, 4) void register_gqa_attention_kernel(
    const int* slot_ids, scalar_t* tmp_out, float* tmp_scores, float* exp_sums,
    float* max_logits, const scalar_t* queries, const uint16_t* kv_cache,
    const int* block_tables, const int* kv_len_tables, int layer_idx,
    int num_layers, int num_kv_heads, int max_blocks, int unified_page_size,
    int max_context_len, int query_stride, float scale,
    const float* alibi_slopes) {
  using namespace register_gqa_detail;
  static_assert(kThreads == 128 && kHeadSize == 128 && kQueriesPerKv == 4);
  static_assert(kPartitionSize > 0);
  const int sequence = blockIdx.y;
  const int kv_head = blockIdx.x;
  const int partition = blockIdx.z;
  const int partition_stride = gridDim.z;
  const int num_heads = num_kv_heads * 4;
  const int first_head = kv_head * 4;
  const int table_offset =
      (slot_ids[sequence] * num_layers + layer_idx) * num_kv_heads + kv_head;
  const int high_length = kv_len_tables[table_offset * 2];
  const int low_length = kv_len_tables[table_offset * 2 + 1];
  const int context_length = high_length + low_length;
  int high_begin = 0, high_end = high_length;
  int low_begin = 0, low_end = low_length;
  if (context_length > kPartitionSize) {
    const int high_partitions = DIVIDE_ROUND_UP(high_length, kPartitionSize);
    const int low_partitions = DIVIDE_ROUND_UP(low_length, kPartitionSize);
    if (partition >= high_partitions + low_partitions) {
      return;
    }
    if (partition < high_partitions) {
      high_begin = partition * kPartitionSize;
      high_end = MIN(high_begin + kPartitionSize, high_length);
      low_end = 0;
    } else {
      high_end = 0;
      low_begin = (partition - high_partitions) * kPartitionSize;
      low_end = MIN(low_begin + kPartitionSize, low_length);
    }
  } else if (partition != 0) {
    return;
  }
  const int first_token =
      high_end > high_begin ? high_begin : high_length + low_begin;
  const int token_count = high_end - high_begin + low_end - low_begin;
  const int* block_table = block_tables + table_offset * max_blocks;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  __shared__ SharedWorkspace workspace;
  __shared__ float warp_maxima[4][kWarps];
  __shared__ float warp_sums[4][kWarps];
  __shared__ quant_meta_type value_metadata[kWarps][60];

  // End this scope before the four-query PV accumulator becomes live.
  {
    float local_maxima[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
    {
      const int dimension_thread = lane / kGroupsPerWarp;
      float query_values[4][kDimensionsPerThread];
#pragma unroll
      for (int query = 0; query < 4; ++query) {
        const scalar_t* source = queries + sequence * query_stride +
                                 (first_head + query) * 128 +
                                 dimension_thread * kDimensionsPerThread;
#pragma unroll
        for (int dimension = 0; dimension < kDimensionsPerThread; ++dimension) {
          query_values[query][dimension] = to_float(source[dimension]);
        }
      }
      compute_keys<32, 8>(kv_cache, block_table, false, unified_page_size,
                          high_begin, high_end, 0, first_token, context_length,
                          scale, alibi_slopes, first_head, query_values,
                          workspace, local_maxima);
      compute_keys<60, 4>(kv_cache, block_table + max_blocks - 1, true,
                          unified_page_size, low_begin, low_end, high_length,
                          first_token, context_length, scale, alibi_slopes,
                          first_head, query_values, workspace, local_maxima);
    }
// Publish all four query maxima together. This also waits for every K
// logit write, replacing the separate barrier before the softmax loop.
#pragma unroll
    for (int query = 0; query < 4; ++query) {
      const float warp_max = maximum(local_maxima[query]);
      if (lane == 0) {
        warp_maxima[query][warp] = warp_max;
      }
    }
    __syncthreads();
#pragma unroll
    for (int query = 0; query < 4; ++query) {
      const float maximum_logit =
          maximum(lane < kWarps ? warp_maxima[query][lane] : -FLT_MAX);
      float local_sum = 0.f;
      for (int token = threadIdx.x; token < token_count; token += kThreads) {
        const float value =
            __expf(workspace.probabilities[query][token] - maximum_logit);
        workspace.probabilities[query][token] = value;
        local_sum += value;
      }
      local_sum = sum(local_sum);
      if (lane == 0) {
        warp_sums[query][warp] = local_sum;
      }
      if (threadIdx.x == 0) {
        const int index =
            (sequence * num_heads + first_head + query) * partition_stride +
            partition;
        max_logits[index] = maximum_logit;
      }
    }
    // Each query has a separate row, so all sums can share this barrier.
    __syncthreads();
#pragma unroll
    for (int query = 0; query < 4; ++query) {
      const float exponential_sum =
          sum(lane < kWarps ? warp_sums[query][lane] : 0.f);
      const float inverse_sum = __fdividef(1.f, exponential_sum + 1e-6f);
      for (int token = threadIdx.x; token < token_count; token += kThreads) {
        const float probability =
            workspace.probabilities[query][token] * inverse_sum;
        workspace.probabilities[query][token] = probability;
        tmp_scores[(sequence * num_heads + first_head + query) *
                       max_context_len +
                   first_token + token] = probability;
      }
      if (threadIdx.x == 0) {
        const int index =
            (sequence * num_heads + first_head + query) * partition_stride +
            partition;
        exp_sums[index] = exponential_sum;
      }
    }
  }
  __syncthreads();
  float accumulators[4][kDimensionsPerThread];
#pragma unroll
  for (int query = 0; query < 4; ++query) {
#pragma unroll
    for (int dimension = 0; dimension < kDimensionsPerThread; ++dimension) {
      accumulators[query][dimension] = 0.f;
    }
  }
  compute_values<32, 8, 4>(kv_cache, block_table, false, unified_page_size,
                           high_begin, high_end, 0, first_token, workspace,
                           value_metadata, accumulators);
  compute_values<60, 4, 2>(kv_cache, block_table + max_blocks - 1, true,
                           unified_page_size, low_begin, low_end, high_length,
                           first_token, workspace, value_metadata,
                           accumulators);

  // Probabilities are dead only once every warp has completed PV.
  __syncthreads();
  const int value_group = warp * kGroupsPerWarp + lane / kDimensionThreads;
  const int dimension_thread = lane % kDimensionThreads;
#pragma unroll
  for (int query = 0; query < 4; ++query) {
#pragma unroll
    for (int dimension = 0; dimension < kDimensionsPerThread; ++dimension) {
      workspace.output_partials[value_group][output_column(
          value_group, dimension_thread * kDimensionsPerThread + dimension)] =
          accumulators[query][dimension];
    }
    __syncthreads();
    if (threadIdx.x < 128) {
      float value = 0.f;
#pragma unroll
      for (int group = 0; group < kValueGroups; ++group) {
        value +=
            workspace.output_partials[group][output_column(group, threadIdx.x)];
      }
      const int index =
          ((sequence * num_heads + first_head + query) * partition_stride +
           partition) *
              128 +
          threadIdx.x;
      from_float(tmp_out[index], value);
    }
    __syncthreads();
  }
}
