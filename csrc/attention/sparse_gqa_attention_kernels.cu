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

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "../ops.h"
#include "attention_dtypes.h"
#include "cache_utils.h"
#include "../quantization/kv_cache_quant/kv_cache_quant.h"

#include <cfloat>

namespace vllm {
namespace gqa_reuse {

constexpr int kHeadSize = 128;
constexpr int kQueriesPerKv = 4;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kTileTokens = 32;
constexpr int kPartitionSize = 5760;
constexpr int kQueryThreads = kThreads / kQueriesPerKv;

struct alignas(16) TileWorkspace {
  // K is transposed for the four-thread, strided QK groups. The token axis
  // is also XOR-swizzled by the packed element to avoid staging-store conflicts.
  float keys[kHeadSize / THREAD_GROUP_SIZE_K][kTileTokens][THREAD_GROUP_SIZE_K];
  float values[kTileTokens][kHeadSize];
  float weights[kQueriesPerKv][kTileTokens];
  float warp_max[kThreads / kWarpSize];
  float warp_sum[kThreads / kWarpSize];
};

__device__ __forceinline__ int value_column(int token, int dimension) {
  // Preserve a bank permutation for contiguous 32-dimension PV reads while
  // distributing packed V staging stores across all 32 shared-memory banks.
  return dimension ^ ((dimension & 32) >> 3) ^ ((dimension & 64) >> 6)
      ^ ((token & 1) << 3) ^ (token & 2);
}

__device__ __forceinline__ void store_value_quad(
    float* destination, float first, float second, float third, float fourth,
    int low_xor) {
  // Permute four registers, avoiding a dynamically indexed local array. The
  // swizzle preserves each quad's aligned 16-byte physical destination.
  if (low_xor & 1) {
    float temporary = first;
    first = second;
    second = temporary;
    temporary = third;
    third = fourth;
    fourth = temporary;
  }
  if (low_xor & 2) {
    float temporary = first;
    first = third;
    third = temporary;
    temporary = second;
    second = fourth;
    fourth = temporary;
  }
  *reinterpret_cast<float4*>(destination) = make_float4(first, second, third, fourth);
}

template<int PageTokens, int KeyBits, int ValueBits>
__device__ __forceinline__ void load_kv_tile(
    const uint16_t* kv_cache, int64_t physical_page, int unified_page_size,
    int token_start, int token_count, TileWorkspace& tile) {
  constexpr int kKeyPack = 16 / KeyBits;
  constexpr int kValuePack = 16 / ValueBits;
  constexpr int kKeyPacks = kHeadSize / kKeyPack;
  constexpr int kValuePacks = kHeadSize / kValuePack;
  constexpr int kKeyVectors = kKeyPacks / (THREAD_GROUP_SIZE_K * K_VEC_SIZE);
  constexpr int kPaddedTokens = DIVIDE_ROUND_UP(PageTokens, V_VEC_SIZE) * V_VEC_SIZE;
  constexpr int kKeyMetaBase = DIVIDE_ROUND_UP(PageTokens * kKeyPacks * 2, 32) * 16;
  constexpr int kValueBase = DIVIDE_ROUND_UP(PageTokens * 4 + kKeyMetaBase * 2, 128) * 64;
  constexpr int kValueMetaBase =
      kValueBase + DIVIDE_ROUND_UP(kPaddedTokens * kValuePacks * 2, 32) * 16;
  const int64_t page_base = physical_page * unified_page_size;

  // Walk the packed K layout in its physical order. Every packed K value is
  // unpacked once, then the decoded tile is consumed by all four queries.
  // A fixed tile capacity keeps the packed-index divisions constant even for
  // the 28-token second tile of a low-precision page and partial final pages.
  for (int packed_idx = threadIdx.x;
       packed_idx < kKeyVectors * kTileTokens * THREAD_GROUP_SIZE_K * K_VEC_SIZE;
       packed_idx += kThreads) {
    const int vector_idx = packed_idx / (kTileTokens * THREAD_GROUP_SIZE_K * K_VEC_SIZE);
    const int inner = packed_idx % (kTileTokens * THREAD_GROUP_SIZE_K * K_VEC_SIZE);
    const int token = inner / (THREAD_GROUP_SIZE_K * K_VEC_SIZE);
    const int chunk = (inner / K_VEC_SIZE) % THREAD_GROUP_SIZE_K;
    const int packed_element = inner % K_VEC_SIZE;
    if (token < token_count) {
      const int page_token = token_start + token;
      const int offset = ((vector_idx * PageTokens + page_token) * THREAD_GROUP_SIZE_K + chunk)
                           * K_VEC_SIZE + packed_element;
      const quant_meta_type metadata = *reinterpret_cast<const quant_meta_type*>(
          kv_cache + page_base + kKeyMetaBase + page_token * 2);
      float decoded[kKeyPack];
      unpack_and_dequant(kv_cache[page_base + offset], KeyBits,
                         to_float(metadata.scale), to_float(metadata.zero_point), decoded);
      #pragma unroll
      for (int i = 0; i < kKeyPack; ++i) {
        tile.keys[(vector_idx * K_VEC_SIZE + packed_element) * kKeyPack + i]
                 [token ^ (packed_element * 2)][chunk] =
            decoded[i];
      }
    }
  }

  // K from the preceding tile is already dead. This barrier also waits for
  // every preceding V reader before any thread overwrites the V buffer.
  __syncthreads();

  // V uses [pack, token-vector, dimension-group, vector-element]. Tile starts
  // are even, including the second tile in each 60-token low-precision page.
  constexpr int kTileTokenVectors = DIVIDE_ROUND_UP(kTileTokens, V_VEC_SIZE);
  constexpr int kValueGroups = 8;
  constexpr int kRowsPerGroup = kHeadSize / kValueGroups;
  constexpr int kPacksPerGroup = kRowsPerGroup / kValuePack;
  for (int packed_idx = threadIdx.x;
       packed_idx < kPacksPerGroup * kTileTokenVectors * kValueGroups * V_VEC_SIZE;
       packed_idx += kThreads) {
    const int packed_element = packed_idx % V_VEC_SIZE;
    const int dimension_group = (packed_idx / V_VEC_SIZE) % kValueGroups;
    const int token_vector = (packed_idx / (V_VEC_SIZE * kValueGroups)) % kTileTokenVectors;
    const int pack = packed_idx / (V_VEC_SIZE * kValueGroups * kTileTokenVectors);
    const int token = token_vector * V_VEC_SIZE + packed_element;
    if (token < token_count) {
      const int page_token = token_start + token;
      const int offset = pack * kPaddedTokens * kValueGroups
          + ((token_start / V_VEC_SIZE + token_vector) * kValueGroups + dimension_group)
              * V_VEC_SIZE + packed_element;
      const quant_meta_type metadata = *reinterpret_cast<const quant_meta_type*>(
          kv_cache + page_base + kValueMetaBase + page_token * 2);
      float decoded[kValuePack];
      unpack_and_dequant(kv_cache[page_base + kValueBase + offset], ValueBits,
                         to_float(metadata.scale), to_float(metadata.zero_point), decoded);
      #pragma unroll
      for (int quad = 0; quad < kValuePack / 4; ++quad) {
        const int dimension = dimension_group * kRowsPerGroup + pack * kValuePack + quad * 4;
        const int column = value_column(token, dimension);
        store_value_quad(&tile.values[token][column & ~3],
            decoded[quad * 4], decoded[quad * 4 + 1],
            decoded[quad * 4 + 2], decoded[quad * 4 + 3], column & 3);
      }
    }
  }
}

__device__ __forceinline__ float warp_max(float value) {
  #pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, mask));
  }
  return value;
}

__device__ __forceinline__ float warp_sum(float value) {
  #pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
    value += __shfl_xor_sync(0xffffffff, value, mask);
  }
  return value;
}

template<int PageTokens, int KeyBits, int ValueBits>
__device__ __forceinline__ void attend_region(
    const uint16_t* kv_cache, const int* block_table, bool reverse_pages,
    int unified_page_size, int region_start, int region_end, int score_offset,
    float scale, float alibi_slope, int context_length,
    const float (&query_values)[kHeadSize / THREAD_GROUP_SIZE_K],
    float* query_scores, TileWorkspace& tile, float& running_max, float& running_sum,
    float& output_first, float& output_second) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int query = threadIdx.x / kQueryThreads;
  const int query_thread = threadIdx.x % kQueryThreads;
  const int query_warp = warp % 2;
  const int token_group = lane % (kWarpSize / THREAD_GROUP_SIZE_K);
  const int dimension_chunk = lane / (kWarpSize / THREAD_GROUP_SIZE_K);

  for (int start = region_start; start < region_end;) {
    const int page = start / PageTokens;
    const int page_start = start % PageTokens;
    const int count = MIN(kTileTokens, MIN(PageTokens - page_start, region_end - start));
    const int64_t physical_page = reverse_pages ? block_table[-page] : block_table[page];
    load_kv_tile<PageTokens, KeyBits, ValueBits>(
        kv_cache, physical_page, unified_page_size, page_start, count, tile);

    float local_max = -FLT_MAX;
    // All warp lanes participate in each shuffle, including partial tiles.
    #pragma unroll
    for (int iteration = 0; iteration < kTileTokens / 16; ++iteration) {
      const int token = query_warp * 8 + token_group + iteration * 16;
      float dot = 0.f;
      if (token < count) {
        #pragma unroll
        for (int i = 0; i < kHeadSize / THREAD_GROUP_SIZE_K; ++i) {
          const int packed_element = (i / (16 / KeyBits)) % K_VEC_SIZE;
          dot = fmaf(query_values[i],
              tile.keys[i][token ^ (packed_element * 2)][dimension_chunk], dot);
        }
      }
      dot += __shfl_down_sync(0xffffffff, dot, 8);
      dot += __shfl_down_sync(0xffffffff, dot, 16);
      if (dimension_chunk == 0 && token < count) {
        const int score_token = score_offset + start + token;
        const float logit = dot * scale + (alibi_slope != 0.f
            ? alibi_slope * (score_token - context_length + 1) : 0.f);
        tile.weights[query][token] = logit;
        query_scores[score_offset + start + token] = logit;
        local_max = fmaxf(local_max, logit);
      }
    }
    local_max = warp_max(local_max);
    if (lane == 0) {
      tile.warp_max[warp] = local_max;
    }
    // Also makes all decoded V writes visible before weighted-value reads.
    __syncthreads();

    // Both warps of a query derive the same online state in registers.
    const float updated_max = fmaxf(running_max,
        fmaxf(tile.warp_max[2 * query], tile.warp_max[2 * query + 1]));
    const float rescale = __expf(running_max - updated_max);
    running_max = updated_max;

    float local_sum = 0.f;
    if (query_thread < count) {
      const float weight = __expf(tile.weights[query][query_thread] - running_max);
      tile.weights[query][query_thread] = weight;
      local_sum = weight;
    }
    local_sum = warp_sum(local_sum);
    if (lane == 0) {
      tile.warp_sum[warp] = local_sum;
    }
    __syncthreads();
    running_sum = running_sum * rescale
        + tile.warp_sum[2 * query] + tile.warp_sum[2 * query + 1];

    output_first *= rescale;
    output_second *= rescale;
    #pragma unroll
    for (int token = 0; token < kTileTokens; ++token) {
      if (token < count) {
        const float weight = tile.weights[query][token];
        output_first = fmaf(weight,
            tile.values[token][value_column(token, query_thread)], output_first);
        output_second = fmaf(weight,
            tile.values[token][value_column(token, query_thread + kQueryThreads)], output_second);
      }
    }
    // The next tile may overwrite K immediately; its loader barrier waits
    // for all four queries to finish V before overwriting that separate buffer.
    start += count;
  }
}

template<typename scalar_t>
__global__ void sparse_gqa_attention_kernel(
    const int* slot_ids, scalar_t* tmp_out, float* tmp_scores,
    float* exp_sums, float* max_logits, const scalar_t* queries,
    const uint16_t* kv_cache, const int* block_tables, const int* kv_len_tables,
    int layer_idx, int num_layers, int num_kv_heads, int max_blocks,
    int unified_page_size, int max_context_len, int query_stride, float scale,
    const float* alibi_slopes) {
  const int sequence = blockIdx.y;
  const int kv_head = blockIdx.x;
  const int partition = blockIdx.z;
  const int partitions_stride = gridDim.z;
  const int num_heads = num_kv_heads * kQueriesPerKv;
  const int slot = slot_ids[sequence];
  const int table_offset = (slot * num_layers + layer_idx) * num_kv_heads + kv_head;
  const int high_length = kv_len_tables[table_offset * 2];
  const int low_length = kv_len_tables[table_offset * 2 + 1];
  const int context_length = high_length + low_length;
  int high_start = 0;
  int high_end = high_length;
  int low_start = 0;
  int low_end = low_length;
  const bool partitioned = context_length > kPartitionSize;
  if (partitioned) {
    const int high_partitions = DIVIDE_ROUND_UP(high_length, kPartitionSize);
    const int low_partitions = DIVIDE_ROUND_UP(low_length, kPartitionSize);
    if (partition >= high_partitions + low_partitions) {
      return;
    }
    if (partition < high_partitions) {
      high_start = partition * kPartitionSize;
      high_end = MIN(high_start + kPartitionSize, high_length);
      low_end = 0;
    } else {
      high_end = 0;
      low_start = (partition - high_partitions) * kPartitionSize;
      low_end = MIN(low_start + kPartitionSize, low_length);
    }
  } else if (partition != 0) {
    return;
  }

  __shared__ TileWorkspace tile;

  const int query = threadIdx.x / kQueryThreads;
  const int query_thread = threadIdx.x % kQueryThreads;
  const int dimension_chunk = (threadIdx.x % kWarpSize) / (kWarpSize / THREAD_GROUP_SIZE_K);
  const int head = kv_head * kQueriesPerKv + query;
  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head];
  const scalar_t* query_ptr = queries + sequence * query_stride + head * kHeadSize;
  float query_values[kHeadSize / THREAD_GROUP_SIZE_K];
  #pragma unroll
  for (int i = 0; i < kHeadSize / THREAD_GROUP_SIZE_K; ++i) {
    query_values[i] = to_float(query_ptr[dimension_chunk * (kHeadSize / THREAD_GROUP_SIZE_K) + i]);
  }
  float output_first = 0.f;
  float output_second = 0.f;
  float running_max = -FLT_MAX;
  float running_sum = 0.f;
  float* query_scores = tmp_scores + (sequence * num_heads + head) * max_context_len;
  const int* block_table = block_tables + table_offset * max_blocks;
  attend_region<32, 8, 4>(kv_cache, block_table, false, unified_page_size,
      high_start, high_end, 0, scale, alibi_slope, context_length,
      query_values, query_scores, tile, running_max, running_sum, output_first, output_second);
  attend_region<60, 4, 2>(kv_cache, block_table + max_blocks - 1, true, unified_page_size,
      low_start, low_end, high_length, scale, alibi_slope, context_length,
      query_values, query_scores, tile, running_max, running_sum, output_first, output_second);

  const float inv_sum = __fdividef(1.f, running_sum + 1e-6f);
  scalar_t* output = tmp_out + ((sequence * num_heads + head) * partitions_stride + partition) * kHeadSize;
  from_float(output[query_thread], output_first * inv_sum);
  from_float(output[query_thread + kQueryThreads], output_second * inv_sum);
  const int first_token = high_end > high_start ? high_start : high_length + low_start;
  const int token_count = high_end - high_start + low_end - low_start;
  for (int token = query_thread; token < token_count; token += kQueryThreads) {
    const int index = first_token + token;
    query_scores[index] = __expf(query_scores[index] - running_max) * inv_sum;
  }
  if (query_thread == 0) {
    const int index = (sequence * num_heads + head) * partitions_stride + partition;
    exp_sums[index] = running_sum;
    max_logits[index] = running_max;
  }
}

template<typename scalar_t>
void launch(torch::Tensor& slot_ids, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& tmp_scores, torch::Tensor& query,
    torch::Tensor& kv_cache, int layer_idx, int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& kv_len_tables, int max_context_len,
    int max_partitions, const c10::optional<torch::Tensor>& alibi_slopes) {
  const dim3 grid(num_kv_heads, query.size(0), max_partitions);
  sparse_gqa_attention_kernel<scalar_t><<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      slot_ids.data_ptr<int>(), reinterpret_cast<scalar_t*>(tmp_out.data_ptr()),
      tmp_scores.data_ptr<float>(), exp_sums.data_ptr<float>(), max_logits.data_ptr<float>(),
      reinterpret_cast<const scalar_t*>(query.data_ptr()),
      reinterpret_cast<const uint16_t*>(kv_cache.data_ptr()),
      block_tables.data_ptr<int>(), kv_len_tables.data_ptr<int>(), layer_idx,
      block_tables.size(1), num_kv_heads, block_tables.size(3), kv_cache.size(1),
      max_context_len, query.stride(0), scale,
      alibi_slopes.has_value() ? alibi_slopes->data_ptr<float>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace gqa_reuse
}  // namespace vllm

void sparse_paged_attention_gqa(
    torch::Tensor& slot_ids, torch::Tensor& positions, torch::Tensor& out,
    torch::Tensor& exp_sums, torch::Tensor& max_logits, torch::Tensor& tmp_out,
    torch::Tensor& tmp_scores, torch::Tensor& query, torch::Tensor& kv_cache,
    int layer_idx, int num_kv_heads, float scale, torch::Tensor& block_tables,
    torch::Tensor& kv_len_tables, int max_context_len, int num_bits_k_high,
    int num_bits_v_high, int num_bits_k_low, int num_bits_v_low,
    int meta_k_high, int meta_v_high, int meta_k_low, int meta_v_low,
    int k_vec_size, int v_vec_size, int num_tokens_per_page_high,
    int num_tokens_per_page_low, const c10::optional<torch::Tensor>& alibi_slopes) {
  TORCH_CHECK(num_bits_k_high == 8 && num_bits_v_high == 4 &&
      num_bits_k_low == 4 && num_bits_v_low == 2 && meta_k_high == 1 &&
      meta_v_high == 1 && meta_k_low == 1 && meta_v_low == 1,
      "GQA reuse attention requires K8V4/K4V2 with one quantization chunk");
  TORCH_CHECK(k_vec_size == K_VEC_SIZE && v_vec_size == V_VEC_SIZE &&
      num_tokens_per_page_high == 32 && num_tokens_per_page_low == 60,
      "GQA reuse attention requires vector sizes 4/2 and high/low page sizes 32/60");
  TORCH_CHECK(query.dim() == 3 && query.size(2) == 128 && num_kv_heads > 0 &&
      query.size(1) == num_kv_heads * 4,
      "GQA reuse attention requires head size 128 and four query heads per KV head");
  TORCH_CHECK(query.stride(2) == 1 && query.stride(1) == 128,
      "GQA reuse attention requires contiguous query head dimensions");
  TORCH_CHECK(!alibi_slopes.has_value() || (alibi_slopes->scalar_type() == at::ScalarType::Float &&
      alibi_slopes->is_contiguous() && alibi_slopes->numel() >= query.size(1)),
      "GQA reuse attention requires contiguous FP32 ALiBi slopes for every query head");
  TORCH_CHECK(block_tables.dim() == 4 && kv_len_tables.dim() == 4 && kv_cache.dim() == 2,
      "Unexpected GQA reuse attention cache metadata dimensions");
  TORCH_CHECK(out.is_contiguous() && exp_sums.is_contiguous() && max_logits.is_contiguous() &&
      tmp_out.is_contiguous() && tmp_scores.is_contiguous() && kv_cache.is_contiguous() &&
      block_tables.is_contiguous() && kv_len_tables.is_contiguous() &&
      slot_ids.is_contiguous() && positions.is_contiguous(),
      "GQA reuse attention output, workspace and cache tensors must be contiguous");
  TORCH_CHECK(query.size(0) > 0 && max_context_len > 0,
      "GQA reuse attention requires a nonempty batch and positive max_context_len");
  const int max_partitions = max_context_len <= vllm::gqa_reuse::kPartitionSize ? 1 :
      DIVIDE_ROUND_UP(max_context_len, vllm::gqa_reuse::kPartitionSize) + 1;
  const int64_t query_heads = query.size(0) * query.size(1);
  TORCH_CHECK(out.numel() >= query_heads * 128 && tmp_scores.numel() >= query_heads * max_context_len,
      "GQA reuse attention output or score buffer is too small");
  TORCH_CHECK(tmp_out.numel() >= query_heads * max_partitions * 128 &&
      exp_sums.numel() >= query_heads * max_partitions &&
      max_logits.numel() >= query_heads * max_partitions,
      "GQA reuse attention partition buffers are too small");
  if (query.scalar_type() == at::ScalarType::Half) {
    vllm::gqa_reuse::launch<uint16_t>(slot_ids, exp_sums, max_logits, tmp_out,
        tmp_scores, query, kv_cache, layer_idx, num_kv_heads, scale,
        block_tables, kv_len_tables, max_context_len, max_partitions, alibi_slopes);
  } else if (query.scalar_type() == at::ScalarType::BFloat16) {
    vllm::gqa_reuse::launch<__nv_bfloat16>(slot_ids, exp_sums, max_logits, tmp_out,
        tmp_scores, query, kv_cache, layer_idx, num_kv_heads, scale,
        block_tables, kv_len_tables, max_context_len, max_partitions, alibi_slopes);
  } else {
    TORCH_CHECK(false, "Unsupported GQA reuse attention dtype: ", query.scalar_type());
  }
  sparse_paged_attention_gqa_reduce(out, exp_sums, max_logits, tmp_out, tmp_scores,
      slot_ids, block_tables, kv_len_tables, kv_cache, layer_idx, num_kv_heads, max_context_len);
}
