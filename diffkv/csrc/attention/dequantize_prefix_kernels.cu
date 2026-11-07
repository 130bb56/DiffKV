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

// Chunked prefill reads the retained prefix of every (request, KV head) once:
// this kernel dequantizes the packed K8V4 and K4V2 pages into FP16 rows and
// appends the chunk's own K/V rows behind them, so FA2 can attend to prefix
// and chunk together. Grid (kv_heads, requests, slices); each warp converts
// one page or copies one 32-token chunk slice. The lane mapping and page
// offsets mirror compute_keys and compute_values in the register kernels.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "../ops.h"
#include "../checks.h"
#include "attention_dtypes.h"
#include "cache_utils.h"
#include "../quantization/kv_cache_quant.h"

namespace diffkv {
namespace dequantize_prefix_detail {

constexpr int kThreads = 128;
constexpr int kDimensionThreads = 8;
constexpr int kDimensionsPerThread = 128 / kDimensionThreads;
constexpr int kGroupsPerWarp = 32 / kDimensionThreads;
constexpr int kWarps = kThreads / 32;
constexpr int kSlicesPerBlock = kWarps;
constexpr int kChunkSliceTokens = 32;

// Store one thread's 16 contiguous dimensions with two 16-byte writes.
__device__ __forceinline__ void store_dimensions(
    uint16_t* row, const float (&values)[kDimensionsPerThread]) {
  uint4 words[2];
  uint16_t* halves = reinterpret_cast<uint16_t*>(words);
#pragma unroll
  for (int dimension = 0; dimension < kDimensionsPerThread; ++dimension) {
    from_float(halves[dimension], values[dimension]);
  }
  reinterpret_cast<uint4*>(row)[0] = words[0];
  reinterpret_cast<uint4*>(row)[1] = words[1];
}

template <int PageTokens, int KeyBits>
__device__ __forceinline__ void dequantize_key_page(const uint16_t* kv_cache,
                                                    int64_t page_base, int page,
                                                    int region_length,
                                                    int logical_offset,
                                                    uint16_t* out) {
  constexpr int kPack = 16 / KeyBits;
  constexpr int kPacks = 128 / kPack;
  constexpr int kMetaBase = DIVIDE_ROUND_UP(PageTokens * kPacks * 2, 32) * 16;
  constexpr int kVectors = kDimensionsPerThread / (kPack * K_VEC_SIZE);
  const int lane = threadIdx.x % 32;
  const int token_group = lane % kGroupsPerWarp;
  const int dimension_thread = lane / kGroupsPerWarp;
  const int original_key_group = dimension_thread / 2;
  const int first_vector = (dimension_thread % 2) * kVectors;

  for (int token = token_group; token < PageTokens; token += kGroupsPerWarp) {
    const int region_token = page * PageTokens + token;
    if (region_token >= region_length) {
      continue;
    }
    const quant_meta_type meta = *reinterpret_cast<const quant_meta_type*>(
        kv_cache + page_base + kMetaBase + token * 2);
    const float scale_key = to_float(meta.scale);
    const float zero_key = to_float(meta.zero_point);
    float values[kDimensionsPerThread];
#pragma unroll
    for (int vector = 0; vector < kVectors; ++vector) {
      const int offset = (((first_vector + vector) * PageTokens + token) *
                              THREAD_GROUP_SIZE_K +
                          original_key_group) *
                         K_VEC_SIZE;
      const k_vec_type packed =
          *reinterpret_cast<const k_vec_type*>(kv_cache + page_base + offset);
#pragma unroll
      for (int packed_element = 0; packed_element < K_VEC_SIZE;
           ++packed_element) {
        float decoded[kPack];
        unpack_and_dequant(packed.data[packed_element], KeyBits, scale_key,
                           zero_key, decoded);
#pragma unroll
        for (int element = 0; element < kPack; ++element) {
          values[(vector * K_VEC_SIZE + packed_element) * kPack + element] =
              decoded[element];
        }
      }
    }
    store_dimensions(out + (logical_offset + region_token) * 128 +
                         dimension_thread * kDimensionsPerThread,
                     values);
  }
}

template <int PageTokens, int KeyBits, int ValueBits>
__device__ __forceinline__ void dequantize_value_page(
    const uint16_t* kv_cache, int64_t page_base, int page, int region_length,
    int logical_offset, uint16_t* out) {
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
  const int token_group = lane / kDimensionThreads;
  const int dimension_thread = lane % kDimensionThreads;

  for (int vector = token_group;
       vector < DIVIDE_ROUND_UP(PageTokens, V_VEC_SIZE);
       vector += kGroupsPerWarp) {
    const int first_token = vector * V_VEC_SIZE;
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
      if (token < PageTokens && region_token < region_length) {
        const quant_meta_type meta = *reinterpret_cast<const quant_meta_type*>(
            kv_cache + page_base + kValueMetaBase + token * 2);
        const float value_scale = to_float(meta.scale);
        const float value_zero = to_float(meta.zero_point);
        float values[kDimensionsPerThread];
#pragma unroll
        for (int pack = 0; pack < kPacksPerThread; ++pack) {
          float decoded[kValuePack];
          unpack_and_dequant(packed_values[pack].data[element], ValueBits,
                             value_scale, value_zero, decoded);
#pragma unroll
          for (int dimension = 0; dimension < kValuePack; ++dimension) {
            values[pack * kValuePack + dimension] = decoded[dimension];
          }
        }
        store_dimensions(out + (logical_offset + region_token) * 128 +
                             dimension_thread * kDimensionsPerThread,
                         values);
      }
    }
  }
}

// Copy one 32-token slice of the chunk's K or V rows (16 bytes per lane).
__device__ __forceinline__ void copy_chunk_slice(const uint16_t* source,
                                                 int64_t source_token_stride,
                                                 int first_token, int count,
                                                 uint16_t* out) {
  const int lane = threadIdx.x % 32;
  constexpr int kPiecesPerToken = 128 / 8;
  for (int piece = lane; piece < kChunkSliceTokens * kPiecesPerToken;
       piece += 32) {
    const int token = first_token + piece / kPiecesPerToken;
    if (token >= count) {
      break;
    }
    const int dimension = (piece % kPiecesPerToken) * 8;
    *reinterpret_cast<uint4*>(out + token * 128 + dimension) =
        *reinterpret_cast<const uint4*>(source + token * source_token_stride +
                                        dimension);
  }
}

__global__ __launch_bounds__(kThreads) void dequantize_prefix_kernel(
    const int* __restrict__ slot_ids, const int* __restrict__ row_offsets,
    const int* __restrict__ block_tables, const int* __restrict__ kv_len_tables,
    const uint16_t* __restrict__ kv_cache,
    const uint16_t* __restrict__ chunk_key,
    const uint16_t* __restrict__ chunk_value,
    const int* __restrict__ chunk_starts, uint16_t* __restrict__ key_out,
    uint16_t* __restrict__ value_out, int layer_idx, int num_layers,
    int num_kv_heads, int max_blocks, int unified_page_size,
    int64_t chunk_token_stride) {
  const int kv_head = blockIdx.x;
  const int request = blockIdx.y;
  const int table_offset =
      (slot_ids[request] * num_layers + layer_idx) * num_kv_heads + kv_head;
  const int high_length = kv_len_tables[table_offset * 2];
  const int low_length = kv_len_tables[table_offset * 2 + 1];
  const int high_pages = DIVIDE_ROUND_UP(high_length, 32);
  const int low_pages = DIVIDE_ROUND_UP(low_length, 60);
  const int* block_table = block_tables + table_offset * max_blocks;
  const int64_t first_row = row_offsets[kv_head * gridDim.y + request];
  uint16_t* keys = key_out + first_row * 128;
  uint16_t* values = value_out + first_row * 128;
  const int slice = blockIdx.z * kSlicesPerBlock + threadIdx.x / 32;
  if (slice < high_pages) {
    const int64_t page_base =
        static_cast<int64_t>(block_table[slice]) * unified_page_size;
    dequantize_key_page<32, 8>(kv_cache, page_base, slice, high_length, 0,
                               keys);
    dequantize_value_page<32, 8, 4>(kv_cache, page_base, slice, high_length, 0,
                                    values);
  } else if (slice - high_pages < low_pages) {
    const int page = slice - high_pages;
    const int64_t page_base =
        static_cast<int64_t>(block_table[max_blocks - 1 - page]) *
        unified_page_size;
    dequantize_key_page<60, 4>(kv_cache, page_base, page, low_length,
                               high_length, keys);
    dequantize_value_page<60, 4, 2>(kv_cache, page_base, page, low_length,
                                    high_length, values);
  } else {
    const int chunk_start = chunk_starts[request];
    const int chunk_length = chunk_starts[request + 1] - chunk_start;
    const int first_token =
        (slice - high_pages - low_pages) * kChunkSliceTokens;
    if (first_token < chunk_length) {
      const int prefix_length = high_length + low_length;
      const int64_t source =
          static_cast<int64_t>(chunk_start) * chunk_token_stride +
          kv_head * 128;
      copy_chunk_slice(chunk_key + source, chunk_token_stride, first_token,
                       chunk_length, keys + prefix_length * 128);
      copy_chunk_slice(chunk_value + source, chunk_token_stride, first_token,
                       chunk_length, values + prefix_length * 128);
    }
  }
}

}  // namespace dequantize_prefix_detail
}  // namespace diffkv

void dequantize_prefix(torch::Tensor& slot_ids, torch::Tensor& row_offsets,
                       torch::Tensor& block_tables,
                       torch::Tensor& kv_len_tables, torch::Tensor& kv_cache,
                       int layer_idx, torch::Tensor& chunk_key,
                       torch::Tensor& chunk_value, torch::Tensor& chunk_starts,
                       torch::Tensor& key_out, torch::Tensor& value_out,
                       int slices) {
  TORCH_CHECK(key_out.is_cuda() && key_out.scalar_type() == at::kHalf,
              "Dequantized keys must be FP16 CUDA");
  const c10::cuda::CUDAGuard device_guard(key_out.device());
  diffkv::check_tensor(slot_ids, key_out, at::kInt, "slot IDs");
  diffkv::check_tensor(row_offsets, key_out, at::kInt, "row offsets");
  diffkv::check_tensor(block_tables, key_out, at::kInt, "block tables");
  diffkv::check_tensor(kv_len_tables, key_out, at::kInt, "KV lengths");
  diffkv::check_tensor(kv_cache, key_out, at::kShort, "packed cache");
  diffkv::check_tensor(chunk_starts, key_out, at::kInt, "chunk starts");
  diffkv::check_tensor(key_out, key_out, at::kHalf, "dequantized keys");
  diffkv::check_tensor(value_out, key_out, at::kHalf, "dequantized values");
  TORCH_CHECK(block_tables.dim() == 4 && kv_len_tables.dim() == 4 &&
                  kv_len_tables.size(3) == 2 &&
                  block_tables.size(0) == kv_len_tables.size(0) &&
                  block_tables.size(1) == kv_len_tables.size(1) &&
                  block_tables.size(2) == kv_len_tables.size(2) &&
                  block_tables.size(3) > 0 && layer_idx >= 0 &&
                  layer_idx < block_tables.size(1),
              "Invalid DiffKV cache table dimensions or layer index");
  TORCH_CHECK(
      kv_cache.dim() == 2 && kv_cache.size(0) > 0 && kv_cache.size(1) == 3392,
      "DiffKV requires int16 [pages, 3392]");
  const int num_kv_heads = block_tables.size(2);
  const int requests = slot_ids.numel();
  TORCH_CHECK(requests > 0 && row_offsets.numel() == requests * num_kv_heads &&
                  chunk_starts.numel() == requests + 1,
              "Row offsets need one entry per KV head and request, chunk "
              "starts one per request plus one");
  TORCH_CHECK(key_out.dim() == 2 && key_out.size(1) == 128 &&
                  key_out.sizes() == value_out.sizes(),
              "Dequantized K/V must be [rows, 128]");
  TORCH_CHECK(chunk_key.is_cuda() && chunk_key.scalar_type() == at::kHalf &&
                  chunk_key.dim() == 3 && chunk_key.size(1) == num_kv_heads &&
                  chunk_key.size(2) == 128 && chunk_key.stride(2) == 1 &&
                  chunk_key.stride(1) == 128 &&
                  chunk_value.sizes() == chunk_key.sizes() &&
                  chunk_value.strides() == chunk_key.strides() &&
                  chunk_key.stride(0) % 8 == 0,
              "Chunk K/V must be FP16 [tokens, kv_heads, 128] with contiguous "
              "heads and a 16-byte aligned token stride");
  TORCH_CHECK(slices > 0, "slices must be positive");
  const dim3 grid(num_kv_heads, requests, slices);
  diffkv::dequantize_prefix_detail::dequantize_prefix_kernel<<<
      grid, diffkv::dequantize_prefix_detail::kThreads, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      slot_ids.data_ptr<int>(), row_offsets.data_ptr<int>(),
      block_tables.data_ptr<int>(), kv_len_tables.data_ptr<int>(),
      reinterpret_cast<const uint16_t*>(kv_cache.data_ptr()),
      reinterpret_cast<const uint16_t*>(chunk_key.data_ptr()),
      reinterpret_cast<const uint16_t*>(chunk_value.data_ptr()),
      chunk_starts.data_ptr<int>(),
      reinterpret_cast<uint16_t*>(key_out.data_ptr()),
      reinterpret_cast<uint16_t*>(value_out.data_ptr()), layer_idx,
      block_tables.size(1), num_kv_heads, block_tables.size(3),
      kv_cache.size(1), chunk_key.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
