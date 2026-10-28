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
#include "../checks.h"
#include "attention_dtypes.h"
#include "cache_utils.h"
#include "../quantization/kv_cache_quant/kv_cache_quant.h"

#include <cfloat>

namespace diffkv_v2 {
namespace gqa_reuse {

constexpr int kHeadSize = 128;
constexpr int kQueriesPerKv = 4;
constexpr int kThreads = 128;
// The kernel masks partial pages, so partitions can split a physical page.
constexpr int kPartitionSize = 240;

#include "sparse_gqa_register_kernels.cuh"
#include "sparse_gqa_reduce_kernels.cuh"

void launch_reduce(torch::Tensor& out, torch::Tensor& exp_sums,
                   torch::Tensor& max_logits, torch::Tensor& tmp_out,
                   torch::Tensor& tmp_scores, torch::Tensor& slot_ids,
                   torch::Tensor& block_tables, torch::Tensor& kv_len_tables,
                   torch::Tensor& kv_cache, int layer_idx, int num_kv_heads,
                   int max_context_len) {
  constexpr int PARTITION_SIZE = kPartitionSize;
  const int max_num_partitions =
      max_context_len <= PARTITION_SIZE
          ? 1
          : DIVIDE_ROUND_UP(max_context_len, PARTITION_SIZE) + 1;
  const dim3 grid(num_kv_heads, out.size(0), 4);
  const dim3 block(256);
  const int shared_bytes = 4 * max_num_partitions * sizeof(float);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto launch = [&](auto scalar_type) {
    using T = decltype(scalar_type);
    register_gqa_reduce_kernel<T><<<grid, block, shared_bytes, stream>>>(
        reinterpret_cast<T*>(out.data_ptr()), exp_sums.data_ptr<float>(),
        max_logits.data_ptr<float>(), reinterpret_cast<T*>(tmp_out.data_ptr()),
        tmp_scores.data_ptr<float>(), block_tables.data_ptr<int>(),
        kv_len_tables.data_ptr<int>(), slot_ids.data_ptr<int>(),
        reinterpret_cast<uint16_t*>(kv_cache.data_ptr()), kv_cache.size(1),
        layer_idx, block_tables.size(1), num_kv_heads, max_context_len,
        block_tables.size(3), max_num_partitions);
  };
  if (out.scalar_type() == at::ScalarType::Half) {
    launch(uint16_t{});
  } else {
    TORCH_CHECK(false, "Unsupported GQA output data type: ", out.dtype());
  }
}

template <typename scalar_t>
void launch(torch::Tensor& slot_ids, torch::Tensor& exp_sums,
            torch::Tensor& max_logits, torch::Tensor& tmp_out,
            torch::Tensor& tmp_scores, torch::Tensor& query,
            torch::Tensor& kv_cache, int layer_idx, int num_kv_heads,
            float scale, torch::Tensor& block_tables,
            torch::Tensor& kv_len_tables, int max_context_len,
            int max_partitions,
            const c10::optional<torch::Tensor>& alibi_slopes) {
  const dim3 grid(num_kv_heads, query.size(0), max_partitions);
  register_gqa_attention_kernel<scalar_t>
      <<<grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
          slot_ids.data_ptr<int>(),
          reinterpret_cast<scalar_t*>(tmp_out.data_ptr()),
          tmp_scores.data_ptr<float>(), exp_sums.data_ptr<float>(),
          max_logits.data_ptr<float>(),
          reinterpret_cast<const scalar_t*>(query.data_ptr()),
          reinterpret_cast<const uint16_t*>(kv_cache.data_ptr()),
          block_tables.data_ptr<int>(), kv_len_tables.data_ptr<int>(),
          layer_idx, block_tables.size(1), num_kv_heads, block_tables.size(3),
          kv_cache.size(1), max_context_len, query.stride(0), scale,
          alibi_slopes.has_value() ? alibi_slopes->data_ptr<float>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace gqa_reuse
}  // namespace diffkv_v2

void sparse_paged_attention_gqa(
    torch::Tensor& slot_ids, torch::Tensor& positions, torch::Tensor& out,
    torch::Tensor& exp_sums, torch::Tensor& max_logits, torch::Tensor& tmp_out,
    torch::Tensor& tmp_scores, torch::Tensor& query, torch::Tensor& kv_cache,
    int layer_idx, int num_kv_heads, float scale, torch::Tensor& block_tables,
    torch::Tensor& kv_len_tables, int max_context_len, int num_bits_k_high,
    int num_bits_v_high, int num_bits_k_low, int num_bits_v_low,
    int meta_k_high, int meta_v_high, int meta_k_low, int meta_v_low,
    int k_vec_size, int v_vec_size, int num_tokens_per_page_high,
    int num_tokens_per_page_low,
    const c10::optional<torch::Tensor>& alibi_slopes) {
  TORCH_CHECK(query.is_cuda() && query.scalar_type() == at::kHalf,
              "DiffKV v2 bringup requires FP16 CUDA queries");
  const c10::cuda::CUDAGuard device_guard(query.device());
  diffkv_v2::check_tensor(slot_ids, query, at::kInt, "slot IDs");
  diffkv_v2::check_tensor(positions, query, at::kInt, "positions");
  diffkv_v2::check_tensor(block_tables, query, at::kInt, "block tables");
  diffkv_v2::check_tensor(kv_len_tables, query, at::kInt, "KV lengths");
  diffkv_v2::check_tensor(kv_cache, query, at::kShort, "packed cache");
  diffkv_v2::check_tensor(out, query, at::kHalf, "output");
  diffkv_v2::check_tensor(tmp_out, query, at::kHalf, "partition output");
  diffkv_v2::check_tensor(exp_sums, query, at::kFloat, "partition sums");
  diffkv_v2::check_tensor(max_logits, query, at::kFloat, "partition maxima");
  diffkv_v2::check_tensor(tmp_scores, query, at::kFloat, "probability scratch");
  TORCH_CHECK(slot_ids.dim() == 1 && positions.dim() == 1 &&
                  slot_ids.numel() == query.size(0) &&
                  positions.numel() == query.size(0),
              "Decode expects one slot and position per query token");
  TORCH_CHECK(!alibi_slopes.has_value(),
              "DiffKV v2 bringup does not support ALiBi");
  TORCH_CHECK(block_tables.dim() == 4 && kv_len_tables.dim() == 4 &&
                  kv_len_tables.size(3) == 2 &&
                  block_tables.size(0) == kv_len_tables.size(0) &&
                  block_tables.size(1) == kv_len_tables.size(1) &&
                  block_tables.size(2) == kv_len_tables.size(2) &&
                  block_tables.size(2) == num_kv_heads &&
                  block_tables.size(3) > 0 && layer_idx >= 0 &&
                  layer_idx < block_tables.size(1),
              "Invalid DiffKV v2 cache table dimensions or layer index");
  TORCH_CHECK(
      kv_cache.dim() == 2 && kv_cache.size(0) > 0 && kv_cache.size(1) == 3392,
      "DiffKV v2 requires int16 [pages, 3392]");
  TORCH_CHECK(
      num_bits_k_high == 8 && num_bits_v_high == 4 && num_bits_k_low == 4 &&
          num_bits_v_low == 2 && meta_k_high == 1 && meta_v_high == 1 &&
          meta_k_low == 1 && meta_v_low == 1,
      "GQA reuse attention requires K8V4/K4V2 with one quantization chunk");
  TORCH_CHECK(k_vec_size == K_VEC_SIZE && v_vec_size == V_VEC_SIZE &&
                  num_tokens_per_page_high == 32 &&
                  num_tokens_per_page_low == 60,
              "GQA reuse attention requires vector sizes 4/2 and high/low page "
              "sizes 32/60");
  TORCH_CHECK(query.dim() == 3 && query.size(2) == 128 && num_kv_heads > 0 &&
                  query.size(1) == num_kv_heads * 4,
              "GQA reuse attention requires head size 128 and four query heads "
              "per KV head");
  TORCH_CHECK(query.stride(2) == 1 && query.stride(1) == 128,
              "GQA reuse attention requires contiguous query head dimensions");
  TORCH_CHECK(!alibi_slopes.has_value() ||
                  (alibi_slopes->scalar_type() == at::ScalarType::Float &&
                   alibi_slopes->is_contiguous() &&
                   alibi_slopes->numel() >= query.size(1)),
              "GQA reuse attention requires contiguous FP32 ALiBi slopes for "
              "every query head");
  TORCH_CHECK(block_tables.dim() == 4 && kv_len_tables.dim() == 4 &&
                  kv_cache.dim() == 2,
              "Unexpected GQA reuse attention cache metadata dimensions");
  TORCH_CHECK(out.is_contiguous() && exp_sums.is_contiguous() &&
                  max_logits.is_contiguous() && tmp_out.is_contiguous() &&
                  tmp_scores.is_contiguous() && kv_cache.is_contiguous() &&
                  block_tables.is_contiguous() &&
                  kv_len_tables.is_contiguous() && slot_ids.is_contiguous() &&
                  positions.is_contiguous(),
              "GQA reuse attention output, workspace and cache tensors must be "
              "contiguous");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(kv_cache.data_ptr()) % alignof(k_vec_type) ==
              0 &&
          (kv_cache.size(1) * sizeof(uint16_t)) % alignof(k_vec_type) == 0,
      "GQA reuse attention requires an 8-byte aligned cache base and page "
      "stride");
  TORCH_CHECK(query.size(0) > 0 && max_context_len > 0,
              "GQA reuse attention requires a nonempty batch and positive "
              "max_context_len");
  const int max_partitions =
      max_context_len <= diffkv_v2::gqa_reuse::kPartitionSize
          ? 1
          : DIVIDE_ROUND_UP(max_context_len,
                            diffkv_v2::gqa_reuse::kPartitionSize) +
                1;
  const int64_t query_heads = query.size(0) * query.size(1);
  TORCH_CHECK(out.numel() >= query_heads * 128 &&
                  tmp_scores.numel() >= query_heads * max_context_len,
              "GQA reuse attention output or score buffer is too small");
  TORCH_CHECK(tmp_out.numel() >= query_heads * max_partitions * 128 &&
                  exp_sums.numel() >= query_heads * max_partitions &&
                  max_logits.numel() >= query_heads * max_partitions,
              "GQA reuse attention partition buffers are too small");
  if (query.scalar_type() == at::ScalarType::Half) {
    diffkv_v2::gqa_reuse::launch<uint16_t>(
        slot_ids, exp_sums, max_logits, tmp_out, tmp_scores, query, kv_cache,
        layer_idx, num_kv_heads, scale, block_tables, kv_len_tables,
        max_context_len, max_partitions, alibi_slopes);
  } else {
    TORCH_CHECK(false,
                "Unsupported GQA reuse attention dtype: ", query.scalar_type());
  }
  diffkv_v2::gqa_reuse::launch_reduce(
      out, exp_sums, max_logits, tmp_out, tmp_scores, slot_ids, block_tables,
      kv_len_tables, kv_cache, layer_idx, num_kv_heads, max_context_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
