#pragma once

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace diffkv {

inline void check_tensor(const torch::Tensor& tensor,
                         const torch::Tensor& reference, at::ScalarType dtype,
                         const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.device() == reference.device(), name,
              " must be on the same CUDA device as the input");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unsupported dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline void check_cache(const torch::Tensor& key, const torch::Tensor& value,
                        const torch::Tensor& cache, const torch::Tensor& slots,
                        const torch::Tensor& tables,
                        const torch::Tensor& lengths,
                        const torch::Tensor& config, int layer) {
  // Writers address tokens through key.stride(0), so K/V may be token rows of
  // a wider tensor (e.g. views of the QKV projection) as long as each row is
  // [KV heads, 128] contiguous and both share the token stride.
  for (const auto& [tensor, name] :
       {std::pair{&key, "key"}, std::pair{&value, "value"}}) {
    TORCH_CHECK(tensor->is_cuda() && tensor->device() == key.device(), name,
                " must be on the same CUDA device as the input");
    TORCH_CHECK(tensor->scalar_type() == at::kHalf, name,
                " has an unsupported dtype");
    TORCH_CHECK(tensor->dim() == 3 && tensor->stride(2) == 1 &&
                    tensor->stride(1) == tensor->size(2) &&
                    tensor->stride(0) >= tensor->size(1) * tensor->size(2),
                name, " rows must be contiguous [KV heads, 128]");
  }
  TORCH_CHECK(key.stride(0) == value.stride(0),
              "key and value must share the token stride");
  check_tensor(cache, key, at::kShort, "packed cache");
  check_tensor(slots, key, at::kInt, "slot IDs");
  check_tensor(tables, key, at::kInt, "block tables");
  check_tensor(lengths, key, at::kInt, "KV lengths");
  check_tensor(config, key, at::kFloat, "compression configuration");
  TORCH_CHECK(
      key.dim() == 3 && key.size(2) == 128 && key.sizes() == value.sizes(),
      "DiffKV requires matching [tokens, KV heads, 128] FP16 K/V");
  TORCH_CHECK(cache.dim() == 2 && cache.size(0) > 0 && cache.size(1) == 3392 &&
                  reinterpret_cast<uintptr_t>(cache.data_ptr()) % 8 == 0,
              "DiffKV requires an aligned int16 [pages, 3392] cache");
  TORCH_CHECK(tables.dim() == 4 && tables.size(0) > 0 && tables.size(3) > 0 &&
                  lengths.dim() == 4 && lengths.size(3) == 2 &&
                  tables.size(0) == lengths.size(0) &&
                  tables.size(1) == lengths.size(1) &&
                  tables.size(2) == lengths.size(2) &&
                  tables.size(2) == key.size(1),
              "Invalid per-request, per-layer, per-head cache tables");
  TORCH_CHECK(layer >= 0 && layer < tables.size(1), "Invalid layer index");
  TORCH_CHECK(slots.dim() == 1 && slots.numel() > 0 && config.dim() == 2 &&
                  config.size(0) == tables.size(0) && config.size(1) == 2,
              "Invalid slot IDs or compression configuration");
}

}  // namespace diffkv
