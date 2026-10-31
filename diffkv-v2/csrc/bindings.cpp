// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include "cache.h"
#include "mem_mgt.h"
#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.attr("build_fingerprint") = DIFFKV_V2_BUILD_FINGERPRINT;
  module.attr("packed_page_bytes") = 6784;
  auto attention = module.def_submodule("ops");
  attention.def("sparse_paged_attention_gqa", &sparse_paged_attention_gqa);
  auto cache = module.def_submodule("cache_ops");
  cache.def("compress_and_append_cache_prompt_phase",
            &compress_and_append_cache_prompt_phase);
  cache.def("compress_and_append_cache_long_prompt_phase",
            &compress_and_append_cache_long_prompt_phase);
  cache.def("compress_and_append_cache_decode_phase",
            &compress_and_append_cache_decode_phase);
  auto memory = module.def_submodule("mem_mgt_ops");
  memory.def("allocate_seqs", &allocate_seqs);
  memory.def("prepare_free_prompt", &prepare_free_prompt);
  memory.def("free_prompt", &free_prompt);
  memory.def("prepare_append_seqs", &prepare_append_seqs);
  memory.def("append_seqs", &append_seqs);
  memory.def("prepare_free_seqs", &prepare_free_seqs);
  memory.def("free_seqs", &free_seqs);
  module.attr("mem_mgt") = memory;
}
