#pragma once
#include <torch/extension.h>

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
    const c10::optional<torch::Tensor>& alibi_slopes);
