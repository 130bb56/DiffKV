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

void sparse_paged_attention_gqa_prefix(
    torch::Tensor& slot_ids, torch::Tensor& positions, torch::Tensor& out,
    torch::Tensor& exp_sums, torch::Tensor& max_logits, torch::Tensor& tmp_out,
    torch::Tensor& tmp_scores, torch::Tensor& query, torch::Tensor& kv_cache,
    int layer_idx, int num_kv_heads, float scale, torch::Tensor& block_tables,
    torch::Tensor& kv_len_tables, int max_context_len, int num_bits_k_high,
    int num_bits_v_high, int num_bits_k_low, int num_bits_v_low,
    int meta_k_high, int meta_v_high, int meta_k_low, int meta_v_low,
    int k_vec_size, int v_vec_size, int num_tokens_per_page_high,
    int num_tokens_per_page_low, torch::Tensor& chunk_lse,
    torch::Tensor& lse_out);

// Dequantize each (request, KV head) retained prefix into FP16 rows
// key_out/value_out[row_offsets[kv_head * requests + request] + token] and
// copy the request's chunk K/V rows behind them. slices * 4 must cover the
// largest high+low page count plus 32-token chunk slice count of any head.
void dequantize_prefix(torch::Tensor& slot_ids, torch::Tensor& row_offsets,
                       torch::Tensor& block_tables, torch::Tensor& kv_len_tables,
                       torch::Tensor& kv_cache, int layer_idx,
                       torch::Tensor& chunk_key, torch::Tensor& chunk_value,
                       torch::Tensor& chunk_starts, torch::Tensor& key_out,
                       torch::Tensor& value_out, int slices);
