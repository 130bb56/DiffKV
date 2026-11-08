#pragma once
#include <torch/extension.h>

void compress_and_append_cache_prompt_phase(
    torch::Tensor& key, torch::Tensor& value, torch::Tensor& score,
    torch::Tensor& kv_cache, torch::Tensor& slot_ids,
    torch::Tensor& block_tables, torch::Tensor& kv_len_tables,
    torch::Tensor& seq_start_loc, torch::Tensor& compress_config_tables,
    const int kv_buffer_size, const int layer_idx, const int num_bits_k_high,
    const int num_bits_v_high, const int num_bits_k_low,
    const int num_bits_v_low, const int num_chunks_k_high,
    const int num_chunks_v_high, const int num_chunks_k_low,
    const int num_chunks_v_low, const int k_vec_size, const int v_vec_size,
    const int num_tokens_per_page_high, const int num_tokens_per_page_low);

void compress_and_append_cache_long_prompt_phase(
    torch::Tensor& key, torch::Tensor& value, torch::Tensor& score,
    torch::Tensor& sorted_indices, torch::Tensor& kv_cache,
    torch::Tensor& slot_ids, torch::Tensor& block_tables,
    torch::Tensor& kv_len_tables, torch::Tensor& seq_start_loc,
    torch::Tensor& compress_config_tables, const int kv_buffer_size,
    const int layer_idx, const int num_bits_k_high, const int num_bits_v_high,
    const int num_bits_k_low, const int num_bits_v_low,
    const int num_chunks_k_high, const int num_chunks_v_high,
    const int num_chunks_k_low, const int num_chunks_v_low,
    const int k_vec_size, const int v_vec_size,
    const int num_tokens_per_page_high, const int num_tokens_per_page_low);

void compress_and_append_cache_decode_phase(
    torch::Tensor& key, torch::Tensor& value, torch::Tensor& positions,
    torch::Tensor& kv_cache, torch::Tensor& slot_ids,
    torch::Tensor& block_tables, torch::Tensor& kv_len_tables,
    torch::Tensor& compress_config_tables, const int max_context_len,
    const int kv_buffer_size, const int layer_idx, const int num_bits_k_high,
    const int num_bits_v_high, const int num_bits_k_low,
    const int num_bits_v_low, const int num_chunks_k_high,
    const int num_chunks_v_high, const int num_chunks_k_low,
    const int num_chunks_v_low, const int k_vec_size, const int v_vec_size,
    const int num_tokens_per_page_high, const int num_tokens_per_page_low);

// Group-maximum DiffKV prompt scores [requests, KV heads, max_prompt_len] from
// the last 64 queries of each whole prompt and FA2's log-sum-exp.
torch::Tensor prompt_scores(torch::Tensor& query, torch::Tensor& key,
                            torch::Tensor& lse, torch::Tensor& cu_seqlens,
                            int64_t max_prompt_len, double scale);

// prompt_scores followed by compress_and_append_cache_prompt_phase for every
// whole prompt of the step; returns the scores.
torch::Tensor score_and_compress_prompts(
    torch::Tensor& query, torch::Tensor& key, torch::Tensor& value,
    torch::Tensor& lse, torch::Tensor& cu_seqlens, int64_t max_prompt_len,
    double scale, torch::Tensor& kv_cache, torch::Tensor& slot_ids,
    torch::Tensor& block_tables, torch::Tensor& kv_len_tables,
    torch::Tensor& compress_config_tables, int kv_buffer_size, int layer_idx,
    int num_bits_k_high, int num_bits_v_high, int num_bits_k_low,
    int num_bits_v_low, int num_chunks_k_high, int num_chunks_v_high,
    int num_chunks_k_low, int num_chunks_v_low, int k_vec_size, int v_vec_size,
    int num_tokens_per_page_high, int num_tokens_per_page_low);
