# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunked prefill attention over the packed DiffKV cache.

Every (request, KV head) keeps its own retained prefix, so prefix lengths differ
per head. For each KV head group the layer runs one FA2 varlen call over the
requests' prefill chunks: its 4 query heads are the GQA group, its keys are the
dequantized prefix followed by the chunk (rows built by dequantize_prefix).
FA2 masks the chunk causally and leaves the prefix fully visible (bottom-right
alignment). A Triton kernel then scores every key token and the prefix token
scores are added into the cache pages, as decode does; chunk token scores are
returned for the compression step.

A step whose chunks all start their prompt takes the direct path: FA2 on the
activations and the score kernel in the original [tokens, heads] layout.

Scores follow the original prompt kernel: per query head, the probabilities a
key receives from the last SCORE_WINDOW (64) queries of the chunk are summed,
and the token score is the maximum over the group's four heads. Prefix tokens
get their cached position advanced by the unscored chunk queries, so the decode
kernel's mean (score / queries since caching) counts only scoring queries.
"""

from dataclasses import dataclass

import torch

from diffkv.ops import PAGE_SIZE, QUANT_ARGS, get_extension
from vllm.triton_utils import tl, triton

HEAD_SIZE = 128
QUERIES_PER_KV = 4
HIGH_PAGE_TOKENS = 32
LOW_PAGE_TOKENS = 60
# Only the last SCORE_WINDOW queries of a chunk score tokens, as the original
# prompt kernel used the last NUM_TOKENS_SCORE prompt queries.
SCORE_WINDOW = 64
# Prompt-phase writers keep the last KV_BUFFER_SIZE tokens at high precision.
KV_BUFFER_SIZE = 64
# 32-bit word offsets of the score and position fields inside a 6784-byte
# page (page_layout in evaluations/attention/correctness.py, halved).
HIGH_SCORE_BASE = 1600
LOW_SCORE_BASE = 1568
HIGH_POSITION_BASE = HIGH_SCORE_BASE + HIGH_PAGE_TOKENS
LOW_POSITION_BASE = LOW_SCORE_BASE + LOW_PAGE_TOKENS
PAGE_WORDS = PAGE_SIZE // 2
# dequantize_prefix: one warp per page or per 32-token chunk slice, 4 per block.
SLICES_PER_BLOCK = 4
CHUNK_SLICE_TOKENS = 32


@triton.jit
def _head_column_sum(
    q_ptr,
    lse_ptr,
    keys_t,
    offs_m,
    offs_d,
    mask_m,
    visible,
    scale,
    q_token_stride,
    lse_token_stride,
):
    query = tl.load(
        q_ptr + offs_m[:, None] * q_token_stride + offs_d[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )
    lse = tl.load(lse_ptr + offs_m * lse_token_stride, mask=mask_m, other=float("inf"))
    probabilities = tl.exp(tl.dot(query, keys_t) * scale - lse[:, None])
    return tl.sum(tl.where(visible, probabilities, 0.0), axis=0)


@triton.jit
def _column_score_kernel(
    q_ptr,
    k_ptr,
    lse_ptr,
    cu_q_ptr,
    cu_k_ptr,
    out_ptr,
    scale,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    k_seq_stride,
    lse_head_stride,
    lse_token_stride,
    out_token_stride,
    out_head_stride,
    out_seq_stride,
    SCORE_WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PACKED_OUT: tl.constexpr,
):
    """DiffKV prompt-phase score of every key token in a sequence.

    For each of the 4 query heads of the group, sum exp(q . key * scale - lse)
    over the last SCORE_WINDOW queries of the chunk that see the key; the
    token score is the maximum over the heads. Key j is visible to query i
    when j <= i + (len_k - len_q): the prefix fully, the chunk causally.
    The key sequence index is kv_head * k_seq_stride + sequence. With
    PACKED_OUT the output is indexed by packed key token, otherwise by
    sequence and the token's position within it.
    """
    sequence = tl.program_id(0)
    kv_head = tl.program_id(1)
    key_block = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + sequence)
    q_end = tl.load(cu_q_ptr + sequence + 1)
    k_index = kv_head * k_seq_stride + sequence
    k_start = tl.load(cu_k_ptr + k_index)
    k_end = tl.load(cu_k_ptr + k_index + 1)
    len_q = q_end - q_start
    len_k = k_end - k_start
    shift = len_k - len_q
    key_begin = key_block * BLOCK_N
    if key_begin >= len_k:
        return
    offs_n = key_begin + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, 128)
    mask_n = offs_n < len_k
    keys = tl.load(
        k_ptr
        + (k_start + offs_n)[:, None] * k_token_stride
        + kv_head * k_head_stride
        + offs_d[None, :],
        mask=mask_n[:, None],
        other=0.0,
    )
    keys_t = tl.trans(keys)
    head_q = q_ptr + q_start * q_token_stride + (kv_head * 4) * q_head_stride
    head_lse = lse_ptr + (kv_head * 4) * lse_head_stride + q_start * lse_token_stride
    first_scored = tl.maximum(len_q - SCORE_WINDOW, 0)
    first_query = tl.maximum(tl.maximum(key_begin - shift, 0), first_scored)
    sum0 = tl.zeros([BLOCK_N], dtype=tl.float32)
    sum1 = tl.zeros([BLOCK_N], dtype=tl.float32)
    sum2 = tl.zeros([BLOCK_N], dtype=tl.float32)
    sum3 = tl.zeros([BLOCK_N], dtype=tl.float32)
    for query_begin in range((first_query // BLOCK_M) * BLOCK_M, len_q, BLOCK_M):
        offs_m = query_begin + tl.arange(0, BLOCK_M)
        mask_m = (offs_m < len_q) & (offs_m >= first_scored)
        visible = (offs_m[:, None] + shift >= offs_n[None, :]) & mask_m[:, None]
        sum0 += _head_column_sum(
            head_q,
            head_lse,
            keys_t,
            offs_m,
            offs_d,
            mask_m,
            visible,
            scale,
            q_token_stride,
            lse_token_stride,
        )
        sum1 += _head_column_sum(
            head_q + q_head_stride,
            head_lse + lse_head_stride,
            keys_t,
            offs_m,
            offs_d,
            mask_m,
            visible,
            scale,
            q_token_stride,
            lse_token_stride,
        )
        sum2 += _head_column_sum(
            head_q + 2 * q_head_stride,
            head_lse + 2 * lse_head_stride,
            keys_t,
            offs_m,
            offs_d,
            mask_m,
            visible,
            scale,
            q_token_stride,
            lse_token_stride,
        )
        sum3 += _head_column_sum(
            head_q + 3 * q_head_stride,
            head_lse + 3 * lse_head_stride,
            keys_t,
            offs_m,
            offs_d,
            mask_m,
            visible,
            scale,
            q_token_stride,
            lse_token_stride,
        )
    score = tl.maximum(tl.maximum(sum0, sum1), tl.maximum(sum2, sum3))
    if PACKED_OUT:
        out_ptr += k_start * out_token_stride
    else:
        out_ptr += sequence * out_seq_stride
    tl.store(
        out_ptr + offs_n * out_token_stride + kv_head * out_head_stride,
        score,
        mask=mask_n,
    )


@triton.jit
def _scatter_prefix_scores_kernel(
    scores_ptr,
    cu_q_ptr,
    cu_k_ptr,
    prefix_len_ptr,
    high_len_ptr,
    table_offset_ptr,
    block_tables_ptr,
    score_cache_ptr,
    position_cache_ptr,
    num_requests,
    max_blocks,
    page_words,
    high_score_base,
    low_score_base,
    high_position_base,
    low_position_base,
    SCORE_WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Add each prefix token's score into its page score field and advance its
    cached position by the chunk queries that did not score it, so the decode
    kernel's query count stays equal to the number of scored queries.

    Sequence s is request s % num_requests of KV head s // num_requests. Pages
    follow the decode order: high-precision pages from the table start,
    low-precision pages from the table end.
    """
    sequence = tl.program_id(0)
    block = tl.program_id(1)
    request = sequence % num_requests
    len_q = tl.load(cu_q_ptr + request + 1) - tl.load(cu_q_ptr + request)
    unscored = tl.maximum(len_q - SCORE_WINDOW, 0)
    k_start = tl.load(cu_k_ptr + sequence)
    prefix_len = tl.load(prefix_len_ptr + sequence)
    high_len = tl.load(high_len_ptr + sequence)
    table = block_tables_ptr + tl.load(table_offset_ptr + sequence) * max_blocks
    offs = block * BLOCK + tl.arange(0, BLOCK)
    mask = offs < prefix_len
    scores = tl.load(scores_ptr + k_start + offs, mask=mask, other=0.0)
    is_high = offs < high_len
    low_token = offs - high_len
    high_page = tl.load(table + offs // 32, mask=mask & is_high, other=0)
    low_page = tl.load(
        table + max_blocks - 1 - low_token // 60, mask=mask & (~is_high), other=0
    )
    page_base = tl.where(is_high, high_page, low_page) * page_words
    slot = tl.where(is_high, offs % 32, low_token % 60)
    score_address = (
        page_base + tl.where(is_high, high_score_base, low_score_base) + slot
    )
    position_address = (
        page_base + tl.where(is_high, high_position_base, low_position_base) + slot
    )
    previous = tl.load(score_cache_ptr + score_address, mask=mask, other=0.0)
    tl.store(score_cache_ptr + score_address, previous + scores, mask=mask)
    position = tl.load(position_cache_ptr + position_address, mask=mask, other=0)
    tl.store(position_cache_ptr + position_address, position + unscored, mask=mask)


def column_scores(
    query,
    key,
    lse,
    cu_q,
    cu_k,
    max_len_k,
    scale,
    out,
    *,
    kv_heads,
    q_head_stride,
    k_head_stride,
    k_seq_stride,
    lse_head_stride,
    lse_token_stride,
    out_token_stride,
    out_head_stride,
    out_seq_stride=0,
    block_m=32,
    block_n=64,
):
    grid = (cu_q.numel() - 1, kv_heads, triton.cdiv(max_len_k, block_n))
    _column_score_kernel[grid](
        query,
        key,
        lse,
        cu_q,
        cu_k,
        out,
        scale,
        query.stride(0),
        q_head_stride,
        key.stride(0),
        k_head_stride,
        k_seq_stride,
        lse_head_stride,
        lse_token_stride,
        out_token_stride,
        out_head_stride,
        out_seq_stride,
        SCORE_WINDOW=SCORE_WINDOW,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        PACKED_OUT=out_seq_stride == 0,
        num_stages=1,
        num_warps=4,
    )


@dataclass(frozen=True)
class LayerPlan:
    """Row layout of the G*R (KV head, request) key sequences for one layer."""

    cu_k: torch.Tensor
    max_len_k: int
    max_prefix: int
    slices: int
    row_offsets: torch.Tensor
    chunk_rows: torch.Tensor
    prefix_lengths: torch.Tensor
    high_lengths: torch.Tensor
    table_offsets: torch.Tensor
    total_rows: int
    group_rows: list[tuple[int, int]]
    group_cu_k: list[torch.Tensor]
    group_max_len_k: list[int]


@dataclass(frozen=True)
class ChunkPlan:
    """Per-step bookkeeping shared by every layer."""

    tokens: int
    num_requests: int
    kv_heads: int
    max_seqlen: int
    slot_ids: torch.Tensor | None
    layers: list[LayerPlan] | None


def plan_chunk_prefill(
    cu_seqlens_cpu, slot_ids_cpu, kv_len_tables_cpu, device, kv_heads=None
):
    """Lay out the step's sequences from the per-layer, per-head prefix lengths.

    cu_seqlens_cpu: chunk boundaries in request order. slot_ids_cpu: the
    cache slot of each request. kv_len_tables_cpu: int32 [requests, layers,
    kv_heads, 2] retained (high, low) lengths per request slot, or None when no
    request has a prefix (every chunk starts its prompt).
    """
    num_requests = len(cu_seqlens_cpu) - 1
    if len(slot_ids_cpu) != num_requests:
        raise ValueError("One slot per request is required")
    lengths = [cu_seqlens_cpu[r + 1] - cu_seqlens_cpu[r] for r in range(num_requests)]
    if min(lengths) <= 0:
        raise ValueError("Every prefill chunk must contain at least one token")
    tokens = cu_seqlens_cpu[-1]
    if kv_len_tables_cpu is None:
        if kv_heads is None:
            raise ValueError("kv_heads is required without prefix tables")
        return ChunkPlan(tokens, num_requests, kv_heads, max(lengths), None, None)
    kv_len = kv_len_tables_cpu
    if kv_len.shape[0] != num_requests or kv_len.shape[3] != 2:
        raise ValueError("kv_len_tables_cpu must be [requests, layers, kv_heads, 2]")
    num_layers, kv_heads = kv_len.shape[1], kv_len.shape[2]
    chunk_slices = [
        (length + CHUNK_SLICE_TOKENS - 1) // CHUNK_SLICE_TOKENS for length in lengths
    ]
    layers = []
    for layer in range(num_layers):
        high = kv_len[:, layer, :, 0].tolist()
        low = kv_len[:, layer, :, 1].tolist()
        seq_len_k, prefix_list, high_list, slices = [], [], [], 0
        for g in range(kv_heads):
            for r in range(num_requests):
                prefix = high[r][g] + low[r][g]
                seq_len_k.append(prefix + lengths[r])
                prefix_list.append(prefix)
                high_list.append(high[r][g])
                pages = (high[r][g] + HIGH_PAGE_TOKENS - 1) // HIGH_PAGE_TOKENS
                pages += (low[r][g] + LOW_PAGE_TOKENS - 1) // LOW_PAGE_TOKENS
                slices = max(slices, pages + chunk_slices[r])
        cu_k = (
            torch.tensor([0] + seq_len_k, dtype=torch.int32).cumsum(0).to(torch.int32)
        )
        chunk_rows = torch.cat(
            [
                torch.arange(
                    int(cu_k[s]) + prefix_list[s], int(cu_k[s + 1]), dtype=torch.int64
                )
                for s in range(kv_heads * num_requests)
            ]
        )
        table_offsets = torch.tensor(
            [
                (slot_ids_cpu[s % num_requests] * num_layers + layer) * kv_heads
                + s // num_requests
                for s in range(kv_heads * num_requests)
            ],
            dtype=torch.int32,
        )
        group_rows, group_cu_k, group_max_len_k = [], [], []
        for g in range(kv_heads):
            begin, end = g * num_requests, (g + 1) * num_requests + 1
            relative = cu_k[begin:end] - cu_k[begin]
            group_rows.append((int(cu_k[begin]), int(cu_k[end - 1])))
            group_cu_k.append(relative.contiguous().to(device))
            group_max_len_k.append(max(seq_len_k[begin : end - 1]))
        layers.append(
            LayerPlan(
                cu_k.to(device),
                max(seq_len_k),
                max(prefix_list),
                max(1, (slices + SLICES_PER_BLOCK - 1) // SLICES_PER_BLOCK),
                cu_k[:-1].contiguous().to(device),
                chunk_rows.to(device),
                torch.tensor(prefix_list, dtype=torch.int32, device=device),
                torch.tensor(high_list, dtype=torch.int32, device=device),
                table_offsets.to(device),
                int(cu_k[-1]),
                group_rows,
                group_cu_k,
                group_max_len_k,
            )
        )
    return ChunkPlan(
        tokens,
        num_requests,
        kv_heads,
        max(lengths),
        torch.tensor(slot_ids_cpu, dtype=torch.int32, device=device),
        layers,
    )


def _direct_prefill(query, key, value, cu_seqlens, plan, scale, fa_version):
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    out, lse = flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=plan.max_seqlen,
        max_seqlen_k=plan.max_seqlen,
        softmax_scale=scale,
        causal=True,
        fa_version=fa_version,
        return_softmax_lse=True,
    )
    scores = torch.empty(
        (plan.tokens, plan.kv_heads), dtype=torch.float32, device=query.device
    )
    column_scores(
        query,
        key,
        lse,
        cu_seqlens,
        cu_seqlens,
        plan.max_seqlen,
        scale,
        scores,
        kv_heads=plan.kv_heads,
        q_head_stride=query.stride(1),
        k_head_stride=key.stride(1),
        k_seq_stride=0,
        lse_head_stride=lse.stride(0),
        lse_token_stride=lse.stride(1),
        out_token_stride=scores.stride(0),
        out_head_stride=scores.stride(1),
    )
    return out, scores, lse.transpose(0, 1)


def chunk_prefill_attention(
    query,
    key,
    value,
    cu_seqlens,
    plan,
    kv_cache,
    block_tables,
    kv_len_tables,
    layer_index,
    scale,
    fa_version=2,
):
    """Attention output and scores for the prefill chunks of one layer.

    query, key, value: FP16 [tokens, heads, 128] activations of this layer for
    all chunks, concatenated in request order with boundaries cu_seqlens.

    Returns (output FP16 [tokens, heads, 128], chunk token scores FP32
    [tokens, kv_heads], log-sum-exp FP32 [tokens, heads]).
    """
    if plan.layers is None:
        return _direct_prefill(query, key, value, cu_seqlens, plan, scale, fa_version)
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    layer = plan.layers[layer_index]
    tokens, heads, _ = query.shape
    kv_heads = plan.kv_heads
    device = query.device
    key_rows = torch.empty(
        (layer.total_rows, 1, HEAD_SIZE), dtype=torch.float16, device=device
    )
    value_rows = torch.empty_like(key_rows)
    get_extension().ops.dequantize_prefix(
        plan.slot_ids,
        layer.row_offsets,
        block_tables,
        kv_len_tables,
        kv_cache,
        layer_index,
        key,
        value,
        cu_seqlens,
        key_rows.view(-1, HEAD_SIZE),
        value_rows.view(-1, HEAD_SIZE),
        layer.slices,
    )
    output = torch.empty_like(query)
    lse = torch.empty((heads, tokens), dtype=torch.float32, device=device)
    for g in range(kv_heads):
        begin, end = layer.group_rows[g]
        group_heads = slice(g * QUERIES_PER_KV, (g + 1) * QUERIES_PER_KV)
        _, group_lse = flash_attn_varlen_func(
            q=query[:, group_heads],
            k=key_rows[begin:end],
            v=value_rows[begin:end],
            out=output[:, group_heads],
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=layer.group_cu_k[g],
            max_seqlen_q=plan.max_seqlen,
            max_seqlen_k=layer.group_max_len_k[g],
            softmax_scale=scale,
            causal=True,
            fa_version=fa_version,
            return_softmax_lse=True,
        )
        lse[group_heads] = group_lse
    scores = torch.empty((layer.total_rows,), dtype=torch.float32, device=device)
    column_scores(
        query,
        key_rows,
        lse,
        cu_seqlens,
        layer.cu_k,
        layer.max_len_k,
        scale,
        scores,
        kv_heads=kv_heads,
        q_head_stride=query.stride(1),
        k_head_stride=0,
        k_seq_stride=plan.num_requests,
        lse_head_stride=lse.stride(0),
        lse_token_stride=lse.stride(1),
        out_token_stride=1,
        out_head_stride=0,
    )
    sequences = kv_heads * plan.num_requests
    if layer.max_prefix > 0:
        _scatter_prefix_scores_kernel[(sequences, triton.cdiv(layer.max_prefix, 256))](
            scores,
            cu_seqlens,
            layer.cu_k,
            layer.prefix_lengths,
            layer.high_lengths,
            layer.table_offsets,
            block_tables,
            kv_cache.view(torch.float32),
            kv_cache.view(torch.int32),
            plan.num_requests,
            block_tables.shape[3],
            PAGE_WORDS,
            HIGH_SCORE_BASE,
            LOW_SCORE_BASE,
            HIGH_POSITION_BASE,
            LOW_POSITION_BASE,
            SCORE_WINDOW=SCORE_WINDOW,
            BLOCK=256,
        )
    chunk_scores = (
        scores[layer.chunk_rows].view(kv_heads, tokens).transpose(0, 1).contiguous()
    )
    return output, chunk_scores, lse.transpose(0, 1)


def prompt_attention(query, key, value, cu_seqlens, max_seqlen, scale, out=None):
    """Causal FA2 over whole prompts; returns the output and log-sum-exp."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        out=out,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=scale,
        causal=True,
        fa_version=2,
        return_softmax_lse=True,
    )


def prompt_scores(query, key, lse, cu_seqlens, max_seqlen, scale):
    """Group-maximum token scores laid out [requests, KV heads, max_seqlen]."""
    return get_extension().cache_ops.prompt_scores(
        query, key, lse, cu_seqlens, max_seqlen, scale
    )


def compress_prompts(
    key,
    value,
    scores,
    cu_seqlens,
    kv_cache,
    slot_ids,
    block_tables,
    kv_len_tables,
    compress_config_tables,
    layer_idx,
):
    """One prompt-phase writer call for every request of the step."""
    get_extension().cache_ops.compress_and_append_cache_prompt_phase(
        key,
        value,
        scores,
        kv_cache,
        slot_ids,
        block_tables,
        kv_len_tables,
        cu_seqlens,
        compress_config_tables,
        KV_BUFFER_SIZE,
        layer_idx,
        *QUANT_ARGS,
    )


def prompt_prefill(
    query,
    key,
    value,
    cu_seqlens,
    max_seqlen,
    scale,
    kv_cache,
    slot_ids,
    block_tables,
    kv_len_tables,
    compress_config_tables,
    layer_idx,
    out=None,
):
    """Attention, scoring and compression of whole prompts, one launch each.

    Every request's prompt is complete in this step, so the keys are the
    step's own activations. FA2 returns the log-sum-exp; one extension call
    turns it into per-token scores laid out [requests, KV heads, max_seqlen]
    (prompt_scores) and compresses all requests into the cache with the
    prompt-phase writer (compress_prompts). key and value may be strided
    views; the kernels read their token stride.
    """
    out, lse = prompt_attention(query, key, value, cu_seqlens, max_seqlen, scale, out)
    scores = get_extension().cache_ops.score_and_compress_prompts(
        query,
        key,
        value,
        lse,
        cu_seqlens,
        max_seqlen,
        scale,
        kv_cache,
        slot_ids,
        block_tables,
        kv_len_tables,
        compress_config_tables,
        KV_BUFFER_SIZE,
        layer_idx,
        *QUANT_ARGS,
    )
    return out, scores
