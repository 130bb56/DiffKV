# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill latency of one layer for whole prompts (chunked prefill off).

One step holds `batch` requests whose prompts of `prompt` tokens are all
processed in that step. Query, key and value are views of one QKV projection
output, as the vLLM attention layer receives them. Every config includes writing
the prompt's KV into its cache:

- diffkv_serving: the current DiffKV backend path (backend.py), per request:
  copy KV into the raw prompt buffer, FA2, final-64 query scores with a matmul
  and softmax per KV head, prompt-phase compression into the packed cache.
- diffkv: diffkv.prefill.prompt_prefill, one FA2 varlen call returning the
  log-sum-exp, one score kernel and one prompt-phase compression call.
- fa2_fp16: reshape_and_cache_flash into a paged FP16 cache and paged FA2,
  as the vLLM FLASH_ATTN backend does.
- flashinfer_fp8: reshape_and_cache_flash into a paged fp8_e4m3 cache and the
  FlashInfer paged prefill kernel with FP16 queries, as the vLLM FLASHINFER
  backend does on SM86.

With --parts, prompt_prefill's attention, score and compression launches are
also timed separately (diffkv_part_*).

Cache allocation, request admission and FlashInfer planning are untimed.
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from attention.benchmark_latency import BLOCK_SIZE, Config, measure, timing_backend
from attention.correctness import (
    HEAD_SIZE,
    LAYER_IDX,
    SCALE,
    default_run_dir,
    metadata,
)

QUERIES_PER_KV = 4


def qkv_inputs(lengths, kv_heads, seed=23, query_scale=0.12, aligned=0.0):
    """Query, key and value as strided views of one [tokens, QKV] tensor.

    With `aligned` > 0, every query of a KV head group gets `aligned` times a
    shared direction and every key a random multiple of it, so a key's logit
    carries a per-token offset of about aligned**2 standard deviations for
    all queries alike; tokens with a negative offset score low everywhere.
    """
    generator = torch.Generator().manual_seed(seed)
    heads = kv_heads * QUERIES_PER_KV
    tokens = sum(lengths)
    q_size = heads * HEAD_SIZE
    kv_size = kv_heads * HEAD_SIZE
    qkv = torch.empty((tokens, q_size + 2 * kv_size), dtype=torch.float16)
    query = torch.randn((tokens, q_size), generator=generator) * query_scale
    key_value = torch.randn((tokens, 2 * kv_size), generator=generator) * 0.5
    if aligned:
        direction = torch.randn((kv_heads, 1, HEAD_SIZE), generator=generator)
        offsets = torch.randn((tokens, kv_heads, 1), generator=generator)
        query += (aligned * direction).expand(-1, QUERIES_PER_KV, -1).reshape(-1)
        key_value[:, :kv_size] += (
            aligned * offsets * direction.squeeze(1) / HEAD_SIZE**0.5
        ).reshape(tokens, kv_size)
    qkv[:, :q_size] = query
    qkv[:, q_size:] = key_value
    qkv = qkv.cuda()
    query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)
    starts = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return (
        query.view(tokens, heads, HEAD_SIZE),
        key.view(tokens, kv_heads, HEAD_SIZE),
        value.view(tokens, kv_heads, HEAD_SIZE),
        starts,
    )


def admit_requests(lengths, kv_heads):
    """A cache controller with one admitted prompt per length, at prefill start."""
    from diffkv.cache import DiffKVCacheController
    from diffkv.ops import PAGE_BYTES, PAGE_SIZE

    num_layers = LAYER_IDX + 1
    num_kv_caches = num_layers * kv_heads
    pages = 16
    for prompt in lengths:
        pages += (-(-prompt // 32) + 1) * num_kv_caches
        pages += -(-prompt * num_kv_caches * (HEAD_SIZE * 2 * 2 + 4 * 4) // PAGE_BYTES)
    controller = DiffKVCacheController(
        max_num_requests=len(lengths),
        num_layers=num_layers,
        num_kv_heads=kv_heads,
        max_model_len=max(lengths),
        arena=torch.zeros((pages, PAGE_SIZE), dtype=torch.int16, device="cuda"),
    )
    request_ids = [f"r{index}" for index in range(len(lengths))]
    for request_id, prompt in zip(request_ids, lengths):
        controller.begin_request(request_id, prompt)
    return controller, request_ids


def serving_prefill(controller, request_ids, starts, query, key, value, output):
    """The DiffKV backend prefill of one layer, prompts complete in this step.

    Mirrors DiffKVImpl.forward for prefill requests with prefix_start 0,
    including the score reset that begin_request performs.
    """
    from diffkv.backend import DiffKVImpl, accumulate_prompt_scores, causal_prefill

    impl = SimpleNamespace(num_kv_heads=key.shape[1])
    for row, req_id in enumerate(request_ids):
        start, end = starts[row], starts[row + 1]
        raw = controller.raw_prompt(req_id)[LAYER_IDX]
        raw[0].copy_(key[start:end])
        raw[1].copy_(value[start:end])
        output[start:end].copy_(causal_prefill(query[start:end], raw[0], raw[1], SCALE))
        scores = controller.prompt_scores(req_id)[LAYER_IDX]
        scores.zero_()
        accumulate_prompt_scores(
            scores, query[start:end], raw[0], 0, end - start, SCALE
        )
        DiffKVImpl._compress_prompt(impl, controller, req_id, LAYER_IDX, raw, scores)
    return output


def cache_bytes(controller):
    return controller.kv_cache.numel() * controller.kv_cache.element_size()


def make_diffkv_serving(batch, prompt, kv_heads):
    lengths = [prompt] * batch
    query, key, value, starts = qkv_inputs(lengths, kv_heads)
    controller, request_ids = admit_requests(lengths, kv_heads)
    output = torch.empty_like(query)
    bounds = starts.tolist()

    def run(q, k, v):
        serving_prefill(controller, request_ids, bounds, q, k, v, output)

    run(query, key, value)
    return Config(
        run,
        (query, key, value),
        output,
        cache_bytes(controller),
        {
            "backend": "DIFFKV",
            "path": "diffkv.backend (DiffKVImpl.forward prefill branch)",
            "parts": [
                "per request: key and value copy into the raw prompt buffer",
                "per request: FA2 (causal_prefill)",
                "per request: final-64 query scores (accumulate_prompt_scores)",
                "per request: compress_and_append_cache_prompt_phase",
            ],
        },
        {},
    )


def make_diffkv(batch, prompt, kv_heads):
    from diffkv.prefill import prompt_prefill

    lengths = [prompt] * batch
    query, key, value, starts = qkv_inputs(lengths, kv_heads)
    controller, request_ids = admit_requests(lengths, kv_heads)
    slot_ids = controller.metadata(request_ids).slot_ids
    cu_seqlens = starts.cuda()
    output = torch.empty_like(query)

    def run(q, k, v):
        prompt_prefill(
            q,
            k,
            v,
            cu_seqlens,
            prompt,
            SCALE,
            controller.kv_cache,
            slot_ids,
            controller.block_tables,
            controller.kv_len_tables,
            controller.compress_config_tables,
            LAYER_IDX,
            out=output,
        )

    run(query, key, value)
    return Config(
        run,
        (query, key, value),
        output,
        cache_bytes(controller),
        {
            "backend": "DIFFKV",
            "path": "diffkv.prefill.prompt_prefill",
            "parts": [
                "FA2 varlen over all requests with log-sum-exp",
                "column score kernel (Triton), scores [requests, KV heads, prompt]",
                "compress_and_append_cache_prompt_phase for all requests",
            ],
        },
        {},
    )


def make_diffkv_parts(batch, prompt, kv_heads):
    """prompt_prefill's three launches as separate configs, on the same inputs."""
    from diffkv.prefill import compress_prompts, prompt_attention, prompt_scores

    lengths = [prompt] * batch
    query, key, value, starts = qkv_inputs(lengths, kv_heads)
    controller, request_ids = admit_requests(lengths, kv_heads)
    slot_ids = controller.metadata(request_ids).slot_ids
    cu_seqlens = starts.cuda()
    output = torch.empty_like(query)
    _, lse = prompt_attention(query, key, value, cu_seqlens, prompt, SCALE, output)
    scores = prompt_scores(query, key, lse, cu_seqlens, prompt, SCALE)

    def attention(q, k, v):
        prompt_attention(q, k, v, cu_seqlens, prompt, SCALE, output)

    def score(q, k, lse):
        prompt_scores(q, k, lse, cu_seqlens, prompt, SCALE)

    def compress(k, v, scores):
        compress_prompts(
            k,
            v,
            scores,
            cu_seqlens,
            controller.kv_cache,
            slot_ids,
            controller.block_tables,
            controller.kv_len_tables,
            controller.compress_config_tables,
            LAYER_IDX,
        )

    compress(key, value, scores)
    parts = {
        "diffkv_part_attention": (attention, (query, key, value), "prompt_attention"),
        "diffkv_part_scores": (score, (query, key, lse), "prompt_scores"),
        "diffkv_part_compress": (compress, (key, value, scores), "compress_prompts"),
    }
    return {
        name: Config(
            function,
            arguments,
            output,
            cache_bytes(controller),
            {"backend": "DIFFKV", "path": f"diffkv.prefill.{path}"},
            {},
        )
        for name, (function, arguments, path) in parts.items()
    }


def paged_layout(batch, prompt):
    pages_per_request = -(-prompt // BLOCK_SIZE)
    pages = batch * pages_per_request
    tables = torch.arange(pages, dtype=torch.int32, device="cuda").reshape(
        batch, pages_per_request
    )
    slots = torch.cat(
        [
            torch.arange(prompt, dtype=torch.int64)
            + request * pages_per_request * BLOCK_SIZE
            for request in range(batch)
        ]
    ).cuda()
    return pages_per_request, pages, tables, slots


def make_fa2(batch, prompt, kv_heads):
    from vllm import _custom_ops
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    lengths = [prompt] * batch
    query, key, value, starts = qkv_inputs(lengths, kv_heads)
    _, pages, tables, slots = paged_layout(batch, prompt)
    shape = (pages, BLOCK_SIZE, kv_heads, HEAD_SIZE)
    key_cache = torch.zeros(shape, dtype=torch.float16, device="cuda")
    value_cache = torch.zeros_like(key_cache)
    cu_seqlens_q = starts.cuda()
    seqused_k = torch.full((batch,), prompt, dtype=torch.int32, device="cuda")
    output = torch.empty_like(query)
    scales = torch.ones((), dtype=torch.float32, device="cuda")

    def run(q, k, v):
        _custom_ops.reshape_and_cache_flash(
            k, v, key_cache, value_cache, slots, "auto", scales, scales
        )
        flash_attn_varlen_func(
            q=q,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=prompt,
            max_seqlen_k=prompt,
            seqused_k=seqused_k,
            block_table=tables,
            softmax_scale=SCALE,
            causal=True,
            fa_version=2,
        )

    run(query, key, value)
    return Config(
        run,
        (query, key, value),
        output,
        key_cache.numel() * key_cache.element_size() * 2,
        {
            "backend": "FLASH_ATTN",
            "fa_version": 2,
            "cache_dtype": "float16",
            "parts": ["reshape_and_cache_flash", "FA2 varlen, paged KV"],
            "page_tokens": BLOCK_SIZE,
        },
        {},
    )


def make_fp8(batch, prompt, kv_heads):
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    from vllm import _custom_ops

    lengths = [prompt] * batch
    query, key, value, starts = qkv_inputs(lengths, kv_heads)
    pages_per_request, pages, _, slots = paged_layout(batch, prompt)
    # The serving baseline resolves the LBNHC layout, exposed to FlashInfer as NHD.
    arena = torch.zeros(
        (pages, BLOCK_SIZE, kv_heads, 2 * HEAD_SIZE),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    key_cache, value_cache = arena.split(HEAD_SIZE, dim=-1)
    scales = torch.ones((), dtype=torch.float32, device="cuda")
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        qo_indptr=starts,
        paged_kv_indptr=torch.arange(batch + 1, dtype=torch.int32) * pages_per_request,
        paged_kv_indices=torch.arange(pages, dtype=torch.int32, device="cuda"),
        paged_kv_last_page_len=torch.full(
            (batch,), (prompt - 1) % BLOCK_SIZE + 1, dtype=torch.int32
        ),
        num_qo_heads=query.shape[1],
        num_kv_heads=kv_heads,
        head_dim_qk=HEAD_SIZE,
        page_size=BLOCK_SIZE,
        causal=True,
        sm_scale=SCALE,
        window_left=-1,
        logits_soft_cap=0.0,
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.float16,
    )
    output = torch.empty_like(query)

    def run(q, k, v):
        _custom_ops.reshape_and_cache_flash(
            k, v, key_cache, value_cache, slots, "fp8_e4m3", scales, scales
        )
        wrapper.run(
            q,
            (key_cache, value_cache),
            q_scale=1.0,
            k_scale=1.0,
            v_scale=1.0,
            out=output,
        )

    run(query, key, value)
    return Config(
        run,
        (query, key, value),
        output,
        arena.numel() * arena.element_size(),
        {
            "backend": "FLASHINFER",
            "cache_dtype": "fp8_e4m3",
            "query_dtype": "float16",
            "parts": [
                "reshape_and_cache_flash (fp8_e4m3)",
                "BatchPrefillWithPagedKVCacheWrapper, causal, paged KV",
            ],
            "flashinfer_resolved_backend": wrapper._backend,
            "scale_handling": "per-tensor key and value scale 1.0, no calibration",
            "page_tokens": BLOCK_SIZE,
        },
        {},
    )


CONFIGS = {
    "diffkv_serving": make_diffkv_serving,
    "diffkv": make_diffkv,
    "fa2_fp16": make_fa2,
    "flashinfer_fp8": make_fp8,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prompts", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--repeat-ms", type=int, default=500)
    parser.add_argument(
        "--parts",
        action="store_true",
        help="also time prompt_prefill's attention, score and compression launches",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS),
        default=list(CONFIGS),
        help="configs to time; the default times all of them in one process, "
        "which is the only way to compare them on one device",
    )
    parser.add_argument(
        "--output", type=Path, help="default: logs/<UTC>-prefill-b<batch>/latency.json"
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = default_run_dir(f"prefill-b{args.batch}") / "latency.json"
    if min(args.batch, args.kv_heads, args.repeat_ms, *args.prompts) < 1:
        parser.error("Batch, heads, repeat-ms and prompt lengths must be positive")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    from diffkv.ops import get_extension

    torch.set_num_threads(2)
    extension = get_extension()
    rows = []
    for prompt in args.prompts:
        configs = {
            name: CONFIGS[name](args.batch, prompt, args.kv_heads)
            for name in args.configs
        }
        if args.parts:
            configs.update(make_diffkv_parts(args.batch, prompt, args.kv_heads))
        samples = {name: [] for name in configs}
        order = list(configs)
        for name in [*order, *reversed(order)]:
            config = configs[name]
            samples[name].extend(
                measure(config.function, config.arguments, args.repeat_ms)
            )
        medians = {name: statistics.median(values) for name, values in samples.items()}
        row = {
            "batch": args.batch,
            "prompt": prompt,
            "kv_heads": args.kv_heads,
            "cuda_graph": False,
            "configs": {
                name: {
                    "details": config.details,
                    "cache_object_bytes": config.cache_bytes,
                    "median_us": medians[name],
                    "samples_us": samples[name],
                }
                for name, config in configs.items()
            },
        }
        if {"diffkv", "flashinfer_fp8"} <= medians.keys():
            row["diffkv_over_fp8_ratio"] = medians["diffkv"] / medians["flashinfer_fp8"]
        rows.append(row)
        print(json.dumps({"prompt": prompt, "median_us": medians}), flush=True)
        del configs
        torch.accelerator.empty_cache()
    report = {
        "status": "completed",
        "metadata": metadata(extension),
        "command": [sys.executable, *sys.argv],
        "timing_backend": timing_backend(),
        "execution_mode": "eager",
        "cuda_graph": False,
        "cold_l2_cache": True,
        "timing_scope": "CUDA events around a direct eager call; L2 flush excluded; "
        "host launch gaps within the event interval are included",
        "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING"),
        "flashinfer_extra_cudaflags": os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS"),
        "operation": "prefill of one layer for whole prompts (chunked prefill "
        "off): attention plus writing the prompt KV into each config's cache",
        "comparison_scope": "one layer for one step; no accuracy check here "
        "(correctness.py checks diffkv against diffkv_serving)",
        "configs_timed": list(args.configs),
        "excluded": [
            "cache allocation",
            "request admission",
            "FlashInfer plan",
            "compilation",
            "input generation",
        ],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"status": report["status"], "output": str(args.output)}, indent=2)
    )


if __name__ == "__main__":
    main()
