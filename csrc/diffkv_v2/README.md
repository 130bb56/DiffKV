# DiffKV v2 native operators

These operators port the production decode and cache format from DiffKV commit
`c489dc1`, including the v2 register-GQA main kernel and split score reducer.
They compile into `_diffkv_v2_C`, with private `diffkv_v2` CUDA namespaces. They
do not replace vLLM's `_C` extension or its separate `FLASH_ATTN_DIFFKV` backend.

The supported configuration is FP16, head dimension 128, four query heads per
KV head, K8V4 high precision, K4V2 low precision, one quantization chunk, and
32/60 tokens per physical page. Unsupported configurations fail explicitly.
ALiBi, block sharing, and BF16 are outside this initial integration.

Build from the editable checkout using the active vLLM environment:

```sh
MAX_JOBS=2 .venv/bin/python -m vllm.diffkv_v2.build_ops
```

The CUDA toolkit must match the active PyTorch CUDA major version. Set
`CUDA_HOME` before importing the build helper when a nondefault toolkit is
needed. `TORCH_CUDA_ARCH_LIST` permits compilation without a visible GPU.
`VLLM_DIFFKV_V2_BUILD_DIR` can relocate the build cache. Source, loader, Python
ABI, PyTorch, compiler, and GPU architecture participate in its fingerprint.
The build preserves the original `-O2` setting without enabling fast math.

`vllm.diffkv_v2.ops.get_extension()` exposes the legacy positional signatures:

- `ops.sparse_paged_attention_gqa`, including importance-score updates.
- `cache_ops.compress_and_append_cache_prompt_phase`,
  `cache_ops.compress_and_append_cache_long_prompt_phase`,
  `cache_ops.compress_and_append_cache_decode_phase`.
- `mem_mgt_ops.allocate_seqs`, `prepare_free_prompt`, `free_prompt`,
  `prepare_append_seqs`, `append_seqs`, `prepare_free_seqs`, `free_seqs`.

Cache storage is contiguous `int16[pages, 3392]`, 6,784 bytes per page, with an
8-byte aligned base. Block tables are contiguous CUDA int32 tensors with shape
`[slots, layers, KV heads, table capacity]`. High pages grow from the left and
low pages from the right after prompt compaction. Length and allocation-count
tables have shape `[slots, layers, KV heads, 2]`. Scalar compression thresholds
are FP32 `[slots, 2]`. K and V passed to writers must be contiguous FP16 tensors.

The short-prompt writer consumes FP32 scores `[batch, query heads, max prompt]`.
The long-prompt writer consumes scores already maximized across GQA heads,
`[batch, KV heads, max prompt]`, and sorted int32 indices of the same shape.
The decode writer uses preceding-step importance to prune, downgrade, and
append. Attention then updates importance using the maximum probability across
the four query heads. The stored position field tracks importance age and is
not necessarily the original token position.

The caller owns valid slot IDs, table capacity, exclusive request ownership,
free-ring accounting, prefix sums, and resetting a freed slot's metadata.
Prompt-release prefix workspaces need `batch * layers * KV heads + 1` entries;
decode append and complete-request release need twice that head count plus one.
All kernels run on PyTorch's current stream, guarded to the input device.

Kernel validation and a controlled FP16 comparison live in
`benchmarks/diffkv_v2/kernel_check.py` and `benchmarks/diffkv_v2/latency.py`:

```sh
.venv/bin/python -m benchmarks.diffkv_v2.kernel_check --output kernel-check.json
.venv/bin/python -m benchmarks.diffkv_v2.latency --output kernel-latency.json
```

The FP64 oracle checks outputs, repeated importance increments, and unchanged
non-score cache bytes. The latency comparison feeds the same dequantized,
retained KV set to the target vLLM paged FlashAttention 2 kernel. It isolates
kernel execution and does not estimate the quality or serving benefit of
pruning the full original context. DiffKV timing includes its score update.
Both sides use CUDA graphs and cold-L2 timing through FlashInfer, with the
actual CUPTI or CUDA-event timing method recorded. Input generation,
allocation, compression, and transfers are excluded. The two arms use their
production configurations without claiming a globally tuned kernel optimum.
