"""Exercise packed DiffKV decode attention without loading model weights.

Run directly so CUDA-free hosts report explicit skips:
    PYTHONPATH=. .cache/baselines/fp16-env/bin/python \
        tests/kernels/test_diffkv_decode_attention.py

The reference uses one full FP64 softmax over logical K/V. It does not use
the CUDA partition mapping, partial softmax sums, or the prompt cache kernel.
"""

from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


HEAD_SIZE = 128
QUERY_HEADS_PER_KV = 4
PAGE_SIZE = 3392  # uint16 elements
LAYER_IDX = 1
SCALE = HEAD_SIZE**-0.5
QUANT_ARGS = (8, 4, 4, 2, 1, 1, 1, 1, 4, 2, 32, 60)
PARTITION_SIZE = 5760
MODES = ("legacy", "gqa")


def _align(value, boundary):
    return (value + boundary - 1) // boundary * boundary


def page_layout(k_bits, v_bits, tokens):
    """Describe the public unified-page format, in uint16 elements."""
    k_packs, v_packs = HEAD_SIZE * k_bits // 16, HEAD_SIZE * v_bits // 16
    k_meta = _align(tokens * k_packs * 2, 32) // 2
    value = _align(k_meta * 2 + tokens * 4, 128) // 2
    padded_tokens = _align(tokens, 2)
    v_meta = value + _align(padded_tokens * v_packs * 2, 32) // 2
    score = v_meta + _align(tokens * 4, 32) // 2
    return SimpleNamespace(k_meta=k_meta, value=value, v_meta=v_meta,
                           score=score, position=score + tokens * 2,
                           k_packs=k_packs, v_packs=v_packs,
                           padded_tokens=padded_tokens)


def _pack_codes(codes, bits):
    width = 16 // bits
    grouped = codes.reshape(codes.shape[0], HEAD_SIZE // width, width).to(torch.int64)
    shifts = torch.arange(width, dtype=torch.int64) * bits
    return torch.sum(grouped << shifts, dim=-1).to(torch.int16)


def _write_region(cache, pages, count, bits, generator, score_mask,
                  adversarial_max=False, compute_reference=True):
    """Pack integer codes independently of the CUDA cache-append kernels."""
    k_bits, v_bits, tokens = bits
    layout = page_layout(k_bits, v_bits, tokens)
    k_codes = torch.randint(0, 1 << k_bits, (count, HEAD_SIZE), generator=generator)
    v_codes = torch.randint(0, 1 << v_bits, (count, HEAD_SIZE), generator=generator)
    # Different region logits and different query heads make incorrect low
    # partition normalization and reading another head before rescaling visible.
    k_codes[:, :8] = 3 if k_bits == 8 else 14
    if adversarial_max and k_bits == 8:
        k_codes[:, 0] = 0
        if count > 1:
            k_codes[1, 0] = 255
    k_scales = torch.full((count,), 1 / 64 if k_bits == 8 else 1 / 4,
                          dtype=torch.float16)
    v_scales = torch.full((count,), 1 / 128 if v_bits == 4 else 1 / 2,
                          dtype=torch.float16)
    k_zeros = torch.full((count,), -2, dtype=torch.float16)
    v_zeros = torch.full((count,), -0.5, dtype=torch.float16)
    # Dyadic metadata makes dequantization exactly representable in FP32,
    # keeping the reference independent of quantizer rounding differences.
    keys = values = None
    if compute_reference:
        keys = k_codes.double() * k_scales.double()[:, None] + k_zeros.double()[:, None]
        values = v_codes.double() * v_scales.double()[:, None] + v_zeros.double()[:, None]
    k_packed, v_packed = _pack_codes(k_codes, k_bits), _pack_codes(v_codes, v_bits)
    addresses = []
    for page_idx, page in enumerate(pages):
        begin, end = page_idx * tokens, min((page_idx + 1) * tokens, count)
        n = end - begin
        # K storage axes: [pack row, token, thread group, vector element].
        k_page = torch.full((tokens, layout.k_packs), 0x3555, dtype=torch.int16)
        k_page[:n] = k_packed[begin:end]
        k_page = k_page.reshape(tokens, 4, layout.k_packs // 16, 4)
        cache[page, :tokens * layout.k_packs] = k_page.permute(2, 0, 1, 3).reshape(-1)
        # V storage axes: [pack row, token pair, thread group, token in pair].
        v_page = torch.full((layout.padded_tokens, layout.v_packs), 0x3555,
                            dtype=torch.int16)
        v_page[:n] = v_packed[begin:end]
        v_page = v_page.reshape(layout.padded_tokens // 2, 2, 8, layout.v_packs // 8)
        cache[page, layout.value:layout.value + v_page.numel()] = v_page.permute(3, 0, 2, 1).reshape(-1)
        half_view = cache[page].view(torch.float16)
        half_view[layout.k_meta:layout.k_meta + 2 * n].reshape(n, 2)[:] = torch.stack(
            (k_scales[begin:end], k_zeros[begin:end]), dim=1)
        half_view[layout.v_meta:layout.v_meta + 2 * n].reshape(n, 2)[:] = torch.stack(
            (v_scales[begin:end], v_zeros[begin:end]), dim=1)
        # Start with nonzero, token-specific scores to distinguish += from =.
        cache[page].view(torch.float32)[layout.score // 2:layout.score // 2 + n] = (
            torch.arange(begin, end, dtype=torch.float32) % 13 + 1) * 1e-5
        cache[page].view(torch.int32)[layout.position // 2:layout.position // 2 + n] = torch.arange(begin, end)
        score_mask[page, layout.score:layout.score + 2 * n] = True
        addresses.extend(page * (PAGE_SIZE // 2) + layout.score // 2 + offset for offset in range(n))
    return keys, values, torch.tensor(addresses, dtype=torch.int64)


def build_fixture(lengths_by_seq=((33, 61), (95, 119)), kv_heads=2,
                  seed=17, use_alibi=False, adversarial_max=False, compute_reference=True):
    """Return CPU packed inputs and a partition-independent full-softmax oracle.

    Helpers are also imported by the model-free attention microbenchmark.
    score_indices address the flattened FP32 view of kv_cache.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to build packed attention inputs")
    if not lengths_by_seq:
        raise ValueError("At least one sequence is required")
    generator = torch.Generator().manual_seed(seed)
    seqs, layers = len(lengths_by_seq), 2
    slots = max(4, seqs + 2)
    query_heads = kv_heads * QUERY_HEADS_PER_KV
    block_capacity = max((h + 31) // 32 + (l + 59) // 60
                         for h, l in lengths_by_seq) + 3
    blocks = slots * layers * kv_heads * block_capacity
    cache = torch.full((blocks, PAGE_SIZE), 0x3555, dtype=torch.int16)
    score_mask = torch.zeros_like(cache, dtype=torch.bool)
    tables = torch.randperm(blocks, generator=generator).reshape(
        slots, layers, kv_heads, block_capacity).to(torch.int32)
    lengths = torch.zeros((slots, layers, kv_heads, 2), dtype=torch.int32)
    selected_slots = (torch.tensor((3, 1)[:seqs], dtype=torch.int32) if seqs <= 2 else
                      torch.arange(1, seqs + 1, dtype=torch.int32).flip(0))
    query = (torch.randn((seqs, query_heads, HEAD_SIZE), generator=generator) * 0.12).half()
    for head in range(query_heads):
        query[:, head, :8] = (1.25, -1.0, 0.4, -0.6)[head % 4]
    if adversarial_max:
        query.zero_()
        query[:, :, 0] = 512
    slopes = (torch.linspace(0, 0.0003, query_heads, dtype=torch.float32)
              if use_alibi else None)
    expected_output = (torch.empty((seqs, query_heads, HEAD_SIZE), dtype=torch.float64)
                       if compute_reference else None)
    score_updates, score_indices, logical_kv = {}, {}, {}
    max_context_len = 0
    for seq, (high, low) in enumerate(lengths_by_seq):
        slot = int(selected_slots[seq])
        for head in range(kv_heads):
            # Give every KV head a distinct length and partial-page boundary.
            h, l = max(0, high - head), max(0, low - 2 * head)
            if h + l == 0:
                raise ValueError("Every KV head needs at least one retained token")
            lengths[slot, LAYER_IDX, head] = torch.tensor((h, l), dtype=torch.int32)
            max_context_len = max(max_context_len, h + l)
            regions, addresses = [], []
            for count, bits, pages in (
                (h, (8, 4, 32), tables[slot, LAYER_IDX, head, :(h + 31) // 32]),
                (l, (4, 2, 60), tables[slot, LAYER_IDX, head].flip(0)[:(l + 59) // 60]),
            ):
                keys, values, indices = _write_region(cache, pages.tolist(), count, bits,
                                                       generator, score_mask,
                                                       adversarial_max=adversarial_max,
                                                       compute_reference=compute_reference)
                regions.append((keys, values))
                addresses.append(indices)
            if not compute_reference:
                continue
            keys = torch.cat([region[0] for region in regions])
            values = torch.cat([region[1] for region in regions])
            q = query[seq, head * 4:(head + 1) * 4].double()
            logits = q @ keys.T * SCALE
            if slopes is not None:
                logits += slopes[head * 4:(head + 1) * 4].double()[:, None] * (
                    torch.arange(h + l).double() - h - l + 1)
            weights = torch.softmax(logits, dim=-1)
            expected_output[seq, head * 4:(head + 1) * 4] = weights @ values
            score_updates[seq, head] = weights.max(dim=0).values
            score_indices[seq, head] = torch.cat(addresses)
            logical_kv[seq, head] = (keys, values)
    return SimpleNamespace(slot_ids=selected_slots,
                           positions=torch.tensor([high + low + 137 for high, low in lengths_by_seq], dtype=torch.int32),
                           query=query, kv_cache=cache, block_tables=tables,
                           kv_len_tables=lengths, alibi_slopes=slopes,
                           max_context_len=max_context_len,
                           expected_output=expected_output,
                           score_updates=score_updates, score_indices=score_indices,
                           score_mask=score_mask, logical_kv=logical_kv)


def device_inputs(fixture, device="cuda"):
    result = {name: getattr(fixture, name).to(device) for name in
              ("slot_ids", "positions", "kv_cache", "block_tables", "kv_len_tables")}
    # A wider backing allocation exercises the nontrivial q_stride path.
    seqs, heads, size = fixture.query.shape
    backing = torch.empty((seqs, heads + 2, size), dtype=fixture.query.dtype, device=device)
    result["query"] = backing[:, :heads]
    result["query"].copy_(fixture.query)
    result["alibi_slopes"] = (fixture.alibi_slopes.to(device)
                               if fixture.alibi_slopes is not None else None)
    return result


def allocate_workspaces(fixture, device="cuda"):
    seqs, heads, size = fixture.query.shape
    capacity = (1 if fixture.max_context_len <= PARTITION_SIZE else
                (fixture.max_context_len + PARTITION_SIZE - 1) // PARTITION_SIZE + 1)
    return dict(
        out=torch.empty((seqs, heads, size), dtype=torch.float16, device=device),
        exp_sums=torch.full((seqs, heads, capacity), float("nan"), device=device),
        max_logits=torch.full((seqs, heads, capacity), float("nan"), device=device),
        tmp_out=torch.full((seqs, heads, capacity, size), float("nan"), dtype=torch.float16, device=device),
        tmp_scores=torch.full((seqs, heads, fixture.max_context_len), float("nan"), device=device))


def attention_args(fixture, inputs, workspace):
    return (inputs["slot_ids"], inputs["positions"], workspace["out"],
            workspace["exp_sums"], workspace["max_logits"], workspace["tmp_out"],
            workspace["tmp_scores"], inputs["query"], inputs["kv_cache"],
            LAYER_IDX, fixture.query.shape[1] // 4, SCALE, inputs["block_tables"],
            inputs["kv_len_tables"], fixture.max_context_len, *QUANT_ARGS,
            inputs["alibi_slopes"])


@unittest.skipUnless(torch is not None, "PyTorch is required for the CPU packed-layout check")
class PackedCacheReference(unittest.TestCase):
    def test_adversarial_softmax_reference_stays_finite(self):
        fixture = build_fixture(((32, 0), (31, 0)), adversarial_max=True)
        self.assertTrue(torch.isfinite(fixture.expected_output).all())
        for update in fixture.score_updates.values():
            self.assertTrue(torch.isfinite(update).all())
            self.assertEqual(float(update[1]), 1.0)

    def test_empty_precision_regions(self):
        fixture = build_fixture(((193, 0), (0, 181)))
        self.assertTrue(torch.isfinite(fixture.expected_output).all())
        for key, indices in fixture.score_indices.items():
            self.assertEqual(len(indices), len(fixture.score_updates[key]))
            self.assertGreater(len(indices), 0)

    def test_encoded_cache_matches_independent_logical_values(self):
        fixture = build_fixture()
        # Decode selected scalar dimensions directly from bit addresses. This
        # deliberately avoids the encoder's reshape/permute inverse so a wrong
        # page layout cannot silently give the GPU and oracle different inputs.
        dimensions = (0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 95, 96, 126, 127)
        for (seq, head), (keys, values) in fixture.logical_kv.items():
            slot = int(fixture.slot_ids[seq])
            high, low = fixture.kv_len_tables[slot, LAYER_IDX, head].tolist()
            table = fixture.block_tables[slot, LAYER_IDX, head]
            for is_low, count, bits, logical_start in (
                (False, high, (8, 4, 32), 0), (True, low, (4, 2, 60), high)):
                k_bits, v_bits, tokens = bits
                layout = page_layout(*bits)
                for token in sorted({0, count // 2, count - 1, min(tokens - 1, count - 1),
                                     min(tokens, count - 1)}):
                    if token < 0:
                        continue
                    page_in_region, offset = divmod(token, tokens)
                    page = int(table[-1 - page_in_region] if is_low else table[page_in_region])
                    half = fixture.kv_cache[page].view(torch.float16)
                    for dimension in dimensions:
                        pack, element = divmod(dimension, 16 // k_bits)
                        vector, pack_in_vector = divmod(pack, 4)
                        group, row = divmod(vector, layout.k_packs // 16)
                        k_address = row * tokens * 16 + offset * 16 + group * 4 + pack_in_vector
                        k_code = ((int(fixture.kv_cache[page, k_address]) & 0xffff) >>
                                  (element * k_bits)) & ((1 << k_bits) - 1)
                        actual_k = k_code * float(half[layout.k_meta + offset * 2]) + float(
                            half[layout.k_meta + offset * 2 + 1])
                        self.assertEqual(actual_k, float(keys[logical_start + token, dimension]))
                        pack, element = divmod(dimension, 16 // v_bits)
                        group, row = divmod(pack, layout.v_packs // 8)
                        v_address = (layout.value + row * layout.padded_tokens * 8 +
                                     (offset // 2) * 16 + group * 2 + offset % 2)
                        v_code = ((int(fixture.kv_cache[page, v_address]) & 0xffff) >>
                                  (element * v_bits)) & ((1 << v_bits) - 1)
                        actual_v = v_code * float(half[layout.v_meta + offset * 2]) + float(
                            half[layout.v_meta + offset * 2 + 1])
                        self.assertEqual(actual_v, float(values[logical_start + token, dimension]))


@unittest.skipUnless(torch is not None and torch.cuda.is_available(),
                     "CUDA is required for the real DiffKV decode attention kernels")
class DecodeAttentionCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vllm._C import ops
        cls.ops = ops
        if not hasattr(ops, "sparse_paged_attention_gqa"):
            raise RuntimeError("Rebuild vllm._C to include sparse_paged_attention_gqa")

    def check_case(self, lengths, use_alibi=False, adversarial_max=False):
        fixture = build_fixture(lengths, use_alibi=use_alibi,
                                adversarial_max=adversarial_max)
        original_scores = fixture.kv_cache.view(torch.float32).reshape(-1)
        for mode in MODES:
            with self.subTest(lengths=lengths, mode=mode):
                inputs = device_inputs(fixture)
                workspace = allocate_workspaces(fixture)
                args = attention_args(fixture, inputs, workspace)
                op = (self.ops.sparse_paged_attention if mode == "legacy" else
                      self.ops.sparse_paged_attention_gqa)
                for repeat in range(1, 3):
                    op(*args)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(workspace["out"].cpu().double(),
                                               fixture.expected_output,
                                               atol=1e-3, rtol=3e-4)
                    actual_cache = inputs["kv_cache"].cpu()
                    actual_scores = actual_cache.view(torch.float32).reshape(-1)
                    for key, indices in fixture.score_indices.items():
                        # Compare increments rather than allowing an initial
                        # accumulated score to hide a bad probability.
                        actual_increment = actual_scores[indices] - original_scores[indices]
                        torch.testing.assert_close(actual_increment.double(),
                                                   fixture.score_updates[key] * repeat,
                                                   atol=5e-8, rtol=5e-4)
                    self.assertTrue(torch.equal(actual_cache[~fixture.score_mask],
                                                fixture.kv_cache[~fixture.score_mask]),
                                    "Attention modified payload, metadata, positions, or unrelated cache pages")
                    self.assertTrue(torch.equal(inputs["kv_len_tables"].cpu(), fixture.kv_len_tables))
                    self.assertTrue(torch.equal(inputs["block_tables"].cpu(), fixture.block_tables))

    def test_partial_high_and_low_pages(self):
        self.check_case(((33, 61), (95, 119)))

    def test_medium_mixed_context(self):
        self.check_case(((479, 480), (959, 960)))

    def test_around_5760_tokens(self):
        self.check_case(((2879, 2880), (2880, 2880), (2881, 2880)))

    def test_low_partition_starts_after_partial_high_region(self):
        self.check_case(((33, 5761), (5761, 1201)))

    def test_above_original_5760_partition(self):
        self.check_case(((2879, 2882), (5761, 119)))

    def test_high_only_and_low_only(self):
        self.check_case(((193, 0), (0, 181)))

    def test_partitioned_alibi(self):
        self.check_case(((33, 5761), (5761, 1201)), use_alibi=True)

    def test_softmax_maximum_outside_first_thread_group(self):
        # Token 1's thread group must participate in the warp maximum. If its
        # logit is missed, exp(logit - incorrect_max) overflows at this gap.
        self.check_case(((32, 0), (31, 0)), adversarial_max=True)

    def test_unsupported_gqa_ratio_is_rejected(self):
        fixture = build_fixture()
        inputs = device_inputs(fixture)
        workspace = allocate_workspaces(fixture)
        # Two query heads per KV head are valid attention inputs, but this
        # shared-K/V specialization deliberately supports exactly four.
        inputs["query"] = inputs["query"][:, ::2].contiguous()
        workspace = {name: tensor[:, ::2].contiguous()
                     for name, tensor in workspace.items()}
        args = attention_args(fixture, inputs, workspace)
        with self.assertRaises(RuntimeError):
            self.ops.sparse_paged_attention_gqa(*args)
        self.assertTrue(torch.equal(inputs["kv_cache"].cpu(), fixture.kv_cache))


if __name__ == "__main__":
    unittest.main(verbosity=2)
