"""Check actual DiffKV prompt-cache score/token pairing on CUDA.

Run without loading model weights:
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .cache/baselines/fp16-env/bin/python \
    tests/kernels/test_diffkv_prompt_scores.py
"""

import unittest

import torch

from vllm._C import cache_ops


def cache_offsets(k_bits, v_bits, tokens):
    """Return half-element offsets for the current one-group cache format."""
    def align(value, boundary):
        return (value + boundary - 1) // boundary * boundary

    key_meta = align(tokens * (128 // (16 // k_bits)) * 2, 32) // 2
    value = align(key_meta * 2 + tokens * 4, 128) // 2
    padded_tokens = align(tokens, 2)
    value_meta = value + align(padded_tokens * (128 // (16 // v_bits)) * 2, 32) // 2
    score = value_meta + align(tokens * 4, 32) // 2
    position = score + tokens * 2
    return key_meta, value_meta, score, position


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the real cache kernel")
class PromptCacheScores(unittest.TestCase):
    @torch.inference_mode()
    def check_pairing(self, prompt_lengths):
        device = "cuda"
        slots, layers, kv_heads, query_heads = 4, 2, 8, 32
        selected_slots, layer = (1, 3), 1
        max_length, block_capacity = max(prompt_lengths), 10
        blocks = slots * layers * kv_heads * block_capacity
        cache = torch.full((blocks, 3392), -1, dtype=torch.int16, device=device)
        # Nontrivial physical pages, slots, layer and GQA heads expose accidental
        # indexing through the batch dimension or a single query head.
        permutation = torch.randperm(blocks, generator=torch.Generator().manual_seed(17))
        tables = permutation.reshape(slots, layers, kv_heads, block_capacity).to(device, torch.int32)
        lengths = torch.full((slots, layers, kv_heads, 2), -1, dtype=torch.int32, device=device)
        scores = torch.zeros((2, query_heads, max_length), dtype=torch.float32)
        original_scores = []
        keys, values = [], []
        for seq, length in enumerate(prompt_lengths):
            by_head = []
            positions = torch.arange(length)
            for head in range(kv_heads):
                rank = (positions * 37 + head * 11 + seq * 7) % 128
                means = torch.where(rank < 8, 0.00004, 0.002 + rank * 0.0001)
                source_scores = means * torch.clamp(length - positions, max=64)
                by_head.append(source_scores)
                for query in range(4):
                    scores[seq, head * 4 + query, :length] = source_scores * ((query + 1) / 4)
            original_scores.append(torch.stack(by_head))
            # Each token's exact half-precision K/V minimum encodes its source
            # position. Cached adjusted position alone cannot identify old
            # prompt tokens, because they all become length - 64.
            identity = positions[:, None] + seq * 512 + torch.arange(kv_heads)[None, :] * 128
            ramp = torch.linspace(0, 1, 128)[None, None, :]
            keys.append((identity[:, :, None] + ramp).half())
            values.append((-identity[:, :, None] + ramp).half())
        key = torch.cat(keys).to(device)
        value = torch.cat(values).to(device)
        starts = [0, prompt_lengths[0], sum(prompt_lengths)]
        cache_ops.compress_and_append_cache_prompt_phase(
            key, value, scores.to(device), cache,
            torch.tensor(selected_slots, dtype=torch.int32, device=device), tables, lengths,
            torch.tensor(starts, dtype=torch.int32, device=device),
            torch.tensor([[0.02, 1.0]] * slots, dtype=torch.float32, device=device),
            64, layer, 8, 4, 4, 2, 1, 1, 1, 1, 4, 2, 32, 60)
        torch.cuda.synchronize()
        cache_cpu = cache.cpu()
        cache_half = cache_cpu.view(torch.float16)
        cache_float = cache_cpu.view(torch.float32)
        cache_int = cache_cpu.view(torch.int32)
        tables, lengths = tables.cpu(), lengths.cpu()
        checked = {"high": 0, "low": 0}
        mismatches = []
        for seq, prompt_length in enumerate(prompt_lengths):
            slot = selected_slots[seq]
            for head in range(kv_heads):
                source = original_scores[seq][head]
                queries = torch.clamp(prompt_length - torch.arange(prompt_length), max=64)
                mean = source / queries
                eligible = torch.arange(prompt_length) < prompt_length - 64
                pruned = eligible & (mean < 0.02 / prompt_length)
                low = eligible & ~pruned & (mean < 1.0 / prompt_length)
                expected_lengths = [int((~pruned & ~low).sum()), int(low.sum())]
                self.assertEqual(lengths[slot, layer, head].tolist(), expected_lengths)
                high_blocks = (expected_lengths[0] + 31) // 32
                seen = []
                for region, count, bits, tokens, page_start in (
                    ("high", expected_lengths[0], (8, 4), 32, 0),
                    ("low", expected_lengths[1], (4, 2), 60, high_blocks),
                ):
                    k_meta, v_meta, score_offset, position_offset = cache_offsets(*bits, tokens)
                    for index in range(count):
                        page = int(tables[slot, layer, head, page_start + index // tokens])
                        offset = index % tokens
                        identity = float(cache_half[page, k_meta + offset * 2 + 1])
                        value_identity = -float(cache_half[page, v_meta + offset * 2 + 1])
                        self.assertEqual(identity, value_identity, "Cached K and V came from different tokens")
                        original_position = int(identity) - seq * 512 - head * 128
                        self.assertIn(original_position, range(prompt_length))
                        self.assertFalse(bool(pruned[original_position]))
                        self.assertEqual(bool(low[original_position]), region == "low")
                        seen.append(original_position)
                        effective_position = prompt_length - min(prompt_length - original_position, 64)
                        self.assertEqual(int(cache_int[page, position_offset // 2 + offset]), effective_position)
                        actual = float(cache_float[page, score_offset // 2 + offset])
                        expected = float(source[original_position])
                        checked[region] += 1
                        if actual != expected:
                            mismatches.append({"seq": seq, "kv_head": head, "region": region,
                                               "source_position": original_position,
                                               "actual_score": actual, "expected_score": expected})
                self.assertEqual(sorted(seen), torch.where(~pruned)[0].tolist())
        self.assertGreater(checked["high"], 0)
        if max_length > 64:
            self.assertGreater(checked["low"], 0)
        mismatched = {region: sum(row["region"] == region for row in mismatches)
                      for region in ("high", "low")}
        self.assertEqual(len(mismatches), 0, f"Wrong score/token pairing, checked={checked}, "
                         f"mismatches={mismatched}, examples={mismatches[:8]}")

    def test_unsorted_buffer_only_prompt(self):
        self.check_pairing((64, 37))

    def test_sorted_power_of_two_prompt(self):
        self.check_pairing((128, 111))

    def test_sorted_non_power_of_two_prompt(self):
        self.check_pairing((193, 165))


if __name__ == "__main__":
    unittest.main(verbosity=2)
