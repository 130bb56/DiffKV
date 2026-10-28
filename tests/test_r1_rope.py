"""CPU reference checks for the vLLM 0.2.6 Llama 3.1 compatibility patch.

Only CUDA placement and the extension import are replaced in these checks.
The production class and max-length validation are compiled from their sources.
Run with .cache/baselines/fp16-env/bin/python -m unittest discover -s tests
-p test_r1_rope.py.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / ".cache/baselines/vllm-1db83e3/vllm"
SCALING = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


def load_rotary(path):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if not isinstance(node, ast.ImportFrom)
                 or node.module != "vllm._C"]
    namespace = {"ops": SimpleNamespace()}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def load_max_len():
    path = BASELINE / "config.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef)
                 and node.name == "_get_and_verify_max_len"]
    namespace = {"Optional": Optional, "PretrainedConfig": PretrainedConfig,
                 "logger": SimpleNamespace(warning=lambda *args: None)}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["_get_and_verify_max_len"]


class Llama3RopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = load_rotary(
            BASELINE / "model_executor/layers/rotary_embedding.py")
        cls.diffkv = load_rotary(
            ROOT / "vllm/model_executor/layers/rotary_embedding.py")
        cls.config = LlamaConfig(
            hidden_size=4096, num_attention_heads=32,
            num_key_value_heads=8, max_position_embeddings=131072,
            rope_theta=500000.0, rope_scaling=SCALING.copy())
        cls.hf = LlamaRotaryEmbedding(cls.config, device="cpu")

    def make_rope(self, namespace, scaling=SCALING, max_position=8704):
        original_arange = torch.arange

        def cpu_arange(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original_arange(*args, **kwargs)

        with patch.object(torch, "arange", side_effect=cpu_arange):
            return namespace["get_rope"](
                128, 128, max_position, 500000.0,
                rope_scaling=scaling.copy() if scaling else None)

    def test_inverse_frequencies_match_hugging_face_and_diffkv(self):
        rope = self.make_rope(self.baseline)
        other = self.make_rope(self.diffkv)
        original_arange = torch.arange

        def cpu_arange(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original_arange(*args, **kwargs)

        with patch.object(torch, "arange", side_effect=cpu_arange):
            actual = rope._compute_inv_freq(500000.0)
            diffkv = other._compute_inv_freq(500000.0)
        torch.testing.assert_close(actual, self.hf.inv_freq, rtol=0, atol=0)
        torch.testing.assert_close(actual, diffkv, rtol=0, atol=0)

    def test_cached_cos_sin_match_hugging_face(self):
        rope = self.make_rope(self.baseline, max_position=131072)
        positions = torch.tensor([[0, 1, 127, 512, 4096, 8191, 8703, 131071]])
        for dtype in (torch.float32, torch.float16):
            reference_cos, reference_sin = self.hf(
                torch.empty(1, dtype=dtype), positions)
            actual_cos, actual_sin = rope.cos_sin_cache[positions].chunk(2, -1)
            actual_cos = actual_cos.repeat(1, 1, 2).to(dtype)
            actual_sin = actual_sin.repeat(1, 1, 2).to(dtype)
            torch.testing.assert_close(actual_cos, reference_cos, rtol=0, atol=0)
            torch.testing.assert_close(actual_sin, reference_sin, rtol=0, atol=0)

    def test_legacy_scaling_and_unscaled_routes_still_construct(self):
        for scaling, expected in (
            (None, "RotaryEmbedding"),
            ({"type": "linear", "factor": 2.0},
             "LinearScalingRotaryEmbedding"),
            ({"type": "dynamic", "factor": 2.0},
             "DynamicNTKScalingRotaryEmbedding"),
            ({"type": "yarn", "factor": 2.0,
              "original_max_position_embeddings": 4352},
             "YaRNScalingRotaryEmbedding"),
            ({"type": "llama3", **{k: v for k, v in SCALING.items()
                                   if k != "rope_type"}},
             "Llama3RotaryEmbedding"),
        ):
            with self.subTest(scaling=scaling):
                self.assertEqual(type(self.make_rope(
                    self.baseline, scaling)).__name__, expected)

    def test_llama3_reuses_baseline_execution_methods(self):
        base = self.baseline["RotaryEmbedding"]
        llama3 = self.baseline["Llama3RotaryEmbedding"]
        self.assertIs(llama3.forward, base.forward)
        self.assertIs(llama3._compute_cos_sin_cache,
                      base._compute_cos_sin_cache)

    def test_max_length_respects_llama3_advertised_context(self):
        verify = load_max_len()
        self.assertEqual(verify(self.config, None), 131072)
        self.assertEqual(verify(self.config, 8704), 8704)
        with self.assertRaises(ValueError):
            verify(self.config, 131073)

    def test_max_length_preserves_legacy_scaling(self):
        verify = load_max_len()
        for key in ("type", "rope_type"):
            config = SimpleNamespace(max_position_embeddings=4096,
                                     rope_scaling={key: "linear", "factor": 2})
            self.assertEqual(verify(config, None), 8192)
        yarn = SimpleNamespace(
            max_position_embeddings=8192,
            rope_scaling={"type": "yarn", "factor": 2,
                          "original_max_position_embeddings": 4096})
        self.assertEqual(verify(yarn, None), 8192)


if __name__ == "__main__":
    unittest.main()
