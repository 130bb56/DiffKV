"""Compare GQA K/V-dequantization reuse against the corrected v1 baseline."""

import hashlib
import json
from pathlib import Path

from diffkv_v1 import apply_variant as apply_v1


VARIANTS = ("gqa-reference", "gqa-reuse")


def apply_variant(engine, worker, variant, folder):
    """Apply v1 and select the reference or shared-K/V kernel before warmup."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown DiffKV GQA variant: {variant}")
    previous = getattr(engine, "_diffkv_gqa_manifest", None)
    if previous is not None:
        if previous["variant"] != variant:
            raise RuntimeError("Changing GQA variants requires a fresh engine and process")
        return previous

    from vllm._C import ops

    if not hasattr(ops, "sparse_paged_attention_gqa"):
        raise RuntimeError("DiffKV GQA reuse requires rebuilding the CUDA extension")
    attentions = [layer.self_attn.attn
                  for layer in worker.model_runner.model.model.layers]
    if not attentions or any(attn.head_size != 128 or attn.num_queries_per_kv != 4
                             for attn in attentions):
        raise ValueError("DiffKV GQA reuse requires head size 128 and four queries per KV head")
    folder = Path(folder).resolve()
    base = apply_v1(engine, worker, "no-sync-bulk-lengths", folder / "v1")
    for attention in attentions:
        attention.use_gqa_reuse = variant == "gqa-reuse"
    manifest = {
        "variant": variant,
        "reuse_queries_per_kv": 4 if variant == "gqa-reuse" else 1,
        "quantization": "K8V4/K4V2, one group, high page 32, low page 60",
        "base_variant": base,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Both variants include v1 Python optimizations; the GQA kernel shares K/V dequantization.",
    }
    (folder / "ablation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    engine._diffkv_gqa_manifest = manifest
    return manifest
