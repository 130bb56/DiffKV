"""Apply DiffKV v1 Python optimizations to an existing engine before warmup."""

import hashlib
import inspect
import json
from pathlib import Path
import textwrap


def replace_method(owner, name, replacements, folder):
    original = getattr(owner, name)
    source = textwrap.dedent(inspect.getsource(original))
    patched = source
    counts = {}
    for old, new in replacements:
        count = patched.count(old)
        if not count:
            raise RuntimeError(f"Expected expression missing from {owner.__name__}.{name}: {old}")
        counts[old] = count
        patched = patched.replace(old, new)
    namespace = {}
    filename = folder / f"{owner.__name__}.{name}.py"
    filename.write_text(patched)
    exec(compile(patched, str(filename), "exec"), original.__globals__, namespace)
    setattr(owner, name, namespace[name])
    return {"method": f"{owner.__module__}.{owner.__name__}.{name}",
            "replacement_counts": counts, "patched_source": str(filename),
            "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "patched_sha256": hashlib.sha256(patched.encode()).hexdigest()}


def apply_variant(engine, worker, variant, folder):
    """Apply a DiffKV ablation once in this process and save its source manifest.

    Call after worker creation and before warmup or measurement. The returned
    dictionary is also written to ``folder / 'ablation.json'``. Reapplying the
    same variant to this engine returns that manifest without patching again;
    changing variants requires a fresh engine and process. This helper has no
    benchmark-runner dependency, so the offline runner can import it safely.
    """
    if variant not in ("original", "no-sync", "no-sync-bulk-lengths"):
        raise ValueError(f"Unknown DiffKV variant: {variant}")
    previous = getattr(engine, "_diffkv_ablation_manifest", None)
    if previous is not None:
        if previous["variant"] != variant:
            raise RuntimeError(
                f"DiffKV variant {previous['variant']} is already applied to this engine; "
                f"cannot apply {variant}")
        return previous
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "variant": variant, "output_dir": str(folder), "patches": [],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Process-local Python changes only; CUDA binary is unchanged.",
    }
    if variant != "original":
        attention_type = type(worker.model_runner.model.model.layers[0].self_attn.attn)
        replacements = [("torch.cuda.synchronize()", "")]
        if variant == "no-sync-bulk-lengths":
            replacements.append(("input_metadata.max_context_len[self.layer].item()",
                                 "input_metadata.max_context_len[self.layer]"))
        manifest["patches"].append(replace_method(attention_type, "forward", replacements, folder))
        manifest["patches"].append(replace_method(type(engine), "step", replacements[:1], folder))
        if variant == "no-sync-bulk-lengths":
            runner_type = type(worker.model_runner)
            original_prepare = runner_type._prepare_decode

            def prepare_decode(self, *args, **kwargs):
                tokens, positions, metadata = original_prepare(self, *args, **kwargs)
                # The complete vector was already computed before the first
                # layer. Read all 32 lengths in one transfer and wait once.
                metadata.max_context_len = metadata.max_context_len.cpu().tolist()
                return tokens, positions, metadata

            runner_type._prepare_decode = prepare_decode
            bulk_source = textwrap.dedent(inspect.getsource(prepare_decode))
            bulk_path = folder / "prepare_decode.bulk_lengths.py"
            bulk_path.write_text(bulk_source)
            manifest["patches"].append({
                "method": f"{runner_type.__module__}.{runner_type.__name__}._prepare_decode",
                "patched_source": str(bulk_path),
                "original_sha256": hashlib.sha256(
                    textwrap.dedent(inspect.getsource(original_prepare)).encode()).hexdigest(),
                "patched_sha256": hashlib.sha256(bulk_source.encode()).hexdigest()})
            manifest["bulk_lengths"] = "one Tensor.cpu().tolist() per decode, replacing 32 Tensor.item() calls"
    (folder / "ablation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    engine._diffkv_ablation_manifest = manifest
    return manifest
