#!/usr/bin/env python3
"""Refresh offline source snapshots and literal diffs from a local FP16 checkout.

Run from any directory. Does not download or modify implementation sources.
"""

import argparse
import difflib
import hashlib
from html import escape, unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess

DOCS = Path(__file__).resolve().parents[1]
ROOT = DOCS.parents[1]
FP16_COMMIT = "1db83e31a2468cae37f326a642c0a4c4edbb5e4f"
GROUPS = {
    "serving": [
        "vllm/entrypoints/api_server.py", "vllm/entrypoints/llm.py",
        "vllm/engine/arg_utils.py", "vllm/engine/async_llm_engine.py",
        "vllm/engine/llm_engine.py", "vllm/sequence.py",
        "vllm/core/scheduler.py", "vllm/core/orchestrator.py",
        "vllm/core/block_manager.py", "vllm/worker/worker.py",
        "vllm/worker/model_runner.py", "vllm/worker/cache_engine.py",
        "vllm/config.py", "vllm/model_executor/input_metadata.py",
        "vllm/model_executor/model_loader.py",
        "vllm/model_executor/models/__init__.py",
    ],
    "kernels": [
        "vllm/model_executor/layers/attention.py",
        "vllm/model_executor/layers/sparse_attention_big_kernel.py",
        "vllm/model_executor/layers/triton_flash_attention.py",
        "vllm/model_executor/layers/triton_fused_softmax_sum.py",
        "csrc/attention/attention_kernels.cu",
        "csrc/attention/attention_utils.cuh", "csrc/attention/cache_utils.h",
        "csrc/attention/sparse_attention_kernels.cu",
        "csrc/cache_kernels.cu", "csrc/long_prompt_cache_kernels.cu",
        "csrc/mem_mgt_kernels.cu", "csrc/pybind.cpp",
        "csrc/quantization/kv_cache_quant/kv_cache_quant.h",
        "csrc/quantization/kv_cache_quant/uint8_quant.cuh",
        "csrc/quantization/kv_cache_quant/uint4_quant.cuh",
        "csrc/quantization/kv_cache_quant/uint2_quant.cuh",
        "csrc/quantization/kv_cache_quant/uint1_quant.cuh",
        "setup.py",
    ],
}

SOURCE_DESCRIPTIONS = {
    "vllm/entrypoints/api_server.py": "HTTP 요청에서 prompt, sampling 설정, KV 양자화와 pruning 설정을 읽습니다.",
    "vllm/entrypoints/llm.py": "Python LLM.generate API에서 요청을 엔진에 전달합니다.",
    "vllm/engine/arg_utils.py": "엔진 초기화에 사용할 명령행 인자와 설정을 구성합니다.",
    "vllm/engine/async_llm_engine.py": "비동기 요청 큐와 engine loop, 응답 stream을 관리합니다.",
    "vllm/engine/llm_engine.py": "요청 등록, scheduler 호출, 모델 실행 결과와 종료 조건을 처리합니다.",
    "vllm/sequence.py": "요청의 token 상태와 KV 양자화, pruning 설정을 보관합니다.",
    "vllm/core/scheduler.py": "실행할 요청을 선택하고 필요한 메모리를 계산합니다.",
    "vllm/core/orchestrator.py": "Worker 초기화와 local, Ray Worker 호출을 담당합니다.",
    "vllm/core/block_manager.py": "KV 페이지의 할당, 확장, 반환을 호출하고 page table과 free list를 관리합니다.",
    "vllm/worker/worker.py": "모델과 cache를 초기화하고 모델 실행, 페이지 반환을 호출합니다.",
    "vllm/worker/model_runner.py": "Token, position, InputMetadata를 준비하고 모델을 실행합니다.",
    "vllm/worker/cache_engine.py": "모든 layer가 공유하는 INT16 KV page tensor를 할당합니다.",
    "vllm/config.py": "모델, cache 설정과 양자화된 KV 페이지 크기를 계산합니다.",
    "vllm/model_executor/input_metadata.py": "Attention에 넘길 slot, KV table, 길이, 양자화 설정을 묶습니다.",
    "vllm/model_executor/model_loader.py": "모델 클래스를 선택하고 weight를 읽습니다.",
    "vllm/model_executor/models/__init__.py": "모델 architecture 이름을 구현 모듈과 클래스에 연결합니다.",
    "vllm/model_executor/layers/attention.py": "기존 attention layer의 prefill과 paged decode 호출 경로입니다.",
    "vllm/model_executor/layers/sparse_attention_big_kernel.py": "Prefill과 decode에서 Triton 함수, cache CUDA op, attention CUDA op를 호출합니다.",
    "vllm/model_executor/layers/triton_flash_attention.py": "Prefill attention 출력과 최근 64개 query의 QK 값을 계산합니다.",
    "vllm/model_executor/layers/triton_fused_softmax_sum.py": "최근 64개 query의 softmax 확률을 합해 초기 cache score를 계산합니다.",
    "csrc/attention/attention_kernels.cu": "FP16 paged attention V1, V2와 partition 출력 병합을 구현합니다.",
    "csrc/attention/attention_utils.cuh": "QK dot product와 thread 간 reduction에 쓰는 함수를 제공합니다.",
    "csrc/attention/cache_utils.h": "K/V vector 크기, thread group 크기와 정렬 보조 함수를 정의합니다.",
    "csrc/attention/sparse_attention_kernels.cu": "KV 역양자화, QK, softmax, AV, partition 출력 병합과 cache score 누적을 구현합니다.",
    "csrc/cache_kernels.cu": "Prefill의 저장 대상 분류와 양자화, decode의 token 제거, 재양자화, 새 K/V 저장을 구현합니다.",
    "csrc/long_prompt_cache_kernels.cu": "긴 prefill에서 Python의 정렬 결과를 받아 K/V를 양자화해 저장합니다.",
    "csrc/mem_mgt_kernels.cu": "페이지 할당, 확장, 반환과 page table 갱신을 구현합니다.",
    "csrc/pybind.cpp": "Python에서 호출하는 ops, cache_ops, mem_mgt_ops를 등록합니다.",
    "csrc/quantization/kv_cache_quant/kv_cache_quant.h": "Bit 수에 따라 K/V pack, unpack, 양자화와 역양자화 함수를 선택합니다.",
    "csrc/quantization/kv_cache_quant/uint8_quant.cuh": "8bit K/V 값의 양자화, uint16 packing과 역양자화를 구현합니다.",
    "csrc/quantization/kv_cache_quant/uint4_quant.cuh": "4bit K/V 값의 양자화, uint16 packing과 역양자화를 구현합니다.",
    "csrc/quantization/kv_cache_quant/uint2_quant.cuh": "2bit K/V 값의 양자화, uint16 packing과 역양자화를 구현합니다.",
    "csrc/quantization/kv_cache_quant/uint1_quant.cuh": "1bit K/V 값의 양자화, uint16 packing과 역양자화를 구현합니다.",
    "setup.py": "CUDA 소스 목록과 vllm._C 빌드를 정의합니다.",
}


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "data-path" in attrs:
            self.refs.append((attrs.get("data-version", "diffkv"),
                              attrs["data-path"], int(attrs["data-line"])))


def expand_references(document, baseline):
    """Write copyable source paths, with one reference per row."""
    pattern = r'<(?P<tag>button|a|code) class="ref"(?P<attrs>[^>]*)>(?P<label>.*?)</(?P=tag)>'

    def replace(match):
        attributes = re.sub(r'\s+href\s*=\s*"[^"]*"', "", match["attrs"])
        attrs = {key: unescape(value) for key, value in
                 re.findall(r'([\w-]+)="([^"]*)"', attributes)}
        label = attrs.get("data-label", unescape(match["label"]))
        attributes = re.sub(r'\s+(?:data-label|title)="[^"]*"', "", attributes)
        directory = ROOT if attrs.get("data-version", "diffkv") == "diffkv" else baseline
        location = str(directory / attrs["data-path"]) + ":" + attrs["data-line"]
        label = escape(label, quote=True)
        return (f'<code class="ref"{attributes} data-label="{label}" title="{label}">'
                f'{escape(location)}</code>')

    document = re.sub(pattern, replace, document, flags=re.DOTALL)
    return re.sub(
        r'(<code class="ref"[^>]*>[^<]*</code>)\s*(?:,|<br\s*/?>)\s*'
        r'(?=<code class="ref")',
        r'\1\n', document)


def highlight_diff(patch):
    """Color diff lines in HTML while preserving the patch's exact text."""
    output = []
    in_hunk = False
    for line in patch.splitlines(keepends=True):
        kind = ""
        if line.startswith("@@"):
            kind = "diff-hunk"
            in_hunk = True
        elif not in_hunk and line.startswith(("--- ", "+++ ")):
            kind = "diff-file"
        elif line.startswith("+"):
            kind = "diff-add"
        elif line.startswith("-"):
            kind = "diff-remove"
        if kind:
            content = line.rstrip("\r\n")
            ending = line[len(content):]
            output.append(f'<span class="{kind}">{escape(content)}</span>' + ending)
        else:
            output.append(escape(line))
    return "".join(output)


def embed_diffs(document, diffs):
    """Keep full patches readable in previews that disable JavaScript."""
    for group, files in diffs.items():
        start = f"<!-- BEGIN STATIC DIFFS: {group} -->"
        end = f"<!-- END STATIC DIFFS: {group} -->"
        blocks = [start, f'<div class="static-diffs" data-static-diff-group="{group}">',
                  f'<p class="caption">변경된 {len(files)}개 파일의 전체 diff입니다.</p>']
        for diff in files:
            path = escape(str(ROOT / diff["path"]))
            label = escape(diff["path"] + " unified diff", quote=True)
            status = " | 원본에 없는 파일" if diff["new"] else ""
            blocks.extend([
                '<details class="diff-file-entry">',
                '<summary><span class="diff-description">'
                + escape(SOURCE_DESCRIPTIONS[diff["path"]])
                + f'</span><code class="diff-path">{path}</code></summary>',
                f'<p class="diff-summary">+{diff["added"]} / -{diff["removed"]}{status}</p>',
                f'<pre class="diff-output" tabindex="0" aria-label="{label}"><code>'
                + highlight_diff(diff["patch"]) + '</code></pre>',
                '</details>',
            ])
        blocks.extend(['</div>', end])
        replacement = "\n".join(blocks)
        if start in document:
            pattern = re.escape(start) + r'.*?' + re.escape(end)
        else:
            pattern = r'<div data-diff-group="' + re.escape(group) + r'">.*?</pre></div>'
        document, count = re.subn(pattern, lambda _: replacement, document, flags=re.DOTALL)
        if count != 1:
            raise ValueError(f"Expected one {group} diff section, found {count}")
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        default=ROOT / ".cache/baselines/vllm-1db83e3")
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    if git(baseline, "rev-parse", "HEAD") != FP16_COMMIT:
        raise SystemExit("Baseline HEAD must match " + FP16_COMMIT)
    if git(baseline, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Baseline tracked sources must be clean")

    document = expand_references((DOCS / "index.html").read_text(), baseline)
    references = References()
    references.feed(document)
    sources = {"diffkv": {}, "fp16": {}}
    diffs = {}
    for group, paths in GROUPS.items():
        diffs[group] = []
        for path in paths:
            before_path, after_path = baseline / path, ROOT / path
            before = before_path.read_text() if before_path.exists() else ""
            after = after_path.read_text()
            sources["diffkv"][path] = after.splitlines()
            if before_path.exists():
                sources["fp16"][path] = before.splitlines()
            patch_lines = difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile="a/" + path if before_path.exists() else "/dev/null",
                tofile="b/" + path, n=3)
            patch = "".join(line if line.endswith("\n")
                            else line + "\n\\ No newline at end of file\n"
                            for line in patch_lines)
            if patch:
                lines = patch.splitlines()
                diffs[group].append({
                    "path": path, "patch": patch,
                    "added": sum(s.startswith("+") and not s.startswith("+++") for s in lines),
                    "removed": sum(s.startswith("-") and not s.startswith("---") for s in lines),
                    "new": not before_path.exists(),
                })
        (DOCS / "diffs").mkdir(exist_ok=True)
        (DOCS / "diffs" / (group + ".diff")).write_text(
            "\n".join(item["patch"] for item in diffs[group]))

    document = embed_diffs(document, diffs)

    for version, path, line in references.refs:
        if path not in sources[version]:
            directory = ROOT if version == "diffkv" else baseline
            sources[version][path] = (directory / path).read_text().splitlines()
        if not 1 <= line <= len(sources[version][path]):
            raise SystemExit(f"Invalid reference: {version} {path}:{line}")

    manifest = {
        "diffkv_commit": git(ROOT, "rev-parse", "HEAD"),
        "fp16_commit": FP16_COMMIT,
        "diffkv_tracked_changes": git(ROOT, "status", "--porcelain", "--untracked-files=no"),
        "scope": "Selected serving and kernel files, not a whole-repository diff",
        "source_sha256": {
            version: {path: hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
                      for path, lines in files.items()}
            for version, files in sources.items()
        },
    }
    # Hash original bytes, including the source's actual final-newline convention.
    for version, files in sources.items():
        directory = ROOT if version == "diffkv" else baseline
        for path in files:
            manifest["source_sha256"][version][path] = hashlib.sha256((directory / path).read_bytes()).hexdigest()
    (DOCS / "assets").mkdir(exist_ok=True)
    (DOCS / "assets/evidence.js").write_text(
        "// Generated by scripts/generate_evidence.py. Do not edit.\n"
        + "window.IMPLEMENTATION_EVIDENCE = "
        + json.dumps({"manifest": manifest, "sources": sources, "diffs": diffs}, ensure_ascii=False)
        + ";\n")
    (DOCS / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (DOCS / "index.html").write_text(document)
    print(f"Validated {len(references.refs)} source references; "
          f"saved {sum(map(len, sources.values()))} source snapshots and "
          f"{sum(map(len, diffs.values()))} file diffs.")


if __name__ == "__main__":
    main()
