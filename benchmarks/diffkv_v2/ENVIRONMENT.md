# Reproducing the vLLM 0.32.0 bringup environment

The source baseline is tag `v0.32.0`, commit
`5df63a7203cdb9328d824a27b682e77c7b9928be`. Runtime packages use official
precompiled vLLM release artifacts with this checkout installed editable;
the installed distribution reports `0.1.dev1+g5df63a720.precompiled`.
The source commit and source manifest identify the implementation.

| Component | Verified local environment |
| --- | --- |
| Host | Ubuntu 18.04.6, glibc 2.27 |
| GPUs, driver | 2 x NVIDIA GeForce RTX 3090, driver 525.147.05 |
| Container | Existing local `cute-dsl:4.8.1-torch`, glibc 2.41 |
| Python | 3.12.15, `.cache/bringup/vllm032/.venv` inside the container |
| PyTorch | `2.13.0+cu129`, runtime SymPy `1.14.0` |
| Tokenization | Transformers `5.19.0`, tokenizers `0.23.3` |
| CUDA compiler | NVIDIA nvcc `12.9.86`, `.cache/bringup/cuda-toolkit-12.9` |
| Build settings | `TORCH_CUDA_ARCH_LIST=8.6`, `MAX_JOBS=2` |

The inspected local image ID is
`sha256:ca18e2f9134073dc2cc5d9899d7d9974e321008ecc4ab7745477eeb5508befe8`.
Its `RepoDigests` list is empty; this is a local image identity, not a registry
digest. The tracked wrapper defaults to this image ID. Set
`DIFFKV_CONTAINER_IMAGE` to select another prepared image explicitly.

The CUDA compiler came from NVIDIA's CUDA 12.9.1 redistributable manifest,
component `cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.9.86-archive.tar.xz`
(SHA256 `7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35`).
The toolkit directory also links CUDA runtime and CCCL headers from the runtime
environment's NVIDIA packages. It is not the host's older CUDA toolkit.

## Running on this host

The original local launcher is `bash .cache/bringup/run-container.sh <command>`.
Its tracked counterpart resolves the checkout directory from its own location:

```bash
DIFFKV_GPU=0 bash benchmarks/diffkv_v2/run-container.sh \
  .cache/bringup/vllm032/.venv/bin/python benchmarks/diffkv_v2/run.py \
  --backend fp16 --gpu 0 --model .cache/models/DeepSeek-R1-Distill-Llama-8B \
  --mode natural --output evaluation-logs/vllm032-math/fp16-natural50.json
```

`DIFFKV_GPU` selects the host GPU. With one device exposed, `--gpu 0` selects
that device inside the container. The wrapper runs as the invoking UID/GID,
mounts the checkout at the same path, sets `CUDA_HOME`, prepends the prepared
Python environment and toolkit to `PATH`, and keeps caches/configuration under
the checkout. The prepared virtual
environment and toolkit contain absolute paths, so moving them requires
recreating the environment and header links.

`CUDA_MODULE_LOADING=LAZY` is required for the verified driver/runtime
combination: eager loading produced a CUDA named-symbol error during the first
allocation; lazy loading passed the same probe. The wrapper sets this before
Python starts. It also uses `OMP_NUM_THREADS=2`, offline model access,
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, and disables usage reporting.
`XDG_CACHE_HOME` is `<checkout>/.cache`, `VLLM_CONFIG_ROOT` is
`<checkout>/.cache/bringup/vllm-config`, `TRITON_CACHE_DIR` is
`<checkout>/.cache/bringup/triton`, and `FLASHINFER_WORKSPACE_BASE` is the checkout
itself. The launcher also supplies local uv, pre-commit, and npm cache paths.

For FP8 with FlashInfer on driver 525.147.05, pass the compiler override after
the wrapper so it is applied inside the container:

```bash
DIFFKV_GPU=0 bash benchmarks/diffkv_v2/run-container.sh \
  env FLASHINFER_EXTRA_CUDAFLAGS='--compress-mode=none' \
  .cache/bringup/vllm032/.venv/bin/python benchmarks/diffkv_v2/run.py \
  --backend fp8 --gpu 0 --model .cache/models/DeepSeek-R1-Distill-Llama-8B \
  --mode natural --output evaluation-logs/vllm032-math/fp8-natural50.json
```

FlashInfer's compressed fatbinary build (`--compress-mode=size`) failed with
`device kernel image is invalid` on this host. Rebuilding with compression
disabled completed the FP8 smoke test, two requests with eight output tokens each;
the [result metadata](../../logs/bringup/kv8-fp8-smoke.json) records the override.
[NVIDIA's nvcc 12.9 documentation](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-compiler-driver-nvcc/index.html#compress-mode-default-size-speed-balance-none-compress-mode)
documents the compression compatibility restriction for drivers predating CUDA
12.4 and defines `none` as disabling compression. The benchmark records the
effective `FLASHINFER_EXTRA_CUDAFLAGS` in its runtime provenance.

The native FP16 baseline resolved to `FLASH_ATTN`, FlashAttention 2. All runs use
FP16 model weights, eager execution, chunked prefill, synchronous scheduling,
and the same workload/settings in [README.md](README.md). Baseline KV is FP16;
DiffKV uses K8V4/K4V2 packed KV. The comparison graphics clock is 1395 MHz,
observed on GPU 0 during the active baseline run. Performance results belong
in the run artifacts, not this environment description.

## CPU grading and other hosts

The archived grader uses a separate `.cache/bringup/grading/.venv` with
[grading-requirements.txt](grading-requirements.txt). Its SymPy `1.13.1` conflicts
with PyTorch 2.13's `sympy>=1.13.3` requirement, so the serving environment keeps
SymPy `1.14.0`. Run the analyzer with the grading interpreter:

```bash
bash benchmarks/diffkv_v2/run-container.sh \
  .cache/bringup/grading/.venv/bin/python benchmarks/diffkv_v2/analyze.py \
  --workload benchmarks/diffkv_v2/fixtures/math50.json \
  --fp16 evaluation-logs/vllm032-math/fp16-natural50.json \
  --diffkv evaluation-logs/vllm032-math/diffkv-natural50.json \
  --output evaluation-logs/vllm032-math/correctness50.json
```

Docker is specific to this older host. On a modern compatible host, use a native
environment with the matching vLLM source/artifacts, PyTorch, and CUDA toolkit.
`vllm.diffkv_v2.ops.get_extension()` builds the isolated operators against the
active Python/PyTorch/CUDA ABI; it records source and compiler fingerprints.
Use the source manifest described in [README.md](README.md) before final runs.
