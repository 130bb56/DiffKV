# DiffKV bringup (vLLM 0.32.0)

vLLM v0.32.0(`5df63a7`) 위에 DiffKV attention과 KV cache 관리를 붙인 코드입니다. DiffKV 전용 파일은 모두 이 디렉터리에 있습니다. 아래 경로는 repo 루트(`/home/130bb56/DiffKV`) 기준입니다.

## 디렉터리

```text
diffkv/
├── run_docker.sh                     container 안에서 명령 실행
├── python/diffkv/                    DiffKV Python 코드 (backend, cache 관리, admission, model runner hook, extension build, prefill attention과 압축)
├── csrc/                             DiffKV CUDA kernel과 binding
├── docs/DiffKV.pdf                   논문 (git 제외)
└── benchmarks/
    ├── math500/
    │   ├── generate.py               backend 하나로 문항 생성, 요청별 시간 기록
    │   ├── generation_utils.py       backend 옵션, 문항 읽기, 지표 요약
    │   ├── record_source_hashes.py   실행 전 코드 해시 기록
    │   ├── test_generation.py        CPU test
    │   ├── problems/                 Correctness test를 위한 benchmark dataset
    │   ├── correctness/              run_correctness.sh, score_outputs.py (답 추출, 정답 판정, 집계)
    │   ├── serving/offline/
    │   ├── serving/online/           
    │   └── logs/
    └── attention/
        ├── run_latency.sh            batch별 decode latency 실행
        ├── benchmark_latency.py      DiffKV, vLLM FA2 FP16, FlashInfer FP8 decode latency benchmark
        ├── run_prefill.sh            batch별 prefill latency 실행 (chunked prefill 끔, prompt 전체를 한 step에서 처리)
        ├── benchmark_prefill.py      layer 하나의 prefill latency: 현재 DiffKV backend 경로, DiffKV prompt_prefill (python/diffkv/prefill.py), vLLM FA2 FP16, FlashInfer FP8. 모두 cache 쓰기 포함
        ├── correctness.py
        ├── profile_nsys.py           nsys로 kernel latency breakdown
        └── logs/
```

## vLLM 쪽 수정 파일

| 파일 | 내용 |
| --- | --- |
| `vllm/v1/attention/backends/registry.py` | `DIFFKV` backend 등록 |
| `vllm/v1/core/sched/scheduler.py` | DiffKV cache manager 선택, preempt 때 물리 page 반납, worker 보고로 사용량 갱신 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | `DiffKVSpec` 등록 |
| `vllm/v1/worker/gpu/model_runner.py` | request 추가, 제거, preempt, batch 시작과 끝마다 DiffKV 호출 |
| `vllm/v1/outputs.py` | worker가 잰 cache 사용량을 scheduler로 보내는 `diffkv_cache_report` 필드 |

## 실행 환경

- host가 Ubuntu 18.04(glibc 2.27)라서 torch 2.13이 로드되지 않음 → 모든 명령은 `diffkv/run_docker.sh`로 container 안에서 실행
- venv: `.venv` (container 안 Python 3.12). `site-packages/diffkv.pth`가 `diffkv/python`과 `diffkv/benchmarks`를 import 경로에 넣음
- CUDA toolkit: `.toolchains/cuda-12.9`, uv: `.toolchains/uv`
- Nsight Systems: `run_docker.sh`가 host의 `/opt/nvidia/nsight-systems-cli/2024.4.1`을 container에 read-only로 mount (환경 변수 `DIFFKV_NSYS_DIR`로 변경)
- 모델: `models/DeepSeek-R1-Distill-Llama-8B`
- GPU: 환경 변수 `DIFFKV_GPU` (기본 0)

## 실행 명령

repo 루트에서 실행합니다.

| 작업 | 명령 |
| --- | --- |
| extension build | `bash diffkv/run_docker.sh python -m diffkv.build_ops` |
| kernel 정확도 검사 | `bash diffkv/run_docker.sh python diffkv/benchmarks/attention/correctness.py` (decode: FP64 oracle, prefill: prompt_prefill을 현재 backend 경로와 비교) |
| attention latency | `bash diffkv/benchmarks/attention/run_latency.sh [batch ...]` (기본 1 8 32, KV 길이 1024 2048 4096 8192) |
| prefill latency | `bash diffkv/benchmarks/attention/run_prefill.sh [batch ...]` (기본 batch 1 8 32, prompt 256 1024, `DIFFKV_PROMPTS`로 변경, `DIFFKV_PREFILL_PARTS=1`이면 prompt_prefill의 attention, score, 압축 launch도 따로 잼) |
| attention profile (Nsight Systems) | `bash diffkv/benchmarks/attention/run_profile.sh [batch ...]` (기본 1 8 50, KV 길이 512 2048 8192) |
| correctness | `bash diffkv/benchmarks/math500/correctness/run_correctness.sh <fp16\|diffkv\|fp8> <sample50\|all500>` |
| offline serving | `bash diffkv/benchmarks/math500/serving/offline/run_offline_serving.sh <fp16\|diffkv\|fp8>` |
| 두 실행 비교 | `bash diffkv/run_docker.sh python diffkv/benchmarks/math500/serving/offline/compare_offline_serving.py --reference <기준 outputs.json> --candidate <비교 outputs.json> --reference-backend <backend> --candidate-backend <backend>` |
| harness test | `bash diffkv/run_docker.sh python diffkv/benchmarks/math500/test_generation.py` |

## 측정 조건

| 항목 | correctness | offline serving | attention latency |
| --- | --- | --- | --- |
| 생성 | EOS까지, greedy, 최대 8192 token | 50문항 동시 도착, 요청별 출력 길이 고정(`ignore_eos`) | decode 1 step |
| 출력 길이 | 모델이 정함 | bringup FP16 EOS 실행의 길이(`output_lengths_sample50.json`, 합계 185,093 token) | 해당 없음 |
| SM clock | 고정 안 함 | 1395 MHz 고정 | 1395 MHz 고정 |
| GPU telemetry | 없음 | 200 ms 간격 | 200 ms 간격 |

- 엔진 설정 (모든 backend 같음): eager, chunked prefill off (prompt 전체를 한 step에서 처리, DiffKV는 `prompt_prefill`), async scheduling off, prefix caching off, max_model_len 9216, max_num_batched_tokens 9216, max_num_seqs 64, gpu_memory_utilization 0.9
- DiffKV: K8V4와 K4V2, threshold `[0.02, 1.0]`, buffer 64 token
- FP8: FlashInfer, `fp8_e4m3` KV cache, scale 1.0
- clock 고정은 docker(`--cap-add=SYS_ADMIN`)로 `nvidia-smi -lgc 1395,1395` 실행

## 제한

- online serving 측정 스크립트 없음
