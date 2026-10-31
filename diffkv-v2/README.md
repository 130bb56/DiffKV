# DiffKV v2 bringup (vLLM 0.32.0)

vLLM v0.32.0(`5df63a7`) 위에 DiffKV v2 attention과 KV cache 관리를 붙인 코드입니다. DiffKV 전용 파일은 모두 이 디렉터리에 있습니다. 아래 경로는 repo 루트(`/home/130bb56/DiffKV`) 기준입니다.

## 디렉터리

```text
diffkv-v2/
├── run_docker.sh                     container 안에서 명령 실행
├── python/diffkv_v2/                 DiffKV Python 코드 (backend, cache 관리, admission, model runner hook, extension build)
├── csrc/                             DiffKV CUDA kernel과 binding
├── docs/DiffKV.pdf                   논문 (git 제외)
└── benchmarks/
    ├── math500/
    │   ├── generate.py               backend 하나로 문항 생성, 요청별 시간 기록
    │   ├── generation_utils.py       backend 옵션, 문항 읽기, 지표 요약
    │   ├── record_source_hashes.py   실행 전 코드 해시 기록
    │   ├── test_generation.py        CPU test
    │   ├── problems/                 math500_all.json (500문항), math500_sample50.json (50문항)
    │   ├── correctness/              run_correctness.sh, score_outputs.py (답 추출, 정답 판정, 집계)
    │   ├── serving/offline/          run_offline_serving.sh, compare_offline_serving.py, output_lengths_sample50.json
    │   ├── serving/online/           (측정 스크립트 없음)
    │   └── logs/                     결과 (git 제외)
    └── attention/
        ├── run_latency.sh            batch별 decode latency 실행
        ├── benchmark_latency.py      DiffKV v2, vLLM FA2 FP16, FlashInfer FP8 decode latency
        ├── correctness.py            FP64 기준값과 DiffKV kernel 출력 비교
        ├── profile_nsys.py           Nsight Systems 측정과 kernel별 시간 분석
        └── logs/                     결과 (git 제외)
```

## vLLM 쪽 수정 파일

| 파일 | 내용 |
| --- | --- |
| `vllm/v1/attention/backends/registry.py` | `DIFFKV_V2` backend 등록 |
| `vllm/v1/core/sched/scheduler.py` | DiffKV cache manager 선택, preempt 때 물리 page 반납, worker 보고로 사용량 갱신 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | `DiffKVSpec` 등록 |
| `vllm/v1/worker/gpu/model_runner.py` | request 추가, 제거, preempt, batch 시작과 끝마다 DiffKV 호출 |
| `vllm/v1/outputs.py` | worker가 잰 cache 사용량을 scheduler로 보내는 `diffkv_cache_report` 필드 |
| `vllm/config/vllm.py` | `DIFFKV_V2`일 때 Model Runner V2 사용 |

## 실행 환경

- host가 Ubuntu 18.04(glibc 2.27)라서 torch 2.13이 로드되지 않음 → 모든 명령은 `diffkv-v2/run_docker.sh`로 container 안에서 실행
- venv: `.venv` (container 안 Python 3.12). `site-packages/diffkv_v2.pth`가 `diffkv-v2/python`과 `diffkv-v2/benchmarks`를 import 경로에 넣음
- CUDA toolkit: `.toolchains/cuda-12.9`, uv: `.toolchains/uv`
- 모델: `models/DeepSeek-R1-Distill-Llama-8B`
- GPU: 환경 변수 `DIFFKV_GPU` (기본 0)

## 실행 명령

repo 루트에서 실행합니다.

| 작업 | 명령 |
| --- | --- |
| extension build | `bash diffkv-v2/run_docker.sh python -m diffkv_v2.build_ops` |
| kernel 정확도 검사 | `bash diffkv-v2/run_docker.sh python diffkv-v2/benchmarks/attention/correctness.py` |
| attention latency | `bash diffkv-v2/benchmarks/attention/run_latency.sh [batch ...]` (기본 1 8 50) |
| correctness | `bash diffkv-v2/benchmarks/math500/correctness/run_correctness.sh <fp16\|diffkv\|fp8> <sample50\|all500>` |
| offline serving | `bash diffkv-v2/benchmarks/math500/serving/offline/run_offline_serving.sh <fp16\|diffkv\|fp8>` |
| 두 실행 비교 | `bash diffkv-v2/run_docker.sh python diffkv-v2/benchmarks/math500/serving/offline/compare_offline_serving.py --reference <기준 outputs.json> --candidate <비교 outputs.json> --reference-backend <backend> --candidate-backend <backend>` |
| harness test | `bash diffkv-v2/run_docker.sh python diffkv-v2/benchmarks/math500/test_generation.py` |

## 측정 조건

| 항목 | correctness | offline serving | attention latency |
| --- | --- | --- | --- |
| 생성 | EOS까지, greedy, 최대 8192 token | 50문항 동시 도착, 요청별 출력 길이 고정(`ignore_eos`) | decode 1 step |
| 출력 길이 | 모델이 정함 | bringup FP16 EOS 실행의 길이(`output_lengths_sample50.json`, 합계 185,093 token) | 해당 없음 |
| SM clock | 고정 안 함 | 1395 MHz 고정 | 1395 MHz 고정 |
| GPU telemetry | 없음 | 200 ms 간격 | 200 ms 간격 |

- 엔진 설정: eager, chunked prefill on, async scheduling off, prefix caching off, max_model_len 9216, max_num_batched_tokens 9216, max_num_seqs 64, gpu_memory_utilization 0.9
- DiffKV v2: K8V4와 K4V2, threshold `[0.02, 1.0]`, buffer 64 token
- FP8: FlashInfer, `fp8_e4m3` KV cache, scale 1.0
- clock 고정은 docker(`--cap-add=SYS_ADMIN`)로 `nvidia-smi -lgc 1395,1395` 실행

## 결과 저장 위치

| 실행 | 디렉터리 | 파일 |
| --- | --- | --- |
| correctness | `benchmarks/math500/logs/correctness/<UTC>-<sample50\|all500>-<backend>/` | `source_hashes.json`, `outputs.json`, `console.log`, `grades.json` |
| offline serving | `benchmarks/math500/logs/serving/offline/<UTC>-sample50-<backend>/` | `source_hashes.json`, `gpu_telemetry.csv`, `outputs.json`, `console.log` |
| 두 실행 비교 | correctness 또는 serving/offline 아래 `<UTC>-compare-<기준>-vs-<비교>/` | `comparison.json` |
| attention latency | `benchmarks/attention/logs/<UTC>-latency-b<batch>/` | `source_hashes.json`, `gpu_telemetry.csv`, `latency.json`, `console.log` |
| kernel 정확도 검사 | `benchmarks/attention/logs/<UTC>-correctness/` | `correctness.json` |
| Nsight Systems | `benchmarks/attention/logs/<UTC>-nsys-b<batch>-kv<길이>/` | `cuda_events.json`, `kernel_attribution.json` |

경로는 모두 `diffkv-v2/` 아래입니다.

## 기존 결과

| 디렉터리 | 내용 |
| --- | --- |
| `benchmarks/math500/logs/correctness/20261029-sample50/` | 50문항 FP16, DiffKV v2, FP8 EOS 실행과 채점 |
| `benchmarks/math500/logs/correctness/20261030-all500-fp8/` | 500문항 FP8 EOS 실행과 채점 |
| `benchmarks/math500/logs/serving/offline/20261029-sample50/` | 출력 길이 고정 실행. old engine의 temperature 0.6 실행 길이(`output_lengths_used.json`)로 잰 값 |
| `benchmarks/attention/logs/20261029-latency/` | attention latency 요약. CUDA graph ON: `fp8-kernel-latency-b*.log`, OFF: `kernel-eager-b*.log` |

기존 결과는 모두 container 안에서 측정했습니다.

## 제한

- Nsight Systems 측정: `run_docker.sh`가 nsys를 container에 넣지 않아 지금은 그대로 실행되지 않음
- online serving 측정 스크립트 없음
