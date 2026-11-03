# DiffKV bringup (vLLM 0.32.0)

vLLM v0.32.0(`5df63a7`) 위에 DiffKV attention과 KV cache 관리를 붙인 코드입니다. DiffKV 전용 파일은 모두 이 디렉터리에 있습니다. 아래 경로는 repo 루트(`/home/130bb56/DiffKV`) 기준입니다.

## 디렉터리

```text
diffkv/
├── run_docker.sh                     container 안에서 명령 실행
├── python/diffkv/                    DiffKV Python 코드 (backend, cache 관리, admission, model runner hook, extension build)
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
    │   ├── serving/offline/          run_offline_serving.sh, analyze_nsys_steps.py (nsys step별 GPU 시간 분해)
    │   ├── serving/online/           
    │   └── logs/
    └── attention/
        ├── run_latency.sh            batch별 decode latency 실행
        ├── benchmark_latency.py      DiffKV, vLLM FA2 FP16, FlashInfer FP8 decode latency benchmark
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
- git worktree에서 실행할 때: worktree 루트의 `.venv`, `.toolchains`, `models`, `.cache`와 `vllm/` 안의 미리 빌드된 파일(`git status --ignored vllm`으로 보이는 `.so`, `_version.py`, `third_party/` 등)을 기본 checkout으로 가리키는 symlink로 만들어 둠. `run_docker.sh`는 기본 checkout을 read-only로(`.cache`만 쓰기 가능) 같이 mount하고, `PYTHONPATH`로 worktree의 `diffkv`와 `vllm`을 먼저 import함

## 실행 명령

repo 루트에서 실행합니다.

| 작업 | 명령 |
| --- | --- |
| extension build | `bash diffkv/run_docker.sh python -m diffkv.build_ops` |
| kernel 정확도 검사 | `bash diffkv/run_docker.sh python diffkv/benchmarks/attention/correctness.py` |
| attention latency | `bash diffkv/benchmarks/attention/run_latency.sh [batch ...]` (기본 1 8 32, KV 길이 1024 2048 4096 8192) |
| attention profile (Nsight Systems) | `bash diffkv/benchmarks/attention/run_profile.sh [batch ...]` (기본 1 8 50, KV 길이 512 2048 8192) |
| correctness | `bash diffkv/benchmarks/math500/correctness/run_correctness.sh <fp16\|diffkv\|fp8> <sample50\|all500>` |
| offline serving | `bash diffkv/benchmarks/math500/serving/offline/run_offline_serving.sh <fp16\|diffkv\|fp8>` |
| offline serving step 분해 (Nsight Systems) | `DIFFKV_NSYS_STEPS="<첫 step> <step 수>" bash diffkv/benchmarks/math500/serving/offline/run_offline_serving.sh <fp16\|diffkv\|fp8>`: 측정 구간의 engine step(0부터 셈) 중 지정한 구간만 trace하고 그 뒤 실행을 멈춤. 결과 디렉터리 이름에 `-nsys-s<첫 step>-n<step 수>`, `trace.nsys-rep`, `trace.sqlite`, `step_breakdown.json`(step별 wall 시간, GPU busy와 idle 시간, kernel 종류별 시간, 동기화 API 호출 수) 추가 |
| 두 실행 비교 | `bash diffkv/run_docker.sh python diffkv/benchmarks/math500/serving/offline/compare_offline_serving.py --reference <기준 outputs.json> --candidate <비교 outputs.json> --reference-backend <backend> --candidate-backend <backend>` |
| harness test | `bash diffkv/run_docker.sh python diffkv/benchmarks/math500/test_generation.py` |

## 측정 조건

| 항목 | correctness | offline serving | attention latency |
| --- | --- | --- | --- |
| 생성 | EOS까지, greedy, 최대 8192 token | 50문항 동시 도착, 요청별 출력 길이 고정(`ignore_eos`) | decode 1 step |
| 출력 길이 | 모델이 정함 | bringup FP16 EOS 실행의 길이(`output_lengths_sample50.json`, 합계 185,093 token) | 해당 없음 |
| SM clock | 고정 안 함 | 1395 MHz 고정 | 1395 MHz 고정 |
| GPU telemetry | 없음 | 200 ms 간격 | 200 ms 간격 |

- 엔진 설정: eager, chunked prefill on, async scheduling off, prefix caching off, max_model_len 9216, max_num_batched_tokens 9216, max_num_seqs 64, gpu_memory_utilization 0.9
- DiffKV: K8V4와 K4V2, threshold `[0.02, 1.0]`, buffer 64 token
- FP8: FlashInfer, `fp8_e4m3` KV cache, scale 1.0
- clock 고정은 docker(`--cap-add=SYS_ADMIN`)로 `nvidia-smi -lgc 1395,1395` 실행

### offline serving의 step별 동시 decode 요청 수

`serving/offline/output_lengths_sample50.json`의 요청별 출력 길이에서 계산한 값. step은 측정 구간의 engine step(0부터 셈), 요청 수는 출력 길이가 그 step보다 긴 요청 수. DiffKV가 처음 약 50 step 동안 새 요청을 step마다 하나씩만 받는 것과 FP8의 preemption은 빠져 있음. 실제 값은 nsys 실행의 `outputs.json` `profile.per_step`에 step마다 기록됨. 문맥 길이는 prompt 길이(32~776 token, 중앙값 83)에 step을 더한 값

| step | 0~783 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 5000 | 6000 | 7000 | 7348~8191 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 동시 decode 요청 | 50 | 46 | 40 | 27 | 25 | 20 | 19 | 17 | 14 | 13 | 12 | 11 |

nsys step 분해 구간 (2026-11-03): 동시 decode 요청이 가장 많은 구간, 중간 구간, 가장 적은 구간에서 32 step씩

| 구간 | `DIFFKV_NSYS_STEPS` | 동시 decode 요청 |
| --- | --- | --- |
| 최대 | `740 32` | 50 |
| 중간 | `3000 32` | 20 |
| 최소 | `7800 32` | 11 |

## 제한

- online serving 측정 스크립트 없음
