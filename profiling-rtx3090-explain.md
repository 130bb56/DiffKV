# profiling-rtx3090.md의 실행 명령어와 집계 방법

이 문서는 [profiling-rtx3090.md](/home/130bb56/DiffKV/profiling-rtx3090.md)에 있는 **DiffKV v1, GPU0, 14개 조건**의 측정 과정을 설명합니다. 어떤 명령으로 캡처했고, 어떤 파일이 생성되었으며, 원자료를 어떻게 계산해 각 표에 넣었는지를 기록합니다.

대상 결과 디렉터리는 다음과 같습니다.

```text
/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433
```

데이터의 흐름은 다음과 같습니다.

```text
plan.json의 14개 조건
  -> nsys profile + profile_offline_nsys.py
  -> 조건별 .nsys-rep, worker/profile.json, worker 실행 결과 JSON
  -> nsys export
  -> 조건별 .sqlite
  -> analyze_offline_nsys.py
  -> 조건별 .summary.json
  -> update_report.py의 render()
  -> profiling-rtx3090.md, 조건별 .kernels.csv, all-kernels.csv, all-results.json
```

## 1. 어떤 실행을 측정했는가

조건은 [plan.json](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/plan.json)에 있습니다. 기존 GPU0 평가에서 DiffKV v1과 FP16이 모두 완료된 조건을 선택했고, 이 결과 문서에는 **DiffKV v1 실행**을 기록했습니다. FP16 가중치를 사용한다는 의미와 FP16 vLLM 경로를 실행한다는 의미는 구분해야 합니다. 여기서는 FP16 가중치를 사용하는 DiffKV v1 경로를 실행했습니다.

| Batch | 입력 길이, 토큰 | 출력 길이, 토큰 | 조건 수 |
|---:|---|---|---:|
| 8 | 512, 1024, 2048 | 512, 1024, 2048 | 9 |
| 16 | 512, 1024 | 512, 1024 | 4 |
| 32 | 512 | 512 | 1 |

각 행에서 입력 길이와 출력 길이를 교차 조합합니다. 총 14개이며, 각 조건을 새 프로세스에서 한 번 측정했습니다. GPU 실행은 물리 GPU0에서 순차적으로 수행했습니다.

| 항목 | 실제 설정 |
|---|---|
| 모델, tokenizer | `/home/130bb56/DiffKV/.cache/models/Meta-Llama-3-8B` |
| 입력 | `logs/performance/llama-matched-inputs.json`의 길이별 고정 token prompt를 batch 전체에 복제 |
| backend, variant | `diffkv`, `no-sync-bulk-lengths` |
| weight dtype, load format | `float16`, `safetensors` |
| GPU | 물리 GPU0, NVIDIA GeForce RTX 3090 |
| tensor parallel | 1 |
| 실행 방식 | eager, `enforce_eager=True` |
| 엔진 한계 | `max_model_len=4096`, `max_num_seqs=32`, `max_num_batched_tokens=8192`, `max_paddings=4096` |
| 메모리 설정 | `swap_space=0`, `gpu_memory_utilization=0.9`, `kv_buffer_size=64` |
| 생성 설정 | `temperature=0`, `ignore_eos=True`, `n=1`, `max_tokens=조건별 출력 길이` |
| DiffKV 생성 설정 | `quant_configs=[8,4,4,2]`, `quant_groups=[1,1,1,1]`, `compress_configs=[0.02,1.0]` |
| seed | 42 |
| warmup | 동일 batch, 입력 길이에서 출력 16토큰, 측정에서 제외 |
| 측정 횟수 | 조건당 1회 |

설정의 실제 저장 위치는 조건별 worker 결과 JSON의 `common_engine_args`, `backend_engine_args`, `sampling`, `diffkv_generate_args`입니다. 실행 중 모든 decode step의 batch가 지정한 B인지, preemption이 없는지도 worker가 기록합니다.

## 2. 실행 명령어

### 2.1 전체 조건의 실행을 담당한 스크립트

작업 디렉터리는 `/home/130bb56/DiffKV`입니다. 전체 캡처는 [collect.py](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/collect.py)가 담당하고, 완료된 캡처의 변환과 분석은 [analyze_all.py](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/analyze_all.py)가 담당합니다. 두 스크립트의 진입 명령은 다음과 같습니다. 분석 스크립트는 캡처 완료를 기다리므로 캡처와 함께 실행할 수 있습니다.

```bash
cd /home/130bb56/DiffKV

.cache/baselines/fp16-env/bin/python \
  logs/performance/nsys-workloads-20261018-151433/collect.py
```

```bash
cd /home/130bb56/DiffKV

.cache/baselines/fp16-env/bin/python \
  logs/performance/nsys-workloads-20261018-151433/analyze_all.py
```

아래 명령어들은 완료된 결과를 만든 절차를 설명하는 용도입니다. 새 측정에는 별도 결과 디렉터리와 새 worker 디렉터리를 사용해야 합니다. `profile_offline_nsys.py`는 기존 worker 디렉터리를 재사용하지 않습니다.

### 2.2 조건 하나를 캡처한 실제 명령어

다음은 **B8, 입력 512, 출력 1024**의 [command.json](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl1024.command.json)에 저장된 인자를 shell 형태로 옮긴 것입니다. `collect.py`가 지정한 작업 디렉터리와 `PYTHONPATH`도 함께 표시했습니다.

```bash
cd /home/130bb56/DiffKV

PYTHONPATH=/home/130bb56/DiffKV \
/opt/nvidia/nsight-systems-cli/2024.4.1/target-linux-x64/nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --sample=none \
  --cpuctxsw=none \
  --output=/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl1024 \
  /home/130bb56/DiffKV/.cache/baselines/fp16-env/bin/python \
  /home/130bb56/DiffKV/benchmarks/profile_offline_nsys.py \
  --backend diffkv \
  --gpu 0 \
  --batch 8 \
  --input-length 512 \
  --output-length 1024 \
  --output-dir /home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl1024-worker
```

| 인자 | 이 측정에서의 역할 |
|---|---|
| `--trace=cuda,nvtx,osrt` | CUDA API, GPU kernel, 메모리 작업, NVTX 구간, OS runtime 이벤트를 수집합니다. |
| `--capture-range=cudaProfilerApi` | Python 코드의 `cudaProfilerStart()`부터 수집합니다. 모델 로딩과 warmup은 수집 범위 밖에 둡니다. |
| `--capture-range-end=stop` | `cudaProfilerStop()`에서 해당 캡처를 끝냅니다. |
| `--sample=none`, `--cpuctxsw=none` | CPU sampling과 CPU context switch trace를 비활성화합니다. |
| `--backend diffkv`, `--gpu 0` | DiffKV 실행을 선택하고, worker가 `CUDA_VISIBLE_DEVICES=0`으로 물리 GPU0을 사용합니다. |
| `--batch`, `--input-length`, `--output-length` | 측정할 요청 수, 입력 토큰 수, 요청별 생성 토큰 수를 지정합니다. |
| `--output` | Nsight report 파일의 경로 접두사입니다. |
| `--output-dir` | NVTX phase metadata와 worker 실행 결과를 저장할 디렉터리입니다. |

### 2.3 Report를 SQLite로 변환한 명령어

`analyze_all.py`는 캡처가 완료된 조건마다 다음 명령을 실행합니다.

```bash
/opt/nvidia/nsight-systems-cli/2024.4.1/target-linux-x64/nsys export \
  --type sqlite \
  --output /home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl1024.sqlite \
  /home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl1024.nsys-rep
```

`.nsys-rep`에 들어 있는 trace를 SQL로 읽을 수 있는 `.sqlite`로 변환합니다. 이 단계에서 모델을 다시 실행하지 않습니다.

### 2.4 SQLite를 분석한 명령어

```bash
/home/130bb56/DiffKV/.cache/baselines/fp16-env/bin/python \
  /home/130bb56/DiffKV/benchmarks/analyze_offline_nsys.py \
  /home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl1024.sqlite \
  --capture-metadata /home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl1024-worker/profile.json \
  --output /home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl1024.summary.json
```

분석 코드는 SQLite를 읽기 전용으로 열고, NVTX, CUDA API, GPU 이벤트의 시각과 대응 관계를 집계합니다. 이 단계에서는 모델을 다시 실행하지 않습니다.

### 2.5 표와 CSV를 생성한 처리

`analyze_all.py`는 조건별 분석 후 [update_report.py](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/update_report.py)를 불러와 `render()`를 호출합니다. `render()`는 `plan.json`과 존재하는 모든 조건의 `.summary.json`을 읽고, 결과 MD, CSV, 전체 조건 JSON을 갱신합니다.

같은 렌더러를 단독으로 호출하는 진입점은 다음과 같습니다. 이 명령은 `profiling-rtx3090.md`도 다시 작성합니다.

```bash
/home/130bb56/DiffKV/.cache/baselines/fp16-env/bin/python \
  /home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/update_report.py
```

## 3. 어떤 파일에 무엇이 들어 있는가

대표 조건을 `b8-isl512-osl512`로 잡으면 주요 파일의 배치는 다음과 같습니다.

```text
/home/130bb56/DiffKV/
├── profiling-rtx3090.md                      # 표와 설명을 정리한 결과
├── profiling-rtx3090-explain.md              # 이 문서
├── benchmarks/
│   ├── profile_offline_nsys.py              # NVTX 표시, 측정 범위 지정
│   ├── run_offline_e2e.py                   # 모델, 입력, 생성 처리 실행
│   └── analyze_offline_nsys.py              # SQLite에서 시간과 phase 집계
└── logs/performance/nsys-workloads-20261018-151433/
    ├── plan.json                           # 14조건, 측정 설정, 선정 근거
    ├── collect.py                          # 조건별 순차 캡처
    ├── analyze_all.py                      # export, 분석, render 실행
    ├── update_report.py                    # JSON에서 MD, CSV 생성
    ├── progress.json                       # 캡처 진행 상태
    ├── measurement-audit.json              # 전체 조건의 집계 정합성 확인 결과
    ├── all-results.json                    # 전체 조건의 summary 통합
    ├── all-kernels.csv                     # 전체 조건, 전체 scope의 커널 목록
    ├── b8-isl512-osl512.nsys-rep             # Nsight 원본 report
    ├── b8-isl512-osl512.sqlite               # SQL로 읽기 위해 변환한 trace
    ├── b8-isl512-osl512.summary.json         # 조건별 집계값, 검사 결과
    ├── b8-isl512-osl512.kernels.csv          # 조건별, scope별 전체 커널
    ├── b8-isl512-osl512.capture.log          # 프로파일러, worker의 표준 출력
    └── b8-isl512-osl512-worker/
        ├── profile.json                    # NVTX engine_step과 scheduler phase 대응
        └── b8-isl512-osl512-diffkv.json      # 모델 설정, 입력 hash, GPU, 생성 실적
```

첫 조건 이외의 13조건에는 실행 인자를 저장한 `<case>.command.json`, 변환 로그 `<case>.export.log`, 분석 로그 `<case>.analysis.log`도 있습니다. 커널 목록과 수치의 근거는 CSV, summary, SQLite에서 확인합니다.

| 확인할 내용 | 파일, 필드 |
|---|---|
| 무엇을 실행했는가 | `plan.json`, `<case>.command.json`, worker 결과의 engine 설정 |
| 어떤 GPU, 입력, 빌드였는가 | worker 결과의 `physical_gpu`, `gpu_uuid`, `input_sha256`, `input_manifest_sha256`, `local_build` |
| 전체의 몇 초, 몇 %인가 | `.summary.json`의 `wall_ms`, `wall_breakdown` |
| prefill과 decode 시간 | `.summary.json`의 `phases.prefill`, `phases.decode` |
| 전체 커널 이름, 호출 수, 합계 시간 | `.kernels.csv`, `all-kernels.csv` |
| 커널 grid, block, 레지스터 수, shared memory | `.summary.json`의 `top_kernels[].configurations` |
| 개별 CUDA launch와 GPU 이벤트 | `.sqlite`, 또는 `.nsys-rep`의 타임라인 |
| 분류, 귀속, 합계의 정합성 | `.summary.json`의 `checks`, `warnings`, 전체 조건의 `measurement-audit.json` |

## 4. 어디부터 어디까지를 E2E로 측정했는가

[profile_offline_nsys.py](/home/130bb56/DiffKV/benchmarks/profile_offline_nsys.py:173)는 모델 로딩과 warmup 이후, 측정 대상 생성 호출을 다음 순서로 감쌉니다.

```text
cudaProfilerStart()
  NVTX measured_generation 시작
    generate()
    torch.cuda.synchronize()    # 이 생성의 GPU 처리 완료를 기다림
  NVTX measured_generation 종료
torch.cuda.synchronize()        # NVTX 범위 밖의 추가 동기화
cudaProfilerStop()
report 저장, 결과 JSON 저장
```

전체 시간 T는 `NVTX_EVENTS`의 `measured_generation`에서 `end - start`로 구합니다. SQLite 시각의 단위는 ns이므로 `wall_ms = (end - start) / 1,000,000`입니다. 생성에는 요청 등록, prefill, decode, sampling, detokenization, 요청 해제를 포함합니다.

**결과 MD의 E2E에는 worker 결과 JSON의 `elapsed_s`를 사용하지 않았습니다.** worker 타이머는 프로파일러 종료 처리를 포함하기 때문입니다. 첫 조건에서 worker의 `elapsed_s`는 27.766137초이고, 표에 사용한 NVTX 구간은 14.085702474초입니다. report 저장을 포함하지 않는 후자를 사용했습니다.

## 5. SQLite에서 무엇을 읽었는가

[analyze_offline_nsys.py](/home/130bb56/DiffKV/benchmarks/analyze_offline_nsys.py:128)가 주로 읽는 테이블은 다음과 같습니다.

| 테이블 | 주요 필드 | 용도 |
|---|---|---|
| `NVTX_EVENTS` | `start`, `end`, `text`, `textId`, `globalTid` | E2E 범위, engine step의 CPU 범위 식별 |
| `StringIds` | `id`, `value` | kernel, CUDA API, NVTX 이름의 ID를 문자열로 변환 |
| `CUPTI_ACTIVITY_KIND_KERNEL` | `start`, `end`, `demangledName`, `correlationId`, `deviceId`, `globalPid`, 실행 구성 | 개별 GPU kernel의 시간, 이름, CUDA API와 대응 관계 확보 |
| `CUPTI_ACTIVITY_KIND_MEMCPY` | `start`, `end`, `correlationId`, `bytes`, `copyKind` | GPU 복사 시간과 전송량 확보 |
| `CUPTI_ACTIVITY_KIND_MEMSET` | `start`, `end`, `correlationId`, `bytes` | GPU memset 시간 확보 |
| `CUPTI_ACTIVITY_KIND_RUNTIME` | `start`, `end`, `globalTid`, `correlationId`, `nameId` | CUDA API에서 launch를 수행한 engine step 식별 |

분석 코드는 `CUPTI_ACTIVITY_KIND_DRIVER`가 있으면 동일하게 읽습니다. 이번 실측 SQLite에는 RUNTIME 테이블이 있습니다.

GPU 이벤트는 `measured_generation`과 겹치는 것으로 제한하고, `deviceId=0`과 대상 프로세스의 `globalPid`로 선택합니다. worker에는 물리 GPU0만 노출하므로 해당 GPU가 CUDA의 논리 device 0입니다.

## 6. Prefill, decode에 어떻게 귀속했는가

**실제 scheduler의 판정을 저장하고, CUDA launch와의 대응 관계로 나눴습니다.**

1. `engine.step()` 전체를 `engine_step_0`, `engine_step_1`, ...의 NVTX로 감쌉니다. 이 외부 범위는 scheduling과 output 처리를 포함합니다.
2. scheduler가 반환한 `prompt_run`을 읽고, 해당 step의 phase를 `prefill` 또는 `decode`로 `worker/profile.json`의 `engine_steps`에 기록합니다. 내부에는 `prefill`, `decode_step_N`의 NVTX도 표시합니다.
3. GPU kernel의 `correlationId`로 대응하는 CUDA API를 찾습니다.
4. CUDA API의 시작 시각과 `globalTid`를 이용해 이를 포함하는 외부 `engine_step_N`을 찾습니다.
5. 그 step의 metadata에 기록한 phase로 kernel의 실행 시간 전체를 귀속합니다.

GPU는 비동기로 실행됩니다. GPU 시작 시각을 CPU의 prefill, decode 범위와 비교하면 다른 step으로 분류할 가능성이 있어, launch를 수행한 step을 추적했습니다. 외부와 내부 NVTX를 함께 더하면 중복 계산되므로, 이번 집계는 외부 `engine_step_N`만 CPU 범위로 사용했습니다.

Prefill은 여러 step에 나뉠 수 있습니다. 입력이 나뉜 조건은 해당 step을 모두 prefill로 합산했습니다. 이번 decode는 모든 조건에서 `출력 길이 - 1`step입니다. 첫 출력 토큰은 prefill에서 생성하기 때문입니다.

## 7. 각 표를 어떻게 계산했는가

### 7.1 전체 생성 시간의 분해

이 표는 `.summary.json`의 **`wall_breakdown`**에서 만들었습니다. 모든 kernel, memcpy, memset의 시작과 종료를 시간순으로 정렬하고, 각 시간 조각을 다음 우선순위에 따라 하나의 열에만 배정합니다.

| 시간 조각의 상태 | 배정할 열 |
|---|---|
| 한 kernel 분류만 실행 중 | 해당 분류의 열. 같은 분류의 kernel끼리 겹쳐도 한 번만 셉니다. |
| 서로 다른 kernel 분류가 동시 실행 중 | `GPU kernel 중첩` |
| kernel 없이 memcpy 또는 memset 실행 중 | `Copy, memset만 실행` |
| kernel, memcpy, memset 기록 없음 | `GPU 실행 없음` |

Kernel과 copy가 겹친 시간은 kernel에 배정합니다. 이렇게 하면 각 열이 서로 중복되지 않고, 합계가 E2E 시간 T, 비율 합계가 100%가 됩니다.

분류에는 [classify_kernel()](/home/130bb56/DiffKV/benchmarks/analyze_offline_nsys.py:30)의 이름 규칙을 순서대로 적용합니다.

| 표의 열 | 내부 분류 | 이름에 따른 분류 예시 |
|---|---|---|
| GEMM | `gemm_including_split_k_reduce` | `gemm`, `cutlass`, `splitkreduce`, `split_k`, `gemv` |
| Attention | `attention_main`, `attention_reduce`, `attention_other_or_fp16` 합계 | sparse paged attention 본체, vLLM reduce, `attn_fwd` 등 |
| KV 갱신 | `kv_cache` | `cache`, `quantize`, `evict`, `compact`, `prun` 등 |
| 기타 GPU | `other` | 위 규칙에 해당하지 않는 kernel. RMSNorm, SiLU 등도 포함합니다. |

Attention의 reduce, 본체, 기타 attention, KV, GEMM 순서로 판정합니다. GEMM에는 split-K 집계 kernel도 포함합니다. 따라서 GEMM 열에는 행렬곱 본체 이외의 집계 시간도 들어 있습니다. 분류 기준은 kernel 이름이며, FLOPs나 전송량을 측정해 분류한 것은 아닙니다.

셀은 `시간ms / 1000`초와 `그 시간ms / wall_ms × 100`%로 표시합니다. 이번 14조건은 kernel끼리 겹친 시간이 0이므로, 분류별 호출 시간 합계와 이 표에 사용한 kernel 구간 시간은 일치했습니다.

### 7.2 Prefill과 decode

| 열 | summary 참조값, 계산 |
|---|---|
| Prefill, decode 횟수 | `phases.<phase>.scope_count` |
| Phase latency | 해당 phase의 외부 `engine_step_N` 범위 합계인 `cpu_scope_ms` |
| Phase GPU kernel | 해당 phase로 귀속한 kernel 호출 시간 합계인 `kernel_sum_ms` |
| Latency의 전체 비율 | `cpu_scope_ms / wall_ms × 100` |
| GPU kernel의 전체 비율 | `kernel_sum_ms / wall_ms × 100` |
| Decode latency, ms/step | `phases.decode.cpu_scope_ms / phases.decode.scope_count` |
| Decode GPU, ms/step | `phases.decode.kernel_sum_ms / phases.decode.scope_count` |

`cpu_scope_ms`는 CPU에서 감싼 범위의 경과 시간이며, CPU가 계산한 시간만을 뜻하지 않습니다. 구간 안의 GPU 대기도 포함합니다. CPU 범위와 GPU kernel 시간은 겹치므로 더해서 E2E를 계산하지 않습니다. Engine step 바깥의 요청 등록 등은 E2E에 포함되고, phase latency 합계에는 포함되지 않습니다.

### 7.3 Decode의 연산 분류

`phases.decode.categories`를 GEMM, Attention, KV, 기타로 묶습니다.

```text
ms/decode step = decode에 귀속한 분류별 kernel 시간 합계ms / decode step 수
괄호 안의 %   = decode에 귀속한 분류별 kernel 시간 합계ms / E2E wall_ms × 100
```

괄호 안의 분모는 **E2E 전체 시간**입니다. 이 표에는 prefill, copy, GPU 실행이 없는 구간이 없으므로 네 열의 % 합계가 100%가 되지 않습니다.

### 7.4 E2E 전체, prefill, decode 각각의 kernel 실행 시간

각 scope에서 같은 이름의 kernel을 묶고, 호출 시간 합계가 큰 순서로 정렬합니다. MD는 상위 5개를 표시하고, CSV는 나머지도 저장합니다.

| 항목 | 계산 |
|---|---|
| 호출 수 | 해당 scope의 같은 이름 kernel 호출 건수 |
| 합계, 초 | 같은 이름 kernel의 `end - start` 합계 / 1,000,000,000 |
| 해당 범위 GPU 합 대비 % | 같은 이름 kernel 시간 합계 / 해당 scope의 전체 kernel 시간 합계 × 100 |
| E2E 대비 % | 같은 이름 kernel 시간 합계 / E2E wall 시간 × 100 |
| 호출 평균, μs | 같은 이름 kernel 시간 합계ms × 1000 / 호출 수 |

CSV의 `scope_wall_pct`는 E2E 행에서는 E2E wall 시간, Prefill과 Decode 행에서는 해당 phase의 CPU 범위 합계를 분모로 사용합니다. `scope_gpu_kernel_sum_pct`와 `whole_wall_pct`는 각각 표의 “해당 범위 GPU 합 대비 %”와 “E2E 대비 %”입니다.

**E2E 행과 Prefill, Decode 행을 함께 합산하면 중복 계산됩니다.** E2E는 같은 kernel 실행을 이미 포함합니다. 일반적으로 kernel 호출 시간 합계에는 동시 실행분도 들어가므로, wall 시간 분해와 구분해야 합니다.

## 8. 실제 수치를 한 행 따라가기

예시는 **B8, 입력512, 출력512**입니다. [summary JSON](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl512.summary.json), [kernel CSV](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/b8-isl512-osl512.kernels.csv)를 참조합니다.

| 항목 | 원자료, 계산 | MD 표시 |
|---|---|---|
| 전체 시간 | `wall_ms = 14085.702474` | 14.086초 |
| GEMM | `10655.404987 / 14085.702474 × 100` | 10.655초, 75.65% |
| Attention | `(1168.508987 + 78.478751 + 15.538397) ms` | 1.263초, 8.96% |
| KV 갱신 | `408.004322 ms` | 0.408초, 2.90% |
| 기타 GPU | `448.610134 ms` | 0.449초, 3.18% |
| Copy, memset만 실행 | `14.677459 + 0.150383 ms` | 0.015초, 0.11% |
| GPU 실행 없음 | `1296.329054 ms` | 1.296초, 9.20% |
| Decode latency | `13202.694272 / 511 ms` | 25.837 ms/step |
| Decode GPU | `11934.275945 / 511 ms` | 23.355 ms/step |
| Decode GEMM | `9869.476680 / 511 ms`, 비율은 `9869.476680 / 14085.702474 × 100` | 19.314 ms/step, 70.07% |

최상위 `cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x64_32x4_tn_align8`는 16,352회 호출됐고, 합계는 5020.275458 ms입니다.

```text
호출 평균         = 5020.275458 × 1000 / 16352 = 307.012932 μs
E2E 대비 비율     = 5020.275458 / 14085.702474 × 100 = 35.640931%
GPU 합계 내 비율  = 5020.275458 / 12774.545578 × 100 = 39.299053%
```

같은 kernel 시간이라도 분모가 달라 35.64%와 39.30%라는 두 값이 나옵니다.

## 9. 직접 파일을 읽어 확인하는 방법

아래는 기존 결과를 읽는 예시입니다. 첫 조건의 전체 시간, 분해값, phase 값, 검사 결과를 표시합니다.

```bash
cd /home/130bb56/DiffKV

.cache/baselines/fp16-env/bin/python - <<'PYCODE'
import json
from pathlib import Path

run = Path('logs/performance/nsys-workloads-20261018-151433')
s = json.loads((run / 'b8-isl512-osl512.summary.json').read_text())
print('E2E초:', s['wall_ms'] / 1000)
for row in s['wall_breakdown']:
    print(row['name'], row['time_ms'], row['whole_wall_pct'])
for phase in ('prefill', 'decode'):
    p = s['phases'][phase]
    print(phase, 'step 수:', p['scope_count'],
          'latency초:', p['cpu_scope_ms'] / 1000,
          'GPU kernel초:', p['kernel_sum_ms'] / 1000)
print('checks:', s['checks'])
PYCODE
```

전체 조건 CSV에서 특정 조건의 Decode만 선택하고 시간순으로 표시하는 예시입니다. `scope`를 `E2E 전체` 또는 `Prefill`로 바꾸면 각각을 확인할 수 있습니다.

```bash
cd /home/130bb56/DiffKV

.cache/baselines/fp16-env/bin/python - <<'PYCODE'
import csv
from pathlib import Path

path = Path('logs/performance/nsys-workloads-20261018-151433/all-kernels.csv')
with path.open(newline='') as f:
    rows = [r for r in csv.DictReader(f)
            if r['case'] == 'b8-isl512-osl512' and r['scope'] == 'Decode']
for r in sorted(rows, key=lambda r: float(r['time_ms']), reverse=True)[:10]:
    print(r['calls'], r['time_ms'], r['whole_wall_pct'], r['kernel'])
PYCODE
```

## 10. 집계에서 확인한 내용과 해석 범위

[measurement-audit.json](/home/130bb56/DiffKV/logs/performance/nsys-workloads-20261018-151433/measurement-audit.json)과 각 summary의 `checks`에서 전체 14조건을 확인했습니다.

| 확인 항목 | 결과 |
|---|---|
| wall 분해 합계와 `measured_generation` | 전체 14조건 일치 |
| phase별 kernel 호출 수, 시간 합계와 E2E | 전체 14조건 일치 |
| kernel phase 귀속률, 호출 수와 시간 | 전체 14조건 모두 100% |
| metadata step 수와 NVTX 외부 범위 수 | 전체 14조건 일치 |
| GPU 이벤트의 E2E 경계에서 잘린 건수 | kernel, 메모리 작업 모두 전체 14조건 0건 |
| decode batch, decode step 수, preemption | 지정 B 유지, 출력 길이−1step, preemption 없음 |
| GPU, 입력 manifest, 실행 코드, 확장 바이너리 기록 | 같은 GPU0, 같은 manifest, runner, 확장 바이너리 |
| 전체 kernel 기록 수, CSV 행 수 | 6,237,564회 호출, scope별 CSV 1,379행 |

이는 측정 범위와 집계 정합성의 확인입니다. 모델 출력의 수치적 정확성을 검증하는 테스트는 이번 프로파일링 실행에 포함하지 않았습니다.

이 결과는 각 조건에서 어떤 처리가 얼마의 시간을 사용했는지 보여줍니다. 이 14조건의 결과에는 Nsight Compute 카운터 측정, theoretical peak 비교, 연산량과 전송량에 따른 시간 하한 계산은 포함하지 않았습니다. 따라서 “어떤 GEMM이 E2E의 35.64%를 사용했다”는 사실은 확인할 수 있지만, 그 비중만으로 비효율이나 개선 가능한 병목을 판정하지 않았습니다.

“GPU 실행 없음”은 대상 프로세스, 대상 GPU에 kernel, copy, memset 기록이 없는 시간입니다. 그 원인이 Python, CPU 처리, 동기화 중 무엇인지는 이 열만으로 특정할 수 없습니다. Nsight Systems를 붙인 wall 시간에는 계측 부하가 포함됩니다.

각 터미널에서 먼저 실행합니다.

```bash
cd /home/130bb56/DiffKV
PROFILE_RUN=/home/130bb56/DiffKV/logs/performance/nsys-base1395-repeat3-20261019-001155
```

**터미널 1: 모델 실행과 GPU 모니터링**

```bash
python -u $PROFILE_RUN/collect.py
```

중단 후 재개할 때는 위 명령 대신 실행합니다.

```bash
python -u $PROFILE_RUN/collect_remaining.py
```

**터미널 2: 완료된 trace 분석**

터미널 1과 동시에 실행합니다.

```bash
python -u $PROFILE_RUN/analyze_all.py
```

**두 작업이 모두 끝난 뒤: 터미널 1에서 순서대로 실행했던 명령**

```bash
# 한 조건 재측정
python -u $PROFILE_RUN/collect_pair_alignment.py

# 로그와 집계 검증
python $PROFILE_RUN/audit_telemetry.py --require-complete
python -u $PROFILE_RUN/audit_results.py

# TOBE 표 저장
python $PROFILE_RUN/render_report.py --publish
```
