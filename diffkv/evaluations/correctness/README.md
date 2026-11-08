## Correctness 결과

공통 조건: DeepSeek-R1-Distill-Llama-8B, RTX 3090, clock 고정 없음, greedy, 문항당 1개 생성

| 데이터셋 | backend (GPU) | 코드 | chunked prefill | 출력 상한 (OSL) | max_model_len | 정답 | 출력 상한에서 잘린 문항 | 답 없음 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MATH-500 500문제 (11/05) | fp16 (GPU1) | main e08375cff7 | on | 8192 | 9216 | 404/500 (80.8%) | 80 | 68 |
| MATH-500 500문제 (11/05) | diffkv (GPU0) | main e08375cff7 | on | 8192 | 9216 | 404/500 (80.8%) | 79 | 71 |
| MATH-500 500문제 (11/05) | fp8 (GPU0) | main e08375cff7 | on | 8192 | 9216 | 397/500 (79.4%) | 88 | 83 |
| MATH-500 500문제 (11/07) | fp16 (GPU0) | bringup-prefill-opt-main 4d30e41fab | off | 8192 | 9216 | 398/500 (79.6%) | 86 | 72 |
| MATH-500 500문제 (11/07) | diffkv (GPU1) | bringup-prefill-opt-main 4d30e41fab | off | 8192 | 9216 | 389/500 (77.8%) | 98 | 79 |
| MATH-500 500문제 (11/07) | fp8 (GPU0) | bringup-prefill-opt-main 4d30e41fab | off | 8192 | 9216 | 401/500 (80.2%) | 79 | 69 |
| MATH-500 500문제 (11/08) | fp16 (GPU0) | bringup-prefill-opt-main 4d30e41fab | off | 8192 | 9216 | 398/500 (79.6%) | 86 | 72 |
| MATH-500 500문제 (11/08) | fp16 (GPU1) | bringup-prefill-opt-main 4d30e41fab | on | 8192 | 9216 | 404/500 (80.8%) | 80 | 68 |
| GPQA-diamond 197문항 (11/06) | fp8 (GPU0) | main e08375cff7 + 커밋 전 GPQA 변경 | on | 8192 | 9216 | 45/197 (22.8%) | 130 | 130 |
| GPQA-diamond 197문항 (11/06) | diffkv (GPU1) | main e08375cff7 + 커밋 전 GPQA 변경 | on | 8192 | 9216 | 42/197 (21.3%) | 132 | 133 |
| GPQA-diamond 197문항 (진행 중, 약 61분 시점) | fp8 (GPU0) | main e08375cff7 + 커밋 전 GPQA 변경 | on | 16384 | 17,408 | 중간 16/75 (21.3%) | 44/75 | 44/75 |
| GPQA-diamond 197문항 (진행 중, 약 61분 시점) | diffkv (GPU1) | main e08375cff7 + 커밋 전 GPQA 변경 | on | 16384 | 17,408 | 중간 28/96 (29.2%) | 49/96 | 49/96 |

### GPQA 출력 상한 8192와 16384 비교 (16384 실행에서 지금까지 끝난 같은 문항)

| backend (GPU) | 끝난 문항 | 정답, 출력 상한 8192 | 정답, 출력 상한 16384 | 잘림, 출력 상한 8192 | 잘림, 출력 상한 16384 |
| --- | --- | --- | --- | --- | --- |
| fp8 (GPU0) | 75 | 17 | 16 | 50 | 44 |
| diffkv (GPU1) | 96 | 24 | 28 | 58 | 49 |

### 용어와 설정
- 출력 상한(OSL): 요청 하나가 생성하는 최대 token 수, 그 안에 EOS가 없으면 그 자리에서 멈춤 → 최종 답을 쓰기 전에 끝나서 오답
- max_model_len: prompt와 출력의 합의 상한, 출력 상한 + 1024 token(prompt 몫)으로 설정
- 모델 자체 상한: 131,072 token (config.json `max_position_embeddings`)
- 메모리상 max_model_len 상한 (gpu_memory_utilization 0.9, 실행 로그의 KV cache 용량): FP16 42,880, fp8 85,776, DiffKV 28,782 token
- GPQA 127번 제외: prompt 2,813 token이 1024 token을 넘어 max_model_len 초과
- GPQA 8192 실행의 잘린 출력: fp8 130개 중 127개, diffkv 132개 중 125개가 `</think>` 전에 잘림, fp8 70개와 diffkv 74개는 끝부분이 같은 token 묶음의 반복
- GPQA 공식 수치: DeepSeek 49.0% (최대 생성 32,768 token, temperature 0.6, 문항당 64개 생성)
- MATH-500 fp8 실행: outputs.json에 working_tree_dirty true로 기록됨, 시작 시 source_hashes.json에는 diffkv와 vLLM 수정 파일의 변경 없음

### FP16 점수가 11/05 404에서 11/07 398로 바뀐 원인: chunked prefill

11/05와 11/07 사이에 vLLM 쪽 변경은 없다(`git diff --name-only e08375cff7 4d30e41fab -- vllm/`가 0개 파일, 변경은 `diffkv/`와 `.gitignore`뿐). FP16 실행에서 달라진 것은 engine 설정 `enable_chunked_prefill` 하나이고, 11/08에 같은 commit에서 이 설정만 바꿔 500문제를 양쪽으로 돌렸다.

| 실행 | 코드 | chunked prefill | 정답 | 잘린 문항 | 답 없음 | 잘리지 않은 문항만의 정답률 |
| --- | --- | --- | --- | --- | --- | --- |
| 20261102T091120Z-all500-fp16 | 2b2461805f | on | 404/500 (80.8%) | 80 | 68 | 0.962 |
| 20261105T095243Z-all500-fp16 | main e08375cff7 | on | 404/500 (80.8%) | 80 | 68 | 0.962 |
| 20261108T041330Z-all500-fp16-chunked | bringup-prefill-opt-main 4d30e41fab | on | 404/500 (80.8%) | 80 | 68 | 0.962 |
| 20261107T144320Z-all500-fp16 | bringup-prefill-opt-main 4d30e41fab | off | 398/500 (79.6%) | 86 | 72 | 0.961 |
| 20261108T041325Z-all500-fp16 | bringup-prefill-opt-main 4d30e41fab | off | 398/500 (79.6%) | 86 | 72 | 0.961 |

- chunked prefill on 실행 세 개는 서로 다른 commit 세 개에서 나왔는데도 출력 token이 500/500 모두 같다. off 실행 두 개도 서로 500/500 같다. FP16 출력은 commit이 아니라 이 설정에만 달라진다
- `2b2461805f`와 `4d30e41fab` 사이에는 `vllm/` 아래 4개 파일이 바뀌었지만 전부 DiffKV 모듈 이름 변경이다 (`diffkv_v2` → `diffkv`, `DIFFKV_V2` → `DIFFKV`)
- 두 설정 사이에서는 500문제 중 198개만 출력이 같고 302개가 갈린다. 처음 달라지는 token 위치는 최소 6, 중앙값 920, 최대 5361이고, 0번(첫 생성 token)인 경우는 0개다. prompt는 최대 776 token이고 max_num_batched_tokens가 9216이라 prompt가 쪼개지는 일은 없으므로 prefill 출력은 양쪽이 같고, 차이는 decode 중에 생긴다
- vLLM V1에서 `enable_chunked_prefill=False`는 새 요청의 prompt가 그 step에 남은 token 예산을 넘으면 요청을 쪼개는 대신 그 step의 요청 투입을 멈춘다(vllm/v1/core/sched/scheduler.py:1180). 그래서 step마다 함께 묶이는 요청 집합과 preemption이 달라진다(preemption 2,776 대 2,630)
- 잘리지 않은 문항만의 정답률은 0.961과 0.962로 같다. 점수 차이 6개는 전부 잘린 문항 수 차이(86 대 80)에서 온다
- 그래서 11/05 수치와 11/07 수치를 나란히 놓고 읽으면 안 된다. 같은 설정 안에서 비교하면 chunked prefill on은 FP16 404, DiffKV 404, FP8 397이고, off는 FP16 398, DiffKV 389, FP8 401이다

### 실행 비교 방법

채점까지 끝난 실행 디렉터리 두 개 이상을 넣으면 문항별로 비교한다. 위 표의 수치가 이 명령의 출력이다.

```bash
bash diffkv/run_docker.sh python diffkv/evaluations/correctness/compare_runs.py \
  diffkv/evaluations/logs/correctness/<실행 A> diffkv/evaluations/logs/correctness/<실행 B>
```

### 결과 디렉터리
- /home/130bb56/DiffKV/diffkv/evaluations/logs/correctness/20261105T095243Z-all500-fp16
- /home/130bb56/DiffKV/diffkv/evaluations/logs/correctness/20261105T095243Z-all500-diffkv
- /home/130bb56/DiffKV/diffkv/evaluations/logs/correctness/20261105T103508Z-all500-fp8
- /home/130bb56/DiffKV/diffkv/evaluations/logs/correctness/20261106T071051Z-diamond197-fp8
- /home/130bb56/DiffKV/diffkv/evaluations/logs/correctness/20261106T071051Z-diamond197-diffkv
- /home/130bb56/DiffKV/diffkv/evaluations/logs/correctness/20261106T102814Z-diamond197-osl16384-fp8
- /home/130bb56/DiffKV/diffkv/evaluations/logs/correctness/20261106T102814Z-diamond197-osl16384-diffkv
- /home/130bb56/DiffKV/diffkv/evaluations/logs/correctness/20261102T091120Z-all500-fp16
- /home/130bb56/DiffKV/diffkv/evaluations/logs/correctness/20261107T144320Z-all500-fp16
- /home/130bb56/DiffKV/diffkv/evaluations/logs/correctness/20261107T144320Z-all500-diffkv
- /home/130bb56/DiffKV/diffkv/evaluations/logs/correctness/20261107T161730Z-all500-fp8
- /home/130bb56/DiffKV/diffkv/evaluations/logs/correctness/20261108T041325Z-all500-fp16
- /home/130bb56/DiffKV/diffkv/evaluations/logs/correctness/20261108T041330Z-all500-fp16-chunked