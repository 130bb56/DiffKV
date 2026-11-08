## Correctness 결과

공통 조건: DeepSeek-R1-Distill-Llama-8B, RTX 3090, clock 고정 없음, greedy, 문항당 1개 생성

| 데이터셋 | backend (GPU) | 코드 | 출력 상한 (OSL) | max_model_len | 정답 | 출력 상한에서 잘린 문항 | 답 없음 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MATH-500 500문제 (11/05) | fp16 (GPU1) | main e08375cff7 | 8192 | 9216 | 404/500 (80.8%) | 80 | 68 |
| MATH-500 500문제 (11/05) | diffkv (GPU0) | main e08375cff7 | 8192 | 9216 | 404/500 (80.8%) | 79 | 71 |
| MATH-500 500문제 (11/05) | fp8 (GPU0) | main e08375cff7 | 8192 | 9216 | 397/500 (79.4%) | 88 | 83 |
| GPQA-diamond 197문항 (11/06) | fp8 (GPU0) | main e08375cff7 + 커밋 전 GPQA 변경 | 8192 | 9216 | 45/197 (22.8%) | 130 | 130 |
| GPQA-diamond 197문항 (11/06) | diffkv (GPU1) | main e08375cff7 + 커밋 전 GPQA 변경 | 8192 | 9216 | 42/197 (21.3%) | 132 | 133 |
| GPQA-diamond 197문항 (진행 중, 약 61분 시점) | fp8 (GPU0) | main e08375cff7 + 커밋 전 GPQA 변경 | 16384 | 17,408 | 중간 16/75 (21.3%) | 44/75 | 44/75 |
| GPQA-diamond 197문항 (진행 중, 약 61분 시점) | diffkv (GPU1) | main e08375cff7 + 커밋 전 GPQA 변경 | 16384 | 17,408 | 중간 28/96 (29.2%) | 49/96 | 49/96 |

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

### 결과 디렉터리
- /home/130bb56/DiffKV/diffkv/evaluations/logs/correctness/20261105T095243Z-all500-fp16
- /home/130bb56/DiffKV/diffkv/evaluations/logs/correctness/20261105T095243Z-all500-diffkv
- /home/130bb56/DiffKV/diffkv/evaluations/logs/correctness/20261105T103508Z-all500-fp8
- /home/130bb56/DiffKV/diffkv/evaluations/logs/correctness/20261106T071051Z-diamond197-fp8
- /home/130bb56/DiffKV/diffkv/evaluations/logs/correctness/20261106T071051Z-diamond197-diffkv
- /home/130bb56/DiffKV/diffkv/evaluations/logs/correctness/20261106T102814Z-diamond197-osl16384-fp8
- /home/130bb56/DiffKV/diffkv/evaluations/logs/correctness/20261106T102814Z-diamond197-osl16384-diffkv