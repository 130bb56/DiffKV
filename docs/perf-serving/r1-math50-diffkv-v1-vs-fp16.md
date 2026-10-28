# R1-Distill-Llama-8B, MATH-500 50문제: DiffKV-v1과 FP16 vLLM 비교

- 성능 측정: 2026-10-24 KST, GPU0에서 한 번에 하나씩 실행
- 최초 실행과 정확도 측정: 2026-10-23 KST ([report.md](../../evaluation-logs/r1-math50-20261023/report.md))
- 확인할 내용: DeepSeek-R1-Distill-Llama-8B에서 DiffKV-v1이 FP16 vLLM보다 유의미하게 빠른가, 정확도를 유지하는가

## 용어
- **FP16**: KV 압축이 없는 vLLM 0.2.6 baseline입니다.
- **DiffKV-v1**: 이 저장소의 DiffKV에 `no-sync-bulk-lengths` 설정을 적용한 것입니다. attention forward와 engine step에 있던 불필요한 `torch.cuda.synchronize()` 호출 12개를 없앴습니다. 또 decode step마다 layer별 문맥 길이를 읽던 `.item()` 호출 32번을 `.cpu().tolist()` 한 번으로 바꿨습니다. 둘 다 Python 수준의 변경이고, CUDA 커널은 바꾸지 않습니다([optimization-v1.md](../../optimization-v1.md)).
- **natural 실행**: EOS가 나오면 멈추고, 최대 8192 token까지 생성하는 실행입니다. 정확도와 출력 길이는 이 실행에서 잽니다.
- **OSL 고정 실행**: DiffKV-v1을 `ignore_eos=True`로 돌리고, 요청별 `max_tokens`를 FP16 natural 실행에서 그 요청이 생성한 token 수로 둔 실행입니다(`run_r1_math.py --mode length_replay`). 50개 요청의 출력 길이가 FP16과 정확히 같으므로, 같은 양을 생성할 때의 속도를 비교할 수 있습니다. 출력이 EOS 뒤로 이어지거나 답 전에 잘리므로 채점하지 않습니다.

## 요약

| 지표 | FP16 (natural 실행) | DiffKV-v1 | FP16 대비 |
|---|---:|---:|---:|
| 평균 E2E, DiffKV-v1 OSL 고정 실행 (초) | 240.85 | 161.27 | 1.49배 빠름 |
| 평균 E2E, DiffKV-v1 natural 실행 (초) | 240.85 | 179.66 | 1.34배 빠름 |
| 처리량, natural 실행 (token/s) | 291.7 | 464.3 | 1.59배 |
| 정확도, 채점 오류 수정 후 | 42/50 (84%) | 40/50 (80%) | -4%p |

시간 지표의 배율은 FP16 값을 DiffKV-v1 값으로 나눈 것이고, 처리량은 DiffKV-v1 값을 FP16 값으로 나눈 것입니다.

- **속도**: DiffKV-v1이 빠릅니다. FP16은 KV cache가 가득 차서 recompute preemption이 67번 일어났고, DiffKV-v1은 0번입니다.
- **사전에 정한 성능 기준**: OSL 고정 실행의 평균 E2E가 1.4배 이상이어야 하고, 1.49배로 통과했습니다. natural 실행 기준으로는 1.34배이며, DiffKV-v1이 5.4% 더 많은 token을 생성한 영향입니다.
- **사전에 정한 정확도 기준**: 정확도 하락 5%p 이내, 출력 길이 비율 0.9~1.1입니다. -4%p와 1.054로 둘 다 통과했습니다.
- **해석의 한계**: 50문제, sampling seed 1개 결과입니다. 두 쪽 정확도가 같다고 말할 근거는 부족합니다.

## 실험 설정

| 항목 | 값 |
|---|---|
| 모델 | deepseek-ai/DeepSeek-R1-Distill-Llama-8B, revision 6a6f4aa4197940add57724a7707d069478df56b1 |
| 데이터 | HuggingFaceH4/MATH-500 test, revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be. `random.Random(42).sample(range(500), 50)`로 선택했고, 길이로 거르거나 다시 뽑지 않았습니다. |
| prompt | 문제 뒤에 "Please reason step by step, and put your final answer within \boxed{}."를 붙이고 chat template을 적용한 뒤 token ID로 고정했습니다. 길이는 32~776 token, 평균 107.7 token입니다. |
| 요청 도착 | 50개 요청을 첫 engine step 전에 같은 도착 시각으로 등록했습니다. |
| sampling | temperature 0.6, top_p 0.95, seed 42, 최대 출력 8192 token |
| engine 설정 (양쪽 같음) | float16, max_model_len 9216, max_num_batched_tokens 9216, max_num_seqs 64, max_paddings 4096, gpu_memory_utilization 0.9, swap 0, eager, tensor parallel 1 |
| warmup | 요청 50개를 32 token씩 생성했고, 측정에서 뺀 뒤 RNG를 다시 초기화했습니다. |
| FP16 | vLLM 0.2.6 (commit 1db83e3)에 llama3 RoPE 패치를 적용했습니다. KV cache는 2431 block × 16 = 38,896 token입니다. |
| DiffKV-v1 | 이 저장소 commit bdefda1에 `no-sync-bulk-lengths`를 적용했습니다. high precision K8V4, low precision K4V2, quant group [1,1,1,1], `compress_configs` [0.02, 1.0], `kv_buffer_size` 64입니다. |
| 소프트웨어 | PyTorch 2.6.0+cu124, Transformers 4.51.3, Triton 3.2.0, xformers 0.0.29.post3 |
| GPU | RTX 3090 24GB, GPU0 (UUID f2139c13-2e60-9ae2-f007-9965bca6b295), 자동 clock |

- max_model_len과 max_num_batched_tokens는 처음에 8704로 계획했습니다. 가장 긴 prompt(776 token)와 출력 상한 8192 token을 함께 담도록 양쪽 모두 9216으로 올렸습니다.
- **llama3 RoPE 패치**: vLLM 0.2.6은 Llama-3.1 계열이 쓰는 `rope_type: llama3`를 지원하지 않습니다. 그래서 DiffKV에 있는 `Llama3RotaryEmbedding`을 baseline으로 옮겼습니다([패치](../../benchmarks/patches/vllm-0.2.6-llama3-rope.patch)). 바뀌는 것은 config의 최대 길이 계산과, 초기화 때 만드는 RoPE 주파수 표입니다. 실행 중에 쓰는 CUDA 커널은 그대로입니다.

## 측정 방법
지표는 [run_r1_math.py](../../benchmarks/run_r1_math.py)의 정의를 따릅니다.

| 지표 | 정의 |
|---|---|
| TTFT | 공통 도착 시각부터 첫 출력 token까지의 시간입니다. scheduler 대기 시간이 들어 있습니다. |
| TPOT | (마지막 token 시각 - 첫 token 시각) / (출력 token 수 - 1)입니다. preemption으로 멈춘 시간이 들어 있습니다. |
| E2E | 공통 도착 시각부터 요청 완료까지의 시간입니다. 모델 로딩과 warmup은 빠집니다. |
| 전체 완료 시간 | 첫 engine step 시작부터 마지막 요청 완료까지의 시간입니다. |
| 처리량 | 생성 token 합계를 전체 완료 시간으로 나눈 값입니다. |
| 평균 decode batch | decode step마다의 batch 크기를 평균한 값입니다. |

실행 순서는 다음과 같습니다. 모두 GPU0에서 앞 실행이 끝난 뒤 다음 실행을 시작했습니다.

1. FP16 natural 실행: 14:56~15:07
2. DiffKV-v1 OSL 고정 실행: 15:07~15:14
3. DiffKV-v1 natural 실행: 15:14~15:21

## 성능 결과

| | FP16 natural | DiffKV-v1 OSL 고정 | DiffKV-v1 natural |
|---|---:|---:|---:|
| 평균 E2E (초) | 240.85 | 161.27 | 179.66 |
| E2E p50 (초) | 206.51 | 142.88 | 137.54 |
| 전체 완료 시간 (초) | 596.65 | 367.15 | 394.97 |
| 생성 token 합계 | 174,055 | 174,055 | 183,382 |
| 처리량 (token/s) | 291.7 | 474.1 | 464.3 |
| 평균 TTFT (초) | 2.06 | 0.85 | 0.86 |
| 평균 TPOT (ms) | 80.9 | 46.1 | 48.8 |
| recompute preemption | 67 | 0 | 0 |
| 평균 decode batch | 11.13 | 21.24 | 22.38 |
| 최대 KV cache 사용률 | 100% | 69.7% | 68.9% |
| prefill step 시간 합계 (초) | 30.13 | 1.23 | 1.25 |
| decode step 시간 합계 (초) | 564.19 | 364.57 | 392.31 |

| FP16 대비 배율 | OSL 고정 실행 | natural 실행 |
|---|---:|---:|
| 평균 E2E | 1.49배 | 1.34배 |
| 전체 완료 시간 | 1.63배 | 1.51배 |
| 처리량 | 1.63배 | 1.59배 |

- FP16의 KV cache는 38,896 token이라 50개 요청의 문맥을 동시에 담지 못합니다. KV cache 사용률이 100%에 닿아 recompute preemption이 67번 일어났고, 평균 decode batch는 11.1입니다.
- DiffKV-v1은 KV cache 사용률이 최대 69.7%라 preemption이 없습니다. 평균 decode batch는 21~22로 FP16의 약 2배입니다.
- FP16의 prefill step 시간 30.13초 중 26.97초는 preemption된 요청을 다시 prefill한 시간입니다(prefill step 40개 중 37개). 처음 50개 요청의 prefill 3 step은 3.17초입니다.
- TTFT 차이에는 두 엔진의 prefill 경로 차이도 들어 있습니다. 같은 첫 prefill batch(요청 28개)가 FP16에서 1.37초, DiffKV-v1에서 0.62초 걸렸습니다. 이 차이의 원인은 분석하지 않았습니다.
- natural 실행의 배율이 OSL 고정 실행보다 낮은 것은 DiffKV-v1이 token을 5.4% 더 생성했기 때문입니다(183,382 대 174,055).

### 2026-10-23 측정과의 비교
10/23에는 FP16 natural 실행(GPU0)과 DiffKV-v1 natural 실행(GPU1)을 동시에 돌렸습니다. FP16 측정 613초 중 뒤쪽 416초가 DiffKV-v1 실행과 겹쳤습니다. 10/24에는 이 겹침을 없애려고 GPU0에서 하나씩 다시 실행했습니다.

| 평균 E2E (초) | 10/23 | 10/24 | 변화 |
|---|---:|---:|---:|
| FP16 natural (GPU0) | 249.30 | 240.85 | -3.4% |
| DiffKV-v1 OSL 고정 (GPU0) | 167.25 | 161.27 | -3.6% |
| DiffKV-v1 natural | 186.01 (GPU1) | 179.66 (GPU0) | -3.4% |
| OSL 고정 실행 배율 | 1.491배 | 1.493배 | |

- 양쪽이 비슷하게 빨라져서 배율은 변하지 않았습니다. 두 날의 차이 원인은 확인하지 않았고, GPU clock과 온도는 기록하지 않았습니다.
- 세 실행 모두 10/23 실행과 출력 token이 50개 요청 모두 같습니다. 그래서 10/23 실행으로 잰 정확도가 10/24 실행에도 그대로 해당합니다.

## 정확도 결과 (natural 실행)

| | FP16 | DiffKV-v1 |
|---|---:|---:|
| 채점기 결과 | 40/50 (80%) | 39/50 (78%) |
| 채점 오류 2건 수정 후 | 42/50 (84%) | 40/50 (80%) |
| 평균 출력 token | 3481.1 | 3667.6 |
| 8192 token 상한 도달 | 6 | 8 |
| 답 추출 실패 | 6 | 7 |

- **채점기**: 저장소의 `vllm/dataset/math_parser.py`에서 `extract_answer`로 마지막 `</think>` 뒤 텍스트의 답을 추출합니다(last-number fallback 없음). 그 답을 `vllm/dataset/math_grader.py`의 `math_equal`로 정답과 비교합니다. 답 추출 실패는 채점기가 답을 찾지 못한 요청 수입니다.
- **채점 오류 2건**: 문제별 출력을 직접 읽어 확인했습니다.
  - math500-379 (FP16): 출력은 `\boxed{12^{\mathrm{th}}}`인데 `12^{\mathrm{}}`로 추출되어 오답 처리됐습니다. 정답은 12입니다.
  - math500-456 (양쪽): 정답 "-2, 1"을 `\boxed{1}`과 `\boxed{-2}`로 나눠 썼는데, 채점기가 마지막 box만 읽어서 오답 처리했습니다.
- **출력 길이 비율**: DiffKV-v1 평균을 FP16 평균으로 나누면 1.054입니다.
- **결과가 갈린 문제** (수정 후): FP16만 맞힌 문제가 3개(math500-279, 412, 194), DiffKV-v1만 맞힌 문제가 1개(math500-301)입니다. 412와 194는 DiffKV-v1이 8192 token 상한에 걸린 문제입니다. 39문제는 둘 다 맞혔고, 7문제는 둘 다 틀렸습니다.
- **통계**: 결과가 갈린 문제가 4개(3 대 1)뿐이라 McNemar exact test의 p는 0.63입니다. 두 쪽 정확도가 다르다고 말할 수 없고, 같다고 말할 근거도 부족합니다.
- **출력 길이 차이**: 요청별 출력 길이 비율(DiffKV-v1 / FP16)은 0.29배(math500-057)부터 4.60배(math500-214, 8192 상한)까지 넓게 퍼져 있습니다. 평균 5.4% 차이가 DiffKV-v1 때문인지 sampling 분산 때문인지는 50문제로 구분할 수 없습니다.
- 논문에 보고된 이 모델의 FP16 MATH 정확도는 88.8%입니다. 이번 실험에서는 출력 상한 8192 token 때문에 FP16에서도 6문제가 답을 내기 전에 잘렸습니다.

## 검증한 항목
- **FP16 RoPE 패치**
  - [test_r1_rope.py](../../tests/test_r1_rope.py) 6개 테스트가 통과합니다. inv_freq와 cos/sin 표를 Transformers 4.51.3과 비교하고, 기존 scaling 경로와 최대 길이 계산이 그대로인지 확인합니다.
  - GPU에서 첫 prompt 3개를 128 token씩 greedy로 생성했을 때, 출력 token 384개가 HF Transformers와 모두 같습니다([parity-check.json](../../evaluation-logs/r1-math50-20261023/parity-check.json)).
- **weight**: 두 safetensors 파일의 SHA256이 Hugging Face LFS 값과 같습니다(run-status.json의 `verified_model_files`).
- **결과 audit**: [analyze_r1_math.py](../../benchmarks/analyze_r1_math.py)의 검사를 모두 통과했습니다. 검사 항목은 workload hash, 요청 ID, engine 설정, sampling 설정, latency 값의 일관성, EOS 정책, 같은 GPU UUID, 요청별 출력 길이 일치입니다([analysis.json](../../evaluation-logs/r1-math50-20261023/solo-rerun-20261024/analysis.json)).
- **결정성**: 10/24 재실행 3개는 10/23 실행과 출력 token이 50개 요청 모두 같습니다.
- **prompt score 버그와의 관계**: DiffKV의 prefill score 커널([triton_flash_attention.py:131](../../vllm/model_executor/layers/triton_flash_attention.py))은 prompt 길이가 64의 배수이고 64보다 길 때 race condition이 있어서, 어떤 token을 pruning하고 low precision으로 저장할지가 틀린 score로 정해집니다. 이 workload에는 그런 길이의 prompt가 없습니다. DiffKV-v1 실행에서는 preemption이 없었으므로 recompute로 prefill을 다시 한 경우도 없습니다. 따라서 이 결과는 이 버그의 영향을 받지 않았습니다.

## 제한 사항
- 50문제, sampling seed 1개, 경로별로 1회씩 측정했습니다. 실행이 결정적이라 같은 조건으로 다시 돌려도 분산은 알 수 없습니다.
- vLLM 0.2.6은 sampling 난수를 엔진 전체에서 하나로 씁니다. batch 구성이 다르면 seed가 같아도 요청별로 받는 난수가 달라지므로, FP16과 DiffKV-v1에서 같은 문제가 같은 난수를 받지 않습니다.
- 50개 요청을 한 번에 넣은 burst 실험입니다. 요청이 시간 간격을 두고 도착하는 online serving 조건은 재지 않았습니다.
- 비교 대상은 DiffKV가 fork한 vLLM 0.2.6(eager 모드, CUDA graph 없음)입니다. 최신 vLLM과 비교한 결과는 아닙니다.
- GPU는 자동 clock 상태였고, clock과 온도는 기록하지 않았습니다.
- DiffKV-v1 결과는 확장 파일 `vllm/_C.cpython-311-x86_64-linux-gnu.so`(sha256 `36e8386cf4628a77957e225f7204bfa769816d1eb9c0b2df15a5412be9267d01`)로 잰 값입니다. 이 파일은 2026-10-24 15:57 KST에 다시 빌드되어, 지금 저장소에 있는 파일과 다릅니다. FP16 확장 파일의 sha256은 `83bf34e2772d2946e1d0dd0b34b47f018f866b6d9d5887d39e2821adb2429878`입니다.

## 파일
- 10/24 재측정: [solo-rerun-20261024/](../../evaluation-logs/r1-math50-20261023/solo-rerun-20261024)
  - `fp16-natural.json`, `diffkv-replay.json`(OSL 고정 실행), `diffkv-natural.json`, `analysis.json`
  - 각 실행의 `.log`, step별 기록 `.steps.jsonl`, 실행에 쓴 source patch와 DiffKV-v1 변경 소스
- 10/23 실행: [r1-math50-20261023/](../../evaluation-logs/r1-math50-20261023)
  - `workload.json`(고정 입력), `provenance.json`(모델과 데이터 출처), `parity-check.json`(HF greedy 비교), `report.md`(최초 보고서)
- 코드: [prepare_r1_math.py](../../benchmarks/prepare_r1_math.py), [run_r1_math.py](../../benchmarks/run_r1_math.py), [analyze_r1_math.py](../../benchmarks/analyze_r1_math.py), [validate_r1_greedy.py](../../benchmarks/validate_r1_greedy.py)

## 재현 명령
저장소 루트에서 실행합니다. 세 실행은 같은 GPU에서 차례로 돌려야 합니다. 실행 전에 `vllm/_C.cpython-311-x86_64-linux-gnu.so`의 sha256이 위 값과 같은지 확인해야 같은 커널로 잴 수 있습니다.

```bash
P=.cache/baselines/fp16-env/bin/python
E=evaluation-logs/r1-math50-20261023
O=$E/solo-rerun-20261024
M=.cache/models/DeepSeek-R1-Distill-Llama-8B

$P -u benchmarks/run_r1_math.py --backend fp16 --gpu 0 \
  --workload $E/workload.json --model $M --output $O/fp16-natural.json
$P -u benchmarks/run_r1_math.py --backend diffkv --gpu 0 --mode length_replay \
  --lengths-from $O/fp16-natural.json \
  --workload $E/workload.json --model $M --output $O/diffkv-replay.json
$P -u benchmarks/run_r1_math.py --backend diffkv --gpu 0 \
  --workload $E/workload.json --model $M --output $O/diffkv-natural.json
.venv/bin/python -u benchmarks/analyze_r1_math.py --workload $E/workload.json \
  --fp16 $O/fp16-natural.json --diffkv $O/diffkv-natural.json \
  --replay $O/diffkv-replay.json --output $O/analysis.json
```
