# Evaluation

RTX 3090 availability is assessed for one 24 GB GPU with FP16 model weights, using the repository's supported architectures and weight memory requirements.

| Model-Parameters | RTX 3090 | Evaluation |
|---|:---:|:---:|
| Llama-2-7B | O | X |
| **Llama-3-8B, Base / Instruct** | O | **O** |
| Llama-3-70B | X | X |
| **Mistral-7B** | O | **O** |
| Mixtral-8x7B | X | X |
| Qwen2.5-7B | O | X |
| Qwen2.5-32B | X | X |
| QwQ-32B | X | X |
| Qwen3-8B | O | X |
| Qwen3-32B | X | X |
| Qwen3-30B-A3B | X | X |

Evaluation indicates whether FP16 / DiffKV comparison results are recorded in this document: O = evaluated, X = not evaluated.

Models in the 0.5B range were excluded because they are too small. FP16 / DiffKV comparisons in this document were conducted on Llama-3-8B, Base / Instruct, and Mistral-7B. Mistral-7B was measured in one offline synthetic case, (ISL 512, OSL 512, B 8).

Llama3-8B Instruct and Base have effectively the same inference performance for these fixed synthetic workloads. Only correctness tests will be rerun with Instruct.

DiffKV runs Qwen3-8B using its `Qwen3ForCausalLM` implementation, but the original FP16 vLLM baseline does not support this architecture, so Qwen3-8B is excluded from the FP16 / DiffKV comparisons.

성능 수치는 Latency, Time은 낮은 값, Throughput, Accuracy, Correct는 높은 값을 **굵게** 표시했습니다.

## Llama3-8B Base, Offline Synthetic E2E, ISL/OSL [512, 1024, 2048], B [8, 16, 32]

- Started: 2026-10-10T17:11:12+09:00
- Model: /home/130bb56/DiffKV/.cache/models/Meta-Llama-3-8B
- GPU: 0
- Workload: **ISL [512, 1024, 2048], OSL [512, 1024, 2048], B [8, 16, 32]**
- Measured batches per case: 1
- Warmup output length: 16
- Logs: /home/130bb56/DiffKV/logs/performance/fixed-grid-20261010-171112-437414

Command:

```bash
/home/130bb56/DiffKV/.cache/baselines/fp16-env/bin/python -u \
  /home/130bb56/DiffKV/benchmarks/run_offline_e2e.py
```

Each cell is **FP16 / DiffKV**. TTFT and TPOT are request means, averaged across measured batches.

Prefill Throughput = input tokens / summed prefill engine.step time. Decode Throughput = B * (OSL - 1) / summed decode engine.step time; the first token is generated during prefill. Phase times include scheduling, model execution, sampling and detokenization. E2E Latency measures the complete generation call, including request registration and cleanup, excluding model loading and warmup. OOM means insufficient memory for the requested fixed workload.

| Case (ISL, OSL, B) | TTFT (ms) | TPOT (ms/token) | Prefill Throughput (tok/s) | Decode Throughput (tok/s) | E2E Latency (s) |
|---|---:|---:|---:|---:|---:|
| **(2048, 2048, 32)** | OOM / 8146.223 | OOM / 76.089 | OOM / 4517.730 | OOM / 438.696 | OOM / 163.902 |
| **(2048, 1024, 32)** | OOM / 8358.083 | OOM / 73.325 | OOM / 4404.322 | OOM / 478.227 | OOM / 83.370 |
| **(2048, 512, 32)** | OOM / 8365.874 | OOM / 76.704 | OOM / 4399.837 | OOM / 500.847 | OOM / 47.562 |
| **(1024, 2048, 32)** | OOM / 4489.622 | OOM / 62.572 | OOM / 4559.283 | OOM / 522.735 | OOM / 132.575 |
| **(1024, 1024, 32)** | OOM / 4498.562 | OOM / 56.591 | OOM / 4551.010 | OOM / 593.487 | OOM / 62.391 |
| **(1024, 512, 32)** | OOM / 4509.372 | OOM / 55.706 | OOM / 4540.138 | OOM / 635.231 | OOM / 32.975 |
| **(512, 2048, 32)** | OOM / 2666.025 | OOM / 55.143 | OOM / 4611.399 | OOM / 585.248 | OOM / 115.545 |
| **(512, 1024, 32)** | OOM / 2655.829 | OOM / 48.032 | OOM / 4627.475 | OOM / 678.879 | OOM / 51.792 |
| **(512, 512, 32)** | **2619.500** / 2656.260 | 46.405 / **45.379** | **4692.221** / 4626.286 | 716.401 / **733.640** | 26.334 / **25.845** |
| **(2048, 2048, 16)** | OOM / 4622.271 | OOM / 53.668 | OOM / 4429.192 | OOM / 305.974 | OOM / 114.481 |
| **(2048, 1024, 16)** | OOM / 4620.869 | OOM / 51.563 | OOM / 4428.889 | OOM / 327.675 | OOM / 57.370 |
| **(2048, 512, 16)** | OOM / 4620.977 | OOM / 52.407 | OOM / 4430.555 | OOM / 340.727 | OOM / 31.401 |
| **(1024, 2048, 16)** | OOM / 2684.174 | OOM / 46.276 | OOM / 4579.988 | OOM / 349.185 | OOM / 97.412 |
| **(1024, 1024, 16)** | **2644.795** / 2685.816 | 44.659 / **42.836** | **4645.789** / 4576.175 | 365.485 / **381.470** | 48.332 / **46.507** |
| **(1024, 512, 16)** | **2649.852** / 2696.930 | 41.729 / **41.653** | **4638.382** / 4557.424 | 400.183 / **401.228** | **23.974** / 23.982 |
| **(512, 2048, 16)** | OOM / 1761.886 | OOM / 42.192 | OOM / 4649.662 | OOM / 379.386 | OOM / 88.130 |
| **(512, 1024, 16)** | **1744.422** / 1768.775 | **37.273** / 38.400 | **4697.466** / 4631.553 | **429.481** / 416.860 | **39.875** / 41.052 |
| **(512, 512, 16)** | **1741.977** / 1767.772 | **32.750** / 36.470 | **4704.056** / 4634.183 | **488.829** / 438.930 | **18.478** / 20.404 |
| **(2048, 2048, 8)** | **2688.685** / 2753.758 | **41.442** / 41.597 | **4570.443** / 4463.014 | **195.167** / 194.482 | **87.521** / 87.902 |
| **(2048, 1024, 8)** | **2685.524** / 2757.479 | 41.243 / **39.543** | **4575.448** / 4457.873 | 198.243 / **207.080** | 44.879 / **43.210** |
| **(2048, 512, 8)** | **2692.173** / 2761.592 | **37.396** / 39.454 | **4564.204** / 4450.458 | **224.558** / 212.533 | **21.803** / 22.923 |
| **(1024, 2048, 8)** | **1761.621** / 1786.330 | **37.193** / 37.640 | **4651.625** / 4586.002 | **215.180** / 212.617 | **77.896** / 78.835 |
| **(1024, 1024, 8)** | **1750.640** / 1782.885 | **33.679** / 34.959 | **4680.802** / 4594.866 | **237.635** / 228.929 | **36.205** / 37.546 |
| **(1024, 512, 8)** | **1755.058** / 1786.481 | **31.195** / 33.719 | **4669.036** / 4585.614 | **256.549** / 237.340 | **17.697** / 19.017 |
| **(512, 2048, 8)** | **893.911** / 908.161 | **34.283** / 35.600 | **4583.306** / 4510.327 | **233.432** / 224.798 | **71.073** / 73.782 |
| **(512, 1024, 8)** | **890.304** / 913.379 | **31.811** / 33.393 | **4601.880** / 4484.570 | **251.577** / 239.661 | **33.434** / 35.075 |
| **(512, 512, 8)** | **893.304** / 909.539 | **28.176** / 32.224 | **4586.436** / 4503.502 | **284.057** / 248.362 | **15.292** / 17.376 |

FP16은 27개 case 중 13개가 OOM이었고, DiffKV는 27개 모두 완료했습니다. 이는 DiffKV가 더 큰 workload를 VRAM에 담을 수 있었음을 보여줍니다. 두 경로 모두 완료한 case에서는 생성 길이와 동시 요청 수가 같으므로, TPOT와 E2E Latency를 직접 비교할 수 있습니다. 예를 들어 (512, 512, 8)은 DiffKV가 더 느렸고, (512, 512, 32)은 비슷했습니다.

## Llama3-8B Base, Offline GSM8K Correctness, 1319 Questions, FP16 / DiffKV

아래 명령은 측정 당시 기록입니다. 당시 FP16 실행 파일 `run_gsm8k_fp16.py`는 현재 [run_gsm8k_correctness.py](./benchmarks/run_gsm8k_correctness.py#L588)의 `--fp16-mode worker`로 통합했습니다. 과거 명령과 결과는 그대로 보존합니다.

FP16 command:

```bash
/home/130bb56/DiffKV/.cache/baselines/fp16-env/bin/python -u \
  /home/130bb56/DiffKV/benchmarks/run_gsm8k_fp16.py \
  --worker \
  --output-dir /home/130bb56/DiffKV/logs/correctness/gsm8k-20261009
```

DiffKV command:

```bash
/home/130bb56/DiffKV/.venv/bin/python -u \
  /home/130bb56/DiffKV/benchmarks/run_gsm8k_correctness.py \
  --model llama --gpu 0 \
  --output-dir /home/130bb56/DiffKV/logs/correctness/gsm8k-20261009 \
  --batch-size 8 --max-tokens 1024
```

Historical evaluation of **the full GSM8K test set, 1319 questions**.

- Model: Meta-Llama-3-8B, FP16 weights in both paths
- GPU: NVIDIA GeForce RTX 3090, device 0
- Prompt: 8-shot completion, input length 680 to 843 tokens
- Batch size: 8, active requests decrease as answers finish
- Maximum output length: 1024 tokens, generation stops at EOS or a configured stop string
- Sampling: greedy, temperature 0, seed 42
- FP16: original vLLM, K16V16
- DiffKV: K8V4/K4V2, thresholds [0.02, 1.0], KV buffer size 64
- Accuracy: exact numeric answer match after removing commas

| Path | Correct | Accuracy (%) | Generated Tokens | Throughput (tok/s) | Time (s) |
|---|---:|---:|---:|---:|---:|
| FP16 | **726 / 1319** | **55.04** | 149012 | **88.51** | **1683.58** |
| DiffKV | 698 / 1319 | 52.92 | 140422 | 82.04 | 1711.73 |

DiffKV는 생성량이 5.76% 적었는데도 Time은 1.67% 더 길었습니다. 생성량 감소와 시간 증가가 함께 반영돼 Throughput이 FP16보다 낮게 나왔습니다.

## Llama3-8B Base, Online MATH, 100 Questions, inf req/s, Fixed OSL 4096

Command:

```bash
.cache/baselines/fp16-env/bin/python -u benchmarks/run_online_e2e.py \
  --num-requests 100 --request-rate inf --gpu 0
```

Workload: **MATH, 100 questions, inf req/s, Fixed OSL 4096**. GPU 0, all requests submitted concurrently.

Same prompt token IDs and scheduled arrival times in both paths. Input lengths: min 11, mean 45.700, max 183. Selected subjects: algebra. The sample is drawn without replacement from the previously used 1000 MATH inputs.

Engine: FP16 weights, max_model_len=8192, max_num_seqs=100, max_num_batched_tokens=8192, max_paddings=4096, GPU memory utilization=0.9, swap=0, eager. FP16 KV: K16V16. DiffKV: K8V4/K4V2, thresholds [0.02, 1.0], KV buffer 64. Greedy, ignore_eos=True. One 16-token warmup request excluded, one measured run, all submitted requests awaited.

Online 표의 TTFT, TPOT, E2E Latency는 요청 100개의 평균입니다. E2E Latency는 각 요청을 보내고 마지막 token을 받을 때까지의 시간, Total Time은 요청 100개가 모두 끝날 때까지의 시간입니다. Request Throughput = 100 / Total Time입니다. Prefill Throughput과 Decode Throughput은 각 engine 구간에서 처리한 tokens를 해당 구간의 시간으로 나눈 값입니다.

| Metric | FP16 | DiffKV |
|---|---:|---:|
| TTFT (ms) | 1673.716 | **1011.103** |
| TPOT (ms/token) | **165.917** | 239.211 |
| Prefill Throughput (tok/s) | **3491.548** | 2688.575 |
| Decode Throughput (tok/s) | 373.815 | **546.767** |
| E2E Latency (s) | **681.105** | 980.581 |
| Request Throughput (req/s) | **0.082** | 0.079 |
| Generated Tokens | 409,600 | 409,600 |
| Total Time (s) | **1220.760** | 1260.151 |

두 경로 모두 409,600 tokens를 생성했습니다. DiffKV의 평균 동시 Decode 요청 수는 46.751개, FP16은 17.253개였습니다. DiffKV는 한 Decode step이 더 오래 걸렸지만, 한 번에 처리한 요청이 더 많아 Decode Throughput이 높았습니다.

KV cache가 거의 가득 차면서 요청의 cache를 버리고 context를 다시 계산하는 Recompute가 발생했습니다. 횟수는 FP16 269회, DiffKV 900회, 다시 계산한 tokens는 각각 334,536개, 1,334,792개였습니다. Prefill 시간은 FP16 97.122초, DiffKV 498.168초였습니다. Decode에서 절약한 347.439초보다 Prefill에서 추가된 401.046초가 커서 Total Time이 더 길었습니다.

TPOT에는 생성 도중의 대기와 Recompute 시간도 포함됩니다. Decode Throughput은 Decode step 시간만 사용하므로, Recompute에 쓴 Prefill 시간이 드러나지 않습니다. 그래서 Decode Throughput이 높아도 TPOT와 E2E Latency가 더 나쁠 수 있습니다.

- [fp16 server.txt](evaluation-logs/online-math-20261010-211506-938223/fp16/server.txt)
- [fp16 client.txt](evaluation-logs/online-math-20261010-211506-938223/fp16/client.txt)
- [fp16 responses.txt](evaluation-logs/online-math-20261010-211506-938223/fp16/responses.txt)
- [fp16 correctness.txt](evaluation-logs/online-math-20261010-211506-938223/fp16/correctness.txt)
- [diffkv server.txt](evaluation-logs/online-math-20261010-211506-938223/diffkv/server.txt)
- [diffkv client.txt](evaluation-logs/online-math-20261010-211506-938223/diffkv/client.txt)
- [diffkv responses.txt](evaluation-logs/online-math-20261010-211506-938223/diffkv/responses.txt)
- [diffkv correctness.txt](evaluation-logs/online-math-20261010-211506-938223/diffkv/correctness.txt)

## Llama3-8B Base, Online MATH, 100 Questions, inf req/s, Minerva 4-shot, Max OSL 4096

Command:

```bash
.cache/baselines/fp16-env/bin/python -u benchmarks/run_online_e2e.py \
  --num-requests 100 --request-rate inf --prompt-style minerva-4shot \
  --max-output-len 4096 --gpu 0
```

Workload: **MATH, 100 questions, inf req/s, Minerva 4-shot, Max OSL 4096**. GPU 0, all requests submitted concurrently.

Same prompt token IDs and scheduled arrival times in both paths. Input lengths: min 624, mean 658.960, max 796. Selected subjects: algebra. The sample is drawn without replacement from the previously used 1000 MATH inputs.

Engine: FP16 weights, max_model_len=8192, max_num_seqs=100, max_num_batched_tokens=8192, max_paddings=4096, GPU memory utilization=0.9, swap=0, eager. FP16 KV: K16V16. DiffKV: K8V4/K4V2, thresholds [0.02, 1.0], KV buffer 64. Greedy, ignore_eos=False. One 16-token warmup request excluded, one measured run, all submitted requests awaited.

| Metric | FP16 | DiffKV |
|---|---:|---:|
| TTFT (ms) | 11627.356 | **8221.597** |
| TPOT (ms/token) | **87.355** | 128.269 |
| Prefill Throughput (tok/s) | **4163.175** | 4117.213 |
| Decode Throughput (tok/s) | 329.221 | **395.075** |
| E2E Latency (s) | **56.920** | 68.548 |
| Request Throughput (req/s) | 0.360 | **0.406** |
| Generated Tokens | 81,665 | 89,945 |
| Total Time (s) | 277.564 | **246.605** |
| Accuracy (%) | **27.00 (27/100)** | 23.00 (23/100) |

DiffKV의 평균 동시 Decode 요청 수는 21.929개, FP16은 12.293개였습니다. 평균 Decode step 시간은 55.507 ms, 37.340 ms로 DiffKV가 더 길지만, 동시에 처리한 요청 수가 더 많아 Decode Throughput은 높았습니다.

FP16은 KV cache 부족으로 Recompute가 13회 발생했고, DiffKV는 0회였습니다. 전체 Decode step 수도 FP16 6,634회, DiffKV 4,097회로 달랐습니다. 더 많은 요청을 함께 진행한 DiffKV가 마지막 요청까지 먼저 끝냈지만, 개별 요청의 TPOT와 평균 E2E Latency는 더 높았습니다.

출력 길이도 달랐습니다. 생성량은 FP16 81,665 tokens, DiffKV 89,945 tokens였고, 최대 길이 4096에 도달한 요청은 각각 17개, 19개였습니다. E2E Latency는 요청별 시간의 평균이고, Total Time은 마지막 요청까지 모두 끝난 시간이므로 두 값의 대소 관계가 같을 필요는 없습니다.

- [fp16 server.txt](evaluation-logs/online-math-20261010-222958-866373/fp16/server.txt)
- [fp16 client.txt](evaluation-logs/online-math-20261010-222958-866373/fp16/client.txt)
- [fp16 responses.txt](evaluation-logs/online-math-20261010-222958-866373/fp16/responses.txt)
- [fp16 correctness.txt](evaluation-logs/online-math-20261010-222958-866373/fp16/correctness.txt)
- [diffkv server.txt](evaluation-logs/online-math-20261010-222958-866373/diffkv/server.txt)
- [diffkv client.txt](evaluation-logs/online-math-20261010-222958-866373/diffkv/client.txt)
- [diffkv responses.txt](evaluation-logs/online-math-20261010-222958-866373/diffkv/responses.txt)
- [diffkv correctness.txt](evaluation-logs/online-math-20261010-222958-866373/diffkv/correctness.txt)

## Llama3-8B Instruct, Online MATH, 100 Questions, inf req/s, Minerva 4-shot, Max OSL 4096

Command:

```bash
.cache/baselines/fp16-env/bin/python -u benchmarks/run_online_e2e.py \
  --model /home/130bb56/DiffKV/.cache/models/Meta-Llama-3-8B-Instruct \
  --num-requests 100 --request-rate inf --prompt-style minerva-4shot \
  --max-output-len 4096 --max-model-len 8192 --max-num-seqs 100 \
  --seed 42 --gpu 0 \
  --output-dir /home/130bb56/DiffKV/evaluation-logs/online-math-instruct-20261011-003428 \
  --report /home/130bb56/DiffKV/evaluation.md
```

Workload: **MATH, 100 questions, inf req/s, Minerva 4-shot, Max OSL 4096**. GPU 0, all requests submitted concurrently.

Same prompt token IDs and scheduled arrival times in both paths. Input lengths: min 624, mean 658.960, max 796. Selected subjects: algebra. The sample is drawn without replacement from the previously used 1000 MATH inputs.

Engine: FP16 weights, max_model_len=8192, max_num_seqs=100, max_num_batched_tokens=8192, max_paddings=4096, GPU memory utilization=0.9, swap=0, eager. FP16 KV: K16V16. DiffKV: K8V4/K4V2, thresholds [0.02, 1.0], KV buffer 64. Greedy, ignore_eos=False. One 16-token warmup request excluded, one measured run, all submitted requests awaited.

| Metric | FP16 | DiffKV |
|---|---:|---:|
| TTFT (ms) | 11726.764 | **8221.294** |
| TPOT (ms/token) | **92.738** | 133.791 |
| Prefill Throughput (tok/s) | **4148.354** | 4079.876 |
| Decode Throughput (tok/s) | 315.325 | **392.425** |
| E2E Latency (s) | **34.402** | 72.077 |
| Request Throughput (req/s) | **0.632** | 0.391 |
| Generated Tokens | 43,506 | 92,655 |
| Total Time (s) | **158.321** | 255.535 |
| Accuracy (%) | 41.00 (41/100) | **42.00 (42/100)** |

DiffKV의 생성량은 92,655 tokens로 FP16의 43,506 tokens보다 2.13배 많았습니다. 최대 길이 4096에 도달한 요청도 7개에서 20개로 늘었습니다. TPOT까지 92.738 ms에서 133.791 ms로 늘어 평균 E2E Latency가 34.402초에서 72.077초로 길어졌습니다.

평균 동시 Decode 요청 수는 10.155개에서 22.602개로 늘었습니다. DiffKV는 Decode 시간이 137.639초에서 235.854초로 길어졌지만, 생성량의 증가폭이 더 커서 Decode Throughput이 높게 나왔습니다. 이 값은 여러 요청의 합산 처리량입니다.

FP16은 Recompute가 5회, DiffKV는 0회였고, Prefill 시간은 16.710초, 16.151초로 비슷했습니다. 이 실험의 시간 증가는 주로 Decode에서 발생했습니다. 두 경로의 출력 길이가 달라 동일한 생성량의 속도를 비교한 결과는 아닙니다.

- [fp16 server.txt](evaluation-logs/online-math-instruct-20261011-003428/fp16/server.txt)
- [fp16 client.txt](evaluation-logs/online-math-instruct-20261011-003428/fp16/client.txt)
- [fp16 responses.txt](evaluation-logs/online-math-instruct-20261011-003428/fp16/responses.txt)
- [fp16 correctness.txt](evaluation-logs/online-math-instruct-20261011-003428/fp16/correctness.txt)
- [diffkv server.txt](evaluation-logs/online-math-instruct-20261011-003428/diffkv/server.txt)
- [diffkv client.txt](evaluation-logs/online-math-instruct-20261011-003428/diffkv/client.txt)
- [diffkv responses.txt](evaluation-logs/online-math-instruct-20261011-003428/diffkv/responses.txt)
- [diffkv correctness.txt](evaluation-logs/online-math-instruct-20261011-003428/diffkv/correctness.txt)

## Mistral-7B v0.1, Offline Synthetic E2E, ISL 512, OSL 512, B 8, FP16 / DiffKV

- Started: 2026-10-13T13:39:34+09:00
- Model: /home/130bb56/DiffKV/.cache/models/Mistral-7B-v0.1
- GPU: 0
- Workload: **ISL 512, OSL 512, B 8**
- Measured batches per case: 1
- Warmup output length: 16
- Logs: /home/130bb56/DiffKV/evaluation-logs/mistral-quick-20261013

Command:

```bash
/home/130bb56/DiffKV/.cache/baselines/fp16-env/bin/python -u /home/130bb56/DiffKV/benchmarks/run_offline_e2e.py \
  --inputs /home/130bb56/DiffKV/logs/performance/mistral-matched-inputs.json \
  --isls 512 \
  --osls 512 \
  --batches 8 \
  --backends fp16 diffkv \
  --gpu 0 \
  --repeats 1 \
  --warmup-tokens 16 \
  --output-dir /home/130bb56/DiffKV/evaluation-logs/mistral-quick-20261013 \
  --report /home/130bb56/DiffKV/evaluation.md
```

Each cell is **FP16 / DiffKV**. TTFT and TPOT are request means, averaged across measured batches.

Prefill Throughput = input tokens / summed prefill engine.step time. Decode Throughput = B * (OSL - 1) / summed decode engine.step time; the first token is generated during prefill. Phase times include scheduling, model execution, sampling and detokenization. E2E Latency measures the complete generation call, including request registration and cleanup, excluding model loading and warmup. OOM means insufficient memory for the requested fixed workload.

| Case (ISL, OSL, B) | TTFT (ms) | TPOT (ms/token) | Prefill Throughput (tok/s) | Decode Throughput (tok/s) | E2E Latency (s) |
|---|---:|---:|---:|---:|---:|
| **(512, 512, 8)** | **866.254** / 893.473 | **22.704** / 34.030 | **4729.734** / 4584.481 | **352.550** / 235.172 | **12.468** / 18.283 |

두 경로 모두 8개 요청에서 각각 512 tokens를 생성했고, 모든 Decode step의 동시 요청 수도 8개로 같았습니다. Recompute는 두 경로 모두 0회였습니다. Prefill 시간은 0.866초, 0.893초로 비슷했지만, Decode 시간은 11.596초, 17.383초로 DiffKV가 더 길었습니다. 따라서 이 case의 E2E Latency 차이는 Decode 구간에서 발생했습니다.


## Llama3-8B Base, Offline Synthetic, optimized DiffKV, Run 20261016-134916-315827

- Started: 2026-10-16T13:49:16+09:00
- Model: /home/130bb56/DiffKV/.cache/models/Meta-Llama-3-8B
- GPU: 0
- Physical GPUs: [0]
- GPU mapping: all 27 cases use GPU 0
- DiffKV variant: no-sync-bulk-lengths
- ISL: [512, 1024, 2048], OSL: [512, 1024, 2048], B: [8, 16, 32]
- Measured batches per case: 1
- Warmup output length: 16
- Logs: /home/130bb56/DiffKV/logs/performance/offline-optimized-grid-20261016-134916-315827
- Resume command: /home/130bb56/DiffKV/.cache/baselines/fp16-env/bin/python /home/130bb56/DiffKV/benchmarks/resume_offline_gpu0.py --run-dir /home/130bb56/DiffKV/logs/performance/offline-optimized-grid-20261016-134916-315827

위 resume 명령은 당시 실행 기록입니다. `resume_offline_gpu0.py`는 현재 삭제했으며, 현재 [run_offline_e2e.py](./benchmarks/run_offline_e2e.py#L742)의 `--case-indices`로 조건을 선택해 새 output directory에서 실행할 수 있습니다. 기존 결과 폴더 재개 명령과는 구분합니다.

Each cell is **FP16 / DiffKV**. All 27 cases were measured on physical GPU 0. The original 13 GPU 0 cases are retained; the 14 former GPU 1 cases have been replaced with GPU 0 remeasurements.

TTFT and TPOT are request means. Prefill Throughput = input tokens / summed prefill engine.step time. Decode Throughput = B * (OSL - 1) / summed decode engine.step time. E2E measures the complete generation call, excluding loading and warmup. Lower latency and higher throughput are **bold**. OOM means insufficient capacity for the fixed workload. Engine limits remain max_model_len=4096, max_num_seqs=32, max_num_batched_tokens=8192.

| Case (ISL, OSL, B) | TTFT (ms) | TPOT (ms/token) | Prefill Throughput (tok/s) | Decode Throughput (tok/s) | E2E Latency (s) |
|---|---:|---:|---:|---:|---:|
| (2048, 2048, 32) | OOM / 8282.602 | OOM / 63.193 | OOM / 4447.146 | OOM / 533.284 | OOM / 137.640 |
| (2048, 1024, 32) | OOM / 8343.259 | OOM / 63.423 | OOM / 4415.340 | OOM / 561.042 | OOM / 73.225 |
| (2048, 512, 32) | OOM / 8139.391 | OOM / 63.565 | OOM / 4523.455 | OOM / 626.131 | OOM / 40.622 |
| (1024, 2048, 32) | OOM / 4404.923 | OOM / 50.309 | OOM / 4646.063 | OOM / 653.315 | OOM / 107.389 |
| (1024, 1024, 32) | OOM / 4472.874 | OOM / 48.011 | OOM / 4577.691 | OOM / 705.653 | OOM / 53.588 |
| (1024, 512, 32) | OOM / 4443.084 | OOM / 46.442 | OOM / 4605.281 | OOM / 777.030 | OOM / 28.176 |
| (512, 2048, 32) | OOM / 2620.674 | OOM / 44.724 | OOM / 4689.964 | OOM / 722.918 | OOM / 94.172 |
| (512, 1024, 32) | OOM / 2631.999 | OOM / 40.176 | OOM / 4668.479 | OOM / 814.515 | OOM / 43.733 |
| (512, 512, 32) | **2615.300** / 2631.166 | 41.627 / **38.368** | **4699.714** / 4669.303 | 802.180 / **873.813** | 23.888 / **22.238** |
| (2048, 2048, 16) | OOM / 4575.167 | OOM / 43.881 | OOM / 4471.380 | OOM / 376.335 | OOM / 94.400 |
| (2048, 1024, 16) | OOM / 4581.346 | OOM / 43.699 | OOM / 4468.597 | OOM / 390.347 | OOM / 49.286 |
| (2048, 512, 16) | OOM / 4585.709 | OOM / 43.876 | OOM / 4464.867 | OOM / 415.917 | OOM / 27.007 |
| (1024, 2048, 16) | OOM / 2661.017 | OOM / 39.209 | OOM / 4617.344 | OOM / 412.834 | OOM / 82.922 |
| (1024, 1024, 16) | **2627.915** / 2658.425 | 40.136 / **35.680** | **4676.753** / 4621.905 | 407.523 / **459.831** | 43.689 / **39.160** |
| (1024, 512, 16) | **2632.449** / 2663.908 | 36.785 / **34.765** | **4669.750** / 4612.950 | 456.477 / **484.705** | 21.431 / **20.430** |
| (512, 2048, 16) | OOM / 1740.503 | OOM / 35.905 | OOM / 4706.784 | OOM / 445.857 | OOM / 75.238 |
| (512, 1024, 16) | **1744.270** / 1750.370 | 34.071 / **32.555** | **4697.893** / 4680.256 | 469.872 / **491.763** | 36.599 / **35.055** |
| (512, 512, 16) | **1731.394** / 1751.878 | **28.622** / 30.565 | **4732.836** / 4676.229 | **559.344** / 523.772 | **16.358** / 17.371 |
| (2048, 2048, 8) | **2691.344** / 2738.041 | 39.542 / **35.926** | **4565.182** / 4488.145 | 204.656 / **225.567** | 83.635 / **76.279** |
| (2048, 1024, 8) | **2679.427** / 2732.789 | 36.255 / **33.036** | **4587.371** / 4496.315 | 226.188 / **248.978** | 39.769 / **36.529** |
| (2048, 512, 8) | **2672.750** / 2741.562 | 33.042 / **32.869** | **4599.301** / 4480.495 | 255.705 / **257.528** | 19.558 / **19.538** |
| (1024, 2048, 8) | **1750.457** / 1770.361 | 34.500 / **32.168** | **4681.313** / 4627.371 | 231.971 / **248.800** | 72.373 / **67.619** |
| (1024, 1024, 8) | **1747.783** / 1771.066 | 30.585 / **29.410** | **4688.487** / 4625.525 | 261.673 / **272.139** | 33.037 / **31.858** |
| (1024, 512, 8) | **1750.960** / 1785.715 | **26.774** / 27.712 | **4679.955** / 4587.581 | **298.938** / 288.812 | **15.433** / 15.947 |
| (512, 2048, 8) | **890.936** / 894.174 | 31.853 / **30.421** | **4598.670** / 4580.886 | 251.256 / **263.094** | 66.095 / **63.167** |
| (512, 1024, 8) | **889.689** / 895.483 | 27.851 / **27.848** | **4605.128** / 4574.192 | 287.371 / **287.408** | **29.382** / 29.385 |
| (512, 512, 8) | **885.592** / 896.845 | **24.364** / 26.239 | **4626.413** / 4567.243 | **328.516** / 305.032 | **13.336** / 14.305 |

Run status: completed. Recorded GPU 0 cases: 27/27.

Worker results: capacity_limited=13, completed=41.



## Llama3-8B Base, Offline Synthetic, ISL 512, OSL 4096, optimized DiffKV, Run 20261016-150620-020460

- Started: 2026-10-16T15:06:20+09:00
- Model: /home/130bb56/DiffKV/.cache/models/Meta-Llama-3-8B
- GPU: 0
- DiffKV variant: no-sync-bulk-lengths
- Physical GPUs: [0]
- GPU mapping: b8-isl512-osl4096: 0, b4-isl512-osl4096: 0
- Engine limits: max_model_len=4608, max_num_seqs=8, max_num_batched_tokens=8192
- ISL: [512], OSL: [4096], B: [8, 4]
- Measured batches per case: 1
- Warmup output length: 16
- Logs: /home/130bb56/DiffKV/logs/performance/long-context-20261016-osl4096-gpu0
- Command: /home/130bb56/DiffKV/.cache/baselines/fp16-env/bin/python /home/130bb56/DiffKV/benchmarks/run_offline_e2e.py --isls 512 --osls 4096 --batches 8 4 --backends fp16 diffkv --gpu 0 --diffkv-variant no-sync-bulk-lengths --skip-diffkv-on-fp16-oom --repeats 1 --warmup-tokens 16 --output-dir logs/performance/long-context-20261016-osl4096-gpu0 --report evaluation.md

Each cell is **FP16 / DiffKV**. TTFT and TPOT are request means, averaged across measured batches. Among pairs of completed numeric results, lower latency and higher throughput are **bold**; exact ties are both bold.

Prefill Throughput = input tokens / summed prefill engine.step time. Decode Throughput = B * (OSL - 1) / summed decode engine.step time; the first token is generated during prefill. Phase times include scheduling, model execution, sampling and detokenization. E2E Latency measures the complete generation call, including request registration and cleanup, excluding model loading and warmup. OOM means insufficient memory for the requested fixed workload.

| Case (ISL, OSL, B) | TTFT (ms) | TPOT (ms/token) | Prefill Throughput (tok/s) | Decode Throughput (tok/s) | E2E Latency (s) |
|---|---:|---:|---:|---:|---:|
| (512, 4096, 8) | **863.421** / 886.640 | **29.538** / 32.555 | **4745.243** / 4619.821 | **270.963** / 245.845 | **121.821** / 134.198 |
| (512, 4096, 4) | **456.135** / 462.411 | 30.517 / **29.672** | **4491.398** / 4429.332 | 131.118 / **134.852** | 125.424 / **121.971** |

Run status: completed. Recorded cases: 2.

Worker results: completed=4.

FP16, DiffKV 모두 OOM 없이 완료했습니다. 모든 실행에서 요청당 4096토큰, decode 4095회, 고정 batch, preemption 0회를 확인했습니다. FP16 OOM 시 해당 batch의 DiffKV를 건너뛰도록 설정했으며, 실제 skip은 없었습니다. [검증과 조건별 계산](/home/130bb56/DiffKV/logs/performance/long-context-20261016-osl4096-gpu0/analysis.json).


## Run 20261016-161730-125499

- Started: 2026-10-16T16:17:30+09:00
- Model: /home/130bb56/DiffKV/.cache/models/Meta-Llama-3-8B
- GPU: 0
- DiffKV variant: no-sync-bulk-lengths
- Physical GPUs: [0]
- GPU mapping: b32-isl512-osl512: 0, b16-isl512-osl1024: 0, b8-isl512-osl2048: 0
- ISL: [512, 1024, 2048], OSL: [512, 1024, 2048], B: [8, 16, 32]
- Measured batches per case: 1
- Warmup output length: 16
- Logs: /home/130bb56/DiffKV/logs/performance/v1-refactor-validation-20261016-161706/offline
- Command: /home/130bb56/DiffKV/.cache/baselines/fp16-env/bin/python /home/130bb56/DiffKV/benchmarks/run_offline_e2e.py --gpu 0 --diffkv-variant no-sync-bulk-lengths --case-indices 8 16 24 --repeats 1 --warmup-tokens 16 --output-dir /home/130bb56/DiffKV/logs/performance/v1-refactor-validation-20261016-161706/offline --report evaluation.md

Each cell is **FP16 / DiffKV**. TTFT and TPOT are request means, averaged across measured batches. Among pairs of completed numeric results, lower latency and higher throughput are **bold**; exact ties are both bold.

Prefill Throughput = input tokens / summed prefill engine.step time. Decode Throughput = B * (OSL - 1) / summed decode engine.step time; the first token is generated during prefill. Phase times include scheduling, model execution, sampling and detokenization. E2E Latency measures the complete generation call, including request registration and cleanup, excluding model loading and warmup. OOM means insufficient memory for the requested fixed workload.

| Case (ISL, OSL, B) | TTFT (ms) | TPOT (ms/token) | Prefill Throughput (tok/s) | Decode Throughput (tok/s) | E2E Latency (s) |
|---|---:|---:|---:|---:|---:|
| (512, 512, 32) | **2516.794** / 2585.833 | **33.046** / 37.946 | **4882.729** / 4751.925 | **1019.955** / 883.215 | **19.405** / 21.977 |
| (512, 1024, 16) | **1711.645** / 1740.348 | **28.500** / 31.882 | **4787.444** / 4707.206 | **561.776** / 502.153 | **30.868** / 34.356 |
| (512, 2048, 8) | **885.966** / 894.773 | **30.225** / 30.255 | **4624.412** / 4577.829 | **264.791** / 264.531 | **62.757** / 62.828 |

Run status: completed. Recorded cases: 3.

Worker results: completed=6.


### 파일 정리 후 재검증

평가 파일을 5개로 정리한 뒤, 기존 최적화 표에서 FP16과 DiffKV가 모두 완료됐던 3조건을 GPU0에서 각각 1회 재측정했습니다. 위 표가 새 결과이며, 기존 표의 수치는 보존했습니다.

| Case (B, ISL, OSL) | 경로 | 기존 TPOT → 재측정 (ms/token) | 변화 | 기존 E2E → 재측정 (s) | 변화 |
|---|---|---:|---:|---:|---:|
| (32, 512, 512) | fp16 | 41.627 → 33.046 | -20.61% | 23.888 → 19.405 | -18.77% |
| (32, 512, 512) | diffkv | 38.368 → 37.946 | -1.10% | 22.238 → 21.977 | -1.17% |
| (16, 512, 1024) | fp16 | 34.071 → 28.500 | -16.35% | 36.599 → 30.868 | -15.66% |
| (16, 512, 1024) | diffkv | 32.555 → 31.882 | -2.07% | 35.055 → 34.356 | -1.99% |
| (8, 512, 2048) | fp16 | 31.853 → 30.225 | -5.11% | 66.095 → 62.757 | -5.05% |
| (8, 512, 2048) | diffkv | 30.421 → 30.255 | -0.55% | 63.167 → 62.828 | -0.54% |

- 6개 실행 모두 완료했습니다. GPU UUID, engine 설정, 입력, sampling, 패키지 버전, 고정 decode batch, 생성 길이를 기존 기록과 대조했습니다. Preemption은 모두 0회였습니다.
- 세 조건에서 FP16과 DiffKV의 출력 SHA256이 같으며, 각각 기존 결과의 출력 SHA256과도 일치했습니다. DiffKV CUDA extension hash도 기존과 같습니다.
- Correctness: 실제 prompt-cache CUDA 커널 회귀 테스트 3개가 모두 통과했습니다.
- 코드 이동 전후 v1 적용 함수와 표 작성 함수의 AST가 같고, 기존 결과로 생성한 Markdown도 byte 단위로 같았습니다. 온라인 통합은 프로토콜 24개, MATH 채점 self-test, 모의 HTTP 실행 연결 검사를 통과했습니다.
- 성능 수치는 기존 기록과 동일하게 재현되지는 않았습니다. 위 변화율은 파일 정리로 얻은 성능 개선으로 해석하지 않습니다. 마지막 DiffKV 측정 중 SW Thermal Slowdown이 Active인 상태를 관찰했으며, 과거의 throttle reason 기록이 없어 수치 차이 전체의 원인을 확정하지 않았습니다.
- [상세 검증 결과](./logs/performance/v1-refactor-validation-20261016-161706/validation.json)
- [CUDA correctness 출력](./logs/performance/v1-refactor-validation-20261016-161706/correctness.txt)
- [측정 중 GPU 상태 관찰](./logs/performance/v1-refactor-validation-20261016-161706/gpu-state-observation.json)
