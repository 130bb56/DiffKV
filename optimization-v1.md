# DiffKV에 적용한 최적화, AS-IS와 TO-BE

현재 적용한 옵션은 `no-sync-bulk-lengths`이며, 아래 **두 최적화**를 함께 적용합니다. 이 문서의 FP16 비교 대상은 저장소에 있는 기존 vLLM baseline입니다. 현재 구현은 [diffkv_v1.py](./benchmarks/diffkv_v1.py#L32)이며, 기존 `diagnose_decode_overhead.py`에서 최적화 적용 부분을 분리했습니다. 과거 측정 폴더의 소스와 hash는 당시 기록을 유지합니다.

| 적용한 최적화 | AS-IS | TO-BE | 적용 위치 |
|---|---|---|---|
| 불필요한 명시적 GPU 동기화 제거 | Llama3-8B decode당 attention 96회, engine 4회, 합계100회 대기 | 대상 `torch.cuda.synchronize()` 호출 삭제 | `SparsePagedAttention.forward()`, `LLMEngine.step()` |
| Context 길이 읽기 일괄화 | 각 layer가 GPU에서 `.item()`으로 길이를 읽어 decode당32회 전송과 대기 | 길이 벡터를 CPU로 한 번 복사한 뒤 각 layer는 CPU 정수를 사용 | `ModelRunner._prepare_decode()`, `SparsePagedAttention.forward()` |

첫 번째 변경은 **prefill과 decode 모두**에 적용됩니다. 두 번째 변경은 **decode 준비와 decode attention**에 적용됩니다.

## 1. GPU 작업이 끝날 때까지 CPU를 멈추던 호출 제거

**실제 변경은 `torch.cuda.synchronize()`를 실행하지 않도록 한 것입니다.** 이 함수는 GPU 작업이 끝날 때까지 호출한 CPU 스레드를 기다리게 합니다. 기존 코드에서는 시간 측정 지점마다 이 호출이 들어 있습니다.

현재 패치는 해당 호출을 빈 문자열로 치환해 삭제합니다.

### AS-IS → TO-BE 실제 변경

아래 diff의 `-` 줄은 삭제하는 호출입니다. 각 호출 다음의 시간 측정 코드는 유지합니다.

~~~diff
@@ -341,2 +341,2 @@
-            torch.cuda.synchronize()
             t0 = time.time()

@@ -385,2 +385,2 @@
-            torch.cuda.synchronize()
             t1 = time.time()

@@ -450,2 +450,2 @@
-            torch.cuda.synchronize()
             t2 = time.time()
~~~

각 지점은 다음 코드에 있습니다.

| 변경 지점 | 기존 코드 |
|---|---|
| QKV projection과 RoPE 이후, attention 진입 시점 | [sparse_attention_big_kernel.py, 341행](./vllm/model_executor/layers/sparse_attention_big_kernel.py#L341) |
| Cache compression 이후 | [sparse_attention_big_kernel.py, 385행](./vllm/model_executor/layers/sparse_attention_big_kernel.py#L385) |
| Attention kernel 이후 | [sparse_attention_big_kernel.py, 450행](./vllm/model_executor/layers/sparse_attention_big_kernel.py#L450) |

예를 들어 cache compression 이후의 대기는 실제 실행 흐름을 다음처럼 바꿉니다.

- **AS-IS:** CPU가 cache kernel 실행을 요청한 다음 `torch.cuda.synchronize()`에서 GPU 완료를 기다립니다. 완료된 뒤 CPU가 attention kernel 실행 요청으로 진행합니다.
- **TO-BE:** CPU가 cache kernel 실행을 요청한 다음, 이 명시적 대기 없이 attention kernel 실행 요청으로 진행할 수 있습니다. GPU는 같은 CUDA stream에 들어온 cache kernel, attention kernel을 순서대로 실행합니다.

따라서 줄인 것은 **커널 사이에 끼어 있던 CPU의 GPU 완료 대기**입니다. Cache 압축과 attention 계산 코드는 그대로 실행합니다. `time.time()`과 시간 변수도 그대로 남으며, 이 타이머 자체는 최적화 내용이 아닙니다.

Llama3-8B는 32 layers이므로 attention의 세 호출은 decode당 `3 * 32 = 96`회입니다. 추가로 `LLMEngine.step()`에 [542행](./vllm/engine/llm_engine.py#L542), [549행](./vllm/engine/llm_engine.py#L549), [560행](./vllm/engine/llm_engine.py#L560), [570행](./vllm/engine/llm_engine.py#L570)의 네 호출이 있습니다. **이 두 함수에서 decode당 합계 100회의 명시적 동기화 호출을 제거했습니다.**

Prefill의 `forward()`에 있는 [187행](./vllm/model_executor/layers/sparse_attention_big_kernel.py#L187) 등의 명시적 동기화에도 같은 치환을 적용합니다. Sampling의 CPU 결과 읽기와 benchmark 측정 경계의 동기화는 유지합니다.

### 이 변경을 적용하는 코드

[diffkv_v1.py, 59행](./benchmarks/diffkv_v1.py#L59)의 실제 치환 정의입니다.

~~~python
replacements = [("torch.cuda.synchronize()", "")]
~~~

[63행](./benchmarks/diffkv_v1.py#L63)과 [64행](./benchmarks/diffkv_v1.py#L64)에서 각각 attention과 engine 함수에 적용합니다.

~~~python
manifest["patches"].append(replace_method(attention_type, "forward", replacements, folder))
manifest["patches"].append(replace_method(type(engine), "step", replacements[:1], folder))
~~~

`replace_method()`는 [20행](./benchmarks/diffkv_v1.py#L20)에서 함수 코드의 문자열을 치환하고, [24행](./benchmarks/diffkv_v1.py#L24)에서 변경한 함수를 만든 뒤, [25행](./benchmarks/diffkv_v1.py#L25)에서 실행 중인 클래스의 메서드로 설정합니다.

계산과 인자를 포함한 함수 전체는 아래 원문에서 확인할 수 있습니다.

| 함수 | AS-IS 전체 함수 | 실험 당시 저장한 TO-BE 전체 함수 |
|---|---|---|
| Attention | [원본 forward()](./vllm/model_executor/layers/sparse_attention_big_kernel.py#L109) | [실행에 사용한 forward()](./logs/performance/target-ablation-20261016-130112/diffkv-no-sync-bulk-lengths/SparsePagedAttention.forward.py#L1) |
| Engine | [원본 step()](./vllm/engine/llm_engine.py#L531) | [실행에 사용한 step()](./logs/performance/target-ablation-20261016-130112/diffkv-no-sync-bulk-lengths/LLMEngine.step.py#L1) |

실험 당시 저장한 파일은 측정 기록으로 유지합니다. 현재 패치는 동기화 호출을 삭제하며, 이전 스냅샷의 자리표시자는 추가하지 않습니다. TO-BE attention 전체 함수에는 2절의 context 길이 읽기 변경도 함께 들어 있습니다. Kernel이 사용하는 stream은 [attention](./csrc/attention/sparse_attention_kernels.cu#L1233), [cache](./csrc/cache_kernels.cu#L1679)에서 확인할 수 있습니다.

## 2. Context 길이의 GPU→CPU 읽기 일괄화

### AS-IS

[ModelRunner._prepare_decode(), 236행](./vllm/worker/model_runner.py#L236)은 GPU의 `kv_len_tables`에서 layer별 최대 KV 길이 벡터를 계산합니다.

~~~python
# ModelRunner._prepare_decode()
max_context_len = torch.amax(
    torch.sum(kv_len_tables[slot_ids], dim=3), dim=(0, 2))
~~~

이 벡터를 `InputMetadata`로 전달한 뒤, [각 layer의 attention, 355행](./vllm/model_executor/layers/sparse_attention_big_kernel.py#L355)이 다음처럼 읽습니다.

~~~python
# SparsePagedAttention.forward()
max_context_len = input_metadata.max_context_len[self.layer].item()
~~~

GPU tensor에서 CPU 정수로 읽는 `.item()`이 layer마다 실행되므로, Llama3-8B에서는 decode당32회의 scalar 전송과 대기가 발생합니다.

### TO-BE

GPU에서 길이 벡터를 계산하는 기존 `_prepare_decode()`를 호출한 다음, 반환된 metadata의 길이 벡터를 한 번에 CPU로 읽습니다. 실제 [wrapper 코드, 66행](./benchmarks/diffkv_v1.py#L66)은 다음과 같습니다. 주석만 생략했습니다.

~~~python
runner_type = type(worker.model_runner)
original_prepare = runner_type._prepare_decode

def prepare_decode(self, *args, **kwargs):
    tokens, positions, metadata = original_prepare(self, *args, **kwargs)
    metadata.max_context_len = metadata.max_context_len.cpu().tolist()
    return tokens, positions, metadata

runner_type._prepare_decode = prepare_decode
~~~

이후 [61행의 치환](./benchmarks/diffkv_v1.py#L61)으로 각 layer에서 `.item()`을 제거합니다.

~~~python
# AS-IS: GPU tensor에서 scalar 읽기
max_context_len = input_metadata.max_context_len[self.layer].item()

# TO-BE: 이미 CPU로 복사된 리스트에서 정수 읽기
max_context_len = input_metadata.max_context_len[self.layer]
~~~

CPU가 기다리는 동작 자체를 없앤 것이 아니라, **32번으로 나뉜 scalar 읽기를 한 번의 vector 읽기로 합친 변경**입니다. 읽는 대상은 context 길이 metadata이며, `.cpu().tolist()`는 동기식 복사입니다.

실제 실행에 적용해 저장한 코드도 확인할 수 있습니다.

- [TO-BE decode 준비 wrapper](./logs/performance/target-ablation-20261016-130112/diffkv-no-sync-bulk-lengths/prepare_decode.bulk_lengths.py#L1)
- [TO-BE attention의 CPU 정수 읽기](./logs/performance/target-ablation-20261016-130112/diffkv-no-sync-bulk-lengths/SparsePagedAttention.forward.py#L247)

## 3. 기존 FP16 vLLM과의 차이

FP16 경로는 pruning을 하지 않아, 같은 request의 KV 길이가 모든 layer와 head에서 같습니다. CPU에서 관리하는 sequence 길이로 context 길이를 알 수 있습니다.

[FP16 ModelRunner, 167행](./.cache/baselines/vllm-1db83e3/vllm/worker/model_runner.py#L167)에서 길이를 가져오며, [188행](./.cache/baselines/vllm-1db83e3/vllm/worker/model_runner.py#L188)에서 최대값을 계산합니다. 아래는 관련 부분을 모아 발췌한 코드입니다.

~~~python
# Request별 CPU 길이 수집
seq_len = seq_data.get_len()
context_len = seq_len if self.sliding_window is None else min(
    seq_len, self.sliding_window)
context_lens.append(context_len)

# Batch의 최대 길이도 CPU 정수로 계산
max_context_len = max(context_lens)
~~~

각 layer는 이 CPU 정수를 그대로 사용하므로 layer마다 `.item()`으로 GPU 값을 읽는 과정이 없습니다. 커널 입력용 `context_lens` 배열은 [228행](./.cache/baselines/vllm-1db83e3/vllm/worker/model_runner.py#L228)에서 별도로 GPU에 전달합니다. 이번 비교는 eager 실행입니다.

~~~python
context_lens = torch.tensor(context_lens,
                            dtype=torch.int,
                            device=device,
                            pin_memory=pin_memory)
~~~

반면 DiffKV는 pruning 이후 실제 보관 길이가 layer와 KV head마다 달라지며, 현재 구현은 이를 GPU의 `kv_len_tables`에서 관리합니다.

| 경로 | Context 길이 준비와 읽기 |
|---|---|
| FP16 | CPU에서 최대 길이 계산, 각 layer가 정수 사용 |
| 기존 DiffKV | GPU에서 길이 벡터 계산, 각 layer가 `.item()`으로 읽기 |
| 최적화 DiffKV | GPU 벡터를 CPU로 한 번 복사, 각 layer가 리스트 정수 사용 |

## 4. 실제 최적화 적용 방식과 범위

[벤치마크 worker, 591행](./benchmarks/run_offline_e2e.py#L591)은 DiffKV backend에만 `apply_variant()`를 호출합니다. 모델 로드 후, warmup 전에 실행합니다.

~~~python
if backend == "diffkv" and variant != "original":
    from diffkv_v1 import apply_variant
    result["python_variant"] = apply_variant(
        engine, w, variant, args.output_dir / f"{case['id']}-diffkv.ablation")
    result["measurement"]["diffkv_variant"] = variant
~~~

[replace_method(), 10행](./benchmarks/diffkv_v1.py#L10)은 원본 함수의 코드를 읽어 변경한 뒤, 실행 프로세스 안에서 클래스 메서드를 교체합니다. 따라서 원본 엔진 파일에는 AS-IS 코드가 남아 있고, 실행된 TO-BE 함수는 각 실험의 ablation 폴더에 별도로 저장됩니다.

| Variant | 명시적 sync 제거 | Context 길이 일괄 읽기 |
|---|---|---|
| `original` | 적용하지 않음 | 적용하지 않음 |
| `no-sync` | 적용 | 적용하지 않음 |
| `no-sync-bulk-lengths` | 적용 | 적용 |

이 두 최적화의 변경 범위는 Python 실행 경로의 대기와 metadata 전송입니다. FP16에는 두 변경 모두 적용되지 않습니다. DiffKV의 CUDA kernel 계산, 양자화 설정, pruning 정책은 이 두 최적화에서 변경하지 않았습니다.
