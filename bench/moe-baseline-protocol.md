# RegesCore-35B llama.cpp reference protocol

Status: preregistered before timed GPU work.

## Question

Measure the llama.cpp reference arm for RegesCore-1.0-35B-UD-Q4_K_S-BARO.gguf on the 20-prompt set. This receipt is independent of the engine lanes.

## Fixed arm

- Model: `~/Models/RegesCore-1.0-35/RegesCore-1.0-35B-UD-Q4_K_S-BARO.gguf`
- Input: `bench/mtp-prompts/p*.txt`, passed as text to llama.cpp. The sibling `.tokens` files are not passed to llama.cpp. They are our tokenised form and are retained only for comparison context.
- Context: 4096 tokens
- Logical batch: 2048
- Physical microbatch: 512
- CPU threads: 8
- GPU layers: 99 requested; effective loaded value must be read from llama.cpp startup output
- Flash attention: on
- KV cache: f16 for K and V
- Parallel slots: 1
- Sampling: temperature 0, top_k 1, seed 1
- Generation: 64 tokens per prompt
- Cache policy: `cache_prompt=false` on every request, one server process for the round

## Verbatim launch and request

The timed launch is performed from the worktree root after a fresh whiteboard-head check:

```text
gpu-wait run --priority 20 --preemptible --vram 22 -- bash bench/moe-baseline.sh .work/moe-base/run-<timestamp>
```

The script launches exactly:

```text
~/llama.cpp/build/bin/llama-server -m ~/Models/RegesCore-1.0-35/RegesCore-1.0-35B-UD-Q4_K_S-BARO.gguf -c 4096 -ngl 99 -fa on -ctk f16 -ctv f16 -b 2048 -ub 512 -t 8 -np 1 --no-cont-batching --host 127.0.0.1 --port <port>
```

For each prompt, the script sends this request to `/completion`:

```json
{"prompt":"<contents of prompt .txt>","n_predict":64,"temperature":0,"top_k":1,"seed":1,"ignore_eos":true,"cache_prompt":false,"return_tokens":true}
```

## Repetitions and metrics

After the server reports healthy and the effective arm settings have been captured, each prompt gets one untimed warmup request followed by five measured requests. Requests are sequential and `cache_prompt=false`. The first measured request is retained, not discarded. Each response's `timings.predicted_per_second` is the decode tok/s value, and `timings.prompt_per_second` is the prefill tok/s value. The script records all raw responses and computes per-prompt medians across the five measured requests.

The primary decode result is the median of the 20 per-prompt medians, with the minimum and maximum of those 20 values. Prefill is reported separately using the same aggregation. Per-prompt rows include decode and prefill values.

## Arm receipt requirements

The receipt records the llama.cpp commit, binary version/build, build flags as reported by the binary, ROCm/driver, GPU, model size and first 16 sha256 characters, effective context/batch/threads/sampling/gpu-layer values, server props, and the exact invocation. The protocol records the warm-cache state: model weights are loaded once by the server and subsequent requests run with the OS page cache and VRAM resident. No cold-cache number is substituted or implied.

## Void conditions

- Missing or contradictory effective-parameter read-back before the first timed request.
- Any GPU launch not mediated by the specified `gpu-wait` command, or a concurrent GPU workload reported by the wait wrapper.
- Wrong model, model hash, llama.cpp binary, prompt set, prompt source, context, batch, threads, KV type, GPU-layer setting, sampling, or generation length.
- Failed server health, incomplete response, fewer than 64 generated tokens, non-finite timing, or fewer than five measured repetitions for any prompt.
- Any prompt response that does not report both decode and prompt timing.

No prediction is made about the engine lane. A later comparison must use the same model, prompt text, generation length, sampling policy, and explicit warm or cold cache state.
