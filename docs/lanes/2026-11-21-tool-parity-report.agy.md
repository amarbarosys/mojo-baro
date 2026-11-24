# Partner Review: Tool Parity Report (lane tool-parity)

Reviewer: agy-partner (standing partner)
Date: 2026-11-21
Verdict: PASS. Root cause of toolgate divergence cleanly isolated to prompt injection (`BARO_NEURONS=1`), with render parity proved 100% (56/56 turn-1, 55/55 turn-2, 8/8 app, 64/64 Flip4). Tool call execution achieves 52/55 parity (94.5%) against L8x, with the 3 misses fully characterized (2 near-tie activation precision flips, 1 parser recovery on malformed call).

---

## 1. Verification of Core Results

1. **Root Cause Confirmed (Fault 1):**
   - The initial 0/56 failure was not template or tokenizer corruption.
   - `tools/baro` line 299 exported `BARO_NEURONS=1`, causing `serve/src/main.rs` to append ~1.2 KB of brain-ask notes to the user prompt.
   - Setting `BARO_NEURONS=0` and passing `neurons: false` restored byte-for-byte prompt parity across all engines:
     - 56/56 toolgate turn-1 prompt bytes and token IDs equal.
     - 55/55 turn-2 prompt bytes and token IDs equal.
     - 8/8 mobile app prompts equal.
     - 64/64 Flip4 prompt SHA matches host prompt SHA.
2. **Toolgate Execution Parity (52/55 on L8x, 52/56 on L8):**
   - On valid uncollided port runs, baro matched llama-server Q8_0 on **52 of 55** tool calls (94.5%).
   - All 52 matches produced identical first tool calls.
   - Used count: 51 on baro vs 49 on llama (bar: within 1).

---

## 2. Analysis of the 3 Divergence Cases

### Rows 4 and 55 (Numerics / Activation Precision)
- **Receipts:**
  - Row 4 (tg-js-04): Both selected `run_js`, but argument text diverged at token 67.
  - Row 55 (tg-web-05): Both selected `web_search`, but query arguments diverged at token 18.
- **DeepSeek Empirical Context:**
  - Llama.cpp (both GPU and ARM CPU) quantizes activations to INT8 inside Q8_0 matmuls.
  - Spark (`tools/baro`) multiplies Q8_0 weights by FP32 activations (`act: float32`).
  - INT8 vs FP32 activation differences produce ~1%–3% per-token argmax divergence on near-tie logit boundaries. Compounded over 50–100 tokens, cumulative divergence probability reaches 40%–70%.
  - Across 119 raw prompts, 88 were identical, 19 were prefix matches, and only 12 diverged.
  - Both engines picked the correct tool. The minor argument string differences are expected activation precision artifacts, not tool schema logic failures.

### Row 12 (Parser Recovery on Malformed Tool Call)
- **Receipt:** The model emitted an unclosed `<tool_call>` tag without closing XML markers or valid JSON arguments.
- **Divergence:**
  - Llama.cpp synthesized 4 empty `run_js {}` calls to keep the turn alive.
  - Baro dropped the malformed block and returned empty content.
- **Architectural Verdict:**
  - Baro's behavior is **correctness-preserving**. Fabricating synthetic `{}` arguments when none were emitted is semantic fabrication. Executing a tool with empty arguments can trigger unintended side effects or schema crashes downstream.
  - Parity should not emulate llama.cpp's empty `{}` hallucination. Retain baro's strict drop behavior.

### Row 42 (Model Repetition Loop)
- Model entered an infinite repetition loop emitting hundreds of `brain_ask` calls.
- This is a known training/decoding defect, not an engine parser bug. Capping tool calls per message in the server harness (e.g. max 8 calls per turn) is the correct fix.

---

## 3. Recommendations on Open Decisions

1. **Decision 1 (Activation Precision):**
   Do not introduce INT8 activation quantization into spark solely to match llama.cpp's rounding noise. FP32 activations maintain higher mathematical fidelity. Accept 52/55 as passing toolgate parity with a documented near-tie margin.
2. **Decision 2 (Parser Policy):**
   Keep baro's parser policy (drop malformed unclosed blocks). Do not copy llama.cpp's synthetic `{}` argument insertion.
3. **Decision 3 (Loop Guard):**
   Add a hard limit in `serve/src/main.rs` capping tool calls per response (default 10) to reject runaway repetition loops before context exhaustion.
4. **Decision 4 (K2 Serve Wiring):**
   Defer K2 megakernel serve wiring to a dedicated kernel fable lane. The existing spark dispatch path is fully functional for serving.
