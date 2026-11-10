# wavegroup

Llama inference in Mojo on the 7900 XTX (gfx1100) that decodes 1.37x faster than llama.cpp (same stint, faster on 21/21
prompts) and gives the same greedy ids on 13/21 prompts for all 64 tokens; on 6 of the other 8, llama.cpp itself splits
between its own configs at similar tokens (`docs/m4.md`).
Mojo is the compiler; wavegroup is the model wiring, one planned memory arena and the launch order, on top of our own
kernels and mojo-baro's, this repo two levels up (`lab/wavegroup/`, merged 2026-10-26, history kept). CLI name: `wg`.

## Llama 3.2 1B Instruct Q4_K_M, measured 2026-10-26

| | wavegroup | llama.cpp ca3d5a3e1 (ROCm) | tinygrad (same GGUF) |
|---|---|---|---|
| greedy tokens, `tests/ref_story.txt` | 64/64 equal | reference | 64/64 equal |
| decode | 586 tok/s (400 tokens) | 441 tok/s (tg64) | 45.5 default, 133.6 JITBEAM=2 |
| prefill, 511 tokens | 23.4 ms warm | 27.0 ms (pp512) | not measured |
| weights in VRAM | 800 MB, as stored in the GGUF | | fp16 dequantised at load |

Same-stint decode (`tools/stint.sh`, one gpu-wait job, 21 prompts x 64 greedy tokens, llama-server in its fast config
-fa on / KV f16 / ignore_eos, drift control 1.004): wavegroup median 567.9 tok/s, llama.cpp 415.2, ratio median 1.367
(min 1.332, max 1.383). Receipt: `docs/receipts/2026-10-26-stint.txt`. The 586 / 441 row above is two harnesses.

## Role next to mojo-baro (2026-10-26)

wavegroup is the bench where a lever is proved on one model before it moves into mojo-baro's spark engine. Moved so far:
fused gate/up GEMV, split-K QKV GEMV, batched prefill, rope_freqs, fused decode rope (1.028x, byte-equal), split decode attention (from 128 tokens of context: 4K context 159 -> 424 tok/s, `docs/receipts/2026-10-26-att-split-landed.txt`). Same stint after those: mojo-baro 558.3 tok/s, wavegroup
571.1, llama.cpp 411.9; both engines give the same greedy ids on 21/21 prompts. `tools/stint.sh` takes a second engine arm
(`ALT=tools/baro_arm.sh`). Receipts: `docs/receipts/2026-10-26-stint-baro-head.txt`, `docs/receipts/2026-10-26-baro-prefill.txt`.

## Run

```
tools/setup.sh                                              # own .venv (not the repo root one): Mojo 1.3.0 + MAX 26.8.0, nightly 2026102505
tools/mojo.sh -D WMMA_TB=1 -D WMMA_C_F16=0 -o .work/decode kernels/llama/decode.mojo
gpu-wait run --timeout 300 -- .work/decode MODEL.gguf "The capital of France is" 24 [noprefill][mega]
```

Full gate (tests, ref_story, three-arm stint): `BARO_ENGINE=spark BARO_PACK=pack tools/gate.sh`.

The prompt is text or a comma separated id list. Text goes through the GGUF's own BPE tokenizer (mojo-baro
`serve/tokenizer.mojo`, BOS added), and the output is printed as ids and then as text. Tokenizer gate (CPU):
`test_tok MODEL.gguf tests/tok_corpus.txt tests/tok_ref.txt`, 9/9 lines equal to `llama-tokenize` ids and round trip.
Text mode stops at the GGUF EOS id (checked every 32 tokens, so up to 31 extra are computed); id mode always runs N.
Greedy only; context up to 512 tokens.

## Layout

- `kernels/llama/`: `weights` (GGUF mmap, one upload), `plan` (arena by liveness from a step list), `llama` (forward
  over T rows: GEMVs at decode, dequant + WMMA GEMM at prefill), `rope` (rope + KV append), `attn_split` (split decode
  attention at head size 64), `decode` (cache, loop, device argmax), `mega` (persistent kernel, opt-in, ~7% slower).
- `kernels/gemm/`: WMMA GEMM and its runner; `kernels/gemv/`: split-K multi-segment GEMV and its bandwidth bench.
- `tools/`: setup, build wrapper, sweeps, `ref_tokens.sh` (llama.cpp reference ids), `tinygrad_decode.py` (baseline),
  `kernel_table.py` (per-kernel time from rocprofv3).
- `docs/`: milestones (`m1`..`m4`), plans and measured levers. History: `~/Brain/wavegroup/`.
