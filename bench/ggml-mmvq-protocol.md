# ggml mul_mat_vec_q (Q8_0) vs amar_gemv_q8 on the Spark m=1 shapes

Preregistered 2026-09-08 ~20:05, before any run. Question: is our GEMV faster per shape than
llama.cpp master's (b10860) own kernel, or is the 1.20x decode win all launch count / graph?

## Arms, one stint, rocprofv3 kernel trace around each
- **ggml**: `bench/ggml-mmvq/mmvq_shapes` -- public ggml backend API, one `ggml_mul_mat`
  (Q8_0 [K,N] x F32 [K,1]) per shape, 20 warm + 200 timed iters; per-kernel us from the
  trace (`quantize_q8_1` + `mul_mat_vec_q<Q8_0,1,...>`), wall us/iter printed as a cross-check.
- **ours**: `.work/spark/spark-engine` 8-token run under the same rocprofv3 flags; `amar_gemv_q8`
  rows by shape (already measured 2026-09-08 17:00: 33.5/33.5/33.1 ffn+down, 21.0 qkv,
  15.6 o, 3.75 headgate, 386 lmhead us).
- Shapes: headgate 16x2560, o 2560x4096, qkv 6144x2560, ffn_gate 10240x2560, down 2560x10240,
  lmhead 131072x2560. Power cap / vddgfx read back first.

## Frozen prediction
- ggml `mul_mat_vec_q` on ffn_gate (26.2 MB): **30-40 us** (700-870 GB/s), plus quantize_q8_1
  ~2 us. Ours 33.5. Ratio ours/theirs per shape **0.9-1.15** on the three big shapes.
- Small shapes (headgate, o): theirs faster by up to 2x -- their block geometry covers more
  rows per block; ours is one wave per row with a fixed 4-unroll.
- lmhead: within 10 % either way (pure bandwidth, 335 MB).
- Verdict rule: if ours < theirs on all three big shapes, the kernel hook into llama.cpp is
  worth an M; if within +-10 % the decode win is launch-path only and the hook is closed.

## Result (2026-09-08 ~20:15, one stint, 290 W / -100 mV, llama.cpp b10860, engine c29f4a5)

Per-kernel median us from rocprofv3 (`.work/ggml-prof/`, `.work/spark/prof2/`); ggml's
`quantize_q8_1` (1.5-2.8 us) is an extra launch on their side and is not in the ratio.

| shape | ggml mul_mat_vec_q | ours amar_gemv_q8 | ours/ggml |
|---|---|---|---|
| headgate 16x2560 | 3.24 | 3.71 | 1.15 |
| o 2560x4096 | 18.60 | 15.71 | 0.85 |
| qkv 6144x2560 | 31.64 | 21.18 | 0.67 |
| ffn_gate 10240x2560 | 43.96 | 33.60 | 0.76 |
| down 2560x10240 | **24.04** | **33.60** | **1.40** |
| lmhead 131072x2560 | 499.5 | 388.2 | 0.78 |

Prediction: ggml on ffn_gate predicted 30-40 us, measured 44 (slower than predicted); ratio
predicted 0.9-1.15 on the big shapes, measured 0.67-0.78 on two of three -- ours is faster
than predicted relative to theirs. Miss: `down` (K=10240, N=2560) -- theirs 1.40x faster.
Ours is one wave per row: 2560 rows = 2560 waves over 96 CUs (~27 waves/CU, 4 waves per
block => ~7 blocks/CU), too little in flight to hide the 10 KB-per-row stream; their
kernel spreads long rows over more threads. Wall us/iter in the harness (52-88 us) is
graph-compute overhead per call, not kernel time.

Verdict: on 4 of 6 shapes ours is faster (0.67-0.85), so the ggml hook is worth an M for
those shapes only; **the actionable finding is on our side**: `down` is occupancy-limited
and a split-K (2-4 way, atomic or 2-stage) on K=10240 should recover ~10 us/layer =
~0.36 ms/token (~5 %). Goes into the round-2 brief ahead of the argmax item.

## Correction (2026-09-08 ~21:00) — the first table's ggml arm was Infinity-Cache-hot

The harness re-ran ONE 28 MB weight 220 times per shape; 28 MB fits the 96 MB IC, so the
`down` row (24.0 us = 1.16 TB/s, above HBM peak) was cache, not kernel. Harness now rotates
8 disjoint weight copies per shape (222 MB > IC; lm-head alone is 357 MB). Cold re-run, same
stint as our trace:

| shape | ggml cold | ours | ours/ggml |
|---|---|---|---|
| headgate 16x2560 | 3.28 | 3.71 | 1.13 |
| o 2560x4096 | 18.50 | 15.71 | 0.85 |
| qkv 6144x2560 | 27.88 | 21.18 | 0.76 |
| ffn_gate 10240x2560 | 42.64 | 33.60 | 0.79 |
| down 2560x10240 | 34.42 | 33.60 | 0.98 |
| lmhead 131072x2560 | 498.9 | 388.2 | 0.78 |

The `down` inversion is gone (parity). Chasing it before this correction: split-K KS=2/4/5
(37-40 us, slower: the K-tail loop loses the 4-unroll, and summation order changes flipped
gate tokens — text pos 38 / chat pos 2), LDS-staged bf16 activation (44.5 us, slower, LDS
occupancy). Both reverted, both closed. Verdict stands: ours faster on 5/6 shapes (0.76-0.85
on the ones that carry bytes), ggml hook not worth it; nothing to fix on `down`.
Ledger: `m.ledger/mojo-baro.md` 2026-09-08.
