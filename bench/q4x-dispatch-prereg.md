# Prereg: lane q4x-dispatch (frozen before any timed run)

Goal: fewer dispatches per decode token in `serve/q4x.mojo`. Baseline table (main eb6b49f + counter 0a84175,
`.work/count-main.log`, p01, GSQ-RCO Q2_0 pack): **1244 dispatches/token** (1189 kernels + 55 copies).
hyper-connection (hc) block per sublayer = gnorm, down GEMV, inject fmv, up GEMV, hcmix = 5 dispatches, 97 hc mixes per token.

## Fusions (one concern per commit, each with a bit-equality parity check against the unfused kernels)
| id | change | dispatches/token |
|---|---|---|
| F5 | skip `indexer.k_proj` fmv when the sparse indexer is off (default dense; the result is only read when `Q4X_SPARSE` is set) | -12 |
| F7 | hc down GEMV + inject fmv in one kernel (T = 1, BF16 weights) | -96 |
| F2 | hc up GEMV + hcmix in one kernel (T = 1, BF16 weights; a wave computes the 4 gate rows of one output element) | -97 |
| F1 | `ffn_out` folded into the next `gnorm` (pending combine reads the expert outputs directly; unfused flush before PLE and under `Q4X_DUMP`) | -47 |

Per-element arithmetic and reduction trees are kept, so the ids must be byte-identical to main.

## Prediction (frozen)
- dispatches/token: 1244 -> **992** (-252, -20%). Checked by the counter build, not by timing.
- A = main binary `.work/speed/q4x-head`: **20.4 ms/token** (serve16k report: 20.0 to 20.8 at the 16K floor, 20 prompts).
- B = lane binary: **19.7 ms/token** (B/A 0.966), bounds 19.2 to 20.4.
  Basis: lane q4x-r2 removed 397 dispatches for about -1 ms (about 2.5 us each, noisy); 252 x 2.5 us = 0.63 ms, plus the removed
  tiny-kernel ramps. Risks that eat it: the fused up kernel runs 2560 waves of 4 dots instead of 10240 waves of 1; kernel busy
  time may rise.
- Identity: ids equal on 20 of 20 prompts (4 runs each, A B B A). Teacher-forced gate `bench/q4x-gate.sh OUT 20 32`: PASS (bar ours >= ctrl - 2 pp).

## Kill line (frozen)
B/A (median over 20 prompts, same-job A B B A, NG 64, window 16384) **> 0.97 = no win** (verdict MISS, or KILLED if the
fused kernels also fail parity). Identity FAIL on any prompt = KILLED regardless of speed.
A post hoc rule may annotate the verdict, never flip it.
