# qwen35moe decode performance round (preregistered 2026-09-15, before any kernel change)

Bar: llama.cpp RegesCore-35B Q4_K_S decode 109.445 tok/s, 20-prompt median
(`exchange/lane-MOE-BASE-report.md`). Ours today: 42.66 served, 40.0 in
`BARO_PROFILE=1` on p09 (`.work/m5/moe-prof1.log`), engine `-D BARO_MODEL=qwen35moe`
at `97059f0`, pack `.work/moe-w1/pack` rebuilt from the GGUF this session
(733 tensors, 21,005,191,680 bytes, same size as W1's receipt).

## Receipt: where the 22 ms per token go

rocprofv3 kernel trace of one decode (p09, prefill 19 + 63 tokens, tracer on;
`.work/moe-perf/trace/moe_results.db`, summary `.work/moe-perf/kernels-per-call.txt`):

| kernel | per call | calls per token | ms per token | bytes per call | rate |
|---|---|---|---|---|---|
| `moe_gate_up_q4k_pack` (8 experts x 512 rows, gate and up) | 132 us | 40 | 5.3 | 9.4 MB | 71 GB/s |
| `moe_matmul_q8_0_m1` (ssm/attn projections, three variants) | 34 to 46 us | ~160 | 6.2 | 9 to 18 MB | ~320 GB/s |
| `amar_moe_down_q4k` (8 experts, 2048 rows, K=512) | 83 us | 40 | 3.3 | 4.7 MB | 57 GB/s |
| `amar_moe_router_top8` (grid 1, 256 threads) | 73 us | 40 | 2.9 | 2 MB f32 logits already computed | latency |
| `moe_gate_up_q8_0` shared expert + `amar_moe_sig_gate` (grid 1) | 28 + 19 us | 40 | 1.9 | 3.3 MB | |
| everything else (norms, delta, gates, adds) | | | ~1.5 | | |

Bytes per token from the pack index: experts 0.78 GB, ssm/attn projections
1.36 GB, head 0.54 GB, about 2.7 GB; at the q8 row kernel's 855 GB/s that is
3.2 ms, so the floor is near 300 tok/s and llama.cpp's 109 is not a ceiling.

Cause, read from the kernel bodies (`kernels/moe.mojo`): `q8_0_row_dot`,
`q4k_row_dot` and `q6k_row_dot` stride k by lane and call a per-element
`*_value` that loads the block scale(s) and ONE quant byte per lane per
iteration, so a wave moves 32 bytes per load instruction and re-reads scales
per element. The dense path's `q4_dot_blocks` loads 16 bytes per lane per
block and reads each scale once. The router top-8 is a single 256-thread block
doing a serial selection; the sigmoid gate is a single wave.

## Arms and order (each its own commit, each gated before the next)

- **R1 experts, q4_k block dot.** New `q4k_dot_blocks` sharing the
  per-element arithmetic of `q4k_value` exactly (same `d*sc*q - dm*mn`, same
  bf16 round-trip per element) but loading 16 quant bytes per lane and the
  144-byte super-block header once per block; `moe_gate_up_q4k_pack` and
  `amar_moe_down_q4k` call it. `amar_moe_down_q6k` (3 layers) same treatment
  or left as is, stated. Prediction: gate+up 132 -> under 30 us, down 83 ->
  under 20 us per call; token 22 -> about 15 ms.
- **R2 projections, q8_0 block dot.** `q8_0_row_dot` loads 16 quant bytes per
  lane per half-block and the f16 scale once per block, same per-element
  bf16 rounding. Prediction: 34 to 46 -> 12 to 20 us per call (near the
  dense q8 row kernel's rate for the same bytes); token about 15 -> 10 ms.
- **R3 latency kernels.** Router top-8 as a wave-parallel selection (8 rounds
  of warp argmax over 256 logits, or a block-wide bitonic top-k); sigmoid gate
  folded into the router launch. Prediction: 73 + 19 -> under 10 us per
  layer; token about 10 -> 7 ms, i.e. about 140 tok/s.

## Gates, every arm

1. `kernels/test_moe_block.mojo` parity vs `tools/moe-ref.py` (fixtures
   regenerated this session into `.work/gguf/`): expert ids exact 8/8, routed,
   shared and y rel < 5e-3, the frozen bar from W3 gate 1.
2. 20-prompt teacher-forced agreement vs llama.cpp (`bench/moe-gate2-force.sh`
   shape, reference ids from llama-server on the same GGUF): mean within
   +-0.5 of 53.20 (P14 bar; accumulation order may move single tokens,
   dequant math may not). Below 52.70 the arm is reverted.
3. `./run-tests.sh` and `tools/ci-checks.sh` green; `isa-receipt` census of
   the changed kernels recorded (VGPR, scratch) as a receipt, not a target.
4. tok/s: 20-prompt median (P4), `bench/ab-prompts.sh` base vs arm with
   `AB_ENGINE_B`, both `-D BARO_MODEL=qwen35moe`, `BARO_MEGA=0 BARO_SPEC=0`,
   clock and power cap read back, identity per prompt (GENERATED equal is
   NOT required here, gate 2 is the identity: record equal/unequal per prompt).
5. Kill line per arm: median below +10% over its base is a no-op and is
   reverted with the numbers in this file.

Fixed for the round: no change to the router math, the top-8 renormalisation,
the shared-expert gate formula, or the pack format. Prefill (m>1) keeps its
path; only m=1 decode kernels change.

### R1 result (2026-09-15): landed

`q4k_dot_blocks` (16 quant bytes per lane, super-block header once per block,
per-element `d*sc*q - dm*mn` with the bf16 round-trip unchanged) in
`moe_gate_up_q4k_pack` and `amar_moe_down_q4k`; `amar_moe_down_q6k` (3
layers) left as is. Gate 1: `test_moe_block` PASS (routed rel 9.6e-8, y
1.3e-4, ids 8/8). Gate 2: 20-prompt mean **52.85/64** against the 53.20
baseline reproduced in the same stint (`.work/moe-perf/base/gate2`), inside
the +-0.5 band; per prompt 53 58 56 49 57 45 58 57 40 56 48 59 55 53 50 44
44 56 60 59. Gate 3: run-tests and ci-checks green at commit. Gate 4:
20-prompt tok/s_gen median **42.88 -> 55.98 (1.306x)**, ranges
42.77..42.95 vs 55.22..56.09, sclk med 3268 MHz, 290 W / -100 mV, GENERATED
equal on 15/20 prompts (accumulation order, as allowed). `BARO_PROFILE=1`
p09: ffn 19.5 -> 12.7 ms per token. Predicted "token 22 -> about 15 ms":
measured 23.3 -> 17.9 ms, the expert kernels alone landed at prediction.

### R2 result (2026-09-15): landed

`q8_0_row_dot` rewritten: two lanes per 34-byte block, one 16-byte quant
load per lane (2-byte aligned, the compiler emits `global_load_b128`, 92
VGPRs, 0 spills), scale read once per block, per-element `(d*q)` bf16
round-trip unchanged; all five callers (ssm/attn projections, shared
expert, sigmoid gate input) inherit it. Gate 1: `test_moe_block` PASS.
Gate 2: 20-prompt mean **53.00/64** (band 52.70..53.70); per prompt 53 58
56 49 57 45 57 57 39 56 48 59 55 53 50 47 45 57 60 59. Gate 4: 20-prompt
tok/s_gen median **55.97 -> 71.89 (1.284x)** against R1, ranges
55.86..56.06 vs 71.67..72.14, sclk med 3277 MHz, 290 W / -100 mV, GENERATED
equal 17/20. `BARO_PROFILE=1` p09: ssm 9.0 -> 5.7 ms per token, ffn
12.7 -> 11.3 (shared expert), profile-mode 50.6 -> 64.5 tok/s.

### R3 result (2026-09-15): landed

Router top-8 as one wave (each lane owns 8 experts in registers; softmax
max and sum by warp reductions; eight rounds of warp argmax with
strict-greater, lowest-index tie-break, the same selection order as the
serial scan it replaces; renormalisation unchanged), dispatched at
`block_dim = 32`; sigmoid gate reads 8 f32 per lane. Gate 1: `test_moe_block`
PASS (its launch adapted to the one-wave router). Gate 2: 20-prompt mean
**53.15/64**; per prompt 55 58 56 50 57 45 57 57 40 56 50 59 54 53 50 46 44
56 60 60. Gate 4: 20-prompt tok/s_gen median **71.82 -> 93.46 (1.301x)**
against R2, ranges 71.52..72.19 vs 92.12..93.80, sclk med 3276 MHz, 290 W /
-100 mV, GENERATED equal 16/20. `BARO_PROFILE=1` p09 per token: attn 1.6,
ssm 5.8, ffn 7.0, head 0.8 ms; profile-mode 64.5 -> 81.7 tok/s.

### Round summary

42.88 -> 55.98 -> 71.89 -> 93.46 tok/s_gen (20-prompt medians, each arm A/B
against its predecessor in one stint), 2.18x, agreement 53.20 -> 52.85 ->
53.00 -> 53.15 within the band throughout. llama.cpp's bar 109.4: 0.39x ->
0.85x. Next levers, from the R3 profile: the ssm sub-block's per-layer small
kernels (conv, gates, l2, delta, about eight launches per layer) and the
remaining expert time; a fresh rocprofv3 trace is the first step of any
follow-up round, not this file's numbers.

Gate 3 on the committed tree at `bf45322` + `44bf333`: `./run-tests.sh`
(through gpu-wait) exit 0, 104 PASS lines; `tools/ci-checks.sh` green for every
tracked source, its one failure being an untracked in-progress LatentOS probe
(`bench/latentos-ipc-probe.mojo`, link error) belonging to the B3 lane, which
was told to land it green or excluded.
