# turbo_rows round A: preregistration (2026-11-16)

Change: A rows to int8 + f32 scale per 32-block, sudot4 dot against Q4_0 nibbles, int8 LDS stage.
Two arms: QA-LDS (quantize while staging, per block) and QA-PRE (separate pre-kernel writes int8 A + scales).

Gate (fixed before any speed number):
1. Teacher-forced top-1 agreement >= 99% vs the shipped bf16 turbo_rows path over the stable prompt set,
   a same-arm repeat of the baseline beside it (27B near-tie flips p03/p08 class are read against that repeat).
2. Logit cosine >= 0.999 on spot checks.
3. Kernel test (`kernels/test_turbo_rows_q8.mojo`): QA-LDS and QA-PRE bit-identical, both within 1e-5 of sum|terms| of a host f64 reference over the read-back int8 rows (integer block dots are exact; the f32 sum order is not, so "= 0" was not testable). Amended before any measurement.
Only an arm passing 1-3 gets a speed read: bench_turbo_rows m = 1, 2, 3 on gate and down shapes, cold cache,
against the shipped kernel (80.8 / 82.7 us at m = 3). Kill if m = 3 is not faster than shipped on both shapes.

## Result (2026-11-16): KILLED

Agreement (9B q4 pack, `bench/force-ab.sh`, `BARO_MEGA=0 BARO_EMBD_HOST=1`, 20 prompts x 64 forced ids, `.work/roundA/`):
cand `BARO_TURBO_Q8=1` 1271/1280 = 99.30% (8 prompts lose 1-2 ids); same-arm repeat q0 vs q0 1280/1280 = 100%.
Item 1 passes; item 2 (logit cosine) UNVERIFIED, no cosine tool on this path. Item 3 passed (`test_turbo_rows_q8`).

Speed (`bench_turbo_rows`, 9B q4 pack, 8 tensors, 11 rounds, median us; the 27B pack is gone after the ~/Models trim):

| shape | m | shipped TURBO | Q8LDS | Q8PRE (incl. quantize launch) |
|---|---|---|---|---|
| ffn_gate 17408x5120 | 1 | 62.6 | 67.7 | 65.3 |
| ffn_gate | 3 | **89.2** | 119.8 | 106.0 |
| ffn_down 5120x17408 | 1 | 64.3 | 73.4 | 67.3 |
| ffn_down | 3 | **88.6** | 125.1 | 106.6 |

Kill rule (m = 3 not faster on both shapes) fires for both arms. PRE beats LDS everywhere: the per-block
in-stage quantize (32-wide absmax + divide per row, every block) costs more than one extra launch.
Mechanism probe (`tools/isa-receipt.py` + `isa-hot` + `isa-loops`, `.work/roundA/isa/`):
- VGPR suspect REFUTED: amar_turbo_rows_q8 75-78 VGPRs vs amar_turbo_rows 104-155. More waves per SIMD, still slower.
- LDS arm hot loop ~1100-1300 instr (~600 extra VALU per chunk: the in-stage absmax/divide/round): explains LDS < PRE.
- PRE hot loop 157-368 instr, 24-72 v_dot4, VALU 58-168: VALU no longer binds, yet it loses. Per loop 12-36 s_waitcnt,
  19-61 s_delay_alu, 0-3 VOPD dual vs 37-165 in the shipped loop: each row's ds_load -> wait -> 4-deep dot chain runs
  serial. Candidate (not yet measured): latency-bound dependent sudot4 chains with LDS loads not batched ahead.
  If reopened: load all MR rows' A words for a block before the first dot, split dlo/dhi into independent chains.
