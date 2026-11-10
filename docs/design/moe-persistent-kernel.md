# R6 design: the qwen35moe persistent token kernel (2026-09-15, not built)

Preregistration and prediction: `bench/moe-persist-protocol.md` R6 (1216
launches -> 1, gaps 4.0 -> under 0.5 ms, about 130 tok/s). Opened only if
R4's result confirms the gap accounting (that file's falsifier).

## What the launch path does per MoE layer today (from `serve/window.mojo`)

SSM layer (30 of 40): rmsnorm+cast -> three q8_0 projections
(`moe_matmul_q8_0_m1` over raw 34-byte blocks: qkv 17.8 MB, gate 8.9,
later out 8.9) -> `r_qf`/`r_kv` reshapes -> conv, l2 norm, reduce_gates,
delta_step, gated_out -> out projection -> residual add; then the MoE FFN:
widen -> router GEMV (256 x 2048 f32, skinny m1 kernel) -> top-8 (one wave)
-> gathered gate+up (q4_k, 8 x 512 rows) -> cast -> down (q4_k, 2048 cols;
q6_k on layers 34, 38, 39) -> sigmoid gate (one wave) -> shared gate+up
(q8_0, 512 rows) -> cast -> shared down (q8_0) -> add3. Attention layer (10
of 40): rmsnorm -> q/k/v q8_0 projections -> split, head norms, rope, KV
append, attention, out projection, residual, then the same MoE FFN. About
30 launches per layer, 1216 per token, median gap 3.1 us.

## The persistent kernel

`amar_mega_token` shape (`kernels/mega.mojo`): G = 96 blocks, bounded grid
barrier with fail word, one launch per token, phases separated by barriers,
block-strided work distribution, weights addressed from the `off` table.
The MoE profile needs its own body (`mega_body_moe`) because its layer
weight order is the lexical MoE table (`moe_base`, `routed_base`,
`extra_base` in `window.mojo`), its projections are raw q8_0 blocks, and
its FFN is the expert path. Phase list per layer, barrier after each unless
noted:

1. norm + cast to bf16 into LDS (`stage_rms`, no barrier: every block
   computes the full row, as the dense kernel does).
2. projections: q8_0 row dots (`q8_0_row_dot` as landed in R2, 16 B per
   lane) block-strided over the output rows; SSM layers write qkv, gate;
   attention layers write q, k, v. Barrier.
3. SSM: conv, l2, reduce_gates, delta, gated_out as the dense kernel's
   `ssm_phases` already fuse them (same kernels, MoE dims: H = 2048, the
   conv/state widths from `ssm.mojo`); attention: split, head norms, rope,
   KV append, attention, as `attn_phases`. Barrier(s) as in the dense body.
4. out projection (q8_0) + residual into X. Barrier.
5. router: every block computes the 256 logits? No: block 0 computes the
   router GEMV rows it owns and top-8 in one wave; cheaper: block-strided
   router rows (256 rows over 96 blocks, one wave each) into an LDS-free
   global logits row, barrier, then block 0 wave 0 runs the R3 top-8 into
   global idx/wt. Barrier.
6. gathered gate+up over 8 x 512 rows (4096 waves of work over 96 x 16
   resident waves: each wave loops rows, R1 dot; the R5b lesson: one row per
   wave, no dual dot), silu*up written as bf16; shared-expert gate+up (512
   rows, q8_0) in the same phase by block-striding both work lists; sigmoid
   gate by one wave in the same phase. Barrier.
7. down: 2048 output columns x 8 experts (the R5 pair dot, 17.7 us receipt,
   is the natural inner loop here), plus shared down (q8_0) in the same
   phase; add routed + sigmoid * shared + residual into X. Barrier.
Per layer: about 7 barriers; per token 280 plus the head, at the measured
0.6 to 1.1 us per barrier (`bench_gridbar`) = 0.2 to 0.3 ms against the 4.0
ms of launch gaps today.

## Residency and registers

Block of 512 threads (16 waves) as the dense kernel; VGPR budget under 192
without the `flat_work_group_size` lift, or declare it and accept fewer
waves per SIMD: the ISA receipt decides the ceiling before launch
(`persistent-kernel-gfx11`: ceiling from the receipt, never launch past
it). The q4_k block partial holds 4 x 16 f32 temporaries; the q8_0 dot 16;
LDS: the bf16 activation row (4 KB at H = 2048) plus the R1 header path.

## Gates (from the protocol) and the extra one this kernel needs

`test_moe_block`-style parity for the whole layer against the launch path
(`BARO_MEGA=0`): identity of X after each layer on one prompt (the dense
kernel's `BARO_DUMP` compare), then the 20-prompt agreement band, the
20-prompt tok/s A/B with clocks, `isa-loops` fingerprint recorded, the
NOT-RESIDENT fail word read on every run. Kill line +5%; the prediction is
+25 to +30% (about 130 tok/s).

## Effort

L: `mega_body_moe` about 250 lines reusing `ssm_phases`/`attn_phases`/
`stage_rms` and the R1/R2/R5 dots; registry dispatch and `window.mojo`
switch about 40 lines (the lane's files, by request); GPU minutes per gate.

## Correction after reading the MoE launch path in full (2026-09-15, later)

The MoE SSM block is not the dense one and `ssm_phases` cannot be reused:
- inner width `NH_V * SSTATE = 4096` against `H = 2048` (the 2026-09-12 W3
  fix), so qkv is `CONV x H` with CONV sized for the 4096 inner, gate is
  `4096 x H`, and `ssm_out` is `H x 4096` (`decode_base + 9`);
- the A and B gates are f32 tensors (`moe_base + 3`, `+ 4`, `ssm_gate_w_layout`)
  consumed by their own GEMV kernel, not q8 rows as in the dense body;
- conv, l2, reduce_gates, delta (`NH_V = 32`, `SSTATE = 128`) and gated_out
  keep the dense shapes and can be reused;
- the launch path also switched to the dense q8 row kernel for the
  projections ran 8% slower than the raw-block dot (R6a), so the
  projection phase must carry the raw-block dot or prove the dense one is
  not slower inside the persistent kernel.

Revised effort: **XL** (own `ssm_phases_moe`, own projection phases with the
raw-block dot, FFN expert phases, MoE offset walk, dump parity per layer).
Staging that keeps every step a short experiment:
1. R6.0 (M, host side, no grid barrier): fuse the per-layer elementwise
   chains on the launch path (norm + cast, casts, add3, the small SSM
   kernels) into 2 to 3 launches per layer; measured by the launch counter
   (`bench/moe-launch-count.sh`) and the 20-prompt A/B. Prediction from the
   timeline: about 400 launches removed, 1.3 ms, +12%.
2. R6.1 (L): persistent kernel for the attention layers' sub-blocks only is
   not possible (one launch per token or nothing), so the first kernel
   milestone is the full body at parity on one prompt with `BARO_DUMP`
   layer compares, speed not gated.
3. R6.2 (M): speed gates as preregistered.
Decision on opening the XL lane is the maintainer's (`docs/NEXT-PLAN.md` GPU rule:
each gate under 10 minutes still holds; the cost is engineering sessions,
not GPU).
