# Split-K on the N=4096 phases (frozen 2026-09-06, after the q4 ALU round, before any change)

Follows `bench/q4-alu-protocol.md` (closed at 118.3 no-spec, `784caa5`). Pool named there: the three
residual-add GEMM phases with N=4096 output rows (ssm out K=4096, attn o K=4096, ffn down K=12288) =
2.35 ms of the 8.33 ms token, streaming at 0.7-0.75x the wide phases' rate because 4096 rows over
1536 wave-slots (96 blocks x 16 waves) is 2.67 rows per wave: a third of the waves run a third row
while the rest wait at the barrier.

## Design (both paths change, identity by construction)

- Work item = (row, half): 8192 items per phase, 5.33 per wave, K split in two halves of 32-blocks
  (64+64 for K=4096, 192+192 for K=12288). Item order is row-major with the two halves adjacent, so
  both halves of a row land in the same block, in waves 2k and 2k+1.
- Megakernel (`split_gemm_add`, Q4 only): half 0 writes its partial p0 to `Pk[row]` and publishes a
  per-row flag (release, tag = barrier generation + 1); half 1 waits on the flag (acquire, bounded spin
  + fail word like `grid_barrier`), clears it, and writes `X = X + (p0 + t1)`. No extra grid barrier.
- Launch path: `amar_matmul_skinny_q4rowb[UNROLL, MR, KSPLIT=2]` writes `Cp[part, r, row]`;
  `amar_skinny_reduce_add[.., NSPLIT=2]` computes `acc = (0 + p0) + p1; X = X + acc`. Same real
  expression `X + (p0 + p1)`, same lane-to-block mapping (one shared inline dot body
  `q4_dot_blocks` over a block range), so the bits agree.
- q8 packs untouched (Q4=False keeps the unsplit loop; q8 launch unchanged).

## Stages

| stage | check |
|---|---|
| K1 kernel | `test_mega_block` m=1 q4 bit-identical (its launch emulation uses the split kernels too); q8 cases unchanged |
| K2 engine | q4 pack mega == launch spec 0/1; launch vs `tools/model-ref.py` 64/64 (the numerics moved: the token stream is the judge); run-tests green |
| K3 A/B | 20 prompts, old binary (`.work/engine-a2`) vs new, one stint, clock-probe; identity 20/20; per-phase profile |

## Frozen prediction

Pool phases at 0.72 -> 0.92 of the wide rate: ssm out 583 -> 460, attn o 162 -> 130, ffn down 1600 ->
1280 us = -475 us on 8331 = **118.3 -> 125-127 no-spec (+6%)**. Costs counted against it: the flag
handshake and the second X read (negligible), the half-1 waves' spin (short: both halves start
together). Land >= 123 (+4%); close < 121 (+2%). Stop rules: any K1 mismatch = the two paths' block
mapping differs, fix the mapping, never the check; K3 spread > 5% = re-run, not a verdict.

## Result (2026-09-06; receipts `results/q4-splitk/`, split build kept at `.work/engine-splitk`, code reverted)

| stage | receipt |
|---|---|
| K1 | PASS: `test_mega_block` m=1 q4 bit-identical with the split on both paths (0 mismatches); q8 cases unchanged |
| K2 | PASS: q4 pack mega == launch spec 0/1; launch vs model-ref 64/64; run-tests green; VGPR 256, spills 77 -> 85 |
| K3 no-spec | same stint, 3042 MHz med: **a2 119.03 (spread 4.8%) -> split 108.10 (2.8%), 0.908x** (identity vs a2 differs on 3/20 prompts, expected: the numerics moved; mega == launch stays 20/20) |
| K3 k=2 | 149.76 -> 150.88 (1.007x) |
| per-phase, same stint, 2 runs | ssm out 588 -> 757 (+29%), attn o 156 -> 227 (+45%), ffn down 1607 -> 1897 (+18%); ssm in-GEMM (unsplit) 1071 -> 1264 (+18%); **delta 395 -> 219 (-45%, code unchanged, unexplained)**; token 8320 -> 9114 us |

**Verdict: CLOSED, below the close line (121) - the split loses 9%. Reverted** (the shared inline dot body
`q4_dot_blocks` it needed is kept, `902d081`, identity by construction, perf unchanged at 119.6/120.3).

Why the prediction was wrong: halving K per item halves the memory depth of each wave. A half-row of
K=4096 is 64 blocks = exactly one UNROLL-2 iteration per lane: two 16-byte loads in flight, then a
16-wide reduce + 5 shuffle steps + the flag handshake, per item. The old row had two iterations of
pipelined loads. The N=4096 phases were never limited by the 2.67-rows-per-wave tail alone (that is an
11% effect: blocks 0-63 run 3 groups, 64-95 run 2); the rest of their 0.72x is per-phase fixed cost
(barrier + ramp) on a 24-50 us phase, which finer items make worse, not better. The first version of
the handshake with an acquire load was worse still (it invalidates the CU's L0/L1 per row; the block
re-fetches A from L2 thousands of times a phase); the L2-atomic version is what was measured.

Lead left open: the unsplit delta phase ran 219 instead of 395 us in the split build with identical
delta code (delta loop 2257 vs 2521 instructions, spills 3 vs 4 - not a 1.8x). If that state is
reproducible it is worth 2% on its own.

Remaining pools after this round (from the a2 profile, 8.32 ms/token): GEMM phases at 0.85x the q8
phases' rate = 0.7 ms, but the two levers tried (ALU, K-split) are spent and bytes-in-flight needs
registers the 512-thread block does not have; attention (306 us on 16 blocks) + rmsc (260) = 0.25 ms.
A 150-tok/s token is 6.67 ms; the megakernel at m=1 is at 8.32 with ~0.5 ms of credible pool left =
~127. Beyond that the structure has to change (fewer, fatter phases per layer, or a 256-thread block
at G=192 with the VGPR budget the 192-block residency receipt allows), which is a new round, not a tweak.
