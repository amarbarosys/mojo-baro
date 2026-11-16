# Few-row q4 GEMV for the verify window: race report (2026-11-15)

Plan: `~/.claude/plans/snoopy-mapping-codd.md`. Shapes: Qwythos-9B q4 pack, ffn gate 12288x4096 and down 4096x12288, real blk.0..7
weights rotated over 8 tensors, 11 rounds of 8 launches, medians (`bench/bench_turbo_rows.mojo -D BARO_H=4096 -D BARO_FFN=12288`).
Predictions were NOT frozen before the race (miss recorded in `docs/rules.md`); the plan's falsifiers are applied after the fact.

## Step 0 (on main)
- `-D BARO_TURBO=1` became the default (`3ba6c8b`): k=2 154.80 -> 168.88 tok/s_gen, 1.091x, ids 20/20.
- Turbo UNROLL 4: +8% per GEMV in the bench, 0.988 in the engine (20 prompts): not wired (`e3a6dd7`). MR 4 / UNROLL 4 mismatches row 3 (open).

## Arm B: `amar_fewrow_wmma` (kernels/fewrow.mojo)
16 weight rows per block, 8 waves split K, activations as the WMMA A operand (m <= 16 rows, zero rows past m), q4 nibbles as
unsigned bf16 (exact, built by truncating the f32 bit pattern), the -8 offset through one WMMA against a ones vector, row scale per
lane (no shuffles), fixed-order LDS reduce over the 8 waves. fp64 host check: max rel 3.5e-5 (gate), 1.05e-4 (down).

| arm | gate us | down us |
|---|---|---|
| q4rowb m=1 UNROLL 2 (launch path) | 48.2 | 64.9 |
| turbo m=3 UNROLL 2 (window default) | 49.3 | 53.9 |
| WMMA v1 (shuffled scales, f32->bf16 rounding) m=3 D=4 | 117.2 | 137.0 |
| WMMA v2 m=1 / m=3 / m=8, D=4 | 61.3 / 61.9 / 63.6 | 81.1 / 83.1 / 85.2 |
| WMMA v2 m=1 / m=3 / m=8, D=8 (225-234 VGPR) | 60.0 / 60.4 / 60.9 | 72.0 / 76.5 / 76.5 |

Verdict: falsifier "WMMA m=3 within 1.10x of q4rowb m=1" FAILED (1.25x gate). Not wired. The curve is flat in m (m=8 = m=1), so it is
the candidate for windows of 8+ rows (k >= 7, `--parallel` batches), not k=2. Next lever if reopened: lanes 16-31 load a second block
and exchange halves with `permlanex16` (today both halves load the same 16 B, half the bytes per load instruction).

## Arm A (bit-exact)
Not built: its depth lever (UNROLL 4) already lost in the engine; K-split needs the m=1 reduction order and was left out.

## Merge receipts
- run-tests.sh under gpu-wait: PASS run-tests, FAIL: 0 (none), rc=0, pack .work/engine-pack-q4.
- wavewait: NOT CAPTURED. amar_fewrow_wmma is on no engine path (bench only); bench/ww.sh captures engine kernels. Capture when it gets wired.
