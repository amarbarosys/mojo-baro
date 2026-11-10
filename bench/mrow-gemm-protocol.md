# Multi-row window protocol — frozen before first run

> Binds `bench/PROTOCOL-RULES.md` P1-P6. Frozen 2026-09-04 on `41d0361`
> (+ the rules commit). Nothing below was measured before it was written
> except the two facts marked (measured).

## Problem on the record (measured)

- A 1-row decode pass costs ~15 ms; a 2-row speculative window ~1.28x that;
  a 4-row prefill window 43 ms = 2.9x (`.work/mr2-A.log`, `41d0361`).
  Bandwidth says 1.05-1.1x: the weights are read once either way.
- Real-prompt MTP at k=2 is 100.7 median (1.47x); llama.cpp 123.5 (1.66x).
  The gap is the window cost, not acceptance (ours 69%, theirs 58%).

## Diagnosis to be confirmed by stage M0, not assumed

`amar_matmul_skinny_q8row[UNROLL, MR]` streams weights once per wave-row but
loads the A slice per row per step from global (L2): A traffic per weight
byte = 2 x MR (bf16 A vs int8 W). At MR = 4 that is 8x the weight stream
through L2. Second suspect: the SSM sub-block runs its 5 kernels per row per
layer (24 layers x 5 x (m-1) extra launches per window).

## Stages

- **M0 receipt (lane B, first).** `bench/bench_coldcache_mrow.mojo`: the
  current q8row at MR = M for M in 1, 2, 4, 8 on the ffn shape (N=12288,
  K=4096), cold-cache 8-buffer rotation as in `bench_coldcache_q8row.mojo`,
  fp64 host check on every row. Prints us and the M-row/1-row ratio. This is
  the P5 receipt and the baseline for M1.
- **M1 kernel (lane A).** `amar_matmul_skinny_q8mrow[MR]`: block stages its
  A rows (MR x K bf16, <= 64 KB at MR = 8) into LDS once, waves stream their
  weight rows from global as q8row does, A comes from LDS. Same fp32 accumulate
  and reduction order per row as q8row so the m=1 path and the engine tokens
  stay bit-identical; gate = M0 bench parity (fp64 host) and engine 64/64.
- **M2 SSM fold (lane C).** The five SSM kernels take `m` and loop rows
  inside (recurrence stays sequential per row, same op order); engine dispatch
  drops the per-row loop for the SSM sub-block. Bit-exact by construction;
  gate = `test_ssm_block` and engine 64/64 on the 20-prompt set.
- **M3 engine (driver).** Registry dispatch to q8mrow for m > 1; MTP sweep
  and race per P4.

## Frozen predictions

| stage | prediction | land rule |
|---|---|---|
| M0 ratio today (m=4 / m=1) | 2.4-3.2x | recorded |
| M1 ratio (m=2, m=4, m=8) | <= 1.15x, <= 1.25x, <= 1.5x | m=4 <= 1.3x AND parity |
| M2 | ssm sub-block per extra row -0.5 to -0.8 ms | 64/64 all 20 prompts |
| M3 real prompts k=2 median | 100.7 -> **112-122** | >= 110 AND 100/100 identity |
| M3 race prompt k=4 | 145.6 -> **165-185** | recorded |
| M3 vs llama.cpp MTP real text | 0.78x -> 0.90-1.0x | recorded |
| falsifier | M1 m=4 ratio > 1.6x after LDS staging: the A-traffic diagnosis is wrong; stop, profile with rocprof, re-preregister | |

## Result

### M0 (2026-09-04, lane B `4af9e5c`, driver re-run under gpu.lock)

`flock .work/gpu.lock ./.work/bench_coldcache_mrow`, grid 1536 x 256,
ROW_WAVES=8, UNROLL=4, ffn_gate N=12288 K=4096, ITERS=200, fp64 parity on
every row: correct=true for all four arms.

| MR | us | GB/s (weight bytes) | ratio to MR=1 |
|---|---|---|---|
| 1 | 88.3 | 605 | 1.00 |
| 2 | 92.2 | 580 | 1.04 |
| 4 | 112.9 | 473 | 1.28 |
| 8 | 201.2 | 266 | 2.28 |

**Frozen m=4 prediction 2.4-3.2x was wrong: measured 1.28x.** The 2.9x
4-row prefill window and the 1.28x 2-row speculative window are therefore
NOT the GEMM at m=2 (1.04x here). The A-traffic diagnosis is falsified for
m <= 4; it holds only toward m=8. Lane A's LDS-staged q8mrow (`d7f765c`,
branch `lane-mrow-kernel`): m=4 1.29x, m=8 2.08x, bit-identical outputs;
no gain at the window sizes the engine uses, not wired into the engine.
Stage M1 is re-scoped to an ablation (traffic vs ALU vs occupancy at m=4/8)
before any second kernel attempt; M2 (SSM per-row fold) is now the lead
lever for the 2-row window.

### Per-window cost split (2026-09-04, driver, race prompt, BARO_PROFILE=1)

1-row decode window vs 2-row speculative window (k=1), ms per window,
profile syncs inflate absolutes, deltas are the signal:

| block | 1-row | 2-row | delta |
|---|---|---|---|
| attn | 1.55 | 1.76 | +0.21 |
| ssm | 4.76 | 5.82 | +1.06 |
| ffn | 8.41 | 9.34 | +0.93 |
| head | 1.54 | 1.61 | +0.07 |
| draft path (process + draft) | 0 | 2.22 | +2.22 |

Sum +4.4 ms on 16.3 = 1.27x, matching the measured window cost. Lever order
by size: draft path 2.2 (blk.32 layer + 1.06 GB q8 LM head + argmax + host
sync per window), SSM per-row 1.06 (M2, lane C), ffn 0.93 (GEMM m=2 is
1.04x = 0.34 of it; the rest is in the SPLITK reduce / swiglu at m rows,
unmeasured), attn 0.21. Draft-path split and the q4 draft head are the
next preregistered items; draft quantization cannot change output tokens,
only acceptance, so it is not gated by bit-exactness.

### M1 ablation (2026-09-04, lane A stint 2, `.work/briefs/status-mrowA.md`)

Original q8row, cold-cache ffn shape, ratio to FULL MR=1 (70.5 us):
FULL m=4 1.46x, m=8 2.71x. NOTRAFFIC (all rows read A row 0): 1.38x /
2.53x. NOFMA (all rows load A, only row 0 accumulates): **1.06x / 1.11x**.
OCC (ROW_WAVES=4): 1.42x / 2.69x. VGPR 89-133 across variants, no spills.
**The multi-row cost is FMA count, not traffic and not occupancy.** The
kernel is compute-bound above m=2: per weight element it does one dequant
and MR fp32 SIMD FMAs on the bf16 activations. LDS staging (stint 1) could
not help and did not. Lever, preregistered as M1b below: packed int8 dot
products (activations quantized to q8 per 32-block at window start, weights
already q8, `v_dot4_i32_i8`-class math, 4 MACs/lane/op) for m >= 3 only;
m <= 2 keeps the bf16 path and its bit-exactness.

### M2 (2026-09-04, lane C `871faca`, merged `9808a9d`)

Five SSM kernels take `m`, row loop inside; `amar_ssm_delta_step` is
VGPR-spill-bound (192 VGPR, 122 spills at MR=1; a runtime row loop made it
288) so it is instantiated per MR via `delta_dispatch`. `test_ssm_block`
m=1/m=4 bit-exact PASS. p09 k=2: 77.6 -> 78.4 (+1%), `profile: ssm` 0.273
-> 0.267 s. Small, as the split predicted (1.06 ms of which launches were a
fraction). The 122-spill delta step at m=1 is a standing defect of the
decode path itself: `bench/ssm-occupancy-protocol.md` was frozen for it
and never run; lane C stint 2 runs it.

### Draft-path split (2026-09-04, lane B stint 2, `BARO_PROFILE=3`, merged `03f6a13`)

Per 2-row window, race prompt k=1: layer 0.61, **head 1.51**, argmax 0.21,
accept (copies + sync + compare) 0.04, other 0.02 = 2.40 ms. p09 k=1 the
same within 0.1. k=2 doubles everything except accept: draft steps are
one row each, never batched. The host sync is 2% and not a lever. The
draft LM head (1.06 GB q8 read per drafted token) is 63% of the draft path.

## M1b, M4 frozen (2026-09-04, after the ablation, before any run)

| stage | prediction | land rule |
|---|---|---|
| M1b q8 x q8 dot kernel, cold-cache ffn shape | m=4 <= 1.15x, m=8 <= 1.4x of bf16 m=1 | parity vs fp64 on dequantized q8 activations <= 2e-3 max_rel; engine 64/64 race AND >= 18/20 identity vs arm A on the prompt set with m>=3 windows only |
| M1b engine effect | prefill 4-row 43 ms -> <= 25 ms; k=4 race 146 -> >= 160 | recorded |
| M4 q4 draft head (`bench/draft-q4-protocol.md`) | draft path 2.4 -> <= 1.7 ms/window; real-prompt k=2 median 100.7 -> **108-115**; acceptance within 3 points of q8 draft | >= 106 AND acceptance >= 66% |
| falsifier M1b | m=4 > 1.3x: int8 dot path does not beat fp32 FMA on this card; stop | |
| falsifier M4 | acceptance drops > 5 points: q4 draft too lossy, try q6/q5 pack before giving up | |

## Item 2 diagnosis — ffn stage split, frozen before the run (2026-09-05)

Milestone item 2 asks where the +0.93 ms/window the ffn sub-block costs at m=2
goes, given the q8row GEMM's own row-scaling ratio is only 1.04x. Item 1
already removed the SSM from suspicion (1.15x/window for twice the rows,
`bench/ssm-mrow-protocol.md`).

Instrument: new `BARO_PROFILE=4` in `serve/engine.mojo`, the exact shape of the
existing `BARO_PROFILE=2` SSM split, over the six ffn stages: `rmsnorm`,
`gemm_gate`, `gemm_up`, `swiglu`, `gemm_down`, `r_add`. Serialized by
`ctx.synchronize()`, so stage sums exceed the true sub-block time and no
`tok/s` from a profile-4 run may be quoted. Arms: A `BARO_SPEC=0` (m=1, 63
windows), B `BARO_SPEC=1 BARO_SPEC_K=1` (m=2, 32 windows), 3 runs each, first
dropped, median of 2, spread gate <5%.

Predictions, frozen before the run:

1. The three `gemm_*` stages have B/A <= 1.15 each. They are weight-bound and
   the weights are read once per window regardless of m.
2. `swiglu` has B/A >= 1.7. Its grid is `ceildiv(m * FFN, 256)`, so m=2 doubles
   both the element count and the P-buffer traffic (2 x m x 12288 f32 read,
   m x 12288 bf16 written).
3. The largest single absolute increase (ms/window) comes from `swiglu`, not
   from any GEMM.
4. `rmsnorm` and `r_add` together add < 0.10 ms/window.

If prediction 3 holds, item 2's fix is a fused swiglu that consumes the two
split-K partials directly instead of round-tripping `Pg`/`Pu` through memory,
and the ceiling stays 115-120. If instead the GEMMs carry the increase, the
1.04x kernel-level row scaling measured in M0 does not survive in the engine
and item 2 is re-scoped to that discrepancy before any kernel is written.

### Run record and verdict (2026-09-05, HEAD 99ca57a)

Instrument as frozen. Each arm rebuilt the engine in the same command and
printed `build exit: 0` + sha256 (`6e604da2…` across all six runs). Arm
identity read back per file: three `BARO_SPEC: False`, three `BARO_SPEC: True`
with `accepted 32`. Pack: `.work/engine-pack-q8d` (`.work/engine-pack-q8` had
been removed from `.work/` by then; both arms used the same pack and
`BARO_DRAFT_Q4` was unset, so the draft head is q8 in both). GPU exclusive.
`gemm_gate` kept runs: A 172.06 / 186.00 ms, B 101.45 / 98.63 ms.

| stage | A (m=1) ms/win | B (m=2) ms/win | B/A | Δ ms/win |
|---|---|---|---|---|
| rmsnorm | 0.8077 | 0.8218 | 1.02 | +0.014 |
| gemm_gate | 2.8417 | 3.1264 | 1.10 | **+0.285** |
| gemm_up | 2.8853 | 3.0516 | 1.06 | **+0.166** |
| swiglu | 0.5753 | 0.5947 | 1.03 | +0.019 |
| gemm_down | 2.9882 | 3.3191 | 1.11 | **+0.331** |
| r_add | 0.5721 | 0.5858 | 1.02 | +0.014 |
| total | 10.6704 | 11.4993 | 1.08 | +0.829 |

**Item 2's premise is falsified.** The three GEMMs carry **+0.78 ms of the
+0.83 ms** — 94% of the ffn window loss. `swiglu` scales 1.03, not the >= 1.7
predicted, and is the third-smallest absolute contributor; predictions 2 and 3
are both wrong. Predictions 1 and 4 hold: no GEMM exceeds 1.15, and
rmsnorm + r_add together add 0.028 ms.

The +0.93 ms this protocol attributed to "per-row elementwise, launches,
staging" is none of those things. It is the q8row GEMM's own row scaling,
which is 1.06-1.11 in the engine against the 1.04 measured cold-cache at the
same ffn shape in M0. Small per call, but there are three of them per layer.

Consequences, by the rule frozen above:

- The planned fix (a fused swiglu consuming the split-K partials directly)
  would target 0.019 ms/window. It is not worth writing.
- **The 115-120 tok/s ceiling for item 2 is not supported** and should not be
  quoted. Item 1 cleared the SSM, this clears the ffn elementwise; what is left
  is GEMM row scaling, and no window rework moves it.
- Item 2 is re-scoped to one question before any kernel is written: why does
  `amar_matmul_skinny_q8row[4, 2]` cost 1.06-1.11x its m=1 self in the engine
  when the same kernel at the same shape costs 1.04x in the cold-cache bench?
  Candidates: the bench's NBUF=8 rotation is a different cache regime from a
  weight streamed once per window; the engine's A operand is 2 rows of a live
  activation rather than a fixed fixture. That is a bench-vs-engine
  reconciliation, not an optimisation.

### Correction after the diagnostic audit (2026-09-05, `ee30176`)

The measurement above stands; **the verdict I drew from it does not.** Two
errors, both mine:

1. **The "1.06-1.11x in the engine vs 1.04x in the bench" discrepancy was
   never established.** M0's own repeat pairs, in `.work/briefs/status-mrowB.md`
   lines 33 and 40-41, are 88.370/91.551 (ratio 1.036) and 80.791/91.501
   (ratio **1.133**): the m=2 time is stable while the m=1 denominator moves
   8 us between repeats. 1.04 is one draw from that pair, not a kernel
   constant, and the engine's 1.06-1.11 sits inside the range. I compared a
   number against a baseline that had already contradicted itself in the same
   file.
2. **`gemm_down` is not the same shape as gate/up** — N=4096/K=12288 against
   N=12288/K=4096 - so quoting its 1.11 beside the others as one "row scaling"
   figure compares two shapes as if they were one.

Also wrong in framing: I reported "+0.83 ms/window" as a loss against an
implicit 1.0x ideal. Per useful row the ffn sub-block at m=2 costs 0.54x its
m=1 self. M0's own per-row series is 88.3 / 46.1 / 28.2 / 25.2 us for
m = 1/2/4/8: weight amortisation already wins. **Multi-row batching is not
falsified.** What limits it is that speculative decode must minimise time per
*accepted* token, so rejected rows can erase the throughput gain - a width
constraint, not a dead direction.

What survives from the run: the ffn elementwise stages are not where the m=2
window cost goes (`swiglu` 1.03, rmsnorm + r_add +0.028 ms/window together),
so the fused-swiglu idea remains not worth writing. Everything I said about a
bench-vs-engine kernel discrepancy, and the claim that the 115-120 tok/s
ceiling is unsupported, is **withdrawn**: neither was established by this
measurement.

The discriminating measurement, per the audit: interleaved m=1/m=2 pairs from
one binary, engine layouts, all three FFN shapes, device timestamps beside host
wall time, with clocks and spread recorded.

## M3-dot frozen (2026-09-05, before any run; code + this block in one pre-run commit)

Scope: trunk FFN gate/up/down only, opt-in `BARO_DOT=1`, windows with m >= 3
(k=2 spec windows are m=3, race k=4 windows m=5, prefill m up to 8). Draft
layer, attn/qkv/z/head GEMMs stay q8row. Engine prints `BARO_DOT:` (P1
read-back). Cost added per SSM/attn layer at m >= 3: two
`amar_quantize_q8_rows` launches (m x K bf16 read, m x K int8 + m x K/32 f16
written -- noise next to 42 MB of weights).

Baseline to beat: 20-prompt k=2 `BARO_DRAFT_Q4=1` median **104.18**
(`draft-q4-protocol.md`); race k=4 145.6; both on `.work/engine-pack-q8d`.

| item | prediction | land rule |
|---|---|---|
| race prompt k=4, dot=1 vs dot=0 | 145.6 -> 152-165 (m=5 windows; bench MR=4: dot 1.14x vs bf16 1.25-1.39x, MR=8: 1.45x vs 2.3-2.46x) | `GENERATED` identical to dot=0 arm on race prompt |
| 20-prompt k=2 median, dot=1 | 104.18 -> 103-108 (weak: dot MR=2 was par with bf16; m=3 interpolates a small win) | >= 104.2 AND >= 18/20 per-prompt identity vs arm A |
| prefill_s, longest prompt (59 tok) | drops >= 20% (prefill windows m=8) | recorded |
| acceptance | within 2 points of dot=0 (trunk logits move at m>=3; drafts unchanged) | recorded |

Falsifier: k=2 median < 104.18 AND race < 152 -> FFN-only dot wiring loses to
its quantize launches at engine shapes; `BARO_DOT` stays default 0, item
closed, no further dot wiring without a new preregistration.

Non-bit-exact change (int8 dot replaces bf16 FMA on m>=3 trunk windows):
identity gates above are the guard; arm A (m=1) is untouched by construction.

### M3-dot run record and verdict (2026-09-05, `b892a25`, .work/dot-race5-*, .work/mtp-dot{0,1})

Arms on `.work/engine-pack-q8d`, `BARO_DRAFT_Q4=1`, `BARO_DOT:` read back in
every log. Race: 5 interleaved pairs, k=4. Sweep: `bench/mtp-prompts.sh`, k=2,
20 prompts, arm-per-flag.

| item | prediction | measured | verdict |
|---|---|---|---|
| race k=4 median | 152-165 | dot=1 **158.1** (dot=0 arm: 147.3; +7.3%), identity 5/5 | HOLDS |
| 20-prompt k=2 median | 103-108, land >= 104.2 AND >= 18/20 | **102.93** (dot=0 arm: 103.68), identity **16/20** (p04/p09/p13/p20) | **LAND RULE FAILS, both legs** |
| prefill_s longest (p17, 59 tok) | drops >= 20% | 0.305 -> 0.241, **-21%** | HOLDS |
| acceptance | within 2 points | 0.6918 -> 0.6860 | HOLDS |

Falsifier (k=2 < 104.18 AND race < 152) does NOT fire: the int8-dot path is
real at m >= 5 and in prefill, but at the shipping config (k=2, spec windows
m=3) its GEMM win does not cover two quantize launches per layer plus the
identity cost of non-bit-exact m=3 verify windows.

**Verdict: `BARO_DOT` stays default 0.** Code remains as a preregistered
opt-in. If a k >= 3 config or prefill wall time ever matters, the next
preregistration should test threshold m >= 4 (k=2 spec windows back to
bit-exact bf16, prefill/race keep the dot win); not run now because the
shipping config gains nothing from it.
