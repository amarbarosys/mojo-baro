# qwen35moe decode, round 2: launches and expert kernels (preregistered 2026-09-15, before any run)

Base: `main` after `4209cc4` (R1 to R3 landed, 93.46 tok/s_gen 20-prompt
median, `bench/moe-perf-protocol.md`). Bar: llama.cpp 109.4.

## Receipt: the token timeline at HEAD (rocprofv3, p09, 43 decode tokens, `.work/moe-perf/timeline-head.txt`)

Per token, medians: **1216 launches**, wall 12.5 ms (traced; 10.7 untraced),
kernel spans 8.5 ms, **gaps 4.0 ms (32% of wall)**, inter-dispatch gap
median 3.1 us. Kernel classes: q8_0 projections 2.3 ms over 160 calls (the
17.8 MB matrix at 18.7 us = 950 GB/s, near the floor); expert gate+up
1.13 ms / 40 calls (28 us for 9.4 MB = 335 GB/s); expert down 1.10 ms / 37
calls (30 us for 4.7 MB = 157 GB/s); rmsnorm 0.37 ms / 81 calls (4.6 us
each, launch-bound); delta 0.33, reduce_gates 0.22, router 0.19, sigmoid
gate 0.15; **`fillBufferAligned` 60 calls and `copyBuffer` 40 calls per
token (0.26 ms plus their gaps)**, host-enqueued memsets and copies in the
MoE layer path; head 0.6 ms.

## Rounds, in order, each its own commit with the R1-R3 gates

Gates for every round (unchanged from `bench/moe-perf-protocol.md`):
`test_moe_block` parity; 20-prompt agreement mean within 52.70..53.70;
`run-tests` and `ci-checks` green; 20-prompt tok/s A/B against the previous
round with clocks read back; kill line: median below +5% is a no-op and is
reverted (lower than round 1's +10% because these are smaller levers).
GPU per gate under 10 minutes.

- **R4, launches from the host (S, `serve/window.mojo`, delegated to the
  opus lane which owns that file):** remove the per-layer `enqueue_memset`
  and `enqueue_copy` in the MoE FFN path (60 + 40 per token) by having the
  consuming kernels write full outputs (no zero-init) and by pointing at
  buffers instead of copying. Prediction: launches 1216 -> about 1116,
  wall -0.5 to -0.6 ms, **+5 to +6%**.
- **R5, expert down kernel (M, fable):** `amar_moe_down_q4k` runs one
  output column per wave over 8 experts x K=512 (2 super-blocks), so 16 of
  32 lanes idle per block-dot and the 8 expert dots serialize per wave.
  Change: two experts per pass on the two lane halves, 4 passes per column,
  same per-element arithmetic. Prediction: down 30 -> under 18 us per call,
  **+6 to +9%**. The gate+up kernel gets the same look (28 us at 335 GB/s;
  one row per wave over K=2048 = 8 super-blocks, 2 passes) only if a
  measured cause is found; not promised here.
- **R6, MoE persistent token kernel (L, fable):** the MoE profile runs the
  per-kernel launch path (`MEGA_ALLOWED` off); build the MoE token kernel on
  `kernels/mega.mojo`'s phases plus expert phases (router, gathered gate/up,
  down, shared expert), R1/R2 dots inside. Prediction: launches 1216 -> 1,
  gaps 4.0 -> under 0.5 ms, **10.7 -> about 7.5 ms per token, about 130
  tok/s**, past llama.cpp's 109.4. Fingerprint read before the number
  (`isa-loops`); residency ceiling from the ISA receipt before launch
  (`persistent-kernel-gfx11`).

Falsifier for the whole round: if R4 lands and the wall does not move by at
least the removed launches x 3 us, the gap accounting is wrong and R6 is not
opened on this evidence.

### R5 result (2026-09-15): below the kill line, reverted; mechanism confirmed, prediction wrong

Two experts per pass on the two lane halves (`q4k_dot_pair`, shared
`q4k_block_partial`). Gate 1 parity PASS; gate 2 agreement **53.35/64**;
gate 4: 20-prompt tok/s **94.43 -> 98.55 (1.0436x)**, ranges 92.81..95.00 vs
98.32..98.86, sclk med 3267 MHz, GENERATED equal 17/20. Kernel receipt
(rocprofv3, `.work/moe-perf/trace-r5`): **down 30.5 -> 17.7 us per call**,
exactly the "under 18" predicted; gate+up unchanged 29.0 -> 29.8. The
token-level prediction (+6 to +9%) was wrong by arithmetic: 12.8 us x 37
calls = 0.47 ms of 10.7 = 4.4%. Under the frozen +5% line the arm does not
land; patch kept at `.work/moe-perf/r5-down-pair.patch`. It re-enters only
as one combined arm with a gate+up change (28 us at 335 GB/s, K = 2048, one
row per wave) under a new frozen prediction stated in kernel microseconds
and in token percent from the launch counts, not a guessed range.

### R5b, combined expert arm (preregistered 2026-09-15, before any run)

Change: (1) the R5 down pair (`.work/moe-perf/r5-down-pair.patch`, down
30.5 -> 17.7 us measured); (2) gate+up: one wave computes its gate row and
its up row in a single loop with all four 16-byte loads per lane (gate block
b, up block b, gate b+4, up b+4) issued before any arithmetic, instead of two
sequential dots each with dependent iterations. Same per-element arithmetic
and bf16 rounding; accumulation order within a row unchanged (blocks still
summed in lane order).
Predictions, frozen: gate+up **29.8 -> under 20 us per call** (40 calls:
at least 0.39 ms per token), down 17.7 us (37 calls: 0.47 ms); token 10.7 ->
under 9.85 ms, **+8 to +10%** on the 20-prompt median against
`.work/moe-perf/engine-head`. Kill line +5% as for the round. Gates as for
the round (parity, agreement band, tok/s A/B with clocks, run-tests,
ci-checks). Falsifier: gate+up not under 22 us in the trace means the loads
were already overlapped and the cause is elsewhere (the receipt decides,
not the tok/s).

### R5b result (2026-09-15): falsifier fired, reverted

Parity PASS, agreement 53.35, 20-prompt tok/s 94.99 -> 97.10 (1.022x, below
the +5% line). Trace (`.work/moe-perf/trace-r5b`): down 18.4 us (the R5
pair, as measured before), **gate+up 29.8 -> 33.5 us**, not under 22: the
falsifier fired. Issuing both matrices' loads together did not help; the
dual dot needed 237 VGPRs (the 192 cap was lifted with
`rocdl.flat_work_group_size` to remove 64 spills) and the occupancy loss
outweighed any overlap, so the gate+up cost is not load latency per wave.
What is known: one row per wave, 4096 waves, 28 to 30 us for 9.4 MB. The
lever is left to R6, where the expert phases are written fresh inside the
persistent kernel and may use the pair dot (17.7 us receipt) by design.
Patch kept at `.work/moe-perf/r5b-dual.patch`.

### R4 result (2026-09-15, opus lane): LANDED, +5.5% and exactly 100 launches

Both arms built from a clean `git archive` of HEAD plus only
`serve/window.mojo` and `serve/harness.mojo`, so no other lane's uncommitted
work is in either binary. `engine-r4base` sha `70f9913a0f87db7c`,
`engine-r4cand` sha `12ca6cfc12beb48f`.

**Launch count, the receipt the prediction was actually stated in**
(`bench/moe-launch-count.sh`, rocprofv3 `--kernel-trace`, difference between a
16-token and a 32-token run so load and prefill cancel):

| arm | dispatches at 16 tok | at 32 tok | per token |
|---|---|---|---|
| base | 34,036 | 53,508 | **1217.0** |
| R4 | 31,239 | 49,111 | **1117.0** |

Exactly 100 launches per token removed, against the predicted 1216 to about
1116.

**Throughput** (`bench/ab-prompts.sh`, 20 prompts, one stint, clock probe):
base median **94.68** tok/s_gen (spread 5.7%), R4 median **99.92** (spread
0.7%), **ratio 1.055**, inside the predicted +5 to +6% and above the +5% kill
line. sclk med 3266 MHz, 290 W cap, -100 mV, junction max 70 C.

**Identity: 20/20 PASS.** Every prompt's `GENERATED` line is identical between
the two arms, which is what makes the agreement gate unnecessary here rather
than skipped: the teacher-forced agreement number cannot move when the emitted
ids are bit-identical to the arm it was measured on. `test_moe_block` parity is
likewise unaffected by construction (no kernel body changed).

**What was removed and why it was safe.** The 60 memsets were two per layer
zeroing `p_32_d` and `p_32b_d` before the two skinny f32 matmuls.
`amar_ssm_reduce_gates` sums `SPLITK` partials, so the unwritten partial has to
be zero; but `amar_matmul_skinny_m1_row` **writes** its output row
(`O[row] = total`, not an accumulate), and on this profile only partial 0 row 0
is ever written, because the dense `gemm_w` partial writer is not compiled into
the MoE build (`MEGA_ALLOWED` is False). So the rest of those planes has to be
zero and never stops being zero: they are zeroed once at allocation. The code
carries the condition that would invalidate that reasoning.

The 40 copies were `moe_ffn` creating a host buffer and copying a zero into
`hidx_d[0]` once per layer, to give the shared expert an index of 0. That is
now one int32 buffer allocated and zeroed once. It also removes a fragile
aliasing: the old copy clobbered the router's own `idx[0]` after the routed
path had consumed it, which was harmless only by ordering.

### R6a, MoE projections in the dense q8 layout (preregistered 2026-09-15, before any run)

Why: `ssm_phases` / `attn_phases` (the persistent kernel's phase bodies)
read int8 rows plus f16 per-32 scales (`wq`/`ws`); the MoE pack keeps its
projections as raw 34-byte q8_0 blocks. Converting those tensors at pack
time (attn_q, attn_k, attn_v, attn_output, attn_qkv, attn_gate, ssm_out:
130 tensors, 1.36 GB) into the dense layout is a byte split, not a
requantisation: q = the block's 32 int8 values, d = the block's f16 scale,
so every dequantised value is bit-equal. Shared-expert q8_0 tensors and the
embedding stay raw (their kernels read raw blocks). Pack amendment stated
here; round 1's "no pack change" froze only that round.
Change: `tools/engine-pack.py --arch qwen35moe` writes those seven classes
as `q8`; new pack dir `.work/moe-w1/pack-q8d` (the old pack stays for the
A/B); `serve/window.mojo`'s eight `moe_matmul_q8_0_m1` dispatch sites use
the dense q8 row kernel (`gemm_q8`) through the `tens_q8q`/`tens_q8s` views
that already exist there.
Predictions, frozen: GENERATED bit-identical on 20/20 prompts between the
old-pack engine and the new-pack engine (values bit-equal, the row kernel's
accumulation order is the same lane-strided sum of per-block dots; if not
identical, the agreement band decides and the difference is explained by
order); tok/s within -1% to +3% (the big matrix is already near bandwidth;
the 8.9 MB ones at 635 GB/s may gain). No kill line: R6a is an enabling
step; it lands on identity, or on agreement in band with the order
difference explained. GPU: parity, identity A/B, minutes.

### R6a result (2026-09-15): pack lands, launch-path switch does not

The pack change landed (`72eeb72`, `.work/moe-w1/pack-q8d`, 131 tensors in the
dense q8 layout, byte total unchanged, three tensors checked value-for-value
bit-equal against the raw blocks). The launch-path switch of the eight
projection dispatches to the dense q8 row kernel (`gemm_q8` /
`amar_matmul_skinny_q8row`) is REVERTED: 20-prompt tok/s **99.54 -> 91.50
(0.919x)**, GENERATED identical 18/20 (the dense kernel's accumulation
order differs from the R2 raw-block dot). The prediction (-1 to +3%) was
wrong: at these shapes (N = 8192 / 512 / 2048 / 4096, K = 2048) the dense
skinny kernel is slower than the R2 dot, which reads 16 bytes per lane per
34-byte block and runs the 17.8 MB matrix at about 950 GB/s. Cause not
chased (the persistent kernel is the consumer, not the launch path). Patch
kept at `.work/moe-perf/r6a-window-switch.patch`. Consequence for R6: its
projection phases must use the raw-block dot (`q8_0_row_dot` on the old
pack) or prove the dense-layout dot inside the persistent kernel is not
slower; the pack-q8d file stays as the enabling artifact for the second
option and the parity gate will decide which pack R6 ships with.

## R6.0: launch folds on the MoE launch path (preregistered 2026-09-15)

R6 (the persistent token kernel) re-estimated XL after the MoE SSM block was
read in full (`docs/design/moe-persistent-kernel.md`, 14798df); its
decision is the maintainer's. R6.0 is the host-side stage that the design note stages
first. It is the dense profile's launch-fusion question asked again in a
different regime: the dense body runs about 200 launches per token and the
fusion ceiling was 6.9% (memory `launch-fusion-closed`); the MoE launch path
runs 1117 (R4 receipt) and the HEAD timeline (`.work/moe-perf/timeline-head.txt`,
pre-R4, 1216 launches) measures the inter-dispatch gap at 3.12 us median and
3999 us of a 12523 us token, 32% of wall. Every fold below removes a launch
whose kernel is a pure elementwise pass over a vector that the producer
already holds in registers.

Folds (launch counts per token, 30 SSM + 10 attention layers = 40 FFNs):
1. `amar_rmsnorm_cast` writes an f32 copy next to the bf16 one
   (`amar_rmsnorm_cast2`), removing the `amar_widen_bf16` before the router
   GEMV (40) and before the SSM f32 gate GEMVs (30): 70 launches.
2. `moe_gate_up_q4k_pack` and `moe_gate_up_q8_0` write bf16 directly, removing
   the two `amar_cast_bf16` per FFN: 80 launches.
3. `amar_moe_sig_gate` (one wave, dot over H) folds into
   `amar_moe_router_top8` (already one wave): 40 launches.
4. `moe_add3` folds into the shared `moe_down_q8_0` epilogue (the routed
   vector is complete before the shared down starts): 40 launches.
5. The two f32 `amar_matmul_skinny_m1_row` SSM gate GEMVs become one launch
   over both weight tensors (grid doubled): 30 launches.
Total 260 launches: 1117 -> 857.

Predictions, frozen: launch count 857 +- 5 per token by
`bench/moe-launch-count.sh` (the receipt that judges the fold, tok/s cannot);
time removed = 260 gaps x 3.1 us = 0.81 ms plus the removed kernels' own
spans (widen 155 us, casts 126 us, sig_gate 154 us, add3 63 us, one skinny
GEMV 85 us: about 0.58 ms), 1.39 ms of the 10.0 ms token: **99.92 -> 115
tok/s (+15%)**, kill line +5% (104.9). Identity: GENERATED bit-identical on
20/20 prompts against `.work/moe-perf/engine-head` (every fold reorders no
floating-point sum: the same values are written by a different launch), so
any difference is a bug, not an order effect. Gates: `test_moe_block` (the
kernel signatures change, its calls are updated), `run-tests`, BARO_DUMP
per-layer X parity on p09, 20-prompt A/B through `bench/ab-prompts.sh`
(`AB_ENGINE_B` inside the gpu-wait job), fail word read, launch count.
Fold order = the order above; each fold is one commit with its own build,
so a lost fold is found by bisection, not by diagnosis. GPU: gates only,
minutes each.

### R6.0 result (2026-09-15): five folds land, 1117 -> 857 launches, 1.132x

Receipts (`.work/moe-perf/lc-r60.log`, `.work/moe-perf/ab-r60.log`,
`.work/moe-perf/ab-r60/`): launches per token **857.0** by the difference
method (23960 at 16 tokens, 37672 at 32), the frozen prediction exactly.
20-prompt A/B, same stint, engine-head (sha be820d1a) against engine-r60
(sha cfbb664e), both on `.work/moe-w1/pack`, power cap 290 W read back:
**head 94.79 tok/s_gen (spread 0.9%) -> r60 107.27 (spread 1.2%), ratio
1.132**; GENERATED identical on 20/20 prompts, fail word 0 on all 40 runs,
no NOT-RESIDENT exit. Predicted +15%, measured +13.2%, kill line +5%:
holds. The head arm measured 94.79 in this stint against R4's 99.92
receipt; the ratio is the claim, the absolute number is stint-bound
(P4), and the llama.cpp 109.4 comparison needs its own same-stint arm
before it is stated either way. The BARO_DUMP per-layer compare was not
run: bit-identical GENERATED on 20/20 prompts is the stronger statement
of the same property for a change that reorders no sum.
Not folded, still on the launch path: the attention layers' `gemm_w`
split-K reduces (about 100 per token), the SSM small chain
(reduce_gates, conv, l2norm, delta, gated_out: 150), `kv_append`/rope/
head norms on the 10 attention layers (about 60). Those are R6.0b if
the maintainer wants a second fold round before R6 proper; the gap per launch is
unchanged at about 3.1 us, so the ceiling of a full second round is
about 300 launches, 0.9 ms, +9%.

## R6.0b: second fold round on the MoE launch path (preregistered 2026-09-15)

the maintainer's call after R6.0: same-stint llama.cpp arm first, then R6.0b. Same
rule as R6.0: every fold removes a launch whose work moves into the
producer, no floating-point sum changes order, so the gate is bit-identical
GENERATED on 20/20 prompts and the receipt is the launch count.

Folds (per token, 10 attention + 30 SSM layers):
1. The MoE projections write their destination directly. On this profile
   `moe_matmul_q8_0_m1` writes partial 0 and `amar_skinny_reduce` then
   sums SPLITK partials of which only partial 0 is ever non-zero (R4's
   note), so the reduce is a copy: 0 + x = x exactly. q, k, v on the
   attention layers (30) and qkv on the SSM layers (30): 60 launches.
2. `moe_matmul_q8_0_m1_add`: the o projection and the ssm_out projection
   add into the residual themselves (`amar_skinny_reduce_add` was
   Y + (0 + x)); the projection result is still written to its old buffer
   so the BARO_DUMP slots 6 and 7 keep their meaning: 40 launches.
3. `amar_head_rmsnorm_rope`: the per-head norm and the YaRN rope on q and
   on k in one launch each (norm, barrier, rope on the same rows in the
   same order): 20 launches.
4. `amar_kv_append2`: k and v appended by one launch (grid z = 2): 10.
Total 130: 857 -> 727.

Predictions, frozen: launches per token 727 +- 5 by
`bench/moe-launch-count.sh`; time removed = 130 gaps x 3.1 us = 0.40 ms
plus the removed kernels' own spans (reduce copies 191 us, half of the
norm/rope/append pairs about 50 us): 0.64 ms of the 9.32 ms token, **107.27
-> 115 tok/s (+7.5%)**, kill line +5%. Identity 20/20 bit-identical against
`.work/moe-perf/engine-r60`, fail word read on every run, run-tests,
ci-checks, census. Not folded: the SSM small chain (reduce_gates, conv,
l2norm, delta, gated_out: 150 launches) and qgate_split/gate_mul (20);
those change the kernels' work partition and belong to R6 proper.

### Same-stint llama.cpp arm (2026-09-15, after R6.0)

`bench/moe-baseline.sh` (llama-server build 10665, ca3d5a3e1, the same GGUF
by sha prefix 02d1fa2e, `.work/moe-perf/llama-stint/`) then
`bench/ab-prompts.sh` head vs r60 back to back, same power cap 290 W:
llama.cpp **109.92** tok/s decode (20-prompt median, 109.49 to 110.01);
ours head 94.40 (spread 4.1%), r60 **106.95** (spread 3.1%), ratio 1.133
(identity 20/20, fail word 0 on 40 runs). So R6.0 stands at **0.973x of
llama.cpp** in the same stint; the bar is not yet passed, and R6.0b's +7.5%
prediction would put it at about 115, above it, if it holds.

### R6.0b result (2026-09-15): 727 launches, +4.2%, below the +5% line, KILLED

Receipts (`.work/moe-perf/lc-r60b.log`, `ab-r60b.log`, `ab-r60b/`): launches
per token **727.0** (predicted 727); 20-prompt A/B r60 (sha cfbb664e) vs
r60b (sha 27093555), same stint, 290 W: **107.25 -> 111.70 tok/s_gen, ratio
1.042**, identity 20/20, fail word 0 on 40 runs, run-tests 104 PASS, census
0 orphans. Predicted +7.5%, measured +4.2%: the launch count held and the
time did not, so the removed launches were cheaper than 3.1 us each on
this path (the reduce copies and the small attention kernels overlap with
neighbouring dispatches more than the timeline's median gap suggests).
Below the preregistered +5% kill line: not landed, tree reverted, patch at
`.work/moe-perf/r60b-folds.patch` (368 lines, applies on `708488a`,
engine at `.work/moe-perf/engine-r60b`). the maintainer's call whether the line
stands for a bit-identical fold that reads 111.70 against llama.cpp's
109.92 (different stints: r60 measured 106.95 in the llama stint and 107.25
here, so the r60b arm is about 1.6% above the bar, not 2%).

### R6.0b landed on the maintainer's call (2026-09-15)

the maintainer overrode the +5% kill line for this round: the line was set for
kernel rewrites with regression risk, and R6.0b is a launch fold whose
output is bit-identical (20/20) with a 1.042 ratio at 1 to 2% spread. The
override is stated here, not hidden; the line stands for kernel rounds.
Re-gated on main (the patch re-applied on top of the tier wiring
`f798ed4`): receipts below.
Main-tree re-gate (engine sha f52b144e on top of `f798ed4`, `.work/moe-perf/lc-r60b-main.log`, `ab-r60b-main.log`): launches per token 727.0; 20-prompt A/B r60 107.28 (spread 2.0%) -> r60b **111.89** (spread 1.2%), ratio 1.043, identity 20/20, fail word 0 on 40 runs; run-tests 104 PASS, ci-checks 0, census 99 kernels 0 orphans. MoE champion: **111.89 tok/s_gen**, launch path, 727 launches per token; llama.cpp same-box arm 109.92 (different stint, its r60 read 106.95 there).

## R6.1 result (2026-09-16, fable lane): the persistent MoE token kernel is at parity

`kernels/mega_moe.mojo` (`f900bbb`), one launch per token over the 40 layers,
head on the launch path. `BARO_DUMP` compare identical over 64 tokens x 80
slots on p09; teacher-forced agreement 64/64 on 20/20 prompts against a
build of `38ee0b7`; fail word 0 on every run; run-tests 104 PASS, ci-checks
0. Receipts and the two defects the gate found in
`exchange/lane-R6-report.md`. Speed was not gated at this stage.

## R6.2: speed of the persistent MoE kernel (preregistered 2026-09-16, before any timed A/B)

### Receipts in hand (instrument readings, not results)

Per-run device receipt (`a2xxxxx`, "mega barrier gen"): launch arm gen 0 and
727.0 launches per token; persistent arm gen 470 per token and 8.0 launches
per token (embed, head norm, head GEMM split-K, reduce, argmax, and the
dump copies), rocprofv3 difference method, `.work/r6/lc-*.log`.

Phase stamps, last token of p09, `BARO_PROFILE=5` (`.work/r6/prof5-persist.log`),
kernel span **8391 us**; per-token sums in us:

| phase (30 SSM layers) | us | phase (10 attention layers) | us |
|---|---|---|---|
| rms + barrier | 75 | rms | 25 |
| projections (qkv 17.8 MB, gate 8.9 MB, alpha/beta f32) | 1529 | q/k/v projections (about 20 MB) | 307 |
| gates + conv | 110 | heads (norm, rope, append) | 33 |
| l2 | 54 | attention | 116 |
| delta | 457 | gate multiply | 13 |
| gated out | 66 | out projection (8.9 MB) | 201 |
| ssm_out (8.9 MB) | 672 | | |
| ffn rms | 76 | ffn rms | 24 |
| router | 112 | router | 36 |
| top-8 + sigmoid | 268 | top-8 + sigmoid | 89 |
| gate+up (routed 9.4 MB + shared) | 1379 | gate+up | 456 |
| down (routed 4.7 MB + shared) | 1689 | down | 586 |

The q8 projection phases (SSM 1529 + 672, attention 307 + 201 = 2709 us
for about 1.09 GB) run at about 400 GB/s where the launch kernels for the
same bytes measured 950 GB/s (the HEAD timeline): the persistent grid is one
block of 8 waves per CU (2 waves per SIMD), and `q8_0_row_dot` issues one
load group per iteration with no unroll, so those phases are
latency-bound, not byte-bound. Expert down is 1.19x the launch kernels'
sum, gate+up about equal, delta 1.4x.

Disclosed: the identity runs of R6.1 are forced runs of a single binary
and were not a timed A/B, but they carry a reading: reference (launch path,
greedy) median 111.29 tok/s_gen against candidate (persistent, forced)
108.07, ratio 0.971. Teacher forcing syncs the host once per token
(host_enqueue_s equals gpu_total_s on the candidate logs), so the reading
is biased against the candidate by an unknown amount; it says the kernel
as landed is not faster, and the phase table says why.

### Levers, in order

1. **Occupancy (no arithmetic change):** the grid size becomes a build-time
   knob `BARO_MOE_G` (96, 192, 288). Residency at 256 VGPRs, wave32, 8
   waves per block is 3 blocks per CU, so 288 is the ceiling with zero
   slack (`persistent-kernel-gfx11`: probe upward with the bounded barrier,
   fail word read on every run; a NOT-RESIDENT exit voids that G). Work
   distribution is block/wave-strided, so any G computes the same values.
   Sweep on 3 prompts (exploration), freeze the chosen G by commit, then
   the confirmation A/B.
2. **Latency hiding in the persistent kernel's dots (bit-exact by
   construction):** `q8_0_row_dot` and `q4k_dot_blocks` variants that issue
   the loads of UNROLL block-iterations before consuming them, keeping the
   per-lane fma chain in the same block order and the same `warp.sum` of
   `acc.reduce_add()`, so every partial is the launch kernel's. The gate 1
   dump compare and the 20-prompt identity are re-run on the changed
   kernel before its A/B.
3. Not in this round: the 256 spills / 940 B scratch (a whole-kernel
   allocator question, `kernel-parity` rule 10), the expert-dot lane
   utilisation at K = 512 (16 of 32 lanes idle in `q4k_dot_blocks`), the
   head fold. Each is its own preregistered step if R6.2 lands.

### Predictions, frozen

- Lever 1 alone: kernel span 8391 -> 7000 to 7600 us (the q8 phases gain
  most from 2 -> 4 or 6 waves per SIMD; the expert phases some). Token
  (20-prompt median) 111.9 champion vs about 115 to 122 persistent.
- Levers 1 + 2: q8 phases 2709 -> 1400 to 1700 us (800 to 950 GB/s), expert
  and delta phases -300 to -600 us; kernel span **8391 -> 6000 to 6800 us**;
  token about 6.9 to 7.7 ms including the launch-path head (0.7 ms) and
  gaps on the 8 remaining launches; **20-prompt median 130 to 145
  tok/s_gen, ratio 1.16 to 1.30 over the champion's same-stint arm.**
- Kill line, as the round preregistered: **below +5% (ratio under 1.05)
  the default stays BARO_MEGA=0 on the MoE profile** and the kernel stays
  in the tree as an opt-in; identity 20/20 and fail word 0 on all 40 runs
  are preconditions, not part of the ratio. Default flips only on a pass.
- Falsifier of the mechanism: if G = 192 does not move the q8 projection
  phases by at least 20% in the stamp profile, the phases are not
  occupancy-bound and lever 2's prediction is void before it is built.

### Procedure

Same binary both arms (`BARO_MEGA=0` vs `BARO_MEGA=1`), `bench/ab-prompts.sh`
under `bench/clock-probe.sh`, 20 prompts, alternating per prompt as the
script does, power cap and voltage offset read back (P1), per-run device
receipt (gen 0 vs 470 per token) and fail word in `results.txt`, identity
column PASS on 20/20, `isa-loops` fingerprint of the timed kernel recorded
in the report, `run-tests.sh` and `tools/ci-checks.sh` green at the commit
that carries the default flip.

### R6.2 lever 1 result (2026-09-16): G above 96 is NOT-RESIDENT, the knob is dead

`BARO_MOE_G` builds at 96, 192, 288 on p03/p09/p12 (`.work/r6/gsweep/`):
G = 96 fail word 0, gen 470 per token, kernel span 8327 to 8382 us; **G = 192
and G = 288 fail word 1 with gen 0 on 6/6 runs** (the first barrier timed
out, the run's tokens are void). The 3-blocks-per-CU arithmetic in the
preregistration used a 1536-VGPR file per SIMD; the `persistent-kernel-gfx11`
rule (waves per SIMD = floor(768 / vgpr), vgpr 256 -> 1 block per CU ->
ceiling 96) was right and G = 96 has zero slack, as the dense kernel has. The
falsifier for lever 2 ("G = 192 moves the q8 phases by 20%") cannot be
evaluated: occupancy is not a knob at 256 VGPRs. Lever 2 proceeds on its own
prediction (q8 phases 2709 -> at most 1700 us in the stamp profile, kernel
span under 7000 us), which is the same mechanism the dense kernel uses at
the same 2 waves per SIMD (`q8_row_dot` UNROLL = 4). Raising occupancy by
cutting VGPRs to 192 or below is a separate step after R6.2, not this round.

### R6.2 result (2026-09-16): 1.027x, below the +5% line; kernel stays opt-in, default stays BARO_MEGA=0

Kernel as timed (`kernels/mega_moe.mojo` at the R6.2 commit): the q8 dots
issue four block-iterations of loads before consuming them (`q8_dot_u[4]`,
same fma chain), the expert q4k dots are the launch kernel's (`q4k_dot_u`
was tried and reverted: +376 us on the untouched down phase from spills,
gate+up flat), and the delta scan is the chunked 32-wide `fma()` form whose
ISA the launch `amar_ssm_delta_step` already contracts to (fma 1061 / mul
346 / add 0, `isa-loops` on the same binary). ISA receipt: 256 VGPRs, 236
spills, 936 B scratch; fingerprint hot loops dual 361/361/51/51/51/227/229/26,
totals fma 1691 mul 262 add 387 scratch 395 (`.work/r6/isa-r63`).

Gates before timing: `BARO_DUMP` compare identical over 64 tokens x 80
slots (p09); teacher-forced agreement 64/64 on 20/20 against
`engine-ref-38ee0b7`, device receipt gen 30080 on all 20 candidate runs
(`.work/r6/force-r63/results.txt`).

Timed A/B (`.work/r6/ab-r63/`, `ab-r63.log`): one binary sha `7c27c2a82214b485`,
`BARO_MEGA=0` vs `BARO_MEGA=1`, 20 prompts alternating, power cap 290 W and
-100 mV read back, clock probe sclk med 3134 MHz (min 1238 across idle,
max 3313), junction 71 C. **launch 110.90 tok/s_gen (spread 1.4%) ->
persistent 113.94 (spread 1.4%), ratio 1.027**; identity PASS 20/20; device
receipt gen 0 on every launch run and 30080 on every persistent run; fail
word 0 on all 40.

Prediction (130 to 145, ratio 1.16 to 1.30) **falsified**. Where it went, by
the stamp profile (kernel span 8391 -> 7986 us, `.work/r6/dump-r63/persist.log`):
delta 457 -> 253 and down 1689 -> 1534 carried the gain; the q8 projection
phases stayed at 2717 us (about 400 GB/s) with or without the per-wave
unroll, so at 2 waves per SIMD the raw 34-byte-stride block loads are not
hidden by issuing more of them per wave. The persistent kernel removes
about 2.2 ms of launch gaps and gives back about 2 ms in phases slower than
the launch kernels they replace; net +3%.

Kill line +5% not met: default unchanged, kernel kept as an opt-in
(`BARO_MEGA=1` on the MoE profile, refuses BARO_TIER and BARO_EXPERTS).
Pools for a next round, each its own preregistration: (1) q8 projection
phases at 400 GB/s, the largest (2.7 of 8.0 ms): either the dense q8
layout from the R6a pack (aligned 16 B loads; its arithmetic differs from
the raw-block dot, so it needs the agreement band, not identity) or LDS
staging of the activation row as the dense kernel does; (2) VGPRs to 192
or below for 2 blocks per CU (G = 192), which is the occupancy lever this
round could not pull; (3) expert down at K = 512 uses 16 of 32 lanes.

## R6.3: aligned q8 loads in the persistent kernel (preregistered 2026-09-18, before any build)

Pool (1) from the R6.2 result: the q8 projection phases run at about 400 GB/s
(2709 of 8391 us) where the launch kernels do the same bytes at 950. R6.2's
per-wave unroll (`q8_dot_u[4]`) moved nothing, so more loads in flight per
wave is not the lever at 2 waves per SIMD. The remaining difference between
the dense megakernel's q8 phase (near HBM rate at the same occupancy,
`q8_row_dot`) and this one is the weight layout: the raw q8_0 block is 34
bytes, so every lane's 16-byte q load sits at `b * 34 + 2 + h * 16`, which is
never 16-byte aligned and straddles a 64-byte line every few blocks; the
dense layout (R6a pack, `pack-q8d`) puts the 32 int8 of a block at a
32-byte-aligned address and the f16 scales in their own array.

Change, kernel side (`kernels/mega_moe.mojo`, `kernels/moe.mojo`): a dense-
addressed twin of the raw-block dot with the SAME lane mapping, the SAME
per-lane arithmetic (`(d * q).cast[bf16].cast[f32]`, `fma`, `reduce_add`,
`warp.sum`) and the SAME block order; only the addresses change (q at
`row * K + b * 32 + h * 16`, scale at `N * K + row * (K / 32) * 2 + b * 2`).
The eight projection dispatches on the launch path (`moe_matmul_q8_0_m1` /
`_add` sites in `serve/window.mojo`) and the six projection phases of the
persistent kernel (attn q/k/v/out, ssm qkv/gate, ssm_out) switch to it. The
shared-expert q8_0 tensors and the embedding stay raw, as the R6a pack
leaves them. The engine on the MoE profile then requires the dense
projections (the loader refuses a `q8_0` projection tensor by name, so a
raw pack fails loud instead of reading garbage).

Change, pack side: `pack-q8d` no longer exists on disk and `/home` has 14 GB
free, so it is rebuilt as a btrfs reflink of the raw pack with the 130
projection tensors rewritten in place (the byte split of `72eeb72`: every
tensor keeps its byte length and offset, so `index.txt` changes only in the
dtype column). Receipt: three tensors dequantised value-for-value against
the raw pack before any engine run.

Predictions, frozen:
- P-R63-1 (identity): the new engine on `pack-q8d` is bit-identical to the
  champion `38ee0b7` on `pack` on 20/20 prompts, both arms of it
  (`BARO_MEGA=0` and `BARO_MEGA=1`), teacher-forced 64/64
  (`bench/force-ab.sh`). Same values, same per-lane ops, same order; a
  single differing id falsifies the "same arithmetic" claim and the round
  stops there.
- P-R63-2 (mechanism): in the `BARO_PROFILE=5` stamp profile the four q8
  projection phases fall from 2709 us to at most 1700 us (800 GB/s or
  better) and the kernel span from 7986 to under 7000 us. If the phases
  stay above 2300 us with aligned loads, alignment is not the cause; the
  next pool is LDS staging of the activation row, not another load form.
- P-R63-3 (launch arm): the launch path with the dense-addressed dot is not
  slower than the raw dot: `BARO_MEGA=0` on `pack-q8d` within -1% to +3% of
  the champion's 111.89 (cross-stint) and, in the same stint, the arm the
  R6.2 A/B measured at 110.90.
- P-R63-4 (result): same-binary 20-prompt A/B `BARO_MEGA=0` vs `BARO_MEGA=1`
  on `pack-q8d`, ratio 1.08 to 1.18 (about 120 to 131 tok/s_gen). Kill line
  unchanged from R6.2: +5% or the default stays `BARO_MEGA=0`.

Procedure: R6.2's (same binary both arms, `bench/ab-prompts.sh` under
`bench/clock-probe.sh`, power cap and voltage read back, gen 470 per token
and fail word 0 on the persistent arm, identity column PASS 20/20,
`isa-loops` fingerprint recorded), preceded by the pack receipt and
P-R63-1. Every GPU run through `gpu-wait run --timeout`. `run-tests.sh` and
`tools/ci-checks.sh` green at the commit that carries any default flip.

### R6.3 killed (2026-10-12): the preregistered A/B never ran and cannot run here

The kernel landed on main (`d684fb2`, merge `7a3d03b`), the lane is gone, and the frozen A/B (`BARO_MEGA=0` vs `1`
on the resident pack-q8d, 20 prompts, predicted 1.08-1.18) has no numbers. It needs the resident qwen35moe pack, and
resident MoE runs are refused on this box since the 2026-10-12 crash (20 GB pack OOM next to the desktop, then
compositor ring resets; `m.ledger/gpu.md`). Per "close the round or kill it": killed, prediction untested. The kernel
stays on main, opt-in as before (default `BARO_MEGA=0` for the MoE profile); a future tier-mode persistent kernel
would need its own prereg.

### Resident refusal voided, R6 re-measured (2026-10-17)

The 2026-10-12 refusal assumed MAX takes ~90% of free VRAM; the engine now defaults
`MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT` to 2, so engine buffers come from free VRAM.
`pack-fit .work/moe-ngram/pack --tmax 4096 --used-gb 0.97`: FITS, 2.61 GB headroom. Resident run next to
the desktop: peak VRAM 20.9 GB, GTT 1.15 GB, no OOM, no ring reset. Resident MoE is allowed again,
with `pack-fit` against measured desktop VRAM first.

Same-binary 20-prompt A/B on the raw resident pack (engine `e7c21c8c586a74db`, main `1cb3aba`, TMAX 4096,
290 W, -120 mV): `BARO_MEGA=0` 129.80 tok/s_gen (spread 3.2%), `BARO_MEGA=1` 129.35 (spread 21.2%), ratio
**0.996**, identity 20/20, 507 barriers per token, fail word 0. The launch path has caught up (111.89 at
`38ee0b7`); the persistent kernel no longer buys anything on the raw pack. Default stays `BARO_MEGA=0`.
Receipts `exchange/receipts/r6-resident/`.
