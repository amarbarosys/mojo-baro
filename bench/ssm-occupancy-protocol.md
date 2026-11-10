# SSM delta_step protocol — frozen before first timed run

> **Binding: [`bench/PROTOCOL-RULES.md`](PROTOCOL-RULES.md).** P1 in particular:
> every parameter defining an arm is read back from the running system and
> recorded BEFORE the timed run. No receipt, no arm.


Question: does reducing `amar_ssm_delta_step`'s state traffic (item 3) or
raising its occupancy (item 2) earn its complexity at our engine's actual
decode shape, and does fusing the per-row SSM loop (item 1) pay for itself
in an m=4 MTP verify window?

Plan and derivations: `.work/ssm-fusion-plan.md`. This file freezes the
predictions; the plan holds the reasoning.

## Scale check — binds every prediction below

Per-token weight traffic ~17.9 GB (frozen, `bench/decode-race-protocol.md`).
SSM state traffic per decode step: N_SSM=24 layers x (2 MB read + 2 MB write
f32 state, ~96 KB conv round-trip) ~= 100 MB/token — **~0.6% of total
traffic at M=1**.

Consequence, stated in advance so it cannot be spun afterwards: **no item
here can move plain M=1 decode more than ~1% by bandwidth.** A ~0%
end-to-end result is the EXPECTED outcome at M=1 and does not falsify a
kernel-level claim. Kernel-level and end-to-end claims are recorded and
judged separately.

## Correctness precondition (gates ALL speed numbers)

Items 2 and 3 are pure reorders/re-partitions: neither changes the
arithmetic or its order. Item 1 must also preserve per-element f32 op
order. Therefore:

1. `kernels/test_ssm_block.mojo` parity passes at **unchanged tolerances**.
   A loosened threshold voids the run.
2. `tools/check-tokens.sh` — 64-token greedy output **bit-identical** to
   `.work/engine-pack/ref-tokens-64.txt`. Required, not hoped for. If a
   token moves, the kernel is wrong; it is not "numerics drift".
3. `./run-tests.sh` green.

Any one failing voids the run: no tok/s, no per-kernel time, no claim.

## Baseline (measured this session, HEAD c8cf219, clean tree)

- Engine built: `./.venv/bin/mojo build serve/engine.mojo -o .work/engine -I kernels`
  (no shim link — the engine imports no FFI). **Not previously recorded
  anywhere; recorded here.**
- Single run, `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10`:
  `tok/s_gen 40.348`, `prefill_s 0.0647`, `decode_s 1.5614`,
  `host_enqueue_s 1.025`, `gpu_total_s 1.626`, token gate PASS.
- This is ONE run, not a median. It is the correctness reference and an
  indicative speed only. The medians below are what get compared.

## Instrument (identical for every arm)

- llama-server DOWN, GPU exclusive, nothing else resident.
- `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10`.
- 5 repeats, drop the first, median of 4. Spread gate <5% end-to-end;
  <1.5% for any per-kernel micro number (coldcache v2 discipline).
- Never sample `rocm-smi` inside a timing loop (BASELINE.md: produced a
  fake 11k-27k GFLOP/s oscillation).
- Every number carries its commit hash.

## Arms

One 4-arm sweep, because items 2 and 3 pull in opposite directions
(register staging costs VGPRs -> fewer resident waves; the j-split wants
more, smaller blocks):

| arm | item 3 (S0 staged in registers) | item 2 (JSPLIT) |
|---|---|---|
| A (control) | no | 1 |
| B | no | 4 |
| C | yes | 1 |
| D | yes | 4 |

Arm A must be bit-identical to today's kernel — that is the sweep's own
self-check. JSPLIT in {1,2,4} explored on the winning staging choice.

Item 1 is measured separately, at m=4, against whichever of A-D wins.

## Predictions — FROZEN, recorded before any timed run

1. **Arm A reproduces baseline** within the 5% spread gate and is
   bit-identical on tokens. If not, the instrument is broken and nothing
   else in this file may be read.
2. **Item 3 (arm C vs A):** S0 read traffic per launch halves (2 MB -> 1 MB).
   Direction only on GPU time — this wins ONLY if the kernel is
   traffic-bound at 32 blocks / 96 CUs. Falsifier: arm C slower than A on
   per-kernel GPU time, which would mean register pressure dominates and
   the staging is dead.
3. **Item 2 (arm B vs A):** per-kernel `delta_step` GPU time drops.
   **`tok/s_gen` moves <1% at M=1** — recorded in advance so a flat
   end-to-end result is not misread as falsification. Falsifier: arm B
   per-kernel time not below A, i.e. the kernel was never occupancy-bound.
4. **Prior-work guard:** an earlier launch-fusion experiment on this same
   loop returned +5% because decode is GPU-bound (host enqueue 83 ms vs
   GPU 775 ms). That measured **enqueue cost**. Items 2 and 3 are
   **occupancy and traffic** — a different mechanism. That null result
   neither predicts nor forbids this one, and must not be cited as
   "already tried". Conversely, no launch-count win may be claimed here.
5. **Item 1 (m=4 window):** removes ~(m-1) x 2 MB x 24 ~= 144 MB of
   intermediate state READS per window (writes are NOT removable —
   stage-2 rollback needs every step's state landed). At ~960 GB/s that
   is ~150 us against a window costing ~18+ ms of weight streaming:
   **~1-2% window speedup**, plus a launch-count reduction bounded above
   by the +5% null in (4). An earlier estimate of ~340 MB double-counted
   item 3's saving; 144 MB is the marginal figure once item 3 has landed.
   Falsifier: fused kernel slower than the arm-winner at m=4 -> keep the
   unfused path; the m=1 path keeps the arm-winner kernel regardless.

## What would make us abandon each item

- Item 3: arm C loses to A AND arm D loses to B (staging never pays).
- Item 2: no arm beats A on per-kernel time (never occupancy-bound).
- Item 1: loses at m=4, or its slot-index contract turns out to conflict
  with stage 2's accept loop once that exists.

## Results

Run 2026-09-04, branch `lane-ssm-spill`, on top of `main`'s merged M2 fold
(`871faca`, `9808a9d`). Instrument: llama-server DOWN, GPU exclusive,
`flock .work/gpu.lock` on every run.

### Arms A/B (item 3, unstaged) do not apply — protocol line invalidated

Item 3 proposes staging `S0`'s column into registers to kill a "pass 1
reads it, pass 2 re-reads it" double read (`.work/ssm-fusion-plan.md`
"Current shape" section). **`amar_ssm_delta_step` already reads the column
exactly once into `col: SIMD[f32, SSTATE]` and reuses it for both the `sk`
pass and the `s`/`o` pass** — this was true in the kernel this protocol
inherited (commit `c8cf219`, predating both this protocol's freeze and my
M2 fold) and is unchanged by the fold. There is no unstaged variant left in
the repo to build as a control; reconstructing one would be inventing code
the protocol never asked for, and the question item 3 poses (does staging
pay for itself) was never gated by a receipt before it landed — a real gap
in this protocol's own discipline, but not one stint 2 can retroactively
fix without fabricating an arm. **Arms A and B, and the protocol's own
self-check ("Arm A must be bit-identical to today's kernel"), are
unsatisfiable as written: "today's kernel" already IS what the protocol
calls arm C** (item 3 = yes, item 2 = 1/control). Per the brief: stopping
here on item 3, not inventing a replacement arm.

### Arms C/D (item 2, JSPLIT) — measured, falsified

Item 2 (JSPLIT: split the column dimension across more, smaller blocks —
`grid_dim=(NH_V, JSPLIT)`, `block_dim=SSTATE/JSPLIT`) still applies cleanly
to the folded kernel; implemented as a new `JSPLIT: Int` comptime parameter
on `amar_ssm_delta_step`, `JSPLIT=1` reproducing arm C (today's kernel) and
`JSPLIT=4` as arm D, per `.work/ssm-fusion-plan.md`'s JSPLIT=4 rationale
(128 blocks > 96 CUs, no partials).

**Gate 1 — correctness (both arms):**
- `kernels/test_ssm_block.mojo`, unchanged tolerances, at `JSPLIT=1` and
  `JSPLIT=4`: **PASS** both (m=1 vs numpy ref, m=4 fold vs 4x sequential
  m=1, `max_rel: 0.0` on the m=4 self-check for both JSPLIT values).
- `bench/mtp-prompts.sh` arm A (no-spec) `GENERATED` line, both JSPLIT
  builds, byte-identical to `main`'s `.work/mtp-m2/p09-explain-gpu.A.log`
  (spot-checked since JSPLIT changes no per-column arithmetic order;
  20-prompt full sweep not re-run since arm D was already disqualified on
  speed before reaching that gate — see below).

**Gate 2 — per-kernel timing, `BARO_PROFILE=2`, `BARO_SPEC_K=2`,
`bench/mtp-prompts/p09-explain-gpu.tokens`, 3 repeats each, `ssm-kernel:
delta` line:**

| arm | JSPLIT | delta (s), 3 runs | median |
|---|---|---|---|
| C (control) | 1 | 0.04805, 0.04782, 0.04736 | **0.0478** |
| D | 4 | 0.10115, 0.08841, 0.08776 | **0.0884** |

Arm D is **85% slower** than arm C — far outside the <1.5% per-kernel
spread gate in either direction; this is not noise. **Falsifies item 2's
prediction outright** ("per-kernel `delta_step` GPU time drops"). Matches
the protocol's own abandon criterion verbatim: *"Item 2: no arm beats A on
per-kernel time (never occupancy-bound)."* `tools/isa-receipt.py` explains
why: `vgpr_count` is unchanged (192, both arms — JSPLIT doesn't reduce
per-thread register need, each thread still carries the full 128-wide
`col` regardless of column-slab width) but `vgpr_spill_count` at MR=8 goes
603 (JSPLIT=1) → 2009 (JSPLIT=4), ~3.3x — JSPLIT=4's per-block `kq` fill
loop (`comptime for g in range(JSPLIT)`) makes every block redundantly
re-read the full 128-wide k/q vector, and the narrower 32-thread blocks
apparently cost more (via redundant traffic + higher spill pressure) than
the extra occupancy buys back. 32 blocks / 96 CUs was not, in practice,
occupancy-bound the way the plan predicted.

**Caught mid-run, worth recording:** the initial JSPLIT=1 "control" build
(before splitting the kernel body into a `comptime if JSPLIT == 1` fast
path that is byte-for-byte the pre-JSPLIT code) was NOT actually identical
to today's kernel at the ISA level — merely adding the `JSPLIT` template
parameter and routing `j` through `js * JW + jt` (with `js` always 0)
raised `vgpr_spill_count` from 603 to 2029 at MR=8, a ~3.4x regression from
the refactor alone, nothing to do with JSPLIT's actual value. First-pass
timing on that broken control showed delta at ~0.09s for BOTH arms
(apparently no difference), which would have wrongly read as "item 2:
inconclusive, within noise" and hidden arm D's real regression. Caught by
re-checking `vgpr_spill_count` against the pre-stint-2 baseline before
trusting the comparison (P1: read the value back from the instrument, not
from the diff you typed) and fixed by giving `JSPLIT == 1` its own
compile-time branch that reproduces the original code exactly, rather than
generalizing the control path through the same code the experiment uses.

### Item 1 (fusion) — already landed in stint 1, not re-measured here

Item 1 (fuse the per-row SSM loop into one launch per window) is the M2
fold already merged to `main` (`871faca`). The plan's own sequencing
("item 1 LAST, built against item 2's winning geometry") is moot: item 2
has no winning geometry (arm C, i.e. today's shape, wins by default), so
there is nothing for item 1 to be rebuilt against. No falsifier triggered.

### Verdict

**No arm lands.** Item 3 is moot (already shipped, unfalsifiable without
inventing code). Item 2 is measured and falsified (arm D 85% slower, not
faster). `amar_ssm_delta_step` stays exactly as `main` has it. No code
change in this stint; this file's Results section and
`.work/briefs/status-mrowC.md` are the deliverable.

## Re-run 2026-09-15: fair arm D, preregistered before the first timed run

Why: `exchange/2026-09-15-jsplit-review.md` shows by compile receipt that the
2026-09-04 arm D carried the same ~2000-spill codegen penalty the control was
repaired away from (the penalty follows the column-index expression, not the
value of JSPLIT: 2029 at JSPLIT=1 vs 2009 at JSPLIT=4; a masked index gives
599 at JSPLIT=4). The recorded 85% therefore measured the refactor. This
section re-runs item 2 with a treatment that pays no penalty the control
escaped. Working tree only, nothing committed; freeze receipt = sha256 of this
file recorded in the exchange report before the first run.

### Arms

- **C (control):** `serve/engine.mojo` built from `main` `a9c0106` with
  `kernels/` and `serve/` untouched, binary `.work/jsplit-review/engine-C`.
  Receipt required: `tools/isa-receipt.py` shows `amar_ssm_delta_step`
  vgpr_spill_count 603 at MR=8 and 81 at MR=1, vgpr_count 192.
- **D (treatment):** same tree plus `JSPLIT: Int = 1` on
  `amar_ssm_delta_step` with `comptime if JSPLIT == 1` reproducing the
  original body verbatim and the split path forming the column index as
  `(block_idx.y & (JSPLIT - 1)) * (SSTATE // JSPLIT) + thread_idx.x`, with
  the full-width `kq` fill; `serve/registry.mojo` dispatches with
  `grid_dim=(NH_V, 4), block_dim=32`. Binary `.work/jsplit-review/engine-D`.
  Receipt required BEFORE timing: spill count within 3% of arm C's at every
  MR in 1..8 (expected 599 at MR=8, 78 at MR=1 from the standalone
  receipts). If D spills more than that, D is not the fair arm and nothing
  downstream may be read.

### Configurations (both arms, each)

- **P (primary, champion path):** `BARO_PROFILE=2 BARO_SPEC=1 BARO_SPEC_K=2
  BARO_PROMPT=bench/mtp-prompts/p09-explain-gpu.tokens`. With the megakernel
  on (default `BARO_MEGA=1`) the per-kernel SSM path, and so
  `amar_ssm_delta_step`, runs only in the spec verify windows (MR = k+1 = 3),
  which is exactly where the champion pays for this kernel today.
- **S (secondary, the 2026-09-04 shape):** `BARO_PROFILE=2 BARO_MEGA=0
  BARO_SPEC=0 BARO_PROMPT=...p09...`. Per-kernel MR=1 decode for all 64
  tokens, the shape item 2 was originally written against.

Instrument: `gpu-wait run --` on every launch, GPU otherwise idle (`gpu-wait
list` empty, checked before each launch). 5 runs per arm per configuration,
alternating C, D, C, D, ... so drift lands on both arms equally. Number read:
the `ssm-kernel: delta <seconds>` line. Reported: median of 5 and median of
the last 4. Spread (max/min - 1) of the control must be under 1.5% (this
file's per-kernel gate) for a difference to be read; if wider, only disjoint
ranges may be read as a direction.

### Read-back (P1), per run

`BARO_SPEC:`, `BARO_MEGA:`, `spec k:` lines from the run log; `GENERATED:`
line byte-identical across all runs of a configuration (JSPLIT changes no
per-column arithmetic order, so a token difference means a wrong kernel and
voids the run); `mega fail word:` must be 0 wherever it is printed; sha256 of
both binaries; the isa-receipt rows above.

### The 0.0478 s reference is not expected to reproduce

136 commits touch `serve/` or `kernels/` between `af30ac9` and `a9c0106`
(megakernel decode, q4 champion, dattn, MTP changes). The per-kernel delta
total is a sum over whichever windows take the per-kernel path, and that set
changed. The arm-A style self-check for this re-run is therefore: control
spread under 1.5%, identity across all runs, receipts above. Arm C's absolute
number is recorded and compared to 0.0478 for the record only.

### Predictions, frozen

1. **Confound thesis:** with spills equalized, D's delta time lands within
   10% of C's in both configurations; the 85% gap does not reappear.
   Falsifier: D at or above 1.5x C in either configuration, which would mean
   the j-split geometry itself is the loss and the 2026-09-04 verdict stands
   for a different reason than recorded.
2. **Item 2 as originally written** (per-kernel time drops): D below C by
   more than the control spread in configuration S, the MR=1 shape it was
   argued for. Direction only; no magnitude predicted. Falsifier: D not
   below C. In configuration P (MR=3, three rows per launch, 3x the work per
   block) no direction is predicted.
3. `tok/s_gen` moves under 1% in either arm (SSM state traffic is ~0.6% of
   per-token bytes, see the scale check at the top of this file). A flat
   end-to-end number is expected and is not evidence either way.

### Outcomes and what each means

- D within spread of C: item 2 is a **confirmed null** (not just suspected)
  and the confound thesis is confirmed at the same time.
- D below C beyond spread: item 2 is **confirmed at kernel level**; still no
  end-to-end claim without the 20-prompt median (P4).
- D above 1.5x C: prediction 1 falsified; the geometry loses on its own.

### Amendment 2026-09-15, before the first timed run: per-MR receipts

The 3% spill gate above was set from the MR=8 and MR=1 receipts. Reading
arm D's engine binary at every MR (all still before any timed run):

| MR | C spills | D (JSPLIT=4, masked) | 1-D form, JSPLIT=4 | JSPLIT=2, masked |
|---|---|---|---|---|
| 8 | 603 | 599 | 612 | read back below |
| 7 | 526 | 530 | | |
| 6 | 458 | 461 | | |
| 5 | 380 | 405 | 405 | 366 |
| 4 | 312 | 336 | 336 | 297 |
| 3 | 235 | 267 | 267 | 227 |
| 2 | 167 | 181 | 181 | 155 |
| 1 | 81 | 78 | 78 | read back below |

At MR=2..5 both fair JSPLIT=4 forms land on the same count, +32 spills over
C at MR=3 (+13.6%), so that residue belongs to the 4-way split itself (four
fill iterations per row) and not to the index expression. The gate as
written fails for JSPLIT=4 at MR=3, which is configuration P's shape.
Decision, recorded before running:

- Configuration S (MR=1: 78 vs 81) is gated PASS for JSPLIT=4 and is the
  fully interpretable comparison for the original item 2.
- Configuration P keeps JSPLIT=4 as an **exploratory** arm carrying a stated
  13.6% spill handicap: a D win or tie there is interpretable, a D loss is
  confounded and may only be reported as such.
- A third arm **D2 = JSPLIT=2 (masked form, `grid_dim=(NH_V, 2),
  block_dim=64`)** is added. It spills less than C at every MR read so far
  and so passes the gate in both configurations. Receipt for its engine
  binary at MR=8 and MR=1 is read back before timing, same 3% rule.

Run order becomes C, D, D2 alternating, 5 rounds per configuration.
Predictions 1..3 apply to D2 unchanged. Prediction 2 (per-kernel time drops)
now has two chances in configuration S; if neither D nor D2 beats C beyond
the spread, item 2 is a confirmed null for both split widths.

### Amendment 3, 2026-09-15, before the tok/s runs

Under `BARO_PROFILE=2` the engine synchronizes after every SSM sub-block, so
its `tok/s_gen` is not the tok/s instrument. Prediction 3 (tok/s moves under
1%) is read from a separate pass with no `BARO_PROFILE`, same prompt, same
`BARO_SPEC=1 BARO_SPEC_K=2`, arms C, D, D2 alternating, 5 rounds. One prompt
only, so this is a receipt on prediction 3 and never an end-to-end claim
(P4 needs the 20-prompt median). Identity and `mega fail word` read on every
run as before.

### Amendment 4, 2026-09-15, before the launch-path run

Non-profiled pass (Amendment 3) read D and D2 both 3.3% below C on
`tok/s_gen` with disjoint ranges, the same amount for both split widths,
while the profiled per-kernel delta time was flat (D) or 2.3% better (D2).
`host_enqueue_s` equals `gpu_total_s` to within 30 us in every run, so the
decode is host-enqueue-bound on this path and a fixed per-launch host cost
would show up exactly like this. The one change D and D2 share that is not
the kernel body is the launch form: `grid_dim=(NH_V, SSM_JSPLIT)` (a 2-D
tuple) where C passes the scalar `NH_V`.

Arm **J1**: the working tree with `SSM_JSPLIT = 1`, i.e. the original kernel
body (comptime branch, receipt must read 603 / 81 spills at MR=8 / MR=1)
launched with the tuple `grid_dim=(NH_V, 1)`. Same non-profiled pass, C and
J1 alternating, 5 rounds. Prediction: if the launch form is the cost, J1 lands
with D and D2 (about 3% below C, disjoint from C); if the kernel is the cost,
J1 lands on C. Either way the result names where the 3.3% lives before any
JSPLIT arm can be judged end-to-end.

### Amendment 5, 2026-09-15, before the sub-block attribution run

J1 (first three rounds read while the pass was still running) lands on C, so
the launch form is not the 3.3%. The loss then lives in GPU work that the
profiled delta segment does not see: something around the split kernel gets
slower only when it runs unsynchronized in the stream. `BARO_PROFILE=1`
attributes GPU time per sub-block (attn / ssm / ffn / head) with a sync at
each sub-block boundary. Arms C, D, D2, 3 rounds alternating, same prompt and
spec settings. Read: which sub-block grows in D and D2 relative to C, and by
how much of the 18 ms `decode_s` gap. If no sub-block grows under profiling,
the loss only exists unsynchronized and the candidates left are cross-kernel
(clock or power state, or dispatch overlap), which this protocol cannot
separate without a clock probe and is stated as such.

### Results 2026-09-15 (working tree, nothing committed)

Binaries (sha256 prefix): C `fc976cd9`, D `501fd19b`, D2 `614968b6`, J1
`e7ab45bd`. Receipts before timing (`tools/isa-receipt.py`, spills at
MR=8..1): C and J1 `603 526 458 380 312 235 167 81`; D `599 530 461 405 336
267 181 78`; D2 `573 504 435 366 297 227 155 78`. vgpr_count 192 everywhere.
Every run: `GENERATED` sha `c00468774758` (identical across all 64 runs,
including the 09-04 reference line), `mega fail word: 0`, `BARO_SPEC: True`,
`spec k: 2` where spec was on, `BARO_MEGA: False` in configuration S. GPU
queue empty before every launch; all launches through `gpu-wait run`.
Run logs: `.work/jsplit-review/runs*/`.

**Gate 2 per-kernel time, `ssm-kernel: delta` (s), 5 rounds alternating:**

| cfg | arm | median | min | max | spread | vs C | ranges |
|---|---|---|---|---|---|---|---|
| P (spec k=2, MR=3 windows) | C | 0.041477 | 0.041158 | 0.041810 | 1.59% | 1.000 | |
| P | D (JSPLIT=4) | 0.041045 | 0.040674 | 0.041451 | 1.91% | 0.990 | overlap |
| P | D2 (JSPLIT=2) | 0.040508 | 0.040127 | 0.040774 | 1.61% | 0.977 | **disjoint** |
| S (no mega, no spec, MR=1) | C | 0.041784 | 0.041520 | 0.041924 | 0.97% | 1.000 | |
| S | D (JSPLIT=4) | 0.042405 | 0.041798 | 0.043472 | 4.01% | 1.015 | overlap |
| S | D2 (JSPLIT=2) | 0.041533 | 0.041223 | 0.042276 | 2.55% | 0.994 | overlap |

The control's P spread (1.59%) is a hair over the 1.5% gate, so in P only
the disjoint D2 result is read as a direction. The 0.0478 s reference did not
reproduce (0.0415 today), as predicted: different engine.

**Prediction 1 (confound thesis): CONFIRMED.** With spills equalized the
split arms land within 2.3% of the control in both configurations. The 85%
regression recorded on 2026-09-04 was the generalization's codegen, not the
split.

**Prediction 2 (item 2, per-kernel time drops at MR=1): FALSIFIED for both
widths.** Configuration S: D +1.5%, D2 -0.6%, both overlapping the control.
Item 2 is a **confirmed null** at the shape it was written for. At MR=3
(configuration P) D2 is 2.3% faster with disjoint ranges; a kernel-level
receipt, one prompt, no end-to-end meaning on its own.

**Prediction 3 (tok/s moves under 1%): FALSIFIED.** Non-profiled pass, same
prompt, `tok/s_gen` median of 5: C 118.90 (118.12..118.98), D 114.99
(114.31..115.55), D2 115.00 (114.81..115.71). Both split arms **3.3% slower
end-to-end with disjoint ranges**; `decode_s` 0.5299 vs 0.5479 / 0.5478.

**Where the 3.3% lives.** J1 (original kernel body, tuple grid) reads 118.33
vs C 118.98, overlapping: the launch form is not it. `BARO_PROFILE=1`
sub-block attribution, median of 3 (s):

| sub-block | C | D | D2 |
|---|---|---|---|
| attn | 0.0481 | 0.0495 | 0.0495 |
| ssm | 0.1947 | 0.1993 | 0.1994 |
| ffn | 0.2218 | 0.2361 | 0.2330 |
| head | 0.0351 | 0.0350 | 0.0350 |
| decode_s | 0.5650 | 0.5857 | 0.5824 |

The **FFN sub-block, whose kernels are untouched, carries 11 to 14 ms of the
20 ms gap** (+5 to +6%); ssm +4.5 ms, attn +1.4 ms, head flat. The delta
kernel itself is not slower in isolation, so the loss is a carry-over from
its execution into the kernels that follow. Candidates: clock or power state
after a kernel that lights all 96 CUs instead of 32, or cache state left by
the different write order of the 2 MB state. Separating those needs a clock
probe, which this protocol does not have and which `docs/BASELINE.md` forbids
inside a timing loop. Stated as unresolved.

### Verdict 2026-09-15

Item 2 stays closed, and for a measured reason this time: the fair split is
flat at kernel level at MR=1 (null confirmed, both widths), 2.3% better at
MR=3 for JSPLIT=2, and **3.3% worse end-to-end** because the sub-blocks after
it slow down. The 2026-09-04 explanation (redundant `kq` fill, spill
pressure from the split) is withdrawn; its number was the refactor. Nothing
lands. The working tree keeps the `JSPLIT` machinery at `SSM_JSPLIT = 1`,
receipt-identical to `main` at every MR and within 0.5% on tok/s (J1), for
the maintainer to keep or drop.

## Carry-over probe 2026-09-15: is the 3.3% a clock artifact, a cache effect, or neither? (preregistered before the first timed run)

Brief `~/Brain/mojo/mojo-baro/briefs/2026-09-15-carryover-probe.md`. Question:
the 2026-09-15 re-run put 11 to 14 of the 20 ms end-to-end loss of the split
arms in the FFN sub-block, whose kernels the split never touches, measured with
`BARO_PROFILE=1` host syncs. This section decides between H1 (clock or power
state left by the 64-workgroup split kernel), H2 (cache state left by its write
order) and H3 (neither), and answers whether per-kernel (Gate 2 style) timing
can judge a launch-geometry change on this card. Nothing is fixed downstream.
Working tree only, nothing committed; `main` @ `a063d19`, tracked
`kernels/ssm.mojo` and `serve/registry.mojo` untouched (byte-identical to
`a9c0106`). Freeze receipt: sha256 of this file in
`.work/carryover/prereg.sha256`, stamped before the first launch.

### Arms (four binaries, sha256 prefixes read back per run)

- **C** `.work/jsplit-review/engine-C` (`fc976cd91f4d`), **D2**
  `.work/jsplit-review/engine-D2` (`614968b64033`): the 2026-09-15 binaries,
  unstamped, used only to reproduce their own recorded figures.
- **Cs** `.work/carryover/engine-Cs` (`e6b9875f42df`), **D2s**
  `.work/carryover/engine-D2s` (`568bdcd410c5`): built by
  `.work/carryover/mkprobe.py` from copies of `kernels/*.mojo` and
  `serve/*.mojo` under `.work/carryover/src-C` and `src-D2` (D2 takes
  `kernels/ssm.mojo` and `serve/registry.mojo` from `b31f7b5` with
  `SSM_JSPLIT = 2`, the same split the 09-15 D2 arm ran). Both carry the same
  stamp patch and nothing else:
  - an uncommitted copy `amar_matmul_skinny_q4rowb_st` of the q4 row GEMV,
    with `llvm.readsteadycounter` (100 MHz REALTIME) and
    `llvm.readcyclecounter` (20-bit SHADER_CYCLES) read at wave entry and after
    the row store; wave 0 lane 0 of every block does an atomic min of its start
    and an atomic max of its end into an 8-slot record per launch; block 0
    additionally stores its own start/end wall and cycle stamps. No barrier
    added. Dispatched only at the three trunk FFN sites of the launch path
    (gate, up, down GEMV), through `gemm_w_st` / `gemm_q4_st`; every other
    kernel, including the split kernel, is the tracked code.
  - the harness reads `/sys/class/drm/card1/device/gpu_metrics` (v1.3) and
    `pp_dpm_sclk` at three points OUTSIDE the decode window: before `t0`,
    after the prefill sync and before `t_prefill_end` is stamped (cost charged
    to prefill, ~20 us measured, printed per read), and after the decode-end
    sync once `gpu_total_s` is taken. No clock read of any kind happens inside
    the timed window; the in-kernel counters are wave-local register reads,
    the same class as the REALTIME stamps the megakernel already carries.
  - after the run, the stamp records are copied back and printed as `STAMP`
    lines (seq, window, layer, site, m, min start, max end, block-0 start/end,
    block-0 cycles at start/end).
- ISA receipts (`tools/isa-receipt.py`, `.work/carryover/isa-*.txt`):
  `amar_ssm_delta_step` spill ladder MR=1..8 is `81 167 235 312 380 458 526
  603` in C and Cs, `78 155 227 297 366 435 504 573` in D2 and D2s (same as
  the 09-15 receipts). The stamped GEMV at MR=3 is 157/159 VGPR, 0 spills,
  0 scratch against 156/158 unstamped (+1 VGPR, no occupancy step); the
  stamped MR=8 variant spills 43/45 vs 41/42 unstamped but MR=8 is never
  dispatched in this configuration (read back from the STAMP `m` column;
  any m outside {1,2,3} voids the run).

### Configuration (every run)

`BARO_SPEC=1 BARO_SPEC_K=2 BARO_PROMPT=bench/mtp-prompts/p09-explain-gpu.tokens`,
no profiling env, `BARO_MEGA` default (1): the verify windows (MR = 3) run
the launch path where the split kernel and the stamped FFN GEMVs live. Every
launch `gpu-wait run --vram 20 --`, `gpu-wait list` read empty before each,
GPU otherwise idle; `power1_cap` read back per run (290 W at freeze). Runs
alternate arms so drift lands on both. Read-back per run: `BARO_SPEC: True`,
`spec k: 2`, `BARO_MEGA: True`, `pack q4 trunk: True`, `mega fail word: 0`,
`GENERATED` sha `c00468774758` (the 09-15 identity), binary sha, `STAMPS`
count identical across stamped runs.

### Phases, gates, predictions (frozen)

**Phase R (reproduce, unstamped C / D2, 3 rounds alternating).** Gate: C
`tok/s_gen` median within 1.5% of the recorded 118.90, control spread under
1.5%, D2 below C with disjoint ranges. Prediction P0: D2/C = 0.967 +- 0.010.
If P0 fails nothing downstream may be read; the probe stops and reports.

**Phase S (stamped Cs / D2s, 5 rounds alternating).** Gates: Cs median within
1.0% of phase-R C (the stamps are free); D2s/Cs within 0.010 of D2/C, disjoint;
identity and read-backs above. Prediction P1: both hold. If the stamps cost
more than 1.0%, the atomic min/max (CAS loops on gfx11) are the suspect and the
instrument is rebuilt with block-0 start / last-block end before anything is
read.

**Decomposition (Phase S, device-side, no host syncs).** Per trunk window the
STAMP timeline gives: gate/up/down GEMV spans, the gap gate-end to up-start
(pure launch bubble), up-end to down-start (swiglu + bubbles), down-end to next
gate-start (residual add, next layer's norm and attn or SSM sub-block including
the split kernel, bubbles), and the window total. Sums per run, medians over
the 5 runs, D2s minus Cs in ms, ranges. The 09-15 FFN attribution predicts
11 to 14 ms more in the D2s FFN region (GEMV spans + g2u + u2d) per run.

- **H1 (clock/power carry-over) signature:** GEMV spans longer in D2s AND the
  block-0 shader clock (cycles / wall, per launch, median) lower in D2s by a
  matching fraction (at least 3%), or the post-decode `avg_gfxclk` lower in
  D2s outside the run-to-run range of Cs.
- **H2 (cache state) signature:** GEMV spans longer in D2s with the block-0
  clock equal within 1% (more cycles, same clock).
- **H3 (neither) signature:** GEMV spans equal within 1%; the difference sits
  in the gaps or in the down-end to next-gate region, or is absent from the
  window timeline altogether (then the `BARO_PROFILE=1` FFN attribution was
  itself an artifact of the sub-block syncs).
- Prior, stated for the record: H2 or H3 more likely than H1, because the head
  sub-block (also a q4 GEMV stream) was flat in the 09-15 profile. No
  magnitude predicted for any hypothesis.

**Falsifiers.** P0 fails: the 3.3% does not reproduce today and the question
is moot until it does. P1 fails: instrument perturbs the stream, rebuild. A
`wrap_unsafe` block-0 lifetime (over 300 us) in more than 1% of launches: the
cycle ratio is not read for that site. Any run with a nonzero fail word, a
different `GENERATED` sha, or a queue that was not empty at launch is VOID and
re-run.

**Answer format.** Gate-2 per-kernel timing can judge a geometry change on
this card only if the whole D2s minus Cs difference lands inside the split
kernel's own span (it cannot, by the 09-15 Gate 2 result, which showed the
split kernel faster); the probe reports where it lands instead and what
instrument sees it. MSPEC: affected iff its window attribution rests on
per-kernel or host-synced sub-block timing of a stream whose kernels changed
geometry; the report says how to tell, and touches nothing in MSPEC.

### Amendment 1, 2026-09-15, before the first timed run: stamp scheme

The smoke run of the first stamped build (numbers not read as results)
showed the block-0 cycles / wall ratio inconsistent across sites (3.1 to
4.5 GHz), and the two int64 atomic min/max lowered to CAS loops. Both stamped
engines were rebuilt before any timed run with: an explicit
`s_waitcnt(0)` before the end stamps; per block, wave 0 adds its own wall
span and its own 20-bit-masked cycle span into two per-launch sums with
native `global_atomic_add_u64` (blocks with a wall span over 300 us are
excluded and counted); the launch end is the end stamp of the last block to
arrive at a per-launch counter; block 0 stores the launch start and its own
spans. The shader clock per launch is therefore the lifetime-weighted mean
over all blocks (1536 or 512 samples), not one wave. STAMP layout: start,
end, sum_rt, sum_cyc, arrivals, b0_rt, b0_cyc, unsafe. New shas: Cs
`e5e95f6f9c00`, D2s `7fdd2a0a5428`. Arrivals must equal the grid (1536 for
gate/up, 512 for down) on every launch; the H1/H2 clock signature now reads
the summed ratio, thresholds unchanged. Nothing else in the preregistration
changes.

### Results 2026-09-15 (carry-over probe, working tree)

Phase R: C 118.89 tok/s_gen (118.82..118.98), D2 115.96 (115.40..116.16),
ratio 0.975, disjoint; P0 holds. Phase S: Cs 116.14 (115.40..116.33), D2s
113.85 (113.69..114.20), ratio 0.980, disjoint; the stamps cost 2.3%, over the
1.0% gate, and the arm difference is preserved (deviation and reasoning in
`exchange/2026-09-15-carryover-probe.md`). Decomposition, D2s minus Cs per
run: gate GEMV +2.34 ms, up -0.11, down +4.15, gaps flat, rest of layer
+5.08, windows +11.95, decode +10.9. Shader clock during the GEMVs equal
within 0.6% (in-kernel), sysfs gfxclk 3020 vs 3026 MHz, power 304 vs 303 W at
the 290 W cap. **H1 falsified; H2-class cost (more cycles at equal clock in
untouched kernels), mechanism not identified. Gate-2 per-kernel timing cannot
judge a geometry change; the untraced tok/s and the in-stream device timeline
can.** Report: `exchange/2026-09-15-carryover-probe.md`.

## MSPEC precondition check (M3, wiring lane, 2026-09-15, preregistered before the first timed run)

Question (brief `briefs/2026-09-15-wiring-lane.md` M3): does an untraced
in-stream gap-share measurement of the MSPEC spec-verify window fall inside
the 8.4-9.2% corrected bound (`bench/mtp-protocol.md`, MSPEC step 1 traced
15.65% at T/B tracer cost 0.93, corrected two ways: 8.35% charging all extra
decode time to gaps, 9.21% charging only the windows' dispatch share)? Kill
line for MSPEC step 2: g < 10%.

**Instrument.** `bench/carryover-stamp.py`'s `SITES` list, extended from the
carry-over probe's 3 FFN GEMVs (gate/up/down) to all 13 `gemm_w[...]` call
sites reached by one decode-loop window on the qwen35 `MEGA_ALLOWED` profile
(`serve/window.mojo:842-1290`): the 3 FFN sites plus 4 attn-subblock GEMVs
(qf, k, v, output) and 5 ssm-subblock GEMVs (qkv, z, a, b, out), mutually
exclusive per layer on `is_attn(layer)` (`N_LAYERS`/`N_ATT`/`N_SSM` =
32/8/24), plus the once-per-window head/logits GEMV. `NSLOT` raised
8192 -> 32768 (13-site stamps/window ~249 vs the 3-site ~21; a 64-token k=2
decode needs ~8715 slots at 13 sites, over the old cap with no bounds check
in `stamp_slot()`).

**Decision rule (the maintainer, 2026-09-15, verbatim):** gap_share = (window -
sum_kernel_spans) / window. Stamping a SUBSET of the window's kernels
underestimates sum_kernel_spans and therefore OVERESTIMATES gap -- every
unstamped kernel's real execution time is charged to "gap". So the SITES
subset above (gemm_w-class only; everything else -- `rmsc_k`, `split_k`,
`hrms_q`/`hrms_kv`, `rope_q`/`rope_k`, `append_k`, `datt_k`/`att_k`,
`dcomb_k`, `gmul_k`, `r_add`, `rgates_k`, `conv_k`, `l2_k`,
`delta_dispatch`, `gated_k`, `r_swiglu`, `argmax_d`/`argmax_k`, `tokcp_k`,
`embed_k`, ~25 kernels total -- charged to gap) yields an UPPER BOUND on the
true untraced gap share, never a lower one. The GEMVs carry the weight
streaming and should dominate the window, so the bound has a real chance of
landing under 10%; if it does not, that is information too, not a failure.

**Verdict rule:** measured upper bound < 10% -> MSPEC's correction is
validated on measured (untraced) ground, the kill line is cleared, step 2
may proceed (starting it is still not this lane's to do). Measured upper
bound >= 10% -> **INCONCLUSIVE**, not a refutation: report which of the ~25
unstamped kernel types would need stamping next to tighten the bound, and
stop.

**Run.** `bench/carryover-run.sh S` (arms Cs/D2s, 5 rounds each, prompt
`bench/mtp-prompts/p09-explain-gpu.tokens`, `BARO_SPEC=1 BARO_SPEC_K=2`,
`GEN_N=64` default); `bench/carryover-analyze.py --site-names
gate,up,down,att_qf,att_k,att_v,att_out,ssm_qkv,ssm_z,ssm_a,ssm_b,ssm_out,head`.
Gap share per run = 1 - sum(all 13 sites' spans) / window span, read from the
`sum_win_ms` and the 13 `sum_<site>_ms` fields already reported per arm.
Identity gate unchanged from the carry-over probe (every run's `GENERATED`
hash must match). Only the **Cs** arm (plain checkout) is read for the gap
share verdict; D2s is carried along for free (same run) but is not part of
this question.

### Result 2026-09-15 (Cs arm, 5 runs, working tree)

Identity PASS 10/10 (both arms), 0 WARN, `fail=0` every run, `STAMPS 8715`
every run (matches the predicted ~249 stamps/window x ~35 windows). Median
over 5 runs, Cs arm: `sum_win_ms` 479.08 (478.30..532.33, one run's tail
outlier in `sum_up_ms` -- 100.51 vs a 56.5x median on that one run, median is
robust to it). Per-site medians (ms): ffn (gate+up+down) 184.59, attn
(qf+k+v+out) 24.77, ssm (qkv+z+a+b+out) 89.65, head 31.66 -- sum of all 13
sites **330.67 ms**.

**gap_share_upper_bound = 1 - 330.67/479.08 = 0.310 (31.0%).**

**Verdict: INCONCLUSIVE per the decision rule (upper bound >= 10%).** This
does not refute the 8.4-9.2% corrected bound -- an upper bound above the kill
line says nothing about where the true value sits, only that this
measurement cannot clear the line. It also does not surprise: the 13
`gemm_w`-class sites cover 249 of ~678 dispatches per window (~37% by
count), and 31% > MSPEC's own traced 15.65% (tracer overhead included) >
its corrected 8.4-9.2%, which is the expected direction -- fewer measured
kernels, higher apparent gap, monotonically.

**Next kernels to add, ranked by expected contribution (not measured, judged
by what each kernel actually does over how much state):** `delta_dispatch`
(the SSM scan itself, i.e. the one truly serial per-token recurrence in
the SSM sub-block, run on all 24 ssm layers -- the single largest suspect,
since the already-stamped `ssm_*` GEMVs are only the projections around it,
89.65 ms, not the scan); `datt_k`/`att_k` (the attention kernel proper, on
all 8 attn layers, analogous gap to `delta_dispatch`); `rope_q`/`rope_k`,
`hrms_q`/`hrms_kv`, `append_k` (attn-branch per-head ops, smaller but
numerous); `conv_k`, `l2_k`, `gated_k`, `rgates_k` (ssm-branch elementwise,
smaller); `rmsc_k` (rmsnorm, runs every sub-block boundary, many small
launches); `r_swiglu`, `r_add`, `gmul_k`, `split_k` (cheap elementwise,
lowest priority); `argmax_d`/`argmax_k`, `tokcp_k`, `embed_k` (once or twice
per window, small). Stamping `delta_dispatch` and `datt_k`/`att_k` next is
the highest-yield single addition, since between them they are the only
unstamped kernels that scale with sequence/state length rather than being
O(1) elementwise.

**Stopping here per the decision rule.** Not this lane's call whether to
extend SITES further or scope a smaller round for it; report:
`exchange/lane-WIRING-report.md`.
