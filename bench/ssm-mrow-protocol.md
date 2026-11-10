# SSM per-row floor probe — frozen before the first timed run

> **Binding: [`bench/PROTOCOL-RULES.md`](PROTOCOL-RULES.md).** P1 (read-back
> receipts), P4 (decode verdicts are 20-prompt medians), P5 (row scaling),
> P6 (harness before kernel).

Milestone item 1 of `~/Brain/mojo-baro/2026-09-05-engine-next-milestone.md`.

## Question

Of the SSM sub-block's per-window cost at m=2, how much is sequential by
construction? `amar_ssm_delta_step[MR]` carries `for r in range(MR)` inside
one kernel (`kernels/ssm.mojo:193`): the state update of row r+1 reads the
state row r wrote, so its m=2 cost cannot fall below 2x its m=1 cost without
changing the recurrence. Every other stage of the sub-block (the four q8
GEMMs + two reduces, `rgates`, `conv`, `l2`, `gated`, the out GEMM + add) is
row-parallel and should be far closer to 1x.

The answer sets the ceiling for milestone item 2 (multi-row window
1.28x -> 1.1x): the sequential part is a floor no window rework can remove.

## Instrument

`serve/engine.mojo` `BARO_PROFILE=2`, which already synchronizes at the seven
SSM stage boundaries and prints `ssm-kernel: <name> <s> <share>` for
`gemm4+reduce2 / rgates / conv / l2 / delta / gated / out_gemm+add`. No new
kernel and no new bench binary (P6): the split needed for this question
exists.

`BARO_PROFILE=2` inserts a `ctx.synchronize()` between stages, so the stage
times are serialized and their SUM exceeds the unsynchronized sub-block time.
This probe compares stages **within one profiled arm** and compares the SAME
stage across m, both under identical synchronization. Absolute per-stage
milliseconds from this probe may not be added into any end-to-end budget, and
no `tok/s` figure from a `BARO_PROFILE=2` run may be quoted anywhere.

## Arms

Both arms: same binary, same pack, same prompt, `BARO_PROFILE=2`.

- **A (m=1)**: `BARO_SPEC=0`. Every window is one row.
- **B (m=2)**: `BARO_SPEC=1`, `BARO_SPEC_K=1` -> `m = kcfg + 1 = 2`.

Prompt-phase windows also run at m>1; the arms are compared on the decode
phase only, which for this prompt dominates the window count.

## Receipts required before either arm's numbers are recorded (P1)

Read back from the run's own stdout, not from the command line:
`pack loaded in`, `prompt tokens:`, `tokens:`, `spec k:` (arm B must print
1, arm A must not report spec windows), and the engine binary rebuilt in the
same command as the run. Recorded alongside: `gpu-wait gpu` VRAM/power before
the run, and the fact that the CoiOS bge-m3 embedding server is resident
(~3 GB, idle) — it is a deviation from `ssm-occupancy-protocol.md`'s
"GPU exclusive" instrument and is disclosed rather than hidden.

3 repeats per arm, drop the first, median of 2 reported per stage; spread
gate <5% on the delta stage (the quantity under test). A wider spread voids
the arm.

## Predictions — frozen at commit time, before any run

1. `delta` ratio B/A >= 1.8. This is the mechanism claim; the loop is
   sequential and each row's arithmetic is identical.
2. Every row-parallel stage's B/A ratio <= 1.5, and `rgates` + `l2`
   specifically <= 1.2 (their grids are m-independent or nearly so).
3. `delta` share of the profiled sub-block rises from arm A to arm B.

## Decision rule — what the answer buys

Let `d` = delta B/A ratio and `S_delta` = delta's share of the m=2 profiled
sub-block.

- **Sequential-dominated**: `S_delta >= 0.35` and `d >= 1.8` -> the +1.06 ms
  at m=2 is mostly floor; milestone item 2's ceiling is **~105 tok/s**, and
  item 2 is rescoped to the non-delta stages only.
- **Recoverable**: `S_delta <= 0.20` -> the loss is outside the state update;
  item 2's ceiling stands at **115-120 tok/s** and it proceeds as written.
- **Mixed** (anything between): item 2 proceeds but its predicted band is
  restated as 105-115 before it is run, not after.

Prediction on the outcome itself, so it is falsifiable: **mixed**, with
`S_delta` between 0.20 and 0.35 and `d` near 2.0.

## What this probe cannot answer

It does not measure whether a different recurrence formulation (chunked scan,
associative scan over rows) would beat the sequential floor. That is a
separate item; this one only prices the floor as the kernel is written.

## Run record — 2026-09-05, HEAD f37815b (clean tree)

Binary: `.work/engine`, built this session from the same tree
(`./.venv/bin/mojo build serve/engine.mojo -I kernels -o .work/engine`, exit 0);
the only commit between build and run is this protocol file. Disclosed
deviation from P1's "rebuilt in the same command as the run".

Instrument: `gpu-wait run --priority 90 --vram 12`,
`MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=10`, GPU at 42 C / 31 W /
2.97 GB used before the first run — the CoiOS bge-m3 embedding llama-server is
resident and idle throughout, in both arms (disclosed above).

Receipts read back from stdout:

| | arm A | arm B |
|---|---|---|
| `BARO_SPEC` | False | True |
| `spec k` | 2 (unused, spec off) | 1 |
| `prompt tokens` | 5 | 5 |
| `tokens` | 64 | 64 |
| `mtp:` | absent | `drafted 32 accepted 32 k 1` |
| windows normalized by | 63 | 32 |

`accepted 32 / drafted 32` on this prompt means arm B is 32 windows of m=2
exactly, so the per-window normalization is measured, not assumed.

3 runs per arm, first dropped, median of 2. Delta-stage kept runs:
A 43.457 / 45.264 ms, B 33.751 / 32.480 ms — spread 4.1% and 3.9%, inside the
5% gate.

Per-window SSM stage time (ms), and the ratio:

| stage | A (m=1) | B (m=2) | B/A |
|---|---|---|---|
| gemm4+reduce2 | 3.6094 | 3.4014 | 0.94 |
| rgates | 0.6662 | 0.8974 | 1.35 |
| conv | 0.4575 | 0.4866 | 1.06 |
| l2 | 0.4398 | 0.4646 | 1.06 |
| **delta** | **0.7041** | **1.0349** | **1.47** |
| gated | 0.4415 | 0.4751 | 1.08 |
| out_gemm+add | 1.1357 | 1.1937 | 1.05 |
| total (profiled, serialized) | 7.4543 | 7.9537 | 1.07 |

`S_delta(m=2)` = 0.130, `d` = 1.47.

### Verdict

**Prediction 1 is FALSIFIED.** `d` = 1.47, not >= 1.8. The row loop is
sequential in program order but not in cost: the state columns are loaded
into registers once per head and row r+1 reuses them, so the second row adds
arithmetic and the second state write, not a second state read. The floor is
lower than the recurrence's structure suggested.

Prediction 2 holds except for `rgates` (1.35 vs the <=1.2 stated). `rgates`
launches `grid_dim=1, block_dim=NH_V=32` — a single wave; at that size the
number is launch overhead, not work, and it should not have been given a
tight band. Prediction 3 holds (delta share 0.094 -> 0.130).

The outcome prediction ("mixed, `S_delta` in 0.20-0.35, `d` near 2.0") is also
wrong. By the frozen decision rule, `S_delta` = 0.130 <= 0.20 is
**recoverable**: milestone item 2 keeps its 115-120 tok/s ceiling and proceeds
as written.

Stronger result than the rule asks for: the whole SSM sub-block costs only
1.07x per window at m=2 while doing twice the rows. The m=2 loss is not in the
SSM sub-block at all, which is what item 2 assumed and this probe now
supports rather than merely permits.

## Re-run — same day, clean GPU, HEAD 4510aa7

`coios-embed.service` (the resident bge-m3 embedding server) was disabled and
its waiting-room job cancelled, removing the one disclosed deviation from the
first run record. GPU before: 1.43 GB used, 18 W, 47 C, 3% busy, no other job
in the queue.

P1 gap from the first run also closed: each arm's script rebuilds the engine
and prints `build exit: 0` plus `sha256sum .work/engine` in the same command as
the run. Both arms, all six runs: `ed178a7bad2a05c38263f732ac78d07bb531c9d4e5ec157d8c1c189338264274`.
Arm identity re-read per file — three runs with `BARO_SPEC: False`, three with
`BARO_SPEC: True` + `accepted 32`.

Delta-stage kept runs: A 45.182 / 43.950 ms (spread 2.8%), B 33.398 / 32.721 ms
(2.1%).

| stage | A (m=1) | B (m=2) | B/A | B/A, contaminated run |
|---|---|---|---|---|
| gemm4+reduce2 | 3.0430 | 3.4353 | 1.13 | 0.94 |
| rgates | 0.6722 | 0.8896 | 1.32 | 1.35 |
| conv | 0.4564 | 0.4819 | 1.06 | 1.06 |
| l2 | 0.4403 | 0.4648 | 1.06 | 1.06 |
| **delta** | **0.7074** | **1.0331** | **1.46** | 1.47 |
| gated | 0.4465 | 0.4686 | 1.05 | 1.08 |
| out_gemm+add | 1.1368 | 1.1885 | 1.05 | 1.05 |
| total (profiled, serialized) | 6.9026 | 7.9617 | 1.15 | 1.07 |

`S_delta(m=2)` = 0.130, `d` = 1.460. **Verdict unchanged**: prediction 1 stays
falsified, the decision rule still reads **recoverable**, item 2 keeps its
115-120 tok/s ceiling. These are the numbers to cite; the first run record is
kept for the contrast below.

### What the resident embedding server was doing to the numbers

It moved exactly one stage and only in the m=1 arm: `gemm4+reduce2` at m=1 was
3.6094 ms/win contaminated vs 3.0430 clean, **+18.6%**. The four q8 GEMMs are
the only bandwidth-bound stage in the sub-block, so an idle-but-resident model
holding ~1.1 GB and its share of Infinity Cache lands there and nowhere else —
`conv`, `l2`, `gated` and `out_gemm+add` reproduce to within 3%.

The distortion inflated the m=1 baseline, which made the m=2 sub-block look
*cheaper* than it is: total B/A 1.07 contaminated vs 1.15 clean. It flattered
the arm under test. Nothing here was load-bearing for the verdict, since `d`
and `S_delta` both moved <1%, but a contaminated baseline biasing toward the
conclusion is the failure mode P1 exists to catch, and it was caught by
re-running rather than by the spread gate — both contaminated arms passed
their spread gates comfortably.

## Round 2: row scaling at m = 1, 2, 4, 8 (preregistered 2026-09-16, before any GPU run)

Brief `~/Brain/mojo/mojo-baro/briefs/2026-09-16-fable-delta-row-scaling.md`; `docs/A3-PLAN.md`
"the throughput gate, corrected" asked for exactly this receipt before any batching target N.

### Instrument

Same engine binary, rebuilt in the run script and sha256-printed in the same command as every
arm (P1); pack `.work/engine-pack-q4`; prompt `bench/mtp-prompts/p14-history.tokens` (7 tokens,
so the one prompt-replay window of 6 rows is under 2% of the windows counted); 64 generated
tokens. Arms by `BARO_SPEC_K`: m = 1 is `BARO_SPEC=0`; m = 2, 4, 8 are `BARO_SPEC=1` with
k = 1, 3, 7 (the verify window is m = k + 1 rows on every decode step; `delta_dispatch`
instantiates `amar_ssm_delta_step[m]` for every m up to 8, the same kernel). Read-backs
per run from stdout: `BARO_SPEC`, `spec k`, `mtp: drafted D accepted A k K`, `tokens`,
`prompt tokens`, the engine sha. Windows per run: m = 1 has 63; m > 1 has D / K.

Three profile modes, each its own run because each synchronises differently:
`BARO_PROFILE=2` (the seven SSM stages, as round 1), `BARO_PROFILE=4` (the six FFN stages),
`BARO_PROFILE=1` (sub-block totals attn, ssm, ffn, head). Per stage: milliseconds per window
= printed seconds / windows. 3 repeats per (m, mode), first dropped, median of the remaining
two, spread of those two reported; spread over 5% on the delta stage voids that arm. The
serialized sums are compared within a mode across m, never quoted as tok/s (round 1's rule).
Cross-check on the quantity under test: one `rocprofv3 --kernel-trace` run per m, device
time per launch of `amar_ssm_delta_step` as the median over its launches (24 per window).

Round 1's m = 2 row is reproduced first: delta B/A must land in 1.46 +/- 5% (1.39 to 1.53)
on the clean-GPU receipt, else the lane stops and reports.

### Predictions, frozen (linear model c(m) = a + b m fitted to round 1's clean m = 1 and 2, ms per window)

| stage | m = 1 (r1) | m = 2 (r1) | m = 4 predicted | m = 8 predicted | band |
|---|---|---|---|---|---|
| delta | 0.707 | 1.033 | 1.69 (2.4x) | 2.99 (4.2x) | +/- 25% |
| gemm4+reduce2 | 3.043 | 3.435 | 4.22 (1.4x) | 5.79 (1.9x) | +/- 25% |
| conv | 0.456 | 0.482 | 0.53 | 0.64 | +/- 25% |
| l2 | 0.440 | 0.465 | 0.51 | 0.61 | +/- 25% |
| gated | 0.447 | 0.469 | 0.51 | 0.60 | +/- 25% |
| out_gemm+add | 1.137 | 1.189 | 1.29 | 1.50 | +/- 25% |
| rgates (one wave, launch-bound) | 0.672 | 0.890 | 1.32 | 2.20 | +/- 50% |
| SSM serialized total | 6.90 | 7.96 | 10.1 (1.46x) | 14.3 (2.07x) | derived |

Per-row efficiency of the SSM sub-block (m / cost ratio): 1.74 at m = 2, 2.7 at m = 4, 3.9 at
m = 8. Whole decode step from the mode-1 totals: T(2)/T(1) 1.15 to 1.30, T(4)/T(1) 1.5 to 1.8,
T(8)/T(1) 2.2 to 3.0, so the aggregate tokens-per-step ceiling relative to single-stream is
1.5 to 1.7x at m = 2, 2.2 to 2.7x at m = 4, 2.7 to 3.6x at m = 8. FFN stages: the GEMMs
scale like gemm4 (weights dominate), the elementwise ones like conv.

### Decision rule, frozen

- **A2 + A3 as planned** if delta at m = 4 is at most 2.6x its m = 1 cost AND delta's share of
  the serialized SSM sub-block at m = 8 is at most 0.30.
- **Delta kernel round precedes A3** if either bound is crossed: the recurrence then dominates
  the batch and its per-row cost is what A3's target N would be paying for.
- Falsifier of the model itself: any row-parallel stage (not delta, not rgates) more than 1.3x
  its linear prediction at m = 4 means the linear extrapolation is wrong and the aggregate
  ceiling above is withdrawn rather than corrected after the fact.

Measurement only: no kernel changes in this lane. GPU budget: 36 profiled runs plus 4 traced
runs, about 3 s each on the q4 pack, under 3 GPU minutes.

### Round 2 result (2026-09-16): delta is not the batching limiter, the m = 8 GEMMs are; A2 + A3 as planned with N = 4

Receipts `.work/mrow2/` (`bench/ssm-mrow-run.sh`, `bench/ssm-mrow-summarize.py`). Engine
sha `808243c5...`, built and printed in the same command; every run read back
`BARO_MEGA: False`, `BARO_SPEC`, `spec k`, `mtp: drafted/accepted`, `tokens: 64`,
`prompt tokens: 7`. Windows: m = 1 63; m = 2 35 (k 1); m = 4 23.7 (k 3); m = 8 20.4 (k 7).
Every delta-stage spread under 1% (bar 5%). Power cap 290 W read back.

**Two void runs before the valid one, both disclosed.** The dense engine now takes the
megakernel at m = 1 by default (2026-09-06, after round 1), so an m = 1 arm without
`BARO_MEGA=0` runs one launch per token and the stage timers see only the prompt-replay
window: the first run's m = 1 stage times were 20x too small. `BARO_MEGA` was not in the
read-back list; it is now, and the summarizer voids any run without `BARO_MEGA: False`. The
second run was void because the fix script aborted before writing. The valid run is the third.

**Round 1 reproduced** on the m = 2 row: delta 1.068 / 0.748 = 1.43x (band 1.39 to 1.53).

Per-window ms, mode 2 (SSM stages, serialized), ratio vs m = 1, per-row efficiency m / ratio:

| stage | m = 1 | m = 2 | m = 4 | m = 8 |
|---|---|---|---|---|
| gemm4+reduce2 | 2.376 | 2.945 1.24x (1.61) | 3.864 1.63x (2.46) | 10.774 4.53x (1.76) |
| rgates | 0.713 | 0.910 1.28x | 1.363 1.91x | 2.269 3.18x |
| conv | 0.487 | 0.491 1.01x | 0.531 1.09x | 0.578 1.19x |
| l2 | 0.463 | 0.467 1.01x | 0.511 1.10x | 0.556 1.20x |
| **delta** | 0.748 | 1.068 **1.43x** (1.40) | 1.568 **2.10x** (1.91) | 2.460 **3.29x** (2.43) |
| gated | 0.470 | 0.473 1.01x | 0.520 1.11x | 0.576 1.23x |
| out_gemm+add | 0.903 | 1.073 1.19x | 1.268 1.40x | 3.294 3.65x |
| SSM serialized total | 6.161 | 7.428 1.21x (1.66) | 9.625 1.56x (2.56) | 20.507 3.33x (2.40) |
| delta share | 0.037 | 0.045 | 0.048 | 0.029 |

Mode 4 (FFN stages): rmsnorm, swiglu, r_add flat to 1.1x through m = 8; gemm_gate 1.25 /
1.74 / **5.94x**, gemm_up 1.24 / 1.53 / **5.79x**, gemm_down 1.26 / 1.85 / **5.77x** at
m = 2 / 4 / 8. Mode 1 (sub-block totals, the aggregate basis):

| | m = 1 | m = 2 | m = 4 | m = 8 |
|---|---|---|---|---|
| attn | 1.193 | 1.410 1.18x | 1.755 1.47x | 4.240 3.56x |
| ssm | 3.973 | 5.342 1.34x | 7.464 1.88x | 18.337 4.62x |
| ffn | 5.323 | 6.721 1.26x | 9.098 1.71x | 33.399 6.27x |
| head | 0.833 | 0.999 1.20x | 1.267 1.52x | 5.869 7.04x |
| step total | 11.322 | 14.473 1.28x | 19.584 1.73x | 61.845 5.46x |
| **aggregate ceiling vs single stream** | 1.00 | **1.56** | **2.31** | **1.46** |

Cross-check, rocprofv3 device time per launch of `amar_ssm_delta_step[m]` (median over
launches; the m = 1 trace is partial, 867 of 1512 launches, because the traced engine hung at
exit and was killed after 10 minutes on the coordinator's call; m = 2, 4, 8 traces complete):
**10.68 / 21.48 / 41.24 / 75.48 us at m = 1 / 2 / 4 / 8, ratios 2.01 / 3.86 / 7.07.** The
kernel's device time is linear in m. The host-synced stage timer reads 31 us per delta launch
at m = 1 (0.748 ms / 24), so two thirds of round 1's "delta" stage was launch and sync
overhead, and its 1.46x was that overhead amortising, not state columns being reused. Round
1's mechanism explanation is withdrawn; its verdict (delta is a small share of the sub-block)
stands because the kernel is 11 to 75 us against a 4 to 18 ms sub-block.

**Predictions:** delta at m = 4 predicted 1.69 ms (measured 1.568, in band) and at m = 8
2.99 (measured 2.460, in band); gemm4 at m = 4 4.22 (3.864, in band); **at m = 8 every
weight GEMM is 1.8 to 2.4x its linear prediction, so the model's falsifier fired for m = 8
and the linear aggregate ceiling at m = 8 (2.7 to 3.6x) is withdrawn.** The measured ceiling
at m = 8 is 1.46x, below m = 4's 2.31x and barely above m = 2's 1.56x.

**Decision rule, as frozen:** delta at m = 4 is 2.10x (bound 2.6x) and its share at m = 8 is
0.029 (bound 0.30): **A2 + A3 as planned; no delta round before A3.** Recommendation added
by the measurement: **target N = 4, not 8.** At m = 8 the q4 weight GEMMs (`gemm_w`, all
five per layer, plus the head) lose their per-row efficiency (1.35 to 1.8 rows per m = 1
cost against 2.3 to 2.6 at m = 4); that is a GEMM question at the SM = 8 tile, not an SSM
one, and it is the preregistered round that precedes any N = 8 target.
