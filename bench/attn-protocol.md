# Attention protocol — lane-attn (2026-09-08)

Binds to `bench/PROTOCOL-RULES.md` P1-P6. Bars come from
`bench/chat-protocol.md` M0c/M0d. Each round's prediction is frozen and
committed before its build is timed; the Result is written under it.

## Round A — in-block decode attention: both halves of the 512-thread block (frozen 2026-09-08, before its build)

**Mechanism, read from source.** On the split path (`attn_phases`,
`kernels/mega.mojo`, `do_split` at T > att_split), every one of the 96
blocks walks its page span with `attn_head_span`: chunks of HD = 256
positions, one position per thread, threads 256..511 idle in both the K
phase and the V phase, four `barrier()` per chunk. The megakernel runs one
block per CU (vgpr 249 -> 1 block/CU, `persistent-kernel-gfx11`), so during
attention each SIMD holds ONE active wave: every dependent FMA of the
256-deep scalar dot chain and every K/V load stalls its full latency with
nothing to interleave. M0d's autopsy: the split cut attention 3.3-3.7x of a
possible 6x; the remainder is this loop plus the merge barrier (~0.3 ms).

**Change.** `attn_head_span2` (new, `kernels/attn.mojo`), used only when
`do_split`: one iteration covers 2*HD = 512 positions, thread tid scores
position t0 + tid, each 256-thread half keeps its own online-softmax stream
(m, l, o[HD]) over its 256-position sub-chunks with the SAME K-dot form,
exp/rescale form and 8-wide V loop as `attn_head_span`; wave reductions go
through `red[8]` (per half: its four waves). Four barriers per 512
positions. After the loop the upper half parks (m1, l1, o1) in LDS and the
lower half merges the two streams in the split-K merge form (lower first).
An empty sub-chunk (span shorter than 256, odd chunk count) skips the
stream update; barriers stay unconditional. Call site: `attn_phases` picks
span2 under `do_split`, its `scores` LDS grows HD -> 2*HD floats, `sums`
4 -> 8. `attn_head_span`, the exact path (T <= att_split) and
`amar_attn_decode` are byte-for-byte untouched. The dot form is not
re-spelled (kernel-parity rule 1); it is the next pool, not this round.

**Predictions.**
- P-F1 Fingerprint before GPU: q4 `amar_mega_token` vgpr <= 256, spill 0,
  `isa-loops` dual >= 105 / 80 +- 2 / 80 +- 2 / 60 +- 2 (fast class; M0d
  reads 122/84/84/61). The instruction count grows (new function on the
  split path) and is recorded, not banded (M0d deviation). One re-roll by
  spelling; second miss stops the round.
- P-F2 Identity. Default path (split off at T <= 1088): one-shot q4 and q8
  GENERATED bit-exact vs the same-stint HEAD-built reference
  (`.work/engine-ref`), 20-prompt A/B identity 20/20. Forced split
  (`BARO_ATT_SPLIT=1`) at T <= 1088: q4 64/64 and q8 64/64 vs
  `tools/model-ref.py` reference tokens, 20-prompt identity 20/20 vs the
  unsplit path of the same binary (two streams merge: identity, not
  bit-exact). `test_mega_block` (ATT_SPLIT = 0, T = 10, one non-empty
  span, one non-empty half): bit-identical to the launch path (the merge
  weights are exactly 1 and 0). `test_attn_block` PASS (untouched kernel).
- P-F3 Long-context decode, M0d's prompts, `BARO_TMAX=102400`, default
  threshold: 8192 >= 120 tok/s_gen (M0d 117.5; attention 1.0 -> ~0.7 ms),
  32768 >= 95 (M0d 85.0; attention 4.3 -> ~2.6 ms: loop 4.0 -> 2.1-2.5 at
  1.6-1.9x, merge 0.3 unchanged), 100000 >= 62 (M0d 48.7; attention 13.0
  -> ~7.5 ms). Hard floor 32768 >= 90: below it the loop is not the
  latency-bound remainder the autopsy claims (then measure the merge
  barrier and the per-iteration cost before any further in-block change).
  GENERATED at 8192 and 32768 equal to M0d's (`.work/ref/long-*.log`);
  100000 recorded as observation (M0d already differs from M0c there); a
  mismatch at 8192 or 32768 is arbitrated by `tools/model-ref.py` on 8192.
- P-F4 20-prompt median (default path) within +-2 % of the reference
  binary (path unchanged at T <= 1088; only the fingerprint can move it).
- P-F5 `mega fail word: 0` on every run; prefill_s unchanged at 8k/32k/100k
  (prefill attention is not touched in Round A).

**Verification before timing (P1).** Fingerprint first. Every timed arm is
built in the same stint (`.work/engine-ref` from `git archive HEAD`,
`.work/engine-a` from the working tree; sha256 of both in the stint log);
`att split:`, `TMAX:`, `prompt tokens:`, `tok/s_gen`, `mega fail word` read
from each run's stdout; `arm.txt` (power cap, vddgfx, packs) written by
`bench/ab-prompts.sh` before the A/B; `gpu-wait list` pasted into the
status file before the timed stint.

**Gate.** P-F1 before GPU; P-F2 every receipt; P-F4 within band; P-F3
recorded against its predictions (32768 >= 90 is the hard floor, the three
numbers are the claim); P-F5.

### Amendment 1 — premise correction, frozen before any GPU run (2026-09-08 13:5x)

**The mechanism above is wrong at its root.** The ISA metadata of the
built engine (`tools/isa-receipt.py`, `amar_mega_token` q4) reads
`.wavefront_size: 32`, `.max_flat_workgroup_size: 256`: the megakernel block
is `ROW_WAVES = 8` waves of **wave32 = 256 threads**, and `if tid < HD`
(HD = 256) leaves no thread idle. The "512 threads, 256 idle" line of
`bench/chat-protocol.md` M0d assumed wave64 and is wrong (lane-chat owns that
file; the correction is recorded here, approved by the driver 13:4x). The
`attn_head_span2` two-halves design was built, fingerprinted (vgpr 251,
spill 0, dual 126/85/85/64) and **never run**: with 256 threads it would have
skipped every second 256-position sub-chunk. Its diff is kept at
`.work/span2-wrong-premise.patch`; it is not part of the round.

**The real bound, read from the M0d ISA** (`.work/isa-ref/co64.s`, the
attention K loop at 0x7E00 and V loop at 0x8178): the K dot is a runtime
loop of 32 iterations, each `2 x global_load_b128` -> `s_waitcnt vmcnt(1)`
-> 8 `v_fmac` — one 32-byte load pair in flight per wave, 32 memory
latencies per position; the V accumulate issues 8 `global_load_b32` then
`s_waitcnt vmcnt(7..0)` + 8 `v_fmac`, 32 iterations per chunk — 8 loads in
flight. Per chunk that is ~64 dependent memory-latency steps with 8 waves
per CU to interleave them. Measured M0d at 32k: 4.0 ms / (9 attention
layers x 21.3 chunks per span) = ~21 us per 256-position chunk = ~330 ns
per latency step. Latency-bound by construction, as the autopsy said; the
cause is memory-level parallelism, not thread count.

**Change (replaces the one above).** `attn_head_span` itself: the K dot
loads `AK_U = 8` vec8 K slices (256 B) per position into registers before
their FMAs (`for du in range(HD // 64)` + `comptime for u`), and the V
accumulate hoists `AV_U = 32` V loads + 32 LDS score reads per step before
their FMAs, with the 8-wide and 1-wide tails kept for remainders. The
arithmetic order is unchanged: every `acc += q8[j] * k8[j]` and
`o += sc[j] * v[j]` executes in the same sequence as before, so the
change is **bit-exact on both paths** (exact and split) and on
`amar_attn_decode` (launch kernel, `test_attn_block` / `test_prefill`),
which share the function. No LDS change, no call-site change.
Fingerprint of the built kernel (before GPU): q4 `amar_mega_token` vgpr
249 (unchanged), spill 0, instructions 12977 -> 13307, dual 122/84/84/61
(unchanged loop class); the K loop now issues 7 `global_load_b128` under
`s_clause` and steady-states at `vmcnt(6)` (~7 loads in flight), the V
loop issues 32 `global_load_b32` and waits `vmcnt(31)` downward (32 in
flight). Latency steps per chunk ~64 -> ~13.

**Predictions (replace P-F3/P-F4 above; P-F1, P-F2, P-F5 stand).**
Unique KV bytes per decoded token = T x 9 layers x 4 kv heads x 2 x 1 KB
= T x 73.7 KB: 604 MB at 8192, 2.42 GB at 32768, 7.37 GB at 100000; at
800 GB/s that floor is 0.75 / 3.0 / 9.2 ms, and M0d's attention cost
(1.0 / 4.3 / 13.0 ms per token, from the M0c->M0d deltas) already sits at
1.3-1.4x of it. So the 4.9x latency cut is capped by bytes, not by the
loop: predicted attention 0.85 / 3.4 / 10.2 ms (~700 GB/s effective plus
~0.1 ms merge + barrier).
- P-F3' 8192 >= 119 tok/s_gen (M0d 117.5), 32768 >= 91 (M0d 85.0), 100000
  >= 55 (M0d 48.7). Hard floor 32768 >= 88: below it the attention cost
  did not drop by the 0.4 ms the latency cut must give at minimum, and
  the remainder is elsewhere (merge barrier, tails, or the split itself).
  GENERATED at 8192/32768/100000 equal to M0d's (bit-exact change; a
  mismatch anywhere is a bug, not a merge-order observation).
- P-F4' 20-prompt median (default path) within +-2 % of the reference and
  identity 20/20 bit-exact; attention at T <= 1088 is <= 80 MB of KV per
  token (~0.1 ms of ~7.5), so the expected move is +0 to +1 %.
- The first-freeze numbers (P-F3: 120 / 95 / 62) are kept above as the
  record of a prediction made from a false premise; they are not the bar.

**Result.** Round A PASSES every prediction in this amendment.

Two stints: the timed runs on 2026-09-08 13:29-13:39 (`.work/a/stint.txt`),
and the two runs the 13:34 budget stop left undone, on 2026-09-10 01:10
(`.work/a/close.txt`). Binaries throughout: ref `.work/engine-ref`
sha 837ce2553c37aeee (M0d), a `.work/engine-a` sha 417a3ad3d90c8f38.

**P-F3' long decode, q4 pack, `BARO_TMAX=102400`, 64 tokens.**

| length | M0d | floor | Round A | GENERATED |
|---|---|---|---|---|
| 8192 | 117.5 | >= 119 | **121.65** | SAME_AS_M0D |
| 32768 | 85.0 | >= 91 (hard 88) | **98.24** | SAME_AS_M0D |
| 100000 | 48.7 | >= 55 | **63.98** | SAME_AS_M0D |

tok/s_gen. `mega fail word: 0` and `att split: 1088` on every run; prefill_s
8.268 / 51.048 / 298.544, unchanged as expected (the change is in the decode
attention span). The 32k number clears the hard floor by 10 tok/s, so the
remainder is not the merge barrier or the tails.

**P-F4' default 20-prompt A/B** (2026-09-10 01:10, queue empty,
`.work/a/ab2/arm.txt`): `engA=.work/engine-ref engB=.work/engine-a`
shaA 837ce2553c37aeee shaB 417a3ad3d90c8f38, power cap 290000000 uW,
vddgfx -100mV, both arms `BARO_PACK=.work/engine-pack-q4`. ref median
136.75 tok/s_gen spread 0.7 %, a median **138.44** spread 0.7 %,
**ratio 1.012**, identity fails none. Inside the +-2 % band and at the top of
the predicted +0 to +1 % move. This run is the one that REFUSED on 2026-09-08
because `AB_ENGINE_B` does not survive `gpu-wait run`; passing it as
`-- env AB_ENGINE_B=...` is the fix, and the refusal is why no self-comparison
was recorded here (ledger 2026-09-08).

**P-F1 / P-F2 bit-exactness.** `test_mega_block` PASS (m=1 q4/q8, m=3 q8);
one-shot q4 EXACT 138.05 tok/s_gen and q8 EXACT 81.76 against the ref engine;
forced-split arm PASS 64/64 on both packs; forced-split vs unsplit A/B ratio
0.999, identity fails none. `test_attn_block` failed on 2026-09-08 for a
missing fixture in the worktree, not a kernel defect; with `.work/gguf`
symlinked to main's it PASSES on 2026-09-10: "qwen35 gated full-attention
block matches numpy reference".

**Receipt caveat (P1).** `gpu-wait list` at the head of the 13:29 stint was
NOT empty: a `mojo build` job from another lane was waiting and a second job
was running. The long-run numbers above were taken minutes later without a
re-read of the queue, so they carry that risk; the 01:10 A/B, taken with the
queue verified empty, is the clean receipt. Nothing here rests on a
sub-percent difference.

**Not done in this round.** Round B (the f32 wave32 256-thread block design
sketched in the status file) was never started.

## Round C — Round A's unroll re-ported onto the KATT kernel (frozen 2026-09-11, before its build is timed)

**Why a new round.** Round A (`3201685`, branch `lane-attn`) never merged. KATT
(`088c940`) made the head dim a comptime parameter (`HD_`) and replaced the
page-strided V pointer (`vb`, `PGSTR`) with a per-position `kv_off` call, so
Round A's V hunk no longer applies and its numbers were measured on a kernel
main no longer has. Base: main `c92210f` (after the lane-prefill-long merge
`689b445`).

**Change.** Same as Round A amendment 1, re-spelled on the KATT form, in
`attn_head_span` only: the K dot loads `AK_U = 8` vec8 K slices per position
before their FMAs (`for du in range(HD_ // (8 * AK_U))`, valid for HD_ 64 /
128 / 256), and the V accumulate hoists `AV_U = 32` V loads (each through
`kv_off[NAT, HD_, NKVH_]`) + 32 LDS score reads per step before their FMAs,
8-wide and 1-wide tails kept. Arithmetic order unchanged, so bit-exact on
every path that shares the function.

**Predictions.** Reference = `.work/engine` built from the `689b445` tree
(sha recorded in the stint log), candidate = `.work/engine-attn2`.
- P-C1 Fingerprint before GPU: q4 `amar_mega_token` vgpr <= 256, spill 0
  (Round A: 249, 0).
- P-C2 Identity, bit-exact: 20-prompt A/B identity 20/20
  (`bench/ab-prompts.sh`, `-- env AB_ENGINE_B=...` so it survives gpu-wait);
  GENERATED at 8192 / 32768 / 100000 equal to the reference's; run-tests.sh
  green (spark attention parity HD 64/128/256); `test_mega_block` and
  `test_attn_block` PASS.
- P-C3 Long decode, `bench/prefill-prompts/p{8192,32768,100000}.tokens`,
  `BARO_TMAX=102400`, 64 tokens, tok/s_gen candidate / reference:
  8192 >= 1.02, 32768 >= 1.10, 100000 >= 1.15 (Round A on M0d: 1.035 /
  1.156 / 1.31). Hard floor 32768 >= 1.05: below it KATT's per-position
  `kv_off` addressing ate the memory-level parallelism the unroll buys, and
  the V loop needs the page-strided pointer back before any merge.
- P-C4 20-prompt median (default path) within +-2 % of the reference.
- P-C5 `mega fail word: 0` on every run; prefill_s within +-3 % at all three
  lengths (decode attention only).

**Verification before timing (P1).** `gpu-wait list` empty at the head of the
stint and pasted into the stint log; sha256 of both binaries; `att split:`,
`TMAX:`, `prompt tokens:`, `tok/s_gen`, `mega fail word` read from every run's
stdout; `arm.txt` from `bench/ab-prompts.sh`.

**Gate.** P-C1 before GPU; P-C2 every receipt; P-C4 in band; P-C3 against its
predictions (32768 >= 1.05 is the hard floor); P-C5. Merge only if all hold.

**Result.** Round C FAILS P-C3 below its hard floor; the kernel change is not
merged. Stint 2026-09-11 23:4x-23:5x, `gpu-wait list` empty at its head, ref
`.work/engine` sha 38478b34aa4aaf4d (`689b445` tree), cand `.work/engine-attn2`
sha 22d3397a764d4681, receipts `.work/attn-c/`.

| length | ref tok/s_gen | cand | ratio | predicted | GENERATED | prefill_s ref / cand |
|---|---|---|---|---|---|---|
| 8192 | 124.54 | 124.79 | 1.002 | >= 1.02 | SAME | 3.388 / 3.394 |
| 32768 | 101.87 | 102.09 | 1.002 | >= 1.10 (hard 1.05) | SAME | 17.50 / 17.75 |
| 100000 | 68.41 | 68.27 | 0.998 | >= 1.15 | SAME | 79.68 / 79.80 |

`mega fail word: 0`, `att split: 1088`, `TMAX: 102400` on every run.
- P-C1 held: q4 `amar_mega_token` (co79, identified by its 91 nibble masks)
  vgpr 256, spill 0 on both arms; q8 (co80) spills 102 -> 103, pre-existing.
- P-C2 held on every timed receipt: GENERATED equal at all three lengths,
  20-prompt identity 20/20. `run-tests.sh` did not run cleanly in the
  worktree: its shim CMake cache belongs to main's `shim/` (the worktree
  borrowed main's `.work`), so it stopped before any test; not a kernel result.
- P-C4 held: ref median 136.18 spread 1.0 %, cand 136.63 spread 0.9 %, ratio 1.003.
- P-C5 held.
- P-C3 failed: the unroll buys nothing on the current kernel. Main's own 32k
  decode is now 101.9 tok/s_gen against Round A's M0d baseline of 85.0 (and
  Round A's 98.24 with the unroll): the kernel changes merged since then
  (KATT, the lane-prefill-long merge) already took the latency-bound remainder
  Round A was cutting, so hoisting loads no longer moves the number. Per the
  hard-floor clause the next lever is not the loop.
