# Megakernel round (frozen 2026-09-05; engine 68.8 tok/s_gen q8 no-spec, 14.5 ms/token)

Bound by `PROTOCOL-RULES.md`. Follows `launch-fusion-protocol.md`: stage 0
there put the launch floor at 2.4-2.6 us (646 launches/token = 10.8% at the
q8 token), stage 2 (`bench/bench_gridbar.mojo`, `0e3880d`) put a grid-wide
atomic barrier at 0.60 us @96 blocks .. 1.10 us @384. Idea source: Luce
megakernel (Qwen3.5-0.8B, one persistent kernel per token, atomic barrier
between phases, LM head as a second launch). Their token is ~2 ms so
launches were ~half of it; ours is 14.5 ms, so this is a +8-12% round, not
a 1.5x one. the maintainer chose the full per-token shape over per-layer.

## Shape

One persistent kernel per token, m = 1, fixed grid G x 256 threads (8 waves,
= `ROW_THREADS`), G <= resident-block ceiling. Every launch in the decode loop
becomes a phase; phases are separated by the `bench_gridbar` barrier
(agent-scope counter + monotonic generation word, no reset between tokens).
Phase bodies are the EXISTING kernel bodies, hoisted into `@always_inline`
device functions that both the standalone `amar_*` kernels and the megakernel
call, with the block index replaced by a block-strided loop
(`for tile in range(block_idx.x, ntiles, grid_dim.x)`). Per-element
arithmetic and summation order are unchanged, so 64/64 token identity vs the
launch path holds by construction; the launch path stays in the engine as
the parity reference (`BARO_MEGA=0`).

Phases that collapse (not just fuse): the six `amar_skinny_reduce*[..., 1]`
passes are `NSPLIT=1` copies/casts (`Cp[0] -> Y`) -- the GEMM phase writes
its destination directly; `reduce_add` becomes one add at the GEMM tail (same
single addition, same order).

Per-token launches: 646 -> 3 (megakernel, LM-head GEMM over VOCAB=248320 rows,
argmax); stage 3 folds the last two in.

Barriers per token: attn layer 8 (rmsc | q,k,v GEMMs | split+hrms+rope+append
| att | gmul+o-GEMM+add) x 8, ssm layer 7 (rmsc | qkv,z,a,b GEMMs | rgates+conv
| l2 | delta | gated | out-GEMM+add) x 24, ffn 4 (rmsc | gate,up GEMMs |
swiglu | down-GEMM+add) x 32, head 2 = ~360 barriers x ~0.7 us = 0.25 ms.

## Prediction (frozen)

Launch cost removed: 643 x 2.43 us = 1.56 ms. Barriers added: 0.25 ms.
Net 1.31 ms of 14.5 = 9.0%, minus whatever the fixed grid costs the GEMM
phases (stage 0 measures it; ceiling assumes 0).

| stage | prediction | land rule |
|---|---|---|
| S2 engine, m=1 no-spec | **+8% to +12%** (68.8 -> 74.3..77.1 tok/s_gen) | 64/64 identity vs `BARO_MEGA=0` AND 20-prompt median >= +6%, spread < 5%, baseline re-run in the same stint |

Stop rules: **S0a** block-strided q8row at the best resident G loses >= 8% vs
the native launch on the ffn shape -> the GEMM phases eat the launch saving,
close the round at stage 0. **S0b** the combined kernel's VGPR count forces
G < 96 -> close (occupancy, Luce's S_TILE lesson; `delta_step` already spills
122 at m=1, `ssm-occupancy-protocol.md`). **S2** < +4% -> close, code stays
on the branch, receipt in `docs/BASELINE.md`.

## Stage 0: fixed-grid GEMM cost + occupancy (S, `bench/bench_mega_gemm.mojo`)

Cold-cache protocol (`coldcache-protocol.md`: W >= 96 MB rotated). q8row m=1
block-strided at G in {96, 144, 192} (see occupancy receipt) vs native `ceildiv(N, 8)` grid, shapes
(N x K): 4096x4096, 8192x4096, 12288x4096, 4096x12288. Receipt: us per
shape per G, ratio to native, clock via `clock-probe.sh`.

Occupancy receipt: a kernel containing the q8row body + the delta_step body
(no barrier logic needed), built with `mojo build --emit=asm`; read
`.vgpr_count` / `.sgpr_count` / `.lds_size` from the code-object metadata;
resident blocks (wave64, 8 waves/block) = floor(768 / vgprs_granule8) waves per
SIMD x 4 SIMD / 8 waves = blocks per CU, x 96 CUs. Cross-check with
`bench/bench_mega_occ.mojo <G>` (q8row phase + barrier), whose barrier has a
bounded spin (`SPIN_LIMIT`) and a fail word: an overshoot prints
`NOT-RESIDENT` and returns. **NEVER launch an unbounded grid barrier past the
computed ceiling: a deadlocked barrier is not a timeout, it is a gfx ring
hang -> MODE1 reset -> every GL client on the desktop dies (2026-09-05 23:25,
`mega_occ_probe 240`, hard reboot).** Receipt 2026-09-05: vgpr 182 -> 4
waves/SIMD -> 2 blocks/CU -> 192; probe: 96/144/192 complete, 216
NOT-RESIDENT. Stage 0 G set is therefore {96, 144, 192}; 288/384 are out.


### Stage 0 receipt (2026-09-05 23:40, `ce3576d`+grid edit, `.work/mega-stage0.log`)

Arm: 290 W cap, -100 mV, engine-pack-q8 blk.0, NBUF=8 rotation, ITERS=200, 5 reps;
clock-probe sclk med 2802 (2694-3122) MHz, power 281-346 W, Tj 74 C. All G
bit-exact vs native (0 mismatches).

| shape (NxK) | native us | G=96 | G=144 | G=192 | best/native |
|---|---|---|---|---|---|
| 12288x4096 | 71.8 | 64.3 | 64.3 | 64.7 | 0.895 |
| 4096x12288 | 77.4 | 65.3 | 68.6 | 74.8 | 0.845 |
| 8192x4096 | 47.0 | 44.6 | 44.2 | 44.7 | 0.941 |
| 4096x4096 | 27.9 | 24.3 | 24.2 | 24.5 | 0.868 |

**S0a does not fire**: the fixed grid is 5-16% FASTER than the native
`ceildiv(N,8)` launch at G=96 on every shape (one block per CU, no tail
wave); G=192 is worst but still <= native. Best resident G for the
megakernel = 96. This is a gain the frozen prediction assumed to be 0, so
the stage-2 ceiling is +8-12% from launches PLUS the GEMM-phase gain; the
frozen numbers stand as the land rule regardless. -> stage 1.

## Stage 1: one ssm layer as a persistent kernel (L, `kernels/mega.mojo`)

Hoist `matmul_skinny_q8row`, `ssm_*`, `rmsnorm_cast`, `residual_add` bodies
into inline device functions (kernel files stay comment-free per CLAUDE.md);
`amar_mega_ssm_layer` = 7 phases. Gate: `kernels/test_mega_block.mojo`
bit-exact vs the launch sequence on random inputs, layer time at G vs the
14-launch sequence (device events).

### Stage 1 receipt (2026-09-05 23:49, `kernels/mega.mojo` + `kernels/test_mega_block.mojo`)

`amar_mega_ssm_layer`: 7 phases (rmsc | qkv,z,a,b GEMMs | rgates+conv | l2 |
delta | gated | out-GEMM+add), 6 grid barriers, G=96 x 512 threads. GEMM
phases write their destinations directly (NSPLIT=1 reduce collapsed); the
head phases map head -> block 0..31 on threads 0..127 with the original
wave-sum order. ISA: vgpr 192 (spill 121, delta_step's), sgpr 87, LDS 1 KB
-> 1 block/CU at 512 threads, G=96 resident, S0b does not fire.

| gate | result |
|---|---|
| residual X (4096) | 0 mismatches |
| conv window (2 slots x 3 x 8192) | 0 mismatches |
| ssm state (2 x 32 x 128 x 128) | 0 mismatches |
| layer time, 200 iters, synthetic q8 weights | launch(14) 120.8 us, mega 87.1 us, **0.72x** |

Per-layer saving 33.7 us x 24 ssm layers = 0.81 ms/token from the ssm
sub-blocks alone (14 launches x 2.43 us = 34 us predicted; measured 33.7,
so barrier cost is hidden under the fixed-grid GEMM gain). -> stage 2.

## Stage 2: per-token kernel (L-XL)

Add attn phases (`attn.mojo` bodies), ffn, embed, head-norm; device-side
layer loop reading a weight-offset table (`List[Int]` -> device buffer) and
building TileTensors from `wbuf` pointers in-kernel (**verify in stage 1 that
`TileTensor(ptr, layout)` constructs in device code; if not, pass per-layer
pointer structs**). `serve/engine.mojo`: `BARO_MEGA=1` replaces the layer
loop with one `enqueue_function`; LM head + argmax stay as launches. Gate =
the prediction table. Profile receipt: `BARO_PROFILE` per-phase device
timestamps (block 0 thread 0 writes `clock64` at each barrier) so the
post-mortem says which phases moved.

### Stage 2 receipt (2026-09-06 00:24, `b2e3cb4` + `d9468e1`, `bench/mega-prompts.sh`, `.work/mega-p4-run2/`)

`amar_mega_token` (`kernels/mega.mojo`): all 32 layers of one decode token in
one launch, G=96 x 512 threads, ~305 bounded grid barriers/token; weights
addressed on device from the pack offset table; `flat_work_group_size=512`
lifts the VGPR cap to 256 (spills 244 -> 113; the cap of 192 had the GEMM hot
loops spilling, 1.09x SLOWER than launches before it). Head + embed stay as
launches (3 launches/token). Engine switch `BARO_MEGA=1` (m=1, no spec,
decode only; prefill and MTP keep the launch path).

Arm: 290 W cap, -100 mV, engine-pack-q8, 20 prompts x 64 tokens, A and M
interleaved per prompt, one stint; sclk med 3040 MHz, Tj 73 C.

| arm | median tok/s_gen | spread | identity |
|---|---|---|---|
| A `BARO_MEGA=0` | 67.08 | 2.9% | ref |
| M `BARO_MEGA=1` | **79.45** | 1.3% | **20/20 GENERATED equal** |

**+18.4%, S2 LANDS** (rule: >= +6%, spread < 5%, baseline re-run same stint).
Device profile (`BARO_PROFILE=5`, last token): ssm sub-blocks 3.06 ms, attn
1.04 ms, ffn 6.81 ms, layers total 10.9 ms vs 14.5 ms/token before.

First 20-prompt run had 1/20 identity FAIL (p01, token 21): an unrolled
rmsnorm sum-of-squares let the backend contract FMA differently from
`amar_rmsnorm_cast`; 1 ulp in the scale once in 64 tokens. Reverted to the
loop form (`d9468e1`), bit-identical over 64 tokens x 64 dump points
(`BARO_DUMP`). Lesson in `m.ledger/mojo-baro.md` 2026-09-06: same order is
not same result; keep the expression form.

`kernels/test_mega_block.mojo` (4-layer synthetic pack, bit-identical X /
conv / ssm state / KV cache) is the kernel gate; the 20-prompt identity run
is the engine gate. Per-phase receipt (4 synthetic layers, cold pack):
ssm GEMM phase 67-77 us, delta 16, out-GEMM tail 29; ffn gate+up 130-140,
down 68-72; rmsc ~2-3 us x2; attention 60 (GEMMs) + 22 (att). The GEMM
phases sit at their stage-0 fixed-grid numbers, so the remaining headroom
is the head fold (stage 3) and the ffn GEMMs themselves, not the barriers.

## Stage 3: fold LM head + argmax; MR > 1 (M, only if S2 lands)

Head GEMM as a phase (VOCAB rows block-strided, 31040 row-groups), argmax via
per-block partials + last-block reduce; 1 launch/token. MR > 1 for the MTP
window mirrors `gemm_q8`'s dispatch (comptime MR) -- separate prediction,
separate freeze, because the m>1 kernels have different occupancy.

### Stage 3 receipt (2026-09-06 00:52, `082a58e`, `results/mega-stage3/`)

Final norm + head GEMM (VOCAB=248320 rows block-strided) + argmax folded
into `amar_mega_token`: **1 launch per decode token** (embed stays a launch
before it). Argmax: per-wave running best with the `amar_argmax_pos` tie
rule (max value, lowest index), per-block partials, block 0 reduces.
Logits are not materialized on the mega path; MTP/spec keep the launch
path. Kernel gate: argmax token equal, all state bit-identical.

| arm | median tok/s_gen | spread | identity |
|---|---|---|---|
| A `BARO_MEGA=0` | 67.62 | 2.6% | ref |
| M `BARO_MEGA=1` | **81.88** | 1.2% | **20/20** |

**+21.1% vs the launch path** (stage 2 was +18.4%). Device profile, last
token: ssm 3.10 ms, attn 1.06, ffn 6.66, head 1.21 (the 1.02 GB q8 head
read at ~840 GB/s -- bandwidth floor), total 12.05 ms; MR>1 for the MTP
window is a separate freeze. Arm: 290 W / -100 mV, sclk med 3040.

## Not in this round

q4/q8dot weights, prefill, MTP verify width, hipLaunchCooperativeKernel (0.1
us, not needed), Mojo `Semaphore` (unprobed).

### Phase-efficiency receipt (2026-09-06, stage 0 for any GEMM-side follow-up; token kernel with the chunked-reload delta, 234 VGPRs / 0 spills)

Bytes moved / phase time, synthetic 4-layer pack (cold), G=96, m=1. `own` =
block 0's loop time, `wait` = barrier wait after it (straggler + barrier).

| phase | bytes | us | GB/s | note |
|---|---|---|---|---|
| ssm qkv+z+a+b GEMM (1544 groups, 16.1/block) | 50.6 MB | 75 | 675 | |
| ssm out-proj (512 groups, 5.3/block) | 17.4 MB | own 31 + wait 3 | 560 | 6 rounds for 5.33 of work |
| attn q,k,v GEMM (1280 groups) | 42.5 MB | 61 | 695 | |
| attn o-proj (512 groups) | 17.4 MB | own 30 + wait 2.4 | 580 | |
| ffn gate+up (3072 groups, 32/block) | 100.7 MB | 142 | 710 | |
| ffn down (512 groups, K=12288) | 50.3 MB | own 65 + wait 3.5 | 770 | |
| head (31040 groups, 323/block) | 1017 MB | 1208 | 842 | |
| standalone wave-per-row q8row, cold (q8-protocol Q1b) | 100.7 MB | 118 | 855 | reference |
| delta (reload, 32-chunk) | -- | 10.0 | -- | register column: 17.3; bit-exact -- but NOT shipped for m=1, see below |

Reading: the loop reaches the standalone stream rate only when a wave owns
hundreds of rows (head, 842). With 5-32 rows per wave each phase pays its
ramp (all 768 waves issue at once, then the last rows straggle): 560-770
GB/s. Barrier wait itself is 2-4 us (3-10% of the small phases). Summed over
the token this is ~1.3 ms of 12 (the m=1 GEMM phases at 855 would be 11%
faster), but it is phase granularity, not the loop: the only levers are
fewer, larger phases (which the data dependencies mostly forbid) or a
different row distribution for the 512-group GEMMs (6 rounds for 5.33 of
work = 11% idle, fixable only by splitting rows, which breaks bit-exactness).
Not frozen as a round: expected <= +4% for an L change; the head is at 842
and the ffn at 710-770 is the realistic remaining target (+5% of the token
if it reached the head's rate). G=192 / 192-cap ruled out (W3).

**Reload delta in the m=1 kernel: measured and reverted (2026-09-06).** The
synthetic 4-layer test said delta 17 -> 10 us and equal totals; the real
pack, interleaved A/B x5 in one stint, said **80.94 -> 79.71 tok/s_gen
(-1.5%)**: `BARO_PROFILE=5` shows ssm -30 us but **ffn +180 us** and attn
+30 us per token. The ffn phases never touch the delta; the kernel's whole
register allocation moved (256 -> 234 VGPRs, 77 -> 0 spills) and the GEMM
loops got a worse schedule. Rule: a whole-kernel change is judged on the
real pack in one stint, never on the synthetic test alone; the spill count
is not a fitness function. The m=1 kernel keeps the register column
(`RELOAD=False`); the window kernel keeps the reload (it needs the 192 cap).

## Round: rmsnorm fold into the LDS prologue (preregistered 2026-09-08, before any run)

**Change.** On the m=1 q4 path (`LDSA`) every block already reduces the full
H row for the rmsnorm, then writes a 1/G slice of `CurB` (bf16, global),
grid-barriers, and reads row 0 of `CurB` back into `Af` (`stage_a`). The fold
computes the same reduction and writes the full normalised row straight into
`Af` (`stage_rms`), so the rmsc phase and its grid barrier disappear: one per
layer for the ssm/attn phase, one per ffn phase, one for the head = 65
barriers per token. The window kernel (MR=3) and the q8 packs keep `rmsc_phase`
+ barrier untouched. Each block now reads X twice from L2 (16 KB f32) instead
of once plus its `CurB` slice; that is < 1 us per phase.

**Predictions (frozen).**
1. Bit-identical: mega-gate identity stages all PASS (q8/q8d/q4 x spec 0/1,
   ref 64/64); `test_mega_block` PASS. The expression
   `(X * scale * Gn).cast[bf16]().cast[f32]()` is the old `CurB` value
   re-read, same reduction order (EW_THREADS stride, warp.sum, wave sums in
   order), so no bit may move.
2. Real pack, q4, no-spec, 20-prompt median (P4), interleaved A/B x5 in one
   stint, old binary `.work/engine-split` vs new: **+2.0 to +3.5 %
   tok/s_gen** (the 2026-09-08 profile: 65 barriers x ~4 us = 0.26 ms of
   7.52 ms). `BARO_PROFILE=5`: the rmsc+barrier slot shrinks by 3-4 us per
   phase; ssm/attn/ffn GEMM slots unchanged within noise.
3. Failure mode to watch (2026-09-06 rule): the kernel's register allocation
   may move and cost an untouched phase. Judge on the real pack, all phases;
   VGPR/spill census recorded as a receipt, not a fitness function. If the
   real-pack median is < +1 % the round is a no-op and is reverted with the
   numbers in the log.

**Result (2026-09-08, same day): no-op by rule 3, reverted; fold parked on
branch `lane-fold` @ `239c065`.** Prediction 1 held: mega-gate 13/13 PASS,
identity 20/20 on the A/B, q4 vs ref 64/64. Prediction 2 failed: 20-prompt
q4 no-spec, interleaved per prompt in one stint (`bench/ab-prompts.sh` with
`AB_ENGINE_B`, sclk med 3012 MHz, 290 W / -100 mV):

| arm | median tok/s_gen | spread |
|---|---|---|
| `.work/engine-split` (champion `37bca40`) | 130.65 | 8.2% |
| fold | 127.29 | 0.7% |

ratio 0.974. `BARO_PROFILE=5`, last token, median of 5 alternating runs
(us): split ssm 2375 / attn 639 / ffn 3839 / head 636 / total 7509; fold
ssm 2797 / attn 601 / ffn 3671 / head 633 / total 7705. The fold did what
it was built for: rmsc slots ssm 97 -> 70, ffn 124 -> 86, attn 39 -> 24,
and the ffn GEMM/down/post slots 2312/1327/62 -> 2282/1265/43 (about
-250 us per token in the phases it touches). The delta phase (stamps 4>5,
code untouched) went **464 -> 1036 us**: the token kernel's scratch went
352 -> 740 B (spills 355 -> 423), the q4 variant's `scratch_load` count 75
-> 286. Third sighting of the delta allocation lottery (392 -> 219 -> 471
-> 525 -> 464 -> 1036 across builds with the same delta code).

Pin attempt in the same stint: the register-column delta body as a
`@no_inline` device function (`delta_col`). The call is real
(`s_swappc_b64`, `scratch_load` back to 62) and the delta phase returned to
665 us, but the ffn GEMM/down slots went 2310/1331 -> 2492/1588 (+440 us),
total 8035 us. Both variants lose to the champion; stopped after two.

Reading: the fold is worth ~+3 % once the delta phase's register allocation
is pinned by something that does not perturb the GEMM loops. Next action:
check whether `mega_token` carries `rocdl.flat_work_group_size` /
`waves_per_eu` (the board rule: LLVM caps VGPRs at 192 without it), pin the
delta phase there, re-measure the champion alone, then re-apply the fold
from `lane-fold`.

**Follow-up, same day: fold + chunked delta LANDS (`9d00280`, merged `9e6feaa`).**
The fold's loss was the delta phase's allocation, so the m=1 q4 kernel now
takes the chunked delta (`RELOAD=True`, already the window kernel's form,
bit-identical by the same test): 239 VGPRs, 0 scratch, delta 464 -> 269 us,
rmsc slots -60, ffn gate/up +180 (the shared q4 dot loop lost 28 VOPD pairs
and gained 270 `s_delay_alu` with unchanged source -- a schedule change, not
work), token 7498 -> 7358 us. Two other pins measured and dropped in the same
stint: `@no_inline` on the delta column (delta 665, ffn +440, total 8035)
and `@no_inline` on all three phase functions (call/arg overhead 108 us in
the ssm entry, delta 659, total 7739). `rocdl.waves_per_eu` is not settable
from Mojo's `@__llvm_metadata` (integer, tuple and literal forms all
rejected), so the scheduler's occupancy target stays out of reach.

| arm | median tok/s_gen | spread | identity |
|---|---|---|---|
| champion `37bca40` | 130.59 | 34.6% (one cold outlier) | ref |
| fold + chunked delta | **133.93** | 14.1% (one outlier) | 20/20 |

**+2.6 %**, inside the frozen +2.0 to +3.5 % band; gate 13/13
(`.work/mega-gate-b2`), sclk med 3030 MHz, 290 W / -100 mV
(`.work/ab-b2-real`). Two earlier A/Bs of this candidate read 1.001 because
the branch lacked the `AB_ENGINE_B` runner commit and ran the champion
against itself; `arm.txt` names both binaries since, and it is read before
the ratio (ledger 2026-09-08).

## Round: warm the next phase's first weight rows before the grid barrier (preregistered 2026-09-15, before any run)

**Why.** Fresh `BARO_PROFILE=5` on the champion (`083ad17`, p09, 137.5 tok/s_gen,
identity sha `c00468774758`, `.work/m5/prof5.log`): per token 7149 us; ffn
gate+up 2414 us = 750 GB/s, ffn down 1299 us = 697 GB/s, ssm out-proj 570 us
= 707 GB/s, head 626 us = 914 GB/s. Same picture as the 2026-09-06
phase-efficiency receipt above: the layer GEMVs pay a ramp at every phase
start (768 waves issue their first loads into a cold pipeline at once) and the
head, one huge phase, does not. That receipt bounded the whole effect at about
+5% of the token and named "fewer, larger phases" as the lever; the data
dependencies forbid that. This round tries the other way to hide the ramp: a
wave that has finished its rows of the current phase issues the loads of its
first weight tiles of the NEXT phase before entering the grid barrier, so
those lines are in L2 when the phase starts. Loads only; nothing is computed
from them, so the arithmetic and its order are untouched.

**Change** (`kernels/mega.mojo`, m=1 q4 token kernel only; the window kernel
and the launch path untouched): a `touch_row` helper (16 B per lane x UNROLL
per wave, the same addresses the first `q4_dot_lds` iteration reads) called
at three sites: (1) end of the ffn gate/up loop, touching the wave's first
`down` row; (2) end of the ffn down loop in `mega_body`, touching the wave's
first row of the next layer's first GEMM (`o1`: CONV x H for an ssm layer,
QF x H for an attn layer); (3) after the last layer, the wave's first head
row. The touched bytes are folded into a register sink stored only under a
value-dependent condition that the compiler cannot fold, so the loads stay.

**Arms.** A = `.work/engine-base` built from `083ad17` untouched; B = same
tree plus the change. Both built in the same stint; sha256 of both recorded;
`isa-loops` fingerprint of `amar_mega_token` q4 recorded for both (A today:
dual 124/79/79/59, delay 644/420/312, scratch 0, `.work/m5/isa/co79.s`).

**Predictions (frozen).**
1. Bit-identical: `tools/mega-gate.sh` identity stages PASS, `test_mega_block`
   PASS, A/B identity 20/20. Loads only; no bit may move.
2. `BARO_PROFILE=5`, p09, median of 5 alternating runs: ffn down slot
   1299 -> 1240..1270 us; the next-layer first slots (ssm in-proj 1031, attn
   qkv 300) shrink by 20..50 us summed; head 626 within noise. Token
   7149 -> 7020..7090 us.
3. Real pack, q4, no-spec, 20-prompt median (P4), `bench/ab-prompts.sh` with
   `AB_ENGINE_B`, A then B per prompt in one stint, clock read back
   (`bench/clock-probe.sh`), power cap read back: **+1.0 to +3.0 %
   tok/s_gen**. Kill line (the 2026-09-08 rule): median below +1.0 %, or any
   untouched phase slower by more than its noise, is a no-op; the change is
   reverted with the numbers in this file and the fingerprint diff named.
4. Failure mode to watch: the touch changes register allocation and re-rolls
   the dot loop (the 09-06 and 09-08 rounds both lost an untouched phase this
   way; `3824e20` gained one). The fingerprint diff is read BEFORE the tok/s
   number is interpreted; a loop that changed with unchanged source is the
   allocator, and only the real-pack A/B settles it.

**Result (2026-09-15, same stint): no-op by the kill line, reverted; patch kept
at `.work/m5/touch.patch`.** Prediction 1 held: `test_mega_block` PASS, A/B
identity 20/20 (the mega-gate identity stage itself aborted on
`engine-pack-q8: no index.txt`, the q8 pack is no longer on disk; tool rot,
not the kernel, noted for `tools/mega-gate.sh`). Prediction 4 fired first:
with the dot-loop source untouched the q4 token kernel's fingerprint moved,
`isa-loops` dual 124/79/79/59 -> 115/118/84/84, delay 644 -> 533, first-loop
wide loads 29 -> 21, VGPR/scratch census unchanged (256, 0 scratch). Read
before the number, as required. Prediction 3 failed: 20-prompt q4 no-spec,
A then B per prompt in one stint (`bench/ab-prompts.sh`, base `e5804d99` vs
touch `eaf20e9a`, sclk med 2930 MHz, 290 W / -100 mV):

| arm | median tok/s_gen | min..max | spread |
|---|---|---|---|
| base (`083ad17`) | 137.54 | 136.96..137.95 | 0.7% |
| touch | 138.11 | 137.74..138.45 | 0.5% |

ratio 1.0042; per-prompt ratio 1.0016..1.0072, every prompt above 1. A real
but sub-threshold gain, confounded with the re-rolled schedule; under the
frozen rule it does not land. The 2026-09-06 ceiling stands: phase ramp is
worth a few percent at most and no single-site change reaches +1%. Not to be
re-proposed as a warm-touch; a next attempt at the ramp needs a different
mechanism (fewer phases), which the data dependencies forbid at m=1.
