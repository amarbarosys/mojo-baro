# Lane rowbw: window GEMV rows toward 700 GB/s (Qwen3.8-27B Turbo, spec k = 2, q4 pack)

Brief `.work/briefs/rowbw.md` (primary checkout). Branch `lane-rowbw`, worktree `mojo-baro-lanes/rowbw`, Mojo 1.2.0 (2016d0cd, main `.venv`).
Model `~/Models/qwen3.8-27b-turbo-fcf-mtp-q4_K_M/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M-BARO-a4eb549.gguf`, pack `~/.cache/baro/qwen35-eec459db896e8bc6/packs/bc7a6cf2...` (q4, 17.20 GB).
Loaded list: `docs/lanes/rowbw-loaded.md`. Starting point: `docs/lanes/turbokernel-report.md`, 68.72 tok/s (20-prompt median, `BARO_TURBO=1`, job turbo-ab-spec).

## 0. Starting point, from the turbokernel report and its captures (no GPU minute spent)

Winning path: launch path, spec k = 2, m = 3 window plus two draft-head passes; every dense projection and the head go through
`amar_turbo_rows[N, K, MR, 2]` (A rows staged once per block in LDS, bf16, 64 blocks of K per chunk, two barriers per chunk).
Per 3-row window about 34 ms against a 15.3 ms byte floor; the FFN rows are 64 x (2 x 80.8 + 82.7) = 15.6 ms of it at 606-621 GB/s.

Shipped-kernel wavewait (`.work/turbokernel/ww-turbo2/`, gate instance, 90 waves, 3 chunks per wave): barrier 21.8%, vmcnt 32.1%,
VALU 32.9%, lgkmcnt 7.0%, spill 0. Per chunk per wave the trace reads: the A staging load wait (`0x1a38`, 721 cycles), the barrier
after the stage (`0x1a64`, 1006), the barrier before it (`0x1924`, 597), then the weight loads are ISSUED and waited on
(`0x1b48` 1701 + `0x1b54` 488). The weight stream of a chunk never overlaps its own staging or the barriers: about 4500 stall cycles
against about 560 VALU cycles per chunk. Named limiter: serialized latencies inside the chunk (weights behind the stage), not FMA
count and not the barrier alone.

## 1. Plan and frozen predictions (written before the first GPU job, commit `8ae4d81`)

Method per the brief: wavewait on the top rows, limiter per row, one kernel lever at a time, byte test against the m = 1 kernel
(`kernels/test_turbo_rows`), kernel bench `bench/bench_turbo_rows` (P5 receipt m = 1..4), same-job engine A/B over the 20
mtp-prompts (P4) with identity, `bench/ww.sh` after every kernel change (P21), isa-receipt spills on every build. No minimum-gain bar;
adoption rule per lever: byte test PASS, faster at m = 3 on both FFN shapes, engine median ratio above 1.00 with ids equal on the
stable prompts (p03-story flips run to run on this shape: same-arm control before reading a miss).

| step | what | prediction | if it misses |
|---|---|---|---|
| S0 | fresh census: `bench/turbo-ktrace.sh` on the base engine (tree a4eb549 + docs, `BARO_TURBO=1`), resident spec run p01 n = 200, `wavewait rank` + per-instance table by grid size | C1: gate/up 78-84 us, down 80-86, out 5120 x 6144 34-42, head verify (m = 3) 800-960 us, draft head m = 1 unchanged class (about 800 us), rmsnorm_cast 140 per window about 9 us each; window 31-35 ms, GEMV rows >= 70% of kernel time | window > 37 ms or GEMV share < 60%: the gap, not the rows, is the pool; re-read before any kernel edit |
| S1 | `bench/ww.sh` on the base engine: gate MR 3, down MR 3, head MR 3 (instances by hash from S0) | W1: gate reproduces the ww-turbo2 shape (barrier >= 18%, vmcnt >= 28%, weight waits after the second barrier); down (9 chunks) has a higher barrier share than gate; head MR 3 (N = 248320) same shape as gate | barrier < 10% on gate: the earlier capture was perturbed, re-read the limiter |
| A1 | kernel lever 1: issue the chunk's weight loads (bytes + scales, UNROLL 2) before the chunk's first barrier, so they are in flight across the A staging and both barriers; same fma chain | gate m = 3 80.8 -> 66-76 us, down 82.7 -> 68-78; MR 3 VGPR <= 140, spill 0; byte test PASS (36 arms) | > 78 us: the longer live range dropped a wave per SIMD or the aggregate stream is already the limit; read vmcnt vs barrier in the after-capture before any other change |
| A2 | if A1 held and barrier share stays > 10%: weight loads one chunk ahead (depth 2 chunks, +UNROLL x 5 VGPR) | a further -5 to -10% on gate | flat or slower: the register cost; keep A1 |
| B1 | single barrier per chunk: LDS double buffer (2 x 15 KB at MR 3), no A-row pipelining (v2 held the next chunk's A loads in registers and lost on VGPR) | -4 to -8% on gate against the A-result, VGPR within +4 | slower: LDS 30 KB per block caps blocks per WGP below the register-allowed waves; reject, keep the single buffer |
| E1 | engine A/B, same job, 20 prompts, base engine vs candidate (`AB_ENGINE_B`), spec k = 2 | 68.72 -> 71.5-75.5 tok/s (FFN rows 15.6 ms x -12% plus head and out rows), ids equal on the stable prompts, drafted/accepted equal | < 70.5: the window is less GEMV-bound than the census says (host gaps, launch floor); go to the gap census |
| H1 | head and draft rows (the single largest dispatch, 1.65 + 1.16 per window): lever named after S0/S1 | decided after S1, frozen in the run record before its job | |
| R1 | `rmsnorm_cast` x 128 per window folded into the residual-add kernel that precedes it (one block per row, same sum-of-squares order): only if XS-S | -0.5 to -0.8 ms per window (+1.5-2.5%), bit-exact | not bit-exact or larger than S: drop |

Kernel design rule, frozen: `amar_turbo_rows` keeps `q4_dot_blocks`' per-lane block order and fma chain and the 16-lane tree +
warp sum; levers change only when loads are issued and how the LDS stage is synchronized. Losing variants are deleted (P20), not kept
as knobs.

## 2. Run record

### S0 census (job rowbw turbo-ktrace, `.work/rowbw/ktrace-base/`, engine `engine-base-7d75b2e` sha 2f3a..., `BARO_TURBO: True`, p01 n = 200 under the tracer 63.9 tok/s)

`bench/turbo-ktrace.sh` (new, committed): rocprofv3 kernel trace of one resident spec run, `wavewait rank`, per-instance table by grid size (`inst.txt`).
110176 dispatches, 2748.6 ms in kernels = 34.4 ms per window (80 windows); the `amar_turbo_rows` instances are 82% of kernel time.

| instance (grid / 32 = N) | VGPR alloc | per window | avg us | GB/s |
|---|---|---|---|---|
| gate, up 17408 x 5120, MR 3 | 136 | 128 | 80.0 | 626 |
| down 5120 x 17408, MR 3 | 136 | 64 | 81.8 | 613 |
| ssm_out / attn_out 5120 x 6144, MR 3 | 136 | 64 | 40.4 | 438 |
| qkv 10240 x 5120, MR 3 | 136 | 48 | 51.1 | 577 |
| z 6144 x 5120, MR 3 | 136 | 48 | 35.6 | 497 |
| attn q 12288 x 5120, MR 3 | 136 | 16 | 60.2 | 588 |
| head 248320 x 5120, MR 3 (verify) | 136 | 1.65 | 1024.5 | 698 |
| head MR 1 (draft) | 96 | 1.16 | 792.9 | 902 |
| head MR 2 (window tail) | 112 | 0.18 | 803.7 | 890 |
| draft layer gate/up MR 1 | 152 | 4.8 | 111.9 | 448 |
| rmsnorm_cast | 24 | 140 | 9.4 | |
| ssm_delta_step | 256 | 48 | 26.9 | |
| argmax_pos (3 blocks over 248320) | 24 | 3 | 134.8 | |

C1 HELD on every row but the head: gate 80.0 (band 78-84), down 81.8 (80-86), out 40.4 (34-42), window 34.4 ms (31-35), GEMV share 82%
(>= 70%); head MR 3 1024.5 us MISSED its 800-960 band (q4rowb 1054 -> turbo 1024, 3%; the head instance was already at 698 GB/s). Small
kernels seen on the way: `rmsnorm_cast` (one block per row, 20 serial strided loads, 9.4 us for 60 KB) and `argmax_pos` (3 blocks for
3 x 248320 logits, 135 us = 22 GB/s) are latency-serialized loops, 1.7 ms of the 34.4; recorded as levers R0/R2 below the GEMV rows.

### A1 (job rowbw-rows1, `.work/rowbw/rows1/`, test + bench sha in `arm.txt`): MISSED, variants deleted

`kernels/test_turbo_rows` PASS, 63 arms (PF 1 and PF 2 bit-equal to q4rowb). Bench medians (us), gate / down at m = 3: shipped 82.1 / 80.2;
PF 1 (weight loads hoisted above the chunk's first barrier) 92.1 / 88.1 at 158 VGPRs; PF 2 (also the stage unrolled) 85.0 / 81.6 at 129
VGPRs. Both slower, the band was 66-76. Falsifier read: not the wave count alone (PF 2 kept 129 VGPRs and still lost), so the vmcnt waits
of the shipped kernel are queueing behind a saturated pipe, not exposed latency; moving the issue point cannot buy bandwidth here.

ISA of the shipped MR 3 chunk body (`.work/rowbw/isa-rows1/body-a535e0cb73f23c84.s`, 936 lines): per chunk about 600 VALU issues,
of which 256 fma lane-ops (121 `v_fma_f32` + 70 `v_fmac` + 33 `v_dual_fmac`), about 190 `v_and`/`v_lshlrev` and about 100 `v_dual_mov`
for the bf16 -> f32 unpack of the A rows (3 rows x 2 blocks x 2 halves x 16 lanes, plus the `interleave`), 64 `v_cvt_f32_ubyte`;
24 `ds_load_b128` (already the wide form), 2 barriers. The A unpack is about half the VALU work per weight byte and it is paid per row.

### L2 frozen before its job: f32 LDS stage (`PF 1`, stride 36 dwords, `alignment=16` loads and stores)

The unpack moves into the stage (once per block per chunk, 1.5 items per thread) and the dot loop reads f32 straight from LDS: VALU per
chunk about 600 -> about 320, `ds_load_b128` 24 -> 48, LDS per block MR 3: 15 -> 27.6 KB (UNROLL 2) or 13.8 KB (UNROLL 1, 6 chunks on
K 5120). Bit-exact by construction (bf16 -> f32 is exact, same fma chain). The previous lane's f32 variant ran 3.3x slower (258 us) and
was dropped as "a defect in that variant"; this one is checked on the CPU first: the dumped ISA must show 48 `ds_load_b128` and no
narrow LDS loads before the GPU job.

| id | prediction | if it misses |
|---|---|---|
| L2a | gate m = 3 80.8 -> 66-74 us, down 82.7 -> 68-76, byte test PASS; MR 3 VGPR <= 131 | > 80 us with the ISA clean: the LDS pipe (48 b128 per chunk per wave) or the bigger stage is the new wall; read lgkmcnt and barrier in the after-capture |
| L2b | UNROLL 1 f32 (13.8 KB LDS) within 3% of UNROLL 2 f32 | UNROLL 1 >= 5% faster: LDS size caps residency at 27.6 KB; ship UNROLL 1 |
| L2c | m = 1 through PF 1 within 3% of bf16 PF 0 (not VALU-bound at m = 1) | slower: route m = 1 to PF 0, m >= 2 to PF 1 |

### L2 (job rowbw-rows2, `.work/rowbw/rows2/`): MISSED on every shape, variant deleted

Byte test PASS (every arm, f32 stage bit-equal). ISA before the job: 48 `ds_load_b128`, 4 `ds_store_b128`, no narrow LDS op, VALU per
chunk 723 -> 541, MR 3 VGPR 131 -> 131, LDS 15 -> 27.6 KB (UNROLL 1: 13.8 KB, 120 VGPRs). Bench medians (us), m = 3 gate / down:
bf16 shipped 83.3 / 82.2; f32 UNROLL 2 95.1 / 93.2; f32 UNROLL 1 87.4 / 90.3; MR 4 f32 120.5 / 118.1 against 100.1 / 96.9; MR 1 and
MR 2 unchanged (62.3 / 64.3). The band was 66-74: 25% less VALU made the kernel 12-15% slower, so VALU issue is not the limiter either.

Reading of A1 + L2 together, with the MR scaling of the bf16 kernel (MR 1 62.5, MR 2 64.2, MR 3 83.3, MR 4 100.1 us; LDS 5 / 10 / 15 /
20 KB per block; VGPR 94 / 110 / 131 / 147): the kernel falls off between MR 2 and MR 3, where the per-block LDS stage crosses the point
at which the blocks resident per WGP drop (an LDS allocation is held by the block until its slowest wave ends, so wave skew inside a block
keeps blocks resident with idle waves). The f32 stage (27.6 KB) and the earlier v2 (30 KB) lost for the same reason; UNROLL 1 bf16
(7.5 KB) is the one arm that was not slower. Hypothesis H-LDS: residency per SIMD is capped by LDS bytes per row of the stage, not by
VGPRs and not by VALU.

### L3 frozen before its job: waves per block (`WV`, default 8)

One stage of MR x 64 blocks per block, shared by WV rows: 16 waves per block halves the LDS bytes per row (15 KB per 16 rows);
4 waves per block doubles it. Bit-exact by construction (the per-row work is untouched; only the block -> row mapping changes).
Arms: WV 16 and WV 4 (control) at MR 3 on both FFN shapes, WV 16 at MR 1, 2, 4.

| id | prediction | if it misses |
|---|---|---|
| L3a | H-LDS: WV 16 gate m = 3 83 -> 70-78 us, down 82 -> 70-78; WV 4 slower (88-100): monotone in LDS per row | WV 16 not faster and WV 4 not slower: H-LDS falsified, residency is not LDS-capped; next lever is bytes in flight per wave (two rows per wave) |
| L3b | WV 16 at MR 1 within 3% of WV 8 (memory-bound already) | |

### S1 captures (`bench/ww.sh` on `engine-base-7d75b2e`, `.work/rowbw/ww-base-{gate,down,head3}/`, wavewait pin 927b2b3)

| capture | waves | vmcnt | lgkmcnt | VALU | barrier | spill | top stalls per chunk (avg cycles) |
|---|---|---|---|---|---|---|---|
| gate MR 3 (hash e2f71279, iteration 1500) | 92 | 30.8% | 16.6% | 30.3% | 16.7% | 0 | weight wait 1842, stage lgkmcnt(0) 2800 x 1 per wave, barriers 860 + 534, A wait 769 |
| down MR 3 (896bd05a, iteration 700) | 26 | 31.9% | 7.7% | 37.0% | 16.9% | 0 | weight wait 1752, barriers 762 + 555, A wait 545 |
| head MR 3 (3f5c92c2, iteration 20) | 1292 | 23.6% | 6.7% | 40.4% | 22.5% | 0 | barriers 833 + 698, weight wait 1134, A wait 457 |

W1 HELD on the gate (barrier 16.7 against >= 18 predicted, vmcnt 30.8, the weight wait sits after the second barrier as in ww-turbo2);
the down instance did not carry a higher barrier share than the gate (MISSED that detail); the head has the highest VALU and barrier
shares of the three and the lowest vmcnt. All three instances show the same shape: a third of a wave's cycles are VALU, a third memory
waits, a sixth barriers.

### L3 (job rowbw-rows3, `.work/rowbw/rows3/`): MISSED both ways, H-LDS falsified, variant deleted

Byte test PASS (every arm). Bench medians (us) m = 3 gate / down: WV 8 (shipped) 82.8 / 80.2; WV 16 86.1 / 84.4; WV 4 86.3 / 83.9;
UNROLL 1 WV 16 83.8 / 84.5. Both directions slower, so residency is not capped by LDS bytes per row. Three single-variable misses
(load issue point, VALU count, LDS per row) on the same kernel.

### The arithmetic the three misses leave standing

Gate GEMV at m = 3: 17408 rows x 2.5 chunks x about 600 wave-instructions per chunk = 26 M wave-instructions; 192 SIMDs x 2.5 GHz
issue them in about 54 us; the byte floor is 52 us (50.1 MB at 960 GB/s); measured 83 us. VALU and memory each run at about 65% of
their ceiling, so the kernel is bound by neither alone but by their overlap: inside a chunk a wave issues its weight loads only after
the second barrier, waits for them, then does all its VALU, and the chunk barriers hold a block's 8 waves in lockstep, so a block's VALU
phase and memory phase alternate instead of overlapping. A1 kept that serialization (its hoisted loads were waited for before the stage).
L2 cut VALU but doubled the LDS stream. L3 changed how many waves march together. None put weight bytes in flight during the FMAs.

### A2 frozen before its job: next chunk's weight loads in flight across this chunk's FMAs (`PRE 1`)

The weight loads (UNROLL blocks + scales, about 10 VGPRs) for chunk c + 1 are issued right after chunk c's second barrier, before
chunk c's dequant and FMAs; chunk 0's loads are issued before the loop. Same fma chain, same lane order, bit-exact by construction.

| id | prediction | if it misses |
|---|---|---|
| A2a | gate m = 3 83 -> 70-78 us, down 80 -> 70-77, byte test PASS, MR 3 VGPR <= 144 | > 80 us: the overlap model is wrong as well; stop the m = 3 kernel levers here, record the four misses, take the small-kernel levers (R0, R2) and report |
| A2b | m = 1 and m = 2 within 3% of PRE 0 | |

### A2 (job rowbw-rows4, `.work/rowbw/rows4/`): MISSED, variant deleted; the m = 3 kernel levers stop here

Byte test PASS (every arm). The prefetch cost 39 VGPRs (MR 3 131 -> 170, MR 4 147 -> 186), four times the 10 predicted: the compiler
keeps the next chunk's bytes and the current chunk's dequantized values live together. Bench medians (us) m = 3 gate / down: shipped
81.2 / 81.2; PRE 1 88.2 / 82.0; MR 2 PRE 1 76.8 against 63.8. Band was 70-78.

Four frozen single-variable levers on `amar_turbo_rows` at m = 3, all bit-exact, all slower: load issue point (A1), VALU per byte (L2),
LDS per row (L3), weight bytes in flight across the FMAs (A2). The shipped kernel (bf16 stage, 8 waves, UNROLL 2, loads after the
second barrier) is the best of nine variants measured today at 605-626 GB/s on the FFN shapes; m = 1 and m = 2 through the same kernel
read 780-810 GB/s. Not run: two N rows per wave (the mega-kq lever); it needs 48 more accumulator VGPRs on top of 131, and A1/A2 just
showed this kernel losing a wave for 27 and 39. The 700 GB/s target for the m = 3 rows is NOT reached in this lane.

### R0 / R2 frozen before their job: the latency-serialized small kernels

`amar_rmsnorm_cast` (140 per window, 9.4 us for 3 x 5120 f32: one block per row, 20 strided loads per thread each consumed before the
next is issued) and `amar_argmax_pos` (3 per window, 134.8 us for 3 x 248320 logits = 22 GB/s, 970 serial loads per thread): both loops
now issue 4 (rmsnorm, both passes) or 8 (argmax) loads before consuming them, in the same per-thread index order, so the sum of squares
and the strict-greater argmax scan see the same sequence; bit-exact by construction. `kernels/elementwise.mojo` only, 25 lines.

| id | prediction | if it misses |
|---|---|---|
| R0a | kernel trace on the candidate: `rmsnorm_cast` 9.4 -> 3-5 us (x 140 = -0.6 to -0.9 ms per window of 34.4) | > 6 us: the kernel is launch-floor bound at 3 blocks; the lever is then the fold into the residual add (R1), not run here |
| R2a | `argmax_pos` 134.8 -> 25-45 us (x 3 = -0.3 ms) | > 60: 3 blocks cannot pull 3 MB; split the row over blocks (two-stage argmax) |
| E1' | engine A/B base vs candidate, spec k = 2, 20 prompts: 68.72 -> 70.3-71.5 tok/s, ids equal on the stable prompts, drafted/accepted equal | < 69.5: the window gaps absorb the kernel time saved; read the ktrace gap per window |

### R0 / R2 kernel receipt (job rowbw turbo-ktrace on the candidate, `.work/rowbw/ktrace-cand/`, engine `engine-cand-beab99e` sha 7d7aea5b, tree beab99e clean)

| kernel | per window | base avg us | candidate avg us | per window saved |
|---|---|---|---|---|
| `amar_rmsnorm_cast` (3 rows) | 140 | 9.4 | **4.8** | 0.64 ms |
| `amar_rmsnorm_cast` (1 row, draft) | 7 | 9.6 | 5.5 | 0.03 ms |
| `amar_argmax_pos` (3 x 248320) | 3 | 134.8 | **39.6** | 0.29 ms |

R0a HELD (4.8 in 3-5), R2a HELD (39.6 in 25-45): about 0.9 ms of the 34.4 ms of kernels per window (2.7%). Both bit-exact by
construction (same per-thread order); the engine identity gate is the check (A/B below). The run-to-run total kernel time of the two
traces differs by 3% in the other direction (2749 vs 2845 ms, the turbo rows moved with the host load of 31), so the per-kernel averages,
not the totals, are the receipt here.

## 3. Receipts for the merge gate

- `./run-tests.sh` under gpu-wait on this worktree at commit ca46a0c (kernel tree beab99e + docs), `RUN_ALL=1` (0 cached), pack
  `.work/engine-pack-q4`, default engine `.work/engine` built from the same tree: **run-tests rc=0**, last line
  `PASS run-tests: all tests, pack .work/engine-pack-q4` (log `.work/rowbw/run-tests/run-tests.log`, `.work/rowbw/run-tests/rc.txt` = rc=0,
  `.work/rowbw/run-tests/arm.txt`). The `FAIL 1000 / 1000` lines inside the log are the corruption mutants of the state tests proving their
  checkers go red; the summary lines read `FAIL: 0 (none)`.
- `tools/ci-checks.sh` on the lane tree: `all non-GPU checks passed` (`.work/rowbw/ci-checks.log`).

## 4. Engine A/B (E1'), job rowbw-ab, `.work/rowbw/ab-r0r2/` (`arm.txt`, `base.log`, `cand.log`), `bench/ab-resident.sh`

Same job, two resident arms, 20 mtp-prompts, 64 tokens, spec k = 2 per request, `BARO_MEGA=1 BARO_SPEC=1` both arms, TMAX 16384,
pack bc7a6cf2 (q4). Arm A `engine-base-7d75b2e` (sha 42420b4ffa56fe92, the shipped tree a4eb549 + docs, `BARO_TURBO=1`), arm B
`engine-cand-beab99e` (sha 7d7aea5b3ccdb9ed, + R0/R2). Read-backs per arm: `BARO_TURBO: True`, `BARO_MEGA: True`, `BARO_SPEC: True`,
fail word 0, vram_max 21.67 GB, gtt_max 0.6 GB, power cap 290 W, 1-min load 5.7 at start and 7.0 / 7.9 max during the arms (above the
4 the harness calls clean; `LOAD_START_MAX=12` was passed and is in `arm.txt`). Drafted / accepted 1071 / 734 (base) vs 1072 / 733 (cand).

| | base | candidate | ratio |
|---|---|---|---|
| 20-prompt median tok/s_gen (P4) | 65.73 | **67.76** | **1.031** |
| per prompt, 18 stable prompts | | | 1.015 to 1.031, every one faster |
| p01-water (first prompt, cold clocks in the base arm) | 40.12 | 67.18 | 1.675, the known cold-first-prompt artifact |
| p06-translate | 69.38 | 68.34 | 0.985, ids differ from token 27 (not comparable) |
| identity (generated ids equal) | | | 19/20, p06-translate FAIL |

E1' MISSED on the absolute (band 70.3-71.5; both arms ran 4% under the 68.72 reference, whose job had a load of 3.5 to 4.2; this one
ran at 6 to 8 beside other sessions' builds) and HELD on the ratio (+3.1% median, +2.3-4% predicted), in the direction and size the
kernel receipt gives (0.9 ms of a 34.4 ms window = 2.7%). The claim is the same-job ratio, never the absolute of this job (P4).

p06-translate is one of the five prompts (p03, p06, p08, p09, p17) the OrcaSAQ-27B sessions and the turbokernel lane saw flip between two
runs of ONE binary on this shape; the rule is a same-arm control before reading a miss as a kernel bug. Control job rowbw-det-p06
(`.work/rowbw/det-p06/`): the candidate against itself (env differs by a dummy variable) on p03, p06, p08. Result below.

P21 captures for R0/R2: `bench/ww.sh` on both engines with the regex `rmsnorm_cast|argmax_pos` (`.work/rowbw/ww-small-base/`,
`ww-small-cand/`) traced 0 waves: a 3-block kernel does not land on the one WGP SQTT samples (the wavewait skill's own caveat). The
per-kernel averages of the two kernel traces (section 2, R0/R2 receipt) are the before/after receipt for these two kernels.

### Same-arm control (job rowbw-det-p06, `.work/rowbw/det-p06/`, candidate vs candidate, env differs by `BARO_DUMMY=1`, p03 / p06 / p08)

| prompt | c1 vs c2 ids | first differing token |
|---|---|---|
| p03-story | DIFFER | 24 |
| p06-translate | equal | |
| p08-sql | DIFFER | 49 |

One binary against itself flipped two of the three near-tie prompts. For p06 the id sequences across every run on disk split into two
attractors that part at token 27: sequence A = this A/B's base arm, the turbokernel lane's `ab-spec` t0 (TURBO off) and t1 (TURBO on,
kernels identical to this base); sequence B = this A/B's candidate, both control runs, and the turbokernel lane's `base-q4-spec1` launch
and mega arms (unmodified kernels). Both attractors occur in binaries without the R0/R2 change, so the 19/20 is the shape's run-to-run
flip (standing hypothesis from the turbokernel report: a read of memory nobody wrote at this shape), not an effect of the candidate.
Identity verdict: equal on every stable prompt (19/20), the one miss reproduced by the same-arm control population.

## 5. Verdict

- **Shipped:** `kernels/elementwise.mojo` R0 (`amar_rmsnorm_cast`, 4 loads in flight per thread, both passes) and R2 (`amar_argmax_pos`,
  8 loads in flight), bit-exact by construction; kernel trace 9.4 -> 4.8 us and 134.8 -> 39.6 us; engine same-job median ratio **1.031**
  (65.73 -> 67.76 tok/s_gen under a host load of 6-8; 18 of 18 stable prompts between 1.015 and 1.031). Against the lane's reference
  68.72 (load 3.5-4.2) the expected clean-box number is about 70.8; it is NOT measured here and is not claimed.
- **Not reached:** the brief's target, the m = 3 window GEMV rows from 605-626 toward 700 GB/s. Four frozen single-variable levers on
  `amar_turbo_rows` (weight-load issue point, f32 LDS stage, waves per block 16 and 4, next-chunk weight prefetch), nine variants, every
  one bit-exact and every one slower (sections 2 A1, L2, L3, A2). The shipped kernel is the local optimum of this structure. The arithmetic
  that survives: VALU issue (54 us) and bytes (52 us) each sit at 65% of their ceiling inside the measured 83 us, so the loss is overlap,
  and the one overlap lever that stays untested (two N rows per wave, the mega-kq winner) costs 48 accumulator VGPRs on a kernel that just
  lost a wave for 27 and 39. Named for the next lane, not run here.
- **Head rows:** the m = 3 head instance already runs at 698 GB/s and the m = 1 draft head at 902; the brief's 1054 us figure was the
  q4rowb instance, the turbo instance reads 1024.5 us (census). No head-specific lever was run (H1 not opened: the head is the same kernel).
- **Not taken:** R1 (rmsnorm fold into the residual add), the GEMV -> reduce fold (brief: at most 1.04x), `ssm_delta_step` (48 x 26.9 us,
  not a GEMV row), the barrier phase balance of the m = 1 megakernel (no-spec path).
- GPU time: census 2, byte test + bench 4, wavewait 5 (3 + 2 empty), A/B 1, control 1, run-tests 1 = 14 jobs, no job lost to a harness
  error (every gate script dry-ran on the CPU first). `gpu-wait stats --days 1` at report time: last 1d: 502 jobs (410 ok / 79 failed / 13
  cancelled), gpu busy 538m, median queue wait 11s (box-wide, every session).
- Receipts: `docs/lanes/rowbw-loaded.md`, this report's run records, `.work/rowbw/` (ktrace-base, ktrace-cand, rows1-4, ww-base-gate /
  down / head3, ww-small-base / cand, ab-r0r2, det-p06, run-tests, isa-rows1, isa-rows2), captures JSON `.work/rowbw/ww-base-gate/ww.json`,
  `.work/rowbw/ww-base-down/ww.json`, `.work/rowbw/ww-base-head3/ww.json`. Brief mirrored at `.work/briefs/rowbw.md`.
- Not merged (the brief). `lane-merge --report docs/lanes/rowbw-report.md lane-rowbw` result appended below.

### lane-merge receipt

`lane-merge --report docs/lanes/rowbw-report.md lane-rowbw`, run 1 (`.work/rowbw/lane-merge.log`, at 874c95b): worktree OK, branch 11 commits
ahead, trial merge clean, report found, wavewait capture named, run-tests receipt present, ci-checks passed; FAIL receipts on the one
abbreviated model path. Run 2 (`.work/rowbw/lane-merge2.log`, at ed2791c, the path written in full): receipts 24/24 exist, run-tests
receipt present, ci-checks passed, trial merge clean; `FAIL branch: nothing ahead of main` because main had taken the branch meanwhile:
the maintainer merged lane-rowbw into main as c553d58 ("merge lane-rowbw: rmsnorm_cast and argmax_pos overlap, same-job 1.031") at 18:44 on
2026-11-02. The lane did not merge itself (the brief).
