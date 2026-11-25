# verify8-r2: 27B q4 verify step at full bandwidth (2026-11-25)

Brief: `docs/lanes/2026-11-25-verify8-r2-brief.md`. Skills loaded: gpu-kernels, bench-gates, wavewait, mojo-baro, mojo-lean,
memory-optimization, mojo-nightly-lane-builder, lanes, ops.

## State found at lane start

- The nerkyor q4 pack and the DFlash2 draft pack are not on disk (they lived in the removed lane `nerkyor-vs-forge`); `/home`
  has 20.4 GB free, a rebuild takes about 17 GB. ASK sent to the boss. Until answered, kernel work uses the same-shape
  27B q4 pack on disk: `~/.cache/baro/qwen35-eec459db896e8bc6/packs/51077bf8...` (TurboFCFusion bake a4eb549, 506 q4
  tensors, H 5120, FFN 17408, 64 layers). Engine numbers on that pack are not nerkyor numbers and are labelled so.

## K0: row kernels at m up to 8 (kernel bench, before any kernel edit)

Question: `amar_turbo_rows_dot2` (LDS-staged A rows, v_dot2_f32_bf16) is wired for m = 2..4 only. Nobody timed it at
m = 5..8. It reads each weight block once per wave and its m = 3 cost is 65.9 us on the gate shape, so it may already
beat the WMMA route (118.9 us) at m = 8 with no new kernel.

Bench: `bench/bench_turbo_rows.mojo -D VERIFY8=2`, real blk.0..7 FFN weights rotated (400 MB per shape), 8 launches per
sync, 11 batches, 3 repeats, 1 s warm work first. ISA receipt of the build (`tools/isa-receipt.py`): dot2 MR 8 U 2
77 VGPR, 0 spills; exact turbo MR 8 131 VGPR, 0 spills.

### Predictions (frozen by this commit, before the run)

| id | claim | pass if |
|---|---|---|
| K0.1 | dot2 at m = 8 beats the bf16 WMMA route | DOT2 MR 8 U 2 <= 0.80 x WMMA D=4 m=8, both shapes |
| K0.2 | dot2 at m = 8 stays under the 750 GB/s line's cost | gate <= 66.8 us, down <= 66.8 us: predicted FAIL (estimate 80 to 95 us, VALU-bound: 256 dot2 per wave per chunk) |
| K0.3 | dot2 row cost is linear | (m=8 - m=3) / 5 within 2 to 6 us per row on the gate shape |
| K0.4 | exact turbo at m = 8 loses to WMMA | TURBO MR 8 >= WMMA m=8 (about 8 VALU per weight per row) |
| K0.5 | dot2 output at m = 8 is within the dot2 m = 3 error class | host check vs fp64 max rel <= 2e-3 |

### K0 result (`exchange/receipts/verify8-r2/k0-rows-m8.log`, tree 07944fd, one gpu-wait job, pack = TurboFCFusion q4)

us per call, median of 3 repeat medians; GB/s = 50,135,040 B / time.

| arm, m = 8 | gate us | gate GB/s | down us | down GB/s |
|---|---|---|---|---|
| turbo m = 1 (base) | 63.1 | 795 | 65.2 | 769 |
| bf16 WMMA D=4 (today's opt-in 8-row route) | 119.0 | 421 | 127.7 | 392 |
| int8 WMMA incl. quantize (not wired) | 94.6 | 530 | 113.9 | 440 |
| dot2 rows MR 8 U 2 | 202.8 | 247 | 212.0 | 236 |
| dot2 rows MR 8 U 1 | 155.6 | 322 | 159.0 | 315 |
| exact turbo rows MR 8 | 189.0 | 265 | 183.7 | 273 |
| q4rowb MR 8 (default build's 8-row route) | 209.9 | 239 | 217.0 | 231 |

| id | verdict |
|---|---|
| K0.1 | FAIL: dot2 m=8 is 1.70x / 1.66x the WMMA route. The row kernels are instruction-bound in m (16 dot2 per row per block); closed for m > 4 |
| K0.2 | FAIL as predicted (and worse than the estimate: 203 us, not 80 to 95) |
| K0.3 | not evaluated: MR 3..6 lines are in the receipt, the m=8 point already decides |
| K0.4 | PASS: exact turbo MR 8 189 / 184 us |
| K0.5 | PASS: max rel 3.8e-4 gate, 1.8e-3 down (bar 2e-3) |

Reading: only a WMMA keeps the instruction count flat in m. The route to 750 GB/s is fewer VALU per weight in the WMMA
kernel (today about 125 VALU and 4 WMMA per 512 weights), not a row kernel.

## K1: `amar_fewrow_nib`, bit-built bf16(128 + n) fragments in the WMMA geometry

Change (`kernels/fewrow.mojo`): weights enter the WMMA as bf16(128 + n) = `0x4300 | n` built with dword bit ops (the dot2
trick), in pair order; `amar_fw_prep` writes A once per GEMM in the same order plus the per-block row sums, so the two
ones-WMMAs per block go away (`d * (dot - 136 * sum)`); m <= 8 keeps 4 accumulator registers. ISA receipt D=4: 123 VGPR at
D=2, 0 spills; the compiler narrows the dword ops to 16-bit ops (`v_or_b16`, `v_and_b16`), about 87 VALU per block where
44 were intended.

### Predictions (frozen by this commit, before the run)

| id | claim | pass if |
|---|---|---|
| K1.1 | fewer VALU and 2 of 4 WMMAs gone give a modest gain | NIB D=4 incl. prep <= 0.95 x WMMA D=4, both shapes |
| K1.2 | it does not reach the int8 route | NIB D=4 incl. prep >= IU8 on the gate shape |
| K1.3 | the prep launch is small | NIB - NIBNOPREP <= 8 us, both shapes |
| K1.4 | numerics in the dot2 class | host check vs fp64 max rel <= 2e-3 at m = 8 and m = 5 |

## K2: `amar_fewrow_iu4`, the q4 nibbles enter `v_wmma_i32_16x16x16_iu4` with no unpack

Idea: Q4_0 stores a block as 16 bytes of two nibbles; the iu4 WMMA takes 16 unsigned nibbles per lane in 2 VGPRs. The
16 loaded bytes are two B operands as they are: zero VALU to unpack. Activations become P signed 4-bit planes per 32-block
(`amar_fw_prep4`: per-block scale max/(2^(4P-1) - 1), nibbles in the Q4_0 byte order, block sum folded into
`-8 * scale * sum(q)`), one WMMA pair per plane, planes chained through the C operand with a 4-bit shift. Integer exact
for the quantized activations; the only error is the activation rounding (P = 2: int8, the IU8 class; P = 3: 12 bits;
P = 4: 16 bits). ISA receipt (`tools/isa-receipt.py .work/v8r2/bench-iu4`, D = 4): 88 to 90 VGPR, 0 spills; per block
2P WMMA, 4(P-1) shifts, 4 cvt, 8 fma. The bf16 route has about 125 VALU and 4 WMMA per block.

### Predictions (frozen by this commit, before the run)

| id | claim | pass if |
|---|---|---|
| K2.1 | P = 2 beats the int8 route | IU4 P2 incl. prep <= 0.92 x IU8, both shapes |
| K2.2 | P = 3 incl. prep beats the bf16 WMMA route by 25% | <= 0.75 x WMMA D=4, both shapes |
| K2.3 | P = 3 on a prepared A reaches 600 GB/s | <= 83.5 us, both shapes; 750 GB/s (66.8 us) predicted NOT met |
| K2.4 | error falls 16x per plane | max abs err / max abs ref: P2 <= 5e-3, P3 <= 4e-4, P4 <= 4e-5 |
| K2.5 | each extra plane costs under 8 us | P4 - P3 and P3 - P2 <= 8 us (prepared A) |

### K1 and K2 results (`exchange/receipts/verify8-r2/k1-nib.log`, `k2-iu4.log`; trees ef59d81, feb4e58; one gpu-wait job each)

us per call at m = 8, median of 3 repeat medians, same pack and method as K0.

| arm | gate us | gate GB/s | down us | down GB/s | max abs err / max abs ref (gate, down) |
|---|---|---|---|---|---|
| bf16 WMMA D=4 (control, both jobs) | 118.9 / 120.2 | 422 / 417 | 127.9 / 129.8 | 392 / 386 | - |
| int8 WMMA incl. quantize | 92.8 / 95.9 | 540 / 523 | 114.3 / 114.1 | 439 / 440 | 4.0e-3, 3.5e-3 |
| NIB D=4 incl. prep | 95.0 | 528 | 128.0 | 392 | 1.3e-6, 1.2e-6 |
| NIB D=4, prepared A | 91.4 | 548 | 122.6 | 409 | same |
| IU4 P=2 incl. prep / prepared | 92.2 / 88.9 | 544 / 564 | 107.9 / 103.2 | 465 / 486 | 4.0e-3, 3.5e-3 |
| IU4 P=3 incl. prep / prepared | 94.1 / 91.2 | 533 / 550 | 110.4 / 103.9 | 454 / 482 | 3.7e-4, 2.1e-4 |
| IU4 P=4 incl. prep / prepared | 103.6 / 99.8 | 484 / 502 | 124.6 / 119.1 | 402 / 421 | 1.9e-5, 1.3e-5 |

| id | verdict |
|---|---|
| K1.1 | gate PASS (0.80), down FAIL (1.00) |
| K1.2 | FAIL by 2 us (95.0 vs 92.8): about equal to the int8 route, with 3000x smaller error |
| K1.3 | PASS: prep costs 3.6 / 5.4 us |
| K1.4 | PASS: max rel 8.9e-4 / 1.5e-3 |
| K2.1 | FAIL: P2 / IU8 = 0.96 gate, 0.95 down |
| K2.2 | gate FAIL by 3 points (0.78), down FAIL (0.85) |
| K2.3 | FAIL: 91.2 / 103.9 us, not <= 83.5 |
| K2.4 | PASS: 4.0e-3, 3.7e-4, 1.9e-5 |
| K2.5 | P3 - P2 PASS (2.3 / 0.7 us); P4 - P3 FAIL on down (15 us) |

Reading: three different inner loops (bf16 bit-built with about 87 VALU per block, int8, 4-bit with about 25 VALU per
block) all stop near 90 us on the gate shape. The inner-loop VALU count stopped being the limit; what the three share is
the geometry of `amar_fewrow_wmma`: each wave loads the same 16 weight rows in both half-waves (the WMMA reads operand
lanes 0-15), and each lane loads 64 to 100 bytes of activation data per 18 weight bytes from global memory. The down
shape (K = 17408, activation planes 3.4x larger) is the slower one in every arm, which fits the second point.

## Step 2 trace table (rocprofv3 kernel trace, `bench/turbo-ktrace.sh`, p01, 200 tokens, TurboFCFusion q4 pack, tree feb4e58)

Host load at start 21 and 33 (other lanes), so absolute times are high; shares are the use. Receipts
`exchange/receipts/verify8-r2/kt-k2`, `kt-k7`. The DFlash2 window itself is not traced: no draft pack on disk; the
MTP k = 7 window on the FEWROW build is the same 8-row trunk.

| window | kernels busy ms / 200 tok | q4 GEMV or GEMM share | next | dispatches |
|---|---|---|---|---|
| MTP k = 2, m = 3 (`eng-base`, dot2) | 3382 | `amar_turbo_rows_dot2` 85.6% (top five 77.2%) | ssm_delta_step 3.2%, rmsnorm_cast 1.8% | 112,990 |
| MTP k = 7, m = 8 (`eng-fw1`, bf16 WMMA) | 3315 | `amar_fewrow_wmma` 79.6% (FFN gate/up 30.8% at 120.5 us, FFN down 15.8% at 123.8 us) | ssm_delta_step 6.3%, q4rowb 2.5%, rmsnorm_cast 1.5% | 104,825 |

In-engine the two FFN GEMMs run at the bench's times (120.5 / 123.8 us vs 119 / 128), so the bench is a fair proxy for
the 8-row window. The 8-row GEMM is 80% of the verify window's kernel time: a 120 -> 70 us kernel removes a third of it.

## K3: `amar_fewrow_iu4s`, staged planes and distinct weights per half-wave

Change: one wave owns 32 output rows. Lanes 0-15 and 16-31 load different weight rows; the second WMMA group takes the
lane-swapped copy (4 registers). All 8 waves of a workgroup walk the same K chunk and read the activation planes from
LDS, staged once per chunk (64 blocks, 448 B per block at P = 3). No cross-wave reduction: each lane stores its own
outputs. ISA receipt D=2: P2 120 VGPR, P3 157, P4 175, 0 spills; the lane swap compiles to `ds_bpermute_b32`.

### Predictions (frozen by this commit, before the run)

| id | claim | pass if |
|---|---|---|
| K3.1 | the WMMA reads operand lanes 0-15 only, so the unreplicated B operand is correct | host check of arm 63 within the K2 P3 error (<= 4e-4 of max); if it fails the replicated arm 83 is the kernel |
| K3.2 | staged geometry beats the K2 kernel | IU4S P3 prepared <= 0.88 x IU4 P3 prepared, both shapes |
| K3.3 | reaches 600 GB/s on both shapes | <= 83.5 us prepared; 750 GB/s (66.8 us) predicted NOT met |
| K3.4 | the down shape stops being the slow one | down / gate <= 1.06 |

### K3 run 1 (tree 79db210): K3.1 FAIL, no timings

Host check of the first arm (P = 2, unreplicated B) failed at once: output row 0, token 0 matches fp64 (0.07302 vs
0.07306), token 1 does not (-0.324 vs -0.146), token 2 matches. Odd tokens live in lanes 16-31 of the result. So the
WMMA does not read operand lanes 0-15 only: a result lane uses the B data of its own half-wave. `exchange/receipts/verify8-r2/k3-iu4s.log`.

Deviation, stated before run 2: new mode `XM = 2` (crossed halves). If a result lane pairs with the B operand of its own
half, one WMMA group on the unswapped weights yields (even tokens, rows 0-15) in lanes 0-15 and (odd tokens, rows 16-31)
in lanes 16-31, and the group on the lane-swapped weights yields the complement. No selects, each lane's scale is its
own. `XM = 1` (halves replicated through selects) stays as arm 83, the safe form. K3.1 is replaced by K3.1b: arm 63
(crossed) passes the host check within 4e-4 of max. K3.2 to K3.4 stand.

### K3 runs 2 and 3 (trees 313d08c, then the arm-order fix; `exchange/receipts/verify8-r2/k3b-iu4s.log`, `k3c-iu4s.log`)

Run 2: crossed halves are correct (errors equal to K2: 4.0e-3, 3.7e-4, 1.9e-5 of max); arm 83 read a P = 4 scratch as
P = 3 and printed nan, a bench order defect, fixed. Run 3, us per call at m = 8, prepared A, D = 2:

| arm | gate us | down us |
|---|---|---|
| bf16 WMMA D=4 (control) | 118.9 | 127.1 |
| K2 IU4 P=3 D=4 (control) | 89.8 | 105.5 |
| K3 IU4S P=2 / P=3 / P=4 | 115.8 / 125.8 / 141.9 | 169.3 / 201.6 / 250.6 |
| K3 P=3 at D=1 / D=4 | 189.7 / 118.3 | 246.2 / 189.2 |

| id | verdict |
|---|---|
| K3.1 | FAIL (run 1). K3.1b PASS: crossed halves 3.65e-4 / 2.10e-4 of max at P = 3 |
| K3.2 | FAIL: 1.40x / 1.91x the K2 kernel, slower, not faster |
| K3.3 | FAIL |
| K3.4 | FAIL: down / gate = 1.60 |

Reading: K3 is latency-bound. One wave walks all of K for 32 rows, so only N / 32 waves exist (160 on the down shape,
544 on gate) against N / 2 in `amar_fewrow_wmma`; time falls with D (190 -> 126 -> 118 us) and grows 10 to 50 us per
plane. The staged-LDS geometry is closed for this kernel. What K3 proved is the crossed-halves WMMA form.

## K4: `amar_fewrow_iu4x`, K2 geometry (K split over 8 waves, 4 waves per 16 rows) with crossed halves

One workgroup = 32 output rows, 8 waves split K; planes and scales come from global memory as in K2; each lane loads
its own weight row, so one load instruction brings 32 rows where K2 brought 16 twice. N / 4 waves. ISA receipt D=4:
P2 104, P3 131, P4 155 VGPR (D=2: 84 / 128), 0 spills.

### Predictions (frozen by this commit, before the run)

| id | claim | pass if |
|---|---|---|
| K4.1 | correct at the K2 error | host check P3 <= 4e-4 of max, both shapes |
| K4.2 | beats K2 by 10% | IU4X P3 prepared <= 0.90 x IU4 P3 prepared, both shapes |
| K4.3 | 600 GB/s on the gate shape | P3 prepared <= 83.5 us gate; down <= 95 us |

### K4 result (`exchange/receipts/verify8-r2/k4-iu4x.log`, one gpu-wait job)

| arm, m = 8, prepared A | gate us | down us |
|---|---|---|
| bf16 WMMA D=4 | 119.5 | 126.9 |
| K2 IU4 P=3 D=4 | 89.2 | 107.6 |
| K4 IU4X P=2 / P=3 / P=4, D=4 | 89.9 / 93.7 / 102.0 | 108.5 / 112.1 / 118.4 |
| K4 P=3 at D=2 / D=1 | 118.4 / 172.8 | 147.5 / 241.2 |

K4.1 PASS (3.65e-4 / 2.10e-4). K4.2 FAIL (1.05 / 1.04: no gain). K4.3 FAIL.

Reading: time halves from D=1 to D=2 and falls again to D=4 in K4 as in K3. The kernels are bound by the latency of
the weight loads: throughput = waves in flight x weight bytes in flight per wave. K4 doubled the unique bytes per load
and paid for it in registers (131 against 88 VGPR), so fewer waves fit: a wash. The lever is weight bytes in flight
per register, not VALU and not rows per wave.

## K5: `amar_fewrow_iu4d`, split unroll

K4 with two depths: DQ weight blocks loaded ahead (5 registers per block: the DRAM stream), DA activation blocks
(20 registers per block at P = 3: cache-resident, loaded just before use). Tail guards replace the divisibility rule.
ISA receipt P3: DQ 4/8/12/16 at DA 2 = 86 / 114 / 148 / 172 VGPR class, 0 spills in every arm.

### Predictions (frozen by this commit, before the run)

| id | claim | pass if |
|---|---|---|
| K5.1 | correct | host check P3 <= 4e-4 of max, P2 <= 5e-3 |
| K5.2 | deeper weight prefetch beats K2 | best IU4D P3 prepared arm <= 0.90 x IU4 P3 prepared, both shapes |
| K5.3 | 600 GB/s on gate | best P3 prepared arm <= 83.5 us gate, <= 95 us down |
| K5.4 | there is an optimum in DQ | time is not monotone in DQ over 4, 8, 12, 16 (registers cost waves) |

### K5 result (`exchange/receipts/verify8-r2/k5-iu4d.log`, one gpu-wait job): my K4 reading was wrong

us per call, m = 8, P = 3, prepared A (K2 control in the same job: 90.1 gate, 105.9 down):

| DQ (weight blocks in flight) | DA 1 gate / down | DA 2 gate / down |
|---|---|---|
| 4 | 98.0 / 116.2 | 100.4 / 117.5 |
| 8 | 118.3 / 126.9 | 119.7 / 125.6 |
| 12 | 108.8 / 125.8 | 112.1 / 127.0 |
| 16 | 132.9 / 151.4 | 132.4 / 151.1 |

K5.1 PASS. K5.2 FAIL (best arm 1.09x / 1.10x K2). K5.3 FAIL. K5.4 PASS in form (not monotone), but no arm beats DQ 4.

Reading: more weight bytes in flight per wave made the kernel slower, so "bound by the latency of the weight loads" is
withdrawn. K2 at 16 rows per wave, D = 4, 88 VGPR stays the fastest 8-row kernel measured (89 to 91 us gate, 104 to 108
down). I do not know its limiter; five guesses (VALU count, activation traffic, half-wave duplication, wave count,
prefetch depth) were each tested by a kernel and none moved it below 89 us. The next step is the instrument the brief
named, a wavewait capture of K2 against turbo m = 1, not a sixth guess.

## K6: wavewait capture of K2 and two timing-only ablations

Capture `.work/v8r2/att-k2` (`att-capture.sh`, iteration 60 of each kernel, warm, bench-sized launch, mism 0, open 0;
copy under `exchange/receipts/verify8-r2/ww-k2/`). Gate shape:

| kernel | cycles per traced wave | row-blocks per wave | cycles per row-block | mem wait share | vmem loads per wave | barrier share |
|---|---|---|---|---|---|---|
| turbo m = 1 (dispatch 9526) | 23,481 | 160 | 147 | 49.0% | 12 | 27.3% |
| K2 IU4 P=3 D=4 (dispatch 10254) | 62,810 | 320 | 196 | 63.6% | 125 | 5.0% |

Mem wait per weight byte: turbo 4.0 cycles, K2 6.9. VALU is 21.7% of K2's cycles (761 issues per wave); the WMMA is not
in the top rows. K2 asks memory for 392 bytes per lane per 4 blocks of which 72 are weights; the rest is activation
planes and scales that every lane of every wave reads again.

### Predictions (frozen by this commit, before the run)

| id | claim | pass if |
|---|---|---|
| K6.1 | the activation-side loads cost K2 at least 10 us on the gate shape | ABL-A (planes and scales always from blocks 0..3) <= K2 - 10 us |
| K6.2 | with weights hot too the kernel is far below turbo m = 1 | ABL-AQ <= 45 us gate (the compute floor) |

### K6 result (`exchange/receipts/verify8-r2/k6-abl.log`, one gpu-wait job): where K2's 89 us go

| arm, m = 8, P = 3, D = 4, prepared A | gate us | down us |
|---|---|---|
| turbo m = 1 | 62.7 | 65.7 |
| K2 as it is | 89.2 | 104.5 |
| K2, planes and scales always from blocks 0..3 (hot) | 77.8 | 85.0 |
| K2, weights hot too | 66.6 | 61.6 |

K6.1 PASS: the activation-side loads cost 11.4 us (gate) and 19.5 us (down). K6.2 FAIL: with every load hot the kernel
still takes 62 to 67 us, the whole budget of the 750 GB/s line.

Split of K2 (gate / down): 67 / 62 us floor with hot memory, 11 / 23 us for the weights from DRAM, 11 / 20 us for
planes and scales. So the 8-row kernel is not memory-bound first: its floor with no DRAM traffic already equals the
1-row kernel's full time. Two levers follow, in this order:
1. the hot floor (62 to 67 us): per-wave round trips and issue, to be read from a wavewait capture of arm 34, not guessed;
2. activation-side bytes: signed weight nibbles (`q ^ 0x88888888`, B operand signed) remove the `-8 * sum` term and
   halve the scale record (32 -> 16 B per lane per block); planes for 8 live tokens only.
Neither is done in this lane round. K2 is wired as it is.

## E1: K2 wired into the 8-row window (`BARO_FEWROW=2`, `BARO_FW_PLANES` default 3)

`serve/window.mojo::gemm_w8` (the 13 GEMM call sites of the window loop): Q4 GEMMs with m > 4 run `amar_fw_prep4` then
`amar_fewrow_iu4` D=4; scratch in `WindowBufs.fwq_d`, `fws_d` (835 KB + 35 KB at FFN 17408, 16 bytes otherwise).
`BARO_FEWROW` 0 and 1 builds are unchanged in route. ISA receipt of `eng-fw2`: 9 instances, 87 to 90 VGPR, 0 spills.

A/B: `bench/ab-resident.sh`, same tree, two binaries: A = `-D BARO_FEWROW=1` (bf16 WMMA, `eng-fw1b` 802737d0),
B = `-D BARO_FEWROW=2` (`eng-fw2` 055eec15); MTP k = 7 (the 8-row window; the DFlash2 arm needs the missing draft
pack), 20 mtp-prompts, GEN 256, TMAX 16384, BARO_MEGA=0, TurboFCFusion q4 pack. Numbers on this pack are not nerkyor
numbers.

### Predictions (frozen by this commit, before the run)

| id | claim | pass if |
|---|---|---|
| E1.1 | the faster GEMM carries over | median tok/s_gen B / A >= 1.08 (FFN GEMMs are 46.6% of the window's kernel time at 0.75x to 0.82x; the other shapes unmeasured) |
| E1.2 | acceptance does not fall | accepted / drafted of B within 2 points of A |
| E1.3 | ids move no more than the WMMA route's own noise | reported only: prompts whose ids differ between A and B; no bar, because neither arm is the exact kernel and A is not run twice here. The gate for a default is forced agreement with a same-arm control on DFlash2, not run (packs missing) |

### E1 result (`exchange/receipts/verify8-r2/e1/`, tree with the wiring commit, one gpu-wait job)

Read back from the running arms: `spec k: 7` both, `BARO_MEGA: False`, `BARO_SPEC: True`, fail word 0, engines
802737d0 / 055eec15, power cap 290 W. Deviation: host 1-min load was 16.2 at start (max 18.6 in A, 15.2 in B; other
lanes' builds), above the harness limit of 4, so `LOAD_START_MAX=30`; both arms ran in one job under the same load.
VRAM peak 23.3 GB with 0.7 GB in GTT in both arms (TMAX 16384 f32 KV beside the desktop).

| arm | median tok/s_gen (20 prompts) | accepted / drafted |
|---|---|---|
| A bf16 WMMA (`BARO_FEWROW=1`) | 71.76 | 3631 / 10264 (0.354) |
| B iu4 P=3 (`BARO_FEWROW=2`) | 84.97 | 3647 / 10173 (0.358) |

Ratio 1.184, per prompt 0.876 to 1.708. E1.1 PASS, E1.2 PASS. E1.3: ids differ between A and B on 4 of 20 prompts
(p03, p04, p06, p09); no same-arm repeat was run, and this shape is known to flip 2 to 8 prompts between runs of one
binary, so 4 says nothing yet.

## Verdict per step

| step | verdict |
|---|---|
| 2 trace table | DONE for the 3-row and 8-row MTP windows on the TurboFCFusion pack. NOT DONE for DFlash2 on the nerkyor pack: both packs are missing, ASK unanswered at report time |
| 3 8-row GEMM toward 750+ GB/s | NOT MET. Best kernel K2 `amar_fewrow_iu4`: 533 / 454 GB/s incl. prep (550 / 482 prepared) against 421 / 392 for the bf16 WMMA. Its floor with all memory hot is 62 to 67 us, the whole 750 GB/s budget (K6) |
| 3 wiring | LANDED as opt-in `-D BARO_FEWROW=2`: 8-row MTP window 71.76 -> 84.97 tok/s_gen (1.184x, same job). Default route unchanged |
| 3 identity | UNVERIFIED: the gate (teacher-forced agreement, same-arm control, DFlash2) needs the nerkyor q4 pack and the draft pack. Kernel-level: fp64 host check, 3.7e-4 / 2.1e-4 of max at P = 3; not bit-exact by construction |
| 3 m = 3 for MTP | closed without a run: a WMMA kernel is flat in m at about 90 us, the dot2 row kernel is 66 us at m = 3 |
| row kernels at m > 4 (K0), bit-built bf16 (K1), LDS staging (K3), crossed halves (K4), split unroll (K5) | closed with numbers above; the crossed-halves WMMA form is correct and stays in the file as `amar_fewrow_iu4x` for a later geometry |
| 4 launch count of the window | NOT STARTED |
| target 120 to 140 tok/s_gen for DFlash2 | NOT MEASURED (no draft pack). On the MTP 8-row window the round moved 71.8 -> 85.0 |

Not run: `bench/preflight.sh`, `run-tests.sh`, a `bench/ww.sh` capture of the wired kernel in the engine (the capture
here is the bench one), `tools/ci-checks.sh` for the new build flags. The lane is not merge-ready.

## Next, in order

1. Packs: rebuild the nerkyor q4 pack and the DFlash2 draft pack outside a lane worktree, then the DFlash2 trace and
   the D3-style gate with `eng-fw2` against `eng-fw1b` and the q4rowb control.
2. K2's hot floor: wavewait capture of bench arm 34, then the signed-nibble B operand (`q ^ 0x88888888`, drops the sum
   term and 16 of 32 scale bytes) and planes for live tokens only.
3. One `amar_fw_prep4` per distinct A (q/k/v, gate/up and the four SSM GEMMs share theirs): 13 prep launches per layer
   become about 6.
4. Step 4 of the brief.

## Skill, SINDEX and tool changes (addendum)

- `~/Brain/Skills/own/gpu-kernels/references/isa-rdna3.md`: WMMA paragraph corrected (a result lane uses the B data of
  its own half-wave; crossed-halves form; Q4_0 bytes are valid iu4 operands).
- `~/Brain/Skills/own/gpu-kernels/references/inference.md`: "Row kernels stop at m = 4" paragraph (numbers, the
  16-bit narrowing of dword bit ops, N / 32 waves is latency-bound).
- `~/Brain/Skills/own/ops/SKILL.md`: `gpu-wait run` needs `--mem`, `--timed` for host-timed jobs. `skilldex` run: PASS.
- `~/Brain/SINDEX/kernels.md`: two entries (WMMA operand lanes assumed unread; bench scratch shared across layouts).
  `~/Brain/SINDEX/mojo-baro.md`: one entry (pack lived only in a removed lane worktree).
- `docs/rules.md`: three lines, 2026-11-25.
- iTools: none changed.
- Whiteboard: `~/Brain/mojo/mojo-baro/whiteboard.md` holds no card for this lane (grep `verify8-r2`: 0 lines), so none is ticked.

## K7 (after the report-back; the maintainer 2026-11-25: use wavewait, corewait, heaptide)

### Second capture: K2 with its loads hot (`.work/v8r2/att-abl`, tables in `exchange/receipts/verify8-r2/ww-abl/`, mism 0, open 0)

Gate shape, cycles per traced wave: K2 58,517; planes and scales hot 28,002; all hot 23,765. The wave got 2.46x
shorter, the kernel only 1.34x faster (89.2 -> 66.6 us). In the all-hot wave: waits on the (hot) loads 27.3%, wave
start 12.4% (2,949 cycles before the first instruction issues, 524 in the full kernel), final barrier 11.6%,
`v_dual_mov_b32` 9.9% (the zeroed C operands of the plane chain), WMMA 8.4% (120 per wave, 16.6 cycles each),
`v_lshlrev` 7.9%. So in the hot regime the wave waits to issue, and the largest VALU items are the plane chain itself.

### K7 kernel `amar_fewrow_iu4r`

m <= 8 leaves A rows 8-15 of every WMMA unused. K7 puts a second nibble plane there: rows 0-7 plane 2s, rows 8-15
plane 2s + 1 of the same tokens. All planes unsigned plus one sign-bit plane (two's complement), weights signed by
`q ^ 0x88888888` (B operand signed), so the `-8 * sum` term and half the scale record go away. Per block at P = 3:
4 WMMA (was 6), no chained zero operands, 5 loads of 16 B (was 7 loads, 98 B), 16 B scale (was 32). P = 4 costs
6 WMMA (was 8). ISA receipt D=4: P3 84 to 122 VGPR over the instances, 0 spills.

### Predictions (frozen by this commit, before the run)

| id | claim | pass if |
|---|---|---|
| K7.1 | correct at K2's error | host check P3 <= 4e-4, P4 <= 4e-5 of max, both shapes |
| K7.2 | faster than K2 at P = 3 | IU4R P3 prepared <= 0.92 x IU4 P3 prepared, both shapes |
| K7.3 | the hot floor falls | IU4R-HOT <= 0.88 x IU4-HOT, both shapes |
| K7.4 | 16-bit activations at about K2's 12-bit cost | IU4R P4 prepared <= 1.03 x IU4 P3 prepared, both shapes |

### K7 result (`exchange/receipts/verify8-r2/k7-iu4r.log`, one gpu-wait job)

| arm, m = 8, D = 4 | gate us | down us | error of max (gate, down) |
|---|---|---|---|
| K2 IU4 P3 incl. prep / prepared | 93.0 / 89.1 | 108.6 / 105.0 | 3.7e-4, 2.1e-4 |
| K7 IU4R P3 incl. prep / prepared | 84.8 / 80.8 | 117.5 / 111.7 | 3.7e-4, 2.1e-4 |
| K7 IU4R P4 incl. prep / prepared | 88.0 / 84.1 | 110.8 / 104.9 | 1.9e-5, 1.3e-5 |
| K2 all loads hot | 66.7 | 61.2 | - |
| K7 P3 all loads hot | 58.8 | 49.5 | - |

K7.1 PASS. K7.2 gate PASS (0.907), down FAIL (1.064). K7.3 PASS (0.88 / 0.81). K7.4 PASS (0.94 / 1.00): 16-bit
activations now cost what 12-bit did.

Odd point: on the down shape P = 3 (two plane sets) is slower than P = 4 (three sets), with fewer WMMAs and loads. The
plane buffer is row-major `[16, K/32 * NS * 16]`; at K = 17408 and NS = 2 the 16 lanes of one load read addresses
17408 B apart, 1024 modulo 4096, so the 16 rows fall on 4 page offsets. Lead, not a finding. K7b tests it: plane
buffer block-major `[K/32 * NS, 256]`, the 16 lanes of a load read one contiguous 256 B.

### K7b predictions (frozen by this commit, before the run)

| id | claim | pass if |
|---|---|---|
| K7b.1 | block-major planes remove the down-shape penalty | IU4R P3 prepared down <= 100 us |
| K7b.2 | and help the gate shape | IU4R P3 prepared gate <= 78 us |

### K7b result (`exchange/receipts/verify8-r2/k7b-iu4r.log`): the layout lead is false

Block-major planes: P3 prepared 80.5 / 111.2 us (row-major 80.8 / 111.7), hot floor 59.4 / 49.8. K7b.1 FAIL, K7b.2
FAIL. The plane layout does not matter; block-major stays (one contiguous 256 B per load is not worse).

Down shape, split again: K7 111 us = 50 us hot floor + 61 us memory; K2 105 = 61 + 44. The floor fell by 11 us and
the memory part grew by 17. The down shape has 2,560 waves (N / 16 x 8) against 8,704 on gate, each with D = 4
weight blocks in flight; a faster wave drains its 4 blocks sooner and then waits. D = 2 -> 4 took K7 from 137 to 111 us
on down and from 98 to 80 on gate.

## K8: unroll depth sweep of `amar_fewrow_iu4r` (tail guard added, any D is legal)

K5's sweep was not a test of depth: it loaded the planes after the weight wait and so added a round trip per group.
K8 loads everything of D blocks at once, as K2 and K7 do, with D = 4, 5, 6, 8, 10, 12 (17 registers per block at P = 3).

### Predictions (frozen by this commit, before the run)

| id | claim | pass if |
|---|---|---|
| K8.1 | correct with the tail guard | host check at D = 8 and D = 6 at the K7 errors |
| K8.2 | the down shape gains from depth | best D prepared P3 down <= 100 us |
| K8.3 | the gate shape gains from depth | best D prepared P3 gate <= 76 us |
| K8.4 | registers end it | time rises again by D = 12 (spills or fewer waves) |

### K8 result (`exchange/receipts/verify8-r2/k8-depth.log`): the tail guard itself cost time; run void for K8.3

P3 prepared, us gate / down by D: 4: 95.8 / 155.0, 5: 107.5 / 121.7, 6: 110.7 / 101.3, 8: 113.1 / 110.6,
10: 80.9 / 86.9, 12: 90.6 / 90.7. K2 control 91.3 / 105.7. K8.1 PASS (errors as K7).
The D = 4 arm is the K7 kernel plus the guard and went from 80.5 / 111.2 to 95.8 / 155.0, so the guard (a per-block
`min` on the block index and a select) changed the kernel, and the sweep mixes two effects. One point stands: D = 10
gives 86.9 us on the down shape, the best 8-row number on that shape so far (K8.2 PASS); K8.3 and K8.4 are not judged.

Deviation, stated before run K8b: the guard is replaced by an overlapping last group (`cc = min(c, NBW - D)`, block
index stays affine, blocks already counted get scale 0), and it is compiled out when D divides the per-wave block count.
Same predictions K8.2 to K8.4, plus K8b.0: D = 4 returns to 80.5 / 111.2 within 3%.

### corewait on the wired engine (`bench/cw-decode.sh`, `eng-fw2`, MTP k = 7, p02): VOID as a limiter receipt

`gpu_busy_percent` was 0 in 182 of 200 samples: the 1500-token request ended (102.9 tok/s_gen) before most of the
10 s record window. Host cycles in that window: libhsa-runtime 83.1%, libamdhip 8.0%, engine 0.5%
(`.work/v8r2/cw-fw2/cw/dsos.tsv`). The window must sit inside decode: the script needs a request length knob.

### K8b result (`exchange/receipts/verify8-r2/k8b-depth.log`, one gpu-wait job)

K8b.0 PASS: D = 4 is back at 80.7 / 112.2 us. P3 prepared A, us gate / down by D:

| D | 4 | 5 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| gate | 80.7 | 89.1 | 91.8 | 85.1 | **77.9** | 82.5 |
| down | 112.2 | 117.1 | 104.3 | 97.9 | **88.2** | 88.3 |

K2 control 89.5 / 104.2, turbo m = 1 62.7 / 65.0. K8.2 PASS (88.2 <= 100). K8.3 FAIL by 1.9 us (77.9, bar 76).
K8.4 PASS on gate (rises at D = 12, 168 to 216 VGPR), flat on down. The curve is not smooth (D = 5 and 6 lose on gate:
an overlapping tail group reloads blocks), so D = 10 is a measured pick, not a law: 20 blocks per wave on gate is two
exact groups, 68 on down is six groups and an overlap of 2.

K7 at D = 10: 644 / 569 GB/s (prepared A), against 421 / 392 for the bf16 WMMA and 550 / 482 for K2.

### corewait, second recording (`.work/v8r2/cw-fw2b`, `eng-fw2`, MTP k = 7, 4000-token request, record inside decode)

Host cycles: libhsa-runtime 80.4%, libamdhip 9.7%, libc 3.7%, KGEN runtime 3.6%, engine binary under 1%. The host
thread spends four fifths of its cycles in the HSA wait, so the 8-row window is not CPU-bound; about 17% is enqueue
work. The script's `gpu_busy_percent` column read 0 in 173 of 200 samples while the engine was decoding, so that sysfs
sample is not a usable busy meter at 50 ms here and is not quoted.

## E2: K7 wired (`-D BARO_FEWROW=3`, D = 10 where a wave has >= 10 blocks, else 4)

Same tree, two binaries: A = `eng-fw2b` 932950b4 (`BARO_FEWROW=2`, K2), B = `eng-fw3` 85861001 (`BARO_FEWROW=3`, K7).
ISA receipt of `eng-fw3`: 9 instances, 168 to 178 VGPR, 0 spills. Harness and pack as E1.

### Predictions (frozen by this commit, before the run)

| id | claim | pass if |
|---|---|---|
| E2.1 | K7 carries over K2 | median tok/s_gen B / A >= 1.04 |
| E2.2 | acceptance equal | within 2 points |

### E2 result (`exchange/receipts/verify8-r2/e2/`, one gpu-wait job): PASS in form, weak in evidence

Read back: `spec k: 7` both arms, fail word 0, engines 932950b4 / 85861001.

| arm | median tok/s_gen (20 prompts) | accepted / drafted |
|---|---|---|
| A K2 (`BARO_FEWROW=2`) | 75.56 | 3664 / 10043 (0.365) |
| B K7 D=10 (`BARO_FEWROW=3`) | 89.33 | 3662 / 10047 (0.364) |

Ratio 1.182, per prompt 0.776 to 2.241, ids differ on 2 of 20 (p03, p10). E2.1 PASS, E2.2 PASS.
Why I do not trust the size: host 1-min load was 26.5 at start and reached 37.9 (A) and 35.8 (B), other lanes'
builds; arm spread 84% and 68%; the arms ran one after the other; and 1.18 is more than the kernel bench explains
(0.87 / 0.85 on the FFN shapes, which are 47% of the window's kernel time, gives about 1.07 unless the other shapes
gain more). E3 repeats it with alternating 5-prompt blocks (`SEQ=1`) against the bf16 WMMA arm.

### heaptide on the K7 engine (`heaptide run --gpu`, `eng-fw3`, 20 prompts x 256 tokens resident, MTP k = 7)

`exchange/receipts/verify8-r2/ht/report.txt`: GROWTH PASS, slope 0 B/s over the window; leaks 11.4 KiB direct, all in
HIP event creation and code-object loading (one per kernel instance), none in the new scratch. Peak: pack 17.38 GB on
the GPU. The new route adds no growth. Not a finding of this lane, seen in HOTSPOTS: `window::step_window` creates
owning device buffers per window (95,040 allocations in the run), a host-side cost to look at with step 4.

## E3: K7 against the bf16 WMMA, alternating blocks

A = `eng-fw1b` 802737d0 (`BARO_FEWROW=1`, built before the K7 wiring commit; its route did not change),
B = `eng-fw3` 85861001. `SEQ=1 BLOCK=5`, 20 prompts, GEN 256, TMAX 16384, MTP k = 7.

### Prediction (frozen by this commit, before the run)

| id | claim | pass if |
|---|---|---|
| E3.1 | K7 over the bf16 WMMA route | paired median ratio >= 1.25 (kernel bench 0.65 / 0.69 on 47% of kernel time, K2 measured 1.184) |

### E3 result (`exchange/receipts/verify8-r2/e3/`, one gpu-wait job, 4 alternating blocks of 5 prompts)

Read back: `spec k: 7` in all 8 arm runs, fail word 0. Host 1-min load 34.7 at start, 25 to 35 during the blocks.

| statistic | A bf16 WMMA | B K7 D=10 | B / A |
|---|---|---|---|
| median of the arm's 20 tok/s_gen | 70.97 | 78.22 | 1.102 |
| median of the 20 per-prompt ratios | - | - | 1.227 (0.581 to 1.333; B faster on 18 of 20) |

E3.1 FAIL as frozen (paired median 1.227, bar 1.25). ids differ on 5 of 20 between the arms; no same-arm repeat.

What the three engine runs say together: K7 is faster than the bf16 WMMA route in the 8-row window, by about 1.10x on
the ratio of medians and 1.23x on the paired median, measured on a host at load 25 to 35. E1 (K2 over WMMA 1.184) and
E2 (K7 over K2 1.182) ran their arms one after the other at loads 16 to 38; their product, 1.40, is not confirmed by
E3 and is withdrawn as a claim. Number to quote for this round: 70.97 -> 78.22 tok/s_gen, alternating blocks,
TurboFCFusion q4 pack, MTP k = 7, loaded host. A quiet-box repeat is owed.

## Verdict per step, after K7 (replaces the table above where they differ)

| step | verdict |
|---|---|
| 3 8-row GEMM toward 750+ GB/s | NOT MET. Best kernel K7 `amar_fewrow_iu4r` D=10: 644 / 569 GB/s on a prepared A (77.9 / 88.2 us), from 421 / 392 (bf16 WMMA). Hot floor 59 / 50 us at D=4 |
| 3 wiring | LANDED as opt-in `-D BARO_FEWROW=3` (K7) and `=2` (K2). 8-row MTP window 70.97 -> 78.22 tok/s_gen (E3). Default unchanged |
| 3 identity | UNVERIFIED: same gap as before (nerkyor q4 pack and DFlash2 draft pack missing). Kernel fp64 check 3.7e-4 / 2.1e-4 of max at P = 3, 1.9e-5 / 1.3e-5 at P = 4 (P = 4 not timed in the engine) |
| instruments (the maintainer 2026-11-25) | wavewait: two captures, they produced K7 and the time split. corewait: host 80.4% in the HSA wait, not CPU-bound. heaptide: growth PASS, no leak from the new route |

Tool changes since the first report-back: `bench/cw-decode.sh` takes `BARO_PACK`, `REQ_SPEC`, `CW_N` (it pointed at a
removed lane's pack and its window missed the decode); its `gpu_busy_percent` column reads 0 during decode and should
be replaced by the kernel-trace busy share. No iTool changed. SINDEX: one more entry (per-block `min` guard, K8).

## E4: K7 identity gate on the nerkyor pack (driver, frozen before the run)

Packs rebuilt 2026-11-25 in main: `.work/nerkyor/pack-q4` (engine-pack --q4 from nerkyor-27B-BARO-91f72c9.gguf,
pack_sha256 9485055a...), `.work/dflash2/draft-pack` (pack_sha256 4b39f396..., equal to the D2a receipt).
Harness `bench/dflash2-d3.sh .work/v8r2/eng-fw3 .work/v8r2/eng-base PACK OUTDIR`, one gpu-wait job, 20 mtp-prompts,
GEN 256, TMAX 16384. FW1 = K7 (`BARO_FEWROW=3`), FW0 = the shipped route (same-arm control). Bars as the D3 gate:

| id | measure | bar |
|---|---|---|
| G1 | forced agreement pooled, f1 (K7) vs f0 (shipped), both vs the m=1 reference | f1 >= f0 - 0.5 pp |
| R1 | reverse arm f1s tokens per cycle; predicted ids equal f1's | f1s <= 0.75 x f1; 20/20 |
| G3 | live dflash2 vs MTP k=2, median tok/s_gen, same binary (K7), same job | ratio > 1.00 |

Decision rule: G1 and R1 PASS -> `BARO_FEWROW=3` becomes the default; G3 decides whether dflash2 becomes the default
spec path. Any FAIL -> stays opt-in.

### E4 result (`.work/v8r2/e4.log`, one gpu-wait job, nerkyor q4 pack, 20 prompts): PASS

| arm | result |
|---|---|
| ref (m=1, no spec) | 39.79 tok/s_gen |
| f1 K7 forced | agreement 0.99687 (5104/5120, min 0.961), 4.914 tokens/cycle |
| f0 shipped forced (control) | agreement 1.00000, 4.969 tokens/cycle |
| f1s reverse | 3.055 tokens/cycle, predicted ids equal f1 on 20/20 |
| live dflash2 (K7) | 88.42 tok/s_gen, 4.009 tokens/cycle, ids != ref on 6/20 |
| MTP k=2 (K7 binary) | 86.28 tok/s_gen |

G1 PASS (0.99687 >= 0.995), R1 PASS, G3 PASS (1.025). Applied, scoped to what the gate covered: the nerkyor27b profile
builds with `-D BARO_FEWROW=3` and serves `BARO_SPEC=dflash2`, `BARO_MEGA=0` (the gated config). The global
`BARO_FEWROW` default stays 0: the flag also reaches prompt replay of every qwen35 model (docs/settings.toml), and E4
covers only the 27B shape. The turbo27b-coder profile has no DFlash2 draft and runs MTP k=2 (3-row window), where K7
does not apply.

## Merge receipts (driver, 2026-11-25)

`bench/preflight.sh` PASS on 52fbde8; `./run-tests.sh` under gpu-wait: rc=0, FAIL: 0 (none), `PASS run-tests: all tests` (log `.work/run-tests-v8r2.log`). Skips: test_kq_prefill_gemm (no ornith-stint pack), one real-decode row.
