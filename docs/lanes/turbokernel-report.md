# Lane turbokernel: decode kernel for Qwen3.8-27B Turbo (qwen35, Q4_K_M, MTP)

Brief `.work/briefs/turbokernel.md`. Branch `lane-turbokernel`, worktree `mojo-baro-lanes/turbokernel`, Mojo 1.2.0 (main `.venv`).
Model `~/Models/qwen3.8-27b-turbo-fcf-mtp-q4_K_M/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf`
(arch qwen35, 866 tensors). Bake of the same file by lane turbobake: `...-BARO-c71dbdc.gguf`, `tools/baro verify` PASS 64/64 on
2026-11-02 (its run env `BARO_MEGA=1 BARO_SPEC=1 BARO_EMBD_HOST=1`, pack flag `--q4`).

## 0. Shape and roofline (from the pack index, element counts)

| constant | value | source |
|---|---|---|
| H | 5120 | `-D BARO_H=5120` (bake build line) |
| layers | 64 = 48 SSM + 16 attention (every 4th) | `-D BARO_N_LAYERS=64` |
| NQH / NKVH / HD | 24 / 4 / 256, ATT = 6144 | `-D BARO_NQH=24` |
| NH_V / SSTATE | 48 / 128, INNER = 6144, conv rows 10240 (16 K heads + 48 V heads) | `-D BARO_SSM_NH_V=48`, `attn_qkv` 5120 x 10240 |
| FFN | 17408 | `-D BARO_FFN=17408` |
| VOCAB | 248320 | `kernels/mega.mojo` |

Kinds in the K-quant trunk (`engine-pack --kq`): SSM layer `attn_qkv` Q6_K, `attn_gate` (z) Q4_K, `ssm_out` Q4_K, `ffn_gate`/`ffn_up` Q4_K,
`ffn_down` Q6_K; attention layer `attn_q` (q + gate, 5120 x 12288) Q4_K, `attn_k` Q4_K, `attn_v` Q6_K, `attn_output` Q4_K; `output.weight`
q8 (converted), `token_embd` bf16. Pack 18.24 GB (q4 pack of the bake: 17.20 GB).

Bytes per decode token on the kq trunk (Q4_K 0.5625 B/elem, Q6_K 0.8203, q8 1.0625):
48 SSM layers x 251.8 MB + 16 attention layers x 233.7 MB + head 1.35 GB = **17.2 GB**, floor 17.9 ms = **55.8 tok/s at 960 GB/s**;
at the mature-runtime 80% of peak, 44.6 tok/s. The brief's "about 45 tok/s" is already there, so the m = 1 pool is at most about 1.25x
and the speculative window (k = 2 MTP, 3 rows per step reading the weights once) is the larger lever. Measured in step 1.

## 1. Clean baseline (step 1), engine `.work/turbokernel/engine-81a50f7` sha 4aa840f42faf0fba, tree 81a50f7

Build: `.work/turbokernel/build-engine.sh` = the bake's line (`-D BARO_MODEL=qwen35 -D BARO_H=5120 -D BARO_N_LAYERS=64 -D BARO_NQH=24
-D BARO_FFN=17408 -D BARO_SSM_NH_V=48 -D BARO_YARN=0 -D BARO_SEQ_CAP=1 -D BARO_SLOTS=4 -D BARO_DOT_UNROLL=2`), Mojo 1.2.0 (2016d0cd).
Packs: q4 = the bake's pack `~/.cache/baro/qwen35-eec459db896e8bc6/packs/bc7a6cf2...` (17.20 GB, `--q4`, pack-fit headroom 5.6 GB at
TMAX 16384); kq = `.work/turbokernel/pack-kq` (`engine-pack --kq` from the lane tree, 18.24 GB, headroom 3.7 GB).
Harness `bench/turbo-base.sh` (resident arms, `bench/ab-resident.sh`, 20 mtp-prompts, 64 tokens, f32 KV, TMAX 16384 read back,
`BARO_MEGA` / `BARO_SPEC` / pack kind / fail word read back per arm, spec per done line). Host load during the arms was 6 to 9
(other lanes' processes, outside the gpu room): recorded per arm as `load_max`, the launch path (one host enqueue per kernel) is
the arm it can hurt.

| job | pack | spec | launch median (min / max) | mega median (min / max) | mega / launch | ids mega = launch | receipt |
|---|---|---|---|---|---|---|---|
| mvzd0p8262qx | q4 | off | 38.15 (31.09 cold p01 / 38.83) | 44.40 (40.65 cold p01 / 44.96) | 1.164 | 18/20 (p03 token 24, p08 token 49 differ) | `.work/turbokernel/base-q4-spec0/` |

The two identity misses are the prompts the OrcaSAQ-27B sessions already flagged as near-tie prompts (p03, p06, p08, p09, p17 vary
between binaries on that shape while a repeated binary agrees with itself). A same-arm repeat (`turbo-det`, mega twice on p01/p03/p08)
tells run-to-run noise from a path difference; result below.
| mvzd4al2dgu9 | q4 | on (k 2) | 62.38 (48.57 / 77.71) | 62.88 (51.13 / 76.99) | 1.008 | 18/20 (p03, p08) | `.work/turbokernel/base-q4-spec1/`; drafted 1073 accepted 731 (launch), 1063 / 737 (mega): 68% |

Same-arm repeat `.work/turbokernel/det-q4-mega/` (megakernel twice, same binary, env differs only in a dummy variable, p01/p03/p08):
p03-story ids DIFFER between the two runs, p01 and p08 equal. So the 27B shape has a run-to-run nondeterminism on near-tie prompts
(the OrcaSAQ-27B sessions saw p03, p06, p08, p09, p17 vary; the 9B never does). Consequence for this lane: the prompt-level identity
gate is read on the stable prompts and backed by a same-arm control; the kernel-level byte test is the strict check. The cause (a
read of memory nobody wrote at this shape is the standing hypothesis) is a separate concern, flagged here, not hunted in this lane.

Spec arithmetic (q4, k = 2): 62.9 tok/s at 1.68 tokens per step (731 / 1073 accepted + 1) = 37.4 steps/s = 26.7 ms per step for a
3-row window plus the draft head, against 22.5 ms for one m = 1 megakernel token (44.4 tok/s) and a 15.3 ms byte floor
(14.66 GB of weights after the host-side embedding). The window step is at 1.75x the byte floor, the m = 1 token at 1.47x.
| mvzd4al2dgu9 (3rd job) | kq | off | 34.42 (spread 25%, cold p01) | 34.47 (33.5 / 35.0) | 1.002 | 17/20 (p07, p09, p17) | `.work/turbokernel/base-kq-spec0/`; 62% of the kq byte floor on both paths |
| 4th job | kq | on | VOID | VOID | - | - | engine refuses: `kq pack: the MTP draft reads q4/q8 only; set BARO_SPEC=0` (`base-kq-spec1/launch.log`) |

Winner of step 1: **q4 pack + spec, 62.9 tok/s**, every step of it on the launch path at m = 3 (k = 2 drafts + 1) plus two draft-head
passes per window; the m = 1 megakernel is not on that path (`use_mega` needs m == 1 and no window). Second: q4 mega no-spec 44.4.
The kq trunk is 22% slower than the q4 pack at m = 1 (more bytes: Q6_K rows for qkv/down/v plus q8 head) and has no spec, so the
kq megakernel regression (BASELINE 87.9 vs 98.3 on the 9B) is not the lane's lever on this model: both kq paths read 34.4 here.

Host-side split, one prompt p01, q4 pack (`.work/turbokernel/prof-q4/`, receipts not claims; `BARO_PROFILE=1` syncs inflate the
window numbers, ratios are what to read):

| path | per m = 1 token (mega, `BARO_PROFILE=5`, last token) | per 3-row window (spec, `BARO_PROFILE=1`, 24 windows) | window / token |
|---|---|---|---|
| ssm sub-blocks | 6.67 ms (byte floor 3.25) | 14.1 ms | 2.1x |
| attn sub-blocks | 1.81 ms (floor 0.98) | 5.8 ms | 3.2x |
| ffn | 13.69 ms (floor 10.0) | 23.9 ms | 1.75x |
| head | 0.81 ms (floor 0.74) | 1.4 ms | 1.7x |
| MTP process + draft | - | 3.5 ms (`BARO_PROFILE=3`: draft head GEMM 2.05, layer 1.33, argmax 0.48 per window) | - |
| total | 23.0 ms | about 27 ms per window at 2.67 tokens per window on p01 | |

Reading: the spec window is row-scaling-bound (P5 names 1.25x at m = 4 the defect line; here 1.75x to 3.2x at m = 3 against a byte
floor that does not move with m). The specialized kernel target is the m = 3 window on this shape, FFN first (55% of the window).

Correction to the spec arithmetic above: tokens per window = 1 + 2 x acceptance = 2.36 (not 1.68), so 62.9 tok/s is 26.6 windows/s =
**37.5 ms per 3-row window** against 22.5 ms per m = 1 megakernel token (1.67x) and a 15.3 ms byte floor.

## 2. Profile (step 2): kernel trace and SQTT of the winning path

Kernel trace (`rocprofv3 --kernel-trace`, resident run, p01, 200 tokens, k = 2, 56.7 tok/s under the tracer, `.work/turbokernel/ktrace-spec/`,
`wavewait rank`): 110176 dispatches, 63 kernels, kernels busy 3071 ms of a 4533 ms span; 80 windows, so about 36 ms of kernels and 8 ms of
gaps per window (tracer included). Instances mapped by grid size (q4rowb grid = 32 N threads):

| kernel (q4 pack, launch path) | shape N x K | per window | avg us | MB | GB/s | VGPR (isa-receipt) |
|---|---|---|---|---|---|---|
| `q4rowb[2,3]` FFN gate, up | 17408 x 5120 | 2 x 64 | 87.8 | 50.1 | 571 | 126 |
| `q4rowb[2,3]` FFN down | 5120 x 17408 | 64 | 90.9 | 50.1 | 551 | 126 |
| `q4rowb[2,3]` ssm_out / attn_output | 5120 x 6144 | 64 | 44.5 | 17.7 | 398 | 126 |
| `q4rowb[2,3]` ssm qkv | 10240 x 5120 | 48 | 56.5 | 29.5 | 522 | |
| `q4rowb[2,3]` ssm z | 6144 x 5120 | 48 | 38.8 | 17.7 | 456 | |
| `q4rowb[2,3]` attn q | 12288 x 5120 | 16 | 65.6 | 35.4 | 540 | |
| head, verify (m = 3) | 248320 x 5120 | 1.65 | 1054 | 715 | 678 | 126 |
| head, draft (m = 1) | 248320 x 5120 | 1.16 | 803 | 715 | 890 | 95 |
| draft layer FFN gate/up (m = 1) | 17408 x 5120 | 4.8 | 109.8 | 50.1 | 456 | 142 |
| rmsnorm_cast | | 140 | 9.4 | | | |
| ssm_delta_step | | 48 | 27.0 | | | |

One `q4rowb` dispatch of 123 ms (N = 5120, prefill, 14 siblings at 72 us) is a stall outlier, excluded.

SQTT (`bench/turbo-att.sh`, wavewait `lane-spillcost` 3239982, JSON `.work/kfacts/ww/turbo27-q4-mega_token-m1.json`, `turbo27-q4-q4rowb-m3-ffn_gate-17408x5120.json`, `turbo27-q4-q4rowb-m3-ffn_down-5120x17408.json`, `turbo27-q4-q4rowb-m3-out-5120x6144.json`, copies in the primary checkout's `.work/kfacts/ww/`):

| dispatch | waves traced | vmcnt wait | VALU | LDS | barrier | spill | hot loop |
|---|---|---|---|---|---|---|---|
| megakernel m = 1, dispatch 3789 (token 12) | 4 | 49.1% | 23.5% | 5.5% | 11.2% | 0.1% | three q4 dot loops 16 + 16 + 14%, all mem, ~900-cycle waits |
| FFN gate m = 3 `q4rowb[2,3]` | 88 | 69.4% | 23.7% | 0.3% | 0 | 0 | one loop 67%, 601 instructions, seven vmcnt waits per iteration of 760 to 1580 cycles |
| FFN down m = 3 | 26 | (tables in `.work/turbokernel/att-ffn3/`) | | | | 0 | |
| out m = 3 | 28 | | | | | 0 | |

Limiter named with numbers: the window GEMV is latency-bound at 4 waves per SIMD (126 VGPRs) with 32 B of weights per lane in flight
(UNROLL 2) and six A-row loads per iteration each exposing an L2-class latency. Across the instances in the trace, GB/s follows waves per
SIMD: head m = 1 at 95 VGPRs 890 GB/s, gate m = 3 at 126 VGPRs 571, draft gate m = 1 at 142 VGPRs 456. FMA count is not the bound
(VALU 24%).

## 3. Frozen predictions (step 3), before `bench/bench_turbo_rows.mojo` runs

Instrument: `bench/bench_turbo_rows.mojo`, the real 27B q4 pack's `blk.0..7.ffn_gate` (17408 x 5120) and `ffn_down` (5120 x 17408), 8 tensors
rotated (400 MB per shape, above 2 x the 96 MB Infinity Cache), 3 warm passes, 11 rounds of 8 launches, median us, fp64 host check of the
m = 3 FULL arm against dequantized weights on tensor 0. Arms: `q4rowb[2,MR]` FULL at MR 1, 2, 3, 4 (the P5 row-scaling receipt);
MR 3 NOTRAFFIC (every row reads A row 0, same loads, same FMAs); MR 3 NOFMA (rows 1 and 2 load A but do not accumulate);
`q4rowb[4,3]` and `q4rowb[8,3]` (depth, the real kernel at UNROLL 4 and 8, bit-identical by construction).

| id | prediction | if it misses |
|---|---|---|
| F1 | gate FULL m = 3 85-95 us, m = 1 95-115 us (the m = 1 instance has more VGPRs): ratio m3/m1 0.8-1.0 | ratio > 1.15: the m = 3 cost is row work after all; re-read the per-row loads |
| F2 | NOTRAFFIC m = 3 within 5% of FULL m = 3 | NOTRAFFIC >= 15% faster: the A footprint (30 KB per wave, L0 misses) is the limiter and A goes to LDS in the kernel |
| F3 | NOFMA m = 3 within 5% of FULL m = 3 | NOFMA >= 15% faster: VALU-bound, the lever is the dot arithmetic, not depth |
| F4 | UNROLL 4 at m = 3 is 10-25% faster than UNROLL 2 while VGPR <= 128; UNROLL 8 spills or drops to 3 waves and gives back part of it | UNROLL 4 not faster with VGPR <= 128: bytes in flight is not the limiter; stop and re-profile |
| F5 | down shape (640 blocks, A 104 KB per wave): FULL m = 3 85-95 us; NOTRAFFIC 10-25% faster (bigger A footprint) | |

Kernel design rule, frozen: the specialized kernel keeps `q4_dot_blocks`' per-lane block order and fma chain (bit-identical to the m = 1
kernel by construction) and changes only what the arms above prove: depth of weight loads in flight, where A is read from, comptime N / K.

### Run record (job turbo-rows, `.work/turbokernel/rows1/`, bench sha in `arm.txt`, commit 142537c, 11 rounds, medians, us)

| arm | gate 17408 x 5120 | GB/s | ratio | down 5120 x 17408 | GB/s | ratio |
|---|---|---|---|---|---|---|
| FULL m = 1, UNROLL 2 | 62.4 | 803 | 1.00 | 77.8 | 644 | 1.00 |
| FULL m = 2 | 78.8 | 636 | 1.26 | 90.2 | 556 | 1.16 |
| FULL m = 3 | 88.0 | 569 | **1.41** | 100.4 | 499 | **1.29** |
| FULL m = 4 | 104.7 | 479 | 1.68 | 106.7 | 470 | 1.37 |
| NOTRAFFIC m = 3 | 76.9 | 652 | 1.23 | 82.8 | 606 | 1.06 |
| NOFMA m = 3 | 76.9 | 652 | 1.23 | 81.1 | 618 | 1.04 |
| FULL m = 3, UNROLL 4 | 92.6 | 541 | 1.48 | 92.2 | 544 | 1.18 |
| FULL m = 3, UNROLL 8 | 93.5 | 536 | 1.50 | 98.4 | 509 | 1.26 |
| FULL m = 1, UNROLL 4 | 76.9 | 652 | 1.23 | 63.2 | 793 | 0.81 |

Host check m = 3 FULL vs fp64: max rel 2.8e-5 (gate), 3.3e-5 (down). A few arms carry one outlier round (max 142 to 169 us); medians stand.

Verdicts: F1 MISSED (the bench's m = 1 instance runs at 803 GB/s; in the engine the draft-layer m = 1 instance of the same shape runs
456 GB/s at 142 VGPRs, a codegen difference, noted below). F2 and F3 both PARTIAL: on gate, removing the A traffic or removing the
per-row FMAs each recovers 11.1 of the 25.6 us m = 3 penalty, so the two row costs are the same size and serialize; on down either one
alone brings m = 3 to within 6% of m = 1. F4 FALSIFIED for gate (UNROLL 4 and 8 lose, VGPR 136 / 160 drop a wave per SIMD) and
inverted for down (UNROLL 4 wins at m = 1 and m = 3: fewer, longer waves need depth). F5 HELD. P5: the shipped m = 3 window kernel is
1.41x / 1.29x its m = 1 self on the FFN shapes, above the 1.25 defect line.

VALU arithmetic behind F3: per 32-element block a lane issues about 48 dequant instructions plus, per row, 32 fma and 32 bf16 unpack
instructions on 16-wide f32 vectors; at m = 3 that is about 240 VALU instructions per 512 B of weights per wave, a 1.0 TB/s VALU ceiling
on 192 SIMDs at 2.5 GHz, so the row work no longer hides under the memory stream (m = 1: 2.2 TB/s, m = 4: 0.8 TB/s).

## 4. Kernel (step 4) and its frozen predictions

`kernels/turbo_rows.mojo`, `amar_turbo_rows[N, K, MR, UNROLL, AF32]`: the Q4_0 m-row GEMV with the A rows staged ONCE per block
in LDS (bf16, or f32 with AF32 so the per-row unpack disappears), K chunked so the stage fits (K 5120 / 6144 in one chunk, K 17408 in
chunks), weights streamed as `q4_dot_blocks` does, the same lane-to-block mapping, the same fma chain in the same k order and the same
16-lane tree + warp sum, so every output word equals `amar_matmul_skinny_q4rowb`'s (byte test `kernels/test_turbo_rows.mojo` on real
pack rows, m = 1, 2, 3). N and K are comptime (no runtime bounds, exact grid). UNROLL per shape from the bench (2 for K 5120, 4 for K
17408). Wired behind `BARO_TURBO=1` into `gemm_w`'s q4 route for the 27B shapes, read back as `BARO_TURBO:`.

| id | prediction | if it misses |
|---|---|---|
| K1 | gate m = 3: bf16 stage 72-80 us, f32 stage 66-76 us (from 88.0) | > 82 us: LDS latency or occupancy ate the gain, re-profile before any redesign |
| K2 | down m = 3 (UNROLL 4, chunked): 80-88 us (from 100.4) | > 92: chunk re-staging costs more than the L2 round trips it removes |
| K3 | out 5120 x 6144 m = 3: from 44.5 us in-engine to <= 38 | |
| K4 | engine, q4 pack, spec k = 2, 20-prompt same-job median: 62.9 -> 68-74 tok/s with all six shapes wired; ids equal on the stable prompts, byte test PASS | < 66: the window is not GEMV-bound as the trace says; re-read the trace shares |
| K5 | m = 1 through the new kernel: within 3% of q4rowb m = 1 (803 GB/s) | slower: keep q4rowb for m = 1 |

### Run record (job turbo-rows2, `.work/turbokernel/rows2/`, dirty tree over 142537c: kernel + test + bench arms, medians, us)

`kernels/test_turbo_rows`: 33 arms (K 5120 / 6144 / 17408, MR 3 at m 1..3 and MR 1, bf16 and f32 staging, UNROLL 2 and 4), every output
word equals `amar_matmul_skinny_q4rowb`: PASS. Bench byte check on real tensor 0, m 3, two variants: 104448 + 30720 words, 0 mismatches.

| arm (m = 3 unless noted) | gate | GB/s | down | GB/s |
|---|---|---|---|---|
| q4rowb FULL m = 1 (this run) | 62.1 | 807 | 71.8 | 698 |
| q4rowb FULL m = 3 (this run) | 89.0 | 563 | 95.4 | 525 |
| TURBO bf16, UNROLL 2 | **79.2** | 633 | **81.3** | 617 |
| TURBO bf16, UNROLL 4 | 90.9 | 551 | 90.0 | 557 |
| TURBO f32, UNROLL 2 | 258.0 | 194 | 269.4 | 186 |
| TURBO f32, UNROLL 4 | 274.8 | 182 | 277.6 | 181 |
| TURBO bf16 m = 1, UNROLL 4 | 64.1 | 782 | 65.1 | 770 |
| TURBO f32 m = 1, UNROLL 2 | 94.0 | 533 | 97.9 | 512 |

K1 HELD for the bf16 stage (79.2 in 72-80); the f32 stage MISSED by 3.3x, a defect in that variant (LDS instruction width or
bank pattern, checked below), not a verdict on the idea. K2 HELD (81.3 in 80-88) but with UNROLL 2, not 4: UNROLL 4 loses on the
new kernel too (VGPR 139 vs 131, one wave per SIMD less), so `turbo_unroll` is 2 for every K and `turbo_af32` is off. K5: the bf16 m = 1
instance is within 3% on gate (64.1 vs 62.1) and faster on down (65.1 vs 71.8).
Per layer at m = 3 the FFN goes 2 x 89.0 + 95.4 = 273 us to 2 x 79.2 + 81.3 = 240 us (-12%), about 2.1 ms per window of 37.5.

### Megakernel barrier cost on the 27B (the maintainer's question, capture `att-mega` dispatch 3789, 4 waves of blocks >= NH_V)

s_barrier: 37 pcs, 26.39 M cycles = 11.2% of the dispatch (spill 0.1%). The costly ones, each once per layer per wave:

| barrier pc | avg cycles | issues per wave | share | phase (by the adjacent loops) | shape arithmetic |
|---|---|---|---|---|---|
| 0x179a0 | 19635 | 64 | 2.1% | after FFN gate+up | 2176 row groups / 96 blocks = 22.67: a third of the blocks idle one step of 23 (4%) plus stragglers |
| 0x18ba0 | 18152 | 64 | 2.0% | after FFN down | 640 groups / 96 = 6.67: a third idle one step of 7 (14%) |
| 0x1429c | 24087 | 48 (SSM layers) | 2.0% | after the delta scan | 48 heads on 96 blocks, half the grid idle; the kq instance already moved z here (lane mega-barrier), the q4 instance has not |
| 0x1551c | 9284 | 64 | 1.0% | SSM/attn out projection | 640 groups / 96, as the down |
| 0x118ac, 0xc2dc | 18043, 25168 | 24, 16 | 0.7 + 0.7% | attention layers | 28 heads on 96 blocks |
| rest (31 pcs) | | | 2.7% | | |

Reading: the 27B shapes divide badly by 96 (5120 / 8 = 640 and 6144 / 8 = 768 row groups; 768 is exact, 640 and 2176 are not), and
the SSM delta leaves half the grid idle on this path. The recoverable part is about half of the 11.2% (the delta phase by the z move the
kq instance already has, bit-exact; the 640-group phases need either a finer work unit or split work for the remainder step). This is the
m = 1 no-spec path (44.4 tok/s); the lane's winning path (spec) never enters the megakernel, so it is recorded here as the next lever for
no-spec decode, not taken in this lane.

### P21 captures (bench/ww.sh, wavewait pin 927b2b3, iteration 1500 of the matching kernel, spec run on the q4 pack)

| capture | kernel | dispatch | waves | vmcnt wait | VALU | lgkmcnt | barrier | spill |
|---|---|---|---|---|---|---|---|---|
| before, `.work/turbokernel/ww-q4rowb/` (engine-t0-4f4dc20, `-D BARO_TURBO=0`) | `q4rowb[2,3]` FFN gate, hash 79507b81e054c1a1 | iteration 1500 | 88 | 69% | 24% | 1.5% | 0 | 0 |
| after v1, `.work/turbokernel/ww-turbo/` (engine-t1-4f4dc20, `-D BARO_TURBO=1`) | `amar_turbo_rows` gate instance, dispatch 19210 | 90 | 32.2% | 34.5% | 6.8% | **20.4%** | 0 |

Reading of v1: the L2 round trips are gone (vmcnt 69 -> 32%) and the block barriers around the LDS stage took their place (20%): v1
staged each chunk serially (load A, store, barrier) with no weight load in flight. v2 (double-buffered LDS, the next chunk's A loads
issued beside the weight loads, one barrier per chunk) is the response; its receipt follows (job turbo-rows3).

The first A/B chain (jobs turbo-ab-spec, turbo-ab-m1launch, engines t0/t1 at 4f4dc20) was REFUSED by `bench/ab-resident.sh`'s load gate:
1-minute load 55.9 and 17.6 at admission (other sessions' builds), no arm ran, no number recorded. Re-queued behind a load wait.

### v2 (double-buffered LDS, next chunk's A loads beside the weight loads, one barrier per chunk): REJECTED (job turbo-rows3)

Byte test PASS (36 arms), byte check PASS, but at m = 3 it is slower than v1: gate 86.6 us (v1 79.2, same-run q4rowb 88.2), down 86.9
(v1 81.3, q4rowb 101.0). ISA: the MR 3 instance went 131 -> 146 VGPRs and the LDS doubled (30 KB; MR 4 60 KB = one block per WGP). The
pipeline's held registers cost more occupancy than the barrier it removed. m = 1 and m = 2 through v2 were good (62.2, 69.1 vs q4rowb
62.4, 78.8), so the m = 2 window (end of generation) gains either way. Kernel returned to the v1 structure, f32 variant deleted
(P20), one more arm added: UNROLL 1 (fewer live registers, the question is whether a fourth wave per SIMD returns).

### Final kernel receipt (job turbo-rows4, v1 structure, bf16 stage, single buffer, UNROLL 2; medians, us)

| arm | gate | GB/s | vs q4rowb same m | down | GB/s | vs q4rowb same m |
|---|---|---|---|---|---|---|
| q4rowb m = 1 / m = 3 (this run) | 62.0 / 90.6 | 808 / 553 | | 78.1 / 100.7 | 642 / 498 | |
| TURBO m = 3 | **80.8** | 621 | 0.89x | **82.7** | 606 | 0.82x |
| TURBO m = 3, UNROLL 1 | 82.4 | 608 | | 83.4 | 601 | |
| TURBO m = 1 | 62.1 | 807 | 1.00x | 63.7 | 788 | 0.82x |
| TURBO m = 2 | 63.8 | 786 | 0.81x (q4rowb m = 2 78.8, run 1) | 66.4 | 755 | 0.74x (90.2) |
| TURBO m = 4 | 98.3 | 510 | 0.94x (104.7) | 98.1 | 511 | 0.92x (106.7) |

Byte test 36 arms PASS, bench byte check PASS. VGPR: MR 1 94, MR 2 147, MR 3 131, MR 4 139 (q4rowb MR 3: 126); no spills. UNROLL 1 did not
return the fourth wave (131 -> 119 VGPRs is still over 128 at MR 3... it reads 119 at MR 1 only), so the MR 3 instance runs 3 waves per SIMD
either way and wins on the LDS staging alone. Per FFN layer at m = 3: 273 -> 244 us (-10.6%); per 3-row window about 1.9 ms of 37.5.

## 5. Wiring and how to turn it on

`serve/window.mojo`: `comptime TURBO = get_defined_int["BARO_TURBO", 0]() == 1`. In `gemm_w`'s Q4_0 route (the window path's GEMV for
every dense projection, the draft layer and the LM head) a q4 weight with N % 8 == 0 and K % 1024 == 0 and m <= 4 goes to
`amar_turbo_rows[N, K, m, 2]` (comptime N and K from the call site, one instance per shape and width; 40 instances on this model);
anything else keeps `gemm_q4` -> `amar_matmul_skinny_q4rowb`. Default off: the knob is a build flag so a bake carries it in
`baro.run.build.flags` (`-D BARO_TURBO=1`) and the 9B packs are untouched until measured there. Read-back: the engine prints
`BARO_TURBO: True|False` next to `BARO_MEGA:`. The megakernel code objects are byte-equal between the two builds (sha of the three
`amar_mega_token` instances compared, so the m = 1 no-spec path is the same binary in both arms).

P21 after-capture of the final kernel: `.work/turbokernel/ww-turbo2/` (engine-t1-f9fa87a, `bench/ww.sh 'turbo_rows'`, iteration 1500,
gate instance e2f712799355194d, dispatch 19210, 90 waves, 1.99 M cycles vs the before capture's 2.08 M on the same WGP sample):
vmcnt wait 32.1%, VALU 32.9%, barrier 21.8%, lgkmcnt 7.0%, LDS 1.3%, spill 0%. JSON `.work/kfacts/ww/amar_turbo_rows.json` and the
before capture `.work/kfacts/ww/amar_matmul_skinny_q4rowb.json` (here and in the primary checkout). The remaining pool in this
kernel is the barrier share (the serial per-chunk stage); the pipelined form that attacked it lost on registers (v2 above), so the
next attempt must stage with fewer live registers (e.g. the stage issued by one wave while the others compute, or smaller chunks
with the barrier count traded against LDS size).

## 6. Same-job A/B (step 5), tree f9fa87a, q4 pack, 20 mtp-prompts, resident arms, `bench/ab-resident.sh` with `AB_ENGINE_B`

Arms: t0 = `engine-t0-f9fa87a` (`-D BARO_TURBO=0`, sha 5341a557cff845b2), t1 = `engine-t1-f9fa87a` (`-D BARO_TURBO=1`, sha f46fccdc4908cb29),
same tree, same env string per job; read-backs `BARO_TURBO: False` / `True`, `BARO_MEGA`, `BARO_SPEC`, fail word 0, TMAX 16384, per-request
spec counts in the done lines. Jobs `turbo-ab-spec` and `turbo-ab-m1launch` (`.work/turbokernel/ab-spec/`, `ab-m1launch/`).

| path | t0 median tok/s_gen | t1 median | ratio | per prompt | identity | drafted / accepted | load_max t0 / t1 |
|---|---|---|---|---|---|---|---|
| spec k = 2 (`BARO_MEGA=1 BARO_SPEC=1`), the winning path | 61.12 | **68.72** | **1.124** | every prompt faster, 1.032 to 1.266 | 19/20 (p03-story, the run-to-run unstable prompt) | 1074 / 733 vs 1075 / 732 | 3.50 / 4.17 |
| launch m = 1 (`BARO_MEGA=0 BARO_SPEC=0`) | 38.64 | 39.63 | 1.026 | 17 prompts 1.008 to 1.031, three t0 outliers (p12 31.2) | 19/20 (p03-story) | 0 | 8.48 / 7.19 (suspect: host load, launch path) |

K4 HELD (68.72 in 68-74). K5 HELD (m = 1 within 3%, faster). The brief's clean baseline for the same path was 62.88 (job 2 above, the
tree before the merge of main), so the model goes **62.9 -> 68.7 tok/s on its best path**, and the identity miss is the same prompt the
same-arm repeat already showed flipping between two runs of one binary (section 1).

Byte test against the m = 1 kernel: `kernels/test_turbo_rows` PASS (36 arms, every word), in `run-tests.sh`. Row-scaling receipt (P5):
section 4 tables, m = 1 / 2 / 3 / 4 on both FFN shapes for the shipped and the new kernel. ISA lottery: the three megakernel code objects
byte-equal across arms. No minimum-gain bar applied (the brief); the next pool is named in section 2 (barrier share of the stage, the
m = 1 megakernel phase balance) and in the kernel-level numbers (gate m = 3 at 621 GB/s against 808 at m = 1).

## 7. Receipts for the merge gate

- `./run-tests.sh` under gpu-wait on this worktree at commit 2b52654 (kernel tree f9fa87a + docs), pack `.work/engine-pack-q4`, engine
  `.work/engine` built from the same tree: **run-tests rc=0**, last line `PASS run-tests: all tests, pack .work/engine-pack-q4`
  (log `.work/turbokernel/run-tests2/run-tests.log`, `rc.txt` = rc=0). The first run (`run-tests/`) was INCOMPLETE rc=3 with every
  pack-free test PASS: the state-layer test needs `.work/engine`, which lane-prep does not build.
- P21 captures: before `.work/turbokernel/ww-q4rowb/` (q4rowb gate, engine-t0-4f4dc20), after `.work/turbokernel/ww-turbo2/` (amar_turbo_rows
  gate, engine-t1-f9fa87a); JSON `.work/kfacts/ww/amar_matmul_skinny_q4rowb.json` and `.work/kfacts/ww/amar_turbo_rows.json` (here and in the primary checkout).
- Kernel census: 160 kernels, 71 in registry, 0 orphans (`docs/KERNELS.md` regenerated). `tools/kernel-facts.sh` regenerated.
- GPU time: baseline 4 jobs + determinism 1 + profile 1 + kernel trace 1 + ATT 4 + ww 3 + bench 4 + A/B 2 (+ 2 refused at admission,
  + 2 dead on a chain script bug, no GPU minute spent) + run-tests 2. gpu-wait stats --days 1: last 1d: 421 jobs (346 ok / 72 failed / 3 cancelled), gpu busy 449m, median queue wait 2s;RUNS  OK FAIL  WALL~  TOTAL VRAM~  WAIT~ REPO           CMD;   2   1    1    66s   395s  17.9     1s turbokernel    turbo-run-tests;   1   1    0   381s   381s  17.7    78s turbobake      run-tests-job.sh;
- Not merged (the brief). Board card ticked, Brain note `~/Brain/mojo/mojo-baro/2026-11-02-turbokernel-lane.md`, memory `turbokernel-lane-landed`.
- `lane-merge --report docs/lanes/turbokernel-report.md lane-turbokernel`: **READY: lane-turbokernel -> main** (log `.work/turbokernel/lane-merge3.log`;
  the WARN gate-off names the arms that run a knob off, one arm per knob by design). Not merged, as the brief asks.
