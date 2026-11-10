# Q4 phase efficiency round: the nibble path is ALU, not stream (frozen 2026-09-06, before any kernel change)

Follows `bench/q4-protocol.md` (q4 trunk landed at 115.0 tok/s_gen no-spec, `003f1f6`, default pack `b237866`).
Goal of the round: no-spec above 150 tok/s_gen on the q4 pack, 20-prompt median (P4).

## Receipt that motivates it (`results/q4-profile/prof5-{q4,q8}.log`, `BARO_PROFILE=5`, last token, device stamps, us)

Per-phase sums over the layers of each kind; bytes from `.work/engine-pack-q4/index.txt` (Q4_0 = 18/32 B per weight, q8 = 34/32).

| phase | q4 us | q4 MB | q4 GB/s | q8 us | q8 GB/s | note |
|---|---|---|---|---|---|---|
| ssm in-GEMM (qkv+gate+alpha/beta, 24 layers) | 1169 | 684 | 585 | 1645 | 785 | fused 4-GEMM loop |
| ssm out-GEMM (24) | 585 | 226 | 386 | 859 | 498 | N=4096: 2.7 rows per wave |
| attn qkv (8) | 347 | 189 | 544 | 466 | 766 | |
| attn o (8) | 164 | 75 | 460 | 247 | 578 | N=4096 |
| ffn gate+up (32) | 2786 | 1812 | 650 | 4312 | 794 | biggest phase |
| ffn down (32) | 1550 | 906 | 584 | 2238 | 765 | N=4096, K=12288 |
| head (248320 x 4096) | 750 | 572 | 763 | 1212 | 892 | |
| **GEMM total** | **7351** | 4464 | 607 | 10979 | 776 | 85% of the token |
| rmsc (norm+cast, all) | 260 | | | 286 | | |
| conv window + conv/norm | 127 | | | 123 | | |
| delta (ssm state) | 397 | | | 406 | | |
| ssm gate prep | 45 | | | 49 | | |
| attn norm/rope/append + att + gmul | 338 | | | 338 | | |
| swiglu | 61 | | | 56 | | |
| **non-GEMM total** | **1267** | | | 1300 | | 15% |
| token total | 8618 | | | 12264 | | 116.0 / 81.5 tok/s |

Reading: the q4 GEMM phases stream at 0.78x the q8 phases' GB/s on the same shapes. Bytes halved, ALU
did not. Hot-loop mix of the q4 `amar_mega_token` ssm in-GEMM loop (`isa-loops`, engine `b237866`,
2625 instructions, 256 nibbles per iteration): 8.4 VALU per nibble = extraction 1.7 (lshr/bfe/and
per nibble to feed `v_cvt_f32_ubyte0`) + cvt 1.0 + `-8` as `v_add_f32` 1.3 + `v_fma_mix` 1.0 +
accumulate 1.0 + A bf16->f32 unpack 2.2 (`v_mov_b16` pairs, redone per fused GEMM) + address/loop.
Standalone q4row was 0.61x q8row's time for 0.53x the bytes (`bench/draft-q4-protocol.md` Q1'), the
same 0.78 ratio: the loss is the kernel's instruction count, not the megakernel's phase granularity
(that loss is the same 0.75-0.85 on both packs, `bench/megakernel-protocol.md`).

Roof: GEMM bytes 4464 MB at 855 GB/s (standalone q8row stream rate) = 5.22 ms; + non-GEMM 1.27 ms = 6.49 ms = 154 tok/s.
At the q8 phases' own rates (776 GB/s) = 5.75 + 1.27 = 7.02 ms = 142 tok/s.

## Stages

| stage | what | check |
|---|---|---|
| A1 | `amar_matmul_skinny_q4rowb` standalone (ffn shape, `bench/bench_coldcache_mrow.mojo` arm): nibbles via u32 masks (`w & 0x0F0F0F0F`, `(w >> 4) & 0x0F0F0F0F`) so each byte converts with one `v_cvt_f32_ubyteN`; `-8` folded into the scale, `w = fma(nib, d, -8d)` (exact: same real number, one rounding, so bit-identical to `(nib-8)*d`); A unpack as `u32 << 16` / `& 0xFFFF0000` (one op per element); accumulate chain unchanged | bitwise equal to `amar_matmul_skinny_q4row` on every output (host compare, not rel-tol); ISA: `v_cvt_f32_ubyte{1,2,3}` present, no `v_add_f32` in the loop; time vs q8row MR=1 same run |
| A2 | same body in the megakernel (`q8_row_dot[Q4]`), explicit fma chain both sides | `test_mega_block` q4 0 mismatches; engine mega == launch 20/20; 64/64 vs model-ref |
| A3 | 20-prompt A/B old binary (`.work/engine-prev`) vs new, one stint, clock-probe | median, spread, identity 20/20 |
| A4 (only if A3 lands) | non-GEMM: delta 397 + attention 304 + rmsc 260 = 961 us; N=4096 phases at 386-584 GB/s (row granularity: 4096 rows / 1536 wave-rows) | separate freeze |

## Frozen predictions

A1: q4rowb time 0.53-0.57x q8row (from 0.61x); GB/s-equivalent 800+. A3: **115.0 -> 128-138** no-spec
(GEMM phases at 0.78 -> 0.90-1.0 of the q8 rates); land >= 125 (+8.7%); close < 120. The 150 target
needs A4 as well: A1-A3 alone cannot reach it (142 at q8-phase parity).
Stop rules: A1 any bitwise difference -> the form is wrong, fix the form, never widen the check; A1
time > 0.60x -> the ALU hypothesis is wrong, stop and re-profile (P6) before touching the megakernel.

## Result (2026-09-06, `0afe310` A1, `1d55b75` A2; receipts `results/q4-alu/`)

| stage | receipt |
|---|---|
| A1 | `amar_matmul_skinny_q4rowb` bitwise equal to q4row on 12288/12288 rows; 51.16 vs 56.50 us (0.905x), **0.58x q8row** (prediction 0.53-0.57, stop rule 0.60). ISA: `v_mov_b16` 0 (was 452 per loop), but only 6 `v_cvt_f32_ubyte{1,2,3}`: the compiler still extracts most nibbles by shift + `ubyte0` |
| A2 | megakernel + launch dispatch on the b-form: `test_mega_block` q4 0 mismatches; engine mega == launch spec 0/1; 64/64 vs model-ref; 54 kernels / 0 orphans; VGPR 256, spills 77 (unchanged) |
| A3 no-spec | same stint, clock 2977 MHz med, 290 W / -100 mV: **prev 113.36 (spread 3.5%) -> new 118.29 (3.1%), 1.043x, identity 20/20** |
| A3 k=2 | **prev 127.62 -> new 143.06, 1.121x**, identity 20/20 (the launch path's m>1 window gained 95.5 -> 108 launch tok/s). The q4 round's k=2 bar (133) is now met |
| per-phase, same stint, 2 runs each (`prof-cmp-*.log`) | ssm in-GEMM 1165 -> 1068 (-8%), ffn gate+up 2789 -> 2632 (-6%), ffn down 1551 -> 1606 (+3.5%), head 757 -> 690 (-9%), token 8626 -> 8331 us |
| UNROLL 2 -> 4 in the megakernel q4 dot | **dead end**: spills 77 -> 583, 61.4 tok/s (0.515x), reverted. The 256-VGPR cap at 512 threads is the wall; bytes in flight per wave cannot grow by unrolling |

**Verdict: below the frozen land (125), inside the close band (< 120): the A1-A3 stage closes at 118.3 (+4.3%).**
The kernel change stays (faster, identity clean, k=2 +12%). Prediction check: 128-138 predicted, 118.3 measured -
the hypothesis "the nibble path's instruction count is the loss" was only a third right: halving the VALU count per
nibble bought 6-9% on the big phases, not the 20-25% the q8 rates implied. What remains is not ALU:

q4 GEMM phases after A2 (same-stint profile): ssm in 640 GB/s (q8 785), ffn gate+up 689 (794), ffn down 566 (765),
ssm out ~390 (498), attn o ~460 (578), head 829 (892). The N=4096-row phases (ssm out, attn o, ffn down = 2.35 ms of the
8.33) sit at 0.7-0.75 of the wide phases' rate: 4096 rows over 1536 wave-slots = 2.67 rows per wave, a third of the
waves run 3 rows while the rest idle after 2. Bytes in flight per wave at UNROLL 2 (2 x 16 B x 32 lanes = 1 KB, 16 KB per
CU) is also half the q8 path's (32 KB per CU), and unrolling to fix it spills (above).

Next freeze candidates, by size of the pool: (1) N=4096 phases: split K in 2 (or 3) so 8192-12288 half-rows spread evenly
over the 1536 wave-slots, partials reduced in a fixed order on BOTH paths (the launch kernels already carry a
`SPLITK` partial layout + `amar_skinny_reduce`); pool 2.35 ms at 0.72 -> 1.0 of the wide rate = -0.55 ms = **+7%**.
(2) attention 306 us on 16 blocks (80 idle) and rmsc 260 us (64 x 4 us, barrier-latency bound): -0.25 ms = +3%.
(3) GEMM at 855 GB/s needs more bytes in flight without VGPRs: b96 loads or LDS-staged weight prefetch; unquantified.
All three together = ~147 tok/s; 150 is at the edge of everything landing.
