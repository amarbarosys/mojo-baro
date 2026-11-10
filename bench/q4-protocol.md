# q4 weights protocol — frozen before first run

> Same rules as `bench/q8-protocol.md` and `bench/PROTOCOL-RULES.md` P1:
> every arm-defining parameter read back from the running system and
> recorded before a timed run. No receipt, no arm.

Frozen 2026-09-04 on the tree that carries the MTP loop and kernel registry
(`7805393` + README/prompt commits). Nothing below this line was measured
before it was written.

## Why q4 and not the KV cache

Byte model per no-spec token today: 10.7 GB of q8 weights, f32 KV cache
< 20 MB at 128 positions. KV quantization moves nothing at this context
length; weights are the only byte lever left. `bench/q8-protocol.md` line
73 already names int4 as the next one.

## Quant on the record

`Q4_0` (ggml): blocks of 32, 4-bit signed nibbles around a per-block fp16
scale, 18 bytes / 32 weights = 4.5 bits/weight vs Q8_0's 8.5. Chosen over
Q4_K_M because llama.cpp's Q4_0 is a single block format our pack can be
proven byte-equal to (Q0 stage below), which is what made the q8 claim
honest. Q4_K_M is a follow-up round, not this one.

## Stages (each gates the next; stop on first failure)

- **Q0 pack parity.** `tools/engine-pack.py --q4` emits Q4_0 blocks; a
  `tools/q4-check.py` proves them byte-equal to `llama-quantize Q4_0` of the
  same bf16 GGUF. Pass = 0 differing bytes across all 2D weights.
- **Q1 kernel.** `amar_matmul_skinny_q4row[UNROLL, MR]`, wave-per-row like
  q8row, nibble unpack in-register. Cold-cache receipt on the ffn shape
  (`bench/bench_coldcache_q8row.mojo` pattern).
- **Q2 engine.** q4 pack default only if Q3 lands; otherwise `BARO_PACK`
  opt-in. Reference = `tools/model-ref.py` run on the same Q4_0 tensors
  (numpy, fp32 accumulate), plus llama.cpp Q4_0 greedy on the same GGUF.
- **Q3 race.** 5 repeats, discard first, median of 4, both arms, bge-m3 and
  every llama-server stopped. Bar = llama.cpp Q4_0 no-spec and with MTP,
  measured first with `tools/llama-ref-run.sh` on the Q4_0 GGUF.

## Frozen predictions

| stage | prediction | land rule |
|---|---|---|
| Q0 | byte-equal, first try or after one scale-rounding fix | 0 diff bytes |
| Q1 ffn-shape stream | 100 MB-equiv in **34-42 us** (q8row 62.6 us; 0.53x bytes, ALU tax 0-20%) | <= 45 us AND parity vs numpy q4 dequant |
| Q2 identity | 64/64 vs llama.cpp Q4_0 on the 5-token prompt AND >= 18/20 prompts identical on `bench/mtp-prompts/` | recorded; < 16/20 = pack or kernel bug, not "quant noise" |
| Q3 no-spec tok/s_gen | **105-125** (roof at 0.53x bytes ~ 176; 60-71% of roof, q8 reached 74%) | >= 1.4x over 68.8 |
| Q3 MTP tok/s_gen | **170-215** (draft/verify acceptance may drop 5-15 pts on q4) | >= 1.3x over 128.0 |
| vs llama.cpp Q4_0 | no-spec 0.85-1.0x; MTP >= 1.1x | recorded |
| falsifier | Q1 > 50 us or Q3 no-spec < 1.25x: stop, diagnose, re-preregister | |

Draft head stays q8 in this round (0.2 GB/token, not worth a second quant
path); recorded as an asymmetry, not corrected.

## Result (2026-09-06, engine `003f1f6`, receipts `.work/q4-round.log`, `.work/ab-q4*/`, `.work/llama-q4pure*/`)

**No-spec LANDS. k=2 misses its bar by 2.8%.**

| stage | receipt |
|---|---|
| Q0 | PASS (`3744c8f`): `--q4` pack byte-equal to `llama-quantize --pure Q4_0` on blk 0/7/15/31 + head; 6.18 GiB |
| Q1' | q4row 0.61x q8row time cold-cache (UNROLL 2/4/8: 58/56/53 us vs 88) |
| Q2 | PASS (`003f1f6`): `test_mega_block` m=1 q4 0 mismatches; engine q4 pack mega == launch, spec 0 and 1; launch vs `tools/model-ref.py` Q4_0 64/64; **20/20 mega == launch on `bench/mtp-prompts/`** (launch 95.23, mega 114.76 tok/s_gen, 1.205x) |
| Q3 no-spec | 20-prompt same-stint A/B, clock-probe 2877 MHz med, 290 W / -100 mV: **q8 80.83 (spread 1.2%) -> q4 115.02 (2.6%), 1.423x, faster 20/20** |
| Q3 k=2 | **q8 102.29 -> q4 129.25, 1.264x, faster 20/20** (per-prompt spread 42%/36% = acceptance varies by prompt, same as every k=2 receipt); acceptance q8 69.9% -> q4 66.2% |
| llama.cpp Q4_0-pure | ref prompt no-spec 110.0 (3 runs 109.4-110.0); mtp-prompts no-spec median **110.0** (spread 1.0%), MTP k=2 median **169.5** (spread 39%, identity 7/20, acc median 75.5%) |

Land rule: no-spec >= 109 AND 64/64 AND 20/20 -> **met (115.02)**. k=2 >= 133 -> **not met (129.25)**.
Prediction check: no-spec 115-140 -> 115.0 (bottom of the band); k=2 140-175 -> 129.3 (below the band: the
window is not the trunk stream, cf. `bench/mrow-gemm-protocol.md` M0, and the q4 draft head already showed
acceptance costs a point). Versus llama.cpp on the same 20 prompts: no-spec **1.05x ahead** (115.0 vs 110.0),
k=2 **0.76x behind** (129.3 vs 169.5) - the MTP gap is unchanged in kind from `bench/mtp-protocol.md` Result 2.

Deviation from the frozen Q2 form, logged: the addendum named the q4row arithmetic form
`acc += wlo*a_lo + whi*a_hi` as the thing to copy. Copying it verbatim did NOT give identity: the
per-element form (int nibble-8, cvt, one fma_mix product, fma accumulate) was already identical in
both ISAs (read with `isa-loops` after decoding the VOP3 words the ROCm objdump rejects), and the
divergence was the compiler reassociating the two-product accumulate differently inside the fused
megakernel loop (1-2 ulp per sub-block, `tools/dump-diff.py`). Pinning the order on the megakernel
side alone failed twice (either order). Fix = explicit `fma(wlo, a_lo, fma(whi, a_hi, acc))` in BOTH
kernels (`kernels/matmul_skinny.mojo`, `kernels/mega.mojo`); the launch kernel therefore changed and
was re-verified against model-ref. ISA after: q4 `amar_mega_token` vgpr 256, spill 72 (q8 77).

Not done here (the maintainer's call): making the q4 pack the engine default (`BARO_PACK`), README tables.

## Addendum (frozen 2026-09-06, before any q4 trunk run): baselines moved, stages re-cut

Baselines are now the megakernel engine (`BARO_MEGA=1`, `ed8b972`..`87381eb`):
**no-spec 80.9 tok/s_gen** (20-prompt median 81-82; same-stint A/B 80.9),
**k=2 102.4**. Bytes per no-spec token on the q8 pack: 8.69 GB (index sum x
1.0625). Q4_0 trunk: 0.53x -> 4.6 GB. The q4row kernel exists
(`amar_matmul_skinny_q4row[UNROLL, MR]`, landed for the draft head, +1.6% on
the window); its cold-cache stream rate has no receipt of its own here, so
stage Q1' measures it before anything is wired.

| stage | what | check |
|---|---|---|
| Q0 | `tools/engine-pack.py MODEL.gguf OUTDIR --q4` (trunk, same tensor selection as `--q8`) | `tools/q4-check.py` byte-equal to `llama-quantize Q4_0` on every 2D weight it is asked about |
| Q1' | q4row cold-cache receipt, ffn shape, UNROLL 2 vs 4 vs q8row | us per 100 MB-equivalent stream; bandwidth-equivalent GB/s |
| Q2 | engine: pack kind from the index (first 2D weight dtype), every GEMM site dispatches q8/q4; megakernel gets a comptime Q4 flag with `q4_row_dot` in the q4row kernel's exact arithmetic form (`(nib & 0xF) - 8) * d`, `(nib >> 4) - 8) * d`, `acc += wlo*a_lo + whi*a_hi`), registry instantiates both | `test_mega_block` q4 case bit-identical to the launch q4 path; engine `BARO_MEGA=0` on the q4 pack 64/64 vs `tools/model-ref.py` on the Q4_0 tensors; mega == launch on 20 prompts |
| Q3 | 20-prompt A/B in one stint: q8 pack vs q4 pack, no-spec and k=2; llama.cpp Q4_0 bar (`tools/llama-ref-run.sh`, `tools/llama-mtp-prompts.sh`) | below |

Prediction (frozen): no-spec **80.9 -> 115-140** (roof at equal phase
efficiency 153; nibble unpack is ALU the q8 path does not pay); k=2
**102.4 -> 140-175** (acceptance may drop a few points on q4 as it did for
the draft head, 69.9 -> 69.2). Land: no-spec >= 109 (1.35x) AND 64/64 vs
model-ref AND 20/20 mega==launch; k=2 >= 133 (1.3x). Close: no-spec < 100.
Stop rules: Q0 any differing byte -> fix the pack, never the checker; Q1'
q4row slower than 0.75x q8row's time on the same shape -> kernel first, no
wiring; Q2 any mismatch that is not a phase-ordering slip -> report.
