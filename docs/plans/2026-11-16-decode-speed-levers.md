# Plan: decode speed per token, levers 1-6 (2026-11-16, boss-run, minimum LOC)

Consults: `.work/consult/astra.md` (codex gpt-6-astra high), `.work/consult/agy.md` (agy gemini-3.8-flash-high).
Baseline: Qwen3.8-27B TurboFCFusion q4 pack, plain mega 44.40 tok/s, MTP k=2 turbo 68.72 (docs/BASELINE.md:403-452).

## Where the consults agree
- First: FR-Spec 62080-id draft head present in the 27B pack (0 engine LOC, `tools/fr-draft.mojo`): +3-8% if missing.
- Ternary is a trap (measured 197-282 GB/s unpack on gfx1100, bench/ternary-protocol.md; PTQ quality). q3: quality
  trade, ceiling ~1.13-1.29x, L-XL. Rejected.
- LM head off VRAM over PCIe is a trap (25 ms per 715 MB call). Target head stays full vocab.
- Megakernel work helps plain m=1 only: the MTP path runs launched windows (0% on spec).
- Obvious turbo_rows variants already lost (UNROLL 4, f32 LDS, extra waves, prefetch, WMMA m=3, int8 sudot4).

## Where they differ, and how it is settled
- k sweep: agy "skip, dead end (9B data)"; astra "0-LOC sweep, 27B optimum never measured". Settled by running it.
- DFlash2: agy "reject, llama.cpp CUDA DFlash2 only 1.34x (docs/models/orcasaq-2-27b.md:65), +1.06 GB SSM slots";
  astra "only after a cheap falsifier: m=9/D=4 verify probe, break-even tau > 68.72 x cycle". Settled by the probe;
  no drafter code before it passes.
- Mega z-overlap: agy +10-15% plain, astra 1-3%. Settled by one A/B.

## Steps (each ends on a gate; predictions frozen in the commit before the timed run)
P0 (0 LOC) FR read-back on the 27B pack (engine prints FR active + ids) and k = 1, 2, 3 sweep, same pack and
   binary, BARO_SLOTS sized, 20 mtp-prompts, BARO_TMAX >= 16384, ms per emitted token + accepted prefix per position.
P1 (S-M) `kernels/mega.mojo::ssm_phases`: schedule z during delta for Q4, MR=1 (reuse the KQ placement
   `GZ_IN = 0 if KQ else G_Z`). Gate: ids equal 20/20 vs current, same-job A/B plain decode, wavewait capture.
P2 (M) `kernels/turbo_rows.mojo::amar_turbo_rows` m=3: overlap the next chunk's LDS fill with the dot, VGPR <= 128
   (ISA receipt), FMA order unchanged. Gate: kernel parity bit-exact vs current, engine 20-prompt spec A/B.
P3 (S-M, falsifier only) DFlash2 verify probe: `amar_fewrow_wmma` D=4 at 27B shapes, m=9, real weights, plus a
   target-only 9-row window time with fixed tokens. Go to a drafter lane only if measured cycle gives
   break-even tau <= ~4 at the refreshed baseline; else write the no-go into the model skill.

## Out
q3 / ternary packs, LM head offload, adaptive k (only if P0 shows task-dependent crossover).

## Verification
Every step: `bench/ab-resident.sh` same-job A/B, arm read-back (P1 rule), ids or teacher-forced agreement,
numbers + wavewait/ISA receipts in the commit and docs/BASELINE.md.
