# Persistent decode kernel: todo (2026-10-26)

Thesis: one dispatch per token (G=96 blocks x 256 threads, grid barriers between phases) removes the ~0.6 ms/token of
dispatch gaps that are 29% of decode today (profile: `tools/kernel_table.py`, 49b4ded), and the token loop no longer
needs a host sync per token.

Phases per layer (each ends in a bounded grid barrier):
1. rmsnorm, computed redundantly by every block (bit-identical, no barrier) + q/k/v GEMV, wave-strided rows
2. rope q, k + K/V into the paged cache (40 head tasks)
3. attention: one q head per block (`attn_head_body`, 32 of 96 blocks)
4. o GEMV, added into x
5. rmsnorm (redundant) + gate and up for the same row in one wave, act = silu(gate) * up
6. down GEMV, added into x
Then: final norm + lm head; each block keeps its best logit, block 0 reduces them into the device token array.

Safety (skill `persistent-kernel-gfx11`): bounded spin barrier + fail word (mojo-baro `grid_barrier`), G = 96 (the
residency floor at <= 256 VGPRs), `rocdl.flat_work_group_size` 256, every run through gpu-wait, fail word read by the host.

## Todo

1. [x] `kernels/llama/mega.mojo`: weight table (addresses + quant types per layer), phases, kernel.
2. [x] ISA receipt: VGPRs, spills, scratch; residency >= 96 blocks.
3. [x] (replaced by the token gate) One-token check: x after all layers vs `forward` (max abs diff), argmax equal.
4. [x] `decode.mojo` mega path: one launch per token, ids stay on the device, fail word checked at the end.
5. [x] Gate: ref_story 64/64 (prefill and no-prefill); decode tok/s vs ~520; profile.
6. [x] docs, Brain, commit.

## Result (2026-10-26)

- ISA: 132 VGPRs, 0 spills, no scratch, 680 B LDS.
- ref_story 64/64 (prefill and no-prefill); ref_paris same near-tie as the launch path.
- Blocks: G=96 415 tok/s, 144 495, 192 529-534; 240 and 288 NOT-RESIDENT at 64 tokens (bounded barrier, clean fail).
  So G=192 is likely the ceiling (2 blocks per CU): no slack; opt-in only (`decode ... mega`).
- Launch path: 516-524 tok/s. Net gain of the persistent kernel at G=192: ~2%. Gaps are gone, but the phases run
  fewer waves than the launch-path GEMVs, qkv lost its S=4 split, and attention uses 32 of 192 blocks.
- Next inside the kernel: phase timestamps (s.sendmsg.rtn 131), attention split across all blocks (skill: 3.3-3.7x on
  that phase), a fallback to the launch path for the rest of the run when the fail word is set.

## Round 2: split attention inside the kernel (2026-10-26)

Attention as (q head, 32-token span) tasks over all 192 blocks + one merge phase (99 -> 115 barriers per token).
ref_story 64/64, 132 VGPRs, no spills. 400 tokens, interleaved: mega 519.6 / 521.5 tok/s, launch path 555.0 / 559.6.
The launch path (now with split attention and gate+up in one GEMV) stays faster by ~7%; the persistent kernel stays
opt-in. Without phase timestamps the cause is open (candidates: 2 waves per SIMD in the GEMV phases, barrier count,
q/k/v without split-K).
