# Brief: independent audit of the gemma4 decode attention kernel (Codex, read-only, 2026-11-16)

Boss pane: a lane pane. Report file: `$HOME/Projects/mojo/mojo-baro-lanes/gemma4-attn/docs/lanes/2026-11-16-gemma4-attn-codex-audit.md`.
When done: `~/iTools/bin/herd tell a lane pane "DONE codex-audit -> <report file> | findings=N must=M"`.

READ-ONLY: do not edit, build into tracked files, or commit. Another agent owns the code.

## What to audit
`kernels/gemma4_attn.mojo` (decode attention for gemma4 E2B: 8 query heads, 1 KV head, head dim 256 sliding window 512 or
head dim 512 full, scale 1.0, split partials + merge) and `kernels/test_gemma4_attn.mojo` if present, in this worktree.
Spec: `docs/lanes/2026-11-16-gemma4-attn-brief.md`. Helpers it calls: `kernels/dattn.mojo` (dkv_tab_off, NEG),
`kernels/arch.mojo` (readlane), `kernels/bexp.mojo` (exp). Semantics reference: `~/Projects/imports/llama.cpp/src/models/gemma4.cpp`.

## Questions
1. Correctness: softmax/merge math, window bounds, chunk edge masking, the lane reduction and readlane mapping, empty
   waves and empty splits, NaN/inf paths, index math against the KV page layout.
2. Does the test actually catch wrong kernels (would a broken window, wrong scale or wrong KV slot fail it)?
3. Resource risks on gfx1100 (VGPR, LDS, barriers) you can see from the source.
Form your own findings first. Only then read `docs/lanes/2026-11-16-gemma4-attn-review.md` (an earlier review) and mark
each of its points agree / disagree / new.

## Report format
Per finding: file:line, severity (MUST / SHOULD / NOTE), what breaks, a concrete input that shows it, the fix in one line.
