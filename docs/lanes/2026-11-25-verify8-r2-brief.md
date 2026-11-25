# Lane verify8-r2: 27B q4 verify step at full bandwidth

Worktree `~/Projects/mojo/mojo-baro-lanes/verify8-r2`, branch `lane-verify8-r2` (from main cf1af3e). Model: nerkyor Qwen3.8-27B q4 pack (turbo27b-coder).

## Goal

Make speculative decode on the 27B q4 pack faster. Today: MTP k=2 + dot2 = 86.0 tok/s_gen (default), DFlash2 = 73.6 (opt-in). Target for this lane: DFlash2 (or the best spec path) at 120 to 140 tok/s_gen. Not a kill line: under target means find the limiter and optimise.

## Budget (why this target)

- Pack ~15 GB, ~850 GB/s practical -> one full weight read ~18 ms -> ~55 verify steps/s at most.
- DFlash2 tau 3.2 to 3.8 tokens/step -> ceiling ~200 tok/s with today's draft.
- The 8-row q4 GEMM runs at 424/399 GB/s (bf16 WMMA); int8 WMMA 538/447 (not wired); turbo m=1 805/778 (docs/BASELINE.md:422). The verify step is at half bandwidth: that is the lever.

## Steps

1. Read first: docs/BASELINE.md:410-470, docs/lanes/2026-11-16-verify8-report.md, 2026-11-16-dflash2-report.md, 2026-11-16-kdefault-report.md, kernels/turbo_rows.mojo, bench/PROTOCOL-RULES.md. Skills: gpu-kernels, bench-gates, wavewait, mojo-baro.
2. Trace the DFlash2 and MTP verify step on the real pack (rocprofv3 kernel trace + wavewait): per-kernel share, GB/s per GEMM. Write the table before any edit.
3. 8-row (m=8, and m=3 for MTP) q4 verify GEMM toward 750+ GB/s: turbo_rows-style LDS-staged A rows at m=8, or wire the int8 WMMA path. Byte test against the m=1 dot helpers where identity is the gate (CLAUDE.md: bit-exact = byte test); else teacher-forced agreement with a same-arm control and a bar frozen before the run.
4. Then launch count of the verify window (~1,100 kernels/token, GPU 82.6% busy).
5. Closed, do not reopen: spec through the window megakernel (59 < 86), int8 WMMA for speed at m=1 (memory int8-wmma-rate-gfx1100: applies to compute rate; at m=8 bandwidth-bound it is open), JSPLIT, balanced decode.

## Rules

- Every GPU job via `gpu-wait run --timeout S`, cap 2 HIP processes. TMAX >= 16384, max_tokens >= 16000 floor (bench/min16k.sh); short runs, 2K prompts.
- Predictions frozen by commit before each timed run. Decode numbers = 20-prompt median, same-job A/B, arm identity read back. Kernel change names a wavewait capture (bench/ww.sh).
- Kernel files carry zero comments. Commit as work lands, pathspec only, no self-tag lines.
- Brain note ~/Brain/mojo/mojo-baro/ dated, whiteboard card (lane-prep opened it) ticked.

## Deliverable

`docs/lanes/2026-11-25-verify8-r2-report.md`: trace table, each change with before/after tok/s and GB/s, identity receipt, verdict per step (LANDED / UNVERIFIED / closed with why). Receipts copied under exchange/receipts/verify8-r2/. Report back: `herd tell a lane pane "DONE verify8-r2 -> docs/lanes/2026-11-25-verify8-r2-report.md"` (or BLOCK with the path of the blocker note).

## Addendum (the maintainer, 2026-11-25): skills, SINDEX, tools

- Keep the skills this lane uses current: a wrong or missing step in a skill (mojo-baro, gpu-kernels, bench-gates, mojo-dev, wavewait, corewait, heaptide, ...) is fixed in its SKILL.md under ~/Brain/Skills/ the same turn, then `skilldex`.
- Every lesson or mistake -> `~/Brain/SINDEX/<domain>.md` lesson entry (Write/Edit, all fields) + docs/rules.md line.
- iTools may be fixed or extended at will when the lane needs them: edit under ~/iTools/<cat>/<tool>/, header + tool.toml same commit, `./index-gen --write`; never edit a script a running process executes.
- List the skill, SINDEX and tool changes in the report.
