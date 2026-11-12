# Brief: RDNA4 accumulator epilogues + gfx12-layout emulation on gfx1100 (fable, effort high)

## Goal
Every WMMA kernel writes correct rows on gfx12 (RDNA4), and that gfx12 code path runs and is checked on our gfx1100.
No gfx12 card exists here. The gfx11 path must not move a byte.

## Load first (list them in the report)
`.claude/skills/rdna4-port/SKILL.md` (sections 2, 3, 8, 10 are this job), `.claude/skills/rdna4-isa/SKILL.md`
(Lane maps), `.claude/skills/gpu-kernels/SKILL.md` rules #7 to #9, `.claude/skills/mojo-baro/SKILL.md`.
Template: `kernels/test_wmma_rdna34.mojo` (`store_acc`, mutant pattern). Rules: `docs/rules.md` last 6 lines.

## Facts you build on (do not re-derive)
- A/B fragments need no change: the stdlib `mma` slices 16-wide fragments on rdna4 and the product is correct (k
  permutation shared by A and B). Do not touch loaders.
- gfx11 C/D row = `2*reg + half`; gfx12 row = `reg + 8*half` (half = lane // 16, column = lane % 16 both).
- Sites (rdna4-port s2, verify each by reading): `matmul_prefill.mojo:93,169`, `matmul_prefill_lds.mojo:158`,
  `matmul_prefill_kq.mojo:202`, `matmul_mmq.mojo:275` + `mmq_scale_slot` (:24) + the `half * 8` scale row (:199),
  `moe_wmma.mojo:156`, `matmul_wmma_lds.mojo:161`, `matmul_wmma_pipe.mojo:879`, `attn.mojo:747,754,777` (mask
  `t0 + 2*i + half` and the P-to-B repack `bf[2*i]`, s3).

## Step 1: one helper
`kernels/arch.mojo`: `acc_row(i, half) -> Int` = `i + 8*half` if `GFX12_ROWS` else `2*i + half`, where
`comptime GFX12_ROWS = _is_amd_rdna4() or get_defined_bool["BARO_EMU_GFX12_ROWS", False]()` (check the exact
define helper name used in `serve/registry.mojo`). Every site above calls it. Nothing else changes in a kernel.

## Step 2: emulation on gfx1100 (the check that makes step 1 real)
When `BARO_EMU_GFX12_ROWS` is set on a gfx11 build, `arch.mma` must hand back the accumulator in gfx12 row order and
take C in gfx12 row order: convert C gfx12->gfx11 before the gfx11 WMMA and D gfx11->gfx12 after it, with lane
exchanges between the halves (`shuffle_xor 16` / `readlane`, whatever lowers on gfx1100). Under the define, every kernel
then runs its gfx12 epilogue on real hardware. Write the 2x2 mapping in a comment-free helper; derive it from the two
row formulas, test it alone first (extend `test_wmma_rdna34`: emu on, 256/256 within the 2^-19 floor).
For attention (`attn.mojo` P*V), work out from the rdna4-isa Lane maps whether the gfx12 repack is correct under
emulation; if emulation cannot exercise it faithfully, say so and mark that site UNVERIFIED.

## Gates (all on gfx1100, `gpu-wait run --shared --vram 1`, knobs as `env K=V` inside the job)
1. Default build: every parity test in `run-tests.sh` that touches these kernels passes unchanged, and
   `tools/isa-receipt.py` + `isa-diff` of each touched kernel vs the pre-change build: identical ISA. Identical means
   no perf A/B and no wavewait capture are needed (rdna4-port s10); state that in the report.
2. Emulation build (`-D BARO_EMU_GFX12_ROWS=true`): the same parity tests pass within each test's existing bar.
3. Mutant: emulation on, but one site left at `2*i + half`: its test must FAIL. One mutant per kernel family.
4. gfx1201 build of every touched test and of `serve/engine.mojo` still lowers; `tools/rdna4-probe/rdna4-probe.sh`
   table unchanged.
5. One real-engine check: dense prefill, 20 prompts, teacher-forced agreement (`BARO_FORCE`) default build vs
   emulation build, same pack. Bar set before the run: write it in the report first. `BARO_TMAX>=16384`,
   max_tokens>=16000 (CLAUDE.md floor); read the whiteboard GPU holds first; state GPU-minutes before queueing.

## Rules
- Kernel files: ZERO comments, ZERO docstrings (CLAUDE.md). Rationale goes in commit bodies.
- Commit each step with a pathspec on this shared checkout (another lane works in `../mojo-baro-lanes/F0`). No co-author
  or model tag, no em dashes.
- Three failed repairs on a step: write `.work/rdna4-epi-failure.md`, report, stop. No subagents, no forks.
- Hard failure = an error raised by the code under test after its own dependencies are present.
- Report: `docs/lanes/2026-11-12-rdna4-epilogue-report.md`: skills loaded, commits, each gate with receipt path and
  numbers, mutants and their failure counts, every site marked VERIFIED-EMU or UNVERIFIED with the reason.
