# Brief: build the rdna4-isa and rdna4-port skills (Sonnet, effort high)

## Goal
Two skills so any later session can port and debug mojo-baro kernels for RDNA4 (gfx1200/gfx1201) without a card.

## Sources (read in full, in this order)
1. `~/Projects/AMD/docs/rdna4-isa.txt` (707-page AMD RDNA4 ISA guide, pdftotext of `rdna4-isa.pdf`). Compute chapters
   in full; skip graphics (export, interpolation, image sampling, ray tracing) like the rdna3 skill did.
2. `~/Projects/AMD/amd_matrix_instruction_calculator/matrix_calculator.py` class `InstCalcGfx12` (line ~6797) and README.
   Read only, never execute it (auto mode denies running external code).
3. Template to copy: `~/Brain/Skills/own/rdna3-isa/` (SKILL.md, `scripts/isa-op.sh`, `scripts/manual-split.sh`) and
   `~/Projects/AMD/docs/rdna3-isa-feb-2023.txt` for gfx11-vs-gfx12 diffs.
4. Facts already established today: `~/Brain/mojo/mojo-baro/2026-11-12-rdna4-wmma-layout.md`,
   `kernels/test_wmma_rdna34.mojo`, `docs/rules.md` last 3 lines, plan `~/.claude/plans/splendid-forging-catmull.md`.

## Deliverables
A. `~/Brain/Skills/own/rdna4-isa/` (via `~/iTools/bin/skill-new.sh` if it exists, else copy rdna3-isa):
   - `SKILL.md` in the rdna3-isa shape: when to use, lookup, machine facts (CU/WGP, LDS per WG, VGPR budget, wave32/64),
     WMMA + SWMMAC table, lane maps for every dense wave32 type (f16, bf16, iu8, iu4 16x16x16, iu4 16x16x32, fp8/bf8,
     C/D f32 and f16) as formulas `lane = ..., element = ...`, each citing its ISA section and calculator line.
   - Hazards: WMMA data hazards (s7.12.1), split wait counters and every other hazard change vs gfx11,
     removed/renamed instructions vs gfx11 (diff the opcode lists of both manuals with the scripts), dot instructions
     (V_DOT4_I32_IU8 / sudot4 equivalents), s_sendmsg_rtn REALTIME, s_setprio, s_memrealtime.
   - `scripts/isa-op.sh` and `scripts/manual-split.sh` parameterised for the RDNA4 text (footer `of 697`). Run
     manual-split to `~/Brain/research/rdna4-isa/`; check each note links back to the skill.
   - `examples/`: an `llvm-mc -mcpu=gfx1201` smoke `.s` that assembles clean (wmma, dot4, s_wait_* forms). Run it.
B. `~/Brain/Skills/own/rdna4-port/SKILL.md`: the mojo-baro porting checklist. Every line tagged MEASURED (commit),
   DOC (ISA section) or UNSURE. Cover: fragment loaders per dtype, accumulator decode, cross-lane shuffles that assume
   gfx11 replication (permlane, xor-16), LDS layout for the new A/B shapes (b64 loads per half), CU-count sized grids,
   Infinity Cache size in timing rules, the stdlib `mma` slicing bug, ROCm/LLVM minimums.
C. Symlink both into `~/Projects/mojo/mojo-baro/.claude/skills/<name>`, run `skilldex`, and add both to the Route table
   in `~/Brain/Skills/own/gpu-kernels/SKILL.md`.

## Rules
- Layout or encoding claims come from the vendor ISA table or the AMD calculator only; web models and the Modular
  stdlib are leads.
- ISA and calculator disagree: write both, mark CONFLICT, do not pick.
- No commits in mojo-baro except the report (pathspec commit). Brain: commit in `~/Brain` with pathspec.
- No co-author or model tag in commits. No em dashes.
- Report (file): `docs/lanes/2026-11-12-rdna4-skill-report.md`: sections and line ranges read, skills loaded, files
  written, llvm-mc result, every CONFLICT and UNSURE, any fact contradicting today's Brain note.
