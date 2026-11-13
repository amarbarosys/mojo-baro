# Brief: docs sync after rdna4-epi and F3-F5 (sonnet, effort medium, M)

Repo: $HOME/Projects/mojo/mojo-baro, branch main, primary checkout. It is shared: commit with a pathspec only, never a bare
`git commit`. Other files are dirty from other work (grammar/, serve/expert_tier.mojo, exchange/): do not touch or stage them.
Docs only. No code, kernel or test edits. No GPU.

## Load first (list them in your report)
`.claude/skills/mojo-baro/SKILL.md`, `.claude/skills/rdna4-port/SKILL.md`, `.claude/skills/rdna4-isa/SKILL.md`,
`.claude/skills/gpu-kernels/SKILL.md`, `.claude/skills/lanes/SKILL.md`.

## Source of truth (read, do not re-derive)
- `docs/lanes/2026-11-12-rdna4-epilogue-report.md` (gates, sites, design calls)
- `~/Brain/mojo/mojo-baro/2026-11-12-rdna4-epi-handoff.md`
- `git log --oneline 61c0001..HEAD` (rdna4-epi merge adfc0ae, c545a27 gfx11_rows, 408f3aa freivalds/lops, e6ecd41 .lane-prep, F3-F5 face settings)
- `docs/rules.md` last 15 lines; `kernels/arch.mojo` (acc_row, gfx11_rows, mma, EMU_GFX12_ROWS)

## Do
1. README.md: complete and current. Each section must match the tree:
   - RDNA4 status: rows ported; the emulation build `-D BARO_EMU_GFX12_ROWS=true` matches the default build bit for bit, including 20/20 engine prompts; a real card is UNVERIFIED.
   - The face/GUI: settings tabs, presets, Apply, Model tab (read the F3, F4, F5 reports in docs/lanes/).
   - The Configuration table: regenerate it with the repo tool (`tools/settings-check --readme`; read its header first).
   - Build and test commands.
   Check that every command and path you write exists (`ls`, `--help`). Remove claims that are no longer true.
2. Skills (canonical files are ~/Brain/Skills/own/<name>/SKILL.md, reached through the symlinks):
   - rdna4-port: `acc_row(base, i, half)` keeps the base inside the helper. An inline form changed the add order and moved gfx11 ISA, twice (test_prefill, test_gemm). Any reduction over accumulator elements goes through `gfx11_rows` (the attn row sum drifted 1 ulp and flipped engine tokens). Method: the emulation build and `bench/rdna4-epi-gate.sh` (build, isa, run, mutants, engine). Tag lines MEASURED/DOC/UNSURE as the skill already does.
   - gpu-kernels: one short rule line for the reduction-order trap, pointing at rdna4-port.
   - mojo-baro: `.lane-prep` overrides lane-prep's built-in list (fixture names go there); `lane-merge --report PATH`; re-run preflight after any commit before lane-merge (the stamp covers the tree).
   - lanes: only if that merge flow is not already there.
   Then run `skilldex` and `skilldex --check`.
3. docs/:
   - BASELINE.md, only if it states RDNA4 or an affected status.
   - docs/rdna4/README.md (new, short): names the probe receipt, the emulation build and the gate script.
   - CHANGELOG unreleased section: add the RDNA4 rows and face settings lines if absent.

## Rules
No em dashes. Docs in normal grammar. One commit per concern, `git commit -m ... -- <paths>`, conventional subjects, no attribution trailer.
At the end run `bash tools/ci-checks.sh`: it must print `all non-GPU checks passed`; fix any doc drift it reports.
No invented numbers: every number cites a report or receipt.

## Deliverable
`docs/lanes/2026-11-12-docs-sync-report.md`: skills loaded, files changed per commit, the ci-checks result line, anything left UNVERIFIED.
