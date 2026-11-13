# Report: docs sync after rdna4-epi and F3-F5

Brief: `docs/lanes/2026-11-12-docs-sync-brief.md`. Docs only; no code, kernel, test or GPU.

## Skills loaded
`.claude/skills/mojo-baro/SKILL.md`, `lanes/SKILL.md`, `gpu-kernels/SKILL.md`, `rdna4-port/SKILL.md` read in full; `rdna4-isa/SKILL.md` read to line 30 (header, scope, use) only.

## Commits (repo, pathspec)
- 0c22ff0 docs(readme): README.md (face section rewritten from F3-F5 reports: tabs, presets, Apply, Model; RDNA4 paragraph now names `acc_row`, `gfx11_rows`, the gate script and the 20/20 result; Testing table gets face gates, `bench/rdna4-epi-gate.sh`, settings-check; Configuration table regenerated with the built `.work/settings-check --readme`, 13 rows, then `PASS settings-check: 183 settings ... README table current`), docs/amd-family.md (RDNA4 row: rows ported, real card UNVERIFIED).
- ed6777e docs(rdna4): docs/rdna4/README.md (new: probe receipt, emulation build, gate script, real-card report).
- 57983e5 docs(changelog): CHANGELOG.md unreleased 2026-11-12 line (removed "rows not ported", added the gate result and the face settings).
- BASELINE.md: no RDNA4 or face statement, not touched.

## Skills (Brain repo, 5d17df295, pathspec)
- rdna4-port: new section 11 (acc_row keeps the base, gfx11_rows for reductions, emulation build, gate steps, MEASURED/UNSURE tags) plus changelog line.
- gpu-kernels: rule 22 in rules.md (reduction-order trap, points at rdna4-port section 11), hub count 21 to 22 and route line.
- mojo-baro: rules 6 (`.lane-prep` overrides the built-in list, `lane-merge --report`) and 7 (re-run preflight after any commit, exit code not through a pipe); hub lane-cycle line and rule index updated.
- lanes: not changed; the merge flow is already there (rules #2 pipe/exit code, #6 preflight last, #17).
- `skilldex` then `skilldex --check`: `PASS skilldex: current`.

## Checks
- Every command and path written was checked with `ls`/header reads: `tools/baro gui|pack-path`, `bench/rdna4-epi-gate.sh` (steps build, isa, run, mutants, engine), `bench/report.sh`, `tools/rdna4-probe/rdna4-probe.sh`, `face/{gate,chat-gate,settings_gate,dash_gate}.sh`, `.work/settings-check`.
- `bash tools/ci-checks.sh` last line: `all non-GPU checks passed`.

## Left UNVERIFIED / notes
- Real RDNA4 card numbers: UNVERIFIED (as stated in README and docs/rdna4).
- Apply restart on a real engine and `face/dash_gate.sh` (needs a 9B engine): NOT RUN per F5 report; README says so.
- Face gates: the README Testing row says "small GPU job" because `settings_gate.sh` queues under `gpu-wait --shared --vram 1`; I did not run any face gate.
- `tools/settings-check` is not an executable name: the header says `tools/settings-check --readme`, the real binary is built to `.work/settings-check` from `tools/settings-check.mojo`. Header wording is stale (not edited: code file).
- `tools/settings-check.py` and `docs/settings-template.toml` still exist (F3 follow-up, unchanged).
