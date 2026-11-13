# Brief: finish F5 (presets UI, Apply, Model tab) (codex, effort medium, M)

Worktree: `~/Projects/mojo/mojo-baro-lanes/F5`, branch `lane-F5`, rebased on main (F3 + F4 merged). `cd` there first.
Read: `docs/lanes/2026-11-12-face-settings-plan.md` (F5 section, owner decisions), `docs/lanes/2026-11-12-face-F5-report.md`
("NOT RUN / remaining" 1 to 5 and 7 are yours; 6 is the boss's), skills (read each SKILL.md, list them in the report): `.claude/skills/mojo-baro/SKILL.md`, `.claude/skills/mojo-dev/SKILL.md`, `.claude/skills/lanes/SKILL.md`.

## Do
1. Check the preset names in the real `docs/settings.toml` (Balanced/Fast/Quality and the rest) against `face/presets.mojo`.
2. `face/presets_ui.mojo`: Presets tab content on F4's widgets (`face/widgets*.mojo`, `face/settings_ui*.mojo`): preset list per
   group, apply-to-staged, save/delete user preset, Apply button with a confirm that shows `ApplyPlan.text`, disabled when
   `blocked()`. Mount it through F4's tab bar (one call).
3. Apply action: stop engine, run `restart_cmd`, poll `/health`, show the fail word (as `face/dash.mojo` reads it). CPU tests
   use a fake command; no real engine.
4. Model tab: GGUF path field, pack lookup via `tools/baro` (dry, no build), model card from `face/dash.mojo` data.
5. Gate: extend `face/settings_gate.sh` (or a sibling) with offscreen frames for Presets and Model, OCR reads the plan text.
   Run under `systemd-run --user --scope -p MemoryMax=8G`. Never open an image or a window.
6. Wire `presets_test` into the same gate.

## Rules
Mojo only. No GPU. Commit each step with a pathspec, no co-author tag, no em dashes. `tools/ci-checks.sh` and
`bench/preflight.sh` must pass. Update the F5 report (what changed, gate output, UNVERIFIED items). Stuck after 2 tries: write
BLOCKED with evidence in the report and stop. Last line in the pane: `DONE` or `BLOCKED`.
