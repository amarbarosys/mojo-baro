# Lane F5 report: presets + apply plan (IMPLEMENTED, gate passing)

Status: Steps 2 to 6 of the 2026-11-13 F5 brief are implemented. The offscreen gate passes, including the fake restart command test and OCR of the ApplyPlan text.

## Built (commits on lane-F5)
- c941cc5 `face/presets.mojo`, `face/apply.mojo`, `face/presets_test.mojo`, `face/testdata/settings-sample.toml`
- a91127b `startup_staged`: Balanced is staged at face start (owner decision)
- Vendored `toml/` package reads settings and presets (no own reader); build with `-I . -I face`.
- presets.mojo: `presets_from_text`/`load_presets` ([[preset]], value order kept), `apply_preset` (staged Dict, returns changed count), user presets `save_user_preset`/`delete_user_preset` in `~/.config/mojo-baro/presets.toml` (override `MOJO_BARO_PRESETS`; replace by name; dotted keys and quotes round trip).
- apply.mojo: `restart_keys` (from `[[setting]] restart = true`), `plan` (restart list / live list), `ApplyPlan.text` (the plain text for the Apply dialog), `with_fit`/`run_fit` (shells `pack-fit PACK --tmax N`, parses FITS/TIGHT/NO), `blocked()` when NO, `restart_cmd` (env of staged BARO_* + `tools/baro serve MODEL --port N`).

## Gate
- `face/presets_test.mojo`: PASS (fixture, apply, startup default, user round trip incl. replace/delete, plan text, verdict, commands). Build: `./.venv/bin/mojo build -I . -I face face/presets_test.mojo -o .work/presets_test`.
- Full CPU suite: `JOBS=2 bench/preflight.sh > .work/F5-gate.txt`, exit 0, `PASS preflight b7c627152a3b` (evidence `.work/F5-gate.txt`, `.work/preflight/`). First run exited 1: 4 cargo tests, missing `.work/ailab` symlink in the worktree; linked to main's, rerun passed. No test edited.
- Test count before/after: UNVERIFIED as a number. preflight does not list face tests; presets_test is one new standalone test (not wired into preflight/run-tests). Item floor not stated in the plan.
- GPU `run-tests.sh` and the restart read-back: NOT RUN (owner: shared small jobs only).

## Initial handoff state (2026-11-12)
1. After F3 and F4 merge: rebase on main; test against the real `docs/settings.toml`; confirm Balanced/Fast/Quality names exist in it.
2. `face/presets_ui.mojo` (not written): preset list per group on the Presets tab, save/delete buttons, Apply button with confirm showing `ApplyPlan.text`, disabled when `blocked()`; mount from F4's tab bar. Needs F4's widgets.
3. Apply: stop engine, run `restart_cmd`, wait for `/health`, show the fail word from dash.mojo. Not written.
4. Model tab (GGUF picker, pack lookup via `tools/baro`, model card from dash.mojo): not written.
5. Offscreen frame + OCR gate for the Presets/Apply plan text. No PNG produced; none opened (owner asleep: no images or windows). PNG paths: none.
6. GPU later: `gpu-wait run --shared --vram 1`, restart with changed `BARO_TMAX`, read back from the engine log.
7. Wire presets_test into a run-tests or ci step.

## Findings
- Mojo `Dict[]` indexing raises: plan/text/restart_cmd are `raises`.
- A string setting holding "true" or "123" reads back as bool/int after a user-preset round trip (documented in presets.mojo).
- `pack-fit` has no `--used-gb` flag in its usage line (the skill table lists one); only `--tmax` is passed.
- QUESTION for the imkit owner: tabs, slider, dropdown, text field widgets are not in imkit; F4 builds them in `face/`, F5 UI will reuse them.
- Skills loaded: mojo-baro, mojo-dev, and lanes.

## Follow-up execution (2026-11-13)

- Step 1 PASS: all eight required names exist in `docs/settings.toml`: Fast, Balanced, Quality, Chat, Code, Reasoning, Low VRAM, Long context.
- `tools/ci-checks.sh`: PASS, exit 0, `.work/F5-codex-ci.log`.
- `bench/preflight.sh`: PASS, exit 0, `PASS preflight 03243106f1de`, `.work/F5-codex-preflight.log`.
- Existing `.work/presets_test` binary: PASS, exit 0, `.work/F5-codex-presets-test.log`; output includes restart/live plan and FITS verdict. No rebuild was needed because `face/presets.mojo` and its test source were unchanged.
- Skills read: `mojo-baro`, `mojo-dev`, `lanes` from `~/Brain/Skills/own/`.
- Steps 2 to 5 and 7 NOT RUN. `face/widgets.mojo` still renders a placeholder for Presets; no preset selection, save/delete actions, confirmation, or disabled state. `face/app.mojo` has no restart process owner or polling state. Model controls are generic settings fields; there is no GGUF picker or pack lookup. `face/settings_gate.sh` does not produce the requested Presets/Model frames or OCR the Apply plan. The report's step 6 remains the boss's work as assigned.
- No real engine started. The offscreen renderer ran through gpu-wait under the 8 GiB scope; no window or image was opened. UI behavior was checked by rendered frames and OCR.
- Skills required by this lane are now read. The imkit widget question remains: tabs, sliders, fields, and buttons are built locally in `face/widgets.mojo` because imkit exposes primitives only.

## Completed follow-up (2026-11-13)

- `face/presets_ui.mojo`: grouped preset selection, stage action, save/delete user presets, ApplyPlan confirm text, and a muted, guarded Confirm button when `blocked()` is true. Mounted by the app tab bar.
- Apply action stops the face-owned `tools/baro serve` process group, starts `restart_cmd`, polls `/health`, refreshes dash data, and shows the dash fail word. `presets_test` runs a fake command and verifies the health timeout without an engine.
- Model tab has a GGUF path field, `tools/baro pack-path` cache lookup (no build), and model/pack/limits card from `Dash`.
- `face/settings_gate.sh` builds the app and gate, runs settings and preset tests, renders seven offscreen frames under the 8 GiB systemd scope, and OCR checks the Model labels and `ApplyPlan.text`. No window or image was opened.
- Gate: PASS, exit 0, evidence copied to `docs/receipts/face-F5/F5-gate.txt`; raw test and OCR artifacts remain in `.work/F5-gate/`. Step 6 of the 2026-11-13 brief is wired into the same gate.
- No real engine restart was run. The action path was exercised only with a fake command, as required for CPU tests. GPU run-tests and real restart/read-back remain owner work.
- `tools/ci-checks.sh`: PASS, exit 0, `.work/F5-final-ci.log`. `bench/preflight.sh`: PASS, exit 0, `PASS preflight 96ed8367fabb`, `.work/F5-final-preflight.log`.
- Commits: `c1fef35` Apply and pack lookup; `996ec1a` Presets and Model tabs; `b479478` offscreen gate and test wiring.
- UNVERIFIED: no real engine restart/read-back or GPU run-tests. The `claim-check` executable is not installed; the gate receipt records the checked claims.

## Suite (boss, 2026-11-13)
settings_gate PASS (presets_test + 7 frames incl. Apply plan OCR), face gate PASS, chat gate PASS. dash_gate PASS on the live 9B engine (.work/engine-9b built from main, pack engine-pack-q4), skew falsifier rc=1 (`docs/receipts/face-F5/dash-gate.txt`). `./run-tests.sh` under gpu-wait: rc=0, FAIL 0 (`docs/receipts/face-F5/run-tests.txt`). Real engine restart from Apply: PASS (face/apply_gate.sh, `docs/receipts/face-F5/apply-gate.txt`; needed the stop-engine fix and a rebaked 9B bake).
