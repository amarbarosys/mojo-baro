# F4 report: settings tabs + help (2026-11-12, branch lane-F4)

Skills loaded: mojo-baro, mojo-dev, lanes (hub SKILL.md files read via the mojo-baro hub; `.claude/skills` has no symlinks in this worktree).

## Built
- `face/settings.mojo`: `Settings` model. Reads the contract TOML with the vendored `toml/` package (`-I .`), no hand reader. Staged values over an applied base, `restart_count`, Basic/Advanced filter, `sampling_json()` (sampling.* keys that differ from the default). `load_default()` reads `docs/settings.toml` if present, else the fixture; a broken real file raises.
- `face/widgets.mojo`: tab bar (Chat, Sampling, Engine, Model, Front, Presets, across the top), control per type (slider for ranged int/float, on/off button for bool, one button per enum choice, text field for path/string/unranged number), summary under each row, status badge, "staged" / "staged, restart" mark, help panel on the right (summary, help, key, default, range, restart, status), Basic/Advanced toggle, scrolling list.
- `face/app.mojo` mount: `settings_bar` at the top, panes start at `PANE_Y` 52 (were 16), tab 0 draws chat + engine panel as before, other tabs draw `settings_pane`. This is more than one call: `draw` takes the `Settings`, and the chat and dash panes moved down 36 px to make room for the tab row.
- `face/testdata/settings-sample.toml`: 5-entry fixture, plan contract. `face/settings_test.mojo`, `face/settings_gate.mojo` (renders 7 frames), `face/settings_gate.sh`.
- Presets tab is a placeholder line (F5 owns it).

## Gate
Command: `face/settings_gate.sh` after the four face unit tests and the app build, output in `.work/F4-gate.txt` (copy `docs/receipts/face-F4/F4-gate.txt`). Exit 0.
Test count: face unit tests 3 (chat, dash, net) before, 4 after (settings_test, 5 refusals + load + staging). The gate adds 7 OCR frames. No face test is wired into run-tests.sh, ci-checks.sh or preflight.sh; those files are outside F4.
Falsifier: `SETTINGS_GATE_FALSIFY=1` asks for a label that is not drawn, exits non-zero (`FAIL ocr: tab-1b: 'NO SUCH LABEL Temperature' not read`).
OCR: tab names on a 3x crop of the bar (tesseract single-line mode), active tab via the panel title, labels, one summary, one help text ("Needs an engine restart"), default/range/restart lines, staged mark; Sampling basic hides the advanced label.
Mount check: `face --shots` with no front, Chat tab frame OCR reads Chat, Engine, Presets, Model card (`.work/f4-mount/`, not committed).
GPU: one shared job per frame run (`gpu-wait run --shared --vram 1`), card idle at the time. No engine run needed.

## PNG paths for the morning (nothing was opened on the desktop after the owner's note; one early `xdg-open` of tab-1b.png ran before that note arrived, a viewer window may still be open)
`docs/receipts/face-F4/tab-1b.png` Sampling basic, `tab-2a.png` Engine advanced + staged BARO_TMAX + help, `tab-1.png`..`tab-5.png` Sampling, Engine, Model, Front, Presets.

## QUESTIONS / not done
1. Sampling into the chat request: `face/chat.mojo` (F1, out of scope for F4) builds the body in `Chat.submit`. It needs a `var extra: String` field and one line adding `"," + extra` after `"max_tokens":N`; app.mojo then sets `chat.extra = st.sampling_json()` each frame. `sampling_json()` is tested. Boss: allow that 3-line edit?
2. imkit owner: no tab bar, no tooltip, no dropdown usable inside a scroll list; built from buttons and row click in `face/widgets.mojo`.
3. Owner decisions (Balanced default, Reasoning preset, folded thinking block) belong to F3/F5 and chat; nothing for F4.
4. After F3 merges: nothing to change, `load_default()` picks `docs/settings.toml`. Re-run `face/settings_gate.sh` then; the gate pins the fixture, so it still passes. Text fields use the imkit field (ASCII only, 360 px).
5. Tab keyboard navigation and ctrl-click are not built.

## Suite (boss, 2026-11-13, after rebase on F3)
Loader fix: absent default = unset (49 of 182 real entries). settings_gate PASS. `./run-tests.sh` under gpu-wait: rc=0, FAIL 0 (tail `docs/receipts/face-F4/run-tests.txt`).
