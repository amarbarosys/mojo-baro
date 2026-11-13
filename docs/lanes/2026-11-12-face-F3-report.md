# F3 report: settings schema (2026-11-12)

Built: `docs/settings.toml` (182 settings: 172 read by code, 9 request fields, 8 presets), `tools/settings-check.mojo`
(drift check, `--readme`, `--toml FILE`), ci step with two falsifiers, README Configuration table generated.

- Gate: `JOBS=2 bench/preflight.sh` under `systemd-run --user --scope -p MemoryMax=16G`, output `.work/F3-gate.txt`, exit 0 (`PASS preflight b310673dcb5c`). Receipt copy: `docs/receipts/face-F3/preflight.txt`.
- Count before/after (item floor: schema checks + 2 falsifiers): ci-checks 31 OK / 22 steps -> 34 OK / 23 steps, 0 FAIL (`ci-baseline.txt`, `ci-after.txt`); cargo 135 + 24 passed.
- Falsifiers: BARO_TMAX deleted -> `missing from docs/settings.toml: BARO_TMAX`; BARO_POOL renamed -> `not read by the code`; a gone request key fails. All run in ci.
- First preflight failed 4 cargo tests: the worktree had no `.work/ailab/pack-kq` (env, not code). Fixed by symlinking `.work/ailab` to the main checkout; rerun passed. Candidate for `lane-prep`.
- Owner decisions applied: Balanced = spec k=2, f32 KV, 32K, temp 0.7, top_p 0.9 (face falls back to int8 when it does not fit: F5 logic, not a TOML value); Reasoning = thinking on (`chat_template_kwargs.enable_thinking`), max_tokens 16000. No images or windows; F3 makes none.
- GPU: NOT RUN (none needed).

## Notes for F4/F5
- `default` absent = unset. `kind` = env | build | server-env | shell | external | request. Build flags need a rebuild plus restart (`restart = true`).
- Help text is migrated from the template docs (sources and receipts inside); the 13 basic entries and request fields are hand-written. "One measured number" exists for BARO_SPEC (1.104x), KVQ (99.1%), MEGA. Others say unmeasured.
- Not done, out of the item's file list: `tools/profile.mojo` and `tools/profile.py` still read `docs/settings-template.toml`, so the template and `tools/settings-check.py` stay; the new check cross-checks the key sets. Follow-up: point profile at settings.toml, delete both.
- `cpu/engine_cpu.mojo` reads `BARO_THREADS`; neither check scans `cpu/`. Scope gap, left as is.
- Thinking fold toggle is front state (F4), not a setting.

## Suite (boss, 2026-11-13)
`./run-tests.sh` under gpu-wait: rc=0, FAIL 0, all tests with pack `.work/engine-pack-q4` (tail in `docs/receipts/face-F3/run-tests.txt`).
