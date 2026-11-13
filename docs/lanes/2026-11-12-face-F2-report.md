# Lane F2 report (face-dash), 2026-11-12

Branch `lane-F2`, worktree `$HOME/Projects/mojo/mojo-baro-lanes/F2`, based on main 43de064 (F0 merged). Item F2 itself is built and its own gate passes. The full-suite gate (`run-tests.sh` under gpu-wait) was NOT run: Boss stopped all new GPU runs. Status: UNFINISHED against the lane's standing rule.

## What was built
- `face/dash.mojo`: `Dash` state and `dash_pane`: engine state tag, queue depth, model card (model id, pack, tmax/seqs/spec_k/tier), last done line (tok/s, path word, fail word), tok/s spark line (last 60 done lines), VRAM bar. Sources: front `/health`, `/v1/models`, the `timings` object of a completion reply (done line), `/sys/class/drm/card*/device/mem_info_vram_*` (the card with the most VRAM; the counters rocm-smi prints).
- `face/dash_test.mojo`: CPU unit test (format, done-line ingestion, 60-sample ring, 502 fail word, vram reader). PASS.
- `face/dash_gate.sh`: the item's gate (below).
- `face/app.mojo`: pane mounting only: one `dash_pane(...)` call replaces the placeholder text; plus what that call needs (`from dash import ...`, `var dash = Dash()`, `draw` takes `dash, host, port`). Boss said "only the one call"; the extra three lines are the state the call needs. F1 will touch the same `draw` signature: expect a trivial rebase conflict, keep both parameter lists.

## Gate (item F2): PASS
- Command: `face/dash_gate.sh` (real `baro-serve` front + real engine `.work/engine-9b` built from this tree, pack `engine-pack-q4` = Qwythos 9B Q4, `BARO_TMAX=16384`, one `gpu-wait run --vram 16` job, 3 headless frames via `SDL_VIDEODRIVER=offscreen --shots`, tesseract OCR via `img-read`). Exit 0.
- Numbers (receipt `docs/receipts/face-F2/compare.txt`, all 3 frames identical): tok/s panel 143.9 vs done line 143.94 (6 completions, 64 tokens, spec path, fail 0); VRAM panel 14.8 / 24.0 GiB vs rocm-smi 14.35 / 23.98 GiB (3.1% on used, inside 5%).
- Falsifier: `DASH_GATE_SKEW=1.2` exits 1 with `FAIL tok/s: frame 1: panel 143.9 vs done line 172.7209` (`falsifier-skew1.2.txt`).
- Unit test: `dash_test.log` PASS.
- Other receipts there: `run.log` (face stdout incl. done lines), `smi.txt`, `health.json`, `server.log`, `gpu-wait.log`, `face-1.ocr.txt`.

## Test count before / after
Item states no floor. Full suite (`run-tests.sh`) not run (GPU stop), so no before/after PASS count. Added: 1 CPU test (`dash_test`), 1 gate script. CPU preflight: exit 0, `PASS preflight daa6398862ea`, 38 builds (`docs/receipts/face-F2/preflight.txt`). Its first attempt failed in cargo (4 text tests: `.work/ailab` and `.work/front-text` not linked into this worktree, same gap as F0; the failing text_oracle truncated `exchange/receipts/front/text-oracle.jsonl`, restored with git checkout); I symlinked both to main's and it passed (`preflight-first-attempt.txt`).

## Commits (lane-F2)
- 5c5715e feat(face): engine panel, tok/s spark line, VRAM bar, model card
- test(face): dash gate on a live front and engine
- docs(face): F2 report and receipts (last commit on the branch)

## Findings and caveats (UNVERIFIED / gaps)
1. No front endpoint exposes the last done line. The done line goes only to the requester. So the pane shows "no run yet" until something feeds `Dash.on_done(Json)` (F1's chat should call it with the reply `timings` or the SSE final chunk stats) or `FACE_DASH_PROBE=N` is set (the gate uses it; the probe blocks the window thread for the request, gate-only). QUESTION for the maintainer: add a read-only `/v1/last-run` (or a stats field in `/health`) to `serve/` so the panel shows other clients' runs? Out of F2 scope (serve/ not listed).
2. The pack dir has no `tokenizer.json`, so the front's text endpoints answer 503. The gate sends the token ids from `<pack>/prompt-tokens.txt` (`FACE_DASH_PROMPT`). Real window use needs a tokenizer next to the pack.
3. VRAM panel includes the face process's own memory (read while it runs); rocm-smi was read after it exited. Hence 14.8 vs 14.35. Inside 5%, but the margin is that process.
4. The state tag text ("idle") is not read by OCR (on_accent text on a colored pill); the gate's "idle" check is satisfied by the F0 status line. Tag colour and text not machine-verified. the maintainer's look is open, as with F0.
5. Engine binary built here has sha256 prefix 6a64c6468b45db0c (F0's was 59abf5b0f9edabbc; sources differ by branch). tok/s 144 matches F0's 9B median (142 to 144).
6. Shots mode renders 3 frames after one refresh, so the spark line has 6 points in the gate; the line itself is not OCR-checked.
7. Tooling: `gpu-wait` job env lacks PATH entries; the gate uses `/usr/bin/curl` and `/opt/rocm/bin/rocm-smi`. The first gate attempt exited 127 for that reason.
8. Skills loaded: mojo-baro (read), lanes rules via brief (not separately read), mojo-dev not loaded. QUESTIONs for imkit: none.

## Remaining steps (cold session)
1. `git rebase main` once F1 has merged; resolve `face/app.mojo` (`draw` signature and the two call sites: keep F1's chat args and `dash, host, port`).
2. CPU: `JOBS=2 bench/preflight.sh` in the lane again after the rebase (passed once at daa6398862ea before it).
3. GPU (needs Boss's go): `gpu-wait run --timeout 1500 --vram 22 -- env STATE_ENGINE=$PWD/.work/preflight/engine-dense bash -c 'cd $PWD && ./run-tests.sh > F2-gate.txt 2>&1 (under .work)'`; record PASS count against the pre-change count; worktree needs `.work/ailab` and `.work/front-text` symlinked to main's (F0 report).
4. Re-run `face/dash_gate.sh` after the rebase (engine at `.work/engine-9b`: `./.venv/bin/mojo build serve/engine.mojo -I . -I kernels -o .work/engine-9b`, symlink `.work/engine-pack-q4` to main's).
5. Decide caveat 1 (last-run endpoint), then wire `on_done` from F1's stream.
6. the maintainer looks at the real window (`tools/baro gui`), then `lane-status .` and `lane-merge lane-F2 --report` (merge order: F1 then F2).
