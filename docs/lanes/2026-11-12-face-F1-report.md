# Lane F1 report (face-chat), 2026-11-12

Result: the chat pane streams over SSE from the real front, and its own live gate passed. The full suite (`run-tests.sh`) was NOT run, because Boss stopped new GPU runs.

Branch `lane-F1`, worktree `$HOME/Projects/mojo/mojo-baro-lanes/F1`, from main with F0 merged (43de064).
Skills loaded: mojo-baro (SKILL.md and rules.md read), article-craft (for this report). mojo-dev and lanes not loaded.

## What was built (face/chat*.mojo, face/chat-gate.sh, one call in face/app.mojo)
- `face/chat.mojo`: `Chat` pane. Message list, `field` input (Enter sends), Send / Stop / Copy buttons, wheel scroll with follow-tail, streaming over SSE. One non-blocking socket polled in `draw` (no thread). The request is HTTP/1.0 so the SSE body is not chunk-framed; a chunked reply is named as an error. Stop, a finished answer, or the `Chat` going out of scope (window close, `__deinit__`) closes the socket; the front then logs `engine: client dropped request N; cancelling`. `max_tokens` 16000.
- `face/chat_test.mojo`: offline test against an in-process listener that sends one byte at a time (events cut inside blank lines and UTF-8, keep-alive, error event, HTTP 503, no `[DONE]`, chunked refusal, refused connect, Stop closes).
- `face/chat_gate.mojo` and `face/chat-gate.sh`: headless driver (typed text as `Input.text`, Return key, mouse press and release on Stop) and the live gate.
- `face/app.mojo`: left pane is `chat.draw(...)`; `Chat` created in `main`; `win.text_input(ui.field_focused)` on focus change (needed for typing in a real window); `draw` takes `mut chat`. Right pane and status line untouched (F2 owns them).

## Gate
- Own gate: `face/chat-gate.sh .work/F1-gate-live`, exit 0: `PASS chat gate: 2 prompts streamed and OCR-read back, Stop and close each cancelled at the front (2 'client dropped' lines)`.
  Real front `baro-serve` with the llama-3.2-1b BARO engine and pack (main checkout `.work/baro-llama-3.2-1b-instruct-q4_k_m-q4_K_M-BARO-93a8b08`), TMAX 16384, one gpu-wait job (`--shared --vram 4`).
  - 2 prompts through the pane: answers of 167 and 154 bytes; tesseract (`img-read`) read the first 4 words of each answer back from its own frame ("the lighthouse keeper a", "the majestic ship the"); frame 2 also holds turn 1.
  - Baseline: 0 `client dropped` lines after the two normal answers.
  - Stop clicked at 217 bytes: pane reports `stopped` and idle; front log gains `client dropped request 3`.
  - Chat dropped at 205 bytes (window close): front log gains `client dropped request 4`. `/health` 200 after both.
- Falsifier: `FACE_FALSIFY=1` (expectation replaced by words nobody streamed) exits 1: `FAIL ocr: answer 1: 'zebra quartz nobody streamed' not read back`.
- Offline: `PASS chat_test: sse parse, 5 stream shapes, refused, stop closes the socket`.
- Receipts (small text, committed): `docs/receipts/face-F1/live/` (gate.txt, two-prompts.log, stop.log, close.log, dropped-lines.txt, chat-{1,2}.ocr.txt, server.log, inner.log, chat_test.log) and `docs/receipts/face-F1/falsify/`. Frames (PPM and PNG, 2.6 MB each) stay in the worktree `.work/F1-gate-live/`.
- Test count before/after: the item states no floor; no `t` entry added to `run-tests.sh` (face/ is not in it, as for F0). Suite count not measured (see UNVERIFIED).

## Commits (lane-F1)
- 300c49b chat pane with SSE streaming client and offline test
- b1f7d6f mount the chat pane, headless gate driver and live gate
- fix: pass PATH into the gate's gpu-wait job (gpu-wait drops the shell env; `img-read` was not found on the first live run)
- docs: this report and the receipts

## Preflight
`JOBS=2 bench/preflight.sh` was started (`.work/F1-preflight.txt`) and was still in its first step (ci-checks) after about 15 minutes. I stopped it by PID so the docs commit could land (rc 143). NO preflight result: UNVERIFIED.

## UNVERIFIED
- `run-tests.sh` full suite on this tree: not run (GPU stopped by Boss).
- Copy button: the clipboard write goes through SDL3 (`SDL_SetClipboardText` on a second handle to libSDL3). It compiles and the button is drawn; it was never exercised, because headless mode has no SDL video.
- Real window, real keyboard (SDL text events), wheel scroll and the look: only headless frames were checked. the maintainer's look is open.
- Scroll and follow-tail have no assertion (frames show only the tail). A long history rewraps the streaming message every frame (bisection per line); not timed.
- `field` is imkit's fixed 360 px single line: a long prompt shows its tail. QUESTION for imkit: a wider or multi-line field, and a clipboard call in `window.mojo`.
- Only the 1B llama pack was driven; Qwen `<think>` text would show inline in the answer (not tested).
- F0's `face/gate.sh` requires the string "Chat placeholder" in OCR; the placeholder is gone, so that gate now FAILS. Not edited (F0 file, outside this item).

## Remaining steps (cold session)
1. In the worktree, check `.work/F1-preflight.txt`; if it is not a PASS for the current tree, run `JOBS=2 bench/preflight.sh`.
2. `gpu-wait run --timeout 1500 --vram 22 -- env STATE_ENGINE=$PWD/.work/preflight/engine-dense bash -c './run-tests.sh > F1-gate.txt 2>&1 (under .work)'` (rule #2); record the PASS, FAIL and SKIP counts here.
3. In `face/gate.sh`, replace "Chat placeholder" in `need=(...)` with "Type a message below"; run it (GPU `--shared --vram 1`).
4. Copy check: add a `--window` arm to `face/chat_gate.mojo` (Window under `SDL_VIDEODRIVER=offscreen`), click `chat.copy_at`, read back `SDL_GetClipboardText`.
5. `lane-status .`, then `lane-merge lane-F1 --report exchange/lane-F1-report.md` (copy this report there), then `git merge --no-ff lane-F1`. F2 rebases on main after.
6. the maintainer looks at the real window (`tools/baro gui --model ...`).
