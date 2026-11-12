# Lane F0 report (imkit face shell), 2026-11-12

Branch `lane-F0`, worktree `$HOME/Projects/mojo/mojo-baro-lanes/F0`, rebased on main 15f07a0 after the maintainer's answer (KERNELS.md fix).

## Gate
- Command: `JOBS=2 bench/preflight.sh` (no step excused) then `gpu-wait run --timeout 1500 --vram 22 -- env STATE_ENGINE=$PWD/.work/preflight/engine-dense bash -c './run-tests.sh > .work/F0-gate.txt 2>&1'` (rule #2).
- Exit codes: preflight 0 (`PASS preflight 747c9bf5401b`, `.work/F0-preflight3.txt`), run-tests 0 (`PASS run-tests: all tests`), evidence `.work/F0-gate.txt` in the worktree.
- Test count: after = 123 PASS lines, 0 FAIL, 1 SKIP (`test_kq_prefill_gemm`: no kq pack in this worktree (ornith-stint pack-kq absent), environmental). Before: not run separately; F0 edits no `t` entry in run-tests.sh, so the suite is unchanged. The item states no floor.
- F0's own gate: `face/gate.sh` PASS (net test, 3 headless frames, OCR). Falsifier: `FACE_EXPECT_EXTRA="no such words"` exits non-zero.
- First attempt hit a pre-existing red (KERNELS.md row), fixed on main by the maintainer. Second hit worktree env gaps (`.work/ailab`, `.work/front-text` not linked; I symlinked them to main's). A failing `text_oracle` truncated the tracked `exchange/receipts/front/text-oracle.jsonl`; restored with git checkout.

## Commits (lane-F0)
- e39ff50 vendor imkit v0.9.0 + drift check against the tag (ci-checks.sh loop; upstream checkout is 91 commits past the tag, so it compares tag blobs; mutated file fails)
- b7bb3e2 face skeleton (app.mojo, net.mojo, net_test.mojo, gate.sh)
- 310bf9a window-cost harness (window-cost.sh, engine-with-window.sh, app --fps/--frames/--stats)
- bc9d736 `tools/baro gui`, 30 fps default, README (names SDL3 and FreeType)

## Window cost (step 2)
Resident Qwythos 9B Q4_0 (`.work/engine-pack-q4`, engine sha256 59abf5b0f9edabbc), 20 mtp-prompts, 64 tokens, f32 KV, TMAX 16384, `bench/ab-resident.sh` SEQ=1, arm B = face in an offscreen SDL3 window on the same card. Receipts `.work/wc/{fps60,fps60b,fps30,fps15}/` (summary.txt, arm.txt, face-fps.log).

| cap | nowin median | win median | median paired ratio | prompts below 0.95 | face measured fps |
|---|---|---|---|---|---|
| 60 | 143.9 / 142.28 | 143.55 / 142.59 | 0.995 / 0.996 | 7 / 6 of 20 (about 0.87) | 49.5 to 59.3 |
| 30 | 142.35 | 142.38 | 1.001 | 1 | 29.7 |
| 15 | 142.13 | 142.82 | 1.006 | 1 | 15.05 |

Window adds about 715 MiB VRAM (default memory manager size; `baro gui` exports percent 2, not re-measured); gtt unchanged. Picked 30 fps by number. Host load 2.5 to 7 during runs (other lanes), arms interleaved.

## Live front check
Rust `baro-serve` + the same 9B engine under gpu-wait: `/health` 503 then 200; `tools/baro gui --port 18432 --shots` rendered "front: ok, engine idle, queue 0", read back by OCR (`.work/live/`).

## UNVERIFIED / notes
- Cost with a real KDE window (compositor, vsync, GPU-accelerated renderer) not measured: display isolation; offscreen SDL uses the software renderer. the maintainer's look at the real window is still open.
- `baro gui` without `--shots` (opens a window) not run for the same reason. `--model` start-the-front path not exercised.
- rapid-ocr is not installed; tesseract via `img-read` is the gate OCR.
- iTool `seq-decide` fails to load `libAsyncRTMojoBindings.so` (here and under gpu-wait); ab-resident ran all 20 prompts; summary computed by awk. Run `itools-doctor`.
- 60 fps bimodal prompts (about 125 vs 144 tok/s) not diagnosed.
- Skills loaded: mojo-baro (read), lanes rules.md (read); mojo-dev not loaded.
- QUESTIONs for imkit: none.
