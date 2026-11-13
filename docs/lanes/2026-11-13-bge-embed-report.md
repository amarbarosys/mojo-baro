# Lane report: bge-small as the standard embedder (lane-bge-embed, 2026-11-13)

Branch `lane-bge-embed`, worktree `../mojo-baro-lanes/bge-embed`, from main `a5edb35`. Not merged: the owner's session runs `lane-merge`.
Brief: `docs/lanes/2026-11-13-bge-embed-brief.md`.

## Verdict

Steps 1 to 5 and 7: done and checked. Step 6 (machine state): partly done. The auto-mode classifier refused writing under
`~/.config/systemd` and enabling a unit ("unauthorized persistence"), so **`baro-embed.service` is not installed and :8081 has no
listener**. Everything it needs is staged and tested; one command finishes it: `embed/install-service.sh`.

## Skills loaded

`.claude/skills/mojo-baro/SKILL.md`, `mojo-dev`, `lanes`, `bench-gates` (hubs only; the rule files they route to were not
read, so rules #1..#54 numbers were not applied by number).

## What landed (commits on the lane)

| commit | content |
|---|---|
| `d9acea8` | `embed/` (vendored newera bge + wordpiece + oracles, `serve_embed.mojo`, `build.sh`, `norm.tsv`), `serve/src/embed.rs`, routing in `embeddings.rs`, `--embed-only` in `main.rs`, `bench/embed-gate.sh` + fixture, ci-checks drift step, settings, README, CHANGELOG |
| `5f77b68` | staged unit, cois-hub drop-in, `embed/install-service.sh` |
| `9fece08` | gate receipt under `docs/receipts/bge-embed/` (`exchange/` is gitignored) |
| ModelLauncher repo `5543661` | `[models.bge-m3]` commented out with the reason |

## Design (smallest that baro-serve can talk to)

- `embed/serve_embed.mojo`: loads bge-small once, CPU. Line protocol on stdin/stdout: request = texts as hex of UTF-8 bytes
  joined by a space (`-` = empty text); reply = `ok NTOK:HEXVEC ...` (token count, 384 LE float32 in hex, exact) or `err MSG`.
  Rows of one batch run in parallel (`parallelize`). Not the engine wire: the engine wire is a token-id generation protocol
  (`serve/PROTOCOL.md`) and carries no text-to-vector request.
- `serve/src/embed.rs`: `EmbedSidecar` spawns the child on the first request (at boot under `--embed-only`), serialises
  batches, restarts it after a death or a 120 s timeout. 5 unit tests.
- `embeddings.rs` routing: `model` absent or bge-small -> embedder; the loaded chat model's name -> the old hidden-state
  path (unchanged); any other name -> 404 naming the two valid choices. OpenAI response shape unchanged; `/v1/models` also lists bge-small.
- `baro-serve --embed-only --port 8081`: no engine, no pack, no tokenizer, no GPU; fails at boot with `FAIL embed-only: ...` if the child cannot start.
- The cut is 512 tokens (the model's limit). The vendored `wordpiece.mojo` cut at 256, so it carries one patch
  (`encode(max_len=256 default)`); the ci-checks drift step applies the same sed to newera's file before comparing.
- New knobs `BARO_EMBED_{BIN,MODEL,NORM,MAX_TOKENS}` in `docs/settings.toml` and `settings-template.toml` (settings-check PASS, both checkers).
- `bge_gpu.mojo` was **not vendored**: CPU is 0.6 s for 30 texts and the brief makes the GPU arm opt-in only if measured faster
  and identical; I did not measure it. Say if you want it.

## Measured (CPU, 7800X3D)

30 fixture texts in one request: 0.58 to 0.62 s (3 runs). One short text: 10 ms. Embedder RSS 178 MB, peak 272 MB. No VRAM.

## Gate: `bench/embed-gate.sh` (receipt `docs/receipts/bge-embed/embed-gate.txt`)

PASS, 9.5 s, CPU only. 30 texts (short, unicode, multi-line, empty, whitespace, code, 301 tokens, 700 tokens cut at 512).
- baro-serve vs ONNX oracle: 30 texts, min cosine 1.0000000, dims 384, norm 0.9999999 to 1.0000001.
- baro-serve vs brain-ask `embed.mojo`: 26 texts, min cosine 1.0000000.
- **NOT EXERCISED against brain-ask**: rows 16 (multi-line), 17 (empty), 20 (301 tokens), 21 (700 tokens). Its `lines` mode takes one non-empty line and cuts at 256 tokens. Those four are ONNX-only.
- Falsifier: build with mean pooling instead of CLS: min cosine 0.8748 vs ONNX, rejected.
- Arm identity read back from `/proc/<child>/cmdline` and `exe` (model dir, norm table, token cap 512, binary sha) before any number. Rows 20 and 21 separate a 512 cap from a 256 cap (cosine 0.9496 between the two cuts), so an inert `BARO_EMBED_MAX_TOKENS` would fail.
- `/api/embeddings` single text equals batch row 0.
- Also: ci-checks checks `embed/norm.tsv` against `bge_norm_oracle.py` output (equal).

## Finding for the owner

brain-ask cuts at 256 tokens; the server cuts at 512. For a text over 255 tokens the two vectors differ (cosine 0.95 on the two
long fixtures). "Share one embedder" holds for text up to 255 tokens only. brain-ask index chunks are capped at 256 tokens, so the
indexes agree today. If you want CoiOS vectors comparable with brain-ask on long chunks, set `BARO_EMBED_MAX_TOKENS=256`.

## CoiOS (step 6)

Machine-state note with BEFORE (and AFTER): `~/Brain/OS/coios-embed-service.md`.
- Found: `coios-embed.service` already disabled and inactive since 2026-09-05; no job in gpu-wait; nothing on :8081; no `COIOS_EMBED_*` anywhere. `disable` is a no-op, the unit and its drop-in stay.
- Done: `serve_embed`, `norm.tsv` and a copy of the lane's `baro-serve` in `~/.cache/baro/embed/` (13.2 MB copy; after merge point the unit at main's binary). ModelLauncher `[models.bge-m3]` commented (discovery gives that folder port 8082 instead of 8081, measured with `load_models`).
- **Blocked (classifier)**: write `~/.config/systemd/user/baro-embed.service` and `cois-hub.service.d/embed.conf`, `daemon-reload`, `enable --now`. Staged as `embed/systemd/*` and `embed/install-service.sh` (checks prerequisites, installs, disables the old unit, curls a 384-dim answer). `COIOS_EMBED_MAX_CHARS=2000` (about 512 English tokens; the server truncates, so denser text is cut, not refused). The drop-in applies at the next `cois-hub` restart; the installer does not restart the hub (shared MCP endpoint).
- CoiOS end to end: checked against a transient server on :18081 with `COIOS_EMBED_URL/MODEL/MAX_CHARS` set the way the drop-in sets them, on a scratch vault and scratch `COIOS_DATA` (never `cois rebuild` on the live index: it drops every table). `cois rebuild --embed --model bge-small-en-v1.5`: 3 files, 3 embedded, dim=384, `embed_meta` = bge-small-en-v1.5 / 384; `cois recall "which cache distorts GEMM benchmarks"` ranked the Infinity Cache note first (0.925) over unrelated notes (0.393, 0.325). The live :8081 curl is UNVERIFIED until the installer runs.
- Side effect to know: `~/.local/bin/cois-embed` still drives `coios-embed.service` (BGE-M3, GPU, :8081). Starting it after the switch would fight `baro-embed` for the port.

## Receipts

- preflight: `PASS preflight c95a38d7d708` (`bench/preflight.sh`, JOBS=2, includes `cargo test` of baro-serve and ci-checks). A first run failed `tree-changed` because I edited `bench/embed-gate.sh` while it ran; rerun on a quiet tree passed.
- run-tests: `docs/receipts/bge-embed/run-tests.txt` ("run-tests" receipt): `gpu-wait run --timeout 1800 --vram 22 -- ./run-tests.sh`, `PASS run-tests: all tests, pack .work/engine-pack-q4`, `FAIL: 0`, exit 0. No kernel file changed, so this proves the lane broke nothing, not that the embedder works (the embed gate does that).
- Not run: the 20-prompt engine gates (no engine code changed); `bench/ww.sh` (no kernel change).

## For lane-merge

- `bench/gate-scope.tsv` has no row for `embed/`, `serve/src/embed*.rs`, `bench/embed-gate*`: unmatched paths count as `full`. I did not edit that file (a lane cannot narrow its own gate). A `cpu` row for `embed/*` and `bench/embed-gate*` is the owner's call.
- `lane-prep` linked `serve/target/release/baro-serve` to main's binary; I built into `.work/cargo-target` so main's binary is untouched.
- Primary checkout was not touched (its dirty files are others').
- The commit hook refused the harness's `Co-Authored-By` trailer (user CLAUDE.md §7), so no commit carries one.
