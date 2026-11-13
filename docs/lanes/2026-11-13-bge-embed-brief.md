# Brief: bge-small as mojo-baro's standard embedder (sonnet, effort medium, L)

Owner decision 2026-11-13: `/v1/embeddings` and `/api/embeddings` of baro-serve default to bge-small-en-v1.5 in Mojo.
The old path (chat model's last hidden state, `serve/src/embeddings.rs`) stays, opt-in by naming a chat model.
CoiOS and brain-ask then share one embedder.

Work in a lane worktree: `lane-prep` a worktree `../mojo-baro-lanes/bge-embed` on branch `lane-bge-embed` from main.
The primary checkout has other people's dirty files: never touch them. Commit with a pathspec.

## Load first (list in your report)
`.claude/skills/mojo-baro/SKILL.md`, `.claude/skills/mojo-dev/SKILL.md`, `.claude/skills/lanes/SKILL.md`,
`.claude/skills/bench-gates/SKILL.md` (gate authoring, falsifier).

## Facts (do not re-derive)
- Mojo bge-small exists in `~/Projects/newera/tools/`: `bge.mojo` (CPU), `bge_gpu.mojo`, oracles `bge_onnx_oracle.py`,
  `bge_tokenizer_oracle.py`, `bge_cosine_oracle.py`, `bge_norm_oracle.py`. brain-ask uses it: `~/iTools/brain/brain-ask/embed.mojo`
  (query PREFIX "Represent this sentence for searching relevant passages: " is brain-ask's, applied by the caller, not the server).
- Weights: `~/Models/bge-small-en-v1.5` (safetensors + onnx). 384 dims, 512 tokens, CLS pooling, L2-normalized.
- CoiOS client: `~/Projects/knowledge-systems/CoiOS.dev/coios-rs-v.0.0.2/crates/coios-kernel/src/embed.rs`, env
  `COIOS_EMBED_URL` (default http://localhost:8081/v1/embeddings), `COIOS_EMBED_MODEL`, `COIOS_EMBED_MAX_CHARS`.
  CoiOS has no stored vectors yet (`~/.local/share/coios-brain/index.sqlite` has no vec0 table).

## Steps
1. Vendor the newera bge files into mojo-baro (`embed/` at repo root, with the oracles under `tools/` or `embed/oracle/`),
   plus a drift check in `tools/ci-checks.sh` like the uregex one. Idea is our own code (newera is ours).
2. An embed server process in Mojo (`embed/serve_embed.mojo`): loads bge-small once, reads requests on the engine wire or a
   simple line protocol, CPU by default (`bge_gpu` opt-in if it is measured faster and identical). Choose the smallest design
   that baro-serve can talk to; read `serve/PROTOCOL.md` first.
3. baro-serve: `/v1/embeddings` with `model` absent or `bge-small-en-v1.5` -> the embed process (started by baro-serve, or
   standalone `baro-serve --embed-only --port 8081` so it runs with no chat engine). A chat model name -> the old path.
   OpenAI response shape unchanged. Update `serve/src/embeddings.rs` tests.
4. Gate `bench/embed-gate.sh` (CPU, no GPU unless bge_gpu is on): 30 fixed texts (short, 512-token cut, unicode, empty
   string), vectors from baro-serve vs brain-ask's `embed.mojo` vs the ONNX oracle: cosine >= 0.9999 each, dims 384, norm 1.
   Falsifier: a mutated pooling (mean instead of CLS) must fail. Commit it with its receipt.
5. Docs: README (embeddings section), `docs/settings.toml` for any new BARO_* knob (ci-checks settings-check must pass),
   CHANGELOG unreleased.
6. CoiOS switch (machine state: `~/Brain/OS/coios-embed-service.md` note BEFORE and AFTER): write a new user unit
   `baro-embed.service` running the embed-only server on :8081; set `COIOS_EMBED_MODEL=bge-small-en-v1.5` and
   `COIOS_EMBED_MAX_CHARS` to fit 512 tokens where CoiOS reads its env; disable (not delete) `coios-embed.service`;
   point ModelLauncher's `[models.bge-m3]` entry out (comment it, note why). Check: `curl` the endpoint, and one CoiOS embed
   call end to end.
7. `bench/preflight.sh`, `run-tests.sh` under gpu-wait, lane report `docs/lanes/2026-11-13-bge-embed-report.md` with
   "run-tests" receipt, then STOP: the owner's session runs lane-merge. Do not merge.

## Rules
No em dashes. Kernel files carry zero comments (applies if any file lands under kernels/). Mojo, Python only as oracle.
Never delete anything under ~/Models. Three failed repairs on a step: write the failure in the report and stop.
