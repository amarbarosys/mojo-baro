# Brief: nerkyor Qwen3.8-27B in the engine, then score it against Forge-gemma4 E2B (2026-11-16)

Boss pane: a lane pane. Report: `docs/lanes/2026-11-16-nerkyor-vs-forge-report.md`.
Per step: `~/iTools/bin/herd tell a lane pane "DONE nerkyor-vs-forge <step> -> <report> | ..."`.

## Load first
Skills: `qwen38-nerkyor-coder` (the model card digest), `model-bringup` (route: serve, benchmark, eval), `bench-gates`,
`gpu-lane-budget`, `mojo-baro`. Read `docs/BASELINE.md` "Qwen3.8-27B Turbo decode" and the eval table (`baro eval`).

## Inputs
- A: `~/Models/qwen38-27b-nerkyor-coder-q4lynn/Qwen3.8-27B-Coder390-EfficientThink-Q4LynnStyle-Q8MTP.gguf`
  (qwen35 hybrid, same shape as Qwen3.8-27B-TurboFCFusion: H 5120, 64 layers, NQH 24, NH_V 48, FFN 17408, MTP blk.64).
- B: Forge-1 E2B, `~/Models/forge-1-gemma-4-e2b-f16/Forge-1-E2B-F16.gguf`, engine `serve/gemma4.mojo`
  (lane-gemma4-port, merging now; use main once merged, else that worktree).

## Steps
S1 Bring A up: gguf-forensics (MTP tensors present, mixed LynnStyle types), bake/pack with the Qwen3.8 shape flags
   (`tools/baro`, `baro-rebake.sh`), `baro chat -p` greedy sanity. Mixed types our q4 pack refuses -> report, do not
   hand-patch weights. Stop ids 248044 AND 248046 (skill gotcha 1).
S2 Correctness of A: teacher-forced agreement vs llama.cpp (`bench/dense-run.sh` or the repo's qwen35 gate), bar
   frozen before the run.
S3 Speed: 20-prompt median decode (`bench/mtp-prompts`), BARO_TMAX >= 16384, arms read back (P1): A plain, A MTP k=2,
   A MTP k=4 (card default 4), B plain. Acceptance rate per k. Same-job A/B where comparing arms.
S4 Quality: `tools/baro eval` for A and B (ppl + 120-task score next to llama.cpp). If eval lacks a gemma4 path,
   add the smallest hook and say so. Note: A's template injects an xhigh system turn and long reasoning; set
   max_tokens >= 16000, record finish reasons (truncation voids a task, it does not fail it).
S5 Table in the report: model, params, VRAM, ppl, task score, decode tok/s (plain / best MTP), tok/s per GB, one line
   verdict per axis. Numbers name their receipt under `exchange/receipts/nerkyor-vs-forge/`.

## Rules
gpu-wait for every GPU job (no env/stdin carried; `env K=V` inside). No run under 16k. Teacher-forced agreement, never
greedy equality. Read each bench script's Usage line before running it. DFlash2 is out of scope (separate lane).
Stuck twice on one step -> report.
