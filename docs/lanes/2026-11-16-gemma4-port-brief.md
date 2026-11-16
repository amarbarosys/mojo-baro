# Brief: gemma4 (Forge-1 E2B) packer and dense-engine wiring (lane-gemma4-port, 2026-11-16)

Boss pane: a lane pane. Report: `docs/lanes/2026-11-16-gemma4-port-report.md`.
Per milestone and at the end: `~/iTools/bin/herd tell a lane pane "DONE gemma4-port <milestone> -> <report> | gate=... tok_s=..."`.

## Goal
Forge-1-Gemma-4-E2B runs in mojo-baro's dense engine, greedy chat coherent, teacher-forced agreement against llama.cpp
at the bar the repo uses for a new arch (`bench-gates`), decode speed measured against llama.cpp (99 tok/s here).

## Load first (mandatory, before any edit)
Skills: `model-bringup` -> `arch-port` (route row 1), `gguf-forensics`, `gguf-tokenizer-spec`, `gpu-kernels`,
`bench-gates`, `mojo-lean`, `mojo-nightly-lane-builder`, `memory-optimization`, `lanes`. Read `docs/rules.md` and
`~/Brain/SINDEX/` mojo-baro entries. `brain-ask --brief "gemma4 port"` before theory.

## Inputs
- Weights: `~/Models/forge-1-gemma-4-e2b-f16/Forge-1-E2B-F16.gguf` (+ RUN.md). Base HF: `~/Models/gemma-4-e2b-base`.
- Semantics: `~/Projects/imports/llama.cpp/src/models/gemma4.cpp` (PLE per-layer inputs, shared KV with kv_source,
  SWA/full interleave, V rms-norm without weight, rope_freqs on full layers, final logit softcap, layer_output_scale,
  double-wide MLP). Research: `~/Brain/models/2026-11-16-gemma4-support-research.md`, `2026-11-16-forge-1-gemma-4-e2b.md`.
- Per-layer oracle: MAX nightly runs Forge correctly (`.work/gemma4/max_probe.py`, `--no-prefer-module-v3`,
  pool env 85 inside gpu-wait); llama.cpp for logits.
- Attention kernel: `kernels/gemma4_attn.mojo` (merged from lane-gemma4-attn). Contract: report
  `docs/lanes/2026-11-16-gemma4-attn-report.md` "Contract for the wiring": TWO KV pools, S (HD 256, window 512) and
  F (HD 512, full), att_i counted within its pool; a shared layer passes its source layer's slot in that pool;
  t_len counts the new token; g4_gb / g4_nld / g4_bt / g4_nsplit hold the launch picks.

## Milestones (each ends on its gate; commit as each lands)
M1 Packer: GGUF -> baro pack with a per-layer profile (kind S/F, head dim, KV source, PLE slice, MLP width).
   Gate: every tensor accounted for (count + bytes vs gguf-forensics), round-trip dequant equal.
M2 Forward on CPU-free path, prefill only, layer by layer vs MAX/llama.cpp dumps. Gate: per-layer cosine >= 0.9999
   at layers 0, 1, a shared layer, a full layer, last.
M3 Decode with the two-pool KV and the new kernel. Gate: teacher-forced agreement vs llama.cpp on the repo's held-out
   prompt set (context >= 16k, max_tokens >= 16000 per repo gates), plus a reverse arm (shared-KV routing broken
   must fail).
M4 Chat template + tokenizer (gguf-tokenizer-spec parity), `baro chat` greedy sanity, decode tok/s vs llama.cpp
   under gpu-wait, numbers in the commit.

## Rules that bite
- Every GPU job: `gpu-wait run --timeout S -- env ...` (no env/stdin carried). CPU preflight never under gpu-wait.
- Global env sets MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=2: override per job.
- Do not edit `kernels/gemma4_attn.mojo` semantics; a needed change -> report it, own commit, rerun its gate.
- Correctness claim = teacher-forced agreement, never greedy equality. No Python twin for new Mojo code.
- Stuck after 2 tries on one milestone -> report, do not grind.
