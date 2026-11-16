# Brief: gemma4 decode attention kernel for mojo-baro (2026-11-16)

Owner: the main session (Opus) ports gemma4 (Forge-1 E2B) into mojo-baro's dense engine; you write ONE kernel and its
test. Report file: `docs/lanes/2026-11-16-gemma4-attn-report.md` (what landed, commits, test output, open items).

## Mandatory skills: load ALL of these before the first edit (the maintainer, 2026-11-16)
Skill tool, or Read the SKILL.md when the Skill tool does not list it (paths under `~/Brain/Skills/own/` or
`.claude/skills/` in this repo): `gpu-kernels`, `mojo-gpu-fundamentals`, `mojo-syntax`, `mojo-lean`,
`mojo-nightly-lane-builder`, `inference-kernels`, `rdna3-isa`, `kernel-parity`, `bench-gates`, `wavewait`,
`memory-optimization`. Name each one you loaded in the first lines of your report; a skill not loaded is a FAIL item.

## Read first (gates enforce it)
- `~/Brain/Skills/own/rdna3-isa/SKILL.md`, `~/Brain/Skills/own/inference-kernels/SKILL.md` (amd_docs_gate).
- mojo-baro `CLAUDE.md` (gpu-wait carries no env/stdin; 16k floor: tests use T up to >= 16384).
- `kernels/spark_kernels.mojo` `amar_attn_decode_swa_gated` (:244), `amar_attn_split_part` / `amar_attn_split_merge`
  (:291, :328), `kernels/attn.mojo` `attn_head_span`, `kv_off`, `kv_tab_off`; their test `kernels/test_spark_attn.mojo`.
- Reference semantics: `~/Projects/imports/llama.cpp/src/models/gemma4.cpp` (build_attn calls, kv_source_impl) and
  MAX nightly `~/Projects/Modular/max-nightly/.venv/lib/python3.12/site-packages/max/pipelines/architectures/gemma4/`.

## The shape (Forge-1 E2B, GGUF kv)
35 layers, pattern SSSSF x7 (S = sliding window 512, head dim 256, rope base 1e4; F = full, head dim 512, rope 1e6).
8 query heads, 1 KV head (MQA) on both kinds. Attention scale 1.0 (q and k are RMS-normed before rope; no 1/sqrt(d)).
Layers 15..34 compute no K/V: they read the cache of layer 13 (their S kind) or 14 (F kind) at the same positions.

## The kernel
`amar_attn_decode_gemma4[HD_, NQH_=8, WINDOW, ...]` in a NEW file `kernels/gemma4_attn.mojo`, decode (one new token per
sequence), f32 KV (KVT) first:
- one launch per layer; the caller passes the KV slot to read (own layer, or source 13/14 for shared layers), so the
  kernel itself has no notion of sharing beyond "read slot s".
- HD_ in {256, 512}; WINDOW = 512 for S (attend to positions max(0, t-511) .. t), 0 = full for F.
- scale 1.0, softmax in f32, no softcap inside attention (gemma4 caps only the final logits).
- long full-attention contexts: reuse the split-part/merge pattern (or the dattn path) so T = 16384+ does not run one
  block per head serially. Measure where the split pays (T threshold) and say it.
Reuse before write: call/extend the existing span helpers; one generic body over HD_/WINDOW, not two copies.

## Test (Mojo only, no Python twin)
`kernels/test_gemma4_attn.mojo`: CPU brute-force oracle in Mojo (f64 accumulate) vs the kernel, HD 256 and 512, WINDOW
512 and 0, T in {1, 2, 511, 512, 513, 4096, 16384}, random q/K/V, max rel err bar stated BEFORE the run
(suggest 1e-4 vs f64). Include a mutant check (e.g. off-by-one window) that the test must catch. Add it to `run-tests.sh`.
Speed: one bench line per (HD, WINDOW, T in {512, 4096, 16384}) in us, cold where it matters (see bench-gates skill).

## Rules
- Work in your worktree; commit as you go, conventional subjects, why in the body, no self-tag lines.
- Every GPU run: `gpu-wait run --timeout S -- ...`, env inside the command.
- Do not touch serve/, the packer, or other kernels' behaviour; the main session wires the kernel in.
- Stop and report when the test passes plus bench numbers, or after two failed approaches with what you learned.
