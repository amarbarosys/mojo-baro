# GPU and token efficiency plan (2026-09-16)

Rules this plan implements: `bench/PROTOCOL-RULES.md` P15 to P20. Global versions: `~/.claude/CLAUDE.md`
§9 and §17.

## Measured waste, 2026-09-16 (`gpu-wait stats --days 1`)

135 GPU jobs, **34 failed (25%)**, 254 GPU-minutes held. Every failure traced today was a harness defect
that a CPU-side check would have caught before the job took the GPU:

| failure | caught by |
|---|---|
| `run-tests.sh` red on main for 5 hours after the ckpt-api merge (`test_serve_proto` not updated) | P15 preflight build of every gate binary |
| grammar gate reasoning cases: `max_tokens` + prompt > TMAX, HTTP 400 | P15 one-item smoke |
| RegesCore task arm ignored the bake's `baro.run.env`, engine refused to start | P15 one-item smoke |
| Granite row: `quality-run.sh` edited while the sweep was executing it | P16 never edit a running script |
| E13 eval tool did not build (three API drifts), found only when the run was due | P15 preflight build |
| `torch.compile` arm held the GPU 32 min without finishing 30 steps | P16 per-job budget with a loud kill |

Silent-failure paths in committed scripts:
- `bench/quality-sweep.sh` prints `FAILED ... continuing` and exits 0 when a model fails.
- `bench/quality-run.sh` exits 0 on a skipped model, so a skip reads as a pass.

Time not spent on the question being asked:
- **Reference arms rerun every time.** llama.cpp T=0 output for a fixed GGUF, llama.cpp commit and input ids
  is deterministic; the quality sweep re-generated it for every model.
- **Full gates used for iteration.** 66-request grammar corpus per edit; full `run-tests.sh` after a one-file change.
- **Eval generation runs past the answer** (100 of 120 task items are math with a single final integer).
- **Shared prefixes prefilled per request.** 120 task prompts share one system prompt per type; the engine's
  `ckpt` hints were not used.
- **Eval arms with spec off.** T=0 spec output is identical by the accept rule and about 1.1x faster.
- **The GPU is held during CPU work** (tokenizer server start, scoring, report writing) and sits idle between
  jobs while results are read.

Not a problem (measured): builds. The engine compiles in 11 s, a test in 1.5 s, spark.mojo in 14.5 s.

## Plan

Effort in LOC. Each item names the check that makes it done.

| # | item | effort | done when |
|---|---|---|---|
| 1 | **Loud failure** in every gate script: `set -euo pipefail`; sweeps print a final `FAIL n/N` line and exit non-zero if any model failed or was skipped; skip is exit code 2, not 0. Files: `bench/quality-sweep.sh`, `bench/quality-run.sh`. | XS | a sweep with one forced-bad model exits non-zero and names it (P11) |
| 2 | **Preflight**: `bench/preflight.sh GATE...` builds every binary the gates use and runs each gate's `--quick 1` on the smallest model, all before `gpu-wait run`; gate scripts refuse to start inside gpu-wait without the preflight stamp for the current HEAD. | S | preflight catches a deliberately broken `test_serve_proto` call on CPU in under 2 min |
| 3 | **Reference-arm cache** `.work/refcache/<sha256(key)>.json`, key = GGUF sha256 + llama.cpp commit + sampler params + input ids. `quality-task-ids.py llama` and the PPL llama step read and write it; the receipt prints `refcache hit/miss key=`. Ours arm never cached. | S | second quality run of one model shows `refcache hit` and llama.cpp never starts |
| 4 | **Tiered gates**: `--quick N` on `bench/grammar-gate.py` (first N schemas) and `bench/quality-run.sh` (first N items); commits and reports cite only full-gate numbers. | XS | `grammar-gate.py --quick 8` runs in under 1 min |
| 5 | **Eval token budget**: task prompts stop at the answer line (`stop` on the newline after the final answer, same for both arms), `ckpt` hint at the end of the shared system prefix, `spec: true` on our arm after one 120-item identity check against spec off. | S | identical scores with and without the change on one model; tokens generated per item logged before and after |
| 6 | **GPU held only for GPU work**: `quality-run.sh` starts the CPU tokenizer server and scores outside the job; one resident engine per model across a session's gates. | S | sweep job wall time vs summed per-model GPU steps within 10% |
| 7 | **Delete superseded harness**: the old quality task-eval script keeps only the scoring functions quality-task-ids imports, or they move into it; `tools/retired/*` stays as oracle material only if a test imports it. | XS, net negative LOC | `ci-checks.sh` green, no importer lost |
| 8 | **Teacher-forced identity in one prefill** instead of token-by-token decode (engine request mode returning argmax per position). | M, engine | 20-prompt forced identity equal to today's `BARO_FORCE` result, wall time at least 10x lower |
| 9 | **Report GPU use per lane**: every lane report ends with `gpu-wait stats --days 1` jobs / failed / busy minutes for its window. | XS | next lane report carries the line |

Order: 1, 4, 2, 3, 5, 7, 6, 9, then 8 only if an identity gate is on the critical path. Items 1 to 7 are
harness only, no engine or kernel file changes, about 250 LOC added and some removed.

## Status (2026-09-16 night)

| # | status | evidence |
|---|---|---|
| 1 loud failure | BUILT | `quality-sweep.sh` ends `PASS N/N` or `FAIL k/N <keys>` non-zero; SKIP/PRE_HOOK hacks removed; `quality-run.sh` `set -euo pipefail` with a FAIL trap |
| 2 preflight | BUILT | `bench/preflight.sh` (167 s): ci-checks, every run-tests build, both engines, cargo test+build; stamp per tree; `--check` in `quality-run.sh` and `e13-mini.sh`. Negative check: a broken `test_serve_proto` call fails at `run-tests-builds`, `--check` refuses the tree |
| 3 reference cache | BUILT | `~/iTools/bin/refcache` (iTools `1005a49`), used for llama-perplexity and the llama.cpp task arm. Llama-1B QUICK=5: 141 s miss, 75 s hit, identical scores |
| 4 quick gates | BUILT | `QUICK=N bench/quality-run.sh`, `QUICK=N bench/grammar-gate.py` |
| 5 eval budget | PARTLY, by measurement | stop-at-answer: 0.0% of 163,465 tokens come after the answer, not built. Shared-prefix ckpt hints: ~70 prefix tokens vs ~238 decode per item, under 3%, not built. Spec on for dense ours arm: BUILT; identical to spec off on 20/20 JSON items after stop-token truncation (spec emits a few tokens past a stop, checked per window); math items not yet compared. Capped generations hold 60% of all eval tokens (327 of 720 items, 27 correct): lowering the cap changes scores, the maintainer's call |
| 6 GPU only for GPU work | BUILT | `quality-run.sh` runs outside gpu-wait, one job per GPU step; tokenizer server, prep, builds, scoring on CPU |
| 7 delete superseded | BUILT | old quality task-eval script deleted (scoring moved, 1,440 saved answers rescore identically); `tools/retired/gguf-tokenizer.py` and `test_tokenizer.py` deleted; `baro-tokenize.py` kept for `--chat`. `ci-checks.sh` no longer skips grammar-importing benches (26 -> 31 built) |
| 8 one-prefill identity | NOT BUILT, by measurement | a 20-prompt identity A/B decodes ~1,280 tokens per arm, 20-40 GPU s; an M engine change to save under a minute per gate fails P20 |
| 9 GPU use in reports | BUILT | `quality-sweep.sh` prints `gpu-wait stats --days 1`; `~/iTools/bin/gpu-waste --since H` gives failed/cancelled jobs by cause (24 h: 34 failed, 5 cancelled, 91.3 GPU-min) |
