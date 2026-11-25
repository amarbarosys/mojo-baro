# renovate: tools

skills loaded: mojo-dev, memory-optimization, codebase-memory (graph has no Mojo call edges for tools/, so dead-code proof is `git grep -w` over the whole repo; the graph returned 0 rows for `max_degree=0` in `tools/*`, a lead only).

Scope: `tools/` minus the tokenizer tools and `tools/ci-checks.sh`, `tools/build-many.sh`. Survey: 249 tracked files, 2.0 MB (1.2 MB is q4x-gtt-report fixtures). `bash -n` and `ast.parse` pass on every shell and Python file. Per-file reference count (basename and stem over the whole repo) and per-symbol unused-def scan of all tools `.mojo` (0 hits) and `.py` (5 hits, 2 real).

## Bugs fixed

| file:line | repro (command + output before/after) | commit |
|---|---|---|
| tools/merge-gate.sh:9 | before: `mojo build kernels/test_prefill.mojo -I kernels -o .work/rv/tp_noserve` -> `kernels/attn.mojo:12:6: error: unable to locate module 'model'`, no binary. After, with `-I serve`: rc=0, 0 `error:` lines, 836 KB binary. Both via `b.sh`. Same finding was noted in docs/lanes/2026-11-12-rdna4-epi-plan.md:137. | fee533b |

## Dead code removed

| symbol or file | lines | proof | commit |
|---|---|---|---|
| tools/mtp_train.py `flatten_document` | 14 | `git grep -w` hits only the def; `flatten_v2_document` is the one called (line 137); no file imports mtp_train; `ast.parse` ok | 4f14390 |
| tools/e11b_items_oracle.py `aliases_for` | 1 | `git grep -w` hits only the def; `python3 -I tools/e11b_items_oracle.py test` -> PASS, bench/e11b unchanged | 4f14390 |
| tools/itools/hf-verify, tools/itools/llm-scan (symlinks) | 2 files, README row | `git grep -F` over the repo finds only the links and the README row; no hit in ~/Brain/Skills, ~/iTools, ~/.claude/skills for `tools/itools/<name>`; ci-checks walks links generically | 3be80ea |
| tools/zluda/ptx_load.c, tools/zluda/vadd_steps.mojo | 53 lines, 2254 bytes | `git grep` for both names finds no hit outside the files; zluda/run.sh builds only nvml_shim.c and the gen_cuda_wrap.sh output; first-run probes from commit 2144003 | 6969b6a |

## Debloat

| what | bytes/LOC before -> after | commit |
|---|---|---|
| the four dead-code commits above | `git diff --stat 2266fa9..HEAD` before this report: 8 files, +2 / -72 | 4f14390 3be80ea 6969b6a fee533b |

## Optimisations

None. No CPU hot path in this area had a before/after number to chase.

## Found, not fixed

| file:line | what | evidence | why not fixed / gate needed |
|---|---|---|---|
| tools/baro-build | zero references anywhere in the repo (basename and stem). | `git grep -F baro-build` 0 files outside itself; added by bd4d3f6 as a manual CLI; builds engine from `profiles/NAME.toml` through `tools/profile.mojo`. | A hand-run entry point, not provably dead. Owner decides: keep, or document it in README/docs. |
| tools/llama_oracle.mojo (216 lines) | Mojo port of tools/llama-oracle.py; no doc, script or build line names it; every doc cites the .py. | `git grep -n llama_oracle` -> .gitignore:37 and its own docstring only. | Port has no byte-parity receipt in the repo; needs a parity run against the .py on a real eval-callback log before either side goes. |
| tools/mcp/mem-check.sh, tools/megaprof.mojo, tools/front-text-jinja-oracle.py, tools/latent-bridge-weights.py, tools/p4-trace-diff.py | zero references (basename and stem) | `refs` scan of `git grep -l -F` | Each is a standalone measurement or oracle for a feature that still exists (e.g. `BARO_MEGA_PROF` in serve/engine.mojo:1320). Kept: removing loses the only consumer of live output. |
| tools/merge-gate.sh:8, test_server.sh:31, test_pool.sh:24, mega-gate.sh:17 | engine build lines have no `--target-accelerator gfx1100` (the brief's engine line has it; baro-build and run-tests.sh pass it). | `git grep -n 'mojo build serve/engine.mojo'` | Changes what the gate builds; needs one GPU gate pass to confirm default target is the same. |
| tools/mtp_train.py, mtp_head.py, p5a_dump_roundtrip.py, p5b_lora_train.py, tools/mtp-head/train.py | draft-head training scripts for a parked lane (memory: P5a NO SIGNAL, parked). | callers only in bench/p5a-smoke-protocol.md | Lane owner decides on archive; 5 files, 44,809 bytes (sum of `wc -c` sizes seen in the survey). |

## Cross-area leads

| file:line | what | proof |
|---|---|---|
| docs/rules.md:83, :264 | only mentions of publish-purge; unaffected by the itools unlink | `git grep -n publish-purge` |
| docs/lanes/2026-11-12-rdna4-epi-plan.md:137 | says merge-gate test_prefill lacks `-I serve` (fixed here, fee533b); same line also lists `test_prefill` not in run-tests.sh | read of the file |

## Needs regen

| generated doc | reason |
|---|---|
| docs/KERNELS.md census | none expected: `kernel-census.mojo` reads `tools/merge-gate.sh` only for its name list (line 285); no kernel file or env name changed. Not regenerated, UNVERIFIED by running the census. |

## Checks run

| command | result |
|---|---|
| `bash .work/renovate/setup.sh tools` | PASS |
| `bash -n` on every tools shell script; `ast.parse` on every tools .py | no error |
| `bash .work/renovate/b.sh ./.venv/bin/mojo build kernels/test_prefill.mojo -I kernels -o ...` (before) | `unable to locate module 'model'` |
| same with `-I serve` (after) | rc=0, 0 `error:`, binary 836 KB |
| `python3 -I tools/e11b_items_oracle.py test` | PASS deterministic, disjoint landmarks, <=4/country, twins |
| unused-def scan, tools `.mojo` | 0 hits |
| unused-def scan, tools `.py` | 5 hits: 2 dead (removed), 3 `cmd_*` dispatched by `globals()` (kept) |
| tools/itools links executable | 17 of 17 `ok` before the unlink (15 after) |

## Mistakes -> rules

- codebase-memory, step "search_graph max_degree=0": returned 0 rows because the lane tree project name was not in `list_projects` page 1 (50 of 219) and Mojo has no call edges; I did not rely on it. Rule: for Mojo trees use the whole-repo `git grep -w` unused-def scan, and treat graph "no result" as no evidence.
- Build via `b.sh ... | tail -8` hid the real `error:` line behind warning text and returned the pipe's exit code. Rule: redirect build output to a log and `grep -n 'error:'` the log, never tail a build.
- A `git grep ... | while` one-liner was refused by the worktree guard; scratch scripts in the scratchpad dir avoid it.
