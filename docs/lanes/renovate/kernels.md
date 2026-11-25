# renovate: kernels

skills loaded: mojo-dev, gpu-kernels (no kernel logic edited, so no reference file was needed; edits are dead code and comments only)

Scope: `kernels/*.mojo` except `turbo_rows.mojo`, `matmul_skinny.mojo`, `fewrow.mojo` (live lanes). The `kernels/test_*.mojo` files were read for imports but not edited: they are tests, not kernel files, and their docstrings feed `tools/kernel-census.mojo`.

Branch `renovate/kernels`, 2 code commits + this report. `git diff --shortstat` of the 2 code commits: 34 files changed, 37 insertions(+), 336 deletions(-).

## Bugs fixed

None. No repro found for a behaviour bug in the files in scope (see "Found, not fixed").

## Dead code removed

| symbol or file | lines | proof | commit |
|---|---|---|---|
| `G_ALL` (comptime in `kernels/mega.mojo`) | -1 | word grep of the whole repo: only the definition | 07ebca4 |
| `Q8S_FFN` (`kernels/mega_rows.mojo`) | -1 | word grep of the whole repo: only the definition | 07ebca4 |
| 35 unused import names in 16 files: `llvm_intrinsic` x7 (dattn, freivalds, matmul_mmq, matmul_prefill, matmul_wmma_pipe, mega, moe), `kv_off` x5 (mega, mega_moe, mega_moe_rows, mega_rows, spark_kernels), `row_major` x3, `KVPAGE KVPSH KVHSTR attn_head_body N_LAYERS MEGA_ALLOWED` (mega), `barrier stack_allocation` (moe), `q8_0_row_dot` (mega_moe, moe_rows), `q6k_value` (moe_rows), `NTHREADS` (matmul_pipe), `ceildiv` (matmul_wmma_pipe), `lane_id` (moe_wmma), `block_dim tanh dtype` (spark_kernels), `NORM_EPS BASE_FULL ATTN_SCALE` (spark_mega) | -29 +21 (whole commit) | per file: the name occurs nowhere outside its import line; no other `.mojo` file imports it from the changed module (re-export check over all .mojo files); builds pass (Checks run) | 07ebca4 |

Prior-round leads (`q4k_dot_u`, `gemm_q2b3/tq1/tq2`, `embed_sort_k`, `_opt_field`): no longer present in the tree, nothing to do.
Whole-repo scan of all 828 `def`/`comptime`/`struct` names in scope: after the two constants, no name has fewer than 2 occurrences, and the 45 names with exactly 2 occurrences were each opened: all have a live use.

## Debloat

| what | bytes/LOC before -> after | commit |
|---|---|---|
| comments and docstrings of 22 kernel files (CLAUDE.md: kernel files carry zero comments and zero docstrings) | 307 lines deleted, 16 lines rewritten without their trailing comment; `git show --shortstat 3646bfa`: 22 files, 16 insertions, 307 deletions | 3646bfa |

After it, a grep for `#` comments and `"""` over the in-scope non-test kernel files finds none. The pass is string-aware (`#` inside literals kept). Layout notes that sat in those comments (for example the dflash2 header, the gemma4 attention header, `nanprobe` purpose) are in git history; if the orchestrator wants them kept, they belong in `docs/`.

## Optimisations

None. No GPU, no CPU-side hot path in scope.

## Found, not fixed

| file:line | what | evidence | why not fixed / gate needed |
|---|---|---|---|
| `kernels/mega_moe_rows.mojo:56,100,133,312` vs `kernels/mega_rows.mojo:42,86,119,192` | `rms_rows`, `f32_dot_rows`, `q8_dot_rows`, `pick` are the same source twice (about 110 lines). `mega_moe_rows` could import them from `mega_rows` (no import cycle: `mega_rows` imports only `mega`). | body hash over all kernel defs (`dups.pl`), the 4 pairs identical; `diff` of the two ranges shows the same text up to `wq` | Megakernel code placement is a lottery on this card (memory: megakernel-lottery-fingerprint). Needs an ISA diff of `amar_mega_moe_rows` before and after plus the MoE bit-identity gate and a speed A/B on GPU. Not build-provable. |
| `kernels/test_mega_block.mojo:423-428` | does not build: "Could not infer capture convention of the captured value ctx" (53 error lines, all inside the test file) | `.work/rv/kernels_test_mega_block.log`; the test is not in `run-tests.sh` or `bench/preflight.sh` | Not caused by this lane (errors are in the test's own nested def). Fix needs the `errors.md` capture-convention row and a GPU run of the test. |
| `kernels/test_*.mojo` (14 files, 50 names) | unused imports, for example `test_mega_rows.mojo` (23 names, including `amar_mega_rows`), `test_realign.mojo` (8), `test_sample_*` (`exp`, `log`, `is_valid`, `SAMP_CAP`, `amar_sample_probs`) | `imports.pl` output over `kernels/test_*.mojo` | Tests left alone this round (low value, each needs its own build). Lead for a later round. |
| `kernels/bench_h2d.mojo` | standalone pinned H2D probe, no importer, no script builds it | `grep -rnw bench_h2d` finds no importer and no build line in any tracked script outside private lane notes | A probe tool, not dead code in the usual sense. Orchestrator decides: keep, or move to `bench/`. |

## Cross-area leads

| file:line | what | proof |
|---|---|---|
| `tools/gen-profile.mojo` binary at the main checkout `.work/gen-profile` | the prebuilt binary is older than the source: its profile lacks `QK_NORM`, so `serve/spark.mojo` fails to build against it | build of `serve/spark.mojo` with that profile: "module 'profile' does not contain 'QK_NORM'"; rebuilt from this tree it works |

## Needs regen

None. `tools/kernel-census.mojo` (built from this tree, run without `--check`) leaves `docs/KERNELS.md` byte-identical: 183 kernels, 71 in registry, 0 orphans. No env name was removed.

## Checks run

| command | result |
|---|---|
| `b.sh mojo build` of `serve/engine.mojo`, `serve/moe_gate1.mojo`, `serve/spark.mojo` (profile from `gen-profile` of the 1B llama gguf), `bench/bench_decode.mojo` (shim linked), after commit 07ebca4 | all OK |
| same for 17 `kernels/test_*` that import a changed module: attn_block, dattn, freivalds, gemma4_attn, iq4nl, mega_rows, mmq, moe_block, moe_rows, moe_wmma, prefill, q8d, quant, realign, rope_qkv, spark_attn, wmma_pipe_gelu | all OK |
| `b.sh mojo build kernels/test_mega_block.mojo` | FAIL, pre-existing, errors only in the test file (see Found, not fixed) |
| after commit 3646bfa: builds of `serve/engine.mojo`, `serve/gemma4.mojo` (stub profile with the E2B dims), `serve/spark.mojo`, `bench/bench.mojo` (shim linked); a closure script shows these 4 roots import all 22 stripped files | all OK |
| `.work/rv/kernel-census --check` and a write run, then `git diff --stat -- docs/KERNELS.md` | 183 kernels, 71 in registry, 0 orphans; no diff |
| `grep -nE '^\s*#|"""|\s#\s'` over in-scope non-test kernel files | no match |
| not run (brief): GPU, `run-tests.sh`, `bench/preflight.sh` | UNVERIFIED at run time; edits are import names, unused constants and comments, so the compiled kernels should be unchanged, but no ISA diff was taken |

## Mistakes -> rules

- Shared scratchpad path: a sibling worker wrote its own `dead.sh` to the same scratchpad path as mine and the tool reported my write as an "update". Rule: a worker puts scratch files in its own subdirectory of the scratchpad (`scratchpad/<area>/`), never in its root.
- First test build failed with "unable to locate module" (`window`, `registry`): the build line lacked `-I serve`. Rule (mojo-dev, "building"): copy the `-I` set from the matching `run-tests.sh` line; `spark.mojo` also needs a profile dir from `tools/gen-profile` built from the same tree, and the bench binaries need the shim link flags.
- A second 2 min wait per build was queue time in `b.sh` (load above 12), not compile time. Rule: batch all builds into one background `xargs -P2` run from the first call and watch with Monitor.
- Hook behaviour: compound `git` calls with loops or pipes are refused in a worktree; use single plain git commands.
