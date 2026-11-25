# renovate: engine-host

Branch `renovate/engine-host` from 2266fa9. Paths: serve engine host code (not window, tokenizer, serve front).
skills loaded: mojo-dev, codebase-memory, memory-optimization.
Net of the seven code commits (`git diff --shortstat 2266fa9..HEAD`): 15 files changed, 68 insertions(+), 310 deletions(-) (-242 LOC).

## Bugs fixed

| file:line | repro | commit |
|---|---|---|
| none | no failing repro found in this area (see "Found, not fixed") | |

## Dead code removed

| symbol or file | lines | proof | commit |
|---|---|---|---|
| `conv_b`, `conv_acc` comptime aliases (serve/dflash.mojo) | -2 | whole-repo word count 1 each (definition only); launch site uses `amar_df_conv[DH, DGS, ODT, ACC]` directly; engine builds | 9ceb35b |
| `_dequantize_kv_int8` (serve/engine.mojo) | -10 | whole-repo word count 2: definition + a comment in `_dequantize_kv_int8_par`, the function `load_state` calls; engine builds | 0c3d66e |
| unused imports `TensorLayout`, `row_major`, `KV4_ROWS` (engine.mojo) | -2 | one hit per name in the file (the import); engine is star-imported by nothing | 0c3d66e |
| `gemm_q2b3`, `gemm_tq1`, `gemm_tq2` + `matmul_ternary` import (serve/registry.mojo) | -69 | `git grep` over all tracked files: only definitions and prose (docs/CAPABILITIES.md, bench/q4x-prompts-16k/a-capabilities.txt, lane-deadcode reports); kernels stay and are launched directly by kernels/test_ternary_gemm.mojo and bench/bench_coldcache_ternary.mojo, both still build | 93f75a2 |
| 9 unused imports: `OwnedDLHandle` (expert_tier, q4x), `Bitset` (grammar_rt), `unsafe_memset` (latent), `row_bf16` (rows), `listdir` (shard_index), `TensorRef` (wstore), `bitcast` (test_q4x_cpu), `assert_equal` (test_tool_grammar) | -1 net | one hit per name in its file; none of these files is star-imported; builds below | 2763aac |

Prior-round leads checked: `q4k_dot_u`, `embed_sort_k`, `_opt_field`, `greedy_tokenize` are not in this area (already gone or elsewhere); `gemm_q2b3/tq1/tq2` were still present and are removed above.

## Debloat

| what | bytes/LOC before -> after | commit |
|---|---|---|
| serve/prefix.mojo: second SHA-256 replaced by a wrapper over serve/sha256.mojo | 611 -> 512 LOC, 26041 -> 22504 bytes | 318ba93 |
| serve/sample_ref.mojo: `_retained()` + `_argmax_valid()` shared by `sample_row_ref` and `sample_probs_ref` | 355 -> 304 LOC, 11652 -> 10353 bytes | 291ab5e |
| serve/ngram.mojo: `ngram_propose` is the first branch of `ngram_propose_tree` | 43 -> 34 LOC | 6ef2f7d |

Parity proofs: SHA 300-length sweep (0..299 bytes + abc vector) prints identical digests before and after (output sha256 240a0ead907bc698...); sample_ref A/B of old vs new file over 9600 calls (token, prob bits, every probs element; ties, NaN, -inf, all-invalid row; T, top-k, top-p, min-p grid): 0 mismatches; test_ngram all asserts pass.

## Optimisations

| what | number before -> after | command | commit |
|---|---|---|---|
| none | CPU-side lever not found in this area without GPU; not claimed | | |

## Found, not fixed

| file:line | what | evidence | why not fixed / gate needed |
|---|---|---|---|
| serve/engine.mojo:358-381 (`load_state`) | 14 locals (`pos`, `kvn`, `is_v4`, `is_v7`, `t16`, `kv4_bytes`, `kv6`, `t1`, `t2`, `t2b`, `t3`, ...) get an initial value that is overwritten before use | compiler warning "assignment to X was never used" in .work/rv/engine-base.log | cosmetic; the function is on the state-load path (needs the state round-trip gate on GPU) |
| serve/engine.mojo:1542, 2513 | `var terr = String("")` initial value never read | same warning class | cosmetic, same reason |
| serve/engine.mojo:1716 | a `Bool` result is discarded (`setenv`) | warning "'Bool' value is unused" | harmless |
| engine, harness, expert_tier, cpu_worker, latent_mlp, window (serve/ files: 138 of 290 build warnings, 118 of them deprecations; includes window.mojo, not mine) | deprecated API: positional `__getitem__` (about 60), `UnsafePointer` (about 22), `bitcast`/`load`/`store` (about 25), `as_c_string_slice`, `alloc` without Layout | `grep 'warning:' .work/rv/engine-new.log` after the changes: 290 (base 292) | mechanical rename but touches hot host loops in files the GPU gates cover; one batch with an engine identity run |
| serve/shard_index.mojo:103, serve/expert_tier.mojo:85 | `open64` fd is never closed after `mmap` (one fd leaked per shard / per registered file) | read of the two functions; mapping survives close, so only an fd-table leak, bounded by shard count | no repro of a failure (limit not reachable with current packs); a `close` after mmap is a one-line fix to batch with a GPU run |
| serve/moe_gate1.mojo | W3 gate 1 program: no script, test list or doc references it except generated docs/settings.toml | `git grep moe_gate1` over all tracked files; it still builds for gfx1100 (BUILD PASS) | it is a runnable gate, not dead; owner decides if it still belongs |
| serve/wstore.mojo (+ kernels/test_wstore.mojo) | only importer is its own manual test; docs/BIG-MODELS.md calls it P1 groundwork with P2 "not written yet" | module scan: `wstore <-` only kernels/test_wstore.mojo and docs | design call, left |
| serve/implants.mojo vs serve/harness.mojo + serve/window.mojo | `implants.mojo` header says engine.mojo keeps "its own copy of the loader and hook" (same arithmetic twice) | file header; bodies are not byte-identical | needs a spark and engine gate to prove one implementation; not touched |
| spark build | `serve/spark.mojo` was not rebuilt after the grammar_rt/latent_mlp-adjacent import edit | bench/fixtures/llama1b-profile/profile.mojo lacks `QK_NORM`, `tools/profile.mojo` lacks H/FFN/..., so no profile in the tree builds spark here | UNVERIFIED for spark; grammar_rt compiles inside engine and test_tool_grammar |

## Cross-area leads

| file:line | what | proof |
|---|---|---|
| bench/bench_latent_handoff.mojo:87 | `strip_ws` is a body-identical copy of `serve/identity.mojo:33 strip_ws`, and the bench already imports `identity` | body-hash duplicate scan (tools/ bench/ kernels/ serve/) |
| kernels/test_sample_device.mojo:231, bench/sample-spark-device.mojo:93 (also `load_logits` x4, `crit` x4, `path_exists` x3) | duplicated test helpers | body-hash scan: identical normalised bodies |
| serve/spark.mojo:820 read_prompt, kernels/test_realign.mojo:28, bench/bench_hidden_dtype.mojo:75 | three identical `read_prompt` | body-hash scan |
| kernels (24 warnings each in dflash_kernels.mojo and doorbell.mojo, 16 in moe.mojo, 13 in state.mojo, 10 in turbo_rows.mojo) | deprecation / unused-assignment warnings | .work/rv/engine-base.log (kernels owner) |

## Needs regen

| generated doc | reason |
|---|---|
| docs/CAPABILITIES.md | ternary paragraph says registry.mojo defines `gemm_q2b3/tq1/tq2` (removed in 93f75a2) |
| docs/KERNELS.md census | `amar_matmul_skinny_q2b3row/tq1row/tq2row` no longer list serve/registry.mojo as a user |
| docs/settings.toml (sources lists) | none expected; no env name changed |

## Checks run

All through `.work/renovate/b.sh` slots; no GPU, no engine run.

| command | result |
|---|---|
| mojo build --target-accelerator gfx1100 serve/engine.mojo (base, 2266fa9) | PASS, 292 warnings |
| same, after commits up to 2763aac minus ngram (prefix, dflash, registry, engine, imports) | PASS, 290 warnings |
| mojo build .work/rv/shatest.mojo before and after the prefix change, outputs diffed | identical (sha256 240a0ead...) |
| ab_sample (old vs new sample_ref, 9600 calls) | PASS, 0 mismatches |
| kernels/test_sample_ref.mojo build + run | PASS: host reference sampler |
| serve/test_ngram.mojo build + run | PASS (9 asserts) |
| serve/test_latent_ipc.mojo build + run | PASS (LATENT_IPC_HEADER_PASS) |
| serve/test_fsm_wire.mojo build + run | PASS |
| serve/test_q4x_cpu.mojo build + run | PASS |
| serve/test_tool_grammar.mojo build + run | PASS |
| builds only: serve/q4x.mojo (gfx1100), tools/llama-slot-to-state, tools/latent-recv, tools/shard-index-check, kernels/test_prefix, kernels/test_wstore, kernels/test_latent, kernels/test_expert_tier, kernels/test_ternary_gemm, bench/bench_coldcache_ternary, serve/moe_gate1.mojo | PASS |
| serve/spark.mojo build | not run to completion: no usable model profile in the tree (see Found, not fixed); UNVERIFIED |
| engine rebuild after the ngram commit (6ef2f7d) | not run; signature unchanged, test_ngram builds against the new file |

## Mistakes -> rules

- Tool/skill step: the scratchpad directory is shared by all ten workers; another worker overwrote my `dead.sh` in it mid-session. Rule: every worker uses its own subfolder of the scratchpad (here `scratchpad/eh/`) from the first write.
- Tool step: a `git` command in a compound form (`cd && git grep | ...`, `$(...)` plus pipes) is refused in a worktree-isolated agent. Rule: one plain `git` call per Bash call, or run the pipeline from a script file written with Write and keep `git` out of it.
- mojo-dev step "build a bench or tool that imports serve/spark.mojo": the skill gives no profile to use; the fixture profile `bench/fixtures/llama1b-profile` is stale (no `QK_NORM`). Rule for the skill: name the profile source (`tools/gen-profile.mojo` for the model in hand) in the spark build line.
- Editing a shell script while a background job runs it: `sed -i` makes a new inode so the running copy is unaffected, but the second chain step used the old flags. Rule: start a new script file for changed flags, never patch a running one.
- Not done (outside my paths): SINDEX entry and docs/rules.md lines for the four items above; the orchestrator folds them in.
