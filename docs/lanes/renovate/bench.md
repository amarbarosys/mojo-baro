# renovate: bench

skills loaded: mojo-dev, bench-gates, toolscript (global). Not loaded: gpu-kernels and memory-optimization (no kernel or hot-path
change in this area), codebase-memory (bench/ is an excluded subtree of the graph, `not_indexed_dir`; every no-use claim below is a
whole-repo `grep -rIlF` over all file types, `.git .work .venv target` excluded).
Totals before this report commit (`git diff --shortstat 2266fa9..HEAD`, `git rev-list --count`): 16 commits, 43 files, +129 -728 lines.
Method: 216 `bench/*.sh` scripts plus 49 harness `.mojo` files. `bash -n` on all scripts: clean. No `shellcheck` on the box. Gates were
read for four defects (a verdict that is not the exit status, an empty set that passes, a `set -e` trap that hides the FAIL line, a
stale binary measured after a failed build), then every fix was shown on a stub: before the fix the stub exits 0 or runs the stale
binary, after it exits 1. Stubs are CPU only; no GPU, no engine, no `gpu-wait` was used.

## Bugs fixed

| file:line | repro (command and result, before -> after) | commit |
|---|---|---|
| bench/claims-verify.sh:68 | `claims-verify.sh OUT all nonesuch` -> `PASS 0/0`, rc=0 -> `FAIL 0/0: no step matched`, rc=1 | 3e43d06 |
| bench/claims-verify.sh:46 | `GPU_WAITING_ROOM_JOB=1 bash -x claims-verify.sh OUT bundle nonesuch` entered the 5 CPU steps (preflight, build, router-affinity, minja, pwa-wiring) inside the GPU job; inner run now gets phase `gpu`, 0 CPU steps | 3e43d06 |
| bench/preflight.sh:11 | stamp pipeline over a list whose last sorted path is tracked but deleted: rc=1 `FAIL preflight stamp: file hashing failed` -> rc=0 with a stamp | 9535921 |
| bench/force-ab-serve.sh:76 | stub ref + stub cand at `forced agreement: 0 / 64` on 20 prompts: `min 0.0%`, rc=0 -> `FAIL: forced agreement under 100.0% on p01-water ...`, rc=1; 64/64 stays rc=0 | 929634f |
| bench/force-ab.sh:45 | a result set with no scored prompt skipped the bar (`pct and min(pct)`); now fails (`not ok`) | 929634f |
| bench/mspec-trace.sh:19-38 | stub engine without the megakernel line: 100 bad receipt rows, rc=0 -> `FAIL receipts 100: ...`, rc=1; a complete stub stays rc=0 | fa809fc |
| bench/spark-ab.sh:49 | the script ended in `wait` on the killed llama-server: `bash -c 'sleep 30 & p=$!; kill $p; wait $p'` gives 143, so the exit status was always 143 and a tokenizer-id FAIL or a Python traceback was invisible; now exits with the summary's status, 1 on tokenizer FAIL, and refuses a missing engine (rc=1 FAIL line, run) | c966475 |
| bench/ssm-mrow-run.sh:17 | stub toolchain whose build exits 1 and a stale engine: `build exit: 1`, stale engine ran 36 profile runs, rc=0 -> `FAIL build`, rc=1, binary removed first; non-zero run exits now end in `FAIL runs: n` | fc6431c |
| bench/fp16-templates.sh:30 | jsonl pair with `correct=false`: `INCORRECT at [256]`, rc=0 -> rc=1 (all-correct stays rc=0) | 0f32393 |
| bench/race-fp16.sh:30 | stub bench printing `"correct": false` entered the medians, rc=0 -> `FAIL race-fp16: kernel output not correct in a@1 a@2`, rc=1 | 0f32393 |
| bench/fp16-sizes.sh:16 | syntax-error source plus stale binary printing gflops 123456: error text then `512: samples [123456] ... OK`, rc=0 and a .jsonl row -> `FAIL build`, rc=1, no row, no stale binary (run through `b.sh`) | 12fb183 |
| bench/check-cert-hook.sh:44,55; cert-ab.sh:32; q4x-serve-check.sh:164 | `n=$(grep -c PAT f)` at a count of 0 under `set -euo pipefail`: rc=1 with no output, so the `fail ... "0 of 21 requests answered"` line below never ran -> guarded, the reason line prints (exit status unchanged, still 1). Shell semantics shown on the same assignment; the four gates themselves need a GPU, UNVERIFIED by a run | 06f5b21 |
| bench/prefill-long-run.sh, prefill-long-stage0.sh, carryover-run.sh, latent-handoff.sh | 7 `gpu-wait run` calls had no `--timeout` (P16 wall budget): added 1800 s (3600 s for the E8 handoff), generous caps not measured; `carryover-run.sh Q` exited 0 running nothing, now rc=2 | 49a749d |
| bench/ab-prompts.sh:52 | `AB_KILL_AT=0`, arm B with other ids (stubs): rc=0 -> `FAIL ab-prompts: identity differs on p01-water p02-python-fib`, rc=1; equal ids rc=0; default kill-line path (exit 3) untouched | e29caa5 |
| bench/latent-bridge-dump.sh | documented command `bench/latent-bridge-dump.sh SRC SEL` -> `Permission denied` rc=126 (mode 644) -> runs (mode 755) | b7e01ab |
| bench/quality-sweep.sh:8 | `mapfile < <(jq ...)` on an unreadable models json: `PASS 0/0`, rc=0 (stub tree) -> `FAIL setup: no model keys`, rc=1 | 1afc360 |
| bench/q4x-prefill-gate.sh:44 | `N_PROMPTS=0` would print `PASS 0/0` (read from the code, UNVERIFIED by a run: needs the binary and the GGUF); now fails when n is 0 | 1afc360 |

Regression test for the claims-verify, force-ab, force-ab-serve, ab-prompts and race-fp16 rows: `bench/tests/gate-exit-selftest.sh`, 9 stub cases in 7 s, `PASS 9/9`
(commit 83a721d, which also adds the shell-trap list to bench/PROTOCOL-RULES.md P16). It is not wired into `run-tests.sh` or `ci-checks` (those are outside this area): needs a one-line call there.

## Dead code removed

| symbol or file | lines | proof | commit |
|---|---|---|---|
| bench/moe-gate-resident.sh | 30 | `cd` into `mojo-baro-lanes/MOE` (absent), no `set -e`, reference ids `.work/moe-w3/...` and pack `.work/moe-w1/pack` absent, prints a mean with no bar. Whole-repo grep finds two history sentences in bench/moe-protocol.md (annotated), no caller | bd87359 |
| bench/fr-gen-prompts.txt | 634 | no reader: grep for `fr-gen`, `gen-prompts` and the file name finds nothing outside the file; the generator its receipt note names (`bench/fr-gen-corpus.sh`) is not in the tree | bd87359 |
| 46 unused names in `from X import` lines of 17 harness files (list: `git show <commit>`) | 28 added, 37 removed | per name: no occurrence in the file outside the import line; cross-file check: the names other benches import from `bench_latent_handoff` and `bench_prefill` are definitions there, none of the removed names; all 17 files rebuilt (see Checks run) | cfd797b |

## Debloat

| what | bytes/LOC before -> after | commit |
|---|---|---|
| fr-gen-prompts.txt + moe-gate-resident.sh | 47344 B, 664 lines removed | bd87359 |
| unused imports | 37 -> 28 lines over 17 files | cfd797b |

## Optimisations

none (CPU-side scripts only; no timed hot path in this area was touched).

## Found, not fixed

| file:line | what | evidence | why not fixed / gate needed |
|---|---|---|---|
| bench/ab-resident.sh:64-79 | header says "exit 1 on any FAIL"; the closing Python prints `identity fails: [...]` and exits 0 | stub-free reading; arms with another env can differ on purpose | semantics call: is an identity FAIL a failure of the A/B or a column? the maintainer or the A/B owner decides |
| bench/moe-launch-count.sh:32 | `BARO_PACK=.work/moe-w1/pack` hardcoded after `$lcenv`, so the env cannot override it, and the dir does not exist | `ls .work/moe-w1` absent | needs the current MoE pack path and a GPU run |
| bench/det-repeat.sh | no `set`, `| tail -1` hides a failing run, no reference anywhere | whole-repo grep: none | a hand tool; delete or fix is a call for its owner |
| bench/dense-run.sh:38 | `ok tok "20/20 tokenized"` is typed, wrong when PROMPTS narrows the set | line read; the verdict itself is gated by MIN_N | cosmetic, one-line fix left for the next GPU-gate edit |
| bench/ornith-run.sh vs qwythos-v2-run.sh | about 70 shared lines (setup, llama server start, agreement loop) in two scripts (P20) | `diff` after renaming the model | merge needs a GPU gate on both models |
| results/e8/ (8.4 MB of 8.7 MB in results/) | per-batch `bNN.json` + `.raw.json` beside `-merged.json` for each round | `du -ak results/e8` | receipt retention is a policy call; README promises receipts are kept |
| 7 scripts (b4-cross-host, checkpoint-api, replay-m, state-timing, t3b-gate, t3b-quad, turn-api) | comments cite `docs/design/latent-os/*.md` of the private AMDHQ repo, absent here | `ls docs/design` | public readers cannot follow; reword when AMDHQ goes private-only |
| compiler warnings in the 17 rebuilt harnesses | `'__add__' is deprecated, use 'unsafe_offset'` (bench_fixedm_kernel_invariance:38,49), `ref 'r' was never used` (bench_q4dot:101) | build logs `.work/build-many/renovate-bench/*.log` | needs the same build to rebuild, low value; next round |

## Cross-area leads

| file:line | what | proof |
|---|---|---|
| ~/Brain/Skills/own/bench-gates/rules.md, SKILL.md "Top rules" | add the shell-trap list now in bench/PROTOCOL-RULES.md P16 (verdict must set the exit status; `grep -c` under `set -e`; a final `wait` on a killed server; `|| true` after a build; `PASS 0/0`) | six of them found in this round, each with a stub repro in the table above |
| run-tests.sh or tools/ci-checks.sh | call `bench/tests/gate-exit-selftest.sh` (7 s, CPU) | the selftest is unwired |
| tools/ci-checks.sh | builds every `bench/*.mojo` with `def main`; a new unused-import check would keep the 46 from coming back | `unused.py` heuristic found 46 in 17 of 49 files |

## Needs regen

none. No env name, kernel file, settings key or capability changed; `docs/KERNELS.md` census lists `kernels/*.mojo`, none touched.

## Checks run

| command | result |
|---|---|
| `bash -n` on every `bench/*.sh` (216) | clean, run before the edits; each edited script re-checked after |
| `bash bench/tests/gate-exit-selftest.sh` | PASS 9/9 (7 s) |
| stub repros listed under Bugs fixed (before and after) | as listed; each before-run shown rc=0 or stale, each after-run rc=1 |
| `b.sh bash bench/fp16-sizes.sh bad.mojo zz 512` | `FAIL build`, rc=1, no stale binary, no .jsonl |
| `tools/build-many.sh renovate-bench` over the 17 import-cleaned files, `--target-accelerator gfx1100`, `-I . -I kernels -I serve -I bench`, shim link, `JOBS=2` through `b.sh` | 16 of 16 plain builds ok; latentos-ipc-probe.mojo (carries the ci-checks needs marker) fails the plain link as ci-checks expects and builds ok with -lamdhip64 -L/opt/rocm/lib |
| `bench/preflight.sh --check` | `FAIL preflight: no passing ...` (the stamp itself ran; full preflight not run, forbidden for workers) |
| full `bench/preflight.sh`, `run-tests.sh`, any GPU gate | NOT RUN (worker rules); UNVERIFIED for every gate's real-engine behaviour |

## Mistakes -> rules

| what broke | rule |
|---|---|
| my first `ab-prompts.sh` stub run passed `x y` as the env strings; the script runs `env BARO_PROMPT=... $envA engine`, so `x` was a command, every log was empty and the table read `nan` | read the arm contract (env strings are `K=V` words) before judging a stub result: bench-gates top rule "read the arm file before any ratio" applies to stubs too |
| a verification run through `b.sh` sat in the slot queue past the 120 s foreground limit and was killed once at `timeout 300` while the box load was 12+ | start slot-wrapped checks with `run_in_background` and an until-loop on the log; set `B_MAXLOAD` only for a trivial parse-error build |
| compound shell lines that mention git together with `$VAR`, pipes or a commit message holding a git word were refused by the worktree guard | one plain `git add`/`git commit -F FILE` per call, literal paths, message in a file |
| `mcp__hub__run_js` has no `require` and `tools.sh` needs a folder alias | in run_js use `tools.read` and `tools.sh('cd /abs && cmd')`; toolscript skill gotcha |
