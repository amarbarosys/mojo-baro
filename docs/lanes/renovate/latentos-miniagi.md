# renovate: latentos-miniagi

Area: `latentos/`, `miniagi/` (without `miniagi/latent/tok*`, without any private path), `ncm/`. Branch `renovate/latentos-miniagi`, base 2266fa9.

skills loaded: mojo-dev, memory-optimization, codebase-memory, toolscript. Not loaded: gpu-kernels, bench-gates (no kernel or GPU change, no GPU run).

Method: whole-repo identifier census of every `def/fn/struct/comptime` name of the area (997 names, `git grep -w` outside the definition line and outside `.md`), repeated-6-line-window duplicate scan, path-existence scan of every path named in the area's comments and scripts, compiler warning read of `run.mojo`, `bridge.mojo`, `ncm.mojo` builds, CPU gates of latentos.

## Bugs fixed

| file:line | repro | commit |
|---|---|---|
| none | no bug with a repro found | |

## Dead code removed

| symbol or file | lines | proof | commit |
|---|---|---|---|
| `TransferSession._same_chunk` (latentos/transport.mojo) | -19 | body line-for-line equal to module-level `same_encrypted_chunk` (transport.mojo:166-183); one caller, now calls the function | 403b390 |
| dead stores: `bridge.mojo` idx_offsets `text` initialiser, `lora.mojo` fold pre-loop `acc`, `nlook.mojo` bwd `q`, `run.mojo` `argv()[0]` read and loop var `r` | -3 net | compiler warning "was never used" for each; build rc=0 after | dc70c81 |

Census result: no function, constant or struct of the area has zero uses (only false positive: `lops.mojo` `__call__`). The 2026-11-15 leads (`q4k_dot_u`, `gemm_q2b3/tq1/tq2`, `embed_sort_k`, `_opt_field`, `greedy_tokenize`) are outside this area or already gone (`embed_sort_k`: no match in `lops.mojo` any more).

## Debloat

| what | LOC before -> after | commit |
|---|---|---|
| transport.mojo duplicate chunk compare | 519 -> 500 | 403b390 |
| sys.mojo `_fill_sockaddr_un` shared by `sys_bind_unix` and `sys_connect_unix` | 431 -> 426 | 83d0a86 |
| comment fixes (ncm.mojo, ncm_pack.py, args-check.sh): ncm_pack.py path, ncm_dump.py path, stale "run.mojo does not build" | 0 net (comments) | d00e760 |

Total `git diff --stat 2266fa9 HEAD` before this report: 9 files, +14 / -42.
No large artifact in the area: biggest tracked files are `lops.mojo` 114 KB, `run.mojo` 108 KB, `lmodel.mojo` 81 KB (985 KB for the 100 area files without private paths).

## Optimisations

| what | number | commit |
|---|---|---|
| none | no CPU-side hot path with a before/after number in this area | |

## Found, not fixed

| file:line | what | evidence | why not fixed / gate needed |
|---|---|---|---|
| miniagi/latent/lops.mojo:1089,1090,1100,1103,1167,1168,1179,1180,1184,1185 | 10 never-used assignments (`bk bv ad bk bq bd ap ad bd bq`) in GPU kernels | compiler warning "was never used", run.mojo build log | GPU kernels: a codegen change needs the identity gate plus a wavewait capture and a GPU batch |
| miniagi/latent/run.mojo:397 | `var e = 0` initial assignment never used | compiler warning | needs the branch structure rewritten (declare without init); skipped, behaviour-neutral but not a one-token change |
| miniagi/latent/run.mojo:26 | comment names `miniagi/latent/train-resume.sh`, file does not exist (`resume-gate.sh` does) | `git ls-files` | run.mojo is embedded as source in every self-describing ckpt; comment edit changes the embedded bytes, left to the lane that rebuilds the trainer |
| latentos/boot/s1_models.mojo, latentos/sys.mojo:239 | `sys_mount` calls extern `latentos_mount`, the C shim `tools/latent-os/mount_shim.c` is not in this repo (`.mojo-upgrade` lines 12-14: it lives in AMDHQ) | `git grep latentos_mount` | s1_models cannot link from this repo alone; the standalone gate passes only because no test reaches `sys_mount`. Needs a decision: vendor the shim or document it |
| miniagi/latent/{args-check,arith-adapt,ckpt-gate,lr-sweep,mat-gate}.sh, ncm/usage-check.sh | no reference to the script name anywhere else in the repo | `git grep -l` per stem: 0 files | committed manual gates and audit rows, not callers; not deleted (rule: verifiers stay committed) |
| latentos/boot/shadow.mojo:19, boot/s1_models.mojo:29 | identical `log_line` helper in two separate boot binaries | duplicate scan | two binaries, a shared import would couple the boot actors; left |
| latentos/transport.mojo `_write_all` | copies the whole payload list (`data.copy()`) before each send | source | at most 64 KiB per chunk, no number measured; left |
| miniagi/**, ncm/ncm.mojo | about 130 toolchain deprecation warnings (positional `__getitem__`, `bitcast`, `as_c_string_slice`, docstring capital) | run.build.log: 144 warnings in run.mojo build | mechanical, high churn across hot kernels; no behaviour value |

## Cross-area leads

| file:line | what | proof |
|---|---|---|
| tools/gguf_reader.mojo:71,86 | two never-used assignments to `n` | warning in the `ncm.mojo` build (imports `tools/gguf_reader.mojo`) |

## Needs regen

None. No env name, kernel file or setting changed.

## Checks run

| command | result |
|---|---|
| `b.sh bash latentos/tests/standalone.sh` (after the edits) | STANDALONE_REFERENCE_PASS (reference_lifecycle, router_fixture, ipc_small, crypto_smoke, golden_binding, transport_smoke) |
| `b.sh bash latentos/tests/fsm_fixes.sh` | fsm_fixes PASS, fsm_fixes_recv PASS, boot_result PASS |
| `b.sh mojo build miniagi/latent/run.mojo -D LAT_H=8 -I miniagi/latent -I miniagi/train -I kernels` before / after | rc=0 / rc=0; never-used warnings 15 -> 11 |
| `b.sh mojo build miniagi/bridge/bridge.mojo -I miniagi/bridge -I miniagi/latent -I miniagi/train -I kernels` before / after | rc=0 / rc=0; never-used warnings 1 -> 0 |
| `b.sh mojo build ncm/ncm.mojo -I tools -I kernels --target-accelerator gfx1100` (after the comment edit) | built, only deprecation and `gguf_reader.mojo` warnings |
| not run: ncm/check.sh, ncm/closure.sh, miniagi gates | need GPU; the changes there are comments or dead stores that the builds prove |

The first standalone run overlapped my first two edits (it queued behind other workers' builds), so the second run above is the receipt for the final tree.

## Mistakes -> rules

- The scratchpad directory named in the task is shared by all workers of the lane: my first `dead.sh` there was overwritten by another worker's file of the same name (491 B instead of mine). Rule: workers keep scripts in `<worktree>/.work/<area>/`, never in the shared scratchpad.
- First `lora.mojo` edit deleted `var acc = List[Float64]()` but the later `acc = ...` inside the loop needed a declaration (mojo: "implicit declaration of 'acc' is not allowed"). Warning "assignment to X was never used" refers to the FIRST assignment only; rule: after removing a flagged initialiser, check for a later bare assignment and make it the `var`, then build before commit (mojo-dev rules #13 step: read the build log).
- The `b.sh` slot wrapper queues silently when other workers hold the slots: a `run_in_background` job with an empty log is waiting, not failed. Rule: wait on `rc=` in the log with a bounded `until` loop instead of reading the empty log.
- `mcp__hub__run_js` `tools.sh` has a 60 s limit; whole-repo `git grep` censuses must run as a background script with `xargs -P8`.
