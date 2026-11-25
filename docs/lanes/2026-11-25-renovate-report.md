# Lane renovate report (2026-11-25)

**Ten Sonnet workers found and fixed 25 bugs with a repro, and the tree lost 932 net lines at the merge point; 18 of the
bugs were gates that could pass on bad input. Nothing was merged to main.** Branch `lane-renovate`, base `2266fa9` (main `cf1af3e` + brief),
69 commits. Brief: `docs/lanes/2026-11-25-renovate-brief.md`. Per-area detail: `docs/lanes/renovate/<area>.md` (10 files).

## Proof

| check | base (2266fa9) | merged tree |
|---|---|---|
| `bench/preflight.sh` | ci-checks, rdna4-probe, 43 builds PASS; cargo 150 pass / 4 fail | `PASS preflight 6a5f0af3319a` at `f1a74fb`; run again after the two report commits (docs only), stamp in the Brain note |
| cargo tests | 154 pass after the fixture repair (below) | 155 pass, 0 fail (+1 new test), second suite 24 pass |
| `./run-tests.sh` under gpu-wait (GPU, 22 GB, `STATE_ENGINE` = the lane's engine-dense) | not run on the base | rc 0, `PASS run-tests: all tests, pack .work/engine-pack-q4`, FAIL 0 |

- The base was red for an environment reason, not code: main's `.work/ailab/pack-kq` pointed into a pack cache dir that had been
  replaced that night. The boss repointed the link; the same 4 tests then passed with no change. Receipt `.work/renovate/base-receipt.md`.
- The GPU suite ran once, on the merged tree before the report commits (docs only after it). Two skips, both environment:
  `test_kq_prefill_gemm` (no kq pack at main's `.work/ornith-stint/pack-kq`) and the real-decode row of one test
  (`.work/draft-logits.bin` absent). The suite does not count the first as INCOMPLETE.
- No perf claim on any kernel or engine path, so no A/B and no wavewait capture: the kernel and engine edits are deletions
  of unused names, comment removal, and one hash helper swap with a 300-length digest sweep.
- Not gated on the GPU: the 18 bench gate fixes were shown on CPU stubs (before/after) by the worker; no real-engine gate run.

## Numbers

| | before | after |
|---|---|---|
| lines, whole diff | | +1383 / -2315 at the merge point (149 files) |
| lines, code only (no docs, no testdata) | | +515 / -1659 (110 files) |
| duplicate testdata removed | 13 files, 592 lines | 0 |
| tracked bytes | 75,911,668 | 75,917,830 at the merge point (+6,162: ten area reports outweigh the deletions) |
| kernel census | 183 kernels, 68 in registry | 183 kernels, 68 in registry, 0 orphans |
| engine build warnings (dense) | 292 | 290 |

## Per area

| area | bugs fixed (repro in the area report) | dead code and debloat | optimise |
|---|---|---|---|
| serve-front | `exact_prefix_hash` panic when `pos` is past the token list (`07484f3`, test fails then passes) | byte-copy dir `serve/testdata/anthropic` (`3471e88`), test helpers into one `test_net.mojo` (`0d35a75`), 1 unused import | none |
| engine-host | none with a repro | duplicate SHA-256 in `prefix.mojo` (`d58013c`, -99 lines), dead `_dequantize_kv_int8` (`2d0dc2d`), unwired ternary wrappers (`df5dffa`, -69), `sample_ref` shared helper (`79b0166`, 9600-call A/B, 0 mismatches), 2 aliases, 12 imports | none |
| kernels | none | 2 unused constants, 35 unused import names (`4aae1a8`); comments out of `matmul*.mojo` and `elementwise.mojo` (`ac318c9`, -160 lines) | none |
| bench | 18 gate bugs (`4d98770` to `2296c7d`): gates that printed PASS on 0 cases, ignored a failed build and timed the stale binary, lost their FAIL line to a zero-count grep, or returned the killed server's status | 46 unused import names in 17 harness files (`b606b23`), unread `fr-gen-prompts.txt` | none |
| tools | `merge-gate.sh` built `test_prefill` without `-I serve` (`ee1d1c9`) | 2 Python helpers, 2 itools links, 2 zluda probes | none |
| latentos-miniagi | none | duplicate chunk compare and sockaddr filler in latentos (`c581de9`, `1994c90`), 4 dead stores | none |
| embed-face | `.5`, `inf`, `NaN` written as bare TOML corrupted the user presets file and stopped the face (`d7ce193`) | none found | none |
| parsers | surrogate-pair `\u` decode, control bytes in enum/const/key literals, default `maxItems` below `minItems` (`2b5e6ca`, `ccaeb43`, `d69be14`; new vocab-free test `905073a`: 5 FAIL before, PASS after) | 7 unused imports | `load_vocab` about 9% faster, same checksum (`597e5cc`) |
| platform | none provable | 37 stale file:line cites in `skills/mojo-baro-operate` replaced by section cites (`b26b8cc`); engine_cpu build warnings 1 to 0, binary identical | none |
| docs-ci | `ci-checks.sh` checked protocol rules P1-P20 of 34 (`bb47139`); clone and discussion links named a GitHub org that returns 404 (`6171d1b`, `gh repo view amarbarosys/mojo-baro` is PUBLIC) | dead links 37 to 31, four claims in METHOD the tree contradicts, stale README and CONTRIBUTING rows | none |

## Orchestrator decisions on worker output

- **Held out, not merged:** `ngram_propose` rewritten as the first branch of `ngram_propose_tree` (worker commit `6ef2f7d`). It sits
  on the speculative decode path, the worker did not rebuild the engine after it, and it saves 9 lines.
- **Merged in part:** the kernels worker stripped comments from 22 kernel files (307 lines). The repo rule names only
  `kernels/matmul*.mojo` and `kernels/elementwise.mojo`; the boss confirmed that scope. The other 14 files keep their comments.
- **Reverted:** the removal of `bench/moe-gate-resident.sh` (`9db6a2e`). The frozen `bench/moe-protocol.md` cites it and
  `tools/ci-checks.sh` failed on the dangling path.
- **Regenerated after the merges:** `docs/KERNELS.md` census (`6a62737`), the ternary paragraph of `docs/CAPABILITIES.md` (`f1a74fb`).
  The body of `6a62737` names CAPABILITIES.md but changed only KERNELS.md; `f1a74fb` says so.

## Found, not fixed (the highest-value items; the full lists are in the area reports)

| where | what | gate it needs |
|---|---|---|
| `kernels/mega_moe_rows.mojo` vs `kernels/mega_rows.mojo` | `rms_rows`, `f32_dot_rows`, `q8_dot_rows`, `pick` are the same source twice (about 110 lines) | ISA diff + megakernel identity + same-job A/B (code placement changes speed on this card) |
| `kernels/test_mega_block.mojo:423` | does not build (capture convention), and no suite builds it | fix + one GPU run of the test, then add it to `run-tests.sh` |
| grammar tests | not in `run-tests.sh` or the preflight | add the build lines; CPU only |
| `grammar/regex.mojo` used by `compile_string` | `.` and negated classes accept `"` and raw control bytes inside a JSON string: invalid JSON from a valid schema pattern (read from code, not run) | a schema case through the engine |
| `minja/render.mojo:461` | floats truncated to Int in comparisons and arithmetic | `tools/minja-gate.sh` on the corpus |
| `serve/shard_index.mojo:103`, `serve/expert_tier.mojo:85` | fd never closed after `mmap` (bounded by shard count) | batch with a GPU run |
| `face/chat.mojo` | the Sampling tab changes nothing in a chat request (`sampling_json` has no caller); an empty assistant turn stays in history after a failed request | design call + one request on a live front |
| `face/app.mojo` | `startup_staged` (Balanced as the startup default) is never called | offscreen face gates |
| `tools/profile.py` | the Python oracle fails 4 profiles the Mojo port passes | decide which side is right |
| `.github/workflows/checks.yml:20` | CI installs `mojo==1.0.0`, the repo pins 1.2.0 | a CI run |
| `bench/ab-resident.sh` | header says exit 1 on any FAIL, the script exits 0 on identity fails | semantics decision |
| `bench/moe-gate-resident.sh`, `bench/moe-launch-count.sh:32` | need a lane tree or a pack path that is gone | current MoE pack path + GPU run |
| `tools/merge-gate.sh`, `test_server.sh`, `test_pool.sh`, `mega-gate.sh` | engine build lines lack `--target-accelerator gfx1100` | one gate pass to confirm the default target |
| `run-tests.sh` state layer | the external latentos check printed a compile error for one test and still reported it as passed (`LatentHeader` has no `dtype_name`); present before this lane | fix in the external check |
| `run-tests.sh:46` | a missing kq pack skips `test_kq_prefill_gemm` without the INCOMPLETE exit 3 | decide if the skip counts |
| deprecations | 118 of 290 engine build warnings are deprecated API in hot host loops | one batch with an engine identity run |
| zero-reference tools | `tools/baro-build`, `tools/llama_oracle.mojo`, 5 standalone probes, 5 parked draft-head training scripts (44,809 bytes), `serve/moe_gate1.mojo`, `kernels/bench_h2d.mojo`, `serve/src/handoff.rs` | owner decision; none is provably dead |
| `results/e8/` | 8.4 MB of per-batch JSON beside the merged files | receipt retention decision |
| vendored, report only | `imkit/`, `uregex/`, `toml/`, `bento/`: unused imports and PNGs listed in the area reports | upstream |

## Skills, ledger and tools changed (brief addendum)

- **iTools** `7283b2f`: `lane-prep` fails on a dangling link inside a fixture dir, two levels deep (header and `tool.toml`
  updated, `index-gen --write`). Test: clean fixture PASS rc 0, planted link FAIL rc 1.
- **Skills** (Brain `2886d3c08`): `lanes/rules.md` rules 63 to 65 (deletion briefs name the reference check and frozen records;
  build throttles use slots, not load average; Workflow workers start with one setup script and prefixed scratch files);
  `lanes/SKILL.md` rule count. In the repo: `skills/mojo-baro-operate` (`b26b8cc`).
- **Ledger** `~/Brain/SINDEX/mojo-baro.md`, three entries dated 2026-11-25: a dependent commit ran after its Edit was denied;
  a worker deleted a script a frozen protocol cites; a load-average throttle starved the workers. Same lines in `docs/rules.md`.

## How it ran

One Workflow (`wf_acc08fc3-d5a`): 10 Sonnet workers at high effort, each in its own worktree on `renovate/<area>`, no GPU,
builds through a 2-slot wrapper. 2.0 M worker tokens, 1146 tool calls, 77 minutes. All ten branches came back clean, with
no self-tag lines. The orchestrator read the behaviour-changing diffs, cherry-picked 63 worker commits and added 6.

## Next

Merge to main waits for the maintainer's go: `lane-merge --report docs/lanes/2026-11-25-renovate-report.md lane-renovate` from the
primary checkout, then `git merge --no-ff`. The worker worktrees under main's `.claude/worktrees/wf_acc08fc3-d5a-*` and the
`renovate/*` branches can go after that.
