# Brief: away-run Sonnet lane (2026-11-15)

Boss pane: a lane pane. Report file: `docs/lanes/2026-11-15-away-sonnet-report.md`. When a job ends:
`~/iTools/bin/herd tell a lane pane "DONE <job> -> docs/lanes/2026-11-15-away-sonnet-report.md"`.

## Before any code (mandatory)
- Load skills: mojo-syntax, mojo-lean, mojo-gpu-fundamentals, memory-optimization, kernel-parity, inference-kernels, gpu-lane-budget, mojo-nightly-lane-builder (Skill tool / `~/.claude/SKILLDEX/SKILLDEX.md`).
- Read: repo `CLAUDE.md`, `docs/rules.md`, `~/Brain/SINDEX/mojo-baro.md` (last 40 entries), `bench/PROTOCOL-RULES.md` P1, P4, P27.
- Work in a worktree: `git worktree add -b lane-away-sonnet ../mojo-baro-lanes/away-sonnet main && lane-prep ../mojo-baro-lanes/away-sonnet`, never the primary checkout.
- Batch reads and searches through the hub toolscript `run_js` (CLAUDE.md §4); `brain-ask` before any theory; `code-reuse find` before new code.

## Team
- You may run one Haiku sub-assistant (herdr pane or Agent tool, `model haiku`, effort low) for reads, greps, log extraction. You direct it; its output is a file you check.
- No other workers. Stay cheap: smallest run that answers, CPU first, every GPU job `gpu-wait run --timeout S` with env inside `bash -c`.

## Job 1: FSM rows not exercised (docs/state-machines.md, section "Status 2026-11-10")
46 rows have code; these were never exercised by a run. For each: a reproduction on the CPU (fake/stub engine) or a short GPU job, test committed, row status updated in `docs/state-machines.md`:
1. live non-zero `fail` word -> HTTP 502 (B-M1): find or add a test-only knob that forces it.
2. KV pool exhaustion -> 429 (B-M7).
3. `captured` (B-M13), `grejected` on a served grammar (C6), B-M4, B-M10, spark half of B-M14.
4. C15 shim check: run it on the card.
5. A-M13: text between tool calls is dropped (partial); fix + cargo test.
B-M2's probe cannot fail on the pre-fix engine: write a probe that can, or mark it NOT TESTABLE with the reason.

## Job 2: Qwen2.5-0.5B emits token 0 (board card, BASELINE.md "0.5B" rows)
Per-layer comparison on ONE prompt against `llama-oracle.py` (or the f64 forward) to find the first divergent layer/op. Report the layer, op and evidence; fix only if XS/S and the spark agreement gate on the other spark models stays equal. Anything beyond: write it up, stop.

## Rules
- Every mistake (yours, a tool's, Haiku's): `what broke -> the rule` in `docs/rules.md` + a lesson-entry in `~/Brain/SINDEX/mojo-baro.md` (Edit tool, 5 fields), same turn.
- Commit per concern on the lane branch, conventional subject + why body, no attribution lines, no push. Merge to main only with `lane-merge lane-away-sonnet` passing.
- `BARO_TMAX >= 16384` on every engine run. Read back every arm parameter from the log.
- Board: tick/annotate the matching card in `~/Brain/mojo/mojo-baro/whiteboard.md` the same turn a job ends; dated Brain note per job.
- Housekeeping at the end: remove scratch files you made, close the Haiku pane, `lane-status .`, sweep orphan processes.
- Stuck after 2 tries on one item: write what you saw to the report, skip to the next.

## Job 3 (after job 2): port `tools/q4x-gtt-trace.py` to Mojo
Python stays as the oracle in `tools/`; Mojo output byte-identical on a recorded input (mojo-port skill: `mojo-port verify`). Commit both, the gate script, and the board card line 96 note.

## Job 4: step 0c byte test (board card at line ~189, "[BVR 2026-10-12 ... step 0c")
One dot product: gfx1100 `warp.sum` order against a CPU 32-lane tree in the same order; byte test file in `kernels/`, in `run-tests.sh` if it is GPU. Report the bits that differ, if any.
