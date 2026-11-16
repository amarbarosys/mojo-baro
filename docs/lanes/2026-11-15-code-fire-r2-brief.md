# Brief: code-fire round 2, better synapses, act on findings in mojo-baro (2026-11-15)

Boss pane: a lane pane. Model: Sonnet, effort high; Haiku sub-assistant for reads. Round 1: `docs/lanes/2026-11-15-code-fire-report.md`
(bars 1, 4, 5 MET; dead coverage 24.2% and held-out bugs 1 of 5 MISSED). Tools: `~/iTools/brain/{repo-net,code-graph-net,code-fire}`.
Bars are the SAME as round 1 (Brain `~/Brain/mojo/mojo-baro/2026-11-15-code-fire.md`), never re-cut; write a "round 2" section
there BEFORE the first round-2 fire, naming what changed.

## A. Synapses the graph misses (iTools, code-graph-net; each its own commit)
1. Name-reference edges: a symbol whose name appears as a word in another symbol's body (method call on a value, comptime
   argument `f[k]`, `comptime k = NAME[...]`) -> edge, resolved like the note links (exact, unique suffix; ambiguous skipped and counted).
   Read bodies from the repo by the graph's line ranges; one ripgrep pass, never a loop of greps.
2. Lifecycle and dunder methods (`__init__`, `__deinit__`, `__moveinit__`, `write_to`, trait methods) count as entry points.
3. Check: round-1 false positives (`exchange/receipts/code-fire/dead-check.tsv`, 319 "name referenced") drop; dead coverage
   re-measured against the same oracle; planted function still found.

## B. Cost seeds (hot.tsv had 1 seed)
- Cost from what exists first: `bench/ww-golden/*`, `tools/isa-receipt.py` spills/VGPR per kernel, `docs/kernel-facts/*.md`.
- Then wavewait captures for the top 8 kernels by decode time that lack a golden: `bench/ww.sh KERNEL ENGINE PACK OUT` under
  `gpu-wait run --timeout 600` (short runs, env inside bash -c). Re-fire; hot.tsv must name >= 8 cost seeds.

## C. Act on findings in mojo-baro (lane worktree: `git worktree add -b lane-codefire2 ../mojo-baro-lanes/codefire2 main && lane-prep ...`)
1. Dead code: for each "truly dead" Mojo symbol (round 1: amarbaro_light, embed_sort_k, _opt_field, q4k_dot_u, gemm_q2b3/tq1/tq2,
   plus any new ones), read it, check with the graph (`trace_path` inbound, `search_code` for string launches) and grep; delete
   only when nothing references it. Kernel files carry zero comments. Commit body: loc before/after.
2. Duplicates (`dup-clusters.tsv`: view_bf16 x9, greedy_tokenize x4, cleanup x4, ...): one shared helper imported where
   the copies were, only where the bodies are byte-equal or differ by a parameter; test/bench helpers only, no kernel change.
3. Gate: `JOBS=2 bench/preflight.sh` PASS, `./run-tests.sh` under gpu-wait green, `lane-merge lane-codefire2` from the
   primary checkout, merge if READY (if denied, report READY and stop).

## D. Re-fire and report
Held-out bug eval and dead-check again with A+B; report the delta vs round 1 per bar. Typed synapse weights only if A+B
leave bar 3 missed, as a follow-up arm with its own control.

Rules: graph is the standard (project `home-user-Projects-mojo-mojo-baro`; re-index after the merge). Do not touch
`serve/window.mojo` (boss nondeterminism probe). Every mistake -> docs/rules.md + SINDEX lesson-entry (Edit tool). Board card
on ~/Brain/mojo/mojo-baro/whiteboard.md. Report `docs/lanes/2026-11-15-code-fire-r2-report.md`, then
`~/iTools/bin/herd tell a lane pane 'DONE code-fire-r2 -> docs/lanes/2026-11-15-code-fire-r2-report.md'`.

## Change (the maintainer, 2026-11-15): part C is taken by the boss
Skip C entirely (no lane-codefire2 worktree, no mojo-baro edits). Do A, B, D; D re-fires against main after the boss merges C.
