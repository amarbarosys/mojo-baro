# code-fire report (2026-11-15)

**Two of five frozen bars missed: the network fires deterministically in 60 s, but its dead-code list and bug ranking are limited by edges the graph does not have.** Dead code covers 23 of the oracle's 95 Mojo functions (24.2%, bar 90%); a held-out bug site reaches the top 50 for 1 of 5 sites (bar 3 of 5), though the random control did worse in all 5.

Brief `2026-11-15-code-fire-brief.md` (plan + addendum). Frozen bars: Brain `~/Brain/mojo/mojo-baro/2026-11-15-code-fire.md`
(66983f1e7, before the first fire). iTools commits: df9b68a (step 1), b5d7967 (2), 4101ad8 (3), 69e56eb (4), step 5 below.
Receipts: `exchange/receipts/code-fire/` (outputs live in `.work/code-fire/`, which is not storage). No mojo-baro code changed.

## Verdict against the frozen bars

| # | Bar | Result |
|---|---|---|
| 1 | Determinism, two runs `cmp`-identical | **MET**, 4 of 4 reports (`determinism.txt`) |
| 2 | Dead code >= 90% of the oracle's Mojo functions outside tests; every extra classed; planted function appears | coverage **MISSED 23 of 95 = 24.2%**; planted **MET** (`plant.out`); extras classed by text grep, only part of the 13 "truly dead" read by hand: **UNVERIFIED** as "every extra read" |
| 3 | Held-out bug site in top 50 for >= 3 of 5, control worse | **MISSED 1 of 5** (`attn_phases` rank 3); control worse in 5 of 5 (met) |
| 4 | Layer A links non-empty, known pair, GDScript unchanged | **MET** (0 -> 4186 call links, `attn_phases` -> `dattn_split_body`, street-sim GDScript links 3572 `cmp`-identical) |
| 5 | One full fire under 5 min on CPU | **MET** 3502 spreads in 59.9 to 65.7 s (16 jobs, `brainnet spread`) |

Missed bars are reported, not re-cut. Typed synapse weights were not tried (frozen: only as a follow-up after a miss).

## Findings

- **Dead code: the graph is the limit, not the walk.** `dead.tsv` lists 450 Function/Method with no directed path from an entry
  seed (tests, tools/, bench/, is_entry). Of the oracle's 72 misses, 14 are tools/bench entries by my seed rule and 58 are
  Methods under a live class (lifecycle `__init__`/`__deinit__` have no CALLS edge, so the oracle's degree 0 is wrong there, not
  `dead.tsv`). Of 410 extras: **319 false positive** (name referenced in source, graph edge missing: method calls on a value,
  comptime arguments), **78 launched by string**, **13 truly dead by grep** (7 Mojo, 4 C shim, 2 shell), 0 test-only. Of the
  7 Mojo: `amarbaro_light`, `embed_sort_k`, `_opt_field`, `q4k_dot_u` are definition-only; `gemm_q2b3/tq1/tq2` have two more text
  mentions I did not read. The classifier is a grep of the name: ambiguous names inflate "referenced" (62 FP and 54 string hits are
  names defined more than once). Use `dead.tsv` as a lead list, not a delete list.
- **Bug candidates: a weak signal.** Held-out rank real/control: `attn_phases` 3/144, `amar_rmsnorm_cast2` 365/2299,
  `amar_turbo_rows` 349/1106, `save_state` 236/368, `amar_ssm_delta_rows` 1198/2426 (4337 candidates). The network beats random on
  every site but ranks only the megakernel attention site high, and `attn_phases` is also a most-linked, high-churn symbol, so
  its rank 3 partly reflects centrality. Top candidates now: `serve.engine.pair_admit`, `kernels.mega.ssm_phases`,
  `kernels.mega_moe_rows.mega_moe_body`, `serve.q4x.Q4X`. `mega_moe_body` (671 lines) and `ssm_phases` sit in the Pg-partials
  nondeterminism neighbourhood: a pointer, not evidence.
- **Hot / duplicates:** one cost seed only, so `hot.tsv` is the neighbourhood of `amar_mega_token` (top: `step_window`,
  `test_mega_block.run_case`, `mega_body`). `dup-clusters.tsv`: 115 SIMILAR_TO clusters, mostly duplicated test/bench helpers
  (`view_bf16` x9, `greedy_tokenize` x4, `cleanup` x4); no kernel duplicate worth a lane. More goldens add seeds.

## What was built

- Layer A fix: `code-links.sh` matched only `func NAME(`; a no-match `git grep` also killed it silently on any non-Godot repo.
- Layer B `code-graph-net build`: 6965 symbols (Function 4397, Method 1412, Class 506, Struct 94, .mojo/.py Module 556), 18023
  synapses: CALLS 12165, USAGE 3343, DEFINES_METHOD 1380, IMPORTS 807, LAUNCH 645, SIMILAR_TO 223, SEMANTICALLY_RELATED 340, TESTS 65,
  OVERRIDE 32. Graph gap closed: `serve.window.step_window -> amar_mega_window` (0 callers before); comptime aliases are global
  (`comptime a = K[...]` in registry.mojo, launched elsewhere as `registry.a`). 9836 links: lives-in to layer A chunks (line
  spans from the chunker's word count) and note -> symbol. Build 39 s, embed on the CPU (30 s).
- Addendum, note -> symbol: backticked names in SINDEX, `docs/rules.md`, Brain `mojo/mojo-baro/*.md`: **249 resolved by exact name,
  104 by unique suffix, 92 ambiguous (skipped), 1482 unknown** (distinct note x name pairs; unknowns are flags, files, env vars,
  prose). 259 symbols carry a note link; 366 carry a fix commit (blame, 90 days).
- Signals (`signals.sh`, 1 s): churn>0 2360, fixes>0 366, notes>0 259, tests>0 1260, cost>0 **1** (one wavewait golden, `amar_mega_token`).
- `code-fire run|risk|eval|dead-check|plant`.
- Merged network: layer B joins the Brain network when `refresh.sh` next runs (rate limit, trailing run pending at report time);
  code-fire reads the layer B store directly, so the reports do not depend on it.

## Deviations (decided before the runs, none after a result)

1. `code-links`: Mojo/Python keep same-file calls, GDScript stays cross-file only (else street-sim cannot stay `cmp`-identical); street-sim total grew 3572 -> 3877 from its Mojo sidecar (0 removed).
2. Embed on the CPU (`newera .work/bge/bge`), not GPU: the brief guardrail said NO GPU, the embed is 30 s.
3. `dead.tsv` = directed reachability, not spread zero: undirected spread reaches a dead caller through its live helper.
4. Site 3 (KQ dead-branch prologue drift) has no text in repo, rules, notes or commit bodies; replaced by `save_state` (SSM state of another prompt, 511cdb4). `heldout-sites.tsv` says so.
5. Hiding a held-out site also hides the symbols of notes naming it that name <= 8 symbols (lesson-sized); whiteboard/SINDEX/rules stay (hiding them would erase the seed set). The code-fire Brain note names the sites by design and is excluded from the link builder.
6. Candidate score = activation x sim x (1 + churn) / (1 + tests); sim = 1 + (bug sites among the 10 nearest embedding neighbours, cos >= 0.6)/10 (`brainnet knn` prints pairs, not cosines); tests are not candidates.
7. IMPORTS edges start at File nodes in this graph; mapped to the file's Module neuron.

## Mistakes -> rules (`docs/rules.md`, SINDEX tooling)

Four silent exits from one cause (`set -e` + pipefail: no-match grep, head/awk closing a pipe); `brainnet knn` empty because every
neuron shares the first path component; the bars note would have leaked the held-out sites into note links; the dead classifier
scanned only .mojo/.py while the graph holds .rs/.js/.c ("truly dead" 90 -> 13).

## Next

- Add wavewait goldens for the hot kernels (cost seeds are 1) and re-fire; try typed weights only after that (follow-up, not a bar).
- Feed the truly-dead Mojo leads to a cleanup pass; fix the graph's missing method-call edges before trusting the other 319.
