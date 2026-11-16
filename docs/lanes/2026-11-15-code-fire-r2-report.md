# code-fire round 2 report (2026-11-15)

**Better synapses cut the dead list's false positives by 73% and tripled its truly-dead Mojo leads, but both missed bars stay missed: dead coverage against the oracle fell to 11.7% (round 1: 24.2%) and held-out bugs stay at 1 of 5.** Part C (acting on the findings in mojo-baro) moved to the boss, so no mojo-baro code changed and D runs again after the boss merges C. Part B.2 (new wavewait captures) was not run: the brief's NO GPU guardrail wins.

Brief `docs/lanes/2026-11-15-code-fire-r2-brief.md` (with its 032fd18 change). Bars are the round-1 bars, unchanged; the round-2 design was frozen
before the first fire in Brain `~/Brain/mojo/mojo-baro/2026-11-15-code-fire.md` (53a3e493c). iTools commits: 682ee38 (round-1 classifier
fix, was uncommitted), 59781cc (A1 REF edges), a20c98b (A2 dead walk), 1801453 (B cost seeds), bef53c1 (plant fix).
Receipts: `exchange/receipts/code-fire-r2/` (small text, copied from `.work/code-fire/`). Round-1 receipts: `exchange/receipts/code-fire/`.

## Delta per bar

| # | Bar | Round 1 | Round 2 |
|---|---|---|---|
| 1 | Determinism, two runs `cmp`-identical | MET 4 of 4 | **MET** 4 of 4 (`determinism-r2.txt`) |
| 2 | Dead >= 90% of the oracle's Mojo functions outside tests; every extra classed; planted function found | coverage 23 of 95 = 24.2% MISSED; planted MET | coverage **11 of 94 = 11.7% MISSED**; planted **MET** (`plant-r2.out`); extras classed by grep, not each read by hand: UNVERIFIED as "every extra read" |
| 3 | Held-out bug in the top 50 for >= 3 of 5, control worse | 1 of 5 MISSED, control worse 5 of 5 | **1 of 5 MISSED**, control worse 5 of 5 |
| 4 | Layer A links, known pair, GDScript unchanged | MET | not touched, not re-run (links 9836, same store build) |
| 5 | One fire under 5 min on CPU | MET 60 to 66 s | **MET** 3510 spreads in 72.8 to 81.6 s (more edges, slower walk) |

## A. Synapses

- **REF edges** (A1, `code-graph-net`): a word of a symbol's own lines that names another symbol. Edges 18023 -> 22544 (REF 14362: 14281 unique,
  81 resolved by a `prefix.word` penultimate match; 6076 ambiguous pairs over 291 names, listed in `edges.amb.tsv`, no REF edge). One perl pass,
  each line owned by its innermost symbol. The first sweep, with no kind or import filter, gave 37k REF with `i`, `f`, `range` as 1000-edge
  hubs; found by reading the top targets by in-degree before any fire, fixed by the filters (methods only after a `.`, bare words only to
  same-file or imported Function/Class/Struct, `builtins.*` never).
- **Dead walk** (A2, `code-fire`): REF edges and the ambiguous edges count as reachability (a delete list errs to keep). Deviation from
  "lifecycle methods count as entry points": class -> method only for lifecycle names (dunders, `write_to`, `write_repr_to`, `copy`), because a
  seed would make a dead class live through its own `__init__`. A live class no longer makes its other methods live.
- **A3 check** against round 1's classified extras: of the **319** "name referenced" false positives, **232 left `dead.tsv`**; the 13 truly-dead
  leads all stay; the planted function is found.

| dead-check extras (beyond the oracle) | round 1 | round 2 |
|---|---|---|
| `dead.tsv` rows | 450 | 200 |
| false positive (name referenced) | 319 | 121 (24 of them names defined more than once) |
| launched by string | 78 | 21 |
| truly dead by grep | 13 (7 Mojo) | 34 (18 Mojo, 16 C/shell/other) |

The 34 are in `truly-dead.tsv`. Mojo: `amarbaro_light`, `embed_sort_k`, `_opt_field`, `q4k_dot_u`, `gemm_q2b3/tq1/tq2` (round 1) plus
`Vocab.bytes_of`, `Ui.caption/scroll_row/set_scale/table_begin`, `Json.opt_f/opt_s`, `TokenKind.WHITESPACE`, `PatternT.findall_cps/findall_groups/group_index`.
Classifier: a grep of the name (no reference outside the definition); lead list for the boss's part C, not a delete list.

## Why coverage fell (the bar was always going to miss)

The oracle (`search_graph max_degree=0`) counts CALLS edges only. Of its 83 misses, by cause (`oracle-miss-classes.txt`):

| cause | count |
|---|---|
| lifecycle method of a live class (`__init__`, `__deinit__`, ...): reachable by design | 73 |
| unique REF in-edge (a method named in reachable code) | 6 |
| tools/bench entry by my seed rule | 4 |

REF edges and the conservative ambiguous edges can only remove entries from `dead.tsv`, so the raw coverage had to fall. Not a bar: against the
21 oracle functions that are not lifecycle methods, `dead.tsv` holds 11 (52%). The 6 REF-reached methods are not proven live: a unique method
name used on a value of another type (a `Dict.items` call matching `Json.items`) counts as a reference.

## Held-out bugs (bar 3, unchanged)

| site | round 1 real / control | round 2 real / control |
|---|---|---|
| `attn_phases` | 3 / 144 | **2** / 54 |
| `amar_rmsnorm_cast2` | 365 / 2299 | 131 / 1689 |
| `amar_turbo_rows` | 349 / 1106 | 467 / 1042 |
| `save_state` | 236 / 368 | 591 / 776 |
| `amar_ssm_delta_rows` | 1198 / 2426 | 1367 / 2776 |

Candidates 4337 -> 4766. Two sites improved, three got worse: the denser graph spreads activation further. The control beats the real run
nowhere, but `attn_phases` control 54 shows how central that site is. Top candidates now: `kernels.mega.ssm_phases`, `serve.engine.pair_admit`,
`kernels.mega_moe_rows.mega_moe_body`, `serve.q4x.Q4X`: pointers, not evidence.

## B. Cost seeds

`hot.tsv` seeds 1 -> **10** (brief: >= 8) without a GPU run: wavewait cycles from `bench/ww-golden` plus three earlier captures
(`ww-cycles.tsv`: `amar_mega_token`, `amar_matmul_skinny_q4rowb`, `amar_turbo_rows`), and ISA pressure from `docs/kernel-facts` (vgpr >= 128
or a spill: 37 symbols carry a kernel-facts ISA line; with the 3 cycle captures, 10 pass the seed rule). Weight = max(cycles / largest, isa / 512), a rank, not a unit.
Seeds: `amar_attn_prefill_wmma`, `amar_dattn_split`, `amar_matmul_skinny_q4rowb`, `amar_mega_token`, `amar_mega_window`, `amar_moe_down_q4k_dot`,
`amar_topn_probs`, `amar_ssm_delta_chunk_w`, `amar_ssm_delta_step`, `amar_turbo_rows`. **Not done: B.2.** Eight new wavewait captures need the GPU,
the brief's guardrail says NO GPU work, so the top-8-by-decode-time captures are open and the hot ranking is ISA-led (top: `step_window`,
`run_case`, `mega_body`). A GPU round can take them with `bench/ww.sh` under `gpu-wait run --timeout 600`.

## Not done / follow-ups

- **D after C:** re-fire after the boss merges C: `CGN_NOREFRESH=1 ~/iTools/brain/code-graph-net/code-graph-net.sh build ~/Projects/mojo/mojo-baro &&
  ~/iTools/brain/code-graph-net/signals.sh . && ~/iTools/brain/code-fire/code-fire.sh run . && ... dead-check . && ... eval .` (the build re-indexes the graph).
  This report is the pre-C state of main (HEAD 032fd18).
- **Typed weights:** bar 3 is missed, so the follow-up arm is open. It needs weighted edges in `brainnet graph` (weight 1 today); a cheaper
  first arm is an edge-type ablation (drop REF, drop SIMILAR_TO) with its own control. Not built.
- **Merged network:** these runs used `CGN_NOREFRESH=1`; the Brain network gets the new layer when `refresh.sh` next runs.
- mojo-baro files touched: this report, `exchange/receipts/code-fire-r2/`, `docs/rules.md` (3 rules). No code.

## Mistakes -> rules (`docs/rules.md`, SINDEX `mojo-baro`)

REF edges built without filters made hubs (caught by the in-degree read before the fire); `plant` failed silently on a cross-device clone
(caught by `bash -x`); a coverage bar against an oracle wrong by class cannot be met by a better tool.
