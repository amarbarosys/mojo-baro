# Graph refresh report (2026-11-15)

Brief: `docs/lanes/2026-11-15-graph-refresh-brief.md`. Read-only on code; NO GPU work.

## Project

- Name: `home-user-Projects-mojo-mojo-baro` (root `$HOME/Projects/mojo/mojo-baro`, branch main). Existed before this run; brief expected none.
- Other `list_projects` rows citing the same root: `mojo-baro` and `mojo-baro-live-plan`. Not touched; duplicate rows to clean up.
- Mode: full, async. Started 2026-11-15T20:51:34Z, succeeded 20:51:36Z.

## Counts

- Nodes 33353, edges 85278. Status ready.
- Doc links: 3293 MENTIONS; unresolved: 206 ambiguous, 226 below_bar_tier, 346 missing, 1 graph_gap, 1 test_only_target.
- Watch: disabled (re-index needed to refresh).

## dattn hits (`search_graph name_pattern=.*dattn.*`, 46 rows)

Code (in `kernels/`, none in `serve/window.mojo`):
- `home-user-Projects-mojo-mojo-baro.kernels.dattn.amar_dattn_exact` (kernels/dattn.mojo:165-192)
- `...kernels.dattn.amar_dattn_split` (kernels/dattn.mojo:483-502)
- `...kernels.dattn.amar_dattn_combine` (kernels/dattn.mojo:508-516)
- `...kernels.dattn.dattn_span` (73-162), `dattn_load_span` (196-231), `dattn_bcast` (235-239), `dattn_step` (243-302), `dattn_nsplit` (306-308), `dattn_split_body` (312-417), `dattn_combine_body` (421-479)
- `...kernels.dattn_harness.launch_dattn` (kernels/dattn_harness.mojo:45-79)
- Modules: `kernels.dattn` (1-517), `kernels.dattn_harness` (1-80), `kernels.test_dattn` (1-131)

Non-code: `bench/bench_dattn.mojo`, `bench/dattn-*.sh`, `bench/dattn-*-protocol.md`, `tools/dattn-ref.py`, `docs/kernel-facts/amar_dattn_{combine,exact,split}.md`, `docs/NEXT-PLAN.md` sections.

## Coverage (`check_index_coverage`, full diagnostics)

| file | status | detail |
|---|---|---|
| serve/window.mojo | no_recorded_issue | no partial/skipped ranges |
| serve/harness.mojo | partial | parse_partial lines 274-274 (re-read those lines before relying on graph) |
| kernels/mega.mojo | no_recorded_issue | no partial/skipped ranges |

Index-wide (`index_status`):
- parse_partial: 51 files (46 omitted from the listing). Samples: bench/ailab/llama.sh:67, bench/ailab/serve.sh:178, bench/bench_turbo_rows.mojo:89, bench/cpu-gemv.mojo:52, bench/draft_dump.mojo:109,206.
- parse_unusable (whole file): 7 files, incl. miniagi/latent/chat_extract.py (untracked), latentos/tests/fixtures/binding.manifest, tests/minja-corpus/lily-cybersecurity-7b-v0.2.jinja, bench/ailab/render-edge/s01-r1.json, s02-r1.json, s03-r2.json, s04-r2.json.
- skipped: 0.
- Caveat: coverage is best-effort; "no issue" does not prove completeness.
