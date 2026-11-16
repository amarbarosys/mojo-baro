# Brief: refresh the codebase-memory graph of mojo-baro (2026-11-15)

Boss pane: a lane pane. Model: Haiku, effort low. Read-only on code; no commits needed.

1. `list_projects`: find the project whose root_path is `$HOME/Projects/mojo/mojo-baro` (none -> it is new).
2. `index_repository` on `$HOME/Projects/mojo/mojo-baro` (full refresh; main changed today: serve/harness.mojo BARO_POISON,
   serve/window.mojo, docs, lane-retarget merge 14bc4e5).
3. `index_status` until done; then check: `search_graph(name_pattern=".*dattn.*")` returns functions in `serve/window.mojo` or `kernels/`,
   and `check_index_coverage` for `serve/window.mojo`, `serve/harness.mojo`, `kernels/mega.mojo`.
4. Write `docs/lanes/2026-11-15-graph-refresh-report.md`: project name, node/edge counts, the dattn hits (qualified names), coverage result
   per file (any partial/skipped ranges listed).
5. Report: `~/iTools/bin/herd tell a lane pane 'DONE graph-refresh -> docs/lanes/2026-11-15-graph-refresh-report.md'`, then stop.
