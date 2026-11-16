# Brief: rebase, gate and merge four old lanes (2026-11-15)

Boss pane: a lane pane. Model: Sonnet, effort high; Haiku sub-assistant allowed for reads. the maintainer's decision: rebase + gate + merge
each of `lane/anthropic` (worktree ../mojo-baro-lanes/anthropic), `port/densepair`, `port/wiring`, `port/engine-pack`
(worktrees ../mojo-baro-lanes/port-*). Order: anthropic, densepair, wiring (depends on densepair? check), engine-pack.

Per lane:
1. In its worktree: `git rebase main`. Conflicts: resolve keeping main's behaviour and the lane's feature; never drop main's
   changes. A conflict you cannot resolve without a design call -> stop that lane, write why, go to the next.
2. Write `exchange/lane-<NAME>-report.md` (git add -f): what the lane does, commits, checks run with results.
3. `JOBS=2 bench/preflight.sh` in the worktree; then `./run-tests.sh` under `gpu-wait run --timeout 2400` (env inside
   `bash -c`, BARO_TMAX >= 16384). Kernel-changing lanes (densepair, wiring) need a wavewait capture
   (`bench/ww.sh KERNEL ENGINE PACK OUT`, goldens `bench/ww-golden/`) and a 20-prompt `bench/ab-resident.sh` A/B vs main with
   REQ_SPEC=true; numbers go in the report. A speed claim names wavewait/corewait/heaptide and the LACT profile (AI-2500-uv125).
4. `lane-merge <branch>` FROM THE PRIMARY CHECKOUT (~/Projects/mojo/mojo-baro), then `git merge --no-ff <branch>` there. If the
   merge is denied, report READY and stop that lane.
5. After a merge: remove the worktree (copy any ignored receipts/reports into exchange/receipts/<lane>/ first; the gate blocks
   otherwise) and delete the branch.

Rules: graph is the standard (codebase-memory project `home-user-Projects-mojo-mojo-baro`, re-index after each merge);
every mistake -> docs/rules.md + SINDEX lesson-entry (Edit tool); board card on ~/Brain/mojo/mojo-baro/whiteboard.md;
another session runs GPU jobs too: gpu-wait queues, never bypass it. Do not touch serve/window.mojo's uncommitted PGNS
probe in the primary checkout (boss work in flight): commit by pathspec only.

Report: `docs/lanes/2026-11-15-old-lanes-gate-report.md` (per lane: merged / READY-not-merged / stopped + why, numbers), then
`~/iTools/bin/herd tell a lane pane 'DONE old-lanes -> docs/lanes/2026-11-15-old-lanes-gate-report.md'`.
