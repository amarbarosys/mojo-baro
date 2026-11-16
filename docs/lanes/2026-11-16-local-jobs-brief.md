# Brief: local models on real mojo-baro jobs (overnight, 2026-11-16)

Boss pane: a lane pane. Report: `docs/lanes/2026-11-16-local-jobs-report.md`. Model: Sonnet high; Haiku sub-assistant
(split pane) for reads and log digests. Per job: `~/iTools/bin/herd tell a lane pane "DONE local-jobs <job> -> <file> | ..."`.

## Engines (our own, served through gpu-wait; one at a time, the GPU is shared with timed lanes)
- Forge-gemma4 E2B: `serve/gemma4.mojo` via baro-serve (lane-gemma4-port, merging; `bench/gemma4-chat.sh` shows the
  launch). Small: fits beside other jobs only if gpu-wait admits it.
- nerkyor Qwen3.8-27B coder (q4 pack from lane-nerkyor-vs-forge) once its S3 lands; Qwythos-9B q4 (`.work/engine-pack-q4`).
- Serve with BARO_TMAX auto (0405c80), max_tokens >= 16000, record finish reasons. Never hold the GPU idle: start,
  run the batch, stop (rules 17-19: record PIDs, setsid, kill by anchored exe).

## Jobs (the maintainer's picks)
J1 Reviewers/finders: feed one kernel or serve module per request (kernels/turbo_rows.mojo, kernels/mega.mojo
   ssm_phases/ffn_phases, serve/window.mojo gemm_w/step_window, kernels/fewrow.mojo). Ask: bugs, dead code, slow
   paths, with file:line. Output `.work/local-jobs/findings-<model>.md`. YOU verify each finding against source
   before it goes into the report (mark VERIFIED / FALSE); count true-positive rate per model.
J2 Summaries: digest today's lane reports and bench logs (`docs/lanes/2026-11-16-*`, `.work/speed/*.log`) into a
   one-page `.work/local-jobs/digest.md`; you spot-check 5 claims.
J3 Gated Mojo writing (prior verdict 2026-10-12: local 27B wrote 0/19 surviving Mojo proposals; retry with today's
   models): 3 small tasks with existing gates, e.g. a missing test case row, a tools/ helper, a doc-derived check.
   A proposal counts only if it compiles and its gate passes; report survivors per model. No kernel edits on main.

## Rules
Toolscript (`mcp__hub__run_js`) for every read batch. gpu-wait for every engine. Memory cap your processes
(`systemd-run --user --scope -p MemoryMax=10G`). Nothing commits to main except findings you VERIFIED and fixed with a
gate (own commit each). Stop and report if an engine crashes twice.
