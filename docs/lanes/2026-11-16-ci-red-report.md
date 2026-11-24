# Report: fix the red `checks` workflow (lane-ci-red, 2026-11-16)

Brief: `docs/lanes/2026-11-16-ci-red-brief.md`. Failing run: `gh run view 43729809356 -R amarbarosys/mojo-baro --log-failed`.

## Cause

The `non-gpu` job failed at one step:

```
== docs/OPERATE-AUDIT.md matches docs/settings.toml and docs/operate-capabilities.toml, four planted defects fail (tools/operate-audit-check.sh)
  FAIL FAIL build: see .work/operate-audit-build.log
```

`tools/operate-audit-check.sh` line 7 hardcoded the compiler path:

```sh
MOJO=${MOJO:-./.venv/bin/mojo}
```

The CI runner has no `.venv`: the workflow installs `mojo==1.0.0` with `pip` to the system Python, so `mojo` is on
`PATH` but `./.venv/bin/mojo` does not exist. The parent `tools/ci-checks.sh` resolves the compiler correctly
(`MOJO=./.venv/bin/mojo; [ -x "$MOJO" ] || MOJO=mojo`) but does not `export` it, so the child script re-resolved it
to the nonexistent `.venv` path. `timeout 300 ./.venv/bin/mojo build ...` then failed with
`No such file or directory`, and the step printed `FAIL build` and exited 1.

Local runs passed because the developer checkout has `.venv/bin/mojo` (Mojo 1.2.0). The tok-check step passed
(80 implemented, 0 differ), matching the brief.

## Fix

One line in `tools/operate-audit-check.sh`: use the same `[ -x ]` fallback to `mojo` on `PATH` that
`ci-checks.sh` uses.

```sh
MOJO=${MOJO:-./.venv/bin/mojo}; [ -x "$MOJO" ] || MOJO=mojo
```

This keeps an explicitly set `MOJO` (the `bench/check-cert-hook.sh` convention), uses the repo `.venv` when it
exists, and falls back to `mojo` on `PATH` otherwise. The check now runs both in CI (no `.venv`) and standalone.

## Local repro

Reproduced the CI condition in the lane worktree by hiding `.venv` (the worktree links the main checkout's `.venv`
via `lane-prep`; `mojo` on `PATH` still resolves to it):

```
$ mv .venv .venv.hidden && tools/operate-audit-check.sh
FAIL build: see .work/operate-audit-build.log        # before the fix
$ cat .work/operate-audit-build.log
timeout: failed to run command './.venv/bin/mojo': No such file or directory
```

After the fix, same condition:

```
$ mv .venv .venv.hidden && tools/operate-audit-check.sh
PASS operate-audit
```

Full non-GPU suite in the worktree, `CI_CACHE=0 tools/ci-checks.sh`:

```
all non-GPU checks passed
CI_CHECKS_EXIT=0
```

## Commit

`721b437 fix(ci): resolve mojo from PATH when .venv is absent` on branch `lane-ci-red`, worktree
`~/Projects/mojo/mojo-baro-lanes/ci-red`. Not pushed.

Push command for the boss:

```
git -C ~/Projects/mojo/mojo-baro-lanes/ci-red push -u origin lane-ci-red
```
