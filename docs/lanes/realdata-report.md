# realdata lane report

Branch `lane-realdata`. Scope: `miniagi/latent` only (family hold-out script, arithmetic task with a Mojo checker, predictions, results). No change to kernels, serve or the engine.

## Gate
run-tests rc=0 under `gpu-wait run --timeout 1800`, pinned toolchain (uv sync venv, Mojo 1.2.0 2016d0cd), log `.work/run-tests.log`, env `BARO_PACK=~/Projects/mojo/mojo-baro/.work/engine-pack-q4`, engine `.work/engine` built from this tree (`.work/engine-build.log`), `baro-serve` copied from the primary checkout (main HEAD 0074feb equals this branch's merge base; `git diff HEAD main -- serve kernels` is empty).
First pass `.work/run-tests-1-rc3.log` ended rc=3: all pack-free tests passed, `test_prefix` and the state layer were skipped for lack of a q4 pack. That is the script's designed incomplete exit, not a failure; the second pass has no skips.

## Results
Numbers, tables and caveats: `miniagi/latent/RESULTS-realdata.md`. Predictions frozen before the runs: `PREDICTIONS-family.md`, `PREDICTIONS-arith.md`. Run logs: `.work/spec/arith-folds.out`, `.work/spec/arith-pilot.out`, `.work/spec/fam-add.out`, `.work/spec/fam-xor.out`.
