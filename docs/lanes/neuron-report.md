# lane-neuron report

Branch `lane-neuron`. Task: R3 neuron hops, falsifier first. Full result and numbers: `miniagi/latent/PREDICTIONS-neuron.md`
(frozen before the grid) and `miniagi/latent/RECEIPTS-neuron.md`.

## Result

Real beats shuffled and none on 3 of 3 seeds (110-112): 98.3/98.9/98.8% vs 26.2/24.0/25.5% vs 31.5/31.9/32.0%, no-lookup
ceiling 31.8%. The win needs a readout loss on the true link (aux 0.5). With aux 0 real equals none (32%), also at 3x updates and
lr 3e-4 (seed 110 only). fp32 neuron gradients match the float64 torch oracle at 1e-4; the bf16 path is 7% off at b=64 (1% without
neuron rows), UNVERIFIED, no bf16 gradient claim.

## Merge gate

- Toolchain: lane `.venv` is a real `uv sync` Mojo 1.2.0 venv (not committed, ignored). The engine for the state layer was built
  with it (`.work/engine`); `serve/target/release/baro-serve` is a symlink to the primary build (ignored).
- run-tests: `./run-tests.sh` under `gpu-wait run --timeout 2400`, BARO_PACK = the primary checkout q4 pack (set by env, not in this lane), rc=0.
  Log: `.work/neuron/run-tests.log` (last line `rc=0`).
- First attempt ended rc=3 (state layer skipped, no engine in the lane): `.work/neuron/run-tests-first-rc3.log`. Fixed by building the
  engine, not by changing the gate. Every pack-free test passed in both runs; no failure on this lane.
- The lane touches only `miniagi/latent/`, `tools/neuron-data.mojo`, `tools/neuron_oracle.py`, `docs/CAPABILITIES.md`, this file.
