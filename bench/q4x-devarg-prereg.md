# q4x-host: preregistration of lever 2, device argmax at the token boundary (lane q4x-host), frozen by this commit

Data: `exchange/lane-q4x-host-timeline.md`: the gap after the last kernel of a token is 3.85 ms/token of 34 (32 gaps, traced, host only):
`logits()` downloads 248,320 f32 (1 MB), copies them element by element into a `Vec`, then `argmax(lg)` scans them again on the host.
`verify()` already runs the same head launch plus `amar_q4x_argmax_p/_r` on the device and reads back one id.

## Lever
`Q4X_DEVARG=1` (default off): the plain decode loop in `run()` calls `Q4X.step_id(tok) -> Int` (forward, head launch, device argmax,
one 4-byte readback) instead of `step()` + host `argmax`. Same head GEMV launch and the same argmax tie rule (first index of the maximum,
strict greater) as the prompt path and `verify()`, so ids must be identical.

## Arms (bench/q4x-ab.sh, same job, one binary, order A B B A)
A = `""`, B = `Q4X_DEVARG=1`. 20 prompts, NG 64, `Q4X_TMAX=16384`, resident `Q4X_STDIN`, warm list. Read-back: engine prints `q4x: devarg N`;
q4x-ab.sh compares it with the env string. The receipt carries the PCIe root-port link (`bench/pcie-link.sh`); a degraded link
does not change this lever's mechanism (host CPU, one 4-byte readback instead of 1 MB) but it inflates A, so B/A reads smaller than on a full link.

## Predictions (A about 30 ms/token)
| quantity | prediction | band |
|---|---|---|
| gap after the token's last kernel | 3.85 -> 0.5 ms | 0.3 to 1.5 ms |
| B/A | 0.93 | 0.88 to 0.98 |

Kill line (brief): B/A > 0.95. Identity must PASS on every run (INVALID otherwise). Teacher-forced `bench/q4x-gate.sh 20 32` is run with
`Q4X_DEVARG=1` as well (the force path takes the same branch).

## Budget
LOC <= 30 in `serve/q4x.mojo`. Bytes: none (reuses `vids`, `vprob`, `vidh`, `amp`).
