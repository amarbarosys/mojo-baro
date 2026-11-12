# q4x-host: preregistration of lever 1, parallel miss DMA (lane q4x-host), frozen by this commit

Data: `exchange/lane-q4x-host-timeline.md`. A missed expert costs 3 serial SDMA copies of 460,800 B at about 3.1 GB/s each (latency
bound: PCIe 4 x16 moves 460 KB in 20 us), and the GPU waits for the chain after the router: 11.3 ms/token of 34 (idle 60%).

## Lever
`Q4X_MISSPAR=N` (default 0 = today's single-stream copies, no change). The three pieces of every missed expert are issued round-robin on N
extra streams (`create_stream`, as `serve/expert_tier.mojo` does) right after the router doorbell; one event per stream is recorded after
the layer's last copy and the main stream waits on it before the expert GEMMs. Hazard argument: the doorbell proves the main stream is
drained, so no earlier kernel reads a slot that a copy overwrites; slots hit by this layer are protected by `slot_ref`. Values copied are
the same bytes, so ids must be identical.

## Arms (bench/q4x-ab.sh, same job, one binary, order A B B A)
A = `""`, B = `Q4X_MISSPAR=3`. 20 prompts, NG 64, `Q4X_TMAX=16384`, resident `Q4X_STDIN` runs, warm list. Read-back: engine prints
`q4x: misspar N streams`, checked against the env string by q4x-ab.sh.

## Predictions (A is about 30 ms/token in this protocol)
| quantity | prediction | band |
|---|---|---|
| B/A (median of per-prompt medians) | 0.85 | 0.78 to 0.95 |
| gap after router with 1 miss (rocprof diagnostic, p03) | 451 -> 190 us | 150 to 300 us |
| idle ms/token | 20.5 -> 14 | 12 to 17 |

Kill line (brief): B/A > 0.95 = no win, lever dropped, N and stream count reported. Identity must PASS on every run (INVALID otherwise).
Also required: `Q4X_MISSPAR` unset must give A's binary behaviour (plain decode B/A <= 1.01 follows from the same job).
Mechanism of a miss: if the gap does not shrink with N the copies are bandwidth-bound (host memory or SDMA count), not latency-bound.

## Budget
LOC <= 40 in `serve/q4x.mojo`. Bytes: no new buffers, N streams and N events.
