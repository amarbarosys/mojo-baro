# Teleport KV T3a: KV pages between two engine processes over HIP IPC, frozen 2026-10-09

Bench: `bench/kv_ipc_bench.mojo` (two processes, fork), 5 KV pages = 40 MiB f32 (E12 size, same as
`bench/latent_ingest_bench.mojo`), 8 iterations, first 2 warmup. A: `KvIpcStage.fill` (kernel copy into a raw
hipMalloc stage) then sends the 64-byte handle. B: `ingest_kv_ipc` (shim open, kernel copy into its own pages, close).
Baseline: the memfd path in B's process at the same size (`mint_kv_latent` + `ingest_kv_latent`).

Checks: B's K and V pages byte-equal to A's pattern (every float compared on the host) on every iteration.
Predictions: B ingest min < 2 ms (open + 40 MiB at >= 200 GB/s); A fill min < 0.5 ms; memfd mint + ingest min >= 10 ms;
handoff (fill + socket + ingest) at least 10x faster than memfd.

## Result (2026-10-09, gpu-wait, binary built from this commit's sources)
40 MiB (5 pages K+V), iterations 2-7: A fill min 0.115 ms (HELD < 0.5), B ingest min 0.306 ms (HELD < 2; first
iteration 5.0 ms = first open), memfd mint+ingest min 6.42 ms (prediction >= 10 MISSED: the memfd path is faster
than E12's 60+37 ms after its pinned-stage fix), handoff fill+ingest 0.42 ms = **15.2x** the memfd path (HELD >= 10x).
Every float of K and V equal on all 8 iterations. Not in the handoff number: the 64-byte socket round trip.
Not done (T3b): an engine pair exchanging a real checkpoint through the LatentOS agent and continuing (BARO_FORCE).

## Correction (2026-10-09, review by pane a lane pane)
The 15.2x above is a WARM-handle number: one stage and one handle were reused for all 8 iterations and the warmup
skip removed the only fresh open (iteration 0 ingest 5.0 ms). Cold handoff estimate 0.115 + 5.0 ms = ~1.25x memfd.
The bench now runs a cold arm (fresh stage + handle per iteration) beside the warm arm; result below when measured.

## Cold arm result (2026-10-09, same binary family, fresh KvIpcStage + handle every iteration 8-15)
Cold handoff (fill incl. hipMalloc + IPC get, then open + copy + close) min 0.512 ms, max 0.600 ms; memfd min 7.22 ms:
**14.1x cold**, 19.7x warm (reused handle). Every float equal, 16/16 iterations. The 5.1 ms at iteration 0 is a
once-per-process cost (first open / first kernel), not a per-handle one: iterations 8-15 each open a handle never seen
before at 0.25-0.31 ms. The review's ~1.25x estimate assumed 5 ms per fresh handle; measured, it is not.
