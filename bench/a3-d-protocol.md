# A3(d): admission and preemption (2026-09-17)

This note is frozen before the A3(d) admission check. It follows A3(c)
commit `477fea2` and keeps the coordinator-owned row-kernel seam out of this
slice.

## Invariant

The single `baro-serve` process admits multiple distinct request ids, chooses
which queued request advances at each host scheduling boundary, and returns
each result to its original client. Mixed prompt lengths must not change any
request's greedy token stream. A request that is not yet started remains
queued; cancellation/preemption is not claimed until a resumable per-row
state owner exists.

## Frozen check

`bench/a3-d-admission-gate.sh` sends five requests with distinct prompt
lengths concurrently to one server and reads `/health` while work is active.
It compares every response token stream to the matching single-request
reference, checks five distinct response ids, five terminal responses, the
server's admission receipt, and clean shutdown. It records the row-scaling
receipt `bench/ssm-mrow-protocol.md` round 2 (`75db5f1`) as context, but does
not convert the current serial admission path into a throughput claim.

## Stopping point

If the worker cannot expose all five ids before the first terminal response,
stop with `ENGINE:` and name the queue/reader seam. If an identity mismatch
appears, stop before timing and retain the response bodies. If preemption
requires a kernel or state-layout change, leave it to the A3(c) kernel owner.
