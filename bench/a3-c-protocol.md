# A3(c): one decode launch for four resident requests (2026-09-17)

This note is frozen before A3(c) implementation and timing. It builds on
merge commit `a96f6c4` and the A3(b) resident-state layout. It does not
re-measure the row-scaling prerequisite: the receipt is
`bench/ssm-mrow-protocol.md`, round 2, commit `75db5f1`.

## Invariant

One target decode launch advances four different request ids in four rows.
Each row reads its own token history, KV page table and resident recurrent
state, and uses its own `SampleParams`. The launch receipt must print all four
ids from the running engine. Four serialized calls in one host function do not
count.

## Frozen row-scaling receipt and prediction

The inherited round-2 receipt measured the launch path at m=1, 2, 4, 8 with
`BARO_MEGA=0`. Its mode-1 step totals were 11.322 ms at m=1 and 19.584 ms at
m=4, an aggregate ceiling of 2.31x and the frozen target N=4. The m=8
falsifier already fired, so no N=8 claim is in scope.

The throughput gate predicts aggregate batch throughput at 2.08x to 2.54x
the single-request 20-prompt median, using 2.31x with a +/-10% band. The
single-request median must remain within +/-2% of its pre-batch value. A result
below 2.08x is a below-bar result, not a rounded pass. A result above 2.54x
requires receipt audit before adoption.

## Staged pre-kernel gate

The coordinator-owned row-kernel seam is not present in this tree. Before
timing the one-launch candidate, the host harness gate is therefore a
two-request identity check using one `a3c2` batch marker followed by two
request rows. The engine allocates both row descriptors and advances the rows
with serial `step_window` calls inside one batch step, printing both ids and
their slots. This gate proves request parsing, descriptor population, private
state/page-table selection, readback and receipt coverage; it explicitly makes
no one-launch or throughput claim. The one-launch gate below remains frozen for
the coordinator's kernel implementation.

## Frozen gate

`bench/a3-c-gate.sh` uses the 20 exact-token prompts in
`bench/mtp-prompts/p*.tokens`, cyclically groups them into five four-row
batches, and generates 64 greedy tokens per request at temperature 0. The
reference arm is single-request and reference-cached. The candidate arm is
one engine process with five four-row launches. `BARO_MEGA=0`, speculative
decoding off, q4 pack, and the effective TMAX, row count, ids, and prompt
lengths must be read back from engine output.

The gate checks every batch row against its single-request token stream,
requires five launch receipts containing four distinct ids, requires
40/40 terminal rows, and checks clean EOF. It records aggregate candidate
tokens per candidate decode time, the single-request median, both hashes, the
row-scaling receipt path, and the effective arm values.

## Falsifiers and stopping point

The gate fails if any row differs, an id is missing or duplicated, a launch
receipt has fewer than four ids, the engine reports serialized launches, or
the single-request regression exceeds +/-2%. The throughput prediction fails
below 2.08x; no number is promoted without its read-back receipt.

If the existing launch kernels cannot address four per-row page-table bases,
positions and recurrent-state regions without changing `kernels/*.mojo`, stop
before timing and write `ENGINE:` with the exact missing ABI. Do not relabel
same-sequence MTP rows as distinct requests and do not substitute four
serialized `step_window` calls.
