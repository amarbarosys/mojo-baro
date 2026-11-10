# A3(b): two resident sequence states (2026-09-17)

This note is frozen before implementation. It proves memory isolation only;
decode remains serial. The synced base is merge commit `3474e64`, and the
dense engine must be rebuilt from that source before any GPU job.

## Invariant

Two requests own independent KV page tables, KV pages, convolution state,
SSM state, token history, ring position, prefix chain, and sampler state in
one engine process. The scheduler may switch at a window boundary. A switch
must not overwrite either request's resident state.

## Frozen gate

`bench/a3-b-gate.sh` runs one dense q4 engine with two request ids in one
process. It submits request A, admits request B while A is active, then
alternates them at a window boundary until each has generated 64 tokens. The
same two requests run alone at `temperature=0` for references through
`refcache`; each resident response must be byte-identical to its own reference.

The identity arm uses 20 real prompt pairs from
`.work/a2-prompts/L8192/p*.tokens`, pairing each prompt with the next prompt
cyclically, with the existing prompt lengths and `n=64`. The receipt must show both ids assigned distinct sequence slots and a
boundary with both ids resident. It must report identity `20/20` for A and
`20/20` for B, with no missing or duplicate terminal response.

The second arm sets `BARO_KVTAB=reverse` for the resident page-table mapping.
The receipt must show a non-identity physical mapping for both sequence page
tables and the same `20/20` identity result for A and B. An identity table
read by mistake is therefore a falsifier, not a passing shortcut.

Every arm reads back the engine ready line, request ids, slot ids, page counts,
mapping mode, boundary admission, generated count, finish reason, and clean
shutdown. No throughput claim is made here.

## Prediction and falsifier

Prediction: the existing A2 block-table addressing remains bit-exact when each
request receives a private page-table region and private recurrent-state
region. Reverse mapping will match identity because the kernels already read
the table.

Falsifiers: either sequence differs from its single-request reference; a
request's output contains another id's tokens; both ids share a slot or page;
the reverse arm reports an identity mapping; the boundary receipt is absent;
or the engine fails to shut down cleanly.

## Stopping point

If the first boundary cannot retain both sequence states in device memory,
stop before any timing and write `ENGINE:` with the exact missing state or
allocation seam. Do not substitute host checkpoint restore for resident
state. If identity passes but reverse mapping cannot be made non-identity,
stop and report the unverified page-table path without claiming A3(b).
