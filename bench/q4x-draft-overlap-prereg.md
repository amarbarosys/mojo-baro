# q4x MTP-draft expert overlap: preregistration (lane draft-overlap, plan item 3 falsifier), frozen by this commit

Question: do the experts the MTP draft's router picks predict the experts the verify pass needs, so that issuing their
copies after the draft would remove verify misses?

This is a COUNT, not a timing: valid on the 2.5 GT/s PCIe link (boss, 2026-10-21). No timing is claimed or used.

## Structural fact read from the code (before any run)
The MTP head is one extra block (`blk.48`, `MTP_L = NL`) with its OWN 512-expert bank (`mtp_ffn`, `mexp`, 1350 MB resident,
`amar_q4x_mtp_tab`). Its router ids index that bank, not the target layers' banks. An id match with target layer l
is therefore a statistical coincidence unless training correlated the two banks. The falsifier measures exactly that.

## Instrument (behind `-D Q4X_LOG_DRAFT_EXPERTS=1`, default off, no kernel code)
- `mtp_ffn`: after the draft router, copy the selection rows to the host (a sync) and OR the 10 ids per row into the
  window's draft set D (all rows of all passes of this `draft()` call; ids 0..511).
- `ffn` during `verify()` only: for every target layer l, V_l = ids routed to by any row of the window.
- `fetch` during `verify()` only: count every verify miss and its bytes; a miss (l, e) with e in D is "saved".
- Controls on the same windows: C1 = the previous window's D against this window's V and misses (a stale draft);
  C2 = chance, mean |D| / 512.
Definitions: recall_l = sum |V_l and D| / sum |V_l| (overall: summed over l); precision = |V_l and D| / |D|;
saved = verify misses with e in D, as a fraction of all verify misses, in count and in bytes.
Runs: `bench/q4x-draft-overlap.sh`, one binary, one gpu-wait job, 20 prompts `bench/q4x-prompts`, NG 64,
`Q4X_TMAX=16384`, K = 2 and 4, `Q4X_SPEC_P` 0.5, warm list r3a-warm-l01, GGUF and MTP paths of lane-q4x-mtp.
Read-back per run: `kv 16384`, expert-cache slots, `spec rounds ... k K pmin P`, the arm line of the overlap summary.

## Predictions (frozen)
1. Overall recall (draft ids that cover verify ids): 4% (range 2% to 10%), K=2 and K=4 alike; at or within 1 percentage
   point of both controls C1 and C2 (no draft-specific signal). P(recall >= 50%) < 3%.
2. Saved fraction of verify miss bytes: 5% (range 2% to 12%); P(>= 10%) = 20%.
3. No layer reaches 50% recall (the largest per-layer recall stays under 25%).
4. Verdict: KILL on both lines for both K (probability 90%).

## Rule (set before the run)
Kill line from the plan: overall recall < 50% or saved < 10% of verify miss bytes -> STOP, no prefetch code.
Only if both clear their lines, at both K, with the draft signal at least 10 points above C1, the lane reports
"alive" and the prefetch lane needs its own prereg. A control within 1 point of the measured value voids any claim
that the draft carries information.
Budget: engine +about 60 LOC under the comptime flag, harness 50 LOC, no buffers added when the flag is off.
