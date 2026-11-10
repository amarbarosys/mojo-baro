# N-gram draft tree beside the MTP chain: stage 0 (simulation), frozen 2026-10-09

Bound by `PROTOCOL-RULES.md`. Follows `draft-tree-protocol.md` (MTP tree closed at stage 0: a tree of MTP
drafts multiplies draft cost without changing per-position acceptance). The open question here is different:
an n-gram tree is draft-free (no MTP layer, no head read), so its branches cost only verify rows. Does adding
`ngram_propose_tree` branches to the k=2 MTP window raise tokens per millisecond?

## Inputs
- Trace: `bench/mtp-prompts.sh ENGINE OUT 2` with `BARO_SPEC_DBG=1` on the 20 `bench/mtp-prompts`, engine at HEAD,
  greedy (generated ids are identical across arms, so acceptance of any proposal is exact against them).
- Second set: the code prompts among them (names with py, code, fib, json, sql, rust or bash), reported separately.

## Simulation (`tools/ngram-tree-sim.mojo, imports serve/ngram.mojo`, CPU)
Per k=2 window start t (from the trace): MTP accepted a_m (from the trace). Tree = `ngram_propose_tree(history,
limit=L, branches=B)` over prompt + ids[:t]; a_n = longest branch prefix matching ids[t:]. Window yields
1 + max(a_m, a_n) tokens. Verify rows r = 3 + sum of branch lengths (capped at 8, MR limit). Window cost
c(r) = 23.1 + 5.0 * (r - 3) ms (stage-0 receipt: 23.1 / 28.4 / 31.4 ms at 3 / 4 / 5 rows; slope from k=2 to k=3).
Window starts are the traced ones (drift from extra accepted tokens ignored, the approximation draft-tree-protocol.md also states). Arms: B in {1,2,3}, L in {2,3,4}.

## Frozen predictions
- Whole set: best arm within +-2% of static k=2 (102.6 simulated tok/s); point +0.5%.
- Code subset: best arm +3% to +12% over k=2; point +6%.
- Reason: n-gram hits are concentrated on repeated spans (code, lists); on prose they cost rows and rarely accept.

## Decision
Stage 1 (build the tree verify in the m-row window, `S1` in the plan) only for an arm whose simulated median beats
k=2 on the set it would be enabled for. A null is recorded here with the numbers.

## Result, frozen arms (2026-10-09, trace `.work/claims/s0-trace`, 20/20 identity PASS, engine `593b988`+attn revert)
Simulated median tok/s, base k=2 = 100.11 (all) / 109.29 (code, n=5). Every arm loses: B1 L2 0.848x / 0.842x,
B1 L4 0.787x / 0.753x, B3 L4 0.772x / 0.713x. Both predictions MISSED (point +0.5% / +6%). Cause: min_width=1
proposes on almost every window (any 1-token suffix repeats), so the tree pays rows nearly always and rarely accepts.

## Amendment A (frozen before its run): propose only on a strong match
Arms B=1, L=3, min_width W in {3, 4, 6} (the tree is empty, zero extra rows, unless a W-token suffix repeats).
Predictions: all set best arm 0.99x to 1.02x (point 1.00x); code best arm 1.00x to 1.06x (point 1.02x).

## Result, amendment A
W=3 0.944x / 0.932x, W=4 0.966x / 1.000x, W=6 0.969x / 1.000x. Predictions: all MISSED low (band 0.99-1.02),
code at the band floor (1.00x, i.e. the tree never fires). **Verdict: CLOSED AT STAGE 0, no arm beats k=2.**
Scope of the null: 64-token greedy answers to the 20 short prompts. Long repeated spans (code edits, quoting a
long context) are not in this set; reopening needs a trace of that workload and a new frozen entry, not a re-run.
