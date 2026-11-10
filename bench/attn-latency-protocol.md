# Attention phase latency (frozen 2026-09-06, after the structural round, before any change)

State: 125.4 tok/s_gen (`7a4e357`), 7.85 ms/token. The attention compute phase (`attn_phases` stamps 3->4) is 300-306 us
per token over 8 layers = 38 us per layer for T ~ 70 positions, on 16 of 96 blocks (one head each, `tid < HD` = 256
threads active). Its work is tiny (16 heads x 70 positions x 256 dims x 2 = 0.6 MFLOP, 1.4 MB of KV): the time is latency.
Two serial chains per thread: the score dot over HD=256 (256 scalar loads of `Kc[kvh, t, d]` per position, one thread per
position, only ~70 threads busy) and the output loop over T (one coalesced V load per position, `o +=` chain).

## Design

1. One `@always_inline` head body in `kernels/attn.mojo`, called by `amar_attn_decode` (launch) and the megakernel phase:
   identity by construction (the two copies are textually the same computation today; the shared body first, unchanged
   arithmetic, must pass the gate with no numeric change).
2. Inside the body: score dot with 8-wide vector loads of K (32 B per instruction instead of 4), scalar accumulation in
   the same order; output loop with the V loads hoisted 8 at a time, `o +=` chain unchanged. If the compiler contracts
   differently after the restructure both kernels move together; `tools/model-ref.py` 64/64 stays the judge.
3. Not in scope: moving to a wave-per-position reduction (changes the summation order; only if 1-2 do not land).

## Stages / checks

| stage | check |
|---|---|
| A0 shared body, no arithmetic change | `test_mega_block` all cases 0 mismatches; engine mega == launch spec 0/1; 64/64 vs model-ref |
| A1 vector K loads + hoisted V loads | same gates; per-phase profile attn 3->4 |
| A2 | 20-prompt A/B HEAD vs A1, one stint, clock-probe, fail word 0/20 |

## Frozen prediction

attn phase 306 -> 120-170 us per token: **125.4 -> 127-128 (+1.5-2.5%)**. Land >= +1.2%; close < +0.5%. This is a
small round by design: the pool is 3.9% of the token. Stop rules: A0 any mismatch = the two copies were not the same
computation, find the difference, do not paper over it; A1 mismatch vs model-ref = the contraction moved the tokens,
report and revert.

## Result (2026-09-06, `results/attn-latency/`)

| stage | receipt |
|---|---|
| A0 shared body | gate all cases 0 mismatches (no numeric change) |
| A1 | gate 0 mismatches; engine mega == launch spec 0/1; 64/64 vs model-ref; fail word 0 in 40/40 runs |
| A2 | same stint, 2988 MHz med: **pre 125.14 (spread 2.2%) -> attn 130.74 (1.2%), 1.045x**, identity 20/20 |
| attn phase 3->4 | 295 / 605 us (the old form was also unstable run to run) -> **92 / 92** |

**LANDS: +4.5% (predicted +1.5-2.5%; the old phase's bad runs were worse than its good ones).** 8.18 -> 7.50 ms/token.
Both kernels moved together (shared body), tokens unchanged vs the fp32 reference: the contraction did not move.
