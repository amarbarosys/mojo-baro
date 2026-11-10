# B4 stage 2: expert locality on the 35B MoE

Preregistered 2026-09-15, before the trace mode was written, at commit
`ae4c5c5`. Binds `bench/PROTOCOL-RULES.md`. Stage 1 (PCIe bandwidth and
expert bytes) is `exchange/2026-09-15-p3-b4-stage1-report.md` at `b7f7db5`.

## The question, and why the tier is not built first

`docs/NEXT-PLAN.md` B4 claims a 100B-class MoE interactive on one 24 GB card
by streaming experts from host RAM. Stage 1 measured the two numbers that
bound it: **28.5 to 28.8 GB/s sustained H2D** on this box (link read back at
x16, 16 GT/s) and **0.573 GB of routed expert bytes per token** for our 35B
(0.791 GB with the shared expert and router), which scales to 1.64 to 2.26 GB
per token for a 100B-class model at the same expert width.

Everything after that rests on one unmeasured quantity: **what fraction of a
token's selected experts is already resident.** MoE-Infinity found under 5% of
experts repeat within a request for a roughly 100-expert model; ours has 256
per layer and picks 8. If locality is that low here, a host-resident tier
cannot pay for itself and B4's build order changes.

So stage 2 measures locality first, from the real engine on the real prompt
set, and only then decides whether the tier is worth building. The
measurement needs the router's choices, not a tier: a tier built before this
number exists would be an engine change resting on an assumption.

## Method

1. **Trace.** `BARO_EXPERTS=<path>` in `serve/window.mojo` accumulates the
   top-8 expert ids `moe_ffn` already computes, per layer, per token, into one
   device buffer, copied to the host once per request. No per-layer sync, so
   the trace run is not a timed run and does not pretend to be one. The 20
   prompts of `bench/mtp-prompts/`, 64 tokens each, `BARO_SPEC=0 BARO_MEGA=0`.
2. **Replay.** `bench/moe-locality.py` replays an LRU per layer over the trace
   at a sweep of capacities (experts resident per layer, from 8 to 256), and
   reports for each: hit rate, expert bytes moved per token at that capacity,
   the resident VRAM those experts occupy, and the implied milliseconds per
   token at stage 1's measured 28.5 GB/s.
3. **Cross-check** that the trace is the engine's real behaviour: the token
   ids the traced run emits must equal the untraced run's on all 20 prompts.
   A trace that changed what the model did would describe a different engine.

## P1 read-back

Engine sha256 built in the same command as the run; `BARO_EXPERTS` echoed by
the run; `prompt tokens`, `tokens: 64` per prompt; the trace's own row count
(20 prompts x 64 tokens x 40 layers x 8 ids) checked against what the replay
reads, so a truncated trace fails loudly rather than producing a hit rate over
fewer tokens than claimed.

## Frozen predictions

1. **Hit rate at a resident half (128 of 256 experts per layer) is below
   70%.** The router is trained to spread load; if this comes back above 90%,
   the routing is far more repetitive than the literature suggests and that
   result is itself the finding.
2. **Sequential repeat rate (the same expert chosen for the same layer on two
   consecutive tokens) is between 5% and 40%.** This is the number a prefetch
   from the current layer's router would exploit, and MoE-Infinity's under 5%
   for a 100-expert model is the low anchor.
3. **No capacity below 256 per layer makes the 100B-class 30 tok/s target fit
   on transfer time alone.** Stage 1 put the required residency at 41 to 58%
   for that target; prediction 1 says a resident half does not deliver a
   hit rate high enough to cover it.

A prediction that fails here is a finding, not a defect, as long as the trace
cross-check passed: this stage measures the model's behaviour, it does not
change it.

## Kill line

If the hit rate at a resident half is below 30%, the host-resident tier is not
built: the streaming claim then depends on prefetch alone and B4's build order
is rewritten around prediction 2 rather than around a cache.

## Result (2026-09-15): two of three predictions falsified, the tier pays

Trace: `.work/b1/experts.txt`, engine `.work/b1/engine-trace` built from HEAD
plus the trace mode, 20 prompts x 64 tokens x 40 layers x top-8 =
**51,200 rows, none missing** (the replay refuses a short trace). Replay:
`bench/moe-locality.py`, bytes per expert 1.77 MB (gate + up + down, q4_k),
bandwidth 28.5 GB/s from stage 1.

**Cross-check first: the traced run emits exactly what the untraced run
emits, on all 20 prompts.** So the trace describes this engine and not an
instrumented variant of it.

**Sequential repeat: 39.3%** of 403,200 picks (same expert, same layer, two
consecutive tokens). A uniform-random router of 256 experts choosing 8 would
give 3.1%, so the routing is strongly structured, and MoE-Infinity's "under 5%
of experts repeat within a request" for a roughly 100-expert model does not
describe this model. Prediction 2 (5% to 40%) held, at the top of its band.

| resident per layer | resident VRAM | hit rate | GB/token | ms/token at 28.5 GB/s |
|---|---|---|---|---|
| 8 | 0.6 GB | 28.1% | 0.407 | 14.3 |
| 16 | 1.1 GB | 49.8% | 0.285 | 10.0 |
| 32 | 2.3 GB | 65.1% | 0.197 | 6.9 |
| **64** | **4.5 GB** | **76.7%** | **0.132** | **4.6** |
| 96 | 6.8 GB | 79.8% | 0.115 | 4.0 |
| 128 | 9.1 GB | **80.4%** | 0.111 | 3.9 |
| 256 (all) | 18.1 GB | 80.5% | 0.110 | 3.9 |

**Prediction 1 is falsified.** A resident half gives **80.4%**, not under 70%.
The ceiling is 80.5% and it is reached at 128; a quarter-cache (64 per layer,
4.5 GB, which fits beside the 5.2 GB trunk) already gets 76.7%. The reason is
visible in the same table: with an unlimited cache the hit rate is still
80.5%, so **only about 100 of the 256 experts per layer are ever touched in a
64-token request**, and they are touched repeatedly.

**Prediction 3 is falsified too, with a caveat that matters more than the
prediction.** At 80% hit rate the 100B-class arithmetic from stage 1 (1.64 GB
routed per token) drops to about 0.32 GB and 11 ms, which fits a 30 tok/s
budget on transfer time. But that applies OUR router's locality to a model we
have not traced. The honest statement is: **on this 35B, a 4.5 GB expert cache
covers three quarters of expert traffic**; whether a 100B-class model routes as
repetitively is an assumption until one is traced.

**Kill line: not reached.** It was "below 30% at a resident half"; measured
80.4%. The host-resident tier is worth building, and stage 3's prefetch has a
concrete target: the 19.5% that a perfect within-request cache still misses
are cold first-touches, which is exactly what a router-driven prefetch of the
next layer can hide.

Caveats, stated rather than buried: the cache is reset per prompt, so this is
within-request locality over 64 tokens and a longer request would score higher;
and the bytes-per-expert figure averages q4_k, ignoring the three layers whose
`down` is q6_k, which biases bytes slightly low.
