# Draft depth / draft tree round -- CLOSED AT STAGE 0 BY MEASUREMENT (2026-09-06)

Bound by `PROTOCOL-RULES.md`. Candidate raised after the megakernel round:
deeper or adaptive MTP drafting (NInfer reports 4.6 tok/round at 92%
acceptance on Qwen3.5-27B; we ship k=2 linear). Stage 0 = measure the
acceptance curve and the window cost curve before freezing any design.

## Stage 0 receipt (`bench/mtp-prompts.sh`, 20 prompts, `BARO_SPEC_DBG=1`, engine `eb09d12`, `BARO_MEGA=1` default)

| arm | median tok/s_gen | spread | windows | acceptance | tokens/window | ms/window |
|---|---|---|---|---|---|---|
| no spec | 80.74 | 1.5% | -- | -- | 1 | 12.4 |
| k=1 | 98.86 | 22.8% | 712 | 81.3% | 1.80 | 18.2 |
| k=2 | 102.34 | 41.6% | 543 | 69.9% | 2.36 | 23.1 |
| k=3 | 95.43 | 59.4% | 473 | 58.7% | 2.71 | 28.4 |
| k=4 | 94.19 | 70.6% | 433 | 50.7% | 2.96 | 31.4 |

Per-position acceptance P(accept >= j), k=4 windows: **0.77, 0.54, 0.38, 0.27**
(k=3: 0.77, 0.55, 0.38; k=2: 0.79, 0.56). n_acc is bimodal (k=4: 101 windows
accept 0, 119 accept all 4): acceptance is a property of the text span, not a
per-position coin.

## Policy simulation on the k=4 traces (draft chain is deterministic given the prefix, so min(n_acc4, k) is exact per window start; start positions drift with policy -- approximation stated)

| policy | median tok/s | spread |
|---|---|---|
| static k=2 | 102.6 | 46% |
| static k=3 | 97.3 | 62% |
| static k=4 | 97.0 | 68% |
| AIMD 1..4, +1 on full window, k=n_acc+1 on miss | 100.4 | 50% |
| AIMD 1..4, +1 on full, k=max(1,n_acc) on miss | 99.5 | 48% |
| AIMD 2..4 | 101.4 | 58% |
| AIMD 1..4, +1 on full, halve on miss | 100.3 | 52% |

**No static or adaptive depth beats k=2.** The marginal draft position is
worth 0.38 then 0.27 tokens and costs ~5 ms (trunk m+1 row: FMA-bound per
`mrow-gemm-protocol.md`; draft step: MTP layer + 1 GB head read). A cheaper
draft head moves it by +1.6% (`draft-q4-protocol.md`, q4 head, landed). A
tree does not change the per-position acceptance; it multiplies the draft
cost. NInfer's 4.6 tok/round comes from a draft model that accepts 92% at
depth, i.e. the model, not the engine.

## Verdict

Closed at stage 0. k=2 linear stays the shipping config. What would reopen
it: a draft path under ~1 ms/step (then k=3 breaks even at P3=0.38), or a
better MTP head (model-side). Cost side is measured, not modelled: the
window-cost column above is the receipt.
