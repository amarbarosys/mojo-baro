# Flash slim, round 3: keep whole co-activation groups instead of single experts (prereg)

FROZEN by the commit that adds this file, before any round 3 run. Round 2 (`bench/flash-slim-r2-protocol.md`, receipts
`exchange/receipts/flash-slim-r2/`) reached in-domain perplexity ratio 0.994 at keep fraction 0.823 (B-v0.99) and
failed its size bar (<= 0.80). The co-activation sweep (`exchange/receipts/flash-slim-coact/`) found that experts
cluster in every layer (k = 8: intra-part co-activation 0.532 vs 0.124 random) and align with domains in the middle
layers.

## Question

Experts in one co-activation group fire on the same tokens. Does keeping or dropping them as a block lose less than
dropping single experts, so that a group keep set reaches the same perplexity with fewer experts?

## Keep sets

Groups: graph-shard k = 16 per layer on the round 1 co-activation graph (8 categories x 1024 tokens,
`bench/flash-slim-coact.sh ... 16`, about 32 experts per group). G-c: per layer, rank the groups by their in-domain
hits in the round 2 trace (the four trace2 batches, 8192 skills + papers tokens), keep groups in that order until
they cover c of the layer's in-domain hits. c in {0.99, 0.995}. `tools/flash-slim gkeep`.

## Arms

One gpu-wait job, reference and eval batch from round 1 (reused, same binary), one masked arm per keep set with its
mask read back from its own expert trace, scored on the CPU with next-token perplexity.

## Bars

1. Mask read-back: 0 picks outside the keep set, else the arm is void.
2. PASS: an arm at in-domain perplexity ratio <= 1.01 with keep fraction <= 0.80 (round 2's bar).
3. Reported: G against round 2's A and B curves at matched keep fraction (linear interpolation between their points).

## Prediction (frozen)

- 32-expert groups mix domains, so covering 99% of in-domain hits takes most groups: G-0.99 keep fraction 0.85 to
  0.93, G-0.995 0.90 to 0.96.
- In-domain perplexity ratio 0.99 to 1.01 at both levels.
- Bar 2 FAILS (keep fraction above 0.80); at matched keep fraction G is no better than B.

## Budget

GPU: one job, 2 masked arms of about 20 s plus loads, under 2 minutes. CPU: gkeep, two scorings.
