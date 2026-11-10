# Flash slim, round 4: calibrated keep below 99% (prereg)

FROZEN by the commit that adds this file, before any round 4 keep set is computed or run. Round 2's calibrated rule
(`bench/flash-slim-r2-protocol.md`, B) reached in-domain perplexity ratio 0.994 at keep fraction 0.823 (B-v0.99) and
0.997 at 0.880 (B-v0.995); its size bar (<= 0.80) failed. Round 3's whole groups needed 0.948.

## Question

Does the calibrated rule at a lower held-out coverage reach keep fraction <= 0.80 while the in-domain perplexity
ratio stays <= 1.01?

## Keep sets and arms

B-c exactly as round 2 (rank on even trace2 sequences, per-layer size from the odd sequences' coverage, keep the top
k on all hits), c in {0.985, 0.98, 0.97}, on round 2's four trace files (reused). Reference and eval batch from
round 1 (reused, same binary). One gpu-wait job, one masked arm per keep set, mask read back from its own expert
trace, next-token perplexity on the CPU.

## Bars

1. Mask read-back: 0 picks outside the keep set, else the arm is void.
2. PASS: an arm at in-domain perplexity ratio <= 1.01 with keep fraction <= 0.80 (round 2's bar).
3. Reported: agreement, KL, per-category perplexity, expert misses; the full B curve (rounds 2 and 4).

## Prediction (frozen)

- Keep fractions: B-0.985 0.78 to 0.81, B-0.98 0.75 to 0.79, B-0.97 0.70 to 0.75.
- In-domain perplexity ratio: B-0.985 1.000 to 1.015, B-0.98 1.005 to 1.03, B-0.97 1.02 to 1.06.
- Bar 2 is a coin flip at B-0.98; my call is a narrow FAIL (1.01 to 1.02 at about 0.77).

## Budget

GPU: one job, 3 masked arms of about 20 s plus loads, under 3 minutes. CPU: keep sets, three scorings.
