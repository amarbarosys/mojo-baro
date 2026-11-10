# Flash slim, round 2: a bigger in-domain trace and per-layer keep sizes (prereg)

FROZEN by the commit that adds this file, before any round 2 run. Round 1 (`bench/flash-slim-protocol.md`, receipts
`exchange/receipts/flash-slim-r1/`) failed its agreement bar at every keep level; its perplexity view put the best
point at keep fraction 0.836, in-domain perplexity ratio 1.018. Round 1's loss came from held-out text routing to
experts its 2048-token in-domain trace never saw.

## Question

Does a 4x larger and more varied in-domain trace (A), and a keep size per layer chosen on held-out trace data (B),
move the quality-per-size curve: lower perplexity ratio at the same keep fraction, or the same ratio with fewer
experts?

## Data

- Eval batch and reference arm R: unchanged from round 1 (`eval.ids` sha256 `116c8b0c21735f5e`, R hidden rows from
  run1, same q4x binary, identity PASS there). Every round 2 number is comparable with round 1.
- Trace: in-domain only, 32 sequences of 256 tokens: 16 skills, 16 papers, one document (or a run of consecutive
  documents) per sequence, spread evenly over each corpus, never an eval document (skills index % 7 != 3, papers
  paper % 50 != 25). Four batches of 8 sequences (4 skills + 4 papers), 2048 tokens each: 8192 in-domain tokens
  against round 1's 2048, and batches of 2048 because q4x prefill grouping grows with the square of the batch.
  Built by `bench/flash-slim-prompts.sh OUT trace2`.

## Keep sets

- A-c: per layer the smallest expert set covering c of all trace hits (round 1's rule on the new trace), c in
  {0.99, 0.995, 1.0}.
- B-c: per layer, rank experts on the even-numbered trace sequences (fit half), take the smallest k whose top-k
  covers c of the hits on the odd-numbered sequences (val half), at least 10; the kept set is the top k ranked on
  all trace hits. c in {0.99, 0.995}. The size adapts to how well each layer's ranking generalizes.

## Arms

One gpu-wait job: four trace arms (T1..T4, `Q4X_EXPERTS`), then one masked arm per keep set on the eval batch, each
reading its mask back from its own expert trace (0 picks outside the set), scored against R on the CPU.

## Bars

1. Mask read-back: 0 picks outside the keep set in every masked arm, else that arm is void.
2. PASS: some arm reaches in-domain perplexity ratio <= 1.01 (the maintainer's 99% quality) at keep fraction <= 0.80
   (smaller than round 1's best point). Agreement against the round 1 bar (97.34%) is reported, not gating.
3. Reported: per-arm keep fraction, in-domain and per-category perplexity ratio, agreement, KL, expert misses; B
   against A at matched keep fraction.

## Prediction (frozen)

- Unhit experts shrink with 4x the tokens: A-1.0 keep fraction 0.90 to 0.95, ppl ratio <= 1.01.
- A-0.99: keep fraction 0.68 to 0.74, ppl ratio 1.08 to 1.13 (round 1 at 0.668: 1.158).
- B-0.99: keep fraction 0.70 to 0.78, ppl ratio 1.05 to 1.10; B beats A at matched keep fraction.
- Bar 2 most likely FAILS: the closest arm at or under 0.80 lands near 1.03 to 1.06.

## Budget

GPU: one job, about 4 x 15 s trace arms + 5 x 20 s masked arms + loads, under 5 minutes. CPU: prompts, keep sets,
five scorings of 12 s each, one at a time (RAM).
