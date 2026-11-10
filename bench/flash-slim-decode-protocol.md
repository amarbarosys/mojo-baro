# Flash slim, decode A/B: what the B-v0.99 keep set buys at decode (prereg)

FROZEN by the commit that adds this file, before any decode run. Rounds 1-4 measured quality in batched prefill and
expert misses only there. This measures the benefit: decode speed and expert misses with the B-v0.99 keep set
(round 2, keep fraction 0.823, `exchange/receipts/flash-slim-r2/keep-v0.99.txt`) against full Flash.

## Scope

- RAM is out of scope for a mask: q4x pins the whole 35 GB expert region at load (`hipHostRegister`, run logs show
  RSS 38 GB) whatever the keep set. RAM residency needs the slicer; RSS and MemAvailable are recorded, not judged.
- Decode only, teacher-forced, so both arms decode the same tokens and route on the same inputs.

## Data (fresh: in no trace and no eval batch)

20 in-domain documents: skills 2nd..11th documents with index % 7 == 3 (the eval rule; the eval batch used only the
1st), papers 2nd..11th papers with paper % 50 == 25. Per document, from token 256 on (past the front matter): a
64-token prompt and the next 64 tokens forced (`Q4X_FORCE`). Built by `bench/flash-slim-prompts.sh OUT decode`.

## Arms (one gpu-wait job, one q4x process per prompt per arm, same binary)

A: full Flash. B: `Q4X_KEEP` = keep-v0.99. Order alternates per prompt (A then B, then B then A). Per run: `gen ms/token`
(decode loop only), `expert misses` / tokens, the argmax ids at every forced step, wall, RSS, MemAvailable, VRAM free.

## Bars and reports

1. Read-back: the B log names the keep count (20237 of 24576), the A log has no keep line; every run prints its ids.
2. Reported (exploratory, no perf claim: P4's 20-prompt median is met, but this is a mask, not a shippable model):
   median ms/token per arm and their ratio, misses per token per arm, top-1 agreement of B's argmax with A's over the
   1280 forced positions (a second, larger in-domain quality read than rounds 1-4), per-prompt rows.

## Prediction (frozen)

- Misses per token: B 10% to 25% below A (round 2 prefill, B-v0.99: 22529 -> 19614, -13%; decode keeps a smaller
  working set per step, so slightly more).
- Decode ms/token: B/A 0.95 to 0.99.
- Agreement of B with A over the 1280 forced positions: 92% to 96% (round 2 eval: 93.9%).

## Budget

GPU: 40 processes of about 4 s load + 128 steps at about 20 ms, under 6 minutes, one job.
