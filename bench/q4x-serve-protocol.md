# q4x serve round: resident stdin mode, chunked prefill, default warm list, smaller cache margin (prereg)

FROZEN by the commit that adds this file, before any run of the new binary. Four engine changes in `serve/q4x.mojo`:

1. `Q4X_STDIN=1`: one process serves one request per stdin line (`TAG N_GEN TOK...`), engine state reset between
   requests, expert cache kept (port of `lane-latent-bridge` `6fc3a74`, stdin part only).
2. `Q4X_PREFILL=1` runs a single-sequence prompt as forwards of at most `Q4X_CHUNK` tokens (default 2048), state and KV
   continued across chunks the way `verify()` continues them; batch buffers sized `min(Q4X_TMAX, Q4X_CHUNK)`.
3. Default warm list: `FIRST_SHARD.warm` when it exists (here a link to `exchange/receipts/q4x-strata/r3a-warm-l01.txt`);
   `Q4X_WARM=0` turns it off.
4. `ECACHE_MARGIN` 1.5 GB -> 0.5 GB: the expert cache takes 1 GB more VRAM.

## Arms (one gpu-wait job, `bench/q4x-serve-check.sh`)

OLD = main binary built from `dea023d` (`.work/flash-slim/q4x`, sha256 95dcd9dfddc7e809). NEW = this round's binary.
Prompts: `bench/q4x-prompts/*.prompt` (20, frozen 2026-10-09), 64 generated tokens each.

- I1 identity: OLD vs NEW argv, `Q4X_DUMP=1` block sums of p01 + 4 tokens, warm off on both.
- I2 reset identity: NEW resident run ends with p01 again; its ids equal the run's first p01 and a fresh argv NEW p01.
- S1 per-request wall: OLD argv, one process per prompt, first 5 prompts. NEW resident, 20 prompts, one process.
- R1 NEW resident default (warm on, margin 0.5 GB). R2 NEW resident `Q4X_WARM=0`. R3 NEW resident warm on,
  `Q4X_ECACHE_GB=17` (about the old margin's cache). Per request: gen ms/token, misses/token, slots read back.
- C1 chunked prefill: one 8192-token single-sequence prompt (`.work/flash-slim/prompts/trace.ids`, 8 documents
  concatenated), `Q4X_TMAX=8320`, 16 generated tokens: NEW default chunk 2048 vs NEW `Q4X_CHUNK=8192` (one forward,
  the old behaviour). Final-norm hidden of all 8192 positions (`Q4X_HIDDEN_F32`) compared byte for byte.

## Bars

1. I1 block sums identical, I2 ids identical, C1 hidden bytes and ids identical. Any miss voids the round's speed reads.
2. Parameter read-back per arm: slot count line, warm line (present in R1/R3, absent in R2), chunk visible as cap in the
   expert cache line.
3. No minimum gain; every number reported against its prediction.

## Prediction (frozen)

- S1: OLD 12 to 35 s wall per request (load dominates); NEW 1.5 to 3 s per request after one 15 to 35 s load.
- R1 steady state (median of requests 2..20): misses/token 30% to 60% below request 1; gen ms/token 0.88 to 0.97 of
  the OLD argv median over its 5 prompts.
- R2 vs R1: request 1 misses/token 30% or more above R1's request 1; steady state within 5% of R1.
- R3 vs R1: R1 has 4% to 7% more slots; steady-state misses/token 2% to 8% lower in R1; gen ms/token within 2%.
- C1: hidden identical; chunked 4 to 7 ms/token, one-forward 30 to 70 ms/token (the cliff seen at 8 x 1024).

## Budget

GPU: one job, about 4 loads of 15 to 35 s, 3 resident arms of about 30 s decode, 5 OLD processes, the 8192 one-forward
arm about 4 to 9 minutes (last in the job, so a timeout loses nothing else). `--timeout 1800`.

## Amendment 1 (after run 1, before run 2; frozen by its commit)

Run 1 (`exchange/receipts/q4x-serve/run1-receipt.txt`) passed every identity bar. Two reads need action:

- C1 falsified change 2 as a default: one 8192-token forward ran 3.87 ms/token, 4 x 2048 chunks 5.98 (misses 22581 vs
  79483, each chunk refetches its experts). The cliff at 8 x 1024 is a multi-sequence effect, not a length effect.
  `Q4X_CHUNK` now defaults to `Q4X_TMAX` (one forward, the old behaviour) and stays as an opt-in for a TMAX whose batch
  buffers do not fit.
- R1 vs R3 (change 4): R1 had 6.6% more slots and 17% fewer steady-state misses, but 10% slower gen ms/token; R1 was
  the first resident arm, after five OLD processes, and R2 (same margin, warm off) ran 18.41. Run order is the
  suspected confound.

Run 2 (`MODE=margin`, binary rebuilt with the chunk default): I1 again, then R1 (default) and R3 (`Q4X_ECACHE_GB=17`)
resident in order R1 R3 R3 R1, GTT used read before and after each arm (the resident-pack GTT spill).

Prediction: R1/R3 gen ms/token 0.97 to 1.00, misses 0.80 to 0.92; GTT used does not grow by more than 0.1 GB in R1.
