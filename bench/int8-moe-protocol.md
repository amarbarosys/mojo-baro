# int8 KV on the MoE engine (qwen35moe): agreement gate, frozen 2026-10-09

Dense int8 KV passed A2S2 (`exchange/lane-A2S2-report.md`: min 96.9%, means 99.2-99.5% at 8k/16k/32k).
The MoE build (`-D BARO_MODEL=qwen35moe -D BARO_KVQ=int8`, Gemini 09-28 wiring) never ran a gate.

- Gate: `bench/a2-gate.sh` (teacher-forced, refcached reference), pack = RegesCore q8d (`tools/moe-pack-q8d.py`),
  tier mode `BARO_TIER=64 BARO_TIER_PINNED=1 BARO_TIER_ZC=1`, bar MIN_PCT 93.75 (A2S2's P14 bar).
- Iteration: QUICK=3 (32k set). Claim: full 60 prompts x 3 sets.
- Predictions: QUICK min >= 95%, mean 98% (MoE has 10 attention layers of 40, fewer int8 reads than dense);
  decode after 32k int8 / f32 in [0.98, 1.10], point 1.03 (tier mode is expert-bound, KV bytes are a smaller share).

## Result, QUICK=1 (2026-10-09)
Tier with the doorbell on: reference arm dies, "expert tier: doorbell for layer 0 never rang (5 s)" (also without ZC).
`BARO_TIER_BELL=0`: p01 at 32k, forced agreement 62/64 = **96.9%** (bar 93.75, prediction >= 95 HELD on n=1), decode after
32k 60.92 tok/s (int8 arm; the f32 ratio needs the full run). Full 60-prompt claim run: not yet.
## Result, full gate 2026-10-11 (resumable gate `203ce72`, doorbell fix `13e153f`, bell ON)
**FAIL at the frozen bar.** FAILFAST stopped the candidate at L16384 p11-email, forced agreement 59/64 = **92.2%** < 93.75.
Before it: L8192 20/20 prompts, min 93.8%, mean 98.3%; L16384 11 prompts, mean 98.4%. Prediction "min >= 95%" MISSED;
"mean 98%" HELD on the 31 scored. 29 prompts (L16384 p12-p20, all of L32768) not run. Decode medians are mixed-session
(a pause/resume run), not a claim. Receipts `exchange/receipts/int8-moe-2026-10-11/`.
Next, before any repair: the same-arm control (P5b lesson). Two f32 binaries differing only in a no-op build flag, same gate,
same bar: if f32-vs-f32 also dips under 93.75 on p11, the bar is unpassable for this set, not int8.
### Same-arm control 2026-10-11
f32 engine vs the same f32 binary (one trailing byte, sha d97259803b39), same gate, FAILFAST=0: **60/60 at 100.0%** at 8k/16k/32k,
p11-email 64/64 in every set. The gate path (state files, tier, doorbell) is deterministic, so p11's 92.2% is int8 KV's own effect,
not gate noise. Not answered: whether any small f32 numeric change (summation order) would also dip; dense int8 passed at min 96.9.
Verdict: MoE int8 KV FAIL stands; next = find where int8 loses p11 (per-layer KV error at 16k, which of the 10 attention layers).
### Where int8 loses (2026-10-11, `tools/state_diff.py --moe`, f32 vs int8 doc state at 16k, CPU)
No single bad layer. Each attention layer's own int8 error (quant-only: f32 K/V quantised per token row) is flat at 0.9-1.9%;
the measured K/V difference compounds with depth, 1.07% at layer 0 (equal to quant-only, so the decoder is right) to 7.9% K /
12.0% V at layers 6-8; conv 2.4%, SSM 1.2% carried. **Corrected 2026-10-11:** these came from numpy f32 norms, which undercount
over ~83M squares (K norm 11% low); f64 (`tools/state_diff.mojo`, oracle fixed): layer 0 1.070e-2, K 5.74% / V 5.52% overall,
per-layer values within 1% relative of the above. Conclusion unchanged. Offline, finer scale groups cut the per-step error only 1.4-1.9x (group 64:
0.77/0.83%, group 32: 0.65/0.67%, 4x/8x scale bytes). Levers to test: group-32 scales; f32 for the first attention layers
(the error source that every later layer inherits). Receipt `exchange/receipts/int8-moe-2026-10-11/state-diff-L16384.txt`.

## Round 2 prereg, frozen 2026-10-11 before any code or GPU run: group scales (`-D BARO_KVG`)
Change: one f32 scale per KVG values of a K/V row (today KVG = 256 = one per row; 256 stays the default and byte-identical).
Instrument: resumable `bench/a2-gate.sh`, MIN_PCT 93.75, tier env, ref = f32 `engine-moe`; speed `bench/ab-resident.sh SEQ=1`.
Sweep (one job, FAILFAST=0): KVG 64 and 32 on L16384 p01-p11 + L32768 p01-p05; winner gets the full 60.
Predictions (mechanism: per-layer quant-only error 1.0% -> 0.77% / 0.65% measured offline, compounding ~linearly in depth):
- KVG 32 full gate: min agreement >= 93.75 on all 60 (point 95.5); p11@16k >= 93.75. KVG 64: p11@16k in [92.2, 95.5].
- Decode after 32k, KVG32 / KVG256 in [0.97, 1.01], point 0.99 (attention is a small share of a tier token; extra scale loads).
- KV bytes per token vs KVG 256: KVG 64 +4.6%, KVG 32 +10.8% (row 260 B -> 272 / 288 B), read back from the engine echo.
Falsifiers: KVG 32 min < 93.75 -> fix 2 (f32 first attention layers); KVG256 build not byte-identical in ids -> layout bug.
Adoption: no minimum gain; passes gate + same-job A/B measured -> KVG becomes the int8 default.
### Round 2 result (2026-10-11, branch `lane-kvg`, prereg `78638d8`)
- KVG 256 build: all 176 kernel objects byte-identical code to the old int8 engine (ISA dump); full gate 60: p11@16k 92.2% (same as round 1), others >= 93.8; f32 rebuild vs old f32 60/60 at 100%.
- KV bytes/token (engine echo): 10400 / 10880 / 11520 at KVG 256 / 64 / 32: **+4.6% / +10.8% HELD.**
- Own error per layer (16k, f64 state_diff): K layer 0 1.07% -> 0.61% at KVG 32; deep drift K layer 8 7.9% -> 6.9%, V layer 5 9.2% -> 8.1%.
- Agreement, same 22 prompts (L16384 + L32768 p01-p11): **mismatched tokens 22 / 22 / 23 of 1408** at KVG 256 / 64 / 32.
  The failures move: KVG 64 and 32 fail p09@32k (59/64) which KVG 256 passed. **Prediction "KVG 32 min >= 93.75" MISSED.**
- Reading: a 43% cut of the per-layer source error cut the deep drift ~13% and moved agreement 0%. Agreement sits on a
  perturbation floor (~1.6% of forced positions flip) that does not scale with the int8 error, so fix 2 (f32 first layers)
  attacks the same source and is predicted to land on the same floor. Not tested: whether any small f32 perturbation (another
  summation order) also lands there; that is the cheapest falsifier before fix 2.
- Not landed: the KVG code (opt-in knob, no measured benefit). **Branch `lane-kvg` (`7d8db27`) KILLED 2026-10-12**: round 3 showed the
  agreement loss is the perturbation floor, so finer scales cannot move it; the branch stays only as an archive.

## Round 3 prereg, frozen 2026-10-12 before any GPU run: f32 perturbation floor (falsifier before fix 2)
Question: is the ~1.6% of forced positions that int8 flips a floor that ANY non-bit-exact change hits (near-ties in
the reference), or int8's own error? Round 2 cut the per-layer source error 43% and moved agreement 0 tokens.
Arms (no rebuild): candidate = the f32 reference binary plus one appended byte, run with a candidate-only knob
(`CAND_ENV`, `bench/a2-gate.sh`), reference records cached (ref `a5b795c979ab6cbd`, pack `7df0c2049970d4bd`).
- P1 `BARO_PF_SSM=chunk`: the dense chunk delta scan in prefill, pure f32 (no bf16 in `amar_ssm_delta_chunk_w`), a sum
  reorder in every SSM layer of the whole document prefill (the scan differs from the decode step in 87% of outputs).
- P2 `BARO_ATT_SPLIT=1`: split-K decode attention at every position, a reorder in the 64 forced positions only.
Scope: same 22 prompts as round 2 (`SETS="16384 32768" N=11`), FAILFAST=0, MIN_PCT 93.75, tier env. Read-back (P1):
`att split: 0` in the P2 receipts; P1 has no echo, so P1 counts as exercised only if its records differ from the
reference somewhere (any forced mismatch, or decode ids); a 0/1408 P1 with no other difference is NOT EXERCISED.
Predictions (int8 KVG 256: 22 of 1408 mismatched):
- P1: mismatches in [11, 44], point 18 (floor hypothesis: a reorder that drifts the 16k state lands near int8's count).
- P2: mismatches in [0, 22], point 5 (smaller perturbation scope: decode only, state untouched).
Decision rule: P1 >= 11 -> floor confirmed: the 93.75 min-over-60 bar measures the reference's near-tie density, not
KV quality; fix 2 is NOT built and the bar goes back to the maintainer with the control distribution. P1 <= 5 -> floor
falsified, the flips are int8's -> fix 2 (f32 first attention layers). 6..10 -> inconclusive, reported as such.
### Round 3 result (2026-10-12, job `mv4e9sf288dk`, prereg `2f8cdd1`)
Receipts `exchange/receipts/int8-moe-2026-10-12/`. Both arms: f32 KV echoed, cand sha 6c3af878, ref records reused.
| arm | mismatched of 1408 | prompts with a miss | min agreement | prediction |
|---|---|---|---|---|
| int8 KVG 256 (round 2) | 22 | 12 | 92.2% (p11@16k 59/64) | |
| P1 `BARO_PF_SSM=chunk`, f32 | **20** | 13 | 95.3% (p11@16k 61/64) | [11, 44], point 18: **HELD** |
| P2 `BARO_ATT_SPLIT=1`, f32 (`att split: 0` read back; default 1088) | **0** | 0 | 100% | [0, 22], point 5: held, point missed low |
- 9 of P1's 13 missing prompts are int8's too (16k p01 p02 p05 p10 p11, 32k p03 p05 p09 p11): the same near-tie prompts.
- A pure f32 sum reorder in the prefill scan, ~1e-7 relative per op, flips as many forced tokens as int8 KV (~1% per layer):
  agreement on this set does not scale with the perturbation once the 16k-32k state drifts at all. A reorder confined to the
  64 decode positions flips none, so the flips come from state drift over the long prefill, not from the decode step.
- **Decision (frozen rule, P1 >= 11): floor confirmed. Fix 2 is not built.** The 93.75% min-over-prompts bar sits one token
  from the floor's own worst prompt (P1 p11@16k 61/64; the bar allows 4 misses, int8 had 5). The bar goes back to the maintainer with
  this control. Decode medians are not a speed claim (one job, no A/B).

## Round 4 prereg, frozen 2026-10-12 before the control run: the bar against an f32 control (the maintainer's option 1)
Bar replaced (the maintainer 2026-10-12): the per-prompt min (93.75%) measured a prompt's near-tie density (round 3). New bar:
int8 KV passes if its total mismatched forced tokens over the 60-prompt gate are within noise of an f32 non-bit-exact
control on the same 60: **int8 <= C + 2 * sqrt(int8 + C)** (two-sigma, counts treated as Poisson), C = control total.
- Control: round 3's P1 arm (`BARO_PF_SSM=chunk`, f32, cand sha 6c3af878) on all 60 prompts (8k, 16k, 32k x 20),
  FAILFAST=0; its 22 recorded prompts are reused, 38 run.
- int8 side is ALREADY MEASURED and on record, stated here before C is known: KVG 256 full gate, **56 of 3840**
  (8k 22, 16k 14, 32k 20; `exchange/receipts/int8-moe-2026-10-11/kvg-g256-results.txt`). So the bar resolves to:
  int8 passes iff **C >= 37** (C = 37: 37 + 2*sqrt(93) = 56.3 >= 56; C = 36: 55.2 < 56).
- Prediction: C in [30, 80], point 50 (P1 on the 22 prompts: 20 vs int8 22; scale to 60: ~51).
- Pass -> MoE int8 agreement accepted; adoption still needs the speed A/B (`bench/ab-resident.sh SEQ=1`, decode after 32k,
  prediction [0.98, 1.10] from the 2026-10-09 prereg). Fail (C <= 36) -> int8's loss exceeds the floor; fix 2 is back on.
### Round 4 result (2026-10-12, job `mv4emip673z4`, prereg `0c3df4c`): int8 agreement PASSES the control bar
Control (f32, `BARO_PF_SSM=chunk`) on all 60: **C = 57 of 3840** (8k 20, 16k 18, 32k 19); int8 KVG 256 56 (22, 14, 20).
Bound C + 2 sqrt(56 + C) = 78.3 >= 56: **PASS.** Prediction C in [30, 80], point 50: HELD. The f32 control itself fails the
old per-prompt bar (p13-haiku@8k 57/64 = 89.1%, worse than any int8 prompt), which is why the gate exits 1. 24 prompts miss
in both arms, 12 only in the control, 5 only in int8. Receipts `exchange/receipts/int8-moe-2026-10-12/r4-*`.
Speed (adoption step, prediction frozen 2026-10-09: decode after 32k int8 / f32 in [0.98, 1.10], point 1.03): one job,
`bench/a2-gate.sh` SETS=32768 N=5 FAILFAST=0, four candidate arms ABBA (f32, int8, int8, f32), each a fresh one-byte copy
so no candidate record is reused; per arm the median forced-decode tok/s over the 5 prompts; ratio = int8 / f32 of the
pair means. Same tier env, same pack.
### Speed A/B result (2026-10-12, job `mv4eqkho4rus`, 620 s): UNDECIDED, one arm out of family, cause unknown
Per-prompt forced-decode tok/s after 32k (p01-p05), receipts `r4ab-*`:
| arm | p01 | p02 | p03 | p04 | p05 | median | prompt prefill s |
|---|---|---|---|---|---|---|---|
| f32a | 61.23 | 55.11 | 64.87 | 60.53 | 56.04 | 60.53 | ~0.30 |
| int8a | 66.44 | 59.23 | 70.39 | 65.38 | 60.47 | 65.38 | 0.29-0.37 |
| int8b | 38.97 | 48.34 | 32.52 | 34.05 | 53.86 | 38.97 | 0.36-0.63 |
| f32b | 58.18 | 54.69 | 64.36 | 60.25 | 56.07 | 58.18 | ~0.30 |
- int8a / f32a per prompt 1.075-1.085 (tight); f32b / f32a 0.95-1.00. int8b / int8a 0.46-0.89 (scattered): out of family.
- int8a and int8b did identical work (same tier hit counts per request, same agreement per prompt), so int8b's loss is
  outside the engine; its prompt prefill also doubled (host side). Not the queue (no overlapping job), not this session
  (one grep in the window), not the concurrent heaptide crash tests (those ran during int8a). No per-second load or clock
  record exists for the job, so the cause is not known and int8b is not explained away.
- Pair-mean ratio with int8b: 0.88; without it, int8a against both f32 arms: 1.10. The frozen band [0.98, 1.10] cannot be
  judged from this run. Verdict: UNDECIDED; adoption not claimed.
### Speed A/B rerun (2026-10-12, job `mv4gorxvwjed`, ABAB, host sampler `bench/host-sample.sh`): **1.077x, prediction band HELD**
Same method as above; arms f32c, int8c, f32d, int8d, fresh one-byte copies. Receipts `r5ab-*` (per-arm results, host.csv).
| arm | p01 | p02 | p03 | p04 | p05 | median |
|---|---|---|---|---|---|---|
| f32c | 61.23 | 54.95 | 64.55 | 60.41 | 56.25 | 60.41 |
| int8c | 66.60 | 59.20 | 70.22 | 65.25 | 60.42 | 65.25 |
| f32d | 61.65 | 55.21 | 64.87 | 60.78 | 56.52 | 60.78 |
| int8d | 66.55 | 59.19 | 70.28 | 65.24 | 60.35 | 65.24 |
- Repeat arms in family: f32d/f32c 1.005-1.007, int8d/int8c 0.999-1.001 per prompt. Host busy median 18-19% in every arm.
- int8 / f32 per prompt (pair means) 1.071-1.086, **median 1.077**. Prediction [0.98, 1.10], point 1.03: **band HELD**, point
  missed high (KV bytes 40960 -> 10400 per token mattered more at 32k than the expert-bound estimate assumed). The first run's
  int8a / f32a (1.075-1.085) matches; its int8b stays unexplained and is superseded, not explained away.
- GPU clock log: none (iTools gpu-live-log died on PATH inside the job, fixed in iTools `5e861ea`); host log complete.
**Verdict: MoE int8 KV passes agreement (control bar) and is 1.077x faster decoding after 32k.** Status as dense int8:
WORKS, DEFAULT-OFF (`-D BARO_KVQ=int8`); making it the MoE default is a separate call.
