# Item 3: MoE miss diagnostic preregistration

## Question

For RegesCore-35B (`qwen35moe`), how do expert miss fraction and miss bytes per
token differ between the fixed 47-token request and its 5,536-token context
request? Record VRAM free after load. The prior S4 speed comparison mixed context
and prompt effects, so this run is diagnostic and makes no speed claim.

## Frozen setup

- Model: `RegesCore-1.0-35B-UD-Q4_K_S-BARO-3c6c25c.gguf`.
- Inputs: `.work/decode-probe/{short,long}.req`, 256 generated tokens each;
  SHA-256 and prompt token counts go in the receipt.
- Run: `bench/decode-probe.sh` direct, serve, and harness arms; no optional grid
  binaries. Same requests and engine build across arms.
- Record per-request `expert misses / references`, MB/token, CPU expert rows,
  generated tokens, GTT, and loaded VRAM usage. Keep raw logs and the command in
  `.work/`.
- No code change in this diagnostic. LOC budget: 0.

## Prediction

Prior capacity-64 measurement: 77.43% hits and 0.1656 GiB misses/token. Predict
about 22.6% misses and 0.166 GiB/token in both contexts; short-to-long miss
fraction within 5 percentage points and bytes/token within 10%. Context can
change routing, so a miss is useful evidence, not a failed run.

## Stop and score

- Do not launch unless pack identity matches the frozen model and estimated GPU
  headroom is at least 1 GiB at `BARO_TMAX=8192`.
- Stop on engine error, nonzero mega fail word, GTT above 4 GiB, or the
  `gpu-wait` 1,500-second timeout. Retain partial logs and mark the result
  incomplete; do not retry the same tree.
- This diagnostic scores MISS if either metric is outside its prediction band,
  otherwise WIN. Neither verdict selects an implementation lever.
- After the counts, preregister one cheapest lever separately before its A/B.
  The A/B kill line is B/A above 0.95; the final gates remain `./run-tests.sh`
  and the `moe-persist` teacher-forced agreement bar.

## Addendum A: tier-mode correction, frozen before diagnostic rerun

The first probe used a full resident pack (`tier tensors: 0`), so it emitted no
miss counters. It is invalid for this question and contributes no measurement.
Use the verified tier split of the same frozen GGUF, with `BARO_TIER=64`,
`BARO_TIER_PINNED=1`, `BARO_TIER_ZC=1`, and `BARO_MEGA=0`. Run only the direct
arm with `MOE_MISS_ONLY=1`; tier mode rejects the persistent mega arm. Keep the
same short/long requests, token counts, prediction bands, stop lines, and
scoring above. The receipt will include the tier pack identity and split
verification output.

## Addendum B: cache-capacity A/B, frozen before timing

Diagnostic result at cap 64: the warmed short request missed 19,316/81,920
expert references (23.58%) and fetched 34,707,898,368 B over 256 decode steps
(0.1262 GiB/token). The warmed long request missed 15,234/68,160 references
(22.35%) and fetched 27,362,721,792 B over 213 steps (0.1196 GiB/token). Both
miss fractions are in band; miss bytes/token are below the preregistered
0.166 GiB prediction by more than 10%, so the diagnostic verdict is MISS.
`rocm-smi` during the loaded cap-64 run reported 11.75 GiB free.

Choose the cheapest available lever: raise the existing GPU expert cache from
64 to 128 experts/layer. `pack-fit` estimates 9.91 GiB headroom at cap 128;
the measured cap-64 free VRAM also covers the additional 4.58 GB cache. No code
change. This should reduce miss traffic and improve decode latency.

- A: `BARO_TIER=64`; B: `BARO_TIER=128`.
- One `gpu-wait` job, same engine/pack, same direct M0 mode and requests; run A
  then B. Compare the mean ms/token for the two warmed 5,536-token requests
  (requests 3 and 4 in each arm). Preserve all raw logs and token IDs.
- Prediction: B/A ms/token 0.90–0.95, with identical generated IDs.
- Kill/score line: stop after B if B/A ms/token > 0.95; score MISS. Score WIN
  only at or below 0.95 with token identity intact. A token mismatch fails the
  accuracy gate at any speed.
- If WIN, run `./run-tests.sh` and the `moe-persist` teacher-forced agreement
  gate (64/64 tokens on 20/20 prompts). If MISS, report the chosen lever as
  falsified; do not test another lever in this lane.
