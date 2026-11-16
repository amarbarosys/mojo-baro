# Prereg: model A on the engine, S2 agreement bar and S3 predictions (frozen 2026-11-16, before any agreement or speed run)

Decision from the boss (2026-11-16): A plain runs on the kq pack now. Then MTP runs on a `--q4` pack built from A (Q4_K, Q5_K, Q6_K and
Q8_0 sources dequantised and requantised to q4_0 by `tools/engine-pack.mojo`, the path TurboFCFusion Q4_K_M runs at 68.7 tok/s with MTP).
The kq MTP guard (`serve/engine.mojo:1766`) is not lifted; its size is reported in the lane report.

## What "agree" means

Teacher-forced agreement: llama.cpp greedy on the A GGUF produces 64 ids per prompt; our engine is forced along them
(`BARO_FORCE`) and counts its own argmax hits. Never greedy equality. Reference: llama.cpp 956591f, CPU (`-ngl 0`), because a GPU
reference differs from CPU on 6 of 8 prompts (`bench/dense-run.sh`). Prompts: the 20 `bench/mtp-prompts/p*.tokens`. Engine: this tree,
`BARO_SPEC=0`, `BARO_TMAX=16384`. A prompt counts as exercised when its 64 forced ids ran with no NOT-RESIDENT and no fail word (P18:
a verdict needs 20 prompts and at least 15 exercised).

## Arms and bars

| arm | pack | bar (all of them) |
|---|---|---|
| K | `--kq`, Q4_K/Q6_K raw, Q5_K/Q8_0 rounded to q8 | median >= 0.97 and min >= 0.90 against llama.cpp |
| Q | `--q4`, everything requantised to q4_0 | median >= 0.90, min >= 0.80, and median >= K median - 0.08 |
| control | K twice, same binary, prompts p03 p06 p08 p09 p17 | tells a near-tie flip from a real miss |

A prompt where the K control disagrees with itself is a near-tie prompt: it stays in the table, it leaves the min, and the report says so
(the 27B shape flips run to run on p03, p06, p08, p09, p17, `docs/BASELINE.md` turbokernel section). Q below its bar is reported as lossy,
not retuned: the bar does not move after the run.

Why these numbers: K keeps the file's own Q4_K and Q6_K bytes and only rounds Q5_K to q8, so it should sit near llama.cpp. Q requantises
the Q8_0 layers 0 to 3 and every Q6_K tensor down to q4_0, a real quality loss; 0.90 median is the line below which it is not a faithful
copy of A.

## S3 predictions (decode, 20 prompts, 64 tokens, TMAX 16384, same-job arms)

| arm | prediction | basis |
|---|---|---|
| A plain, kq pack | 28 to 34 tok/s | Turbo kq trunk 34.4 at 17.2 GB; A has more q8 bytes (Q5_K and Q8_0 become q8) |
| A plain, q4 pack, launch | 36 to 40 tok/s | Turbo q4 launch 38.15 |
| A MTP k=2, q4 pack | 55 to 70 tok/s | Turbo 62.4 to 68.7 at 68% acceptance |
| A MTP k=4, q4 pack | within 10% of k=2, acceptance per step lower | card default is 4, engine window is m = k+1 |
| B (Forge E2B) plain | not predicted: engine port not merged when this is written | |

Acceptance prediction: A's MTP head is the official head, the body is post-trained; llama.cpp measured 0.88 to 0.90 on the card, q4
requantisation lowers it: expect 0.60 to 0.75 per drafted token on the q4 pack.

## Amendment 1 (2026-11-16, before any agreement or speed run)

The kq pack of A is 23,544,193,024 B (Q5_K and Q8_0 become q8, which is 1.0625 B per element against 0.56 for Q4_K). `pack-fit
.work/nerkyor/pack-kq --tmax 16384 --used-gb 1.2` says NO: 25.22 GB needed, pool 24.55 GB, over by 0.66 GB. The K arm cannot run at 16k,
so the K bar, the K control and the third Q clause (median >= K median - 0.08) are void. The q4 pack is 17,204,711,424 B and FITS with
5.68 GB headroom. Q is judged against llama.cpp alone: median >= 0.90 and min >= 0.80. The prediction "A plain, kq pack 28 to 34 tok/s"
is void for the same reason. Q controls itself: Q twice, same binary, on p03 p06 p08 p09 p17.

Disclosure rule: any change to an arm, a bar or the prompt set after this commit goes into the report with its reason.
