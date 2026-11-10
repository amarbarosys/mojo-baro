# Same-stint decode bar against llama.cpp (2026-10-12)

Preregistered before any run. Script `bench/llama-stint.sh`.

## Why
The README's "1.19x ahead" (130.7 vs 110.0, 2026-09-06) is the last same-stint ratio. Our dense champion has since
moved to 148.80 tok/s_gen (`c5034a4`, 2026-10-12) and llama.cpp has moved five weeks; 148.80 / 110.0 is not a receipt.

## Arms (one gpu-wait job, bench/clock-probe.sh around it)
- **ours**: `serve/engine.mojo` at HEAD (`-D BARO_MODEL` default qwen35; the only engine change since `c5034a4` is
  the grammar sampling path), pack `.work/engine-pack-q4` (built from the GGUF below, checked through its
  identity.json), `BARO_SPEC=0`, megakernel default, GEN_N 64, one process per prompt, KV f32 (the dense default).
- **llama1**: llama.cpp master `08ffba21e` (2026-10-12), built HIP gfx1100 with the flags of the existing build
  (ROCWMMA_FATTN, HIP_GRAPHS, NO_VMM, Release).
- **llama2**: llama.cpp `1744c6bde` (2026-09-08), the existing `~/llama.cpp-master` build: whether llama.cpp itself
  moved since the 110.0 bar.
- Both llama arms: `llama-server` fast config `-ngl 99 -fa on -ctk f16 -ctv f16 -b 2048 -ub 512 -t 8 -np 1`, the
  same token ids (`bench/mtp-prompts/p*.tokens`, 20 prompts) through `/completion`, n_predict 64 with
  `ignore_eos` (added after run 1: llama2 stopped at 48 on an EOS; ours always decodes 64), temperature 0,
  top_k 1, cache_prompt false, tok/s = `timings.predicted_per_second`, one untimed request first.
- Weights: `Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf` (Q4_0 both sides).
- Order: ours (20), llama1 (20), llama2 (20), ours-tail (first 5 prompts again, drift control).
- Read-back: power cap, vddgfx, load; engine sha; each server's build line, offloaded layers, flash_attn, KV type.

## Frozen predictions
- ours median: **146 - 151** tok/s_gen.
- llama1 median: **108 - 128** tok/s.
- llama2 median: **106 - 114** tok/s.
- ours / llama1: **1.15 - 1.38**.
- Drift control: ours-tail / ours on the same prompts **0.98 - 1.02**.

## Reading
- The headline is ours / llama1, same stint, 20-prompt medians; it replaces the README's 1.19x row.
- Drift control outside 0.98 - 1.02: the ratio is reported UNVERIFIED with the drift beside it, no rerun to clean up.
- Same-64-ids counts are printed for information only (greedy 64-token equality is not a valid identity gate:
  CLAUDE.md, llama.cpp f16 KV fails its own f32 reference).
- Not measured here: speculative decode (llama.cpp's MTP read 169.5 vs our 129.3 on 2026-09-06; our k=2 is 150.96
  since), prefill, other weights (q8, Q4_K_M).

## Run 1 (script faults, no result)
Stopped by its own checks: llama2 predicted 48 of 64 tokens on one prompt (EOS, no ignore_eos), the llama read-back
lines were NOT READ (this llama.cpp logs load lines only at -lv 4) and the clock probe sampled nothing (rocm-smi
not on the job's PATH). Fixed in the script; the full run is repeated, no arm number from run 1 is quoted.

## Result (run 2, 2026-10-12 09:44, job mv4y4omkak44, receipts `exchange/receipts/llama-stint/`)
Read-back: power cap 290 W, vddgfx -120 mV, load 4.90 at start; engine sha 7d262e53dbd1c5fa, pack from the Q4_0-pure
GGUF; both servers offloaded 34/34, flash_attn enabled, K/V f16; sclk median 3005 MHz over 123 busy samples.

| arm | median | spread | prediction | |
|---|---|---|---|---|
| ours | 148.89 tok/s_gen | 1.4% | 146 - 151 | HIT |
| llama1 `08ffba21e` | 111.34 tok/s | 28.5% | 108 - 128 | HIT (low end) |
| llama2 `1744c6bde` | 112.04 tok/s | 2.4% | 106 - 114 | HIT |
| ours / llama1 | **1.337** | | 1.15 - 1.38 | HIT, ours faster 20/20 |
| drift (ours-tail / ours) | 0.9977 | | 0.98 - 1.02 | HIT |

- llama1's spread is two consecutive prompts, p14 89.02 and p15 80.48 (every other prompt 110.4-112.5). The queue
  held only this job then; the cause is outside the queue and unknown. Medians do not depend on them (llama1 and
  llama2 agree within 0.6%).
- llama.cpp did not get faster between 1744c6bde and today's master on this model and config; the move from 1.19x
  (2026-09-06) to 1.34x is our engine (130.7 -> 148.89).
- Same 64 ids: 5/20 for both builds, information only.

## Run 3: q8 (preregistered 2026-10-26, before any run)
Why: the README's q8 row (68.8 vs 74.1, 0.94x, 2026-09-04) is the oldest row llama.cpp wins and was never a
same-stint ratio.
- Weights (amended before any run): the HF checkpoint carries no MTP block, and the pack needs one. Trunk =
  `convert_hf_to_gguf.py --no-mtp --outtype q8_0` from the bf16 HF checkpoint; `blk.32.*` (MTP) grafted from the
  Q4_0-pure GGUF (`.work/q8-stint/graft.py`, block_count 33, nextn_predict_layers 1). Both sides read the grafted
  file (`LS_GGUF`). BARO_SPEC=0 and llama.cpp without a draft never run the MTP block, so the trunk is the arm.
- ours: `serve/engine.mojo` at HEAD, pack `tools/engine-pack.mojo GGUF OUT --q8` (`LS_PACK`), checked through its
  identity.json; `BARO_TMAX=16384` (16K floor) read back from the engine log.
- llama1: the existing `~/Projects/imports/llama.cpp` build, fast config as above, `-c 16384` (16K floor).
- n_predict 64 with ignore_eos: fixed-length decode timing, the `bench/min16k.sh` fixed-ids class.
- Frozen predictions: ours **66 - 90** tok/s_gen; llama1 **70 - 82** tok/s; ours / llama1 **0.85 - 1.15**;
  drift **0.98 - 1.02**.

### Result (run 3, 2026-10-26, receipts `exchange/receipts/llama-stint-q8/`)
Read-back: power cap 290 W, vddgfx -120 mV, load 17.59 at start (another lane on the CPU); engine sha 24482b43143aecb2,
pack from the grafted Q8_0 GGUF; TMAX 16384 and f32 KV in every engine log; llama.cpp `ca3d5a3e1` offloaded 34/34,
flash_attn enabled, K/V f16, 16384 cells; sclk median 3006 MHz.

| arm | median | spread | prediction | |
|---|---|---|---|---|
| ours | 80.38 tok/s_gen | 9.9% (p01 72.79, the first run of the job; all others 80.14-80.73) | 66 - 90 | HIT |
| llama1 | 75.46 tok/s | 1.7% | 70 - 82 | HIT |
| ours / llama1 | **1.065** | | 0.85 - 1.15 | HIT, ours faster 19/20 |
| drift | 0.9947 | | 0.98 - 1.02 | HIT |

- The README's 0.94x (2026-09-04) is replaced: q8 moved 68.8 -> 80.38 on our side, 74.1 -> 75.46 on llama.cpp's.
- Same 64 ids 6/20, information only.

## Run 4: Ornith-1.5-9B from Q4_K_M (preregistered 2026-10-26, before any run)
Why: the README's Ornith row (80.8 vs 88.8, 0.91x, 2026-09) is the last row llama.cpp wins, from an older engine.
- Weights: `~/Models/ornith-1.5-9b-q4_K_M/Ornith-1.5-9B-Q4_K_M.gguf` (Q4_K 223, Q6_K 35, MTP block present); llama.cpp
  reads it as is.
- Two jobs, one per pack, each with its own llama1 arm: **4a** `engine-pack --q8` (the old row's arm, about 2x the
  weight bytes of llama.cpp's side), **4b** `engine-pack --q4` (closest bytes). Same engine binary, settings as run 3.
- Frozen predictions: 4a ours **76 - 86**, llama1 **84 - 100**, ratio **0.78 - 0.98**; 4b ours **138 - 154**,
  llama1 **84 - 100**, ratio **1.40 - 1.75**; drift **0.98 - 1.02** in both.
- 4b is a requantization (Q4_K/Q6_K -> q4_0 rules): its speed is a claim, its output quality is not measured here.

### Result (run 4, 2026-10-26, receipts `exchange/receipts/llama-stint-ornith/`)
Same engine (sha 24482b43143aecb2) and read-backs as run 3: TMAX 16384 in every engine log, llama.cpp 34/34
offloaded, flash_attn on, K/V f16, 16384 cells; load 3.65 (4a) and 6.33 (4b) at start, sclk median 2992 / 2994 MHz.

| arm | ours | llama1 | ours / llama1 | drift | predictions |
|---|---|---|---|---|---|
| 4a q8 pack | 80.48 (spread 7.6%) | 93.40 (1.0%) | **0.862**, faster 0/20 | 0.9976 | all HIT |
| 4b q4 pack | 146.93 (spread 17.5%) | 93.07 (1.1%) | **1.579**, faster 20/20 | 0.9987 | all HIT |

- Both spreads come from p01, the first run of each job (74.81 and 121.96); the other prompts sit within 1%.
- 4a: llama.cpp moved 88.8 -> 93.40 since 2026-09 and our q8 arm stayed at 80.5: the old row's gap is wider, 0.91x -> 0.86x.
  It reads twice the weight bytes; the same-bytes arm is 4b.
- 4b requantizes Q4_K/Q6_K to q4_0: speed only. Same 64 ids 0/20 (4a 3/20); its agreement is not measured here.

## Run 5: Ornith q4-pack agreement (preregistered 2026-10-26, before any run)
Teacher-forced agreement (`BARO_FORCE`) against llama.cpp's own greedy 64 ids (Q4_K_M as is, -c 16384, f16 KV,
ignore_eos), 20 prompts, script `bench/ornith-agree.sh`. Arms: **control** = run 4a's q8 pack (measured 98.4% median
in 2026-09, P14), **candidate** = run 4b's q4 pack; same engine binary.
- Gate valid only if the control reaches median >= 95.0%; else VOID, no verdict on the candidate.
- Candidate PASS: median >= 95.0% and min >= 85.0% over the 20 prompts; else FAIL.
- Frozen predictions: control median **96 - 99.5%**; candidate median **92 - 98%**, min **80 - 92%**.

### Result (run 5, 2026-10-26, job mvouq03g2b2f, receipts `exchange/receipts/llama-stint-ornith/agree/`)
| arm | median | min | prediction | |
|---|---|---|---|---|
| control (q8 pack) | 97.66% | 90.62% | median 96 - 99.5 | HIT; gate valid (>= 95) |
| candidate (q4 pack) | 92.97% | 79.69% | median 92 - 98, min 80 - 92 | median HIT, min MISS (p10-recipe 51/64) |

**Verdict FAIL** (bar median >= 95.0, min >= 85.0). The q4 pack loses 4.7 points of median agreement to the q8 pack
on every prompt but p16: requantizing Q4_K/Q6_K to q4_0 rules costs accuracy. Its 1.58x (run 4b) is speed only, not
a claim. Next: keep the file's own K-quant bytes on the qwen35 path (spark's native K-quant kernels, 2026-10-19) rather
than requantize.

## Run 6: Ornith native K-quant pack (preregistered 2026-10-26, before any run)
`engine-pack --kq` keeps the file's Q4_K / Q6_K rows; the engine reads them on the launch path only
(`window.gemm_kq` over `amar_gemv_kq`), megakernel, batched prefill and MTP draft off. Script `bench/kq-check.sh`:
- **R (regression, the routing change):** old engine sha 24482b43143aecb2 vs new, q8 and q4 packs, `BARO_MEGA=0` and
  default, 5 prompts each: generated ids equal 20/20 runs (ids only: the Bool -> kind change leaves every non-kq kernel
  and argument as it was).
- **Agreement:** `bench/ornith-agree.sh`, control q8 pack, candidate kq pack, same bar as run 5 (median >= 95.0,
  min >= 85.0). Prediction: candidate median **96 - 100%**, min **88 - 100%**.
- **Speed:** `bench/llama-stint.sh` with the kq pack. Prediction: ours **70 - 110** tok/s_gen, llama1 **88 - 98**,
  ratio **0.75 - 1.15**.

### Result (run 6, 2026-10-26, receipts `exchange/receipts/llama-stint-ornith/kq/`)
Engine sha a3b03a30b4ad94a3 (`715b6f0`). Load 37.25 (agreement) and 12.27 (stint) at start, another lane on the CPU.

| check | measured | prediction | |
|---|---|---|---|
| R regression | ids equal 20/20 | 20/20 | PASS |
| agreement candidate (kq) | median 98.44%, min 90.62% | 96 - 100, 88 - 100 | HIT, **gate PASS** |
| agreement control (q8) | median 97.66% | gate valid >= 95 | valid |
| ours (kq pack) | 96.64 tok/s_gen, spread 1.1% | 70 - 110 | HIT |
| llama1 | 93.25 tok/s | 88 - 98 | HIT |
| ours / llama1 | **1.036**, faster 20/20 | 0.75 - 1.15 | HIT |
| drift | 1.0036 | 0.98 - 1.02 | HIT |

The native K-quant pack agrees with llama.cpp better than our q8 pack does (same bytes as llama.cpp's side) and runs
1.036x llama.cpp on the launch path alone, megakernel off. The megakernel K-quant dot is the next lever.

## Run 7: Ornith kq pack on the megakernel (preregistered 2026-10-26, before any run)
Main `cf78faa` (merge `c12f003`): `engine-pack --kq` trunks decode on the m=1 megakernel by default (K-quant row dot
with two superblocks and two rows in flight, rms folded into per-block LDS). Engine `.work/engine` sha 59abf5b0f9edabbc,
same pack `.work/ornith-stint/pack-kq`, same llama-server (sha 5d954eb24804be38), same script `bench/llama-stint.sh`.
Identity: the megakernel gives ids equal 20/20 with the launch path (`exchange/receipts/mega-kq/mega11-check.out.txt`),
so run 6's agreement result (median 98.44%) carries over; no agreement re-run.
Read-back after the run, from every ours log: `BARO_MEGA: True`, `pack kq trunk: True`, `TMAX: 16384`.
- **Prediction:** ours **92 - 104** tok/s_gen (lane gate 99.55), llama1 **88 - 98**, ratio **1.00 - 1.12**, drift
  **0.98 - 1.02**.

### Result (run 7, 2026-10-26, receipts `exchange/receipts/llama-stint-ornith/mega/`)
Engine sha 59abf5b0f9edabbc, load 2.05 at start. Read-back: BARO_MEGA True, pack kq trunk True, TMAX 16384 in 20/20 ours logs.

| check | measured | prediction | |
|---|---|---|---|
| ours (kq pack, megakernel) | 99.53 tok/s_gen, spread 1.1% | 92 - 104 | HIT |
| llama1 | 94.14 tok/s, spread 1.7% | 88 - 98 | HIT |
| ours / llama1 | **1.057**, faster 20/20 | 1.00 - 1.12 | HIT |
| drift | 1.0031 | 0.98 - 1.02 | HIT |

The megakernel adds 3.0% over run 6's launch path (96.64), the same gain the lane gate measured (1.028x).

## Run 8: Ornith kq prefill baseline (preregistered 2026-10-26, before any run)
The kq pack replays the prompt one row at a time through the m=1 path (`replay rows: N-1`; heaptide, corewait and
wavewait: GPU-bound per row, the `gemv_kq` GEMVs ~84% of the cycles, `~/Brain/mojo/mojo-baro/2026-10-26-three-tools-kq-prefill.md`).
This run is the baseline for the batched K-quant GEMM round, not a claim. Script `bench/kq-prefill-stint.sh`: same
engine (sha 59abf5b0f9edabbc), pack and gguf as run 7, llama-bench pp at 1024 and 16384 (-ngl 99 -fa 1, 2 reps).
- **Prediction:**
  - ours 1024: **6 - 10 s**; ours 16384: **110 - 220 s**;
  - llama 1024: **0.25 - 0.45 s**; llama 16384: **5 - 12 s**;
  - ours/llama time: **15 - 40x** at both lengths.

### Result (run 8, 2026-10-26, receipts `exchange/receipts/llama-stint-ornith/prefill/`)
Engine sha 59abf5b0f9edabbc, llama-bench sha 12ce6210edc0d895, load 0.80 at start. Read-back passed on all 4 ours runs.

| length | ours (mean of 2) | llama-bench | ours/llama time | predictions |
|---|---|---|---|---|
| 1024 | 7.625 s | 0.300 s | 25.4x | 6-10 / 0.25-0.45 / 15-40: HIT |
| 16384 | 124.2 s | 5.711 s | 21.8x | 110-220 / 5-12 / 15-40: HIT |

Baseline for the batched K-quant GEMM round. Per prompt row: ~7.4 ms at 1K and ~7.6 ms at 16K, flat, so attention growth is
not the cost; the m=1 weight pass per row is.

## Run 9: kq batched prefill on dequant + WMMA (preregistered 2026-10-26, before any code or run)
Plan: prefill on kq packs through `pf_dequant_kq` (f16 scratch) + `amar_matmul_wmma_pipe` (Spark's path, `2c936ea`), the
m=1 GEMV for outputs below the WMMA tile. Not bit-exact with the replay (WMMA summation order), so:
- **Agreement gate** (fail-closed, before any timing): `bench/ornith-agree.sh` with PREFIX = `bench/prefill-prompts/p1024.tokens`
  on every one of the 20 prompts. Control = the same kq pack with `BARO_PREFILL=0` (replay); candidate = prefill on.
  PASS iff **candidate median >= control median - 1.0 pt AND candidate min >= 85%**. Read back: `prefill rows` > 0 on the
  candidate, `replay rows` = N-1 on the control.
- **Speed:** `bench/kq-prefill-stint.sh` at 1024 and 16384, same pack, gguf and llama-bench as run 8. Prediction: ours
  1K **0.4 - 1.2 s**, 16K **6 - 11 s**, ours/llama time **1.0 - 2.0x** at both lengths.
- **Decode unchanged:** generated ids equal 20/20 on mtp-prompts against main 59abf5b0; run 7 median within 1%.

### Result (run 9, 2026-10-26, lane `lane-kq-prefill`, receipts `exchange/receipts/kq-prefill/`)
Engine sha 96518fde0a4377a1 (lane, `2062cce`), pack and gguf as run 8, llama-bench sha 12ce6210edc0d895. Read-back on
every ours log: `pack kq trunk: True`, `TMAX: 16384`, `prefill rows: 1023 / 16383`, prompt length. The report is
`exchange/lane-KQ-PREFILL-report.md`.

| check | measured | prediction / bar | |
|---|---|---|---|
| speed 1024 | ours 0.288 s, llama 0.298 s, 0.97x | ours 0.4 - 1.2 s, 1.0 - 2.0x | MISS (faster than the band) |
| speed 16384 | ours 5.036 s, llama 5.689 s, 0.89x | ours 6 - 11 s, 1.0 - 2.0x | MISS (faster than the band) |
| agreement, PREFIX p1024 | candidate median 93.75, min 78.12; control median 93.75, min 78.12 | median >= control - 1.0 AND min >= 85 | median part PASS, min part FAIL on BOTH arms (p09, p16 at 50/64 in each) |
| decode unchanged | replay ids 20/20 = main, median 99.21 vs 99.36 (0.9985); prefill arm 20/20, 99.41 | ids 20/20, within 1% | HIT (run 7 median 99.53 within 1%) |
| dequant share (BARO_PROFILE=1, 1K) | 0.12 of the chunk | > 0.10 names the follow-up | follow-up: in-kernel K-quant decode in the LDS GEMM |

Speed stint `stint2/` ran alone; the first stint (`stint/`, 0.286 / 5.076 s) overlapped a run-tests job and is kept
as a receipt only. Run 8 to run 9: 7.625 -> 0.288 s (26.5x) and 124.2 -> 5.04 s (24.7x). The agreement bar's
min >= 85 was set on bare mtp prompts; with the 1024-token prefix the unchanged replay arm reads 78.12 on the same
two prompts, so that half of the bar does not test the change (same shape as P5b). The candidate's per-prompt counts
differ from the control on 4 of 20 prompts (+1, +1, +1, -1; 1178 vs 1177 of 1280).

**Bar amendment (2026-10-26, after the data, stated as such).** The min half is replaced by a control-relative min:
PASS iff candidate median >= control median - 1.0 AND candidate min >= control min - 1.0. Precedent: P5b's
forced-agreement bar, amended to aggregate-vs-control when the unpatched base failed the same floor
(`docs/CAPABILITIES.md`, `c24dd32`). Run 9 on the amended bar: median 93.75 >= 92.75, min 78.12 >= 77.12: **PASS**.
The original min >= 85 result stays recorded above as FAIL. `bench/ornith-agree.sh` with PREFIX now applies the amended bar.

## Run 10: K-quant decode inside the LDS GEMM's B staging (preregistered 2026-10-26, before any code or run)
Plan: drop the f16 scratch round trip of run 9 (`kq_dequant` + `kq_gemm`). The Q4_K / Q6_K unpack (the GEMV's
`kq_lane16` order) moves into the B-staging of the LDS-pipelined prefill GEMM, as `amar_matmul_prefill_q4` does for Q4_0
(`kernels/matmul_prefill.mojo`, `bench/prefill-protocol.md` R5). Run 9 measured dequant at 0.12 of the 1K chunk.
- **Kernel byte test:** the new kernel against run 9's dequant + WMMA path on real Ornith rows (Q4_K and Q6_K, m 1, 17,
  128, 1024), fp64 oracle `tools/kq-prefill-ref.py`, bar `2^-8 |ref| + 2^-19 sum |a w|` (run 9's oracle bar).
- **Agreement:** `PREFIX=bench/prefill-prompts/p1024.tokens bench/ornith-agree.sh`, control `BARO_PREFILL=0`, amended run 9
  bar: candidate median >= control median - 1.0 AND candidate min >= control min - 1.0.
- **Speed:** `bench/kq-prefill-stint.sh` at 1024 and 16384, same pack, gguf and llama-bench as runs 8 and 9, alone on the
  card (`--vram 22`). Prediction: ours **1K 0.22 - 0.30 s, 16K 4.2 - 5.2 s**; ours/llama time **0.75 - 0.95x** at 1K and
  **0.75 - 0.92x** at 16K. A same-job A/B against run 9's path (knob) gives the change's own ratio: predicted 1.03 - 1.15x
  at 1K.
- **VRAM:** the 142.6 MB of run 9 scratch goes away when the new path is the only kq route (predicted -100 to -143 MB).
- **Decode unchanged:** ids 20/20 on mtp-prompts against main; run 7 median within 1%.

### Run 10 result (2026-10-26, lane `kq-mmq`, `exchange/lane-KQ-MMQ-report.md`)
Engine `.work/engine` sha `a24fe08d50b51cbb` (lane `4810a11` + docs), main engine from `1055e6d`, pack `.work/ornith-stint/pack-kq`,
receipts `exchange/receipts/kq-mmq/`.
- **Kernel byte test:** `test_kq_prefill_gemm` + `tools/kq-prefill-ref.py` PASS, 12 cases, lds arm worst err/bar 0.50 on
  ref16 (run 9's wmma arm 0.50), lds vs wmma max 7.7e-5; the `ACC` instance doubles its own output bit for bit.
- **Agreement** (PREFIX p1024, control `BARO_PREFILL=0`): candidate median 93.75 min 78.12 = control 93.75 / 78.12,
  **PASS** on the amended bar; per prompt +1 +1 +1 -1 against the control (`agree/`), same on the knob build (`agree-p1/`).
- **Speed** (`kq-prefill-stint.sh`, alone on the card): **1024 tokens 0.259 s vs llama-bench 0.298 s = 0.87x; 16384
  tokens 4.539 s vs 5.662 s = 0.80x** (run 9: 0.288 / 5.036 s, 0.97x / 0.89x). Predictions 0.22-0.30 s, 4.2-5.2 s,
  0.75-0.95x, 0.75-0.92x: **all four inside the band**. Same-job A/B of the two routes at `8d60892` (`ab/`): 1024 tokens
  0.287 vs 0.297 s = **1.033x**, inside the 1.03-1.15x band at its floor; the dequant share measured alone on the card
  was 0.063 of the chunk, not run 9's overlapped 0.12, so the band's premise was inflated. The kernel then gained
  another 1.11x on the gate shape from the hoisted loads and the U = 8 unroll (`bench-kernel-*.txt`).
- **VRAM** (`kq-vram-check.sh`): after_alloc_bufs 15,944,912,896 -> 15,802,306,560 B (**-142.6 MB**), after_first_token
  -142.4 MB; predicted -100 to -143 MB.
- **Decode unchanged:** `bench/kq-decode-check.sh` (main arm `BARO_PREFILL=0`, since run 9 main prefills kq packs too): replay ids 20/20 equal main, median 99.83 vs 99.80 tok/s_gen (1.0003); prefill arm also 20/20, 100.01 (1.0022). **PASS**.
- `./run-tests.sh` PASS (`run-tests.log`, alone under `--vram 22`).
