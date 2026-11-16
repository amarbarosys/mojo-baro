# Report: nerkyor Qwen3.8-27B vs Forge-gemma4 E2B (lane nerkyor-vs-forge, 2026-11-16)

Brief: `docs/lanes/2026-11-16-nerkyor-vs-forge-brief.md`. Receipts: `exchange/receipts/nerkyor-vs-forge/`.
Skills loaded: qwen38-nerkyor-coder, model-bringup, bench-gates, gpu-lane-budget, mojo-baro, gguf-forensics.

## S1 (in progress): model A cannot run with MTP on today's engine

File A has no Q4_0 tensor, so the q4 pack refuses it, and the kq pack that takes it refuses MTP. A plain decode can run;
the two "A MTP" arms of S3 cannot, unless the engine guard is lifted.

File: `Qwen3.8-27B-Coder390-EfficientThink-Q4LynnStyle-Q8MTP.gguf`, 866 tensors, arch qwen35, 65 blocks (blk.64 = MTP layer),
sha256 `5e2a1784...f57eb`. The card hash in the skill belongs to the `-MTP.gguf` sibling, not this file, so the card check is not done.
Tensor-type table: `exchange/receipts/nerkyor-vs-forge/s1-type-table.txt` (gguf-sweep: 0 NaN, 0 Inf, verdict clean).

| finding | evidence |
|---|---|
| "Q4LynnStyle" is a mixed K-quant layout: Q4_K / Q5_K / Q6_K / Q8_0 body, Q6_K `output.weight`, Q4_K `token_embd`, BF16 `ssm_alpha/beta`. No Q4_0. | s1-type-table.txt |
| Layers 0 to 3 are Q8_0 throughout, middle layers Q4_K / Q5_K, `ffn_down` mostly Q6_K. | same |
| MTP layer blk.64: every 2D tensor is Q8_0, `nextn.eh_proj` included. | same |
| The q4 pack (`--q4`) needs every 2D weight Q4_0 and cannot take A. Not hand-patched, per the brief. | BASELINE "Q4_0 bakes serve a q4 pack" |
| `--kq` keeps raw Q4_K / Q6_K bytes. Q5_K and Q8_0 tensors are rounded to q8 (`kq_mixed` pack, megakernel forced off). | `tools/engine-pack.mojo:8,434`, `serve/engine.mojo:1755` |
| The engine refuses MTP on every kq pack: `kq pack: the MTP draft reads q4/q8 only; set BARO_SPEC=0`. | `serve/engine.mojo:1764-1766` |

Same gap hit TurboFCFusion Q4_K_M on 2026-11-02 (kq 34.4 tok/s, spec VOID).

Boss decision: plain first, then make `--q4` take A (option c). Option a (lift the kq MTP guard) is sized, not built.

### The kq pack of A does not fit, so the q4 pack is the only way to run A at 16k

| pack | bytes | pack-fit at TMAX 16384, 1.2 GB desktop |
|---|---|---|
| kq (`--kq`, Q5_K and Q8_0 rounded to q8) | 23,544,193,024 | NO, needs 25.22 GB in a 24.55 GB pool, over by 0.66 GB |
| q4 (`--q4`, everything requantised to q4_0) | 17,204,711,424 | FITS, headroom 5.68 GB |

Receipt `s1-pack-fit.txt`. The kq plain arm and the kq agreement arm are void. Option c needed no engine change: `tools/engine-pack.mojo` already
dequantises Q4_K / Q5_K / Q6_K / Q8_0 and requantises to q4_0 under `--q4`; only `tools/baro`'s auto-pick (all Q4_0) kept A out. The prereg carries
Amendment 1 for this, committed before any run (36c65a3).

### Option a, size only: MTP on a kq pack

`gemm_kq` (`serve/window.mojo:443`) issues one m=1 GEMV per row of A, so an m = 3 window reads every weight three times and speculation cannot pay.
`kernels/gemv_kq.mojo` has a multi-row kernel (`amar_gemv_kq_rows`, test `kernels/test_gemv_kq_rows.mojo`) that `gemm_kq` does not call. Lifting the
guard means: call it from `gemm_kq` for m <= 4 with a byte-exact test against the per-row path, give the draft head a kq-pack branch (`draft_q4`
is false there), then remove `serve/engine.mojo:1764-1766` and gate with teacher-forced agreement. Estimate L (about 150 to 400 lines, one kernel
integration and one gate round), unmeasured. For A it would also need a smaller kq pack: the 23.5 GB one does not fit, so it only helps models whose
body is Q4_K / Q6_K.

## S2: the q4 pack of A against llama.cpp (teacher-forced, bar frozen first)

Prereg `docs/lanes/2026-11-16-nerkyor-vs-forge-prereg.md` (9c068cc, Amendment 1 36c65a3). Reference: llama.cpp 956591f, CPU, greedy, 64 ids per
prompt (`s2-llama-cpu-ref.tsv`; p17 ended at 37 ids). Engine `501fff57...` built from tree 0046a27, q4 pack, `BARO_SPEC=0`, `BARO_TMAX=16384`,
one resident process, forced per request. Receipts `s2-agree-q4/`, `s2-agree-q4-control/`, `s2-agree-q4-falsifier/`.

Verdict: the q4 pack is a usable copy of A. Median 0.9375 (bar 0.90 passes). The raw minimum is 0.7812 on p09 against a 0.80 bar: it misses, and the
frozen near-tie rule decides it. The control (same binary, second process) moved p09 from 50/64 to 51/64 and p06 from 60/64 to 61/64, so both are near-tie
prompts and leave the minimum; the minimum over the other 18 is 0.8438 (p08, 54/64 both runs), which passes.

| prompt | agree | prompt | agree | prompt | agree | prompt | agree |
|---|---|---|---|---|---|---|---|
| p01 | 61/64 | p06 | 60/64 (control 61) | p11 | 60/64 | p16 | 60/64 |
| p02 | 63/64 | p07 | 62/64 | p12 | 61/64 | p17 | 32/37 |
| p03 | 57/64 (control 57) | p08 | 54/64 (control 54) | p13 | 58/64 | p18 | 64/64 |
| p04 | 63/64 | p09 | 50/64 (control 51) | p14 | 58/64 | p19 | 64/64 |
| p05 | 63/64 | p10 | 62/64 | p15 | 60/64 | p20 | 60/64 |

Disclosures: (1) the control ran p03, p06, p08, p09, not p17 (my glob left it out); p17 is 0.8649, above both bars, so no verdict moves. (2) The K arm and its
bars are void (kq pack does not fit). (3) The Q-versus-llama.cpp loss is the cost of requantising the Q8_0 layers 0 to 3 and every Q6_K tensor to q4_0; it is
not split by tensor type here.

## S3: decode speed (20 mtp-prompts, 64 tokens, TMAX 16384, resident arms)

A on its q4 pack with MTP runs at 68.03 tok/s at k=2, 1.61 times its plain decode; B decodes plain at 116.11 tok/s, 1.7 times A's best.

| arm | median tok/s_gen | read-back and notes | receipt |
|---|---|---|---|
| A plain, megakernel | 42.15 (spread 6.9%) | fail word 0, load 4.1 to 4.5; first attempt of this arm hit NOT-RESIDENT beside a co-tenant and is void (`first-plain-pair-mega-INVALID.out`) | `s3-plain-pair/` |
| A plain, launch path | 38.91 | ids differ from the megakernel arm on p09 only (near-tie prompt) | `s3-plain-pair/` |
| A MTP k=2, launch | 68.03 | acceptance 748 / 1038 = 0.721 per drafted token; spread 46% is prompt-to-prompt, not noise | `s3-spec-k2-k4/` |
| A MTP k=4, launch | 63.42 | acceptance 859 / 1608 = 0.534; load_max 8.1 in this arm; ids differ from k=2 on p03, p06, p09 (near ties) | `s3-spec-k2-k4/` |
| B plain (Forge-1 E2B F16) | 116.11 (min 110.95, max 116.85) | gemma4-port engine `db2333f3`, unmerged, its own pack and Gemma-tokenized prompt ids; ready tmax 16384 | `s3-b-forge/` |

Against the prereg: A plain on the q4 pack 38.91 (predicted 36 to 40), k=2 68.03 (55 to 70), k=4 within 10% of k=2 (it is 6.8% lower), acceptance 0.721 (0.60 to
0.75). The kq plain prediction is void (pack does not fit). Disclosures: all A arms ran one binary (`BARO_SLOTS=6` so k=4 fits, engine sha `6d4479ea`); the plain pair and
the spec pair are two jobs on a quiet box, not one job, so the 1.61x is a cross-job ratio; k=4 is slower because acceptance falls faster than the window grows.

## S4: task score and perplexity

Run with `.work/nerkyor/quality.sh` (receipt copy in `exchange/receipts/nerkyor-vs-forge/`), not `tools/baro eval`: that needs a BARO bake with `baro.run.*` keys, A has none
and B cannot be baked (no gemma4 arch in `tools/bake.sh`). The script reuses `bench/quality-task-ids.py` unchanged in method: 120 e8 tasks (100 math, 20 json), identical
prompt ids to both arms, thinking off. Our arm for A is the q4 pack with MTP k=2; for B the gemma4-port engine. The cap is 16000 tokens (the brief's floor); the harness default is
300.

| | ours (engine) | llama.cpp, same GGUF | finish reasons | receipt |
|---|---|---|---|---|
| A at cap 300 | 13 / 120 | 11 / 120 | ours 93 of 120 `length`, llama.cpp 90 of 120 `limit`: void, no information | `s4-A-cap300/` |
| A at cap 16000 | 83 / 120 (math 81, json 2) | 89 / 120 (math 87, json 2) | 120 `stop`, 120 `eos`, none void | `s4-A-cap16000/` |
| B at cap 16000 | 78 / 120 (math 76, json 2) | 78 / 120 (math 76, json 2) | 118 `stop` + 2 `length`; llama.cpp 118 `eos` + 2 `limit`: 2 void each | `s4-B-cap16000/` |

Reading it:
- A's q4 pack against llama.cpp on the same weights: minus 5.0 pp. 76 tasks correct in both, 7 only on ours, 13 only on llama.cpp; sign test on the 20 discordant tasks p = 0.26,
  so the loss from requantising to q4_0 is suggestive, not established at n = 120. Only 3 of 120 outputs are identical text (about 500 tokens each).
- B on our engine against llama.cpp: identical score, 82 of 120 identical outputs.
- A against B, both on our engine: 83 against 78. 69 correct in both, 14 only A, 9 only B; sign test p = 0.40. No significant task-score difference.
- json is 2 / 20 in all four arms, both models: the json tasks do not separate anything here; only the math score is informative.
- Thinking is off in this harness, so it measures A's direct answers, not the reasoning mode the card scores. The two B tasks that hit the cap are digit loops.

Perplexity (llama.cpp `llama-perplexity`, wikitext-2 test, 8 chunks of 512, the `bench/quality-run.sh` setting): A 6.4619 +/- 0.3443, B 23.4533 +/- 1.7315. The two models use
different tokenizers (248,320 and 262,144 tokens), so per-token perplexity is not a ranking between them. Our engine's own perplexity (`quality-ppl-run.py`) was not run: it
needs a bake, and the gemma4 engine has no logprobs wiring (Open items below). Receipts `s4-A-cap300/ppl-llama.log`, `s4-B-cap16000/ppl-llama.log`.

## S5: the table

VRAM is peak minus base from the same job (base read before the job), the S4 engine arm, TMAX 16384. tok/s per GiB divides the median by that VRAM.

| | A: nerkyor Q4LynnStyle (q4 pack) | B: Forge-1 E2B F16 |
|---|---|---|
| params | 27B class (qwen35, 64 layers, H 5120), plus a 0.42B MTP layer | 4.6B stored weights (9.27 GB f16, E2B with per-layer embeddings) |
| VRAM | 17.54 GiB (17,958 MiB) | 9.46 GiB (9,685 MiB) |
| ppl (llama.cpp, wikitext-2) | 6.46 (not comparable across tokenizers) | 23.45 |
| task score, our engine, /120 | 83 (69.2%); llama.cpp 89 | 78 (65.0%); llama.cpp 78 |
| decode tok/s, plain / best MTP | 42.15 / 68.03 (k=2) | 116.11 / none (no MTP head) |
| tok/s per GiB, plain / best MTP | 2.40 / 3.88 | 12.27 / none |
| agreement with llama.cpp (teacher-forced) | median 0.9375, min 0.78 (p09, near tie), the other 19 at or above 0.84 | min 100% over 17 prompts, 98.4% at 15k (gemma4-port report) |

One line per axis:
- Speed: B is 2.8 times faster than A plain and 1.7 times faster than A with MTP.
- Memory: B needs 54% of A's VRAM, which leaves room for a 131k window (6.5 GB at 131k on llama.cpp, `RUN.md`).
- Efficiency: B delivers 12.3 tok/s per GiB against 3.9 for A at its best.
- Task quality: A leads by 5 of 120 tasks, not significant (p = 0.40); json says nothing, thinking is off.
- Fidelity of our engine to the file: B matches llama.cpp exactly on tasks; A's q4 pack is 5 pp below llama.cpp, not significant at n = 120, with a lossy requantisation behind it.

## Open items and what was not done

- Option a (kq MTP guard) sized in S1, not built. A's kq pack does not fit at 16k anyway.
- A's reasoning mode (thinking on, 16000 cap) not run: at A's lengths (card P50 3k to 5k tokens) 120 tasks cost hours; a subset run is the next step if the boss wants it.
- Our engine's perplexity for A (needs a bake, `tools/bake.sh` + a reference id file) and for B (needs logprobs in `serve/gemma4.mojo`) not run.
- DFlash2 out of scope, per the brief. B is not merged: its numbers come from the gemma4-port lane's build.
- The dflash2 and gemma4 lanes shared the card during this lane; every timed arm above records its load and VRAM base in its receipt.

## Merge gate (2026-11-16)

- main merged into the lane (`docs/rules.md` conflict, both sides kept, 6b63808).
- `bench/preflight.sh` under a 24G scope: PASS preflight ad266257c48f. The first run failed `ci-checks` on `serve/test_tune.mojo`; that test passes on its own and
  `tools/ci-checks.sh` passes on a rerun, so the failure was transient (cause unverified; the lane has no `serve/` or `kernels/` diff against main).
- `./run-tests.sh` under gpu-wait (job mwj0quok2f4v, `RUN_ALL=1`, `STATE_ENGINE=.work/preflight/engine-dense`): run-tests PASS, all tests, rc=0, FAIL: 0 (log `.work/run-tests.log`).
