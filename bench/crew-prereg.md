# Latent crew, experiment 1 (preregistered 2026-10-17, before any run)

Question: does a crew of 3 latent workers plus 1 speaker beat one model, and does the latent handoff beat a text handoff?
Models: the same OrcaSAQ-27B bake on all 4 engines. Weights shared over HIP IPC (`bench/ipcn-ab.sh`, 4 engines, 20.93 GiB measured). Workers A, B, C: SLOTS=2 lean engines, spec off. Speaker D: spec on.

## Structural facts (measured earlier, receipts in `~/Projects/hardware/AMDHQ/runs/latent-os/`)
- SSM recurrent state of two engines cannot be merged. Exact state handoff is sequential and one-to-one (E12 PASS, KV 79/120 = T 79/120).
- Raw HIDDEN injection failed (E8 round 5). The soft arm picks the greedy token on 19 of 20 rows, so it is nearly text.
- So the speaker can ingest the state of ONE worker. The other workers reach it as text. No exact 3-way latent merge exists.

## Task set (void conditions first)
- Generated, exact-scored tasks (variable tracking and multi-step arithmetic, length L tunable). The old `e8_tasks.json` saturates (math 20/20) and is not used.
- Step 0: single-model sweep over L. Pick the L where arm S scores 40 to 60 percent on 100 items. Outside the band: void, retune L.
- Relative gates need `T >= S + 3 pp`. Else void (a null handoff passes a relative gate).

## Arms (same prompts, same seed, one job)
| arm | what | rules out |
|---|---|---|
| S | one engine, one pass | baseline |
| SC | 3 workers solve alone, majority vote, no speaker | gain from 3x compute only |
| T | 3 workers write text thoughts, speaker reads all and answers | text crew |
| L | speaker ingests state of worker 1 (E12 path), plus text of workers 2 and 3 | latent handoff |
| Lx | as L, state taken from a different item | gain from any state, not the right one |
| Tn | as T, thoughts replaced by norm-matched noise text length | gain from extra tokens |

## Frozen predictions
1. S in 40 to 60 percent at the chosen L (step 0).
2. SC >= S + 3 pp. T >= SC (speaker adds over a vote): band 0 to +6 pp.
3. L within 3 pp of T (E12 says state equals text). Gain of L is speaker latency: time from last worker token to first speaker token, predicted below 0.5 of T.
4. Lx below L by at least 5 pp. If Lx matches L, the state carries nothing: void.
A miss means optimize, not drop.

## Gates
Identity of engine outputs per arm against a same-binary control (p03 p06 p08 p09 p12 p15 p17 vary run to run on 27B). Fail word read on every run. Every engine's import/export read-back line checked. Clean GPU (under 1.5 GB VRAM) before the job. Verdict computed by hand from raw counts, paired Wald interval.

## Cost (state before queueing)
Step 0: 1 engine, 100 items x 5 L values, short answers: about 10 GPU-minutes. Main run: 4 engines, 100 items x 6 arms: about 25 GPU-minutes.

## Amendment 1 (2026-10-18, frozen before the 20-item run; smoke runs smoke-m6d only)
The first design failed in smoke runs, so the setup changed. Predictions 1 to 4 above stand; only the instrument changed.
- Tasks: `tools/crew_tasks` mode mod1000 (three-digit products), L=6. Step 0 on one engine: L=3 15/20, L=4 13/20, L=6 8/20 (40%, lower edge of the band). S is re-measured inside the crew run; if S falls outside 40 to 60 percent the run is void.
- Workers: S-style step-by-step text (three role lines), cap 1500 tokens. Terse workers scored 0/20 each (p1-L12) and were dropped.
- Speaker: answers directly after the notes (prefill `ANSWER:`, 12 tokens). Single model S writes full reasoning (1500 tokens).
- Engines: speaker D and worker 1 on f32 KV with a 5376 window (state handoff carries f32 KV only). Workers 2 and 3 on the int8 KV engine, window 2560. Weights shared over IPC across both binaries (measured: loads, 23.8 GiB total).
- Identity gate: answer agreement, not token ids. Restored runs depend on the checkpoint position (T3 vs T2 ids equal 2/3, answers 3/3). Floor = T3~T2. Gate = L~T2 within the floor.
- Prediction 3 (latency) restated: speaker time L / T below 0.5 is the claim; smoke gave 0.91 on 3 items.
- Cost: about 40 GPU-minutes (S 1500 plus 3 workers 1500 tokens per item at roughly 75 tok/s total, 20 items).

## Amendment 2 (2026-10-18, frozen before the run): short decisive test of the handoff primitive
the maintainer: no more long tests. The 40 GPU-minute crew run (run1) was cancelled at start. `bench/crew-handoff.sh` replaces it for now: 2 engines, N=20 items, L=6, the worker note is the generator's exact worked solution (no worker text generated), speaker answers in 12 tokens. About 6 GPU-minutes.
Scope: this tests only whether a worker's result reaches the speaker through the state handoff. It does NOT test whether a crew beats one model; that needs the long run and is not claimed.
Predictions (frozen):
1. T0 (no note) at most 6/20. Tn (noise note) at most 6/20. Lx (state of another item) at most 6/20.
2. T (note as text) at least 18/20. Floor T3~T2 answer agreement at least 18/20.
3. L (note as state) answer agreement with T2 at least 18/20, and L correct at least 18/20.
4. Median wall time L at most T.
Void: T below 18/20 (the speaker cannot use even a text note). Verdict PASS needs 1, 2 and 3. Latency (4) is reported, not gated.

## Amendment 3 (2026-10-18, frozen before the run): protocol compliance repeat on fresh IDs, N=10, about 60 seconds
Skills applied: latent-os-bench, latent-transfer-eval (fresh held-out IDs with choices frozen, engine build identity, paired test), gate-authoring `latent-controls.md` (one result line per arm, fingerprint of the injected state), latent-bridge-design (fail closed on tampered state). Same script, seed 12 (fresh tasks), N=10. Run 1 (handoff2, seed 11, N=20) stays the primary result.
Bars scaled to N=10 (frozen): T and L correct at least 9/10; L~T2 and floor T3~T2 agreement at least 9/10; T0, Tn, Lx each at most 2/10; exact McNemar L vs Lx p below 0.01; tamper `pack` and `prefix_hash` each HTTP 409; fingerprint line (pos, bytes, prefix_hash) present for every L and Lx item; engine and pack identity in the arm file.
Arm 0 note (latent-os-bench rule): T0 sits at the floor by design here. This is a mechanism test, not an accuracy-gain test, so the mid-range rule for arm 0 is not applied; it applies to the crew accuracy run (single model 8/20 = 40% at L=6, step 0).
