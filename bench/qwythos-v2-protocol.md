# Qwythos-9B-v2-MTP-Q6_K vs llama.cpp (teacher-forced agreement + tok/s)

Preregistered 2026-09-16, before any GPU run on this model, tree at `17707a7`. Binds to
`bench/PROTOCOL-RULES.md` P1-P6. Method = `bench/ornith-protocol.md` step 3a (teacher-forced
agreement) and step 3b (decode tok/s_gen, 20-prompt median) exactly, reused as-is; G1/G2/G3
and steps 3c/3d are out of scope for this lane (brief `briefs/2026-09-16-qwythos-v2-agreement.md`).
Question: does the q8-repacked bake `Qwythos-9B-v2-MTP-Q6_K-BARO-04867e2.gguf` decode correctly
against llama.cpp running the source GGUF natively, and at what speed?

## Why this model reuses Ornith's method unchanged

`~/Models/qwythos-9b-v2-mtp-q6_k/Qwythos-9B-v2-MTP-Q6_K.gguf` (bake
`Qwythos-9B-v2-MTP-Q6_K-BARO-04867e2.gguf`, same dir) is `qwen35`, the same architecture and
shape family as Ornith-1.5-9B and the Qwythos champion (H=4096, FFN=12288, 33 blocks,
NQH=16, NKVH=4, HD=256, VOCAB 248320, tokenizer sha matching Qwythos per
`exchange/lane-moe-report.md`), K-quant sourced (quant histogram: 185 Q6_K, 184 F32, 73 Q8_0,
no bf16 tensor to pack directly), so `tools/engine-pack.py --q8` takes the same
`dequantize_kquant` -> bf16 -> q8 path Ornith's protocol validated (G1: packed-q8 vs gguf-py
dequantise of the source K-quant bytes, PASS). This lane does not re-run G1/G2/G3: nothing about
the K-quant->q8 path or `BARO_FORCE` changed since Ornith's round, and the model-library lane
(2026-09-16, `~/Models/library/REPORT.md`) already produced this bake and flagged it
UNVERIFIED-correctness for exactly this reason, no llama.cpp agreement receipt exists yet. This
lane produces that receipt.

## Arms

- **ours**: `.work/engine` built from current `serve/engine.mojo` (`BARO_FORCE` already landed,
  no kernel change this lane), `BARO_PACK=.work/qv2/engine-pack-q8`
  (`tools/engine-pack.py Qwythos-9B-v2-MTP-Q6_K-BARO-04867e2.gguf .work/qv2/engine-pack-q8 --q8`,
  the same flag the bake itself was verified against), greedy (`BARO_SPEC=0`).
- **llama.cpp**: `~/llama.cpp/build/bin/llama-server -m Qwythos-9B-v2-MTP-Q6_K-BARO-04867e2.gguf
  -c 8192 -ngl 99 -fa on -b 2048 -ub 512 -t 8 -ctk q8_0 -ctv q8_0`, greedy (`temperature 0,
  top_k 1`), token-id prompt, its own native K-quant kernels. `-ctk q8_0 -ctv q8_0` (not f16-KV)
  deliberately, per the CLAUDE.md caveat this brief cites: llama.cpp's own f16-KV config fails
  its own f32 reference at 5/7 lengths on this family, so f16-KV is not a trustworthy reference
  config here and Ornith's already-validated q8_0/q8_0 config is reused instead of re-deriving one.
- Prompt set: `bench/mtp-prompts/*.tokens` (20 prompts), `GEN_N = 64`.
- Read-back before any number is read (P1), both arms:
  - **ours**: engine build 0 errors; `BARO_PACK` tensor count printed at load (bake's own
    `n_tensors` is 442, same as Ornith's pack: expect the same count printed); `BARO_FORCE` id
    count printed when set, must equal 64 per prompt.
  - **llama.cpp**: `/props` `ctk`/`ctv` both `q8_0`, `n_ctx` 8192; `/completion` response
    `prompt_n == N` (token count of the input) on every request.
  - **both**: llama-server down while `ours` holds the GPU and vice versa, throughout.

## Frozen predictions

**Agreement (step 3a).** Basis: Ornith-1.5-9B, same architecture, same K-quant->q8 chain, same
`BARO_FORCE` method, measured median 63/64 (98.4%), range 57-64. Against that anchor, the
CLAUDE.md caveat is specific to the Qwythos family, not Ornith: "llama.cpp's own f16-KV config
fails its own f32 ref at 5/7 lengths" on Qwythos is a documented identity fragility this model
carries that Ornith's round never tested for. That does not directly predict *this* check (we are
not using f16-KV, and the fragility was about llama.cpp's own internal KV-cache-dtype identity,
not about K-quant->q8 agreement), but it is evidence this family runs closer to the edge of
numerical agreement than Ornith did, so the band is wider and the point estimate lower than
Ornith's outcome, not copied from it.

**Prediction: 45-85% median agreement.** Falsifier: <25% median (one of the two paths doing
something qualitatively wrong, not just accumulating more rounding error) or >95% median (would
say the f16-KV fragility caveat has no bearing here at all, worth a note either way but not a
correctness problem). Not gated pass/fail per CLAUDE.md (no greedy-identity threshold is frozen
past 64 ids either way), reported alongside the read-back, same as Ornith's 3a.

**Decode speed, ours (step 3b).** Basis: the model-library bake's own one-prompt receipt
(`baro.hw.tok_s_gen_20p` embedded key, despite the name a single-prompt greedy number per the
bake's own `baro.hw.config`: "q8 pack, greedy, one-prompt") reads **80.82 tok/s_gen**, and
Ornith's independently-measured 20-prompt median on the identical q8 dot-loop infrastructure
(same shape class, same kernel commit family) was 80.78: the two numbers agree to within 0.05%
despite being different models and different measurement granularity, which is the strongest
available anchor. **Prediction: ours 72-90 tok/s_gen.** Falsifier: outside 55-105 (either a
regression the shared q8 path should not have, or the bake receipt was measuring something this
lane's harness does not reproduce).

**Decode speed, llama.cpp (step 3b).** Ornith's llama.cpp arm (Q4_K_M, roughly 4.83 bits/weight
average) measured 88.8 tok/s_gen, below its own predicted band, flagged in
`bench/ornith-protocol.md` as the protocol's own scaling assumption being the miss, not either
engine. This bake is Q6_K (roughly 6.56 bits/weight average), about 1.36x more stream bytes than
Q4_K_M at the same shape. Scaling Ornith's measured 88.8 by that byte ratio (not re-deriving from
Q8_0, learning from the prior round's miss): 88.8 / 1.36 = 65.3. **Prediction: llama.cpp 50-80
tok/s_gen.** Falsifier: outside 35-100, given P6: the byte-scaling argument itself was already
shown fragile once this round, so the falsifier stays wide rather than tight.

No ratio claim is frozen between the two arms.

## Not in this round

G1 (K-quant->q8 dequant fidelity) and G2 (`BARO_FORCE` no-op when unset), both already PASSED
in `bench/ornith-protocol.md` on the same code path, nothing touched since; G3 (kernel
self-consistency vs `model-ref.py`), same reasoning, not re-run; step 3c (MTP identity) and 3d
(chat smoke), out of scope per the brief, which asks for 3a+3b only. Any kernel change (none
needed, `BARO_FORCE` already exists). Requantising through any path other than `--q8`.

## Result (2026-09-16, engine sha `17707a7`, `bench/qwythos-v2-run.sh` into `.work/qv2/run/`)

Read-back (P1): pack build printed `packed 442 tensors` (matches Ornith's pack tensor count and
the bake's own `n_tensors: 442`, same tensor layout family); engine build 0 errors; `/props`
confirmed `n_ctx: 8192` (`ctk`/`ctv` were not present in llama-server's `default_generation_settings`
block this llama.cpp build returns, a receipt gap for the next round, same class Ornith's own
round left open for slot count; the `-ctk q8_0 -ctv q8_0` flags were passed and the server never
refused to start on them, which is weaker evidence than a printed value); `BARO_FORCE:` printed
the correct id count on every one of the 20 forced runs (64 for 19 prompts, 43 for `p17-summarize`,
matching how many tokens llama.cpp actually returned for that prompt, not a fixed 64); `prompt
tokens:` on the no-spec engine runs matched each prompt file's own word count (P1: `prompt_n == N`)
on all 20; llama-server down while `ours` held the GPU and vice versa throughout (sequential
curl-then-engine per prompt, same alternation pattern as `bench/ornith-run.sh`).

- **Step 3a (teacher-forced agreement vs llama.cpp): median 99.2% (63.5/64), range 90.6-100%
  (58/64 to 64/64).** This is ABOVE the frozen 45-85% band and past the 95% falsifier: the
  Qwythos f16-KV caveat's fragility does not carry over to this check, the K-quant->q8 chain
  agrees with llama.cpp's native K-quant path even more closely than it did on Ornith (98.4%).
  My own prediction was wrong, stated plainly rather than rounded down to fit the band; the
  falsifier existed to catch exactly this and it did its job. Six prompts hit 64/64; the low
  outlier is `p19-numbers` at 58/64 (90.6%), still nowhere near the 25% failure line.
- **Step 3b (tok/s_gen, 20-prompt median): ours 80.77 (range 80.38-80.96), llama.cpp 81.54
  (range 80.36-82.54).** `ours` is inside the predicted 72-90 band and matches the bake's own
  one-prompt receipt (80.82) to within 0.06%, the same cross-check tightness Ornith's round
  showed. `llama.cpp` landed just above the predicted 50-80 band (inside the 35-100 falsifier,
  so not a falsification) at **1.0096x llama.cpp/ours**, essentially at parity. The bit-width
  scaling argument (Q6_K carrying ~1.36x the stream bytes of Ornith's Q4_K_M, so llama.cpp should
  be markedly slower here) was wrong a second time in this protocol family: Ornith's own round
  already showed the same style of scaling argument overshooting (predicted 110-150, measured
  88.8). Two rounds now show llama.cpp's decode throughput on this 9B/33-layer shape landing
  within about 10% of `ours`' q8 arm regardless of the source quant format (Q4_K_M here, Q6_K
  there), which reads as a fixed per-token cost (attention, kernel launches) dominating over
  bits-per-weight bandwidth scaling at this model size, not a property of either engine's tuning.
  No further scaling prediction is made from this pattern without tracing it.

**Verdict: Qwythos-9B-v2-MTP-Q6_K decodes correctly through the K-quant->q8 engine pack.** Median
agreement (99.2%) is the highest recorded in this protocol family so far, and decode speed lands
within a few percent of llama.cpp running the same weights natively. The correctness gap the
model-library lane flagged (`~/Models/library/REPORT.md`, "UNVERIFIED-correctness") is closed.
`~/Models/library/models/qwythos-9b-v2-mtp-q6_k__Qwythos-9B-v2-MTP-Q6_K-BARO-04867e2.json` updated
(`correctness_verified: true`), `~/Models/library/INDEX.md` correctness table row updated, and
`README.md`'s model table gained a Qwythos-v2 row (`README.md:24`) since this result passes.
