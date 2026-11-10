# Quality rows: perplexity + task eval vs llama.cpp, per served model

Preregistered 2026-09-16, before any GPU run, at the lane-quality worktree tree
(`.work/wt/quality`, branch `lane-quality`, base `b7969eb`). Binds to
`bench/PROTOCOL-RULES.md` P1-P6. Brief: `briefs/2026-09-16-quality-rows-lane.md`.
Question: for every served model, does the engine's perplexity and small-task
accuracy agree with llama.cpp running the same weights, within a band set from
that model's own existing teacher-forced agreement receipt?

Answers project-review item 4 (`~/Brain/mojo/mojo-baro/2026-09-16-project-review.md`):
correctness today is measured as agreement with ourselves and llama.cpp, never
as task quality. This adds one quality axis (does the model answer questions
right) alongside the existing identity axis (does the engine reproduce the
model's own distribution).

## Scope: 10 models, not "11 + 1"

The brief names "the 11 library-verified bakes ... plus RegesCore-35B MoE".
`~/Models/library/INDEX.md`'s own correctness table is the cited source of
truth and it lists 13 `-BARO-` entries, of which 3 are excluded (2 `FAIL`
Spark-Q4 bakes, 1 `superseded` duplicate of the Qwythos champion) and MiniCPM5
is separately `BLOCKED` (no tokenizer, brief already excludes it). That leaves
**10** distinct valid bakes: RegesCore is already one of them, not a separate
11th on top of 11 others. Reconciled here rather than asked: the discrepancy
is a miscount against INDEX.md's own table, not a scope decision.

| # | model | family (arch) | gguf for llama.cpp | agreement basis (source) |
|---|---|---|---|---|
| 1 | Llama-3.2-1B-Instruct-Q4_K_M | llama | `~/Models/llama-3.2-1b-instruct-q4_K_M/Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 95.3-100%, 20/20 prompts (README) |
| 2 | lily-cybersecurity-7b-v0.2-Q6_K | llama | `~/Models/lily-cybersecurity-7b-v0.2-q6_k/lily-cybersecurity-7b-v0.2-q6_k.gguf` | 96.9-100%, 20/20 (README) |
| 3 | Qwen2.5-7B-Instruct-Q4_K_M | qwen2 | `~/Models/qwen2.5-7b-instruct-gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf` | 96.9-100%, 20/20 (README) |
| 4 | Qwen2.5-Coder-7B-Instruct-Q4_K_M | qwen2 | `~/Models/qwen2.5-coder-7b-instruct-q4_k_m.gguf` | same family as #3, no separate receipt found |
| 5 | Granite-4.2-3B-BF16 | granite | `~/Models/granite-4.2-3b-bf16/granite-4.2-3b-BF16.gguf` | 98.4-100%, 20/20 (README) |
| 6 | Ornith-1.5-9B-Q4_K_M | qwen35 | `~/Models/ornith-1.5-9b-q4_K_M/Ornith-1.5-9B-Q4_K_M.gguf` | 98.4%, 20/20 (README, `bench/ornith-protocol.md`) |
| 7 | Qwythos-9B-v2-MTP-Q6_K | qwen35 | `~/Models/qwythos-9b-v2-mtp-q6_k/Qwythos-9B-v2-MTP-Q6_K.gguf` | 99.2% median, range 90.6-100% (`bench/qwythos-v2-protocol.md`) |
| 8 | Qwythos-9B-Claude-Mythos-5-1M-MTP-BF16 (champion) | qwen35 | BARO bake itself (original deleted, superseded-bake cleanup 2026-09-16; tail-identity checked equal at bake time) | established, no single numeric median found in Brain/README, treated as the family baseline (#6/#7), not independently tighter |
| 9 | RegesCore-1.0-35B-UD-Q4_K_S | qwen35moe | BARO bake itself (original deleted, same cleanup) | 53.20/64 = 83.1% mean, 20 prompts (README, `bench/moe-persist-protocol.md`) |
| 10 | Spark-X2.5-4B-Q8_0 | spark2_5 | BARO bake itself (original deleted, same cleanup) | none found (README lists Spark with no agreement %), treated as unestablished |

Engines: `serve/engine.mojo` for qwen35/qwen35moe (#6-9), `serve/spark.mojo`
for llama/qwen2/granite/spark2_5 (#1-5, #10). Reference: llama.cpp ROCm build
`~/llama.cpp/build/bin/` (HIP backend, gfx1100, same as `bench/dense-run.sh`).

For #8/#9/#10 the pre-bake original GGUF no longer exists on disk (whiteboard
2026-09-16: "Superseded bakes deleted by the maintainer ... 5 files, tail-identity
checked first"). llama.cpp loads the `-BARO-<sha>` file directly for these
three; the self-describing bake is a strict superset of the original tensors
(extra `baro.*` KVs only, `gguf-verify` **PASS** on each), so "same GGUF" for
the arm-identity requirement holds by construction. First read-back on each
of these three arms includes confirming llama.cpp's own KV parse did not
choke on the added KVs (props/log, not assumed).

## Item 1: instrument

### Perplexity

**Text**: `~/Models/quant-lab/wikitext-2-raw/wiki.test.raw`
(WikiText-2 test split, Salesforce/MetaMind, CC BY-SA 3.0; fetched by
`~/Models/llama.cpp/scripts/get-wikitext-2.sh`, the same script and source
llama.cpp's own perplexity docs use). `sha256sum`:
`173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08` (1,290,590
bytes, 4358 lines). Committed by reference (path + sha), not copied into the
repo: it is a 1.2 MB third-party corpus already vendored under `~/Models/`.

**Context and chunking, identical both arms**: `-c 512`, `--chunks 8`
(non-overlapping, `--ppl-stride 0`, llama.cpp's default sliding-off mode),
the first part of the file tokenized by each model's OWN tokenizer, so token
count varies per tokenizer and is recorded per arm per model, never assumed
equal. Per chunk: 512 tokens, NLL summed over positions 1..511 (511
predicted tokens; position 0 has no left context in either arm's
convention), 8 chunks = 4088 predicted tokens total per arm per model. PPL =
exp(sum_NLL / 4088).

**ours**: `serve/engine.mojo` (qwen35/qwen35moe) or `serve/spark.mojo`
(the rest) under `BARO_SERVE=1`, one resident process per model, one request
per chunk: `{"id":N,"prompt":[first_token],"n":511,"spec":false,
"top_logprobs":1,"force":[the other 511 reference ids]}`. `top_logprobs":1`
is the trigger, not the data source: it routes the T<=0 decode through
`window.mojo`'s sampling branch (`amar_sample_row`, argmax-equivalent at
T<=0, P-K2) instead of the megakernel/`argmax_k` path, which is the only
branch that dumps the full pre-penalty logits row (`BARO_DUMP_LOGITS_DIR`,
`window.mojo:1534-1544`, landed today in `4c6d728` items 3-4). `force`
teacher-forces every position to the reference token regardless of the
model's own argmax (`serve/PROTOCOL.md`, `bench/ornith-protocol.md`
amendment), so what gets dumped at step `hn` is the model's actual
next-token distribution over the true prefix, before it is overwritten:
exactly the forced-token logprob perplexity needs. No engine edit here,
every piece (per-request `force`, `top_logprobs`, `BARO_DUMP_LOGITS_DIR`) is
an existing flag/field that landed before this protocol. `BARO_SPEC=0`
pinned (P1: `BARO_FORCE`/per-request `force` requires no-spec, engine raises
otherwise). Host script (`bench/quality-ppl-run.py`) drives stdin/stdout one
request at a time, never pipes all requests at once: send request, block
until that id's `"done":true` line, THEN read+score `$DUMP/row-0.bin`
through `row-510.bin` (raw VOCAB x f32, pre-penalty, `log_softmax` computed
host-side in numpy against the actual chunk token at `hn+1`), then clear
`$DUMP` before the next chunk's request. The dump directory is a single
fixed path for the whole resident process (`BARO_DUMP_LOGITS_DIR` is read
once per decode step, not per request) and every request's `hn` restarts at
0, so an unread or uncleared dump from chunk K would be silently overwritten
by chunk K+1's rows if the two ever raced; serial request/response with an
explicit clear between is what rules that out.

**llama.cpp**: `llama-perplexity -m GGUF -f wiki.test.raw -c 512 --chunks 8
-ngl 99 -fa on -ctk f16 -ctv f16 -t 8`, same flags `bench/dense-run.sh` uses
for the decode arm. Reads its own `Final estimate: PPL = X +/- Y` line.

**Verdict basis**: `predicted_ppl_ratio = ours_PPL / llama.cpp_PPL`, band set
per model from the agreement basis column above.

### Task eval

**Set**: `bench/data/e8_tasks.json`, already in this repo, 120 items (100
`math` from GSM8K, MIT license, via `tools/gsm8k-parquet-to-jsonl.py`; 20
`json` schema-extraction, synthetic/in-house). Exact-match scoring already
defined and reused unmodified from `bench/e8_score.py`
(`extract_last_int`/`parse_json_obj`/`subset_match`): math scores on the
last integer in the (stripped) response, json on an exact dict match against
`expected`. `bench/quality-task-ids.py` imports these functions directly
rather than redefining them, so the two eval paths (E8's raw-dump harness,
this one's live HTTP harness) never diverge on what "correct" means.

**Prompts, identical both arms**: `messages = [{"role":"system","content":
SYS}, {"role":"user","content": task["prompt_text"]}]`, `SYS` extracted once
per task type from `e8_tasks.json`'s own `full_prompt` (the text between
`<|im_start|>system\n` and `<|im_end|>`, same for every item of that type),
POSTed to each engine's own `/v1/chat/completions` (`serve/PROTOCOL.md`;
llama-server's OpenAI-compatible route) with `temperature:0`,
`max_tokens:300`. Each engine renders `messages` through its OWN loaded
tokenizer's chat template (minijinja on our side, llama.cpp's own Jinja on
theirs): "identical chat template" here means both sides render the SAME
model's template, not a shared literal string. A prompt-token-count mismatch
on the first task of a model is read back and logged as a receipt, not
assumed away.

**Scoring pre-processing** (before `e8_score.py`'s functions, python,
`bench/quality-task-ids.py`): strip `<think>...</think>`, strip a single
outer triple-backtick fence keeping its contents; for `json` tasks, extract
the first balanced `{...}` object. Same function, same order, both arms.

**Verdict basis**: `delta_pp = ours_exact_pct - llama.cpp_exact_pct` (also
reported per type, math/json, mirroring `e8_score.py`'s table), band set per
model from the agreement basis column.

## Item 2: predicted bands (frozen, this commit)

| # | model | PPL ratio band (ours/llama.cpp) | task delta band (pp) |
|---|---|---|---|
| 1 | Llama-3.2-1B | 0.90 to 1.10 | +/-8 |
| 2 | lily-7B | 0.90 to 1.10 | +/-8 |
| 3 | Qwen2.5-7B | 0.90 to 1.10 | +/-8 |
| 4 | Qwen2.5-Coder-7B | 0.90 to 1.10 | +/-8 |
| 5 | Granite-4.2-3B | 0.90 to 1.10 | +/-8 |
| 6 | Ornith-1.5-9B | 0.90 to 1.10 | +/-8 |
| 7 | Qwythos-v2-MTP-Q6_K | 0.85 to 1.15 | +/-10 |
| 8 | Qwythos champion BF16 | 0.85 to 1.15 | +/-10 |
| 9 | RegesCore-35B MoE | 0.60 to 1.40 | +/-20 |
| 10 | Spark-X2.5-4B | 0.75 to 1.25 (no basis, informational) | +/-15 (no basis, informational) |

Basis: models with a 95% or higher teacher-forced identity receipt get the
tight band (0.90-1.10 PPL, +/-8pp task); the two MTP/qwen35 models with a
single-digit percent of non-agreeing positions but no dense-family-wide
receipt get one notch wider (0.85-1.15, +/-10pp); RegesCore's 83.1% mean
agreement (the weakest receipt in the repo, `README.md` line 26) gets a band
wide enough that a real defect, not the sampler-path noise already known to
sit at the 53/64 level, is what fails it. Spark has no existing receipt at
all, so its row is explicitly informational: a Spark result outside its band
updates the band and flags "no basis", it does not by itself read as a
defect the way a tight-band model's miss would.

**PASS rule**: PPL ratio inside its band AND |delta_pp| inside its band.
Either alone outside band gives a **MISS**, reported with both numbers, not
silently averaged into a pass. A MISS is not automatically an engine defect:
CLAUDE.md's read-the-receipt-first rule applies here too, confirm arm
identity (which gguf, which tokenizer, prompt token counts matched) before
proposing a cause.

## Item 3: run order and read-backs

Per model, in order: build engine + pack (`bench/quality-build.sh`), then
perplexity ours (8 requests, one resident process), then perplexity
llama.cpp (one `llama-perplexity` run), then task eval ours (120 HTTP
requests, `baro-serve`), then task eval llama.cpp (120 HTTP requests,
`llama-server`), then score, then write `.work/quality/<model>/result.json`,
then delete that model's pack and dump scratch (disk floor: 40 GB free on
`/home`), then the next model.

Read-backs recorded per arm per model (P1): our engine's printed
`BARO_SPEC:`, `prompt tokens:`, `TMAX:`/`kv dtype:` lines; llama.cpp's
`/props` (task eval) or startup log (`n_ctx`, backend) for perplexity;
dumped-row count per chunk (must equal 511, a short count is a silent
truncation, not a smaller valid sample); HTTP prompt token counts from both
`timings`/`usage` fields on the first task of each model.

All 10 rows, agree or not, get recorded (item 3 of the brief: "whether or
not they agree"). GPU via `gpu-wait run --vram <GB> --priority 20 --`,
whiteboard head and `gpu-wait list` checked before every launch.

## Amendment (mid-lane, coordinator-approved 2026-09-16): spark perplexity deferred

Live on Llama-3.2-1B: `serve/spark.mojo` at this worktree's base has no
`top_logprobs`/penalty wiring at all (`sample.temperature > 0` is the only
branch that samples; a `top_logprobs:1` request is silently accepted and
decodes greedy, 0 rows dumped -- confirmed, not theorized). `bench/quality-
build.sh` was also rebuilding from the gguf's *embedded* `baro.kernel.commit`
(every current bake predates today's sampling-lane logprob landing), which
would have hidden the same gap behind a stale-build symptom even on the
dense path; fixed to build `serve/engine.mojo` from this worktree's current
HEAD instead of the bake's own history.

Coordinator decision (asked as item 1's stop condition): both continue now
and wait. This round: **task eval on all 10 models**, **perplexity on the 4
dense-family models only** (Ornith, Qwythos-v2, Qwythos champion, RegesCore
-- `serve/engine.mojo`, confirmed working). The 6 spark-family rows (Llama-
3.2-1B, lily-7B, Qwen2.5-7B, Qwen2.5-Coder-7B, Granite-4.2-3B, Spark-X2.5-4B)
get task eval only this round, perplexity marked **"perplexity pending"**
(not `UNVERIFIED`, not a `MISS`) in the draft table and in
`result.json` (`ppl_ours: null`). Sampling lane `a lane pane` is wiring
`top_logprobs`/`amar_topn_probs` into `spark.mojo` after its current fix (at
main `02cab13`, ahead of this worktree's base, spark already returns 400 on
`top_logprobs` rather than silently ignoring it); once that lands, this
worktree rebases and the 6 spark perplexity rows run under this same
protocol and bands table, no new brief. `BARO_DUMP_LOGITS_DIR` is being
moved out of `window.mojo` into the harness by that lane's own review; same
env var name, `bench/quality-ppl-run.py` needs no change, confirm after
rebase rather than assume.

`bench/quality-bands.json`'s spark rows and predicted bands stand unchanged
for when their perplexity runs; only the run order and this round's BASELINE
table cell (`perplexity pending`, not a verdict) change.

## Amendment 2 (2026-09-16 eve, before any scored run; lane taken over by the main session after the 14:33 OOM kill)

Found while resuming, before any row was scored (the only prior GPU use is the Ornith pilot, which
VOIDed at the dump step):

1. **Scoring window matched to llama-perplexity.** `tools/perplexity/perplexity.cpp` scores only the
   second half of each chunk: `first = n_ctx/2`, logits at positions 256..510 predicting tokens
   257..511, 255 tokens per chunk, 2,040 over 8 chunks, and replaces each chunk's token 0 with BOS when
   the vocab adds BOS. The frozen text above ("positions 1..511, 4088 tokens") would have compared a
   full-window NLL against a second-half NLL and biased the ratio upward. Ours now scores exactly the
   same 2,040 positions from the same chunk tokens.
2. **One token stream for both PPL arms.** The wiki text is tokenized once per model by a CPU-only
   `llama-server -ngl 0` on the reference GGUF (`/tokenize`, `add_special: true`), chunked, BOS
   substituted exactly as perplexity.cpp does; ours receives those ids. Read-back: first 8 ids of chunk
   0 and the token count, per model.
3. **Task eval: identical token ids, one GPU arm at a time.** Our Rust server has no tokenizer for 6 of
   the 10 models (no `tokenizer.json`), and the committed harness ran both servers on the GPU together
   (a 35B MoE plus llama-server does not fit in 24 GB). Now: the CPU llama-server renders each task
   through the model's own chat template (`/apply-template`, Jinja) and tokenizes it (`/tokenize`,
   `add_special: false`, `parse_special: true`); both arms receive those ids at T=0, max 300 tokens.
   Ours: the engine binary over its stdin protocol, `stop` = the model's end-of-generation ids (EOS
   and, when present as single tokens, `<|im_end|>`, `<|eot_id|>`, `<|endoftext|>`,
   `<|end_of_text|>`). llama.cpp: GPU `llama-server` `/completion` with the same ids. Output ids of both
   arms are detokenized by the same CPU server and scored by the unchanged scoring functions. This is
   stricter than "each engine renders its own template": the prompts are identical by construction.
4. **Ours arm engine and pack.** Task eval uses the engine and pack `tools/baro serve` caches for that
   bake (`~/.cache/baro/<id>/`, what a user runs; 10/10 passed 64/64 ref tokens today,
   `exchange/lane-BAROSERVE-report.md`). Perplexity (4 dense/MoE models) uses `serve/engine.mojo` built
   from this worktree's HEAD (`-D BARO_MODEL=qwen35moe` for RegesCore) on that same cached pack; its
   first request per model replays the bake's `baro.run.prompt.tokens` and must reproduce
   `baro.run.ref.tokens` 64/64, else the row is VOID.
5. **Perplexity host math** in numpy (log-softmax over the dumped f32 row); the pure-Python loop was
   hours per model.

Bands, verdict rule, models and the spark perplexity deferral are unchanged.

## Amendment 3 (2026-09-16 eve, after the Ornith pilot's ours task arm, before any task score was read)

The pilot's ours arm hit the 300-token cap on 104 of 120 items: Ornith reasons in `<think>` and is cut
off before answering, so the task score would measure the cap, not the model, and each arm took about
7.5 min. Task prompts are now rendered with `chat_template_kwargs: {enable_thinking: false}` (E8 round
5's no-think setting; templates without the variable ignore it), max_tokens stays 300. The llama.cpp arm
runs 8 parallel slots (`-np 8`, `-c 16384`, so 2,048 context per slot); every item is still one
independent T=0 request with the same ids. The Ornith pilot's task arm is discarded and rerun under this
amendment. Its perplexity numbers stand (ours 9.6124, llama.cpp 9.1251).

## Result (2026-09-16 eve)

| model | engine | PPL ours / llama.cpp (ratio) | task ours / llama.cpp | delta | verdict |
|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct-Q4_K_M | spark | pending (spark logprobs landed after this sweep) | 22/120 / 24/120 | -1.7 pp | task PASS |
| lily-cybersecurity-7b-v0.2-Q6_K | spark | pending (spark logprobs landed after this sweep) | 13/120 / 15/120 | -1.7 pp | task PASS |
| Qwen2.5-7B-Instruct-Q4_K_M | spark | pending (spark logprobs landed after this sweep) | 46/120 / 46/120 | +0.0 pp | task PASS |
| Qwen2.5-Coder-7B-Instruct-Q4_K_M | spark | pending (spark logprobs landed after this sweep) | 21/120 / 20/120 | +0.8 pp | task PASS |
| Qwythos-9B-Claude-Mythos-5-1M-MTP-BF16 | dense | 9.301 / 8.410 (1.106) | 70/120 / 80/120 | -8.3 pp | PASS |
| RegesCore-1.0-35B-UD-Q4_K_S | moe | 6.233 / 6.242 (0.999) | 11/120 / 12/120 | -0.8 pp | PASS |

Every row that ran is inside its frozen band. Scope cut by the maintainer mid-sweep: Ornith, Qwythos-v2 and
Spark-X2.5 skipped (`.work/quality/SKIP`); Granite VOID (`quality-run.sh` edited while the sweep was
executing it, bash re-read shifted lines, its prep step failed). RegesCore's first task arm VOIDed because
the harness ignored the bake's `baro.run.env` (`BARO_MEGA=0 BARO_SPEC=0`); fixed and rerun, row above
is the rerun. Qwythos compares our q4 pack with llama.cpp on BF16 (see docs/BASELINE.md).
