# Report: model skills for the five runnable bakes (sonnet, 2026-11-13)

## Skills loaded
Pilot report checklist (`2026-11-13-llm-skills-pilot-report.md`), Brain `llms-skills/README.md`, `tools/llm-skills-probe.sh`,
`tools/llm-skills-index.mojo`; repo rules from `docs/rules.md`. No new skill symlinks were needed.

## Result
Five skills built, each with `SKILL.md`, `card.kv`, `system_prompt.md`, `chat_template.jinja`, `baro.json`, `gguf-meta.json`,
`docs/`, `runs/` under `~/Brain/Skills/llms-skills/<key>/`. One gpu-wait job per model, in the brief's order, `BARO_NEURONS=0`,
window 32768 (read back from `/health`), 10 probes each. `llms.db` after rebuild: **6 models** (5 new + qwythos-9b-v2), **208 file rows**
(baro 6, card 6, chat_template 6, doc 7, gguf_meta 6, run 165, skill 6, system_prompt 6).
No bake was made, nothing under `~/Models` was touched.

| model (bake) | probes | decode tok/s_gen median | prefill (8K recall probe) | `baro.run.env` | pack flags |
|---|---|---|---|---|---|
| llama-3.2-1b-instruct (1d6e2dc) | 8/10 | 529 (16000-token run 335) | 6451 rows 2.19 s | BARO_MEGA=0 BARO_SPEC=0 | --dense |
| spark-x2.5-4b (1d6e2dc) | 7/10 | 134 (16000-token run 103) | 8038 rows 8.66 s | BARO_MEGA=0 BARO_SPEC=0 | none (q8 pack) |
| qwen2.5-7b-instruct (1d6e2dc) | 10/10 | 126 | 7727 rows 12.5 s | BARO_MEGA=0 BARO_SPEC=0 | --dense |
| granite-4.2-3b (1d6e2dc) | 8/10 | 153 | 6754 rows 8.58 s | BARO_MEGA=0 BARO_SPEC=0 | --dense |
| ornith-1.5-9b (cf78732) | 9/10 | 71 | 8030 rows 3.23 s | BARO_MEGA=0 BARO_SPEC=0 | --q8 |

Decode = median of the engine's `tok/s_gen` over the 9 to 10 requests (a receipt, not a P4 20-prompt median). Receipts per model:
`~/Brain/Skills/llms-skills/<key>/runs/` (`summary.tsv`, `summary-live.tsv`, `*.req.json`, `*.res.json`, `server.log`, `engine-toks.txt`, `health.json`).

## Per model
**Llama-3.2-1B-Instruct, 8/10.** FAIL p03 snail (wrong arithmetic, ran to `max_tokens` 16000, 47.8 s: model failure), FAIL p04 tools
(HTTP 500 `chat_template: too many arguments (in chat:61)`, template `tojson(indent=4)` rejected by the Mojo minja). No thinking mode.
Answers "Llama" to identity. Fastest model by 4x.

**Spark-X2.5-4B, 7/10.** FAIL p04 (12 tokens in its own `<tool_call>NAME<arg_key>` format, empty content, no `tool_calls`; matches the
2026-11-07 Brain note), FAIL p05 (JSON malformed: `{"name":"}Marta Kowalski",...`; a quote lands inside the value; do not use
`response_format` with it until checked), FAIL p07 (thinking used all 16000 tokens, 155 s, no `</think>`). Prefill is slow (about 930 tok/s).
No vendor card on disk (gguf name is "Hf_Format"); only `generation_config.json` (T1.0, top_p 0.95, top_k off) and the gguf are DOC.

**Qwen2.5-7B-Instruct, 10/10.** Parsed tool call, strict JSON, code runs, 7727-token recall. No thinking mode. Default system text
"You are Qwen, created by Alibaba Cloud..." comes from the template. Slowest prefill of the set (about 620 tok/s).

**Granite-4.2-3B, 8/10.** FAIL p04 (HTTP 500 `too many arguments (in chat:8)`, template `tojson(ensure_ascii=False)`), FAIL p08 (French
but 19 words against a 15-word cap). Thinking on by default; the greedy primes question cost 1473 thinking tokens. Card sampler is T1.0,
top_p 0.95; the probe used 0.6/0.95/20 (not the card values), so quality verdicts are provisional.

**Ornith-1.5-9B, 9/10.** FAIL p08 (16 words, cap 15). Tool call parsed, JSON, code, 8K recall pass. Without a system message it
answers "I'm Qwen ... Tongyi Lab"; the Ornith persona prompt is only in `~/Models/ornith-1.5-9b-q4_K_M/*.system.txt`, not in the gguf or template
(behavior with that prompt not probed). Only bake here on kernel cf78732 (the brief's choice); qwen35 engine honors `BARO_TMAX`.

## Gotchas found by running it (each cost a run or a rescore)
1. **Dense bakes ignore `BARO_TMAX`.** llama, qwen2, granite and spark2_5 embed `comptime TMAX = 4096` in `profile.mojo`; the first chain
   run failed all four at the probe's own `/health` tmax check (4096 vs 32768) in about 10 s each, no GPU time spent. The repo floor is 16384, so for
   those four the probes ran an engine rebuilt from the bake's own cached sources with `TMAX = 32768`
   (`.work/fanout/cache/<id>/engine-k1d6e2dc`, same flags `-D WMMA_TB=1 -D WMMA_C_F16=0`, packs symlinked from `~/.cache/baro`, selected by `BARO_CACHE`).
   The gguf files are untouched. **Deviation from the brief: the served engine for these four is not the stock cache engine.** Decode and prefill
   numbers are from that rebuilt engine; a stock bake serves at most 4096 total tokens. A real fix is a rebake with a larger TMAX (owner's session).
2. **Minja rejects `tojson` arguments** (`indent=`, `ensure_ascii=`), so any template that uses them (Llama 3.2, Granite 4.2) fails with HTTP 500 as
   soon as `tools` are sent. Qwen2.5 and the qwen35 templates use bare `tojson` and work. Not fixed here (renderer lives in `minja/`).
3. **Probe checker bug (mine, fixed).** Thinking was cut with `sed '1,/<\/think>/d'`, which deletes an answer on the same line as `</think>`. Live scoring failed
   Spark p01 and p10 on right answers. Fixed (`${ans##*</think>}`) and all five models were re-scored offline with the new `PROBE_RESCORE=1`
   (no server, saved responses); only Spark changed (7 PASS after, 5 before). `summary-live.tsv` keeps the first scoring. Rule added to `docs/rules.md`, SINDEX entry in `~/Brain/SINDEX/tooling.md`.
4. Thinking arrives inline in `message.content` for every thinking model here (Spark, Granite, Ornith); no `reasoning_content` field (as in the pilot).
5. The dense engine's prefill is slow (620 to 930 tok/s for 3B to 7B) against 2500 to 2900 tok/s for the 1B and the qwen35 9B. Not investigated.

## Could not verify
Tool round trips, `BARO_KVQ=int8`, recall above 8K, other quants and older bakes, vision (Ornith mmproj exists), MTP/spec, the Ornith system prompt, card
sampler values for Granite/Ornith, a rate (each probe is one draw at T0.6), P4 20-prompt decode medians. No Spark-X2.6 bake (waits for the owner).
GPU hold check: `gpu-wait list` empty at launch; each job ran under `gpu-wait run --vram 16`, stopped by process group at the end.

## Changes in the repo
`tools/llm-skills-probe.sh`: `BARO_CACHE` pass-through into the queued job and server, `</think>` strip fix, `PROBE_RESCORE`. Rule lines in
`docs/rules.md`. Brain: five skill folders, `index.md`, `llms.db`, README section "Dense bakes: embedded TMAX 4096".
