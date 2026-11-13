# Report: model skills pilot, Qwythos-9B-v2 (sonnet, 2026-11-13)

## Skills loaded
`.claude/skills/mojo-baro/SKILL.md`, `.claude/skills/model-bringup/SKILL.md`, `.claude/skills/ops/SKILL.md`; from them the ops
rules #1 to #8 (gpu-wait carries nothing) and model-bringup rules #17 to #20 (server lifecycle).

## Layout (made)
`~/Brain/Skills/llms-skills/`: `README.md` (rule, layout, how to add a model), `extract.sh` (gguf header to
`chat_template.jinja`, `baro.json`, `gguf-meta.json`), `index.md` and `llms.db` (generated), and `qwythos-9b-v2/` with
`SKILL.md`, `card.kv`, `system_prompt.md`, `chat_template.jinja`, `baro.json` (99 keys; `*.src.*` values stored as len + sha256, the
sources are in the gguf), `gguf-meta.json`, `docs/` (base model card and GGUF repo card, fetched with `hf download --include README.md`),
`runs/` (30 small files, 0.2 MB). Builder: `tools/llm-skills-index.mojo` (Mojo; writes index.md, pipes SQL to the sqlite3 CLI).
Probe driver: `tools/llm-skills-probe.sh` (one gpu-wait job, self-re-exec like `face/apply_gate.sh`).
`llms.db` verified by query: 1 model row, 38 file rows (kinds baro, card, chat_template, doc, gguf_meta, run, skill, system_prompt).
Two Brain files were not asked for: `card.kv` (the builder's input row; Mojo has no cheap JSON parse) and `extract.sh`.

## Probe results (bake eeb082e, `tools/baro serve`, BARO_NEURONS=0, BARO_TMAX=32768, about 3 GPU minutes per run, 2 clean-ish runs)
Receipts: `~/Brain/Skills/llms-skills/qwythos-9b-v2/runs/` (`summary.tsv`, `*.req.json`, `*.res.json`, `server.log`, `engine-toks.txt`).

| id | probe | prompt tok | gen tok | tok_s_gen | verdict |
|---|---|---|---|---|---|
| p01 | chat, thinking on | 57 | 359 | 42.5 | PASS (Canberra) |
| p02 | chat, thinking off | 59 | 38 | 70.6 | PASS |
| p03 | snail puzzle, thinking on | 87 | 567 | 57.9 | PASS (Answer: 8) |
| p04 | tool call | 313 | 26 | 29.5 | PASS (parsed tool_calls get_weather Paris) |
| p05 | json_schema strict | 64 | 23 | 36.4 | PASS |
| p06 | 8K recall (needle mid) | 8060 | 12 | 68.6 | PASS (prefill 8026 rows 2.66 s) |
| p07 | code, thinking on | 71 | 6487 | 58.9 | FAIL (code raises IndexError on 5 of 5 strings; 110 s) |
| p08 | system prompt (French, 15 words) | 63 | 23 | 71.1 | PASS (one clipped word) |
| p09 | identity | 50 | 19 | 71.1 | PASS (by hand: "I am Qwythos ... Empero AI") |
| p10 | greedy primes | 54 | 270 | 68.8 | PASS (no loop) |

Verdict 9 of 10 PASS; the FAIL is a real model failure, rechecked by running the extracted code by hand. Small-request
tok_s_gen is low because the engine divides by a short decode; the 6487-token run is the decode figure: 58.9.

## Findings by running it (each cost a run)
1. `tools/baro serve` exports `BARO_NEURONS=1`: the front appends ~600 tokens of `<reference>` Brain notes to every user
   message. First run: a one-line question carried 712 prompt tokens, the model talked about "the reference information",
   the tool probe returned an invented weather. All probes were rerun with `BARO_NEURONS=0`. First-run receipts kept as `runs/neurons-on-*`.
2. Thinking comes inline in `message.content` (opening `<think>` is in the prompt; no `reasoning_content`). My first
   checks read the whole content and could pass on text inside the thinking; the script now cuts at `</think>`.
3. baro-serve has no sampling defaults; every request must carry the card's temperature, top_p, top_k.
4. Prompt + max_tokens must fit TMAX: a 16786-token prompt (my filler was 26 tokens a line, not 13) at TMAX 32768 got 400.
5. Sampling is seeded: two runs of p07 produced the identical 6487 tokens. Run-to-run variation needs a different seed or prompt, not a rerun.
6. File name says Q4_0 and `baro.run.env` says BARO_SPEC=0; the engine pack is q8 (10.73 GB device), so speed here is not the MTP speed.

## Could not verify
- Vision (mmproj), multi-turn tool round trip with a tool result, recall beyond 8K, 16K+ prompts, `BARO_KVQ=int8` agreement,
  MTP speculation speed, other quants (Q6_K file, `3c6c25c` older bake), greedy loop rate (one probe only; the 0% claim is DOC).
- Benchmark numbers (MMLU, GPQA, HumanEval) are the vendor's, internal harness, DOC only.
- The probe set is small: one sample per probe at T0.6, so PASS/FAIL per probe is one draw, not a rate.
- `tok_s` in the db (63) is the median of the ten engine `tok/s_gen` values, not a 20-prompt median (P4), so it is a receipt, not a champion claim.
- No GPU hold on the board was found at launch (`gpu-wait list` empty); the whiteboard was grepped, not read in full.

## Checklist for the next models
1. Recall first: this skill library, model cards in `docs/models/`, `~/Models/<dir>/README.md`; note existing bakes and shas.
2. CPU: `tools/baro serve MODEL --no-serve` (cache hit or build); `extract.sh MODEL <key>`; `hf download ... --include README.md`.
3. `pack-fit` before the GPU job when the pack is above 10 GB; declare `--vram`.
4. Run the probe (`tools/llm-skills-probe.sh`) with `BARO_NEURONS=0`; adapt probes to the model (no tools: drop p04; no thinking: drop think-on cases); keep 8K recall and the system-prompt check.
5. Read the first response by hand before trusting a PASS: prompt_tokens for a one-line question must be below ~100.
6. Strip thinking before checking answers; verify code by executing it; count a PASS only after the check could fail.
7. Write SKILL.md with every claim tagged DOC (source) or MEASURED (receipt); say what was not probed.
8. Rebuild index and db with the builder, commit Brain by pathspec, commit mojo-baro scripts by pathspec.
9. Next models in the queue by value: the daily-use ones with a BARO bake (Spark/OrcaSAQ 27B notes exist in `docs/models/`; start by folding them in).
10. Open item for the maintainer: `BARO_NEURONS=1` as the `tools/baro serve` default hurts evaluations; consider default 0 for non-chat callers (not changed here).
