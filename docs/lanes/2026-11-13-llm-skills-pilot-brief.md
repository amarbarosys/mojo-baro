# Brief: model skills pilot, one model end to end (sonnet, effort medium, M)

New rule (owner, 2026-11-13): every model under `~/Models` gets a skill for AI agents on how to use it well, built
**while the model runs here**, from its docs, chat template, system prompt and self-describing (BARO) info, all indexed,
with a database. This brief is the pilot on ONE model; the layout you make becomes the template for every other model.

Pilot model: Qwythos-9B v2, bake `~/Models/qwythos-9b-v2-mtp-q6_k/Qwythos-9B-v2-MTP-Q4_0-BARO-eeb082e.gguf`
(its serve cache is built; `tools/baro serve` works, verified by `face/apply_gate.sh`).

## Load first (list them in the report)
`.claude/skills/mojo-baro/SKILL.md`, `.claude/skills/model-bringup/SKILL.md`, `.claude/skills/ops/SKILL.md` (gpu-wait rules).

## Layout (create)
`~/Brain/Skills/llms-skills/` with:
- `README.md`: the rule, the layout, how to add a model.
- `<model-key>/` per model (key = lowercase family-size-variant, e.g. `qwythos-9b-v2`):
  - `SKILL.md`: front matter (name, description that says when an agent should pick this model), then: what it is good
    and bad at (from docs AND from your runs), the exact chat template behaviour (thinking on/off, tool calls, stop
    tokens), the system prompt it expects, sampling settings that work (from docs, checked), context limits, speed here
    (tok/s from the run log), how to serve it (`tools/baro serve ...`), gotchas seen in the runs. Every claim tagged
    DOC (with source) or MEASURED (with receipt path).
  - `chat_template.jinja` (extracted from the gguf `tokenizer.chat_template`), `system_prompt.md` (default or documented),
    `baro.json` (every `baro.*` kv of the bake, from the gguf header), `gguf-meta.json` (arch, shape, quant, ctx),
    `docs/` (model card README from Hugging Face, fetched with the crawl4ai MCP tool or `hf download --include README.md`),
    `runs/` (small text receipts of your probe runs).
- `index.md` (one line per model) and a database `llms.db` (SQLite: table models(key, path, sha, arch, params, quant,
  ctx, tok_s, good_at, bad_at, skill_path, updated), table files(key, kind, path)). A small committed builder rebuilds both
  from the folders: write it in Mojo (`tools/llm-skills-index.mojo` in mojo-baro) per the repo rule; SQLite through the
  `sqlite3` CLI is acceptable for writes.

## Running the model
Under `gpu-wait run` only (CLAUDE.md: env and redirects go inside the queued command). Use `tools/baro serve BAKE --port N`
and the OpenAI API (`/v1/chat/completions`). Respect the repo floor: `BARO_TMAX >= 16384`, `max_tokens >= 16000`.
Probe set (keep it small, about 10 requests, one gpu-wait job): plain chat, thinking on vs off, a tool call, a JSON-schema
output, a long-context recall at ~8K, a code task, a system-prompt-following check. Save each request and response under
`runs/`. Stop the server at the end of the job.

## Rules
No em dashes. Brain is a git repo: commit there with a pathspec. mojo-baro: commit the builder with a pathspec, never
stage other dirty files. Do not delete or move anything under ~/Models.

## Deliverable
`docs/lanes/2026-11-13-llm-skills-pilot-report.md` (mojo-baro): skills loaded, the layout, the probe results table,
what you could not verify, and a proposed checklist for the next models.
