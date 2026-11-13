# Brief: model skills for the runnable bakes (sonnet, effort medium, M)

Follow the pilot exactly: `docs/lanes/2026-11-13-llm-skills-pilot-report.md` (its checklist), layout and builder
from `~/Brain/Skills/llms-skills/README.md`, `tools/llm-skills-index.mojo`, `tools/llm-skills-probe.sh`. Same rules
as the pilot brief `docs/lanes/2026-11-13-llm-skills-pilot-brief.md` (gpu-wait only, TMAX >= 16384, max_tokens >= 16000,
receipts under runs/, DOC/MEASURED tags, never touch ~/Models, commit with pathspecs).

Models, one gpu-wait job each, in this order (newest bake per model; all pass `tools/baro serve --no-serve`):
1. Llama-3.2-1B-Instruct  `~/Models/*/Llama-3.2-1B-Instruct-Q4_K_M-BARO-1d6e2dc.gguf`
2. Spark-X2.5-4B           `~/Models/*/Spark-X2.5-4B-Q4_K_M-BARO-1d6e2dc.gguf`
3. qwen2.5-7b-instruct     `~/Models/*/qwen2.5-7b-instruct-q4_K_M-BARO-1d6e2dc.gguf`
4. granite-4.2-3b          `~/Models/*/granite-4.2-3b-BF16-BARO-1d6e2dc.gguf`
5. Ornith-1.5-9B           `~/Models/*/Ornith-1.5-9B-Q4_K_M-BARO-cf78732.gguf`
Run with `BARO_NEURONS=0` (the pilot found the default pads ~600 tokens); record that in each SKILL.md.
For each model also record the bake's `baro.run.env` and pack flags, and the decode tok/s, so speed settings are visible.
A model whose probes fail to start: write what failed in its SKILL.md as UNVERIFIED and continue with the next.
Do not bake new models (Spark-X2.6 waits for the owner's session).

Deliverable: `docs/lanes/2026-11-13-llm-skills-fanout-report.md`: per model, probes passed/failed, tok/s, run env, gotchas;
`llms.db` row count after the rebuild.
