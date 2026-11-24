# Lane tok4-agy (agy, Gemini 3.8 Flash High): partner of the codex tok4 lane

Same goal and rules as `docs/lanes/2026-11-24-tok4-astra-brief.md` (read it whole first). Split of work:
- codex (pane a lane pane, name tok4): levers 1-2 (code-weighted sample, new data in the sample).
- you (tok4agy): levers 3-4 (V 8192 vs 12288 cost, whole-word coverage before stage 2, tool names as single ids) AND the
  parity gate for every candidate either lane makes: `tokv3-gate.sh`, llama.cpp `llama-tokenize` equal ids, template
  render parity, baro.apk pre-type (the apk uses llama.cpp: superbpe pre-type must be known to its build).
Collaborate through files: write your arm table to `.work/tok4/agy-arms.tsv` and read codex's `.work/tok4/sw/*/score.tsv`.
Message codex with `~/iTools/bin/herd tell a lane pane "ASK tok4 <topic> -> <path>"` (HM/1). Your report:
`docs/lanes/2026-11-24-tok4-agy-report.md`; then `herd tell <driver pane> "DONE tok4agy -> <report>"`. CPU only.
