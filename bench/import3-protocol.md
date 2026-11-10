# Import 3 models: MiniCPM5-2B, vibethinker-1.5b, Qwen3.8-27B-OBLITERATED (2026-10-18)

Question: does baro serve these three GGUFs from `~/Models` with output that agrees with llama.cpp?
The arch field matches a supported family; nothing else was checked before this round.

## Arms
| model | arch | path | gate |
|---|---|---|---|
| MiniCPM5-2B Q4_K_M | llama | `model-import --arch spark --gguf` (profile, pack, gate, bake) | `bench/dense-run.sh`, 20 prompts x 64 forced tokens |
| vibethinker-1.5b Q4_K_M | qwen2 | same | same |
| Qwen3.8-27B-OBLITERATED Q4_K_M | qwen35, id `qwen35-eec459db896e8bc6` (= OrcaSAQ-2-27B) | engine `.work/crew/engine-kv8-fix2` (the OrcaSAQ serve engine), pack `tools/engine-pack.mojo --q4` | `bench/llama-agree.sh`, 20 prompts x 64 tokens, OrcaSAQ-2-27B in the same job as control |

Window: llama-server `-c 16384`, engine `BARO_TMAX=16384` / profile `TMAX=16384` (min16k guard in both gates).
64 forced ids per prompt is the measure (the fixed-id exemption in `bench/min16k.sh`).

## Bars (frozen before any run)
- spark arms: forced agreement >= 90% on every one of 20 prompts (`model-import` default `MIN_PCT 90`).
- OBLITERATED: median >= control median - 3 pp AND minimum >= control minimum - 5 pp, control = OrcaSAQ-2-27B
  with the same engine, window and prompts in the same job (its 2026-10-17 reading was 93.75 / 81.25).
- Any engine error, NOT-RESIDENT, or a missing agreement line = FAIL for that arm.

## Predictions
- MiniCPM5: may refuse at read-back if its `tokenizer.ggml.pre` has no regex in `serve/tokenizer.mojo`; else PASS.
- vibethinker: PASS (qwen2 is gated on Qwen2.5-0.5B/7B).
- OBLITERATED: PASS, within 2 pp of the control median (same base, same shape).
