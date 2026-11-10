# Model settings card: OrcaSAQ-2-Cyber-27B-Uncensored (2026-10-17)

Template: `~/iTools/llm/model-library/model-settings-card.md`. Every number cites a receipt in `exchange/`.

## Identity

- Model: orcarouter/OrcaSAQ-2-Cyber-27B-Uncensored-GGUF (qwen35 hybrid: 48 SSM + 16 attention layers, dense FFN, MTP head)
- Model file and quant: `~/Models/OrcaSAQ-2-Cyber-27B-Uncensored-GGUF/OrcaSAQ-2-27B-Uncensored.gguf`, mixed IQ4_XS body, Q5_K qkv/v, Q6_K output and embedding
- Model SHA256: 4ab6bdf8a9008869abf1630fb92d04b5439805f96b56692962fa240b1d439977 (header sha 5b89bccc2dd62e4ff2c623068e47bf37b43d53ac00bc5dd7ad41cae2f428fecd, chat template sha c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041)
- Architecture: H 5120, 64 layers (block_count 65 with the NextN layer), 24 heads / 4 KV heads, key length 256, FFN 17408, vocab 248320, context 262144
- Runtime and version: mojo-baro at e658f85 or later, self-describing bake `OrcaSAQ-2-27B-Uncensored-BARO-e658f85.gguf`
- Host and GPU: 7900 XTX gfx1100, 24 GB, 290 W cap, ROCm 7.2.4
- Date: 2026-10-17

## Load and memory

- Context: TMAX 1088 default (f32 KV, about 131 KB per token). `BARO_TMAX=4096` ran the 490 MB MAX pool out of memory; stay at 1088 to 2176.
- Slots: `-D BARO_SEQ_CAP=1`, `-D BARO_SLOTS=4` (kcfg + 2; the engine raises if k is larger)
- Pack: `--q4`, 17.20 GB on disk, 15.0 GB on the device with `BARO_EMBD_HOST=1`
- Peak VRAM: **17.02 GB** after buffers (was 25.65 GB at defaults, 20.93 at SEQ_CAP=1); `exchange/lane-ORCA27B-VRAM2-SSM-report.md`, `exchange/orca27b-vram-report.md`
- Host RAM: 2.54 GB pinned for the embedding table

## Best settings (what the bake carries)

```
build flags  -D BARO_SEQ_CAP=1 -D BARO_SLOTS=4 -D BARO_DOT_UNROLL=2     (+ shape flags from tools/model-id.py)
run env      BARO_MEGA=1 BARO_SPEC=1 BARO_EMBD_HOST=1
pack         --q4 (tools/engine-pack.mojo)
serve        tools/baro serve OrcaSAQ-2-27B-Uncensored-BARO-e658f85.gguf [--rebuild]
```

| setting | value | evidence |
|---|---|---|
| SEQ_CAP=1 | -4.72 GB | VRAM report, identity 18/20 (2 prompts vary run to run) |
| SLOTS=4 | -0.784 GB, speed 0.998x | `exchange/lane-ORCA27B-VRAM2-SSM-report.md` |
| BARO_EMBD_HOST=1 | -2.54 GB, speed parity (1.014x with SLOTS=4) | same-job 20-prompt A/B, `bench/orca27b-lean-ab.sh` |
| BARO_DOT_UNROLL=2 | no-spec 44.47 to 46.03 tok/s (+3.5%), neutral with spec (73.52 vs 73.55) | `exchange/mega-unroll-report.md` |
| BARO_SPEC=1, k=2 | 44.47 to 70.18 tok/s_gen (1.58x), 87% of drafts accepted | `exchange/receipts/orca27b-spec/` |

## Do not use

- `BARO_SPEC_K=3`: ratio 1.011, 95% CI [0.970, 1.029] (`exchange/spec-k3-report.md`)
- `-D BARO_MEGA_G=80/64`: 0.906x and slower (`exchange/mega-g-report.md`)
- `BARO_CP` / prefill chunk 512: 0.997x, CI below parity (`exchange/lane-ORCA27B-VRAM2-BUFS-report.md`)
- `BARO_NGRAM=1` (MoE only, 0.76x), `BARO_DOT=1`, `BARO_MEGA_WIN=1` (0.761x)
- `-D BARO_DOT_UNROLL=4`: register spill, 15.81 tok/s; `=3`: 44.67

## Prompt and generation

- Chat template: embedded in the GGUF (sha above); greedy decode is the verified path
- Thinking / tool-call format: not measured here (UNVERIFIED)
- Sampling: T>0 runs the megakernel without speculation (about 44.5 to 46 tok/s)

## Evidence

- Launch command: `tools/baro serve ~/Models/OrcaSAQ-2-Cyber-27B-Uncensored-GGUF/OrcaSAQ-2-27B-Uncensored-BARO-e658f85.gguf`
- Verify: `tools/gguf-verify.sh FILE OUTDIR` under `gpu-wait`: PASS 64/64 on the e658f85 bake (`.work/bake27/verify5.log`)
- Quality result: teacher-forced agreement with llama.cpp on the same GGUF, 20 prompts x 64 tokens: **93.75% median, 81.25% minimum** (`.work/bake27/agree/`, run with `bench/llama-agree.sh`)
- Throughput result: greedy with speculation about 73.5 tok/s_gen on 8 stable prompts, served-mode 20-prompt medians 69 to 70 (`exchange/spec-k3-report.md`); bandwidth 653 GB/s = 68% of 960 GB/s without speculation
- Known failures or caveats: p03, p06, p08, p09, p12, p15, p17 vary run to run on this model (open defect, `exchange/receipts/orca27b-spec/nondeterminism.txt`); the Q6_K to q4 conversion of `output.weight`, `attn_qkv`, `attn_v` has not been isolated as a cause of the 6% disagreement
- Status: `PARTIAL` (speed and memory VERIFIED; quality against llama.cpp measured once; thinking and tool calls UNVERIFIED)

## Resource pages

- Official model page: https://huggingface.co/orcarouter/OrcaSAQ-2-Cyber-27B-Uncensored-GGUF (card: 15.7 GB, llama.cpp 20.5 tok/s, 27.6 with DFlash2, CUDA)
- Local evidence: `docs/BASELINE.md`, `~/Brain/mojo/mojo-baro/2026-10-17-orcasaq-27b-bringup.md`

## Serving: template, system prompt, windows, team (2026-10-18)

Receipts under `exchange/receipts/{crew,prompt,naked,selfdesign,maxctx,team,template}`; reports `exchange/lane-CREW-report.md`, `exchange/lane-SELFDESIGN-report.md`.

- **Chat template:** `tools/baro serve MODEL.gguf --chat-template-file serve/templates/qwen35.jinja` (the bake still carries the original). Byte-identical to the GGUF template on 22 of 22 cases in jinja2 and minja; adds `latent_notes`, `latent_instruction`, `latent_part` through `chat_template_kwargs`. Checks: `tools/template-equiv.sh`, `tools/template-equiv-jinja2.py`, `tools/template-latent-check.sh`, `bench/template-serve-check.sh`. The server renders with minijinja.
- **System prompt and tools:** `serve/prompts/orcasaq-system.md` (ASD-STE100, team, latent handoff built in; the role is the first line of the user message) and `serve/prompts/orcasaq-tools.json` (11 tools). `tools/prompt-check.sh MODEL.gguf 2000` (prefix 1,898 tokens), `tools/ste-check.py`. Rule checks run with thinking ON; with thinking off the model ignores the 409 rule.
- **Windows:** f32 KV costs 131,072 B/token, int8 KV (`-D BARO_KVQ=int8`) 33,280 B/token. One int8 engine loads to 163,840 tokens (24,455 MiB), fails at 196,608. Usable next to the desktop: about 131,072. Agreement int8 vs f32: mean 99.1%, min 96.9% on 20 prompts (short range; depth untested). The state handoff carries f32 KV only. Floor for every test: window >= 16384, max_tokens >= 16000 (`bench/min16k.sh`).
- **Latent handoff:** `/v1/checkpoints` then `/v1/ipc/fork`; 224 MB per handle, at most 8 live (delete after use), HTTP 409 on a tampered or mismatched handle. Short notes are faster as text.
- **Engine fix:** a 1-token prompt with speculation on used to kill the engine; it now runs without speculation for that request.
- **Open:** state-vs-text break-even, int8 KV quality at depth, team gain on a task the solo model fails, an executor for the latent tools beyond `crew_assign` (`bench/team-run.sh` is the first).

## Interactive chat in omp (2026-10-18)

- **Launch:** `bench/orca27b-chat.sh [omp args]`. If port 8330 is closed, it starts `bench/orca27b-serve.sh` (one int8 KV engine behind `baro-serve`, omp provider `orca27b`, gpu-wait priority 90, `--vram 23`, 8 h timeout, log `.work/chat/serve.log`), waits up to 240 s for `/health`, then runs `omp --model orca27b/pack`.
- **Window:** the chat launcher sets `W=163840`. Use `W=131072` next to a heavy desktop. Started directly, `orca27b-serve.sh` defaults to 65,536. The preflight refuses to start if `vram_used` is 2,560 MiB or more. It prints `engine_sha`, `window` and `template_sha`.
- **System prompt:** `serve/prompts/orca27b-omp-system.md` (a frozen snapshot of omp's generated prompt, with em dashes removed) + `serve/prompts/orca27b-memory.md` + the live `~/Brain/orca27b/MEMORY.md`. The engine has no system prompt of its own; the client supplies it.
- **Tools:** the model reads its tool list from the frozen prompt text, not from omp. A new MCP server stays invisible until it is added to that file (it answered "no MCP tools" before 7a2874a added the `amarbaro-devtools` block).
- **Memory:** `~/Brain/orca27b/` holds one fact per file, indexed by `MEMORY.md`. The model writes the files itself with its write tool (83d99a3; the model asked for continuity between conversations).
- **Status:** UNVERIFIED. No session after 7a2874a or 83d99a3 has been recorded showing that the model sees the devtools tools or that a memory file was written and read back in a later conversation.
