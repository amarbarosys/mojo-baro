---
name: mojo-baro-operate
description: Operate mojo-baro, a local LLM server for AMD RDNA3 GPUs, as a user would. Use it to serve a GGUF model, talk to the OpenAI or Ollama API, tune speed or context length, run many sessions, serve embeddings, fit a model into low VRAM, open the GUI, or drive the MCP server. Use it when someone asks "why is it slow", "which setting does X", "how do I turn on Y", or hits an HTTP 429, 502, 501, 404 or 503. It covers the basic settings, the eight presets, what not to enable, and how to verify each change.
metadata:
  owner: amarbaro.org labs
  license: Apache-2.0
---

# mojo-baro-operate

Audience: a person or agent that runs mojo-baro. Development notes are out of scope.

## When to use

- Serve a GGUF model on the local GPU (`tools/baro serve`).
- Speed, context length, memory, or concurrent sessions are the question.
- Embeddings, the GUI (`tools/baro gui`), or the MCP server.
- An HTTP error or a slow answer. See Troubleshooting.
- "Which setting does X" or "why did my change do nothing". See Verify a change.

## Quick start

Source: README.md:30-112.

1. Hardware: AMD Radeon RX 7900 XTX (gfx1100, 24 GB) with ROCm 7.2 (README.md:36-37). Other RDNA3 cards build, tuning unmeasured (README.md:36).
2. Install and check:
   ```sh
   git clone https://github.com/amarbarosys/mojo-baro && cd mojo-baro
   uv sync
   (cd serve && cargo build --release)
   bench/preflight.sh
   ```
   `uv sync` pins `max[all]==26.7.0` (Mojo 1.2.0) in a repo-local `.venv` (README.md:42-43, 45-52). `bench/preflight.sh` needs no GPU and no weights (README.md:54).
3. Serve a model. The first run for a shape takes minutes; the engine and pack go to `~/.cache/baro/<shape-id>/` (README.md:59, 62-66).
   ```sh
   tools/baro serve ~/Models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --port 8080
   ```
4. Talk to it. Use `max_tokens` of 16000 or more (`docs/operate-capabilities.toml` chat row; README.md:71-74). Streaming uses SSE (README.md:79-81). Ollama clients use the same port (README.md:84-85).
5. The GUI: `tools/baro gui [--model M.gguf] [--port N] [--fps N]` (README.md:88-112). Needs `libSDL3.so.0` and `libfreetype.so.6` (README.md:110-111).

## Capabilities and settings

Every capability has an enable step, a default, a best standard with its measured reason and source, a status word, the GUI control (or MISSING) and a check. They are in `references/capabilities.md` (generated from `docs/operate-capabilities.toml`, so it cannot drift); the full table with every setting is `docs/OPERATE-AUDIT.md` in the repo. Index of ids:

chat, completions, sampling, penalties, thinking, tools, json-schema, multiturn, fail-word, spec, mega, kvq, tmax, prefill, parallel, tier, cert, embeddings, checkpoints, router, audio, vision, pwa, ollama, face, profiles, cpu, neurons, rdna4, health, cancel, mojo-front, tune, shared-exp.

Read the row of the capability you are about to change, then the Verify section below. Status words: WORKS, PARTIAL, OFF (works, default off), FAILS.

## Presets per goal

These are starting points. The presets in docs/settings.toml say themselves that several values are a convention or unmeasured; measure on your card before you rely on one.

- **Fast chat**: preset Fast or Balanced (docs/settings.toml). Set `BARO_SPEC=1`, `BARO_SPEC_K=2`, `BARO_FR=1` (FR-Spec draft head, default; needs the pack FR table, which tools/baro adds), `BARO_MEGA=1`, `BARO_PREFILL=1`, `BARO_TMAX=16384`. On a K-quant pack under Mojo 1.2 set `BARO_MEGA=0` (docs/operate-capabilities.toml, mega row).
- **Long context**: profile `profiles/turbo27b-coder.toml`: `tools/baro serve MODEL --profile turbo27b-coder`. Int8 KV is a build flag, so it comes from the profile's `[build]` section, not an env variable. Run `pack-fit` first (profiles/turbo27b-coder.toml; docs/operate-capabilities.toml, profiles row).
- **Many sessions**: build the engine with `-D BARO_SEQ_CAP=N -D BARO_SLOTS=2`, then run `baro-serve --engine E --pack KQPACK --parallel N --kv-pool P`. Dense K-quant packs only. These are flags, not env (docs/operate-capabilities.toml, parallel row).
- **Embeddings**: `baro-serve --embed-only --port 8081` after `embed/build.sh` (docs/operate-capabilities.toml, embeddings row).
- **Low VRAM**: int8 KV (a build flag, as above) and `BARO_TMAX=16384`. For a MoE pack, `BARO_TIER=64`. Run `pack-fit` before, and `rocm-smi --showmeminfo vram` (docs/settings.toml preset Low VRAM; docs/operate-capabilities.toml, tier and tmax rows).
- **Correctness-checked**: preset Quality (`BARO_SPEC=0`, greedy, f32 KV) and `BARO_CERT=SEED`. Compare with `tools/baro verify MODEL.gguf` (docs/settings.toml preset Quality; docs/operate-capabilities.toml, cert row).

## What NOT to turn on

Each line gives the reason and its source.

- q4x MTP verify window: FAILS (docs/CAPABILITIES.md:28).
- WMMA attention and the dense chunk scan in MoE prefill (`BARO_PF_ATT`, `BARO_PF_SSM`): non-exact, FAIL identity (docs/CAPABILITIES.md:104).
- A requantised q4 pack of a Q4_K_M model: 92.97% agreement, FAILS. Use the K-quant pack (README.md:175).
- `BARO_NGRAM` on MoE: works but slower (docs/CAPABILITIES.md:113).
- `BARO_MEGA_WIN` and `BARO_DOT`: default off (docs/CAPABILITIES.md:428, 432).
- `presence_penalty` and `frequency_penalty`: inert (docs/operate-capabilities.toml, penalties row).
- The Mojo front: experimental (docs/operate-capabilities.toml, mojo-front row).
- `BARO_CERT` as a default: decode ratio 1.023, 0.23 ms per token (docs/operate-capabilities.toml, cert row).
- `BARO_SHARED_EXP`: 3.6x the cost of a plain exp (docs/CAPABILITIES.md:17).
- `tool_choice: auto` as a tool filter: it does not restrict names (docs/operate-capabilities.toml, tools row).
- The CPU engine for speed: 6 to 7 tok/s (docs/operate-capabilities.toml, cpu row).

## Troubleshooting by symptom

Rows from README.md:454-461.

| symptom | cause | fix |
|---|---|---|
| a plain "hello" costs about 480 prompt tokens and the reply is a `brain_ask` tool call | `tools/baro serve` exports `BARO_NEURONS=1` (`tools/baro:299`), so a chat request without tools gets the retrieval loop (the bare `baro-serve` binary treats an unset variable as off) | send `"neurons": false` per request, or start with `BARO_NEURONS=0` (the `chat` tool of baro-mcp sends `neurons:false` unless told otherwise) |
| `baro: not a BARO file: no baro.kernel.commit` | the model is a plain GGUF | serve a BARO-stamped file (`*-BARO-<commit>.gguf`) |
| `pack/profile mismatch: profile H 4096 ... implies H 5120` | the engine was compiled for another model shape | use `tools/baro serve`, or build with the shape flags from `tools/model-id.py MODEL.gguf --json` |
| `no tokenizer.json loaded` (HTTP 503 on chat) | the pack has no `tokenizer.json` | pass `--tokenizer`, or rebuild the pack with `tools/baro serve` |
| decode is 10x slower than the tables, no error | weights spilled from VRAM to GTT because something else holds VRAM | read `gtt_used` in `/sys/class/drm/card*/device/`; free VRAM or lower `BARO_TMAX` |
| out of memory with VRAM free | MAX's small-allocation pool is 2% of free VRAM | set `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT` higher |
| `/health` 503 for a long time after start | the pack is still loading; the port opens first | wait for 200 (README.md:460) |
| the engine prints `ready`, exits 0, served nothing | its stdin was empty | feed requests on stdin, or let a front start it |

HTTP codes:

- **429**: past the KV pool in `--parallel` mode. Not a failure (docs/operate-capabilities.toml, parallel row).
- **502**: `engine_invalid_tokens`, a non-zero fail word. Read `timings.fail` (docs/operate-capabilities.toml, fail-word row).
- **501**: embeddings through the chat model on the spark engine or on a `BARO_SEQS > 1` engine. Use `--embed-only` (docs/operate-capabilities.toml, embeddings row).
- **404**: `/api/pull` for a model other than the loaded pack (docs/CAPABILITIES.md:128), and `/v1/embeddings` with a model name that is neither bge-small nor the loaded chat model (README.md:353-354).
- **503**: `/health` while loading or dead, and chat with no tokenizer (README.md:460; docs/operate-capabilities.toml, health row).

## How to verify a change

Read-back endpoints and their field names are listed with file:line in serve/src/main.rs:484-529.

| setting | where to read it back |
|---|---|
| `BARO_TMAX` | `GET /health` `limits.tmax` (serve/src/main.rs:497-499) |
| `BARO_PACK` | `GET /health` `pack` (serve/src/main.rs:502) |
| `BARO_SEQS`, `BARO_POOL` | `GET /health` `limits.seqs`, `pool`, `capacity` (serve/src/main.rs:494-501) |
| `BARO_SPEC_K`, `BARO_TIER` | `GET /health` `limits.spec_k`, `limits.tier` (serve/src/main.rs:497-499) |
| `BARO_SPEC`, `BARO_MEGA`, `BARO_PREFILL` | response `timings.path` (docs/operate-capabilities.toml, sampling and mega rows) |
| `BARO_KVQ` | no read-back endpoint; use the response timings or the engine log |
| `sampling.*` | the response `timings`; the seed check: same seed twice gives the same ids (docs/operate-capabilities.toml, sampling row) |

`/props` in the README says it reports the KV type (README.md:310-315). The handler at serve/src/main.rs:506-519 returns `n_ctx`, `model` and the template capabilities only. Use `/health` or the engine log for KV type.

## The MCP server

`baro-mcp` (`tools/mcp/baro_mcp.mojo`) is a stdio MCP server over the running front. Build: `mojo build tools/mcp/baro_mcp.mojo -I . -I face -I tools/mcp -o .work/baro-mcp`. Add it: `claude mcp add -e BARO_MCP_URL=http://127.0.0.1:8080 -e BARO_MCP_REPO=$PWD baro-mcp -- $PWD/.work/baro-mcp`. It reads `docs/settings.toml` and `docs/operate-capabilities.toml` on each call.

- Read: `status`, `models`, `settings`, `capabilities`, `explain_setting`.
- Control: `chat`, `embed`, `checkpoint_list`, `checkpoint_fork`, `set_and_restart`, `preset`. `checkpoint_export` and `checkpoint_import` return "binary download/upload not supported".
- `set_and_restart` takes env and server-env settings only, validates each value, restarts the engine as the GUI Apply does, and reports `took_effect` per key from the new process's environment. Build flags (`BARO_KVQ`) and request settings are refused.
- Failed calls return `isError`, never a protocol error. Test: `tools/mcp/test_mcp.sh`.

## See also

- [references/settings.md](references/settings.md): the Basic settings and the eight presets, one line each.
- `docs/OPERATE-AUDIT.md` in the repo: the full capability, setting and flag tables with the GUI column and the gaps.
- [references/capabilities.md](references/capabilities.md): every capability, generated.
