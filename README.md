# mojo-baro

An LLM inference engine for AMD RDNA3 GPUs. Every GPU kernel is written in
[Mojo](https://www.modular.com/mojo). One RX 7900 XTX serves chat over OpenAI-,
Anthropic- and Ollama-compatible HTTP APIs, from a GGUF file, with no Python at run time.

| at a glance (7900 XTX, 20-prompt medians unless noted) | |
|---|---|
| Qwythos-9B Q4_0, plain decode | **148.89 tok/s, 1.34x llama.cpp** (111.34, same job) |
| Qwythos-9B q4, MTP k=2 with turbo window GEMVs (default) | **168.88 tok/s**, ids identical to the base build 20/20 |
| Qwythos-9B Q8_0 | 80.38 tok/s, 1.07x llama.cpp |
| Ornith-1.5-9B Q4_K_M, native K-quant pack | 99.53 tok/s, 1.06x llama.cpp; prefill 0.80x of llama.cpp's time at 16k |
| Qwen3.8-27B q4, MTP k=2 (dot2 window, default) | 86.0 tok/s |
| RegesCore-35B MoE | 111.89 tok/s, level with llama.cpp |
| Qwen3.8-Flash-Next, 74.5 GB, 512 experts from host RAM | 50 to 59 tok/s |
| Llama-3.2-1B (generic engine) | 562.8 tok/s, 1.36x llama.cpp |

Every number has a preregistered protocol in `bench/` and a check you can run. The
toolchain moved to Mojo 1.2.0 on 2026-10-29; rows dated before that are older-toolchain
numbers (see [Results](#results)).

The project is about the kernels. Consumer RDNA3 cards are where most people have
24 GB of VRAM, and the vendor libraries are not tuned for them. This repo measures how
much of that gap is real.

**Have an AMD GPU that is not a 7900 XTX?** Your card is the most useful thing you can
contribute. See [docs/amd-family.md](docs/amd-family.md): fifteen minutes, no model
weights needed.

**Contents:** [Quick start](#quick-start) · [How it works](#how-it-works) ·
[Results](#results) · [What runs](#what-runs) · [The server and its API](#the-server-and-its-api) ·
[Clients](#clients-terminal-window-browser) · [Configuration](#configuration) ·
[Other backends](#other-backends-cpu-nvidia-rdna4-phone) · [Testing](#testing) ·
[Measuring](#measuring) · [Troubleshooting](#troubleshooting) ·
[How numbers get into this repo](#how-numbers-get-into-this-repo) · [Layout](#layout) ·
[Contributing](#contributing) · [License](#license)

## Quick start

### What you need

| | |
|---|---|
| GPU | AMD Radeon RX 7900 XTX (gfx1100, 24 GB). Other RDNA3 cards should build; their tuning is unmeasured |
| OS | Linux with ROCm 7.2 (`/dev/kfd` readable by your user) |
| Toolchain | [`uv`](https://docs.astral.sh/uv/), Rust (cargo) for the HTTP front, CMake and a C++ compiler for the hipBLASLt shim, `jq` and `curl` for the scripts |
| RAM | 32 GB is enough for 9B models; the 35B MoE, the 27B models and the 512-expert model keep part of their weights in host RAM |
| Weights | a GGUF file of a [supported architecture](#what-runs). No weights ship with this repo |

Nothing is installed machine-wide. `uv sync` creates a repo-local `.venv` that pins
`max[all]==26.7.0` (Mojo 1.2.0).

### Install and check

```sh
git clone https://github.com/amarbarosys/mojo-baro && cd mojo-baro
uv sync                              # repo-local .venv with the pinned Mojo/MAX toolchain
(cd serve && cargo build --release)  # the HTTP front, serve/target/release/baro-serve
bench/preflight.sh                   # CPU only: CI checks, every test builds, both engines build
```

`bench/preflight.sh` needs no GPU and no weights. If it passes, the toolchain is right.

### Serve a model

```sh
tools/baro serve ~/Models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --port 8080
```

On the first run for a model shape this takes a few minutes: `tools/baro` reads the
architecture and dimensions from the GGUF header, compiles an engine for exactly that
shape, and converts the weights into a pack. Both go to `~/.cache/baro/<shape-id>/`
(override with `$BARO_CACHE`). Every later run of the same file starts in seconds, and
another checkpoint of the same shape reuses the compiled engine. An unsupported
architecture or tokenizer is refused before any build starts.

The context window is sized automatically: unless you set `BARO_TMAX`, the engine measures
free VRAM, divides by the KV bytes per token and caps the result at the model's trained
context.

### Talk to it

```sh
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "Name the capital of France in one word."}],
  "max_tokens": 16000
}' | jq -r '.choices[0].message.content'
```

Streaming uses server-sent events, the same as the OpenAI API:

```sh
curl -sN localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a haiku about GPUs."}],"stream":true}'
```

Any OpenAI client works with `base_url="http://localhost:8080/v1"` and any API key.
Anthropic SDK clients and Claude Code work against `/v1/messages` on the same port.
Ollama clients work against `/api/*`. See [Clients](#clients-terminal-window-browser) for
the terminal client, the window and the browser page.

## How it works

```
 HTTP client (curl, OpenAI / Anthropic SDK, Claude Code, Ollama client, tools/baro chat, face)
      |
      v
 baro-serve  (Rust, serve/src/)              baro-front (Mojo, serve/front/, experimental)
   chat template, tokenizer, tool calls,       same routes, same engine protocol
   JSON-schema masks, queue, cancel,
   per-message checkpoints, /health
      |
      |  one JSON line per request on stdin, one JSON line per token on stdout
      |  (serve/PROTOCOL.md)
      v
 engine process (Mojo), compiled for one model shape
   serve/engine.mojo   qwen35 dense and qwen35moe (MoE): the tuned engine
   serve/spark.mojo    generic dense: llama, qwen2, qwen3, granite, spark2_5
   serve/q4x.mojo      qwen4exp, 512 experts streamed from host RAM
   cpu/engine_cpu.mojo decode on the CPU, no GPU (qwen35 dense)
      |
      v
 kernels/*.mojo   GEMV and WMMA GEMM per quant format, flash attention, gated-delta SSM scan,
                  MoE router, device sampler, persistent megakernels (single and pair)
```

**One engine per shape, compiled.** Layer count, hidden size, head counts and the
architecture are compile-time constants (`-D BARO_H=4096 ...`), so loops have fixed trip
counts and the compiler can unroll and schedule them. The cost: an engine built for a 9B
pack refuses a 27B pack (`pack/profile mismatch ... implies H 5120`). `tools/baro` hides
this by keying its cache on the shape.

**Packs, not GGUF at run time.** `tools/baro` converts the GGUF once into a pack: one flat
`pack.bin` with weights laid out for the kernels that read them, an `index.txt`, the
tokenizer and an `identity.json` that names the source file by sha256. A pack loads with
one sequential read. Pack formats: q4 and q8 (requantized), native K-quant (`--kq`, keeps
the GGUF's Q4_K / Q6_K blocks and decodes them in the kernel) and split packs for MoE
experts in host RAM ([`docs/ENGINE-PACK.md`](docs/ENGINE-PACK.md)).

**Decode paths.** The engine reports which one served each request (`timings.path`):

| path | what it is |
|---|---|
| `mega` | one persistent kernel launch per token for the whole forward pass (dense default) |
| `launch` | one launch per op; the reference path and the MoE default |
| `spec` | speculative decode with the model's own MTP draft head: k tokens drafted, one verify window; output identical to greedy. The window's Q4_0 GEMVs run `amar_turbo_rows` (LDS-staged A rows, bit-identical) |
| `pair` | two sessions decoded by one R = 2 megakernel on dense packs (`--parallel`) |
| `prefill` | the prompt in batched chunks through WMMA GEMMs |

DFlash2 block drafting (`BARO_SPEC=dflash2`) is opt-in: it matches llama.cpp's port and
reads 1.022x of MTP k=2, a thin margin, so MTP stays the default.

**Memory.** KV is f32 by default or int8 (`BARO_KVQ=int8`, dense only, 33,280 bytes per
token on the 9B, RULER 100.0 at 64k and 128k). MoE experts can live in host RAM with a
VRAM cache per layer (`BARO_TIER`), filled by in-kernel reads from pinned host memory.
Each message boundary of a chat is checkpointed, so the next turn prefills only the new
message.

**State.** Session state saves to and loads from files, and moves between engine processes
through `latentos/` (memfd and HIP IPC). `spark.mojo` can hold a second model in the same
process (`BARO_TWINS=2`, `BARO_PACK_B`) and hand KV from one to the other.

## Results

### Decode against llama.cpp

Same box, same GGUF, greedy, no speculation on either side.

| model, weights | llama.cpp tok/s | mojo-baro tok/s | ratio |
|---|---|---|---|
| **Qwythos-9B, Q4_0 both sides, 20-prompt median, 2026-10-12** | **111.34** | **148.89** | **1.34x** |
| Qwythos-9B, Q8_0 both sides, 20-prompt median, same stint, 2026-10-26 | 75.46 | 80.38 | 1.07x |
| Ornith-1.5-9B Q4_K_M, our q8 pack (2x llama.cpp's weight bytes), same stint, 2026-10-26 | 93.40 | 80.48 | 0.86x |
| Ornith-1.5-9B Q4_K_M, our q4 pack (requantized; FAILS agreement, 92.97% median), same stint, 2026-10-26 | 93.07 | 146.93 | 1.58x, not a claim |
| Ornith-1.5-9B Q4_K_M, native K-quant pack (`--kq`, launch path), agreement 98.44% median, 2026-10-26 | 93.25 | 96.64 | 1.04x |
| **Ornith-1.5-9B Q4_K_M, native K-quant pack on the megakernel, same stint, 2026-10-26** | **94.14** | **99.53** | **1.06x** |
| Ornith-1.5-9B Q4_K_M, K-quant pack, **prompt prefill** 1024 / 16384 tokens (seconds) | 0.298 / 5.66 | 0.259 / 4.54 | 0.87x / 0.80x of the time |
| Qwen2.5-7B Q4_K_M, native K-quant path (3-prompt quick run, 2026-10-21) | 121.2 | 132.8 | 1.10x |
| Llama-3.2-1B Q4_K_M, spark.mojo, 21-prompt medians, same job, 2026-10-26 | 412.9 | 562.8 | 1.36x |
| RegesCore-35B MoE (different stints) | 109.92 | 111.89 | 1.02x |

The Q4_0 row is one same-stint job ([`bench/llama-stint-protocol.md`](bench/llama-stint-protocol.md)):
ours faster on 20/20 prompts. llama.cpp ran in its fast configuration (flash attention,
f16 KV); ours ran with f32 KV. A ratio is only quoted as a claim when both sides were
measured in the same stint; the other rows say so.

**Toolchain.** The rows above were measured on Mojo 1.1.0 or earlier. The move to Mojo
1.2.0 / MAX 26.7.0 (2026-10-29) was gated on builds and tests, not on throughput. Of these
rows only the K-quant megakernel was re-timed: on 1.2 it reads 88.13 tok/s against the
launch path's 98.29 on the same pack, ids 20/20, cause open
([`docs/BASELINE.md`](docs/BASELINE.md) "Megakernel on K-quant packs"). Native K-quant
packs therefore default to the launch path (`BARO_MEGA=0`).

### Speculative decode

Speculation with the model's own MTP head is on by default and gives output identical to
plain greedy decode ([`bench/mtp-protocol.md`](bench/mtp-protocol.md)).

| model, arm (20-prompt medians) | tok/s_gen |
|---|---|
| Qwythos-9B q4, k=2, launch window, base build | 154.80 |
| **Qwythos-9B q4, k=2, `amar_turbo_rows` window (default since 2026-11-15)** | **168.88** (1.091x, ids 20/20) |
| Qwythos-9B q4, T=0.7 top_p 0.9, k=2 | 147.15 (spec off: 109.19) |
| Qwen3.8-27B q4, k=2, exact turbo window | 73.3 |
| **Qwen3.8-27B q4, k=2, dot2 window GEMVs (default since 2026-11-16)** | **86.0** (1.222x over 70.4 same job) |
| Qwen3.8-27B-TurboFCFusion Q4_K_M, k=2 with turbo rows | 68.72 (61.12 same job) |

The dot2 window is not bit-exact; it became the default on a 120-task eval
(87/120 against 79/120 exact). Sampling runs on the device inside the decode loop and
composes with speculation: sampled decoding with speculation (147.15) is faster than
greedy decoding without it (134.97). On real text llama.cpp's own speculation was still
ahead of ours at k=2 on the 9B (123.5 against 100.7, measured before the turbo window);
the 5-token race prompt alone is not a claim.

### Prefill

Within about 1.3x of llama.cpp from 8k to 32k context (17.42 s against 13.18 s at 32k),
using f16 WMMA flash attention and a one-wave SSM scan. On a native K-quant pack the
prompt runs batched through a WMMA GEMM that decodes Q4_K / Q6_K in its B staging and
beats llama.cpp's prefill time (table above). On spark.mojo, batched prefill (128-row
chunks) is 10 to 16x faster than token-by-token on 350-token prompts, with identical ids.

### A cold-cache vendor beat at the decode shape

Single-token decode streams weights out of VRAM. At **M=1, K=4096, N=12288**, with
weights rotated out of the 96 MB Infinity Cache
([`bench/coldcache-protocol.md`](bench/coldcache-protocol.md)):

| kernel | us per launch |
|---|---|
| `amar_matmul_skinny_v2` CPT=8 | 137.8 to 138.9 |
| **`amar_matmul_skinny_m1` CPT=8** | **121.2 to 121.8** |
| hipBLASLt f16 | 122.1 to 123.3 |

The ranges do not overlap; the margin is about 1%. The cause is occupancy, not bytes: one
row drops the LDS staging and shrinks the accumulator, so more waves stay resident. It
took four preregistered rounds, three of which falsified their own predictions.

### An fp16 WMMA GEMM ahead of hipBLASLt

Two LDS buffers, one barrier per K-step, a two-deep global prefetch, XOR-swizzled A,
188 VGPR and no spills; the tile shape is picked per launch
([`bench/wmma-fp16-protocol.md`](bench/wmma-fp16-protocol.md)):

| size | 256 | 512 | 1024 | 1536 | 2048 | 3072 | 3584 |
|---|---|---|---|---|---|---|---|
| **ours, GFLOP/s** | 6372 | 30642 | 74824 | 93632 | 91300 | 99307 | 105786 |
| hipBLASLt | 6288 | 26324 | 63623 | 69332 | 80203 | 97224 | 85671 |
| ratio | 1.01 | 1.16 | 1.18 | 1.35 | 1.14 | 1.02 | 1.24 |

These sizes fit in the Infinity Cache for both sides, so this is a warm-cache comparison.
4096³ is left out: its buffers total 100.7 MB, over the cache, and the same binary read
anywhere from 91k to 99k GFLOP/s. The same GEMM has a fused bias + GELU + fp16 epilogue
(`amar_matmul_wmma_pipe_gelu`, byte-equal to the GEMM followed by a separate GELU); it made
the GPU embedder 20.9% faster.

## What runs

Agreement is teacher-forced, over 20 prompts, against llama.cpp on the same GGUF unless the
row says otherwise (min / mean, in percent). The date is when that row was last measured.

| model | quant | arch, engine | agreement (min / mean) | decode tok/s | date |
|---|---|---|---|---|---|
| Qwythos-9B (reference model) | Q4_0, BF16 | `qwen35`, engine.mojo | PPL and task PASS | 148.89 plain, 168.88 MTP k=2 | 2026-11-15 |
| Qwythos-9B-v2-MTP | Q6_K | `qwen35`, engine.mojo | 99.2 median | | 2026-09-16 |
| Ornith-1.5-9B | Q4_K_M | `qwen35`, engine.mojo, `--kq` pack | 98.44 median | 99.53 | 2026-10-26 |
| Qwen3.8-27B (nerkyor) | q4 | `qwen35`, engine.mojo | | 42.1 plain (3 prompts), 86.0 MTP k=2 (20 prompts) | 2026-11-16 |
| Qwen3.8-27B-TurboFCFusion | Q4_K_M | `qwen35`, engine.mojo | ids 19/20 vs base | 68.72 MTP k=2 | 2026-11-02 |
| OrcaSAQ-2-Cyber-27B | IQ4_XS mix | `qwen35`, engine.mojo | 81.25 / 93.75 | 46.03 plain, 69 to 70 MTP k=2 | 2026-10-17 |
| Qwen3.8-27B-OBLITERATED | Q4_K_M | `qwen35`, engine.mojo | 79.69 / 91.41 (control 81.25 / 93.75) | 39.3 plain, 63.6 MTP k=2, 3-prompt | 2026-10-22 |
| RegesCore-1.0-35B | UD-Q4_K_S | `qwen35moe`, engine.mojo | 83.1 mean; PPL and task PASS | 111.89 | 2026-09-15 |
| Qwen3.8-Flash-Next (74.5 GB, 512 experts) | UD-IQ1_M | `qwen4exp`, q4x.mojo | 97.66 (llama.cpp CPU control 97.34) | 50 to 59 | 2026-10-21 |
| Qwen2.5-7B-Instruct | Q4_K_M | `qwen2`, spark.mojo | 95.3 / 99.4 | 130.0 to 132.8 | 2026-10-21 |
| Qwen2.5-Coder-7B-Instruct | Q4_K_M | `qwen2`, spark.mojo | task PASS | | 2026-09-16 |
| MiniCPM5-2B | Q4_K_M | `llama`, spark.mojo | 92.2 / 98.0 | 233.3 | 2026-10-19 |
| vibethinker-1.5B | Q4_K_M | `qwen2`, spark.mojo | 100 / 100 against an f64 forward of the same GGUF | 385.8 | 2026-10-19 |
| Llama-3.2-1B-Instruct | Q4_K_M | `llama`, spark.mojo | ids equal to an independent engine 21/21, llama.cpp 13/21 | 562.8 | 2026-10-26 |
| lily-cybersecurity-7b | Q6_K | `llama` (Mistral), spark.mojo | 96.9 min | 94.1 to 95.0 | 2026-09 |
| granite-4.2-3b | BF16 | `granite`, spark.mojo | 98.4 min | 164.5 to 169.0 | 2026-09 |
| Spark-X2.5-4B | Q8_0 | `spark2_5`, spark.mojo | no agreement run on record | | |
| Qwen2.5-0.5B-Instruct | Q4_K_M | `qwen2`, spark.mojo | 68.8 / 85.0, UNVERIFIED | | 2026-09-17 |

- vibethinker is gated against the exact model, not llama.cpp: llama.cpp's CPU run
  disagrees with an f64 forward of the same file on 12 of 64 tokens of one prompt.
- Qwen2.5-0.5B (H=896) stays below the bar; llama.cpp's own Q4_K_M reads 82.8 / 91.4
  against its bf16 on the same prompts.
- The 27B models fit in 24 GB with the embedding table in host RAM.
- spark.mojo renders chat and tool calls the same as llama.cpp's template output
  (tool-parity gate 52/55).
- **Vision** (MoE only): images are encoded on the CPU by a libmtmd sidecar and fed with
  interleaved mrope ([`docs/vision.md`](docs/vision.md)).
- Rows dated 2026-09 have not been re-measured since.

Identity against llama.cpp is checked by **teacher-forced agreement**: both engines are fed
the same token ids and the top-1 predictions are compared. Greedy token equality is not
used past about 256 tokens, because llama.cpp's own f16-KV configuration fails it against
its own f32 reference.

The full list of features, each with its status word and the check behind it, is
[`docs/CAPABILITIES.md`](docs/CAPABILITIES.md). Failed and default-off features are listed
there on purpose. Model weights are not distributed with this repo.

## The server and its API

`baro-serve` (Rust, `serve/src/`) keeps one engine process alive and turns HTTP into the
engine's line protocol. `tools/baro serve` starts it for you; by hand:

```sh
serve/target/release/baro-serve --engine ENGINE --pack PACKDIR --tokenizer PACKDIR/tokenizer.json \
  [--host 127.0.0.1] [--port 8080] [--parallel N] [--engine-env KEY=VALUE ...]
```

### Endpoints

| route | what it does |
|---|---|
| `POST /v1/chat/completions` | chat, OpenAI shape, `stream: true` for SSE |
| `POST /v1/completions` | raw completion; `prompt` may be text or a list of token ids |
| `POST /v1/messages`, `/v1/messages/count_tokens` | Anthropic Messages API, native (no proxy): tools, thinking, SSE events; Claude Code runs against it |
| `GET /v1/models`, `/models` | the loaded model |
| `POST /v1/cancel` | stop a running request by id |
| `POST /v1/fork` | several branches from one prompt prefill |
| `POST /apply-template` | the exact text and token ids a chat request would prefill |
| `POST /tokenize`, `/detokenize` | the server's own tokenizer, bit-equal to llama.cpp's on the same GGUF |
| `GET /health` | 200 only when an engine can take work; 503 while loading or after the engine died |
| `GET /props` | engine limits: window, speculation depth, KV type |
| `POST /v1/embeddings`, `/api/embeddings` | embeddings: bge-small-en-v1.5 by default ([below](#embeddings)) |
| `/api/chat`, `/api/generate`, `/api/tags`, `/api/ps`, `/api/show`, `/api/pull`, `/api/version` | Ollama-compatible |
| `POST /tool/crawl`, `GET /brain/pull`, `POST /brain/push` | companion-app routes: a page fetch through a local crawl4ai, and sync of an agent notes folder (`BARO_CRAWL4AI`, `BARO_BRAIN_DIR`) |
| `GET /` | a chat page in the browser |

`/health` answers with a status code, not only a body: a load balancer or script can poll
it. The port opens before the weights finish loading, so "port open" does not mean
"ready". The full contract is [`serve/PROTOCOL.md`](serve/PROTOCOL.md).

### Request features

- **Sampling:** `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `presence_penalty`,
  `frequency_penalty`. Sampling runs on the GPU inside the decode loop and composes with
  speculation. `temperature: 0` is greedy. `BARO_DEFAULT_TOP_K/P` and `BARO_DEFAULT_MIN_P`
  set server defaults for requests that leave them out.
- **Tool calls** come back in the OpenAI `tool_calls` shape (or Anthropic `tool_use`
  blocks on `/v1/messages`), streamed as they form. `tool_choice` takes `auto`, `none`,
  `required` or a named function.
- **Structured output:** `response_format` with a JSON schema is enforced by a token mask
  on the device: the model cannot emit a token that breaks the schema
  ([`docs/grammar.md`](docs/grammar.md)).
- **Multi-turn:** each message boundary is checkpointed, so a follow-up turn prefills only
  the new message. `usage.baro.cached_tokens` says how many prompt tokens came from the
  cache.
- **Timings:** every response carries `timings` with tok/s, the decode `path`, and the
  engine's `fail` word. A run with a non-zero fail word returns HTTP 502
  `engine_invalid_tokens`, never a 200 with bad text.
- **Concurrency:** one request decodes at a time and the rest queue, except on dense packs
  with `--parallel N`: sessions share one weight copy and the shared prompt prefix and
  decode together (R = 2 pair megakernel, batched steps on K-quant packs). Past the KV
  pool the server answers HTTP 429
  ([`docs/design/multi-session-serve.md`](docs/design/multi-session-serve.md)).
- **Engine probes:** a request can ask for the final hidden state (`hidden`), the top-K
  logits (`logits_topk`) or a sentence embedding (`embed`).

### Embeddings

`/v1/embeddings` and `/api/embeddings` answer with **bge-small-en-v1.5** (384 dimensions,
CLS pooling, L2-normalised, 512 tokens) when `model` is absent or names it. It runs in a
Mojo child process (`embed/serve_embed.mojo`), so it needs no chat engine:
`baro-serve --embed-only --port 8081` serves it alone. `input` is a string or an array of
strings; `encoding_format` is `float` or `base64`. Naming the loaded chat model returns
the chat model's last hidden state instead.

```bash
embed/build.sh                      # serve_embed + norm.tsv into ~/.cache/baro/embed
baro-serve --embed-only --port 8081 # needs ~/Models/bge-small-en-v1.5
curl -s localhost:8081/v1/embeddings -d '{"input":["hello","world"]}' -H 'content-type: application/json'
```

`bench/embed-gate.sh` checks 30 fixed texts against the ONNX model (cosine >= 0.9999) and
rejects a mean-pooling build.

### Several engines: `baro-router`

`serve/src/bin/router.rs` puts several `baro-serve` nodes behind one port and routes by
model and load, judging each node by its `/health` code. `tools/baro multi-serve
MODEL.gguf --devices 0,1` starts one node per GPU.

### The Mojo front (experimental)

`serve/front/` is a second HTTP front written in Mojo on the same engine protocol: chat and
completion routes (tool calls included), `/health`, `/props`, `/v1/models`, `/v1/cancel`,
`/tokenize`, `/detokenize`. On a real 27B engine it gave the same final answers as the
Rust front on 5 prompts (`bench/front-real-check.sh`). It has no prefix cache, no Ollama or
Anthropic routes and no `--parallel` yet. Use `baro-serve` unless you work on the front.

## Clients: terminal, window, browser

- **`tools/baro chat`**: terminal client. Streams the reply, shows the thinking steps
  (`--hide-think` hides them) and runs tool calls (`--tools`, `--tool-exec`).
- **`tools/baro gui [--model MODEL.gguf] [--port N] [--fps N]`**: the face
  (`face/app.mojo`, on the vendored `imkit/` GUI). It uses the front on the port, or starts
  `baro serve` for `--model`. Tabs:
  - **Chat**: streams over SSE (Send, Stop, Copy), with an engine panel (state, queue,
    model card, fail word, tok/s, VRAM bar).
  - **Sampling, Engine, Front**: one control per setting from
    [`docs/settings.toml`](docs/settings.toml), with help text, default, range and restart
    need. A change is staged, never applied silently.
  - **Presets**: Fast, Balanced, Quality; Chat, Code, Reasoning; Low VRAM, Long context;
    plus your own (`~/.config/mojo-baro/presets.toml`). **Apply** shows a plan with a
    `pack-fit` verdict, restarts the engine with the staged values and waits for
    `/health`.
  - **Model**: GGUF path, pack lookup in the cache, the model, pack and limits card.

  The face is checked offscreen (`face/gate.sh`, `face/chat-gate.sh`,
  `face/settings_gate.sh`: frames read back with OCR). It needs `libSDL3.so.0` and
  `libfreetype.so.6`. The frame cap defaults to 30 fps: at 60 a resident 9B engine lost
  about 13% on some prompts (`face/window-cost.sh`). `scripts/install-desktop.sh` adds a
  desktop entry and icon.
- **Browser**: `GET /` on the server port is a chat page that shows thinking and names
  the served model.

## Configuration

Settings come from three places, in this order of precedence:

1. **Environment variables** read by the engine (`BARO_*`). The Rust front passes its own
   environment to the engine and adds `--engine-env KEY=VALUE`.
2. **Profiles** in `profiles/*.toml`: a named set of settings for one use, such as
   `turbo27b-coder.toml` (27B as a 131k-context coding server).
   `tools/baro serve --profile NAME`.
3. **Build flags** (`-D NAME=VALUE`), which fix the model shape and some kernel choices at
   compile time. `tools/baro` sets these from the GGUF.

Every setting the code reads is listed, with type, default, range, restart need and help
text, in [`docs/settings.toml`](docs/settings.toml), the single source the face reads.
`tools/settings-check.mojo` fails if that file and the source disagree, and generates the
table below. The ones you are most likely to touch:

<!-- settings-table:begin (generated by tools/settings-check --readme; edit docs/settings.toml) -->

| setting | default | what it does |
|---|---|---|
| `BARO_DEFAULT_TOP_K` | unset | baro-serve: top-k for requests that leave it out (llama-server's default: 40 / 0.95 / 0.05). Unset: untruncated. |
| `BARO_DEFAULT_TOP_P` | unset | baro-serve: top-p for requests that leave it out (llama-server's default: 40 / 0.95 / 0.05). Unset: untruncated. |
| `BARO_DEFAULT_MIN_P` | unset | baro-serve: min-p for requests that leave it out (llama-server's default: 40 / 0.95 / 0.05). Unset: untruncated. |
| `BARO_KVQ` | `f32` | KV cache dtype: f32, or int8 (dense only; RULER 100.0 at 64k and 128k; state files become BAROST03 = pool pages, 32k load 0.061 s, P7-int8-pool-sta... |
| `BARO_TMAX` | `16384` | Context window the KV pool is sized for, in tokens (engine default 1088, baro-serve 32768, face default 16384). |
| `BARO_MEGA` | `1` | Persistent megakernel; default 1 on dense, 0 on MoE and on native K-quant packs (the table shows the dense value). 0 selects the launch path. |
| `BARO_PREFILL` | `1` | Batched prefill (up to BARO_PREFILL_C rows per chunk). |
| `BARO_SPEC` | `1` | Speculative decoding with the model's MTP draft head (dense |
| `BARO_TIER` | unset | Experts in host RAM with this many cached per layer in VRAM (64 = 5.2 GB) |
| `BARO_SEQS` | `1` | Sequences in flight on one engine. |
| `BARO_PACK` | `.work/moe-w1/pack` | Weight pack directory the engine loads (pack.bin + index.txt, experts.bin for a tier pack). |
| `BARO_CACHE` | `~/.cache/baro` | Directory for compiled engines, manifests and packs (tools/baro, default ~/.cache/baro/<id>/). |
| `BARO_POOL` | `1` | Number of engine processes baro-serve spawns |

<!-- settings-table:end -->

**Size the window against free VRAM.** The engine allocates its buffers from free VRAM, so
the desktop and other GPU programs count. Read used VRAM first
(`rocm-smi --showmeminfo vram`), then size `BARO_TMAX` so the pack plus KV fits;
`BARO_KVQ=int8` halves the KV part. `pack-fit PACK --tmax N --used-gb G` answers before a
launch. When a pack does not fit, tier mode keeps MoE experts in host RAM.

## Other backends: CPU, NVIDIA, RDNA4, phone

| target | state |
|---|---|
| **CPU** (`cpu/engine_cpu.mojo`) | correct and slow. Dense qwen35 decode, op for op the GPU launch path, same line protocol. 1280 of 1280 teacher-forced ids agree with the GPU build (`bench/check-cpu-engine.sh`). About 7 tok/s at 8 threads on a quiet 7800X3D; prefill replays the decode step; no K-quant path yet |
| **NVIDIA** | port in progress, never run on an NVIDIA card. `serve/gpu_rt` loads HIP or CUDA by target (`BARO_CUDART_LIB`); int8 dots lower to `dp4a` on NVIDIA and WMMA paths refuse to build there. `tools/zluda/` runs `sm_XX` Mojo builds on the 7900 XTX through ZLUDA (partial: MAX kernels run, a mixed-sign dp4a patch is in the tree). UNVERIFIED |
| **RDNA4** (gfx1200, gfx1201: RX 9060 XT, 9070, 9070 XT, Radeon AI PRO R9700) | builds, never run on a card. Every WMMA epilogue goes through `acc_row` in `kernels/arch.mojo`; the gfx12 accumulator row order is emulated on gfx1100 (`-D BARO_EMU_GFX12_ROWS=true`, `bench/rdna4-epi-gate.sh`): bit for bit equal, 20 of 20 engine prompts agree ([`docs/rdna4/README.md`](docs/rdna4/README.md)). The megakernel turns itself off on cards with fewer than 96 compute units |
| **Phone GPU** (Adreno 840) | Mojo has no Adreno backend; the phone engine drives OpenCL C kernels from Mojo. Qwen3-8B Q4_0 decode 13.88 tok/s against llama.cpp OpenCL 11.56 (1.17x), at about 98% of the measured read ceiling; no batched GPU prefill yet. Tooling in `phone/`, report [`docs/phone-adreno.md`](docs/phone-adreno.md) |
| other RDNA3 cards | same ISA, should build; tuning (96 CUs, 96 MB Infinity Cache) unmeasured. [docs/amd-family.md](docs/amd-family.md) |

If you own an RDNA4 card, `bench/report.sh --rdna4` (no weights needed) writes one local
text file: card, intrinsic table, a WMMA numeric check, kernel log. Paste it into a
"hardware report" issue; a failing report is the most useful one.

## Testing

| command | needs | what it checks |
|---|---|---|
| `bench/preflight.sh` | CPU | `tools/ci-checks.sh`, every test binary builds, both engines build. Writes a stamp other gates check |
| `./run-tests.sh` | GPU | kernel parity tests (each multi-row kernel against the m=1 decode kernel, byte for byte), the kernel census, the Anthropic adapter |
| `cargo test` in `serve/` | CPU | the Rust front against a stub engine |
| `serve/front/test_*.mojo` | CPU | the Mojo front against fake engines |
| `bench/check-cpu-engine.sh` | GPU + CPU | the CPU tier against the GPU engine, with a self-check and a mutant that must fail |
| `bench/front-real-check.sh` | GPU | both fronts on one real engine |
| `face/settings_gate.sh`, `face/chat-gate.sh`, `face/gate.sh` | small GPU job | the face offscreen; frames read back with OCR |
| `bench/rdna4-epi-gate.sh` | CPU build, GPU run | gfx12 accumulator rows emulated on gfx1100: ISA identity, parity, planted-mutant kills, engine agreement |
| `bench/embed-gate.sh` | CPU | the embedder against the ONNX model |
| `tools/settings-check.mojo` | CPU | `docs/settings.toml` against the code and the Configuration table |

CI (`.github/workflows/checks.yml`) runs the CPU checks. GPU jobs on the author's machine
run under a local queue (`gpu-wait run`); on your machine, run the inner command directly.

## Measuring

Three instruments; every perf or energy claim names which one produced it:

| instrument | measures | entry |
|---|---|---|
| **wavewait** | per-instruction stall causes of a GPU kernel from a real SQTT trace; energy share per kernel class | `bench/ww.sh KERNEL ENGINE PACK OUT`, goldens in `bench/ww-golden/` |
| **corewait** | host CPU time and energy per instruction (hardware counters, RAPL) | `corewait record` |
| **heaptide** | host and VRAM memory growth, leaks | `heaptide run --gpu` |

A kernel change carries a wavewait capture against its golden. Throughput is decided by
a 20-prompt same-job A/B (`bench/ab-resident.sh`): wavewait says why, the A/B says how
much. `bench/trace-pair.sh` takes rocprofv3 kernel traces of two engine arms in one job.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `pack/profile mismatch: profile H 4096 ... implies H 5120` | the engine was compiled for another model shape | use `tools/baro serve`, or build with the shape flags from `tools/model-id.py MODEL.gguf --json` |
| `no tokenizer.json loaded` (HTTP 503 on chat) | the pack has no `tokenizer.json` | pass `--tokenizer`, or rebuild the pack with `tools/baro serve` |
| decode is 10x slower than the tables, no error | weights spilled from VRAM to GTT (system RAM over PCIe) because something else holds VRAM | read `gtt_used` in `/sys/class/drm/card*/device/`; free VRAM or lower `BARO_TMAX` |
| out of memory with VRAM free | MAX's small-allocation pool is 2% of free VRAM | set `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT` higher |
| `/health` 503 for a long time after start | the pack is still loading; the port opens first | wait for 200 |
| the engine prints `ready`, exits 0, served nothing | its stdin was empty | feed requests on stdin, or let a front start it |
| wrong results while several GPU programs run | four or more active HIP processes on one card corrupt results, even finite ones | run at most two GPU jobs at a time |
| text differs between two runs at temperature 0 | near-tie tokens flip between decode paths and with a warm prefix cache | compare with teacher-forced agreement, not text equality |

## How numbers get into this repo

Every performance claim is preregistered: the question, the instrument, the predicted
range and the falsifier are committed **before** the run, and the result is recorded
against them whether or not it agreed ([`bench/PROTOCOL-RULES.md`](bench/PROTOCOL-RULES.md)).
The reference arm gets the same treatment as ours: cold-cache rotation, clock probe and a
read-back of every arm-defining parameter from the running system.

An early baseline here recorded a 2x lead over hipBLASLt. It was measuring an untuned
vendor call: fixing three defects in our own shim took hipBLASLt from 2497 to 5201 GFLOP/s
and erased the lead. **A vendor baseline that looks easy to beat is a bug in your harness
until proven otherwise.**

[`docs/BASELINE.md`](docs/BASELINE.md) holds every verified number and known trap;
[`docs/rules.md`](docs/rules.md) holds every lesson as a `what broke -> rule` line.
`bench/` is a lab notebook, not a product: its scripts assume the author's machine
(llama.cpp at `$HOME/llama.cpp`, models under `$HOME/Models`, a local GPU queue). Read it
for the method; expect to edit paths before it runs elsewhere.

## Layout

| | |
|---|---|
| `kernels/` | Mojo GPU kernels and their parity tests ([`docs/KERNELS.md`](docs/KERNELS.md), generated) |
| `serve/` | the engines, tokenizer, prefix cache, GPU runtime shim; `serve/src/` the Rust front and router, `serve/front/` the Mojo front |
| `cpu/` | the CPU tier: ops, weight unpacking, thread pool, `engine_cpu.mojo` |
| `embed/` | bge-small-en-v1.5 in Mojo, the process `baro-serve` starts for `/v1/embeddings` |
| `face/`, `imkit/` | the window (`tools/baro gui`) and the vendored immediate-mode GUI |
| `grammar/`, `minja/`, `uregex/`, `toml/`, `json/` | JSON-schema grammar, chat-template renderer, regex engine, TOML and JSON readers (all Mojo) |
| `latentos/` | state handoff between engine processes (memfd and HIP IPC) |
| `miniagi/` | a Mojo training stack on the same card: trainer, flash attention forward and backward, WMMA GEMM kernels, tokenizer tools |
| `profiles/` | named settings sets for `tools/baro serve --profile` |
| `bench/` | benchmark harnesses and the frozen protocols |
| `shim/` | C++ hipBLASLt shim behind a C ABI, the vendor reference |
| `tools/` | `baro` CLI, GGUF packer, reference implementations, gates, ZLUDA tooling |
| `phone/` | Adreno tooling and OpenCL race kernels |
| `docs/` | current truth: BASELINE, CAPABILITIES, KERNELS, settings, design notes |

The tokenizer is Mojo, bit-equal to llama.cpp on the same GGUF
([`docs/TOKENIZER.md`](docs/TOKENIZER.md)).

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) lists where help is wanted, the check that decides
each item, and the template every change fills in.

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party material is named in
the NOTICE: the vendored `toml/` reader (DataBooth, Apache-2.0) and the pre-tokenizer
regexes in `serve/pretok-table.json` (from llama.cpp, MIT). llama.cpp is fetched at a
pinned commit, never redistributed.
