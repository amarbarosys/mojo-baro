# mojo-baro

An LLM inference engine for AMD RDNA3 GPUs, with every GPU kernel written in
[Mojo](https://www.modular.com/mojo). It serves chat over an OpenAI-compatible
(and Ollama-compatible) HTTP API on one RX 7900 XTX.

On matched Q4_0 weights it decodes the dense Qwythos-9B **1.34x faster than
llama.cpp** (148.89 against 111.34 tok/s, 20-prompt median, same job). On the
35B MoE it is level with llama.cpp. On q8 it is 1.07x ahead (same stint, 2026-10-26). From a Q4_K_M file it is
1.06x ahead on a native K-quant pack (agreement 98.44%, megakernel, same
stint, 2026-10-26, Mojo 1.1; on the current Mojo 1.2 pin that megakernel is
slower than the launch path, see below). Every number here has a preregistered protocol in `bench/`
and a check you can run.

The project is about the kernels. Consumer RDNA3 cards are where most people
have 24 GB of VRAM, and the vendor libraries are not tuned for them. This repo
measures how much of that gap is real.

**Have an AMD GPU that is not a 7900 XTX?** Your card is the most useful thing
you can contribute. See [docs/amd-family.md](docs/amd-family.md): fifteen
minutes, no model weights needed.

**Contents:** [Quick start](#quick-start) · [How it works](#how-it-works) ·
[Results](#results) · [What runs](#what-runs) · [The server and its API](#the-server-and-its-api) ·
[Configuration](#configuration) · [CPU tier](#cpu-tier-decode-without-a-gpu) ·
[Testing](#testing) · [Troubleshooting](#troubleshooting) ·
[How numbers get into this repo](#how-numbers-get-into-this-repo) · [Layout](#layout) ·
[Portability](#portability) · [Contributing](#contributing) · [License](#license)

## Quick start

### What you need

| | |
|---|---|
| GPU | AMD Radeon RX 7900 XTX (gfx1100, 24 GB). Other RDNA3 cards should build; their tuning is unmeasured |
| OS | Linux with ROCm 7.2 (`/dev/kfd` readable by your user) |
| Toolchain | [`uv`](https://docs.astral.sh/uv/), Rust (cargo) for the HTTP front, CMake and a C++ compiler for the hipBLASLt shim, `jq` and `curl` for the scripts |
| RAM | 32 GB is enough for 9B models; the 35B MoE and the 27B models use host RAM for part of their weights |
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

On the first run for a model shape this takes a few minutes: `tools/baro` reads the architecture
and dimensions from the GGUF header, compiles an engine for exactly that shape, and converts the
weights into a pack. Both go to `~/.cache/baro/<shape-id>/` (override with `$BARO_CACHE`). Every
later run of the same file starts in seconds, and another checkpoint of the same shape reuses the
compiled engine. An unsupported architecture or tokenizer is refused before any build starts.

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

Any OpenAI client works with `base_url="http://localhost:8080/v1"` and any API key. Ollama
clients work against the same port (`/api/chat`, `/api/generate`, `/api/tags`). `tools/baro chat`
is a terminal client.

`tools/baro gui [--model MODEL.gguf] [--port N] [--fps N]` opens the face (`face/app.mojo`, built on the vendored
`imkit/`): chat on the left, the engine panel on the right, in one window. It uses the front already on the port, or
starts `baro serve` for `--model`. Skeleton only for now: placeholder text and the front's `/health` line. It needs the
system libraries `libSDL3.so.0` (Arch: `pacman -S sdl3`) and `libfreetype.so.6` (`pacman -S freetype2`); the face checks
both at start and names the missing one. The frame cap defaults to 30 fps: at 60 a resident 9B engine lost about 13% on 6
to 7 of 20 prompts, at 30 and 15 on 1 (`face/window-cost.sh`).

## How it works

```
 HTTP client (curl, OpenAI SDK, Ollama client, tools/baro chat)
      |
      v
 baro-serve  (Rust, serve/src/)            baro-front (Mojo, serve/front/, experimental)
   chat template, tokenizer, tool calls,     same routes, same engine protocol
   JSON-schema masks, queue, cancel,
   per-message checkpoints, /health
      |
      |  one JSON line per request on stdin, one JSON line per token on stdout
      |  (serve/PROTOCOL.md)
      v
 engine process (Mojo), compiled for one model shape
   serve/engine.mojo   qwen35 dense and qwen35moe (MoE): the tuned engine
   serve/spark.mojo    generic dense: llama, qwen2, granite, spark2_5
   serve/q4x.mojo      qwen4exp, 512 experts streamed from host RAM
   cpu/engine_cpu.mojo decode on the CPU, no GPU (qwen35 dense)
      |
      v
 kernels/*.mojo   GEMV and WMMA GEMM per quant format, flash attention, gated-delta SSM scan,
                  MoE router, device sampler, the persistent megakernel
```

**One engine per shape, compiled.** Layer count, hidden size, head counts and the architecture are
compile-time constants (`-D BARO_H=4096 ...`), so loops have fixed trip counts and the compiler
can unroll and schedule them. The cost: an engine built for a 9B pack refuses a 27B pack
(`pack/profile mismatch ... implies H 5120`). `tools/baro` hides this by keying its cache on the
shape.

**Packs, not GGUF at run time.** `tools/baro` converts the GGUF once into a pack: one flat
`pack.bin` with weights laid out for the kernels that read them, an `index.txt`, the tokenizer and
an `identity.json` that names the source file by sha256. A pack loads with one sequential read.
Pack formats: q4 and q8 (requantized), native K-quant (`--kq`, keeps the GGUF's Q4_K / Q6_K blocks
and decodes them in the kernel) and split packs for MoE experts in host RAM.

**Decode paths.** The engine reports which one served each request (`timings.path`):

| path | what it is |
|---|---|
| `mega` | one persistent kernel launch per token for the whole forward pass (dense default) |
| `launch` | one launch per op; the reference path and the MoE default |
| `spec` | speculative decode with the model's own MTP draft head: k tokens drafted, one verify pass; output identical to greedy |
| `prefill` | the prompt in batched chunks through WMMA GEMMs |

**Memory.** KV is f32 by default or int8 (`BARO_KVQ=int8`, dense only, 33,280 bytes per token on
the 9B). MoE experts can live in host RAM with a VRAM cache per layer (`BARO_TIER`). Each message
boundary of a chat is checkpointed, so the next turn prefills only the new message.

## Results

### Decode against llama.cpp

Same box, same GGUF, greedy, no speculation on either side.

| model, weights | llama.cpp tok/s | mojo-baro tok/s | ratio |
|---|---|---|---|
| **Qwythos-9B, Q4_0 both sides, 20-prompt median, 2026-10-12** | **111.34** | **148.89** | **1.34x** |
| Qwythos-9B, Q8_0 both sides, 20-prompt median, same stint, 2026-10-26 | 75.46 | 80.38 | 1.07x |
| Ornith-1.5-9B Q4_K_M, our q8 pack (2x llama.cpp's weight bytes), same stint, 2026-10-26 | 93.40 | 80.48 | 0.86x |
| Ornith-1.5-9B Q4_K_M, our q4 pack (requantized; FAILS agreement, 92.97% median), same stint, 2026-10-26 | 93.07 | 146.93 | 1.58x, not a claim |
| Ornith-1.5-9B Q4_K_M, native K-quant pack (`--kq`, launch path), agreement 98.44% median, same stint, 2026-10-26 | 93.25 | 96.64 | 1.04x |
| **Ornith-1.5-9B Q4_K_M, native K-quant pack on the megakernel (default), same stint, 2026-10-26** | **94.14** | **99.53** | **1.06x** |
| Ornith-1.5-9B Q4_K_M, native K-quant pack, **prompt prefill** 1024 / 16384 tokens (seconds, K-quant decode inside the WMMA GEMM, run 10), 2026-10-26 | 0.298 s / 5.66 s | 0.259 s / 4.54 s | 0.87x / 0.80x of the time |
| Qwen2.5-7B Q4_K_M, native K-quant path (3-prompt quick run, 2026-10-21) | 121.2 | 132.8 | 1.10x |
| RegesCore-35B MoE (different stints) | 109.92 | 111.89 | 1.02x |

The Q4_0 row is one same-stint job ([`bench/llama-stint-protocol.md`](bench/llama-stint-protocol.md)):
ours faster on 20/20 prompts. llama.cpp ran in its fast configuration (flash
attention, f16 KV); ours ran with f32 KV. A ratio is only quoted as a claim when
both sides were measured in the same stint; the other rows are older or shorter
runs and say so.

**Toolchain.** The rows above were measured on Mojo 1.1.0 or earlier. The repo now pins
Mojo 1.2.0 / MAX 26.7.0 (2026-10-29); that move was gated on builds and tests,
not on throughput, and of these rows only the K-quant megakernel was re-timed: on 1.2 it reads
87.90 tok/s against the launch path's 98.29 on the same pack (`mega-kq-check`,
2026-10-30), 88.13 after the delta-phase fold, ids 20/20
([`docs/BASELINE.md`](docs/BASELINE.md) "Megakernel on K-quant packs", cause
open). `BARO_MEGA=0` selects the launch path. Until the other rows are
re-timed on 1.2, read them as numbers of the older toolchain.

- **Speculative decode** with the model's own MTP head is on by default and
  gives output identical to plain greedy decode: 1.104x at k=2 as a 20-prompt
  median ([`bench/mtp-protocol.md`](bench/mtp-protocol.md)).
- **Prefill** is within about 1.3x of llama.cpp from 8k to 32k context
  (17.42 s against 13.18 s at 32k), using f16 WMMA flash attention and a
  one-wave SSM scan. On a native K-quant pack the prompt runs batched through a
  WMMA GEMM that decodes Q4_K / Q6_K in its B staging: 1024 tokens in 0.259 s and 16384 in 4.54 s
  against llama.cpp's 0.298 and 5.66 s ([`bench/llama-stint-protocol.md`](bench/llama-stint-protocol.md) run 10).
- **Sampling** runs on the device inside the decode loop and composes with
  speculation: T=0.7, top_p 0.9 at 147.15 tok/s.

### A cold-cache vendor beat at the decode shape

Single-token decode streams weights out of VRAM. At **M=1, K=4096, N=12288**,
with weights rotated out of the 96 MB Infinity Cache
([`bench/coldcache-protocol.md`](bench/coldcache-protocol.md)):

| kernel | us per launch |
|---|---|
| `amar_matmul_skinny_v2` CPT=8 | 137.8 to 138.9 |
| **`amar_matmul_skinny_m1` CPT=8** | **121.2 to 121.8** |
| hipBLASLt f16 | 122.1 to 123.3 |

The ranges do not overlap; the margin is about 1%. The cause is occupancy, not
bytes: one row drops the LDS staging and shrinks the accumulator, so more waves
stay resident. It took four preregistered rounds, three of which falsified
their own predictions.

### An fp16 WMMA GEMM ahead of hipBLASLt

Two LDS buffers, one barrier per K-step, a two-deep global prefetch,
XOR-swizzled A, 188 VGPR and no spills; the tile shape is picked per launch
([`bench/wmma-fp16-protocol.md`](bench/wmma-fp16-protocol.md)):

| size | 256 | 512 | 1024 | 1536 | 2048 | 3072 | 3584 |
|---|---|---|---|---|---|---|---|
| **ours, GFLOP/s** | 6372 | 30642 | 74824 | 93632 | 91300 | 99307 | 105786 |
| hipBLASLt | 6288 | 26324 | 63623 | 69332 | 80203 | 97224 | 85671 |
| ratio | 1.01 | 1.16 | 1.18 | 1.35 | 1.14 | 1.02 | 1.24 |

These sizes fit in the Infinity Cache for both sides, so this is a warm-cache
comparison. 4096³ is left out: its buffers total 100.7 MB, over the cache, and
the same binary read anywhere from 91k to 99k GFLOP/s.

## What runs

Fifteen models across six architectures. Agreement is teacher-forced, over
20 prompts, against llama.cpp on the same GGUF unless the row says otherwise
(min / mean, in percent). The date is when that row was last measured.

| model | quant | arch, engine | agreement (min / mean) | decode tok/s | date |
|---|---|---|---|---|---|
| Qwythos-9B (reference model) | Q4_0, BF16 | `qwen35`, engine.mojo | PPL and task PASS | 148.89 (spec off), 150.96 (MTP k=2) | 2026-10-12 |
| Qwythos-9B-v2-MTP | Q6_K | `qwen35`, engine.mojo | 99.2 median | | 2026-09-16 |
| Ornith-1.5-9B | Q4_K_M | `qwen35`, engine.mojo, `--kq` pack | 98.44 median against llama.cpp (teacher-forced, 20 prompts) | 99.53 | 2026-10-26 |
| OrcaSAQ-2-Cyber-27B | IQ4_XS mix | `qwen35`, engine.mojo | 81.25 / 93.75 | 46.03 (spec off), 69 to 70 (MTP k=2), 20-prompt medians | 2026-10-17 |
| Qwen3.8-27B-OBLITERATED | Q4_K_M | `qwen35`, engine.mojo | 79.69 / 91.41 (control 81.25 / 93.75) | 39.3 (spec off), 63.6 (MTP k=2), 3-prompt medians | 2026-10-22 |
| RegesCore-1.0-35B | UD-Q4_K_S | `qwen35moe`, engine.mojo | 83.1 mean; PPL and task PASS | 111.89 | 2026-09-15 |
| Qwen3.8-Flash-Next (74.5 GB, 512 experts) | UD-IQ1_M | `qwen4exp`, q4x.mojo | 97.66 (llama.cpp CPU control 97.34) | 17 to 20 ms/token (50 to 59 tok/s) | 2026-10-21 |
| Qwen2.5-7B-Instruct | Q4_K_M | `qwen2`, spark.mojo | 95.3 / 99.4 | 130.0 to 132.8 | 2026-10-21 |
| Qwen2.5-Coder-7B-Instruct | Q4_K_M | `qwen2`, spark.mojo | task PASS | | 2026-09-16 |
| MiniCPM5-2B | Q4_K_M | `llama`, spark.mojo | 92.2 / 98.0 | 233.3 | 2026-10-19 |
| vibethinker-1.5B | Q4_K_M | `qwen2`, spark.mojo | 100 / 100 against an f64 forward of the same GGUF | 385.8 | 2026-10-19 |
| Llama-3.2-1B-Instruct | Q4_K_M | `llama`, spark.mojo | 95.3 min (range 95.3 to 100), before the rope_freqs fix | 558.3 to 562.8 (21-prompt medians) | 2026-10-26 |
| lily-cybersecurity-7b | Q6_K | `llama` (Mistral), spark.mojo | 96.9 min (range 96.9 to 100) | 94.1 to 95.0 | 2026-09 |
| granite-4.2-3b | BF16 | `granite`, spark.mojo | 98.4 min (range 98.4 to 100) | 164.5 to 169.0 | 2026-09 |
| Spark-X2.5-4B | Q8_0 | `spark2_5`, spark.mojo | no agreement run on record | | |
| Qwen2.5-0.5B-Instruct | Q4_K_M | `qwen2`, spark.mojo | 68.8 / 85.0, UNVERIFIED | | 2026-09-17 |

- vibethinker is gated against the exact model, not llama.cpp: llama.cpp's
  CPU run disagrees with an f64 forward of the same file on 12 of 64 tokens of
  one prompt, so a llama.cpp bar is unreachable for it.
- Qwen2.5-0.5B (H=896) stays below the bar; llama.cpp's own Q4_K_M reads
  82.8 / 91.4 against its bf16 on the same prompts.
- The 27B models fit in 24 GB with the embedding table in host RAM (17.0 GB
  peak VRAM); context stays at 1088 to 2176 tokens.
- spark.mojo, 2026-10-26: prompt prefill runs batched (128-row chunks, dequant + WMMA GEMM), 10 to 16x
  faster on 350-token prompts, generated ids identical to token-by-token on all six spark models in
  `~/Models`; fused gate/up and split-K QKV GEMVs (+4.2%, +2.5% decode on Llama 1B); `rope_freqs` is applied
  (Llama 3.x was rotating at the wrong frequencies): Llama 1B greedy ids now equal wavegroup's on 21/21 prompts
  and llama.cpp's on 13/21. Agreement for the Llama row has not been re-run since that fix. Receipts:
  `lab/wavegroup/docs/receipts/2026-10-26-*`.
- Lab gate (`lab/wavegroup/tools/gate.sh`, every llama.cpp window at 16384), 2026-10-26: spark Llama 1B
  562.8 tok/s against llama.cpp 412.9 (1.364x, 21-prompt medians, same job), ids equal to wavegroup on 21/21.
- Rows dated 2026-09 have not been re-measured since; `bench/quick-speed.sh`
  on 2026-10-21 put the dense Q4_0 arm at the same 1.34x ratio.

Identity against llama.cpp is checked by **teacher-forced agreement**: both
engines are fed the same token ids and the top-1 predictions are compared, over
20 prompts. Greedy token equality is not used past about 256 tokens, because
llama.cpp's own f16-KV configuration fails it against its own f32 reference.

The full list of features, each with its status word and the check behind it,
is [`docs/CAPABILITIES.md`](docs/CAPABILITIES.md). Failed and default-off
features are listed there on purpose.

Model weights are not distributed with this repo.
## The server and its API

`baro-serve` (Rust, `serve/src/`) keeps one engine process alive and turns HTTP into the engine's
line protocol. `tools/baro serve` starts it for you; to start it by hand:

```sh
serve/target/release/baro-serve --engine ENGINE --pack PACKDIR --tokenizer PACKDIR/tokenizer.json \
  [--host 127.0.0.1] [--port 8080] [--parallel N] [--engine-env KEY=VALUE ...]
```

### Endpoints

| route | what it does |
|---|---|
| `POST /v1/chat/completions` | chat, OpenAI shape, `stream: true` for SSE |
| `POST /v1/completions` | raw completion; `prompt` may be text or a list of token ids |
| `GET /v1/models`, `/models` | the loaded model |
| `POST /v1/cancel` | stop a running request by id |
| `POST /v1/fork` | copy a session's state to a second engine |
| `POST /tokenize`, `/detokenize` | the server's own tokenizer, bit-equal to llama.cpp's on the same GGUF |
| `GET /health` | 200 only when an engine can take work; 503 while loading or after the engine died |
| `GET /props` | engine limits: window, speculation depth, KV type |
| `/api/chat`, `/api/generate`, `/api/tags`, `/api/ps`, `/api/show`, `/api/version` | Ollama-compatible |
| `POST /api/embeddings` | embeddings |

`/health` answers with a status code, not only a body: a load balancer or script can poll it.
The port opens before the weights finish loading, so "port open" does not mean "ready".

### Request features

- **Sampling:** `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `presence_penalty`,
  `frequency_penalty`. Sampling runs on the GPU inside the decode loop and composes with
  speculation. `temperature: 0` is greedy.
- **Tool calls** come back in the OpenAI `tool_calls` shape. `tool_choice` takes `auto`, `none`,
  `required` or a named function.
- **Structured output:** `response_format` with a JSON schema is enforced by a token mask on the
  device: the model cannot emit a token that breaks the schema.
- **Multi-turn:** each message boundary is checkpointed, so a follow-up turn prefills only the new
  message. `usage.baro.cached_tokens` says how many prompt tokens came from the cache.
- **Timings:** every response carries `timings` with tok/s, the decode `path`, and the engine's
  `fail` word. A run with a non-zero fail word returns HTTP 502 `engine_invalid_tokens`, never a
  200 with bad text.
- **Concurrency:** one request decodes at a time and the rest queue, except on a dense K-quant
  pack with `--parallel N`: N sessions share one weight copy and the shared prompt prefix and
  decode together in batched steps. Past the KV pool the server answers HTTP 429.
  [`docs/design/multi-session-serve.md`](docs/design/multi-session-serve.md).

### Several engines: `baro-router`

`serve/src/bin/router.rs` puts several `baro-serve` nodes behind one port and routes by model and
load. It judges a node by its `/health` code. `tools/baro multi-serve MODEL.gguf --devices 0,1`
starts one node per GPU.

### The Mojo front (experimental)

`serve/front/` is a second HTTP front written in Mojo, built on the same engine protocol. It
serves the chat and completion routes (tool calls included), `/health`, `/props`, `/v1/models`,
`/v1/cancel`, `/tokenize` and `/detokenize`. On 2026-11-10 it served
the real engine for the first time (Turbo 27B pack, `bench/front-real-check.sh`): the same final
answers as the Rust front on 5 prompts, the same decode path, 503 during load, a clean SIGTERM.
It has no prefix cache, no Ollama routes, no router support beyond `/health` and no `--parallel` yet. Use `baro-serve` unless you
are working on the front itself.

The contract between either front and the engine is [`serve/PROTOCOL.md`](serve/PROTOCOL.md).

## Configuration

Settings come from three places, in this order of precedence:

1. **Environment variables** read by the engine (`BARO_*`). The Rust front passes its own
   environment to the engine and adds `--engine-env KEY=VALUE`.
2. **Profiles** in `profiles/*.toml`: a named set of settings for one use, such as
   `turbo27b-coder.toml` (27B as a 131k-context coding server). `tools/baro serve --profile NAME`.
3. **Build flags** (`-D NAME=VALUE`), which fix the model shape and some kernel choices at
   compile time. `tools/baro` sets these from the GGUF.

Every setting the code reads is listed, with its type, default and the models it applies to, in
[`docs/settings-template.toml`](docs/settings-template.toml). `tools/settings-check.py` fails if
that file and the source disagree. The ones you are most likely to touch:

| setting | default | what it does |
|---|---|---|
| `BARO_TMAX` | 1088 in the engine, 32768 under `baro-serve` | context window the KV pool is sized for, in tokens |
| `BARO_KVQ` | `f32` | `int8` halves KV memory on dense models (agreement 99.1% mean on 20 prompts) |
| `BARO_SPEC` | 1 | speculative decode with the MTP head (dense; 1.104x as a 20-prompt median) |
| `BARO_MEGA` | 1 dense, 0 MoE | the persistent megakernel; `0` selects the launch path |
| `BARO_PREFILL` | 1 | batched prefill; `0` replays the prompt one row at a time |
| `BARO_TIER` | unset | MoE experts in host RAM, this many cached per layer in VRAM (needs a split pack) |
| `BARO_SEQS` | 1 | sequences in flight on one engine |
| `BARO_CACHE` | `~/.cache/baro` | where `tools/baro` keeps compiled engines and packs |

**Size the window against free VRAM.** The engine allocates its buffers from free VRAM, so the
desktop and other GPU programs count. Read used VRAM first (`rocm-smi --showmeminfo vram`), then
size `BARO_TMAX` so the pack plus KV fits in what is left; `BARO_KVQ=int8` halves the KV part.

## CPU tier: decode without a GPU

`cpu/engine_cpu.mojo` runs the dense qwen35 decode step on the CPU, op for op the same as the GPU
launch path. It speaks the same line protocol, so either front can serve it.

```sh
./.venv/bin/mojo build cpu/engine_cpu.mojo -I . -I kernels -I cpu -I serve -o engine_cpu
BARO_THREADS=8 ./engine_cpu PACKDIR < requests.jsonl
```

Status, honestly: it is **correct and slow**. Teacher-forced against the GPU build, 1280 of 1280
ids agree (20 prompts x 64 ids, `bench/check-cpu-engine.sh`, GPU side built with
`-D BARO_SHARED_EXP=1` and `BARO_MEGA=0`). Decode is 6 to 7 tok/s with 8 threads on a quiet
7800X3D and drops to 0.48 tok/s next to one busy 9-core job, because the worker pool spins
instead of parking. Prefill replays the decode step (59 tokens take 8.6 s), there is no K-quant
path yet, and only the dense qwen35 shape runs. It has run on one host only.

## Testing

| command | needs | what it checks |
|---|---|---|
| `bench/preflight.sh` | CPU | `tools/ci-checks.sh`, every test binary builds, both engines build. Writes a stamp other gates check |
| `./run-tests.sh` | GPU | kernel parity tests (each new kernel against the m=1 decode kernel, byte for byte) and the kernel census |
| `cargo test` in `serve/` | CPU | the Rust front against a stub engine |
| `serve/front/test_*.mojo` | CPU | the Mojo front against fake engines |
| `bench/check-cpu-engine.sh` | GPU + CPU | the CPU tier against the GPU engine, with a self-check and a mutant that must fail |
| `bench/front-real-check.sh` | GPU | both fronts on one real engine |

GPU jobs on the author's machine run under a local queue (`gpu-wait run`). On your machine,
run the inner command directly.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `pack/profile mismatch: profile H 4096 ... implies H 5120` | the engine was compiled for another model shape | use `tools/baro serve`, or build with the shape flags from `tools/model-id.py MODEL.gguf --json` |
| `no tokenizer.json loaded` (HTTP 503 on chat) | the pack has no `tokenizer.json` | pass `--tokenizer`, or rebuild the pack with `tools/baro serve` |
| decode is 10x slower than the tables, no error | weights spilled from VRAM to GTT (system RAM over PCIe) because something else holds VRAM | read `gtt_used` in `/sys/class/drm/card*/device/`; free VRAM or lower `BARO_TMAX` |
| out of memory with VRAM free | MAX's small-allocation pool is 2% of free VRAM | set `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT` higher |
| `/health` 503 for a long time after start | the pack is still loading; the port opens first | wait for 200; a large pack takes seconds from page cache and longer from disk |
| the engine prints `ready`, exits 0, served nothing | its stdin was empty | feed requests on stdin, or let a front start it |
| text differs between two runs at temperature 0 | near-tie tokens flip between decode paths and with a warm prefix cache | compare with teacher-forced agreement, not text equality (see below) |

## How numbers get into this repo

Every performance claim is preregistered: the question, the instrument, the
predicted range and the falsifier are committed **before** the run, and the
result is recorded against them whether or not it agreed
([`bench/PROTOCOL-RULES.md`](bench/PROTOCOL-RULES.md)).

An early baseline here recorded a 2x lead over hipBLASLt. It was measuring an
untuned vendor call: fixing three defects in our own shim took hipBLASLt from
2497 to 5201 GFLOP/s and erased the lead. **A vendor baseline that looks easy
to beat is a bug in your harness until proven otherwise.**

`bench/` is a lab notebook, not a product. Its scripts assume the author's
machine (llama.cpp at `$HOME/llama.cpp`, models under `$HOME/Models`, a local
GPU job queue). Read it for the method; expect to edit paths before it runs
elsewhere. Some protocols cite lane reports under `exchange/`; those live in
the private development history this repo was squashed from.

## Layout

| | |
|---|---|
| `kernels/` | Mojo GPU kernels and their parity tests ([`docs/KERNELS.md`](docs/KERNELS.md), generated) |
| `serve/` | the engines, tokenizer, prefix cache; `serve/src/` the Rust front and router, `serve/front/` the Mojo front |
| `cpu/` | the CPU tier: ops, weight unpacking, thread pool, `engine_cpu.mojo` |
| `profiles/` | named settings sets for `tools/baro serve --profile` |
| `face/`, `imkit/` | the window (`tools/baro gui`) and the vendored immediate-mode GUI under it (mojo-imkit v0.9.0) |
| `grammar/`, `minja/`, `uregex/`, `toml/` | JSON-schema grammar, chat-template renderer, regex engine, TOML reader (all Mojo) |
| `bench/` | benchmark harnesses and the frozen protocols |
| `shim/` | C++ hipBLASLt shim behind a C ABI, the vendor reference |
| `tools/` | GGUF packer, reference implementations, gates |
| `latentos/` | state handoff between engine processes (memfd and HIP IPC) |
| `docs/` | [`BASELINE.md`](docs/BASELINE.md) holds every verified number and known trap |

The tokenizer is Mojo, bit-equal to llama.cpp on the same GGUF
([`docs/TOKENIZER.md`](docs/TOKENIZER.md)).

## Portability

Everything is measured on one card, and RDNA3 specifics matter: wave size 32
(not 64 as on CDNA), 64 KB LDS per block, tile parameters swept for 96 compute
units and a 96 MB Infinity Cache. Other RDNA3 cards use the same ISA, so the
kernels should build, but whether the tuning holds is unmeasured.
[docs/amd-family.md](docs/amd-family.md) asks for exactly that.

**RDNA4 (gfx1200, gfx1201: RX 9060 XT, 9070, 9070 XT, Radeon AI PRO R9700): builds, never run on a card.** The
engine and every AMD intrinsic it uses compile for both targets (`tools/rdna4-probe/rdna4-probe.sh`, receipt in
`docs/rdna4/`), and `bench/preflight.sh` keeps it that way. RDNA4 halved the WMMA input fragments and moved the
accumulator rows; the fragments are handled, the accumulator rows of the prefill kernels are not ported yet, so
prefill output on RDNA4 is expected to be wrong until that lands. The megakernel turns itself off on cards with
fewer compute units than its 96-block grid. If you own one of these cards, `bench/report.sh --rdna4` (no weights
needed) writes one local text file: card, intrinsic table, a WMMA numeric check, kernel log. Paste it into a
"hardware report" issue; a failing report is the most useful one.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) lists where help is wanted, the check that
decides each item, and the template every change fills in.

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party material
is named in the NOTICE: the vendored `toml/` reader (DataBooth, Apache-2.0) and
the pre-tokenizer regexes in `serve/pretok-table.json` (from llama.cpp, MIT).
llama.cpp is fetched at a pinned commit, never redistributed.
