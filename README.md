# mojo-baro

An LLM inference engine for AMD RDNA3 GPUs, with every GPU kernel written in
[Mojo](https://www.modular.com/mojo). It serves chat over an OpenAI-compatible
HTTP API on one RX 7900 XTX.

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

## Building

Requires ROCm 7.2, CMake, Rust (for the server) and
[`uv`](https://docs.astral.sh/uv/). Nothing is installed machine-wide: `uv sync`
creates a repo-local `.venv` pinning `max[all]==26.5.0` (Mojo 1.0.0).

```sh
uv sync              # repo-local .venv with the pinned Mojo/MAX toolchain
bench/preflight.sh   # CPU only: CI checks, every test and both engines build
./run-tests.sh       # needs the card: parity tests and the kernel census
```

Serve any supported GGUF with one command:

```sh
tools/baro serve MODEL.gguf [--port 8080] [--chat-template-file PATH] [--rebuild]
```

It reads the architecture and dimensions from the GGUF header, builds the
engine and weight pack on a cache miss (`~/.cache/baro/`, override with
`$BARO_CACHE`), and reuses them for every checkpoint of the same shape. An
unsupported architecture or tokenizer is refused before any build starts.

## The server

`baro-serve` (Rust, `serve/src/`) keeps one engine process alive and speaks
the OpenAI API: `/v1/chat/completions` and `/v1/completions` with SSE
streaming, `/v1/models`, `/v1/cancel`, `/v1/fork`, `/tokenize`, `/detokenize`,
`/health`, plus an Ollama-compatible API.

- Tool calls come back in the OpenAI `tool_calls` shape; `tool_choice` takes
  `auto`, `none`, `required` or a named function. `response_format` with a JSON
  schema is enforced by a device token mask.
- Each message boundary is checkpointed, so a follow-up turn prefills only the
  new message.
- One request decodes at a time, except on a dense K-quant pack with
  `--parallel N`: N sessions share one weight copy and the shared prompt prefix,
  and decode together in batched steps (past the KV pool: HTTP 429).
  [`docs/design/multi-session-serve.md`](docs/design/multi-session-serve.md).

`tools/baro chat` is a terminal client. The contract between server and engine
is [`serve/PROTOCOL.md`](serve/PROTOCOL.md).

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
| `serve/` | the engines, tokenizer, prefix cache and the Rust server |
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

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) lists where help is wanted, the check that
decides each item, and the template every change fills in.

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party material
is named in the NOTICE: the vendored `toml/` reader (DataBooth, Apache-2.0) and
the pre-tokenizer regexes in `serve/pretok-table.json` (from llama.cpp, MIT).
llama.cpp is fetched at a pinned commit, never redistributed.
