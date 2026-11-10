# Engine <-> server protocol

`serve/engine.mojo` (qwen35, qwen35moe) and `serve/spark.mojo` (the dense
families: llama, qwen2, granite, spark2_5) are both single-request decoders
that speak this same line protocol under `BARO_SERVE=1`, sharing the
byte-scanner reader and request parser in `serve/serve_proto.mojo` rather
than each defining their own wire format. `serve/src` (`baro-serve`, Rust)
keeps one engine process alive (either binary, selected with `--engine`) and
turns HTTP requests into addressed engine lines. Device execution remains
one request at a time in this step. This file is the
contract between the two, and the seam where the process boundary is later
replaced by a C ABI. Not every optional field is acted on by every engine:
spark has no draft head (`spec` is parsed and ignored, `spec_k` in its
`ready` line is always `0`) and no prefix-checkpoint chain (`ckpt` is parsed
and ignored); `stop` and the required id/prompt/n fields are honored by
both.

## Process model

```
client --HTTP--> baro-serve --stdin JSON lines--> engine (BARO_SERVE=1)
                            <--stdout JSON lines--
```

- The server spawns `.work/engine` (`--engine`) with `BARO_SERVE=1` and
  `BARO_PACK=<pack>` (`--pack`), inheriting its own environment, so every
  other `BARO_*` knob (`BARO_SPEC`, `BARO_SPEC_K`, `BARO_MEGA`,
  `BARO_DRAFT_Q4`, ...) still reaches the engine.
- The engine loads the pack and allocates every buffer **once**, prints the
  ready line, then blocks on stdin.
- Device requests are strictly serial: the engine advances one request at a
  time. The server worker may write later addressed request lines before the
  current request's terminal line; HTTP requests wait in a bounded queue (64)
  and get `503` beyond it.
- Stdin EOF ends the engine: the request loop exits and the process
  returns 0. That is the clean shutdown; the server closes stdin on
  SIGINT and kills the child only if it has not exited after 30 s.
- `BARO_SERVE=0` (default) is the unchanged one-shot path: prompt from
  `BARO_PROMPT`/`<pack>/prompt-tokens.txt`, `GEN_N` tokens, `GENERATED:`
  line, draft receipt, exit. Every existing gate runs that path.

## A3(a) wire-admission delta

The A3(a) wire permits the server to write a second request line before the
first request has printed its terminal `done` or `error` line. The engine
still advances one request at a time in this item; this is admission and
addressing, not device batching. The engine keeps each accepted request's id,
and every `tok`, `done`, or `error` line carries that id so the server can
route output to the waiting HTTP response without dropping lines for another
request.

The old contract's strict-serial statement above therefore changes only at
the stdin/stdout boundary: HTTP work may be admitted ahead of completion,
while device execution remains serial until A3(c). A malformed line remains a
log/error and must not be mistaken for a request. The A3(a) gate records the
second request's admission before the first terminal line, then checks both
id streams to completion.

## Lines the engine prints (stdout)

One JSON object per line, no embedded newlines. Any line that does not
start with `{`, is not a JSON object, or fails the field checks below is a
**log line** and is forwarded to the server's stderr — never acted on.
The engine keeps printing its usual diagnostics (`pack loaded in`,
`tok/s_gen:`, `GENERATED:`, ...) between the JSON lines.

| line | when | fields |
|---|---|---|
| `{"ready":true,"tmax":128,"mrows":8,"kmax":8,"spec_k":2,"pack":"..."}` | once, after init | `tmax` = KV/token capacity: `len(prompt)+n <= tmax` or the request is rejected. `mrows` = prefill chunk, `kmax` = max draft width, `spec_k` = the draft width in effect (`spec-k.txt` / `BARO_SPEC_K`). |
| `{"id":ID,"tok":T}` | per generated token, in order | `T` is the token id chosen at the next position (greedy at `temperature` 0, sampled otherwise). In spec mode a verified window emits its accepted tokens plus the correction token together. |
| `{"id":ID,"done":true,"n":N,"prefill_s":..,"decode_s":..,"tok_s":..,"finish":"length"\|"stop"\|"cancelled"}` | after the last token | `n` = tokens actually generated (`<=` request `n`: less than requested when `finish` is `"stop"` or `"cancelled"`), `tok_s` = `(n-1)/decode_s` (0 when n<=1), matching the one-shot `tok/s_gen`. With spec: `"drafted"`, `"accepted"`, `"k"`. |
| `{"id":ID,"embed":[F,...]}` | once, only when the request carries `"embed":true`; after the first token line, before `done` | `H` floats: the post-final-norm hidden state of the LAST PROMPT TOKEN, L2-normalized on the engine side (P0a-e). The request runs with the head unfolded (as a sampled request does), because the folded head never writes that row; tokens are unchanged. Send `n:1`, `spec:false`. The spark engine and the multi-sequence path (`BARO_SEQS > 1`) answer an error line instead. Gate: `bench/p0ae-embed-gate.sh` (cosine against llama.cpp `--pooling last`). |
| `{"id":ID,"hidden":[F,...]}` | once after each generated token when the request carries `"hidden":true` | `H` raw f32 post-final-norm values for the hidden row that produced the preceding token. Dense/MoE engine only. The request uses the non-spec launch path. |
| `{"id":ID,"logits_topk":[{"id":T,"logit":F},...]}` | once after each generated token when the request carries `"logits_topk":K` | The top `K` raw pre-penalty logits, `1 <= K <= 20`, ordered by descending value. Dense/MoE engine only. |
| `{"id":ID,"error":"..."}` | instead of tokens | request rejected before any GPU work: bad JSON, missing/negative fields, empty prompt, `n < 1`, `len(prompt)+n > tmax`. The loop continues. |

`id` is echoed from the request. The engine parses `id` first, so an error
for an unparseable line carries `"id":0`.

### State words (2026-11-10)

Every terminal state has a word on the wire, so a client reads a field and never infers a state from
a hang, a log line or a number. All fields are additive: a reader that ignores them sees the old line.

| Line | Field | Values | Meaning |
|---|---|---|---|
| ready | `"tier"` | `off`, `page-cache`, `hot`, `pinned`, `pinned+zc`, `pinned+cpu`, each with an optional `+mmap` | expert-tier mode of this engine |
| ready | `"seqs"` | N | sequences this engine decodes at once |
| done | `"fail"` | N | megakernel fail word; non-zero = the tokens of this request are invalid (the front answers `502 engine_invalid_tokens`) |
| done | `"finish"` | adds `"error"` | the request ended on an engine error (KV pool exhausted, lens base missing); an `error` line with the same `id` came first |
| done | `"path"` | `mega`, `mega_win`, `spec`, `launch`, `pair`, `seqs` | decode path that ran |
| done | `"grammar"` | `none`, `wait`, `match`, `done` | grammar matcher phase at the end; with a grammar also `"gdraws"`, `"gaccepted"`, `"grejected"` |
| done | `"captured"` | N | positions written under `BARO_CAPTURE` |
| done | `"load_tier"` | `hot`, `file`, `ipc` | where the restored state came from |
| error | text | named reasons | unknown `latent_id`, fork on an empty chain, a prompt past the window after a prefix extension and a lens request without its base answer with an error line; the engine stays alive |

`grejected > 0` means a masked draw produced a token the grammar refused: a host/device desync, reported
by the front as the warning `grammar_rejected_tokens`. An idle or stale `{"cancel":ID}` is dropped with
the log line `cancel: id N is not decoding, dropped`; one parser serves every engine.

Not exercised by a run on the card: a live non-zero `fail` (no knob forces the barrier timeout), KV pool
exhaustion, `captured`, and `grejected` on a served grammar. Their fields are tested on CPU with injected values.

HTTP side (`baro-serve`): `/health` carries `engine_state` (`starting`, `idle`, `busy`, `dead`,
`stopping`) and per engine `{state, depth, reason}`; the port answers `503` with phase `starting` or
`warming` while the pack loads, so `curl -sf /health` succeeds only on a ready server. A dead engine in a
pool is skipped. A client that drops its stream cancels its request. A stream that ends with no terminal
event is `502` (SSE: an error chunk before `[DONE]`). Turn, fork and chat responses carry `state_source`
(`none`, `explicit`, `created`, `auto`) and `tier_source`. The router's `/health` says `ok`, `degraded`,
`starting` or `down`.

## Lines the server writes (stdin)

```
{"id":ID,"prompt":[INT,...],"n":N,"spec":BOOL,"stop":[[INT,...],...],"ckpt":[INT,...]}\n
{"cancel":ID}\n
```

- `id`: unsigned integer, assigned by the server (monotonic, starts at 1).
- `prompt`: non-empty list of non-negative token ids, in order.
- `n`: tokens to generate, `>= 1` (an upper bound: `stop` or `cancel` can end
  generation sooner, see below).
- `spec`: optional; `true`/`false` selects MTP speculative decode for this
  request. Absent => the engine's `BARO_SPEC` default.
- `stop`: optional, default `[]`. A list of token-id sequences; once the
  tail of the tokens generated so far equals any of them, the engine stops
  (that sequence's tokens are still emitted as `tok` lines and counted in
  `n`) and reports `"finish":"stop"`. Checked once per decode window (not
  mid-window), and only synced back from the GPU when `stop` is non-empty --
  a request with no `stop` pays nothing for this. `serve/src/main.rs` builds
  `stop` from the tokenizer's own EOS-like ids (each a length-1 sequence)
  plus any caller `stop` string, tokenized.
- `ckpt`: optional, default `[]` (M1b role-boundary checkpoints). Token
  positions where `serve/prefix.mojo` should also take a prefix checkpoint,
  in addition to the periodic 1024-token grid and the prompt-end point.
  `serve/src/main.rs` sends, for a chat request, the token length after
  each message when the conversation so far is rendered with no generation
  prompt (`Text::role_boundaries`) -- a later turn's full prompt starts
  with exactly those same bytes, since history does not change, so these
  positions are restore points for it. Index 0 (by convention the system
  prompt) is saved pinned against eviction; every hint is saved as a
  role-boundary checkpoint, evicted only after every periodic-grid one. A
  hint that misses a real tokenization boundary wastes a checkpoint slot,
  never corrupts a restore -- `lookup` still requires the stored SHA-256
  hash (salted per pack, so two packs never share a checkpoint) to match.
- `temperature`/`top_p`/`top_k`/`min_p`/`seed`/`presence_penalty`/
  `frequency_penalty`: optional (C3 control block). Parsed into
  `SampleParams`. `temperature > 0` makes `serve/engine.mojo` draw with
  the device sampler (`kernels/sample.mojo`, `amar_sample_row`) inside the
  decode loop (M5 `8ddd477`), composed with speculation by the accept rule
  in `amar_spec_accept` (A1 `b3c0d90`, `bench/spec-sample-protocol.md`);
  `top_p`, `top_k`, `min_p` and `seed` apply there. The two penalties are
  parsed and carried but not acted on. `serve/sample_ref.mojo` is the host
  reference, matched against the device kernels. `serve/spark.mojo` parses
  the block and still decodes greedy. Absent, the request line is identical
  to before this field existed.
- `tool_re` / `tool_re0` (strings, regex in `grammar/regex.mojo` syntax):
  optional, written by `baro-serve` when the request has `tools`, no
  `response_format`, and a template wire it parses (`<function=` XML or JSON
  in `<tool_call>`). `serve/engine.mojo` (single and `BARO_SEQS>2` paths)
  scans the output for `<tool_call>` and then draws the text through the end
  of the tool name masked by `tool_re` (`\s*<function=(hold|turn)>`), so a
  call can only name a tool of the request; `tool_re0` (the name alone) is
  armed at the first draw for `tool_choice` "required". The matcher lets go
  once the name ends; arguments and plain text stay free. Tool requests run
  with spec and the megakernel off on the single path; on the seqs path the
  slot takes its own m = 1 step for the name tokens only. `serve/spark.mojo`
  ignores both fields.
- `schema` (a JSON Schema object) and `reasoning` (bool, default true):
  optional, written by `baro-serve` from `response_format` and
  `chat_template_kwargs.enable_thinking`. `serve/engine.mojo` compiles the
  schema into a grammar matcher (`serve/grammar_rt.mojo`, `grammar/`) and
  draws every generated token with `amar_sample_row_masked`, at any
  temperature (masked argmax at `temperature <= 0`). With `reasoning` true
  the mask starts after `</think>` appears in the output. The request runs
  with spec, the megakernel and truncation (`top_p`/`top_k`/`min_p`) off,
  stops with `"finish":"stop"` when the document is complete, and the
  engine logs `grammar masked draws: N accepted: N` per request
  (`bench/grammar-protocol.md`). Spark uses the same grammar runtime and masked sampler.
- `hidden`: optional bool for the HTTP completion routes. When true, the wire
  emits a raw hidden row after each generated token. `logits_topk`: optional
  integer from 1 to 20 for the same routes. The wire emits raw pre-penalty
  top-k logits after each generated token. These options force non-spec decode
  and are refused by `serve/spark.mojo`.
- `{"cancel":ID}`: a second line shape, written to the same stdin at any
  point while `ID` is decoding (the request line for the *next* id is never
  written before this one's `done` line, so a stray line mid-request can
  only be this). The engine polls for it once per decode window (`poll(0,
  POLLIN, timeout=0)`, never blocking, never touching the GPU queue); a
  match breaks the loop and the `done` line reports `"finish":"cancelled"`
  with `n` = tokens actually generated. A cancel for a request that is not
  the one currently decoding (already finished, or still queued on the
  server side) is silently dropped -- cancelling a queued request is not
  supported.

The engine's parser is a byte scanner, not a JSON library: keys must be
double-quoted, values must be plain integers / `true` / `false` / arrays of
the shapes above, and no other structure is interpreted. Extra keys are
ignored.

## Per-request state

Before each request the engine zeroes the conv/SSM slot rings, both KV
caches (trunk and draft) and the megakernel counters, and resets the
position to 0, so the request runs exactly as the one-shot path would.
Verified: three back-to-back requests (no-spec, spec, no-spec) on the q4
pack each reproduce `ref-tokens-64.txt` (`tools/test_server.sh`,
`.work/lane/gate1.log`).

## Multi-session engine (`BARO_SEQS=N`, N > 2, dense packs)

`baro-serve --parallel N [--kv-pool P]` starts the engine with `BARO_SEQS=N`, `BARO_KV_SHARE=1`
and `BARO_KV_POOL=P`; the engine must be built `-D BARO_SEQ_CAP>=N` (`-D BARO_SLOTS=2` keeps the
SSM ring at two entries per slot: 32 slots hold 3.2 GB of recurrent state instead of 14.5 GB). The
wire is unchanged: every request line is admitted at once, `tok`/`done`/`error` lines carry their
id, and lines of different ids interleave. Up to N requests hold a slot each; more wait in the
engine's pending list. Per loop the engine admits into free slots, runs every pending prefill
(`BARO_SEQS_PF=all`), then one decode step for every decoding slot in groups of up to 8 rows
(`serve/rows.mojo`). Each request restores the longest prefix checkpoint (its KV pages shared
read-only, the partial page copied, the SSM state copied into its slot), takes further pages one at
a time as it decodes, and frees them at `done`.

- Admission takes pages for the prompt plus `BARO_SEQS_RESERVE` (512) decode tokens. When they are not
  free while other requests run, the line waits in the engine's queue until pages free up (no
  checkpoint is evicted for it); with nothing else running the engine evicts checkpoints and, if that
  is not enough, answers `{"id":ID,"error":"kv pool full: ..."}` (`429`). `baro-serve --max-inflight Q`
  (64) is the hard bound: past Q requests waiting or running, a new one gets `429` before it reaches the
  engine. A slot that outgrows its reserve and cannot get its next page ends with
  `kv pool exhausted at pos N` (`502`).
- Served per slot: prompt, `n`, `stop`, `ckpt`, `temperature`/`top_p`/`top_k`/`min_p`/`seed`
  (device sampler per row, same seed and counter as the single-sequence path). Refused with an error
  line: `schema`, `embeds`, `embed`, `hidden`, `logits_topk`, `top_logprobs`, penalties, state
  save/load, control lines. No speculation, no megakernel.
- Identity: a request's ids equal the single-sequence engine's ids for the same prompt **and the same
  restore point** (`timings.cached`). A deeper restore (an identical earlier prompt left a prompt-end
  checkpoint) replays fewer rows through the kq WMMA prefill and can flip a near-tie, on either engine.
  Gate: `bench/ailab/` (`baseline.sh`, `parity.sh` keyed on state and restore point, `burst.sh`).

Streaming costs one host copy + synchronize per generated token on the
non-spec path (spec mode reuses the accept-window copies it already
makes): about 0.5% of tok/s_gen on the 5-token receipt prompt.

## HTTP surface (`baro-serve`)

| route | body | notes |
|---|---|---|
| `GET /health` | | `status`, `queue` (waiting+running), `tokenizer` (bool), `limits` (the ready line), `pack` |
| `GET /v1/models` | | one model, id = pack directory name |
| `POST /v1/completions` | `prompt` (string, or array of token ids), `max_tokens` (omitted: up to the window, TMAX minus prompt), `stream`, `spec` (extension), `stop` (string or array of strings), `temperature`/`top_p`/`top_k`/`min_p`/`seed`/`presence_penalty`/`frequency_penalty`/`logprobs`, plus `hidden` and `logits_topk` | response adds `choices[0].tokens`, `timings`, and requested top-level `hidden` or `logits_topk` arrays. In SSE, each latent row is a data chunk after its token chunk. |
| `POST /v1/chat/completions` | `messages`, `max_tokens`/`max_completion_tokens`, `stream`, `spec`, `stop`, the same C3 sampler and latent fields, `tools`, `chat_template_kwargs`, `response_format` | needs the tokenizer; template from `tokenizer-meta.json` (`chat_template`, Jinja via minijinja + pycompat) else ChatML. `tools` and `chat_template_kwargs` are passed to the template (A5 `0950c6f`); non-streaming output returns parsed `choices[0].message.tool_calls`, while SSE emits OpenAI-shaped function-name and argument fragments as the model's `<tool_call>` wire arrives. `response_format` `{"type":"json_schema","json_schema":{"schema":...}}` is enforced on dense/MoE and Spark engines. Latent rows follow the same SSE and non-streaming shapes. `serve/spark.mojo` engines refuse latent fields with 400. |
| `POST /v1/fork` | `prompt` (as `/v1/completions`), `branches`: array of `{max_tokens, spec, stop, sampler fields}` | runs the branches in order on one engine; branch 0 prefills and checkpoints the prompt, later branches restore it (B5 `a400c67`, `bench/fork-protocol.md`). Response `{"object":"fork","model","prompt_tokens","branches":[{index,id,text,tokens,finish_reason,usage,timings}]}`. |
| `POST /v1/cancel` | `{"id": "cmpl-7"}` / `{"id": "chatcmpl-7"}` (the response `id`, or the SSE `id` field of its first chunk -- read while the request is still streaming) | `{"cancelled": bool}`; `true` only if that request was the one actively decoding. A queued-but-not-started or already-finished id returns `false`. |
| `POST /tokenize` | `{"content": "...", "add_special": false}` | `{"tokens": [...]}` |
| `POST /detokenize` | `{"tokens": [...]}` | `{"content": "..."}` |

### Ollama compatibility surface (P0a)

The Ollama routes share the existing `baro-serve` listener and model selected by
`--pack`; there is no second Ollama port.  The compatibility surface is:

| route | request and response contract |
|---|---|
| `GET /api/tags` | Returns one loaded model in Ollama's `models` array. |
| `GET /api/ps` | Returns the loaded model while its engine is alive, otherwise an empty `models` array. |
| `GET /api/version` | Returns the baro compatibility version. |
| `POST /api/show` | Accepts `model` or `name`; returns model metadata for the loaded pack. |
| `POST /api/pull` | Confirms the already loaded pack using Ollama NDJSON status frames. `stream:false` returns one success object. A different model returns `404`; importing or hot-swapping packs remains outside the server. |
| `POST /api/chat` | Accepts Ollama `messages`, `options`, and `stream`. The default stream is `true`; streamed responses are newline-delimited JSON. `stream: false` returns one final JSON object. |
| `POST /api/generate` | Accepts `prompt`, optional `system`, `raw`, `options`, and `stream`. The default stream is `true`; streamed responses are newline-delimited JSON. `stream: false` returns one final JSON object. |
| `POST /api/embeddings` | Accepts a string `prompt` or `input` and returns one Ollama-shaped embedding vector from the P0a-e wire. Arrays of strings are also accepted. Spark and `BARO_SEQS > 1` refusals return `501` with `embeddings_pending`. |
| `POST /v1/embeddings` | Accepts a string or array of strings in `input` and returns an OpenAI-shaped list of L2-normalized P0a-e vectors. `encoding_format`: `float` (default) or `base64` (little-endian f32 bytes). |

The supported Ollama `options` mapping is `num_predict` to maximum generated
tokens, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, and `stop` to the
existing sampler and stop fields. `repeat_penalty` and `num_ctx` are accepted
for client compatibility but are not applied. `images`, tools, and format are
accepted where present but are not implemented by this item. A client must not
infer that an accepted option took effect without a read-back contract.

Ollama stream chunks use `content-type: application/x-ndjson`, one JSON object
per line, and finish with `done: true`, counts, durations, and the generated
context. OpenAI routes continue to use SSE and are otherwise unchanged.
The HTTP embedding wrapper is gated by `bench/p0a-contract-gate.sh` gate 4.
The engine-vector parity gate remains `bench/p0ae-embed-gate.sh`: it compares
the last-prompt-token vector with llama.cpp `--pooling last`, checks the 4096
float dimension, normalization, determinism, identity with and without the
flag, and malformed-flag refusal.

`stream: true` returns SSE: one `data:` chunk per token (`text` delta and
`tokens: [id]`), a final chunk with `finish_reason` + `usage` + `timings`
(+ the full `tokens` list on completions), then `data: [DONE]`.

Errors are `{"error": {"message", "type", "code"}}` with the HTTP status:
`400` bad request (empty prompt, over-length, bad JSON), `503` no
tokenizer / queue full / engine gone, `502` engine error mid-request.

## P3a speech input

`POST /v1/audio/transcriptions` accepts an OpenAI-compatible multipart form. The required
`file` part is an audio file; `model`, `language`, and `response_format` are optional fields.
The server accepts `response_format=text` and returns the transcript as plain text, or
`response_format=json` and returns `{"text":"..."}`. The default model is the coordinator's
large Whisper model, and the default language is `en`.

The first request starts one `whisper-server` sidecar on loopback with the selected model,
English language, beam size 5, 8 threads, and GPU enabled when the gate admits it. The server
proxies the file to the sidecar's `/inference` endpoint, waits for the JSON result, and forwards
only the requested response shape. The sidecar stops after the idle timeout and is restarted for
the next request. It never starts an LLM engine for an audio-only gate.

The sidecar's effective model path, language, beam size, thread count, device, and listening
address are logged before the first transcription and copied into the gate receipt. A malformed
multipart request is `400`; a sidecar start or request failure is `502`; an unavailable model or
sidecar capacity failure is `503`. The fixed P3a fixture gate compares the normalized `text`
field or plain response byte for byte against the same `whisper-cli` model and settings.

## P3b speech output

`POST /v1/audio/speech` accepts JSON `input`, optional `model` and `voice`, `response_format` set
to `wav`, and `speed` set to `1`. It runs the configured TTS command with arguments
`TEXT VOICE OUTPUT_WAV`, defaulting to the local Piper runner, and returns `audio/wav`.

### Tokenizer files (owned by the tokenizer lane)

`<pack>/tokenizer.json` (HF `tokenizers` format; loaded with the Rust
`tokenizers` crate, oniguruma regex backend) and, optionally,
`<pack>/tokenizer-meta.json` with `chat_template` (string),
`bos_token_id`/`eos_token_id` (or `bos_id`/`eos_id`), `bos_token`/
`eos_token` (strings), `add_bos` (bool). Stop ids = `eos_token_id` plus
`<|im_end|>` / `<|endoftext|>` when the vocabulary has them. Without
`tokenizer.json` the server starts anyway and the text endpoints return
`503`; `--tokenizer PATH` points at a file elsewhere. `--chat-template-file
PATH` overrides the pack's `tokenizer-meta.json` template for this process.
The file is UTF-8 raw Jinja. It is validated at startup and receives the same
render context as an embedded template. Without the flag, the embedded
template and then the existing ChatML fallback are unchanged.

## The C-ABI seam

Everything above the engine is written against three operations, and the
process boundary is the only thing the line protocol adds:

| stdin/stdout today | C ABI later |
|---|---|
| spawn + `ready` line | `baro_engine_t* baro_open(const char* pack)` returning the same limits struct |
| request line | `int baro_generate(baro_engine_t*, const uint32_t* prompt, size_t n_prompt, uint32_t n, bool spec, baro_token_cb cb, void* user)` — `cb(user, token)` per token, return = done stats or a negative error code |
| stdin EOF | `void baro_close(baro_engine_t*)` |

`serve/src/engine.rs` is the only file that knows about the child
process; `Engine::submit` / `Event` are the interface the handlers use and
would be re-implemented over the ABI (one worker task still serialises
calls, since the engine stays single-request). `serve/src/protocol.rs`
(the line parser and its tests) is the part that disappears. Rust must
bind Mojo's exported C symbols only — never the C++ hipBLASLt shim
(`docs/BASELINE.md`, "Layers").

## Verifying

```
tools/test_server.sh [OUTDIR]      # builds, then: clippy, cargo test, /health,
                                   # completion == ref-tokens-64, SSE == ref,
                                   # 2 queued == ref, 400 on over-length, SIGINT shutdown
```

Runs under `gpu-wait` by itself. Receipts land in `OUTDIR/SUMMARY.txt`
(default `.work/server-test/`).
