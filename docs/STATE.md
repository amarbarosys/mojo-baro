# The state layer: checkpoints, forks, turns, sharing

What a user meets. Every number here has a receipt in AMDHQ `runs/latent-os/` (dates 2026-09-24
unless noted); the design is `docs/design/latent-os/` there. Gates: `bench/state-gates.sh`,
`bench/state-timing.sh`, `bench/turn-api.sh`, `bench/replay-m.sh`.

## What a checkpoint is

The engine's state after a prefix: the KV pages of the attention layers, the conv and SSM state
of the recurrent layers, and the token list, at position `len(prompt) - 1` (the last prompt token
is the one whose forward produces the first output). Saved as one file:

| magic | KV payload | written by | 32k size |
|---|---|---|---|
| `BAROST01` | f32 | `BARO_STATE_INT8=0` | 2,150 MB |
| `BAROST02` | int8 per KVHSTR block + f32 scale | default (f32-KV engine) | 577 MB |
| `BAROST03` | the int8 pool's own pages | `-D BARO_KVQ=int8` engine | 585 MB |

A fork from a checkpoint produces the same greedy ids as recomputing the prompt (28-check gate
`checkpoint-api.sh` on every commit; C2-long at 8k and 32k; the int8-pool engine's ids equal the
f32 engine's in every state-timing cell). The engine refuses a file from another pack (salt), a
different slot layout, or another KV format.

## Start a server

```
BARO_PACK=.work/e11b-qwythos-pack BARO_CKPT_DIR=.work/ckpts BARO_TMAX=32768 \
  serve/target/release/baro-serve --engine .work/engine --pack .work/e11b-qwythos-pack --port 8080 \
  [--tokenizer tokenizer.json --chat-template-file chat_template.jinja]   # for text endpoints
```

Knobs (all in `docs/settings-template.toml`, group `state`, each read back in the engine log):
`BARO_CKPT` slots, `BARO_CKPT_DIR`, `BARO_CKPT_API` live cap (507 past it), `BARO_CKPT_DISK_BYTES`
(507 past it, no file left), `BARO_STATE_INT8`, `BARO_REPLAY_M` (rows per replay window, 4).

## Create, fork, turn

```
POST /v1/checkpoints           {"prompt": [ids] | "text", "ttl_s": 600}
  -> {"created": true, "checkpoint": {"id": "lat:state@pos7999:..", "pos": 7999, "bytes": .., "identity": {..}}}
POST /v1/checkpoints/ID/fork   {"branches": [{"prompt_suffix": "text" | [ids], "max_tokens": 32, "temperature": 0}]}
  -> branches[i]: tokens, text, restored, cached_tokens (== pos), tier ("hot" | "file"), timings
POST /v1/turn                  {"checkpoint": ID, "suffix": "text" | [ids], "max_tokens": 32}
  -> text, tokens, ttft_s (front-measured, request in -> first generated token), tier, timings
GET  /v1/checkpoints           GET /v1/checkpoints/ID       DELETE /v1/checkpoints/ID
```

`hot`: the engine still holds the prefix (nothing replayed from a lower position since), the file
is not read. `file`: the file is read through a pinned stage (int8 32k 0.20 s on the f32-KV
engine, 0.061 s on the int8 engine). The fork request line carries only the suffix; the engine
reads the prefix's token list from the file header.

Measured, 8k prefix, greedy, Qwythos Q4_0 on the 7900 XTX:

| case | time to first token |
|---|---|
| fresh 15-token suffix, prefix hot | 0.072 s (restore 3 ms + replay 68 + front 1) |
| the same suffix again (a repeated template turn) | 0.027 s |
| fresh suffix, prefix from file, int8 engine | 0.107 s |
| 32k prefix, fresh suffix, hot | 0.085 s |

The replay of the suffix rows is the cost: 4 rows per window, ~17 ms per window. A suffix of
at most 8 tokens is two windows.

## Chat route

`POST /v1/chat/completions` with `"checkpoint": true` (non-streaming) creates the rendered
prompt's checkpoint first and returns its id as `checkpoint`; with `"checkpoint_id": ID` the
rendered prompt must start with the checkpoint's tokens (409 otherwise) and only the rest is
sent. Ids are the same as a plain chat call (`turn-api.sh` chat section).

## Share across processes and hosts

Two servers on one host sharing `BARO_CKPT_DIR` see each other's checkpoints (S1,
`ckpt-share.sh`; the sidecar JSON next to the file carries id, pos, tokens, identity, TTL).

Across hosts: `GET /v1/checkpoints/ID/file` returns the raw file with `X-Baro-Sha256` and the
sidecar without the token ids in `X-Baro-Meta` (at 32k the ids are 142 KB, over curl's header cap) and with them at `GET /v1/checkpoints/ID/meta`; `POST /v1/checkpoints/import` (multipart parts `meta`, `file`,
optional `sha256`) registers it on the other server, refusing a different pack or weights
(409, field named). Shipping the state beats recomputing the prompt when the link is fast enough:
int8 32k wins on 1 GbE (7.2 s vs 16.4 s, B4-long); the router that picks per request (`handoff.rs`)
is not wired to a route yet (plan 12 P4). A state file is the prompt in another encoding: ship it
over an encrypted or private link only (LCGuard, 07).

Two engines on one GPU, without a file (T3b, AMDHQ `runs/latent-os/T3b-2026-10-12.md`): start both
servers with `BARO_SHIM_LIB` pointing at `libamarbaro_shim.so` (`cmake -S shim -B .work/shim-build`),
f32 KV, and engines built with `-D BARO_SEQ_CAP=1` so two fit (8.8 GB each at `BARO_TMAX=9216`).

```
POST A /v1/ipc/fork   {"checkpoint": ID, "target": "127.0.0.1:PORT_B", "suffix": [ids] | "text", "max_tokens": 32}
```

A's engine copies the checkpoint into a reusable device stage and hands B a 320-byte message (the
256-byte LatentHeader plus a HIP IPC handle); B refuses a different pack or prefix with 409 before
touching the handle, otherwise copies the state GPU to GPU and answers as a fork. The answer carries
`ttft_s` (A's request in to B's first token), `tier: "ipc"` and `ipc.{bytes, export_s}`. 8k
checkpoint: 581 MB moved, B ingest 8 ms, median TTFT 88 ms against 121 ms for B forking the same
checkpoint from the shared file. Instruments: `POST /v1/ipc/sha {pos, prefix_hash}`, `GET
/v1/ipc/stats` (opens, live and pooled stages). Stages are pooled, never freed: on ROCm an
allocation exported for IPC is not reclaimed by `hipFree`. Same host and GPU only; the LatentOS agent
as the relay is not built (link 3).

One weight copy for the pair: start A with `BARO_WEIGHTS_EXPORT=FILE` (it loads pack.bin into an
exportable allocation and writes a 112-byte record: size, pack salt, IPC handle) and B with
`BARO_WEIGHTS_IMPORT=FILE` (it checks size and salt against its own pack, refusing before any HIP call,
then maps A's buffer in 4 ms and never reads pack.bin). A must outlive B. The 9B pair holds 11.68 GB
above the desktop instead of 18.32; ids equal the separate-weights run and VRAM returns to base when
both exit (T3b step 2a, `T3B_WEIGHTS=shared bench/t3b-gate.sh`).

`BARO_EMBD_HOST=1` keeps `token_embd` (2.03 GB of bf16 on the 9B, one row read per token) in mapped
pinned host memory that the embed kernels read over PCIe; VRAM holds the other 4.60 GB of the pack.
Decode 146.46 vs 146.89 tok/s (0.997), ids equal. With shared weights the pair holds 9.77 GB above
the desktop (T3b step 2b, `T3B_EMBD=host`); both engines of a pair must agree on the knob, the weight
record's size check refuses otherwise. Not for `realign`: its full-table gather reads all 2.03 GB over
PCIe per call.

Four engines on one weight copy fit in 13.31 GB above the desktop (`bench/t3b-quad.sh`) and decode
correctly together **only with `BARO_MEGA=0`** (114 tok/s aggregate, 29 to 34 per engine). With the
megakernel on, engines decoding at the same time trip its bounded grid barrier (`mega fail word: 1
NOT-RESIDENT` in the log) and the request still returns those invalid tokens with HTTP 200: run
co-resident engines with `BARO_MEGA=0`.

## LatentOS reference contract

The canonical package now exposes `LatentOSClient` operations `describe`, `create`, `fork`,
`resume`, `release`, `export`, and `import`. `execution_checkpoint` is the only representation
accepted by `resume`; `semantic_latent` is accepted only by the separate semantic-consume path
after its directional binding manifest is validated. The immutable parent and mutable branch test,
weight-owner count, byte baseline, expiry, quota, identity, and v1 semantic-kind refusal run with:

```
latentos/tests/standalone.sh --backend reference
```

The gate copies only `latentos/` to a temporary directory, builds the reference client, router,
golden binding fixture, AEAD chunk/relay fixture, and legacy IPC checks, then records the linked
dependency list when `LATENTOS_STANDALONE_OUT` is set. `bench/latent-binding-check.sh` is CPU-only
and compares a rectangular sender→receiver affine export against the Mojo loader; it does not
claim semantic quality.

Cross-host state uses the versioned `latentos/transport.mojo` session and the libsodium
XChaCha20-Poly1305 shim. The relay stores ciphertext and chunk offsets only; it has no content key.
The older `ipc.mojo` TCP helpers remain legacy plaintext framing with an unkeyed FNV check and
must not be used for untrusted state. Relay expiry, peer identity, and the two-physical-host gate
still require the external network deployment described in the product plan.

`Yahai`'s Baro client now uses the real `/v1/checkpoints` and `/v1/checkpoints/{id}/fork` API,
rejects empty answers/tokens/usage, and explicitly rejects the unsupported cache-only draft flag.
The former `/v1/latent/*` routes return 410 with the checkpoint migration target; they no longer
claim a successful handle or enter a nonterminal engine exchange.

The Granite→Llama Spark adapter compiles with `serve/semantic_latent.mojo`. Compile both supported
profiles with `bench/semantic-latent-build.sh`; a missing profile exits 3 and is never treated as
a pass. The adapter accepts only a versioned semantic manifest plus an f32 projected input, checks
geometry, payload SHA-256, the required-authority flag, and deployment-provided sender/receiver/
runtime/tokenizer/scope/layer/injection/task pins; it prints `semantic consume_count` and can dump
the sender's final-norm f32 row via `BARO_SEMANTIC_DUMP`. This is a live-buffer instrumentation
seam, not a semantic-quality receipt; the required sender-only training/evaluation run remains
separate. Set the `BARO_SEMANTIC_*_SHA256`, scope, layer, injection-site, task-domain, revision and
epoch pins before one-shot use; a mismatch fails before the first request. The current Spark file
seam is intentionally one-shot: serve mode refuses a process-global semantic file instead of
replaying it across unrelated chats. A room adapter must provide a per-request handle before
enabling BARO_SERVE.

## Metrics

`GET /metrics` (Prometheus text): `baro_checkpoint_forks_total{tier}`, `baro_turns_total`,
`baro_checkpoint_imports_total`, `baro_checkpoints_live`, `baro_checkpoint_bytes`,
`baro_turn_ttft_seconds` histogram.

## Not yet

Cross-host routing per request (P4); the MoE profile's int8 pool; a checkpoint that survives an
unrelated request on the device (P2c); NVMe-cold load numbers (all receipts are page-cache hot).
