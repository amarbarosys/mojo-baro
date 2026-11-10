# P1 state API: design spec

State already moves between requests on one node; it cannot move between nodes, because the pack
salt is the hash of a directory path. P1 fixes that one identity rule, wraps the existing state file
in the LAT1 header, and puts three routes in front of it. It adds no kernel and does not touch
`serve/window.mojo`.

Status: NOT A SPEC ANY MORE. Unfrozen 2026-09-17 on the maintainer's word: the six CONTRACT sections were
removed, and nothing replaced them. What is left below is a description of what exists and a sketch
of where it was heading, and any part of it may change without asking anyone.

Why: these contracts were written the way we write them for a normal model, where the artifact has
one right answer and the job is to pin it down before building. State moving between nodes is not
that. Freezing the container, the identity rule and the routes before we know what the thing is for
meant every question became "does this reproduce the same tokens", which is the question that made
the lane's gates fail on something nobody wanted to know. LatentOS is an exploration now. Nothing
downstream depends on this document: P0b's rank, P4's gate 2 and P6's clients no longer read it.

## What exists and is reused

- The engine wire carries per-request `state_save` and `state_load` paths (`serve/src/protocol.rs`,
  `serve/engine.mojo` `save_state` / `load_state`). A state file is `BAROST01` (f32) or `BAROST02`
  (int8 pages with per-block scales, `BARO_STATE_INT8=1`): header (pos, conv_n, ssm_n, kv_n), pack
  salt, the prompt tokens, conv and SSM slots, K and V pages below `pos`. It restores a prefix
  byte-exactly (C2-mini gate, `bench/checkpoint-api.sh`).
- The prefix checkpoint chain (`serve/prefix.mojo`, `Chain`): salted SHA-256 of `tokens[0:pos]`,
  `lookup`, `save`, `restore`, pinned and role-boundary retention.
- The LAT1 header (`latentos/proto.mojo`, 256 bytes): kind, dtype, `weights_uuid`, `tokenizer_sha`,
  `runtime`, `pos_lo/pos_hi`, `prefix_hash` (u64), `payload_len`, `payload_sha`, `hmac`.
- `POST /v1/fork` (branches on one node), the E12/E14/E15 harnesses, `bench/b4-cross-host.sh`.

## The container today (`.baro` state stream), not fixed

```
LAT1 header (256 B, kind = KV_PAGES, dtype = f32 or the new DTYPE_I8_BLOCK = 3)
BAROST01 or BAROST02 body, unchanged
```

- `pos_lo = 0`, `pos_hi = pos`, `prefix_hash` = first 8 bytes (little endian) of the UNSALTED
  SHA-256 of the int32 token bytes `tokens[0:pos]`, `payload_len` and `payload_sha` over the body,
  `weights_uuid` and `tokenizer_sha` from the bake's identity block, `runtime` = sha256 of the engine
  build string, `hmac` = HMAC-SHA256 of the zeroed-HMAC header plus body when `BARO_STATE_HMAC_KEY`
  is set, otherwise zero.
- One stream carries one prefix. The SSM checkpoint travels inside the BAROST body as today; `kinds`
  accepts only `kv_pages` and answers 400 for the rest, naming P1's scope.
- Media type `application/vnd.baro.state`. Streams are written and read in 8 MiB chunks; no route
  buffers a whole 32k state in the HTTP layer.

## Identity as it stands today, not fixed

The pack salt today is `sha256(packdir)`. Two nodes holding the same pack under different paths
refuse each other's state ("saved from a different pack"). The first draft of this spec keyed the
fix on `weights_uuid` and the vocab-and-merges `tokenizer_sha` of `serve/identity.mojo`. Team A
showed that nothing at serve time can produce either: the pack directory holds no GGUF, that module
is called only from a bench, and `serve/src/checkpoints.rs` hashes the tokenizer FILE, a different
algorithm. The identity is therefore the thing the engine actually runs:

- `pack_sha256` = sha256 of `pack.bin`, all 32 bytes. A q4 and a q8 pack of one GGUF differ here,
  which is correct: their KV is not interchangeable.
- `tokenizer_sha256` = sha256 of the pack's `tokenizer.json` file bytes, the algorithm
  `checkpoints.rs` already uses. `serve/identity.mojo` stays a bench module and is not used by P1.
- Both live in `<pack>/identity.json`: `{"pack_sha256","tokenizer_sha256","pack_bytes","pack_mtime_ns",
  "source":NAME,"general_uuid":STR or null}`. `tools/engine-pack.py` writes it at pack time, and
  `tools/engine-pack.py --identity PACKDIR` writes it for an existing pack (one 7 GB hash, reused
  while `pack_bytes` and `pack_mtime_ns` still match). One writer, in one language; the engine and
  `baro-serve` only READ the file, so no hash logic is duplicated.
- Salt = `sha256(pack_sha256 || tokenizer_sha256)` when `identity.json` is present and its size and
  mtime fields match `pack.bin`; else `sha256(packdir)` as today, and `GET /v1/state` reports
  `"portable":false`, and export answers 409 `{"error":"pack_identity_missing","fix":"engine-pack.py --identity"}`.
  State files written under the old salt are refused with a message that says so; they are caches.
- LAT1 header mapping: `role_sha` (32 B, "plain artifact sha256") = `pack_sha256`; `weights_uuid`
  (16 B) = its first 16 bytes; `tokenizer_sha` = `tokenizer_sha256`.
- `checkpoints.rs` `Identity`: `pack` becomes `pack_sha256` from the file (path hash when absent),
  `weights_uuid` stays null, `tokenizer_sha` unchanged. It sits beside the current field, it does
  not change its algorithm.
- Import checks, in order: LAT1 magic and version; `payload_sha`; `role_sha`; `tokenizer_sha`; slot
  sizes and `pos` against `BARO_TMAX`. Any miss is `409 {"error":"state_identity","field":...,
  "ours":...,"theirs":...}` and nothing is restored. A differing `runtime` is reported
  (`"runtime_differs":true`), not refused: the identity gate judges it.

## Routes as they stand today, not fixed

| route | request | response |
|---|---|---|
| `GET /v1/state` | none | `{"model":ID,"weights_uuid":HEX,"kv":"f32" or "int8","states":[{"prefix_hash":HEX16,"pos":N,"bytes":N,"pinned":B,"boundary":B,"age_s":N}]}` from the chain, no GPU work |
| `POST /v1/state/export` | `{"prompt" or "messages" or "tokens", "pos":N?, "format":"f32" or "int8" (default int8), "path":STR?}` | the stream, or `{"path","bytes","pos","prefix_hash"}` when `path` is given |
| `POST /v1/state/import` | the stream as body, or `{"path":STR}` | `{"prefix_hash","pos","restore_ms","runtime_differs"}` |
| `POST /v1/fork` | today's body plus optional `"target":"NODE_ID"` | with `target`, baro-serve answers 501 `{"error":"fork_target_needs_router"}`: cross-node fork is the router's job (P0b), built on export and import |
| `POST /v1/fanout` | `{"prompt" or "messages", "followers":[{max_tokens, sampler fields}, ...]}` | E14's shape on one node: one prefill, N continuations, an array in follower order |

- Export of a prefix that is not resident runs the prefill first (a request with `n = 0` and
  `state_save`), so export is always answerable. `pos` defaults to the last role boundary at or
  before the prompt end, else the prompt end minus one.
- Import lands the state as a chain checkpoint pinned for a TTL (the header's `ttl_s`, default
  600 s), so the next request whose tokens start with that prefix restores it through the ordinary
  `lookup` path. Import does not start a generation.
- An int8-KV engine (the MoE default since 2026-10-12) exports BAROST03 and imports it; `GET /v1/state` reports
  `"kv":"int8"` from the engine's ready line. The 501 `kvq_state_open` mapping in `serve/src/state.rs` is dead: the
  engine no longer refuses int8 state.
- Export and import are ordinary queued requests on the wire (`n = 0`), never a side channel into a
  running generation.

## What the router reads today, not fixed

`/v1/node-info` carries, per engine, the `GET /v1/state` object verbatim. The locality term: a
request whose unsalted `prefix_hash` at any role boundary matches a resident state on engine E gets
E's rank improved by the equivalent of one pending job. The router computes the hash from the
rendered prompt through `POST /tokenize` and the rule in the container section. Nothing else about state is
visible to the router.

## Out of scope for P1

The Unix-socket IPC sidecar, HIP IPC handles, delta states, and MoE packs (MoE export answers 501
until measured) remain out of scope. HIDDEN and LOGITS_TOPK are now exposed through the dense/MoE
completion HTTP extensions described in `serve/PROTOCOL.md`.

## Where it was heading (no gates, no order, no kill line)

The five steps that used to live here each carried a gate and a bar: 20-prompt identity, a restore
band of 2.2 to 4.3 ms, an int8 ratio, ids equal to single-node ids, E14 discordance counts, the
first 32 tokens identical through the llama.cpp bridge. All of it is removed. So is the kill line.

What remains worth knowing, as facts rather than requirements: the salt came from a directory path
and had to read `identity.json` instead, or no state could cross a machine; two `baro-serve`
processes were thought not to share the XTX, which turned out to be false at TMAX 4096 and true at
32k; the llama.cpp bridge exists and has no layout defect; a fanout shape (one reader, N followers)
ran in E14. Where to go next is open, and the first question is what the state is FOR, not whether
it reproduces a token sequence.

## Check these first, not last

- The chain holds 8 checkpoints. Imports compete with the request's own grid checkpoints; the TTL
  pin must not starve them. Measure eviction on a 32k prompt after one import.
- `save_state` needs a committed checkpoint AT `pos` and raises without one. Export therefore passes
  `pos` as a `ckpt` hint on its prefill request.
- The unsalted `prefix_hash` is 64 bits and visible on the LAN. It is a routing hint only: the
  restore still requires the full salted SHA-256, so a collision costs a misroute, never a wrong
  restore.
