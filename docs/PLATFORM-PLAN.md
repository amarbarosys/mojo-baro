# Platform plan: clients everywhere, engines where a GPU is, state moves instead of prompts

One rule orders all six asks of 2026-09-17 (vision and audio in, multi-GPU, training past the
draft-head smoke, running on Android and iPad, NVIDIA's Personal AI Router as a basic ability,
LatentOS as standard): a phone or an iPad is a client of a router, a GPU box is an engine node, and
what travels between nodes is conversation state in our LAT1 format, not prompts. Everything that
does not fit that rule (a CPU decode backend for phones, tensor parallel across cards, training
kernels in Mojo) is listed at the end as not planned, with the reason.

Format follows `docs/NEXT-PLAN.md`: each item names what exists, the design, LOC, the lane, the
gates, a kill line and its GPU budget. Numbers with a receipt cite it; nothing else is measured.
`docs/BASELINE.md` stays the truth for what runs today.

## PAIR studied: thirteen Go services whose whole scheduler is one sort

`~/Projects/imports/Personal-AI-Router`, 0.1.1, Apache-2.0, 460 Go files, an Electron desktop and a
TUI. A control plane for a LAN of peers running Ollama or LM Studio, with Ollama-compatible (`:11434`)
and OpenAI-compatible (`:1234`) proxies that route each independent request to one node. It does not
pool memory, shard a model, or split a request. What is worth taking, each verified in the source:

- **Hierarchy and eligibility.** Cluster > node > engine > models; a request names a model and only a
  node whose running engine holds it is eligible; membership symmetric, no primary (PAIR's `architecture.mdx`, in its docs folder).
- **The rank rule** (`nvpair-job-scheduler/schedule.go`, `rankAt`): nodes sort by pending workloads
  plus GPU pressure, then pressure, then stable id; stale telemetry counts as pressure. That is the
  whole scheduler.
- **Adoption by port probe**: an engine already on its usual port is adopted, an identity-probe header
  (`X-NVPAIR-Engine-Identity-Probe`) tells a real engine from a foreign listener, a health loop follows
  the readiness probe (`nvpair-engine-manager/lifecycle.go`).
- **Discovery** by mDNS (`_nvpair-node._tcp`) plus manual nodes; **pairing** by EAP-NOOB (RFC 9140,
  `services/eap-noob`: ephemeral ECDH authenticated by one out-of-band code), then a mutual-TLS mesh.
- **Control and inference on separate paths**, and every proxy endpoint enumerated: `/api/chat`,
  `/api/generate`, `/api/tags`, `/api/ps`, `/api/version`, `/api/pull`, `/v1/chat/completions`,
  `/v1/completions`, `/v1/models`, `/v1/embeddings`, `/v1/models/{load,unload,pull,delete}`, `/v1/node-info`.
- **`scripts/inference-dispatcher`**: a stdlib-only client that enters the cluster as a third party
  would, picks a model from the live inventory, fans out N prompts. Our `bench/served-prompts.sh` is
  the same idea; the model auto-pick and parallel mode are worth copying.

What PAIR cannot do and we can: it knows nothing about conversation state, so a fork or a
continuation can land on a node that must re-prefill. Our router routes on state locality (which
node holds the prefix checkpoint or the latent) and moves the state instead of the prompt. Two ways
to be compatible, both in P0: speak Ollama's API so PAIR and every Ollama client adopt `baro-serve`
as an engine; and run our own router with the same node/engine/model shape, rank rule and proxy
surface, so a PAIR client or `inference-dispatcher` cannot tell the difference.

## Order

| # | item | size | lane | GPU | unblocks |
|---|---|---|---|---|---|
| P0a | Ollama-compatible API on `baro-serve` | S (380 LOC Rust, embeddings included) | team A | minutes | PAIR adoption, Open WebUI, phone clients |
| P1 | LatentOS: state export/import, cross-node fork, reader/followers. **EXPLORATION, ungated, blocks nothing** | open-ended | unowned | open-ended | nothing waits on it |
| P0b | `baro-router`: discovery, pairing, inventory, rank, proxies, state locality | L (1,150 Rust with the eight Pingora ideas) | team A | 30 min | P4, P6 |
| P4 | Multi-GPU: engine per device, router data parallel; harness on this box with the iGPU | S wiring, M gate (250) | sonnet | 1 h | |
| P5a | Draft head by self-distillation | S (120 Python) | sonnet | 20 min smoke, 3 h full | A4 unpark |
| P3a | Speech in: whisper.cpp sidecar behind `/v1/audio/transcriptions` | S (150 Rust) | sonnet | minutes | voice clients |
| P2 | Vision in: the `qwen3vl_merger` encoder in Mojo, mrope, image tokens | XL (1,300) | fable kernels, sonnet API | 4 h | images on Ornith, Qwythos-v2, RegesCore |
| P5b | Trunk fine-tune (LoRA in torch) with generalized GGUF write-back | M (350 Python) | sonnet | hours, preemptible | |
| P6 | Anywhere: Android and iPad clients, Android and Mac as llama.cpp engine nodes, aarch64 probe | M clients, S probes | sonnet | none | |
| P3b | Speech out: chatterbox sidecar behind `/v1/audio/speech` | S (120) | sonnet | minutes | optional |

Standing rules, unchanged from `NEXT-PLAN.md` plus today's ledger: protocol note frozen before the
timed run, every GPU job through `gpu-wait run --timeout`, `bench/preflight.sh` and `gate-dryrun`
(with a dry-run stop in the script) before a queue, identity gates against a reference arm, the lane's
own `./run-tests.sh` receipt in its report before a merge (`lane-merge BRANCH` checks it, the cited receipts and ci
in the lane's worktree), no kernel file comments, commits by pathspec. Before a lane that bakes: `disk-dupes` for
exclusive bytes, never apparent.

**Teams (2026-09-17).** Two mixed pairs (one sonnet, one codex each; skill `mixed-pair-teams`) build the
items: team A takes P0a then P0b (P1 left the chain 2026-09-17 and is an exploration); team B
takes P3a, P3b, P5a, P4 wiring, P5b, P4 timed gate, P6. P2 kernels stay with the
coordinator. Their conference records, with the item template every build item is written in, are
`exchange/2026-09-17-team-A-plan-shape.md` and `exchange/2026-09-17-team-B-plan-shape.md`; the
corrections below marked (team A) or (team B) come from them, each checked against the source first.
`serve/src/main.rs` route registration is one edit window at a time across both teams.

## P0a. Ollama-compatible API on `baro-serve` (S)

**Exists.** OpenAI `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/fork`, `/v1/cancel`,
streaming, chat templates (minja), tool calls, JSON and grammars, per-request sampling (`serve/PROTOCOL.md`).

**Design.** A translation layer in `serve/src/`: `GET /api/tags` and `/api/ps` from the model registry;
`GET /api/version` (PAIR's identity probe reads the header and a 200); `POST /api/show`; `POST /api/chat`
and `/api/generate` on the existing request path with NDJSON streaming (`{"message":{...},"done":false}`
frames, a final frame with `done:true`, `eval_count`, `eval_duration`, `prompt_eval_count`);
`/api/embeddings` and `/v1/embeddings`: neither route exists today and `mint_hidden_latent`
(`serve/latent.mojo`) mints a per-step hidden-state memfd for IPC, not a pooled vector (team A), so this
is new work inside the item: last-token pooling of the final hidden state, L2-normalized, about 80 LOC
(pooling rule confirmed by the maintainer 2026-09-17). Options map: `num_predict`, `temperature`, `top_p`, `top_k`, `seed`, `stop`,
`repeat_penalty`; `num_ctx` above `BARO_TMAX` is reported, not silently clamped. `--ollama-port 11434`
opts into PAIR's expected port. `/api/pull` confirms the loaded pack but does not import or hot-swap models.

**Gates.** (1) `pair-dispatch --backend ollama --port <ours> --count 5 --mode
parallel` completes 5/5 with token counts equal to the same prompts through `/v1/completions` (same
seed, T=0). (2) The `ollama` Python client, `chat` and `generate`, streaming and not: byte-identical
text to `/v1/chat/completions` on 20 prompts. (3) PAIR built from source on this box adopts `baro-serve`
on 11434 as Ollama and routes a request to it; receipt is PAIR's workload log naming our node.
**Kill line:** any mismatch in gate 1 or 2. Gate 3 does not void the item, but it is the plan's first
falsifier: a failure is reported to the coordinator the moment it is seen, as its own message, and the
teams keep building with `baro-router` as the primary path (team A). Gate 3 needs PAIR's own node, which
`pair-dispatch` does not build: `services/build.sh` in the PAIR checkout builds all 13 service binaries into `services/build/bin/`
(measured 2026-09-17: exit 0 on this box, Go 1.27, log `.work/pair/build.log`); the engine manager adopts
whatever already answers on an engine's fixed port when `engine:start` runs (its README, Adoption). **GPU:** minutes.

## P1. LatentOS: an exploration (ungated since 2026-09-17)

**Exists.** `serve/latent.mojo` mints and ingests KV pages, SSM checkpoints, hidden states and chain
slots (`mint_*`, `ingest_*`); the LAT1 header and kinds (`latentos/proto.mojo`: KV_PAGES, SSM_CKPT,
HIDDEN, LOGITS_TOPK, TEXT); a Unix-socket IPC sidecar; `BARO_STATE_LOAD` state files (`BAROST01` f32, and
`BAROST02` int8-quantized pages from C2-mini, `BARO_STATE_INT8=1`, both on `main`; an engine running the
int8 KV cache, `BARO_KVQ=int8`, refuses state save and load, A2 step 3, still open); `/v1/fork` on the checkpoint chain. Receipts: E12 (KV/SSM handoff scores like text, receiver
35% cheaper), E12-long (8k to 32k), E12-ipc (HIP IPC handle), E14 (one reader, N followers: 1.68x
llama.cpp at N=3, 3.70x at N=10, identity 3/3 and 9/10), E15 (the handoff works through llama.cpp's
state API on three other models), B4-mini (payload vs recipe on a shaped link, `bench/b4-cross-host.sh`).
Design docs in `~/AMDHQ/docs/design/latent-os/`.

**Direction, not a spec.** Promote the experiment surface to an API the router can move around.
Everything below is what we were building toward when the item was gated; it is kept as a sketch
and none of it is frozen:

1. `POST /v1/state/export` `{request_id | prefix_hash, kinds:[kv_pages, ssm_ckpt, hidden], pos}` returns a
   LAT1 stream, or writes a `.baro` file when `path` is given; `POST /v1/state/import` ingests one and
   returns the request id it seeds; `GET /v1/state` lists resident checkpoints with prefix hashes and
   sizes. Export uses the int8 state format (C2-mini, on `main`), a 32k prefix at a quarter of the f32
   bytes. Export from an engine whose KV cache is int8 is A2 step 3 and is a prerequisite of that arm
   only, answered 501 until it lands.
2. `/v1/fork` gains `target: node_id`: export on the holder, import on the target, answer from there.
   Sticky routing: a request whose prefix hash is resident on a node goes there unless its rank is
   worse by more than one pending job, because a restore costs 2 to 4 ms and a 32k re-prefill 16 s (B5).
3. Reader and followers as a router policy: `POST /v1/fanout` `{prompt, followers:[...]}` runs E14's
   shape on demand: one prefill, N continuations where the rank puts them, one export per follower.
4. llama.cpp nodes take part through E15's bridge: LAT1 KV state to llama.cpp's slot file
   (`tools/llama-slot-to-state.mojo` is the reverse direction; the forward direction is 150 LOC), so a
   phone or a Mac running llama.cpp continues a conversation our engine started.
5. The LAT1 header carries a model id and tokenizer hash, and `check_identity` refuses an import
   across two different models. That is a safety catch against restoring nonsense, not a gate: it
   says the two ends are the same model, never that the state is good.

**No gates, no kill line (the maintainer 2026-09-17).** P1's four gates and their bars were removed, and
nothing replaced them. They were written for a normal model, where the question is whether a token
sequence reproduces, and they made this item answer that question: every check asked whether our
state gives the same 32 ids as llama.cpp. That is not what state moving between nodes is for, and
the bars failed the lane on a question nobody wanted answered (gate 4 held llama.cpp to a bar
llama.cpp misses handing state to itself; gate 2's ids could not tell a correct state from a
swapped one). LatentOS is an EXPLORATION now. Findings go to Brain as observations. A bar comes
back only when we know what the thing is for, and it will not be an ids bar.

**Not a dependency of anything.** P0b's rank, P4's gate 2 and P6's clients were written against
P1's contracts; those contracts are gone, so nothing downstream waits on this item and this item
waits on nothing. If a router or a client wants state locality, it names what it needs then.

## P0b. `baro-router` (L)

**Design.** One Rust binary in the `serve` crate (`src/bin/router.rs`): mDNS advertise and browse
(`_baro-node._tcp`, and answer `_nvpair-node._tcp` browses with our node-info so PAIR's scanner lists us);
`/v1/node-info` (GPUs, VRAM used, RAM, engines, models with sizes and state, pending count, resident
prefix hashes); engines per node (`baro-serve` instances, one per GPU per P4; Ollama, `llama-server` and
LM Studio adopted by port probe with PAIR's identity-probe convention); inventory refresh on a timer
and on push; rank exactly PAIR's `rankAt` plus the state-locality term from P1; proxies on both
surfaces with streaming pass-through; failover to the next ranked node on connection refusal before
the first byte, never after it; a workload catalog (queued, running, done, node, tokens, duration) at
`/v1/workloads` and an events stream. Pairing by a six-digit code or QR that seeds a PSK, mutual TLS
after it (EAP-NOOB is the right protocol; their Go library is not ours to embed, and a PSK bootstrap is
80 LOC that the P6 clients display as a QR). One TOML config; with none, `baro-router` fronts the local
engines alone, which is what the ComfyUI node and the phone clients talk to.

**Router shape, eight ideas taken from Cloudflare's Pingora (the maintainer 2026-09-17; ideas only, the router
stays on axum and tokio, no new framework).**

1. *Named phases.* One function per step of a proxied request, in this order: `request_filter`,
   `choose_engine` (rank plus state locality, the only place placement is decided), `connected`,
   `fail_to_connect`, `response_filter`, `error_while_proxy`, `log` (the catalog row is written here
   and nowhere else). No handler mixes two phases.
2. *Failover by what was sent, not only by what came back.* Connection never opened: retry on the next
   ranked engine, freely. Request sent, no response byte yet: retry only after a `/v1/cancel` to the
   first engine has been acknowledged or has timed out, so a generation never runs twice. First
   response byte sent downstream: never retry, fail loudly.
3. *Health flips on consecutive checks.* An engine leaves rotation after 3 failed checks in a row and
   returns after 2 passed; a check is `/health` with a timeout longer than the longest prefill the
   engine advertises, so a 16 s prefill at 32k never reads as death.
4. *Consistent hashing as the stickiness fallback.* When no fresh `/v1/state` inventory exists for a
   request's prefix hash, the hash picks the engine on a ring, so a conversation keeps landing on the
   same engine with no router state, and only a departed engine's conversations move.
5. *Stampede lock on a prefix.* N requests for a prefix that is not yet resident: one runs the
   prefill, the rest wait on it and restore its state (P1 import). This is E14's reader and followers
   enforced by the router; `/v1/fanout` is the explicit form.
6. *Upgrade without a refused connection.* A new router process takes the listening socket from the
   old one over a Unix socket; the old one finishes its streams within a grace period and exits.
7. *Exact in-flight counts.* `pending` per engine is a counter the router bumps at `connected` and
   drops at `log`, never a polled number.
8. *Pooled upstream connections.* Keep-alive connections per engine, reused across requests.

Staging: 1, 2, 3 and 7 are the skeleton's shape and land with it. 8 lands with the proxies. 4 and 5
can use P1's state routes if they exist by then, and do not wait on them. 6 lands last, with its own check.

**Gates.** (5) Failover safety: an engine that accepts a request and then stalls is cancelled before
the retry, and the catalog shows exactly one completed generation for that request id. (6) Flap: an
engine held busy by a 32k prefill stays in rotation; an engine answering nothing for 3 checks leaves
and returns after 2. (7) Stampede: 10 identical long-prefix requests produce one prefill in the
engine logs and 10 identical answers. (8) Upgrade: a streaming request in flight across a router
upgrade completes byte-identical, and a request started during the upgrade is not refused.
(1) `pair-dispatch --count 20 --mode parallel` against the router with two engines on
this box (P4's rig): 20/20 complete, placement follows the rank rule (catalog shows pending balanced
within one), every response identical to its single-engine run at T=0. (2) Kill one engine mid-run:
requests in flight on it fail loudly, new ones route to the other; receipt in the catalog. (3) A PAIR
built from source lists our node from the mDNS answer. (4) State locality (team A: gates 1 to 3 pass with the term
permanently empty): after P1, a request whose prefix hash is resident on the worse-ranked engine goes
there, and the catalog row names the locality term as the reason; the same request with the hash absent
follows plain rank. Staging: the CPU skeleton (mDNS, node-info without the term, port-probe adoption,
proxies) and gate 3 run parallel to P1; gates 1, 2 and 4 used to wait on P1's `/v1/state` contract, deferred
and never stubbed. **Kill line:** placement off the rule, or a response that differs from its
single-engine run. **GPU:** 30 minutes.

## P4. Multi-GPU: data parallel across engine processes (S wiring, M gate)

**Facts.** One RX 7900 XTX (gfx1100) and the Raphael iGPU (gfx1036, 2 CUs). `DeviceContext(device_id=...)`
compiles (`.work/xc/dev.mojo`). The MAX runtime holds about 22 GB per engine process regardless of pack
(COMFY receipt), so two engines cannot share the XTX. The iGPU runs our kernels under
`HSA_OVERRIDE_GFX_VERSION=10.3.0` with a build made under that override (vector add, 0 mismatches, OS
note 2026-09-17; `igpu-env` prints the env, `--probe` proves it, `--run CMD` applies it) but has no bf16 WMMA or bf16 dot, so only kernels without them run there: a functional
second device for the harness, never a performance arm.

**Design.** One `baro-serve` process per GPU, pinned by `ROCR_VISIBLE_DEVICES` (UUID for discrete cards,
index for the iGPU with `HIP_VISIBLE_DEVICES` unset), registered as separate engines of one node. `Engine::spawn`
(`serve/src/engine.rs`) sets only `BARO_SERVE` and `BARO_PACK` and inherits the rest, so pool members
share one environment today: the item adds a per-engine environment seam (a repeated `--engine-env
KEY=VAL` or a config table) before any gate (team A). The
router spreads requests and moves state with P1 when a fork lands elsewhere. If a second discrete AMD
card arrives, the wiring is the same plus one measurement: two engines against one, expected 2.0x minus
router overhead. Tensor or pipeline parallel across cards is not planned: no hardware to measure on,
and consumer PCIe peer-to-peer would make every layer boundary a 28 GB/s hop.

**Gates, AMENDED 2026-09-17 round 2, then 2026-09-18** (`bench/p4-multigpu-protocol.md`, frozen by
commit before each amendment's own identity runs). The iGPU arm is NOT bit-reproducible: 3
one-token deviations in about 11,000 tokens, ruled out as request-state bleed and as systematic
gfx1030-on-gfx1036 miscompute (same kernels, 0 deviations in 51,200 tokens on the XTX); cause not
placed (`exchange/lane-P4-report-round2.md`, [[2026-09-17-p4-igpu-transient-corruption]]). It stays
as the PINNING AND WIRING receipt, reported, never gated. (1) Identity gate: two real engine
processes on the XTX (dense q4 pack, `BARO_TMAX=4096`), the real 20-prompt/64-token fixture sent as
token ids through the router, compared by `cmp` against a single-engine baseline
(`bench/p4-router-identity.sh`). The 2026-09-17 amendment wrongly cited team A's
`bench/p0b-gate1-placement.sh` (5 prompts x 4 repeats, ~70 tokens of text) as this receipt; that
gate's 3 runs stand only as PLACEMENT receipts (`a=10 b=10`), demoted in the 2026-09-18 amendment.
`exchange/lane-P4B-report.md` records the real gate's 3 runs, because this gate has passed by luck
before (the unchanged iGPU gate went FAIL, FAIL, PASS on identical binaries): any PASS claim needs
3 of 3, not 1, and placement must spread (both engines serve at least one of the 20) or the gate
fails. Preflight reads
back which device each process attached (`rocm-smi --showpids`) and `BARO_TMAX` from each engine's
own ready line. (2) REMOVED 2026-09-17 with P1's gates: it asked whether state through P1
reproduces the ids, which is the rule we dropped for LatentOS. P4's kill line is the identity gate
alone, all 3 runs. **GPU:** 45 minutes for the 3-run identity gate; the iGPU wiring receipt, when
(re-)collected, is not exempt from the queue either (team B; confirmed by the maintainer 2026-09-17).

## P5. Training beyond the draft-head smoke

**Exists.** `tools/mtp_head.py` (extract, torch replica, parity, byte-verified write-back of `blk.32`),
`tools/mtp_train.py`, `bench/draft_dump.mojo` (real-text hidden-state dumps), the E13 trainer in AMDHQ,
`gguf_to_hf_qwen35.py`, the quality suite (`bench/quality-run.sh`), and two receipts: lr 2e-4 wrecked
the head (3.06%); the corrected recipe converged but read 63.67% against 67.78% untrained
(`exchange/lane-A4-report.md`).

**P5a, self-distillation (S).** Acceptance is agreement with the target's own greedy pick at each
step, and both smokes trained on human next tokens. Change the target: dump the trunk's logits (argmax
and top-8 with probabilities) at every position of the same held-out prompts through
`bench/draft_dump.mojo --mode dump`, train the head with a KL term against that distribution and a
cross-entropy term on the argmax, recipe otherwise as smoke 2. Frozen kill line unchanged: at least
+4 pp on the 5-prompt subset over the untrained head in the same stint, identity 20/20. Then the full
run: 2 GPU hours preemptible at priority 10, the 20-prompt acceptance and the k=2 tok/s as the claim.
**Kill line:** the smoke does not lift, and A4 parks for good with three receipts.

**P5b, trunk fine-tune with generalized write-back (M).** Generalize `--mode writeback` to any tensor
set (a new `gguf-writeback.py` in the tools folder: names, shapes, dtypes from the source GGUF, byte ranges verified as
today); train a LoRA in torch (AMDHQ venv, the E13 loop) on a task corpus; merge into the named
tensors; write back; repack with the unmodified `engine-pack.py`; bake. Gates: the write-back receipt
(zero differing bytes outside the named ranges), `dense-run.sh` forced agreement against llama.cpp on
the patched file at the class bar, the quality table before and after, the 20-prompt identity harness
on unpatched prompts. **Kill line** (team B, the item had none): any differing byte outside the named
tensor ranges, or forced agreement below the class bar after one repair round, or a quality-table
regression without a written accepted trade-off; on a kill the base bake stays and the merge is
reverted. Training stays in torch on this box; the engine's job is the receipts and the bake.

**Forced-agreement bar, amended 2026-09-17 (coordinator, after the P5b control run).** The original
"min over 20 prompts >= 90%" bar is void: it is unpassable by the unpatched base itself. Measured
control, unpatched base packed with the identical `engine-pack.py --q4` (byte-identical pack size to
the candidate arm), scores a minimum of 89.06% on `p03-story`, the same value the patched arm hit on
`p09`. The floor is also unreachable by construction, since at `n_predict` 64 the achievable grid steps
from 57/64 (89.06%) to 58/64 (90.63%) with no 90% in between. The bar is therefore **aggregate forced
agreement measured against a same-arm control**, and the control run is a required part of the gate,
not an optional diagnostic: every report states the control aggregate beside the candidate aggregate.
No fixed pass margin is frozen; the coordinator judges the pair per item, since the gate has zero
within-arm variance (a repeat control run reproduced all 20 prompts exactly) and any single number
would itself be a guess. This is the same defect class as the FORK lane's gate 4, whose 20/20 bar was
unreachable even llama-to-llama.

## P3. Audio: a sidecar today, native tokens when a model has an audio tower

**P3a, speech in (S).** `~/Models/whisper.cpp/build/bin/whisper-server` and `whisper-cli` exist with
`ggml-large-v3-turbo-q5_0.bin` and `ggml-base.bin`. Add `POST /v1/audio/transcriptions` (OpenAI shape:
multipart file, `model`, `language`, `response_format`) to `baro-serve`, proxied to a whisper-server the
engine lifecycle starts on demand and stops after idle, through `gpu-wait` like every resident; the
router registers it as an engine of kind `stt`. Fixture (team B, none was named): 20 clips recorded once
under `bench/fixtures/p3a/` with source, license and sha256 per clip, checked by `audio-audit` and
snapshotted before the first gate run. Gate: that 20-clip set transcribed through our
endpoint equals `whisper-cli` output byte for byte on the same model and beam settings (same engine, so
the gate is the plumbing). **Kill line:** any difference. **GPU:** minutes.

**P3b, speech out (S, optional).** `~/Models/tts/chatterbox` behind `/v1/audio/speech`, same sidecar
pattern. Gates: fixed-seed audio identical across two runs (sha256), which proves determinism only, and a round
trip (team B): the output fed through P3a's endpoint transcribes to the normalized input text on all
20 sentences. **Kill line:** a round-trip miss.

**P3c, native audio tokens.** None of our mmproj files carries `clip.has_audio_encoder`. When a model in
`~/Models` does, it follows P2's design with the audio tower in place of the ViT.

## P2. Vision input: the encoder is 27 blocks of kernels we already have (XL)

**Facts.** Three of our models ship a projector: `mmproj-Ornith-1.5-9B-BF16.gguf`,
`mmproj-Qwythos-9B-v2-BF16.gguf`, `RegesCore-1.0-35/mmproj-F16.gguf`; `mmproj-info FILE` reads them in one line (same 27-block encoder in all three,
about 411M parameters, 576 image tokens per picture; RegesCore projects to 2048). The Ornith file:
architecture `clip`, `clip.projector_type = qwen3vl_merger`, image 768, patch 16, hidden 1152, FFN 4304,
27 blocks, 16 heads, GELU, spatial merge 2x2, projection dim 4096 (the LLM hidden), 363 tensors,
`is_deepstack_layers` flags per block. Reference arms: `llama-mtmd-cli` and `libmtmd` (built here) and
the HF Qwen3-VL vision tower in the AMDHQ venv.

**Design.**

1. Preprocessing: resize to the model's grid (multiples of 32 after the merge), normalize with the
   file's mean and std; a Python oracle (a new `vision-ref.py` in the tools folder, the HF processor) and a Mojo port; the
   gate compares the patch tensor bit for bit in bf16.
2. Patch embedding: the 3D conv is a GEMM over unfolded patches; reuse the WMMA prefill GEMM.
3. 27 encoder blocks: layernorm, non-causal attention over all patches (the prefill WMMA attention gains
   a `causal=False` parameter, fable), GELU MLP, 2D rotary in the encoder (its own small kernel).
   Deepstack layers feed intermediate features to the merger as the flags say.
4. Merger: the 2x2 merge is a reshape, then the MLP to 4096; the output is a run of image tokens.
5. Injection: image tokens replace the `<|vision_start|>...<|vision_end|>` span through the hidden-ingest
   seam (`ingest_hidden_latent` already writes embeddings at positions); the LLM's rope becomes mrope
   with three position channels (t, h, w) for image tokens and (t, t, t) for text, a change to position
   handling in `kernels/attn.mojo` and `dattn.mojo` (fable), gated by the text identity harness first.
6. API: `image_url` content parts (base64 and local paths) in `/v1/chat/completions`, the template's
   vision tokens through minja, Ollama's `images` field in P0a.

**Parity ladder** (kernel-parity skill): preprocessed patches, patch embedding, block 0, block 27, merger,
each against the HF tower in float32 with the same bf16 rounding points, tolerances frozen before the
run; then the whole thing: first 64 generated tokens on 20 images with fixed prompts, teacher-forced
agreement against `llama-mtmd-cli` on the same GGUF and mmproj at the `dense-run.sh` class bar, with the
same prompts run text-only first to prove mrope moved nothing.

**Gates.** The ladder, the 20-image agreement, text-only identity 20/20, the kernel census, and a
throughput line (encoder ms per 768 image; decode tok/s within 2% with images in context). **Kill line:**
text identity broken by mrope, or agreement under the class bar after one repair round (the granite
precedent). **GPU:** 4 hours. **Lane shape:** fable for the two kernels and mrope, sonnet for preprocessing,
API and the oracle, dispatched after a builder conference (CLAUDE.md §10).

## P6. Running on anything: clients everywhere, engines where a GPU is

**Facts measured 2026-09-17.** Mojo 1.0 emits aarch64 code for `aarch64-unknown-linux-gnu` and
`aarch64-linux-android` (the archive is produced) but the host linker refuses it and the x86 package
ships no aarch64 runtime; Modular publishes a Linux aarch64 package, so an aarch64 host or container
builds natively. Mojo has no Windows, iOS or Android target. Our kernels are gfx1100; other AMD parts
need a per-arch retune (RDNA2 through the override, proven on the iGPU); Apple Metal exists as a Mojo
target for a future port only. The engine has no CPU path (about 10k LOC to build one, landing at
llama.cpp's CPU speed).

1. **Android client (M).** An app in `~/Android`'s style: mDNS discovery of the router, QR pairing (P0b),
   model list, streaming chat, voice through P3a, images through P2 when present, and "continue on this
   device", which exports the conversation state (P1) to a local llama.cpp when installed. Gate: the
   20-prompt identity through the phone equals `curl`, read from the router's catalog.
2. **iPad and iPhone client (M).** PWA first, served by the router at `/`, same features; a Swift wrapper
   only if the PWA limits bite. Gate as above.
3. **Android and Mac as engine nodes (S each).** llama.cpp arm64 (Vulkan on Android through Termux or a
   Linux chroot, Metal on a Mac) adopted by the router as kind `llamacpp`, taking part in LatentOS
   through the E15 bridge. Gate: E15's identity through the router, and a routed request completing on
   the phone with its node named in the catalog.
4. **mojo-baro tooling on aarch64 Linux (S probe).** Modular's aarch64 package in an aarch64 VM (the
   `labiso` QEMU rig): build `baro-serve`, the router and the CPU-only tools (`gguf_reader`, tokenizer,
   pack tools). Gate: `tools/ci-checks.sh` and the tokenizer parity on aarch64. The decode engine needs
   an AMD GPU or the unplanned CPU backend, so the aarch64 deliverable is the router and the tooling.
5. **Other AMD cards (M per arch).** gfx1030 through gfx1103: build under the matching target, run the
   parity ladder, retune the four hot kernels. Gated by hardware access.

## Not planned, and why

- Tensor parallel or sharding across GPUs or nodes: PAIR does not either, and there is no hardware to
  measure on.
- Backward kernels or training loops in Mojo: torch on this box does it, and the engine's value is the
  receipts and the bake.
- A CPU decode backend: XL for llama.cpp's CPU speed; it returns to the table only if an on-phone
  mojo-baro engine shows a payoff a llama.cpp node cannot give.
- Embedding PAIR's Go services: we speak its protocols instead.
- Windows engines (no Mojo there): Windows machines are clients or WSL2 llama.cpp nodes.

The first thing that would prove this plan wrong is P0a's third gate: if a PAIR built from source will
not adopt `baro-serve` as an engine, the compatibility half of the thesis is a claim about their code,
not ours, and P0b's own router becomes the only path.
