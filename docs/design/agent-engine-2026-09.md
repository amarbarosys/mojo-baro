# baro agent engine — design (2026-09-08)

Status: design for the `lane-chat` build. Sources: local code (llama.cpp
`tools/server`, `common/sampling`, `common/chat`; deer-flow `config.yaml`,
harness client), NVIDIA docs read 2026-09-08 (TensorRT-LLM `features/kvcache`,
`kv-cache-compression`, `speculative-decoding`, `sampling`, `overlap-scheduler`,
blog 12 guided+speculative; Dynamo `agentic-harness-support` blog, KV offload;
kvpress; RULER), and the 2026-09-08 web consult
(`exchange/web-model-consult-2026-09-08.md`). Claims marked (measured) come
from this repo's receipts; (doc) from the sources above; (inferred) is ours.

## 0. Thesis

The unit of work is the agent turn, not the token. A harness (DeerFlow,
Claude Code, OpenClaw, Codex-style loops) restarts from the same 8-52k prefix
dozens of times per task, appends a tool result, and needs to know as early as
possible that a tool call is ready. Incumbents optimise tok/s for many tenants
on datacenter GPUs; this engine optimises, on one 24 GB card:

| metric | today | target | how it is measured |
|---|---|---|---|
| TTFT, 8k prefix, warm checkpoint | n/a (1088 cap) | < 60 ms | `prefill rows:` receipt, P4 median |
| TTFT, 8k prefix, cold | n/a | < 1.2 s (bf16 prefill), < 0.6 s (int8) | same |
| time-to-tool-dispatch after last token of the call | end of stream | at the closing tag, typed event | tap timestamps |
| decode tok/s_gen, q4 9B | 133.9 (measured) | 140+ (dot-loop schedule) | `bench/ab-prompts.sh` |
| determinism | greedy only | same bytes for same request+seed, spec on/off | CI byte-compare |
| receipts per response | `usage.baro` with `cached_tokens` and `prefill_rows` | plus decode ms, acceptance, checkpoint hit, prefix churn | `usage.baro` |

## 1. Shape table and instantiation

One source, per-shape instantiation from the gguf header (128-bit structural
id: arch, dims, layer pattern, expert count, quant ABI) → `Engine[Shape]`.
Compile-time: tile shapes, unroll, LDS budgets, head dims, block-quant size.
Runtime: layer count, T, page tables, expert ids. The 9B is the first shape:

| | 9B (this lane) | 27B (next) | Flash-Next MoE (later) |
|---|---|---|---|
| layers | 24 GDN + 8 attn | header | header, experts on CPU |
| H / FFN / vocab | 4096 / 12288 / 248320 | header | header |
| attn | GQA 16q/4kv, D=256 | | + indexer, compress 4 |
| SSM state | 32 heads x 128x128 f32 + conv 4x6144 = 26.25 MiB | | |
| weights q4 | 6.2 GB | ~15 GB | CPU-resident experts |

## 2. Memory plan (the tricks we already have, applied)

VRAM tiers on 24 GB: pack 6.2 GB; KV pages; SSM ring slots; scratch. Host:
pinned checkpoints and cold pages. Disk: optional cold tier.

Discipline carried over from the kernel rounds (all measured in this repo):
- Activation row staged once per block into LDS, chunk-major, `ds_load_b128`
  per lane (`stage_a`/`stage_rms`): no per-row VMEM re-read; 31 KB LDS budget
  (48 KB was 1.5 % faster and tripped residency once — keep 31).
- K=12288 operands stay packed bf16 in LDS (24 KB) and unpack at the point of
  use; f32 would not fit.
- Weights layer-major so a chunk's tiles stay warm in Infinity Cache; any
  single-buffer timing at W >= 96 MB is invalid (IC contamination).
- q4 b-form dot: u32 nibble masks, -8 folded into the scale, one-op bf16
  unpack; bit-exact by construction. Wave-per-row template for q8.
- Pack load 0.85 s for 10.7 GB (measured): reload on demand is cheap enough
  that idle VRAM release (ollama keep-alive semantics) is affordable.
- Every GPU job through `gpu-wait` (VRAM-aware queue); the engine reports
  `mega fail word` every run.
- SSM state in ring slots (`sl` slots, `rg` ring index): the checkpoint is a
  slot copy, not a recompute.

New tiers:
- KV pages 128 tokens, K8/V4 default (1664 B/token/layer, 13.3 KB/token over 8
  layers: 1.33 GB at 100k, 13.3 GB at 1M), K4/V4 switch (9.2 GB at 1M) gated
  by the long-context test in §9. bf16 KV mode kept for bit-exact gates.
- Checkpoints (26.25 MiB f32 each) in pinned host RAM: 32 per slot = 840 MiB;
  copy at PCIe 4 x16 is ~1.1 ms each way (inferred from 25 GB/s).
- Cold-page codec (doc, TRT-LLM): pages moving to host/disk may be re-encoded;
  we keep f32/K8 on host (byte-exactness), consider a disk codec only later.

## 3. Prefix reuse for a hybrid model

Mechanism (doc, TRT-LLM KV Cache System + Mamba snapshot boundaries; llama.cpp
`get_common_prefix` + `n_cache_reuse` shifting):
- Radix tree over 128-token KV pages keyed by (prefix hash, salt); a lookup on
  every request returns the longest matched page chain.
- SSM snapshots only at boundaries: every 1024 tokens (periodic) and at every
  prompt end (end offset 0), plus explicit offsets from the start (the 7.9k
  system-prompt end is the anchor every agent call restarts from). A KV match
  is usable only up to the last snapshot boundary at or before it; the rest is
  replayed. This is the TRT-LLM rule for hybrid Mamba models, adopted verbatim.
- Checkpoint = {position, prefix_hash(tokens[0:position]), model/ABI hash,
  generation, conv[24], delta[24], page-chain ref}. Restore requires the hash,
  never position alone.
- Retention: prioritised LRU, priority 1-100 (default 35), request-settable
  (TRT `KvCacheRetentionConfig`); anchors pinned at 100; never trim a diverged
  prefix, fall back to the nearest earlier snapshot. Only leaf pages evict.
- Salting: `cache_salt` mixed into the page hash so two harnesses or users
  never share pages by accident (doc, TRT-LLM).
- Prompt-stability guard (doc, Dynamo): a varying token near position 0 (a
  billing header, a timestamp, a session id) defeats every cache. The engine
  reports `prefix_churn` in receipts (position of the first differing token
  vs the previous request from the same salt) and offers a strip rule per
  harness. Dynamo measured 168 ms vs 912 ms TTFT on a 52k prompt from one
  unstable line.
- llama.cpp's `n_cache_reuse` (shift a matching chunk to a new position) is
  NOT adopted: RoPE-shifting KV is inexact and the SSM state cannot be
  shifted at all.

CI, byte-exact: cold A||B vs restore(A)+replay(B) — conv, delta, KV pages and
next-token logits `memcmp`-equal at positions 0/1/1023/1024/1025/7900/8191/
8192, with mutations at first token, checkpoint±1, last token; a corrupted
hash must fail restoration.


### 3a. What the incumbents' merged PRs settled (read 2026-09-08, bodies in `.work/design/prs/`)
- **Snapshot placement.** Fixed intervals waste memory and still miss the fork;
  TensorRT-LLM #18272 measured agent traces (178 subagent groups): siblings
  share 87-98 % of each other's prompt and always diverge before the end, so
  the valuable snapshot is at the **branch point**, decided at lookup time on
  the diverging request, aligned down to the page. With only prompt-end
  snapshots 7 % of all prompt tokens matched in attention but were pruned for
  want of a recurrent snapshot. We adopt: prompt-end + branch-point snapshots,
  periodic 1024 only as a safety net, `per_conversation` policy for chat.
- **One forward, not two.** vLLM #52789: a checkpoint inside the prompt used
  to split the whole model forward at the checkpoint boundary; splitting only
  the recurrent kernel at that offset inside one forward gave 9-25 % TTFT.
  Our chunked prefill must checkpoint mid-chunk without a second pass over
  the attention/FFN blocks.
- **Restore points survive cancellation.** ollama #17901: agent clients cancel
  long prefills (timeouts shorter than a 40k prefill); a cancelled prefill
  keeps every restore point it crossed so the retry resumes. Every trie node
  carries snapshots spanning exactly its edge from creation (#14887 trie).
  Adopt both; cancel is a first-class path in M1, not an error.
- **Checkpoints at message boundaries.** llama.cpp #24176/#25472: a checkpoint
  at the start of every user message, evicting checkpoints within a
  min-step of each other. Our boundary set = every role boundary in the
  rendered template + prompt end + branch points.
- **Recurrent rollback for speculation.** llama.cpp #26623/#28466: a state
  copy per draft token, roll back to the last accepted; models without
  snapshots corrupt under multi-sequence. Our ring slots already hold one
  state per draft position; the invariant is written down and tested.
- **Adaptive draft length.** ollama #16791: a controller picks the draft
  length that maximises committed tokens/s from measured per-position
  acceptance and forward cost, backs off to plain decode, one host sync per
  round, draft head with its own prefix-cached KV. Replaces our fixed k=2.
- **Reproducible page hashes.** vLLM prefix caching: parent hash + block
  tokens + extra (salt, LoRA, media); `sha256_cbor` for cross-process
  reproducibility. Our hash is canonical bytes, never a language-specific
  pickle.
- **Grammar across the reasoning boundary.** vLLM #44993: under speculation
  the `</think>` marker and the first post-marker tokens must enter the FSM
  exactly once; two silent failure modes were found. Test this case
  explicitly in M6.
- **Streaming cadence.** Dynamo #13975: emit tool-call deltas on every parser
  delta, never buffer to a quiet chunk; #6422: the reasoning state machine
  cycles normal→reasoning→normal per `<think>` block and handles
  `</think>text<think>` inside one chunk.

## 4. Decode

- Megakernel per token stays the m=1 engine (133.9 tok/s measured). Pools:
  the shared q4 dot-loop schedule (VOPD 102 -> 74 with unchanged source), read
  by `isa-loops` before any A/B.
- Long context: GQA-grouped decode kernel, one KV head + its four Q heads per
  wave, K/V loaded once, 32 lanes per 256-dim row, 8 dims per lane, split-K
  over 64-page spans with (m, l, O) partials and one merge workgroup per head;
  no LDS in decode (consult, brief 08).
- Speculation: MTP k=2 (measured 145 race / 100 real prompts). Add the
  model-free drafters the agent workload rewards (doc, TRT-LLM): suffix
  automaton over generated tokens (threshold 4) and NGram prompt-lookup —
  tool results and file contents get echoed, and both are exact under greedy.
  Under temperature: Leviathan rejection sampling with penalties applied
  identically to draft and target; counter-based RNG; adaptive off-switch by
  EWMA of effective tok/s. Relaxed acceptance in the thinking phase
  (TRT `use_relaxed_acceptance_for_thinking`, top-k + delta) is an opt-in.


Reference for the prefill GEMM round (not this lane): vLLM #41394 ships a
native gfx1100 W4A16 kernel pair — scalar `v_dot2_f32_f16` for M < 16,
`v_wmma_f32_16x16x16` for M >= 16 (v1 one wave 16x16 tile, v2 two waves
32x16 double-buffered LDS); on a 7900 XTX, fp16 WMMA was a wash against
scalar+fdot2 end-to-end despite a 47 % microbench win, and the authors name
"multi-wave WMMA with LDS-shared A" as the unbuilt 10-15 %. llama.cpp's int8
MMQ remains the path to copy for Q4 x Q8; vLLM's is W4A16.

## 5. Sampler

Device-side chain in llama.cpp's default order (verified in `common/common.h`):
penalties (repeat/freq/presence over last-N) → DRY → top-n-sigma → top-k →
typical-p → top-p → min-p → XTC → temperature → dist. Temperature 0 = the
current argmax fast path. Defaults come from the gguf generation config
(TRT `generation_config: auto`), Qwen thinking preset temp 0.7 / top_p 0.8 /
top_k 20 / min_p 0; DeerFlow sends temperature 0 and `extra_body.repeat_penalty
1.15` (verified in `~/Projects/deer-flow/config.yaml`). OpenAI names map:
`frequency_penalty`, `presence_penalty`, `seed`, `logprobs`.

## 6. Guided decoding

JSON schema / regex / EBNF / structural tags (doc, TRT-LLM guided-decoding).
Grammar state is advanced per draft token and rolled back on rejection so it
composes with speculation (doc, TRT blog 12). The mask is computed on the CPU
for step n while the GPU runs step n-1 (see §8) and applied on device before
sampling. Tool-call arguments use the tool's JSON schema as the grammar when
`tool_choice` forces a tool.

## 7. Serving contract

APIs: OpenAI chat completions (DeerFlow, LangChain), OpenAI Responses (Codex
style), Anthropic Messages (Claude Code, OpenClaw). Fidelity list from Dynamo
(doc), adopted as conformance tests:
- `GET /v1/models` and `GET /v1/models/{id}` with slashed ids; `input_tokens`
  in `message_start`; tolerate `cache_control` annotations (later: honour them
  as pin/TTL hints).
- One owner for reasoning parsing. Template-native `reasoning_content` when
  the Qwen3 template reads it. Interleaved `<think>` / `<tool_call>` segments
  keep their order on replay; prior thinking behind tool calls is not
  truncated in agent turns (Dynamo measured 167 ms vs 322 ms TTFT from a
  mutated thinking prefix).
- Tool calls stream as deltas AND as a typed `event: tool_call_dispatch` side
  channel the moment the closing tag is parsed, so the harness runs the tool
  while decode continues. State-machine lexer over the raw tag stream
  (llama.cpp `common/chat.cpp` Qwen3/Hermes parsers as reference; grammar-
  constrained arguments when a schema exists).
- Request extension `baro` (mirrors Dynamo `nvext.agent_hints`): `priority`,
  `expected_output_tokens`, `speculative_prefill` (start prefilling the
  shared prefix while the tool runs), `cache_salt`, `retention_priority`.
- Errors: context overflow returns 400 with llama.cpp's
  `exceed_context_size` shape (verified) — DeerFlow's reaction is a test, not
  an assumption.

## 8. Host loop

Overlap scheduler (doc, TRT-LLM): launch step n, then do the CPU work of step
n-1 (detokenise, tag lexer, grammar advance, receipts) while the GPU runs.
Today the loop syncs per token (`host_enqueue_s == gpu_total_s`, measured);
the token on device is 7.36 ms, the wall is 7.66 (measured) — the overlap
recovers most of that 0.3 ms. Stop/cancel: 64-byte control block in pinned
host memory {generation, cancel, state, produced}, polled once per token on
device; EOS set {248046 `<|im_end|>`, 248044 `<|endoftext|>`}; stop strings
matched across token boundaries on the host side.

## 9. Tests and gates

- Existing: `tools/merge-gate.sh`, `tools/mega-gate.sh`, identity 64/64 vs
  `tools/model-ref.py`, 20-prompt P4 A/B (`bench/ab-prompts.sh`, read
  `arm.txt` first).
- Byte-exact restore test (§3). Determinism test: same request + seed, spec
  on/off, 3 runs, byte-equal.
- Long context: RULER subset generated locally (NIAH single/multi-key,
  variable tracking, common-words aggregation; effective length = the largest
  T with score >= the 4k baseline threshold), run through the OpenAI endpoint
  at 8k/32k/64k/128k for bf16, K8/V4, K4/V4. A KV format ships only if its
  effective length equals bf16's.
- Harness smokes: `tools/deerflow-smoke.py` (write_file → read_file → answer,
  call 2 prefill < 200 tokens), Claude Code and OpenClaw against the Anthropic
  endpoint, a tool-call conformance corpus (parallel calls, optional params,
  nested JSON, calls after thinking).

## 10. Milestones (each preregistered, each its own gate)

| M | scope | size | gate |
|---|---|---|---|
| M0 | runtime T, paged bf16 KV, GQA-grouped decode kernel, chunked prefill to 128k | L | identity at T <= 1088 unchanged; TTFT table 8k/32k/100k vs llama.cpp |
| M1 | checkpoints + radix pages + retention + salt + churn receipt | L | byte-exact restore CI; DeerFlow call 2 < 200 prefill tokens |
| M2 | control block, EOS/stop, limits, overlap host loop | M | cancel mid-token; wall - device < 0.1 ms/token |
| M3 | sampler chain on device + rejection-sampled MTP + SA/NGram drafters | L | determinism test; acceptance receipts; P4 A/B no regression at temp 0 |
| M4 | tool/reasoning lexer, dispatch event, three APIs, receipts | L | harness smokes; conformance corpus |
| M5 | K8/V4 and K4/V4 KV, RULER gate | M | effective length = bf16 |
| M6 | guided decoding with grammar rollback | M | schema corpus under spec on/off |

Not in this lane: the 27B port (shape table first), Flash-Next experts,
multi-tenant batching, the int8 prefill GEMM (its own round; M0 uses the bf16
prefill and reports the gap).
