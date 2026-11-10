# Next plan (2026-09-15): five capability gaps and six headline capabilities

Status: DRAFT. Part A is detailed. Part B is filled in after web research
brief 4 (`exchange/2026-09-15-web-research-4.md`) lands, then the whole plan
is dispatched to an opus build lane under the fable coordinator. Effort in
LOC (XS < 20, S < 60, M < 150, L < 400, XL beyond). Every item names the
check that makes it done (CLAUDE.md s18); every timed claim is preregistered
in its `bench/*-protocol.md` before the first run (P1, P4).

Ground truth today (`docs/BASELINE.md`): Qwythos-9B q4 136.37 tok/s_gen
greedy, 150.96 with MTP k=2; 32k decode 101 tok/s; qwen35moe 93.46 tok/s
(llama.cpp 109.4); prefill within 1.3x of llama.cpp at 8k to 32k; four dense
families served; sampling on device with two shapes refused pending rounds;
self-describing bakes verified at `8184f7d`; LatentOS KV/SSM handoff 27x
faster than re-prefill at 32k.


## GPU rule for every item (the maintainer, 2026-09-15): short experiments that prove

The GPU is in constant use. No lane gets GPU-days. Every claim is proven by
a preregistered experiment that holds the GPU for minutes, not hours:
- Each gate below is sized to run in under 10 minutes of GPU time
  (one-prompt receipts for direction, the 20-prompt median only for a
  landing decision, never for exploration).
- Anything longer (training, sweeps) runs as `gpu-wait run --preemptible
  --priority 10` in slices that checkpoint every few minutes, so interactive
  work at `--priority 90` preempts it and nothing is lost; it never holds the
  card outright.
- A build lane reports GPU minutes used per gate in its report; a gate that
  needs more than 10 minutes is split or redesigned before it runs.
- Reads, builds, ISA receipts, parity tests on fixtures and host oracles are
  CPU and free; do them first and often.

## Part A: the five gaps

### A1. Sampled speculation (M, sonnet; after the C3 tail round, S)

Rule (Leviathan 2211.17192, Chen 2302.01318, both apply truncation to
target and draft before the rule): accept draft token x with probability
min(1, p(x)/q(x)) on the adjusted distributions, resample from
norm(max(0, p - q)) on the first rejection, bonus token from p. Host
reference `serve/sample_ref.mojo::spec_accept_ref` exists; device
`kernels/sample.mojo::amar_spec_accept` exists. Wire both into the MTP
verify window under `temperature > 0` (today spec is forced off there).
Prerequisite: the C3 tail round (53-bit uniform for the Gumbel key on both
sides, preregistered in `bench/chat-protocol.md`), else the untruncated
distribution carries the 2^-24 floor.
Gates: (1) T=0 byte-identical on `ref-tokens-64` and the 20-prompt set;
(2) device accept == host reference per draw on fixed seeds; (3) 20000-draw
chi-square of emitted tokens against the target distribution on three real
rows at T=1 and at T=0.7/top_p 0.9, p=0.001; (4) 20-prompt tok/s at T=0.7
with spec on within 5% of the T=0 spec gain; (5) refusal for the untruncated
shape lifted in the same commit as the tail fix.
Fallback: llama.cpp's sample-and-compare (unbiased per token, lower
acceptance), S.

### A2. Paged and quantized KV (L, fable for `kernels/dattn.mojo`, sonnet for the allocator)

**LANDED 2026-09-17: step 1 paging `09a8cd9`/`acf92e5` (16-token blocks, block table, dattn through the table), step 2 int8 KV `0fae7bb` opt-in `-D BARO_KVQ=int8` (`exchange/lane-A2S2-report.md`), 60-prompt identity harness `bench/a2-gate.sh` (`652a86c`).**

Step 1 paging: 16-token blocks, per-request block table (vLLM 2309.06180),
`dattn` split kernel reads through the table; prefix checkpoints and the
MTP (k+1)-slot ring keep working. Step 2 quantization on KIVI's axes
(2402.02750): keys per channel quantized before RoPE (KVQuant 2401.18079),
values per token, int8 with per-block scales first, int4 as a second arm.
Gates: forced agreement (`BARO_FORCE`) 20/20 at 8k, 16k, 32k for each arm;
64k and 128k RULER subsets within 24 GB; decode after p32768 not below
101 tok/s; prefix checkpoint restore byte-identical; the MoE profile
unaffected. This lane is also LatentOS C2 (compressed checkpoints).

**Harness landed first (2026-09-16, fable): `bench/a2-prompts.sh` (three shared-document sets, 8k/16k/32k, 20 prompts each, prefixes of one another) and `bench/a2-gate.sh` (both engines resident, per-request `force` and `ckpt`, reference ids cached the P17 way, `QUICK=N` iteration subset, `MIN_PCT` bar). Verified on two builds that must agree: 60/60 prompts at 100%, 59/60 restored a prefix checkpoint, full gate 91 s wall with the reference cached, quick subset 65 s cold; decode after 32k 101.56 tok/s, the bar above. The 64k and 128k RULER subsets stay a once-per-merge-candidate run.**

**Step 1 (paging) LANDED 2026-09-17 (`09a8cd9`, `exchange/2026-09-17-A2-step1-report.md`): block table through every KV kernel, identity and reverse tables both 60/60, decode after 32k 100.7 vs 101.6, `serve/kvpage.mojo` allocator. Step 2 (int8 KV) LANDED 2026-09-17 (`80d4b2c`, opus lane, `exchange/lane-A2S2-report.md`): opt-in `BARO_KVQ=int8`, bar 93.75 from llama.cpp q8_0 and our bf16, min 96.9 under identity and reverse tables, 32k decode 1.171x, RULER 100 at 64k/128k. Step 3 = int8 state files and checkpoints, int4 (KIVI axes) as its own lane.**

### A3. Continuous batching (XL; plan first, sonnet on the scheduler, fable on the m>1 GEMV)

**IN PROGRESS 2026-09-17 (codex lane, `docs/A3-PLAN.md`, `exchange/lane-A3-report.md`): (a) wire admission `3474e64`, (b) two resident sequence states `a96f6c4`, (c) host row harness and (d) admission gate `0413455` merged. Left: the per-row kernel seam (`exchange/lane-A3-c-kernel-request.md`, five row kernels taking a RowDesc pointer) before any one-launch batching or throughput claim.**

Orca (OSDI 2022): iteration-level scheduling and selective batching, i.e.
attention per request, linear layers batched as rows. The engine already
runs m up to 8 rows per launch and the Rust front already has a request
queue and `BARO_POOL`. Needs A2 (block tables let different lengths share a
batch). Design decisions for the plan stage: prefill chunking granularity
(Sarathi), preemption by recompute (vLLM's own ablation favours it at small
block sizes), per-request sampler state.
Gates: N=4 concurrent clients through `baro-serve`, each output identical
to its single-request run at T=0; aggregate tok/s at m=4 against the P5
row-scaling receipt (delta phase scales 1.46x at m=2, so predict, do not
assume); no regression of the single-request 20-prompt median.

### A4. Trained draft head (M code; GPU in preemptible slices, sonnet)

**PARKED 2026-09-17 (`f3acaab`, `exchange/lane-A4-report.md`): tooling landed (extract, torch replica, real-text dumps, training, byte-verified write-back), baseline reproduced on the real harness (65.97%), two smokes: lr 2e-4 wrecked the head (3.06%), the corrected recipe converged but reads 63.67% against 67.78% untrained. Open decision: retrain against the target's own greedy picks (that is what acceptance measures), or park.**

Acceptance is the lever only through training: DeepSeek-V3's trained MTP
head reaches 85 to 90% second-token acceptance (2412.19437) against our 42%.
E13's trainer, converter and Mojo projector are built and verified
(`~/AMDHQ`, mojo-baro `761472a` `627e8da` `ca10db1`); retarget them to
next-token prediction from the trunk's last hidden state. Expected tokens
per pass (1 - a^(K+1)) / (1 - a) divided by the window cost t(K)
(Sequoia 2402.12374) is the frozen prediction; at a=0.8, K=2: 2.44 / 1.45.
Proof before any long run: a 10-minute training smoke on 2000 sequences
must move acceptance on 5 prompts above the untrained 42% (frozen: >= 50%
or the item is killed); only then a preemptible, checkpointed run in
slices. Gates: acceptance on the 20-prompt set; 20-prompt tok/s at k=2 and
k=3; T=0 identity unchanged (the draft never changes the target's argmax).
Caveat from EAGLE Table 3: on the MoE model the win is bounded by extra
expert reads per verified token; predict it, do not promise it.

### A5. Tool calling and structured output through the grammar engine (M, sonnet)

`grammar/` has JSON-schema and regex to PDA, token-mask fill, accept /
snapshot / rollback. Wire the mask into the device sampler per step
(mask before Gumbel), expose `response_format` (json_schema) and `tools` on
`/v1/chat/completions`, pass `chat_template_kwargs` through to minijinja
(Granite 4.2: `enable_thinking`, `low_effort`; same field name as vLLM and
llama.cpp). Spec decode stays on: rollback on rejected drafts is what the
snapshot API is for.
Gates: schema-valid JSON on 50 prompts at T=0 and T=0.7; a real tool-call
round trip through the server with the call parsed and the tool result fed
back; spec on with at least one rollback exercised; T=0 identity unchanged
when no grammar is set.


### A6. Sampler cost at real vocab (M, fable; found by A1's gate 4)

**LANDED 2026-09-16 (`54fb4bb`, `exchange/2026-09-16-A6-report.md`): the cost was the megakernel bypass at T > 0, not the sampler; no-spec T=0.7 132.30 vs greedy 134.65 same stint, T=0 byte-identical. A6.3 `39f6a4c`: penalties, top_logprobs and grammar on the same route, grammar cost 1.253x -> 1.010x.**

Sampling itself costs 19% of decode at this vocab (T=0.7 no-spec 109.19 vs
greedy no-spec 134.97 tok/s, `bench/spec-sample-protocol.md`): the device
sampler scans 248320 logits per draw where the argmax path is one reduction.
KSAMP-c's design (sampled window, exact compaction, `bench/chat-protocol.md`)
is the fix shape; it was measured on synthetic rows only. Gates: the C3 device
test unchanged (per-token host equality, chi-square on three real rows), T=0
byte-identical, and 20-prompt no-spec T=0.7 within 5% of greedy.

### A0. Prerequisites and hygiene

- Served-path timing of the MoE model after R1 to R3 (one gpu-wait launch;
  the 93.46 is the one-shot engine).
- MoE persistent path (L, fable): a stamp timeline of one MoE token first
  (`bench/carryover-stamp.py`), then move the expert phases into the
  megakernel; research put llama.cpp's remaining edge in launch structure.
- B3 C-language IPC probe (S): decides Mojo `external_call` ABI versus
  driver before IPC handoff is called dead.

## Part B: six headline capabilities (detailed from `exchange/2026-09-15-web-research-4.md`)

Each: prior art and its number; our claim; the hard limit; the reviewer's
check; effort; model; dependencies; the short GPU proof.

### B1. A model file that runs itself and proves its numbers (M to L, opus; no kernel work)

Prior art: llamafile bundles engine and weights for behavioural
reproducibility ("originally observed behaviors can be reproduced
indefinitely") but embeds no hardware receipt, no kernel source, no
rebuild-and-verify; MLPerf verifies claims by a 90-day human audit;
Sigstore model-signing proves provenance of bytes, not performance. No
format found that embeds the kernel source behind a benchmark number.
Our claim: a `*-BARO-<sha>.gguf` rebuilds its engine, verifies its tokens
and prints its hardware receipt on any RDNA3 card with one command, and
can run with no checkout. Hard limit: none in format (GGUF ignores unknown
keys, verified); llamafile's 4 GB Windows loader cap does not apply since we
do not make the file an executable, we make the file self-sufficient.
Build: `baro run MODEL.gguf` = extract sources, build with the pinned
toolchain from `uv sync`, run `gguf-verify`, serve; signed `baro.hw.*`
receipt (Sigstore-style, optional); a `baro.hw.receipts` ledger key that
`gguf-verify --append` extends with a verified card's result. Reviewer's
check: on a second RDNA3 card, the file alone reproduces the 64/64 tokens
and prints a receipt whose card differs from ours. GPU proof: one
`gguf-verify` run per bake, under a minute; the contributor flow in
`docs/amd-family.md` is the multi-card proof. Depends on nothing.

### B2. Agents that share memory, not messages (M, opus with the latent-os-bench skill; GPU minutes per item)

Prior art: LatentMAS (2511.20639) concatenates attention KV between agents
in one Python process, 2.6 to 7x over text; Interlat sends last hidden
states, 24x less communication; CacheGen/CacheBlend/DistServe move KV
across nodes with task-quality proxies. None hands off SSM state, none
crosses processes, none checks continuation identity. Our claim: N
follower agents inherit one reader's full KV plus SSM state across
processes (and hosts, on the A3 rig) in milliseconds, with the continuation
identical to the reader's own. E12/E12-long already prove the primitive
(27x at 32k, 40/40 hashes). Build: E14 (`09-roadmap.md` C1): one reader,
three then ten followers over a 32k document, wall clock end to end, versus
text re-prefill and versus llama.cpp's re-prefill; then the same on two
hosts (B4 of the roadmap). Hard limit: receiver must be architecturally
identical (no adapter), and payload size (2 GiB at 32k) bounds cross-host
until C2 compressed checkpoints (= A2's q8 KV). Reviewer's check:
continuation diff, not task accuracy: follower output equals the reader's
own continuation token for token. GPU proof: each E14 arm is under 5
minutes (one prefill, ten short decodes). Depends on nothing; C2 later.

### B3. One million tokens of context on a consumer GPU (L, fable on `dattn`, opus on paging; A2 taken to the end)

Prior art: Jamba (8x smaller KV, 256k released, 1M trained, on 80 GB);
KVQuant 1M on an A100-80GB with 3-bit KV. Nobody has shown 1M on 24 GB.
Arithmetic (research 4): 10 attention layers x 4 KV heads x 256 dims:
bf16 40 KB/token = 38 GiB at 1M (does not fit); q8 20 KB/token = 19 GiB
(fits only with a tiny resident model); q4 10 KB/token = 9.5 GiB (fits with
the q4 9B trunk, 5.2 GB). So the claim needs q4 KV (KIVI axes: keys per
channel pre-RoPE, values per token) plus paging; SSM state is constant.
RULER's own warning binds the claim: report effective length (the point
accuracy drops below threshold), and note RULER finds SSM-family models
lag Transformers. Reviewer's check: RULER (all task classes, not needle
alone) at 128k, 256k, 512k, 1M with the effective length reported, plus
forced agreement at each length against a bf16-KV run of the same prompt
where memory allows. GPU proof, staged: q8 KV at 32k (forced agreement,
minutes); q4 KV at 32k; 128k RULER subset (tens of minutes, preemptible
slices); only then longer. Depends on A2.

### B4. A 100B-class MoE on 24 GB by streaming experts from host RAM (XL; opus on the RAM tier and prefetch, fable on the gathered-expert kernel)

Prior art: Mixtral-Offloading (LRU + speculative prefetch, 2 to 3 tok/s on a
3060), Fiddler (cold experts computed on CPU, 1.26x), MoE-Infinity
(A5000-24GB, PCIe 4.0 at 24 GB/s measured, 3.1 to 16.7x, but on a 15.7B
model; finds under 5% of experts repeat within a request for ~100-expert
models). Nobody has shown a 100B-class MoE interactive on one 24 GB card.
Hard limit: sustained PCIe 4.0 x16 about 24 to 28 GB/s; per-token expert
bytes for our 35B MoE are 0.78 GB (from the pack index), so a 100B-class
model with the same expert width and top-8 moves 1 to 2 GB per token
uncached, i.e. 40 to 80 ms per token if nothing is resident and prefetch
hides nothing. The claim therefore rests on locality (what fraction of the
selected experts are already resident) and on prefetching the next layer's
experts from the current router (Pre-gated MoE). Reviewer's check: on the
real target model, sustained tok/s, cache hit rate per step, bytes moved
per token against an independently measured PCIe bandwidth (the three
numbers MoE-Infinity reports). Build order: (1) measure PCIe sustained
bandwidth on this box (CPU, minutes); (2) instrument our 35B MoE with a
host-resident expert tier and an LRU of resident experts, keep the trunk in
VRAM, report hit rate on the 20 prompts (minutes); (3) prefetch from the
current layer's router logits for the next layer; (4) only then a larger
model. GPU proof at each stage is a 20-prompt run. Depends on the MoE
persistent path only for the final speed, not for the feasibility proof.

### B5. Forking a conversation into parallel branches at near-zero cost (M, opus; sampler and grammar already built)

Prior art: Hydragen (exact shared-prefix attention, up to 32x throughput,
benefit grows with prefix length and shrinks as branches diverge), Tree of
Thoughts (search logic, no systems answer), SGLang fork over a radix cache.
Our claim: fork at any token into N branches with no recompute of the
prefix (prefix checkpoint restore is 7.6 ms), each branch a first-class
request with its own sampler and grammar state (snapshot/rollback), and
each branch's continuation identical to a from-scratch run of that branch
at T=0. Hard limit: per-branch KV grows independently after the fork, so
N is bounded by VRAM until A2 paging; attention over shared prefix plus
unique suffixes is Hydragen's decomposition, which the split `dattn` kernel
can express later (fable) but is not needed for the identity claim.
Reviewer's check: cost of N forks versus N recomputes at prefix lengths
1k, 8k, 32k, and a token-for-token identity diff per branch. Build: a
`/v1/fork` endpoint (or `n` on completions) on top of prefix checkpoints
and per-request sampler state from A1/A5; best-of-N and a tool-call tree
search as the demo. GPU proof: minutes per run. Depends on A5 for grammar
per branch; on A2 for N beyond a handful.

### B6. Measurement as a product (S to M, opus for the write-up and tooling; no GPU beyond receipts)

Prior art: MLPerf audits; NEUTRINO-style in-kernel instrumentation papers;
the observation that host-synced per-kernel timing cannot separate a
kernel's own cost from what the preceding kernel left in cache and clock
state matches our carry-over probe. Our claim: a published method (the
preregistration rules, the in-stream stamp tool, the carry-over result, the
self-describing file with receipts) that lets anyone reproduce or refute
every number in `docs/BASELINE.md` on their own card. Build: a
`docs/METHOD.md` that states the rules and the two negative results
(per-kernel timing versus geometry changes; the 64-vocab sampler gates that
missed a real-vocab defect), `tools/gguf-verify.sh` as the entry point, and
`bench/carryover-stamp.py` documented as the attribution instrument.
Reviewer's check: a third party runs the verify command and the stamp tool
on their card and gets receipts in the same format. Depends on B1.

## Build order for the opus lane

Non-kernel work in this order, each gated and committed before the next:
A0.1 served-path MoE timing (minutes); A0.3 B3 C-language IPC probe (S);
B1 `baro run` and receipt ledger; A5 grammar and tool calling with
`chat_template_kwargs`; A1 sampled speculation (after the coordinator lands
the C3 tail round, a kernel-file change); B2 E14 demo; B5 fork endpoint;
B6 method write-up; A2 allocator side (block tables) while the coordinator
does the `dattn` side; A3 plan document only (no build); B4 stages 1 and 2.
Kernel files (`kernels/*.mojo`) are never edited by the lane: it stops and
reports, the coordinator (fable) does that part.
