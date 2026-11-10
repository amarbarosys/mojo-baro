# mojo-baro capabilities

Every row carries its own date and check. The newest rows (2026-10-21) are first; older rows were not re-measured unless they say so. The current model list with dates is in the README "What runs" table.

## Claims status (rewritten 2026-10-09)

Read this section first: where it contradicts a later paragraph, this one is newer. Every row names its check or says what is missing. Unchecked features from 2026-09-27..28 were
deleted from the code on 2026-10-22, not listed.

2026-10-21 rows (newest; each names its check):

- **q4x decode dispatch fusion: WORKS, DEFAULT-ON** (`d422318`): 1244 -> 992 dispatches per decode token, same-job
  B/A 0.9536 at the 16K window, 20 prompts NG 64, identity 20/20, `bench/q4x-gate.sh` 20x32 PASS 99.38% vs 98.91%.
  `Q4X_UNFUSED=1` restores the old launches. Measured on a degraded PCIe link (2.5 GT/s): the ratio holds, absolutes do not.
  Warm, 16 GT/s (2026-10-21): NG 256, 20 prompts, B/A 0.889 (paired 0.901), identity 20/20; ~0.947 against the second A run alone (the first ran in leftover host load).
  Gate proof: `bench/check-fuse-mutants.sh` (`d181a44`) catches 3/3 planted fused-kernel bugs.
- **Shared exp/log/rsqrt/sin/cos, CPU == GPU bits: WORKS, DEFAULT-OFF** (`-D BARO_SHARED_EXP=1`, `kernels/bexp.mojo`):
  40017/40017 Float32 inputs bitwise equal, max 1 ulp, `bench/check-shared-math.sh` (mutant caught). fexp32 costs 3.6x
  std exp in an exp-bound kernel (newera `af914d0`).
- **CPU backend parity: PARTIAL** (`ada7420`): one q4 dot (12288 real rows) and one dense attention layer bitwise equal
  to the GPU, `bench/check-cpu-parity.sh` PASS with mutant. A full CPU decode is not built.
- **Per-token Freivalds certificate: PARTIAL** (`ada7420`, `bench/check-certificate.sh`): KAPPA 0.125, 336/336 honest
  checks pass, 7/7 kernel mutants caught, 15.9 us host check per matrix. Engine hook (lane-cert-hook, `BARO_CERT=SEED`,
  layer 3, launch path, default off): token identity 20/20 prompts, 2560/2560 cert lines pass 7/7, planted GEMV mutant
  fails 64/64 tokens, `check-fuse-mutants` and `run-tests.sh` green. Decode B/A **1.023** at 16.0 GT/s (20 prompts, ABBA,
  `bench/cert-ab.sh`), over the 1.01 kill line: the check as shipped (serial after the token sync) is KILLED for
  default-on; 0.23 ms per token.
- **q4x MTP verify window: FAILS** (lane-q4x-mtp `74f9868`, not merged): B/A 1.069 (K=2), 1.095 (K=4), identity PASS;
  measured on the degraded link, so the number is inflated, the cause (verify rows scale like decode steps, more
  distinct experts) stands. `mtp_pass` window bound fix merged (`9b5dcb5`).
- **MTP-draft expert prefetch: KILLED** (lane-draft-overlap `0614239`): draft-router experts match verify experts in
  7.36% of routes (40 arms), saved miss bytes 7.33%; kill lines 50% / 10%.
- **Q4X_MISSPAR, Q4X_DEVARG: PRESENT, UNMEASURED** (lane-q4x-host `99709a9`, not merged): speed runs need a 16 GT/s link.
- **Quick baseline** (`bench/quick-speed.sh`, 3 prompts, not a stint): dense 9B Q4_0 1.34x llama.cpp, Qwen2.5-7B
  Q4_K_M native K-quant 1.10x (README row).

The four ideas, now real work
- **Teleport KV (GPU to GPU state handoff): PARTIAL.** The 2026-09-28 `BARO_TELEPORT_KV` switch only read and
  printed an env var. Real path: HIP IPC through the shim `amarbaro_ipc_open` (`fda465b`) and
  `serve/latent_ipc.mojo` (`KvIpcStage`, `ingest_kv_ipc`). 40 MiB of KV pages between two processes: 0.51 ms
  with a fresh stage and handle (14.1x the memfd path, 7.22 ms), 0.37 ms reused (19.7x), every float equal
  16/16 (`bench/kv_ipc_bench.mojo`). A mapped buffer reads at 430 GB/s in a kernel. Missing: a live engine pair
  handing state through the LatentOS agent under a `BARO_FORCE` gate (T3b).
- **miniagi latent trainer (1.59M params, 8-hop pointer task): WORKS** (`miniagi/README.md`). Phase C entry 3
  PASS 99.73/99.56/99.54% in 5.9-8.5 s; parity gate `miniagi/latent/dcheck.sh` (fp32 exact, bf16 inside torch's
  draw spread, every case, seed 13 PASS `3ed64bd`). Soft thoughts show no gain over text CoT at this scale;
  self-training (`star=1`) collapses to chance (smoke `4396610`). Jacobian lens readout of hidden states on
  Qwen2.5-0.5B: hop visible 27.4% vs 3.6% chance (`465b69a`).
- **miniagi neuron hops (1.59M + W_a/W_b, 2-3 link wikinet questions): WORKS WITH A READOUT SIGNAL, NOT WITHOUT** (`miniagi/latent/PREDICTIONS-neuron.md`, `RECEIPTS-neuron.md`).
  Frozen grid, seeds 110-112, 1500 updates, aux 0.5: real 98.3/98.9/98.8% vs shuffled 26.2/24.0/25.5% vs none 31.5/31.9/32.0% (no-lookup ceiling 31.8%); bar met 3 of 3.
  Real with aux 0 stays at 32% (1500 updates, lr 1e-3) and 28% at 4500 updates, lr 3e-4 (seed 110): the lookup does not train from answer loss alone. fp32 path = torch oracle at 1e-4; bf16 path 7% off, UNVERIFIED.
- **Latent recurrence: UNVERIFIED, no signal yet.** `BARO_BI_RECURRENCE_STEPS` (09-28) was read and printed only.
  `BARO_LATENT_COT_STEPS` in `serve/spark.mojo` is a real loop but needs a model trained for it. The frozen
  latent ladder (run40b, receipt `exchange/receipts/latent-ladder/`): CoT 77.5, latent R8k32 75.0, no-CoT 70.0,
  SHUF 0, FLAW 32.5 (the control failed, so no verdict). Next: a second frozen round; LoRA training only on signal.
- **Tree speculation: CLOSED AT STAGE 0.** The "EAGLE-2" claim had no caller for `ngram_propose_tree`. The
  simulation that does call it (`tools/ngram-tree-sim.mojo`, `bench/ngram-tree-protocol.md`) puts every arm at
  0.71-0.97x of k=2 MTP on 20 prompts. The MTP draft tree was closed earlier (`bench/draft-tree-protocol.md`).
- **int8 KV: dense WORKS, DEFAULT-OFF; MoE WORKS, DEFAULT-OFF (2026-10-12: agreement within an f32 reorder control, 56 vs 57 of
  3840; decode after 32k 1.077x, `bench/int8-moe-protocol.md` rounds 3-4); spark UNVERIFIED.** History: Dense passed A2S2 (min 96.9% forced
  agreement, 1.171x decode after 32k). MoE (2026-10-11): 60-prompt gate FAIL, p11@16k 92.2% < 93.75 (f32 same-arm
  control 60/60 at 100%); one scale per 32 values cut the per-layer error 43% and left agreement unchanged
  (22/22/23 mismatches of 1408). Confirmed 2026-10-12: a pure f32 prefill-scan reorder (`BARO_PF_SSM=chunk`) flips 20 of the
  same 1408 on mostly the same prompts, so the gate's min bar sits on a perturbation floor, not on int8's error; bar under review
  (`bench/int8-moe-protocol.md` round 3). Spark int8 (`8bc4010`) has no gate. State export and import work on an int8
  MoE server (BAROST03, round trip in `bench/tests/baro-moe-live.sh`, 2026-10-12). The MoE bake serves in tier mode:
  a bake whose `baro.run.env` sets `BARO_TIER` makes `tools/baro serve` split its pack into trunk + `experts.bin`
  (`tools/pack-split-experts.py`, under `~/.cache/baro/<id>/tier-<sha8>`), because the resident pack OOMs next to the
  desktop. `RegesCore-1.0-35B-UD-Q4_K_S-BARO-7d42feb.gguf` carries that env and int8 reference ids (the bake's own
  default build; int8 vs f32 is the agreement gate's question, not bake verify's): `tools/baro verify` PASS 64/64,
  `tools/baro-serve-gate.sh` 64/64 cold and warm (`exchange/receipts/moe-bake-int8/`). The `e40d9ca` bake still
  carries f32 ids and fails verify at position 8 by design. Re-reference with
  `BARO_REBAKE_REF=ids BARO_RUN_ENV=... tools/baro-rebake.sh OLD.gguf`.

Other 2026-09-28 / v0.4.0 rows, current status
- **Per-shape prefill tile table (`BARO_TUNE`): PARTIAL, opt-in** (merge `19a474e`). Tile choice per prefill
  GEMM shape, measured once, carried in the GGUF as `baro.tune.pf`, read at boot. +0.4% prefill at 128 tokens,
  null at 256..8192 (`exchange/lane-TILE-TUNE-report.md`; sweep drift about 30% within a job, only paired A/Bs
  count). Bake path UNVERIFIED. Replaces the false "microsecond JIT hot-patching" claim (`15d1a8b` read metadata
  and printed).
- **Adaptive KV pruning (score <= 1e-5 in `amar_attn_decode`): REVERTED** (`fb9f92f`). It flipped p13-haiku;
  decode no-spec 151.9/152.2 without it vs 150.8/152.9 with it. Receipt `exchange/receipts/attn-revert/`.
- **Self-describing GGUF closure gate: WORKS** (`0a5fcbf`), part of `tools/ci-checks.sh`.
- **Prefill WMMA ping-pong tiles (`34670b2`): real code, UNVERIFIED** (no speed arm, no agreement gate).
- **int8 MMQ prefill (`07fb0dd`): real, parity-tested, not the default**; measured 0.91x of bf16 on the same
  schedule, so no speed case.
- **MLA decode (`amar_attn_mla_decode`): PRESENT BUT UNUSED.** Imported, never called; no model here uses MLA.

Engine and kernels
- **Batched MoE prefill (qwen35moe): WORKS, DEFAULT-OFF (`BARO_PREFILL=1`), bit-identical to
  replay.** Prompt rows go through `moe_prefill_forward` in 1024-row chunks instead of one m=1
  window per token. Check: `bench/moe-prefill-identity.sh`, PASS 23/23 prompts equal over 64
  greedy tokens in tier mode (256-row chunks, so long prompts cross chunk boundaries) and in
  resident mode, 12 of the 23 exercise prefill; receipts `exchange/receipts/MOEPF/`. Kernel
  receipts: `kernels/test_moe_rows.mojo` (16 outputs bit-exact vs the m=1 kernels),
  `kernels/test_ssm_rows.mojo` (delta scan bit-exact over 37 rows). Speed, tier mode, engine
  clock, 2 repeats, a timing job and not yet the speed gate: 1k 524 to 531 tok/s, 8k 428 to 432,
  32k 246 to 247, against replay at about 67. Long context: one served chat request of 75,052
  tokens through `baro-serve` in tier mode answered a needle at 0.95 of the prompt, first token
  after 651 s (`bench/moe-prefill-needle.sh`, PASS). Report `exchange/lane-MOEPF-report.md`.
  - **Tier mode streams one layer of experts per chunk** into two VRAM staging slots, the next
    layer copying on a second `DeviceStream` while the current one computes (`BARO_PF_OVERLAP`,
    default on, +1.04 GB VRAM). The 64-slot expert cache is bypassed and left untouched.
  - **WMMA attention and the dense chunk scan in MoE prefill: DEFAULT-OFF, FAIL identity.**
    `BARO_PF_ATT=wmma` reads 97.92% teacher-forced agreement against a 99% bar
    (`bench/moe-prefill-agree.sh`, control arm 64/64); `BARO_PF_SSM=chunk` flips a greedy token
    at index 4. Both reorder sums. Exact attention is the long-context limiter (32k: 133 s vs
    84 s with WMMA).
- **Resident MoE pack on a 24 GB card: PARTIAL, short context only.** MAX's memory pool is about
  90% of free VRAM; the 21 GB pack leaves 0.08 to 0.4 GB, ran out of memory at an 8k f32 KV cache
  and ran slower than tier mode at 1k. Tier mode (`BARO_TIER=64 BARO_TIER_PINNED=1
  BARO_TIER_ZC=1`) is the MoE configuration past 1k context. Check: `pack-fit` (iTools).
- **MoE prompt-lookup (ngram) speculation: WORKS, DEFAULT-OFF, slower.** `BARO_NGRAM=1`
  (`ad327e8`, `9cbb683`): identity 20/20, acceptance 37.7%, 76.8 tok/s vs 101.1 spec-off, 0.76x.
- **`BARO_SEQ_CAP` build flag: WORKS** (`92784e9`), the KV sequence count; `-D BARO_SEQ_CAP=1`
  is what lets one 131072-token sequence fit.
- **R6.3, dense-addressed q8 dot for the MoE projections (`d684fb2`): PRESENT, UNMEASURED.** The
  launch path uses it and passes the prefill identity gate above; the preregistered persistent-
  kernel A/B (`0931e90`) has no numbers. Round killed 2026-10-12 (`bench/moe-persist-protocol.md`): the A/B needs the resident MoE pack, refused on this box since the 10-12 crash.

Serving (all from the 2026-09-18 board entries; receipts lost unless a test suite is named)
- **`POST /v1/audio/speech` (TTS): WORKS**, WAV through a configurable runner, Piper verified.
- **PWA browser voice into `/v1/audio/transcriptions`: WORKS** against the whisper sidecar.
- **Streaming tool-call deltas: WORKS**, OpenAI-shaped name and argument fragments from the
  `<tool_call>` grammar; newline-after-tag fix `6b21092`.
- **`hidden: true` and `logits_topk: K` on completions: WORKS**, JSON and SSE.
- **LAT1 state HMAC (`BARO_STATE_HMAC_KEY`): WORKS**, unsigned or wrong-key imports rejected.
- **Ollama `POST /api/pull`: WORKS** for the loaded pack, 404 otherwise (Rust tests 81 + 11).
- **Router prefix affinity with cold-miss locking: WORKS** (router unit tests 13, CPU HTTP gate).
- **Spark JSON schema enforcement: IMPLEMENTED, live gate UNVERIFIED.**
- **Chat template file overrides (`8978f10`), request bodies up to 64 MiB (`ff8e523`): WORKS**
  by their commits; no separate gate named.
- **`tools/baro eval`, `eval-all`, `multi-serve --devices`: WORKS as CLI wiring**; the two-device
  token identity gate on real hardware is UNVERIFIED.
- **LatentOS agent `--daemon` loop: WORKS** (build and liveness gate); the vendored
  `latentos/agent.mojo` carries this block on top of upstream and `ci-checks` allows for it.
- **Settings template, all 62 `BARO_*` settings, with `tools/settings-check.py`: WORKS** as a
  ci check (`d163eb5`, `0108788`).

Verification
- `bench/preflight.sh` (CPU) before any gate; gates take `EXPLORE=1` for runs that may not
  claim (verdict `UNVERIFIED`, exit 3). `./run-tests.sh`: rc=0 on `ea57b60`'s tree under
  `gpu-wait`, 2026-09-19, including the two new parity tests.
  Builds go through `tools/build-many.sh` (since `10bf32f`): 4 jobs x 4 threads, skipped when a
  content key (every `*.mojo`/`shim/` file, Mojo version, command) matches a prior success. Cold
  128.7 s (serial was 170-310 s), unchanged tree about 21 s; `BUILD_CACHE=0`, `JOBS=N` override.
- **`tools/baro --profile NAME`: WORKS as wiring** (`0a44cd4`): `profiles/*.toml` checked against the
  settings template, read by `tools/profile.mojo`, which matches its Python oracle on the three seed
  profiles; its TOML reader is DataBooth/mojo-toml vendored as `toml/` (Apache-2.0), so a clone
  builds it. `--target-accelerator gfx1100` is pinned
  in `tools/baro`, `bench/preflight.sh`, `bench/quality-run.sh` and `ci-checks` (`88c27cf`).


## How to read this document

A one-line-per-capability map of everything below, with the status words and the
switches, is the "What it can do" section of the [README](../README.md).

This is a capability reference, not a plan and not a sales sheet. The rule it is
written under: a capability appears here with the check that proves it, or it
appears here marked as unproven. Nothing is listed because it is intended,
designed, or nearly finished.

Status words mean exactly one thing each:

- **WORKS** or **PROVEN**: a named check passed. The check and its receipt path
  are cited on the same line. You can rely on it.
- **PARTIAL**: works inside a stated limit. The limit is part of the capability
  and is written into the same line, never as a footnote. Outside that limit,
  treat it as unproven.
- **DEFAULT-OFF**: implemented but not on. The environment variable and its
  default are named, and so is the reason it is off. Several of these measured
  at or below their own preregistered kill lines, which is why they ship off.
- **PRESENT BUT UNUSED**: code exists and may even be registered, but no
  execution path calls it. Not a capability. Listed so nobody rediscovers it
  and assumes otherwise.
- **CLAIMED**: asserted somewhere in prose with no check behind it. Treat as a
  lead, not a fact.
- **FAILS**, **PARKED**, **KILLED**, **BROKEN**: what is wrong, the number that
  says so, and what to use instead where something else works.

Failed and parked items are kept in this document deliberately. They are as much
a part of what mojo-baro is today as the passing ones, and a reader who came away
believing a failed feature works because the document was tactful would be worse
off than one who read nothing. Where a number was retracted or called provisional
by the repo's own docs, that label is carried through verbatim rather than
quietly upgraded.

Three conventions inherited from the repo's rules are worth restating, because
they explain why some entries look pedantic:

1. A green build, a passing unit test, an HTTP 200, and an active systemd unit
   are not evidence that a feature works. The check has to exercise the artifact
   the way a user meets it.
2. A reference implementation is not a reference. Vendor and baseline arms get
   the same cold-cache, clock and parameter read-back discipline as ours, and
   several gates in this repo were found unpassable by their own baseline.
3. Passing a flag is not evidence it took effect. Arm-defining parameters are
   read back from the running system, and a few switches in this codebase are
   silently inert under conditions named below.

## What mojo-baro is

A from-scratch GPU inference stack for AMD gfx1100 (Radeon RX 7900 XTX), written
in Mojo, with a Rust serving layer around it. It is not a llama.cpp wrapper and
not a PyTorch deployment: the kernels, the attention and SSM paths, the sampler,
the pack format and the bake pipeline are all its own, and llama.cpp appears
throughout only as the differential oracle that its correctness gates are
measured against.

## Engine and kernels

Read at HEAD `2208bdb` (2026-09-17). Sources: `docs/BASELINE.md`,
`docs/KERNELS.md` (auto-generated by `tools/kernel-census.mojo`),
`serve/engine.mojo`, `serve/spark.mojo`, `serve/model.mojo`,
`serve/model_qwen35.mojo`, `serve/model_qwen35moe.mojo`,
`serve/registry.mojo`, `serve/window.mojo`, `serve/expert_tier.mojo`,
`kernels/*.mojo`, `docs/ternary-quant-notes.md`, `docs/mtp-notes.md`.

### There are two engines, not one (three binaries since qwen4exp)

`serve/engine.mojo` is not a multi-architecture engine. It compiles into
exactly one of two hardcoded comptime profiles, selected by `BARO_MODEL`
in `serve/model.mojo`:

- `qwen35` (dense): `serve/model_qwen35.mojo`. H=4096, 32 layers, 24 gated-
  delta-net (SSM) layers plus 8 full-attention layers, `MEGA_ALLOWED=True`.
  Has the MTP draft head (`blk.32`, NextN). Runs Qwythos-9B-Claude-Mythos.
  Shape comes from the GGUF as `-D BARO_*` flags (`tools/model-id.py`), 9B
  defaults. **OrcaSAQ-2-27B (IQ4_XS GGUF, packed --q4, 17.2 GB): WORKS,
  UNVERIFIED vs llama.cpp** (2026-10-17): H 5120, 64 layers (48 SSM + 16
  attention), NH_V 48, no YaRN, coherent greedy text. 13-token prompt, 64
  tokens, no spec: 44.3 tok/s gen with the mega persistent kernel (default;
  needs ATT <= INNER, `MEGA_FITS`), 39.7 without, llama-bench tg64 41.3 in
  the same stint. Mega and launch path emit identical tokens. No MTP spec,
  no agreement gate vs llama.cpp yet.
- `qwen35moe` (MoE): `serve/model_qwen35moe.mojo`. H=2048, 40 layers, 30
  SSM plus 10 attention, `MEGA_ALLOWED=False`, plain RoPE (the GGUF
  declares no rope scaling, so FREQ_SCALE=1.0, not YaRN). Sparse MoE FFN
  via `kernels/moe.mojo`, top-8 router. Runs RegesCore-1.0-35B.

Both profiles are variants of one architecture family, the Qwen3.5 hybrid
SSM/attention design. There is no registration mechanism for a third
family; adding one means writing a third `model_*.mojo` and wiring it into
`serve/model.mojo` by hand.

`serve/spark.mojo` is a second, separate binary: a generic dense GQA
transformer with no SSM layers, driven by a `profile.mojo` generated per
model by the model-import tool (materializes under `.work/dense/*/profile.mojo`,
`.work/sample-spark/*/profile.mojo`). This is the path that runs imported
non-hybrid models: Llama-3.2-1B-Instruct, Qwen2.5-7B/Coder-7B-Instruct,
lily-cybersecurity-7b (profile `spark2_5`, has QKV bias and a gate that the
other spark models do not). `bench/dense-run.sh` also builds
`serve/spark.mojo` under the confusing name "dense-run".

`serve/q4x.mojo` (2026-10-09) is a third binary, for `qwen4exp`
(Qwen3.8-Flash-Next). Hyper-connections replace the residual at every layer, so
it is a separate forward rather than a third `model_*.mojo` profile: GEMVs on the
GPU through `kernels/q4x_kernels.mojo` (one kernel per ggml type, over the P6 row
dots), and since 2026-10-10 (lane q4x-r1) the small ops too: the residual stream
lives in VRAM and the host keeps only the router top-10 (one 512-float download
per layer), expert slot staging and the sparse indexer selection past 2051 tokens
(`exchange/q4x-r1-report.md`). Weights from the mmap'd shards via
`serve/shard_index.mojo`. Status WORKS (v0): 20-prompt teacher-forced agreement
99.06% vs control 98.91% on the GSQ-RCO Q2_0 pack (`bench/q4x-gate.sh`), 17
ms/token median in the gate run (119 before lane q4x-r1, 37.5 after it, 24 after q4x-r3a:
expert misses DMA from the hipHostRegister'd shard mmap, warm list `Q4X_WARM`,
`exchange/lane-Q4X-R3A-report.md`; 23 after q4x-r2: fused launches, device router with a
coherent doorbell, `exchange/lane-Q4X-R2-report.md`; 18 after q4x-r3b: Q2_0 unpack vectorized and the delta
step in four groups, `exchange/lane-Q4X-R3B-report.md`; 17 after q4x-r3c: two-rows-per-wave K-quant lanes,
`exchange/lane-Q4X-R3C-report.md`; kernel busy is 12.7 of the 17). Kernel parity:
`kernels/test_q4x_hostops.mojo`. Batched prefill (2026-10-12, lane latent-flash P1, `exchange/lane-latent-flash-p1-report.md`):
`Q4X_PREFILL=1` runs the prompt as one layer-major forward (`Q4X_SEQ=N` for several sequences per batch), decode is the
same forward with one token; one GEMM kernel `amar_q4x_gemm[GT, R, KQ2]` over a tile table gives R tokens per weight load
with the m=1 per-element order, so the final-norm hidden of every position is byte-identical to `step()`
(`bench/q4x-prefill-gate.sh`, PASS 20/20) and decode's bits did not move. `Q4X_HIDDEN=path` appends the hiddens (bf16
rows, f32 with `Q4X_HIDDEN_F32=1`). Speed prefill 4.58 ms/token at T=2048 (prereg 2.8) and 5.23 at T=8192 as 4 x 2048 (prereg 0.7), decode 23.4 / 34.4 ms/token in the same jobs; expert bytes 11.7 / 3.5 MB/token (`bench/q4x-prefill-ab.sh`, same job, 20-prompt median). Long context: dense attention is the default past 2051 tokens
and passes the 2766-token gate at 100%; the sparse indexer is opt-in (`Q4X_SPARSE=1`, 87.5% on
the same gate: the block cut moves with f32 drift, kernel verified, report section). Not served over HTTP; CLI only (`q4x SHARD1 N_GEN TOK...`). Resident mode (2026-10-14, `bench/q4x-serve-protocol.md`): `Q4X_STDIN=1` serves one request per stdin line (`TAG N_GEN TOK...`, answer `out TAG ids ...`), state reset between requests and the expert cache kept: 21 requests in 39-48 s wall for one process against 6.7-7.5 s per request as one process each, steady-state misses 28 per token against 90 cold, ids equal to the argv run (`bench/q4x-serve-check.sh`, I1/I2 PASS). Default warm list `FIRST_SHARD.warm` (here a link to `exchange/receipts/q4x-strata/r3a-warm-l01.txt`, `Q4X_WARM=0` off): first-request misses 90.6 -> 33.5 per token. Expert cache margin 0.5 GB (was 1.5): +6.6% slots, steady misses -17%, gen ms/token order-confounded (first resident arm of a job runs slow; second pair 0.985, 17 of 19 prompts faster), GTT flat. `Q4X_CHUNK=N` runs a single-sequence prefill as N-token forwards, byte-identical but 1.5x slower at 8192 as 4 x 2048 (5.98 vs 3.87 ms/token: each chunk refetches its experts), so opt-in for a Q4X_TMAX whose batch buffers do not fit; one 8192-token forward runs 3.87 ms/token, so the 60 ms/token cliff at 8 x 1024 is a multi-sequence effect. Latent input steps
(2026-10-12, latent-flash E8 retry): `Q4X_VEC=rows.f32 Q4X_VEC_IDS=id,...` feeds f32 vectors in place of the token_embd
row after the prompt (the id still enters the PLE n-gram hash); fed the token's own row it is byte-identical to `step()`
(`bench/q4x-vec-gate.sh`, PASS 3/3 with a falsifier). `Q4X_SOFT=m` runs m soft steps, softmax(top-K logits) . token_embd
rows through the engine's dequant, rescaled to `Q4X_SOFT_NORM`; K defaults to 1024 because top-64 missed cos 0.999 on
1 of 20 prompts (`bench/q4x-soft-check.sh`, PASS 20/20 at K=1024 vs latent-wa's full-vocabulary soft). A K=1024 step
costs about 0.9 s (host top-K and sum, unoptimized).
VRAM margin (2026-10-18, lane open3): the expert cache takes `vram_free() - Q4X_ECACHE_MARGIN_MB` (default 2048; the constant was 512 MiB). With 0.5 GiB free next to the desktop the kernel moved the engine's 3.4 and 15 GB buffers to GTT: direct reclaim, prefill 75 s to as much as 582 s at 16K, decode 21 to 285 ms/token in a spike run. Interleaved A/B at a 2,048-id prompt: margin 512 spiked in 6 of 6 runs, the default in 0 of 6; the default costs about 7% decode and 17% more expert misses (`exchange/lane-open3-report.md`, `bench/q4x-gtt-trace.sh`). `Q4X_ECACHE_MARGIN_MB=512` restores the old cache size.

**HTTP, the 16K floor and long-context decode (2026-10-18, lane q4x-serve16k, `exchange/lane-q4x-serve16k-report.md`).**
`BARO_SERVE=1 BARO_PACK=FIRST_SHARD` makes q4x speak the baro-serve line protocol (`serve/PROTOCOL.md`): `ready` with `tmax=Q4X_TMAX`, `{id,tok}` per
greedy token, `{id,done,...}` after every diagnostic line (baro-serve reads engine stdout only while a request is in flight), `stop` sequences,
`{cancel:ID}` polled per token; sampling, penalties, logprobs, hidden, embed and state_save/load answer an error line (greedy only). Launch:
`baro-serve --engine .work/q4x --pack SHARD1 --tokenizer tokenizer.json --engine-env Q4X_PREFILL=1 --engine-env Q4X_TMAX=16384`
(`tokenizer.json` from `gguf-tokenizer-json SHARD1 DIR`). Gate `bench/q4x-serve-check.sh MODE=http`: curl `/v1/completions` streaming at
`max_tokens` 16000, ids equal the `Q4X_STDIN` ids (p01 511, p02 16000), streamed text equals `/detokenize` of the ids, `POST /v1/cancel` stops the
engine at 148 to 155 ids, `/health` tmax read back 16384, GTT flat 0.45 GB. 16K floor: a one-forward 16384-token prefill at `Q4X_TMAX=16640` peaks at
23.5 GB VRAM, spilled 12.3 and 16.4 GB to GTT in the 2 runs that finished and the third faulted (GPU memory access fault), so 16K runs use `Q4X_CHUNK=8192` (VRAM peak 23.41 to 23.46 GB);
teacher-forced agreement at a 16384-token prompt, 2 prompts x 32 ids (`bench/q4x-gate.sh`, `bench/q4x-prompts-16k`): OURS 96.88% vs CTRL 96.88%, self-forced
control 32/32 (prereg `exchange/q4x-serve16k-prereg-item2.md`); transient GTT peaks of 15.2 and 4.2 GB in 2 of 6 chunked runs, no throughput change, cause open.
Decode attention was the 16K cost: `amar_q4x_attn` 54.7 of 87 to 94 ms/token (rocprofv3, `bench/q4x-trace.sh`); three byte-exact kernels
(`amar_q4x_attn_sc`, `_sm`, `_out`, from `Q4X_ATTSPLIT` tokens, default 512, 0 = old path) take 16K-context decode from 77.9 to 27.4 ms/token
(`bench/q4x-ab.sh`, same job A B B A, ids equal), and a 16000-token HTTP generation from 48.1 to 20.2 ms/token mean; short context is unchanged
(20 prompts, 1.0017). Short-context gap split under the profiler (20 prompts): busy 10.9 of 23.4 ms/token, launch-floor gaps 4.2 ms (1213), host waits
5.8 ms (2.4 after the router), memcopies 3.0 ms. Prefill at 16384 tokens: 5.1 to 8.4 ms/token as 2 x 8192.

A reader who treats `engine.mojo` as one general engine will misplan any
work involving a third architecture, or any work on imported models,
because that work is entirely in the other binary.

### Quantization: end to end vs present-but-unused

Weight GEMM dispatch in `serve/engine.mojo`/`serve/spark.mojo` is a hard
binary switch (`serve/window.mojo:gemm_w`): q4 (block-32 int4 plus fp16
scale) or q8 (block-32 int8 plus fp16 scale). Nothing else reaches the
dense engine's forward pass as a weight format. **The spark engine is the
exception since 2026-10-19 (main `d480700`):** `tools/engine-pack.py --dense`
writes Q4_K / Q6_K gguf blocks as they are (dt `q4k`, `q6k`, split QKV) and
`serve/spark.mojo` runs them through `amar_gemv_kq` with f32 activations;
q8 entries still take the q8 path. Numbers: `docs/BASELINE.md` "Spark engine:
native K-quant weights, f32 activations"; protocol `bench/sparkq4k-protocol.md`.

- **q4: WORKS.** Default pack (`BARO_PACK` default `.work/engine-pack-q4`,
  built by `tools/engine-pack.py --q4`). All current champion decode
  numbers are q4 (see performance section).
- **q8: WORKS, opt-in via pack choice.** Built by `tools/engine-pack.py
  MODEL.gguf OUTDIR --q8`, checked bit-equal to `llama-quantize Q8_0` by
  `tools/q8-check.py`. 68.8 tok/s_gen, 64/64 greedy identity vs llama.cpp
  Q8_0 and the bf16 reference (verified, `docs/BASELINE.md`).
- **bf16 weights: REMOVED.** `docs/BASELINE.md` states directly: "The bf16
  pack path (41.7 tok/s) is gone from the engine; its numbers stay in
  `bench/q8-protocol.md`." Activations are still carried as bf16 tiles;
  there is no bf16 weight path.
- **f16 weights: never existed in the engine.** f16 appears only as the
  block scale dtype for q4/q8 and in standalone WMMA GEMM microbenchmarks
  (`kernels/matmul_wmma_pipe.mojo`, `kernels/matmul_wmma_lds.mojo`) used
  only from `bench/bench_fp16*.mojo`. Neither engine calls them.
- **int8 MMQ prefill (dequant-in-kernel): PARTIAL.** `kernels/matmul_mmq.mojo`
  (`amar_matmul_lds_q4`, `amar_quant_q8`) has a parity test
  (`kernels/test_mmq.mojo`) and an ablation bench
  (`bench/bench_prefill_abl.mojo`, `bench/prefill-protocol.md` R5), but the
  default prefill dispatch in `serve/registry.mojo` is
  `amar_matmul_prefill_q4`/`_q8`/`amar_matmul_prefill_lds`, not this kernel.
- **int8 WMMA (for GEMM speed): closed, not adopted.** Measured to issue
  at the same rate as bf16 on gfx1100; MMQ lost to bf16 on the same
  schedule. Not proposed again per this repo's own memory of the result.
- **MoE expert weights: a separate axis, native GGUF K-quant.** MoE
  weights are not repacked into baro's own q4/q8; the pack is "Q4_K
  experts, Q8_0 projections, Q6_K head" (`docs/BASELINE.md`, MoE round 1),
  read by dedicated kernels `amar_moe_down_q4k`, `amar_moe_down_q4k_zc`
  (zero-copy), `amar_moe_down_q6k`, `amar_moe_gate_up_q4k`
  (`kernels/moe.mojo`).
- **KV cache dtype (`BARO_KVQ`, comptime `-D` flag, `kernels/attn.mojo`):**
  - `f32`: default, WORKS.
  - `bf16`: PRESENT BUT UNUSED as a working path. Fails the 20-prompt
    identity gate; recorded in this repo's memory as a known-bad arm.
  - `int8`: DEFAULT-OFF, dense-only. Build with `-D BARO_KVQ=int8`;
    `serve/engine.mojo` raises at comptime if `IS_MOE and KVT !=
    DType.float32` ("the MoE megakernel writes f32 KV"). Off by default
    because it is a build-time opt-in, not a runtime default. Measured:
    decode after 32k context 117.69 tok/s vs 100.54 f32 (1.171x), short
    context 1.011x, RULER niah_single 100.0 at 64k and 128k
    (`docs/BASELINE.md`, "int8 KV since 2026-09-17", commit `80d4b2c`,
    verified). `BARO_STATE_SAVE` refuses to checkpoint when KV is int8,
    an unimplemented case, not a policy off-switch.
- **Ternary (Q2_B3, TQ1_0, TQ2_0): PRESENT BUT UNUSED.**
  `kernels/matmul_ternary.mojo` implements all three formats.
  `serve/registry.mojo` defines wrapper functions `gemm_q2b3`, `gemm_tq1`,
  `gemm_tq2` for them. Neither `serve/engine.mojo` nor `serve/spark.mojo`
  calls any of the three wrappers; a search for those names in both files
  returns nothing. `docs/ternary-quant-notes.md` states this directly:
  "Correctness only; no throughput has been measured for any of these
  kernels and none may be claimed without the `bench/coldcache-protocol.md`
  preregistration flow." The kernels are registered and parity-tested
  (`kernels/test_ternary_gemm.mojo`, `bench/bench_coldcache_ternary.mojo`),
  which is a correctness claim only, not a decode-path claim.

### Kernel inventory

From `docs/KERNELS.md` (auto-generated, kept in sync by policy: every
`amar_*` kernel must be reachable from `serve/registry.mojo`, a bench, or
a test).

| category | kernel(s) | file |
|---|---|---|
| GEMM, dense fp32 (naive/tiled/register-tile/double-buffered) | `amar_matmul_naive`, `_tiled`, `_regtile`, `_dbuf`, `_ldst`, `_vec4` | `matmul.mojo`, `matmul_dbuf.mojo`, `matmul_ldst.mojo`, `matmul_vec4.mojo` |
| GEMM fp16 WMMA, benchmark-only | `amar_matmul_wmma_lds`, `amar_matmul_wmma_pipe` | `matmul_wmma_lds.mojo`, `matmul_wmma_pipe.mojo` |
| GEMV/skinny weight-native, decode (q4/q8) | `amar_matmul_skinny*` (q4row/q4rowb, q8row/q8b/q8dot, m1/m1_row/m1_row2/v2), `amar_skinny_reduce*` | `matmul_skinny.mojo` |
| GEMM prefill (m>1), q4/q8 plus LDS variant | `amar_matmul_prefill_q4/q8`, `amar_matmul_prefill_lds`, `amar_prefill_swiglu_bf16` | `matmul_prefill.mojo`, `matmul_prefill_lds.mojo` |
| int8 MMQ, dequant-in-kernel prefill | `amar_matmul_lds_q4`, `amar_quant_q8` | `matmul_mmq.mojo` |
| Ternary GEMV | `amar_matmul_skinny_q2b3row/tq1row/tq2row` | `matmul_ternary.mojo` |
| Attention, decode/prefill | `amar_attn_decode`, `amar_attn_prefill`, `amar_attn_prefill_wmma`, `amar_kv_append(2)`, `amar_head_rmsnorm(_rope)`, `amar_rope_yarn`, `amar_qgate_split`, `amar_gate_mul(_cast)` | `attn.mojo` |
| Attention, generic decode, split-K ("dattn") | `amar_dattn_exact`, `amar_dattn_split`, `amar_dattn_combine` | `dattn.mojo` |
| Attention, Spark family (SWA plus gate) | `amar_attn_decode_swa_gated`, `amar_rope_kv_append`, `amar_rope_plain`, `amar_gemv_q8`, `amar_argmax_part/final`, `amar_bias_add`, `amar_embed_lookup_f32` | `spark_kernels.mojo` |
| SSM / gated-delta-net scan | `amar_ssm_conv(_chunk)`, `amar_ssm_delta_step`, `amar_ssm_delta_chunk(_w)`, `amar_ssm_gates(_rows)`, `amar_ssm_gated_out(_bf16/_rows_bf16)`, `amar_ssm_qk_l2norm(_rows)`, `amar_ssm_reduce_gates` | `ssm.mojo` |
| Elementwise, norm, embed, dequant | `amar_rmsnorm(_cast/_cast2)`, `amar_embed_lookup(_pos)`, `amar_quantize_q8_rows`, `amar_swiglu`, `amar_softmax_rows`, `amar_rope_rows`, `amar_tok_copy/remap`, `amar_argmax_pos/row`, `amar_cast_bf16`, `amar_widen_bf16`, `amar_residual_add` | `elementwise.mojo`, `ssm.mojo` |
| Sampler | `amar_sample_row(_masked)`, `amar_sample_probs(_masked)`, `amar_topn_probs`, `amar_apply_penalties`, `amar_spec_accept` | `sample.mojo` |
| MoE (router, gate/up, down, zero-copy) | `amar_moe_router_top8(_sig)`, `amar_moe_gate_up(_q4k)`, `amar_moe_sig_gate`, `amar_moe_down(_q4k/_q4k_zc/_q6k)` | `moe.mojo` |
| Megakernel, persistent, one launch per token (dense) | `amar_mega_token`, `amar_mega_window` | `mega.mojo` |
| MoE megakernel, persistent, opt-in | `amar_mega_moe_token` | `mega_moe.mojo` |
| Realign (embedding gather/reduce) | `amar_realign_gather`, `amar_realign_reduce` | `realign_kernels.mojo` |
| FFI shim binding to hipBLASLt | (C ABI wrapper, no `amar_*` name) | `amarbaro.mojo` |

### Decode and prefill features

- **Speculative decode / MTP: WORKS on dense, unconditionally off on MoE.**
  `BARO_SPEC` defaults to `"1"`, but the actual gate in `serve/engine.mojo`
  is `spec_env = MEGA_ALLOWED and getenv("BARO_SPEC","1")=="1"`, and
  `MEGA_ALLOWED` is `False` for `qwen35moe`. Setting `BARO_SPEC=1` on the
  MoE profile has no effect: it has no draft head at all. Draft head is
  `blk.32` (NextN). Window cap `KMAX = SM = 8` (`kernels/matmul_skinny.mojo`);
  default k=2 (`BARO_SPEC_K` / `spec-k.txt`). Verified, 20-prompt median,
  k=2, dense q4: 100.7 tok/s vs llama.cpp Q8_0 123.5 (0.78x); a k=4
  race-prompt figure of 145.6 vs 109.8 (1.33x) is explicitly called "an
  instrument receipt, never a verdict" in `docs/BASELINE.md` (P4), and an
  earlier headline of 127.96 tok/s (1.89x) measured on that same race
  prompt is superseded. Sampling composes with speculation since commit
  `b3c0d90`: T=0.7/top_p 0.9 spec-on median 147.15 tok/s_gen vs spec-off
  109.19 (verified, `bench/spec-sample-protocol.md`).
- **Megakernel: DEFAULT-OFF on MoE, on by default on dense.** `BARO_MEGA`
  defaults to `"1"` if `MEGA_ALLOWED` else `"0"`. On dense, m=1 decode
  without speculation runs as one persistent launch
  (`kernels/mega.mojo::amar_mega_token`); verified receipt (2026-09-06):
  67.13 to 81.98 tok/s_gen (+22%). On MoE, `BARO_MEGA=1` runs an analogous
  persistent kernel (`kernels/mega_moe.mojo`, rounds R6/R6.1) that measured
  **at or below its own preregistered kill line**: R6 A/B measured 110.90
  to 113.94 tok/s_gen, a ratio of 1.027, against a preregistered +5% kill
  line, so it stays off by default (`BARO_MEGA=0` on MoE). R6.1's speed is
  stated as "not measured yet" in `docs/BASELINE.md`.
- **`BARO_MEGA_WIN` (megakernel on the speculative window): DEFAULT-OFF.**
  Default `"0"`. Measured worse than the launch path it would replace,
  0.761x (115.39 vs 151.68 tok/s_gen on k=2 spec, dense q4), so it stays
  off.
- **`BARO_DOT` (int8-dot FFN path): DEFAULT-OFF.** Default `"0"`, and the
  code further gates it `and not pack_q4`, so it can only ever apply on
  the q8 pack. This repo's own memory records the k=2 case as closed, "no
  signal", not to be re-proposed.
- **Chunked delta (SSM): WORKS**, part of the dense q4 m=1 kernel
  (`amar_ssm_delta_chunk(_w)`, `RELOAD=True`, 239 VGPRs, 0 scratch),
  landed with the 133.9 tok/s_gen round, commit `9e6feaa`, verified.
- **rmsnorm fold: WORKS**, per-phase rmsnorm and quantize folded into the
  consuming GEMV's LDS prologue (`stage_rms`), removing 65 barriers per
  token, same `9e6feaa` round, verified.
- **Zero-copy expert tier: DEFAULT-OFF, requires pinning.**
  `BARO_TIER_ZC` defaults `"0"` and only takes effect if
  `BARO_TIER_PINNED=1` (that one defaults `"1"`, `serve/expert_tier.mojo`).
  When on, a missed q4_k expert is read directly from pinned host memory
  by the gate/up and down kernels instead of a DMA copy. Verified: 67.71
  to 72.02 tok/s_gen (1.064x), identity 20/20. The three q6_k down layers
  still DMA; not zero-copy. Off by default because it is a newer, opt-in
  path layered on top of pinning.
- **KV cache: paged, block table.** `serve/kvpage.mojo` (`PageTable`),
  128-token pages, since commit `09a8cd9`. Dtype per `BARO_KVQ` (see
  quantization section). `BARO_KVTAB=identity|reverse` selects the
  page-table indirection arm used as a gate falsifier, not a user-facing
  setting.
- **Max context / TMAX: PARTIAL, runtime override of a small compile-time
  default.** Comptime default `TMAX = 1088` (`serve/registry.mojo`,
  `serve/moe_pack.mojo`), but `serve/engine.mojo` reads `BARO_TMAX` at
  runtime and sizes the KV pool (`alloc_bufs(ctx, pack, tmax)`) from that
  value, not the comptime constant. 32k decode is verified: dattn-merge
  numbers show decode after 8k context 112.70 to 124.48 tok/s_gen, after
  32k 85.22 to 101.36. The int8 KV round measured RULER niah_single 100.0
  at 64k and 128k, so 128k context has been run, but only with `BARO_TMAX`
  raised above the 1088 default and with the VRAM to back it; it is not
  what a default-configuration run gets.
- **Attention split-K / dattn: PARTIAL, threshold-gated.**
  `amar_dattn_split`/`amar_dattn_exact`/`amar_dattn_combine`
  (`kernels/dattn.mojo`) engage only above `BARO_ATT_SPLIT_T` cached
  tokens (default equal to `TMAX`, so 1088 unless `BARO_TMAX` is also
  raised), or unconditionally if `BARO_ATT_SPLIT=1`. `docs/BASELINE.md` is
  explicit this path is not the source of the 136.37 tok/s_gen champion
  number: "the new decode attention runs only above T = 1088."
  Merged as part of `3824e20` (lane-dattn).
  - **32k/128k support in one line: WORKS, but not out of the box.**
    The paths above compose into working 32k/128k decode, verified as
    cited, only when `BARO_TMAX` is raised at build/run time; the
    committed defaults (TMAX=1088, split threshold=TMAX) are sized for
    short-context decode speed, not long context.
- **Batched MoE prefill: WORKS, DEFAULT-OFF.** See "Changes since 2026-09-17" at the top.
- **WMMA prefill attention: PARTIAL, not merged into the main tree.**
  `amar_attn_prefill_wmma` (`attn.mojo`) is registered (`attpw_k`) and used
  in the prefill-long-context lane (`docs/prefill-long-ctx-2026-09-11.md`,
  WMMA attention plus one-wave SSM scan), but `docs/BASELINE.md` states
  this lane is "not merged" for the 17.4s/32k prefill figure it reports.

### Performance numbers, with source label carried verbatim

Figures below are quoted from `docs/BASELINE.md` unless a different source
is named. `docs/BASELINE.md` states its own freshness bound: "Last
verified: 2026-09-15 for the qwen35moe decode row..., 2026-09-12 for the
long-context prefill rows..., and 2026-09-02 for the fp16 WMMA rows....
Rows not named here were not re-measured on those dates."

**Dense qwen35, q4, no-spec, 20-prompt median, VERIFIED:**
- 136.37 tok/s_gen, commit `3824e20` (2026-09-11, merge of lane-dattn),
  identity 20/20. Current champion.
- 133.9 tok/s_gen, commit `9e6feaa` (2026-09-08), identity 20/20.
  Superseded by the row above, not retracted.

**q8 arm, VERIFIED:** 68.8 tok/s_gen, 64/64 greedy identity vs llama.cpp
Q8_0 and the bf16 reference; llama.cpp Q8_0 no-spec bar 74.1.

**Speculative decode (dense, k=2, 20 real prompts, median), VERIFIED:**
100.7 tok/s_gen vs llama.cpp Q8_0 123.5 (0.78x, behind). k=4 race-prompt
figure 145.6 vs 109.8 (1.33x) is explicitly NOT A VERDICT per
`docs/BASELINE.md` P4 ("a single-prompt speculative number is an
instrument receipt, never a verdict"). A prior race-prompt figure of
127.96 tok/s_gen (1.89x, acceptance 50/53) is RETRACTED (superseded, its
tail inflates acceptance to about 94%).

**Sampled plus speculative composition, VERIFIED, one stint, 20-prompt
medians:** T=0.7/top_p 0.9 spec-on 147.15, spec-off 109.19; T=0 spec-on
150.24, spec-off 134.97.

**Megakernel sampled decode, VERIFIED, commit `54fb4bb` (2026-09-16):**
no-spec T=0.7/top_p 0.9 132.30 tok/s_gen vs no-spec T=0 134.65 in the same
stint (98.3%), was 108.32 before this round.

**MoE, RegesCore-35B:**
- Round 1, VERIFIED (2026-09-15): 93.46 tok/s_gen, 20-prompt median (was
  42.88). llama.cpp 109.4 on the same GGUF.
- R6.0b champion, VERIFIED, commit `38ee0b7` (2026-09-15): 111.89
  tok/s_gen, 20-prompt median, spread 1.2%, bit-identical to the previous
  engine on 20/20 prompts. Up from 107.28 (R6.0, `1e270b5`) and 94.79 (R4,
  `7145b71`). Landed BELOW its own preregistered +5% kill line (measured
  ratio 1.043x same-stint for R6.0, kept only "on the maintainer's recorded
  override"). llama.cpp on the same GGUF: 109.92.
- Expert tier pinned, VERIFIED (2026-09-17): 67.12 tok/s_gen (experts in
  host RAM) vs 48.57 (page-cached). This baseline predates the R6.0b
  launch-fusion round; not directly comparable to 111.89 without checking
  which optimizations stack.
- Zero-copy misses, VERIFIED (2026-09-17), opt-in `BARO_TIER_ZC=1`: 67.71
  to 72.02 tok/s_gen (1.064x).
- R6 persistent MoE megakernel, VERIFIED but BELOW its +5% kill line:
  110.90 to 113.94 tok/s_gen (1.027x). Stays off by default.

**Long-context decode (dense, dattn merge, `3824e20`), VERIFIED:** after
8k context, 112.70 to 124.48 tok/s_gen; after 32k, 85.22 to 101.36.

**int8 KV, VERIFIED, commit `80d4b2c` (2026-09-17), opt-in:** decode after
32k context, 117.69 tok/s_gen vs 100.54 f32 (1.171x); short context
1.011x; RULER niah_single 100.0 at 64k and 128k.

**Kernel microbenchmarks, fp16 WMMA, VERIFIED (2026-09-02):**
- 512x512x512 fp32, 200 iterations: `amar_matmul_regtile` about 5130
  GFLOP/s vs hipBLASLt about 5200 (a tie, trading places by size).
- fp16 pipe kernel: 2048^3, 93841 GFLOP/s vs hipBLASLt 81698; 4096^3,
  97957 vs 89716 (+9.2%).
- Roofline: R = 125.4 TFLOP/s peak; the pipe kernel at 4096^3 runs 89000
  to 91000 GFLOP/s, 0.71 R at 2.6 GHz clock.

**RETRACTED or PROVISIONAL, do not cite as current without the label:**
- Regtile "about 2x faster than hipBLASLt": RETRACTED. `docs/BASELINE.md`:
  "An earlier version of this file recorded regtile as ~2x faster than
  hipBLASLt. That was measuring an untuned vendor call, not a fast
  kernel."
- D1 power cap result (290 to 402 W bought +3%): PROVISIONAL. All arms ran
  at an invalid problem size (4096^3, inside the Infinity Cache
  contamination zone); needs a re-run at 3584^3 before it can be cited.
- fp16 GEMM 0.791 R roofline claim: RETRACTED/superseded. It came from an
  invalid 4096^3 measurement (100.7 MB working set against a 96 MB
  Infinity Cache); the valid figure at 3584^3 is 0.842 R.
- Ternary kernel throughput: UNVERIFIED, in fact not measured at all.
  "No throughput has been measured for any of these kernels"
  (`docs/ternary-quant-notes.md`); do not report a ternary tok/s number
  because none exists.
## Serving, APIs and clients

### HTTP endpoints (`baro-serve`, current commit `a01ce69`/`41f60f8` on `main`)

- Serves an OpenAI-shaped text completion API (`POST /v1/completions`, `POST /v1/chat/completions`,
  streaming via SSE, minja chat templates, tool calls parsed from the model's `<tool_call>` wire
  format, JSON-schema `response_format` enforcement, logprobs). WORKS: P0a gates 1 to 4 PASS,
  including a real `ollama` Python client and a from-source PAIR build routing a request to it
  (`exchange/lane-P0A-report.md`). Streaming tool-call deltas now emit OpenAI-shaped function-name
  and argument fragments from the same `<tool_call>` wire format. Unit gate: `cargo test`.
- JSON-schema `response_format` enforcement now works on Spark profiles too. The shared grammar
  runtime masks the Spark sampler before truncation, tracks `enable_thinking`, and stops on a
  terminated schema document. Source gate: Spark Mojo build reaches the changed path, then is
  blocked here by the host's unknown GPU architecture.
- Serves an Ollama-shaped API (`GET /api/tags`, `/api/ps`, `/api/version`, `POST /api/show`,
  `/api/chat`, `/api/generate`, `/api/embeddings`). WORKS: P0a gates 1 to 3 PASS, including PAIR's
  engine-manager adopting `baro-serve` on port 11434 and routing a live request to it. PARTIAL:
  `repeat_penalty` and `num_ctx` are parsed so a real client does not 400, but neither is wired to
  anything on the engine side (no multiplicative penalty exists, and `num_ctx` is only reported via
  the ordinary exceed-context error, never clamped). `images` and `tool_calls` on the Ollama request
  side are accepted but unused. `POST /api/pull` confirms the loaded pack with Ollama-compatible NDJSON
  status frames, or returns 404 for a different model. It does not import or hot-swap packs.
- Serves embeddings (`POST /v1/embeddings`, `POST /api/embeddings`), last-token hidden state,
  L2-normalized on the engine. WORKS: gated against `llama-embedding --pooling last
  --embd-normalize 2`, cosine minimum 0.9882, retrieval 8/8 (`exchange/lane-P0AE-report.md` gate 2,
  after a gate 1 FAIL from a stale folded-head bug that has since been fixed). PARTIAL: refuses with
  501 on the `serve/spark.mojo` engine and on any `BARO_SEQS > 1` engine; those refusal code paths
  compile but were never exercised by a live request, left UNVERIFIED in that report.
- Completion routes accept Baro latent extensions, `hidden: true` streams one raw post-final-norm
  hidden row per generated token, and `logits_topk: K` streams raw pre-penalty top-k logits. WORKS
  on the dense/MoE engine through SSE and non-streaming responses, with `K` bounded to 20. Spark
  refuses these fields with 400. HTTP shape gate: `bench/p6-pwa-gate.sh`, latent receipt
  `.work/p14-latent-p6-gate-2/SUMMARY.txt`.
- Serves speech-in transcription (`POST /v1/audio/transcriptions`, multipart WAV, proxied to a
  whisper-server sidecar). WORKS: 20/20 fixtures transcribed byte-for-byte equal to `whisper-cli` on
  the same model and beam settings (`exchange/lane-P3A-report.md`).
- Serves speech-out (`POST /v1/audio/speech`) through a configurable WAV TTS runner. The default
  local runner is Piper, with `BARO_TTS_BIN`, `BARO_TTS_VOICE`, `BARO_TTS_OUT_DIR`, and
  `BARO_TTS_TIMEOUT_SECS` overrides. HTTP plumbing gate: `bench/p6-pwa-gate.sh`; Chatterbox
  quality and speech-in round-trip remain unverified.
- Serves named state checkpoints as first-class objects (`POST/GET /v1/checkpoints`,
  `GET/DELETE /v1/checkpoints/{id}`, `POST /v1/checkpoints/{id}/fork`): run a prompt once, save full
  engine state to a file, later fork branches that load it and prefill only the suffix. WORKS: this
  is the mechanism P0b's gate 4 (below) exercises live.
- Serves single-node LatentOS state export/import (`GET /v1/state`, `POST /v1/state/export`,
  `POST /v1/state/import`, P1). WORKS: round trip (both a raw file and a LAT1-wrapped stream)
  reproduces the same `prefix_hash` and restores in about 2 ms (`exchange/lane-P1-report.md`).
  int8-KV engines (the MoE default) export BAROST03 and import it (2026-10-12, one prompt, live check). NOT RUN: the
  plan's own 20-prompt, both-format, restore-band gate never ran; only single-prompt smokes exist.
  A real bug was found and fixed on the way: `Chain::lookup` always reserves the prompt's last
  token as the live decode seed, so an imported file's own `pos` needed replaying its last token
  once more before restore would clear that bar. Every state-file import returned 502 before the
  fix (`0df9230`).
- Does not serve fan-out (`POST /v1/fanout`, one prefill feeding N follower continuations). Designed
  in `docs/P1-STATE-API.md`; no route exists in `main.rs` or the router.
- Serves single-node prompt forking (`POST /v1/fork`, branch one shared prompt into N branches off
  the checkpoint chain). WORKS on one node. Cross-node forking via `target:"HOST:PORT"` exists as
  code (export on the source, stream to the target's `/v1/state/import`, relay the answer) but its
  identity gate FAILS; see "Cross-node fork" below.
- Serves a PWA client shell at `/` and `/web/{*path}` (embedded web server). See Clients below.
- Names its states on the wire (2026-11-10, `docs/state-machines.md`, `serve/PROTOCOL.md` section
  State words): `/health` carries `engine_state` and answers `503` while the pack loads; a dead engine
  in a pool is skipped; a dropped client cancels its request; a stream with no terminal event is `502`.
  WORKS in `cargo test` (135 + 23) and the state gates of `run-tests.sh`; the engine-side words are
  checked on the card by `bench/fsm-wire-gate.sh`.

### `baro-front` (`serve/front/`, the Mojo HTTP front, round 1)

- Serves `/health`, `/v1/models`, `/props`, `/tokenize`, `/detokenize`, `/v1/cancel` and the completion
  and chat routes (JSON and SSE) from one 1.14 MB Mojo binary, on the vendored `bento`, `aio`, `json`,
  `posix` and `systemd` libraries. Text (tokenizer, chat template, detokenizer) equals the Rust front
  on 3297 oracle rows. UNVERIFIED against the real engine: every test runs on a stub or a fake engine.
- Does not serve checkpoints, `/v1/turn`, fork, audio, the Ollama routes or embeddings. `baro-serve`
  stays the default front.

### `baro-router` (`serve/src/bin/router.rs`, a separate CPU-only binary, P0b)

- Discovers peers via mDNS (`_baro-node._tcp`) and answers PAIR's `_nvpair-node._tcp` browse so a
  PAIR scanner lists it. WORKS: a from-source PAIR build discovers the router's node and reads
  `/v1/node-info`, run twice, both exit 0 (`exchange/lane-P0B-report.md` gate 3). The fix that made
  this pass requires a human step that does not survive a reboot: the root cause was firewalld's
  `public` zone blocking inbound UDP 5353, opened with `sudo firewall-cmd --zone=public
  --add-service=mdns` without `--permanent`. It reverts on the next firewalld reload or reboot;
  whoever next relies on mDNS discovery on this box must reapply it.
- Ranks and places requests across registered engines (fewest pending requests wins, ties by id),
  and biases placement toward an engine holding a request's resident state (CONTRACT 4 locality,
  computed only for `/v1/completions` and `/api/generate`'s raw prompt; a `messages`-shaped chat
  request has no rendered prompt available to the router and always falls back to plain rank).
  WORKS, live, two real engines: gate 1 (placement) 20/20 via PAIR's own `pair-dispatch` client,
  balanced within one, every response identity-matched against a single-engine baseline at T=0;
  gate 4 (locality) a real `POST /v1/checkpoints` makes a prefix resident, the router picks it up
  within one 5 s probe cycle, and the matching request is placed `locality` not `rank`. PARTIAL,
  and this limit belongs to the capability itself: gates 1 and 2 ran two live 9B `baro-serve`
  processes on one XTX at `BARO_TMAX=4096` only, the sole context length at which two 9B engines fit
  in 24 GB (10.7 GB each). They prove nothing about two engines at 32k context on this box; that
  configuration does not fit at all.
- Proxies both API surfaces with streaming pass-through and fails over a request that never got a
  response byte to the next-ranked engine; a request past its first response byte is never retried,
  it fails loudly instead. WORKS: gate 2, a genuinely SIGKILLed engine's in-flight SSE stream fails
  loudly (curl exit 18, no terminal DONE), and the next request lands on the survivor.
- Uses deterministic rendezvous hashing for a prompt prefix with no resident checkpoint, and
  serializes concurrent cold requests for the same prefix behind one in-flight owner. WORKS,
  router unit tests cover stable placement and waiter release, and the CPU HTTP gate passes at
  `.work/p8-router-affinity-gate/SUMMARY.txt`. Zero-downtime process handoff remains absent.

### Multi-engine and multi-node

- Pools multiple engine processes inside one `baro-serve` (`BARO_POOL=N`, `EnginePool`), routing
  each request to the shortest queue. WORKS: this is the in-process pooling every gate above runs
  on; distinct from the router's cross-process placement.
- Pins a device explicitly per engine process (`--engine-env KEY=VALUE`, repeatable, `KEY=` unsets),
  so a `ROCR_VISIBLE_DEVICES`/`HSA_OVERRIDE_GFX_VERSION` pin is a logged `baro-serve` argument
  rather than only whatever launched it. WORKS as wiring: device read-back confirmed one engine
  pinned to the XTX (`GPU-859baafa301986cb`) and the other to the iGPU
  (`ROCR_VISIBLE_DEVICES=1`, `HSA_OVERRIDE_GFX_VERSION=10.3.0`).
- `tools/baro multi-serve MODEL.gguf --devices 0,1` prepares one cached engine
  and pack, launches one pinned `baro-serve` per listed device, and fronts them
  with `baro-router`. This is the production launch path. The existing two-
  device identity gate remains the hardware correctness bar.
- Does not yet run two GPUs in production. FAILS: the P4 multi-GPU gate hit its kill line. Two
  engines (XTX dense q4, iGPU Qwen2.5-7B under the gfx1030 override) split 6 round-robin prompts;
  `p06-translate` on the iGPU matched its solo run for 6 leading tokens and then diverged, evidence
  of request-state bleeding or a non-determinism defect in the engine/kernel path, not a
  device-pinning or queue failure (`exchange/lane-P4-report.md`). This result predates a relevant
  fix: the gate ran at main parent `c0e9297`, before `f9048f6` (`save_state` no longer lets one
  prompt's export carry another prompt's conv/SSM state, landed later on `main`). P4 is currently
  being re-investigated on branch `lane-p4` against the fixed engine; whether the original
  divergence was caused by that bug or is a separate defect is not yet known. Treat P4 as FAILS,
  re-run in flight, not as a closed result.
- Does not yet move state reliably between nodes. FAILS / UNCLASSIFIED: the cross-node
  `/v1/fork?target=` gate (`exchange/lane-FORK-report.md`) found the K/V-swapped-state falsifier
  only caught 3 of 5 deliberately wrong states (needed at least 4 to certify the ids gate at all),
  so the verdict is FAIL by its own frozen rule, f32 and int8 both reached 20/20 and 16/20
  respectively at every tested link rate (100 Mbit, 1 Gbit, 10 Gbit) but the ids alone cannot prove
  a state was really moved. The 32k timing half of this gate was NOT RUN: two 9B engines do not fit
  on one XTX at that context length. The llama.cpp bridge gate (our exported state resumed by a
  real `llama-server`) scored 16/13/14 of 20 across three models against llama.cpp's own ceiling
  restoring its own bytes (15/16/16, itself short of 20/20), so the verdict is UNCLASSIFIED rather
  than a clean numerics failure, left to the coordinator. A live two-node smoke (three prompts,
  loopback, not the gate) did move state end to end successfully and is explicitly labeled a smoke,
  not a gate pass. A real engine bug was found and fixed by the byte-level check this lane ran
  because the ids gate could not have caught it: `save_state` picked a checkpoint by position
  alone, so one prompt's export could silently carry a different prompt's conv/SSM state under a
  valid sha (fixed `f9048f6` on current `main`, reproduced 7/7 correct after the fix including
  deliberately swapped states). Separately, the forward bridge itself has no layout defect: a
  byte-identical round trip, a falsifier that does catch a swapped file at depth greater than 0,
  most prompts fully identical, and a smooth 1 to 5 percent elementwise K/V difference against
  llama.cpp's own all point the same way.

### Clients

- Android app (`~/Android/baro/`): remote chat client plus an on-device llama.cpp mode. WORKS as a
  remote client: gate 1 (clean build, unit tests) PASS; gate 2 (80 requests through the real client
  against curl) PASS byte-identical. FAILS on-device: gate 3 (5 fixed prompts, on-phone vs a
  llama.cpp desktop CPU reference) is 1/5 wrong after one repair round, the kill line for that gate.
  The one failing prompt diverges at a near-tie logprob (desktop top-2 within 0.02 to 0.08 of each
  other), read as cross-architecture q4_K numerics rather than a bridge bug, but that reading is not
  proven; no x86_64 on-device run exists to separate arch numerics from device code. PARTIAL: gate 4
  (UI, looked at on the phone) PASS with caveats, an automated audit flagged clickable elements as
  unlabeled, likely a Compose semantics false positive, but this was not confirmed with a real
  screen reader. Voice input end to end (record on phone, transcribe through P3a) is UNVERIFIED,
  only unit-tested against a mock. The coordinator has an open decision to make: ship the remote
  client only, or accept the on-device mode as 4/5 exact with the divergence documented; the app
  still ships both tabs as built.
- PWA client, served by `baro-serve` itself at `/`. WORKS on the tested surface: a CPU-only
  fake-engine browser gate PASSED, streamed text lands correctly in the DOM, no horizontal overflow
  at 400px width, offline reload serves a controlled shell with the server stopped, service worker
  registers. WORKS: browser PCM capture, WAV encoding, and `/v1/audio/transcriptions` pass through
  the PWA browser gate with a real whisper sidecar. The web assets are
  719 LOC against the item's own 600 LOC budget, disclosed and not fixed.
- ComfyUI custom node (`comfyui-baro`): chat and JSON-schema-constrained nodes calling `baro-serve`
  over HTTP, time-sliced against ComfyUI's own models. WORKS: VRAM receipt confirms the MAX runtime
  reserves about 22 GB regardless of model size, so no engine can be resident alongside ComfyUI's
  diffusion models, by design the node stops/starts the engine around each call; an end-to-end
  workflow produced an image; the JSON node's grammar gate passed 41/41; a SIGKILL orphan test
  confirmed the sidecar child dies with its parent via PR_SET_PDEATHSIG.
- aarch64 tooling (not the decode engine). WORKS: `baro-serve` cross-compiles to aarch64 and, on a
  real Arch Linux ARM QEMU guest, builds natively; Mojo 1.0.0 installs natively; `tools/ci-checks.sh`
  exits 0; tokenizer parity is 61/61 against `llama-tokenize` on both aarch64 and x86_64. There is no
  decode engine on aarch64: that needs an AMD GPU or an unbuilt CPU backend, so the aarch64
  deliverable is the router and CPU tooling only, and even the router itself was not tested on that
  guest, since it had not been built anywhere in the platform yet when this gate ran.

### Sidecars

- Whisper (speech-in) is a `whisper-server` child process started on demand and stopped after idle
  (default 300 s), driven as a plain HTTP peer on loopback. WORKS, see the audio transcription entry
  above. Two real orphan bugs were found and fixed: `baro-serve` had only a Ctrl-C handler, so a
  plain SIGTERM orphaned the whisper child and once leaked about 24.5 GB of VRAM until manually
  killed, fixed by adding a SIGTERM handler that also runs the sidecar's own shutdown; then
  PR_SET_PDEATHSIG hardening was added so the child dies within 2 seconds even under SIGKILL or an
  OOM kill of the parent, confirmed live.
- Chatterbox (speech-out, P3b) is not wired up. NOT RUN, see the audio speech entry above.
- LatentOS is three different mechanisms under one name, with three different statuses. Reading it
  as a single feature is the main way to get it wrong.
  - **memfd plus SCM_RIGHTS handle handoff, the sidecar proper: WORKS.** `mint_*` writes a payload
    (SSM checkpoint, KV pages, or 8-step hidden vectors) into a `memfd_create` region, seals it
    immutable, and passes the fd with a 256-byte `LatentHeader` over a Unix socket via `sendmsg`
    (`latentos/ipc.mojo`, pure Mojo `external_call`). The receiver `mmap`s the sealed fd read-only.
    Wired to the engine by `BARO_LATENT_SOCK` (`serve/latent.mojo`, commit `2e7b5d3`); unset, nothing
    happens. Proven twice: `test_latent` in `run-tests.sh` round-trips a checkpoint byte-exact
    in-process, and the L1 gate `tools/latent-gate.sh` does it live across two processes over the
    real socket (1 handle exported, 1 received, ingested payload byte-identical at 52,690,944 bytes,
    with a negative control confirming the gate cannot pass vacuously). This moves bytes through
    host memory, not GPU to GPU.
  - **HIP IPC handle handoff, GPU to GPU: KILLED** (`47e2896`, confirmed `d060a72`).
    `hipIpcOpenMemHandle` returns `hipErrorInvalidValue` in the receiving process on every run
    through Mojo's `external_call`, while a parallel C probe built with `hipcc`
    (`bench/latentos-ipc-probe.c`) succeeds and sha256-matches every time. Cause isolated to Mojo's
    FFI not implementing the SysV by-value struct ABI for arguments over 16 bytes, not a driver or
    permissions problem. Not planned to be revisited in Mojo.
  - **HTTP `.baro` state files: WORKS on one node**, and is a separate mechanism from the socket
    sidecar despite reusing the LAT1 header. This is the P1 export/import API above.
  - `latentos-agent` as a live daemon: WORKS. `--daemon` now keeps the process resident after boot,
    runs `stage_l7_serve_step` once per second for store eviction and watchdog heartbeats, and writes
    the manifest before serving. Gate: `bench/latentos-agent-gate.sh`. Cross-host transport
    (`tcp_listen`/`tcp_connect`/`send_tcp_latent`) remains untested.
  - Measured experiments, kept for their numbers rather than as gate passes: **E12** KV handoff
    scores identically to full re-prefill on task accuracy at 8k/16k/32k. **E14** one reader with N
    followers PASSES at N=3 (3/3) and FAILS at N=10 (9/10), the miss diverging at token 62 of 64
    after both arms already agree on the answer; wall clock beat llama.cpp's own cold total at both
    sizes (1.68x at N=3, 3.70x at N=10). **E15**, the llama.cpp state bridge, did NOT MEET its own
    20/20 bar on any of three models (16/20, 13/20, 14/20) against a llama.cpp control that itself
    only reaches 15/16/16, so it is UNCLASSIFIED rather than a clean failure. The pure layout
    round-trip underneath it is solid (byte-identical, 32/32 ids).
  - **Status framing changed on 2026-09-17 (`0eb6ef2`, merged `50f9d34`).** The four P1 gates, their
    bars, the kill line and the "ids must match" pass/fail rule were removed on the maintainer's call, with
    nothing put in their place, because that criterion was the wrong test: gate 4 held llama.cpp to a
    bar llama.cpp misses restoring its own state, and gate 2's ids could not tell a correct restored
    state from a deliberately K/V-swapped one (2 of 5 wrong states still produced matching ids). The
    measurements and the falsifiers were kept, since the falsifiers are what caught the real
    `save_state` cross-prompt export bug. This is a descope of the pass/fail framing, not a proof or
    a disproof. No plan-level LatentOS gate is PASS-and-live today.
  - The HTTP HIDDEN and LOGITS_TOPK streams now exist as dense/MoE completion extensions. State-transfer
    HMAC is enforced when both nodes set the same non-empty `BARO_STATE_HMAC_KEY`; with a key set,
    unsigned or wrong-key state is rejected, while keyless mode accepts only unsigned state.
## Models, pipeline and verification

Current as of `main` at `2208bdb`, after team A's merge `a01ce69` (P0b CPU
router gates 1-4) and team B's merge `41f60f8` (P5b LoRA write-back, P4
multi-gpu gate), with the P5b bar amendment at `c24dd32`.

### Model roster: the quality axis, read honestly

`bench/quality-models.json` lists 11 servable model configurations. Do not
read that count as "11 validated models." Only **2 of the 11 have both a
perplexity PASS and a task PASS**: the Qwythos champion and RegesCore-35B.
Everything else is either perplexity-BLOCKED by construction, VOID from a
harness accident, or simply not run this round.

The quality harness is exposed as one command: `tools/baro eval MODEL-KEY`
runs one row, `tools/baro eval-all` runs the roster, and `--dry-run` validates
the key without reserving a GPU. These commands reuse `bench/quality-run.sh`
and preserve its per-step `gpu-wait` reservations.

| model | engine | quant | PPL | task | verdict |
|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | spark | Q4_K_M | BLOCKED | PASS (-1.7 pp) | task only |
| lily-cybersecurity-7b-v0.2 | spark | Q6_K | BLOCKED | PASS (-1.7 pp) | task only |
| Qwen2.5-7B-Instruct | spark | Q4_K_M | BLOCKED | PASS (0.0 pp) | task only |
| Qwen2.5-Coder-7B-Instruct | spark | Q4_K_M | BLOCKED | PASS (+0.8 pp) | task only |
| Granite-4.2-3B | spark | BF16 | BLOCKED | VOID | needs a full rerun |
| Ornith-1.5-9B | dense (qwen35) | Q4_K_M | not run | not run | skipped this round, "cut for time" |
| Qwythos-9B-v2 | dense (qwen35) | Q6_K, MTP | not run | not run | skipped this round |
| Qwythos champion | dense (qwen35) | BF16, MTP | PASS (1.106 ratio) | PASS (-8.3 pp) | **both PASS** |
| Qwythos p5b-patched | dense (qwen35) | BF16 base + LoRA on blk.24-31 ffn_down, served q4 | PASS (1.102 ratio) | PASS (-7.5 pp) | both PASS, see training section |
| RegesCore-1.0-35B | moe (qwen35moe) | Q4_K_S | PASS (0.999 ratio) | PASS (-0.8 pp) | **both PASS**, weakest agreement basis in the repo (83.1% mean) |
| Spark-X2.5-4B | spark2_5 | Q8_0 | BLOCKED | not run this round | no agreement basis on record at all |

Numbers from `docs/BASELINE.md` "Quality vs llama.cpp" and
`bench/quality-protocol.md` "Result" (2026-09-16 sweep), cross-checked
against `bench/quality-models.json` and `bench/quality-bands.json`.

**What "BLOCKED" means mechanically.** `serve/spark.mojo`, at the commit
this sweep ran against, has no `top_logprobs`/penalty wiring at all: a
`top_logprobs:1` request is silently accepted and the engine decodes greedy,
so zero logit rows get dumped. Perplexity scoring needs the full pre-penalty
logits row at every position (`bench/quality-protocol.md` "Item 1:
instrument"), so there is nothing to score. This affects every spark-family
model (Llama-3.2-1B, lily-7B, Qwen2.5-7B, Qwen2.5-Coder-7B, Granite-4.2-3B,
Spark-X2.5-4B), which is why the block is per-family, not per-model.
`bench/quality-protocol.md`'s own amendment says the fix is already scoped:
lane `a lane pane` was wiring `top_logprobs`/`amar_topn_probs` into `spark.mojo`
as of that note. **To unblock**: confirm that wiring landed at the current
commit (`grep -n top_logprobs serve/spark.mojo`), then rerun the 6 spark
perplexity rows under the existing protocol and bands, no new brief needed
per the amendment's own text.

**Granite's VOID row is a data-integrity problem, not a footnote.**
`bench/quality-protocol.md`'s Result section states plainly:
"`quality-run.sh` edited while the sweep was executing it, bash re-read
shifted lines, its prep step failed." That is not a model defect and not a
skip; it is a corrupted run whose output must not be read as evidence about
Granite one way or the other. There is currently no valid PPL or task row
for Granite-4.2-3B anywhere in this repo's tracked results. The row needs a
clean rerun, not a note explaining the old one, before anyone cites a
Granite quality number.

Outside `quality-models.json`, `~/Models/library/INDEX.md`'s own
correctness table lists 13 total `-BARO-` bakes; 3 are excluded from the
quality scope (2 `FAIL` Spark-Q4 bakes, 1 `superseded` duplicate of the
Qwythos champion), and MiniCPM5 is separately BLOCKED for having no
tokenizer. That reconciles to the 10 rows above plus the new p5b-patched
row, not a bigger hidden roster.

### Import and bake pipeline

There is no `model-import` script in this repo; the chain is wrapped, with a receipt per step, by the iTool `~/iTools/bin/model-import` (`~/iTools/llm/model-import`, 2026-09-17). `tools/engine-pack.py` used to
reference a `model-import.sh` in a comment, which does not exist anywhere in
the current tree (a repo-wide search came back empty); that comment was
corrected in `3444a8d`. The pipeline that actually runs today, driven by
`bench/dense-run.sh` and `tools/bake.sh`:

1. `tools/gen-profile.mojo` (spark family only) generates a `profile.mojo`
   from the source GGUF's dims/rope/activation, since `serve/spark.mojo` is
   not arch-specialized at compile time the way `serve/engine.mojo` is.
2. `tools/engine-pack.mojo` (qwen35 trunk `--q8`/`--q4`/`--q4-draft`,
   `--dense`, `--arch qwen35moe`) or `tools/spark-pack.mojo` (spark2_5, whose
   per-head attention gate the engine packer refuses) builds the runtime weight
   pack: one flat binary plus a text index and `identity.json`. Both are
   byte-identical to their Python oracles (`tools/engine-pack-check.sh`,
   `tools/spark-pack-check.sh`) and are built strict IEEE
   (`--fp-mode contract=off`) and run from the bake's closure by
   `tools/pack-run.sh`. On the six bakes' models the Mojo engine packer runs
   1.2-2.5x faster at 0.3-0.96 GB peak RSS against the oracle's 1.2-12.3 GB
   (`exchange/receipts/engine-pack-mojo/check.txt`). The ternary codecs
   `--q2b3`/`--tq1`/`--tq2` stay in `tools/engine-pack.py`. `tools/q8-check.py` proves
   the `--q8` path bit-equal to `llama-quantize Q8_0`.
3. `tools/embed-files.py` prints the source-file closure a bake must embed.
   It deliberately never embeds `serve/engine.mojo`/`serve/spark.mojo`
   themselves; those are pulled from git at the bake's own commit by the
   verify/closure tools, so a tampered or dirty-tree bake is detectable.
4. `tools/gguf-embed.py` writes a NEW gguf (the source file is never
   touched) carrying `baro.kernel.*` (arch, commit, file list, full source
   text per file) and, for self-serving bakes, `baro.run.*` (harness path
   and sha256, prompt tokens, reference ids, env, pack tool and flags).
5. `tools/bake.sh` orchestrates steps 3 and 4, and refuses to bake from a
   dirty `kernels/`, `serve/*.mojo`, `grammar/`, `uregex/`, `minja/`, or
   `latentos/` tree.
6. `tools/gguf-closure.sh` rebuilds the engine from a bake's own embedded
   sources and gates it on the bake's own embedded reference tokens.
7. `tools/gguf-verify.sh` runs step 6, then prints this card's gfx/driver/
   ROCm/power-cap next to the bake's `baro.hw.*` expectations. Its own
   comment is explicit that the tok/s comparison it prints "is a receipt,
   never a pass/fail."
8. `tools/gguf-receipt.py` is the only writer of the verified-receipt
   ledger, and only from `tools/baro verify --append`. A `tools/baro run`
   number is explicitly labelled self-reported and never becomes a receipt.
9. `tools/baro` is the CLI wrapping all of this: `run` (self-contained,
   builds engine and pack from the file alone, no checkout needed beyond
   the script), `verify` (pulls the harness from git and byte-compares it
   to the embedded copy before trusting anything), `serve` (adds a
   persistent engine/pack cache keyed by `tools/model-id.py`'s structural
   id plus the gguf's own sha256), `receipts` (lists the ledger).

PROVEN self-contained: the Qwythos champion and RegesCore bakes at commit
`8184f7d` both pass `tools/gguf-verify.sh` 64/64 with rebuilt tok/s_gen
close to the embedded 20-prompt median (`docs/BASELINE.md`, "Self-describing
bakes 2026-09-15"). One earlier bake of the same champion model
(`Qwythos-...-BF16-BARO-8184f7d`, since superseded) is on record in
`~/Models/library/INDEX.md` as "fixture-assisted", meaning it silently used
a local `.work/engine-pack-q4` fixture instead of its own embedded
prompt/ref tokens. That specific file's closure claim was wrong; the row
that supersedes it is the one to trust.

The dense-import agreement gate (`bench/dense-run.sh`, `MIN_PCT` default 90,
fixed at `b4271cd` after a period where PASS meant only "ran") shows
Qwen2.5-7B-Instruct at 96.9/99.4 PASS, bake verified 64/64 from the file.
Qwen2.5-0.5B-Instruct sits UNVERIFIED at 68.8/85.0, below the floor even
though llama.cpp's own q4_K_M-vs-bf16 class bar on the same prompts is
82.8/91.4; root cause is open ("our numerics ... or a shape-specific defect
on NQH 14 / NKVH 2 / H 896"). No 0.5B bake has shipped.

### Training and finetuning

**Continuous-thought training (`miniagi/latent/`): Phase C entry 3 PASS (99.54-99.73%, 5.9-8.5 s, seeds 110-112); entry 1 FAILED on wall. Greedy text CoT matches the soft eval within 0.08 pp.**
8-hop pointer chasing, 2 layers, 8 heads, 1.59M params: text chain-of-thought training read out
through soft-mixture thoughts (softmax(z/0.5) E fed back, no tokens). Fresh seeds 101-103: 99.66 /
99.80 / 99.34% on 4096 test cases, process wall 8.22 / 10.30 / 8.34 s; the preregistered <= 10 s per
seed fails on seed 102, run next to another project's GPU job (10.4 vs 8.0 ms/step idle).
Entry 2 (seeds 104-106, card reserved): 99.51 / 99.63 / 99.51%, binary wall 6.43-6.96 s, but the wall half is
NOT MEASURED: its frozen 50-update load probe read 8.71 > 8.5 ms (probe miscalibrated).
Raw-hidden Coconut: 93.5% at 1500 updates (unmatched); matched H=8 short+nobot 6000 updates: 99.71/96.63/99.88% at lr 1e-3, 32-35% at lr 3e-3 (`exchange/miniagi-rawctl-lr-prereg-2026-10-10.md`). Prereg and result: `exchange/miniagi-latent-phasec-prereg-2026-10-09.md`.

**Mojo-native training, dense byte-LM (`miniagi/train/`): WORKS at toy scale
(2026-10-09).** Every forward and backward kernel is Mojo (no autodiff exists
in MAX): one `gemm[ta, tb]` body serves forward, dX and dW; RMSNorm, SwiGLU,
RoPE, embedding, softmax-CE and causal attention backward; AdamW with fp32
weights and bf16 moments. Model: V 256, D 256, 4 heads, SwiGLU 688, 2 blocks,
1.65M params. Gate `tools/train-gate.sh` (oracle `tools/train_oracle.py`,
torch CPU in `.work/oracle-venv`): 62 parity cases incl. 20 per-tensor model
grads (worst 1.2e-6 of max), 200-step loss within 1e-5 over steps 0-19 and
within 2x torch's own 1e-7-perturbation drift over the run, second run
bit-identical. `Model[fast=True]` puts the forward weight GEMMs on fp16 WMMA:
1.10x (33.0 k to 36.4 k bytes/s at T=512); backward stays fp32 until loss
scaling has its own gate. Naive kernels, no perf claim. It is the M0+M1 cut of
a mini-AGI rebuild (`~/Brain/research/mini-AGI/2026-10-09-analysis.md`).

The other three lanes, three different outcomes. Read the status word first, then the
number behind it.

**P5b, LoRA write-back: PASS, but only after the gate's own floor was
amended.** `tools/p5b_lora_train.py` (129 lines, hand-written, no peft)
trains a rank-16/alpha-32 LoRA on `mlp.down_proj` of `blk.24..31` of the
Qwythos champion BF16 base, 1,000 GSM8K documents, loss 1.0927 to 0.8530
(job `mu5bgdqs4db9`). `tools/gguf-writeback.py` patches only the named
tensor byte ranges (`outside_range_diff_bytes=0`, verified). Forced
agreement of the patched engine against llama.cpp came back at 96.89%
aggregate over 20 prompts, with a per-prompt minimum of 89.06% on
`p09-explain-gpu`, 0.94 points under the frozen 90% min-over-20 floor.

The coordinator then ran the same gate against the **unpatched base**, and
it also failed the 90% floor, at 89.06% on a different prompt
(`p03-story`), aggregate 97.31%. Two reasons, not noise: at `n_predict` 64
the achievable score grid has no 90% step (57/64 = 89.06%, 58/64 = 90.63%),
and which single prompt sits worst is not a stable property across arms
(control re-run twice, identical 1230/1264 both times, so the swing is real
and reproducible, just not a defect). The floor was amended at `c24dd32`
("amend P5b's forced-agreement bar to aggregate-vs-control"), and the merge
at `41f60f8` records the result as "P5b LoRA write-back PASS on the amended
aggregate-vs-control bar." **If you re-run this or any similar forced-
agreement gate and the unpatched base has never been measured against the
same floor, do not trust a PASS or a FAIL from the patched arm alone.** The
min-over-20 number this gate originally used was unpassable by construction
at this generation length; anyone who re-applies the old 90% min-over-20
bar without first checking a same-arm control will hit the same wall for
the same non-reason. Quality table and 20/20 unpatched identity both PASS
cleanly, no caveat needed there.

**P5a, self-distillation draft head: PARKED, kill line hit, not a defect.**
`tools/mtp_head.py` retrains `blk.32` (the NextN draft head) as itself,
reusing HF's own decoder-layer classes rather than a hand-rolled
reimplementation. The first training attempt was VOID: a bug in
`bench/draft_dump.mojo` produced a dump where every record had the same
`target_argmax=0`, and the training smoke's loss collapsed to near-zero by
learning that constant. Caught by a validator repair that added a
distinct-argmax-count check. After the dump was fixed and a corrected
recipe was used (lr 2e-5, 16-document gradient accumulation, clip 1.0, 10%
warmup, after a more aggressive first recipe had measurably damaged the
head), the acceptance gate on the required p01-p05 subset showed **+2.30
percentage points** (70.08% vs baseline 67.78%) against a **required +4
percentage points** to pass. Result: NO SIGNAL. `exchange/lane-P5A-report.md`
is explicit that no gate-breaking failure was observed; the effect is
simply too small to justify shipping the head. Parked, not killed for
cause.

**P4, multi-GPU dispatch: FAIL, recorded as failed.** Per-process
`--engine-env` wiring and per-child-PID device receipts landed
(`02f899c`, `48d01d5`, `6b688fc`). The two-device (XTX plus iGPU)
round-robin gate ran for real, both engines healthy, device pinning
confirmed by PID and `ROCR_VISIBLE_DEVICES` read-back, and then failed at
`p06-translate`: the iGPU's solo-stream and split-stream requests, same
prompt, same temperature 0, same long-lived process, agreed for 6 tokens
then diverged, one degenerating into a repeated token. `exchange/lane-P4-report.md`
calls this "evidence of request-state bleeding or another engine/
kernel-level determinism defect, not a device-pinning or queue failure."
Merged to `main` at `41f60f8` only after the job reached terminal state,
and explicitly not claimed as passing. The router-backed placement gate and
the state-movement gate are both still blocked or unrun; no claim exists
for either.

### Tools that are broken or drifted: landmines for the next person

**`~/iTools/bin/refcache` is the landmine. Do not use it for this repo's
gates.** It now resolves to a rewritten tool
(`~/iTools/harness/refcache/refcache.sh`) with positional `key`/`get`/
`put`/`ls` subcommands. `bench/quality-run.sh` was written against the
older `--key`/`--key-file`/`--out -- CMD` interface, which still exists and
still works at `~/iTools/dev/refcache/refcache.sh`, just unwired from the
`bin/` shim that a plain `refcache` on PATH would find. This drift caused a
real, confirmed failure: the P5b patched quality row would not run until
`bench/quality-run.sh`'s two call sites were repointed at `dev/refcache`
(commit `ef2c7dd`). **Use the `dev/refcache` interface explicitly by path
until the two tools are reconciled; do not trust `refcache` resolved from
PATH.**

**There is no `model-import` script in this repo. FIXED 2026-09-17.** (The iTool `~/iTools/bin/model-import` wraps the chain from outside the tree.)
Two places pointed at one: `tools/engine-pack.py`'s comment named a
`model-import.sh`; the historical `POST /api/pull` 501 also named a
`model-import.py`. Neither file exists in the tree, under `tools/` or anywhere
else. The endpoint now confirms the loaded pack, while imports use the
documented chain in `3444a8d`: `gen-profile.mojo` /
`engine-pack.mojo` (or `spark-pack.mojo`) / `embed-files.py` / `gguf-embed.py` /
`bake.sh`.

### Verification system: what to run and when

**`run-tests.sh`** needs the GPU card. It builds the C++ shim, then builds
and runs eight Mojo test binaries in sequence: `test_gemm` (fp16 GEMM vs a
host reference through the C ABI), `test_prefix` (byte-exact prefix
checkpoint restore against the real engine path), `test_sample_ref` (a
pure-CPU sampler reference, no accelerator), `test_sample_pen` (penalties
and top-N logprobs, device vs host at real vocabulary width),
`test_sample_mask` (grammar-masked sampling), `test_spark_attn` (KATT
head-dimension parity at HD 64/128/256 against a numpy float64 oracle),
`test_latent` (builds both the mint and ingest sides of the LatentOS
sidecar together, since a prior break went undetected for a full commit
range when only one side was built), and `test_serve_proto` (the
request-line schema slice), finishing with `kernel-census --check`. No
wall-clock receipt for a full run was found in this repo; expect several
sequential builds plus short GPU runs, not a multi-minute sweep.

**`tools/ci-checks.sh`** needs no GPU, only a CPU Mojo build for the kernel
census. Twelve checks: no orphaned `amar_*` kernel; `docs/KERNELS.md`
matches a fresh census; pre-tokenizer regexes match llama.cpp; all Python
sources parse; all shell scripts parse; `.github` YAML is valid;
`bench/PROTOCOL-RULES.md` still contains sections `P1` through `P6` (see
caveat below); the vendored `tools/gguf_reader.mojo` matches its iTools
upstream; the vendored `uregex/`, `minja/` packages match their upstream
repos and `latentos/` is the one copy (AMDHQ `src/latentos` points here); every repo-relative path referenced from `docs/*.md`
or `bench/*.md` actually exists; `results/*.json` hardware receipts are
well-formed; every `bench/*.mojo` with a `main()` still compiles (this is
the check whose absence once let `bench_launch_floor.mojo` drift silently
for a week). No wall-clock receipt was found for this either; it is mostly
static analysis plus one CPU build, so it is the cheap check to run first,
before `run-tests.sh`.

**Blind spot in this check, FIXED 2026-09-17**: step 7 verified only that
`PROTOCOL-RULES.md` still had `P1` through `P6`, while the file had grown to
`P20`, so dropping any of `P7` through `P20` passed silently. Now checks the
full range (`d67c793`), verified by a negative control that renames `P17` and
produces `FAIL PROTOCOL-RULES.md lost rules: P17`. The bound is the constant
`PROTOCOL_RULES_N` at the top of the script; raise it when a rule is added.

**Which identity check to use.** Three tools exist and they answer
different questions; do not substitute one for another.

- **Teacher-forced agreement (`BARO_FORCE`, per-request `force`)** is the
  standing gate for anything past about 256 generated ids. It measures
  whether the engine's own next-token distribution agrees with a reference
  at every forced position, not whether two engines produce the same free-
  running text. This is the only identity check valid at long generation
  lengths or under any kind of quantization mismatch between arms.
- **Greedy 64-token equality is never a valid gate past roughly 256 ids.**
  This is a standing rule, not a preference: llama.cpp's own f16-KV
  configuration fails its own f32 reference at 5 of 7 tested lengths
  (`bench/PROTOCOL-RULES.md` P14). A free-running greedy match is fragile
  to any single-token divergence compounding forward; forced agreement at
  each position is not, because every position is re-anchored to the
  reference regardless of what came before.
- **20-prompt identity (`bench/mtp-prompts.sh`)** is the basis for any
  performance claim (tok/s, speedup, acceptance rate), never a single
  prompt. `bench/PROTOCOL-RULES.md` P4's own worked example: a headline of
  128 tok/s and 1.17x over llama.cpp, measured on one 5-token race prompt
  at 94% acceptance, fell to 82 tok/s and 0.66x on the 20-prompt set. Both
  numbers were correct measurements; only one described the engine.
- **Byte checks beat id checks for saved state.** Two of five KV-swapped
  save/restore states kept the correct 32 token ids while still exporting
  the wrong prompt's SSM state underneath; only a byte-level compare of the
  state file caught it. Use a byte check, not an id check, whenever the
  artifact under test is a saved state rather than a token stream.

**A gate bar must be validated against a threshold some known-good
configuration can actually clear (P14), and this is the rule the P5b
episode exists to teach the next person.** Before freezing a forced-
agreement floor, measure that same floor against an unpatched, same-shape
control on the same hardware. If you skip that step and only measure the
arm under test, a floor that is unreachable by construction (as the 90%
min-over-20 floor was, at generation length 64, where the achievable score
grid steps from 89.06% to 90.63% with nothing in between) reads as a defect
in whatever you are testing, when it is really a property of the floor
itself. Write the same-arm control into the gate from the start rather
than adding it after a false alarm.

**Preregistration and cold-cache discipline** for anything claiming a
number: `bench/PROTOCOL-RULES.md` P1 through P20 bind every protocol in
`bench/`. The two most load-bearing for a first-time gate author are P1
("passing a parameter is not evidence it took effect"; read every arm-
defining value back from the running system before trusting a timed run)
and P7 (the arm file identifying which binary ran must be written by the
run itself, from inside the code path that executes that arm, never
assembled alongside it). `bench/coldcache-protocol.md` is the single-buffer
GEMM cold-cache rotation protocol; single-buffer GEMM timings at working
sets of 96 MB or more are invalid on this card, Infinity Cache
contamination.

## Known gaps in this document

- P4 is being re-investigated on branch `lane-p4` as this was written. Its
  status here may be stale in the direction of pessimism if the re-run passes.
- Wall-clock costs for `run-tests.sh` and `tools/ci-checks.sh` are not recorded
  anywhere in the repo, so this document cannot state them. `ci-checks.sh`
  measured about two and a half minutes on 2026-09-17, dominated by compiling
  the bench sources.
- The spark-family perplexity block may already be lifted. The fix was scoped to
  a lane that was wiring `top_logprobs` into `serve/spark.mojo`; confirm with
  `grep -n top_logprobs serve/spark.mojo` before assuming the six BLOCKED rows
  are still blocked.
- This document was assembled from a read of the tree at `2208bdb` plus the
  merged lane reports. Where a lane report and the code disagreed, the code won.

## Maintaining this file

Update it when a capability's status word changes, not when code changes. A
refactor that does not move something between WORKS, PARTIAL, DEFAULT-OFF,
PRESENT BUT UNUSED, FAILS and PARKED does not need an edit here. A gate that
newly passes, a default that flips, or a number that gets retracted does.

The status word and its evidence travel together. If you cannot name the check,
the correct edit is to change the status word, not to drop the citation.
