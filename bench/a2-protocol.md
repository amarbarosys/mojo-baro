# A2 step 1: paged KV through a block table (preregistered 2026-09-17, before its build)

Plan `docs/NEXT-PLAN.md` A2. Lane `lane-A2`, fable (kernel scope). Binds `bench/PROTOCOL-RULES.md`.
Gate harness `bench/a2-gate.sh` (landed `652a86c`, verified: 60/60 at 100%, 91 s wall cached).

## What exists

The KV pool is already page-major with 128-token pages: `kv_off` (`kernels/attn.mojo`) and
`dkv_off` (`kernels/dattn.mojo`) compute `(((t >> 7) * NAT + att_i) * NKVH + kvh) * KVHSTR +
(t & 127) * HD`, so the physical page is the logical page. Every KV read and write in the dense engine
goes through one of those two functions (attn: append, append2, decode, prefill, prefill_wmma,
head_span; dattn: exact, split, load_span; mega token and window), plus the MoE megakernel and the
spark profile's attention. The pool is `ceil(TMAX / 128)` pages, one request at a time, so a block
table is an identity map today; its value is A3 (N requests share the pool) and B3 (pages leave VRAM).

## Change

- `kv_off` and `dkv_off` take a device table `tab: UnsafePointer[Int32]`; physical page =
  `tab[t >> 7]`. Page size stays **128, not the plan's 16**: `dattn_load_span` loads each 8-token span
  from one base address and walks it linearly, so spans must stay inside a page and the lookup is one
  load per span, not per token; 16-token pages would multiply the table by 8 and force a per-token
  lookup in every loop for a fragmentation benefit that does not exist at N <= 4 on one card.
- Every kernel that touches KV takes the table as one more pointer argument; every launch site passes
  `b.kvtab_d`. The MoE megakernel and the spark profile get the same argument and an identity table
  (one rule, no special cases). Kernel files stay comment-free.
- Host: a new `kvpage.mojo` under `serve/` (named this way until the lane lands, for the
  dangling-reference check), a `PageTable` (free list over the pool's physical pages, `alloc(n)`,
  `free`, `identity()`, `reverse()`, `upload(ctx)`), owned by the engine's window state; `kvtab_d`
  (`int32[tpages]`) in `WindowBufs`. `BARO_KVTAB=identity|reverse` selects the mapping at start-up
  and is echoed (P1). State save gathers logical pages through the table; state load resets the table
  to identity and copies as before. Prefix checkpoints are unchanged (KV stays in place; the table is
  per request and identity while there is one request).

## Predictions (frozen)

- P-A2a identity table: `bench/a2-gate.sh MAIN CAND` 60/60 at 100% (MIN_PCT 100), 59/60 restored,
  decode after 32k within 1% of the main engine's number from the same stint (main measured 101.56
  today: candidate >= 100.5).
- P-A2b **reverse table** (`BARO_KVTAB=reverse`, logical page p on physical page tpages-1-p): the same
  gate 60/60 at 100% against the main engine. This is the only prediction that proves the table is
  read: a kernel that ignores it would read the wrong physical page for every logical page but one and
  agreement would collapse past the first 128 tokens.
- P-A2c short context: `bench/ab-prompts.sh` main vs candidate, T=0, megakernel, spec off, 20
  prompts: candidate within the standing +-2% band of main (main 135.07 today).
- P-A2d prefill: the first request of each set in the a2 gate (the one that prefills 8k, then 8k more,
  then 16k more) within 2% of the main engine's `prefill_s` for the same request.
- P-A2e `run-tests.sh` exit 0 with `kernels/test_mega_block.mojo` bit-identical (tools/mega-gate.sh
  kernel step), `tools/ci-checks.sh` exit 0, the MoE engine and the spark engine build and pass their
  standing checks in run-tests with identity tables.
- P-A2f state round trip: `BARO_STATE_SAVE` under the reverse table, `BARO_STATE_LOAD` into an
  identity table, the continuation's forced agreement 64/64 on one prompt (instrument receipt, P4).

## Kill line

Any identity miss under the identity table is a bug and is fixed before anything else. Decode after
32k below 99% of main means the page lookup did not hoist out of the token loop: restructure the
dattn span base per page and re-run; below 97% after that, step 1 is not merged as written and the
report says why. Receipts on every timed run: engine sha256, `BARO_KVTAB:` echo, `TMAX:`, `kv pages`,
`bench/clock-probe.sh` line, arm parameters from the running engine.

## Result (2026-09-17, change `09a8cd9`, merged tree `d3a0bf0`, engine `367ab62323748a3a`)

Receipts `.work/a2/gates/`, `.work/a2/gates2/`, `.work/a2/gates/postmerge-quick/`, ISA `.work/a2/isa/`;
report `exchange/2026-09-17-A2-step1-report.md`. Preflight PASS before every job.
- P-A2a PASS: identity table 60/60 at 100.0%, 59/60 restored, decode after 32k **100.67** (main
  101.56, 99.1%, bar 100.5). `BARO_KVTAB: identity kv pages: 256`.
- P-A2b PASS: reverse table 60/60 at 100.0%, 59/60 restored, decode after 32k **100.71**. The
  kernels read the table.
- P-A2c PASS: main 136.95 vs paged 136.07 tok/s_gen, ratio 0.994, identity 20/20.
- P-A2d PASS: first-request prefills equal within one request's noise (8064 ids 3.06 s on both).
- P-A2e PASS with one pre-existing red: run-tests exit 0 (103 kernels, 0 orphans), ci-checks exit 0,
  MoE and spark engines build; `test_mega_block` m=1 arms bit-identical, its m=3 window-kernel parity
  arm FAILS identically on main's kernels (958/12288 residual mismatches, max 0.0078; G=192 residency
  probe aborts): pre-existing, behind `BARO_MEGA_WIN=0`, recorded on the board, not chased.
  `test_attn_block` compiles but has no fixture on any tree (needs `tools/attn-ref.py` dumps): not run.
- P-A2f PASS: save under reverse (pos 12, 1 page), load under identity, forced agreement 64/64.
- ISA: q4 token megakernel dual 124/79/79/59, 0 spills (unchanged); dattn split 165/0; prefill WMMA
  spills 28 -> 25.
- Post-merge (main's MoE stage 3 merged into the lane): quick gate 3/3 at 100.0%, decode after 32k
  101.05. Kill line not reached.

# A2 step 2: int8 KV on the block table (preregistered 2026-09-17, before any int8 GPU run)

Brief `~/Brain/mojo/mojo-baro/briefs/2026-09-17-A2-step2-int8-kv.md`. Lane `lane-a2s2`, opus.
Code `80d4b2c` (built, never run on the GPU at freeze time); bar harness `f45a7dd`.

## The bar (P14), measured before this section was written

llama.cpp side: `bench/a2-bar-llama.sh .work/a2s2/bar/full` (tool `tools/llama-force.cpp`, llama.cpp
`ca3d5a3e1`, `Qwythos-9B-...-Q4_0-pure.gguf` sha256 `6e0ae811ca9e4c49`, n_ctx 32768, FA on, one
decode step per forced token). The f32 arm is generated greedy, then forced against itself as the
P11 self-check: **3840/3840**. A first version forced the 63 reference ids as one batch and read
127/128 against f32 itself (llama.cpp kernels are not batch-invariant); it was discarded.

Our side: `MIN_PCT=0 bench/a2-gate.sh engine-ref engine-bf16` (ref `a8ab6d2903176a42` = `ed8c7f7`,
bf16 `38ee0084a8e8ac14` = `80d4b2c -D BARO_KVQ=bf16`, identity table, receipts `.work/a2s2/gates/bar-bf16/`,
`BARO_KVQ: bf16 kv bytes/token: 32768`, 290 W cap).

| arm vs its own f32 KV | 8k min / mean | 16k min / mean | 32k min / mean | at 100% (8k/16k/32k) |
|---|---|---|---|---|
| llama.cpp q8_0 (block-32 scales) | 96.9 / 98.52 | 95.3 / 98.12 | **93.8** / 97.97 | 7 / 6 / 4 of 20 |
| llama.cpp f16 | 100 / 100 | 100 / 100 | 100 / 100 | 20 / 20 / 20 |
| ours bf16 | 98.4 / 99.8 | 96.9 / 99.7 | 98.4 / 99.6 | (gate table) |

llama.cpp's f16 equals its f32 because its flash-attention kernel computes in f16 either way, so its
"f32 reference" is not a true f32 reference; q8_0 still moves 16 of 20 prompts at 32k off it.
Decode in the same stint (ref, f32 KV): 123.26 / 115.19 / **101.38** tok/s at 8k/16k/32k; bf16:
127.97 / 123.20 / **114.83**. Halving the KV bytes bought 13.3% at 32k: the attention bytes are on
the decode path at long context (the chat lane's "not bytes" finding was measured at 16 of 96 blocks
occupied, before the split kernel).

**Frozen: `MIN_PCT=93.75`** (60/64, the known-good int8 configuration's worst prompt) at all three
lengths, and **set mean >= 97.9** at every length (its worst set mean). Both must hold.

## Layout (arm 1, per-row int8, post-RoPE)

- `BARO_KVQ=f32|bf16|int8` comptime (`get_defined_string`, default f32) sets `KVT`; echoed at start-up
  as `BARO_KVQ: <q> kv dtype: <t> kv pool bytes: <n> kv bytes/token: <n>`.
- One (page, attention layer, KV head) stride = 128 rows x 256 int8, then 128 float32 scales:
  `KVPAD = KVPAGE * 4`, `KVHSTR = 32768 + 512` bytes. The row stride in elements stays `HD`. Scale of
  row t = `absmax / 127`, byte offset `row + KVPAGE*HD + (t & 127) * (4 - HD)`.
- **Page tail, not a separate plane**: the tail is inside the page, so the block table, pool sizing
  (`KVHSTR` already carries `KVPAD`), state paging and a future page move (A3, B3) carry the scales
  with no new buffer, no new kernel argument and no second address function. A separate plane would
  add two pointers to every KV kernel and a second table walk per read.
- Quantize at append (`amar_kv_append`, `append2`, the q4/q8 token megakernels and the window
  megakernel): wave `warp.max(abs(x))` -> barrier -> row max -> `round-half-away(x * 127 / max)`;
  thread 0 writes the scale. Dequantize at read: split kernel `dattn_load_span` (int8 8-wide load x
  row scale, so `dattn_step` sees f32 as before), exact `dattn_span` / `attn_head_span` (scale hoisted
  per key row, per value token), `amar_attn_prefill` (per row), `amar_attn_prefill_wmma` (dequantize
  then f16).
- f32 builds are unchanged: measured before freeze, `isa-diff` ref vs `80d4b2c` default build, every
  kernel's VGPR/spill/private identical, token megakernel 124/79/79/59 SAME, window and split SAME.
- Refusals: state save/load and LatentOS KV pages raise on non-f32 KV (quantized state files are
  step 3); the MoE engine and spark raise at start-up under non-f32 `BARO_KVQ`.
- int8 ISA (recorded, not gated): q4 token megakernel dual 124/79/79/59 -> 135/84/84/60 (fast class,
  loop 1 >= 115), spills 0; split kernel 165 -> 180 VGPR, 0 spills; prefill WMMA spills 25 -> 16;
  window megakernel spills 42 -> 51 and 48 -> 60 (behind `BARO_MEGA_WIN=0`).

## Predictions (frozen)

All gates on engines built from the freeze commit, `BARO_TMAX=32768`, 290 W cap, preflight PASS.

- **P-S2a int8, identity table**: `MIN_PCT=93.75 bench/a2-gate.sh engine-ref engine-int8` exits 0;
  set means 98.5 at each length, band [97.0, 99.5]; restored 59/60.
  `BARO_KVQ: int8 ... kv bytes/token: 16640` (0.254 of f32's 65536) in `cand.out`.
- **P-S2b int8, reverse table** (`BARO_KVTAB=reverse`): same gate exits 0, and the per-prompt
  forced-agreement counts are **identical** to P-S2a's for all 60 prompts (dequantization reads the
  scale through the same table as the row; any difference is an address bug, not noise).
- **P-S2c decode after 32k** (int8, the gate's median): **120 tok/s, band [108, 128]**, and >= 101
  (the A2 bar). Arithmetic: ref 9.86 ms/token at 32k, bf16 8.71 ms (bytes x0.5 took 1.15 of the 2.45
  ms that 32k adds over short context), so bytes x0.254 leaves ~0.62 ms plus dequantization.
  Same-stint f32 default candidate for the ratio: P-S2e.
- **P-S2d short context**: `bench/ab-prompts.sh` ref vs int8 (`AB_ENGINE_B`), spec off, 20 prompts,
  one stint under `bench/clock-probe.sh`: int8/ref tok/s_gen ratio **0.985, band [0.97, 1.00]**
  (the megakernel's extra quantize barrier on 8 attention layers); greedy identity recorded, not
  gated (lossy KV). Standing +-2% band: below 0.98 is reported as a miss.
- **P-S2e f32 default unchanged**: `bench/a2-gate.sh engine-ref engine-f32` 60/60 at 100% (MIN_PCT
  100), decode after 32k within 1% of the reference's own same-stint number.
- **P-S2f tests**: `run-tests.sh` exit 0 and `tools/ci-checks.sh` exit 0 on the default build; the MoE
  and spark engines build (preflight). Under int8, `BARO_STATE_SAVE` exits non-zero with the
  `state files store f32 KV` error.
- **P-S2g RULER** (once, on the merge candidate): `niah_single` N=5 at 65536 and 131072 through our
  engine, int8 vs f32 on the same prompts, both fit 24 GB (int8 pool 2.03 GB at 131072 vs 8.4 GB):
  int8 score within 20 points (one prompt) of f32 at each size.

## Kill line

Any P-S2e miss is a bug in the f32 path and is fixed first. P-S2b not identical: address bug, fixed
before any other number is read. **Arm 1 misses the bar** (any prompt < 93.75 or a set mean < 97.9 at
any length, on either table): arm 2, KIVI axes (keys per channel before RoPE, RoPE at read, values
per token), under its own preregistered addendum; if arm 2 misses too, int8 KV is not viable on this
model at these lengths, the code stays opt-in off, and that is the result. Decode after 32k < 101:
the lane reports the miss with the split kernel's VGPR and the loads per token, and the int8 arm is
not recommended as a default for long context. Receipts on every timed run: engine sha256,
`BARO_KVQ:` and `BARO_KVTAB:` lines, `TMAX:`, arm file, power cap.

## Result (2026-09-17, code `80d4b2c`, engines int8 `dcf1564e082c5286`, f32 `829c25a55d4c5421`)

Report `exchange/lane-A2S2-report.md`; receipts `.work/a2s2/`. Preflight PASS before every gate.
- P-S2a PASS: identity table min 96.9 at 8k/16k/32k, means 99.3 / 99.2 / 99.5, restored 59/60,
  `kv bytes/token: 16640`.
- P-S2b PASS: reverse table per-prompt counts identical to P-S2a on all 60.
- P-S2c PASS: decode after 32k 117.69 (reverse 118.17) vs f32 default 100.54, 1.171x.
- P-S2d MISS on the frozen band (faster): ref 136.07, int8 137.62, ratio 1.011, inside +-2%.
- P-S2e PASS: f32 default 60/60 at 100%, decode after 32k 100.54; ISA identical to the reference.
- P-S2f PASS: run-tests exit 0, preflight (ci-checks, MoE and spark builds) PASS, int8 state save
  exits 1 with the f32-only error.
- P-S2g PASS on what ran: niah_single N=5 int8 100.0 at 65536 and 131072, f32 100.0 at 65536; f32 at
  131072 fits (started, 8.7 GB pool) and its prompts were not run (int8 at the ceiling, the maintainer's call).
Kill line not reached; arm 2 not run.
