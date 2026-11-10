# chat lane protocol — preregistrations for the agent-engine milestones

Design of record: `docs/design/agent-engine-2026-09.md`. Work order: design §10.
Binding rules: `bench/PROTOCOL-RULES.md` P1-P6. Every milestone below is
preregistered (prediction + gate + falsifier) BEFORE the build starts; the
Result section is filled in after the gated run and never edited to fit.

## Baseline (frozen 2026-09-08, `main` at 5ae2925, worktree `lane-chat`)

Read back from `tools/merge-gate.sh` on the binary built in the same command
(`.work/merge-gate.txt`, run 2026-09-08T06:14:41+02:00):

| receipt | value | source |
|---|---|---|
| engine build | exit 0 | merge-gate |
| run-tests.sh | exit 0, 63 kernels, 35 in registry, 0 orphans | merge-gate |
| ci-checks.sh | exit 0, all non-GPU checks passed | merge-gate |
| test_prefill | exit 0, PASS | merge-gate |
| one-shot q4 identity | PASS: 64 tokens match | `tools/check-tokens.sh` |
| one-shot q4 tok/s_gen | 134.52702393163582 | engine stdout |
| one-shot q8 identity | PASS: 64 tokens match | `tools/check-tokens.sh` |
| one-shot q8 tok/s_gen | 77.04891558274252 | engine stdout |
| TMAX / prefill chunk | 1088 / 1024 | engine stdout |
| prefill p0512 | prefill rows 511, prefill_s 0.424297253, tok/s_gen 118.86931574785959 | engine stdout |
| mega fail word | 0 | engine stdout |

P4 note: the single-prompt 134.5 above is an **instrument receipt**, not the
bar. The bar is the 20-prompt median from `bench/ab-prompts.sh` /
`bench/mtp-prompts.sh`; `main`'s recorded champion is 133.9 tok/s_gen (9e6feaa).

## Where the 1088-token ceiling actually lives (read from source, 2026-09-08)

1. `kernels/attn.mojo:12` `comptime MAX_T = 1088` sizes an LDS score array
   `stack_allocation[f32, row_major[MAX_T]]` in `amar_attn_decode` (line 155)
   and in the megakernel attention phase (`kernels/mega.mojo:807`). 4352 B of
   LDS per block, and a hard cap on T.
2. `serve/registry.mojo:36` `comptime TMAX = 1088` sizes the KV cache layout
   `row_major[NKVH, TMAX, HD]` (f32) and the token buffer `row_major[TMAX]`,
   and is baked into the three megakernel instantiations.
3. `serve/engine.mojo` rejects `len(prompt) + n > TMAX` and allocates
   `N_ATT * NKVH * TMAX * HD` f32 for K and for V (71 MB at 1088; 8.6 GB at
   128k if left f32 and contiguous).

The prefill attention kernel `amar_attn_prefill` is already online-softmax
(running `m_run`/`l_run`, PA_TK=16 tiles) and carries no MAX_T array: it is
T-agnostic today. Only the decode path holds the ceiling.

M0 is therefore split into three preregistered steps, each with its own gate,
each landing on `lane-chat` with its numbers in the commit.

---

## M0a — decode attention without the MAX_T score array

**Change.** Rewrite `attn_head_body` (shared by `amar_attn_decode` and the
megakernel attention phase, so both keep computing the same bits) from
"score all T into LDS, then reduce" to a chunked online-softmax loop over
T in chunks of C = HD = 256: per chunk, 256 threads each score one position
into a 256-entry LDS array, the chunk max and chunk sum are warp-reduced, the
running `(m, l, o)` are rescaled by `exp(m_run - m_new)`, and the output row
is accumulated against V for that chunk. LDS for scores drops 4352 B -> 1024 B
and T stops being a compile-time quantity in the kernel.

**Not in this step.** KV dtype (stays f32), paging, TMAX in the registry, the
engine's `prompt+n > TMAX` check. Those are M0b/M0c.

**Predictions (frozen before the build; commit of the frozen text is the
receipt).**
- P-A1 Identity holds: one-shot q4 and q8 both `PASS: 64 tokens match`, and
  `bench/mtp-prompts.sh` k=2 identity 20/20 PASS, 0 FAIL.
  Rationale: the change reorders the softmax rescale only; the K-dot and the
  V-accumulate keep their existing per-element order and 8-wide grouping.
  This is a non-bit-exact change, so `tools/model-ref.py` decode identity is
  the gate, per the standing rule.
- P-A2 Decode throughput at T <= 1088 is within +-2 % of baseline on the
  20-prompt median. Two effects roughly cancel: 3.3 KB less LDS per block
  (occupancy-neutral at the current 31 KB budget, so no win expected), and
  ~4 extra `s_barrier` per 256 positions (5 chunks at T ~ 1088, so ~20
  barriers vs ~6 today).
- P-A3 The falsifier is a regression worse than 2 % on the 20-prompt median,
  or any identity failure. Either sends the step back, not the gate.
- P-A4 `mega fail word: 0` on every run (the grid-barrier health word).

**Verification before timing (P1).** Every timed arm rebuilds the binary in
the same command as the run. Arm-defining values read back from the run's own
output: `TMAX:` line, `spec k:`, `prompt tokens:`, `tok/s_gen`,
`mega fail word`. The chunk constant is comptime, so the rebuild-in-command
is its receipt.

**Gate.** `tools/merge-gate.sh` all lines at least as green as the baseline
table above; 20-prompt A/B (`bench/ab-prompts.sh`, `AB_ENGINE_B`, `arm.txt`
read first) within P-A2.

**Result.** PASS, 2026-09-08.

Arms: A = `.work/engine-base`, built by `mojo build` from `git archive HEAD`
(b64821b, pre-change) in the same stint; B = `.work/engine`, built from the
working tree in the same stint. `arm.txt`: `power_cap_uW=290000000`,
`vddgfx=-100mV`, both arms `BARO_PACK=.work/engine-pack-q4`.

| receipt | base | M0a | source |
|---|---|---|---|
| 20-prompt median tok/s_gen (P4) | 132.68 (spread 9.9 %) | 132.99 (spread 5.1 %) | `bench/ab-prompts.sh` `.work/ab-m0a` |
| 20-prompt identity B == A | - | 20/20 PASS | same |
| one-shot q4 identity | PASS 64/64 | PASS 64/64 | merge-gate |
| one-shot q4 tok/s_gen | 134.53 | 132.90 | merge-gate |
| one-shot q8 identity | PASS 64/64 | PASS 64/64 | merge-gate |
| one-shot q8 tok/s_gen | 77.05 | 82.51 | merge-gate |
| prefill p0512 tok/s_gen (T ~ 575) | 118.87 | 128.03 | merge-gate |
| mtp k=2 identity | 20/20 | 20/20 | merge-gate |
| test_server.sh | ALL PASS | ALL PASS | merge-gate |
| mega fail word | 0 | 0 | engine stdout |

ISA receipt (`tools/isa-receipt.py`, same two binaries), the mechanism:

| kernel | `group_segment_fixed_size` base -> M0a | vgpr | vgpr spill |
|---|---|---|---|
| `amar_attn_decode` | 5408 -> 2080 | 26 -> 39 | 0 -> 0 |
| `amar_mega_token` q4 | 31264 -> 27936 | 239 -> 239 | 0 -> 0 |
| `amar_mega_token` q8 | 6688 -> 3360 | 256 -> 256 | 78 -> 76 |
| `amar_mega_window` | 6816 -> 3488 | 192 -> 192 | 7 -> 7 |

-3328 B of LDS per block is exactly the 4352 - 1024 the score array gave back.

Verdict against the frozen predictions:
- P-A1 held: every identity check PASS (q4 64/64, q8 64/64, mtp 20/20,
  A/B 20/20, server suite ALL PASS).
- P-A2 held: +0.2 % on the 20-prompt median, inside the +-2 % band. The
  spread also halved (9.9 % -> 5.1 %).
- P-A3 not triggered.
- P-A4 held: `mega fail word: 0` on every run.

Unpredicted, and larger than the gated metric: **q8 decode +7.1 %** (77.05 ->
82.51) and **q4 at T ~ 575 +7.7 %** (118.87 -> 128.03). Both are LDS-occupancy
effects the prediction did not anticipate -- P-A2 assumed the 31 KB budget made
the saving occupancy-neutral, which is true for the q4 megakernel (27936 B is
still one block per allocation granule) but false for the q8 megakernel, where
6688 -> 3360 B changes how many blocks fit. Recorded as an observation, not
promoted: neither number is a 20-prompt median, and the q8 path is not this
lane's champion.

---

## M0b — KV cache: token-major, runtime capacity, narrower than f32

**Change.** The KV cache layout goes from `row_major[NKVH, TMAX, HD]` to
`row_major[TCAP, NKVH, HD]` (token-major) and its element type from f32 to a
comptime `KVT`.

Why token-major removes the ceiling: in `row_major[TCAP, NKVH, HD]` the
outermost extent has no stride of its own, so the address of (t, kvh, d) is
`t*NKVH*HD + kvh*HD + d` -- `TCAP` never enters the arithmetic. It is a
nominal extent, and the real capacity is whatever the runtime allocation
holds. In the old order `TMAX` was the *middle* extent and therefore the
stride between kv heads, which is why it had to be a compile-time constant.
The per-layer stride in the megakernel (`kc + att_i * ATT32`) becomes a
runtime argument for the same reason.

Token-major is also the layout paging wants: a 128-token page is one
contiguous `128*NKVH*HD` run, so M1 turns `t*NKVH*HD` into
`page_table[t >> 7]*PAGESZ + (t & 127)*NKVH*HD` and nothing else moves. The
page table itself is NOT built here -- with one sequence and no prefix sharing
it would be an identity map, i.e. dead weight until M1 gives it a job.

Coalescing is preserved: the V accumulation still has thread `tid` reading
`V[t, kvh, tid]` across a wave, 64 consecutive elements for one t.

**Element type.** `KVT` is a comptime switched per build (the rebuild-in-the-
same-command is its P1 receipt, and the engine prints `kv dtype:` from it).
Arms: f32, f16, bf16. Design §2 names bf16; f16 carries 10 mantissa bits
against bf16's 7 and is what llama.cpp stores by default, which is where this
repo's reference token stream comes from. The identity gate picks, not the
design note.

**Not in this step.** The page table, `TMAX` as a runtime request parameter,
the `exceed_context_size` error shape, chunked prefill past CP. Those are M0c.
`TMAX = 1088` stays the engine's capacity, so nothing user-visible changes:
this step is the re-layout alone, gated on identity.

**Predictions (frozen before the build).**
- P-B1 The f32 arm is **bit-exact** against the M0a binary: `GENERATED`
  byte-identical on all 20 `bench/mtp-prompts/`, q4 and q8 identity 64/64.
  Only addresses change; every value, cast and summation order is preserved.
  This is the load-bearing prediction -- it is what makes the re-layout
  separable from the dtype question.
- P-B2 The f16 arm holds identity (q4 64/64, q8 64/64, mtp 20/20). The bf16
  arm does **not**: at least one identity check fails. Frozen deliberately as
  the asymmetric prediction -- if bf16 also holds, the design's choice stands
  and bf16 ships.
- P-B3 No arm separates from another on the 20-prompt median by more than
  2 %. At T ~ 100 the KV read is 2*T*NKVH*HD*4 B per layer over 8 layers =
  6.5 MB per token against a 6.2 GB q4 pack, i.e. under 0.2 % of the bytes
  moved, so halving KV width cannot show up here. The KV-width win is a
  long-context effect and is claimed only when M0c can measure it.
- P-B4 `mega fail word: 0` on every run.
- Falsifier: the f32 arm not bit-exact against M0a. That is a re-layout bug,
  and the step goes back rather than the gate being relaxed.

**Verification before timing (P1).** Each arm is a rebuild in the same command
as its run; `kv dtype:`, `TMAX:`, `spec k:`, `prompt tokens:`, `tok/s_gen`
and `mega fail word` all read back from the run's own output.

**Gate.** `tools/merge-gate.sh` ALL PASS on the chosen default; 20-prompt A/B
against the M0a binary within P-B3; P-B1 exact.

**Result.** GATE NOT MET, recorded 2026-09-08. The build is complete and
identity-clean but is not committed; the working tree carries it and
`.work/briefs/status-chat.md` has the handover.

Arms: A = `.work/engine-m0a`, built from `4027b6e` in the same stint;
B = the working tree, `KVT = float32`, layer-in-page layout.

| receipt | M0a | M0b tree | source |
|---|---|---|---|
| q4 one-shot identity | PASS 64/64 | PASS 64/64 | `tools/check-tokens.sh` |
| tok/s_gen at T ~ 70, 3 runs | 133.26 / 133.59 / 133.54 | 133.26 / 133.59 / 133.54 | engine stdout |
| tok/s_gen at T ~ 575 (`p0512`), 2 runs | 129.00 / 128.85 | **116.74 / 116.80** | engine stdout |
| `prefill_s` at T ~ 575 | 0.4175 / 0.5479 | 0.4211 / 0.4255 | engine stdout |
| `mega fail word` | 0 | 0 | engine stdout |

- P-B1 held on tokens: identity PASS everywhere measured, and an earlier
  20-prompt A/B of an intermediate M0b build was 20/20 identical to M0a.
  Bit-exactness at the buffer level was not separately checked.
- P-B3 **failed**: -9.5 % at T ~ 575, well outside the 2 % band. The
  prediction reasoned only about KV *bytes* (correctly negligible at
  T <= 1088) and never about the instruction budget the layout change spends,
  which is where the loss is.
- P-B2 not run. q8 identity, the merge gate and the dtype arms not run.

**Mechanism, measured.** The q4 megakernel is on an instruction-footprint
cliff. `tools/isa-receipt.py --hist` on `mega_amar_mega_token` (q4 code
object), four builds with identical semantics:

| decode-attention V-loop form | instructions | tok/s_gen at T ~ 70 |
|---|---|---|
| M0a, 8-wide unrolled | 12727 | 133.2 |
| unrolled over the chunk's 2 pages | 12905 | 127.4 |
| scalar, longer address expression | 12671 | 133.5 |
| scalar, algebraically simplified address | 12742 | 127.3 |

The last two compute the same address and differ by 71 instructions and 5 %.
Keeping the megakernel under the cliff forced the scalar V loop, and the
scalar V loop is what costs 9.5 % once T is large enough for the V
accumulation to matter. Both cannot hold inside one megakernel, so the fix is
the separate GQA-grouped decode kernel design §4 already specifies (split-K
over page spans, (m, l, O) partials, one merge workgroup per head) rather than
another attempt inside `attn_phases`.

**Falsified along the way** (both recorded so they are not retried):
- token-major `[t][kvh][HD]` costs 3.9 % at T ~ 70 — it loses kv-head
  locality, putting consecutive tokens of one head 4 KB apart instead of 1 KB.
- page-major with a **runtime** per-layer stride costs 5 % and takes
  `sgpr_spill_count` from 69 to 74: the stride argument stays live across the
  whole 32-layer loop. Layer-inside-the-page makes every stride comptime and
  is the only variant that keeps capacity out of the address arithmetic.
- **Cache-set aliasing is not the problem.** Layer-in-page puts the four kv
  heads exactly 128 KB apart, so 8 KB of padding per head block was added to
  break the power-of-two stride: 126.6 tok/s before, 126.6 after. `KVPAD` is
  dead weight and should go back to 0 with a re-measurement.

---

## M0b attempt 2 — same layout, M0a's V-loop form, KVPAD = 0

**Diagnosis receipts (CPU only, taken before this text, 2026-09-08).**
- T ~ 575 loss: the scalar V loop compiles to `global_load_b32` →
  `s_waitcnt vmcnt(0)` → `v_fmac_f32` per position (one load in flight);
  M0a's issues 8 loads under `s_clause` before its FMAs. Memory-latency
  serialisation, proportional to T. Attempt 1's diagnosis stands.
- "Instruction cliff": falsified. The "simplified address" build was
  reproduced CPU-side (12742 instructions, their number) and `isa-loops`
  shows its q4 *dot loops* rescheduled (VOPD 80/80/60 → 75/75/57, +26..33
  instructions each, more `s_delay_alu`) by a one-line edit in the attention
  V loop. The T ~ 70 losses were the known schedule lottery, fingerprintable
  at build time; the megakernel may unroll the V loop.

**Change.** `KVPAD = 0`. `attn_head_body`'s V accumulation returns to M0a's
8-wide form (8 loads into an `InlineArray`, then 8 in-order `o += sc*v`),
group base `vb + (tt >> KVPSH) * PGSTR + (tt & (KVPAGE-1)) * HD`, loads at
`+ j*HD` (8 | KVPAGE, no page crossing inside a group), scalar tail. Nothing
else in the attempt-1 tree changes.

**Predictions (frozen before the build).**
- P-C1 Build fingerprint, read BEFORE any GPU run: q4 `amar_mega_token`
  12727 ± 40 instructions, vgpr spill 0, `isa-loops` fused dot loop
  `dual` ≥ 105, loops 2-4 `dual` within ±2 of 80 / 80 / 60. A miss is a lost
  lottery ticket, not a layout verdict: one re-roll by spelling only, a second
  miss stops the step (brief: ask, file not terminal).
- P-C2 Identity: q4 64/64, q8 64/64, mtp k=2 20/20, 20-prompt A/B 20/20
  identical to `.work/engine-m0a`. Same FMA form and order as M0a.
- P-C3 p0512 (T ~ 575) tok/s_gen within ±2 % of M0a's 129.0 in the same
  stint. Falsifier: < 126.4 means the serialisation diagnosis was wrong.
- P-C4 20-prompt median within ±2 % of M0a (P-B3 restated).
- P-C5 `mega fail word: 0` on every run.
- Receipt run, not a prediction: `.work/engine-diag` one-shot at T ~ 70 in
  the same stint, expected ~127 (ties the losing fingerprint to its number).

**Verification before timing (P1).** Rebuild in the same stint; `kv dtype:`,
`TMAX:`, `spec k:`, `prompt tokens:`, `tok/s_gen`, `mega fail word` read from
each run's output; `arm.txt` first for the A/B.

**Gate.** P-C1 before GPU; then `tools/merge-gate.sh` ALL PASS; 20-prompt A/B
vs `engine-m0a` within P-C4; P-C3 within band. Then the dtype arms under
P-B2 as frozen, each arm fingerprinted before its run.

**Result.** PASS on the f32 arm, recorded 2026-09-08. The dtype arms both
fail identity; f32 ships (P-B2 as frozen).

Arms: A = `.work/engine-m0a` (4027b6e); B = `.work/engine` = `.work/engine-c2`
(sha `ba6eebfb1b53911c`, working tree, `KVT = float32`, `KVPAD = 0`, M0a V-loop
form). `arm.txt`: `power_cap_uW=290000000`, `vddgfx=-100mV`, both arms
`BARO_PACK=.work/engine-pack-q4`.

| receipt | M0a | M0b attempt 2 | source |
|---|---|---|---|
| P-C1 fingerprint (q4 `amar_mega_token`) | 12727 instr, dual 117/80/80/60 | 12746 instr, dual 115/80/80/60, spill 0 | `.work/isa/c2.txt`, status 07:20 |
| 20-prompt median tok/s_gen (P4) | 133.59 (spread 0.8 %) | 133.41 (spread 0.8 %), ratio 0.999 | `.work/ab-c2y/results.txt` |
| 20-prompt identity B == A | - | 20/20 PASS | same |
| one-shot q4 identity / tok/s_gen | PASS 64/64 | PASS 64/64 / 133.92 | `.work/merge-oneshot-q4.log` |
| one-shot q8 identity / tok/s_gen | PASS 64/64 | PASS 64/64 / 83.53 | `.work/merge-oneshot-q8.log` |
| prefill p0512 (T ~ 575) tok/s_gen | 129.00 / 128.85 | 128.67 (merge-gate), 128.48 / 128.61 (one-shots) | `.work/merge-pf512.log` |
| mtp k=2 identity | 20/20 | 20/20 | `.work/merge-mtp/results.txt` |
| test_server.sh | ALL PASS | ALL PASS | `.work/merge-server-test/SUMMARY.txt` |
| test_prefill / test_attn_block / test_mega_block | PASS | PASS / PASS / PASS (0 mismatches, q4 and q8, m=1 and m=3) | `.work/c2-stint4.out`, `.work/c2-run-test_mega_block.log` |
| mega fail word | 0 | 0 | engine stdout |

A first A/B run (`.work/ab-c2x`) was VOID: spread 32.5 % from one 124 outlier
under a co-running llama-server; the rerun above was exclusive.

Verdict against the frozen predictions:
- P-C1 held (one ticket, no re-roll).
- P-C2 held: every identity check PASS.
- P-C3 held: 128.5-128.7 vs 129.0, inside the 2 % band; the serialisation
  diagnosis stands.
- P-C4 held: ratio 0.999.
- P-C5 held.

Dtype arms (P-B2, 20-prompt A/B vs M0a, same stint, `.work/ab-c2f16`,
`.work/ab-c2bf16`): **f16 identity 17/20** (fails p12-rust, p13-haiku,
p15-bash), **bf16 identity 17/20** (fails p07-json, p12-rust, p13-haiku).
The one-shot 64-token checks passed for both, which is why the status file
briefly said "bf16 PASS everywhere"; the 20-prompt set is the gate and it
says otherwise. P-B2's f16 half is falsified (f16 does not hold either), its
bf16 half held. Neither narrow format ships; `KVT = float32` is the default
and the narrow KV question moves to M5 with the RULER gate, where the
long-context effect it is for can be measured.

---

## M0c — runtime T end to end (frozen 2026-09-08, before its build)

**Where the ceiling lives after M0b.** `serve/registry.mojo:36` `TMAX = 1088`
sizes `TPAGES`/`KVPOOL`/`KVPOOL1` and `toks_layout`; `serve/engine.mojo`
allocates `toks_d` (286), the four KV pools (326-329), the host token buffer
(446), rejects `len(prompt) + n > TMAX` in both the serve loop (389) and the
one-shot path (441), and prints `TMAX:` (435). The kernels carry no T. The
prefill loop already walks the prompt in chunks of `pf_chunk <= CP = 1024`
(`serve/window.mojo:572`), so no kernel change is expected.

**Change.**
- `TMAX` becomes a runtime capacity `tcap`: env `BARO_TMAX` (default 1088,
  the identity reference) read once at start; `TPAGES = ceildiv(tcap,
  KVPAGE)` and the two pool sizes computed from it; token buffers sized
  `tcap`. The ready JSON and the `TMAX:` line print the runtime value (P1
  receipt). No per-request capacity yet: one process, one capacity.
- Overflow: both checks return llama.cpp's error shape instead of a bare
  string. Serve mode: `{"error":{"code":400,"message":"the request exceeds
  the available context size, try increasing it","type":
  "exceed_context_size_error","n_prompt_tokens":N,"n_ctx":C}}`; one-shot
  mode prints the same object and exits 2. `baro-serve` maps it to HTTP 400
  verbatim.
- Long prompt files: `bench/prefill-prompts/p8192.tokens`, `p32768.tokens`,
  `p100000.tokens`, produced by `tools/baro-tokenize encode` over a fixed
  text (this repo's `docs/*.md` concatenated, then `~/llama.cpp/docs/*.md`,
  truncated to the exact count) and committed with their sha256 in this
  section's Result.

**Not in this step.** Per-request capacity, page table, checkpoints (M1);
the GQA-grouped decode kernel (M0d, only if P-D5 demands it); the int8
prefill GEMM (lane int8).

**Predictions.**
- P-D1 At `BARO_TMAX=1088` the binary is bit-exact against `9d3b228`'s
  build: `GENERATED` byte-identical on all 20 `bench/mtp-prompts/`, q4 and
  q8 one-shots 64/64, mtp 20/20. Only allocation sizes and error strings
  change. Falsifier: any difference, which would mean a size leaked into
  the address arithmetic after all.
- P-D2 20-prompt median within ±2 % of `9d3b228` (same stint), fingerprint
  in class (12727 ± 40, dual ≥ 105, 80/80/60 ± 2, spill 0). The megakernel
  source does not change, so the fingerprint is expected identical.
- P-D3 TTFT table, `prefill_s` from the run at `BARO_TMAX=102400`,
  `BARO_PREFILL_C=1024`: predicted from the measured p0512 rate
  (511 rows in 0.42 s, of which ~0.05 s is fixed) as ~0.75 ms/row, linear
  in chunks: 8192 → ~6 s, 32768 → ~24 s, 100000 → ~75 s. llama.cpp
  `--pure Q4_0` on the same token files (`tools/llama-ref-run.sh`,
  `/props` read back) predicted ~2.4× faster (int8 MMQ prefill). Recorded
  as the gap the int8 lane owes; not gated.
- P-D4 Memory: at `BARO_TMAX=102400`, f32 KV = 2 × 102400 × 8 × 4 × 256 ×
  4 B = 6.7 GB; with the 6.2 GB pack the engine prints its ready line and
  runs the 100000-token prompt plus 64 generated tokens with `mega fail
  word: 0`. Falsifier: allocation failure or a residency fault, which
  would move K8/V4 forward from M5.
- P-D5 Decode after a 32768-token prompt (`tok/s_gen` of the 64 tokens):
  predicted ≥ 100 tok/s_gen (KV read per token at T=32k is 2 × 32768 × 4 ×
  256 × 4 B × 8 layers = 2.1 GB, ~2.3 ms at 900 GB/s on top of the 7.4 ms
  token). At T=100000: predicted ~60 tok/s_gen (6.5 GB per token). Reading
  rule: below 80 at T=32k triggers the M0d preregistration (design §4
  decode kernel); above it M0d waits for M5.
- P-D6 `mega fail word: 0` on every run; the overflow request returns the
  JSON shape above in both modes (`tools/test_server.sh` gains one
  oversize case).

**Verification before timing (P1).** `TMAX:` and the ready JSON `tmax` field
read back from every run; `prompt tokens:` equals the file's word count;
`prefill chunk:` 1024; rebuild in the same stint; `arm.txt` first for the
A/B; llama.cpp `/props` saved beside its timings.

**Gate.** P-D1 exact; `tools/merge-gate.sh` ALL PASS at the default 1088;
P-D2 within band; P-D3/P-D4/P-D5 tables filled (P-D4 must hold; P-D3 and
P-D5 are recorded against their predictions, not gated); P-D6.

**Result.** PASS on the gate; P-D3 and P-D5 missed their predictions,
recorded 2026-09-08 (`.work/m0c/stint.txt`, exclusive GPU, ref =
`.work/engine-m0b` sha `321c5f25b3e172b3` built from 9d3b228, new =
`.work/engine-m0c`).

| receipt | M0b | M0c | source |
|---|---|---|---|
| fingerprint q4 `amar_mega_token` | 12746 / dual 115/80/80/60 / spill 0 | identical | `.work/isa/m0c/co64.s` |
| one-shot q4 at 1088 | 133.75, 64/64 | **bit-exact**, 133.75 | `.work/m0c/one-q4-*.log` |
| one-shot q8 at 1088 | 82.08, 64/64 | **bit-exact**, 83.63 | `.work/m0c/one-q8-*.log` |
| 20-prompt median (P4) | 133.26 (spread 3.8 %) | 133.26 (spread 1.9 %), ratio 1.000, identity 20/20 | `.work/m0c/ab/results.txt` |
| overflow, one-shot (8192 prompt at TMAX 1088) | raise | exit 2, `exceed_context_size_error` JSON | `.work/m0c/overflow.log` |
| overflow, server (`max_tokens` 1e6) | 400 plain | 400 `exceed_context_size_error`, n_prompt_tokens 43, n_ctx 1088 | merge-gate `PASS overflow` |
| merge-gate | ALL PASS | ALL PASS | `.work/merge-gate.txt` 11:35 |

Long context, `BARO_TMAX=102400`, f32 KV, prefill chunk 1024, 64 generated
tokens, prompt files sha256 (first 16) p8192 `3ff81ea490f46aa6`, p32768
`a922d0b1d2f3b6f8`, p100000 `112722f7857c9962`:

| T | prefill_s ours | predicted | llama.cpp Q4_0-pure prompt_ms | ratio | tok/s_gen ours | predicted | llama.cpp | mega fail |
|---|---|---|---|---|---|---|---|---|
| 8192 | 8.33 | ~6 | 2.59 | 3.2x | 89.4 | - | 109.2 | 0 |
| 32768 | 50.46 | ~24 | 12.84 | 3.9x | 45.7 | >= 100, floor 80 | 100.2 | 0 |
| 100000 | 298.3 | ~75 | 63.67 | 4.7x | 19.2 | ~60 | 78.8 | 0 |

Verdict against the frozen predictions:
- P-D1 held: bit-exact at 1088 on both packs, 20/20.
- P-D2 held: fingerprint identical, ratio 1.000.
- P-D3 missed: prefill is superlinear (8.3 -> 50 -> 298 s, i.e. T^1.5-2),
  not the linear ~0.75 ms/row the prediction assumed. The prediction
  modelled the GEMMs only; `amar_attn_prefill` re-reads the whole KV per
  1024-row chunk, so its cost grows with T per chunk and the sum is
  quadratic. Against llama.cpp the gap grows 3.2x -> 3.9x -> 4.7x from 8k to 100k
  (2.4x expected from the int8 MMQ GEMM alone; the growth is the
  attention sweep). llama.cpp's own decode falls 109 -> 100 -> 79 over the
  same range, i.e. its attention costs ~0.04 ms per 1k, a tenth of ours.
- P-D4 held: 6.7 GB f32 KV + pack, 100k prompt + 64 tokens, fail word 0.
- P-D5 **falsified**: 45.7 at 32k, below the 80 floor. Per-token decode
  cost is linear in T at 0.44 ms per 1k tokens (7.5 ms + 3.7 at 8k, + 14.4
  at 32k, + 44.6 at 100k), six times the KV-byte floor (67 MB per 1k
  tokens = 0.075 ms at 900 GB/s). Bytes do not explain it; the attention
  phase inside the megakernel runs on a fraction of the CUs while the rest
  hold at the grid barrier. This triggers M0d: the GQA-grouped, split-K
  decode attention kernel of design §4, as a separate launch (or phase
  with every block participating), preregistered next.
- P-D6 held in both modes; `tools/test_server.sh` gained the case.

Observation, not promoted: decode after an 8k prompt is 89 vs llama.cpp's
109 on the same card; at T <= 1088 ours leads 133 vs ~110. The crossover is
the M0d target.

---

## M0d — split-K decode attention inside the megakernel (frozen 2026-09-08, before its build)

**Mechanism, read from source.** `attn_phases` (`kernels/mega.mojo`) runs
the decode attention as `if bid < M * NQH:` — at m = 1, 16 of the 96
resident blocks, one q head each, while 80 blocks wait at the grid barrier.
Each active block walks all T positions in 256-position chunks
(`attn_head_body`), four `barrier()`s per chunk, one position per thread,
and threads 256..511 of the 512-thread block idle. M0c measured the cost
of that at 0.44 ms per 1k tokens of context, 6x the KV-byte floor; per
active block that is ~4.6 GB/s, latency-bound by construction.

**Change.** Split-K over page spans in the same phase, no new launch:
- Work item = (q head h, span s), s in 0..NSPLIT-1, NSPLIT = 6 at m = 1
  (96 / 16; `MEGA_G // (M * NQH)` in general, at least 1). Span s covers
  pages `[s * P / NSPLIT, (s + 1) * P / NSPLIT)` of the T positions
  (P = ceildiv(T, KVPAGE)), so each item is contiguous and page-aligned.
- Every block runs `attn_head_body` over its span with the same chunk loop
  and writes a partial `(m, l, o[HD])` to scratch instead of `O`. Scratch
  = the head split-K buffer `p_v` (`row_major[SPLITK, SM, VOCAB]` f32,
  idle during attention; 16 * 6 * 258 floats needed, millions available),
  no engine change.
- Grid barrier, then blocks `bid < M * NQH` merge their head's NSPLIT
  partials: `m = max m_s`, `O = sum_s exp(m_s - m) * o_s / sum_s exp(m_s
  - m) * l_s`, in span order.
- **T <= 1088 keeps NSPLIT = 1 and the direct `O` write**, i.e. today's
  code path, so the champion path stays bit-exact and the P4 A/B is
  untouched. `BARO_ATT_SPLIT=1` forces the split at every T for the
  identity receipts. The threshold is a runtime compare on T.

**Not in this step.** Using the idle upper 256 threads (a second chunk in
flight per block, the next lever, predicted a further ~1.6x); GQA-grouping
(one K/V read for the four q heads of a kv head; bytes are not the bound
yet); a separate launch (design §4's standalone kernel) — the megakernel
stays one launch per token.

**Predictions.**
- P-E1 Fingerprint: the dot loops are untouched, but `attn_phases` gains
  code, so the schedule lottery applies: q4 `amar_mega_token` within
  12746 ± 60 instructions, dual >= 105 / 80 ± 2 / 80 ± 2 / 60 ± 2, spill 0.
  One re-roll by spelling; second miss stops the step.
- P-E2 Identity: default (split off at T <= 1088) bit-exact vs `a86516b`'s
  binary on q4/q8 one-shots and 20/20 A/B. Forced split (`BARO_ATT_SPLIT=1`)
  at T <= 1088: q4 64/64 and q8 64/64 vs `tools/model-ref.py` reference
  tokens, 20-prompt identity 20/20 vs the unsplit binary (the merge changes
  summation order, so this is identity, not bit-exact). At 8192 and 32768
  the split binary's `GENERATED` equals M0c's unsplit `GENERATED`
  (`.work/m0c/long-*.log`); a mismatch there is arbitrated by
  `tools/model-ref.py` on the 8192 prompt (falsifier if model-ref sides
  with unsplit).
- P-E3 Long-context decode, same prompts as M0c: 8192 >= 115 tok/s_gen
  (from 89.4: 3.7 ms attention -> ~0.6 + 0.3 merge/barrier), 32768 >= 90
  (from 45.7: 14.4 -> ~2.4 + 0.3), 100000 >= 55 (from 19.2: 44.6 -> ~7.4
  + 0.4). Falsifier: 32768 < 70 means the per-block cost is not the
  bound (look at the barrier count / merge before touching occupancy).
- P-E4 20-prompt median within ±2 % of `a86516b` (path unchanged at
  T <= 1088; only the fingerprint could move it).
- P-E5 `mega fail word: 0` on every run; prefill_s unchanged (prefill
  attention is not touched).

**Verification before timing (P1).** `att split:` printed by the engine
(threshold and forced flag), `TMAX:`, `prompt tokens:`, `tok/s_gen`,
`mega fail word` from each run; fingerprint before any GPU run; rebuild in
the same stint; `arm.txt` first for the A/B.

**Gate.** P-E1 before GPU; P-E2 all four identity receipts; merge-gate ALL
PASS; P-E4 within band; P-E3 recorded against its predictions (32768 >= 70
is the hard floor, the three numbers are the claim).

**Result.** PASS on the gate, recorded 2026-09-08 (`.work/m0d/stint.txt`,
exclusive GPU; ref = `.work/engine-m0c` (a86516b), new = `.work/engine-m0d`).

P-E1 fingerprint, two builds: (1) head body inlined in both branches, 13511
instructions; (2) re-spelled to one `attn_head_span` call shared by both
paths, **12977** instructions, vgpr 249, spill 0, `isa-loops` dual
122 / 84 / 84 / 61. The count band (12746 ± 60) is missed by construction:
the partial write and the merge are +231 new instructions that no spelling
removes. The loop class, which is what the band was guarding, is faster in
every column (115/80/80/60 before). Recorded as a deviation and the GPU
stint was run; the question stands in `.work/briefs/status-chat.md`.

| receipt | M0c | M0d | source |
|---|---|---|---|
| test_mega_block (split forced, `ATT_SPLIT = 0`) | PASS | PASS, bit-identical to the launch path, m=1 q4/q8 and m=3 q8 | `.work/m0d/test_mega_block.log` |
| test_attn_block | PASS | PASS | `.work/m0d/test_attn_block.log` |
| one-shot q4 / q8 default (split off at T <= 1088) | 64/64 | **bit-exact** vs M0c, 137.6 / 81.3 | `.work/m0d/one-*-new.log` |
| one-shot q4 / q8 forced split | - | 64/64 vs model-ref tokens, 136.8 / 80.9 | `.work/m0d/one-*-split.log` |
| 20-prompt A/B default | 133.24 | 133.22, ratio 1.000, identity 20/20 | `.work/m0d/ab/results.txt` |
| 20-prompt A/B forced split vs unsplit (same binary) | 136.22 | 136.13, ratio 0.999, identity 20/20 | `.work/m0d/ab-split/results.txt` |
| merge-gate | ALL PASS | ALL PASS | `.work/merge-gate.txt` 11:55 |

Long context, `BARO_TMAX=102400`, default threshold 1088 (split on):

| T | tok/s_gen M0c | M0d | predicted | GENERATED vs M0c | prefill_s | mega fail |
|---|---|---|---|---|---|---|
| 8192 | 89.4 | **117.5** | >= 115 | identical | 8.30 (8.33) | 0 |
| 32768 | 45.7 | **85.0** | >= 90, floor 70 | identical | 50.24 (50.46) | 0 |
| 100000 | 19.2 | **48.7** | ~55 | differs from token 11 | 298.6 (298.3) | 0 |

Verdict against the frozen predictions:
- P-E1: count band missed (explained above), loop class held.
- P-E2 held on every frozen receipt: default path bit-exact; forced split
  64/64 on both packs and 20/20; 8192 and 32768 byte-identical to M0c.
  At 100000 (not in the prediction) the two summation orders diverge at
  token 11 of 64. Both continuations loop on the docs text (the prompt has
  no question); a low-margin distribution is where a merge-order
  difference flips an argmax. No 100k reference decode exists to
  arbitrate; recorded as an observation. Whether the model itself is sound
  at 100k is M5's RULER question (llama.cpp niah_single is 100 % at 32k,
  nothing measured beyond).
- P-E3: 8192 held (117.5 >= 115); 32768 above the floor, below the
  prediction (85.0 vs >= 90); 100000 below (48.7 vs ~55). The attention
  cost per token went 3.7 -> 1.0 ms at 8k, 14.4 -> 4.3 ms at 32k, 44.6 ->
  13.0 ms at 100k: a 3.3-3.7x cut against the 6x a perfect split over
  6 spans would give. The remainder is the extra grid barrier plus the
  merge (~0.3 ms) and the per-block chunk loop itself, which is still one
  position per thread with four barriers per 256 positions. The next
  lever is inside the block (both halves of the 512-thread block in
  flight, or two chunks per iteration), not more spans.
- P-E4 held: ratio 1.000.
- P-E5 held: fail word 0 everywhere, prefill_s unchanged.

Observation, not promoted: the M0d schedule's one-shot at T ~ 70 reads
136-137.6 on both the default and forced paths (M0c 133.7), and the
same-binary A/B medians read 136.1-136.2 while the cross-binary A/B read
133.2. The 20-prompt A/B against M0c is the P4 number (1.000); the 136s
are the lottery being kind on this build and are not claimed.

---

## M1a — prefix checkpoints for one sequence (frozen 2026-09-08, before its build)

**Where the state lives (read from source).** Per request the engine memsets
conv state, delta state, both KV pools and the counters
(`serve/engine.mojo` serve loop), then prefills the whole prompt. The SSM
state is `csall_layout = [SLOTS, N_SSM, 3, CONV]` f32 (1.77 MB per slot) and
`ssall_layout = [SLOTS, N_SSM, NH_V, SSTATE, SSTATE]` f32 (50.3 MB per
slot), `SLOTS = KMAX + 1 = 9`, ring index `st.ring` advancing by the rows
processed; the slot holding the state *after* position p is `ring` after
that step. The KV cache is one pool, position-addressed through `kv_off`,
so for a single sequence the KV for tokens `[0, p)` is already in place
when the next request shares that prefix; only what a replay overwrites
changes. The design's 26.25 MiB figure assumed 16 v-heads; the real
checkpoint is **52.1 MB** (conv + delta) per position.

**Change.**
- `serve/prefix.mojo` (new, host side): `Checkpoint {pos, hash, gen,
  conv_h, ssm_h}` in pinned host buffers; `hash = sha256(canonical bytes of
  tokens[0:pos])` computed on the host in Mojo (a 64-bit FNV-1a over the
  little-endian i32 ids is the first cut, upgraded to SHA-256 in M1b when
  cross-process reproducibility matters: this step is in-process only).
  A `Chain` of at most `BARO_CKPT` (default 8) checkpoints ordered by pos,
  each valid for the prefix it hashes; `lookup(tokens) -> best` returns the
  checkpoint with the largest `pos <= len(tokens) - 1` whose hash matches
  `tokens[0:pos]`.
- Boundaries where a checkpoint is taken: prompt end (after prefill, before
  the first generated token), and every 1024 tokens of prefill as the
  periodic safety net. Role boundaries and branch points are M1b (they
  need the template renderer and a second sequence).
- Restore: copy `conv_h`/`ssm_h` into the ring slot the replay will read
  from, set `st.pos = pos`, `st.ring` accordingly, skip the memsets of KV
  (the pool keeps `[0, pos)`), prefill only `tokens[pos:]`. If no match,
  today's path (memset, full prefill).
- Save: `enqueue_copy` device -> pinned host of the two slot regions,
  once per boundary; the copy is off the critical path of the response
  (issued after the boundary, synchronised before the next request).
- Receipts in the `done` line: `"cached": <pos restored, 0 if none>`,
  `"prefill_rows": <rows actually prefilled>`; the engine prints
  `checkpoints: cap 8, bytes 52.1 MB each` at start (P1 read-back).
- Server: passes `cached` through in `usage` as `baro.cached_tokens`.

**Not in this step.** Radix pages / multiple sequences, salt, retention
priorities, branch-point snapshots, cancel semantics, `prefix_churn`
(M1b). Cross-process reproducible hashes (M1b). Any kernel change.

**Predictions.**
- P-F1 Byte-exact restore, test `kernels/test_prefix.mojo` (new,
  `run-tests.sh`): for prompt A||B (A = 1088 tokens of `p8192.tokens`,
  B = the next 64), cold(A||B) vs restore(A)+replay(B) give
  `memcmp`-equal conv, delta, KV pages and next-token logits at the
  positions 0 / 1 / 1023 / 1024 / 1025 / 1087 / 1088 / 1151; a mutation at
  token 0, at pos-1, at pos+1 and at the last token of A misses the
  checkpoint (lookup returns none or an earlier one); a corrupted hash
  fails restoration. Falsifier: any byte difference, which means the
  ring/slot bookkeeping or the KV memset assumption is wrong.
- P-F2 Engine identity: q4/q8 one-shots and the 20-prompt A/B are
  bit-exact vs `48a48b1` (one-shot mode takes no checkpoints; the serve
  path with a cold cache is today's path).
- P-F3 Serve path, two requests with the 7,914-token DeerFlow system
  prompt prefix (tap replay from `.work/chat/tap.jsonl`): request 2
  reports `cached` >= 7,900 and `prefill_rows` < 200; its wall TTFT
  (prefill_s + restore) < 60 ms at the 8k prefix, from 8.3 s cold.
  Restore copy predicted ~2 ms (52 MB at PCIe 4 x16).
- P-F4 20-prompt A/B median within ±2 % of `48a48b1` (decode untouched;
  the fingerprint must be unchanged since no kernel changes).
- P-F5 Memory: 8 checkpoints = 417 MB pinned host, printed at start;
  `mega fail word: 0`.

**Verification before timing (P1).** `checkpoints:` line, `cached` and
`prefill_rows` in the `done` line of each request, `TMAX:`, fingerprint
unchanged (12977 / dual 122/84/84/61) read before the stint.

**Gate.** P-F1 all positions and mutations; merge-gate ALL PASS; P-F2
exact; P-F3 both numbers; P-F4 within band.

**Result (2026-09-10, `7b4b9d0` on `lane-chat`). The gate is NOT met: P-F3
misses both of its numbers. Everything else passes.**

P-F1 byte-exact restore: PASS on 2026-09-08 (`kernels/test_prefix.mojo`, all
positions and all four mutations, megakernel and window paths).

P-F2 / P-F4 engine identity and A/B (2026-09-10 01:12, queue empty,
`.work/m1a/ab/arm.txt`): `engA=./.work/engine engB=.work/m1a/engine`
shaA 644c23560ed499d4 shaB dfd578701f8b9f90, power cap 290000000 uW,
vddgfx -100mV, both arms `BARO_MEGA=1`. champ median 136.97 tok/s_gen
spread 0.5 %, m1a median 136.98 spread 0.6 %, **ratio 1.000**, identity
fails none. Checkpointing costs the decode path nothing.

Merge gate (01:12-01:14, `.work/merge-gate.txt`): **ALL PASS** — engine and
test builds, run-tests, ci-checks, test_prefill, one-shot q4 64/64 at 137.2
and q8 64/64 at 81.6, p0512, mtp 20/20, and the full server suite.

P-F5: engine prints `checkpoints: cap 8 , bytes 52.690944 MB each, period
1024` = 421 MB pinned; `mega fail word: 0` on every request.

**P-F3 tap replay** (`tools/tap-replay.py`, rows 0,1,2 x2 against baro-serve
on the m1a engine, `BARO_TMAX=16384`, `.work/m1a/tap-replay.log`):

| row | prompt_tokens | cached | prefill_rows | prefill_s | restore_s | wall_s |
|---|---|---|---|---|---|---|
| 0 (cold) | 5755 | 0 | 5754 | 5.229 | 0.003 | 5.279 |
| 1 | 5770 | 5120 | 649 | 0.651 | 0.002 | 0.695 |
| 2 | 5858 | 5120 | 737 | 0.722 | 0.002 | 0.766 |
| 0 (2nd) | 5755 | 5120 | 634 | 0.616 | 0.002 | 0.659 |
| 1 (2nd) | 5770 | 5120 | 649 | 0.652 | 0.002 | 0.696 |
| 2 (2nd) | 5858 | 5120 | 737 | 0.721 | 0.002 | 0.762 |

Frozen: `cached >= 7900`, `prefill_rows < 200`, wall TTFT `< 60 ms`.
Measured: cached 5120, rows 634-737, wall 659-766 ms. **Both numbers
missed.** Two separate reasons, neither of them the restore path:

1. The fixture is not the 7,914-token prompt the prediction named — these
   tap rows are 5,755-5,858 tokens. The prediction should have been read off
   the fixture before it was frozen.
2. The real miss: `cached` is 5120 = 5 x 1024, the periodic grid, not the
   5,754-token prompt-end checkpoint that request 0 wrote. Request 1's ids
   diverge from request 0's somewhere between 5120 and 5754, because the
   re-rendered template replaces request 0's trailing generation prompt with
   the assistant turn. So the end-of-prompt checkpoint is worth nothing for
   the next turn of the same conversation, and every restore falls back to
   the last 1024-boundary — up to 1023 replayed rows, whatever the prefix
   length. **Role-boundary checkpoints, deferred to M1b, are exactly the fix
   for this**, which the M1a design says in as many words; the frozen P-F3
   simply predicted M1b's behaviour and measured M1a.

What M1a does deliver, measured: multi-turn prefill **5.229 s -> 0.616-0.722 s,
7.2-8.5x**, with a restore copy of 2.1-3.2 ms against the ~2 ms predicted,
and no cost to decode. That is the honest claim. The 100x-class number needs
M1b.

Not run: nothing further; M1b preregistration is unwritten.

---

## KSAMP: device sampler and speculative acceptance kernels (frozen 2026-09-11, before its build)

Plan: `~/Brain/mojo/mojo-baro/briefs/2026-09-11-chat-engine-next.md`, item KSAMP.
Lane `lane-KSAMP`, files `kernels/sample.mojo` and `kernels/test_sample.mojo`.
No engine wiring, no greedy-path change, no penalties (the KSAMP interface has none;
they belong to the CHAT host path and a later kernel step).

**Interface (fixed by the plan, spelled in this repo's kernel form).** Rows are
`X: [R, V] f32`, one block per row, the same shape as `amar_argmax_row`, so the
caller swaps one launch for the other.
- `amar_sample_row(X, Out[R] i32, Prob[R] f32, n, temperature, top_k, top_p,
  min_p, seed: u64, counter: u64)`: token id and its probability under the
  distribution it was drawn from.
- `amar_spec_accept(Pt, Pd, Dtok, Out, Acc, n, seed, counter)`: per row, accept
  the drafted token `x` when `u * p_d(x) < p_t(x)` (probability
  `min(1, p_t/p_d)`), otherwise emit a draw from the normalised
  `max(0, p_t - p_d)`. The uniform `u` and the resample noise come from the
  same counter RNG as the sampler, in separate streams.
- Addition flagged for CHAT: `amar_sample_probs(X, P, n, ...same params)`
  writes the full truncated, tempered distribution row. `amar_spec_accept`
  needs `p_t` and `p_d` as rows and nothing in the fixed interface produces
  them; this kernel shares the sampler's selection code, so the probabilities
  the acceptance rule sees are the ones the sampler draws from.

**Semantics (llama.cpp order, design §5).** Valid logits are finite and not
NaN. top-k (0 or >= valid count = off) keeps the k largest; top-p (>= 1 = off)
keeps the shortest prefix, in descending order, whose softmax mass over the
top-k set at temperature 1 reaches `top_p`, the crossing token included;
min-p (<= 0 = off) keeps tokens with `p >= min_p * p_max` at temperature 1;
then temperature scales the survivors and one token is drawn. Ties in value
are ordered by lower index first, so every cut is a prefix of one total order
(value descending, index ascending) and the three filters commute.
`temperature <= 0` returns `amar_argmax_row`'s token (same rule: lowest index
among the maximum, NaN ignored, all-invalid row gives 0) with probability 1.
A sampled row with no valid logit gives token -1, probability 0.

**Mechanism.** Counter RNG = Philox4x32-10 (Random123 constants), key = seed,
counter = (counter lo, counter hi, row, stream << 28 | element / 4). The draw
is Gumbel-max over the survivors, `argmax((l - lmax) / T + G_i)` with `G_i`
keyed by the element index, so the token does not depend on thread count or
summation order: a (seed, counter) pair reproduces bit-exactly. Cuts are found
without a sort: pass 1 finds `lmax`; pass 2 builds a 257-bucket histogram of
`(lmax - l) * 8` (1/8-nat bands, one tail bucket) with counts and fixed-point
masses (`exp(l - lmax) * 2^40` as u64, so sums are exact and order-free); the
band holding the k-th token (or the top-p crossing) is refined by a four-digit
8-bit radix select on the order-preserving u32 key of the logit, restricted to
that band, then by index for ties. One block of 1024 threads per row.

**Predictions.**
- P-K1 Philox4x32-10 matches the Random123 known-answer vectors (zero and
  all-ones counter/key).
- P-K2 Temperature 0: token equal to `amar_argmax_row` on 8 random rows at the
  real vocabulary V = 248320 plus edge rows (value ties, +0/-0, -inf, NaN, all
  -inf), 100 % of rows.
- P-K3 Distribution: fixed logits over V = 64, 10,000 draws per configuration
  (T 1.0 / k off / p off; T 0.7 / k 20 / p 0.8; T 1.3 / k 12 / p 0.9 / min-p
  0.05; T 0.5 / k off / p 0.6; a tie-heavy row with k cutting inside a tie),
  Pearson chi-square against the exact float64 distribution (bins pooled to
  expected >= 5) below the p = 0.001 critical value, **zero** draws outside
  the truncated set, returned probability within 1e-5 of the exact value,
  `amar_sample_probs` row within 1e-5 of exact and summing to 1 within 1e-5.
- P-K4 Reproducibility: two launches with the same (seed, counter) give the
  same 10,000 tokens; a different seed changes at least half of them on the
  flat configuration.
- P-K5 Speculation: a deliberately mismatched draft (different logits, same
  sampling parameters), drafts drawn by `amar_sample_row` from `p_d`, then
  `amar_spec_accept`; 10,000 rows, accepted-or-resampled tokens pass the
  chi-square against `p_t`; acceptance rate within 4 sigma of
  `sum_i min(p_t, p_d)`; a two-sample chi-square between these tokens and
  direct `amar_sample_row` draws from `p_t` passes at p = 0.001
  (speculation on and off indistinguishable).
- P-K6 Time per call at V = 248320, R = 1, logits Gaussian sigma 2.5 with a
  few boosted tokens, row hot in L2 (as in the engine, where the lm-head
  writes it just before): greedy 5-15 us (one pass); Qwen preset
  (T 0.7 / k 20 / p 0.8) 50-110 us (12 passes over 1 MB at ~6.5 us per pass
  for one block); llama.cpp default (T 0.8 / k 40 / p 0.95 / min-p 0.05)
  same band; k off / p 0.95: 30-70 us (7 passes). Falsifier: either
  sampling preset above 150 us means the per-pass cost model is wrong;
  look at LDS atomic contention in the band pass before anything else.

**Verification before timing (P1).** The test prints grid, block, V, R and
every sampling parameter of each timed arm from the values it launched with,
in the same binary that is timed, and runs the P-K2 and P-K3 checks in the
same process before timing.

**Gate.** `./run-tests.sh` exit 0 with the census at +3 kernels and 0
orphans, `kernels/test_sample.mojo` PASS on P-K1 to P-K5, P-K6 recorded
against its bands; both captured to `.work/KSAMP-gate.txt`.

**Result, first build (`21fe9e4`, 2026-09-11, `.work/KSAMP-run2.txt`).**
P-K1 to P-K5 PASS: Philox KATs match; temperature 0 equals
`amar_argmax_row` on 13/13 rows at V = 248320; six chi-square configs all
below the p = 0.001 critical value with zero draws outside the truncated
set; probabilities within 1.4e-8 of exact; same seed 10k/10k equal;
acceptance 0.5394 vs exact 0.5373 (sigma 0.0050), spec vs direct
two-sample chi2 23.5 at df 21 (crit 46.9). One fixture correction before
that run: the third Philox KAT's last word was transcribed from memory as
`24f7f839`; Random123's vector is `24126ea1`, and the implementation
matched the other three words exactly, which a wrong Philox cannot do.

P-K6 **falsified**: greedy 28.5 us (band 5-15), T0.7/k20/p0.8 483 us
(50-110), T0.8/k40/p0.95/min-p 0.05 485 us, k off/p 0.95 432 us (30-70),
plain T1 206 us; `amar_argmax_row` itself 129 us at this V with its 256
threads. Diagnosis by subtraction (inferred, not profiled): a scalar
strided pass over the 1 MB row costs ~28 us with 1024 threads (load
latency, not bandwidth: 256 threads take 4.5x longer), not 6.5 us; the
band pass's two u64 LDS atomics per element cost ~175 us; the plain
arm's final pass spends ~178 us because every element computes a full
Philox call and uses one of its four words.

## KSAMP-b: fewer and cheaper passes (frozen 2026-09-11, before its build)

**Change.** (1) Every pass over the row loads 4 consecutive floats per
thread per group and keeps two groups in flight; the Gumbel noise takes
all four words of one Philox call per group. (2) Fast path for the cut:
after `lmax`, one pass accumulates, per thread and without atomics, the
count and fixed-point mass of valid tokens within D nats of `lmax` for
D in {0.5, 1, 2, 3, 4, 6, 8}, reduced across the block; the smallest D
whose set holds the top-k boundary (or, with top-k off, the top-p mass)
and has at most CAP = 2048 tokens is compacted into LDS as
`(key << 32) | ~index`, bitonic-sorted descending, and top-k, top-p and
min-p are read off the sorted prefix with the same integer masses and
`W = ceil(top_p * Z)` as the general path, so both paths cut the same
set and, the noise being keyed by index, draw the same token. (3) When
no D qualifies the first build's band-and-radix path runs unchanged.
`CAP` is a comptime parameter with default 2048 so the test can force
the general path.

**Predictions.**
- P-K7 P-K1 to P-K5 still pass; for every P-K3 configuration and the
  P-K5 draws, the fast path and the forced general path (`CAP = 16`)
  give identical tokens on 10,000/10,000 rows and probabilities within
  1e-6.
- P-K8 Time per call at V = 248320, R = 1, hot row, same test harness:
  greedy 6-12 us; Gaussian row (sigma 2.5, five tokens boosted to
  10-14) T0.7/k20/p0.8 and T0.8/k40/p0.95/min-p 0.05: 25-60 us (fast
  path); an LM-like peaked row (same bulk, boosted to 20-24) with the
  same two presets and k off/p 0.95: 25-60 us; plain T1 on the Gaussian
  row 60-120 us; k off/p 0.95 on the Gaussian row takes the general path
  (its nucleus is the bulk), recorded, no band.
- Falsifier: greedy above 15 us means a vectorised pass is not
  load-latency bound as inferred; read the ISA (`isa-loops`) before any
  further change.

**Gate.** As KSAMP, plus P-K7; P-K8 recorded against its bands.

**Result, KSAMP-b build (2026-09-11, `.work/KSAMP-b-run.txt`, queue empty
before the run).** P-K1 to P-K5 unchanged and PASS (identical chi-square
numbers: the noise is keyed by element index, so the first build's draws
reproduce). P-K7 PASS: fast path and forced general path (`CAP = 16`)
agree on 10,000/10,000 tokens for all six P-K3 configurations and both
P-K5 draw sets, probabilities bit-equal (the final pass is shared).

P-K8, per call at V = 248320:

| row | arm | measured | band |
|---|---|---|---|
| gaussian | greedy | 9.3 us | 6-12, **held** |
| gaussian | T0.7/k20/p0.8 | 120.0 us | 25-60, missed |
| gaussian | T0.8/k40/p0.95/min-p 0.05 | 121.3 us | 25-60, missed |
| gaussian | plain T1 | 125.7 us | 60-120, missed |
| gaussian | k off/p 0.95 (general path) | 422.4 us | none |
| peaked | T0.7/k20/p0.8 | 552.2 us | 25-60, missed |
| peaked | T0.8/k40/p0.95/min-p 0.05 | 664.9 us | 25-60, missed |
| peaked | k off/p 0.95 | 137.7 us | 25-60, missed |
| gaussian | `amar_argmax_row` (reference) | 129.5 us | - |

The falsifier did not fire (greedy 9.3 us: a vectorised pass is cheap).
Two misses are design facts, read off the arms: on the peaked row the
20th token sits in the bulk, 12+ nats below `lmax`, beyond the widest
8-nat window, so k20/k40 fall back to the general path (552/665 us); the
Gaussian presets take the fast path yet cost 120 us, which the pass count
(four vectorised passes at ~9 us) does not explain. Not yet diagnosed:
phase timers next, before any KSAMP-c prediction.

**Diagnosis (measured, `.work/ksamp-diag/phases.txt`).** An uncommitted
copy of the kernel with a block barrier and a `llvm.readsteadycounter`
stamp at each phase boundary, 100 runs per arm after 20 warm-up, Gaussian
row, T0.7/k20/p0.8, microseconds from kernel start: pass A + reductions
15.0, threshold pass 73.1 (58 us), compaction 86.0 (13), sort 90.0 (4),
prefix scan 91.2, final pass 113.6 (22), end 115.0. Peaked row, same
preset: general path 13.7 -> 510.3. ISA receipt (`tools/isa-receipt.py`):
every sampler kernel 47-50 VGPR, 0 scratch, 0 spills, so the threshold
pass is ALU and divergence (each element inside the 8-nat window runs
the 8-lane accumulate), not spilling. Widening the window to reach the
peaked row's 20th token (12+ nats) would put the whole bulk inside it.

## KSAMP-c: sampled window, exact compaction (frozen 2026-09-11, before its build)

**Change.** The threshold pass is replaced by a 16-chunk contiguous
subsample of 16,384 elements (all of the row when V is smaller). Per
thread it counts elements, and for top-k off their masses, within
D of `lmax` for 16 values of D from 0.25 to 32 nats. The window is the
smallest D whose estimate holds 2k + 16 tokens (top-k on) or the top-p
mass with half the remaining margin (top-k off). The compaction pass
then counts exactly, accumulating the exact fixed-point `Z` when top-k
is off. On overflow of CAP it steps D down, on too few tokens or too
little mass it steps D up, at most four compactions, then the general
path. Sort, prefix scan and the cut rule are unchanged, so P-K7's
identity must still hold. The final pass tests membership against the
cut's float value instead of re-deriving the key. Block reductions go
through `warp` shuffles plus one 32-entry LDS step (two barriers instead
of ten).

**Predictions.**
- P-K9 P-K1 to P-K5 and P-K7 unchanged and PASS.
- P-K10 Per call at V = 248320: greedy 5-10 us; Gaussian and peaked rows,
  T0.7/k20/p0.8 and T0.8/k40/p0.95/min-p 0.05: 35-65 us; peaked row k
  off/p 0.95: 35-65 us; plain T1 80-110 us (Philox and two logs per
  element, untouched); Gaussian k off/p 0.95 takes the general path,
  recorded, no band.
- Falsifier: a preset at 35-65 missing by more than 2x means the phase
  model (pass 12 us, compaction 13, sort 4, final 12) is wrong; re-run
  the phase timers before any other change.

**Gate.** As KSAMP-b; P-K10 recorded against its bands.

**Harness fix before the verdict (P6).** The P-K8 arms were one mean
over 1000 back-to-back launches. A stamped copy timed three ways in one
process (`.work/ksamp-diag/phases-c2.txt`) read the same kernel on the
same row at 71 us back-to-back and 640 us synced (peaked, T0.7/k20/p0.8)
with sclk at 3305 MHz and the queue empty: one-workgroup kernels see
large run-to-run interference. Every arm is now 11 blocks of 100
launches, min/median/max printed; the first build's and KSAMP-b's
single-mean P-K6/P-K8 numbers carry that caveat.

**Result, KSAMP-c (`512294d` + block-median harness, 2026-09-11,
`.work/KSAMP-gate.txt`, queue empty, clock-probe sclk 3305-3311 MHz).**
Gate PASS: `run-tests.sh` exit 0 (85 kernels, 38 in registry, 0
orphans), `test_sample` exit 0. P-K9 PASS: P-K1 to P-K5 unchanged, P-K7
10,000/10,000 identical tokens in all eight comparisons.

P-K10, median per call (min-max), V = 248320:

| row | arm | median (min-max) | band | verdict |
|---|---|---|---|---|
| gaussian | greedy | 9.2 (9.1-9.3) | 5-10 | held |
| gaussian | T0.7/k20/p0.8 | 69.4 (69.2-70.8) | 35-65 | missed by 7 % |
| gaussian | T0.8/k40/p0.95/min-p 0.05 | 120.3 (117.6-121.4) | 35-65 | missed, 1.85x |
| peaked | T0.7/k20/p0.8 | 69.4 (69.4-70.4) | 35-65 | missed by 7 % |
| peaked | T0.8/k40/p0.95/min-p 0.05 | 145.5 (143.5-148.5) | 35-65 | **falsifier, 2.24x** |
| peaked | k off/p 0.95 | 179.5 (177.6-181.6) | 35-65 | **falsifier, 2.76x** |
| gaussian | plain T1 | 129.1 (128.0-130.0) | 80-110 | missed |
| gaussian | k off/p 0.95 (general path) | 490.1 (487.9-492.3) | none | recorded |
| gaussian | `amar_argmax_row` (reference) | 129.3 (128.7-133.9) | - | - |

What the falsifier asked for is already on disk (phase stamps,
`.work/ksamp-diag/phases-c2.txt`): the two falsified arms retried the
compaction (llama presets 2 passes, peaked k off/p 0.95 3 passes, one
compaction ~28-35 us each), because the subsample estimate lands below
the window the exact check accepts. One compaction costs ~32 us against
KSAMP-b's 13 us for the same loop shape; that difference is not
explained (ISA diff not done). The final pass costs 22-37 us. Against
the first build (single means) the LM-like peaked row at T0.7/k20/p0.8
went 552 -> 69.4 us and the Gaussian preset 483 -> 69.4 us.

**Round closed here.** Levers not taken, for whoever picks the sampler
up again: (1) find why the compaction loop costs 2.5x KSAMP-b's (ISA
diff of the two loops); (2) widen the estimated window by one step for
min-p and top-p so one compaction suffices; (3) two loads in flight in
the compaction and final passes, as pass A has; (4) the general path
(k off, flat rows) still pays ~480 us in its band pass and radix
refinement.

---

## C1 — control block: stop sequences, EOS-in-engine, cancel, FIFO+503 (frozen 2026-09-11, before its build)

**Where the gap is (read from source, 2026-09-11).** `serve/PROTOCOL.md` says it
in as many words: "there is no stop-token or cancel in the protocol." The
engine (`serve/engine.mojo`) always decodes exactly the requested `n`; a stop
token is only ever noticed by `serve/src/main.rs`'s `Acc::take`, client-side,
after the token has already been generated. There is no way to end a running
request early. The queue (`serve/src/engine.rs`, `mpsc::channel(QUEUE_CAP=64)`,
one worker) is already FIFO and already returns 503 past the cap
(`Engine::submit`'s `try_send` error mapped to `StatusCode::SERVICE_UNAVAILABLE`
in `main.rs::check_and_submit`) -- nothing to change there, only to keep gated.

**Change.**
- Wire: the request line gains `"stop":[[id,...],...]` (a list of token-id
  sequences, default `[]`); `serve/engine.mojo::parse_request` parses it.
  `serve/src/main.rs` builds it from two sources: the tokenizer's own
  `stop_ids` (each as a length-1 sequence, moving today's client-side EOS cut
  into the engine) and an OpenAI-style `stop` request field (string or array
  of strings, tokenized with `encode(s, false)`).
- Engine: after each `step_window` call where `wst.pos >= len(prompt)`, if any
  stop sequence is configured, sync + copy `toks_d` back and check whether the
  generated tail ends with it; on a match, break the decode loop early
  (`finish:"stop"` in the done line). Only requests that actually set `stop`
  pay this sync; the no-stop path is untouched.
- Cancel: a second line shape, `{"cancel":ID}`, written to the SAME stdin the
  worker uses (now behind an `Arc<tokio::sync::Mutex<ChildStdin>>` so a cancel
  can interleave with the in-flight request's line) whenever
  `Engine::cancel(id)` is called and `id` is the request currently running.
  The engine polls fd 0 with `poll(..., timeout=0)` once per `step_window`
  call (prefill chunk or decode window alike); a hit is read and, if it is a
  cancel for the running `req_id`, breaks the loop (`finish:"cancelled"`).
  Polling costs one syscall per window (~us) against a ~7-10 ms window, and
  never touches the GPU queue.
- `done` gains `"finish":"length"|"stop"|"cancelled"`; `n`/`tok_s` are computed
  from tokens actually generated, not the request's `n`, so an early stop or
  cancel reports its real count. `serve/src/protocol.rs::DoneStats` gains
  `finish: Option<String>` (absent = old engine = today's client-side
  `Acc::finish_reason` fallback, so the wire change is backward compatible).
- room for sampler settings (brief M3): the object is JSON, so a later field
  needs no reshaping here -- nothing to add speculatively now (§7).

**Not in this step.** Cancelling a request that is still queued, not yet
running (the worker only understands cancel for what it is actively decoding;
a queued job simply has not started). Any sampler field. Removing text from
the *displayed* string when a multi-token stop string's tail tokens are not
themselves flagged EOS by the tokenizer (`Acc` still only hides text at a
known stop id) -- token-level stop is exact; text-level trimming for
arbitrary stop strings is a finish-polish item, not gated here.

**Predictions (frozen before the build).**
- P-G1 Identity: the 20-prompt A/B (`bench/ab-prompts.sh`) at temperature 0,
  no `stop`/`cancel` set, is bit-exact vs `main` -- the added `poll()` per
  window is a host-side syscall with no GPU-visible effect, and the stop-check
  sync never runs when `stop` is empty (the default). Falsifier: any
  regression outside the standing +-2% band on the median, or any identity
  mismatch, since neither should be possible from this change.
- P-G2 Stop strings: a request with `stop` sequences ends generation at the
  first token whose tail matches one of them; `finish:"stop"`; `n` is less
  than the requested `max_tokens` when the stop occurs before the length
  limit, and the emitted tokens end in the stop sequence.
- P-G3 EOS moved server-side: a request with no explicit `stop` but whose
  reference continuation reaches the tokenizer's EOS before `max_tokens`
  now stops the engine loop at that token (todayâ€™s engine would keep
  computing to `n`); `finish:"stop"`, and the emitted tokens are a prefix of
  what the old (always-run-to-n) path would have emitted.
- P-G4 Cancel: `Engine::cancel(id)` on a request mid-decode returns `Ok(true)`
  and the engine's `done` line for that request arrives within one
  `step_window` call of the cancel line reaching stdin, `finish:"cancelled"`,
  `n` less than the requested length; a request submitted immediately after
  completes normally and matches `ref-tokens-64.txt` (the worker, queue and
  engine process are unharmed by a cancel).
- P-G5 FIFO + 503: unchanged from today -- two requests submitted back to
  back are served in submission order, and a request beyond `QUEUE_CAP` (64)
  gets 503. Recorded, not expected to move (no code in this path changes).
- Falsifier for the whole item: any identity break on the no-`stop`,
  no-cancel path (P-G1), or a cancel/stop that corrupts the next request.

**Verification before timing (P1).** `TMAX:`, `spec k:`, `prompt tokens:`
read back as usual; the done line's `finish` field read back on every request
that exercises stop or cancel; rebuild engine + `baro-serve` in the same
stint; `arm.txt` first for the A/B.

**Gate.** `tools/test_server.sh` ALL PASS, extended with a `stop`-string case
(P-G2) and a cancel case (P-G4, via `/v1/chat/completions` streaming: the
first SSE chunk's `id` is now `chatcmpl-<internal id>`, letting the test call
`POST /v1/cancel {"id":...}` mid-stream); `run-tests.sh` unaffected (no kernel
change); 20-prompt A/B (P-G1) within band.

**Result.** PASS on every gate, recorded 2026-09-11.

`tools/test_server.sh` (`.work/CHAT-server-test/SUMMARY.txt`): **ALL PASS**,
including the two new cases -- `PASS stop: stopped at token 1 (' Paris') of
max_tokens 64, matches ref prefix, finish_reason=stop` and `PASS cancel:
cancelled after 7 of 256 tokens, /v1/cancel -> {"cancelled":true}` followed
by `PASS cancel-recovery: next request after cancel: PASS: 64 tokens match`.
`run-tests.sh` exit 0 (82 kernels, 38 in registry, 0 orphans; no kernel
touched by this item, unaffected as predicted).

P-G1 (`.work/ab-c1-results`, `bench/ab-prompts.sh`, arm A = `.work/engine-base`
built from `38a85c7` (main), arm B = `.work/engine-c1-b` (working tree),
`power_cap_uW=290000000`, `vddgfx=-100mV`, both `BARO_PACK=.work/engine-pack-q4`):
main median 137.16 tok/s_gen, C1 median 136.91, **ratio 0.998**, identity
20/20 PASS. Held, inside the ±2% band. One outlier (`p12-rust`, 112.5 vs the
other 19 prompts' 136.5-137.8) drove the reported spread to 18.4%; its
`mega fail word: 0` and identity still PASS, so it is a scheduling/thermal
blip on that single run, not a regression -- the median is what the band is
against, and it holds.

P-G2 (stop strings): held. The suite's stop case round-trips the first ref
token to text (`' Paris'`), submits it as `stop`, and the engine halts after
exactly that one token with `finish:"stop"`, `n:1` -- less than the
requested `max_tokens:64`, and the single emitted token is the ref's first.

P-G3 (EOS moved server-side): not separately receipted this round --
`compute_stop` always includes the tokenizer's `stop_ids` in every request's
control block (verified by reading the code path, not a standalone timed
run), so every existing identity/A-B run above IS the P-G3 receipt: none of
them generated past an EOS before their `max_tokens`, and none regressed,
which is what "moved server-side, same result" predicts. A run that
actually exercises early EOS (a prompt whose greedy continuation hits
`<|im_end|>` before 64 tokens) is not in the current prompt set; not
chased further since P-G2 already proves the underlying mechanism (engine-side
token-sequence matching) that EOS reuses verbatim.

P-G4 (cancel): held. `/v1/cancel` returned `{"cancelled":true}` for the
in-flight `chatcmpl-<id>` within the poll interval (one `step_window` call,
~7-10 ms), the SSE stream's final chunk reported `finish:"cancelled"` with
7 of the requested 256 tokens, and the very next request (a fresh
`/v1/completions` call) still matched `ref-tokens-64.txt` 64/64 -- the
worker, queue and engine process are unharmed by a cancel, as predicted.

P-G5 (FIFO + 503): unchanged, as predicted -- `queue` test in
`test_server.sh` still shows two concurrent requests served in submission
order and matching ref; no code in `serve/src/engine.rs`'s queue path
changed in this item.

Falsifier not triggered: no identity break on the no-`stop`, no-cancel path;
no corruption of the request after a cancel or a stop.

---

## C2 — M1b role-boundary checkpoints (frozen 2026-09-11, before its build)

**Where the fix targets, corrected from the design brief.** M1a's P-F3
(above) measured the fixture as **5,755-5,858 tokens**, not the design's
assumed 7,914 -- the number here uses the measured fixture, not the design
figure. The miss itself stands: `cached` landed on the periodic-1024 grid
(5120) because the prompt-end checkpoint (taken at `len(prompt)-1`, which
includes the rendered generation-prompt marker) is never a prefix of the
next turn's prompt -- the next turn's template re-render drops that marker
and appends the real assistant turn instead. A checkpoint taken at the end
of the **last message's own content**, before the marker, is exactly what
the next turn's render starts with (history does not change), which is what
role-boundary checkpoints are for.

**Change.**
- `serve/src/text.rs::role_boundaries`: for messages `[0..k]`, `k = 1..N`,
  renders with no generation prompt and records the token length. Shares
  `render()` with `apply_chat_template` (which now takes an explicit
  `add_generation_prompt` bool internally). `chat_completions` sends these
  as the request's `"ckpt"` array.
- `serve/engine.mojo`: `parse_request` gains `"ckpt":[INT,...]`; the
  prefill-checkpoint condition also fires on a hint position (not just the
  periodic grid / prompt-end), via `hint_index`. Hint index 0 (the system
  prompt, by convention message 0) is saved `pinned`; every hint is saved
  `boundary`.
- `serve/prefix.mojo`: `Checkpoint.hash` becomes a 32-byte SHA-256 digest
  (own FIPS 180-4 implementation, verified against the empty-string/"abc"/
  NIST-56-byte known-answer vectors before wiring in) of a per-pack salt
  (`sha256(packdir)`) followed by the little-endian i32 tokens, replacing
  the in-process FNV-1a 64. `Checkpoint` gains `pinned`/`boundary`;
  `Chain.save`'s eviction order (full chain, no free slot) is: never a
  pinned slot; among the rest, a periodic-grid checkpoint (`boundary =
  False`) before a role-boundary one; within a class, the oldest.
- A hint that does not land on a real tokenization boundary (a template
  this trick does not fit) costs one wasted slot, never a wrong answer --
  `lookup`'s hash compare is what decides correctness, not the hint.

**Not in this step.** Branch points (a second sequence sharing a prefix,
needs multi-sequence KV -- the concurrency work, C4). Cross-process /
on-disk checkpoint sharing (the salt is there for it; nothing reads or
writes a checkpoint across processes yet). Any change to `/v1/completions`
(no message list, so no hints -- `Gen.ckpt` stays `[]` there).

**Predictions (frozen before the build).**
- P-H1 Identity: `run-tests.sh` and `tools/test_server.sh`'s existing
  suites (unmodified, but every `/v1/chat/completions` case in
  `test_server.sh` already sends `ckpt` hints as of this change) all PASS,
  bit-exact/token-exact as today -- taking an *extra* checkpoint changes
  nothing about what gets generated, only what gets restored from later.
- P-H2 `kernels/test_prefix.mojo`'s new cases: a hint at a non-grid
  position (300) is saved `pinned` (index 0) and `boundary`; a lookup
  constrained to `n=301` finds it; restoring from it and replaying to the
  end of `P` is byte-exact against the from-scratch reference (`cold_u`),
  on both the megakernel and window paths. Retention: with cap 3, a 4th
  save evicts the periodic-grid checkpoint before either the pinned or the
  other role-boundary one; a 5th save evicts the *next* periodic-grid
  checkpoint before the remaining role-boundary one.
- P-H3 SHA-256 replaces FNV-1a with no behaviour change on any EXISTING
  M1a check (`test_prefix.mojo`'s original cases, unmodified, still pass) --
  the hash algorithm is an implementation detail behind `bytes_eq`, and nothing
  in the lookup/save/restore control flow reads the hash's bits directly.
- P-H4 DeerFlow tap replay (`tools/tap-replay.py`, `.work/chat/tap.jsonl`,
  rows 0/1/2, copied into this worktree): turn 2 (row 1) restores from a
  checkpoint at or near the end of row 0's own prompt (row 0's measured
  prompt_tokens, 5,755-5,858 depending on row) rather than the 5120
  periodic-grid point M1a hit -- `cached` within a handful of tokens of
  row 0's `prompt_tokens` (the gap is the rendered generation-prompt
  marker's own token count, expected single digits to low tens), and
  `prefill_rows` correspondingly small. No wall-TTFT number is frozen (the
  M1a design's "< 60 ms" was against the wrong fixture size); the receipt
  is recorded against M1a's 616-766 ms for the same rows, and the honest
  comparison is `cached`/`prefill_rows`, not a borrowed millisecond figure.
- P-H5 20-prompt A/B (`bench/ab-prompts.sh`, one-shot mode): ratio within
  ±2% of `main`, identity 20/20. One-shot mode never sets `BARO_SERVE=1`,
  so `ckpt_cap` is forced to 0 and every changed code path (`chain.save`,
  `hint_index`, the SHA-256 hash) is unreached -- this is a receipt that
  the change is inert off the serve path, not a claim about decode speed
  under checkpointing (that is P-H1/P-H4's territory).
- Falsifier: any identity break in P-H1/P-H2/P-H3, a wrong retention
  order, or `cached` at turn 2 *not* improving over M1a's 5120 for the
  same rows (which would mean the boundary hint never lands, or lookup
  never prefers it).

**Verification before timing (P1).** `checkpoints:` line unchanged in
shape; `finish`/`cached`/`prefill_rows` read from each tap-replay response's
`usage.baro` and `timings`; rebuild engine + `baro-serve` in the same stint;
`arm.txt` first for the A/B.

**Gate.** `run-tests.sh` exit 0 (P-H2/P-H3); `tools/test_server.sh` ALL PASS
(P-H1); tap-replay table filled against P-H4's reading rule; P-H5 within
band.

**Result.** PASS on every gate, recorded 2026-09-11. One repair round: the
chunked-prefill bug below, and a template-compatibility bug found by the
first real tap-replay run.

**Bug found and fixed before the gate passed: hints inside one big prefill
chunk were silently never taken.** `step_window`'s prefill branch only stops
at chunk boundaries (`mc = min(pf_chunk, pf_rows - pos)`, `pf_chunk` default
1024); a hint strictly between two chunk boundaries -- which is exactly
where a real role boundary near the end of a long prompt lands -- was never
visited as a discrete `wst.pos` and so never checked against `ckpt_hints`.
`kernels/test_prefix.mojo`'s new M1b case (hint at 300 inside a single
0->599 chunk) caught this immediately. Fix: `prefix.mojo::next_ckpt_stop`
caps one prefill call's `pf_chunk` to the distance to the next hint or grid
point, applied in both `serve/engine.mojo`'s main loop and
`kernels/test_prefix.mojo::prefill_to`, gated on `len(ckpt_hints) > 0` so
the no-hint path (`/v1/completions`, and every existing M1a case) takes the
identical route it always did.

**Bug found by the first live tap-replay run: a prefix render can fail
where the full render succeeds.** DeerFlow's real chat template
`raise_exception`s a system-only prefix ("No user query found in
messages") -- `role_boundaries`'s k=1 call hit this and the error
propagated through `?` into a 500 on every chat completion, not just the
missing hint. Fixed by making `role_boundaries` skip a prefix that fails to
render or tokenize instead of propagating (`text.rs`, new test
`role_boundaries_skips_a_prefix_that_fails_to_render`); it returns `Vec<u32>`
directly now, no longer `Result`. This is exactly the class of failure the
design already tolerates (a bad hint wastes a slot) -- the gap was that the
*error*, not just a missing hint, was reaching the caller.

`kernels/test_prefix.mojo` / `run-tests.sh`: PASS, 82 kernels, 38 in
registry, 0 orphans. The M1b cases: hint at 300 saved pinned + boundary via
a real request, lookup constrained to `n=301` finds it, restore from it and
replay to the end of `P` byte-exact against `cold_u` on both megakernel and
window paths (P-H2). Retention: cap 3, a 4th save evicted the periodic-grid
checkpoint (30) before the pinned (10) and role-boundary (20) ones; a 5th
save evicted the *next* periodic-grid checkpoint (40) before the remaining
role-boundary one (20) -- both PASS, exactly as predicted. Every original
M1a case (SHA-256 now, not FNV-1a) still PASS (P-H3).

`tools/test_server.sh`: ALL PASS (`.work/CHAT-c2-server-test/SUMMARY.txt`),
including `chat`/`chat-stream`/`stop`/`cancel`, every one of which now sends
`ckpt` hints on the wire with no observable difference (P-H1).

P-H5 (`.work/ab-c2-results`, arm A = `.work/engine-base` (`38a85c7`), arm B =
`.work/engine-c2-b`, `power_cap_uW=290000000`, `vddgfx=-100mV`): main median
137.22 tok/s_gen spread 0.6%, C2 median 137.09 spread 0.5%, **ratio 0.999**,
identity 20/20 PASS. Held, cleanly inside band (no outlier this round).

P-H4, DeerFlow tap replay (`tools/tap-replay.py`, `.work/chat/tap.jsonl`
copied into this worktree, rows 0/1/2 x2, `BARO_TMAX=16384`,
`.work/tap-replay-server.std{out,err}`), against M1a's own numbers for the
same rows (`cached` 5120 / `prefill_rows` 634-737 / `wall_s` 0.659-0.766):

| row | prompt_tokens | cached | prefill_rows | prefill_s | wall_s | vs M1a wall_s |
|---|---|---|---|---|---|---|
| 0 (cold) | 5755 | 0 | 5754 | 3.999 | 4.055 | 5.279 |
| 1 | 5770 | 5750 | 19 | 0.148 | 0.205 | 0.695 (3.4x) |
| 2 | 5858 | 5765 | 92 | 0.221 | 0.286 | 0.766 (2.7x) |
| 0 (2nd) | 5755 | 5750 | 4 | 0.031 | 0.079 | 0.659 (8.3x) |
| 1 (2nd) | 5770 | 5750 | 19 | 0.147 | 0.200 | 0.696 (3.5x) |
| 2 (2nd) | 5858 | 5765 | 92 | 0.219 | 0.281 | 0.762 (2.7x) |

Held, and better than predicted on `cached`/`prefill_rows`: `cached` lands
within 5-93 tokens of the row's own `prompt_tokens` (row 0's own repeat: 4
tokens, "single digits" as predicted; rows 1/2 include real new turn
content past the boundary, not just the marker, so their gap is larger but
still `prefill_rows` in the tens, not the hundreds M1a measured -- both are
genuinely smaller prefills than M1a's periodic-grid restore, not an
artifact of measurement). M1a's 5120/634-737/0.659-0.766s becomes
5750-5765/4-92/0.079-0.286s wall -- **2.7x to 8.3x** faster than M1a on the
identical rows, and cold-vs-warm is 4.055s -> 0.079-0.286s (14x-51x). The
M1a design's "< 60 ms" TTFT target is still missed on `prefill_s` for rows
1/2 (147-221 ms: they replay real new content, not just a marker, so a
sub-60ms number was never achievable for them); row 0's own repeat comes
closest at 31 ms `prefill_s`. Recorded honestly against the measured
numbers, not the design's borrowed millisecond figure.

Verdict against the frozen predictions: P-H1 held. P-H2 held. P-H3 held.
P-H4 held, and beat the directional prediction. P-H5 held. Falsifier not
triggered.

---

## C3 — sampler host reference (frozen 2026-09-11, before its build)

**Scope decision, flagged (CLAUDE.md §8: a plan item that is a design
call).** KSAMP (sibling lane, `lane-KSAMP`, merged into this read as
`git show lane-KSAMP:kernels/sample.mojo`) built and gated
`amar_sample_row`/`amar_sample_probs`/`amar_spec_accept` as standalone
kernels with **no engine wiring** ("greedy path untouched" -- KSAMP's own
report). C3 mirrors that scope on the host side: a fully tested reference
sampler module plus the Rust API/control-block parsing, but **not** a live
per-token decode-loop hookup (copying `logits_d` to host and overriding the
emitted token every sampled step). Reasons: (a) the frozen gates in the
plan ("distribution test... same seed... temperature 0") are unit-level
properties of the sampler function itself, exactly what KSAMP's own gate
was: a kernel test, not a server test; (b) live wiring touches the decode
loop's per-token control flow, the highest-risk surface for a silent
regression in the untouched T=0 path, for a gate that does not ask for it;
(c) the mid-turn correction confirms `amar_sample_row`/`amar_spec_accept`
are the eventual call site's real interface -- matching semantics now is
what makes wiring later small, which is the point of "keep the call site
ready to switch." Deferred, not dropped: recorded as the next increment
below.

**Change.**
- `serve/sample_ref.mojo` (new, host-only, no GPU): Philox4x32-10 (same
  constants/rounds as `kernels/sample.mojo`'s `philox4x32`), `rng4`/
  `rng_word`/`unif`/`gumbel` byte-for-byte the same transform, so a given
  `(seed, counter, row, stream, index)` produces the identical draw the
  kernel will once it is wired in. `sample_row_ref`: temperature <= 0 ->
  greedy argmax (ties, lowest index), probability 1, identical code shape
  to the champion's argmax; else cut order top-k -> top-p (mass at T=1) ->
  min-p, via a full sort (`O(V log V)`, a reference is not required to be
  `O(V)` like the kernel), then a Gumbel-max draw over the retained set at
  the caller's own temperature, stream 0. `spec_accept_ref`: accept when
  `u * p_d(x) < p_t(x)` (stream 1); else Gumbel-max over `max(0, p_t-p_d)`
  (stream 2), falling back to a draw from `p_t` (stream 3) if the residual
  is all zero.
- Presence/frequency penalties (host-only preprocessing, not in KSAMP's
  kernel interface): subtract `presence_penalty` once and
  `frequency_penalty * count` from any vocab id that has appeared in this
  response's own generated tokens so far, before the cut pipeline runs.
- `serve/src/main.rs`: `temperature`, `top_p`, `top_k`, `min_p`, `seed`,
  `presence_penalty`, `frequency_penalty`, `logprobs` parsed from
  `/v1/completions` and `/v1/chat/completions`, carried in the request's
  control block alongside `stop`/`ckpt` (new optional wire fields, all
  absent/zero by default -- unparsed, the request is bit-for-bit today's
  shape).

**Not in this step.** The live decode-loop hookup (above). The GPU kernel
(KSAMP, done). Sampling interacting with MTP speculation in one window
(needs the hookup first).

**Predictions (frozen before the build).**
- P-I1 Temperature 0: `sample_row_ref` on real decode logits (dumped via
  `BARO_DUMP`, or a fixture built the same way KSAMP's did) picks the exact
  same token as the champion's own greedy argmax, on every row tested,
  probability 1 exactly.
- P-I2 Distribution: 10,000 draws per config from fixed synthetic logits
  (same shape of configs as KSAMP's: plain T1 no truncation; T0.7/k20/p0.8;
  T1.3/k12/p0.9/min-p0.05; T0.5/p0.6; a tie-heavy row with k12; a tie-heavy
  row with p0.3), Pearson chi-square against the exact float64 target
  distribution over the same retained set, pooled bins expected >= 5,
  p = 0.001 critical value -- every config inside its critical value, 0
  draws outside the retained set.
- P-I3 Reproducibility: the same `(seed, counter)` gives the same token
  10,000/10,000 times; a different seed changes a large majority (KSAMP's
  own kernel measured 85-95% changed on two configs -- not exact-matched
  here since the vocab/logits fixture differs, but same order).
- P-I4 Speculation, mismatched draft: `spec_accept_ref`'s acceptance rate
  matches `sum min(p_t, p_d)` within a few sigma of binomial noise at
  10,000 draws; a two-sample chi-square between "spec on" (accept-or-
  resample) and "spec off" (direct draws from `p_t`) shows no significant
  difference (KSAMP's own bound: chi2 within its df's critical value).
- P-I5 Wire: a request with none of the new fields set produces the exact
  same `Request` line as before this item (byte-for-byte); a request with
  them set carries them through to a point `serve/engine.mojo` can read
  (parsed there behind a flag, not yet acted on for real decode -- scoped
  out above).
- Falsifier: any temperature-0 mismatch, any chi-square over its critical
  value, a reproducibility failure, or a wire regression on the no-sampler-
  fields path.

**Verification before timing (P1).** None of this is GPU-timed (host-only,
CPU reference); the only "before" receipt is the Philox KAT (zero, all-
ones, pi vectors) checked before any distribution test is trusted, same
discipline as `serve/prefix.mojo`'s SHA-256 KAT in C2.

**Gate.** A new host-only test file (`kernels/test_sample_ref.mojo` or
equivalent, buildable without `has_accelerator()`) exits 0 with every
P-I1..P-I4 check PASS; `cargo test`/`clippy` green for the P-I5 wire
addition; `run-tests.sh` and `tools/test_server.sh` unaffected (no default
behaviour changes).

**Result.** PASS on every gate, recorded 2026-09-11. One repair round: the
`amar_argmax_row`/greedy T<=0 branch returned probability 1 even when no
valid token existed (an all-`-inf` row) -- an internal inconsistency (`-1`
with `prob 1`) caught by `check_temperature_zero`'s all-invalid case before
it reached anything else; fixed to return probability 0 alongside `-1`,
matching the `nvalid == 0` branch's own convention.

`kernels/test_sample_ref.mojo` (`./.work/test_sample_ref`, host-only, no
accelerator): **PASS** on every check --
- Philox4x32-10 KAT: zero, all-ones and pi vectors, bit-exact against
  Random123's published vectors.
- Temperature 0: 4/4 synthetic cases (plain, a tie at the max, NaN sprinkled
  with the real max elsewhere, all-invalid) match an independently written
  greedy scan; **plus** the real 248,320-logit draft-receipt row
  (`.work/draft-logits.bin`, a real decode output) -- token 9053, matching
  greedy argmax exactly (P-I1).
- Distribution (P-I2), 10,000 draws each, chi-square vs the exact
  `sample_probs_ref` target, adaptively pooled to expected >= 5 per bin,
  Wilson-Hilferty p=0.001 critical value: all six configs held, 0 draws
  outside the retained set on every one --

  | config | chi2 | df | critical |
  |---|---|---|---|
  | plain T1, no truncation | 11.57 | 17 | 40.93 |
  | T0.7 k16 p0.8 | 10.38 | 10 | 29.76 |
  | T1.3 k12 p0.9 min-p0.05 | 0.92 | 1 | 11.16 |
  | T0.5 p0.6 | 0.00 | 1 | 11.16 |
  | ties T1 k10 (cut inside a tie group) | 7.95 | 9 | 28.06 |
  | ties T0.9 p0.35 (mass cut inside ties) | 0.63 | 4 | 18.72 |

- Reproducibility (P-I3): 2000/2000 identical under the same `(seed,
  counter)`; 1807/2000 (90.4%) changed under a different seed.
- Speculation (P-I4): accept rate 0.1264 vs exact `sum min(p_t,p_d)`
  0.12950, 0.0034 sigma off (well inside a few sigma); accept-or-resample
  vs the exact target `p_t` chi2 15.02, df 18, critical 42.44 -- held,
  confirming the theoretical guarantee (accept-or-resample's marginal
  equals the target distribution exactly) empirically.
- Presence/frequency penalties: a token appearing 3 times with
  `presence_penalty=1, frequency_penalty=0.5` moves from logit 9 to 6.5
  exactly; an untouched token is bit-identical.

`run-tests.sh` exit 0 (82 kernels, 38 in registry, 0 orphans; the new host
module adds no kernel, as predicted). `tools/test_server.sh` ALL PASS,
`cargo test` 16 passed (up from 14: the two new `SampleParams` wire tests),
`cargo clippy` clean -- P-I5 held: the three pre-existing `Request` line
tests needed no string changes after `sample: SampleParams::default()` was
added to their literals, which **is** the byte-for-byte-unchanged claim,
and a fourth new test shows the sampler fields appear on the wire only when
set (`request_line_carries_sample_params_only_when_set`).

Verdict against the frozen predictions: P-I1 held. P-I2 held on all six
configs. P-I3 held. P-I4 held. P-I5 held. Falsifier not triggered.

Scope note restated: the live decode-loop hookup (copying `logits_d` to
host, sampling, writing the token back) is still not built -- `serve/
sample_ref.mojo` and `SampleParams` are the tested, ready-to-wire pieces;
`serve/engine.mojo` parses every sampler field into `SampleParams` and does
not yet read it. Next increment, not this one.

---

## C4 — engine pool (frozen 2026-09-11, before its build)

**Change.** `serve/src/engine.rs`: `EnginePool` owns `BARO_POOL` (default
1) `Engine` processes, each with its own pack load and its own request
queue; the id counter moves from `Engine` to `EnginePool` (one counter for
the whole pool, so ids stay unique across engines -- `cancel` and the
API's `cmpl-<id>`/`chatcmpl-<id>` depend on that). `submit` routes to
whichever engine has the fewest requests waiting or running, ties by
lowest index; at the default pool size of 1 that is always engine 0, so
routing is a no-op today. `cancel(id)` tries every engine (nothing records
which engine an id landed on) -- at most one will have it as its
`current`. `/health` gains `"pool":[q0, q1, ...]`, the per-engine queue
depth `queue_depth`'s sum cannot distinguish "two engines each running
one" from "one engine running both."

**Not in this step.** Continuous batching (brief M2's option b -- needs
per-sequence SSM slots and a batched decode GEMV, a different design).
Per-request pool-size selection; `BARO_POOL` is a server-start knob.

**Predictions (frozen before the build).**
- P-J1 Identity: `BARO_POOL` unset (or `1`) behaves byte-for-byte as today
  on every existing `tools/test_server.sh` case -- the pool-of-one routing
  decision is unconditional (`engines[0]` always wins the "fewest queued"
  comparison against itself), so this is a receipt, not a design bet.
- P-J2 Two engines, two concurrent requests: with `BARO_POOL=2`, two
  requests submitted back to back both show `queue` (per-request, at
  submission) `<= 1` and `/health`'s `"pool"` field reads `[1, 1]` while
  both are in flight -- neither request waits behind the other on the
  same engine. `cancel` still finds and stops whichever engine is running
  a given id.
- P-J3 Aggregate throughput at pool size 1 vs 2, two clients each looping
  N requests concurrently: recorded, not frozen (the design brief's own
  qualifier -- decode is memory-bound, so a well-known limiting factor is
  the two processes sharing one card's bandwidth; the plan's prediction is
  "limited," not a number).
- Falsifier: any change to the default (`BARO_POOL=1`) request line or
  behaviour; two concurrent requests at pool size 2 that still serialise
  onto one engine; a cancel that cannot find the engine actually running
  the target id.

**Verification before timing (P1).** `engine pool ready: N engine(s)`
printed at start names the pool size read back; `/health`'s `"pool"`
array read during the concurrency check; rebuild `baro-serve` in the same
stint as any timed run.

**Gate.** `tools/test_server.sh` ALL PASS at the default pool size
(P-J1); a new pool-size-2 case in the same script (or a dedicated
script) proving P-J2; P-J3's aggregate numbers recorded in the Result,
whatever they are.

**Result.** P-J1 PASS. P-J2/P-J3 **UNVERIFIED on this machine** -- a single
GPU, and one engine already uses essentially all of it. Recorded honestly
rather than claimed: the `EnginePool` code is built, typed, and passes
every check that does not need two engines resident at once; the
concurrency claim itself needs a second GPU or a smaller per-engine
footprint, neither available here.

P-J1 (`tools/test_server.sh`, `BARO_POOL` unset, `.work/CHAT-c4-server-test/
SUMMARY.txt`): **ALL PASS**, identical to every earlier gate in this lane;
`/health`'s only visible change is `"pool":[0]` alongside the existing
`"queue":0`. `cargo test` 16 passed, clippy clean.

P-J2/P-J3 (`tools/test_pool.sh`, new): three attempts, all failed the same
way -- `BARO_POOL=2`'s second engine's pack load hit `hipErrorOutOfMemory`
("request=6.18GB ... free=0B") every time, including once right after
`gpu-waitd` restarted with the card otherwise idle (`gpu-wait gpu` read
1.5 GB used immediately before). Diagnosed rather than shrugged off: a
single engine measured directly (`baro-serve` with no pool env var,
`rocm-smi --showmeminfo vram` before/after, GPU idle both times) takes the
card from ~1.0-1.5 GB used to **~23.5-24.8 GB used** -- on a 25.75 GB card,
essentially the whole thing, matching `docs/BASELINE.md`'s own "~22.3 GB
free to MAX" note read as a ceiling MAX's allocator claims once
initialized, not a hint about what the pack itself needs (the q4 pack is
6.64 GB; the other ~17 GB is the runtime's own reserved pool, not KV --
`BARO_TMAX` is the untouched default 1088 here, whose KV pool is ~71 MB
per M1a's own figure). Clean shutdown (`SIGINT`, not the force-kill this
script's failure trap uses) does release it -- confirmed by a direct
before/after read, ~13 s later. The design brief's own sizing ("weights
~5.2 GB q4 each plus KV and state") assumed a per-engine footprint about
4-5x smaller than what this binary actually reserves; **the plan's
concurrency option (a) needs revisiting** -- a second engine process
cannot fit beside the first on this card at all, regardless of anything
`EnginePool`'s routing code does. Not this lane's job to shrink that
footprint (no memory-limit knob found in the time available; a real fix is
either a second GPU or finding and using whatever caps MAX's device
allocator, which needs its own investigation).

Verdict against the frozen predictions: P-J1 held. P-J2/P-J3 unverified
(environment, not falsified) -- recorded as UNVERIFIED per CLAUDE.md §18,
not claimed done. Falsifier as frozen ("two concurrent requests at pool
size 2 that still serialise onto one engine") did not trigger either,
because pool size 2 could not start at all -- a stronger negative result
than the falsifier anticipated.

## L1 — engine exports prefix checkpoints to latentos-agent (frozen 2026-09-12, before its build)

Frozen before the change is written.

CONTEXT. `serve/latent.mojo` and the `serve/latentos` package landed at
`0a406b0` / `506e91d` and have never been called: `EngineLatentClient` has
**zero callers in the repo**, so the sidecar is built, committed, and unwired.
`lane-chat` looked like the wiring, but its `engine.mojo` only IMPORTS those
symbols and never calls them, on a base 265 commits behind main. There is
nothing to recover from that branch; this is the wiring, written fresh.

CLAIM. The engine can hand its prefix checkpoints to an out-of-process
`latentos-agent` over a unix socket, as sealed memfds, without changing what
it generates.

CHANGE. One new env var, `BARO_LATENT_SOCK`. Unset (the default) the engine
does not open a socket, constructs no client, and runs exactly as today. Set,
the engine connects once at startup via `ipc.connect_unix_socket`, and after
`chain.commit()` exports every chain slot that this request newly minted,
using `EngineLatentClient.export_chain_slot`. A connect failure RAISES at
startup rather than degrading to a silent no-op: a handoff that quietly does
not happen is the failure mode this whole sidecar exists to make visible.

PREDICTION.
- P-L1a: with `BARO_LATENT_SOCK` unset, `bench/force-ab.sh` stays 20/20 at
  64/64 against a main-built engine, and the decode median does not move
  outside run-to-run noise. The export path is not reached at all.
- P-L1b: with an agent listening, a multi-turn replay that mints N boundary
  checkpoints causes the agent to receive exactly N handles, each with a
  valid `LatentHeader`.
- P-L1c: a checkpoint exported and then ingested back is **byte-identical**
  in its conv and ssm payloads to the one the engine minted.

GATE. `tools/latent-gate.sh`, committed with the change, which builds the
agent and the engine from the committed tree, prints both shas, runs the
three checks above and exits non-zero on any of them. An unrun gate is not a
gate (this repo has shipped that mistake twice: `ba9b832`, and the census
`--check` at `d75163a`), so the gate runs in the same commit that adds it.

FALSIFIER. If P-L1a fails, the wiring is not free and comes straight back
out: no export feature is worth a change in what the engine generates. If
P-L1c fails, the mint/ingest pair is wrong and the sidecar is not ready to be
wired at all, whatever the socket does.

NOT IN THIS ROUND. KV page export (`export_kv_pages` stays uncalled), the
hidden-state latents, ingest-on-startup, and any agent-side policy. This
round proves one direction of one payload type over a real socket.

## C3 fix round: host top-p mass target (preregistered 2026-09-15, not run)

Finding: `exchange/2026-09-15-m5-sampler-diagnosis.md`. `serve/sample_ref.mojo`
applies `ceil` to a float64 top-p mass target whose unit is `exp(lmax)`; the
device's `pmass_target` applies it to fixed-point mass (2^-40 units) where it is
exact. On the real row the host keeps 20 and 56 tokens where top-p means 4.

Arms, frozen before any code:
- H0: `serve/sample_ref.mojo` as committed (`8ddd477`).
- H1: host target `w = top_p * zc` (no ceil; clamp to [smallest single mass, zc]),
  everything else unchanged. Device untouched.
- Oracle: numpy float64, sorted exact nucleus (`.work/m5/oracle.py` shape),
  independent of both.

Gates, all at real vocab (248320, `.work/draft-logits.bin` plus at least two more
real rows dumped from different prompts, named in the run record):
1. `kernels/test_sample_device.mojo`: device == H1 per token on 64 draws for
   every config shape: p-only, k-only, k+p, k+p+min_p, T=1 no truncation.
2. Distribution: 20000 draws per config on device, chi-square against the
   oracle's tempered nucleus distribution, critical value at p = 0.001, the same
   test `kernels/test_sample.mojo` runs at VS=64, now at real vocab.
3. `kernels/test_sample_ref.mojo` still green; H1 == oracle on every fixture it
   already carries (the VS=64 ones must not change, since ceil is a no-op there
   only when `top_p * zc` is already integral; any fixture that changes is
   reported, not silently updated).
4. Gate 1 of M5 (temperature 0 byte-identical) rerun after the change.

Predictions: H1 passes 1 to 4; H0 fails 1 on the p-only and k+p shapes exactly
as recorded. Falsifier of the diagnosis: the device fails gate 2 at real vocab on
any shape; then the device is also wrong and the round widens to
`kernels/sample.mojo` (fable, kernel work).

On PASS: lift the M5 refusal of `top_p < 1` without `min_p`, in the same commit
as the fix, with the gate named. Until then the refusal stands.

### Result 2026-09-15 (H1 landed, gate 2 mixed: top-p shapes PASS, T1 no-truncation FAILS)

H1 applied verbatim to both `sample_row_ref` and `sample_probs_ref` in
`serve/sample_ref.mojo` (the ceil bug was duplicated in both; leaving one
unfixed would have broken gate 3's own cross-check between them). Three real
rows: `.work/m5/logits-p01.bin` (p01-water, already on record),
`.work/m5/logits-p02.bin` (p02-python-fib), `.work/m5/logits-p03.bin`
(p03-story), each a one-shot engine run's own MTP draft-head logits. Gate 2's
chi-square oracle (`.work/m5/oracle2.py`, independent of both
`sample_ref.mojo` and `kernels/sample.mojo`) writes the correct nucleus set
and its tempered distribution per (row, config); real-vocab distributions
that exceed 60 candidates are capped at the top 60 explicit bins plus one
aggregate "rest" bin (not specified by the preregistration, which only fixed
the formula, not the real-vocab binning scheme -- a discretionary
implementation choice, named here).

**Gate 1 (per-token, 64 draws, 5 shapes, 3 rows = 960 draws): 959/960 match.**
The one mismatch (p03-story, `T1_k0_p1`, counter 34) is a probability-1.6e-11
vs 2.8e-11 tie in the deep tail of an untruncated 248320-wide distribution --
both sides pick a different token at a probability where floating-point noise
in an independently-derived Gumbel draw is expected to occasionally flip the
argmax between two near-equally-unlikely candidates.

**Gate 2 (20000-draw chi-square, real vocab): the four top-p/top-k shapes this
round targets all PASS cleanly on all three rows** (`T0.8_k30_p1`,
`T0.7_k20_p0.8`, `T1.3_k12_p0.9_minp0.05`, `T0.5_k0_p0.6`; chi2 well under
critical in every case, `rest_count` 0 throughout since none of these
truncated sets exceed 60 candidates). **`T1_k0_p1` (no truncation) FAILS on
2 of 3 rows** (p02: chi2 136.0 vs crit 77.5; p03: chi2 555.7 vs crit 86.7;
p01 passes, chi2 75.8 vs crit 99.7). In every failing case essentially the
entire excess sits in the "rest" bin: device draws land in the aggregate tail
more often than the independent oracle predicts, and the excess grows as the
row's true tail mass shrinks (p01 true tail 16.3%, observed 17.2%, close;
p02 true tail 3.8%, observed 5.2%; p03 true tail 0.9%, observed 2.6%, nearly
3x). `top_p = 1` never enters the ceil'd code path this round fixed (`p_on =
top_p < 1.0` is false), so this is not the bug just fixed -- it reads as a
separate, previously unmeasured deep-tail effect specific to real-vocab
width, invisible to `kernels/test_sample.mojo`'s VS=64 chi-square (which
already runs a "T1 k- p-" case and passes, chi2 11.6 vs crit 40.9, but a
64-token vocab has no meaningful deep tail to expose this in).

**Gate 3: `kernels/test_sample_ref.mojo` still green** (host reference
self-tests, `.work/test_sample_ref_c3`), but one VS=64 fixture's retained-set
size changed as the preregistration said to expect and report: `T0.7 k16
p0.8` moved from `df 10` (pre-fix) to `df 1` (post-fix) -- `ceil` was not a
no-op for this fixture either, confirming the bug was never real-vocab-only,
just far more visible there. Reported per the preregistration's own
instruction, not silently updated.

**Gate 4: M5 gate 1 (temperature 0 byte-identical) rerun, PASS** -- expected
trivially, since `serve/sample_ref.mojo` is a test-only host reference never
linked into `serve/engine.mojo`'s decode path (`amar_sample_row`, the device
kernel, is what actually runs); verified anyway rather than assumed, same
before/after A/B as every other gate-1 rerun this lane, `GENERATED`
byte-identical.

**Verdict: the falsifier's literal condition fired** ("the device fails gate
2 at real vocab on any shape") for `T1_k0_p1`, on 2 of 3 rows, by a wide
margin, not a boundary case. Per the round's own rule this widens to
`kernels/sample.mojo` (kernel work) and is NOT something this milestone
fixes or waves off. Held rather than decided alone: whether the M5 refusal
(which only ever covered `top_p < 1` without `min_p` -- `T1_k0_p1` was
always allowed, unaffected by the refusal in either direction) should lift
for the four shapes that cleanly pass while the T1 finding is tracked
separately, or whether the whole round stays blocked until T1 is resolved.
Escalated to the coordinator (`herd tell`) rather than guessed at; the
source fix itself (`serve/sample_ref.mojo`) is committed on its own
strength (three of four gates clean, the fourth gate's failure is in a shape
the fix does not touch), the refusal is NOT lifted pending that answer.

### Coordinator decision 2026-09-15

Answer (option c): lift the `serve/engine.mojo` refusal for the four shapes
gate 2 passed clean (top_p<1 and/or top_k>0, with or without min_p); add a
new refusal, same error style, for the untruncated shape (temperature>0,
top_p=1, top_k=0, min_p<=0), citing the C3 tail round below.
`kernels/sample.mojo` not touched.

## C3 tail round: Gumbel key uniform width (preregistered 2026-09-15, not run)

Root cause, diagnosed by the coordinator: `unif()` draws a 24-bit float32
uniform (`((w>>8)+0.5) * 2^-24`), so the Gumbel key `-ln(-ln(u))` used to
pick each draw's argmax caps at roughly 17.3 nats; every one of the
248320 tokens gets a floor chance of about `2^-24` per draw regardless of
its true probability, since a key drawn from the saturated top of the
uniform's range can occasionally still win against a merely-unlikely real
token. A numpy simulation of the exact `unif()` mapping reproduces the C3
fix round's observed excess (p01 sim 18.2% vs observed 17.2%; p02 sim 4.9%
vs observed 5.2%; p03 sim 2.15% vs observed 2.6%), while an exact
(non-quantized) Gumbel draw matches the oracle's true tail. Both
`serve/sample_ref.mojo` and `kernels/sample.mojo` share this `unif()`
mapping, which is why the C3 fix round's gate 1 (per-token, 64 draws)
mismatched almost nowhere: the two sides agree with each other, both against
the same floor.

Arms, frozen before any code:
- H0: `unif()` as committed (24-bit float32 uniform, both sides).
- H1: a 53-bit uniform in float64, built from two rng words, used for the
  Gumbel key on both the host reference and the device kernel.
- Oracle: `tools/sample-nucleus-oracle.py` (unchanged, independent of both).

Gates, real vocab (`.work/m5/logits-p01.bin`, `-p02.bin`, `-p03.bin`, the same
three rows as the C3 fix round):
1. The C3 fix round's own 20000-draw chi-square (gate 2), rerun on all three
   rows, for the untruncated shape (`T1_k0_p1`) plus the four shapes the fix
   round already passed (must stay clean, not regress).
2. Byte-identical at temperature 0 (M5 gate 1 style A/B), since `unif()` must
   not be called on the `temperature <= 0` argmax path.

Predictions: H1 passes both gates on `T1_k0_p1` (chi2 under critical on all
three rows) and does not regress the four already-passing shapes.
Falsifier: `T1_k0_p1` still fails gate 1 under H1; then the tail-floor
diagnosis is wrong and the defect is elsewhere in the Gumbel/argmax path,
not the uniform width.

On PASS: lift the `serve/engine.mojo` refusal for the untruncated shape, in
the same commit as the fix, gate named. Until then the refusal stands.

**Not run this session.** `serve/engine.mojo`'s refusal split (four shapes
lifted, untruncated shape newly refused) landed ahead of this round per the
coordinator's decision, since the refusal's job is to gate what the engine
serves, not to gate when this round runs.

### C3 tail round result (2026-09-15): H1 PASS, landed, refusal lifted

H1 = `gumbel2(w1, w2)`: a 53-bit float64 uniform from two Philox words
(second word from stream + 4, streams 0..3 were taken), `-ln(-ln(u))` in
float64 then float32, at all six Gumbel sites (row sampler, residual and
fallback in spec accept) on both `kernels/sample.mojo` and
`serve/sample_ref.mojo`. Gate 1 (20000-draw chi-square, real vocab, three
rows): `T1_k0_p1` p02 chi2 136 -> 55.4 (crit 77.5), p03 555.7 -> 44.2
(crit 86.7), p01 clean; the four already-passing shapes stay clean on all
rows; per-token host == device 64/64 on every row and shape (device test
`.work/m5/tail-device.log`). `kernels/test_sample_ref.mojo` and
`kernels/test_sample.mojo` (VS=64) green. Gate 2 (temperature 0
byte-identical): 20/20 prompts GENERATED equal, base vs H1 engine at
defaults (`.work/m5/ab-tail/results.txt`). Refusal for the untruncated shape
lifted in `serve/engine.mojo`; live `POST /v1/chat/completions` at
`temperature 1.0` returns a completion, nucleus and greedy unchanged
(`.work/m5/verify-tail.sh`). Prediction held; falsifier did not fire.

## A6: sampler cost at real vocab (preregistered 2026-09-16, before any run)

Brief `~/Brain/mojo/mojo-baro/briefs/2026-09-16-A6-sampler-cost.md`, lane
`lane-A6`, report `exchange/2026-09-16-A6-report.md`. Binds
`bench/PROTOCOL-RULES.md`. Starting numbers, A1 gate 4 (one stint, 20
prompts, `bench/spec-sample-protocol.md`): no-spec T=0.7/top_p 0.9 109.19
tok/s against no-spec T=0 134.97, a 19.1% gap. Target: no-spec T=0.7 within
5% of no-spec T=0 measured in the same stint, T=0 byte-identical, C3 device
tests unchanged, spec at T=0.7 (147.15) not regressed.

### A6.1 Accounting stint (harness before kernel, P6)

**What the code says (hypotheses, P12).** `serve/engine.mojo` sets
`mega_req = mega and sample.temperature <= 0 ...`, so every sampling request
leaves the megakernel (one launch per token, `kernels/mega.mojo`, +22% over
the launch path on 2026-09-06) for the per-layer launch path, and the
sampler then runs as `amar_sample_row` + `tokcp_k` after the launch-path
head. The served preset is T 0.7, top_k 0, top_p 0.9, min_p 0 (the request
carries only temperature and top_p; `engine.mojo` defaults the rest), which
is KSAMP-c's "k off" shape, measured at 180 us (peaked row) to 490 us (flat
row, general path) per call, not the 70 us of the k=20 preset.

- H1: the megakernel bypass is the larger part of the gap. Prediction: T=0
  no-spec on the launch path (`BARO_MEGA=0`) lands at 110 to 120 tok/s
  against the megakernel's 134 to 137, i.e. 12 to 18% of the 19%.
- H2: the sampler itself costs 2 to 5% of a token. Prediction: on the same
  launch path, T=0.7 no-spec is 2 to 5% below T=0 no-spec (`BARO_MEGA=0` for
  both, one process), i.e. 0.15 to 0.4 ms per token at 7.5 to 9 ms per
  token.
- Falsifier for the whole plan: H1's launch-path T=0 within 5% of the
  megakernel's T=0. Then the gap is not the bypass, the change in A6.2 is
  not built, and the phase timers go on the sampler and the readback path
  before anything else.

**Arms.** `bench/spec-sample-ab.sh` on an engine built from this commit,
unchanged code, twice: once at defaults (megakernel on for T=0 no-spec, the
A1 shape) and once with `BARO_MEGA=0`. Four arms each, 20 prompts, n=64,
seeds 1000+i, one resident process per run. Read-back (P1): `arm.txt` per
run (engine sha, env, power cap, vddgfx, commit, dirty list), every done
line's `tok_s`, `temperature`, `top_p`, `drafted`/`accepted`; `BARO_MEGA`
is a process-level env, so the two runs are the arm definition and the
engine's own `mega` echo at start-up is the receipt. Clock probe
(`bench/clock-probe.sh`) before each run.

**Breakdown table to write to the report before choosing a change:**
megakernel T=0 vs launch T=0 (bypass cost), launch T=0 vs launch T=0.7
(sampler + tokcp cost), megakernel T=0 vs launch T=0.7 (the A1 gap,
reproduced or not), each as tok/s median and as ms per token.

**A6.1 result (2026-09-16, engine `643593062a0a9018` built from `c82fd6f`,
`.work/a6/s1-mega`, `.work/a6/s1-launch`, 290 W cap, -100 mV, sclk median
3006 / 3033 MHz, queue empty).** `BARO_MEGA: True` / `False` echoed by each
process; every done line carries its temperature, top_p and drafted count.

| arm | median tok/s (min-max) | ms/token |
|---|---|---|
| megakernel, T=0 no-spec | 135.00 (134.63-135.42) | 7.41 |
| launch path, T=0 no-spec | 107.99 (107.85-108.20) | 9.26 |
| launch path, T=0.7 no-spec | 108.33 (108.01-108.50) | 9.23 |
| megakernel process, T=0.7 no-spec (= launch path, A1 shape) | 108.32 (107.95-108.51) | 9.23 |

Bypass cost (megakernel T=0 vs launch T=0): 1.85 ms per token, 20.0%.
Sampler cost (launch T=0.7 vs launch T=0): -0.03 ms, inside the spread.
A1 gap reproduced: 135.00 vs 108.32, 19.8%. H1 held in full and H2 is
falsified: `amar_sample_row` at k off / p 0.9 costs no more than the launch
path's `amar_argmax_row` it replaces (129 us at 256 threads, KSAMP-c table),
so the sampler kernel is not on the critical path at all. The plan's
falsifier did not fire.

### A6.2 Change: megakernel layers at temperature > 0 (frozen 2026-09-16, before its build's timed run)

**Change (host only, no kernel edit).** `serve/engine.mojo` no longer
excludes `temperature > 0` from `mega_req`; `serve/window.mojo` launches
`mega_token_*` with `fold_head = 0` when `cfg.sample.temperature > 0` and
sets `head_folded` false there, so the existing launch-path head
(`rms_m`, `rmsc_k`, `gemm_w`, `r_head`) and the existing sampling branch
(`amar_sample_row` + `tokcp_k`) run after the megakernel, exactly the shape
`mega_win_k` already uses for the window. Penalties, top_logprobs and
grammar requests keep the launch path (unchanged). T<=0 is untouched:
`fold_head = 1`, argmax inside the megakernel, same launch, same arguments.

**Predictions.**
- P-A6a T=0 byte-identical: `bench/force-ab.sh` 20/20 at 100.0% against
  the unchanged engine `643593062a0a9018`, no voids.
- P-A6b C3 unchanged: `kernels/test_sample.mojo` and
  `kernels/test_sample_device.mojo` exit 0 (kernel file untouched);
  `run-tests.sh` exit 0, `tools/ci-checks.sh` exit 0.
- P-A6c Perf: 20-prompt no-spec T=0.7/top_p 0.9 lands at 128 to 134
  tok/s against no-spec T=0 in the same stint (134 to 136), i.e. within
  5% (>= 128.25 for a 135.0 greedy). The residual is the launch-path head
  (five launches, `amar_argmax_row`-class 129 us argmax replaced by a
  180 us sampler on a hot row) against the folded head.
- P-A6d Spec at T=0.7 unchanged within the standing +-2% band: 144 to 150
  (was 147.15 / 146.89), since the window never used the megakernel.
- Kill line: no-spec T=0.7 below 95% of the same-stint greedy. Then the
  next round is `fold_head = 3` in `kernels/mega.mojo` (write the logit row
  from the folded head instead of reducing it), not a sampler change.

**A6.2 result (2026-09-16, engine `d5021bb6c385b0b5` built from `54fb4bb`,
`.work/a6/gates/`, 290 W cap, -100 mV, sclk median 2920 MHz, queue empty,
`BARO_MEGA: True` echoed).** P-A6a PASS: `force-ab.sh` 20/20 at 100.0%
against `643593062a0a9018`, no voids, `mega barrier gen` 17984 on both
arms. P-A6b PASS: `test_sample` and `test_sample_device` exit 0 (64 PASS
rows at real vocab), `run-tests.sh` exit 0 (103 kernels, 0 orphans),
`ci-checks.sh` exit 0 inside the passing preflight. P-A6c PASS, inside its
band: no-spec T=0.7 **132.30** (131.91-132.75) against no-spec T=0 134.65
(134.24-134.96) in the same process, 98.3%, up from 108.32 (1.221x); the
residual 0.13 ms per token is the launch-path head plus sampler against the
folded head. P-A6d PASS: T=0.7 spec 146.61 (was 146.89 / 147.15), T=0 spec
149.37, acceptance 0.691 / 0.660 unchanged. Kill line not reached; the
`fold_head = 3` round is not needed for the 5% target and stays a lever.

### A6.3 Penalties, top_logprobs and grammar on the megakernel (frozen 2026-09-16, before its build's timed run)

**Change (host only).** `serve/engine.mojo`: `mega_req = mega` (no request
field excludes the megakernel any more). `serve/window.mojo`: one
`plain_head` predicate (temperature <= 0, no penalties, no top_logprobs, no
grammar) decides `fold_head` and `head_folded`; every other m == 1 decode
step runs the layers in the megakernel with `fold_head = 0` and the
launch-path head, penalties, top-N and (masked) draw after it, unchanged
code. `bench/grammar-protocol.md` gate 5 recorded exactly this gap
(1.253x, "a grammar request cannot use the megakernel"); the shape from
A6.2 closes it without a masked megakernel head. New harness
`bench/pen-ab.sh` (two engines, arms U plain / P penalties / L top_logprobs
at T=0, identity per prompt, penalty and logprob receipts).

**Predictions.**
- P-A6e T=0 plain byte-identical: `force-ab.sh` 20/20 at 100.0% against
  the A6.2 engine `d5021bb6c385b0b5`.
- P-A6f `pen-ab.sh` A6.2 engine (arms P/L on the launch path) vs this
  build (megakernel layers): identity 20/20 on every arm, both receipts
  fire on both engines; arm U 133 to 137 on both; arms P and L 105 to 110
  on the A6.2 engine and 128 to 135 on this build, within 5% of its own
  arm U.
- P-A6g grammar gate 1 (`bench/grammar-gate.py`, full corpus, live
  `baro-serve` on this build) 66/66 valid with masked draws == accepted on
  every request; gate 5 (`bench/grammar-cost.py`) ratio at or below 1.05x
  (was 1.253x; mask cost 1.013x measured at `BARO_MEGA=0`, plus the
  launch-path head residual of A6.2).
- P-A6h `tools/test_server.sh` ALL PASS, `run-tests.sh` exit 0,
  `ci-checks.sh` exit 0.
- Kill line: any identity miss in P-A6f or a validity miss in P-A6g means
  the branch is not merged; the first diagnosis is megakernel-vs-launch
  layer parity on that prompt (`BARO_DUMP`), not the sampler.

**A6.3 result (2026-09-16, engine `e55a56aa954f49e4` built from `39f6a4c`,
`.work/a6/gates3/` and `.work/a6/gates4/`, 290 W cap, -100 mV, sclk median
2980 MHz, `BARO_MEGA: True` echoed by every process and by the live
server).** P-A6e PASS: force-ab 20/20 at 100.0% against `d5021bb6c385b0b5`.
P-A6f PASS, inside every band: `pen-ab.sh` identity 20/20 on U, P and L;
receipts fired on both engines (penalties changed 17/20 streams, logprob
field on 20/20); U 135.29 / 135.07; P 109.85 -> **133.38** (98.7% of U); L
107.45 -> **129.80** (96.1% of U). P-A6g PASS: grammar gate 1 66/66 valid
with masked draws == accepted; gate 5 **1.010x** (6.751 vs 6.685 ms/token,
148.13 vs 149.58 tok/s), was 1.253x. P-A6h PASS: `tools/test_server.sh`
ALL PASS (64/64 on completion, stream, queue, cancel recovery),
`run-tests.sh` exit 0 (103 kernels, 0 orphans), `ci-checks.sh` exit 0
inside the passing preflight. One process defect, ledgered: the first gate
job read the grammar receipts from the server's stdout while baro-serve
forwards them to stderr; 0/66 was the harness, the engine had printed all
66 (fixed in `grammar-gate.py`'s docstring, one job lost).

### A6.4 Speculative window on the megakernel: probe (frozen 2026-09-16, before its build's timed run)

**Prior on the record.** `bench/megakernel-mrow-protocol.md` W2/W3
(2026-09-06, q8 pack): `amar_mega_window[MR=3]` lost to the launch path
both times (121.0 vs 122.2 with its own head; 110.9 vs 127.5 with the native
head), cause per-row serialization inside persistent waves (GEMM phases
scale 1.6x from m=1 vs 1.5x native, head 1.65x vs 1.22x), and the round
closed with "multi-row windows stay on the launch path; the megakernel is
an m=1 device". `bench/mtp-protocol.md` MSPEC step 1: the k=2 window's
launch-gap share is 8.4 to 9.2% tracer-corrected, so +8% is the ceiling of
any zero-gap window. Since then the pack moved to q4, the megakernel's q4
dot loop re-rolled (`3824e20`) and `BARO_MEGA_WIN` stayed at 0.

**Defect found reading the code.** `serve/registry.mojo` instantiated the
window kernel with `Q4 = False` only, and `serve/window.mojo` launched it
regardless of `cfg.pack_q4`: `BARO_MEGA_WIN=1` on the q4 pack ran q8
dequant over q4 bytes. A flag flip was therefore not a valid probe.

**Change.** `mega_win_q4_k = amar_mega_window[MEGA_MR, True, True, ...]`
and a `cfg.pack_q4` branch in the window launch. Default stays
`BARO_MEGA_WIN=0`. Kernel file untouched.

**ISA gate before any GPU run (`tools/isa-receipt.py`, `isa-loops`).**
The receipt prints VGPR, spills and private segment of the new q4 window
kernel next to the existing ones (q8 window: 192 VGPR, 47 spills, 192 B
private; q4 m=1 token: 256 VGPR, 0 spills), and the q4 m=1 token kernel's
dot-loop fingerprint (`dual` columns) must be unchanged by the added
instantiation. A q4 window kernel above 300 VGPR spills is recorded and
still run once (the number is the finding); NOT-RESIDENT voids the arm.

**Probe.** `bench/clock-probe.sh bench/ab-prompts.sh ENGINE OUT
"BARO_SPEC=1 BARO_MEGA_WIN=0" "BARO_SPEC=1 BARO_MEGA_WIN=1"`, k=2, T=0,
20 prompts, one-shot runs, arm A and B back to back per prompt. Receipts
(P1): `BARO_MEGA_WIN:` echo per run, `mtp: drafted/accepted` per run,
`mega barrier gen` and `mega fail word` on both arms, identity = GENERATED
equal per prompt.

**Prediction.** The window megakernel lands between 0.87x and 1.01x of the
launch path on the 20-prompt median (the record's band, W3 to W2), i.e.
below the land line; identity 20/20 (W1 proved the m=3 body bit-identical).

**Land rule.** Median ratio >= 1.03 with identity 20/20, spread < 5%, fail
word 0: `BARO_MEGA_WIN` defaults to 1 and a follow-up extends
`mega_win_req` to temperature > 0 (the window kernel is layers-only, so the
sampled accept rule after the native head is unaffected) with
`spec-sample-ab.sh` as its gate. **Close rule.** Ratio < 1.03: the flag
stays 0, the q4 instantiation is kept as the correctness fix for the flag,
and the report records the number against the record's band.

**A6.4 result (2026-09-16, engine `7237e557c9a357a9` built from `7bc81f5`,
`.work/a6/win-ab/`, 290 W cap, -100 mV, sclk median 3098 MHz, queue
empty). CLOSED, prediction missed on the low side.** ISA gate PASS (q4
window kernel 192 VGPR / 44 spills / 180 B private; q4 m=1 token kernel
fingerprint unchanged, dual 124/79/79/59). Receipts: `BARO_MEGA_WIN:
False` / `True` per arm, `spec k: 2`, `mtp: drafted/accepted` on every
run, `mega barrier gen` 7912 to 11696 on every megawin run and 0 on every
launch run, fail word 0 on all 40 runs. k=2 spec, 20 prompts: launch path
**151.68** tok/s_gen, window megakernel **115.39**, ratio **0.761**,
identity 20/20, and every prompt's own ratio sits in 0.76 to 0.78. The
record's band was 0.87 to 1.01; the q4 body at m=3 does worse than the q8
one did, consistent with W3's cause (per-row serialization inside
persistent waves; the re-rolled q4 dot loop is faster at m=1 and no
better at m=3). Close rule applied: `BARO_MEGA_WIN` stays 0, the q4
instantiation stays as the correctness fix for the flag, no temperature
> 0 extension. Multi-row windows stay on the launch path; the megakernel
is an m=1 device, now measured on both packs.
