# Report: DFlash2 speculative decoding (lane-dflash2, 2026-11-16)

Brief: `docs/lanes/2026-11-16-dflash2-brief.md`. Skills loaded: qwen38-nerkyor-coder, gpu-kernels, inference-kernels,
kernel-parity, bench-gates, mojo-lean, mojo-nightly-lane-builder, memory-optimization, rdna3-isa, lanes.

## Flag: the verify window is 8 rows, not 9

llama.cpp `common/speculative.cpp:1100-1102`: "DFlash input is [id_last, <mask> * (block_size-1)]: in-place denoising
yields at most block_size-1 draft tokens". The checkpoint has `dflash_config.block_size = 8`
(`~/Models/qwen38-27b-nerkyor-coder-q4lynn/dflash2-fp8/config.json`), so a DFlash2 cycle drafts 7 tokens and the target
verifies anchor + 7 = **8 rows**. That fits `SM = 8` / `MROWS = 8` as they are: no scoped verify capacity is needed.
SLOTS >= k + 2 = 9 (`serve/engine.mojo:1819`), which is the registry default `KMAX + 1`. The brief's `T_verify9` is
`T_verify8` below.

## D1: 8-row verify window at 27B shapes (falsifier)

Target: nerkyor Q4LynnStyle-Q8MTP requantized to the q4 pack (`../nerkyor-vs-forge/.work/nerkyor/pack-q4`, 505 q4
tensors, pack_sha256 `9485055a...`; in this worktree `.work/nerkyor` is a symlink to that lane's `.work/nerkyor`, so
`.work/nerkyor/pack-q4` is the same pack). The 8-row window runs today as an MTP k=7 window: same trunk, same accept and
ring rollback, a different drafter.

Change: `serve/window.mojo::gemm_w` sends Q4 GEMMs with m > 4 to `amar_fewrow_wmma` D=4 when `-D BARO_FEWROW=1`
(default 0, so no default path changes; m <= 4 stays on turbo). D=4 fits every 27B Q4 shape (K/32 = 160, 192, 544 are
multiples of 32); D=8 fits none. Bench: `bench/bench_turbo_rows.mojo` gains a D=4 arm set for shapes that fail D=8.
Harness: `bench/dflash2-d1.sh` (one gpu-wait job): kernel bench, then three resident A/Bs over the 20 mtp-prompts,
BARO_TMAX 16384, GEN 256 (not max_tokens 16000: 20 prompts x 16000 tokens is hours per arm; short-runs rule).

### Predictions (frozen before the timed run)

| id | claim | pass if |
|---|---|---|
| K1 | WMMA D=4 is flat in m at 27B | m=8 / m=1 <= 1.10, both shapes |
| K2 | WMMA beats today's m>5 path | WMMA m=8 <= 0.70 x q4rowb MR 8, both shapes |
| K3 | 8 rows cost little over the k=2 window GEMV | WMMA m=8 / turbo m=3 in [1.15, 1.45] |
| E1 | FEWROW does not touch m=1 | nospec ids fw1 == fw0, 20/20 |
| E2 | the 8-row window keeps greedy ids | k7 fw1 ids == nospec on >= 18/20 (WMMA order is not bit-exact) |
| E3 | fewrow wins the 8-row window end to end | k7 tok/s fw1 / fw0 >= 1.15 |
| E4 | trunk time of one 8-row window | T_verify8 (attn+ssm+ffn+head per window, profile arm) in [30, 45] ms |
| E5 | break-even acceptance | tau = 68.72 x (T_verify8 + T_draft_est 5 ms) in [2.4, 3.5]: GO (brief stops at tau > 6) |

T_draft_est = 5 ms: 2.30 GB draft weights read once at ~650 GB/s (3.5 ms) + 7 rows through the target LM head (~1.1 ms)
+ selector/conv (small). Revised after D2 measures it.

### D1 result: timing GO, ids gate FAILED for the WMMA route

Receipts: `.work/dflash2/d1/` (run 4, tree 601f691, fw1 `d97aaf7a`, fw0 `bbde51e1`, rows bench `e1ac3099`), one gpu-wait
job, about 14 GPU-minutes. Runs 1-3 are void: two refused at the 1-min load gate (other lanes' builds), one hit a megakernel
grid-barrier timeout (fail word NOT-RESIDENT on 2 of 20 requests) next to a co-tenant GPU job. Run 4 runs every arm with
`BARO_MEGA=0` and `LOAD_START_MAX=20`; load during the arms was 4-8.

Kernel, 27B shapes, real nerkyor weights, us per call (bench median):

| shape | q4rowb m=1 | turbo m=3 | q4rowb MR 8 (today m>5) | WMMA D=4 m=1 | WMMA D=4 m=8 |
|---|---|---|---|---|---|
| ffn_gate 17408 x 5120 | 62.7 | 89.3 | 210.0 | 118.6 | 118.7 |
| ffn_down 5120 x 17408 | 92.2 | 89.0 | 219.0 | 124.6 | 126.1 |

Engine, 20 mtp-prompts, GEN 256, TMAX 16384, resident arms:

| arm | median tok/s_gen | accepted / drafted | ids vs nospec fw0 (m=1 reference) |
|---|---|---|---|
| nospec fw0 (launch path) | 38.92 | - | reference |
| nospec fw1 | 38.55 | - | 3 differ |
| MTP k=2 fw1 (m=3, turbo; route not reached) | 70.94 | 3045 / 4134 (0.737) | 2 differ |
| MTP k=7 fw0 (8 rows on q4rowb MR 8) | 50.15 | 3735 / 9559 (0.391) | 2 differ |
| MTP k=7 fw1 (8 rows on WMMA D=4) | 76.72 | 3806 / 9060 (0.420) | 6 differ |
| MTP k=7 fw1 BARO_PROFILE=1 | 70.67 | 3804 / 9076 | 8 differ |

Profile arm, per 8-row window (about 1296 windows): trunk attn 4.8 + ssm 17.0 + ffn 27.0 + head 1.8 = **T_verify8 50.6
ms** with sub-block syncs; the MTP draft of 7 steps took 10.1 ms. Unprofiled estimate from the k7 fw1 arm: 3.94 tokens per
window at 76.72 tok/s = 51.3 ms per cycle, minus the ~8-10 ms MTP draft = **T_verify8 about 41-43 ms**.

| id | result | verdict |
|---|---|---|
| K1 | m=8 / m=1 = 1.001 (gate), 1.012 (down) | PASS |
| K2 | WMMA m=8 / q4rowb MR 8 = 0.565, 0.576 | PASS |
| K3 | WMMA m=8 / turbo m=3 = 1.33, 1.42 | PASS |
| E1 | nospec fw1 vs fw0: 3 of 20 differ | FAIL: serve mode sets `replay_m=MROWS` (`serve/engine.mojo:891`), so prompt-replay windows of 5-8 rows take the WMMA route. FEWROW is not m=1-only. |
| E2 | k7 fw1: 14/20 equal to the m=1 reference | FAIL. Controls on the shipped kernels: k2 18/20, k7 fw0 18/20, and the two fw1 k7 runs disagree with each other (6 vs 8). The WMMA route adds about 4 flips over this pack's baseline of 2. |
| E3 | k7 fw1 / fw0 = 1.53 | PASS |
| E4 | 50.6 ms profiled, ~42 ms unprofiled | FAIL as frozen (profiled number above 45) |
| E5 | tau = 68.72 x (T_verify8 + 5 ms) = 3.23 (unprofiled) to 3.82 (profiled); 3.33 against this pack's own k=2 (70.94) | PASS of the brief's line (tau <= 6) and of the plan's "<= ~4" |

Findings beyond the predictions:
- Greedy ids on this pack are not stable even on shipped kernels (2 of 20 flip at k=2 and k=7 fw0). An ids-equality gate
  at 20/20 cannot pass here for any arm. A gate for a non-bit-exact window needs the teacher-forced agreement gate
  (`BARO_FORCE`) with a same-arm control (memory: P5b bar unpassable on base).
- MTP at k=7 on the WMMA route already beats MTP k=2: 76.72 vs 70.94 tok/s (1.08x, same job), tau 3.94 at k=7. Plan P0
  (k sweep) should include k = 4..7 with BARO_FEWROW=1.
- The cost of the 8-row window is the SSM (17 ms, linear in m) and the FFN at 400 GB/s WMMA (27 ms). A bit-exact route
  (k7 fw0) leaves tau_break at about 4.7: still under 6.

Verdict: the timing falsifier passes (tau_break 3.2-3.8). The ids part of the gate fails for the WMMA route, and the brief's
D3 gate ("greedy ids equal to no-spec on 20 prompts") cannot pass on this pack for any arm (control 18/20). Decision for
the boss below; D2 (drafter) does not depend on it and continues.

Decision for the boss: D3 gate = teacher-forced agreement vs the m=1 arm with a same-arm control (recommended), or keep
greedy ids and run D3 on the bit-exact q4rowb route (tau_break about 4.7).

**Boss ACK (2026-11-16):** next = D2; D3 gate = teacher-forced agreement with a same-arm control; bar frozen before the
run; smallest tests.

## D2: drafter

### D2a (landed): draft pack from the boss's Q8_0 GGUF

`~/Models/qwen38-27b-nerkyor-coder-q4lynn/Qwen3.8-27B-nerkyor-DFlash2-Q8_0.gguf` (arch `dflash`, 81 tensors, every matrix
Q8_0, norms and conv bases F32) is what llama.cpp's DFlash2 runs, so a pack of it compares like with like. The FP8
safetensors (1375 MiB FP8 + 920 MiB BF16) are not needed. `tools/engine-pack.mojo --arch dflash` reuses the lexical
qwen35moe path and splits every Q8_0 matrix into the engine q8 layout (qs, then f16 scales).
Receipt: `.work/dflash2/draft-pack`, 49 q8 + 32 f32, pack.bin 2,045,465,600 B (GGUF 2,056,414,240 B; the gap is the
GGUF header/tokenizer KV), pack_sha256 `4b39f396...`, 3.40 s, 384 MB RSS.

### Port semantics (llama.cpp, read 2026-11-16)

- Taps: `dflash.target_layers` = 6/20/34/48/62 = target_layer_ids + 1 (`conversion/qwen.py:898-901`): the residual
  stream OUTPUT of target layers 5/19/33/47/61, concatenated to 25600 per token.
- Injection (`src/models/dflash.cpp` graph<false>, embd branch): per verified token at its position p:
  `g = rmsnorm(fc @ feat, enc.output_norm)`; per draft layer `K = rope(rmsnorm_head(Wk g, k_norm), p)`,
  `V = Wv g` (no norm, wv present), written to the draft KV cache (SWA 2048).
- Draft block (`common/speculative.cpp:1340-1360`): tokens `[id_last, mask 248070 x 7]` at positions n..n+7 (n = pos of
  id_last), embedded with the TARGET tok_embd. Per layer: `x0 = rmsnorm(h, attn_norm)`; `dyn = attn_conv_proj @ x0`;
  `x = conv(x0, dyn, attn_conv_base, side 0)`; q/k/v from x, q_norm/k_norm per head, rope; non-causal attention over the
  cache window plus the block; `o = Wo att`; `o = conv(o, dyn, base, side 1)`; `h = h + o`; same pattern around the FFN
  (ffn_norm, ffn_conv_proj, side 0 before the SwiGLU, side 1 after), `h = h + f`. Then `output_norm`, target LM head.
- Two-tap conv (`build_dflash2_conv`): `dyn` reshapes to [n_groups=320, kernel 2, side 2] per token; weight =
  `dyn[g, tap, side]` broadcast over the 16 channels of group g + `base[side, tap, ch]`; out[t] = w0 * x[t] + w1 * x[t-1],
  x[-1] = 0 inside each 8-token block.
- Selector (`build_dflash2_selector`): per position, top-16 of the draft logits (full vocab), unary = those logits;
  `gate = selector_hidden @ h_final` (rank 256); score[i][j] = unary[j] + succ[cand_j] . (pred[prev_i] * gate).
  Position 1's predecessor is the anchor id.
- Greedy walk (`speculative.cpp:1394-1440`): predecessor = 0; for i = 1..7: scores = lattice row i at predecessor;
  predecessor = argmax; draft token = cand[i][predecessor].

### D2 plan (remaining)

| step | files | size | check |
|---|---|---|---|
| D2b taps (built, UNVERIFIED) | `serve/window.mojo`: `tap_d` f32[5, MROWS, H] + `tap_on` in WindowBufs, one copy per tapped layer at the end-of-layer hook of the window loop; `serve/harness.mojo` allocates it only under `BARO_SPEC=dflash2` (else 1 float). Prompt rows reach the taps through window replay (`BARO_PF_ROWS=0`); prefill-path taps are D3 work. | S | engine builds (`c7a2f836`, CPU only). Layer and offset correctness is checked by the D2 gate itself (draft logits vs llama.cpp; the reverse arm shifts the taps one layer); no separate GPU run, per smallest-tests |
| D2c draft forward | new `serve/dflash.mojo`: injection + 5 layers + conv + selector + walk, all on existing q8 GEMV / rmsnorm / rope / attention kernels | L | per-position top-1 and top-16 overlap vs llama.cpp on dumped inputs; reverse arm: taps shifted one layer must drop acceptance |
| D2d reference dumps | own worktree `~/Projects/imports/llama.cpp-lanes/dflash2-dump` (branch dflash2-dump, 2cfcd22b4 + ff30654a3, local only): `LLAMA_DFLASH_DUMP=path` writes `'F' seq n_chunk 25600 \| pos[n] \| f32 feat[n x 25600]` per injection chunk (host feature path forced) and `'L' seq pos0 id_last n_block 272 \| f32 lattice[n_block x 272] \| n \| ids` per draft cycle (272 = 16 cands + 16x16 scores). Harness `bench/dflash2-d2d.sh`: target ngl 52 (boss: 99 and 60 OOM), draft on GPU, 7 drafts, greedy, ctx 16384, 64 tokens, prompts p01 p02 | S | dump files non-empty, records parse |

### D2d result (landed): reference dumps

`.work/dflash2/d2d/` (gpu-wait priority 90, llama dflash2-dump ff30654a3, target ngl 52, draft ngl 99, k 7, greedy, ctx 16384):

| prompt | dump B | F chunks / rows / distinct pos | L cycles | llama drafted / accepted | decoded tokens | tokens per cycle |
|---|---|---|---|---|---|---|
| p01-water | 12,513,764 | 15 / 121 / 83 | 14 | 95 / 53 (55.8%) | 67 | 4.8 |
| p02-python-fib | 10,960,152 | 13 / 106 / 85 | 12 | 81 / 56 (69.1%) | 68 | 5.7 |

Every record parses to EOF (cols 272, block 8, feature width 25600). Rows exceed distinct positions: the server re-injects
verify rows at positions it later overwrites, so the replay keeps the latest feature per position. llama.cpp's tokens per
cycle (4.8, 5.7) sit above the D1 break-even (3.2-3.8): two prompts, a receipt, not a claim. llama.cpp ran 17.8 / 20.6 tok/s
here at ngl 52 (partial offload, not a speed comparison).

### D2c build plan (kernel survey 2026-11-16)

The target kernels bake the target shape in as comptime globals (HD 256, NQH 24, NKVH 4), so attention cannot be reused
for the draft (HD 128, 32 q heads, 8 KV heads, SWA 2048, non-causal). Reuse vs new:

| op | reuse | note |
|---|---|---|
| every projection (fc 25600->5120, q/k/v/o, conv proj 5120->1280, gate/up/down, selector_hidden 5120->256) | `serve/registry.mojo::gemm_q8` (runtime m, n, k; bf16 A, q8 pack) | the draft pack is all q8 |
| rmsnorm (hidden, per-head q/k with n=128) | `kernels/elementwise.mojo::amar_rmsnorm` / `amar_rmsnorm_cast` (runtime n) | per-head = rows x 128 view |
| rope | `kernels/spark_kernels.mojo::amar_rope_plain[NROT 128, NEOX]` (runtime pos, nh) | one launch per row first; NEOX vs normal: read llama's rope type for arch dflash before use |
| SwiGLU | `kernels/elementwise.mojo::amar_swiglu` | |
| LM head over 8 rows | the window's head GEMM (`Wheadq4`, m rows) | target head, full vocab |
| top-16 per row | the sampler top-n kernel behind `topn_ids_d` | check it takes a 248320-wide row |
| draft attention | NEW `amar_df_attn`: one block per q head, 8 query rows, keys = window cells + the block | ~60 LOC |
| two-tap conv | NEW `amar_df_conv`: out[t] = (dyn[t,g,0,side] + base[side,0,c]) x[t] + (dyn[t,g,1,side] + base[side,1,c]) x[t-1], x[-1] = 0 | ~25 LOC |
| selector lattice | NEW `amar_df_lattice`: gather succ/pred codebook rows (q8), score[i][j] = unary[j] + succ[c_j] . (pred[p_i] * gate) | ~50 LOC |
| draft KV | NEW buffers in `serve/dflash.mojo`: K/V f32[5 layers, 2048 ring, 8 heads, 128], latest injection per position wins | |

Parity harness `bench/dflash_parity.mojo` (oracle = the D2d dump): replay F records into the draft KV, run the block at
each L record's pos0 with its anchor, compare top-16 candidate sets per position (top-1 equal, overlap /16), the walked ids,
and score max-abs. Reverse arm: same features shifted one record (wrong positions) must lose top-1 agreement.

Memory: k=7 arms peaked at 21.7 GiB VRAM (SLOTS 9, f32 KV 16k). The draft adds 1.90 GiB plus its KV, so D3 needs
`BARO_KVQ=int8` (saves about 1.5 GiB at 16k) or the MTP head left out of the pack.

### D2c host side (2026-11-16)

Files: `serve/dflash.mojo` (struct `DFlash`: draft pack + ring + buffers, `inject`, `draft`), `bench/dflash_parity.mojo`
(dump reader, CPU walk check, GPU replay), `kernels/dflash.mojo` renamed `kernels/dflash_kernels.mojo` (with `-I kernels
-I serve`, `serve/dflash.mojo` shadows it; same fix as `spark_kernels`). Kernel changes: `amar_df_attn` writes bf16 (its
only reader is the Wo GEMM); `amar_df_conv[H, GS, ODT, ACC]` writes bf16 on side 0 (the next GEMM's A) and accumulates
f32 on side 1 (the residual add); new `amar_df_put` (rows into ring cells).

Semantics re-read from `src/models/dflash.cpp` before the build: `build_attn` stores the block's own K/V before it
attends, so `draft` writes block rows to ring cells pos0..pos0+7 first and the next injection overwrites them; SWA stays
2048 (the `n_swa += 1` at `dflash.cpp:101` is the DSV4 single-pass branch only); the GGUF has no embedding, value or
logit scale; the attention conv's `dyn` comes from `attn_norm(h)` and serves both sides, the FFN conv's from
`ffn_norm(ffn_inp)`.

Step 1 receipt (CPU, 1832d16): p01 F 15 / rows 121 / pos 83 / L 14, p02 13 / 106 / 85 / 12, host walk == dumped ids 26/26.

#### D2c parity gate (frozen before the first GPU run)

One gpu-wait job: `dflash_parity --gpu .work/dflash2/draft-pack ../nerkyor-vs-forge/.work/nerkyor/pack-q4 p01 p02`.
Replay in file order (F -> inject, L -> draft at the record's pos0 and anchor), ring reset per dump. Rows i = 1..7 of the
26 L records (182 rows).

| id | measure | bar |
|---|---|---|
| B1 | forced decision: our argmax at the dumped predecessor == the dumped draft id | >= 0.90 of 182 rows |
| B2 | candidate overlap of our top-16 with the dumped set | mean >= 14.0 / 16 |
| R1 | reverse arm (every feature at position + 1): forced decision rate | <= 0.5 x B1 |
| -  | free greedy walk (cycles with all 7 ids equal, mean equal prefix); score abs diff over (prev id, cand id) pairs present in both lattices (mean, max) | reported, no bar |

The handoff's "top-1 equal" is B1: the dump keeps candidates unordered and has no unary column, so the logits' top-1
cannot be recovered from it; the forced decision is the draft's top-1 at each position on the same path.
Why the bars are below 1.0: our activations enter every q8 GEMM as bf16, llama's as q8_1; llama ran the original GGUF's
LM head, we run the q4 pack's re-quantized head; ggml's softmax and rope kernels differ from ours.

#### D2c result: the forward arm matches llama.cpp; R1 FAILED as frozen

Receipt: `.work/dflash2/parity/run1.log` (tree da0c52d, one gpu-wait job, priority 90, under 1 GPU-minute).

| arm | forced decision | overlap /16 | walk all 7 equal | mean equal prefix /7 | score abs diff mean / max (pairs) |
|---|---|---|---|---|---|
| fwd p01 | 95 / 98 | 15.22 | 11 / 14 | 6.50 | |
| fwd p02 | 81 / 84 | 15.36 | 10 / 12 | 6.75 | |
| **fwd pooled** | **0.967** (176/182) | **15.29** | 21 / 26 | 6.62 | 0.160 / 0.83 (36,822) |
| rev p01 | 51 / 98 | 8.75 | 0 / 14 | 2.00 | |
| rev p02 | 64 / 84 | 9.07 | 4 / 12 | 4.17 | |
| **rev pooled** | **0.632** | 8.90 | 4 / 26 | 3.00 | 1.97 / 22.4 (13,457) |

| id | result | verdict |
|---|---|---|
| B1 | 0.967 | PASS |
| B2 | 15.29 | PASS |
| R1 | 0.632 > 0.5 x 0.967 = 0.484 | FAIL |

Annotation (does not change the verdict): the reverse arm moves every measure that is not forced. Walk all-7 falls from
21 to 4 of 26, overlap from 15.3 to 8.9, score diff mean from 0.16 to 1.97. The forced decision stays at 0.63 because it
gives the lattice the true predecessor, and the selector's pairwise term `succ[c] . (pred[prev] * gate)` alone picks many
continuations. R1 measured a weak falsifier, not a wrong forward pass: score max-abs 0.83 over 36,822 matched pairs on the
forward arm needs every layer, the conv, the ring and the selector to agree with llama.

Decision for the boss: accept D2c on B1/B2 and let D3's acceptance gate carry the reverse control (wrong features must
drop accepted tokens per cycle), or freeze a new reverse bar on an unforced measure (walk all-7 or overlap) and rerun the
same job (under 1 GPU-minute). The draft's time per block is not measured yet: the host top-16 over 7 x 248,320 logits
stands in for a GPU top-k and is D3 work.

**Boss ACK (2026-11-16):** D2c accepted on B1/B2; R1 FAIL stays recorded, no new bar for D2c; D3's reverse control gets
its bar frozen before the run; next = D3.

## D3: integrate

Wiring (616a223): `BARO_SPEC=dflash2` loads the drafter beside the target pack and shares the target's token_embd and LM
head. Each cycle drafts 7 ids at the last committed position, writes them to `toks_d` and verifies them through
`step_window`'s ngram path (greedy accept, ring rollback). After every window the kept rows of `tap_d` go into the draft
ring through `amar_df_gather`, so prompt replay windows feed it too. The taps exist only in the launch-path layer loop,
so dflash2 turns prompt prefill, the megakernel and prefix reuse off. Under `BARO_FORCE` the window carries the reference
tokens and the draft is scored against them: one forced run measures the 8-row verify numerics and the draft's
acceptance. `BARO_DF_SHIFT=k` injects every feature at position + k.

### D3 gate (frozen before the gate run)

Harness `bench/dflash2-d3.sh ENGINE_FW1 ENGINE_FW0 PACK OUTDIR`, one gpu-wait job, 20 mtp-prompts, GEN 256, TMAX 16384,
every arm resident with `BARO_MEGA=0`. Reference = FW1 `BARO_SPEC=0 BARO_PREFILL=0` greedy ids (m=1 decode, prompts
replayed as in the dflash2 arms).

| id | measure | bar |
|---|---|---|
| G1 | forced agreement, pooled over all positions, f1 (FW1 = WMMA 8-row route) vs the same-arm control f0 (FW0 = shipped q4rowb route), both vs the reference | f1 >= f0 - 0.5 pp |
| R1 | reverse arm f1s (f1 + `BARO_DF_SHIFT=1`): forced draft acceptance, tokens per cycle (accepted + 1) | f1s <= 0.75 x f1, and predicted ids equal to f1's on 20/20 (the draft must not reach the verified ids) |
| G3 | live `BARO_SPEC=dflash2` vs MTP k=2, median tok/s_gen, same binary, same job | ratio > 1.00 |
| -  | accepted-prefix histogram (forced and live), live ids vs the reference, draft seconds per arm, f0/f1 per-prompt min agreement | reported, no bar |

Known costs at this stage, not bars: the draft's top-16 runs on the host (8 MB logits copy and a 1.7 M-element scan per
cycle); `kpos` uploads per call; injection launches rope once per row.

### D3 run 1 (tree dc3a3da): G1 PASS, R1 PASS, G3 FAIL

Receipt `.work/dflash2/d3/` (one gpu-wait job, priority 90; fw1 `2917d66b`, fw0 `04594e97`). Smoke before it
(`.work/dflash2/d3-smoke`, 2 prompts, GEN 64): forced 128/128, live ids equal 2/2.

| arm | forced agreement (pooled, min prompt) | tokens/cycle | cycles | draft s | tok/s_gen median |
|---|---|---|---|---|---|
| ref (m=1) | - | - | - | - | 39.86 |
| f1 forced, WMMA route | 0.99824 (5111/5120), 0.984 | 4.98 | 644 | 6.74 | |
| f0 forced, shipped route | 0.99980 (5119/5120), 0.996 | 4.92 | 641 | 7.84 | |
| f1s forced, shift 1 | 0.99824, predicted == f1 20/20 | 3.10 | 644 | 6.76 | |
| live dflash2 | ids differ from ref on 5/20 | 3.94 | 1298 | 13.55 | **68.96** |
| MTP k=2 | accepted 3048/4126 | - | - | - | **72.04** |

| id | result | verdict |
|---|---|---|
| G1 | f1 0.99824 vs f0 0.99980 - 0.005 | PASS |
| R1 | f1s 3.10 <= 0.75 x 4.98 = 3.73; predicted equal 20/20 | PASS |
| G3 | 68.96 / 72.04 = 0.957 | FAIL |

Forced histogram f1 (accepted 0..7): 86 110 59 42 43 33 27 244; live: 335 229 150 104 73 58 51 298. Live tokens per cycle
(3.94) sit below the forced figure (4.98): forced drafts are evenly spaced (one per kept window), live cycles crowd into
hard stretches, where every miss starts a new short cycle.

Limiter: the draft costs 10.4 ms per live cycle (13.55 s / 1298) of a ~57 ms cycle, against the 5 ms estimate. The
host top-16 (8 MB logits copy + a 1.7 M-element scan) is the named suspect. At 4 ms per draft the same acceptance gives
about 69 x 57 / 50.6 = 77.7 tok/s_gen, above MTP k=2. Next: a GPU top-16 feeding the lattice kernel directly, then the
same gate again (bars unchanged) with the D2c parity replay in the same job.

### D3 run 2 (tree 3077687, GPU top-16): G1 PASS, R1 PASS, G3 FAIL; the named lever was wrong

Receipt `.work/dflash2/d3r2/` (parity replay + gate, one gpu-wait job). Parity unchanged: B1 0.967, B2 15.29.

| arm | forced agreement | tokens/cycle | draft s | tok/s_gen median |
|---|---|---|---|---|
| f1 | 0.99844, min 0.988 | 4.98 | 6.63 | |
| f0 | 1.00000 | 4.93 | 7.45 | |
| f1s | 0.99844, predicted == f1 20/20 | 3.10 | 6.65 | |
| live | ids differ from ref on 5/20 | 3.94 | **13.49** (run 1: 13.55) | **70.18** |
| MTP k=2 | | | | **71.79** |

G1 PASS (0.99844 vs 1.00000 - 0.005), R1 PASS (3.10 <= 3.73, 20/20), G3 FAIL (0.978). The GPU top-16 left the draft time
where it was (10.4 ms per cycle): the host top-16 was not the limiter. I named it without a phase split. The
live-arm gain (68.96 -> 70.18) is inside run-to-run spread (MTP k=2 moved 72.04 -> 71.79 with no change).

Candidate cause, NOT measured: the draft's GEMMs. One block reads about 2.4 GB (5 layers x ~330 MB of q8 + the 0.7 GB q4
head), all at m = 8, where q8 rows take `amar_matmul_skinny_q8row[4, SM]` (QV 8). D1 measured the q4 equivalent at about
210 GB/s, not the 650 GB/s the 5 ms estimate assumed; 2.4 GB at 210-300 GB/s is 8-11 ms. Next step proposed: a phase
split of the draft (sync timers per stage, one 2-prompt run, S), then pick the lever by its number. Levers in view: a q4
draft pack on the `amar_fewrow_wmma` route (half the bytes, the 8-row kernel D1 measured at ~380 GB/s; draft quality
must re-pass the D2c gate), or an 8-row q8 WMMA kernel.

**Boss ACK (2026-11-16):** one more lever, chosen from the draft phase split; then lane-merge as opt-in if G1/R1 hold;
G3 0.978 recorded open; stop after this lever.

### Draft phase split (predictions frozen before the run)

`BARO_DF_PROF=1` puts a sync timer after each stage of `draft` and around `inject` (serialised, so the stage sum exceeds
the unprofiled draft). Run: live dflash2, 2 mtp-prompts (p01, p02), GEN 256, TMAX 16384, resident, `BARO_MEGA=0`.

| id | claim | pass if |
|---|---|---|
| P1 | the FFN GEMMs dominate | ffn >= 50% of embed + attn + ffn + head + select |
| P2 | the select stage is small since the GPU top-16 | select < 1.0 ms per draft |
| P3 | the q8 draft GEMMs run far under the 650 GB/s the estimate assumed | attn + ffn bytes (5 x 330 MB) / (attn + ffn ms) < 400 GB/s |

Result (`.work/dflash2/prof/live.log`, tree c32abc5; ms per draft, p01 / p02): embed 0.04 / 0.03, attn 2.49 / 2.46, ffn
5.56 / 5.55, head 1.74 / 1.73, select 0.49 / 0.47; inject 0.91 / 1.00 (after the window, outside the draft).

| id | result | verdict |
|---|---|---|
| P1 | ffn 5.56 of 10.29 ms = 54% | PASS |
| P2 | select 0.48 ms | PASS |
| P3 | 1.77 GB (q8 incl. scales) / 8.03 ms = 220 GB/s; ffn alone 262, attn 128; the q4 head on the WMMA route 411 | PASS |

The q8 GEMMs at m = 8 (`amar_matmul_skinny_q8row[4, SM]`, QV 8) are the limiter. The lever (one, per the ACK): the
int8-dot kernel the trunk already has for m >= 5 (`quant_rows` + `gemm_q8dot`, memory M3-dot: wins at m >= 5) on every
q8 draft GEMM, `BARO_DF_DOT=1` default, 0 = the bf16 route. It quantizes activations per 32 values, as llama's q8_1 does,
so the D2c parity gate runs again.

#### Lever predictions (frozen before the run)

| id | claim | pass if |
|---|---|---|
| L1 | the dot route cuts the q8 draft GEMMs | attn + ffn <= 0.70 x 8.03 ms = 5.62 ms per draft (profile arm, same 2 prompts) |
| L2 | parity holds | D2c B1 >= 0.90 and B2 >= 14.0 on the dot route |
| L3 | the D3 bars | G1, R1 as frozen; G3 judged again (ratio > 1.00) |

### D3 run 3 (tree 6bbfa3c, int8-dot draft): every bar passes

Receipt `.work/dflash2/d3r3/` (parity replay, profile, gate: one gpu-wait job, priority 90).

| id | result | verdict |
|---|---|---|
| L1 | attn + ffn 2.04 + 3.42 = 5.45 (p01), 1.91 + 3.32 = 5.23 (p02) ms per draft, <= 5.62 | PASS |
| L2 | D2c B1 0.9725 (was 0.967 on the bf16 route), B2 15.30 | PASS |
| G1 | f1 0.99844 vs f0 1.00000 - 0.005 | PASS |
| R1 | f1s 3.106 <= 0.75 x 4.967; predicted equal 20/20 | PASS |
| G3 | live **73.58** vs MTP k=2 **71.99** tok/s_gen, ratio **1.022** | PASS |

Draft per live cycle 9.56 s / 1296 = 7.4 ms (run 2: 10.4). Live tokens per cycle 3.95, ids differ from the m=1 reference
on 5/20 (WMMA route; reported, no bar). The G3 margin is thin: the live arm moved 68.96 -> 70.18 between runs 1 and 2
with no draft change (1.8%), MTP k=2 stayed in 71.79-72.04. A claim of "faster than MTP k=2" needs a sequential A/B
(`ab-resident.sh SEQ=1`) before it leaves this report. Open costs, in order: the q4 head on 8 rows (1.73 ms), attention
projections at 128 GB/s (small matrices), inject 0.72 ms, rope launched per row in inject.

### Wavewait captures of the new kernels (merge prep)

`bench/ww.sh KERNEL engine-d3 pack-q4 OUT BARO_SPEC=dflash2 BARO_MEGA=0`, 12th dispatch, p01 prompt; receipts
`.work/dflash2/ww/{amar_df_attn,amar_df_conv,amar_df_lattice}/ww.json` (+ classes.txt, top.txt). No spills in any.

| kernel | top stall classes (share of cycles) | reading |
|---|---|---|
| `amar_df_attn` | waitcnt vmcnt 48.4%, lgkmcnt 11.3%, lds 4.3%, barrier 2.4% | one key per loop trip, its K load waited on before the dot and two barriers per key: latency-bound; tiling keys (several per trip) is the lever if attention ever matters (it is inside attn's 2 ms, mostly GEMMs) |
| `amar_df_conv` | lgkmcnt 39.1%, vmcnt 29.3% | elementwise, one 1.1k-cycle wait on the scalar side/base loads; small |
| `amar_df_lattice` | one lgkmcnt wait of 11.2k cycles (scalar loads of cand/gate), vmem 11% | 16-thread blocks fill half a wave32 and each thread runs a 256-long rank dot serially; part of select's 0.45 ms |

### Merge receipts

- main merged into the lane first (380c526; only `docs/rules.md` overlapped, both sides kept).
- `bench/preflight.sh`: PASS preflight 7c900afb2140 (ci-checks clean after 28621aa: census, settings, OPERATE-AUDIT).
- run-tests: `RUN_ALL=1 ./run-tests.sh` under gpu-wait on tree 28621aa: "PASS run-tests: all tests, pack
  .work/engine-pack-q4", rc=0; log `.work/dflash2/run-tests.log`.
- Default behaviour unchanged: dflash2 runs only under `BARO_SPEC=dflash2`; MTP stays the default spec path.
