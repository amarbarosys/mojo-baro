# MoE batched prefill (lane MOEPF): design and preregistration

Brief: `~/Brain/mojo/mojo-baro/briefs/2026-09-19-moe-batched-prefill.md`. Base `ff8e523`
(lane-r63 head; `main` lacks `BARO_SEQ_CAP`, which gate 3 needs). This file is frozen by its
commit, before any timed run.

## Question

qwen35moe replays every prompt token through the m=1 decode path (`pf_on = not IS_MOE`).
Does a chunked m > 1 prefill reproduce the replay path's greedy tokens on all 20
`bench/mtp-prompts`, in resident and host-RAM expert modes, and what prefill tok/s does it reach
at 1k, 8k and 32k prompt tokens?

Held constant: pack, tokenizer, sampler (greedy), KV format, `BARO_MEGA=0`, the decode path
after the prompt. Treatment: prompt rows 0 .. n-2 go through `moe_prefill_forward` in chunks of
`BARO_PREFILL_C` (default CP = 1024) instead of n-1 calls of the m=1 window.

## Design

1. **Same arithmetic per output element as the m=1 kernels, a token axis added to the grid.**
   Every trunk projection (`moe_matmul_q8d_m1`, `_add`), the router, top-8, the routed gate/up,
   the routed down, the shared expert and the embed get a `_rows` sibling whose thread handles
   (token t, output row r): it calls the SAME dot helper (`q8d_row_dot`, `q4k_dot_blocks`,
   `q6k_row_dot`, `q8_0_row_dot`) with `x_row = t`. Summation order per element is unchanged, so
   these phases are bit-identical to replay by construction and need no tolerance.
   The routed down keeps the m=1 form (one thread sums the token's 8 experts in top-k order,
   `out += WT[j] * dot`), so the accumulation order over experts is unchanged too.
2. **Why not a WMMA grouped GEMM first.** It changes summation order, so identity becomes a
   forced-agreement question with a bar the unpatched base is known to miss (P5b). The row-batched
   form already removes what makes replay slow: 727 launches per token become 727 per chunk, and
   a projection's weights (at most 17 MB) plus a chunk's activations (4 MB) sit inside the 96 MB
   Infinity Cache for the whole launch, which m=1 replay can never do across 40 layers. A grouped
   WMMA GEMM is round 2, only if the numbers below leave a reason for it.
3. **Token order for the experts.** Round 1 runs the routed kernels in (t, j) order, no sort. The
   sort (ATOM's align step) only changes which thread handles which (t, j); it cannot change a
   value. It is round 1b, adopted only if it measures faster, because a layer's 256 experts are
   453 MB and an unsorted grid cannot keep them in the last-level cache.
4. **Attention and SSM reuse the dense chunk kernels** (`attpw_k`, `conv_p`, `deltaw_p`,
   `gates_p`, `l2_p`, `gated_p`) at the MoE dims. These are NOT bit-identical to the decode
   kernels by construction (WMMA attention, chunked scan). Gate 1 decides: if any prompt
   diverges, the offending phase is replaced by a row-batched sibling of the decode kernel and
   the divergence is reported with its token index either way.
5. **Tier mode: one layer of experts streamed once per chunk.** With 1024 rows x 8 experts a
   chunk touches nearly all 256 experts of a layer, so the 64-slot cache and its per-row
   `prepare` are the wrong tool. Prefill bypasses the cache: per layer, the layer's gate/up/down
   blocks are copied from the pinned store into one VRAM staging buffer (about 453 MB, allocated
   only when the tier is active), and the routed kernels address it by expert id exactly as they
   address the resident pack. One kernel set serves both modes; the bytes are the same bytes, so
   identity carries over. Cost: 40 x 453 MB = 18 GB per chunk over PCIe 4.0 x16. The cache is
   left untouched, so decode after prefill starts from the same slots as before.
   Rejected: zero-copy reads in-kernel (every (t, j, r) thread would cross PCIe, thousands of
   times the bytes); per-row `prepare` (a host sync per row per layer, which is replay).

## Gates

G1 identity: `bench/moe-prefill-identity.sh`, 20 prompts x 64 greedy tokens, replay
(`BARO_PREFILL=0`) vs batched, same binary, both modes. PASS = 20/20 equal. A prompt shorter than
PF_MIN + 1 tokens does not exercise prefill and is reported as such, not counted as a pass.
The arm file records `pf_rows` from the engine's own echo per request (P1): a batched arm whose
echo reads 0 rows is VOID.

G2 speed: `bench/moe-prefill-speed.sh`, prompt lengths 1024, 8192, 32768 (fixed ids from
`bench/prefill-prompts`), replay vs batched, both modes, same binary, same stint, alternated,
3 repeats, first run after load discarded. Prefill tok/s = rows / (wall from request accept to
first token), measured by the harness clock outside the engine, with the engine's own number
next to it. Identity of the first 16 generated tokens checked on every timed run. Replay at
32k is one repeat (about 8 minutes per run at 67 tok/s).

G3 long context: one served request of about 100k tokens, tier mode, `-D BARO_SEQ_CAP=1`,
`BARO_TMAX=131072`, needle near the end, shape of `.work/profiles-exp/step0/needle.json`.
PASS = the answer contains the needle value.

G4: `run-tests.sh` and `tools/ci-checks.sh` exit 0 on the committed tree.

## Registered predictions (round 1, bit-exact rows, unsorted experts)

Replay baselines assumed from prior receipts: resident about 112 tok/s, tier about 67 tok/s.

| cell | prediction (prefill tok/s) | mechanism |
|---|---|---|
| resident 1k | 600 to 1500 | launch cost amortized 1024x; trunk GEMMs in cache; experts HBM-bound, unsorted |
| resident 8k | 500 to 1300 | attention share grows |
| resident 32k | 300 to 900 | chunk attention over 32k keys dominates |
| tier 1k | 350 to 900 | resident minus 18 GB per chunk over PCIe (about 0.8 s per chunk at 22 GB/s) |
| tier 8k | 330 to 850 | same |
| tier 32k | 250 to 700 | same |
| G1 | 20/20 resident, 20/20 tier | MoE phases bit-exact; risk sits in attention and SSM chunk kernels |

Confidence order, most to least sure: tier slower than resident at 1k; every cell above 3x
replay; resident 1k inside band; G1 20/20 (least sure, item 4 of the design).

Falsifiers: any cell under 3x its replay number means the row-batched form is not the lever and
round 2 (grouped GEMM) opens with a per-kernel profile (`rocprof-kernels`) first. Tier within
10% of resident means the copy is not on the critical path and the staging design is oversized.
A G1 failure in a phase claimed bit-exact (item 1) is a bug, not a tolerance question.

Adoption: `pf_on` default flips to on for IS_MOE only if G1 is 20/20 in both modes, G3 passes
and G4 is green. No speed threshold (the brief sets none); numbers are reported as measured.

## Outputs

`.work/moepf/<gate>/<stamp>/` (arm.txt, per-prompt logs, summary.tsv), report
`exchange/lane-MOEPF-report.md`.
