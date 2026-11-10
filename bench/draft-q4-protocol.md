# q4 draft head protocol (M4) — frozen before first run, 2026-09-04

Binds `bench/PROTOCOL-RULES.md` P1-P6. Facts: the MTP draft path costs
2.4 ms per window, 63% of it the 1.06 GB q8 LM-head read per drafted token
(`bench/mrow-gemm-protocol.md`, draft-path split). The draft's tokens never
reach the output: a wrong draft costs acceptance, not correctness. So the
draft head may be quantized below the trunk without touching the 64/64 gate.

Plan: `tools/engine-pack.py --q4-draft` appends a Q4_0 copy of
`output.weight` (and optionally the blk.32 ffn/attn 2D weights) as extra
pack entries (18 B / 32 weights, byte-equal to `llama-quantize Q4_0` blocks,
checked by `tools/q4-check.py` as in the q8 round). Kernel
`amar_matmul_skinny_q4row[UNROLL, MR]`: wave-per-row like q8row, nibble
unpack in register, same fp32 accumulate. `blk32_forward` uses it for the
head when the pack has the q4 entry; the trunk never does.

| prediction | value | rule |
|---|---|---|
| q4row ffn-shape cold-cache stream | 100 MB-equiv in 36-44 us (q8row 62.6) | <= 48 us AND fp64 parity on dequantized values |
| draft path per window (k=1) | 2.4 -> 1.5-1.7 ms | recorded |
| real prompts k=2 median | 100.7 -> 108-115 | land >= 106 AND 100/100 identity AND acceptance >= 66% (q8 draft: 69%) |
| race k=4 | 146 -> 155-165 | recorded |
| falsifier | acceptance < 64%: q4 draft too lossy; q6_K/q5_0 pack next, not more q4 tuning | |

## Result

Not yet run.

## M4 landed on the 20-prompt set — 2026-09-05

The M4 result (`a81242a`) was measured on one race prompt. This is its P4
verdict: `bench/mtp-prompts.sh` over all 20 prompts of `bench/mtp-prompts/`,
k=2, arm-per-flag, on a pack rebuilt for the purpose.

**The q4 draft head was inert on the live pack.** `.work/engine-pack-q8` was
built without `--q4-draft`, so `have_q4_draft` was false and `BARO_DRAFT_Q4=1`
silently decayed to the q8 head — a P1 inert parameter, visible only because
the engine prints `BARO_DRAFT_Q4:` and that line is now readable (see below).
Rebuilt as `.work/engine-pack-q8d` with
`tools/engine-pack.py MODEL.gguf .work/engine-pack-q8d --q8 --q4-draft`:
443 entries, 10.52 GiB, trailing `output.weight.q4draft q4` present.

| arm | median tok/s_gen (20 prompts) | drafted | accepted | acceptance |
|---|---|---|---|---|
| no spec (A) | 67.32 / 67.61 | — | — | — |
| k=2, `BARO_DRAFT_Q4=0` | **102.53** | 1054 | 737 | 0.6992 |
| k=2, `BARO_DRAFT_Q4=1` | **104.18** | 1061 | 734 | 0.6918 |

**+1.61%**, faster on 19 of 20 prompts, spread across prompts 88.4-127.4.
Greedy identity PASS on all 40 speculative runs (arm B `GENERATED` equals
arm A's, per prompt).

Acceptance moved as the protocol warned it might: 0.6992 -> 0.6918, i.e. the
q4 draft head is a slightly worse drafter. It still wins because the head is
cheaper by more than the lost acceptance costs. The single-prompt figure that
motivated the item (+2.1%) overstated it; the 20-prompt verdict is +1.61%.

Left as an env flag rather than made default: the default pack
(`.work/engine-pack-q8`) has no q4 entry, so flipping the default would only
change behaviour for packs built with `--q4-draft`, and that choice belongs
with the pack, not the binary.

### Instrument bug found and fixed while taking these receipts

`gpu-wait run` printed nothing before the job's first several lines: the
waiting-room daemon's `op_logs` follow mode initialised its file offsets to the
current size, so output written before the client attached was skipped. The
engine's `BARO_DRAFT_Q4:` and `loading pack:` lines - the read-back for the
parameter under test - were being dropped while the rest of the run looked
complete. For a fast job it lost everything.

Fixed in gpuwaitingroom `6e8ef0c` (follow replays from 0; `gpu-wait logs
--follow` keeps tail-then-follow via `since_end`; regression test verified
failing before the fix), daemon reinstalled and restarted. **Receipts taken
through `gpu-wait run` before 2026-09-05 09:10 may be missing their first
lines** - not wrong, but incomplete. Where a run's parameter read-back matters,
redirect the engine's stdout to a file inside the job rather than trusting the
stream.
