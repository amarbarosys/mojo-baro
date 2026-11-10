# Decode attention wired into the engine (frozen 2026-09-11, before any engine timed run)

Binds: `bench/PROTOCOL-RULES.md` P1-P6, repo `CLAUDE.md`. Follows `bench/dattn-protocol.md`
(kernel round closed BETWEEN: 1.43 / 1.70 / 1.48x vs llama.cpp standalone).

## What changes

- `kernels/mega.mojo` `attn_phases`: the split branch (`pos + 1 > att_split`) runs
  `dattn_split_body` on (row, KV head, split) work items, one grid barrier, then
  `dattn_combine_body`; split count `dattn_nsplit(T, M, MEGA_G)`. The non-split branch keeps
  `attn_head_span` with unchanged arithmetic.
- `serve/window.mojo` launch path: the same split branch through `amar_dattn_split` /
  `amar_dattn_combine` with the same split count, so mega == launch holds on the split path
  by construction (shared device functions).
- Config: KV f32 (the engine's), `DATT_NLD` = 4 from the f32 standalone sweep
  (receipt below: best or tied at both lengths; 8 spills 16 B and is slower), rotation off.
- The split threshold is unchanged (`BARO_ATT_SPLIT_T`, default TMAX 1088): decode with
  T <= 1088 never enters the new code.

## Arms

| arm | binary | built from |
|---|---|---|
| base | `.work/engine-base`, sha256 `18bd27c3efebcf1f` | main's engine sources, built by mega-gate at `7721fe0` |
| new | `.work/engine`, rebuilt inside the stint | the commit carrying this file |

Instrument: the engine's own `tok/s_gen` (64 generated tokens), `GENERATED`, `mega fail word`,
`TMAX`, `att split` lines, all read back per run. Runner: `bench/dattn-wire-run.sh`, one
gpu-wait job, fail-closed.

## Gates (all before any timed run counts)

| gate | check | status at freeze |
|---|---|---|
| G1 numerics | `tools/dattn-ref.py`: every shape incl. the engine's f32, KV {1, 127, 128, 129, 4096}, split ns {1, 8, 64}, NLD {2, 4, 8}, ROT, query rows M {1, 3} | **PASS** 920/920, worst 1.57e-5 (bound 2e-3), multi-row worst 1.56e-5; `.work/dattn/ref-rows.log` |
| G2 fingerprint | CPU-side, before any GPU run (memory `megakernel-lottery-fingerprint`): q4 token megakernel top-4 loop `dual`, spills, LDS | **PASS**: new 124 / 79 / 79 / 59, 0 spills, 256 VGPR, LDS 36.8 KB; base 114 / 78 / 78 / 53, 0 spills, 240 VGPR, 27.9 KB |
| G3 engine gate | `tools/mega-gate.sh` on new: default path bit-identical to the reference tokens on every pack | in the stint |
| G4 split identity | `BARO_ATT_SPLIT=1`, mega vs launch `GENERATED` equal at p0512 and p8192 | in the stint |

## Frozen predictions

- **W1 short context (P4).** 20-prompt `bench/ab-prompts.sh` base vs new: identity 20/20 (the
  new code is not entered); median ratio 0.98-1.02. The megakernel source changed (13.3k to
  15.9k instructions) so the allocator re-rolled; G2 says the dot loops stayed in the fast
  class. Falsifier: ratio < 0.98.
- **W2 agreement at long context (recorded, not gated).** Against an exact-attention arm
  (base with the split threshold above T), new's `GENERATED` common prefix is at least as long
  as base's, at p8192 and p32768.
- **W3 long-context decode**, median of 3 alternating runs, new / base `tok/s_gen`.
  Model: token time = base time - attention_old + attention_new, attention_old from the chat
  lane's receipts (8k 117.5 tok/s = 8.51 ms, 32k 85.0 = 11.76 ms, over a 7.5 ms short-context
  token: 1.0 ms and 4.3 ms of attention over 8 layers), attention_new = 8 x (f32 KV bytes per
  layer / the sweep's standalone rate).
  f32 sweep receipt (`.work/dattn/f32sweep.log`, S0 = 256/16/4 f32, 2000 iterations under
  rocprofv3, rotation >= 402 MB): NLD 4 at 24 splits (the megakernel's count for one row)
  81.86 us at KV 8192 (820 GB/s) and 301.40 us at KV 32768 (891 GB/s); NLD 2 within 1 %,
  NLD 8 16 B scratch and up to 11 % slower.
  - 8k: attention 8 x 81.9 us = 0.65 ms against 1.0 ms: predicted ratio **1.04**, band 1.02-1.07.
  - 32k: attention 8 x 301.4 us = 2.41 ms against 4.3 ms: predicted ratio **1.19**, band 1.10-1.25.
  The base arm is re-measured in the stint; the band is on the ratio, not on the chat-lane numbers.

**Land:** G1-G4 pass, W1 holds, W3 32k ratio >= 1.08. **Close negative:** W1 falsified, or
32k ratio < 1.03. **Between:** report, no land.

## Stop rules

- Any gate fails: fix or report; never re-freeze the reference.
- `mega fail word` non-zero on a run: that run is void (NOT-RESIDENT).
- Spread > 10 % on a long-context arm: that row is void.
- Engine sha in the summary differs from the arms table: the stint is void.

## Result (2026-09-11, stint `.work/dattn-wire/w1`, commit `b7c5041`)

**Verdict: LAND (on `lane-dattn`; merging is the maintainer's call).** G1-G4 pass, W3 32k ratio 1.189
>= 1.08, W2 held. W1's band **missed high** (1.066 against 0.98-1.02); its falsifier (< 0.98)
did not fire, and the land rule is read as "W1 not falsified". That reading is stated here so
it can be overruled.

| gate | result | read from |
|---|---|---|
| G1 numerics | PASS 920/920, worst 1.57e-5 | `.work/dattn/ref-rows.log` |
| G2 fingerprint | PASS: q4 token dual 124/79/79/59, 0 spills, LDS 36.8 KB | `tools/isa-receipt.py`, `isa-loops` |
| G3 mega-gate on new (sha `606e156d60333e5d`) | ALL PASS: q8 / q8d / q4 mega == launch at spec 0 and 1, q8 and q4 vs ref-tokens-64 | `mega-gate/SUMMARY.txt` |
| G4 split identity (`BARO_ATT_SPLIT=1`) | mega == launch 64/64 at p0512 and p8192, fail word 0 | `SUMMARY.txt` |

| prediction | frozen | observed | |
|---|---|---|---|
| W1 20-prompt ratio | 0.98-1.02, identity 20/20 | **1.066** (base 127.96, new 136.41 tok/s_gen, spread 0.5 / 1.4 %), identity 20/20 | MISSED high |
| W2 agreement vs exact | new prefix >= base prefix | 64/64 for base, new and exact at p8192 and p32768 | HELD |
| W3 8k ratio | 1.04 (1.02-1.07) | **1.104** (112.70 to 124.48, spread 0.3 / 0.2 %) | MISSED high |
| W3 32k ratio | 1.19 (1.10-1.25) | **1.189** (85.22 to 101.36, spread 0.3 / 0.3 %) | HELD |

Where the gain comes from. The new code is not entered below T = 1088, so W1's +6.6 % is the
megakernel's re-rolled schedule (hot dot loop dual 114 to 124, s_delay_alu 704 to 644), not
attention. Taking the 20-prompt token time (7.815 base, 7.331 ms new) as the non-attention
part: attention at 8k is 1.058 ms base and 0.702 ms new (predicted 0.65), at 32k 3.919 ms
and 2.535 ms (predicted 2.41). The attention kernel alone is worth **1.04x at 8k and 1.13x at
32k**; the schedule re-roll multiplies every token by about 1.066 on top. The 32k land bar
(1.08) is met by the attention share alone.

Receipts: arm file `ab20/arm.txt` engA = base `18bd27c3efebcf1f`, engB = new `606e156d60333e5d`,
same pack and env; every run prints `att split: 1088` (exact arm 1000000), `TMAX: 33792`,
`mega fail word: 0`; exact attention at 32k 45.43 tok/s (the chat lane's pre-split 45.7).

Caveat on W1: the +6.6 % is a property of this build's register allocation (memory
`megakernel-lottery-fingerprint`). Any later megakernel edit can re-roll it either way; check
the fingerprint before crediting or blaming a short-context change.
