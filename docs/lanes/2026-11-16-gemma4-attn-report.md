# Report: gemma4 decode attention kernel (lane-gemma4-attn, 2026-11-16)

Skills loaded before the first edit: `gpu-kernels`, `mojo-gpu-fundamentals`, `mojo-syntax`, `mojo-lean`,
`mojo-nightly-lane-builder`, `inference-kernels`, `rdna3-isa`, `kernel-parity`, `bench-gates`, `wavewait`,
`memory-optimization`. Also read: `gpu-kernels/rules.md`, `bench-gates/rules.md`, `gpu-lane-budget`,
`llm-benchmark-method`, `reverse-arm-gates`, `determinism-triage`, `persistent-kernel-gfx11`.

## Result

`amar_attn_decode_gemma4` passes its gate (400 of 400 arms, worst error 5.8e-6 against the 1e-4 bar, 110 of 110
mutants caught) and is faster than the existing per-head kernels on every gemma4 shape. Cold numbers, final job,
microseconds per layer call including the host enqueue:

| gemma4 layer | T | new kernel | launches | best existing arm | ratio |
|---|---|---|---|---|---|
| S: HD 256, window 512 | 512 | 13.6 | 1 | 21.7 (split, 2 launches) | 1.6x |
| S: HD 256, window 512 | 16384 | 13.5 | 1 | 18.5 (split, 2 launches) | 1.4x |
| F: HD 512, full | 512 | 17.9 (spread 14.0 to 28.7) | 2 | 27.6 | 1.5x |
| F: HD 512, full | 4096 | 30.5 | 2 | 100.0 | 3.3x |
| F: HD 512, full | 16384 | 91.7 | 2 | 799.3 | 8.7x |

At T 16384 the F layer reads its 67 MB of K and V at 733 GB/s. Where the split pays: full attention from
T about 192 up (T 128: one launch 9.9 us, split 11.0; T 256: one launch 13.6, split 11.8); the best split count is 8
from T 512 to T 16384. The windowed layer never needs a split.

These are kernel numbers on random data. They do not say what the engine gains: that needs the real pack and an
engine A/B, which is the main session's step.

## Predictions, frozen before the first timed run (commit of this file)

Instrument: `test_gemma4_attn bench`, microseconds per launch from the host clock, N launches per sync, 11 blocks,
median; cold = KV rotated over 268 MB of disjoint pages (page table + slot), warm-up 0.4 to 1.5 s before each arm.
Reference arm: the existing per-head kernels at the same shape (`amar_attn_decode_swa_gated`, and
`amar_attn_split_part` + `amar_attn_split_merge`). Bytes floor at 700 GB/s: K + V rows read once.

| shape | bytes | floor | new kernel, cold | reference, cold |
|---|---|---|---|---|
| S: HD 256, window 512, T >= 512 | 1.05 MB | 1.5 us | 6 to 12 us, one launch (nsplit 1, 8 blocks) | 10 to 25 us |
| F: HD 512, full, T 512 | 2.1 MB | 3 us | 8 to 16 us | 12 to 30 us |
| F: HD 512, full, T 4096 | 16.8 MB | 24 us | 50 to 100 us at the best nsplit | 150 to 600 us |
| F: HD 512, full, T 16384 | 67 MB | 96 us | 150 to 300 us at the best nsplit | 400 to 1500 us (split), > 3 ms (one launch) |

Split threshold (full attention): nsplit > 1 starts to pay between T 1024 and T 2048.
Falsifier: the new kernel at its best nsplit is slower than the best reference arm on any F row with T >= 4096.
No adoption bar (repo rule 2026-10-09): a miss means find the limiter, not drop.

## Predictions against the numbers

| prediction | measured | verdict |
|---|---|---|
| S new kernel 6 to 12 us, one launch | 22.2 first build, 12.0 to 13.7 after the block change | MISS, 1.5 us over the band |
| S reference 10 to 25 us | 18.5 to 21.7 | held |
| F T 512 new 8 to 16 us | 12.7 (ladder job), 17.9 (final job, wide spread) | edge |
| F T 4096 new 50 to 100 us | 30.5 | MISS, faster than the band |
| F T 16384 new 150 to 300 us | 91.7 | MISS, faster than the band (my 700 GB/s floor was low) |
| F T 16384 reference split 400 to 1500 us | 799.3 | held |
| F T 16384 reference one launch > 3 ms | 1653 us | MISS |
| split pays from T 1024 to 2048 | from T about 192 | MISS |
| falsifier: new slower than reference on F, T >= 4096 | 3.3x and 8.7x faster | not triggered |

Five of eight bands missed. The one that cost work was S: the first build ran 8 serial cold load round trips per
wave. A kernel trace (rocprofv3 `--kernel-trace --stats`) read 18.1 us device time against a 2.9 us host enqueue,
so the kernel was the limit, not the launch. Blocks of 1024 threads (32 waves per head, 2 steps per wave) and a
one-pass combine took it from 22.2 to 12.0 us in the same shape.

## What landed

| commit | content |
|---|---|
| `4010d4d` | `kernels/gemma4_attn.mojo`, `kernels/test_gemma4_attn.mojo`, first green gate |
| `7495229` | this file with the frozen predictions, the main-session review |
| `465918a` | block size parameter, one-pass combine, `g4_nsplit`, `bench/gemma4-attn.sh`, `run-tests.sh` line |
| this commit | results, receipts under `exchange/receipts/gemma4-attn/`, `docs/rules.md` line |

`dattn.mojo` and `attn.mojo` are not touched: the megakernel shares their device functions. The kernel imports
`dkv_tab_off`, `NEG`, `readlane` and `bexp.exp`; the body is the dattn layout with the lane width generalised to
HD / 32, so one body serves HD 256 and 512. `amar_dattn_split` cannot serve HD 512 (its row needs 64 lanes of 8).

## Contract for the wiring

- Instantiate `amar_attn_decode_gemma4[HD_, WINDOW, NAT, g4_gb(HD_, WINDOW), g4_nld(HD_, WINDOW), g4_bt(HD_, WINDOW),
  QLayout, KLayout, OLayout, PLayout]`. Launch with `grid_dim=(8 // GB, ns)`, `block_dim=g4_bt(...)`,
  `ns = g4_nsplit(WINDOW, t_len)`. When `ns > 1`, launch `amar_attn_merge_gemma4[HD_, PLayout, OLayout]` next with
  `grid_dim=8`, `block_dim=HD_ // 8`.
- Picks: S = GB 1, NLD 8, BT 1024, always one launch. F = GB 2, NLD 4, BT 256, split 1 below T 192, then
  `min(8, T // 64)`.
- Two KV pools, one per head dim (review MUST item): S pool HD 256, F pool HD 512, each with its own `NAT`; `att_i`
  is the slot inside the pool of that kind. A shared layer passes its source's slot (layer 13 in the S pool, 14 in
  the F pool). Pool layout is `dkv_tab_off[HD_, 1, NAT]` with the page table `tab`, f32 only.
- `t_len` counts the new token. The kernel attends to positions `max(0, t_len - WINDOW) .. t_len - 1`, which is
  llama.cpp's `pos_q - pos_k < n_swa`.
- Q and O are `[8, HD_]` f32, read and written through `.ptr`. `Pg` needs `8 * ns * (HD_ + 8)` floats (ns <= 8
  under the rule, 64 in the test).
- Scale is 1.0 with no argument; no softcap. One token per launch (no row axis).
- The kernel loads only rows inside `[t_lo, t_len)`: rows outside may hold anything (the test fills them with NaN).

## Test output

`bench/gemma4-attn.sh test` after the Codex fixes, receipt `exchange/receipts/gemma4-attn/test-final.log` (the bench
tables below come from the build before those fixes, sha `45ee06c7dfbd2a03`; the fixes add a comptime assert and
docstring lines to the kernel, no executable statement; the bench was not rerun, the ISA receipt was):

```
pass lines 400, mutants caught 110
worst pass: hd 512 win 512 T 16384 qs 4 ns 1 rel 5.805605202698661e-06 PASS
weakest t_len / window mutant: hd 256 win 0 T 4096 qs 1 mutant t_len-1 rel 1.2544775370050314 CAUGHT
scale mutants: 48 weakest: hd 256 win 512 T 2 qs 4 mutant scale 1/sqrt(HD) rel 0.7107938154080797 CAUGHT
PASS test_gemma4_attn
```

400 = 2 HD x 2 windows x 7 T x 2 q scales x 7 split counts (1, 2, 3, 7, 16, 64 and the shipped `g4_nsplit`) plus 8
arms with the S and F pools live together. Bar, set before the first run: max over heads of
max|o - ref| / max|ref| <= 1e-4 against an f64 oracle in the same binary. Mutants are real kernels: `WINDOW - 1`
and `WINDOW + 1` builds, a launch with `t_len - 1`, and (Codex finding 1) a launch whose q is multiplied by
1 / sqrt(HD), which is what a kernel with the usual attention scale would compute. The reverse arms (reversed page table, slot 1 of 2, NaN
outside the attended rows) make a kernel that ignores the table, the slot or a window edge read NaN.
The `run-tests.sh` line ran green through the runner's own `t()` helper; the full `run-tests.sh` did not run.

## Bench detail

Final job `exchange/receipts/gemma4-attn/bench-final.log` (11 blocks of up to 200 launches, median, KV rotated
over 268 MB; load average 2.4 at the start, other lanes were building). Cold / hot microseconds:

| HD, window, T | rule split | new, rule | new, one launch | reference one launch | reference split |
|---|---|---|---|---|---|
| 256, 512, 512 | 1 | 13.6 / 10.1 | 13.6 / 10.1 | 61.4 / 32.7 | 21.7 / 16.6 |
| 256, 512, 4096 | 1 | 13.5 / 10.1 | 13.5 / 10.1 | 56.6 / 32.8 | 20.9 / 16.3 |
| 256, 512, 16384 | 1 | 13.5 / 10.1 | 13.5 / 10.1 | 56.0 / 32.3 | 18.5 / 15.9 |
| 512, 0, 512 | 8 | 17.9 / 11.9 | 23.2 / 18.1 | 55.3 / 40.2 | 27.6 / 21.8 |
| 512, 0, 4096 | 8 | 30.5 / 26.8 | 153.9 / 125.5 | 370.8 / 329.6 | 100.0 / 81.2 |
| 512, 0, 16384 | 8 | 91.7 / 81.1 | 588.2 / 467.5 | 1653.1 / 1469.5 | 799.3 / 715.7 |
| 256, 0, 16384 (not a gemma4 shape) | 8 | 81.2 / 69.3 | 612.9 / 519.3 | 1476.2 / 1365.8 | 389.0 / 323.5 |
| 512, 512, 16384 (not a gemma4 shape) | 1 | 25.3 / 17.1 | 25.3 / 17.1 | 69.6 / 40.8 | 27.8 / 21.8 |

Split ladder, F shape, cold us by split count (`ladder.log`):

| T | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| 64 | 7.4 | 9.9 | 9.9 | 10.9 | 13.5 | |
| 128 | 9.9 | 11.5 | 11.0 | 11.8 | 14.0 | 18.0 |
| 256 | 13.6 | 12.9 | 11.8 | 11.9 | 14.0 | 17.9 |
| 512 | 22.6 | 17.4 | 14.1 | 12.7 | 16.0 | 19.1 |
| 1024 | 40.4 | 24.3 | 17.2 | 15.4 | 17.3 | 20.8 |
| 2048 | 83.4 | 40.9 | 26.3 | 21.5 | 25.0 | 29.6 |
| 4096 | 152.0 | 79.0 | 42.8 | 31.0 | 35.7 | 40.2 |
| 8192 | 294.3 | 142.1 | 80.2 | 50.4 | 56.4 | 59.4 |
| 16384 | 588.5 | 268.6 | 145.3 | 91.6 | 100.7 | 102.8 |

Knob sweeps (`sweep-F.log`, `sweep-S.log`), F shape at T 16384 / 4096, best split, cold us: GB 2 NLD 4 92.1 / 31.1
(kept), GB 2 NLD 8 92.3 / 31.4, GB 2 NLD 2 95.8 / 33.3, GB 4 NLD 4 97.7 / 35.7, GB 1 NLD 4 146.0 / 46.0, BT 512
94.7 / 33.7; GB 4 NLD 4 BT 512 spills 19 VGPRs and the script refuses it. S shape, one launch: NLD 8 BT 1024
12.0 to 13.7 across three jobs, NLD 4 BT 1024 12.0, NLD 4 BT 512 13.0, NLD 8 BT 512 21.2. The top S picks tie
inside the job-to-job spread of about 1.5 us; I kept NLD 8 because fewer serial round trips is the mechanism.

### Emitted code of the shipped instances (Codex finding 4)

Read from the device per dispatch (`rocprofv3 --kernel-trace`, columns VGPR_Count, Scratch_Size, LDS_Block_Size)
and from the code object (`tools/isa-receipt.py`, `isa-final.txt`). The trace names the instance each shape
launched, so the mapping is not inferred:

| shipped instance | kernel hash | VGPR (ISA / device) | spills | scratch | LDS | block | grid | device time |
|---|---|---|---|---|---|---|---|---|
| S: HD 256, window 512, GB 1, NLD 8, BT 1024 | `ad74c58e` | 114 / 120 | 0 | 0 B | 33280 B | 1024 | 8 blocks | 13.6 us avg, 8.1 min (T 512) |
| F: HD 512, full, GB 2, NLD 4, BT 256 | `7d19187d` | 223 / 224 | 0 | 0 B | 33280 B | 256 | 4 x 8 blocks | 22.8 us avg, 18.4 min (T 4096, 8 splits) |
| merge, HD 512 | `c9227b8a` | 22 / 24 | 0 | 0 B | 0 B | 64 | 8 blocks | 3.6 us avg, 3.4 min |

Device averages mix the cold and the hot arm of one bench run. Occupancy that follows: F at 224 VGPR holds 6
waves per SIMD (1536 / 224), and 33 KB of LDS allows at most 3 blocks per 128 KB workgroup processor; the best
split of 8 launches 32 blocks, so neither limit binds at the measured optimum. I make no occupancy claim past that.

## Review items

| item | state |
|---|---|
| two KV pools by HD | in the kernel docstring, in this contract, test arm `two_pools` (S and F live, launches interleaved before one sync) |
| VGPR at HD 512 | read from the receipt: 223, 0 spills at NLD 4; NLD 8 at GB 2 is 244 to 246 and no faster |
| 8 serial combine passes | one LDS pass after one barrier when the slab fits 32 KB (all shipped shapes) |
| merge block size | merge threads past HD_ / 8 return; block HD_ // 8 is in the docstring |
| large scores through bexp exp | q scale 4: random scores to std 30, planted scores to 341, worst error 5.8e-6 |
| empty splits | split 64 at T 1 and T 2 in the gate |

Codex audit (`2026-11-16-gemma4-attn-codex-audit.md`, 0 MUST, 1 SHOULD, 3 NOTE):

| finding | state |
|---|---|
| 1 SHOULD: no negative control for the scale | scale mutant in the gate, 48 of 48 caught, weakest 0.71 |
| 2 NOTE: implicit preconditions | kernel docstring: `t_len >= 1`, finite scores well above NEG; not enforced in the kernel |
| 3 NOTE: BT not asserted | `comptime assert BT % WARP_SIZE == 0`; the launch must use BT threads (docstring) |
| 4 NOTE: no emitted-code receipt | table above |

## Open items

- The S layer sits at 13.5 us; a trivial kernel reads about 4 us of device time in the same trace. What is left is
  2 cold round trips, the compute of 8 heads on 8 workgroup processors and the combine. I did not take a wavewait
  capture: with 8 blocks the traced CU may hold no wave. That capture is the next instrument, not another knob.
- The F line at T 512 spread 14.0 to 28.7 in the final job (12.7 in the ladder job). Read it as 13 to 18.
- `g4_nsplit` is fitted on HD 512 full attention. HD 256 full attention wants 16 splits at T 16384 (52.0 against
  81.2 us); no gemma4 layer has that shape.
- KV types: f32 tested. int8 is refused at comptime. bf16 and f16 compile through the cast and have no test.
- T above 16384 is not measured (the bench pool holds one rotation at T 32768, which is a hot number).
- Not run: the full `run-tests.sh`, `bench/preflight.sh`, `lane-merge`. When the kernel gets a launch site in
  `serve/`, add that file to the kernel-census caller list.
- GPU use of this lane: about 39 jobs, about 20 GPU-minutes, none failed (my count from the logs; `gpu-wait stats
  --days 1` reports the whole box: 292 jobs, 16 failed).
- Falsifier for the headline: an engine A/B on the real pack where the F layers at long context do not get
  faster than the split reference path.
