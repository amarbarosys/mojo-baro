# Verify8: keep transposed weights; 800 GB/s is still unproved

2026-11-16. Lane `lane-verify8`, starting commit `39a3f2e`. Report to `a lane pane` from `a lane pane`.

The best tested change is transposing the q4 block records and scales for the existing 8-wave,
D=4 WMMA kernel. It reaches **440.7 GB/s gate and 421.3 GB/s down**, versus **421.6 / 397.7**
for the control. All 8-row sampled fp64 checks pass, and transposition is byte-identical to the
control on every output of tensor 0. Keep this arm. Reducing split-K to 4, 2, or 1 waves does
not improve it. No tested change reaches the requested 1-row bandwidth.

## Diagnosis and its limit

The 8-row kernel is not reading the weights eight times. Its source shares one q4 fragment
across the WMMA activation rows, and the m=1..8 timing is flat. The cost is the fixed fragment
load, conversion, issue/dependency, and cross-wave reduction work. This is **not a demonstrated
2x DRAM-traffic problem**. The reported GB/s is useful packed Q+S bytes divided by wall time,
not a measurement of physical DRAM traffic.

Evidence:

- `serve/window.mojo::gemm_w` sends eligible q4 `m > 4` to `amar_fewrow_wmma`. Turbo uses
  contiguous K-block lanes and LDS activation staging. Fewrow uses output-row lanes, loads
  its A fragments directly, issues four WMMAs per q4 block, then reduces eight K partials.
- At K=5120 the warm fewrow trace has **120 VMEM loads and 2,495 VALU issues per wave**;
  the separately captured warm Turbo m=1 trace has **12 and 926**. The waves process different
  amounts of useful work, so these counts are not an instruction-normalized speed ratio.
- The matched warm gate control assigns **64.5%** of charged cycles to VALU, **9.5%** to
  vmcnt waits, **7.4%** to VMEM loads, and **16.0%** to the final barrier. Cause accounting
  names VALU dependencies at 41%, issue at 12%, and issue wait at 11%. This supports an
  issue/dependency-heavy kernel organization, not a pure bandwidth ceiling.
- The transposed arm still assigns **59.2%** to VALU, **25.8%** to vmcnt waits, and **10.2%**
  to the barrier. It has 138 VGPR and no spills, like the control. Transposition removes a
  small access-layout cost but leaves most of the gap.
- Reducing the wave count fails the short-loop/tail hypothesis as a sufficient fix. Removing
  the two offset WMMAs with bf16 subtraction also loses. Bit-built biased bf16 fragments
  increase static ISA from 790 to 870 instructions. A refill-after-use pipeline grows
  registers from 138 to 251 and loses throughput, consistent with register pressure.

**Root-cause scope:** the receipts establish a fixed WMMA fragment-processing and scheduling
cost, with a small weight-layout component. They do not isolate one instruction or prove a
single cause for the entire 2x gap. The exact remaining cause is **UNRESOLVED**. Do not claim
that the original live-engine 36.5% vmcnt / 21.4% barrier shares prove a DRAM bottleneck or
that removing either share would give a proportional speedup. The live engine and warm
standalone benchmark have different execution conditions; no cause for that difference was tested.

## Measurements

RX 7900 XTX, gfx1100. Queue reported `AI-2500-uv125`, sclk <=2500 MHz, mclk <=1250 MHz,
290 W. No machine settings changed. Mojo 1.2.0. Real nerkyor 27B q4 blk.0..7 FFN weights.
Each call counts 50,135,040 Q+S bytes. Each rotation covers 401,080,320 bytes. Eight launches
per synchronization, 11 timed batches per arm, three repeated arm groups in one process.
One second of Turbo MR3 work precedes timings. Values below are medians of the three medians.
These are host-wall timings, not device-event timings; some batch maxima contain large outliers.

| Arm, m=8 unless stated | Gate us | Gate GB/s | Down us | Down GB/s |
|---|---:|---:|---:|---:|
| Turbo m=1, U=2 | 62.287 | 804.9 | 64.522 | 777.0 |
| Original storage, 8 waves, D=4 | 118.910 | 421.6 | 126.068 | 397.7 |
| Transposed Q/S, 8 waves | **113.763** | **440.7** | **119.004** | **421.3** |
| Original storage, 4 waves | 119.400 | 419.9 | 126.063 | 397.7 |
| Original storage, 2 waves | 125.542 | 399.3 | 147.151 | 340.7 |
| Original storage, 1 wave | 144.316 | 347.4 | 210.718 | 237.9 |
| Transposed Q/S, 4 waves | 114.922 | 436.3 | 123.997 | 404.3 |
| Transposed Q/S, 2 waves | 119.808 | 418.5 | 144.747 | 346.4 |
| Transposed Q/S, 1 wave | 140.882 | 355.9 | 216.787 | 231.3 |
| Center nibbles, remove offset WMMAs | 168.724 | 297.1 | 178.136 | 281.4 |
| Transpose + bit-built biased nibbles | 118.463 | 423.2 | 126.150 | 397.4 |
| Transpose + refill-after-use pipeline | 141.547 | 354.2 | 148.189 | 338.3 |

Predictions were recorded before each new probe in `.work/verify8-astra/prereg.md`.
The 650-800 GB/s transpose prediction, >=15% smaller-wave gain, 550-700 bit-construction
prediction, and 550-650 pipeline prediction all failed. Centering also missed 470-580 GB/s.
The first exploratory run had cold-clock-looking timing drift; its first groups are not used
for the final table. Actual clocks were not sampled, so that drift is not attributed conclusively.

## One recommended change

Keep **transposed Q/S, 8 waves, D=4**, implemented by benchmark ABL=22 / kernel MODE=2.
Predict **430-450 GB/s gate and 415-435 GB/s down** on this workload, based on the repeated
measurements. Do not budget 800 GB/s or a 2x verify-window gain. The measured useful-bandwidth
gains are 4.5% and 5.9%. If these gains applied to all of the reported 75% GEMM share, a 46 ms
window would become roughly 44.2 ms. That is an estimate, not an engine measurement.

The branch contains a working prototype; production routing retains MODE=0. The layout change
for each tensor is:

```text
NB = K / 32
Q_old[n, b, j]  -> Q_new[b, n, j]       j = 0..15
S_old[n, b]     -> S_new[b, n]
Q_new view: row_major[NB, N*16], then vectorize the last dimension by 16
S_new view: row_major[NB, N]
Kernel loads: Qb[blk, row], S[blk, row]
Keep the same activation fragments, four WMMAs, scales, and eight-wave reduction.
```

This sketch describes the measured layout, not a proposed on-disk schema change. The benchmark
creates a scratch transpose. Production adoption needs a loader/pack policy: an additional
full q4 copy would be an unacceptable default for a 27B model on this 24 GiB card. Prefer one
canonical layout with compatible consumers, or prove a bounded alternative before routing it.
No pack schema or engine default was changed here.

**Numerics:** the recommended layout preserves operation order and passed bit equality on
139,264 gate outputs and 40,960 down outputs, all eight activation rows on tensor 0. Its fp64
sample check covers every 97th output channel across all eight rows: 180 gate channels and
53 down channels, max relative errors 3.1893e-5 and 3.2352e-4, below the existing 2e-3 gate.
Timing rotates eight tensors; exhaustive parity currently covers only tensor 0. Centering,
bit construction, and changing split-K can lose bit-exactness. No teacher-forced agreement
run or live verify-window run was performed. The requested 0.005 agreement gate remains
required before adopting an arithmetic-changing arm.

## Reproduce and inspect

```bash
./.venv/bin/mojo build --target-accelerator gfx1100 bench/bench_turbo_rows.mojo \
  -I . -I kernels -I serve -D BARO_H=5120 -D BARO_FFN=17408 -D VERIFY8=1 \
  -o .work/verify8-astra/bench-final
gpu-wait run --timeout 120 --priority 90 -- env \
  BARO_PACK=$HOME/Projects/mojo/mojo-baro-lanes/nerkyor-vs-forge/.work/nerkyor/pack-q4 \
  .work/verify8-astra/bench-final
```

`PASS bench_turbo_rows` exercises every listed arm and both shapes, with fp64 checks before
timing and the exhaustive tensor-0 transpose comparison. Final run: 3.99 s, peak host RSS
1,573,272 KB. Final build: 5.82 s, peak RSS 1,087,144 KB. The experiment adds a 401,080,320-byte
host transpose and device transpose per active shape, unchanged payload size. These figures
include all arms and repacking; they are not a production-memory improvement claim.

Receipts under `.work/verify8-astra/`: `run-final.log`, `summary.txt`, `final.sha256`,
`build-final.log`, `isa-original/`, `isa-pipe/`, `isa-pipe.txt`, and `prereg.md`.
`ww-matched/{classes,top,trans-classes,trans-top}.txt` contains matched warm gate captures,
dispatches 10807 and 11031, 46 and 44 traced waves, zero open waves and zero mismatches.
`ww-final/classes.txt` is the separate warm Turbo capture. A pipeline trace was misidentified
as the runtime fill kernel with zero decoded cycles; it is excluded from causal claims.

Code navigation used the brief's main-checkout graph, generation 2026-11-16T09:00:00Z.
Coverage reported no recorded issues in fewrow, turbo_rows, or window, and a partial bench
parse at line 91; direct source reads covered that gap. The initial fewrow source hashes
matched main and this lane. The experimental generic epilogue supports fewer than eight waves;
the original baseline and experimental baseline have the same main-loop disassembly, but their
epilogue instructions differ. The original supplied timings independently support the baseline range.

Next action: driver review of the transposed prototype and its memory/layout cost. Treat the
near-800 GB/s target and full live-engine causal explanation as open work, not a completed fix.

The repeated register-preflight miss now has an executable CPU gate:
`bench/verify8-resource-gate.sh BINARY KERNEL_SUBSTRING MAX_VGPR`. For this build, target
`4414ee860887c6fd` with a 160-VGPR budget before queueing a changed transpose arm. The gate
checks one matching kernel, register budget, and spills. It passed the 138-VGPR transpose
and rejected the known 251-VGPR pipeline; receipt `resource-gate.txt`. The budget is an
experiment constraint, not a hardware occupancy guarantee.
