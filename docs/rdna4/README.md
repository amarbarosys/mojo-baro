# RDNA4 (gfx1200, gfx1201)

Status: builds and the accumulator rows are ported; no gfx12 card has run any of it. Every numeric and perf claim for a
real RDNA4 card is UNVERIFIED.

- Probe receipt: `probe-2026-11-12.txt` is the output of `tools/rdna4-probe/rdna4-probe.sh`: which AMD intrinsics and
  WMMA widths lower for gfx1100, gfx1200 and gfx1201. `bench/preflight.sh` diffs a fresh run against it.
- Emulation build: `-D BARO_EMU_GFX12_ROWS=true` on a gfx1100 card makes `arch.mma` hand the accumulator back in gfx12 row
  order, so `acc_row` (`kernels/arch.mojo`) decodes rows as `i + 8*half` instead of `2*i + half`. Reductions over
  accumulator elements go through `gfx11_rows` so their add order stays the gfx11 one.
- Gate script: `bench/rdna4-epi-gate.sh` with the steps `build`, `isa`, `run`, `mutants` and `engine` (usage in its header; `mutants`
  plants one gfx11 row formula per kernel family and the test must fail). Receipts copied to `exchange/receipts/rdna4-epi/`. Result of the last run: `docs/lanes/2026-11-12-rdna4-epilogue-report.md`
  (gfx11 ISA byte-equal 8/8, parity 7/7 default and emulation, 13/13 mutants caught, engine agreement 20/20 on 64 forced ids per prompt).
- Real card: `bench/report.sh --rdna4` writes the self-test report an owner can post.
- Method and traps: skills `rdna4-port` and `rdna4-isa`; rules in `docs/rules.md` (acc_row base argument, gfx11_rows).
