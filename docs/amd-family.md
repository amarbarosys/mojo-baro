# Run mojo-baro on your AMD GPU

Every number in this repo comes from one AMD RX 7900 XTX. Whether the kernels
build, run and stay fast on any other AMD card is unmeasured. Not "probably
fine": nobody has checked. A receipt from your card, even a failing one, is the
most useful contribution this project can get.

You need an AMD GPU, a working ROCm install, and about fifteen minutes. **No
model weights are needed.** The benchmark is a self-contained fp16 GEMM, ours
against AMD's hipBLASLt on the same card.

## Where your card stands

| family | gfx target | example cards | expected today |
|---|---|---|---|
| RDNA3, big die | `gfx1100` | RX 7900 XTX, 7900 XT, 7900 GRE, Radeon PRO W7900, W7800 | builds and runs. Measured on the 7900 XTX only; the XT and GRE have fewer compute units and less Infinity Cache, so the tile thresholds may be wrong for them |
| RDNA3, mid die | `gfx1101` | RX 7800 XT, 7700 XT, Radeon PRO W7700 | same ISA family, should build; unmeasured |
| RDNA3, small die | `gfx1102` | RX 7600, 7600 XT, Radeon PRO W7600 | same ISA family, should build; 8 to 16 GB limits which models fit; unmeasured |
| RDNA3 / 3.5 APUs | `gfx1103`, `gfx1150`, `gfx1151` | Radeon 780M / 760M, 890M, Strix Halo 8060S | same ISA family, shared system memory changes the bandwidth picture entirely; unmeasured |
| RDNA4 | `gfx1200`, `gfx1201` | RX 9060 XT, 9070, 9070 XT | not RDNA3. WMMA changed between generations, so the WMMA kernels may not build. `bench/report.sh` warns and still writes a receipt |
| RDNA2 and older | `gfx1030` and below | RX 6900 XT, 6800, 6700 XT | no WMMA units; the fp16 GEMM kernel will not run. A failing receipt still tells us where the line is |
| CDNA (Instinct) | `gfx90a`, `gfx942`, `gfx950` | MI210, MI250, MI300X, MI355X | wave64 hardware; the kernels assume warp size 32. Out of scope for now |

Find your gfx target with `rocminfo | grep -m1 gfx`.

## Steps

1. **Check ROCm sees the card.**

   ```sh
   rocminfo | grep -m1 gfx
   ```

   If this prints nothing, fix ROCm first; nothing below will work.

2. **Clone and set up the toolchain.** Nothing is installed machine-wide: `uv`
   creates a repo-local `.venv` with the pinned Mojo/MAX toolchain.

   ```sh
   git clone https://github.com/amarbaro/mojo-baro && cd mojo-baro
   uv sync
   ```

3. **Run the preflight.** It checks the GPU, ROCm, hipBLASLt and the
   toolchain in seconds, before any long build.

   ```sh
   ./bench/report.sh --check
   ```

4. **Run the benchmark.** About fifteen minutes. Close anything heavy that
   uses the GPU first (games, video encodes, other ML jobs): the numbers are
   only meaningful on an otherwise idle card.

   ```sh
   ./bench/report.sh           # full run, ten sizes
   ./bench/report.sh --quick   # three sizes, if you are short on time
   ```

   It prints a markdown block and writes
   `results/report-<gfx>-<card>-<commit>.json`.

5. **Open a hardware report.** Use the
   [hardware report issue template](../.github/ISSUE_TEMPLATE/hardware-report.yml),
   paste the markdown block, attach the JSON.

6. **File it even if it failed.** A build error on RDNA4, a card that
   throttles, a hipBLASLt that behaves differently: those are results. The
   script writes a receipt marked `"valid": false` with the reason rather than
   nothing.

## Verify a self-describing model file on your card

Every `*-BARO-<sha>.gguf` we publish carries the kernel sources that produced
its numbers (`baro.kernel.src.*`) plus the receipt they were measured under
(`baro.hw.*`: card, driver, ROCm, power cap, the 20-prompt tok/s median and
the protocol file). One command rebuilds the engine from the file alone,
checks its tokens against the reference, and prints your card next to ours:

```sh
gpu-wait run --vram 20 -- tools/gguf-verify.sh ~/Models/.../Qwythos-...-BARO-<sha>.gguf
```

Exit 0 means the sources in the file are complete and reproduce the tokens
on your card. The tok/s line it prints is a receipt for your hardware report,
not a pass or fail: a different card has a different number, and that
difference is the data we want.

## Going further on your card

Optional, for people who want to dig in:

- **Peak WMMA rate.** `bench/wmma-peak.sh` measures your card's WMMA issue-rate
  roofline. It reads the compute-unit count from the running GPU rather than
  trusting the 96 in the source; pass a count to override it.
- **The full test suite.** `./run-tests.sh` builds the hipBLASLt shim and runs
  the kernel parity tests. It is the check that the kernels are numerically
  correct on your hardware, not just fast.
- **Retuning the tile thresholds.** The fp16 GEMM picks its tile shape from how
  many workgroups the grid would launch (`>= 96` blocks takes 128x128, `>= 64`
  takes 64x128, below that 64x64). Those cut-offs were swept to fill 96 compute
  units. On a card with fewer CUs they are the first thing to revisit. Follow the
  preregistration flow in [CONTRIBUTING.md](../CONTRIBUTING.md): commit the
  prediction before the run, then the receipt.

## What a good receipt protects against

Two ways to get a number that looks fine and is not, both of which have hit
this repo:

- **Short warm-up.** The same binary read 66k GFLOP/s after one second of
  warm-up and 91k after ten, only because the clocks had not settled. The
  script warms for 10 s and marks the receipt invalid below that.
- **Cache residency.** A benchmark whose buffers fit in the Infinity Cache
  measures the cache, not VRAM. Receipts record the working-set size, and
  sizes that straddle your card's cache size are the ones to read carefully.
