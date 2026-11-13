# WMMA failure diagnostics report

## Changed

`kernels/test_wmma_rdna34.mojo` now caches reference values and, on mismatch, prints up to eight wrong cells. Each record includes the reference matrix coordinate matching `got` within the existing `2^-19 sum|ab|` tolerance, when one exists, and the lane/register from the current layout. `store_acc`, the PASS line, and the failing `Error` remain unchanged.

Failure diagnostic hot path: host-side reference matching after a mismatch. The success path adds one 256-element float32 reference buffer (1,024 bytes). Build measurement on the same gfx1100 target: baseline 4.16 s / 505,888 KB peak RSS; changed 3.17 s / 481,468 KB. These are compile measurements, not a runtime performance claim.

## Builds and checks

- gfx1100 build: PASS using installed `mojo` (the requested `./.venv/bin/mojo` is absent). 3.17 s / 481,468 KB peak RSS.
- gfx1201 build: PASS. 3.24 s / 499,624 KB peak RSS.
- gfx1100 control on GPU via `gpu-wait run --shared --vram 1 --timeout 120`: PASS, `PASS wmma-gfx: 256/256 within 2^-19 sum|ab|, K=64`.
- gfx12-row-formula mutant on gfx1100 via `gpu-wait`: FAIL as required. Eight reported wrong cells each have a writer and a matching reference coordinate.
- `git diff --check`: PASS.
- `tools/ci-checks.sh`: RUNNING at report preparation, currently at “bench sources compile (no GPU needed to build)”. Result pending.

## Mutant output, first 10 lines

```text
C[0,1] gpu -3.9999995 want -4.0
bad r= 1  c= 0  got= -1.9999983  want= 8.0
writer lane= 16  register= 0
matches want at r2= 0  c2= 3
bad r= 1  c= 1  got= 4.0000024  want= -7.0
writer lane= 17  register= 0
matches want at r2= 2  c2= 1
bad r= 1  c= 2  got= 5.000005  want= -7.0
writer lane= 18  register= 0
matches want at r2= 0  c2= 9
```

## UNVERIFIED

- CI result pending.
- No gfx1201 device run; this machine ran the control and mutant on gfx1100 only.
- No RDNA4 hardware run.
