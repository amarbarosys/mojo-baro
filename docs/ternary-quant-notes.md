# Ternary quantization families (Q2_B3, TQ1_0, TQ2_0)

Current truth for the ternary weight path: what the pack stores, what the
kernels decode, and the two gates that prove both. Correctness only; no
throughput has been measured for any of these kernels and none may be
claimed without the `bench/coldcache-protocol.md` preregistration flow.

## Formats

All three store each weight as one of `{-d, 0, +d}` with one fp16 `d` per
block, `d = fp16(amax)`, code `= roundf(x / amax) + 1` in `{0, 1, 2}` (half
away from zero). They differ only in block size and how the trits are packed.

| family | pack dtype | ggml type | block | payload | block bytes | bpw |
|---|---|---|---|---|---|---|
| Q2_B3 (B3S fork) | `q2b3` | 43 | 128 | 26 B | 28 (`d` first) | 1.75 |
| TQ1_0 (mainline) | `tq1` | 34 | 256 | 52 B (`qs[48]` + `qh[4]`) | 54 (`d` last) | 1.6875 |
| TQ2_0 (mainline) | `tq2` | 35 | 256 | 64 B | 66 (`d` last) | 2.0625 |

**Q2_B3** packs five trits per byte in base 3 (`3^5 = 243`). The 128 trits
form 4 chunks of 32; chunk `c` owns bytes `[6c .. 6c+5]` (30 trits, digit
`t % 5` of byte `6c + t/5`) plus two stragglers in shared byte `24 + (c>>1)`
at digit `2*(c&1) + (t-30)`. Decode of trit `t`: `(qs[byte] / 3^digit) % 3`.
Every byte/digit is a compile-time constant of the index, which is why the
kernel unrolls the whole block.

**TQ1_0** stores five trits per byte as a base-3 number *ceiling-scaled* into
0..255 (`q = (q*256 + 242) / 243`), so trit `n` (most significant first) is
recovered with a wrapping multiply and a shift: `((uint8)(q * 3^n) * 3) >> 8`.
Element order is interleaved: bytes `qs[0..31]` carry elements `n*32 + m`
(`n` = digit, `m` = byte), `qs[32..47]` carry `160 + n*16 + m`, and the four
`qh` bytes carry `240 + n*4 + j` with only four trits each (the value is
multiplied by 3 before scaling so the first trit sits at the top).

**TQ2_0** is plain two bits per weight: byte `qs[j + m]` (`j` in `{0, 32}`)
holds elements `j*4 + l*32 + m` in bits `2l..2l+1`.

Reference C for all of the above is `tools/ternary-ref.c`, byte-for-byte
copies of `quantize_row_*_ref` / `dequantize_row_*` from the staged llama.cpp
sources (`.work/b3s-ref/`, B3S fork for Q2_B3, mainline for TQ). The
`--selftest` of `tools/b3s-check.py` re-diffs those bodies whenever the
staging dir is present.

## Pack layout

`tools/engine-pack.py MODEL.gguf OUTDIR --q2b3|--tq1|--tq2` applies the same
tensor selection as `--q8` (every 2D bf16 weight except `token_embd`) and
stores, per tensor, the block payload bytes verbatim in weight-native order
`[out, in/BLOCK * PAYLOAD]` followed by fp16 scales `[out, in/BLOCK]`. The
scale is split out of the ggml block exactly as the q8 pack splits it, so a
row of `in` weights is `in/BLOCK * PAYLOAD` contiguous bytes and the kernels
address block `b` of row `r` at byte `b * PAYLOAD`. Index dtypes are
`q2b3` / `tq1` / `tq2`; `n_elem` counts weights, scales follow at
`offset + n_elem/BLOCK * PAYLOAD`.

Quantizers are the ggml reference functions vectorized in numpy: codes via
`rint` with exact `+-0.5` ties forced away from zero (roundf semantics), then
one integer matrix product per family against a constant `[BLOCK, PAYLOAD]`
weight matrix that encodes the byte/digit map, then the TQ1_0 ceiling scale.

`serve/engine.mojo` does not consume these dtypes yet; its index parser
rejects them. Wiring a ternary pack into the engine is a separate step (a
ternary-native model also has a different tensor order than the packer's
fixed Qwythos list).

## Kernels

`kernels/matmul_ternary.mojo`, one wave per weight row like
`amar_matmul_skinny_q8row`, template parameter `MR` = activation rows carried
(m <= MR), `Cp[0, r, row]` output:

- `amar_matmul_skinny_q2b3row` — lane `l` owns blocks `l, l+32, ...`; loads
  the 26 payload bytes as scalars, decodes each 32-trit chunk as two 16-wide
  fp32 vectors with constant divide/modulo per element.
- `amar_matmul_skinny_tq1row` — 13 x 4-byte loads joined into the 32/16/4
  byte groups; each digit is one wrapping uint8 multiply and a shift on the
  whole group; the `qh` trits are gathered into one 16-wide vector.
- `amar_matmul_skinny_tq2row` — 4 x 16-byte loads; shift-and-mask per bit
  pair.

All three accumulate `sum(trit * a)` for a block in fp32 (the products are
exact: `+-a` or `0`), then add `blocksum * d` once per block, then
`warp.sum`. The activation is the same bf16 `[M, K]` the q8 path uses.

## Gates

```
python3 tools/b3s-check.py --selftest
    Python packer vs C reference, bit for bit, random ternary-native and
    random fp32 rows (incl. zero blocks and exact +-0.5 ties), 3 families.
python3 tools/b3s-check.py PACKDIR MODEL.gguf name...
    Pack entry payload + scales vs C reference on the bf16 tensor.
python3 tools/b3s-check.py --quant Q.gguf PACKDIR MODEL.gguf name...
    As above plus the llama-quantize TQ1_0/TQ2_0 blocks of Q.gguf
    (`llama-quantize --pure MODEL.gguf Q.gguf TQ1_0`). No Q2_B3 GGUF exists
    on this machine; the C reference is the only oracle for that family.
python3 tools/b3s-check.py --fixture 8 MODEL.gguf blk.0.ffn_gate.weight blk.0.ffn_down.weight
mojo build kernels/test_ternary_gemm.mojo -o .work/test_ternary_gemm -I kernels
gpu-wait run --priority 90 --vram 3 -- bash -c 'cd REPO && ./.work/test_ternary_gemm'
    Kernel parity vs A @ dequant(W)^T from the C dequantizer, K = 4096 and
    12288, MR = 8 and 1, gate max_rel < 1e-2 (measured ~1e-4).
```
