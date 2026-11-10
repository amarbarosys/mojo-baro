# CPU backend design (2026-10-07)

Status: DESIGN, nothing built. Claim: a decode-only CPU engine of about 1,050
LOC can be byte-identical to the GPU m=1 kernels per op, by emulating the
32-lane wave order on the CPU, and it builds with the `mojo` package alone.
Targets: dense qwen35 9B decode on x86_64 and aarch64 Linux, and Android
(step 0b passed 2026-10-07).

## Results so far

- **0a PASS (x86_64, 2026-10-07):** a venv with only `mojo==1.1.0` (692 MB, vs
  1.9 GB for `max[all]`) builds and runs std code. The std library on 1.1 has
  no `parallelize` (it lives in `max.algorithm`, absent here), so threads come
  from `pthread_create` through `external_call` with an `abi("C")` callback:
  8 threads, sum 360 as expected. `cpu/pool.mojo` is therefore in scope.
  `mojo build` prints a `FATAL ... posix_spawn` line (crash-reporter handler
  missing) and still writes a working binary.
- **0b PASS (Android 16, arm64-v8a, 2026-10-07):** `mojo build --emit object
  --target-triple aarch64-unknown-linux-android` works. The object needs ten
  `KGEN_CompilerRT_*` runtime symbols that exist only as x86 glibc libraries;
  `cpu/android/mojo_rt_shim.c` (~40 LOC) implements them and the NDK clang
  links the result. Hello (float loop + print) and the pthread probe both run
  on the phone with the expected output; also on a Galaxy A21s (SM-A217F,
  Android 12, 2.7 GB RAM). Repeatable: `cpu/android/probe.sh`.
  Not yet exercised on the phone: SIMD width, `fma` lowering, mmap of a
  multi-GB file; those are step 1's byte test run on the device.

## Goals, in priority order

1. **Parity**: byte-identical per-op output against the GPU m=1 decode kernels,
   so the CPU engine is a second arm of the same model, not a new model.
2. **Minimum LOC**: reuse every GPU-free module; port only the decode ops.
3. **Bytes**: weights never copied or dequantized; one arena at boot; zero
   allocation per token (`memory-optimization`).
4. **Modular**: builds and runs with the `mojo` package alone. No `max`, no
   ROCm, no `DeviceContext` anywhere in its import graph.

## Module graph

```
                 baro-serve (Rust router, one new flag)
                    | serve_proto over stdin/stdout
        +-----------+------------+
   engine.mojo (GPU)        cpu/engine_cpu.mojo (NEW)
        |                        |
   registry, window,        cpu/ops.mojo   lane-emulated dots, rms, rope, act
   harness, kernels/*       cpu/ssm.mojo   conv1d + delta step
   (max.gpu)                cpu/attn.mojo  decode attention over f32 KV
                            cpu/pack.mojo  index.txt + mmap pack.bin
                            cpu/pool.mojo  thread pool (only if 0a fails)
        \                        /
         shared, already GPU-free (import scan 2026-10-07):
         serve/model.mojo  serve/sample_ref.mojo  serve/tokenizer.mojo
         serve/serve_proto.mojo  uregex/
```

Rule: `cpu/` imports only `std` and the shared row. `tools/ci-checks.sh` gets
a grep that fails if `cpu/` or the shared row imports `max.` or `gpu`.

The pack needs no new format: `index.txt` is `name dtype offset bytes` per
tensor (442 rows for the 9B q4 pack), so `cpu/pack.mojo` maps `pack.bin` and
hands out typed pointers at those offsets.

## Parity: emulate the wave, do not approximate it

GPU m=1 dots (`q4_dot_lds`, `q8_row_dot` in `kernels/mega.mojo`) accumulate
per lane: lane `l` owns blocks `l, l+32, l+64, ...`, keeps a `SIMD[f32,16]`
fma chain (`fma(wlo, a_lo, fma(whi, a_hi, acc))`), then `acc.reduce_add()`,
then `warp.sum` across 32 lanes.

CPU form: walk blocks in memory order, `acc[blk % 32]` updated with the
identical fma chain. Same memory stream as a plain loop (the 32 lanes cover
contiguous blocks), so the emulation costs registers, not bandwidth. Then
`reduce_add` per lane and the same 32-lane tree as `warp.sum` on gfx1100
(order read from the stdlib lowering, step 0c).

Same treatment for every reduction with a fixed GPU order: rmsnorm workgroup
sum, attention softmax max/sum, SSM delta column. Elementwise ops (rope,
silu, residual) are order-free and need only the same formula and fma use.

Gate per op: byte test against the m=1 kernel on real layer inputs
(`kernels/test_*_rows.mojo` pattern). Engine gate: 20-prompt token identity
against the GPU engine through `bench/engine-ab.sh`. An op that cannot be made
byte-exact moves the engine gate to teacher-forced agreement (`BARO_FORCE`)
with a same-arm control and a bar frozen before the run.

## Memory plan

| Buffer | Size (9B, TMAX 4096) | Form |
|---|---|---|
| weights | pack.bin, ~5.4 GB | `mmap` read-only, page cache, zero copy |
| embedding row | H x bf16 | read in place from the mmap |
| activations | ~0.5 MB | one arena, carved at boot |
| KV, attn layers only | N_ATT x 2 x TMAX x KV x f32 | one allocation, f32 like the GPU arm |
| SSM conv + state | fixed per layer | one allocation |
| logits | VOCAB x f32 | in the arena |

Zero allocations in the token loop. Each step records RSS (`VmRSS`, `VmHWM`
from `/proc/self/status`) and tok/s before and after, numbers in the commit
body. Hot path is the q4 GEMV stream; its bound is pack bytes over DRAM
bandwidth.

## LOC budget

| File | LOC |
|---|---|
| cpu/ops.mojo | ~200 |
| cpu/ssm.mojo | ~150 |
| cpu/attn.mojo | ~120 |
| cpu/pack.mojo | ~60 |
| cpu/engine_cpu.mojo | ~300 |
| cpu/pool.mojo (pthread) | ~40 |
| cpu/android/ (runtime shim + probe) | ~40, landed |
| byte tests vs m=1 kernels | ~200 |
| router flag + ci grep | ~20 |
| **total** | **~1,050 (XL)** |

Lower than the first 1,500 estimate because sampler (`sample_ref.mojo`, the
host oracle), tokenizer, protocol and model constants are reused unchanged.
Out of scope for v1: MoE (+~300), batched prefill (prefill runs the decode
loop), MTP, state export, vision, grammar.

## Steps

```
0a  mojo-only build      std-only probe compiles without max     parallelize is in
                                                                 max.algorithm on 1.1;
                                                                 if max is needed,
                                                                 cpu/pool.mojo
0b  Android runtime      trivial mojo binary runs on the phone    gates Android only
0c  warp.sum order       read gfx1100 lowering, write CPU tree    byte test, one dot
1   pack + q4/q8 dots    byte-equal vs q4_dot_lds / q8_row_dot    real layer rows
2   rms, rope, act       byte-equal vs mega.mojo phases
3   ssm step, attn step  byte-equal vs ssm_phases / attn_phases (non-dattn path)
4   engine_cpu loop      20-prompt identity vs GPU engine         bench/engine-ab.sh
5   router flag          baro-serve --engine cpu, chat over curl  user-path check
```

Each step lands as its own commit with its byte test. Step 4 is the first
engine claim; step 5 is the first a user meets.

## Risks

- `warp.sum` on AMD may lower to DPP in an order that is not a plain xor
  tree; then 0c fails and dots drop to the agreement gate.
- Mojo `reduce_add` order is library code, identical on both targets only if
  no target-specific path exists. Step 1's byte test checks it.
- CPU `fma` must lower to a fused instruction (x86 FMA3, aarch64 FMLA); an
  unfused fallback breaks parity silently. The byte test catches it.
- Speed is bandwidth-bound at ~5.4 GB per token: 7800X3D DDR5 at about
  60 GB/s caps near 11 tok/s, a phone lower. llama.cpp CPU on the same
  weights is the reference arm, cold on both sides.
