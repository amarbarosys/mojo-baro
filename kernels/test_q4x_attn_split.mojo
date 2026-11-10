"""Split decode attention (amar_q4x_attn_sc / _sm / _out) against amar_q4x_attn: byte test and cold-cache timing.

Gate G1 of exchange/q4x-serve16k-prereg-item3.md: the bf16 output bytes of the three-kernel path equal the single-kernel path at
T in {1, 2, 255, 256, 257, 1000, 2051, 4096, 16384} (seeded random q, g, K, V; masked = 0). Timing then rotates NBUF distinct K/V
buffers (their total is far above the 96 MB Infinity Cache) at T in {256, 512, 1024, 2048, 4096, 8192, 16384} and prints us/call per
path, which sets the default Q4X_ATTSPLIT threshold.
Build: mojo build kernels/test_q4x_attn_split.mojo -I kernels -o .work/test_q4x_attn_split
"""
from std.math import sqrt
from std.memory import bitcast
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer
from q4x_kernels import amar_q4x_attn, amar_q4x_attn_sc, amar_q4x_attn_sm, amar_q4x_attn_out

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime u8 = DType.uint8
comptime u16 = DType.uint16
comptime i32 = DType.int32
comptime HD = 256
comptime NQ = 24
comptime NKV = 2
comptime TMAX = 16384
comptime NBUF = 6
comptime FP = MutPointer[Scalar[f32], MutAnyOrigin]
comptime BP = MutPointer[Scalar[bf16], MutAnyOrigin]
comptime UP = MutPointer[Scalar[u8], MutAnyOrigin]
comptime IP = MutPointer[Scalar[i32], MutAnyOrigin]


def fp(b: DeviceBuffer[f32], off: Int = 0) -> FP:
    return FP(unsafe_from_address=Int(b.unsafe_ptr()) + 4 * off)


def fill(b: DeviceBuffer[f32], n: Int, seed: UInt64) raises:
    var s = seed
    with b.map_to_host() as h:
        for i in range(n):
            s = s * 6364136223846793005 + 1442695040888963407
            h[i] = Float32(Int((s >> 40) % 2000000)) / 1e6 - 1


def run_old(ctx: DeviceContext, q: FP, g: FP, kc: FP, vc: FP, sc: FP, vis: UP, xq: BP, sst: IP, T: Int) raises:
    ctx.enqueue_function[amar_q4x_attn](q, g, Int32(NQ * HD), kc, vc, sc, Int32(T), vis, xq, Int32(NQ * HD), Int32(T - 1), sst, Int32(0), Int32(NQ), Int32(NKV), 1 / sqrt(Float32(HD)), grid_dim=(NQ, 1), block_dim=HD)


def run_new(ctx: DeviceContext, q: FP, g: FP, kc: FP, vc: FP, sc: FP, xq: BP, sst: IP, T: Int, part: Int = 7) raises:
    if part & 1 != 0:
        ctx.enqueue_function[amar_q4x_attn_sc[NQ // NKV]](q, Int32(NQ * HD), kc, sc, Int32(T), Int32(T - 1), sst, Int32(NKV), 1 / sqrt(Float32(HD)), grid_dim=(NKV, (T + 255) // 256), block_dim=256)
    if part & 2 != 0:
        ctx.enqueue_function[amar_q4x_attn_sm](sc, Int32(T), Int32(T - 1), sst, grid_dim=NQ, block_dim=256)
    if part & 4 != 0:
        ctx.enqueue_function[amar_q4x_attn_out](sc, Int32(T), vc, g, xq, Int32(T - 1), sst, Int32(NQ), Int32(NKV), grid_dim=(NQ, 4), block_dim=64)


def main() raises:
    var ctx = DeviceContext()
    var kvn = TMAX * NKV * HD
    var qd = ctx.enqueue_create_buffer[f32](NQ * HD)
    var gd = ctx.enqueue_create_buffer[f32](NQ * HD)
    var kcd = ctx.enqueue_create_buffer[f32](NBUF * kvn)
    var vcd = ctx.enqueue_create_buffer[f32](NBUF * kvn)
    var scd = ctx.enqueue_create_buffer[f32](NQ * TMAX)
    var visd = ctx.enqueue_create_buffer[u8](TMAX)
    var sstd = ctx.enqueue_create_buffer[i32](4)
    var xo = ctx.enqueue_create_buffer[bf16](NQ * HD)
    var xn = ctx.enqueue_create_buffer[bf16](NQ * HD)
    fill(qd, NQ * HD, 11)
    fill(gd, NQ * HD, 12)
    fill(kcd.create_sub_buffer[f32](0, kvn), kvn, 13)
    fill(vcd.create_sub_buffer[f32](0, kvn), kvn, 14)
    for r in range(1, NBUF):
        ctx.enqueue_copy(dst_buf=kcd.create_sub_buffer[f32](r * kvn, kvn), src_buf=kcd.create_sub_buffer[f32](0, kvn))
        ctx.enqueue_copy(dst_buf=vcd.create_sub_buffer[f32](r * kvn, kvn), src_buf=vcd.create_sub_buffer[f32](0, kvn))
    with sstd.map_to_host() as h:
        for i in range(4):
            h[i] = 0
    ctx.synchronize()
    var q = fp(qd)
    var g = fp(gd)
    var sc = fp(scd)
    var vis = UP(unsafe_from_address=Int(visd.unsafe_ptr()))
    var sst = IP(unsafe_from_address=Int(sstd.unsafe_ptr()))
    var po = BP(unsafe_from_address=Int(xo.unsafe_ptr()))
    var pn = BP(unsafe_from_address=Int(xn.unsafe_ptr()))
    var bad = 0
    for T in [1, 2, 255, 256, 257, 1000, 2051, 4096, 16384]:
        run_old(ctx, q, g, fp(kcd), fp(vcd), sc, vis, po, sst, T)
        ctx.synchronize()
        run_new(ctx, q, g, fp(kcd), fp(vcd), sc, pn, sst, T)
        ctx.synchronize()
        var diff = 0
        with xo.map_to_host() as a:
            with xn.map_to_host() as b:
                for i in range(NQ * HD):
                    if bitcast[u16, 1](SIMD[bf16, 1](a[i]))[0] != bitcast[u16, 1](SIMD[bf16, 1](b[i]))[0]:
                        diff += 1
        print("T", T, "differing bf16 outputs", diff, "of", NQ * HD)
        bad += diff
    if bad != 0:
        raise Error("FAIL test_q4x_attn_split: " + String(bad) + " outputs differ")
    print("G1 PASS test_q4x_attn_split: byte-equal at 9 lengths")
    for T in [256, 512, 1024, 2048, 4096, 8192, 16384]:
        var us = List[Float64]()
        for path in range(5):
            for w in range(30):
                var r = w % NBUF
                if path == 0:
                    run_old(ctx, q, g, fp(kcd, r * kvn), fp(vcd, r * kvn), sc, vis, po, sst, T)
                else:
                    run_new(ctx, q, g, fp(kcd, r * kvn), fp(vcd, r * kvn), sc, pn, sst, T, [0, 7, 1, 2, 4][path])
            ctx.synchronize()
            var t0 = perf_counter_ns()
            for it in range(60):
                var r = it % NBUF
                if path == 0:
                    run_old(ctx, q, g, fp(kcd, r * kvn), fp(vcd, r * kvn), sc, vis, po, sst, T)
                else:
                    run_new(ctx, q, g, fp(kcd, r * kvn), fp(vcd, r * kvn), sc, pn, sst, T, [0, 7, 1, 2, 4][path])
            ctx.synchronize()
            us.append(Float64(perf_counter_ns() - t0) / 1e3 / 60)
        print("T", T, "us/call old", us[0], "split", us[1], "| scores", us[2], "softmax", us[3], "out", us[4])
