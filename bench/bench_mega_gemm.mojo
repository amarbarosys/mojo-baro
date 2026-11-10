"""Stage 0 of bench/megakernel-protocol.md: fixed-grid q8row GEMM cost.

The q8row m=1 body run as a block-strided phase at G in {96,192,288,384}
blocks vs its native ceildiv(N, 8) grid, over the four decode GEMM shapes
from the q8 engine pack (blk.0), NBUF rotation per coldcache-protocol.md.
Correctness: block-strided output must be bit-identical to native.
"""

from max.gpu import block_idx, grid_dim, thread_idx, lane_id, WARP_SIZE
from max.gpu.primitives import warp
from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, TensorLayout, row_major

from matmul_skinny import amar_matmul_skinny_q8row, SM, SPLITK, ROW_WAVES, ROW_THREADS
from tio import load_into, pack_offset

comptime NBUF = 8
comptime ITERS = 200
comptime REPEATS = 5

comptime bf16 = DType.bfloat16
comptime f16 = DType.float16
comptime f32 = DType.float32
comptime i8 = DType.int8
comptime u8 = DType.uint8


def mega_q8row_phase[
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout, PLayout: TensorLayout
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    Q: TileTensor[i8, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    Cp: TileTensor[f32, PLayout, MutAnyOrigin],
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2
    comptime assert S.flat_rank == 2 and Cp.flat_rank == 3
    comptime UNROLL = 4
    comptime QV = 16
    comptime STEP = WARP_SIZE * QV
    var N = Int(n)
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var wave = Int(thread_idx.x) // WARP_SIZE
    var ngroups = ceildiv(N, ROW_WAVES)
    var Qv = Q.vectorize[1, QV]()
    var Av = A.vectorize[1, QV]()
    var g = Int(block_idx.x)
    while g < ngroups:
        var row = g * ROW_WAVES + wave
        if row < N:
            var acc = SIMD[f32, QV](0)
            var kk = 0
            while kk + UNROLL * STEP <= K:
                var qs = Array[SIMD[i8, QV], UNROLL](uninitialized=True)
                var ds = Array[Scalar[f16], UNROLL](uninitialized=True)
                comptime for u in range(UNROLL):
                    var kb = kk + u * STEP
                    qs[u] = rebind[SIMD[i8, QV]](Qv[row, kb // QV + lane])
                    ds[u] = rebind[Scalar[f16]](S[row, (kb + lane * QV) // 32])
                comptime for u in range(UNROLL):
                    var kb = kk + u * STEP
                    var w = qs[u].cast[f32]() * ds[u].cast[f32]()
                    var a = rebind[SIMD[bf16, QV]](Av[0, kb // QV + lane]).cast[f32]()
                    acc += w * a
                kk += UNROLL * STEP
            while kk < K:
                var q = rebind[SIMD[i8, QV]](Qv[row, kk // QV + lane]).cast[f32]()
                var d = rebind[Scalar[f16]](S[row, (kk + lane * QV) // 32]).cast[f32]()
                var a = rebind[SIMD[bf16, QV]](Av[0, kk // QV + lane]).cast[f32]()
                acc += (q * d) * a
                kk += STEP
            var total = warp.sum(acc.reduce_add())
            if lane == 0:
                Cp[0, 0, row] = rebind[Cp.ElementType](total)
        g += Int(grid_dim.x)


def qt[N: Int, K: Int](
    ctx: DeviceContext, q_dev: DeviceBuffer[u8], b: Int
) raises -> Tuple[TileTensor[i8, type_of(row_major[N, K]()), MutAnyOrigin], TileTensor[f16, type_of(row_major[N, K // 32]()), MutAnyOrigin]]:
    comptime QBYTES = N * K + N * (K // 32) * 2
    var qdp = q_dev.unsafe_ptr().unsafe_bitcast[UInt8]()
    var qb = DeviceBuffer[i8](ctx, (qdp + b * QBYTES).unsafe_bitcast[Int8](), N * K, owning=False)
    var sb = DeviceBuffer[f16](ctx, (qdp + b * QBYTES + N * K).unsafe_bitcast[Float16](), N * (K // 32), owning=False)
    return (TileTensor(qb, row_major[N, K]()), TileTensor(sb, row_major[N, K // 32]()))


def run_shape[N: Int, K: Int](ctx: DeviceContext, name: String, a_dev: DeviceBuffer[bf16]) raises:
    comptime QBYTES = N * K + N * (K // 32) * 2
    comptime a_layout = row_major[1, K]()
    comptime q_layout = row_major[N, K]()
    comptime s_layout = row_major[N, K // 32]()
    comptime p_layout = row_major[SPLITK, SM, N]()
    comptime native = amar_matmul_skinny_q8row[4, 1, type_of(a_layout), type_of(q_layout), type_of(s_layout), type_of(p_layout)]
    comptime phase = mega_q8row_phase[type_of(a_layout), type_of(q_layout), type_of(s_layout), type_of(p_layout)]
    comptime GR = ceildiv(N, ROW_WAVES)

    var q_host = ctx.enqueue_create_host_buffer[u8](QBYTES)
    var p_host = ctx.enqueue_create_host_buffer[f32](SPLITK * SM * N)
    var p2_host = ctx.enqueue_create_host_buffer[f32](SPLITK * SM * N)
    ctx.synchronize()
    load_into(".work/engine-pack-q8/pack.bin", q_host.unsafe_ptr().unsafe_bitcast[UInt8](), QBYTES, pack_offset("q8", name))
    var q_dev = ctx.enqueue_create_buffer[u8](NBUF * QBYTES)
    var p_dev = ctx.enqueue_create_buffer[f32](SPLITK * SM * N)
    var p2_dev = ctx.enqueue_create_buffer[f32](SPLITK * SM * N)
    var qdp = q_dev.unsafe_ptr().unsafe_bitcast[UInt8]()
    for b in range(NBUF):
        var qb = DeviceBuffer[u8](ctx, qdp + b * QBYTES, QBYTES, owning=False)
        ctx.enqueue_copy(dst_buf=qb, src_buf=q_host)
    ctx.enqueue_memset(p_dev, 0)
    ctx.enqueue_memset(p2_dev, 0)
    ctx.synchronize()

    var ab = DeviceBuffer[bf16](ctx, a_dev.unsafe_ptr(), K, owning=False)
    var A = TileTensor(ab, a_layout)
    var Cp = TileTensor(p_dev, p_layout)
    var Cp2 = TileTensor(p2_dev, p_layout)

    var grids: List[Int] = [96, 144, 192]

    var t0 = qt[N, K](ctx, q_dev, 0)
    ctx.enqueue_function[native](A, t0[0], t0[1], Cp, Int32(1), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
    ctx.enqueue_copy(dst_buf=p_host, src_buf=p_dev)
    ctx.synchronize()
    for gi in range(len(grids)):
        var G = grids[gi]
        ctx.enqueue_function[phase](A, t0[0], t0[1], Cp2, Int32(N), Int32(K), grid_dim=G, block_dim=ROW_THREADS)
        ctx.enqueue_copy(dst_buf=p2_host, src_buf=p2_dev)
        ctx.synchronize()
        var bad = 0
        for j in range(N):
            if p_host[j] != p2_host[j]:
                bad += 1
        print(name, " N=", N, " K=", K, " G=", G, " mismatches vs native:", bad)

    print("shape rep native_us g96_us g144_us g192_us   (grid native=", GR, ")")
    for rep in range(REPEATS):
        var us = List[Float64]()
        var w0 = perf_counter_ns()
        while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
            for b in range(NBUF):
                var t = qt[N, K](ctx, q_dev, b)
                ctx.enqueue_function[native](A, t[0], t[1], Cp, Int32(1), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
            ctx.synchronize()
        var s0 = perf_counter_ns()
        for it in range(ITERS):
            var t = qt[N, K](ctx, q_dev, it % NBUF)
            ctx.enqueue_function[native](A, t[0], t[1], Cp, Int32(1), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
        ctx.synchronize()
        us.append(Float64(perf_counter_ns() - s0) / 1.0e3 / Float64(ITERS))
        for gi in range(len(grids)):
            var G = grids[gi]
            w0 = perf_counter_ns()
            while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
                for b in range(NBUF):
                    var t = qt[N, K](ctx, q_dev, b)
                    ctx.enqueue_function[phase](A, t[0], t[1], Cp2, Int32(N), Int32(K), grid_dim=G, block_dim=ROW_THREADS)
                ctx.synchronize()
            s0 = perf_counter_ns()
            for it in range(ITERS):
                var t = qt[N, K](ctx, q_dev, it % NBUF)
                ctx.enqueue_function[phase](A, t[0], t[1], Cp2, Int32(N), Int32(K), grid_dim=G, block_dim=ROW_THREADS)
            ctx.synchronize()
            us.append(Float64(perf_counter_ns() - s0) / 1.0e3 / Float64(ITERS))
        print(name, " ", rep, " ", us[0], " ", us[1], " ", us[2], " ", us[3])


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var a_host = ctx.enqueue_create_host_buffer[bf16](12288)
    ctx.synchronize()
    for i in range(12288):
        a_host[i] = Scalar[bf16](Float32((i % 97) - 48) / 64.0)
    var a_dev = ctx.enqueue_create_buffer[bf16](12288)
    ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
    ctx.synchronize()
    run_shape[12288, 4096](ctx, "blk.0.ffn_gate.weight", a_dev)
    run_shape[4096, 12288](ctx, "blk.0.ffn_down.weight", a_dev)
    run_shape[8192, 4096](ctx, "blk.0.attn_qkv.weight", a_dev)
    run_shape[4096, 4096](ctx, "blk.0.attn_gate.weight", a_dev)
