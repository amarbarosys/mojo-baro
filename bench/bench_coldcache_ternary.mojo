"""T0 instrument, sibling of bench_coldcache_q8row.mojo: cold-cache M=1
ffn_gate shape (N=12288, K=4096) for the q8row control plus the three
ternary wave-per-row arms (q2b3row, tq1row, tq2row).

Reads blk.0.ffn_gate.weight from .work/engine-pack-{q8,q2b3,tq1,tq2}
(payload bytes verbatim + fp16 block scales) and A from the existing
.work/gguf/blk_0_ffn_gate_weight.a.bin (row 0). Correctness: fp64 host
reference. The q8 reference is an in-kernel fp64 dot, same as
bench_coldcache_q8row.mojo; the ternary references are precomputed by
bench/gen-ternary-ref.py (C-dequantized production pack, fp64 dot) into
.work/gguf/blk_0_ffn_gate_weight.<fam>.ref1.bin -- run that script first.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, row_major

from matmul_skinny import amar_matmul_skinny_q8row, ROW_WAVES, ROW_THREADS
from matmul_ternary import (
    amar_matmul_skinny_q2b3row, amar_matmul_skinny_tq1row, amar_matmul_skinny_tq2row,
    B3_BLOCK, B3_BYTES, TQ_BLOCK, TQ1_BYTES, TQ2_BYTES,
)
from tio import load_into, pack_offset
from bench_coldcache_q8row import max_rel, max_abs_norm

comptime M = 1
comptime K = 4096
comptime N = 12288
comptime NBUF = 8
comptime ITERS = 200
comptime REPEATS = 10
comptime MR = 1

comptime Q8_NB = K // 32
comptime B3_NB = K // B3_BLOCK
comptime TQ_NB = K // TQ_BLOCK

comptime Q8_QBYTES = N * K + N * Q8_NB * 2
comptime Q2B3_QBYTES = N * B3_NB * B3_BYTES + N * B3_NB * 2
comptime TQ1_QBYTES = N * TQ_NB * TQ1_BYTES + N * TQ_NB * 2
comptime TQ2_QBYTES = N * TQ_NB * TQ2_BYTES + N * TQ_NB * 2

comptime a_layout = row_major[M, K]()
comptime p_layout = row_major[1, MR, N]()

comptime q8_q_layout = row_major[N, K]()
comptime q8_s_layout = row_major[N, Q8_NB]()
comptime b3_q_layout = row_major[N, B3_NB * B3_BYTES]()
comptime b3_s_layout = row_major[N, B3_NB]()
comptime tq1_q_layout = row_major[N, TQ_NB * TQ1_BYTES]()
comptime tq1_s_layout = row_major[N, TQ_NB]()
comptime tq2_q_layout = row_major[N, TQ_NB * TQ2_BYTES]()
comptime tq2_s_layout = row_major[N, TQ_NB]()

comptime bf16 = DType.bfloat16
comptime f16 = DType.float16
comptime f32 = DType.float32
comptime i8 = DType.int8
comptime u8 = DType.uint8

comptime q8kernel = amar_matmul_skinny_q8row[4, MR, type_of(a_layout), type_of(q8_q_layout), type_of(q8_s_layout), type_of(p_layout)]
comptime b3kernel = amar_matmul_skinny_q2b3row[MR, type_of(a_layout), type_of(b3_q_layout), type_of(b3_s_layout), type_of(p_layout)]
comptime tq1kernel = amar_matmul_skinny_tq1row[MR, type_of(a_layout), type_of(tq1_q_layout), type_of(tq1_s_layout), type_of(p_layout)]
comptime tq2kernel = amar_matmul_skinny_tq2row[MR, type_of(a_layout), type_of(tq2_q_layout), type_of(tq2_s_layout), type_of(p_layout)]

comptime GR = ceildiv(N, ROW_WAVES)


def report(kind: String, ref_host: HostBuffer[f32], got: HostBuffer[f32]) -> Bool:
    var rel = max_rel(ref_host, got)
    var absn = max_abs_norm(ref_host, got)
    var ok = rel < 1e-2
    print(kind, "max_rel:", rel, " abs/maxref:", absn, " correct=", ok)
    return ok


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    comptime base = ".work/gguf/blk_0_ffn_gate_weight"
    comptime name = "blk.0.ffn_gate.weight"

    print("grid_dim=", GR, " block_dim=", ROW_THREADS, " ROW_WAVES=", ROW_WAVES, " MR=", MR)

    var a_host = ctx.enqueue_create_host_buffer[bf16](M * K)
    ctx.synchronize()
    load_into(base + ".a.bin", a_host.unsafe_ptr().unsafe_bitcast[UInt8](), M * K * 2)
    var a_dev = ctx.enqueue_create_buffer[bf16](M * K)
    ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
    ctx.synchronize()
    var A = TileTensor(a_dev, a_layout)

    var refq8_host = ctx.enqueue_create_host_buffer[f32](N)
    var refb3_host = ctx.enqueue_create_host_buffer[f32](N)
    var reftq1_host = ctx.enqueue_create_host_buffer[f32](N)
    var reftq2_host = ctx.enqueue_create_host_buffer[f32](N)
    var got_host = ctx.enqueue_create_host_buffer[f32](N)
    var p_host = ctx.enqueue_create_host_buffer[f32](MR * N)
    ctx.synchronize()
    load_into(base + ".q2b3.ref1.bin", refb3_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * 4)
    load_into(base + ".tq1.ref1.bin", reftq1_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * 4)
    load_into(base + ".tq2.ref1.bin", reftq2_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * 4)

    var q8off = pack_offset("q8", name)
    var q8_host = ctx.enqueue_create_host_buffer[u8](Q8_QBYTES)
    ctx.synchronize()
    load_into(".work/engine-pack-q8/pack.bin", q8_host.unsafe_ptr(), Q8_QBYTES, q8off)
    var qp8 = q8_host.unsafe_ptr().unsafe_bitcast[Int8]()
    var sp8 = (q8_host.unsafe_ptr() + N * K).unsafe_bitcast[Float16]()
    for j in range(N):
        var s: Float64 = 0
        for k in range(K):
            var d = Float64(sp8[j * Q8_NB + k // 32])
            s += Float64(qp8[j * K + k]) * d * Float64(a_host[k])
        refq8_host[j] = Float32(s)

    var b3off = pack_offset("q2b3", name)
    var b3_host = ctx.enqueue_create_host_buffer[u8](Q2B3_QBYTES)
    var tq1off = pack_offset("tq1", name)
    var tq1_host = ctx.enqueue_create_host_buffer[u8](TQ1_QBYTES)
    var tq2off = pack_offset("tq2", name)
    var tq2_host = ctx.enqueue_create_host_buffer[u8](TQ2_QBYTES)
    ctx.synchronize()
    load_into(".work/engine-pack-q2b3/pack.bin", b3_host.unsafe_ptr(), Q2B3_QBYTES, b3off)
    load_into(".work/engine-pack-tq1/pack.bin", tq1_host.unsafe_ptr(), TQ1_QBYTES, tq1off)
    load_into(".work/engine-pack-tq2/pack.bin", tq2_host.unsafe_ptr(), TQ2_QBYTES, tq2off)

    var q8_dev = ctx.enqueue_create_buffer[u8](NBUF * Q8_QBYTES)
    var b3_dev = ctx.enqueue_create_buffer[u8](NBUF * Q2B3_QBYTES)
    var tq1_dev = ctx.enqueue_create_buffer[u8](NBUF * TQ1_QBYTES)
    var tq2_dev = ctx.enqueue_create_buffer[u8](NBUF * TQ2_QBYTES)
    var q8p = q8_dev.unsafe_ptr()
    var b3p = b3_dev.unsafe_ptr()
    var t1p = tq1_dev.unsafe_ptr()
    var t2p = tq2_dev.unsafe_ptr()
    for b in range(NBUF):
        var q8b = DeviceBuffer[u8](ctx, q8p + b * Q8_QBYTES, Q8_QBYTES, owning=False)
        ctx.enqueue_copy(dst_buf=q8b, src_buf=q8_host)
        var b3b = DeviceBuffer[u8](ctx, b3p + b * Q2B3_QBYTES, Q2B3_QBYTES, owning=False)
        ctx.enqueue_copy(dst_buf=b3b, src_buf=b3_host)
        var t1b = DeviceBuffer[u8](ctx, t1p + b * TQ1_QBYTES, TQ1_QBYTES, owning=False)
        ctx.enqueue_copy(dst_buf=t1b, src_buf=tq1_host)
        var t2b = DeviceBuffer[u8](ctx, t2p + b * TQ2_QBYTES, TQ2_QBYTES, owning=False)
        ctx.enqueue_copy(dst_buf=t2b, src_buf=tq2_host)
    ctx.synchronize()

    var p_dev = ctx.enqueue_create_buffer[f32](MR * N)
    var Cp = TileTensor(p_dev, p_layout)

    var q8q0 = DeviceBuffer[i8](ctx, q8p.unsafe_bitcast[Int8](), N * K, owning=False)
    var q8s0 = DeviceBuffer[f16](ctx, (q8p + N * K).unsafe_bitcast[Float16](), N * Q8_NB, owning=False)
    ctx.enqueue_memset(p_dev, Float32(0))
    ctx.enqueue_function[q8kernel](
        A, TileTensor(q8q0, q8_q_layout), TileTensor(q8s0, q8_s_layout), Cp, Int32(M), Int32(N), Int32(K),
        grid_dim=GR, block_dim=ROW_THREADS,
    )
    ctx.enqueue_copy(dst_buf=p_host, src_buf=p_dev)
    ctx.synchronize()
    for j in range(N):
        got_host[j] = p_host[j]
    _ = report("q8row  ", refq8_host, got_host)

    var b3q0 = DeviceBuffer[u8](ctx, b3p, N * B3_NB * B3_BYTES, owning=False)
    var b3s0 = DeviceBuffer[f16](ctx, (b3p + N * B3_NB * B3_BYTES).unsafe_bitcast[Float16](), N * B3_NB, owning=False)
    ctx.enqueue_memset(p_dev, Float32(0))
    ctx.enqueue_function[b3kernel](
        A, TileTensor(b3q0, b3_q_layout), TileTensor(b3s0, b3_s_layout), Cp, Int32(M), Int32(N), Int32(K),
        grid_dim=GR, block_dim=ROW_THREADS,
    )
    ctx.enqueue_copy(dst_buf=p_host, src_buf=p_dev)
    ctx.synchronize()
    for j in range(N):
        got_host[j] = p_host[j]
    _ = report("q2b3row", refb3_host, got_host)

    var t1q0 = DeviceBuffer[u8](ctx, t1p, N * TQ_NB * TQ1_BYTES, owning=False)
    var t1s0 = DeviceBuffer[f16](ctx, (t1p + N * TQ_NB * TQ1_BYTES).unsafe_bitcast[Float16](), N * TQ_NB, owning=False)
    ctx.enqueue_memset(p_dev, Float32(0))
    ctx.enqueue_function[tq1kernel](
        A, TileTensor(t1q0, tq1_q_layout), TileTensor(t1s0, tq1_s_layout), Cp, Int32(M), Int32(N), Int32(K),
        grid_dim=GR, block_dim=ROW_THREADS,
    )
    ctx.enqueue_copy(dst_buf=p_host, src_buf=p_dev)
    ctx.synchronize()
    for j in range(N):
        got_host[j] = p_host[j]
    _ = report("tq1row ", reftq1_host, got_host)

    var t2q0 = DeviceBuffer[u8](ctx, t2p, N * TQ_NB * TQ2_BYTES, owning=False)
    var t2s0 = DeviceBuffer[f16](ctx, (t2p + N * TQ_NB * TQ2_BYTES).unsafe_bitcast[Float16](), N * TQ_NB, owning=False)
    ctx.enqueue_memset(p_dev, Float32(0))
    ctx.enqueue_function[tq2kernel](
        A, TileTensor(t2q0, tq2_q_layout), TileTensor(t2s0, tq2_s_layout), Cp, Int32(M), Int32(N), Int32(K),
        grid_dim=GR, block_dim=ROW_THREADS,
    )
    ctx.enqueue_copy(dst_buf=p_host, src_buf=p_dev)
    ctx.synchronize()
    for j in range(N):
        got_host[j] = p_host[j]
    _ = report("tq2row ", reftq2_host, got_host)

    print("rep  q8row_us  q2b3row_us  tq1row_us  tq2row_us")
    for rep in range(REPEATS):
        var w0 = perf_counter_ns()
        while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
            for b in range(NBUF):
                var qb = DeviceBuffer[i8](ctx, (q8p + b * Q8_QBYTES).unsafe_bitcast[Int8](), N * K, owning=False)
                var sb = DeviceBuffer[f16](ctx, (q8p + b * Q8_QBYTES + N * K).unsafe_bitcast[Float16](), N * Q8_NB, owning=False)
                ctx.enqueue_function[q8kernel](A, TileTensor(qb, q8_q_layout), TileTensor(sb, q8_s_layout), Cp, Int32(M), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
            ctx.synchronize()
        var t0 = perf_counter_ns()
        for it in range(ITERS):
            var b = it % NBUF
            var qb = DeviceBuffer[i8](ctx, (q8p + b * Q8_QBYTES).unsafe_bitcast[Int8](), N * K, owning=False)
            var sb = DeviceBuffer[f16](ctx, (q8p + b * Q8_QBYTES + N * K).unsafe_bitcast[Float16](), N * Q8_NB, owning=False)
            ctx.enqueue_function[q8kernel](A, TileTensor(qb, q8_q_layout), TileTensor(sb, q8_s_layout), Cp, Int32(M), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
        ctx.synchronize()
        var q8_us = Float64(perf_counter_ns() - t0) / 1.0e3 / Float64(ITERS)

        w0 = perf_counter_ns()
        while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
            for b in range(NBUF):
                var qb = DeviceBuffer[u8](ctx, b3p + b * Q2B3_QBYTES, N * B3_NB * B3_BYTES, owning=False)
                var sb = DeviceBuffer[f16](ctx, (b3p + b * Q2B3_QBYTES + N * B3_NB * B3_BYTES).unsafe_bitcast[Float16](), N * B3_NB, owning=False)
                ctx.enqueue_function[b3kernel](A, TileTensor(qb, b3_q_layout), TileTensor(sb, b3_s_layout), Cp, Int32(M), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
            ctx.synchronize()
        t0 = perf_counter_ns()
        for it in range(ITERS):
            var b = it % NBUF
            var qb = DeviceBuffer[u8](ctx, b3p + b * Q2B3_QBYTES, N * B3_NB * B3_BYTES, owning=False)
            var sb = DeviceBuffer[f16](ctx, (b3p + b * Q2B3_QBYTES + N * B3_NB * B3_BYTES).unsafe_bitcast[Float16](), N * B3_NB, owning=False)
            ctx.enqueue_function[b3kernel](A, TileTensor(qb, b3_q_layout), TileTensor(sb, b3_s_layout), Cp, Int32(M), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
        ctx.synchronize()
        var b3_us = Float64(perf_counter_ns() - t0) / 1.0e3 / Float64(ITERS)

        w0 = perf_counter_ns()
        while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
            for b in range(NBUF):
                var qb = DeviceBuffer[u8](ctx, t1p + b * TQ1_QBYTES, N * TQ_NB * TQ1_BYTES, owning=False)
                var sb = DeviceBuffer[f16](ctx, (t1p + b * TQ1_QBYTES + N * TQ_NB * TQ1_BYTES).unsafe_bitcast[Float16](), N * TQ_NB, owning=False)
                ctx.enqueue_function[tq1kernel](A, TileTensor(qb, tq1_q_layout), TileTensor(sb, tq1_s_layout), Cp, Int32(M), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
            ctx.synchronize()
        t0 = perf_counter_ns()
        for it in range(ITERS):
            var b = it % NBUF
            var qb = DeviceBuffer[u8](ctx, t1p + b * TQ1_QBYTES, N * TQ_NB * TQ1_BYTES, owning=False)
            var sb = DeviceBuffer[f16](ctx, (t1p + b * TQ1_QBYTES + N * TQ_NB * TQ1_BYTES).unsafe_bitcast[Float16](), N * TQ_NB, owning=False)
            ctx.enqueue_function[tq1kernel](A, TileTensor(qb, tq1_q_layout), TileTensor(sb, tq1_s_layout), Cp, Int32(M), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
        ctx.synchronize()
        var t1_us = Float64(perf_counter_ns() - t0) / 1.0e3 / Float64(ITERS)

        w0 = perf_counter_ns()
        while Float64(perf_counter_ns() - w0) / 1.0e9 < 1.0:
            for b in range(NBUF):
                var qb = DeviceBuffer[u8](ctx, t2p + b * TQ2_QBYTES, N * TQ_NB * TQ2_BYTES, owning=False)
                var sb = DeviceBuffer[f16](ctx, (t2p + b * TQ2_QBYTES + N * TQ_NB * TQ2_BYTES).unsafe_bitcast[Float16](), N * TQ_NB, owning=False)
                ctx.enqueue_function[tq2kernel](A, TileTensor(qb, tq2_q_layout), TileTensor(sb, tq2_s_layout), Cp, Int32(M), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
            ctx.synchronize()
        t0 = perf_counter_ns()
        for it in range(ITERS):
            var b = it % NBUF
            var qb = DeviceBuffer[u8](ctx, t2p + b * TQ2_QBYTES, N * TQ_NB * TQ2_BYTES, owning=False)
            var sb = DeviceBuffer[f16](ctx, (t2p + b * TQ2_QBYTES + N * TQ_NB * TQ2_BYTES).unsafe_bitcast[Float16](), N * TQ_NB, owning=False)
            ctx.enqueue_function[tq2kernel](A, TileTensor(qb, tq2_q_layout), TileTensor(sb, tq2_s_layout), Cp, Int32(M), Int32(N), Int32(K), grid_dim=GR, block_dim=ROW_THREADS)
        ctx.synchronize()
        var t2_us = Float64(perf_counter_ns() - t0) / 1.0e3 / Float64(ITERS)

        print(rep, " ", q8_us, " ", b3_us, " ", t1_us, " ", t2_us)

    print("bytes: q8=", Q8_QBYTES, " q2b3=", Q2B3_QBYTES, " tq1=", TQ1_QBYTES, " tq2=", TQ2_QBYTES)
