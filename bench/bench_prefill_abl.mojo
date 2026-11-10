"""ABLATION RECEIPT (bench/prefill-protocol.md R5-abl), not an arm: prefill GEMM on the ffn shape (N=12288, K=4096, Q4_0 from the q4
pack), bench/prefill-protocol.md kernel table: bf16-WMMA prefill kernel vs
the decode-path wave-per-row kernel looped in 8-row windows (its register
ceiling), n = 16 .. 1024 activation rows, plus (R5) the LDS-pipelined
schedule instantiated for bf16 (bf16-lds) and for int8 activations (mmq,
with the q8 quantiser timed separately as `quant`). NBUF weight copies
rotate so the weight stream is cold (8 x 28 MB > 96 MB Infinity Cache).
Correctness gates before any timing is printed: WMMA vs row loop rel <
1e-4, bf16-lds vs row loop rel < 1e-4, mmq vs bf16-lds relative Frobenius
< 1e-2 and max|a-b|/rms(b) < 5e-2.
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, TensorLayout, row_major

from matmul_skinny import amar_matmul_skinny_q4rowb, amar_skinny_reduce, SM, ROW_WAVES, ROW_THREADS
from matmul_prefill import amar_matmul_prefill_q4, PF_THREADS
from matmul_mmq import amar_quant_q8, amar_matmul_lds_q4, MMQ_THREADS, QT_THREADS, ABL
from tio import load_into, pack_offset
from bench_prefill import lcg, q_tensors, lds_cfg_name, timed

comptime K = 4096
comptime N = 12288
comptime MMAX = 1024
comptime NBUF = 8
comptime ITERS = 16
comptime REPEATS = 5
comptime Q4BYTES = N * K // 2 + N * (K // 32) * 2

comptime a_layout = row_major[MMAX, K]()
comptime a8_layout = row_major[SM, K]()
comptime c_layout = row_major[MMAX, N]()
comptime c8_layout = row_major[SM, N]()
comptime q4_layout = row_major[N, K // 2]()
comptime s_layout = row_major[N, K // 32]()
comptime p_layout = row_major[1, SM, N]()
comptime d_layout = row_major[K // 32, MMAX]()

comptime bf16 = DType.bfloat16
comptime f16 = DType.float16
comptime f32 = DType.float32
comptime u8 = DType.uint8
comptime i8 = DType.int8
comptime i32 = DType.int32


def lds_launch[ADT: DType, WM: Int, WN: Int, TM: Int, TN: Int](
    ctx: DeviceContext, A: TileTensor[ADT, type_of(a_layout), MutAnyOrigin],
    Ad: TileTensor[f32, type_of(d_layout), MutAnyOrigin], An: TileTensor[i32, type_of(d_layout), MutAnyOrigin],
    Q: TileTensor[u8, type_of(q4_layout), MutAnyOrigin], S: TileTensor[f16, type_of(s_layout), MutAnyOrigin],
    C: TileTensor[f32, type_of(c_layout), MutAnyOrigin], m: Int,
) raises:
    comptime BM = WM * TM * 16
    comptime BN = WN * TN * 16
    ctx.enqueue_function[amar_matmul_lds_q4[ADT, WM, WN, TM, TN, False, type_of(a_layout), type_of(d_layout), type_of(d_layout), type_of(q4_layout), type_of(s_layout), type_of(c_layout)]](
        A, Ad, An, Q, S, C, Int32(m), Int32(N), Int32(K), grid_dim=(ceildiv(N, BN), ceildiv(m, BM)), block_dim=MMQ_THREADS,
    )


def lds_dispatch[ADT: DType](cfg: Int, ctx: DeviceContext, A: TileTensor[ADT, type_of(a_layout), MutAnyOrigin],
                 Ad: TileTensor[f32, type_of(d_layout), MutAnyOrigin], An: TileTensor[i32, type_of(d_layout), MutAnyOrigin],
                 Q: TileTensor[u8, type_of(q4_layout), MutAnyOrigin], S: TileTensor[f16, type_of(s_layout), MutAnyOrigin],
                 C: TileTensor[f32, type_of(c_layout), MutAnyOrigin], m: Int) raises:
    if cfg == 0:
        lds_launch[ADT, 2, 4, 2, 2](ctx, A, Ad, An, Q, S, C, m)
    else:
        lds_launch[ADT, 4, 2, 2, 4](ctx, A, Ad, An, Q, S, C, m)


def lds_grid(cfg: Int, m: Int) -> String:
    var bm = 64 if cfg == 0 else 128
    return "grid=(" + String(ceildiv(N, 128)) + "," + String(ceildiv(m, bm)) + ") block=" + String(MMQ_THREADS)


def quant_launch(ctx: DeviceContext, A: TileTensor[bf16, type_of(a_layout), MutAnyOrigin],
                 Aq: TileTensor[i8, type_of(a_layout), MutAnyOrigin], Ad: TileTensor[f32, type_of(d_layout), MutAnyOrigin],
                 An: TileTensor[i32, type_of(d_layout), MutAnyOrigin], m: Int, mpad: Int) raises:
    ctx.enqueue_function[amar_quant_q8[type_of(a_layout), type_of(a_layout), type_of(d_layout), type_of(d_layout)]](
        A, Aq, Ad, An, Int32(m), Int32(K), Int32(mpad), grid_dim=ceildiv(mpad * (K // 32), QT_THREADS), block_dim=QT_THREADS,
    )


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var q_h = ctx.enqueue_create_host_buffer[u8](Q4BYTES)
    var a_h = ctx.enqueue_create_host_buffer[bf16](MMAX * K)
    ctx.synchronize()
    load_into(".work/engine-pack-q4/pack.bin", q_h.unsafe_ptr(), Q4BYTES, pack_offset("q4", "blk.0.ffn_gate.weight"))
    var st = UInt64(99)
    for i in range(MMAX * K):
        a_h[i] = Scalar[bf16](lcg(st))
    var q_d = ctx.enqueue_create_buffer[u8](NBUF * Q4BYTES)
    var a_d = ctx.enqueue_create_buffer[bf16](MMAX * K)
    var c_d = ctx.enqueue_create_buffer[f32](MMAX * N)
    var aq_d = ctx.enqueue_create_buffer[i8](MMAX * K)
    var ad_d = ctx.enqueue_create_buffer[f32](MMAX * (K // 32))
    var an_d = ctx.enqueue_create_buffer[i32](MMAX * (K // 32))
    for b in range(NBUF):
        var dst = DeviceBuffer[u8](ctx, q_d.unsafe_ptr().unsafe_offset(b * Q4BYTES), Q4BYTES, owning=False)
        ctx.enqueue_copy(dst_buf=dst, src_buf=q_h)
    ctx.enqueue_copy(dst_buf=a_d, src_buf=a_h)
    ctx.synchronize()
    var A = TileTensor(a_d, a_layout)
    var C = TileTensor(c_d, c_layout)
    var Aq = TileTensor(aq_d, a_layout)
    var Ad = TileTensor(ad_d, d_layout)
    var An = TileTensor(an_d, d_layout)
    quant_launch(ctx, A, Aq, Ad, An, MMAX, MMAX)
    ctx.synchronize()
    print("ABLATION ABL", ABL, "(0 none, 1 no B global load, 2 no A global load, 3 no per-block epilogue, 4 no barrier, 5 no d8/nu loads) shape N", N, "K", K, "NBUF", NBUF, "ITERS", ITERS, "REPEATS", REPEATS)
    var ns: List[Int] = [256, 512, 1024]
    for ni in range(len(ns)):
        var m = ns[ni]
        var meds3 = List[Float64]()
        for _ in range(REPEATS):
            ctx.synchronize()
            var t0 = perf_counter_ns()
            for it in range(ITERS):
                var qb = q_tensors(ctx, q_d, it % NBUF)
                lds_dispatch[bf16](1, ctx, A, Ad, An, qb[0], qb[1], C, m)
            ctx.synchronize()
            meds3.append(Float64(perf_counter_ns() - t0) / 1e3 / ITERS)
        timed(ctx, m, "bf16-lds", lds_cfg_name(1), lds_grid(1, m), meds3)
        var meds4 = List[Float64]()
        for _ in range(REPEATS):
            ctx.synchronize()
            var t0 = perf_counter_ns()
            for it in range(ITERS):
                var qb = q_tensors(ctx, q_d, it % NBUF)
                lds_dispatch[i8](1, ctx, Aq, Ad, An, qb[0], qb[1], C, m)
            ctx.synchronize()
            meds4.append(Float64(perf_counter_ns() - t0) / 1e3 / ITERS)
        timed(ctx, m, "mmq", lds_cfg_name(1), lds_grid(1, m), meds4)
