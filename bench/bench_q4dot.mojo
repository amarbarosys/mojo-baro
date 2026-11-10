"""Dense Q4_0 skinny GEMM at m rows: amar_matmul_skinny_q4rowb (f32 dequant, the engine's) vs
amar_matmul_skinny_q4dot (int8 activations x nibbles through sdot4). Real Qwythos Q4_0 ffn_gate
weights from .work/e11b-qwythos-pack (blk.0..7, 28.3 MB each, rotated so the 96 MB Infinity Cache
never holds the arm under test). Correctness: q4dot vs q4rowb on blk.0 at every MR (rel 1e-2).
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu import WARP_SIZE
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, TensorLayout, row_major
from elementwise import amar_quantize_q8_rows
from matmul_skinny import amar_matmul_skinny_q4rowb, amar_matmul_skinny_q4dot, ROW_WAVES, ROW_THREADS
from tio import median

comptime N = 12288
comptime K = 4096
comptime NT = 8
comptime MMAX = 4
comptime ROUNDS = 11
comptime QB = N * K // 2
comptime SB = N * (K // 32) * 2
comptime TB = QB + SB
comptime bf16 = DType.bfloat16
comptime f16 = DType.float16
comptime f32 = DType.float32
comptime i8 = DType.int8
comptime u8 = DType.uint8
comptime a_layout = row_major[MMAX, K]()
comptime aq_layout = row_major[MMAX, K]()
comptime as_layout = row_major[MMAX, K // 32]()
comptime q_layout = row_major[N, K // 2]()
comptime s_layout = row_major[N, K // 32]()
comptime p_layout = row_major[1, MMAX, N]()
comptime PACK = ".work/e11b-qwythos-pack/"


def pack_offset(name: String) raises -> Int:
    with open(PACK + "index.txt", "r") as f:
        for line in f.read().splitlines():
            var parts = line.split(" ")
            if String(parts[0]) == name:
                if String(parts[1]) != "q4":
                    raise Error("not q4: " + name)
                return atol(String(parts[2]))
    raise Error("missing " + name)


def load_at(mut dst: HostBuffer[u8], dst_off: Int, path: String, size: Int, skip: Int) raises:
    with open(path, "r") as f:
        _ = f.seek(skip)
        var data = f.read_bytes(size)
        if len(data) < size:
            raise Error("short read " + path)
        for i in range(size):
            dst[dst_off + i] = data[i]


def run_old[MR: Int](ctx: DeviceContext, mut a_d: DeviceBuffer[bf16], mut w_d: DeviceBuffer[u8], t: Int, mut p_d: DeviceBuffer[f32]) raises:
    var qb = DeviceBuffer[u8](ctx, w_d.unsafe_ptr().unsafe_offset(t * TB), QB, owning=False)
    var sb = DeviceBuffer[f16](ctx, w_d.unsafe_ptr().unsafe_offset(t * TB + QB).unsafe_bitcast[Float16](), N * (K // 32), owning=False)
    ctx.enqueue_function[amar_matmul_skinny_q4rowb[2, MR, type_of(a_layout), type_of(q_layout), type_of(s_layout), type_of(p_layout)]](
        TileTensor(a_d, a_layout), TileTensor(qb, q_layout), TileTensor(sb, s_layout), TileTensor(p_d, p_layout), Int32(MR), Int32(N), Int32(K),
        grid_dim=ceildiv(N, ROW_WAVES), block_dim=ROW_THREADS)


def run_new[MR: Int](ctx: DeviceContext, mut aq_d: DeviceBuffer[i8], mut as_d: DeviceBuffer[f16], mut w_d: DeviceBuffer[u8], t: Int, mut p_d: DeviceBuffer[f32]) raises:
    var qb = DeviceBuffer[u8](ctx, w_d.unsafe_ptr().unsafe_offset(t * TB), QB, owning=False)
    var sb = DeviceBuffer[f16](ctx, w_d.unsafe_ptr().unsafe_offset(t * TB + QB).unsafe_bitcast[Float16](), N * (K // 32), owning=False)
    ctx.enqueue_function[amar_matmul_skinny_q4dot[2, MR, type_of(aq_layout), type_of(as_layout), type_of(q_layout), type_of(s_layout), type_of(p_layout)]](
        TileTensor(aq_d, aq_layout), TileTensor(as_d, as_layout), TileTensor(qb, q_layout), TileTensor(sb, s_layout), TileTensor(p_d, p_layout), Int32(MR), Int32(N), Int32(K),
        grid_dim=ceildiv(N, ROW_WAVES), block_dim=ROW_THREADS)


def arm[MR: Int](ctx: DeviceContext, mut a_d: DeviceBuffer[bf16], mut aq_d: DeviceBuffer[i8], mut as_d: DeviceBuffer[f16], mut w_d: DeviceBuffer[u8],
                 mut po_d: DeviceBuffer[f32], mut pn_d: DeviceBuffer[f32]) raises:
    run_old[MR](ctx, a_d, w_d, 0, po_d)
    run_new[MR](ctx, aq_d, as_d, w_d, 0, pn_d)
    ctx.synchronize()
    var ho = ctx.enqueue_create_host_buffer[f32](MMAX * N)
    var hn = ctx.enqueue_create_host_buffer[f32](MMAX * N)
    ctx.enqueue_copy(dst_buf=ho, src_buf=po_d)
    ctx.enqueue_copy(dst_buf=hn, src_buf=pn_d)
    ctx.synchronize()
    var worst: Float64 = 0
    var scale: Float64 = 0
    for i in range(MR * N):
        var r = Float64(ho[i])
        if abs(r) > scale:
            scale = abs(r)
        var d = abs(Float64(hn[i]) - r)
        if d > worst:
            worst = d
    var olds = List[Float64]()
    var news = List[Float64]()
    for _ in range(3):
        for t in range(NT):
            run_old[MR](ctx, a_d, w_d, t, po_d)
            run_new[MR](ctx, aq_d, as_d, w_d, t, pn_d)
    ctx.synchronize()
    for r in range(ROUNDS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for t in range(NT):
            run_old[MR](ctx, a_d, w_d, t, po_d)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        for t in range(NT):
            run_new[MR](ctx, aq_d, as_d, w_d, t, pn_d)
        ctx.synchronize()
        var t2 = perf_counter_ns()
        olds.append(Float64(t1 - t0) / 1e3 / NT)
        news.append(Float64(t2 - t1) / 1e3 / NT)
    var mo = median(olds.copy())
    var mn = median(news.copy())
    print("MR", MR, ": q4rowb", mo, "us (", Float64(TB) / mo / 1e3, "GB/s )  q4dot", mn, "us (", Float64(TB) / mn / 1e3, "GB/s )  ratio", mo / mn, " max|diff|/max|ref|", worst / scale)


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var w_h = ctx.enqueue_create_host_buffer[u8](NT * TB)
    var a_h = ctx.enqueue_create_host_buffer[bf16](MMAX * K)
    ctx.synchronize()
    for t in range(NT):
        var off = pack_offset("blk." + String(t) + ".ffn_gate.weight")
        load_at(w_h, t * TB, PACK + "pack.bin", TB, off)
    var seed = 12345
    for i in range(MMAX * K):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        a_h[i] = (Float32(seed % 2001) / 1000.0 - 1.0).cast[bf16]()
    var w_d = ctx.enqueue_create_buffer[u8](NT * TB)
    var a_d = ctx.enqueue_create_buffer[bf16](MMAX * K)
    var aq_d = ctx.enqueue_create_buffer[i8](MMAX * K)
    var as_d = ctx.enqueue_create_buffer[f16](MMAX * (K // 32))
    var po_d = ctx.enqueue_create_buffer[f32](MMAX * N)
    var pn_d = ctx.enqueue_create_buffer[f32](MMAX * N)
    ctx.enqueue_copy(dst_buf=w_d, src_buf=w_h)
    ctx.enqueue_copy(dst_buf=a_d, src_buf=a_h)
    ctx.enqueue_function[amar_quantize_q8_rows[type_of(a_layout), type_of(aq_layout), type_of(as_layout)]](
        TileTensor(a_d, a_layout), TileTensor(aq_d, aq_layout), TileTensor(as_d, as_layout), Int32(MMAX), Int32(K), grid_dim=(MMAX, K // 32), block_dim=WARP_SIZE)
    ctx.synchronize()
    print("bench_q4dot: Qwythos Q4_0 ffn_gate", N, "x", K, ",", NT, "tensors rotated,", ROUNDS, "rounds; bytes per call", TB)
    arm[1](ctx, a_d, aq_d, as_d, w_d, po_d, pn_d)
    arm[2](ctx, a_d, aq_d, as_d, w_d, po_d, pn_d)
    arm[3](ctx, a_d, aq_d, as_d, w_d, po_d, pn_d)
    arm[4](ctx, a_d, aq_d, as_d, w_d, po_d, pn_d)
