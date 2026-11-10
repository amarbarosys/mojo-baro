"""Prefill GEMM tile sweep, one weight shape per build (bench/tile-tune-protocol.md).

Build per shape: -D TS_N=<rows> -D TS_K=<cols> -D TS_TENSOR=<pack tensor name>.
Six compiled variants (4 wmma tiles, 2 bf16-lds tiles) at every m; the arm
"rule" is the registry.mojo dispatch (gemm_prefill_q4). Weight copies rotate
over >= 256 MB so the stream is cold (96 MB Infinity Cache). Gate before any
timing: every variant bit-exact against the rule arm at that m.
Output rows: m variant tile grid us_med us_min us_max.
"""
from std.math import ceildiv
from std.sys import has_accelerator, get_defined_int, get_defined_string
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, TensorLayout, row_major

from matmul_prefill import amar_matmul_prefill_q4, PF_THREADS
from matmul_prefill_lds import amar_matmul_prefill_lds, LDS_THREADS
from tio import median

comptime N = get_defined_int["TS_N", 4096]()
comptime K = get_defined_int["TS_K", 4096]()
comptime TENSOR = get_defined_string["TS_TENSOR", "blk.3.attn_output.weight"]()
comptime MMAX = 1024
comptime Q4BYTES = N * K // 2 + N * (K // 32) * 2
comptime NBUF = max(8, ceildiv(256 * 1024 * 1024, Q4BYTES))
comptime ITERS = 16
comptime REPEATS = 5
comptime NVAR = 6
comptime PF_LDS_MIN = 128

comptime a_layout = row_major[MMAX, K]()
comptime c_layout = row_major[MMAX, N]()
comptime q4_layout = row_major[N, K // 2]()
comptime s_layout = row_major[N, K // 32]()

comptime bf16 = DType.bfloat16
comptime f16 = DType.float16
comptime f32 = DType.float32
comptime u8 = DType.uint8

comptime AT = TileTensor[bf16, type_of(a_layout), MutAnyOrigin]
comptime QT = TileTensor[u8, type_of(q4_layout), MutAnyOrigin]
comptime ST = TileTensor[f16, type_of(s_layout), MutAnyOrigin]
comptime CT = TileTensor[f32, type_of(c_layout), MutAnyOrigin]


def pack_offset(name: String) raises -> Int:
    with open(".work/engine-pack-q4/index.txt", "r") as f:
        for line in f.read().splitlines():
            var parts = line.split(" ")
            if parts[0] == name:
                if String(parts[1]) != "q4":
                    raise Error("not q4: " + name)
                if Int(parts[3]) != N * K:
                    raise Error(name + " has " + String(parts[3]) + " elems, build says " + String(N * K))
                return Int(parts[2])
    raise Error("missing " + name)


def lcg(mut st: UInt64) -> Float32:
    st = st * 6364136223846793005 + 1442695040888963407
    return Float32(Int((st >> 40) & 0xFFFF)) / 32768.0 - 1.0


def q_tensors(ctx: DeviceContext, q_dev: DeviceBuffer[u8], b: Int) raises -> Tuple[QT, ST]:
    var o = b * Q4BYTES
    var qb = DeviceBuffer[u8](ctx, q_dev.unsafe_ptr().unsafe_offset(o), N * K // 2, owning=False)
    var sb = DeviceBuffer[f16](ctx, q_dev.unsafe_ptr().unsafe_offset(o + N * K // 2).unsafe_bitcast[Float16](), N * (K // 32), owning=False)
    return (rebind[QT](TileTensor(qb, q4_layout)), rebind[ST](TileTensor(sb, s_layout)))


def wmma[WTM: Int, WTN: Int, WAVES_M: Int](ctx: DeviceContext, A: AT, Q: QT, S: ST, C: CT, m: Int) raises:
    comptime BM = WAVES_M * WTM * 16
    comptime BN = (8 // WAVES_M) * WTN * 16
    ctx.enqueue_function[amar_matmul_prefill_q4[WTM, WTN, WAVES_M, False, type_of(a_layout), type_of(q4_layout), type_of(s_layout), type_of(c_layout)]](
        A, Q, S, C, Int32(m), Int32(N), Int32(K), grid_dim=(ceildiv(N, BN), ceildiv(m, BM)), block_dim=PF_THREADS)


def lds[WM: Int, WN: Int, TM: Int, TN: Int](ctx: DeviceContext, A: AT, Q: QT, S: ST, C: CT, m: Int) raises:
    comptime BM = WM * TM * 16
    comptime BN = WN * TN * 16
    ctx.enqueue_function[amar_matmul_prefill_lds[u8, WM, WN, TM, TN, False, type_of(a_layout), type_of(q4_layout), type_of(s_layout), type_of(c_layout)]](
        A, Q, S, C, Int32(m), Int32(N), Int32(K), grid_dim=(ceildiv(N, BN), ceildiv(m, BM)), block_dim=LDS_THREADS)


def launch(v: Int, ctx: DeviceContext, A: AT, Q: QT, S: ST, C: CT, m: Int) raises:
    if v == 0:
        wmma[1, 2, 1](ctx, A, Q, S, C, m)
    elif v == 1:
        wmma[2, 2, 1](ctx, A, Q, S, C, m)
    elif v == 2:
        wmma[2, 2, 2](ctx, A, Q, S, C, m)
    elif v == 3:
        wmma[4, 2, 2](ctx, A, Q, S, C, m)
    elif v == 4:
        lds[2, 4, 2, 2](ctx, A, Q, S, C, m)
    else:
        lds[4, 2, 2, 4](ctx, A, Q, S, C, m)


def rule(m: Int) -> Int:
    if m <= 64:
        return 2
    if m <= PF_LDS_MIN:
        return 3
    return 5


def vname(v: Int) -> String:
    var names: List[String] = ["wmma16x256", "wmma32x256", "wmma64x128", "wmma128x128", "lds64x128", "lds128x128"]
    return names[v]


def vtile(v: Int) -> Tuple[Int, Int]:
    var bm: List[Int] = [16, 32, 64, 128, 64, 128]
    var bn: List[Int] = [256, 256, 128, 128, 128, 128]
    return (bm[v], bn[v])


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var q_h = ctx.enqueue_create_host_buffer[u8](Q4BYTES)
    var a_h = ctx.enqueue_create_host_buffer[bf16](MMAX * K)
    var ref_h = ctx.enqueue_create_host_buffer[f32](MMAX * N)
    var got_h = ctx.enqueue_create_host_buffer[f32](MMAX * N)
    ctx.synchronize()
    with open(".work/engine-pack-q4/pack.bin", "r") as f:
        _ = f.seek(pack_offset(TENSOR))
        var data = f.read_bytes(Q4BYTES)
        if len(data) < Q4BYTES:
            raise Error("short read " + TENSOR)
        for i in range(Q4BYTES):
            q_h[i] = data[i]
    var st = UInt64(99)
    for i in range(MMAX * K):
        a_h[i] = Scalar[bf16](lcg(st))
    var q_d = ctx.enqueue_create_buffer[u8](NBUF * Q4BYTES)
    var a_d = ctx.enqueue_create_buffer[bf16](MMAX * K)
    var c_d = ctx.enqueue_create_buffer[f32](MMAX * N)
    for b in range(NBUF):
        var dst = DeviceBuffer[u8](ctx, q_d.unsafe_ptr().unsafe_offset(b * Q4BYTES), Q4BYTES, owning=False)
        ctx.enqueue_copy(dst_buf=dst, src_buf=q_h)
    ctx.enqueue_copy(dst_buf=a_d, src_buf=a_h)
    ctx.synchronize()
    var A = rebind[AT](TileTensor(a_d, a_layout))
    var C = rebind[CT](TileTensor(c_d, c_layout))
    print("TS shape", TENSOR, "N", N, "K", K, "q4_bytes", Q4BYTES, "NBUF", NBUF, "stream_MB", NBUF * Q4BYTES // (1024 * 1024), "ITERS", ITERS, "REPEATS", REPEATS, "PF_LDS_MIN", PF_LDS_MIN)
    print("m variant tile grid us_med us_min us_max")
    var ms: List[Int] = [16, 32, 64, 128, 256, 512, 1024]
    for mi in range(len(ms)):
        var m = ms[mi]
        var q0 = q_tensors(ctx, q_d, 0)
        launch(rule(m), ctx, A, q0[0], q0[1], C, m)
        ctx.enqueue_copy(dst_buf=ref_h, src_buf=c_d)
        ctx.synchronize()
        for v in range(NVAR):
            launch(v, ctx, A, q0[0], q0[1], C, m)
            ctx.enqueue_copy(dst_buf=got_h, src_buf=c_d)
            ctx.synchronize()
            for i in range(m * N):
                if got_h[i] != ref_h[i]:
                    raise Error("FAIL exact: " + vname(v) + " vs rule at m=" + String(m) + " i=" + String(i))
        for v in range(NVAR + 1):
            var var_id = rule(m) if v == NVAR else v
            var meds = List[Float64]()
            for _ in range(REPEATS):
                ctx.synchronize()
                var t0 = perf_counter_ns()
                for it in range(ITERS):
                    var qb = q_tensors(ctx, q_d, it % NBUF)
                    launch(var_id, ctx, A, qb[0], qb[1], C, m)
                ctx.synchronize()
                meds.append(Float64(perf_counter_ns() - t0) / 1e3 / ITERS)
            var mn = meds[0]
            var mx = meds[0]
            for i in range(len(meds)):
                mn = min(mn, meds[i])
                mx = max(mx, meds[i])
            var t = vtile(var_id)
            var label = "rule=" + vname(var_id) if v == NVAR else vname(v)
            print(m, label, String(t[0]) + "x" + String(t[1]), String(ceildiv(N, t[1])) + "x" + String(ceildiv(m, t[0])), median(meds.copy()), mn, mx)
