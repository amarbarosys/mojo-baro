"""Row-scaling receipt (P5) and limiter ablation for the 27B Turbo window GEMV: amar_matmul_skinny_q4rowb at the two
FFN shapes of the q4 pack (gate 17408 x 5120, down 5120 x 17408), real blk.0..7 weights rotated over 8 tensors (400 MB per
shape, above twice the Infinity Cache), median of 11 rounds of 8 launches after 3 warm passes. Arms per shape: FULL at
MR 1, 2, 3, 4; MR 3 NOTRAFFIC (every row reads A row 0) and NOFMA (rows 1, 2 load A, only row 0 accumulates) through a
bench-local copy of q4_dot_blocks with an ABL knob; the real kernel at UNROLL 4 and 8 (depth, bit-identical by
construction); amar_turbo_rows (ABL 10) at MR 3 UNROLL 2 / 4 and MR 1, 2, 4, byte-checked against q4rowb on tensor 0. The MR 3 FULL arm is checked on tensor 0 against an fp64 host dot over the dequantized rows.
Frozen predictions: docs/lanes/turbokernel-report.md section 3.  env BARO_PACK = the q4 pack dir.
"""
from std.math import ceildiv, fma
from std.memory import bitcast
from std.sys import has_accelerator
from std.os import getenv
from std.sys import get_defined_int
from std.time import perf_counter_ns
from std.utils import StaticTuple
from max.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from max.gpu.primitives import warp
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, TensorLayout, row_major
from matmul_skinny import amar_matmul_skinny_q4rowb, bf16x16_to_f32, ROW_WAVES, ROW_THREADS
from matmul import dtype
from turbo_rows import amar_turbo_rows, amar_turbo_rows_q8, amar_turbo_rows_dot2
from elementwise import amar_quantize_q8_rows
from max.gpu import WARP_SIZE
from fewrow import amar_fewrow_wmma, amar_fewrow_iu8, amar_q8_cols, FW_THREADS

comptime NT = 8
comptime H_D = get_defined_int["BARO_H", 5120]()
comptime FFN_D = get_defined_int["BARO_FFN", 17408]()
comptime MMAX = 8
comptime ROUNDS = 11
comptime bf16 = DType.bfloat16
comptime f16 = DType.float16
comptime f32 = DType.float32
comptime u8 = DType.uint8
comptime u32 = DType.uint32


def q4_dot_abl[
    MR: Int, UNROLL: Int, ABL: Int, ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    Q: TileTensor[u8, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    row: Int, lane: Int, kb0: Int, kb1: Int, M: Int,
) -> Array[Float32, MR]:
    comptime QV = 16
    comptime STEP = WARP_SIZE
    var Qb = Q.vectorize[1, QV]()
    var Av = A.vectorize[1, QV]()
    var acc = Array[SIMD[dtype, QV], MR](fill=SIMD[dtype, QV](0))
    var sink = SIMD[u32, QV](0)
    var kk = kb0
    while kk + UNROLL * STEP <= kb1:
        var bytes_u = Array[SIMD[u8, QV], UNROLL](uninitialized=True)
        var ds = Array[Scalar[f16], UNROLL](uninitialized=True)
        comptime for u in range(UNROLL):
            var blk = kk + u * STEP + lane
            bytes_u[u] = rebind[SIMD[u8, QV]](Qb[row, blk])
            ds[u] = rebind[Scalar[f16]](S[row, blk])
        comptime for u in range(UNROLL):
            var blk = kk + u * STEP + lane
            var d = ds[u].cast[dtype]()
            var w32 = bitcast[u32, 4](bytes_u[u])
            var lo = bitcast[u8, QV](w32 & 0x0F0F0F0F).cast[dtype]()
            var hi = bitcast[u8, QV]((w32 >> 4) & 0x0F0F0F0F).cast[dtype]()
            var dv = SIMD[dtype, QV](d)
            var m8d = SIMD[dtype, QV](d * -8)
            var wlo = fma(lo, dv, m8d)
            var whi = fma(hi, dv, m8d)
            comptime for r in range(MR):
                if r < M:
                    comptime ar = 0 if ABL == 1 else r
                    var a_lo_b = rebind[SIMD[bf16, QV]](Av[ar, blk * 2])
                    var a_hi_b = rebind[SIMD[bf16, QV]](Av[ar, blk * 2 + 1])
                    comptime if ABL == 2 and r > 0:
                        sink ^= bitcast[u32, QV](a_lo_b.cast[f32]()) ^ bitcast[u32, QV](a_hi_b.cast[f32]())
                    else:
                        acc[r] = fma(wlo, bf16x16_to_f32(a_lo_b), fma(whi, bf16x16_to_f32(a_hi_b), acc[r]))
        kk += UNROLL * STEP
    var out = Array[Float32, MR](fill=0)
    comptime for r in range(MR):
        if r < M:
            comptime if ABL == 2 and r > 0:
                out[r] = bitcast[f32](sink.reduce_or())
            else:
                out[r] = warp.sum(acc[r].reduce_add())
    return out^


@__llvm_metadata(`rocdl.flat_work_group_size`=StaticTuple[Int32, 1](Int32(ROW_THREADS)))
def q4rowb_abl[
    UNROLL: Int, MR: Int, ABL: Int,
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout, PLayout: TensorLayout,
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    Q: TileTensor[u8, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32, n: Int32, k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2 and S.flat_rank == 2 and Cp.flat_rank == 3
    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= N:
        return
    var t = q4_dot_abl[MR, UNROLL, ABL](A, Q, S, row, lane, 0, K // 32, M)
    comptime for r in range(MR):
        if r < M:
            if lane == 0:
                Cp[0, r, row] = rebind[Cp.ElementType](t[r])


def median(mut v: List[Float64]) -> Float64:
    sort(v)
    return v[len(v) // 2]


def pack_offset(pack: String, name: String) raises -> Int:
    with open(pack + "/index.txt", "r") as f:
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


struct Shape[N: Int, K: Int]:
    comptime QB = Self.N * Self.K // 2
    comptime SB = Self.N * (Self.K // 32) * 2
    comptime TB = Self.QB + Self.SB
    comptime a_layout = row_major[MMAX, Self.K]()
    comptime q_layout = row_major[Self.N, Self.K // 2]()
    comptime s_layout = row_major[Self.N, Self.K // 32]()
    comptime p_layout = row_major[1, MMAX, Self.N]()


def launch[
    N: Int, K: Int, UNROLL: Int, MR: Int, ABL: Int
](ctx: DeviceContext, a_d: DeviceBuffer[bf16], w_d: DeviceBuffer[u8], t: Int, p_d: DeviceBuffer[f32]) raises:
    comptime Sh = Shape[N, K]
    var qb = DeviceBuffer[u8](ctx, w_d.unsafe_ptr().unsafe_offset(t * Sh.TB), Sh.QB, owning=False)
    var sb = DeviceBuffer[f16](ctx, w_d.unsafe_ptr().unsafe_offset(t * Sh.TB + Sh.QB).unsafe_bitcast[Float16](), N * (K // 32), owning=False)
    var A = TileTensor(a_d, Sh.a_layout)
    comptime TRANS = ABL == 22 or (ABL >= 26 and ABL <= 29) or ABL == 32
    comptime QL = row_major[K // 32 if TRANS else N, N * 16 if TRANS else K // 2]()
    comptime SL = row_major[K // 32 if TRANS else N, N if TRANS else K // 32]()
    var Q = TileTensor(qb, QL)
    var S = TileTensor(sb, SL)
    var P = TileTensor(p_d, Sh.p_layout)
    comptime if ABL == 30 or ABL == 31:
        # round A: int8 A rows. 31 = PRE, timed with its amar_quantize_q8_rows launch. Scratch sits after P in p_d.
        comptime aq_l = row_major[MMAX, K]()
        comptime as_l = row_major[MMAX, K // 32]()
        var base_p = p_d.unsafe_ptr().unsafe_offset(MMAX * N)
        var Aq = TileTensor(DeviceBuffer[DType.int8](ctx, base_p.unsafe_bitcast[Int8](), MMAX * K, owning=False), aq_l)
        var As = TileTensor(DeviceBuffer[f16](ctx, base_p.unsafe_offset(MMAX * K // 4).unsafe_bitcast[Float16](), MMAX * (K // 32), owning=False), as_l)
        comptime if ABL == 31:
            ctx.enqueue_function[amar_quantize_q8_rows[type_of(Sh.a_layout), type_of(aq_l), type_of(as_l)]](
                A, Aq, As, Int32(MR), Int32(K), grid_dim=(MR, K // 32), block_dim=WARP_SIZE)
        ctx.enqueue_function[amar_turbo_rows_q8[N, K, MR, UNROLL, ABL == 31, type_of(Sh.a_layout), type_of(aq_l), type_of(as_l), type_of(Sh.q_layout), type_of(Sh.s_layout), type_of(Sh.p_layout)]](
            A, Aq, As, Q, S, P, Int32(MR), grid_dim=N // ROW_WAVES, block_dim=ROW_THREADS)
    elif ABL == 50:
        ctx.enqueue_function[amar_turbo_rows_dot2[N, K, MR, UNROLL, type_of(Sh.a_layout), type_of(Sh.q_layout), type_of(Sh.s_layout), type_of(Sh.p_layout)]](
            A, Q, S, P, Int32(MR), grid_dim=N // ROW_WAVES, block_dim=ROW_THREADS)
    elif ABL == 40:
        # int8 WMMA: amar_q8_cols (timed with it) then amar_fewrow_iu8. Scratch after P: Aq[16, K] int8, Ast[K/32, 16] f32.
        comptime aq_l = row_major[16, K]()
        comptime as_l = row_major[K // 32, 16]()
        var base_p = p_d.unsafe_ptr().unsafe_offset(MMAX * N)
        var Aq = TileTensor(DeviceBuffer[DType.int8](ctx, base_p.unsafe_bitcast[Int8](), 16 * K, owning=False), aq_l)
        var As = TileTensor(DeviceBuffer[f32](ctx, base_p.unsafe_offset(4 * K), K // 2, owning=False), as_l)
        ctx.enqueue_function[amar_q8_cols[type_of(Sh.a_layout), type_of(aq_l), type_of(as_l)]](
            A, Aq, As, Int32(MR), grid_dim=(16, K // 32), block_dim=WARP_SIZE)
        ctx.enqueue_function[amar_fewrow_iu8[N, K, UNROLL, type_of(aq_l), type_of(as_l), type_of(QL), type_of(SL), type_of(Sh.p_layout)]](
            Aq, As, Q, S, P, Int32(MR), grid_dim=N // 16, block_dim=FW_THREADS)
    elif ABL >= 20 and ABL <= 32:
        comptime NW = 4 if ABL == 23 or ABL == 26 else (2 if ABL == 24 or ABL == 27 else (1 if ABL == 25 or ABL == 28 else 8))
        comptime MODE = 4 if ABL == 32 else (3 if ABL == 29 else (2 if TRANS else (1 if ABL == 21 else 0)))
        ctx.enqueue_function[amar_fewrow_wmma[N, K, UNROLL, type_of(Sh.a_layout), type_of(QL), type_of(SL), type_of(Sh.p_layout), MODE, NW]](
            A, Q, S, P, Int32(MR), grid_dim=N // 16, block_dim=NW * WARP_SIZE)
    elif ABL >= 10:
        ctx.enqueue_function[amar_turbo_rows[N, K, MR, UNROLL, type_of(Sh.a_layout), type_of(Sh.q_layout), type_of(Sh.s_layout), type_of(Sh.p_layout)]](
            A, Q, S, P, Int32(MR), grid_dim=N // ROW_WAVES, block_dim=ROW_THREADS)
    elif ABL == 0:
        ctx.enqueue_function[amar_matmul_skinny_q4rowb[UNROLL, MR, type_of(Sh.a_layout), type_of(Sh.q_layout), type_of(Sh.s_layout), type_of(Sh.p_layout)]](
            A, Q, S, P, Int32(MR), Int32(N), Int32(K), grid_dim=ceildiv(N, ROW_WAVES), block_dim=ROW_THREADS)
    else:
        ctx.enqueue_function[q4rowb_abl[UNROLL, MR, ABL, type_of(Sh.a_layout), type_of(Sh.q_layout), type_of(Sh.s_layout), type_of(Sh.p_layout)]](
            A, Q, S, P, Int32(MR), Int32(N), Int32(K), grid_dim=ceildiv(N, ROW_WAVES), block_dim=ROW_THREADS)


def arm[
    N: Int, K: Int, UNROLL: Int, MR: Int, ABL: Int
](ctx: DeviceContext, name: String, a_d: DeviceBuffer[bf16], w_d: DeviceBuffer[u8], p_d: DeviceBuffer[f32], base: Float64) raises -> Float64:
    comptime Sh = Shape[N, K]
    for _ in range(3):
        for t in range(NT):
            launch[N, K, UNROLL, MR, ABL](ctx, a_d, w_d, t, p_d)
    ctx.synchronize()
    var us = List[Float64]()
    for _ in range(ROUNDS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for t in range(NT):
            launch[N, K, UNROLL, MR, ABL](ctx, a_d, w_d, t, p_d)
        ctx.synchronize()
        us.append(Float64(perf_counter_ns() - t0) / 1e3 / NT)
    var m = median(us)
    var lo = us[0]
    var hi = us[len(us) - 1]
    print(name, " MR", MR, " UNROLL", UNROLL, " ABL", ABL, ": ", m, "us  min", lo, " max", hi, " GB/s", Float64(Sh.TB) / m / 1e3, " ratio_to_m1", m / base if base > 0 else 1.0)
    return m


def host_check[N: Int, K: Int, U: Int = 2, ABL: Int = 0, MR: Int = 3](ctx: DeviceContext, w_h: HostBuffer[u8], a_h: HostBuffer[bf16], a_d: DeviceBuffer[bf16], w_d: DeviceBuffer[u8], p_d: DeviceBuffer[f32]) raises:
    comptime Sh = Shape[N, K]
    launch[N, K, U, MR, ABL](ctx, a_d, w_d, 0, p_d)
    ctx.synchronize()
    var p_h = ctx.enqueue_create_host_buffer[f32](MMAX * N)
    ctx.enqueue_copy(dst_buf=p_h, src_buf=p_d)
    ctx.synchronize()
    var worst: Float64 = 0
    var emax: Float64 = 0
    var amax: Float64 = 0
    var rows = 0
    var r0 = 0
    while r0 < N:
        for r in range(MR):
            var acc: Float64 = 0
            for b in range(K // 32):
                var d = Float64(bitcast[f16](UInt16(w_h[Sh.QB + (r0 * (K // 32) + b) * 2]) | (UInt16(w_h[Sh.QB + (r0 * (K // 32) + b) * 2 + 1]) << 8)).cast[f32]())
                for j in range(16):
                    var byte = Int(w_h[r0 * (K // 2) + b * 16 + j])
                    var wlo = (Float64(byte & 15) - 8) * d
                    var whi = (Float64(byte >> 4) - 8) * d
                    acc += wlo * Float64(a_h[r * K + b * 32 + j].cast[f32]()) + whi * Float64(a_h[r * K + b * 32 + 16 + j].cast[f32]())
            var got = Float64(p_h[r * N + r0])
            var rel = abs(got - acc) / max(abs(acc), 1e-3)
            if rel > worst:
                worst = rel
            emax = max(emax, abs(got - acc))
            amax = max(amax, abs(acc))
            if r0 == 0 and r < 3:
                print("  sample row", r, "got", got, "ref", acc)
        rows += 1
        r0 += 97
    print("host check MR", MR, "ABL", ABL, "vs fp64:", rows, "rows, max rel", worst, " max|err|/max|ref|", emax / amax)
    if (ABL == 40 and emax / amax > 2e-2) or (ABL != 40 and worst > 2e-3):
        raise Error("FAIL host check: max rel " + String(worst))


def byte_check[N: Int, K: Int](ctx: DeviceContext, a_d: DeviceBuffer[bf16], w_d: DeviceBuffer[u8], p_d: DeviceBuffer[f32]) raises:
    var pr_h = ctx.enqueue_create_host_buffer[f32](MMAX * N)
    var pt_h = ctx.enqueue_create_host_buffer[f32](MMAX * N)
    launch[N, K, 2, 3, 0](ctx, a_d, w_d, 0, p_d)
    ctx.enqueue_copy(dst_buf=pr_h, src_buf=p_d)
    ctx.synchronize()
    var bad = 0
    comptime for v in range(2):
        comptime if v == 0:
            launch[N, K, 2, 3, 10](ctx, a_d, w_d, 0, p_d)
        else:
            launch[N, K, 4, 3, 10](ctx, a_d, w_d, 0, p_d)
        ctx.enqueue_copy(dst_buf=pt_h, src_buf=p_d)
        ctx.synchronize()
        for i in range(3 * N):
            if bitcast[u32](pr_h[i]) != bitcast[u32](pt_h[i]):
                bad += 1
    print("byte check turbo vs q4rowb, tensor 0, m 3, two variants:", 2 * 3 * N, "words, mismatches", bad)
    if bad != 0:
        raise Error("FAIL byte check: turbo kernel differs from q4rowb")


def shape_round[N: Int, K: Int](ctx: DeviceContext, pack: String, tensor: String) raises:
    comptime Sh = Shape[N, K]
    var w_h = ctx.enqueue_create_host_buffer[u8](NT * Sh.TB)
    var a_h = ctx.enqueue_create_host_buffer[bf16](MMAX * K)
    ctx.synchronize()
    for t in range(NT):
        var off = pack_offset(pack, "blk." + String(t) + "." + tensor + ".weight")
        load_at(w_h, t * Sh.TB, pack + "/pack.bin", Sh.TB, off)
    var seed = 12345
    for i in range(MMAX * K):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        a_h[i] = (Float32(seed % 2001) / 1000.0 - 1.0).cast[bf16]()
    var w_d = ctx.enqueue_create_buffer[u8](NT * Sh.TB)
    var a_d = ctx.enqueue_create_buffer[bf16](MMAX * K)
    var p_d = ctx.enqueue_create_buffer[f32](MMAX * N + MMAX * K)  # tail: round A int8 scratch
    ctx.enqueue_copy(dst_buf=w_d, src_buf=w_h)
    ctx.enqueue_copy(dst_buf=a_d, src_buf=a_h)
    ctx.synchronize()
    print("shape", tensor, N, "x", K, " bytes per call", Sh.TB, " tensors", NT, " rounds", ROUNDS)
    comptime if get_defined_int["VERIFY8", 0]() != 0:
        # One-change ablations: identical math with transposed storage, or centered nibbles.
        var wt_h = ctx.enqueue_create_host_buffer[u8](NT * Sh.TB)
        var wt_d = ctx.enqueue_create_buffer[u8](NT * Sh.TB)
        ctx.synchronize()
        for t in range(NT):
            var off = t * Sh.TB
            for b in range(K // 32):
                for n in range(N):
                    for j in range(16):
                        wt_h[off + (b * N + n) * 16 + j] = w_h[off + (n * (K // 32) + b) * 16 + j]
                    for j in range(2):
                        wt_h[off + Sh.QB + (b * N + n) * 2 + j] = w_h[off + Sh.QB + (n * (K // 32) + b) * 2 + j]
        ctx.enqueue_copy(dst_ptr=wt_d.unsafe_ptr(), src_ptr=wt_h.unsafe_ptr(), size=NT * Sh.TB)
        ctx.synchronize()
        host_check[N, K, 4, 20, 8](ctx, w_h, a_h, a_d, w_d, p_d)
        host_check[N, K, 4, 21, 8](ctx, w_h, a_h, a_d, w_d, p_d)
        host_check[N, K, 4, 22, 8](ctx, w_h, a_h, a_d, wt_d, p_d)
        host_check[N, K, 4, 23, 8](ctx, w_h, a_h, a_d, w_d, p_d)
        host_check[N, K, 4, 24, 8](ctx, w_h, a_h, a_d, w_d, p_d)
        host_check[N, K, 4, 25, 8](ctx, w_h, a_h, a_d, w_d, p_d)
        host_check[N, K, 4, 26, 8](ctx, w_h, a_h, a_d, wt_d, p_d)
        host_check[N, K, 4, 27, 8](ctx, w_h, a_h, a_d, wt_d, p_d)
        host_check[N, K, 4, 28, 8](ctx, w_h, a_h, a_d, wt_d, p_d)
        host_check[N, K, 4, 29, 8](ctx, w_h, a_h, a_d, wt_d, p_d)
        host_check[N, K, 4, 32, 8](ctx, w_h, a_h, a_d, wt_d, p_d)
        host_check[N, K, 4, 40, 8](ctx, w_h, a_h, a_d, w_d, p_d)
        host_check[N, K, 2, 50, 3](ctx, w_h, a_h, a_d, w_d, p_d)
        var expected = ctx.enqueue_create_host_buffer[f32](MMAX * N)
        var actual = ctx.enqueue_create_host_buffer[f32](MMAX * N)
        launch[N, K, 4, 8, 20](ctx, a_d, w_d, 0, p_d)
        ctx.enqueue_copy(dst_ptr=expected.unsafe_ptr(), src_ptr=p_d.unsafe_ptr(), size=MMAX * N)
        launch[N, K, 4, 8, 22](ctx, a_d, wt_d, 0, p_d)
        ctx.enqueue_copy(dst_ptr=actual.unsafe_ptr(), src_ptr=p_d.unsafe_ptr(), size=MMAX * N)
        ctx.synchronize()
        for i in range(MMAX * N):
            if bitcast[u32](expected[i]) != bitcast[u32](actual[i]):
                raise Error("FAIL transposed byte check at " + String(i))
        print("PASS transposed byte check", MMAX * N)
        # Hold useful work for one second to bring memory clocks out of idle.
        var warm_start = perf_counter_ns()
        while perf_counter_ns() - warm_start < 1_000_000_000:
            for t in range(NT):
                launch[N, K, 2, 3, 10](ctx, a_d, w_d, t, p_d)
            ctx.synchronize()
        for repeat in range(3):
            print("repeat", repeat)
            var base = arm[N, K, 2, 1, 10](ctx, tensor + " TURBO", a_d, w_d, p_d, 0)
            _ = arm[N, K, 4, 8, 20](ctx, tensor + " BASE", a_d, w_d, p_d, base)
            _ = arm[N, K, 4, 8, 21](ctx, tensor + " CENTER", a_d, w_d, p_d, base)
            _ = arm[N, K, 4, 8, 22](ctx, tensor + " TRANSPOSE", a_d, wt_d, p_d, base)
            _ = arm[N, K, 4, 8, 40](ctx, tensor + " IU8", a_d, w_d, p_d, base)
            comptime for mr in range(1, 5):
                _ = arm[N, K, 2, mr, 10](ctx, tensor + " TURBO", a_d, w_d, p_d, base)
                _ = arm[N, K, 2, mr, 50](ctx, tensor + " DOT2", a_d, w_d, p_d, base)
            _ = arm[N, K, 2, 8, 40](ctx, tensor + " IU8", a_d, w_d, p_d, base)
            _ = arm[N, K, 4, 8, 23](ctx, tensor + " WAVES4", a_d, w_d, p_d, base)
            _ = arm[N, K, 4, 8, 24](ctx, tensor + " WAVES2", a_d, w_d, p_d, base)
            _ = arm[N, K, 4, 8, 25](ctx, tensor + " WAVES1", a_d, w_d, p_d, base)
            _ = arm[N, K, 4, 8, 26](ctx, tensor + " TRANS4", a_d, wt_d, p_d, base)
            _ = arm[N, K, 4, 8, 27](ctx, tensor + " TRANS2", a_d, wt_d, p_d, base)
            _ = arm[N, K, 4, 8, 28](ctx, tensor + " TRANS1", a_d, wt_d, p_d, base)
            _ = arm[N, K, 4, 8, 29](ctx, tensor + " MAGIC", a_d, wt_d, p_d, base)
            _ = arm[N, K, 4, 8, 32](ctx, tensor + " PIPE", a_d, wt_d, p_d, base)
        return
    host_check[N, K](ctx, w_h, a_h, a_d, w_d, p_d)
    comptime if (K // 32) % (8 * 8) == 0:
        host_check[N, K, 2, 20](ctx, w_h, a_h, a_d, w_d, p_d)
    var base = arm[N, K, 2, 1, 0](ctx, tensor + " FULL", a_d, w_d, p_d, 0)
    _ = arm[N, K, 2, 2, 0](ctx, tensor + " FULL", a_d, w_d, p_d, base)
    _ = arm[N, K, 2, 3, 0](ctx, tensor + " FULL", a_d, w_d, p_d, base)
    _ = arm[N, K, 2, 4, 0](ctx, tensor + " FULL", a_d, w_d, p_d, base)
    _ = arm[N, K, 2, 3, 1](ctx, tensor + " NOTRAFFIC", a_d, w_d, p_d, base)
    _ = arm[N, K, 2, 3, 2](ctx, tensor + " NOFMA", a_d, w_d, p_d, base)
    _ = arm[N, K, 4, 3, 0](ctx, tensor + " FULL", a_d, w_d, p_d, base)
    _ = arm[N, K, 8, 3, 0](ctx, tensor + " FULL", a_d, w_d, p_d, base)
    _ = arm[N, K, 4, 1, 0](ctx, tensor + " FULL", a_d, w_d, p_d, base)
    byte_check[N, K](ctx, a_d, w_d, p_d)
    _ = arm[N, K, 2, 3, 10](ctx, tensor + " TURBO", a_d, w_d, p_d, base)
    _ = arm[N, K, 1, 3, 10](ctx, tensor + " TURBO", a_d, w_d, p_d, base)
    _ = arm[N, K, 2, 1, 10](ctx, tensor + " TURBO", a_d, w_d, p_d, base)
    _ = arm[N, K, 2, 2, 10](ctx, tensor + " TURBO", a_d, w_d, p_d, base)
    _ = arm[N, K, 2, 4, 10](ctx, tensor + " TURBO", a_d, w_d, p_d, base)
    _ = arm[N, K, 4, 1, 10](ctx, tensor + " TURBO", a_d, w_d, p_d, base)
    _ = arm[N, K, 4, 2, 10](ctx, tensor + " TURBO", a_d, w_d, p_d, base)
    _ = arm[N, K, 4, 3, 10](ctx, tensor + " TURBO", a_d, w_d, p_d, base)
    _ = arm[N, K, 4, 4, 10](ctx, tensor + " TURBO", a_d, w_d, p_d, base)
    comptime for mr in range(1, 5):
        _ = arm[N, K, 2, mr, 30](ctx, tensor + " Q8LDS", a_d, w_d, p_d, base)
        _ = arm[N, K, 2, mr, 31](ctx, tensor + " Q8PRE", a_d, w_d, p_d, base)
    comptime if (K // 32) % (8 * 8) == 0:
        _ = arm[N, K, 1, 1, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
        _ = arm[N, K, 1, 2, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
        _ = arm[N, K, 1, 3, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
        _ = arm[N, K, 1, 4, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
        _ = arm[N, K, 1, 5, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
        _ = arm[N, K, 1, 8, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
        _ = arm[N, K, 2, 1, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
        _ = arm[N, K, 2, 2, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
        _ = arm[N, K, 2, 3, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
        _ = arm[N, K, 2, 4, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
        _ = arm[N, K, 2, 5, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
        _ = arm[N, K, 2, 8, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
        _ = arm[N, K, 4, 1, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
        _ = arm[N, K, 4, 2, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
        _ = arm[N, K, 4, 3, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
        _ = arm[N, K, 4, 4, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
        _ = arm[N, K, 4, 5, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
        _ = arm[N, K, 4, 8, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
        _ = arm[N, K, 8, 1, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
        _ = arm[N, K, 8, 3, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
        _ = arm[N, K, 8, 8, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)
    elif (K // 32) % (8 * 4) == 0:
        # dflash2 D1: 27B shapes fail D=8, D=4 fits; m=8 = the DFlash2 verify window (anchor + 7 drafts).
        # FULL MR 8 = the engine's m > 5 path today (gemm_q4 -> q4rowb[2, SM]).
        host_check[N, K, 4, 20](ctx, w_h, a_h, a_d, w_d, p_d)
        _ = arm[N, K, 2, 8, 0](ctx, tensor + " FULL", a_d, w_d, p_d, base)
        comptime for mr in [1, 3, 4, 5, 8]:
            _ = arm[N, K, 4, mr, 20](ctx, tensor + " WMMA", a_d, w_d, p_d, base)


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var pack = getenv("BARO_PACK", "")
    if pack == "":
        raise Error("FAIL args: BARO_PACK=<q4 pack dir>")
    var ctx = DeviceContext()
    print("bench_turbo_rows pack", pack)
    shape_round[FFN_D, H_D](ctx, pack, "ffn_gate")
    shape_round[H_D, FFN_D](ctx, pack, "ffn_down")
    print("PASS bench_turbo_rows")
