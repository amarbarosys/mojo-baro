"""Parity for the int8 MMQ prefill path (bench/prefill-protocol.md R5).

1. amar_quant_q8 vs a host reference on the same bf16 rows: int8 codes
   bit-exact, block scale d bit-exact, and the bias term -8*d*sum(aq)
   bit-exact (sum of 32 integers in [-127,127] is exact in f32, so every
   summation order agrees). Also checks the accumulator-order permutation of
   the scale arrays by de-permuting them on the host.
2. amar_matmul_lds_q4[int8] vs an fp64 host dot over the SAME device-produced
   quantised activations and the same dequantised q4 weights (32-row subset,
   all 12288 columns): isolates the kernel from the quantisation. Both tile
   configs (128x128, 64x128), the ACC epilogue, and the M-padding path (M=24).
   Gate 1e-3 on the floored metric (bench/prefill-protocol.md R5).
3. amar_matmul_lds_q4[bf16] (the same schedule, bf16 maths of the R4 kernel)
   vs amar_matmul_prefill_q4: rel < 1e-4, both configs, M=1024 and M=24.
4. (1)+(2) end to end vs amar_matmul_prefill_q4 on the same bf16 activations:
   the new rounding step. Gate: relative Frobenius error < 1e-2 and
   max|a-b| / rms(b) < 5e-2 (predicted ~4e-3 and ~2e-2); the floored
   per-element metric is printed for the record only.
"""
from std.math import ceildiv
from std.memory import alloc
from std.sys import has_accelerator

from max.algorithm import parallelize
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from tio import load_into
from matmul_prefill import amar_matmul_prefill_q4, PF_THREADS
from matmul_mmq import amar_quant_q8, amar_matmul_lds_q4, MMQ_THREADS, QT_THREADS

comptime bf16 = DType.bfloat16
comptime f16 = DType.float16
comptime f32 = DType.float32
comptime i8 = DType.int8
comptime i32 = DType.int32
comptime u8 = DType.uint8

comptime K = 4096
comptime N = 12288
comptime MBIG = 1024
comptime MSMALL = 24
comptime NSUB = 32
comptime NB = K // 32
comptime Q4BYTES = N * K // 2 + N * (K // 32) * 2

comptime a_layout = row_major[MBIG, K]()
comptime aq_layout = row_major[MBIG, K]()
comptime d_layout = row_major[NB, MBIG]()
comptime c_layout = row_major[MBIG, N]()
comptime q4_layout = row_major[N, K // 2]()
comptime s_layout = row_major[N, K // 32]()


def pack_offset(index: String, name: String, dt: String) raises -> Int:
    with open(index, "r") as f:
        for line in f.read().splitlines():
            var parts = line.split(" ")
            if parts[0] == name:
                if String(parts[1]) != dt:
                    raise Error("wrong dtype for " + name)
                return Int(parts[2])
    raise Error("missing " + name)


def view[
    dt: DType, LT: TensorLayout
](ctx: DeviceContext, buf: DeviceBuffer[u8], o: Int, n: Int, lt: LT) -> TileTensor[dt, LT, MutAnyOrigin]:
    var b = DeviceBuffer[dt](ctx, buf.unsafe_ptr().unsafe_offset(o).unsafe_bitcast[Scalar[dt]](), n, owning=False)
    var t = TileTensor(b, lt)
    return rebind[TileTensor[dt, LT, MutAnyOrigin]](t)


def lcg(mut st: UInt64) -> Float32:
    st = st * 6364136223846793005 + 1442695040888963407
    return Float32(Int((st >> 40) & 0xFFFF)) / 32768.0 - 1.0


def slot_of(r: Int) -> Int:
    return (r // 16) * 16 + (r % 2) * 8 + (r % 16) // 2


def dequant_q4(q: MutPointer[UInt8, MutUntrackedOrigin], s: MutPointer[Float16, MutUntrackedOrigin], c: Int, k: Int) -> Float64:
    var kb = k // 32
    var e = k % 32
    var b = Int(q[unsafe_offset=c * (K // 2) + kb * 16 + (e % 16)])
    var nib = (b & 0xF) if e < 16 else (b >> 4)
    return Float64(nib - 8) * Float64(s[unsafe_offset=c * (K // 32) + kb].cast[f32]())


def gemm_ref_mmq(
    aq: MutPointer[Int8, MutUntrackedOrigin], ad: MutPointer[Float32, MutUntrackedOrigin],
    q: MutPointer[UInt8, MutUntrackedOrigin], s: MutPointer[Float16, MutUntrackedOrigin],
    rows: MutPointer[Int32, MutUntrackedOrigin], dst: MutPointer[Float32, MutUntrackedOrigin], nrows: Int,
):
    def one(t: Int) {imm aq, imm ad, imm q, imm s, imm rows, imm dst}:
        var r = Int(rows[unsafe_offset=t])
        var sl = slot_of(r)
        for c in range(N):
            var acc = Float64(0)
            for k in range(K):
                var d8 = Float64(ad[unsafe_offset=(k // 32) * MBIG + sl])
                acc += Float64(aq[unsafe_offset=r * K + k]) * d8 * dequant_q4(q, s, c, k)
            dst[unsafe_offset=t * N + c] = Float32(acc)
    parallelize(one, nrows)


def gather_rows(src: MutPointer[Float32, MutUntrackedOrigin], rows: MutPointer[Int32, MutUntrackedOrigin],
                dst: MutPointer[Float32, MutUntrackedOrigin], nrows: Int):
    for t in range(nrows):
        var r = Int(rows[unsafe_offset=t])
        for c in range(N):
            dst[unsafe_offset=t * N + c] = src[unsafe_offset=r * N + c]


def check(name: String, got: MutPointer[Float32, MutUntrackedOrigin],
          want: MutPointer[Float32, MutUntrackedOrigin], n: Int,
          gate: Float64, floor: Float64 = 1e-2) raises:
    var worst = Float64(0)
    var wi = 0
    var tot = Float64(0)
    for i in range(n):
        var e = abs(Float64(got[unsafe_offset=i]) - Float64(want[unsafe_offset=i]))
        var rel = e / (abs(Float64(want[unsafe_offset=i])) + floor)
        tot += rel
        if rel > worst:
            worst = rel
            wi = i
    print(name, "max_rel:", worst, "mean_rel:", tot / Float64(n), "at", wi,
          "got", got[unsafe_offset=wi], "want", want[unsafe_offset=wi])
    if worst > gate:
        raise Error("parity failure: " + name)


def run_quant(ctx: DeviceContext, A: TileTensor[bf16, type_of(a_layout), MutAnyOrigin],
              Aq: TileTensor[i8, type_of(aq_layout), MutAnyOrigin],
              Ad: TileTensor[f32, type_of(d_layout), MutAnyOrigin],
              An: TileTensor[i32, type_of(d_layout), MutAnyOrigin], m: Int) raises:
    ctx.enqueue_function[amar_quant_q8[type_of(a_layout), type_of(aq_layout), type_of(d_layout), type_of(d_layout)]](
        A, Aq, Ad, An, Int32(m), Int32(K), Int32(MBIG),
        grid_dim=ceildiv(MBIG * NB, QT_THREADS), block_dim=QT_THREADS,
    )


def run_lds[ADT: DType, WM: Int, WN: Int, TM: Int, TN: Int, ACC: Bool](
    ctx: DeviceContext, A: TileTensor[ADT, type_of(a_layout), MutAnyOrigin],
    Ad: TileTensor[f32, type_of(d_layout), MutAnyOrigin], An: TileTensor[i32, type_of(d_layout), MutAnyOrigin],
    Q: TileTensor[u8, type_of(q4_layout), MutAnyOrigin], S: TileTensor[f16, type_of(s_layout), MutAnyOrigin],
    C: TileTensor[f32, type_of(c_layout), MutAnyOrigin], m: Int,
) raises:
    comptime BM = WM * TM * 16
    comptime BN = WN * TN * 16
    ctx.enqueue_function[amar_matmul_lds_q4[ADT, WM, WN, TM, TN, ACC, type_of(a_layout), type_of(d_layout), type_of(d_layout), type_of(q4_layout), type_of(s_layout), type_of(c_layout)]](
        A, Ad, An, Q, S, C, Int32(m), Int32(N), Int32(K),
        grid_dim=(ceildiv(N, BN), ceildiv(m, BM)), block_dim=MMQ_THREADS,
    )


def run_q4[WTM: Int, WTN: Int, WAVES_M: Int, ACC: Bool](
    ctx: DeviceContext, A: TileTensor[bf16, type_of(a_layout), MutAnyOrigin],
    Q: TileTensor[u8, type_of(q4_layout), MutAnyOrigin], S: TileTensor[f16, type_of(s_layout), MutAnyOrigin],
    C: TileTensor[f32, type_of(c_layout), MutAnyOrigin], m: Int,
) raises:
    comptime BM = WAVES_M * WTM * 16
    comptime BN = (8 // WAVES_M) * WTN * 16
    ctx.enqueue_function[amar_matmul_prefill_q4[WTM, WTN, WAVES_M, ACC, type_of(a_layout), type_of(q4_layout), type_of(s_layout), type_of(c_layout)]](
        A, Q, S, C, Int32(m), Int32(N), Int32(K),
        grid_dim=(ceildiv(N, BN), ceildiv(m, BM)), block_dim=PF_THREADS,
    )


def frob(name: String, got: MutPointer[Float32, MutUntrackedOrigin], want: MutPointer[Float32, MutUntrackedOrigin],
         n: Int, gate_frob: Float64, gate_max: Float64) raises:
    var se = Float64(0)
    var sw = Float64(0)
    var worst = Float64(0)
    var wi = 0
    var tot = Float64(0)
    var wflo = Float64(0)
    for i in range(n):
        var b = Float64(want[unsafe_offset=i])
        var e = abs(Float64(got[unsafe_offset=i]) - b)
        se += e * e
        sw += b * b
        if e > worst:
            worst = e
            wi = i
        var rel = e / (abs(b) + 1e-2)
        tot += rel
        if rel > wflo:
            wflo = rel
    var rms = (sw / Float64(n)) ** 0.5
    var rf = (se / sw) ** 0.5
    var mx = worst / rms
    print(name, "rel_frobenius:", rf, "max_abs/rms:", mx, "rms(b):", rms, "at", wi,
          "got", got[unsafe_offset=wi], "want", want[unsafe_offset=wi],
          "| floored metric max:", wflo, "mean:", tot / Float64(n))
    if rf > gate_frob or mx > gate_max:
        raise Error("parity failure: " + name)


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var q_h = ctx.enqueue_create_host_buffer[u8](Q4BYTES)
    var a_h = ctx.enqueue_create_host_buffer[bf16](MBIG * K)
    var aq_h = ctx.enqueue_create_host_buffer[i8](MBIG * K)
    var ad_h = ctx.enqueue_create_host_buffer[f32](NB * MBIG)
    var an_h = ctx.enqueue_create_host_buffer[i32](NB * MBIG)
    var c_h = ctx.enqueue_create_host_buffer[f32](MBIG * N)
    var c2_h = ctx.enqueue_create_host_buffer[f32](MBIG * N)
    ctx.synchronize()
    load_into(".work/engine-pack-q4/pack.bin", q_h.unsafe_ptr(), Q4BYTES,
              pack_offset(".work/engine-pack-q4/index.txt", "blk.0.ffn_gate.weight", "q4"))
    var st = UInt64(7)
    for i in range(MBIG * K):
        a_h[i] = Scalar[bf16](lcg(st))

    var q_d = ctx.enqueue_create_buffer[u8](Q4BYTES)
    var a_d = ctx.enqueue_create_buffer[bf16](MBIG * K)
    var aq_d = ctx.enqueue_create_buffer[i8](MBIG * K)
    var ad_d = ctx.enqueue_create_buffer[f32](NB * MBIG)
    var an_d = ctx.enqueue_create_buffer[i32](NB * MBIG)
    var c_d = ctx.enqueue_create_buffer[f32](MBIG * N)
    var c2_d = ctx.enqueue_create_buffer[f32](MBIG * N)
    ctx.enqueue_copy(dst_buf=q_d, src_buf=q_h)
    ctx.enqueue_copy(dst_buf=a_d, src_buf=a_h)
    ctx.synchronize()
    var A = TileTensor(a_d, a_layout)
    var Aq = TileTensor(aq_d, aq_layout)
    var Ad = TileTensor(ad_d, d_layout)
    var An = TileTensor(an_d, d_layout)
    var C = TileTensor(c_d, c_layout)
    var C2 = TileTensor(c2_d, c_layout)
    var Q = view[u8](ctx, q_d, 0, N * K // 2, q4_layout)
    var S = view[f16](ctx, q_d, N * K // 2, N * (K // 32), s_layout)

    run_quant(ctx, A, Aq, Ad, An, MBIG)
    ctx.enqueue_copy(dst_buf=aq_h, src_buf=aq_d)
    ctx.enqueue_copy(dst_buf=ad_h, src_buf=ad_d)
    ctx.enqueue_copy(dst_buf=an_h, src_buf=an_d)
    ctx.synchronize()

    var bad_q = 0
    var bad_d = 0
    var bad_n = 0
    for r in range(MBIG):
        var sl = slot_of(r)
        for kb in range(NB):
            var amax = Float32(0)
            for e in range(32):
                var v = abs(a_h[r * K + kb * 32 + e].cast[f32]())
                if v > amax:
                    amax = v
            var d = amax / 127
            var inv = Float32(0) if amax == 0 else 127 / amax
            var s32 = 0
            for e in range(32):
                var qv = round(a_h[r * K + kb * 32 + e].cast[f32]() * inv)
                if Int(aq_h[r * K + kb * 32 + e]) != Int(qv):
                    bad_q += 1
                s32 += Int(qv)
            if ad_h[kb * MBIG + sl] != d:
                bad_d += 1
            if Int(an_h[kb * MBIG + sl]) != -8 * s32:
                bad_n += 1
    print("quant codes mismatched:", bad_q, " d mismatched:", bad_d, " bias mismatched:", bad_n)
    if bad_q + bad_d + bad_n != 0:
        raise Error("parity failure: amar_quant_q8")

    var rows_p = alloc[Int32](NSUB)
    var st2 = UInt64(31)
    for t in range(NSUB):
        rows_p[unsafe_offset=t] = Int32(Int(lcg(st2) * 400 + 500) % MBIG)
    rows_p[unsafe_offset=0] = 0
    rows_p[unsafe_offset=1] = 1
    rows_p[unsafe_offset=2] = Int32(MBIG - 1)
    var want = alloc[Float32](NSUB * N)
    var got = alloc[Float32](NSUB * N)
    gemm_ref_mmq(aq_h.unsafe_ptr(), ad_h.unsafe_ptr(), q_h.unsafe_ptr(),
                 q_h.unsafe_ptr().unsafe_offset(N * K // 2).unsafe_bitcast[Float16](),
                 rows_p, want, NSUB)

    run_lds[i8, 4, 2, 2, 4, False](ctx, Aq, Ad, An, Q, S, C, MBIG)
    ctx.enqueue_copy(dst_buf=c_h, src_buf=c_d)
    ctx.synchronize()
    gather_rows(c_h.unsafe_ptr(), rows_p, got, NSUB)
    check("mmq 128x128 (WM4 WN2 TM2 TN4) M=1024 vs host fp64", got, want, NSUB * N, 1e-3)

    run_lds[i8, 2, 4, 2, 2, False](ctx, Aq, Ad, An, Q, S, C, MBIG)
    ctx.enqueue_copy(dst_buf=c_h, src_buf=c_d)
    ctx.synchronize()
    gather_rows(c_h.unsafe_ptr(), rows_p, got, NSUB)
    check("mmq 64x128 (WM2 WN4 TM2 TN2) M=1024 vs host fp64", got, want, NSUB * N, 1e-3)

    run_lds[i8, 4, 2, 2, 4, True](ctx, Aq, Ad, An, Q, S, C, MBIG)
    ctx.enqueue_copy(dst_buf=c_h, src_buf=c_d)
    ctx.synchronize()
    gather_rows(c_h.unsafe_ptr(), rows_p, got, NSUB)
    for i in range(NSUB * N):
        want[unsafe_offset=i] = want[unsafe_offset=i] * 2
    check("mmq ACC epilogue (doubles the tile)", got, want, NSUB * N, 1e-3)

    var rows_s = alloc[Int32](NSUB)
    for t in range(NSUB):
        rows_s[unsafe_offset=t] = Int32(t % MSMALL)
    var want_s = alloc[Float32](NSUB * N)
    gemm_ref_mmq(aq_h.unsafe_ptr(), ad_h.unsafe_ptr(), q_h.unsafe_ptr(),
                 q_h.unsafe_ptr().unsafe_offset(N * K // 2).unsafe_bitcast[Float16](),
                 rows_s, want_s, NSUB)
    for i in range(MBIG * N):
        c_h[i] = 0
    ctx.enqueue_copy(dst_buf=c_d, src_buf=c_h)
    run_lds[i8, 4, 2, 2, 4, False](ctx, Aq, Ad, An, Q, S, C, MSMALL)
    ctx.enqueue_copy(dst_buf=c_h, src_buf=c_d)
    ctx.synchronize()
    gather_rows(c_h.unsafe_ptr(), rows_s, got, NSUB)
    check("mmq 128x128 M=24 padding path vs host fp64", got, want_s, NSUB * N, 1e-3)
    for i in range(MSMALL * N, MBIG * N):
        if c_h[i] != 0:
            raise Error("mmq wrote past M")
    run_lds[i8, 2, 4, 2, 2, False](ctx, Aq, Ad, An, Q, S, C, MSMALL)
    ctx.enqueue_copy(dst_buf=c_h, src_buf=c_d)
    ctx.synchronize()
    gather_rows(c_h.unsafe_ptr(), rows_s, got, NSUB)
    check("mmq 64x128 M=24 padding path vs host fp64", got, want_s, NSUB * N, 1e-3)

    run_q4[4, 2, 2, False](ctx, A, Q, S, C2, MBIG)
    run_lds[bf16, 4, 2, 2, 4, False](ctx, A, Ad, An, Q, S, C, MBIG)
    ctx.enqueue_copy(dst_buf=c_h, src_buf=c_d)
    ctx.enqueue_copy(dst_buf=c2_h, src_buf=c2_d)
    ctx.synchronize()
    check("bf16-lds 128x128 M=1024 vs amar_matmul_prefill_q4", c_h.unsafe_ptr(), c2_h.unsafe_ptr(), MBIG * N, 1e-4)
    run_lds[bf16, 2, 4, 2, 2, False](ctx, A, Ad, An, Q, S, C, MBIG)
    ctx.enqueue_copy(dst_buf=c_h, src_buf=c_d)
    ctx.synchronize()
    check("bf16-lds 64x128 M=1024 vs amar_matmul_prefill_q4", c_h.unsafe_ptr(), c2_h.unsafe_ptr(), MBIG * N, 1e-4)
    run_q4[4, 2, 2, False](ctx, A, Q, S, C2, MSMALL)
    run_lds[bf16, 4, 2, 2, 4, False](ctx, A, Ad, An, Q, S, C, MSMALL)
    ctx.enqueue_copy(dst_buf=c_h, src_buf=c_d)
    ctx.enqueue_copy(dst_buf=c2_h, src_buf=c2_d)
    ctx.synchronize()
    check("bf16-lds 128x128 M=24 vs amar_matmul_prefill_q4", c_h.unsafe_ptr(), c2_h.unsafe_ptr(), MSMALL * N, 1e-4)

    run_lds[i8, 4, 2, 2, 4, False](ctx, Aq, Ad, An, Q, S, C, MBIG)
    run_q4[4, 2, 2, False](ctx, A, Q, S, C2, MBIG)
    ctx.enqueue_copy(dst_buf=c_h, src_buf=c_d)
    ctx.enqueue_copy(dst_buf=c2_h, src_buf=c2_d)
    ctx.synchronize()
    frob("mmq+quant vs bf16 prefill kernel (M=1024, all rows)", c_h.unsafe_ptr(), c2_h.unsafe_ptr(), MBIG * N, 1e-2, 5e-2)
    print("PASS")
