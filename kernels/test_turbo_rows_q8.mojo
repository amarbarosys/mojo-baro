"""amar_turbo_rows_q8 (kernels/turbo_rows.mojo, round A) test: random Q4_0 rows and bf16 A rows at the 27B K values,
N = 64, every MR/UNROLL/m the engine dispatches. Two checks per arm:
1. QA-LDS (quantize while staging) and QA-PRE (amar_quantize_q8_rows then stage) give bit-identical words.
2. Both agree with a host f64 reference built from the read-back Aq/As (exact integer block dots, then the scaled sum)
   within 1e-5 of the row's sum of |terms| (the GPU sums in f32, in another order).
The agreement of int8 activations with the bf16 path is NOT this test: that is the teacher-forced gate in the prereg.
"""
from std.sys import has_accelerator
from std.memory import bitcast
from max.gpu.host import DeviceContext
from max.gpu import WARP_SIZE
from layout import TileTensor, row_major
from matmul_skinny import ROW_WAVES, ROW_THREADS
from elementwise import amar_quantize_q8_rows
from turbo_rows import amar_turbo_rows_q8

comptime N = 64
comptime MMAX = 4
comptime bf16 = DType.bfloat16
comptime f16 = DType.float16
comptime f32 = DType.float32
comptime u8 = DType.uint8
comptime i8 = DType.int8


struct Lcg:
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> UInt64:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return self.s >> 33


def check[K: Int, MR: Int, UNROLL: Int](ctx: DeviceContext, m: Int, mut rng: Lcg) raises -> Int:
    comptime NB = K // 32
    comptime QB = N * K // 2
    comptime a_l = row_major[MMAX, K]()
    comptime as_l = row_major[MMAX, NB]()
    comptime q_l = row_major[N, K // 2]()
    comptime s_l = row_major[N, NB]()
    comptime p_l = row_major[1, MMAX, N]()
    var q_h = ctx.enqueue_create_host_buffer[u8](QB)
    var s_h = ctx.enqueue_create_host_buffer[f16](N * NB)
    var a_h = ctx.enqueue_create_host_buffer[bf16](MMAX * K)
    ctx.synchronize()
    for i in range(QB):
        q_h[i] = UInt8(rng.next() & 0xFF)
    for i in range(N * NB):
        s_h[i] = (Float32(Int(rng.next() % 2001) - 1000) / 20000.0).cast[f16]()
    for i in range(MMAX * K):
        a_h[i] = (Float32(Int(rng.next() % 2001) - 1000) / 500.0).cast[bf16]()
    var q_d = ctx.enqueue_create_buffer[u8](QB)
    var s_d = ctx.enqueue_create_buffer[f16](N * NB)
    var a_d = ctx.enqueue_create_buffer[bf16](MMAX * K)
    var aq_d = ctx.enqueue_create_buffer[i8](MMAX * K)
    var as_d = ctx.enqueue_create_buffer[f16](MMAX * NB)
    var pl_d = ctx.enqueue_create_buffer[f32](MMAX * N)
    var pp_d = ctx.enqueue_create_buffer[f32](MMAX * N)
    ctx.enqueue_copy(dst_buf=q_d, src_buf=q_h)
    ctx.enqueue_copy(dst_buf=s_d, src_buf=s_h)
    ctx.enqueue_copy(dst_buf=a_d, src_buf=a_h)
    ctx.enqueue_memset(aq_d, 0)
    ctx.enqueue_memset(as_d, 0)
    ctx.enqueue_memset(pl_d, 0)
    ctx.enqueue_memset(pp_d, 0)
    var A = TileTensor(a_d, a_l)
    var Aq = TileTensor(aq_d, a_l)
    var As = TileTensor(as_d, as_l)
    var Q = TileTensor(q_d, q_l)
    var S = TileTensor(s_d, s_l)
    ctx.enqueue_function[amar_quantize_q8_rows[type_of(a_l), type_of(a_l), type_of(as_l)]](
        A, Aq, As, Int32(m), Int32(K), grid_dim=(m, NB), block_dim=WARP_SIZE)
    ctx.enqueue_function[amar_turbo_rows_q8[N, K, MR, UNROLL, False, type_of(a_l), type_of(a_l), type_of(as_l), type_of(q_l), type_of(s_l), type_of(p_l)]](
        A, Aq, As, Q, S, TileTensor(pl_d, p_l), Int32(m), grid_dim=N // ROW_WAVES, block_dim=ROW_THREADS)
    ctx.enqueue_function[amar_turbo_rows_q8[N, K, MR, UNROLL, True, type_of(a_l), type_of(a_l), type_of(as_l), type_of(q_l), type_of(s_l), type_of(p_l)]](
        A, Aq, As, Q, S, TileTensor(pp_d, p_l), Int32(m), grid_dim=N // ROW_WAVES, block_dim=ROW_THREADS)
    var aq_h = ctx.enqueue_create_host_buffer[i8](MMAX * K)
    var as_h = ctx.enqueue_create_host_buffer[f16](MMAX * NB)
    var pl_h = ctx.enqueue_create_host_buffer[f32](MMAX * N)
    var pp_h = ctx.enqueue_create_host_buffer[f32](MMAX * N)
    ctx.enqueue_copy(dst_buf=aq_h, src_buf=aq_d)
    ctx.enqueue_copy(dst_buf=as_h, src_buf=as_d)
    ctx.enqueue_copy(dst_buf=pl_h, src_buf=pl_d)
    ctx.enqueue_copy(dst_buf=pp_h, src_buf=pp_d)
    ctx.synchronize()
    var bad = 0
    var nz = 0
    for r in range(m):
        for n in range(N):
            var g = pl_h[r * N + n]
            if bitcast[DType.uint32](g) != 0:
                nz += 1
            if bitcast[DType.uint32](g) != bitcast[DType.uint32](pp_h[r * N + n]):
                if bad == 0:
                    print("FAIL arms K", K, "MR", MR, "UNROLL", UNROLL, "m", m, "row", r, "col", n, "lds", g, "pre", pp_h[r * N + n])
                bad += 1
            var want = Float64(0)
            var mag = Float64(0)
            for b in range(NB):
                var tot = 0
                for j in range(16):
                    var byte = Int(q_h[n * (K // 2) + b * 16 + j])
                    tot += ((byte & 0xF) - 8) * Int(aq_h[r * K + b * 32 + j])
                    tot += ((byte >> 4) - 8) * Int(aq_h[r * K + b * 32 + 16 + j])
                var term = Float64(s_h[n * NB + b].cast[f32]()) * Float64(as_h[r * NB + b].cast[f32]()) * Float64(tot)
                want += term
                mag += abs(term)
            if abs(Float64(g) - want) > 1e-5 * mag + 1e-30:
                if bad == 0:
                    print("FAIL want K", K, "MR", MR, "UNROLL", UNROLL, "m", m, "row", r, "col", n, "gpu", g, "ref", want, "mag", mag)
                bad += 1
    if nz == 0:
        print("FAIL K", K, "m", m, ": output all zero, test void")
        return 1
    print("K", K, " MR", MR, " UNROLL", UNROLL, " m", m, ": ", m * N, "words, bad", bad)
    return bad


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var rng = Lcg(11)
    var bad = 0
    comptime for ks in range(3):
        comptime K = 5120 if ks == 0 else (6144 if ks == 1 else 17408)
        for m in range(1, 4):
            bad += check[K, 3, 2](ctx, m, rng)
            bad += check[K, 4, 2](ctx, m, rng)
        bad += check[K, 4, 2](ctx, 4, rng)
        bad += check[K, 3, 4](ctx, 3, rng)
        bad += check[K, 1, 2](ctx, 1, rng)
        bad += check[K, 2, 2](ctx, 2, rng)
    if bad != 0:
        raise Error("FAIL test_turbo_rows_q8: " + String(bad) + " bad words")
    print("PASS test_turbo_rows_q8: arms bit-identical, both within 1e-5 of the exact int8 reference")
