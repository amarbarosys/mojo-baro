"""amar_turbo_rows (kernels/turbo_rows.mojo) byte test against amar_matmul_skinny_q4rowb, the window kernel it replaces:
random Q4_0 rows at the 27B K values (5120, 6144, 17408) with N = 64, UNROLL 2 and 4, m = 1, 2, 3 under MR 3 and MR 4,
m = 2 under MR 2 and m = 1 under MR 1 (every window width the engine dispatches). Every output word must equal the reference's (same fma chain, same lane order); the first
mismatch fails loudly with shape, arm, row and both words.
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.memory import bitcast
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from matmul_skinny import amar_matmul_skinny_q4rowb, ROW_WAVES, ROW_THREADS
from turbo_rows import amar_turbo_rows

comptime N = 64
comptime MMAX = 3
comptime bf16 = DType.bfloat16
comptime f16 = DType.float16
comptime f32 = DType.float32
comptime u8 = DType.uint8


struct Lcg:
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> UInt64:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return self.s >> 33


def check[K: Int, MR: Int, UNROLL: Int](ctx: DeviceContext, m: Int, mut rng: Lcg) raises -> Int:
    comptime QB = N * K // 2
    comptime NS = N * (K // 32)
    comptime a_l = row_major[MMAX, K]()
    comptime q_l = row_major[N, K // 2]()
    comptime s_l = row_major[N, K // 32]()
    comptime p_l = row_major[1, MMAX, N]()
    var q_h = ctx.enqueue_create_host_buffer[u8](QB)
    var s_h = ctx.enqueue_create_host_buffer[f16](NS)
    var a_h = ctx.enqueue_create_host_buffer[bf16](MMAX * K)
    ctx.synchronize()
    for i in range(QB):
        q_h[i] = UInt8(rng.next() & 0xFF)
    for i in range(NS):
        s_h[i] = (Float32(Int(rng.next() % 2001) - 1000) / 20000.0).cast[f16]()
    for i in range(MMAX * K):
        a_h[i] = (Float32(Int(rng.next() % 2001) - 1000) / 500.0).cast[bf16]()
    var q_d = ctx.enqueue_create_buffer[u8](QB)
    var s_d = ctx.enqueue_create_buffer[f16](NS)
    var a_d = ctx.enqueue_create_buffer[bf16](MMAX * K)
    var pr_d = ctx.enqueue_create_buffer[f32](MMAX * N)
    var pt_d = ctx.enqueue_create_buffer[f32](MMAX * N)
    ctx.enqueue_copy(dst_buf=q_d, src_buf=q_h)
    ctx.enqueue_copy(dst_buf=s_d, src_buf=s_h)
    ctx.enqueue_copy(dst_buf=a_d, src_buf=a_h)
    ctx.enqueue_memset(pr_d, 0)
    ctx.enqueue_memset(pt_d, 0)
    var A = TileTensor(a_d, a_l)
    var Q = TileTensor(q_d, q_l)
    var S = TileTensor(s_d, s_l)
    ctx.enqueue_function[amar_matmul_skinny_q4rowb[2, MR, type_of(a_l), type_of(q_l), type_of(s_l), type_of(p_l)]](
        A, Q, S, TileTensor(pr_d, p_l), Int32(m), Int32(N), Int32(K), grid_dim=ceildiv(N, ROW_WAVES), block_dim=ROW_THREADS)
    ctx.enqueue_function[amar_turbo_rows[N, K, MR, UNROLL, type_of(a_l), type_of(q_l), type_of(s_l), type_of(p_l)]](
        A, Q, S, TileTensor(pt_d, p_l), Int32(m), grid_dim=N // ROW_WAVES, block_dim=ROW_THREADS)
    var pr_h = ctx.enqueue_create_host_buffer[f32](MMAX * N)
    var pt_h = ctx.enqueue_create_host_buffer[f32](MMAX * N)
    ctx.enqueue_copy(dst_buf=pr_h, src_buf=pr_d)
    ctx.enqueue_copy(dst_buf=pt_h, src_buf=pt_d)
    ctx.synchronize()
    var bad = 0
    var nz = 0
    for r in range(m):
        for n in range(N):
            var a = bitcast[DType.uint32](pr_h[r * N + n])
            var b = bitcast[DType.uint32](pt_h[r * N + n])
            if a != 0:
                nz += 1
            if a != b:
                if bad == 0:
                    print("FAIL K", K, "MR", MR, "UNROLL", UNROLL, "m", m, "row", r, "col", n, "ref", pr_h[r * N + n], "turbo", pt_h[r * N + n])
                bad += 1
    if nz == 0:
        print("FAIL K", K, "m", m, ": reference output all zero, test void")
        return 1
    print("K", K, " MR", MR, " UNROLL", UNROLL, " m", m, ": ", m * N, "words, mismatches", bad)
    return bad


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var rng = Lcg(7)
    var bad = 0
    comptime for ks in range(3):
        comptime K = 5120 if ks == 0 else (6144 if ks == 1 else 17408)
        for m in range(1, 4):
            bad += check[K, 3, 2](ctx, m, rng)
            bad += check[K, 3, 4](ctx, m, rng)
            bad += check[K, 4, 2](ctx, m, rng)
        bad += check[K, 2, 2](ctx, 2, rng)
        bad += check[K, 1, 2](ctx, 1, rng)
        bad += check[K, 1, 4](ctx, 1, rng)
    if bad != 0:
        raise Error("FAIL test_turbo_rows: " + String(bad) + " mismatching words")
    print("PASS test_turbo_rows: every word equals amar_matmul_skinny_q4rowb")
