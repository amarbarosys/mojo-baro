# Batched prefill helpers for serve/spark.mojo (wavegroup's T-row forward): weights are dequantised into an f16
# scratch and multiplied by f16 rows with the WMMA GEMM; these kernels are the glue around that GEMM.
from std.math import ceildiv, fma, tanh
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major
from matmul_wmma_pipe import amar_matmul_wmma_pipe
from matmul_prefill_kq import amar_matmul_prefill_kq

comptime KQ_U_Q4K = 8
comptime KQ_U_Q6K = 4
from matmul_prefill_lds import LDS_THREADS
from bexp import exp
from max.gpu import global_idx
from gemv_kq import kq_lane16, kq_bytes, GT_Q4K, GT_Q6K
from q4x_kernels import UP

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime f16 = DType.float16
comptime P32 = Pointer[Float32, MutAnyOrigin]
comptime P16 = Pointer[Float16, MutAnyOrigin]


def pf_dequant_kq[GT: Int](w: UP, o: P16, n32: Int32, k32: Int32):
    """n rows of a Q4_K / Q6_K matrix to f16 [n, k], with the GEMV's own unpack."""
    var t = Int(global_idx.x)
    var sbr = Int(k32) // 256
    if t < Int(n32) * sbr * 16:
        var sb = t // 16
        var r = kq_lane16[GT](w, sb * kq_bytes[GT](), t % 16)
        var v = fma(r[0].cast[f32](), SIMD[f32, 16](r[1]), SIMD[f32, 16](-r[2]))
        var at = (sb // sbr) * Int(k32) + (sb % sbr) * 256 + r[3]
        comptime for i in range(16):
            o[at + i] = v[i].cast[f16]()


def pf_dequant_q8(q: Pointer[Int8, MutAnyOrigin], s: P16, o: P16, nk: Int32):
    """The pack's q8 matrix (int8 [n*k], then one f16 scale per 32 along k) to f16."""
    var i = Int(global_idx.x)
    if i < Int(nk):
        o[i] = (Float32(q[i]) * s[i // 32].cast[f32]()).cast[f16]()


def pf_cast16[DT: DType](x: Pointer[Scalar[DT], MutAnyOrigin], o: P16, n: Int32):
    var i = Int(global_idx.x)
    if i < Int(n):
        o[i] = x[i].cast[f16]()


def pf_qkv_split[BIAS: Bool](t: P32, bias: P32, q: P32, k: P32, v: P32, qd: Int32, kvd: Int32, n: Int32):
    """GEMM rows [T, qd + 2 kvd] (+ bias) into Q [T, qd], K [T, kvd], V [T, kvd]: the per-row layouts the kernels index."""
    var i = Int(global_idx.x)
    var w = Int(qd) + 2 * Int(kvd)
    if i >= Int(n) * w:
        return
    var r = i // w
    var c = i % w
    var x = t[i]
    comptime if BIAS:
        x += bias[c]
    if c < Int(qd):
        q[r * Int(qd) + c] = x
    elif c < Int(qd) + Int(kvd):
        k[r * Int(kvd) + c - Int(qd)] = x
    else:
        v[r * Int(kvd) + c - Int(qd) - Int(kvd)] = x


def pf_add(t: P32, x: P32, n: Int32):
    """Residual: x += t."""
    var i = Int(global_idx.x)
    if i < Int(n):
        x[i] += t[i]


def pf_act_mul[GELU: Bool, DT: DType](g: P32, u: P32, o: Pointer[Scalar[DT], MutAnyOrigin], n: Int32):
    """o = act(g) * u, cast to the activation type: the gate/up epilogue of the GEMV path."""
    var i = Int(global_idx.x)
    if i < Int(n):
        var a = g[i]
        comptime if GELU:
            a = 0.5 * a * (1 + tanh(0.7978845608028654 * a * (1 + 0.044715 * a * a)))
        else:
            a = a / (1 + exp(-a))
        o[i] = (a * u[i]).cast[DT]()


def kq_dequant[N: Int, K: Int](ctx: DeviceContext, wa: Int, kind: Int, ws: Int) raises:
    """The Q4_K / Q6_K matrix at address wa [N, K] to f16 rows at ws (run 9's engine route; now the reference arm of
    kernels/test_kq_prefill_gemm.mojo, the engine runs kq_gemm_lds)."""
    var w = UP(unsafe_from_address=wa)
    var o = P16(unsafe_from_address=ws)
    if kind == GT_Q4K:
        ctx.enqueue_function[pf_dequant_kq[GT_Q4K]](w, o, Int32(N), Int32(K), grid_dim=ceildiv(N * K // 16, 256), block_dim=256)
    elif kind == GT_Q6K:
        ctx.enqueue_function[pf_dequant_kq[GT_Q6K]](w, o, Int32(N), Int32(K), grid_dim=ceildiv(N * K // 16, 256), block_dim=256)
    else:
        raise Error("kq prefill: weight kind " + String(kind) + " is not Q4_K / Q6_K")


def kq_gemm[N: Int, K: Int, MP: Int, ADT: DType](ctx: DeviceContext, a: Int, x16: Int, ws: Int, c: Int, m: Int) raises:
    """c[MP, N] f32 rows 0..m-1 = a[m, K] (ADT) times the f16 rows at ws (kq_dequant), through the WMMA GEMM; rows m..
    round up to the tile are scratch. Tile as serve/spark.mojo pf_linear; N = 32 takes a one-warp-wide tile."""
    comptime B128 = ceildiv(N, 128) * ceildiv(MP, 128)
    comptime WM = 2 if N == 32 or B128 < 64 else 4
    comptime WN = 1 if N == 32 else (4 if 64 <= B128 < 512 else 2)
    comptime TN = 4 if B128 >= 512 else 2
    comptime TILE = WN * TN * 16
    comptime assert N % TILE == 0 and K % 32 == 0 and MP % (WM * 32) == 0, "kq prefill GEMM: N below the WMMA tile takes the GEMV"
    var x = P16(unsafe_from_address=x16)
    ctx.enqueue_function[pf_cast16[ADT]](Pointer[Scalar[ADT], MutAnyOrigin](unsafe_from_address=a), x, Int32(m * K), grid_dim=ceildiv(m * K, 256), block_dim=256)
    var A = TileTensor(DeviceBuffer[f16](ctx, x, MP * K, owning=False), row_major[MP, K]())
    var Bm = TileTensor(DeviceBuffer[f16](ctx, P16(unsafe_from_address=ws), N * K, owning=False), row_major[N, K]())
    var C = TileTensor(DeviceBuffer[f32](ctx, P32(unsafe_from_address=c), MP * N, owning=False), row_major[MP, N]())
    ctx.enqueue_function[amar_matmul_wmma_pipe[type_of(A).LayoutType, type_of(Bm).LayoutType, type_of(C).LayoutType, WM, WN, 2, TN, 1, 0]](
        A, Bm, C, Int32(m), Int32(N), Int32(K), grid_dim=(N // TILE, ceildiv(m, WM * 32)), block_dim=WM * WN * 32)


def kq_launch[GT: Int, N: Int, K: Int, ACC: Bool, U: Int, AL: TensorLayout, CL: TensorLayout](
    ctx: DeviceContext, A: TileTensor[bf16, AL, MutAnyOrigin], w: UP, C: TileTensor[f32, CL, MutAnyOrigin], m: Int
) raises:
    comptime assert K % 256 == 0, "kq prefill GEMM: K is whole superblocks"
    comptime if N % 128 != 0:
        comptime assert N == 32, "kq prefill GEMM: N is 32 or a multiple of 128"
        ctx.enqueue_function[amar_matmul_prefill_kq[GT, 8, 1, 1, 2, ACC, U, AL, CL]](
            A, w, C, Int32(m), Int32(N), Int32(K), grid_dim=(1, ceildiv(m, 128)), block_dim=LDS_THREADS)
    else:
        if m <= 64:
            ctx.enqueue_function[amar_matmul_prefill_kq[GT, 2, 4, 2, 2, ACC, U, AL, CL]](
                A, w, C, Int32(m), Int32(N), Int32(K), grid_dim=(N // 128, ceildiv(m, 64)), block_dim=LDS_THREADS)
        else:
            ctx.enqueue_function[amar_matmul_prefill_kq[GT, 4, 2, 2, 4, ACC, U, AL, CL]](
                A, w, C, Int32(m), Int32(N), Int32(K), grid_dim=(N // 128, ceildiv(m, 128)), block_dim=LDS_THREADS)


def kq_gemm_lds[AL: TensorLayout, CL: TensorLayout, //, N: Int, K: Int, ACC: Bool, U4: Int = KQ_U_Q4K, U6: Int = KQ_U_Q6K](
    ctx: DeviceContext, A: TileTensor[bf16, AL, MutAnyOrigin], wa: Int, kind: Int, C: TileTensor[f32, CL, MutAnyOrigin], m: Int
) raises:
    """C[m, N] (+)= A[m, K] bf16 times the Q4_K / Q6_K matrix at address wa [N, K], decoded to f16 inside the LDS-pipelined
    GEMM's B staging (kernels/matmul_prefill_kq.mojo): no weight scratch, no row cast. Tile by m as the q4 route; N = 32 on
    one wave-wide tile. U4 / U6: K-step unroll per kind (steps of one superblock the loop body holds at comptime;
    bench/bench_kq_prefill.mojo sweeps them)."""
    var w = UP(unsafe_from_address=wa)
    if kind == GT_Q4K:
        kq_launch[GT_Q4K, N, K, ACC, U4](ctx, A, w, C, m)
    elif kind == GT_Q6K:
        kq_launch[GT_Q6K, N, K, ACC, U6](ctx, A, w, C, m)
    else:
        raise Error("kq prefill: weight kind " + String(kind) + " is not Q4_K / Q6_K")
