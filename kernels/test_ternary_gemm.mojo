"""Parity check for the three ternary wave-per-row GEMV kernels
(matmul_ternary.mojo: q2b3row = Q2_B3/B3S, tq1row = TQ1_0, tq2row = TQ2_0)
against the C reference codec.

Consumes .work/gguf/ produced by:
  tools/b3s-check.py --fixture 8 <model.gguf> blk.0.ffn_gate.weight blk.0.ffn_down.weight

Per tensor and family the fixture holds the block payload bytes [N, nb*B]
and fp16 scales [N, nb] written by the verbatim ggml quantize_row_*_ref in
tools/ternary-ref.c, a fixed random bf16 A [8, K] (seed 7), and the fp32
reference C = A @ dequant(W)^T where dequant is the C dequantize_row_* of
the same file. Both tensors are exercised so K = 4096 (one block per lane)
and K = 12288 (three blocks per lane, straggler-free) both run; each kernel
is launched at MR = 8 with m = 8 and at MR = 1 with m = 1 (m == MR). Gate: max
relative error < 1e-2 over every checked output (weights decode to exactly
{-d, 0, +d}, so the only disagreement is fp32 summation order).
"""
from std.math import ceildiv
from std.memory import alloc
from std.sys import has_accelerator

from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, row_major

from tio import load_exact
from matmul_skinny import ROW_WAVES, ROW_THREADS
from matmul_ternary import (
    amar_matmul_skinny_q2b3row, amar_matmul_skinny_tq1row,
    amar_matmul_skinny_tq2row,
    B3_BLOCK, B3_BYTES, TQ_BLOCK, TQ1_BYTES, TQ2_BYTES,
)

comptime M = 8
comptime bf16 = DType.bfloat16
comptime f16 = DType.float16
comptime f32 = DType.float32
comptime u8 = DType.uint8


def launch[
    KIND: Int, MR: Int,
    AL: TensorLayout, QL: TensorLayout, SL: TensorLayout, PL: TensorLayout
](
    ctx: DeviceContext,
    A: TileTensor[bf16, AL, MutAnyOrigin],
    Q: TileTensor[u8, QL, MutAnyOrigin],
    S: TileTensor[f16, SL, MutAnyOrigin],
    Cp: TileTensor[f32, PL, MutAnyOrigin],
    m: Int, n: Int, k: Int,
) raises:
    comptime if KIND == 0:
        ctx.enqueue_function[amar_matmul_skinny_q2b3row[MR, AL, QL, SL, PL]](
            A, Q, S, Cp, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    elif KIND == 1:
        ctx.enqueue_function[amar_matmul_skinny_tq1row[MR, AL, QL, SL, PL]](
            A, Q, S, Cp, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    else:
        ctx.enqueue_function[amar_matmul_skinny_tq2row[MR, AL, QL, SL, PL]](
            A, Q, S, Cp, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )


def check[KIND: Int, MR: Int, N: Int, K: Int](
    ctx: DeviceContext, base: String, fam: String
) raises:
    comptime BLOCK = B3_BLOCK if KIND == 0 else TQ_BLOCK
    comptime PB = B3_BYTES if KIND == 0 else (TQ1_BYTES if KIND == 1 else TQ2_BYTES)
    comptime NB = K // BLOCK
    comptime a_layout = row_major[M, K]()
    comptime q_layout = row_major[N, NB * PB]()
    comptime s_layout = row_major[N, NB]()
    comptime p_layout = row_major[1, M, N]()

    var a_host = ctx.enqueue_create_host_buffer[bf16](M * K)
    var q_host = ctx.enqueue_create_host_buffer[u8](N * NB * PB)
    var s_host = ctx.enqueue_create_host_buffer[f16](N * NB)
    var c_host = ctx.enqueue_create_host_buffer[f32](M * N)
    var cref = alloc[Float32](M * N)
    ctx.synchronize()
    load_exact(base + ".tern.a.bin", a_host.unsafe_ptr().unsafe_bitcast[UInt8](), M * K * 2)
    load_exact(base + "." + fam + ".bin", q_host.unsafe_ptr(), N * NB * PB)
    load_exact(base + "." + fam + ".scales.bin", s_host.unsafe_ptr().unsafe_bitcast[UInt8](), N * NB * 2)
    load_exact(base + "." + fam + ".c.bin", cref.unsafe_bitcast[UInt8](), M * N * 4)

    var a_dev = ctx.enqueue_create_buffer[bf16](M * K)
    var q_dev = ctx.enqueue_create_buffer[u8](N * NB * PB)
    var s_dev = ctx.enqueue_create_buffer[f16](N * NB)
    var c_dev = ctx.enqueue_create_buffer[f32](M * N)
    ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
    ctx.enqueue_copy(dst_buf=q_dev, src_buf=q_host)
    ctx.enqueue_copy(dst_buf=s_dev, src_buf=s_host)
    ctx.synchronize()
    var A = TileTensor(a_dev, a_layout)
    var Q = TileTensor(q_dev, q_layout)
    var S = TileTensor(s_dev, s_layout)
    var Cp = TileTensor(c_dev, p_layout)

    comptime AL = type_of(a_layout)
    comptime QL = type_of(q_layout)
    comptime SL = type_of(s_layout)
    comptime PL = type_of(p_layout)

    ctx.enqueue_memset(c_dev, Float32(0))
    launch[KIND, MR, AL, QL, SL, PL](ctx, A, Q, S, Cp, MR, N, K)
    ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
    ctx.synchronize()
    var worst = Float64(0)
    for i in range(MR * N):
        var r = Float64(cref[unsafe_offset=i])
        var err = abs(Float64(c_host[i]) - r) / (abs(r) + 1e-3)
        worst = max(worst, err)
    print(fam, "N=" + String(N), "K=" + String(K), "MR=" + String(MR), "max_rel:", worst)
    if worst > 1e-2:
        raise Error(fam + " parity failure")
    cref.unsafe_free()


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    comptime gate = ".work/gguf/blk_0_ffn_gate_weight"
    comptime down = ".work/gguf/blk_0_ffn_down_weight"

    check[0, 8, 12288, 4096](ctx, gate, "q2b3")
    check[0, 8, 4096, 12288](ctx, down, "q2b3")
    check[0, 1, 12288, 4096](ctx, gate, "q2b3")
    check[0, 1, 4096, 12288](ctx, down, "q2b3")
    check[1, 8, 12288, 4096](ctx, gate, "tq1")
    check[1, 8, 4096, 12288](ctx, down, "tq1")
    check[1, 1, 12288, 4096](ctx, gate, "tq1")
    check[1, 1, 4096, 12288](ctx, down, "tq1")
    check[2, 8, 12288, 4096](ctx, gate, "tq2")
    check[2, 8, 4096, 12288](ctx, down, "tq2")
    check[2, 1, 12288, 4096](ctx, gate, "tq2")
    check[2, 1, 4096, 12288](ctx, down, "tq2")

    print("PASS: q2b3row, tq1row, tq2row match the C reference codec")
