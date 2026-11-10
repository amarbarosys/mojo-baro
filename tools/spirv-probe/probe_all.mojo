"""Emit-only instantiation of every amar_* elementwise kernel.

Built with `--target-accelerator apple-m1 --emit asm`, never run. The layouts here
are baked into the IR as strides, so the shape defines in `host.c` must match them.
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.sys import get_defined_int

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

from elementwise import (
    amar_rmsnorm, amar_rmsnorm_cast, amar_rmsnorm_cast2, amar_swiglu, amar_rope_rows,
    amar_softmax_rows, amar_embed_lookup, amar_embed_lookup_pos, amar_argmax_pos,
    amar_argmax_row, amar_tok_copy, amar_tok_remap, amar_quantize_q8_rows, EW_THREADS,
)

comptime R = 8
comptime H = 4096
comptime V = get_defined_int["V", 5003]()
comptime TV = 1024
comptime K = 64
comptime QM = 8
comptime QK = 4096

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime i32 = DType.int32


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU target"
    var ctx = DeviceContext()

    comptime x_l = row_major[R, H]()
    comptime g_l = row_major[H]()
    comptime f_l = row_major[1, H]()
    comptime flat_l = row_major[R * H]()
    comptime v_l = row_major[R, V]()
    comptime tab_l = row_major[TV, H]()
    comptime k_l = row_major[K]()
    comptime qa_l = row_major[QM, QK]()
    comptime qs_l = row_major[QM, QK // 32]()

    var x_buf = ctx.enqueue_create_buffer[f32](R * H)
    var X = TileTensor(x_buf, x_l)
    var g_buf = ctx.enqueue_create_buffer[f32](H)
    var G = TileTensor(g_buf, g_l)
    var o_buf = ctx.enqueue_create_buffer[f32](R * H)
    var O = TileTensor(o_buf, x_l)
    var ob_buf = ctx.enqueue_create_buffer[bf16](R * H)
    var Ob = TileTensor(ob_buf, x_l)
    var f_buf = ctx.enqueue_create_buffer[f32](H)
    var F = TileTensor(f_buf, f_l)
    var xf_buf = ctx.enqueue_create_buffer[f32](R * H)
    var Xf = TileTensor(xf_buf, flat_l)
    var uf_buf = ctx.enqueue_create_buffer[f32](R * H)
    var Uf = TileTensor(uf_buf, flat_l)
    var of_buf = ctx.enqueue_create_buffer[f32](R * H)
    var Of = TileTensor(of_buf, flat_l)
    var s_buf = ctx.enqueue_create_buffer[f32](R * V)
    var S = TileTensor(s_buf, v_l)
    var tab_buf = ctx.enqueue_create_buffer[bf16](TV * H)
    var Tab = TileTensor(tab_buf, tab_l)
    var toks_buf = ctx.enqueue_create_buffer[i32](K)
    var Toks = TileTensor(toks_buf, k_l)
    var toks2_buf = ctx.enqueue_create_buffer[i32](K)
    var Toks2 = TileTensor(toks2_buf, k_l)
    var a_buf = ctx.enqueue_create_buffer[bf16](QM * QK)
    var A = TileTensor(a_buf, qa_l)
    var aq_buf = ctx.enqueue_create_buffer[DType.int8](QM * QK)
    var Aq = TileTensor(aq_buf, qa_l)
    var as_buf = ctx.enqueue_create_buffer[DType.float16](QM * (QK // 32))
    var As = TileTensor(as_buf, qs_l)

    ctx.enqueue_function[amar_rmsnorm[type_of(x_l), type_of(g_l), type_of(x_l)]](
        X, G, O, Int32(H), Float32(1e-6), grid_dim=R, block_dim=EW_THREADS)
    ctx.enqueue_function[amar_rmsnorm_cast[type_of(x_l), type_of(g_l), type_of(x_l)]](
        X, G, Ob, Int32(H), Float32(1e-6), grid_dim=R, block_dim=EW_THREADS)
    ctx.enqueue_function[amar_rmsnorm_cast2[type_of(x_l), type_of(g_l), type_of(x_l), type_of(f_l)]](
        X, G, Ob, F, Int32(H), Float32(1e-6), grid_dim=R, block_dim=EW_THREADS)
    ctx.enqueue_function[amar_swiglu[type_of(flat_l), type_of(flat_l), type_of(flat_l)]](
        Xf, Uf, Of, Int32(R * H), grid_dim=ceildiv(R * H, EW_THREADS), block_dim=EW_THREADS)
    ctx.enqueue_function[amar_rope_rows[type_of(x_l)]](
        X, Int32(8), Int32(128), Int32(5), Float32(10000.0), grid_dim=R, block_dim=EW_THREADS)
    ctx.enqueue_function[amar_softmax_rows[type_of(v_l)]](
        S, Int32(V), grid_dim=R, block_dim=EW_THREADS)
    ctx.enqueue_function[amar_embed_lookup[type_of(tab_l), type_of(x_l)]](
        Tab, O, Int32(123), Int32(H), grid_dim=(ceildiv(H, EW_THREADS), R), block_dim=EW_THREADS)
    ctx.enqueue_function[amar_embed_lookup_pos[type_of(tab_l), type_of(x_l), type_of(k_l)]](
        Tab, O, Toks, Int32(3), Int32(H), grid_dim=(ceildiv(H, EW_THREADS), R), block_dim=EW_THREADS)
    ctx.enqueue_function[amar_argmax_pos[type_of(v_l), type_of(k_l)]](
        S, Toks, Int32(V), Int32(3), grid_dim=R, block_dim=EW_THREADS)
    ctx.enqueue_function[amar_argmax_row[type_of(v_l), type_of(k_l)]](
        S, Toks, Int32(V), grid_dim=R, block_dim=EW_THREADS)
    ctx.enqueue_function[amar_tok_copy[type_of(k_l), type_of(k_l)]](
        Toks, Toks2, Int32(1), Int32(2), Int32(8), grid_dim=1, block_dim=EW_THREADS)
    ctx.enqueue_function[amar_tok_remap[type_of(k_l), type_of(k_l)]](
        Toks, Toks2, Int32(K), grid_dim=1, block_dim=EW_THREADS)
    ctx.enqueue_function[amar_quantize_q8_rows[type_of(qa_l), type_of(qa_l), type_of(qs_l)]](
        A, Aq, As, Int32(QM), Int32(QK), grid_dim=(QM, QK // 32), block_dim=32)
    ctx.synchronize()
