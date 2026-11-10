"""S2 round 7 microbench: routed-expert gate+up and down, old f32-dequant kernels vs the int8
sdot4 kernels, on the real layer-0 q4_K weights from .work/gguf (tools/moe-ref.py --gguf).
Cold-ish: each call routes to a fresh set of 8 experts (32 rotating sets, 302 MB of gate+up
and 151 MB of down never fit the 96 MB Infinity Cache). ROUNDS x CALLS, per-call medians.
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu import WARP_SIZE
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major
from elementwise import amar_quantize_q8_rows
from moe import (
    moe_gate_up_q4k_pack, amar_moe_down_q4k, moe_gate_up_q4k_dot, amar_moe_down_q4k_dot,
    N_EXP, TOPK, E_FFN, MOE_H, MOE_WAVES, MOE_THREADS,
)
from tio import load_into, median

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime i32 = DType.int32
comptime u8 = DType.uint8
comptime Q4K_BYTES = 144
comptime Q4_G_BYTES = N_EXP * E_FFN * (MOE_H // 256) * Q4K_BYTES
comptime Q4_D_BYTES = N_EXP * MOE_H * (E_FFN // 256) * Q4K_BYTES
comptime SETS = 32
comptime ROUNDS = 11
comptime CALLS = 100
comptime x_2 = row_major[1, MOE_H]()
comptime idx_1 = row_major[TOPK]()
comptime h_1 = row_major[TOPK * E_FFN]()
comptime h_2 = row_major[TOPK, E_FFN]()
comptime o_1 = row_major[MOE_H]()
comptime xq_l = row_major[1, MOE_H]()
comptime xs_l = row_major[1, MOE_H // 32]()
comptime hq_l = row_major[1, TOPK * E_FFN]()
comptime hs_l = row_major[1, TOPK * E_FFN // 32]()


comptime k_gu_old = moe_gate_up_q4k_pack[TOPK, E_FFN, type_of(x_2), type_of(idx_1), type_of(h_1), bf16]
comptime k_dn_old = amar_moe_down_q4k[TOPK, E_FFN, type_of(h_2), type_of(idx_1), type_of(idx_1), type_of(o_1)]
comptime k_qx = amar_quantize_q8_rows[type_of(x_2), type_of(xq_l), type_of(xs_l)]
comptime k_gu_new = moe_gate_up_q4k_dot[TOPK, E_FFN, MOE_H // 256, type_of(idx_1), type_of(h_1)]
comptime k_qh = amar_quantize_q8_rows[type_of(hq_l), type_of(hq_l), type_of(hs_l)]
comptime k_dn_new = amar_moe_down_q4k_dot[TOPK, E_FFN, E_FFN // 256, type_of(idx_1), type_of(idx_1), type_of(o_1)]


def run_old(ctx: DeviceContext, s: Int, mut xb_d: DeviceBuffer[bf16], mut w_d: DeviceBuffer[u8], mut idx_d: DeviceBuffer[i32],
            mut hb_d: DeviceBuffer[bf16], mut wd_d: DeviceBuffer[u8], mut wt_d: DeviceBuffer[f32], mut o_d: DeviceBuffer[f32]) raises:
    var idx_b = DeviceBuffer[i32](ctx, idx_d.unsafe_ptr().unsafe_offset((s % SETS) * TOPK), TOPK, owning=False)
    var idx = TileTensor(idx_b, idx_1)
    var xb2 = TileTensor(xb_d, x_2)
    var hb1 = TileTensor(hb_d, h_1)
    var hb2 = TileTensor(hb_d, h_2)
    var wt = TileTensor(wt_d, idx_1)
    var o1 = TileTensor(o_d, o_1)
    ctx.enqueue_function[k_gu_old](xb2, w_d.unsafe_ptr(), idx, hb1, Int32(MOE_H), Int32(Q4_G_BYTES), grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS)
    ctx.enqueue_function[k_dn_old](hb2, wd_d.unsafe_ptr(), idx, wt, o1, Int32(MOE_H), grid_dim=ceildiv(MOE_H, MOE_WAVES), block_dim=MOE_THREADS)


def run_new(ctx: DeviceContext, s: Int, mut xb_d: DeviceBuffer[bf16], mut w_d: DeviceBuffer[u8], mut idx_d: DeviceBuffer[i32],
            mut hb_d: DeviceBuffer[bf16], mut wd_d: DeviceBuffer[u8], mut wt_d: DeviceBuffer[f32], mut o_d: DeviceBuffer[f32],
            mut xq_d: DeviceBuffer[DType.int8], mut xs_d: DeviceBuffer[DType.float16], mut hq_d: DeviceBuffer[DType.int8], mut hs_d: DeviceBuffer[DType.float16]) raises:
    var idx_b = DeviceBuffer[i32](ctx, idx_d.unsafe_ptr().unsafe_offset((s % SETS) * TOPK), TOPK, owning=False)
    var idx = TileTensor(idx_b, idx_1)
    var xb2 = TileTensor(xb_d, x_2)
    var hb1 = TileTensor(hb_d, h_1)
    var wt = TileTensor(wt_d, idx_1)
    var o1 = TileTensor(o_d, o_1)
    ctx.enqueue_function[k_qx](xb2, TileTensor(xq_d, xq_l), TileTensor(xs_d, xs_l), Int32(1), Int32(MOE_H), grid_dim=(1, MOE_H // 32), block_dim=WARP_SIZE)
    ctx.enqueue_function[k_gu_new](xq_d.unsafe_ptr(), xs_d.unsafe_ptr(), w_d.unsafe_ptr(), idx, hb1, Int32(Q4_G_BYTES), grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS)
    ctx.enqueue_function[k_qh](TileTensor(hb_d, hq_l), TileTensor(hq_d, hq_l), TileTensor(hs_d, hs_l), Int32(1), Int32(TOPK * E_FFN), grid_dim=(1, TOPK * E_FFN // 32), block_dim=WARP_SIZE)
    ctx.enqueue_function[k_dn_new](hq_d.unsafe_ptr(), hs_d.unsafe_ptr(), wd_d.unsafe_ptr(), idx, wt, o1, Int32(MOE_H), grid_dim=ceildiv(MOE_H, MOE_WAVES), block_dim=MOE_THREADS)


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    comptime D = ".work/gguf/"
    var x_h = ctx.enqueue_create_host_buffer[f32](MOE_H)
    var w_h = ctx.enqueue_create_host_buffer[u8](2 * Q4_G_BYTES)
    var wd_h = ctx.enqueue_create_host_buffer[u8](Q4_D_BYTES)
    var idx_h = ctx.enqueue_create_host_buffer[i32](SETS * TOPK)
    var wt_h = ctx.enqueue_create_host_buffer[f32](TOPK)
    ctx.synchronize()
    load_into(D + "moe_x.bin", x_h.unsafe_ptr().unsafe_bitcast[UInt8](), MOE_H * 4)
    load_into(D + "moe_q4k_gate.bin", w_h.unsafe_ptr(), Q4_G_BYTES)
    load_into(D + "moe_q4k_up.bin", w_h.unsafe_ptr().unsafe_offset(Q4_G_BYTES), Q4_G_BYTES)
    load_into(D + "moe_q4k_down.bin", wd_h.unsafe_ptr(), Q4_D_BYTES)
    for s in range(SETS):
        for j in range(TOPK):
            idx_h[s * TOPK + j] = Int32((s * 8 + j * 37 + s * j) % N_EXP)
    for j in range(TOPK):
        wt_h[j] = Float32(0.125)
    var xb_h = ctx.enqueue_create_host_buffer[bf16](MOE_H)
    ctx.synchronize()
    for i in range(MOE_H):
        xb_h[i] = x_h[i].cast[bf16]()
    var xb_d = ctx.enqueue_create_buffer[bf16](MOE_H)
    var w_d = ctx.enqueue_create_buffer[u8](2 * Q4_G_BYTES)
    var wd_d = ctx.enqueue_create_buffer[u8](Q4_D_BYTES)
    var idx_d = ctx.enqueue_create_buffer[i32](SETS * TOPK)
    var wt_d = ctx.enqueue_create_buffer[f32](TOPK)
    var hb_d = ctx.enqueue_create_buffer[bf16](TOPK * E_FFN)
    var o_d = ctx.enqueue_create_buffer[f32](MOE_H)
    var xq_d = ctx.enqueue_create_buffer[DType.int8](MOE_H)
    var xs_d = ctx.enqueue_create_buffer[DType.float16](MOE_H // 32)
    var hq_d = ctx.enqueue_create_buffer[DType.int8](TOPK * E_FFN)
    var hs_d = ctx.enqueue_create_buffer[DType.float16](TOPK * E_FFN // 32)
    ctx.enqueue_copy(dst_buf=xb_d, src_buf=xb_h)
    ctx.enqueue_copy(dst_buf=w_d, src_buf=w_h)
    ctx.enqueue_copy(dst_buf=wd_d, src_buf=wd_h)
    ctx.enqueue_copy(dst_buf=idx_d, src_buf=idx_h)
    ctx.enqueue_copy(dst_buf=wt_d, src_buf=wt_h)
    ctx.synchronize()

    for _ in range(20):
        run_old(ctx, 0, xb_d, w_d, idx_d, hb_d, wd_d, wt_d, o_d)
        run_new(ctx, 1, xb_d, w_d, idx_d, hb_d, wd_d, wt_d, o_d, xq_d, xs_d, hq_d, hs_d)
    ctx.synchronize()
    var olds = List[Float64]()
    var news = List[Float64]()
    for r in range(ROUNDS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for c in range(CALLS):
            run_old(ctx, r * CALLS + c, xb_d, w_d, idx_d, hb_d, wd_d, wt_d, o_d)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        for c in range(CALLS):
            run_new(ctx, r * CALLS + c, xb_d, w_d, idx_d, hb_d, wd_d, wt_d, o_d, xq_d, xs_d, hq_d, hs_d)
        ctx.synchronize()
        var t2 = perf_counter_ns()
        olds.append(Float64(t1 - t0) / 1e3 / CALLS)
        news.append(Float64(t2 - t1) / 1e3 / CALLS)
    var mo = median(olds.copy())
    var mn = median(news.copy())
    var bytes_call = Float64(TOPK * E_FFN * (MOE_H // 256) * Q4K_BYTES * 2 + TOPK * MOE_H * (E_FFN // 256) * Q4K_BYTES)
    print("bench_moe_dot: gate+up+down per call, ROUNDS", ROUNDS, "x CALLS", CALLS, " routed bytes per call", Int(bytes_call))
    print("old (f32 dequant, 2 launches):  median", mo, "us  ->", bytes_call / mo / 1e3, "GB/s")
    print("new (int8 sdot4, 4 launches):   median", mn, "us  ->", bytes_call / mn / 1e3, "GB/s")
    print("ratio old/new", mo / mn)
