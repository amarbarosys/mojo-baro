"""moe_down_q6k_grouped at the engine's prefill chunk shape (T = 1023 tokens, top-8, 256 experts,
E_FFN x H down rows), bench/moeprefill-protocol.md: byte-exact against the m=1 amar_moe_down_q6k
(one launch per token, then moe_down_combine in k order) on random Q6_K bytes, then per-launch
time as the median of ROUNDS blocks of CALLS launches on the grouped kernel alone.
  ./.venv/bin/mojo build bench/bench_moe_q6k_rows.mojo -I kernels -o .work/bench_moe_q6k_rows
  gpu-wait run --timeout 300 -- .work/bench_moe_q6k_rows
"""
from std.math import ceildiv
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from moe import amar_moe_down_q6k, N_EXP, TOPK, E_FFN, MOE_H, MOE_WAVES, MOE_THREADS
from moe_rows import moe_pairs_group, moe_down_q6k_grouped, moe_down_combine
from test_moe_rows import Lcg, fill_bytes, set_f16, view, differ

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime i32 = DType.int32
comptime u8 = DType.uint8
comptime T = 1023
comptime H = MOE_H
comptime MR = 8
comptime NG = T * TOPK // MR + N_EXP
comptime Q6_DROW = (E_FFN // 256) * 210
comptime Q6D = N_EXP * H * Q6_DROW
comptime ROUNDS = 11
comptime CALLS = 20
comptime hh_rows = row_major[T * TOPK, E_FFN]()
comptime hh_one = row_major[TOPK, E_FFN]()
comptime idx_rows = row_major[T, TOPK]()
comptime idx_one = row_major[TOPK]()
comptime x_rows = row_major[T, H]()
comptime x_one = row_major[H]()
comptime ge_layout = row_major[NG]()
comptime gp_layout = row_major[NG * MR]()
comptime gs_layout = row_major[2 * N_EXP]()
comptime dn_layout = row_major[T * TOPK, H]()


def grouped(ctx: DeviceContext, hh_d: DeviceBuffer[bf16], q6_d: DeviceBuffer[u8], ge_d: DeviceBuffer[i32], gp_d: DeviceBuffer[i32], dn_d: DeviceBuffer[f32]) raises:
    ctx.enqueue_function[moe_down_q6k_grouped[MR, E_FFN, type_of(hh_rows), type_of(ge_layout), type_of(gp_layout), type_of(dn_layout)]](
        view(ctx, hh_d, 0, T * TOPK * E_FFN, hh_rows), q6_d.unsafe_ptr(), view(ctx, ge_d, 0, NG, ge_layout),
        view(ctx, gp_d, 0, NG * MR, gp_layout), view(ctx, dn_d, 0, T * TOPK * H, dn_layout), Int32(H),
        grid_dim=(ceildiv(H, MOE_WAVES), NG), block_dim=MOE_THREADS)


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var g = Lcg(0x6B)
    var hh_h = ctx.enqueue_create_host_buffer[bf16](T * TOPK * E_FFN)
    var id_h = ctx.enqueue_create_host_buffer[i32](T * TOPK)
    var wt_h = ctx.enqueue_create_host_buffer[f32](T * TOPK)
    var q6_h = ctx.enqueue_create_host_buffer[u8](Q6D)
    ctx.synchronize()
    for i in range(T * TOPK * E_FFN):
        hh_h[i] = Scalar[bf16](Float32(Int(g.next() % 2001) - 1000) / 811.0)
    for t in range(T):
        for j in range(TOPK):
            id_h[t * TOPK + j] = Int32((Int(g.next() % 32) * TOPK + j) % N_EXP)
            wt_h[t * TOPK + j] = Float32(Int(g.next() % 1000) + 1) / 4000.0
    fill_bytes(g, q6_h.unsafe_ptr(), Q6D)
    for b in range(Q6D // 210):
        set_f16(q6_h.unsafe_ptr(), b * 210 + 208, g)
    var hh_d = ctx.enqueue_create_buffer[bf16](T * TOPK * E_FFN)
    var id_d = ctx.enqueue_create_buffer[i32](T * TOPK)
    var wt_d = ctx.enqueue_create_buffer[f32](T * TOPK)
    var q6_d = ctx.enqueue_create_buffer[u8](Q6D)
    var ge_d = ctx.enqueue_create_buffer[i32](NG)
    var gp_d = ctx.enqueue_create_buffer[i32](NG * MR)
    var gs_d = ctx.enqueue_create_buffer[i32](2 * N_EXP)
    var dn_d = ctx.enqueue_create_buffer[f32](T * TOPK * H)
    var dG_d = ctx.enqueue_create_buffer[f32](T * H)
    var dM_d = ctx.enqueue_create_buffer[f32](T * H)
    ctx.enqueue_copy(dst_buf=hh_d, src_buf=hh_h)
    ctx.enqueue_copy(dst_buf=id_d, src_buf=id_h)
    ctx.enqueue_copy(dst_buf=wt_d, src_buf=wt_h)
    ctx.enqueue_copy(dst_buf=q6_d, src_buf=q6_h)
    for t in range(T):
        ctx.enqueue_function[amar_moe_down_q6k[TOPK, E_FFN, type_of(hh_one), type_of(idx_one), type_of(idx_one), type_of(x_one)]](
            view(ctx, hh_d, t * TOPK * E_FFN, TOPK * E_FFN, hh_one), q6_d.unsafe_ptr(), view(ctx, id_d, t * TOPK, TOPK, idx_one),
            view(ctx, wt_d, t * TOPK, TOPK, idx_one), view(ctx, dM_d, t * H, H, x_one), Int32(H),
            grid_dim=ceildiv(H, MOE_WAVES), block_dim=MOE_THREADS)
    ctx.enqueue_function[moe_pairs_group[MR, T * TOPK, type_of(idx_rows), type_of(ge_layout), type_of(gp_layout), type_of(gs_layout)]](
        view(ctx, id_d, 0, T * TOPK, idx_rows), view(ctx, ge_d, 0, NG, ge_layout), view(ctx, gp_d, 0, NG * MR, gp_layout),
        view(ctx, gs_d, 0, 2 * N_EXP, gs_layout), Int32(T), Int32(NG), grid_dim=1, block_dim=N_EXP)

    grouped(ctx, hh_d, q6_d, ge_d, gp_d, dn_d)
    ctx.enqueue_function[moe_down_combine[TOPK, type_of(dn_layout), type_of(idx_rows), type_of(x_rows)]](
        view(ctx, dn_d, 0, T * TOPK * H, dn_layout), view(ctx, wt_d, 0, T * TOPK, idx_rows), view(ctx, dG_d, 0, T * H, x_rows),
        Int32(H), grid_dim=(ceildiv(H, 256), T), block_dim=256)
    var gh = ctx.enqueue_create_host_buffer[f32](T * H)
    var mh = ctx.enqueue_create_host_buffer[f32](T * H)
    ctx.enqueue_copy(dst_buf=gh, src_buf=dG_d)
    ctx.enqueue_copy(dst_buf=mh, src_buf=dM_d)
    ctx.synchronize()
    var small = 0
    var big = 0
    var maxrel = Float64(0)
    for i in range(T * H):
        var a = Float64(gh[i])
        var m = Float64(mh[i])
        if a != m:
            var rel = abs(a - m) / max(abs(m), 1e-30)
            if rel < 1e-5:
                small += 1
            else:
                big += 1
            if rel > maxrel:
                maxrel = rel
    print("diff elements: rounding-level", small, "larger", big, "max rel", maxrel, "first grouped/m1:", gh[0], mh[0], gh[1], mh[1], gh[H], mh[H])
    differ("down_q6k_grouped vs m=1", gh.unsafe_ptr().unsafe_bitcast[UInt8](), mh.unsafe_ptr().unsafe_bitcast[UInt8](), T * H * 4)

    var med: List[Float64] = []
    for _ in range(ROUNDS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(CALLS):
            grouped(ctx, hh_d, q6_d, ge_d, gp_d, dn_d)
        ctx.synchronize()
        med.append(Float64(perf_counter_ns() - t0) / 1e3 / CALLS)
    sort(med)
    print("shape T", T, "groups", NG, "H", H, "FFN", E_FFN, "experts", N_EXP, "MR", MR)
    print("down_q6k_grouped us/launch median", med[ROUNDS // 2], "min", med[0], "max", med[ROUNDS - 1], "of", ROUNDS, "x", CALLS)
    print("PASS bench_moe_q6k_rows")
