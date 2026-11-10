"""Parity: one decode token through the qwen35moe sparse-MoE block on GPU vs
the numpy reference (tools/moe-ref.py implementing transformers
Qwen3_5MoeSparseMoeBlock).

Gates: the eight selected expert ids EXACT, and routing weights, routed output,
shared-expert output and the block output y all rel < 5e-3. routed and shared
are checked separately so a failure localises to the routed path, the shared
path, or the router itself.

Run tools/moe-ref.py first to write the fixtures into .work/gguf/.
"""
from std.math import ceildiv
from std.memory import alloc, unsafe_memcpy
from std.os import getenv
from std.sys import exit, has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from tio import load_exact
from matmul_skinny import amar_matmul_skinny_m1_row, ROW_WAVES, ROW_THREADS
from ssm import amar_cast_bf16
from elementwise import amar_quantize_q8_rows
from max.gpu import WARP_SIZE
from moe import (
    moe_gate_up_q4k_dot, amar_moe_down_q4k_dot,
    amar_moe_router_top8, amar_moe_sig_gate, amar_moe_gate_up, amar_moe_down,
    amar_moe_gate_up_q4k, amar_moe_down_q4k, q4k_decode_vector,
    moe_gate_up_q4k_zc, amar_moe_down_q4k_zc,
    N_EXP, TOPK, E_FFN, SH_FFN, MOE_H, MOE_WAVES, MOE_THREADS,
)

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime i32 = DType.int32
comptime u8 = DType.uint8

comptime x_1 = row_major[MOE_H]()
comptime x_2 = row_major[1, MOE_H]()
comptime wr_2 = row_major[N_EXP, MOE_H]()
comptime lg_1 = row_major[N_EXP]()
comptime idx_1 = row_major[TOPK]()
comptime wt_1 = row_major[TOPK]()
comptime one_1 = row_major[1]()

comptime wg_2 = row_major[N_EXP * E_FFN, MOE_H]()
comptime wd_2 = row_major[N_EXP * MOE_H, E_FFN]()
comptime h_1 = row_major[TOPK * E_FFN]()
comptime h_2 = row_major[TOPK, E_FFN]()

comptime wgs_2 = row_major[SH_FFN, MOE_H]()
comptime wds_2 = row_major[MOE_H, SH_FFN]()
comptime hs_1 = row_major[SH_FFN]()
comptime hs_2 = row_major[1, SH_FFN]()

comptime o_1 = row_major[MOE_H]()


def check(name: String, got: MutPointer[Float32, MutUntrackedOrigin],
          want: MutPointer[Float32, MutUntrackedOrigin], n: Int,
          gate: Float64 = 5e-3) raises:
    var worst = Float64(0)
    var wi = 0
    for i in range(n):
        var e = abs(Float64(got[unsafe_offset=i]) - Float64(want[unsafe_offset=i]))
        var rel = e / (abs(Float64(want[unsafe_offset=i])) + 1e-2)
        if rel > worst:
            worst = rel
            wi = i
    print(
        name, "max_rel:", worst, "at", wi,
        "got", got[unsafe_offset=wi], "want", want[unsafe_offset=wi],
    )
    if worst > gate:
        raise Error("parity failure: " + name)


def bytes_differ(
    a: MutPointer[UInt8, MutUntrackedOrigin], b: MutPointer[UInt8, MutUntrackedOrigin], n: Int
) -> Int:
    var bad = 0
    for i in range(n):
        if a[unsafe_offset=i] != b[unsafe_offset=i]:
            bad += 1
    return bad


def check_i32(name: String, got: MutPointer[Int32, MutUntrackedOrigin],
              want: MutPointer[Int32, MutUntrackedOrigin], n: Int) raises:
    for i in range(n):
        if got[unsafe_offset=i] != want[unsafe_offset=i]:
            print(name, "mismatch at", i,
                  "got", got[unsafe_offset=i], "want", want[unsafe_offset=i])
            raise Error("parity failure: " + name)
    print(name, "exact over", n)


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    comptime D = ".work/gguf/"

    comptime N_EW = N_EXP * E_FFN * MOE_H
    comptime N_DW = N_EXP * MOE_H * E_FFN
    comptime Q4K_BYTES = 144
    comptime Q4K = 256
    comptime Q4_G_BYTES = N_EXP * E_FFN * (MOE_H // 256) * Q4K_BYTES
    comptime Q4_D_BYTES = N_EXP * MOE_H * (E_FFN // 256) * Q4K_BYTES

    var x_h = ctx.enqueue_create_host_buffer[f32](MOE_H)
    var wr_h = ctx.enqueue_create_host_buffer[f32](N_EXP * MOE_H)
    var wsg_h = ctx.enqueue_create_host_buffer[f32](MOE_H)
    var wg_h = ctx.enqueue_create_host_buffer[bf16](N_EW)
    var wu_h = ctx.enqueue_create_host_buffer[bf16](N_EW)
    var wd_h = ctx.enqueue_create_host_buffer[bf16](N_DW)
    var qg_h = ctx.enqueue_create_host_buffer[u8](Q4_G_BYTES)
    var qu_h = ctx.enqueue_create_host_buffer[u8](Q4_G_BYTES)
    var qd_h = ctx.enqueue_create_host_buffer[u8](Q4_D_BYTES)
    var wgs_h = ctx.enqueue_create_host_buffer[bf16](SH_FFN * MOE_H)
    var wus_h = ctx.enqueue_create_host_buffer[bf16](SH_FFN * MOE_H)
    var wds_h = ctx.enqueue_create_host_buffer[bf16](MOE_H * SH_FFN)
    var idx0_h = ctx.enqueue_create_host_buffer[i32](1)
    ctx.synchronize()

    load_exact(D + "moe_x.bin", x_h.unsafe_ptr().unsafe_bitcast[UInt8](), MOE_H * 4)
    load_exact(D + "moe_wr.bin", wr_h.unsafe_ptr().unsafe_bitcast[UInt8](), N_EXP * MOE_H * 4)
    load_exact(D + "moe_wsg.bin", wsg_h.unsafe_ptr().unsafe_bitcast[UInt8](), MOE_H * 4)
    load_exact(D + "moe_wg.bin", wg_h.unsafe_ptr().unsafe_bitcast[UInt8](), N_EW * 2)
    load_exact(D + "moe_wu.bin", wu_h.unsafe_ptr().unsafe_bitcast[UInt8](), N_EW * 2)
    load_exact(D + "moe_wd.bin", wd_h.unsafe_ptr().unsafe_bitcast[UInt8](), N_DW * 2)
    load_exact(D + "moe_q4k_gate.bin", qg_h.unsafe_ptr(), Q4_G_BYTES)
    load_exact(D + "moe_q4k_up.bin", qu_h.unsafe_ptr(), Q4_G_BYTES)
    load_exact(D + "moe_q4k_down.bin", qd_h.unsafe_ptr(), Q4_D_BYTES)
    load_exact(D + "moe_wgs.bin", wgs_h.unsafe_ptr().unsafe_bitcast[UInt8](), SH_FFN * MOE_H * 2)
    load_exact(D + "moe_wus.bin", wus_h.unsafe_ptr().unsafe_bitcast[UInt8](), SH_FFN * MOE_H * 2)
    load_exact(D + "moe_wds.bin", wds_h.unsafe_ptr().unsafe_bitcast[UInt8](), MOE_H * SH_FFN * 2)
    idx0_h[0] = 0

    var x_d = ctx.enqueue_create_buffer[f32](MOE_H)
    var xb_d = ctx.enqueue_create_buffer[bf16](MOE_H)
    var wr_d = ctx.enqueue_create_buffer[f32](N_EXP * MOE_H)
    var wsg_d = ctx.enqueue_create_buffer[f32](MOE_H)
    var lg_d = ctx.enqueue_create_buffer[f32](N_EXP)
    var idx_d = ctx.enqueue_create_buffer[i32](TOPK)
    var wt_d = ctx.enqueue_create_buffer[f32](TOPK)
    var sg_d = ctx.enqueue_create_buffer[f32](1)
    var idx0_d = ctx.enqueue_create_buffer[i32](1)
    var wg_d = ctx.enqueue_create_buffer[bf16](N_EW)
    var wu_d = ctx.enqueue_create_buffer[bf16](N_EW)
    var wd_d = ctx.enqueue_create_buffer[bf16](N_DW)
    var qg_d = ctx.enqueue_create_buffer[u8](Q4_G_BYTES)
    var qu_d = ctx.enqueue_create_buffer[u8](Q4_G_BYTES)
    var qd_d = ctx.enqueue_create_buffer[u8](Q4_D_BYTES)
    var wgs_d = ctx.enqueue_create_buffer[bf16](SH_FFN * MOE_H)
    var wus_d = ctx.enqueue_create_buffer[bf16](SH_FFN * MOE_H)
    var wds_d = ctx.enqueue_create_buffer[bf16](MOE_H * SH_FFN)
    var h_d = ctx.enqueue_create_buffer[f32](TOPK * E_FFN)
    var hb_d = ctx.enqueue_create_buffer[bf16](TOPK * E_FFN)
    var hs_d = ctx.enqueue_create_buffer[f32](SH_FFN)
    var hsb_d = ctx.enqueue_create_buffer[bf16](SH_FFN)
    var routed_d = ctx.enqueue_create_buffer[f32](MOE_H)
    var shared_d = ctx.enqueue_create_buffer[f32](MOE_H)
    var routed_q4_d = ctx.enqueue_create_buffer[f32](MOE_H)
    var qv_d = ctx.enqueue_create_buffer[f32](Q4K)

    ctx.enqueue_copy(dst_buf=x_d, src_buf=x_h)
    ctx.enqueue_copy(dst_buf=wr_d, src_buf=wr_h)
    ctx.enqueue_copy(dst_buf=wsg_d, src_buf=wsg_h)
    ctx.enqueue_copy(dst_buf=wg_d, src_buf=wg_h)
    ctx.enqueue_copy(dst_buf=wu_d, src_buf=wu_h)
    ctx.enqueue_copy(dst_buf=wd_d, src_buf=wd_h)
    ctx.enqueue_copy(dst_buf=qg_d, src_buf=qg_h)
    ctx.enqueue_copy(dst_buf=qu_d, src_buf=qu_h)
    ctx.enqueue_copy(dst_buf=qd_d, src_buf=qd_h)
    ctx.enqueue_copy(dst_buf=wgs_d, src_buf=wgs_h)
    ctx.enqueue_copy(dst_buf=wus_d, src_buf=wus_h)
    ctx.enqueue_copy(dst_buf=wds_d, src_buf=wds_h)
    ctx.enqueue_copy(dst_buf=idx0_d, src_buf=idx0_h)
    ctx.synchronize()

    var x1 = TileTensor(x_d, x_1)
    var x2 = TileTensor(x_d, x_2)
    var xb1 = TileTensor(xb_d, x_1)
    var xb2 = TileTensor(xb_d, x_2)
    var wr = TileTensor(wr_d, wr_2)
    var wsg = TileTensor(wsg_d, x_1)
    var lg = TileTensor(lg_d, lg_1)
    var idx = TileTensor(idx_d, idx_1)
    var wt = TileTensor(wt_d, wt_1)
    var sg = TileTensor(sg_d, one_1)
    var idx0 = TileTensor(idx0_d, one_1)
    var wg = TileTensor(wg_d, wg_2)
    var wu = TileTensor(wu_d, wg_2)
    var wd = TileTensor(wd_d, wd_2)
    var wgs = TileTensor(wgs_d, wgs_2)
    var wus = TileTensor(wus_d, wgs_2)
    var wds = TileTensor(wds_d, wds_2)
    var h1 = TileTensor(h_d, h_1)
    var hb2 = TileTensor(hb_d, h_2)
    var hs1 = TileTensor(hs_d, hs_1)
    var hsb2 = TileTensor(hsb_d, hs_2)
    var routed = TileTensor(routed_d, o_1)
    var shared = TileTensor(shared_d, o_1)
    var routed_q4 = TileTensor(routed_q4_d, o_1)
    comptime qv_1 = row_major[Q4K]()
    var qv = TileTensor(qv_d, qv_1)

    comptime k_cast = amar_cast_bf16[type_of(x_1), type_of(x_1)]
    ctx.enqueue_function[k_cast](
        x1, xb1, Int32(MOE_H),
        grid_dim=ceildiv(MOE_H, 256), block_dim=256,
    )

    comptime k_router = amar_matmul_skinny_m1_row[
        f32, 2, type_of(x_2), type_of(wr_2), type_of(lg_1)
    ]
    ctx.enqueue_function[k_router](
        x2, wr, lg, Int32(N_EXP), Int32(MOE_H),
        grid_dim=ceildiv(N_EXP, ROW_WAVES), block_dim=ROW_THREADS,
    )

    comptime k_top8 = amar_moe_router_top8[
        type_of(lg_1), type_of(idx_1), type_of(wt_1)
    ]
    ctx.enqueue_function[k_top8](lg, idx, wt, grid_dim=1, block_dim=32)

    comptime k_sig = amar_moe_sig_gate[
        type_of(x_1), type_of(x_1), type_of(one_1)
    ]
    ctx.enqueue_function[k_sig](x1, wsg, sg, Int32(MOE_H), grid_dim=1, block_dim=32)

    comptime k_gu = amar_moe_gate_up[
        TOPK, E_FFN, type_of(x_2), type_of(wg_2), type_of(wg_2),
        type_of(idx_1), type_of(h_1),
    ]
    ctx.enqueue_function[k_gu](
        xb2, wg, wu, idx, h1, Int32(MOE_H),
        grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS,
    )

    comptime k_cast_h = amar_cast_bf16[type_of(h_1), type_of(h_1)]
    ctx.enqueue_function[k_cast_h](
        TileTensor(h_d, h_1), TileTensor(hb_d, h_1), Int32(TOPK * E_FFN),
        grid_dim=ceildiv(TOPK * E_FFN, 256), block_dim=256,
    )

    comptime k_down = amar_moe_down[
        TOPK, E_FFN, type_of(h_2), type_of(wd_2), type_of(idx_1),
        type_of(wt_1), type_of(o_1),
    ]
    ctx.enqueue_function[k_down](
        hb2, wd, idx, wt, routed, Int32(MOE_H),
        grid_dim=ceildiv(MOE_H, MOE_WAVES), block_dim=MOE_THREADS,
    )

    comptime k_gu_s = amar_moe_gate_up[
        1, SH_FFN, type_of(x_2), type_of(wgs_2), type_of(wgs_2),
        type_of(one_1), type_of(hs_1),
    ]
    ctx.enqueue_function[k_gu_s](
        xb2, wgs, wus, idx0, hs1, Int32(MOE_H),
        grid_dim=ceildiv(SH_FFN, MOE_WAVES), block_dim=MOE_THREADS,
    )

    comptime k_cast_hs = amar_cast_bf16[type_of(hs_1), type_of(hs_1)]
    ctx.enqueue_function[k_cast_hs](
        TileTensor(hs_d, hs_1), TileTensor(hsb_d, hs_1), Int32(SH_FFN),
        grid_dim=ceildiv(SH_FFN, 256), block_dim=256,
    )

    comptime k_down_s = amar_moe_down[
        1, SH_FFN, type_of(hs_2), type_of(wds_2), type_of(one_1),
        type_of(one_1), type_of(o_1),
    ]
    ctx.enqueue_function[k_down_s](
        hsb2, wds, idx0, sg, shared, Int32(MOE_H),
        grid_dim=ceildiv(MOE_H, MOE_WAVES), block_dim=MOE_THREADS,
    )
    ctx.synchronize()

    comptime k_q4v = q4k_decode_vector[type_of(qv_1)]
    ctx.enqueue_function[k_q4v](
        qg_d.unsafe_ptr(), qv, grid_dim=1, block_dim=256,
    )
    var qv_got = ctx.enqueue_create_host_buffer[f32](Q4K)
    ctx.enqueue_copy(dst_buf=qv_got, src_buf=qv_d)
    ctx.synchronize()
    var qv_ref = alloc[Float32](Q4K)
    load_exact(D + "moe_q4k_vector.bin", qv_ref.unsafe_bitcast[UInt8](), Q4K * 4)
    check("q4k decoder", qv_got.unsafe_ptr(), qv_ref, Q4K, 1e-6)

    var idx_got = ctx.enqueue_create_host_buffer[i32](TOPK)
    var wt_got = ctx.enqueue_create_host_buffer[f32](TOPK)
    var routed_got = ctx.enqueue_create_host_buffer[f32](MOE_H)
    var shared_got = ctx.enqueue_create_host_buffer[f32](MOE_H)
    ctx.enqueue_copy(dst_buf=idx_got, src_buf=idx_d)
    ctx.enqueue_copy(dst_buf=wt_got, src_buf=wt_d)
    ctx.enqueue_copy(dst_buf=routed_got, src_buf=routed_d)
    ctx.enqueue_copy(dst_buf=shared_got, src_buf=shared_d)
    ctx.synchronize()

    var idx_ref = alloc[Int32](TOPK)
    var wt_ref = alloc[Float32](TOPK)
    var routed_ref = alloc[Float32](MOE_H)
    var shared_ref = alloc[Float32](MOE_H)
    var y_ref = alloc[Float32](MOE_H)
    load_exact(D + "moe_idx_ref.bin", idx_ref.unsafe_bitcast[UInt8](), TOPK * 4)
    load_exact(D + "moe_w_ref.bin", wt_ref.unsafe_bitcast[UInt8](), TOPK * 4)
    load_exact(D + "moe_routed_ref.bin", routed_ref.unsafe_bitcast[UInt8](), MOE_H * 4)
    load_exact(D + "moe_shared_ref.bin", shared_ref.unsafe_bitcast[UInt8](), MOE_H * 4)
    load_exact(D + "moe_y_ref.bin", y_ref.unsafe_bitcast[UInt8](), MOE_H * 4)

    var y_got = alloc[Float32](MOE_H)
    for i in range(MOE_H):
        y_got[unsafe_offset=i] = (
            routed_got.unsafe_ptr()[unsafe_offset=i]
            + shared_got.unsafe_ptr()[unsafe_offset=i]
        )

    check_i32("expert_ids", idx_got.unsafe_ptr(), idx_ref, TOPK)
    check("routing_weights", wt_got.unsafe_ptr(), wt_ref, TOPK, 1e-4)
    check("routed", routed_got.unsafe_ptr(), routed_ref, MOE_H)
    check("shared", shared_got.unsafe_ptr(), shared_ref, MOE_H)
    check("y", y_got, y_ref, MOE_H)

    comptime k_gu_q4 = amar_moe_gate_up_q4k[
        TOPK, E_FFN, type_of(x_2), type_of(idx_1), type_of(h_1)
    ]
    ctx.enqueue_function[k_gu_q4](
        xb2, qg_d.unsafe_ptr(), qu_d.unsafe_ptr(), idx, h1, Int32(MOE_H),
        grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS,
    )
    ctx.enqueue_function[k_cast_h](
        TileTensor(h_d, h_1), TileTensor(hb_d, h_1), Int32(TOPK * E_FFN),
        grid_dim=ceildiv(TOPK * E_FFN, 256), block_dim=256,
    )
    comptime k_down_q4 = amar_moe_down_q4k[
        TOPK, E_FFN, type_of(h_2), type_of(idx_1), type_of(wt_1), type_of(o_1)
    ]
    ctx.enqueue_function[k_down_q4](
        hb2, qd_d.unsafe_ptr(), idx, wt, routed_q4, Int32(MOE_H),
        grid_dim=ceildiv(MOE_H, MOE_WAVES), block_dim=MOE_THREADS,
    )
    ctx.synchronize()
    var routed_q4_got = ctx.enqueue_create_host_buffer[f32](MOE_H)
    ctx.enqueue_copy(dst_buf=routed_q4_got, src_buf=routed_q4_d)
    ctx.synchronize()
    var y_q4 = alloc[Float32](MOE_H)
    for i in range(MOE_H):
        y_q4[unsafe_offset=i] = (
            routed_q4_got[i] + shared_got[i]
        )
    check("q4k routed", routed_q4_got.unsafe_ptr(), routed_ref, MOE_H)

    # S2 round 7: int8 activations (q8 per 32) x q4_K weights through sdot4, weights read once
    # per wave with every block's loads issued before the arithmetic. Not bit-exact with the
    # f32 dequant kernels (int8 activations), so the gate is the f32 reference at 1e-2 here and
    # the 20-prompt agreement band in the round.
    comptime xq_l = row_major[1, MOE_H]()
    comptime xs_l = row_major[1, MOE_H // 32]()
    comptime hq_l = row_major[1, TOPK * E_FFN]()
    comptime hs_l = row_major[1, TOPK * E_FFN // 32]()
    var xq_d = ctx.enqueue_create_buffer[DType.int8](MOE_H)
    var xs_d = ctx.enqueue_create_buffer[DType.float16](MOE_H // 32)
    var hq8_d = ctx.enqueue_create_buffer[DType.int8](TOPK * E_FFN)
    var hs8_d = ctx.enqueue_create_buffer[DType.float16](TOPK * E_FFN // 32)
    var hb1_d = ctx.enqueue_create_buffer[bf16](TOPK * E_FFN)
    var routed_dot_d = ctx.enqueue_create_buffer[f32](MOE_H)
    comptime k_qx = amar_quantize_q8_rows[type_of(x_2), type_of(xq_l), type_of(xs_l)]
    ctx.enqueue_function[k_qx](xb2, TileTensor(xq_d, xq_l), TileTensor(xs_d, xs_l), Int32(1), Int32(MOE_H), grid_dim=(1, MOE_H // 32), block_dim=WARP_SIZE)
    comptime k_gu_dot = moe_gate_up_q4k_dot[TOPK, E_FFN, MOE_H // 256, type_of(idx_1), type_of(h_1)]
    var up_off = Int32(Int(qu_d.unsafe_ptr()) - Int(qg_d.unsafe_ptr()))
    ctx.enqueue_function[k_gu_dot](
        xq_d.unsafe_ptr(), xs_d.unsafe_ptr(), qg_d.unsafe_ptr(), idx, TileTensor(hb1_d, h_1), up_off,
        grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS,
    )
    comptime k_qh = amar_quantize_q8_rows[type_of(hq_l), type_of(hq_l), type_of(hs_l)]
    ctx.enqueue_function[k_qh](TileTensor(hb1_d, hq_l), TileTensor(hq8_d, hq_l), TileTensor(hs8_d, hs_l), Int32(1), Int32(TOPK * E_FFN), grid_dim=(1, TOPK * E_FFN // 32), block_dim=WARP_SIZE)
    comptime k_down_dot = amar_moe_down_q4k_dot[TOPK, E_FFN, E_FFN // 256, type_of(idx_1), type_of(wt_1), type_of(o_1)]
    ctx.enqueue_function[k_down_dot](
        hq8_d.unsafe_ptr(), hs8_d.unsafe_ptr(), qd_d.unsafe_ptr(), idx, wt, TileTensor(routed_dot_d, o_1), Int32(MOE_H),
        grid_dim=ceildiv(MOE_H, MOE_WAVES), block_dim=MOE_THREADS,
    )
    ctx.synchronize()
    var routed_dot_got = ctx.enqueue_create_host_buffer[f32](MOE_H)
    ctx.enqueue_copy(dst_buf=routed_dot_got, src_buf=routed_dot_d)
    ctx.synchronize()
    check("q4k dot4 routed (int8 activations)", routed_dot_got.unsafe_ptr(), routed_ref, MOE_H, 1e-2)
    check("q4k y", y_q4, y_ref, MOE_H)
    print("PASS: raw Q4_K routed experts match the bf16-rounded GGUF oracle")

    # Zero-copy arm (bench/moe-tier-protocol.md): the same eight experts read
    # from the pinned host buffers by the _zc kernels, which fill a slot cache
    # as they go. Three checks: the routed output is bit-identical to the
    # device-resident q4k arm; every filled slot equals its source bytes; a
    # hit-only relaunch from the filled cache reproduces the output again.
    comptime E_G = E_FFN * (MOE_H // 256) * Q4K_BYTES
    comptime E_D = MOE_H * (E_FFN // 256) * Q4K_BYTES
    var hg_h = ctx.enqueue_create_host_buffer[u8](2 * Q4_G_BYTES)
    ctx.synchronize()
    unsafe_memcpy(dest=hg_h.unsafe_ptr(), src=qg_h.unsafe_ptr(), count=Q4_G_BYTES)
    unsafe_memcpy(dest=hg_h.unsafe_ptr().unsafe_offset(Q4_G_BYTES), src=qu_h.unsafe_ptr(), count=Q4_G_BYTES)
    var cg_d = ctx.enqueue_create_buffer[u8](2 * TOPK * E_G)
    var cd_d = ctx.enqueue_create_buffer[u8](TOPK * E_D)
    var slots_h = ctx.enqueue_create_host_buffer[i32](TOPK)
    var hoste_h = ctx.enqueue_create_host_buffer[i32](TOPK)
    var hits_h = ctx.enqueue_create_host_buffer[i32](TOPK)
    ctx.synchronize()
    for j in range(TOPK):
        slots_h[j] = Int32(j)
        hoste_h[j] = idx_got[j]
        hits_h[j] = Int32(-1)
    var slots_d = ctx.enqueue_create_buffer[i32](TOPK)
    var hoste_d = ctx.enqueue_create_buffer[i32](TOPK)
    var hits_d = ctx.enqueue_create_buffer[i32](TOPK)
    ctx.enqueue_copy(dst_buf=slots_d, src_buf=slots_h)
    ctx.enqueue_copy(dst_buf=hoste_d, src_buf=hoste_h)
    ctx.enqueue_copy(dst_buf=hits_d, src_buf=hits_h)
    var slots = TileTensor(slots_d, idx_1)
    var hoste = TileTensor(hoste_d, idx_1)
    var hits = TileTensor(hits_d, idx_1)
    var routed_zc_d = ctx.enqueue_create_buffer[f32](MOE_H)
    var routed_zc = TileTensor(routed_zc_d, o_1)
    comptime k_gu_zc = moe_gate_up_q4k_zc[
        TOPK, E_FFN, type_of(x_2), type_of(idx_1), type_of(h_1)
    ]
    comptime k_down_zc = amar_moe_down_q4k_zc[
        TOPK, E_FFN, type_of(h_2), type_of(idx_1), type_of(wt_1), type_of(o_1)
    ]
    ctx.enqueue_function[k_gu_zc](
        xb2, cg_d.unsafe_ptr(), slots, hoste, hg_h.unsafe_ptr(), h1, Int32(MOE_H),
        Int32(TOPK * E_G), Int32(Q4_G_BYTES),
        grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS,
    )
    ctx.enqueue_function[k_cast_h](
        TileTensor(h_d, h_1), TileTensor(hb_d, h_1), Int32(TOPK * E_FFN),
        grid_dim=ceildiv(TOPK * E_FFN, 256), block_dim=256,
    )
    ctx.enqueue_function[k_down_zc](
        hb2, cd_d.unsafe_ptr(), slots, hoste, qd_h.unsafe_ptr(), wt, routed_zc, Int32(MOE_H),
        grid_dim=ceildiv(MOE_H, MOE_WAVES), block_dim=MOE_THREADS,
    )
    ctx.synchronize()
    var routed_zc_got = ctx.enqueue_create_host_buffer[f32](MOE_H)
    ctx.enqueue_copy(dst_buf=routed_zc_got, src_buf=routed_zc_d)
    var cg_got = ctx.enqueue_create_host_buffer[u8](2 * TOPK * E_G)
    var cd_got = ctx.enqueue_create_host_buffer[u8](TOPK * E_D)
    ctx.enqueue_copy(dst_buf=cg_got, src_buf=cg_d)
    ctx.enqueue_copy(dst_buf=cd_got, src_buf=cd_d)
    ctx.synchronize()
    var zc_bad = 0
    for i in range(MOE_H):
        if routed_zc_got[i] != routed_q4_got[i]:
            zc_bad += 1
    var fill_bad = 0
    for j in range(TOPK):
        var e = Int(hoste_h[j])
        fill_bad += bytes_differ(cg_got.unsafe_ptr().unsafe_offset(j * E_G), qg_h.unsafe_ptr().unsafe_offset(e * E_G), E_G)
        fill_bad += bytes_differ(cg_got.unsafe_ptr().unsafe_offset((TOPK + j) * E_G), qu_h.unsafe_ptr().unsafe_offset(e * E_G), E_G)
        fill_bad += bytes_differ(cd_got.unsafe_ptr().unsafe_offset(j * E_D), qd_h.unsafe_ptr().unsafe_offset(e * E_D), E_D)
    ctx.enqueue_function[k_gu_zc](
        xb2, cg_d.unsafe_ptr(), slots, hits, hg_h.unsafe_ptr(), h1, Int32(MOE_H),
        Int32(TOPK * E_G), Int32(Q4_G_BYTES),
        grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS,
    )
    ctx.enqueue_function[k_cast_h](
        TileTensor(h_d, h_1), TileTensor(hb_d, h_1), Int32(TOPK * E_FFN),
        grid_dim=ceildiv(TOPK * E_FFN, 256), block_dim=256,
    )
    ctx.enqueue_function[k_down_zc](
        hb2, cd_d.unsafe_ptr(), slots, hits, qd_h.unsafe_ptr(), wt, routed_zc, Int32(MOE_H),
        grid_dim=ceildiv(MOE_H, MOE_WAVES), block_dim=MOE_THREADS,
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=routed_zc_got, src_buf=routed_zc_d)
    ctx.synchronize()
    var hit_bad = 0
    for i in range(MOE_H):
        if routed_zc_got[i] != routed_q4_got[i]:
            hit_bad += 1
    print("zc: routed mismatches", zc_bad, " fill byte mismatches", fill_bad, " hit-relaunch mismatches", hit_bad)
    if zc_bad != 0 or fill_bad != 0 or hit_bad != 0:
        print("FAIL: zero-copy arm differs from the device-resident q4k arm")
        exit(1)
    print("PASS: zero-copy q4k experts read from pinned host memory match, and the slots are filled byte-exact")

    # BARO_TIER_CPU (bench/moe-tier-cpu-protocol.md round 2): HE = -2 marks an
    # expert the CPU computes; both _zc kernels skip it. Skipping must equal the
    # same hit-only launch with that expert's router weight set to 0, bit for bit
    # (adding 0 * dot leaves every partial sum unchanged).
    comptime J0 = 3
    var skip_h = ctx.enqueue_create_host_buffer[i32](TOPK)
    var wt_save = ctx.enqueue_create_host_buffer[f32](TOPK)
    ctx.enqueue_copy(dst_buf=wt_save, src_buf=wt_d)
    ctx.synchronize()
    for j in range(TOPK):
        skip_h[j] = Int32(-2) if j == J0 else Int32(-1)
    var skip_d = ctx.enqueue_create_buffer[i32](TOPK)
    ctx.enqueue_copy(dst_buf=skip_d, src_buf=skip_h)
    var skip = TileTensor(skip_d, idx_1)
    var skip_got = ctx.enqueue_create_host_buffer[f32](MOE_H)
    for arm in range(2):
        if arm == 1:
            var wz = ctx.enqueue_create_host_buffer[f32](TOPK)
            ctx.synchronize()
            for j in range(TOPK):
                wz[j] = Float32(0) if j == J0 else wt_save[j]
            ctx.enqueue_copy(dst_buf=wt_d, src_buf=wz)
            ctx.enqueue_copy(dst_buf=skip_d, src_buf=hits_h)
        ctx.enqueue_function[k_gu_zc](
            xb2, cg_d.unsafe_ptr(), slots, skip, hg_h.unsafe_ptr(), h1, Int32(MOE_H),
            Int32(TOPK * E_G), Int32(Q4_G_BYTES),
            grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS,
        )
        ctx.enqueue_function[k_cast_h](
            TileTensor(h_d, h_1), TileTensor(hb_d, h_1), Int32(TOPK * E_FFN),
            grid_dim=ceildiv(TOPK * E_FFN, 256), block_dim=256,
        )
        ctx.enqueue_function[k_down_zc](
            hb2, cd_d.unsafe_ptr(), slots, skip, qd_h.unsafe_ptr(), wt, routed_zc, Int32(MOE_H),
            grid_dim=ceildiv(MOE_H, MOE_WAVES), block_dim=MOE_THREADS,
        )
        ctx.enqueue_copy(dst_buf=skip_got if arm == 0 else routed_zc_got, src_buf=routed_zc_d)
        ctx.synchronize()
    ctx.enqueue_copy(dst_buf=wt_d, src_buf=wt_save)
    ctx.synchronize()
    var skip_bad = 0
    var skip_moved = 0
    for i in range(MOE_H):
        if skip_got[i] != routed_zc_got[i]:
            skip_bad += 1
        if skip_got[i] != routed_q4_got[i]:
            skip_moved += 1
    print("zc skip: mismatches vs weight-0 launch", skip_bad, " outputs that moved vs all-8", skip_moved)
    if skip_bad != 0 or skip_moved == 0:
        print("FAIL: HE = -2 skip is not the weight-0 launch, or skipped nothing")
        exit(1)
    print("PASS: HE = -2 skips the expert in both zc kernels, bit-identical to a zero router weight")
    print("PASS: qwen35moe sparse-MoE block matches numpy reference (m=1)")

    # Feasibility measurement, NOT a preregistered perf claim (CLAUDE.md /
    # bench/PROTOCOL-RULES.md apply to champion claims). The question is whether
    # the top-8-of-256 gather dominates the routed FFN at m=1.
    #
    # The routed path touches 8 x (gate 2.10 + up 2.10 + down 2.10 MB) = 50.3 MB
    # of bf16 expert weights per token per layer. That is under the 96 MB
    # Infinity Cache, so a fixed expert set would sit resident and report a
    # fantasy number. ROTATION defeats that: ARMS disjoint expert sets walk
    # 8 x 50.3 = 402 MB, well past the cache.
    if getenv("BARO_MOE_BENCH", "0") == "1":
        comptime ARMS = 8
        comptime ITERS = 200
        var arm_h = ctx.enqueue_create_host_buffer[i32](ARMS * TOPK)
        for a in range(ARMS):
            for j in range(TOPK):
                arm_h[a * TOPK + j] = Int32(a * TOPK + j)
        var arm_d = ctx.enqueue_create_buffer[i32](ARMS * TOPK)
        ctx.enqueue_copy(dst_buf=arm_d, src_buf=arm_h)
        ctx.synchronize()

        for a in range(ARMS):
            var sel = DeviceBuffer[i32](
                ctx, arm_d.unsafe_ptr() + a * TOPK, TOPK, owning=False
            )
            var seltt = rebind[TileTensor[i32, type_of(idx_1), MutAnyOrigin]](
                TileTensor(sel, idx_1)
            )
            ctx.enqueue_function[k_gu](
                xb2, wg, wu, seltt, h1, Int32(MOE_H),
                grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS,
            )
        ctx.synchronize()

        var arms_rt = Int(getenv("BARO_MOE_ARMS", "8"))
        var t0 = perf_counter_ns()
        for it in range(ITERS):
            var a = it % arms_rt
            var sel = DeviceBuffer[i32](
                ctx, arm_d.unsafe_ptr() + a * TOPK, TOPK, owning=False
            )
            var seltt = rebind[TileTensor[i32, type_of(idx_1), MutAnyOrigin]](
                TileTensor(sel, idx_1)
            )
            ctx.enqueue_function[k_gu](
                xb2, wg, wu, seltt, h1, Int32(MOE_H),
                grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS,
            )
            ctx.enqueue_function[k_cast_h](
                TileTensor(h_d, h_1), TileTensor(hb_d, h_1), Int32(TOPK * E_FFN),
                grid_dim=ceildiv(TOPK * E_FFN, 256), block_dim=256,
            )
            ctx.enqueue_function[k_down](
                hb2, wd, seltt, wt, routed, Int32(MOE_H),
                grid_dim=ceildiv(MOE_H, MOE_WAVES), block_dim=MOE_THREADS,
            )
        ctx.synchronize()
        var t1 = perf_counter_ns()

        var us = Float64(t1 - t0) / 1000.0 / Float64(ITERS)
        var mb = Float64(TOPK) * 3.0 * Float64(E_FFN) * Float64(MOE_H) * 2.0 / 1e6
        print("moe routed block: arms", arms_rt, ":", us, "us/token/layer,", mb, "MB bf16 touched,",
              mb / us * 1000.0, "GB/s")
        print("  40 layers ->", us * 40.0 / 1000.0, "ms/token =",
              1000.0 / (us * 40.0 / 1000.0), "tok/s ceiling from expert traffic alone")

        for a in range(ARMS):
            var sel = DeviceBuffer[i32](
                ctx, arm_d.unsafe_ptr() + a * TOPK, TOPK, owning=False
            )
            var seltt = rebind[TileTensor[i32, type_of(idx_1), MutAnyOrigin]](
                TileTensor(sel, idx_1)
            )
            ctx.enqueue_function[k_gu_q4](
                xb2, qg_d.unsafe_ptr(), qu_d.unsafe_ptr(), seltt, h1, Int32(MOE_H),
                grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS,
            )
            ctx.enqueue_function[k_cast_h](
                TileTensor(h_d, h_1), TileTensor(hb_d, h_1), Int32(TOPK * E_FFN),
                grid_dim=ceildiv(TOPK * E_FFN, 256), block_dim=256,
            )
            ctx.enqueue_function[k_down_q4](
                hb2, qd_d.unsafe_ptr(), seltt, wt, routed_q4, Int32(MOE_H),
                grid_dim=ceildiv(MOE_H, MOE_WAVES), block_dim=MOE_THREADS,
            )
        ctx.synchronize()
        var tq0 = perf_counter_ns()
        for it in range(ITERS):
            var a = it % arms_rt
            var sel = DeviceBuffer[i32](
                ctx, arm_d.unsafe_ptr() + a * TOPK, TOPK, owning=False
            )
            var seltt = rebind[TileTensor[i32, type_of(idx_1), MutAnyOrigin]](
                TileTensor(sel, idx_1)
            )
            ctx.enqueue_function[k_gu_q4](
                xb2, qg_d.unsafe_ptr(), qu_d.unsafe_ptr(), seltt, h1, Int32(MOE_H),
                grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS,
            )
            ctx.enqueue_function[k_cast_h](
                TileTensor(h_d, h_1), TileTensor(hb_d, h_1), Int32(TOPK * E_FFN),
                grid_dim=ceildiv(TOPK * E_FFN, 256), block_dim=256,
            )
            ctx.enqueue_function[k_down_q4](
                hb2, qd_d.unsafe_ptr(), seltt, wt, routed_q4, Int32(MOE_H),
                grid_dim=ceildiv(MOE_H, MOE_WAVES), block_dim=MOE_THREADS,
            )
        ctx.synchronize()
        var tq1 = perf_counter_ns()
        var qus = Float64(tq1 - tq0) / 1000.0 / Float64(ITERS)
        var qmb = Float64(TOPK) * 3.0 * Float64(E_FFN) * Float64(MOE_H) * 144.0 / 256.0 / 1e6
        print("moe q4k routed block: arms", arms_rt, ":", qus, "us/token/layer,", qmb, "MB q4k touched,",
              qmb / qus * 1000.0, "GB/s")
        print("  40 layers ->", qus * 40.0 / 1000.0, "ms/token =",
              1000.0 / (qus * 40.0 / 1000.0), "tok/s ceiling from expert traffic alone")
