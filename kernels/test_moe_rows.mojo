"""Parity: the row-batched MoE kernels (kernels/moe_rows.mojo) against the m=1
kernels they are siblings of, BIT-EXACT, token by token.

The rows kernels call the same dot helpers with x_row = token, so every output
element has the summation order of the m=1 path; this test is the receipt for
that claim (bench/moe-prefill-protocol.md, design item 1). Weights are random
bytes with sane f16 block scales: layout bugs show on random data, and the
real-pack decision is gate 1, not this test.
"""
from std.math import ceildiv
from std.sys import has_accelerator, size_of

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major
from max.gpu import WARP_SIZE
from elementwise import amar_quantize_q8_rows
from state import amar_split_bf16_planes, amar_merge_bf16_planes
from moe import moe_gate_up_q4k_dot, amar_moe_down_q4k_dot

from matmul_skinny import amar_matmul_skinny_m1_row, ROW_WAVES, ROW_THREADS
from moe import (
    amar_moe_router_top8_sig, moe_matmul_q8d_m1, moe_matmul_q8d_m1_add,
    moe_gate_up_q4k_pack, amar_moe_down_q4k, amar_moe_down_q6k,
    moe_gate_up_q8_0, moe_down_q8_0_res,
    N_EXP, TOPK, E_FFN, SH_FFN, MOE_H, MOE_WAVES, MOE_THREADS,
)
from moe_rows import (
    moe_matmul_q8d_rows, moe_matmul_q8d_rows_add, moe_router_top8_sig_rows,
    moe_gate_up_q4k_rows, moe_down_q4k_rows, moe_down_q6k_rows,
    moe_shared_gate_up_q8_0_rows, moe_shared_down_q8_0_res_rows, moe_skinny_f32_rows, moe_matmul_q8d_rows_shared,
    moe_pairs_group, moe_gate_up_q4k_grouped, moe_down_q4k_grouped, moe_down_combine, moe_down_q6k_grouped,
    moe_gate_up_q4k_grouped_dot, moe_down_q4k_grouped_dot,
)

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime i32 = DType.int32
comptime u8 = DType.uint8

comptime T = 5
comptime NE = 16
comptime NOUT = 512
comptime H = MOE_H

comptime a_rows = row_major[T, H]()
comptime a_one = row_major[1, H]()
comptime o_rows = row_major[T, NOUT]()
comptime o_one = row_major[NOUT]()
comptime x_rows = row_major[T, H]()
comptime x_one = row_major[H]()
comptime lg_rows = row_major[T, N_EXP]()
comptime lg_one = row_major[N_EXP]()
comptime idx_rows = row_major[T, TOPK]()
comptime idx_one = row_major[TOPK]()
comptime sg_rows = row_major[T]()
comptime sg_one = row_major[1]()
comptime hh_rows = row_major[T * TOPK, E_FFN]()
comptime hh_one = row_major[TOPK, E_FFN]()
comptime hh_flat = row_major[TOPK * E_FFN]()
comptime hs_rows = row_major[T, SH_FFN]()
comptime hs_one = row_major[1, SH_FFN]()
comptime hs_flat = row_major[SH_FFN]()

comptime wr_layout = row_major[N_EXP, H]()

comptime GMR = 4
comptime NG = (T * TOPK + GMR - 1) // GMR + N_EXP
comptime ge_layout = row_major[NG]()
comptime gp_layout = row_major[NG * GMR]()
comptime gs_layout = row_major[2 * N_EXP]()
comptime dn_layout = row_major[T * TOPK, H]()

comptime Q4_ROW = (H // 256) * 144
comptime Q4_DROW = (E_FFN // 256) * 144
comptime Q6_DROW = (E_FFN // 256) * 210
comptime Q8_ROW = (H // 32) * 34
comptime Q8_DROW = (SH_FFN // 32) * 34


struct Lcg:
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> UInt64:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return self.s >> 33


def fill_bytes(mut g: Lcg, p: MutPointer[UInt8, MutUntrackedOrigin], n: Int):
    for i in range(n):
        p[unsafe_offset=i] = UInt8(Int(g.next() & 0xFF))


def set_f16(p: MutPointer[UInt8, MutUntrackedOrigin], at: Int, mut g: Lcg):
    # 0x2800..0x2BFF: positive, 0.03 to 0.06, never inf or nan.
    p[unsafe_offset=at] = UInt8(Int(g.next() & 0xFF))
    p[unsafe_offset=at + 1] = UInt8(0x28 + Int(g.next() & 0x3))


def differ(name: String, a: MutPointer[UInt8, MutUntrackedOrigin], b: MutPointer[UInt8, MutUntrackedOrigin], n: Int) raises:
    var bad = 0
    var nz = 0
    for i in range(n):
        if a[unsafe_offset=i] != b[unsafe_offset=i]:
            bad += 1
        if a[unsafe_offset=i] != 0:
            nz += 1
    print(name, "bytes", n, "differ", bad, "nonzero", nz)
    if bad != 0:
        var first = -1
        var last = -1
        for i in range(n):
            if a[unsafe_offset=i] != b[unsafe_offset=i]:
                if first < 0:
                    first = i
                last = i
        print("  first differing byte", first, "(element", first // 4, ") last", last, "(element", last // 4, ")")
    if bad != 0:
        raise Error("FAIL rows parity: " + name + " differs from the m=1 kernel")
    if nz == 0:
        raise Error("FAIL rows parity: " + name + " output is all zero, nothing was compared")


def view[
    dt: DType, LT: TensorLayout
](
    ctx: DeviceContext, b: DeviceBuffer[dt], o: Int, n: Int, lt: LT
) -> TileTensor[dt, LT, MutAnyOrigin]:
    var s = DeviceBuffer[dt](ctx, b.unsafe_ptr().unsafe_offset(o), n, owning=False)
    var t = TileTensor(s, lt)
    return rebind[TileTensor[dt, LT, MutAnyOrigin]](t)


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var g = Lcg(0x5EED)

    var a_h = ctx.enqueue_create_host_buffer[bf16](T * H)
    var xf_h = ctx.enqueue_create_host_buffer[f32](T * H)
    var hh_h = ctx.enqueue_create_host_buffer[bf16](T * TOPK * E_FFN)
    var hs_h = ctx.enqueue_create_host_buffer[bf16](T * SH_FFN)
    var lg_h = ctx.enqueue_create_host_buffer[f32](T * N_EXP)
    var gw_h = ctx.enqueue_create_host_buffer[f32](H)
    var res_h = ctx.enqueue_create_host_buffer[f32](T * H)
    comptime Q8D = NOUT * H + NOUT * (H // 32) * 2
    var q8d_h = ctx.enqueue_create_host_buffer[u8](Q8D)
    comptime Q4GU = 2 * NE * E_FFN * Q4_ROW
    var q4gu_h = ctx.enqueue_create_host_buffer[u8](Q4GU)
    comptime Q4D = NE * H * Q4_DROW
    var q4d_h = ctx.enqueue_create_host_buffer[u8](Q4D)
    comptime Q6D = NE * H * Q6_DROW
    var q6d_h = ctx.enqueue_create_host_buffer[u8](Q6D)
    comptime Q8GU = 2 * SH_FFN * Q8_ROW
    var q8gu_h = ctx.enqueue_create_host_buffer[u8](Q8GU)
    comptime Q8D0 = H * Q8_DROW
    var q8d0_h = ctx.enqueue_create_host_buffer[u8](Q8D0)
    var idx0_h = ctx.enqueue_create_host_buffer[i32](1)
    ctx.synchronize()

    for i in range(T * H):
        a_h[i] = Scalar[bf16](Float32(Int(g.next() % 2001) - 1000) / 997.0)
        xf_h[i] = Float32(Int(g.next() % 2001) - 1000) / 613.0
        res_h[i] = Float32(Int(g.next() % 2001) - 1000) / 401.0
    for i in range(T * TOPK * E_FFN):
        hh_h[i] = Scalar[bf16](Float32(Int(g.next() % 2001) - 1000) / 811.0)
    for i in range(T * SH_FFN):
        hs_h[i] = Scalar[bf16](Float32(Int(g.next() % 2001) - 1000) / 811.0)
    for i in range(T * N_EXP):
        lg_h[i] = Float32(Int(g.next() % 20001) - 10000) / 1000.0
    for i in range(H):
        gw_h[i] = Float32(Int(g.next() % 2001) - 1000) / 30000.0
    idx0_h[0] = 0

    fill_bytes(g, q8d_h.unsafe_ptr(), Q8D)
    for b in range(NOUT * (H // 32)):
        set_f16(q8d_h.unsafe_ptr(), NOUT * H + b * 2, g)
    fill_bytes(g, q4gu_h.unsafe_ptr(), Q4GU)
    for b in range(Q4GU // 144):
        set_f16(q4gu_h.unsafe_ptr(), b * 144, g)
        set_f16(q4gu_h.unsafe_ptr(), b * 144 + 2, g)
    fill_bytes(g, q4d_h.unsafe_ptr(), Q4D)
    for b in range(Q4D // 144):
        set_f16(q4d_h.unsafe_ptr(), b * 144, g)
        set_f16(q4d_h.unsafe_ptr(), b * 144 + 2, g)
    fill_bytes(g, q6d_h.unsafe_ptr(), Q6D)
    for b in range(Q6D // 210):
        set_f16(q6d_h.unsafe_ptr(), b * 210 + 208, g)
    fill_bytes(g, q8gu_h.unsafe_ptr(), Q8GU)
    for b in range(Q8GU // 34):
        set_f16(q8gu_h.unsafe_ptr(), b * 34, g)
    fill_bytes(g, q8d0_h.unsafe_ptr(), Q8D0)
    for b in range(Q8D0 // 34):
        set_f16(q8d0_h.unsafe_ptr(), b * 34, g)

    var a_d = ctx.enqueue_create_buffer[bf16](T * H)
    var xf_d = ctx.enqueue_create_buffer[f32](T * H)
    var hh_d = ctx.enqueue_create_buffer[bf16](T * TOPK * E_FFN)
    var hs_d = ctx.enqueue_create_buffer[bf16](T * SH_FFN)
    var lg_d = ctx.enqueue_create_buffer[f32](T * N_EXP)
    var gw_d = ctx.enqueue_create_buffer[f32](H)
    var q8d_d = ctx.enqueue_create_buffer[u8](Q8D)
    var q4gu_d = ctx.enqueue_create_buffer[u8](Q4GU)
    var q4d_d = ctx.enqueue_create_buffer[u8](Q4D)
    var q6d_d = ctx.enqueue_create_buffer[u8](Q6D)
    var q8gu_d = ctx.enqueue_create_buffer[u8](Q8GU)
    var q8d0_d = ctx.enqueue_create_buffer[u8](Q8D0)
    var idx0_d = ctx.enqueue_create_buffer[i32](1)
    ctx.enqueue_copy(dst_buf=a_d, src_buf=a_h)
    ctx.enqueue_copy(dst_buf=xf_d, src_buf=xf_h)
    ctx.enqueue_copy(dst_buf=hh_d, src_buf=hh_h)
    ctx.enqueue_copy(dst_buf=hs_d, src_buf=hs_h)
    ctx.enqueue_copy(dst_buf=lg_d, src_buf=lg_h)
    ctx.enqueue_copy(dst_buf=gw_d, src_buf=gw_h)
    ctx.enqueue_copy(dst_buf=q8d_d, src_buf=q8d_h)
    ctx.enqueue_copy(dst_buf=q4gu_d, src_buf=q4gu_h)
    ctx.enqueue_copy(dst_buf=q4d_d, src_buf=q4d_h)
    ctx.enqueue_copy(dst_buf=q6d_d, src_buf=q6d_h)
    ctx.enqueue_copy(dst_buf=q8gu_d, src_buf=q8gu_h)
    ctx.enqueue_copy(dst_buf=q8d0_d, src_buf=q8d0_h)
    ctx.enqueue_copy(dst_buf=idx0_d, src_buf=idx0_h)

    # Two planes per output: R = rows kernel, M = m=1 kernel looped over tokens.
    var oR_d = ctx.enqueue_create_buffer[f32](T * NOUT)
    var oM_d = ctx.enqueue_create_buffer[f32](T * NOUT)
    var xR_d = ctx.enqueue_create_buffer[f32](T * H)
    var xM_d = ctx.enqueue_create_buffer[f32](T * H)
    var idxR_d = ctx.enqueue_create_buffer[i32](T * TOPK)
    var idxM_d = ctx.enqueue_create_buffer[i32](T * TOPK)
    var wtR_d = ctx.enqueue_create_buffer[f32](T * TOPK)
    var wtM_d = ctx.enqueue_create_buffer[f32](T * TOPK)
    var sgR_d = ctx.enqueue_create_buffer[f32](T)
    var sgM_d = ctx.enqueue_create_buffer[f32](T)
    var hR_d = ctx.enqueue_create_buffer[bf16](T * TOPK * E_FFN)
    var hM_d = ctx.enqueue_create_buffer[bf16](T * TOPK * E_FFN)
    var sR_d = ctx.enqueue_create_buffer[bf16](T * SH_FFN)
    var sM_d = ctx.enqueue_create_buffer[bf16](T * SH_FFN)
    var dR_d = ctx.enqueue_create_buffer[f32](T * H)
    var dM_d = ctx.enqueue_create_buffer[f32](T * H)
    var d6R_d = ctx.enqueue_create_buffer[f32](T * H)
    var d6M_d = ctx.enqueue_create_buffer[f32](T * H)
    var rR_d = ctx.enqueue_create_buffer[f32](T * H)
    var rM_d = ctx.enqueue_create_buffer[f32](T * H)
    ctx.enqueue_copy(dst_buf=xR_d, src_buf=res_h)
    ctx.enqueue_copy(dst_buf=xM_d, src_buf=res_h)
    ctx.enqueue_copy(dst_buf=rR_d, src_buf=res_h)
    ctx.enqueue_copy(dst_buf=rM_d, src_buf=res_h)

    var A = view(ctx, a_d, 0, len(a_d), a_rows)
    var s_off = Int32(NOUT * H)
    comptime g_rows = ceildiv(NOUT, 8)

    # 1. trunk projection, plain and residual-add
    ctx.enqueue_function[moe_matmul_q8d_rows[type_of(o_rows), type_of(a_rows)]](
        A, q8d_d.unsafe_ptr(), view(ctx, oR_d, 0, len(oR_d), o_rows), Int32(NOUT), Int32(H), s_off, grid_dim=(g_rows, T), block_dim=256)
    # 2. router top-8, weights, shared sigmoid gate
    ctx.enqueue_function[moe_router_top8_sig_rows[type_of(lg_rows), type_of(idx_rows), type_of(idx_rows), type_of(x_rows), type_of(x_one), type_of(sg_rows)]](
        view(ctx, lg_d, 0, len(lg_d), lg_rows), view(ctx, idxR_d, 0, len(idxR_d), idx_rows), view(ctx, wtR_d, 0, len(wtR_d), idx_rows), view(ctx, xf_d, 0, len(xf_d), x_rows), view(ctx, gw_d, 0, len(gw_d), x_one), view(ctx, sgR_d, 0, len(sgR_d), sg_rows),
        Int32(H), grid_dim=T, block_dim=MOE_THREADS // MOE_WAVES)
    for t in range(T):
        var At = view(ctx, a_d, t * H, H, a_one)
        var Ot = view(ctx, oM_d, t * NOUT, NOUT, o_one)
        ctx.enqueue_function[moe_matmul_q8d_m1[type_of(o_one), type_of(a_one)]](
            At, q8d_d.unsafe_ptr(), Ot, Int32(NOUT), Int32(H), s_off, grid_dim=g_rows, block_dim=256)
        ctx.enqueue_function[amar_moe_router_top8_sig[type_of(lg_one), type_of(idx_one), type_of(idx_one), type_of(x_one), type_of(x_one), type_of(sg_one)]](
            view(ctx, lg_d, t * N_EXP, N_EXP, lg_one),
            view(ctx, idxM_d, t * TOPK, TOPK, idx_one),
            view(ctx, wtM_d, t * TOPK, TOPK, idx_one),
            view(ctx, xf_d, t * H, H, x_one),
            view(ctx, gw_d, 0, len(gw_d), x_one),
            view(ctx, sgM_d, t, 1, sg_one),
            Int32(H), grid_dim=1, block_dim=MOE_THREADS // MOE_WAVES)
    ctx.synchronize()

    # The router picks among 256 ids; the expert planes here hold NE experts.
    var idx_h = ctx.enqueue_create_host_buffer[i32](T * TOPK)
    ctx.enqueue_copy(dst_buf=idx_h, src_buf=idxR_d)
    ctx.synchronize()
    var ide_h = ctx.enqueue_create_host_buffer[i32](T * TOPK)
    for i in range(T * TOPK):
        ide_h[i] = idx_h[i] % NE
    var ide_d = ctx.enqueue_create_buffer[i32](T * TOPK)
    ctx.enqueue_copy(dst_buf=ide_d, src_buf=ide_h)

    var up_off = Int32(NE * E_FFN * Q4_ROW)
    var sh_up_off = Int32(SH_FFN * Q8_ROW)
    # 3. routed gate/up, routed down (q4k and q6k), shared gate/up, shared down + residual
    ctx.enqueue_function[moe_gate_up_q4k_rows[TOPK, E_FFN, type_of(a_rows), type_of(idx_rows), type_of(hh_rows)]](
        A, q4gu_d.unsafe_ptr(), view(ctx, ide_d, 0, len(ide_d), idx_rows), view(ctx, hR_d, 0, len(hR_d), hh_rows), Int32(H), up_off,
        grid_dim=(ceildiv(TOPK * E_FFN, MOE_WAVES), T), block_dim=MOE_THREADS)
    ctx.enqueue_function[moe_down_q4k_rows[TOPK, E_FFN, type_of(hh_rows), type_of(idx_rows), type_of(idx_rows), type_of(x_rows)]](
        view(ctx, hh_d, 0, len(hh_d), hh_rows), q4d_d.unsafe_ptr(), view(ctx, ide_d, 0, len(ide_d), idx_rows), view(ctx, wtR_d, 0, len(wtR_d), idx_rows), view(ctx, dR_d, 0, len(dR_d), x_rows), Int32(H),
        grid_dim=(ceildiv(H, MOE_WAVES), T), block_dim=MOE_THREADS)
    ctx.enqueue_function[moe_down_q6k_rows[TOPK, E_FFN, type_of(hh_rows), type_of(idx_rows), type_of(idx_rows), type_of(x_rows)]](
        view(ctx, hh_d, 0, len(hh_d), hh_rows), q6d_d.unsafe_ptr(), view(ctx, ide_d, 0, len(ide_d), idx_rows), view(ctx, wtR_d, 0, len(wtR_d), idx_rows), view(ctx, d6R_d, 0, len(d6R_d), x_rows), Int32(H),
        grid_dim=(ceildiv(H, MOE_WAVES), T), block_dim=MOE_THREADS)
    ctx.enqueue_function[moe_shared_gate_up_q8_0_rows[4, SH_FFN, type_of(a_rows), type_of(hs_rows)]](
        A, q8gu_d.unsafe_ptr(), view(ctx, sR_d, 0, len(sR_d), hs_rows), Int32(H), Int32(Q8_ROW), sh_up_off, Int32(T),
        grid_dim=(ceildiv(SH_FFN, MOE_WAVES), ceildiv(T, 4)), block_dim=MOE_THREADS)
    ctx.enqueue_function[moe_shared_down_q8_0_res_rows[4, SH_FFN, type_of(hs_rows), type_of(sg_rows), type_of(x_rows), type_of(x_rows)]](
        view(ctx, hs_d, 0, len(hs_d), hs_rows), q8d0_d.unsafe_ptr(), view(ctx, sgR_d, 0, len(sgR_d), sg_rows), view(ctx, xf_d, 0, len(xf_d), x_rows), view(ctx, rR_d, 0, len(rR_d), x_rows), Int32(H), Int32(Q8_DROW), Int32(T),
        grid_dim=(ceildiv(H, MOE_WAVES), ceildiv(T, 4)), block_dim=MOE_THREADS)
    ctx.enqueue_function[moe_matmul_q8d_rows_add[type_of(a_rows), type_of(o_rows)]](
        A, q8d_d.unsafe_ptr(), view(ctx, xR_d, 0, T * NOUT, o_rows),
        Int32(NOUT), Int32(H), s_off, grid_dim=(g_rows, T), block_dim=256)

    for t in range(T):
        var At = view(ctx, a_d, t * H, H, a_one)
        var It = view(ctx, ide_d, t * TOPK, TOPK, idx_one)
        var Wt = view(ctx, wtR_d, t * TOPK, TOPK, idx_one)
        var Hin = view(ctx, hh_d, t * TOPK * E_FFN, TOPK * E_FFN, hh_one)
        ctx.enqueue_function[moe_gate_up_q4k_pack[TOPK, E_FFN, type_of(a_one), type_of(idx_one), type_of(hh_flat), bf16]](
            At, q4gu_d.unsafe_ptr(), It, view(ctx, hM_d, t * TOPK * E_FFN, TOPK * E_FFN, hh_flat),
            Int32(H), up_off, grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS)
        ctx.enqueue_function[amar_moe_down_q4k[TOPK, E_FFN, type_of(hh_one), type_of(idx_one), type_of(idx_one), type_of(x_one)]](
            Hin, q4d_d.unsafe_ptr(), It, Wt, view(ctx, dM_d, t * H, H, x_one), Int32(H),
            grid_dim=ceildiv(H, MOE_WAVES), block_dim=MOE_THREADS)
        ctx.enqueue_function[amar_moe_down_q6k[TOPK, E_FFN, type_of(hh_one), type_of(idx_one), type_of(idx_one), type_of(x_one)]](
            Hin, q6d_d.unsafe_ptr(), It, Wt, view(ctx, d6M_d, t * H, H, x_one), Int32(H),
            grid_dim=ceildiv(H, MOE_WAVES), block_dim=MOE_THREADS)
        ctx.enqueue_function[moe_gate_up_q8_0[1, SH_FFN, type_of(a_one), type_of(sg_one), type_of(hs_flat), bf16]](
            At, q8gu_d.unsafe_ptr(), view(ctx, idx0_d, 0, len(idx0_d), sg_one), view(ctx, sM_d, t * SH_FFN, SH_FFN, hs_flat),
            Int32(H), Int32(Q8_ROW), sh_up_off, grid_dim=ceildiv(SH_FFN, MOE_WAVES), block_dim=MOE_THREADS)
        ctx.enqueue_function[moe_down_q8_0_res[1, SH_FFN, type_of(hs_one), type_of(sg_one), type_of(sg_one), type_of(x_one), type_of(x_one)]](
            view(ctx, hs_d, t * SH_FFN, SH_FFN, hs_one), q8d0_d.unsafe_ptr(), view(ctx, idx0_d, 0, len(idx0_d), sg_one),
            view(ctx, sgR_d, t, 1, sg_one),
            view(ctx, xf_d, t * H, H, x_one),
            view(ctx, rM_d, t * H, H, x_one),
            Int32(H), Int32(Q8_DROW), grid_dim=ceildiv(H, MOE_WAVES), block_dim=MOE_THREADS)
        ctx.enqueue_function[moe_matmul_q8d_m1_add[type_of(o_one), type_of(a_one), type_of(o_one)]](
            At, q8d_d.unsafe_ptr(), view(ctx, oM_d, t * NOUT, NOUT, o_one),
            view(ctx, xM_d, t * NOUT, NOUT, o_one),
            Int32(NOUT), Int32(H), s_off, grid_dim=g_rows, block_dim=256)
    ctx.synchronize()

    def cmp[dt: DType](name: String, r: DeviceBuffer[dt], m: DeviceBuffer[dt], n: Int) raises {imm ctx}:
        var rh = ctx.enqueue_create_host_buffer[dt](n)
        var mh = ctx.enqueue_create_host_buffer[dt](n)
        ctx.enqueue_copy(dst_buf=rh, src_buf=r)
        ctx.enqueue_copy(dst_buf=mh, src_buf=m)
        ctx.synchronize()
        differ(name, rh.unsafe_ptr().unsafe_bitcast[UInt8](), mh.unsafe_ptr().unsafe_bitcast[UInt8](), n * size_of[Scalar[dt]]())

    var wr_h = ctx.enqueue_create_host_buffer[f32](N_EXP * H)
    ctx.synchronize()
    for i in range(N_EXP * H):
        wr_h[i] = Float32(Int(g.next() % 2001) - 1000) / 9000.0
    var wr_d = ctx.enqueue_create_buffer[f32](N_EXP * H)
    var skR_d = ctx.enqueue_create_buffer[f32](T * N_EXP)
    var skM_d = ctx.enqueue_create_buffer[f32](T * N_EXP)
    ctx.enqueue_copy(dst_buf=wr_d, src_buf=wr_h)
    ctx.enqueue_function[moe_skinny_f32_rows[2, type_of(x_rows), type_of(wr_layout), type_of(lg_rows)]](
        view(ctx, xf_d, 0, T * H, x_rows), view(ctx, wr_d, 0, N_EXP * H, wr_layout), view(ctx, skR_d, 0, T * N_EXP, lg_rows),
        Int32(N_EXP), Int32(H), grid_dim=(ceildiv(N_EXP, ROW_WAVES), T), block_dim=ROW_THREADS)
    for t in range(T):
        ctx.enqueue_function[amar_matmul_skinny_m1_row[f32, 2, type_of(a_one), type_of(wr_layout), type_of(lg_one)]](
            view(ctx, xf_d, t * H, H, a_one), view(ctx, wr_d, 0, N_EXP * H, wr_layout), view(ctx, skM_d, t * N_EXP, N_EXP, lg_one),
            Int32(N_EXP), Int32(H), grid_dim=ceildiv(N_EXP, ROW_WAVES), block_dim=ROW_THREADS)
    ctx.synchronize()
    cmp("skinny_f32", skR_d, skM_d, T * N_EXP)
    # Shared weight loads: MR tokens per thread, T = 5 leaves a clamped tail at MR = 4.
    var oS_d = ctx.enqueue_create_buffer[f32](T * NOUT)
    var xS_d = ctx.enqueue_create_buffer[f32](T * H)
    ctx.enqueue_copy(dst_buf=xS_d, src_buf=res_h)
    ctx.enqueue_function[moe_matmul_q8d_rows_shared[4, False, type_of(o_rows), type_of(a_rows)]](
        A, q8d_d.unsafe_ptr(), view(ctx, oS_d, 0, T * NOUT, o_rows), Int32(NOUT), Int32(H), s_off, Int32(T),
        grid_dim=(g_rows, ceildiv(T, 4)), block_dim=256)
    ctx.enqueue_function[moe_matmul_q8d_rows_shared[4, True, type_of(o_rows), type_of(a_rows)]](
        A, q8d_d.unsafe_ptr(), view(ctx, xS_d, 0, T * NOUT, o_rows), Int32(NOUT), Int32(H), s_off, Int32(T),
        grid_dim=(g_rows, ceildiv(T, 4)), block_dim=256)
    ctx.synchronize()
    # Experts grouped: pairs sorted by expert, one weight load per GMR pairs, top-k combine after.
    var ge_d = ctx.enqueue_create_buffer[i32](NG)
    var gp_d = ctx.enqueue_create_buffer[i32](NG * GMR)
    var gs_d = ctx.enqueue_create_buffer[i32](2 * N_EXP)
    var hG_d = ctx.enqueue_create_buffer[bf16](T * TOPK * E_FFN)
    var dn_d = ctx.enqueue_create_buffer[f32](T * TOPK * H)
    var dG_d = ctx.enqueue_create_buffer[f32](T * H)
    ctx.enqueue_function[moe_pairs_group[GMR, T * TOPK, type_of(idx_rows), type_of(ge_layout), type_of(gp_layout), type_of(gs_layout)]](
        view(ctx, ide_d, 0, T * TOPK, idx_rows), view(ctx, ge_d, 0, NG, ge_layout), view(ctx, gp_d, 0, NG * GMR, gp_layout),
        view(ctx, gs_d, 0, 2 * N_EXP, gs_layout), Int32(T), Int32(NG), grid_dim=1, block_dim=N_EXP)
    ctx.enqueue_function[moe_gate_up_q4k_grouped[GMR, E_FFN, type_of(a_rows), type_of(ge_layout), type_of(gp_layout), type_of(hh_rows)]](
        A, q4gu_d.unsafe_ptr(), view(ctx, ge_d, 0, NG, ge_layout), view(ctx, gp_d, 0, NG * GMR, gp_layout),
        view(ctx, hG_d, 0, T * TOPK * E_FFN, hh_rows), Int32(H), up_off,
        grid_dim=(ceildiv(E_FFN, MOE_WAVES), NG), block_dim=MOE_THREADS)
    ctx.enqueue_function[moe_down_q4k_grouped[GMR, E_FFN, type_of(hh_rows), type_of(ge_layout), type_of(gp_layout), type_of(dn_layout)]](
        view(ctx, hh_d, 0, T * TOPK * E_FFN, hh_rows), q4d_d.unsafe_ptr(), view(ctx, ge_d, 0, NG, ge_layout), view(ctx, gp_d, 0, NG * GMR, gp_layout),
        view(ctx, dn_d, 0, T * TOPK * H, dn_layout), Int32(H), grid_dim=(ceildiv(H, MOE_WAVES), NG), block_dim=MOE_THREADS)
    ctx.enqueue_function[moe_down_combine[TOPK, type_of(dn_layout), type_of(idx_rows), type_of(x_rows)]](
        view(ctx, dn_d, 0, T * TOPK * H, dn_layout), view(ctx, wtR_d, 0, T * TOPK, idx_rows), view(ctx, dG_d, 0, T * H, x_rows),
        Int32(H), grid_dim=(ceildiv(H, 256), T), block_dim=256)
    var dn6_d = ctx.enqueue_create_buffer[f32](T * TOPK * H)
    var d6G_d = ctx.enqueue_create_buffer[f32](T * H)
    ctx.enqueue_function[moe_down_q6k_grouped[GMR, E_FFN, type_of(hh_rows), type_of(ge_layout), type_of(gp_layout), type_of(dn_layout)]](
        view(ctx, hh_d, 0, T * TOPK * E_FFN, hh_rows), q6d_d.unsafe_ptr(), view(ctx, ge_d, 0, NG, ge_layout), view(ctx, gp_d, 0, NG * GMR, gp_layout),
        view(ctx, dn6_d, 0, T * TOPK * H, dn_layout), Int32(H), grid_dim=(ceildiv(H, MOE_WAVES), NG), block_dim=MOE_THREADS)
    ctx.enqueue_function[moe_down_combine[TOPK, type_of(dn_layout), type_of(idx_rows), type_of(x_rows)]](
        view(ctx, dn6_d, 0, T * TOPK * H, dn_layout), view(ctx, wtR_d, 0, T * TOPK, idx_rows), view(ctx, d6G_d, 0, T * H, x_rows),
        Int32(H), grid_dim=(ceildiv(H, 256), T), block_dim=256)
    ctx.synchronize()
    cmp("gate_up_q4k_grouped", hG_d, hM_d, T * TOPK * E_FFN)
    cmp("down_q6k_grouped", d6G_d, d6M_d, T * H)
    cmp("down_q4k_grouped", dG_d, dM_d, T * H)
    cmp("q8d_shared", oS_d, oM_d, T * NOUT)
    cmp("q8d_shared_add", xS_d, xM_d, T * NOUT)
    cmp("q8d_rows", oR_d, oM_d, T * NOUT)
    var xr_h = ctx.enqueue_create_host_buffer[f32](T * NOUT)
    ctx.enqueue_copy(dst_buf=xr_h, src_buf=DeviceBuffer[f32](ctx, xR_d.unsafe_ptr(), T * NOUT, owning=False))
    ctx.synchronize()
    var same_as_input = 0
    for i in range(T * NOUT):
        if xr_h[i] == res_h[i]:
            same_as_input += 1
    print("q8d_rows_add elements equal to the untouched input:", same_as_input, "of", T * NOUT)
    cmp("q8d_rows_add", xR_d, xM_d, T * NOUT)
    cmp("router_idx", idxR_d, idxM_d, T * TOPK)
    cmp("router_wt", wtR_d, wtM_d, T * TOPK)
    cmp("shared_sigmoid", sgR_d, sgM_d, T)
    cmp("gate_up_q4k", hR_d, hM_d, T * TOPK * E_FFN)
    cmp("down_q4k", dR_d, dM_d, T * H)
    cmp("down_q6k", d6R_d, d6M_d, T * H)
    cmp("shared_gate_up", sR_d, sM_d, T * SH_FFN)
    cmp("shared_down_res", rR_d, rM_d, T * H)
    # int8-dot4 arm (S2 round 7): grouped MR-row kernels vs the m=1 dot4 kernels, bit-exact.
    comptime aq_rows = row_major[T, H]()
    comptime as_rows = row_major[T, H // 32]()
    comptime hq_rows = row_major[T * TOPK, E_FFN]()
    comptime hsq_rows = row_major[T * TOPK, E_FFN // 32]()
    var aq_d = ctx.enqueue_create_buffer[DType.int8](T * H)
    var as_d = ctx.enqueue_create_buffer[DType.float16](T * H // 32)
    var hDM_d = ctx.enqueue_create_buffer[bf16](T * TOPK * E_FFN)
    var hDG_d = ctx.enqueue_create_buffer[bf16](T * TOPK * E_FFN)
    var hq_d = ctx.enqueue_create_buffer[DType.int8](T * TOPK * E_FFN)
    var hsq_d = ctx.enqueue_create_buffer[DType.float16](T * TOPK * E_FFN // 32)
    var dDM_d = ctx.enqueue_create_buffer[f32](T * H)
    var dnD_d = ctx.enqueue_create_buffer[f32](T * TOPK * H)
    var dDG_d = ctx.enqueue_create_buffer[f32](T * H)
    ctx.enqueue_function[amar_quantize_q8_rows[type_of(a_rows), type_of(aq_rows), type_of(as_rows)]](
        A, TileTensor(aq_d, aq_rows), TileTensor(as_d, as_rows), Int32(T), Int32(H), grid_dim=(T, H // 32), block_dim=WARP_SIZE)
    for t in range(T):
        var It = view(ctx, ide_d, t * TOPK, TOPK, idx_one)
        var Wt = view(ctx, wtR_d, t * TOPK, TOPK, idx_one)
        ctx.enqueue_function[moe_gate_up_q4k_dot[TOPK, E_FFN, H // 256, type_of(idx_one), type_of(hh_flat)]](
            aq_d.unsafe_ptr().unsafe_offset(t * H), as_d.unsafe_ptr().unsafe_offset(t * (H // 32)), q4gu_d.unsafe_ptr(), It,
            view(ctx, hDM_d, t * TOPK * E_FFN, TOPK * E_FFN, hh_flat), up_off, grid_dim=ceildiv(TOPK * E_FFN, MOE_WAVES), block_dim=MOE_THREADS)
    ctx.enqueue_function[amar_quantize_q8_rows[type_of(hq_rows), type_of(hq_rows), type_of(hsq_rows)]](
        TileTensor(hDM_d, hq_rows), TileTensor(hq_d, hq_rows), TileTensor(hsq_d, hsq_rows), Int32(T * TOPK), Int32(E_FFN), grid_dim=(T * TOPK, E_FFN // 32), block_dim=WARP_SIZE)
    for t in range(T):
        var It = view(ctx, ide_d, t * TOPK, TOPK, idx_one)
        var Wt = view(ctx, wtR_d, t * TOPK, TOPK, idx_one)
        ctx.enqueue_function[amar_moe_down_q4k_dot[TOPK, E_FFN, E_FFN // 256, type_of(idx_one), type_of(idx_one), type_of(x_one)]](
            hq_d.unsafe_ptr().unsafe_offset(t * TOPK * E_FFN), hsq_d.unsafe_ptr().unsafe_offset(t * TOPK * E_FFN // 32), q4d_d.unsafe_ptr(), It, Wt,
            view(ctx, dDM_d, t * H, H, x_one), Int32(H), grid_dim=ceildiv(H, MOE_WAVES), block_dim=MOE_THREADS)
    ctx.enqueue_function[moe_gate_up_q4k_grouped_dot[GMR, E_FFN, H // 256, type_of(ge_layout), type_of(gp_layout), type_of(hh_rows)]](
        aq_d.unsafe_ptr(), as_d.unsafe_ptr(), q4gu_d.unsafe_ptr(), view(ctx, ge_d, 0, NG, ge_layout), view(ctx, gp_d, 0, NG * GMR, gp_layout),
        view(ctx, hDG_d, 0, T * TOPK * E_FFN, hh_rows), up_off, grid_dim=(ceildiv(E_FFN, MOE_WAVES), NG), block_dim=MOE_THREADS)
    ctx.enqueue_function[moe_down_q4k_grouped_dot[GMR, E_FFN, E_FFN // 256, type_of(ge_layout), type_of(gp_layout), type_of(dn_layout)]](
        hq_d.unsafe_ptr(), hsq_d.unsafe_ptr(), q4d_d.unsafe_ptr(), view(ctx, ge_d, 0, NG, ge_layout), view(ctx, gp_d, 0, NG * GMR, gp_layout),
        view(ctx, dnD_d, 0, T * TOPK * H, dn_layout), Int32(H), grid_dim=(ceildiv(H, MOE_WAVES), NG), block_dim=MOE_THREADS)
    ctx.enqueue_function[moe_down_combine[TOPK, type_of(dn_layout), type_of(idx_rows), type_of(x_rows)]](
        view(ctx, dnD_d, 0, T * TOPK * H, dn_layout), view(ctx, wtR_d, 0, T * TOPK, idx_rows), view(ctx, dDG_d, 0, T * H, x_rows),
        Int32(H), grid_dim=(ceildiv(H, 256), T), block_dim=256)
    ctx.synchronize()
    cmp("gate_up_q4k_grouped_dot", hDG_d, hDM_d, T * TOPK * E_FFN)
    cmp("down_q4k_grouped_dot", dDG_d, dDM_d, T * H)

    # K3 union-verify deduplication check: deduplicate (t, j) pairs by unique expert e in uniq[0..nu-1],
    # evaluate each unique expert in a single contiguous pass into contrib[t * TOPK + j], then reduce
    # contrib[t * TOPK + 0..TOPK-1] in original router order j=0..TOPK-1 -> 0-byte diff vs m=1 (dDM_d).
    var seen_e = List[Bool](capacity=NE)
    for _ in range(NE):
        seen_e.append(False)
    var uniq_e = List[Int]()
    for i in range(T * TOPK):
        var e = Int(ide_h[i])
        if not seen_e[e]:
            seen_e[e] = True
            uniq_e.append(e)
    var ge_u_h = ctx.enqueue_create_host_buffer[i32](NG)
    var gp_u_h = ctx.enqueue_create_host_buffer[i32](NG * GMR)
    ctx.synchronize()
    for i in range(NG):
        ge_u_h[i] = Int32(-1)
    for i in range(NG * GMR):
        gp_u_h[i] = Int32(-1)
    var n_u_groups = 0
    for ui in range(len(uniq_e)):
        var target_e = uniq_e[ui]
        var lane = 0
        for p_idx in range(T * TOPK):
            if Int(ide_h[p_idx]) == target_e:
                if lane == 0:
                    ge_u_h[n_u_groups] = Int32(target_e)
                gp_u_h[n_u_groups * GMR + lane] = Int32(p_idx)
                lane += 1
                if lane == GMR:
                    n_u_groups += 1
                    lane = 0
        if lane > 0:
            n_u_groups += 1
    var ge_u_d = ctx.enqueue_create_buffer[i32](NG)
    var gp_u_d = ctx.enqueue_create_buffer[i32](NG * GMR)
    var dnU_d = ctx.enqueue_create_buffer[f32](T * TOPK * H)
    var dDU_d = ctx.enqueue_create_buffer[f32](T * H)
    ctx.enqueue_copy(dst_buf=ge_u_d, src_buf=ge_u_h)
    ctx.enqueue_copy(dst_buf=gp_u_d, src_buf=gp_u_h)
    ctx.enqueue_function[moe_down_q4k_grouped_dot[GMR, E_FFN, E_FFN // 256, type_of(ge_layout), type_of(gp_layout), type_of(dn_layout)]](
        hq_d.unsafe_ptr(), hsq_d.unsafe_ptr(), q4d_d.unsafe_ptr(), view(ctx, ge_u_d, 0, NG, ge_layout), view(ctx, gp_u_d, 0, NG * GMR, gp_layout),
        view(ctx, dnU_d, 0, T * TOPK * H, dn_layout), Int32(H), grid_dim=(ceildiv(H, MOE_WAVES), n_u_groups), block_dim=MOE_THREADS)
    ctx.enqueue_function[moe_down_combine[TOPK, type_of(dn_layout), type_of(idx_rows), type_of(x_rows)]](
        view(ctx, dnU_d, 0, T * TOPK * H, dn_layout), view(ctx, wtR_d, 0, T * TOPK, idx_rows), view(ctx, dDU_d, 0, T * H, x_rows),
        Int32(H), grid_dim=(ceildiv(H, 256), T), block_dim=256)
    ctx.synchronize()
    print("k3_union_verify: unique experts loaded:", len(uniq_e), "of", T * TOPK, "pairs (", n_u_groups, "groups )")
    cmp("k3_union_verify_down_q4k_dot", dDU_d, dDM_d, T * H)

    # K3-4 BF16 sign+exponent / mantissa plane split lossless round-trip check:
    var n_bf16 = T * H
    var hi_d = ctx.enqueue_create_buffer[u8](n_bf16)
    var lo_d = ctx.enqueue_create_buffer[u8](n_bf16)
    var rec_d = ctx.enqueue_create_buffer[bf16](n_bf16)
    ctx.enqueue_function[amar_split_bf16_planes](
        a_d.unsafe_ptr().unsafe_bitcast[UInt16](), hi_d.unsafe_ptr(), lo_d.unsafe_ptr(), Int32(n_bf16),
        grid_dim=ceildiv(n_bf16, 256), block_dim=256)
    ctx.enqueue_function[amar_merge_bf16_planes](
        hi_d.unsafe_ptr(), lo_d.unsafe_ptr(), rec_d.unsafe_ptr().unsafe_bitcast[UInt16](), Int32(n_bf16),
        grid_dim=ceildiv(n_bf16, 256), block_dim=256)
    ctx.synchronize()
    cmp("k3_bf16_plane_split_roundtrip", rec_d, a_d, n_bf16)
    var hi_h = ctx.enqueue_create_host_buffer[u8](n_bf16)
    ctx.enqueue_copy(dst_buf=hi_h, src_buf=hi_d)
    ctx.synchronize()
    var uniq_hi = 0
    var seen_hi = List[Bool](capacity=256)
    for _ in range(256):
        seen_hi.append(False)
    for i in range(n_bf16):
        var b_val = Int(hi_h[i])
        if not seen_hi[b_val]:
            seen_hi[b_val] = True
            uniq_hi += 1
    print("k3_bf16_plane_split: lossless roundtrip 0 diff; distinct sign+exp bytes:", uniq_hi, "of 256")
    print("PASS moe rows parity: 18 outputs bit-exact over", T, "tokens, including K3 union-verify and BF16 plane split")
