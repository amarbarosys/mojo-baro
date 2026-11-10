"""Parity: amar_ssm_delta_rows (one launch, m rows, state kept in registers)
against amar_ssm_delta_step[1] launched once per row through the state ring,
BIT-EXACT on the per-row outputs and on the final state slot. The rows kernel
stages the next row's k/q in LDS while the current row runs.

The batched MoE prefill has to reproduce the replay path's tokens
(bench/moe-prefill-protocol.md); the dense chunk scan amar_ssm_delta_chunk_w
splits the state sum across lanes and does not. Its drift is printed here as
a number, not gated.
"""
from std.sys import has_accelerator

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from ssm import (
    amar_ssm_delta_step, amar_ssm_delta_rows, amar_ssm_delta_chunk_w,
    CONV, NH_V, SSTATE, DC_BLOCKS,
)

comptime f32 = DType.float32
comptime T = 37
comptime SLOTS = 9
comptime N_SSM = 2
comptime SI = 1

comptime sall = row_major[SLOTS, N_SSM, NH_V, SSTATE, SSTATE]()
comptime conv_rows = row_major[T, CONV]()
comptime conv_one = row_major[1, CONV]()
comptime g_rows = row_major[T, NH_V]()
comptime g_one = row_major[1, NH_V]()
comptime o_rows = row_major[T, NH_V, SSTATE]()
comptime o_one = row_major[1, NH_V, SSTATE]()
comptime SBYTES = SLOTS * N_SSM * NH_V * SSTATE * SSTATE


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
    var seed = UInt64(0xD17A)

    var s_h = ctx.enqueue_create_host_buffer[f32](SBYTES)
    var c_h = ctx.enqueue_create_host_buffer[f32](T * CONV)
    var eg_h = ctx.enqueue_create_host_buffer[f32](T * NH_V)
    var be_h = ctx.enqueue_create_host_buffer[f32](T * NH_V)
    ctx.synchronize()
    for i in range(SBYTES):
        seed = seed * 6364136223846793005 + 1442695040888963407
        s_h[i] = Float32(Int((seed >> 33) % 2001) - 1000) / 5000.0
    for i in range(T * CONV):
        seed = seed * 6364136223846793005 + 1442695040888963407
        c_h[i] = Float32(Int((seed >> 33) % 2001) - 1000) / 11000.0
    for i in range(T * NH_V):
        seed = seed * 6364136223846793005 + 1442695040888963407
        eg_h[i] = 0.5 + Float32(Int((seed >> 33) % 1000)) / 2100.0
        seed = seed * 6364136223846793005 + 1442695040888963407
        be_h[i] = Float32(Int((seed >> 33) % 1000)) / 1000.0

    var sA_d = ctx.enqueue_create_buffer[f32](SBYTES)
    var sB_d = ctx.enqueue_create_buffer[f32](SBYTES)
    var sC_d = ctx.enqueue_create_buffer[f32](SBYTES)
    var c_d = ctx.enqueue_create_buffer[f32](T * CONV)
    var eg_d = ctx.enqueue_create_buffer[f32](T * NH_V)
    var be_d = ctx.enqueue_create_buffer[f32](T * NH_V)
    var oA_d = ctx.enqueue_create_buffer[f32](T * NH_V * SSTATE)
    var oB_d = ctx.enqueue_create_buffer[f32](T * NH_V * SSTATE)
    var oC_d = ctx.enqueue_create_buffer[f32](T * NH_V * SSTATE)
    ctx.enqueue_copy(dst_buf=sA_d, src_buf=s_h)
    ctx.enqueue_copy(dst_buf=sB_d, src_buf=s_h)
    ctx.enqueue_copy(dst_buf=sC_d, src_buf=s_h)
    ctx.enqueue_copy(dst_buf=c_d, src_buf=c_h)
    ctx.enqueue_copy(dst_buf=eg_d, src_buf=eg_h)
    ctx.enqueue_copy(dst_buf=be_d, src_buf=be_h)

    comptime RING0 = 4
    # A: the decode kernel, one launch per row, ring advancing by one.
    for r in range(T):
        ctx.enqueue_function[amar_ssm_delta_step[1, type_of(sall), type_of(conv_one), type_of(g_one), type_of(o_one)]](
            view(ctx, sA_d, 0, SBYTES, sall), view(ctx, c_d, r * CONV, CONV, conv_one),
            view(ctx, eg_d, r * NH_V, NH_V, g_one), view(ctx, be_d, r * NH_V, NH_V, g_one),
            view(ctx, oA_d, r * NH_V * SSTATE, NH_V * SSTATE, o_one),
            Int32(RING0 + r), Int32(SI), Int32(SLOTS), grid_dim=NH_V, block_dim=SSTATE)
    # B: the exact rows kernel, one launch.
    ctx.enqueue_function[amar_ssm_delta_rows[type_of(sall), type_of(conv_rows), type_of(g_rows), type_of(o_rows)]](
        view(ctx, sB_d, 0, SBYTES, sall), view(ctx, c_d, 0, T * CONV, conv_rows),
        view(ctx, eg_d, 0, T * NH_V, g_rows), view(ctx, be_d, 0, T * NH_V, g_rows),
        view(ctx, oB_d, 0, T * NH_V * SSTATE, o_rows),
        Int32(RING0), Int32(SI), Int32(SLOTS), Int32(T), grid_dim=NH_V, block_dim=SSTATE)
    # C: the dense chunk scan, for the drift number only.
    ctx.enqueue_function[amar_ssm_delta_chunk_w[type_of(sall), type_of(conv_rows), type_of(g_rows), type_of(o_rows)]](
        view(ctx, sC_d, 0, SBYTES, sall), view(ctx, c_d, 0, T * CONV, conv_rows),
        view(ctx, eg_d, 0, T * NH_V, g_rows), view(ctx, be_d, 0, T * NH_V, g_rows),
        view(ctx, oC_d, 0, T * NH_V * SSTATE, o_rows),
        Int32(RING0), Int32(SI), Int32(SLOTS), Int32(T), grid_dim=DC_BLOCKS, block_dim=32)

    var oA_h = ctx.enqueue_create_host_buffer[f32](T * NH_V * SSTATE)
    var oB_h = ctx.enqueue_create_host_buffer[f32](T * NH_V * SSTATE)
    var oC_h = ctx.enqueue_create_host_buffer[f32](T * NH_V * SSTATE)
    var sA_h = ctx.enqueue_create_host_buffer[f32](SBYTES)
    var sB_h = ctx.enqueue_create_host_buffer[f32](SBYTES)
    ctx.enqueue_copy(dst_buf=oA_h, src_buf=oA_d)
    ctx.enqueue_copy(dst_buf=oB_h, src_buf=oB_d)
    ctx.enqueue_copy(dst_buf=oC_h, src_buf=oC_d)
    ctx.enqueue_copy(dst_buf=sA_h, src_buf=sA_d)
    ctx.enqueue_copy(dst_buf=sB_h, src_buf=sB_d)
    ctx.synchronize()

    var bad_o = 0
    var nz = 0
    var drift = 0
    var pa = oA_h.unsafe_ptr().unsafe_bitcast[UInt32]()
    var pb = oB_h.unsafe_ptr().unsafe_bitcast[UInt32]()
    var pc = oC_h.unsafe_ptr().unsafe_bitcast[UInt32]()
    for i in range(T * NH_V * SSTATE):
        if pa[unsafe_offset=i] != pb[unsafe_offset=i]:
            bad_o += 1
        if pa[unsafe_offset=i] != pc[unsafe_offset=i]:
            drift += 1
        if pa[unsafe_offset=i] != 0:
            nz += 1
    comptime SLOT = N_SSM * NH_V * SSTATE * SSTATE
    var fin = ((RING0 + T) % SLOTS) * SLOT + SI * NH_V * SSTATE * SSTATE
    var bad_s = 0
    var qa = sA_h.unsafe_ptr().unsafe_bitcast[UInt32]()
    var qb = sB_h.unsafe_ptr().unsafe_bitcast[UInt32]()
    for i in range(NH_V * SSTATE * SSTATE):
        if qa[unsafe_offset=fin + i] != qb[unsafe_offset=fin + i]:
            bad_s += 1
    print("delta_rows vs delta_step: outputs differ", bad_o, "of", T * NH_V * SSTATE, " final state differ", bad_s, "of", NH_V * SSTATE * SSTATE, " nonzero outputs", nz)
    print("delta_chunk_w vs delta_step (not gated): outputs differ", drift, "of", T * NH_V * SSTATE)
    if nz == 0:
        raise Error("FAIL ssm rows parity: decode outputs are all zero, nothing was compared")
    if bad_o != 0 or bad_s != 0:
        raise Error("FAIL ssm rows parity: amar_ssm_delta_rows is not bit-exact with amar_ssm_delta_step")
    print("PASS ssm rows parity: bit-exact over", T, "rows through a", SLOTS, "slot ring")
