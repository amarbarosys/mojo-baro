"""Parity: amar_ssm_conv_l2g over m=2 rows in one launch against two m=1
launches through the same conv-state ring, BIT-EXACT on conv output, Eg, Beta
and the conv state, repeated ITERS times (a racy or uninitialised read shows
up as a run-to-run difference or a non-finite value). Build with the target
model's -D BARO_SSM_* flags; OrcaSAQ-27B: -D BARO_SSM_NH_V=48.
"""
from std.sys import has_accelerator
from std.math import isfinite, exp, log1p, sqrt

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, TensorLayout, row_major

from ssm import amar_ssm_conv_l2g, CONV, CONV_GATE_BLOCKS, NH_V, NH_K, KDIM, SSTATE, SSM_EPS
from matmul_skinny import SPLITK

comptime f32 = DType.float32
comptime SLOTS = 9
comptime ITERS = 200
comptime S_N = SLOTS * 3 * CONV
comptime P_N = SPLITK * 2 * NH_V

comptime q2 = row_major[2, CONV]()
comptime q1 = row_major[1, CONV]()
comptime sl = row_major[SLOTS, 1, 3, CONV]()
comptime wl = row_major[CONV, 4]()
comptime p2 = row_major[SPLITK, 2, NH_V]()
comptime p1 = row_major[SPLITK, 1, NH_V]()
comptime g2 = row_major[2, NH_V]()
comptime g1 = row_major[1, NH_V]()
comptime dl = row_major[NH_V]()


def view[
    dt: DType, LT: TensorLayout
](
    ctx: DeviceContext, b: DeviceBuffer[dt], o: Int, n: Int, lt: LT
) -> TileTensor[dt, LT, MutAnyOrigin]:
    var s = DeviceBuffer[dt](ctx, b.unsafe_ptr().unsafe_offset(o), n, owning=False)
    var t = TileTensor(s, lt)
    return rebind[TileTensor[dt, LT, MutAnyOrigin]](t)


def fill(mut seed: UInt64, h: HostBuffer[f32], n: Int, scale: Float32, bias: Float32 = 0):
    for i in range(n):
        seed = seed * 6364136223846793005 + 1442695040888963407
        h[i] = bias + Float32(Int((seed >> 33) % 2001) - 1000) * scale


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var seed = UInt64(0xC0117)
    var q_h = ctx.enqueue_create_host_buffer[f32](2 * CONV)
    var s_h = ctx.enqueue_create_host_buffer[f32](S_N)
    var w_h = ctx.enqueue_create_host_buffer[f32](CONV * 4)
    var a_h = ctx.enqueue_create_host_buffer[f32](P_N)
    var b_h = ctx.enqueue_create_host_buffer[f32](P_N)
    var d_h = ctx.enqueue_create_host_buffer[f32](NH_V)
    var e_h = ctx.enqueue_create_host_buffer[f32](NH_V)
    ctx.synchronize()
    fill(seed, q_h, 2 * CONV, 1e-3)
    fill(seed, s_h, S_N, 1e-3)
    fill(seed, w_h, CONV * 4, 5e-4)
    fill(seed, a_h, P_N, 1e-4)
    fill(seed, b_h, P_N, 1e-4)
    fill(seed, d_h, NH_V, 1e-4)
    fill(seed, e_h, NH_V, 1e-4, -0.5)
    # Row r's split-K partials as their own [SPLITK, 1, NH_V] buffers for the m=1 arm.
    var a1_h = ctx.enqueue_create_host_buffer[f32](2 * SPLITK * NH_V)
    var b1_h = ctx.enqueue_create_host_buffer[f32](2 * SPLITK * NH_V)
    ctx.synchronize()
    for r in range(2):
        for s in range(SPLITK):
            for h in range(NH_V):
                a1_h[(r * SPLITK + s) * NH_V + h] = a_h[(s * 2 + r) * NH_V + h]
                b1_h[(r * SPLITK + s) * NH_V + h] = b_h[(s * 2 + r) * NH_V + h]

    var q_d = ctx.enqueue_create_buffer[f32](2 * CONV)
    var w_d = ctx.enqueue_create_buffer[f32](CONV * 4)
    var a_d = ctx.enqueue_create_buffer[f32](P_N)
    var b_d = ctx.enqueue_create_buffer[f32](P_N)
    var a1_d = ctx.enqueue_create_buffer[f32](2 * SPLITK * NH_V)
    var b1_d = ctx.enqueue_create_buffer[f32](2 * SPLITK * NH_V)
    var d_d = ctx.enqueue_create_buffer[f32](NH_V)
    var e_d = ctx.enqueue_create_buffer[f32](NH_V)
    var sA_d = ctx.enqueue_create_buffer[f32](S_N)
    var sB_d = ctx.enqueue_create_buffer[f32](S_N)
    var oA_d = ctx.enqueue_create_buffer[f32](2 * CONV)
    var oB_d = ctx.enqueue_create_buffer[f32](2 * CONV)
    var gA_d = ctx.enqueue_create_buffer[f32](2 * NH_V)
    var gB_d = ctx.enqueue_create_buffer[f32](2 * NH_V)
    var tA_d = ctx.enqueue_create_buffer[f32](2 * NH_V)
    var tB_d = ctx.enqueue_create_buffer[f32](2 * NH_V)
    ctx.enqueue_copy(dst_buf=q_d, src_buf=q_h)
    ctx.enqueue_copy(dst_buf=w_d, src_buf=w_h)
    ctx.enqueue_copy(dst_buf=a_d, src_buf=a_h)
    ctx.enqueue_copy(dst_buf=b_d, src_buf=b_h)
    ctx.enqueue_copy(dst_buf=a1_d, src_buf=a1_h)
    ctx.enqueue_copy(dst_buf=b1_d, src_buf=b1_h)
    ctx.enqueue_copy(dst_buf=d_d, src_buf=d_h)
    ctx.enqueue_copy(dst_buf=e_d, src_buf=e_h)

    comptime k2 = amar_ssm_conv_l2g[type_of(q2), type_of(sl), type_of(wl), type_of(q2), type_of(p2), type_of(g2), type_of(dl)]
    comptime k1 = amar_ssm_conv_l2g[type_of(q1), type_of(sl), type_of(wl), type_of(q1), type_of(p1), type_of(g1), type_of(dl)]
    var W = view(ctx, w_d, 0, CONV * 4, wl)
    var Dl = view(ctx, d_d, 0, NH_V, dl)
    var El = view(ctx, e_d, 0, NH_V, dl)
    var oA_h = ctx.enqueue_create_host_buffer[f32](2 * CONV)
    var oB_h = ctx.enqueue_create_host_buffer[f32](2 * CONV)
    var sA_h = ctx.enqueue_create_host_buffer[f32](S_N)
    var sB_h = ctx.enqueue_create_host_buffer[f32](S_N)
    var gA_h = ctx.enqueue_create_host_buffer[f32](4 * NH_V)
    var gB_h = ctx.enqueue_create_host_buffer[f32](4 * NH_V)
    var bad_iters = 0
    var nonfinite = 0
    for it in range(ITERS):
        ctx.enqueue_copy(dst_buf=sA_d, src_buf=s_h)
        ctx.enqueue_copy(dst_buf=sB_d, src_buf=s_h)
        ctx.enqueue_function[k2](
            view(ctx, q_d, 0, 2 * CONV, q2), view(ctx, sA_d, 0, S_N, sl), W, view(ctx, oA_d, 0, 2 * CONV, q2),
            view(ctx, a_d, 0, P_N, p2), view(ctx, b_d, 0, P_N, p2), view(ctx, gA_d, 0, 2 * NH_V, g2), view(ctx, tA_d, 0, 2 * NH_V, g2),
            El, Dl, Int32(0), Int32(0), Int32(SLOTS), Int32(2), grid_dim=CONV_GATE_BLOCKS, block_dim=256)
        for r in range(2):
            ctx.enqueue_function[k1](
                view(ctx, q_d, r * CONV, CONV, q1), view(ctx, sB_d, 0, S_N, sl), W, view(ctx, oB_d, r * CONV, CONV, q1),
                view(ctx, a1_d, r * SPLITK * NH_V, SPLITK * NH_V, p1), view(ctx, b1_d, r * SPLITK * NH_V, SPLITK * NH_V, p1),
                view(ctx, gB_d, r * NH_V, NH_V, g1), view(ctx, tB_d, r * NH_V, NH_V, g1),
                El, Dl, Int32(r), Int32(0), Int32(SLOTS), Int32(1), grid_dim=CONV_GATE_BLOCKS, block_dim=256)
        ctx.enqueue_copy(dst_buf=oA_h, src_buf=oA_d)
        ctx.enqueue_copy(dst_buf=oB_h, src_buf=oB_d)
        ctx.enqueue_copy(dst_buf=sA_h, src_buf=sA_d)
        ctx.enqueue_copy(dst_buf=sB_h, src_buf=sB_d)
        ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, gA_h.unsafe_ptr(), 2 * NH_V, owning=False), src_buf=gA_d)
        ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, gA_h.unsafe_ptr().unsafe_offset(2 * NH_V), 2 * NH_V, owning=False), src_buf=tA_d)
        ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, gB_h.unsafe_ptr(), 2 * NH_V, owning=False), src_buf=gB_d)
        ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, gB_h.unsafe_ptr().unsafe_offset(2 * NH_V), 2 * NH_V, owning=False), src_buf=tB_d)
        ctx.synchronize()
        var diff = 0
        for i in range(2 * CONV):
            if not isfinite(oA_h[i]):
                nonfinite += 1
                if nonfinite <= 5:
                    print("non-finite iter", it, "row", i // CONV, "col", i % CONV, oA_h[i])
            if oA_h[i] != oB_h[i]:
                diff += 1
                if diff <= 3 and bad_iters < 3:
                    print("out differ iter", it, "row", i // CONV, "col", i % CONV, oA_h[i], oB_h[i])
        for i in range(S_N):
            if sA_h[i] != sB_h[i]:
                diff += 1
        for i in range(4 * NH_V):
            if gA_h[i] != gB_h[i]:
                diff += 1
        if diff > 0:
            bad_iters += 1
    for r in range(2):
        var nd = 0
        var lo = CONV
        var hi = -1
        for c in range(CONV):
            if oA_h[r * CONV + c] != oB_h[r * CONV + c]:
                nd += 1
                lo = min(lo, c)
                hi = max(hi, c)
        print("last iter row", r, "differ", nd, "cols", lo, "..", hi)
    # Host conv + silu + l2norm reference over both rows through the ring (slot 0 -> 1 -> 2).
    var hr = List[Float32](length=2 * CONV, fill=0)
    var st = List[Float32](length=3 * CONV, fill=0)
    for c in range(CONV):
        for k in range(3):
            st[k * CONV + c] = s_h[k * CONV + c]
    for r in range(2):
        for c in range(CONV):
            var x = q_h[r * CONV + c]
            var acc = st[c] * w_h[c * 4] + st[CONV + c] * w_h[c * 4 + 1] + st[2 * CONV + c] * w_h[c * 4 + 2] + x * w_h[c * 4 + 3]
            hr[r * CONV + c] = acc / (1 + exp(-acc))
            st[c] = st[CONV + c]
            st[CONV + c] = st[2 * CONV + c]
            st[2 * CONV + c] = x
        for hh in range(2 * KDIM // SSTATE):
            var ss: Float32 = 0
            for j in range(SSTATE):
                ss += hr[r * CONV + hh * SSTATE + j] ** 2
            var inv = 1 / sqrt(ss + SSM_EPS)
            if hh < NH_K:
                inv = inv / sqrt(Float32(SSTATE))
            for j in range(SSTATE):
                hr[r * CONV + hh * SSTATE + j] *= inv
    for r in range(2):
        var wa = 0
        var wb = 0
        for c in range(CONV):
            var t = hr[r * CONV + c]
            if abs(oA_h[r * CONV + c] - t) > 1e-4 * abs(t) + 1e-6:
                wa += 1
            if abs(oB_h[r * CONV + c] - t) > 1e-4 * abs(t) + 1e-6:
                wb += 1
        print("conv vs host row", r, ": m=2 arm wrong", wa, " m=1 arm wrong", wb, "of", CONV)
    var ea = 0
    var eb = 0
    for r in range(2):
        for h in range(NH_V):
            var ar: Float32 = 0
            for sp in range(SPLITK):
                ar += a_h[(sp * 2 + r) * NH_V + h]
            var eg = exp(log1p(exp(ar + d_h[h])) * e_h[h])
            if abs(gA_h[r * NH_V + h] - eg) > 1e-5 * abs(eg) + 1e-6:
                ea += 1
            if abs(gB_h[r * NH_V + h] - eg) > 1e-5 * abs(eg) + 1e-6:
                eb += 1
            if (r * NH_V + h) % 8 == 0:
                print("  r", r, "h", h, "host", eg, "A", gA_h[r * NH_V + h], "B", gB_h[r * NH_V + h])
    print("Eg vs host: m=2 arm wrong", ea, " m=1 arm wrong", eb, "of", 2 * NH_V, " sample h0 r0 host/A/B")
    var gd = 0
    for i in range(4 * NH_V):
        if gA_h[i] != gB_h[i]:
            gd += 1
    print("last iter gate values differ", gd, "of", 4 * NH_V)
    print("conv_l2g m=2 vs 2 x m=1 (CONV", CONV, "NH_V", NH_V, "blocks", CONV_GATE_BLOCKS, "):", bad_iters, "of", ITERS, "iters differ,", nonfinite, "non-finite")
    # Keep every device owner alive past its views (ASAP destruction frees at the last named use).
    _ = (q_d.unsafe_ptr(), w_d.unsafe_ptr(), a_d.unsafe_ptr(), b_d.unsafe_ptr(), a1_d.unsafe_ptr(), b1_d.unsafe_ptr(), d_d.unsafe_ptr(), e_d.unsafe_ptr(), sA_d.unsafe_ptr(), sB_d.unsafe_ptr(), oA_d.unsafe_ptr(), oB_d.unsafe_ptr(), gA_d.unsafe_ptr(), gB_d.unsafe_ptr(), tA_d.unsafe_ptr(), tB_d.unsafe_ptr())
    if bad_iters > 0 or nonfinite > 0:
        raise Error("FAIL conv_l2g parity")
    print("PASS conv_l2g parity")
