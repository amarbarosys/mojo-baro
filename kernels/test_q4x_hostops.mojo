"""q4x device kernels vs the f32 host math they replaced (serve/q4x.mojo before c1d8604).

Gate: every f32 output within 1e-5 of the host reference, relative to the output's largest
magnitude; bf16 outputs within 8e-3 (one bf16 ulp). Inputs are seeded LCG values in [-1, 1].
Attention runs at T = 6 (dense and masked) and at T = 2100 masked, past the sparse-indexer window.
Since lane q4x-r2 the fused kernels (combine inside gnorm, conv+heads, delta+post, two-range swiglu,
two-matrix fmv) are checked against the same host math the separate kernels had.
Build: mojo build kernels/test_q4x_hostops.mojo -I kernels -o .work/test_q4x_hostops
"""
from std.math import sqrt, exp, log1p, cos, sin
from std.memory import bitcast
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from q4x_kernels import (
    amar_q4x_fmv, amar_q4x_gnorm, amar_q4x_swiglu, amar_q4x_hcmix, amar_q4x_bcast,
    amar_q4x_ssm_conv, amar_q4x_delta, amar_q4x_qkprep, amar_q4x_attn, amar_q4x_ffn_out,
    amar_q4x_ple_gate, amar_q4x_ple_conv, amar_q4x_router,
    amar_q4x_gemm, amar_q4x_hcdown, amar_q4x_hcup, amar_q4x_gnorm_ffn,
)

comptime Q4X_XR = 16
comptime Q4X_XK = 16384

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime u8 = DType.uint8
comptime u16 = DType.uint16
comptime H = 2560
comptime HC = 4
comptime HCD = HC * H
comptime HD = 256
comptime NQ = 24
comptime NKV = 2
comptime NROT = 64
comptime SD = 128
comptime SK = 16
comptime SV = 48
comptime CONV_CH = 10240
comptime IO_N = 10336 + SV * SD
comptime TOPK = 10
comptime FF = 640
comptime EPS = Float32(1e-6)
comptime ROPE_BASE = Float32(10000000.0)
comptime Vec = List[Float32]
comptime FP = MutPointer[Scalar[f32], MutAnyOrigin]
comptime BP = MutPointer[Scalar[bf16], MutAnyOrigin]
comptime WP = MutPointer[Scalar[u16], MutAnyOrigin]


def fp(b: DeviceBuffer[f32], off: Int = 0) -> FP:
    return FP(unsafe_from_address=Int(b.unsafe_ptr()) + 4 * off)


def bp(b: DeviceBuffer[bf16], off: Int = 0) -> BP:
    return BP(unsafe_from_address=Int(b.unsafe_ptr()) + 2 * off)


struct Rng:
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> Float32:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return Float32(Int((self.s >> 40) % 2000000)) / 1e6 - 1

    def vec(mut self, n: Int) -> Vec:
        var v = Vec(length=n, fill=0)
        for i in range(n):
            v[i] = self.next()
        return v^


def sigmoid(x: Float32) -> Float32:
    return 1 / (1 + exp(-x))


def silu(x: Float32) -> Float32:
    return x / (1 + exp(-x))


def rms_scale(x: Vec, off: Int, n: Int) -> Float32:
    var s = Float64(0)
    for i in range(n):
        s += Float64(x[off + i]) * Float64(x[off + i])
    return 1 / sqrt(Float32(s / Float64(n)) + EPS)


def l2_scale(x: Vec, off: Int, n: Int) -> Float32:
    var s = Float64(0)
    for i in range(n):
        s += Float64(x[off + i]) * Float64(x[off + i])
    return 1 / max(sqrt(Float32(s)), EPS)


def rope(mut x: Vec, off: Int, pos: Int):
    var ts = Float32(ROPE_BASE) ** (Float32(-2.0) / NROT)
    var theta = Float32(pos)
    for i in range(NROT // 2):
        var c = cos(theta)
        var s = sin(theta)
        var x0 = x[off + i]
        var x1 = x[off + i + NROT // 2]
        x[off + i] = x0 * c - x1 * s
        x[off + i + NROT // 2] = x0 * s + x1 * c
        theta *= ts


def up(ctx: DeviceContext, x: Vec) raises -> DeviceBuffer[f32]:
    var b = ctx.enqueue_create_buffer[f32](len(x))
    with b.map_to_host() as h:
        for i in range(len(x)):
            h[i] = x[i]
    return b^


def down(b: DeviceBuffer[f32], n: Int) raises -> Vec:
    var v = Vec(length=n, fill=0)
    with b.map_to_host() as h:
        for i in range(n):
            v[i] = h[i]
    return v^


def down_bf16(b: DeviceBuffer[bf16], off: Int, n: Int) raises -> Vec:
    var v = Vec(length=n, fill=0)
    with b.map_to_host() as h:
        for i in range(n):
            v[i] = h[off + i].cast[f32]()
    return v^


def same(name: String, got: Vec, want: Vec) raises:
    """Lane q4x-dispatch: a fused kernel must give the unfused kernels' exact f32 bits."""
    var bad = 0
    for i in range(len(want)):
        if bitcast[DType.uint32, 1](SIMD[f32, 1](got[i])) != bitcast[DType.uint32, 1](SIMD[f32, 1](want[i])):
            bad += 1
    print("  ", name, "n", len(want), "bit differences", bad)
    if bad > 0:
        raise Error("FAIL " + name)


def up_bf16(ctx: DeviceContext, x: Vec) raises -> DeviceBuffer[u16]:
    var b = ctx.enqueue_create_buffer[u16](len(x))
    with b.map_to_host() as h:
        for i in range(len(x)):
            h[i] = UInt16(bitcast[DType.uint32, 1](SIMD[f32, 1](x[i]))[0] >> 16)
    return b^


def check(name: String, got: Vec, want: Vec, tol: Float64) raises:
    var scale = Float64(0)
    for v in want:
        scale = max(scale, abs(Float64(v)))
    var worst = Float64(0)
    var at = 0
    for i in range(len(want)):
        var e = abs(Float64(got[i]) - Float64(want[i])) / scale
        if e > worst:
            worst = e
            at = i
    print("  ", name, "n", len(want), "max rel err", worst, "at", at, "got", got[at], "ref", want[at])
    if worst > tol:
        raise Error("FAIL " + name)


def main() raises:
    var ctx = DeviceContext()
    var seqsd = ctx.enqueue_create_buffer[DType.int32](2)
    with seqsd.map_to_host() as sh:
        sh[0] = 0
        sh[1] = 1
    var rng = Rng(7)
    var X = ctx.enqueue_create_buffer[bf16](Q4X_XR * Q4X_XK)
    X.enqueue_fill(0)
    var tf = 1e-5
    var tb = 8e-3

    var res = rng.vec(HCD)
    var w = rng.vec(HCD)
    var resd = up(ctx, res)
    var wd = up(ctx, w)
    var od = ctx.enqueue_create_buffer[f32](HCD)
    var dummy = ctx.enqueue_create_buffer[f32](HC)
    ctx.enqueue_function[amar_q4x_gnorm](fp(resd), Int32(HCD), fp(wd), fp(od), Int32(HCD), bp(X), Int32(HCD), fp(od), Int32(H), fp(dummy), Int32(HC), Int32(0), grid_dim=(HC, 1), block_dim=256)
    ctx.synchronize()
    var xn = Vec(length=HCD, fill=0)
    for c in range(HC):
        var s = rms_scale(res, c * H, H)
        for i in range(H):
            xn[c * H + i] = res[c * H + i] * s * w[c * H + i]
    check("gnorm f32", down(od, HCD), xn, tf)
    check("gnorm bf16 row", down_bf16(X, 0, HCD), xn, tb)

    var g = rng.vec(TOPK * FF)
    var u = rng.vec(TOPK * FF)
    var gu = Vec(length=TOPK * 2 * FF, fill=0)
    for e in range(TOPK):
        for c in range(FF):
            gu[e * 2 * FF + c] = g[e * FF + c]
            gu[e * 2 * FF + FF + c] = u[e * FF + c]
    var gud = up(ctx, gu)
    var g2 = rng.vec(FF)
    var u2 = rng.vec(FF)
    var g2d = up(ctx, g2)
    var u2d = up(ctx, u2)
    ctx.enqueue_function[amar_q4x_swiglu](fp(gud), bp(X, 4 * Q4X_XK), Int32(Q4X_XK), Int32(TOPK * FF), fp(g2d), fp(u2d), Int32(FF), bp(X, 14 * Q4X_XK), Int32(FF), Int32(FF), Int32(FF), grid_dim=(TOPK * FF + FF + 255) // 256, block_dim=256)
    ctx.synchronize()
    var act2 = Vec(length=FF, fill=0)
    for i in range(FF):
        act2[i] = silu(g2[i]) * u2[i]
    check("swiglu second range bf16", down_bf16(X, 14 * Q4X_XK, FF), act2, tb)
    var act = Vec(length=TOPK * FF, fill=0)
    var got = Vec(length=TOPK * FF, fill=0)
    for j in range(TOPK):
        var row = down_bf16(X, (4 + j) * Q4X_XK, FF)
        for i in range(FF):
            act[j * FF + i] = silu(g[j * FF + i]) * u[j * FF + i]
            got[j * FF + i] = row[i]
    check("swiglu rows bf16", got, act, tb)

    var gate = rng.vec(HCD)
    var gated = up(ctx, gate)
    var curd = ctx.enqueue_create_buffer[f32](H)
    ctx.enqueue_function[amar_q4x_hcmix](od.unsafe_ptr(), Int32(HCD), gated.unsafe_ptr(), Int32(HCD), curd.unsafe_ptr(), Int32(H), X.unsafe_ptr().unsafe_offset(2 * Q4X_XK), Int32(Q4X_XK), grid_dim=(H // 256, 1), block_dim=256)
    ctx.synchronize()
    var mixed = Vec(length=H, fill=0)
    for i in range(H):
        var acc = xn[i] * sigmoid(gate[i])
        for c in range(1, HC):
            acc = acc + xn[c * H + i] * sigmoid(gate[c * H + i])
        mixed[i] = acc * (1.0 / HC)
    check("hcmix f32", down(curd, H), mixed, tf)
    check("hcmix bf16 row", down_bf16(X, 2 * Q4X_XK, H), mixed, tb)

    var inj = rng.vec(HC)
    var injd = up(ctx, inj)
    var curh = down(curd, H)
    ctx.enqueue_function[amar_q4x_gnorm](resd.unsafe_ptr(), Int32(HCD), wd.unsafe_ptr(), od.unsafe_ptr(), Int32(HCD), X.unsafe_ptr(), Int32(HCD), curd.unsafe_ptr(), Int32(H), injd.unsafe_ptr(), Int32(HC), Int32(1), grid_dim=(HC, 1), block_dim=256)
    ctx.synchronize()
    var res2 = res.copy()
    for c in range(HC):
        var wc = sigmoid(inj[c] * (1.0 / HC)) * 2
        for i in range(H):
            res2[c * H + i] = res2[c * H + i] + curh[i] * wc
    check("gnorm add=1: combine in place f32", down(resd, HCD), res2, 0)
    var xn2 = Vec(length=HCD, fill=0)
    for c in range(HC):
        var sc2 = rms_scale(res2, c * H, H)
        for i in range(H):
            xn2[c * H + i] = res2[c * H + i] * sc2 * w[c * H + i]
    check("gnorm add=1: norm of the combined f32", down(od, HCD), xn2, tf)

    ctx.enqueue_function[amar_q4x_bcast](curd.unsafe_ptr(), Int32(H), od.unsafe_ptr(), Int32(HCD), grid_dim=(HCD // 256, 1), block_dim=256)
    ctx.synchronize()
    var bc = Vec(length=HCD, fill=0)
    for i in range(HCD):
        bc[i] = curh[i % H]
    check("bcast f32", down(od, HCD), bc, 0)

    var y = rng.vec(CONV_CH)
    var st = rng.vec(3 * CONV_CH)
    var cw = rng.vec(4 * CONV_CH)
    var yd = up(ctx, y)
    var std = up(ctx, st)
    var cwd = up(ctx, cw)
    var io = Vec(length=IO_N, fill=0)
    var iod = up(ctx, io)
    var alpha = rng.vec(SV)
    var beta = rng.vec(SV)
    var dt = rng.vec(SV)
    var a = rng.vec(SV)
    var alphad = up(ctx, alpha)
    var betad = up(ctx, beta)
    var dtd = up(ctx, dt)
    var ad = up(ctx, a)
    ctx.enqueue_function[amar_q4x_ssm_conv](yd.unsafe_ptr(), Int32(CONV_CH), std.unsafe_ptr(), cwd.unsafe_ptr(), iod.unsafe_ptr(), Int32(IO_N), alphad.unsafe_ptr(), betad.unsafe_ptr(), dtd.unsafe_ptr(), ad.unsafe_ptr(), seqsd.unsafe_ptr(), Int32(0), Int32(1), grid_dim=(CONV_CH // SD + 1, 1), block_dim=SD)
    ctx.synchronize()
    var x = Vec(length=CONV_CH, fill=0)
    var st2 = st.copy()
    for c in range(CONV_CH):
        var acc = st[c * 3] * cw[c * 4] + st[c * 3 + 1] * cw[c * 4 + 1] + st[c * 3 + 2] * cw[c * 4 + 2] + y[c] * cw[c * 4 + 3]
        x[c] = silu(acc)
        st2[c * 3] = st[c * 3 + 1]
        st2[c * 3 + 1] = st[c * 3 + 2]
        st2[c * 3 + 2] = y[c]
    check("ssm_conv state f32", down(std, 3 * CONV_CH), st2, 0)
    var ioref = Vec(length=IO_N, fill=0)
    for h in range(SK):
        var qs = l2_scale(x, h * SD, SD)
        var ks = l2_scale(x, SK * SD + h * SD, SD)
        for i in range(SD):
            ioref[h * SD + i] = x[h * SD + i] * qs
            ioref[2048 + h * SD + i] = x[SK * SD + h * SD + i] * ks
    for i in range(SV * SD):
        ioref[4096 + i] = x[2 * SK * SD + i]
    for h in range(SV):
        var gl = alpha[h] + dt[h]
        var sp = gl if gl > 20 else log1p(exp(gl))
        ioref[10240 + h] = exp(sp * a[h])
        ioref[10288 + h] = sigmoid(beta[h])
    var iog = down(iod, IO_N)
    var iohead = Vec(length=10336, fill=0)
    var ioheadr = Vec(length=10336, fill=0)
    for i in range(10336):
        iohead[i] = iog[i]
        ioheadr[i] = ioref[i]
    check("ssm_conv+heads io f32", iohead, ioheadr, tf)

    var z = rng.vec(SV * SD)
    var nw = rng.vec(SD)
    var state = rng.vec(SV * SD * SD)
    for i in range(SV * SD * SD):
        state[i] = state[i] * 0.1
    var stated = up(ctx, state)
    var zd = up(ctx, z)
    var nwd = up(ctx, nw)
    var dscale = 1 / sqrt(Float32(SD))
    ctx.enqueue_function[amar_q4x_delta](stated.unsafe_ptr(), iod.unsafe_ptr(), Int32(IO_N), dscale, zd.unsafe_ptr(), Int32(SV * SD), nwd.unsafe_ptr(), X.unsafe_ptr().unsafe_offset(3 * Q4X_XK), Int32(Q4X_XK), seqsd.unsafe_ptr(), Int32(0), Int32(1), grid_dim=(SV, 1), block_dim=4 * SD)
    ctx.synchronize()
    var o = Vec(length=SV * SD, fill=0)
    for h in range(SV):
        var kh = h % SK
        var dec = ioref[10240 + h]
        var bt = ioref[10288 + h]
        for j in range(SD):
            var sacc = Float32(0)
            for i in range(SD):
                var wv = state[h * SD * SD + i * SD + j] * dec
                state[h * SD * SD + i * SD + j] = wv
                sacc += wv * ioref[2048 + kh * SD + i]
            var dl = (ioref[4096 + h * SD + j] - sacc) * bt
            var acc = Float32(0)
            for i in range(SD):
                var wv = state[h * SD * SD + i * SD + j] + ioref[2048 + kh * SD + i] * dl
                state[h * SD * SD + i * SD + j] = wv
                acc += wv * ioref[kh * SD + i]
            o[h * SD + j] = acc * dscale
    check("delta state f32", down(stated, SV * SD * SD), state, tf)
    var oref = Vec(length=SV * SD, fill=0)
    for h in range(SV):
        var ss = Float64(0)
        for j in range(SD):
            ss += Float64(o[h * SD + j]) * Float64(o[h * SD + j])
        var rs = 1 / sqrt(Float32(ss / Float64(SD)) + EPS)
        for j in range(SD):
            oref[h * SD + j] = o[h * SD + j] * rs * nw[j] * sigmoid(z[h * SD + j])
    iog = down(iod, IO_N)
    var og = Vec(length=SV * SD, fill=0)
    for i in range(SV * SD):
        og[i] = iog[10336 + i]
    check("delta+post f32", og, oref, tf)
    check("delta+post bf16 row", down_bf16(X, 3 * Q4X_XK, SV * SD), oref, tb)

    var pos = 5
    var T = pos + 1
    var qkv = rng.vec(NQ * 2 * HD + 2 * NKV * HD)
    var qn = rng.vec(HD)
    var kn = rng.vec(HD)
    var qkvd = up(ctx, qkv)
    var qnd = up(ctx, qn)
    var knd = up(ctx, kn)
    var qd = ctx.enqueue_create_buffer[f32](NQ * HD)
    var gd = ctx.enqueue_create_buffer[f32](NQ * HD)
    var kcd = ctx.enqueue_create_buffer[f32](T * NKV * HD)
    var vcd = ctx.enqueue_create_buffer[f32](T * NKV * HD)
    var kc = rng.vec(T * NKV * HD)
    var vc = rng.vec(T * NKV * HD)
    with kcd.map_to_host() as hb:
        for i in range(T * NKV * HD):
            hb[i] = kc[i]
    with vcd.map_to_host() as hb:
        for i in range(T * NKV * HD):
            hb[i] = vc[i]
    var ts = Float32(ROPE_BASE) ** (Float32(-2.0) / NROT)
    ctx.enqueue_function[amar_q4x_qkprep](qkvd.unsafe_ptr(), Int32(0), qnd.unsafe_ptr(), knd.unsafe_ptr(), qd.unsafe_ptr(), gd.unsafe_ptr(), Int32(NQ * HD), kcd.unsafe_ptr(), vcd.unsafe_ptr(), Int32(pos), seqsd.unsafe_ptr(), ts, Int32(NQ), Int32(NKV), grid_dim=(NQ + NKV, 1), block_dim=HD)
    ctx.synchronize()
    var q = Vec(length=NQ * HD, fill=0)
    var gt = Vec(length=NQ * HD, fill=0)
    for h in range(NQ):
        for i in range(HD):
            q[h * HD + i] = qkv[h * 2 * HD + i]
            gt[h * HD + i] = qkv[h * 2 * HD + HD + i]
        var s = rms_scale(q, h * HD, HD)
        for i in range(HD):
            q[h * HD + i] = q[h * HD + i] * s * qn[i]
        rope(q, h * HD, pos)
    var k = Vec(length=NKV * HD, fill=0)
    for i in range(NKV * HD):
        k[i] = qkv[NQ * 2 * HD + i]
    for h in range(NKV):
        var s = rms_scale(k, h * HD, HD)
        for i in range(HD):
            k[h * HD + i] = k[h * HD + i] * s * kn[i]
        rope(k, h * HD, pos)
    for i in range(NKV * HD):
        kc[pos * NKV * HD + i] = k[i]
        vc[pos * NKV * HD + i] = qkv[NQ * 2 * HD + NKV * HD + i]
    check("qkprep q f32", down(qd, NQ * HD), q, tf)
    check("qkprep gate f32", down(gd, NQ * HD), gt, 0)
    check("qkprep kc f32", down(kcd, T * NKV * HD), kc, tf)
    check("qkprep vc f32", down(vcd, T * NKV * HD), vc, 0)

    var vis = List[UInt8](length=T, fill=1)
    vis[2] = 0
    vis[4] = 0
    var visd = ctx.enqueue_create_buffer[u8](T)
    with visd.map_to_host() as hb:
        for i in range(T):
            hb[i] = vis[i]
    var scd = ctx.enqueue_create_buffer[f32](NQ * T)
    var sc = 1 / sqrt(Float32(HD))
    for masked in range(2):
        ctx.enqueue_function[amar_q4x_attn](qd.unsafe_ptr(), gd.unsafe_ptr(), Int32(NQ * HD), kcd.unsafe_ptr(), vcd.unsafe_ptr(), scd.unsafe_ptr(), Int32(T), visd.unsafe_ptr(), X.unsafe_ptr().unsafe_offset(3 * Q4X_XK), Int32(Q4X_XK), Int32(T - 1), seqsd.unsafe_ptr(), Int32(masked), Int32(NQ), Int32(NKV), sc, grid_dim=(NQ, 1), block_dim=HD)
        ctx.synchronize()
        var oa = Vec(length=NQ * HD, fill=0)
        for h in range(NQ):
            var kh = h // (NQ // NKV)
            var sco = Vec(length=T, fill=0)
            var mx = Float32(-3.0e38)
            for t in range(T):
                var s = Float32(0)
                for i in range(HD):
                    s += q[h * HD + i] * kc[(t * NKV + kh) * HD + i]
                sco[t] = s * sc
                if masked != 0 and vis[t] == 0:
                    sco[t] = Float32(-3.0e38)
                mx = max(mx, sco[t])
            var den = Float32(0)
            for t in range(T):
                sco[t] = exp(sco[t] - mx) if (masked == 0 or vis[t] != 0) else Float32(0)
                den += sco[t]
            for t in range(T):
                var pw = sco[t] / den
                for i in range(HD):
                    oa[h * HD + i] += pw * vc[(t * NKV + kh) * HD + i]
        for i in range(NQ * HD):
            oa[i] = oa[i] * sigmoid(gt[i])
        check("attn bf16 row masked=" + String(masked), down_bf16(X, 3 * Q4X_XK, NQ * HD), oa, tb)

    var TL = 2100
    var kcl = rng.vec(TL * NKV * HD)
    var vcl = rng.vec(TL * NKV * HD)
    var kcld = up(ctx, kcl)
    var vcld = up(ctx, vcl)
    var visl = List[UInt8](length=TL, fill=0)
    var rl = Rng(11)
    for t in range(TL):
        visl[t] = 1 if rl.next() > 0.2 else 0
    var visld = ctx.enqueue_create_buffer[u8](TL)
    with visld.map_to_host() as hb:
        for i in range(TL):
            hb[i] = visl[i]
    var scld = ctx.enqueue_create_buffer[f32](NQ * TL)
    ctx.enqueue_function[amar_q4x_attn](qd.unsafe_ptr(), gd.unsafe_ptr(), Int32(NQ * HD), kcld.unsafe_ptr(), vcld.unsafe_ptr(), scld.unsafe_ptr(), Int32(TL), visld.unsafe_ptr(), X.unsafe_ptr().unsafe_offset(3 * Q4X_XK), Int32(Q4X_XK), Int32(TL - 1), seqsd.unsafe_ptr(), Int32(1), Int32(NQ), Int32(NKV), sc, grid_dim=(NQ, 1), block_dim=HD)
    ctx.synchronize()
    var ol = Vec(length=NQ * HD, fill=0)
    for h in range(NQ):
        var kh = h // (NQ // NKV)
        var sco = Vec(length=TL, fill=0)
        var mx = Float32(-3.0e38)
        for t in range(TL):
            var s = Float32(0)
            for i in range(HD):
                s += q[h * HD + i] * kcl[(t * NKV + kh) * HD + i]
            sco[t] = s * sc if visl[t] != 0 else Float32(-3.0e38)
            mx = max(mx, sco[t])
        var den = Float32(0)
        for t in range(TL):
            sco[t] = exp(sco[t] - mx) if visl[t] != 0 else Float32(0)
            den += sco[t]
        for t in range(TL):
            var pw = sco[t] / den
            for i in range(HD):
                ol[h * HD + i] += pw * vcl[(t * NKV + kh) * HD + i]
    for i in range(NQ * HD):
        ol[i] = ol[i] * sigmoid(gt[i])
    check("attn bf16 row masked T=2100 (past IDX_W)", down_bf16(X, 3 * Q4X_XK, NQ * HD), ol, tb)

    var ye = rng.vec(TOPK * H)
    var wts = rng.vec(TOPK)
    var ysh = rng.vec(H)
    var sg = rng.vec(1)
    var yed = up(ctx, ye)
    var wtsd = up(ctx, wts)
    var yshd = up(ctx, ysh)
    var sgd = up(ctx, sg)
    ctx.enqueue_function[amar_q4x_ffn_out](yed.unsafe_ptr(), wtsd.unsafe_ptr(), yshd.unsafe_ptr(), Int32(H), sgd.unsafe_ptr(), Int32(1), curd.unsafe_ptr(), Int32(TOPK), grid_dim=(H // 256, 1), block_dim=256)
    ctx.synchronize()
    var fo = Vec(length=H, fill=0)
    for i in range(H):
        var acc = Float32(0)
        for j in range(TOPK):
            acc = acc + ye[j * H + i] * wts[j]
        fo[i] = acc + ysh[i] * sigmoid(sg[0])
    check("ffn_out f32", down(curd, H), fo, tf)

    var key = rng.vec(HCD)
    var qry = rng.vec(HCD)
    var val = rng.vec(H)
    var keyd = up(ctx, key)
    var qryd = up(ctx, qry)
    var vald = up(ctx, val)
    var gtsd = ctx.enqueue_create_buffer[f32](HC)
    var keyp = MutPointer[Scalar[f32], MutAnyOrigin](unsafe_from_address=Int(keyd.unsafe_ptr()))
    var keyg = MutPointer[Scalar[f32], MutAnyOrigin](unsafe_from_address=Int(keyd.unsafe_ptr()))
    ctx.enqueue_function[amar_q4x_ple_gate](keyp, qryd.unsafe_ptr(), vald.unsafe_ptr(), Int32(H), keyg, gtsd.unsafe_ptr(), Int32(HC), grid_dim=(HC, 1), block_dim=256)
    ctx.synchronize()
    var gref = Vec(length=HCD, fill=0)
    var gts = Vec(length=HC, fill=0)
    for c in range(HC):
        var s = Float32(0)
        for i in range(H):
            s += key[c * H + i] * qry[c * H + i]
        s = s * (1 / sqrt(Float32(H)))
        var mag = sqrt(max(abs(s), Float32(1e-6)))
        var sgn = Float32(1) if s > 0 else (Float32(-1) if s < 0 else Float32(0))
        gts[c] = sigmoid(sgn * mag)
        for i in range(H):
            gref[c * H + i] = val[i] * gts[c]
    check("ple_gate gts f32", down(gtsd, HC), gts, tf)
    check("ple_gate gated f32 (in place)", down(keyd, HCD), gref, tf)

    var nrm = rng.vec(HCD)
    var pcw = rng.vec(4 * HCD)
    var hist = rng.vec(9 * HCD)
    var nrmd = up(ctx, nrm)
    var pcwd = up(ctx, pcw)
    var histd = up(ctx, hist)
    var pres = rng.vec(HCD)
    var presd = up(ctx, pres)
    ctx.enqueue_function[amar_q4x_ple_conv](nrmd.unsafe_ptr(), keyd.unsafe_ptr(), pcwd.unsafe_ptr(), histd.unsafe_ptr(), presd.unsafe_ptr(), seqsd.unsafe_ptr(), Int32(0), Int32(1), grid_dim=(HCD // 256, 1), block_dim=256)
    ctx.synchronize()
    var out = pres.copy()
    var hist2 = hist.copy()
    for c in range(HCD):
        var hh = c * 9
        var acc = hist[hh] * pcw[c * 4] + hist[hh + 3] * pcw[c * 4 + 1] + hist[hh + 6] * pcw[c * 4 + 2] + nrm[c] * pcw[c * 4 + 3]
        var cv = silu(acc)
        for j in range(8):
            hist2[hh + j] = hist[hh + j + 1]
        hist2[hh + 8] = nrm[c]
        out[c] = out[c] + (gref[c] + cv)
    check("ple_conv res f32", down(presd, HCD), out, tf)
    check("ple_conv hist f32", down(histd, 9 * HCD), hist2, 0)

    var kk = 2560
    var nn = 48
    var wf = rng.vec(kk * nn)
    var xf = rng.vec(kk)
    var wq = ctx.enqueue_create_buffer[u16](kk * nn)
    with wq.map_to_host() as hb:
        for i in range(kk * nn):
            var bits = bitcast[DType.uint32, 1](SIMD[f32, 1](wf[i]))[0]
            hb[i] = UInt16(bits >> 16)
            wf[i] = bitcast[f32, 1](SIMD[DType.uint32, 1](UInt32(hb[i]) << 16))[0]
    var xfd = up(ctx, xf)
    var yfd = ctx.enqueue_create_buffer[f32](nn)
    var wqp = WP(unsafe_from_address=Int(wq.unsafe_ptr()))
    var wqp2 = WP(unsafe_from_address=Int(wq.unsafe_ptr()))
    ctx.enqueue_function[amar_q4x_fmv[1]](wqp, wqp2, Int32(nn), xfd.unsafe_ptr(), Int32(kk), yfd.unsafe_ptr(), Int32(nn), Int32(kk), Int32(nn), Int32(1), grid_dim=((nn + 7) // 8, 1), block_dim=256)
    ctx.synchronize()
    var yref = Vec(length=nn, fill=0)
    for j in range(nn):
        var acc = Float64(0)
        for i in range(kk):
            acc += Float64(wf[j * kk + i]) * Float64(xf[i])
        yref[j] = Float32(acc)
    check("fmv bf16 w f32 x", down(yfd, nn), yref, tf)
    var yf2 = ctx.enqueue_create_buffer[f32](2 * nn)
    ctx.enqueue_function[amar_q4x_fmv[1]](wqp, wqp2, Int32(nn), xfd.unsafe_ptr(), Int32(kk), yf2.unsafe_ptr(), Int32(2 * nn), Int32(kk), Int32(2 * nn), Int32(1), grid_dim=((2 * nn + 7) // 8, 1), block_dim=256)
    ctx.synchronize()
    var yref2 = Vec(length=2 * nn, fill=0)
    for j in range(2 * nn):
        yref2[j] = yref[j % nn]
    check("fmv two matrices", down(yf2, 2 * nn), yref2, tf)
    var NE = 512
    var lg = rng.vec(NE)
    for i in range(NE):
        lg[i] = lg[i] * 8
    lg[77] = lg[5]
    var lgd = up(ctx, lg)
    var belld = ctx.enqueue_create_buffer[DType.int32](16)
    belld.enqueue_fill(0)
    var wtd = ctx.enqueue_create_buffer[f32](TOPK)
    ctx.enqueue_function[amar_q4x_router](fp(lgd), Int32(NE), belld.unsafe_ptr(), fp(wtd), Int32(7), Int32(TOPK), fp(lgd), Int32(0), grid_dim=1, block_dim=NE)
    ctx.synchronize()
    var pr = lg.copy()
    var mxl = pr[0]
    for v in pr:
        mxl = max(mxl, v)
    var den = Float32(0)
    for i in range(NE):
        pr[i] = exp(pr[i] - mxl)
        den += pr[i]
    for i in range(NE):
        pr[i] = pr[i] / den
    var ids = List[Int]()
    var rwts = Vec()
    var tk = List[Bool](length=NE, fill=False)
    var ws = Float32(0)
    for _ in range(TOPK):
        var b = -1
        for i in range(NE):
            if not tk[i] and (b < 0 or pr[i] > pr[b]):
                b = i
        tk[b] = True
        ids.append(b)
        rwts.append(pr[b])
        ws += pr[b]
    ws = max(ws, Float32(6.103515625e-5))
    for j in range(TOPK):
        rwts[j] = rwts[j] / ws
    var badids = 0
    with belld.map_to_host() as hb:
        if Int(hb[0]) != 7:
            raise Error("FAIL router bell seq " + String(hb[0]))
        for j in range(TOPK):
            if Int(hb[1 + j]) != ids[j]:
                badids += 1
                print("  router id mismatch", j, hb[1 + j], ids[j])
    print("   router top-k ids mismatches", badids, "(host order, lowest index on ties)")
    if badids != 0:
        raise Error("FAIL router ids")
    check("router weights f32", down(wtd, TOPK), rwts, 0)
    var kz = up(ctx, Vec(length=NE, fill=0))
    belld.enqueue_fill(0)
    ctx.enqueue_function[amar_q4x_router](fp(lgd), Int32(NE), belld.unsafe_ptr(), fp(wtd), Int32(8), Int32(TOPK), fp(kz), Int32(1), grid_dim=1, block_dim=NE)
    ctx.synchronize()
    with belld.map_to_host() as hb:
        for j in range(TOPK):
            if Int(hb[1 + j]) != ids[j]:
                raise Error("FAIL router keep-all ids differ at " + String(j))
    check("router keep-all weights", down(wtd, TOPK), rwts, 0)
    var kb = Vec(length=NE, fill=0)
    var mk = List[Bool](length=NE, fill=False)
    for j in range(3):
        kb[ids[j]] = Float32.MIN_FINITE
        mk[ids[j]] = True
    var kbd = up(ctx, kb)
    belld.enqueue_fill(0)
    ctx.enqueue_function[amar_q4x_router](fp(lgd), Int32(NE), belld.unsafe_ptr(), fp(wtd), Int32(9), Int32(TOPK), fp(kbd), Int32(1), grid_dim=1, block_dim=NE)
    ctx.synchronize()
    var pm = lg.copy()
    var mxm = Float32.MIN_FINITE
    for i in range(NE):
        if not mk[i]:
            mxm = max(mxm, pm[i])
    var denm = Float32(0)
    for i in range(NE):
        pm[i] = Float32(0) if mk[i] else exp(pm[i] - mxm)
        denm += pm[i]
    for i in range(NE):
        pm[i] = pm[i] / denm
    var mids = List[Int]()
    var mwts = Vec()
    var tkm = mk.copy()
    var wsm = Float32(0)
    for _ in range(TOPK):
        var b = -1
        for i in range(NE):
            if not tkm[i] and (b < 0 or pm[i] > pm[b]):
                b = i
        tkm[b] = True
        mids.append(b)
        mwts.append(pm[b])
        wsm += pm[b]
    wsm = max(wsm, Float32(6.103515625e-5))
    for j in range(TOPK):
        mwts[j] = mwts[j] / wsm
    with belld.map_to_host() as hb:
        for j in range(TOPK):
            if Int(hb[1 + j]) != mids[j] or mk[Int(hb[1 + j])]:
                raise Error("FAIL router keep mask id " + String(j) + ": got " + String(hb[1 + j]) + " want " + String(mids[j]))
    check("router keep-mask weights vs host", down(wtd, TOPK), mwts, 1e-6)
    var mw = down(wtd, TOPK)
    var lgm = lg.copy()
    for i in range(NE):
        if mk[i]:
            lgm[i] = Float32.MIN_FINITE
    var lgmd = up(ctx, lgm)
    belld.enqueue_fill(0)
    ctx.enqueue_function[amar_q4x_router](fp(lgmd), Int32(NE), belld.unsafe_ptr(), fp(wtd), Int32(10), Int32(TOPK), fp(kz), Int32(0), grid_dim=1, block_dim=NE)
    ctx.synchronize()
    with belld.map_to_host() as hb:
        for j in range(TOPK):
            if Int(hb[1 + j]) != mids[j]:
                raise Error("FAIL router masked-logit ids differ at " + String(j))
    check("router keep-mask = masked logits, same kernel", down(wtd, TOPK), mw, 0)
    # lane q4x-dispatch: fused launches against the launches they replace, bit for bit
    var wdn = rng.vec(320 * HCD)
    var win = rng.vec(4 * HCD)
    var xnv = rng.vec(HCD)
    var xnd = up(ctx, xnv)
    var wdnd = up_bf16(ctx, wdn)
    var wind = up_bf16(ctx, win)
    var xx = ctx.enqueue_create_buffer[bf16](HCD)
    with xx.map_to_host() as h:
        for i in range(HCD):
            h[i] = xnv[(i * 7) % HCD].cast[bf16]()
    var Xd = TileTensor(xx, row_major[1, HCD]())
    var tl1 = ctx.enqueue_create_buffer[DType.int32](4)
    with tl1.map_to_host() as h:
        h[0] = 0
        h[1] = 0
        h[2] = 1
        h[3] = 0
    var tlp = MutPointer[Scalar[DType.int32], MutAnyOrigin](unsafe_from_address=Int(tl1.unsafe_ptr()))
    var tab0 = ctx.enqueue_create_buffer[DType.int64](1)
    tab0.enqueue_fill(0)
    var yA = ctx.enqueue_create_buffer[f32](16480)
    var xqA = ctx.enqueue_create_buffer[bf16](H)
    var injA = ctx.enqueue_create_buffer[f32](HC)
    var yB = ctx.enqueue_create_buffer[f32](16480)
    var xqB = ctx.enqueue_create_buffer[bf16](H)
    var injB = ctx.enqueue_create_buffer[f32](HC)
    var wdnp = MutPointer[Scalar[u8], MutAnyOrigin](unsafe_from_address=Int(wdnd.unsafe_ptr()))
    ctx.enqueue_function[amar_q4x_gemm[30, 1, False, type_of(Xd).LayoutType]](Xd, wdnp, tab0.unsafe_ptr(), tlp, tlp, tlp, yA.unsafe_ptr(), xqA.unsafe_ptr(), Int32(HCD), Int32(320), Int32(320), Int32(2 * HCD), Int32(32), Int32(16480), Int32(H), Int32(1), Float32(0.25), Int32(0), grid_dim=(40, 1), block_dim=256)
    var winp = WP(unsafe_from_address=Int(wind.unsafe_ptr()))
    ctx.enqueue_function[amar_q4x_fmv[1]](winp, winp, Int32(HC), xnd.unsafe_ptr(), Int32(HCD), injA.unsafe_ptr(), Int32(HC), Int32(HCD), Int32(HC), Int32(1), grid_dim=(1, 1), block_dim=256)
    ctx.enqueue_function[amar_q4x_hcdown[type_of(Xd).LayoutType]](Xd, wdnp, Int32(2 * HCD), Int32(HCD), Int32(320), yB.unsafe_ptr(), xqB.unsafe_ptr(), Float32(0.25), wind.unsafe_ptr(), xnd.unsafe_ptr(), injB.unsafe_ptr(), Int32(HC), grid_dim=(41, 1), block_dim=256)
    ctx.synchronize()
    same("hcdown y", down(yB, 320), down(yA, 320))
    same("hcdown silu bf16 row", down_bf16(xqB, 0, 320), down_bf16(xqA, 0, 320))
    same("hcdown inject", down(injB, HC), down(injA, HC))

    var wup = rng.vec(HCD * 320)
    var wupd = up_bf16(ctx, wup)
    var xl = ctx.enqueue_create_buffer[bf16](320)
    with xl.map_to_host() as h:
        for i in range(320):
            h[i] = xnv[i * 3].cast[bf16]()
    var Xu = TileTensor(xl, row_major[1, 320]())
    var wupp = MutPointer[Scalar[u8], MutAnyOrigin](unsafe_from_address=Int(wupd.unsafe_ptr()))
    var curA = ctx.enqueue_create_buffer[f32](H)
    var curB = ctx.enqueue_create_buffer[f32](H)
    var xcA = ctx.enqueue_create_buffer[bf16](H)
    var xcB = ctx.enqueue_create_buffer[bf16](H)
    ctx.enqueue_function[amar_q4x_gemm[30, 1, False, type_of(Xu).LayoutType]](Xu, wupp, tab0.unsafe_ptr(), tlp, tlp, tlp, yA.unsafe_ptr(), xqA.unsafe_ptr(), Int32(320), Int32(HCD), Int32(HCD), Int32(640), Int32(32), Int32(16480), Int32(0), Int32(0), Float32(0), Int32(0), grid_dim=(HCD // 8, 1), block_dim=256)
    ctx.enqueue_function[amar_q4x_hcmix](xnd.unsafe_ptr(), Int32(HCD), yA.unsafe_ptr(), Int32(16480), curA.unsafe_ptr(), Int32(H), xcA.unsafe_ptr(), Int32(H), grid_dim=(H // 256, 1), block_dim=256)
    ctx.enqueue_function[amar_q4x_hcup[type_of(Xu).LayoutType]](Xu, wupp, Int32(640), Int32(320), xnd.unsafe_ptr(), curB.unsafe_ptr(), xcB.unsafe_ptr(), grid_dim=(H // 8, 1), block_dim=256)
    ctx.synchronize()
    same("hcup cur", down(curB, H), down(curA, H))
    same("hcup bf16 row", down_bf16(xcB, 0, H), down_bf16(xcA, 0, H))

    var fT = 2
    var fres = rng.vec(fT * HCD)
    var fye = rng.vec(fT * TOPK * H)
    var fwts = rng.vec(fT * TOPK)
    var fsh = rng.vec(fT * H)
    var fsg = rng.vec(fT)
    var finj = rng.vec(fT * HC)
    var fyed = up(ctx, fye)
    var fwtd = up(ctx, fwts)
    var fshd = up(ctx, fsh)
    var fsgd = up(ctx, fsg)
    var finjd = up(ctx, finj)
    var fresA = up(ctx, fres)
    var fresB = up(ctx, fres)
    var foA = ctx.enqueue_create_buffer[f32](fT * HCD)
    var foB = ctx.enqueue_create_buffer[f32](fT * HCD)
    var fxA = ctx.enqueue_create_buffer[bf16](fT * HCD)
    var fxB = ctx.enqueue_create_buffer[bf16](fT * HCD)
    var fcur = ctx.enqueue_create_buffer[f32](fT * H)
    ctx.enqueue_function[amar_q4x_ffn_out](fyed.unsafe_ptr(), fwtd.unsafe_ptr(), fshd.unsafe_ptr(), Int32(H), fsgd.unsafe_ptr(), Int32(1), fcur.unsafe_ptr(), Int32(TOPK), grid_dim=(H // 256, fT), block_dim=256)
    ctx.enqueue_function[amar_q4x_gnorm](fresA.unsafe_ptr(), Int32(HCD), wd.unsafe_ptr(), foA.unsafe_ptr(), Int32(HCD), fxA.unsafe_ptr(), Int32(HCD), fcur.unsafe_ptr(), Int32(H), finjd.unsafe_ptr(), Int32(HC), Int32(1), grid_dim=(HC, fT), block_dim=256)
    ctx.enqueue_function[amar_q4x_gnorm_ffn](fresB.unsafe_ptr(), Int32(HCD), wd.unsafe_ptr(), foB.unsafe_ptr(), Int32(HCD), fxB.unsafe_ptr(), Int32(HCD), fyed.unsafe_ptr(), fwtd.unsafe_ptr(), fshd.unsafe_ptr(), Int32(H), fsgd.unsafe_ptr(), Int32(1), Int32(TOPK), finjd.unsafe_ptr(), Int32(HC), grid_dim=(HC, fT), block_dim=256)
    ctx.synchronize()
    same("gnorm_ffn residual", down(fresB, fT * HCD), down(fresA, fT * HCD))
    same("gnorm_ffn f32", down(foB, fT * HCD), down(foA, fT * HCD))
    same("gnorm_ffn bf16 row", down_bf16(fxB, 0, fT * HCD), down_bf16(fxA, 0, fT * HCD))
    print("PASS test_q4x_hostops")
