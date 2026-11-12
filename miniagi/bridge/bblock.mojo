"""Latent bridge: block + readers, forward and backward on explicit buffers (no trainer state).

Shared by the trainer (bridge.mojo) and P6 serve (`-I miniagi/bridge -I miniagi/latent -I miniagi/train`).
Parameter order: PNAMES, sizes from psizes(cfg); the first NDEC tensors take weight decay.
"""
from std.math import ceildiv, sqrt
from max.gpu.host import DeviceContext, DeviceBuffer
from ops import F, I, THREADS, grid, swiglu_fwd_k, swiglu_bwd_k, add_k
from lops import Gemm, copy_k, bcast_k, colsum_k, sumsq_k, CH
from bkern import *

comptime PNAMES = "wq wk wv wo w1 w3 w2 wp1 wp2 wqa wka wva woa wkb wvb wc z0 esrc gm g1 g2 gz gan ga gb"
comptime WQ = 0
comptime WK = 1
comptime WV = 2
comptime WO = 3
comptime W1 = 4
comptime W3 = 5
comptime W2 = 6
comptime WP1 = 7
comptime WP2 = 8
comptime WQA = 9
comptime WKA = 10
comptime WVA = 11
comptime WOA = 12
comptime WKB = 13
comptime WVB = 14
comptime WC = 15
comptime Z0 = 16
comptime ESRC = 17
comptime GM = 18
comptime G1 = 19
comptime G2 = 20
comptime GZ = 21
comptime GAN = 22
comptime GA = 23
comptime GB = 24
comptime NPAR = 25
comptime NDEC = 16
comptime EPS: Float32 = 1e-6
comptime S_ONE = 0
comptime S_HT2 = 1
comptime S_RA2 = 2
comptime S_AOT2 = 3
comptime S_RB2 = 4
comptime S_CT2 = 5
comptime S_RC2 = 6
comptime S_COS = 7
comptime S_CE = 8
comptime S_HIT = 9
comptime NSC = 16


@fieldwise_init
struct Cfg(ImplicitlyCopyable):
    var d: Int
    var nh: Int
    var ff: Int
    var nz: Int
    var vocab: Int
    var nsrc: Int
    var s1: Int
    var s2: Int
    var nhb: Int
    var nkvb: Int
    var hdb: Int
    var qlo: Int
    var R: Int

    def hd(self) -> Int:
        return self.d // self.nh

    def hdkb(self) -> Int:
        return self.hdb - self.qlo

    def qw(self) -> Int:
        return self.nhb * self.hdb

    def kbw(self) -> Int:
        return self.nkvb * self.hdkb()

    def vbw(self) -> Int:
        return self.nkvb * self.hdb


def psizes(c: Cfg) -> List[Int]:
    var d = c.d
    var s: List[Int] = [d * d, d * d, d * d, d * d, d * c.ff, d * c.ff, c.ff * d, c.s1 * d, c.s2 * d, d * d, d * d, d * d, d * d,
                        d * c.kbw(), d * c.vbw(), d * d, c.nz * d, 3 * d, d, d, d, d, d, d, c.nhb]
    return s^


def ptrs(c: Cfg, base: F) -> List[F]:
    var s = psizes(c)
    var out = List[F]()
    var o = 0
    for i in range(len(s)):
        out.append(base.unsafe_offset(o))
        o += s[i]
    return out^


def ntotal(c: Cfg) -> Int:
    var n = 0
    for v in psizes(c):
        n += v
    return n


def ndecay(c: Cfg) -> Int:
    var s = psizes(c)
    var n = 0
    for i in range(NDEC):
        n += s[i]
    return n


@fieldwise_init
struct Run(ImplicitlyCopyable):
    var ra: Bool
    var rb: Bool
    var rc: Bool
    var single: Bool
    var own: Bool
    var wa: Float32
    var wb: Float32
    var wc: Float32
    var lam: Float32
    var wsc: Float32


@fieldwise_init
struct Bat(ImplicitlyCopyable):
    """One micro-batch on the device: packed source rows, answer rows, tables (4 ints per item: qoff nq koff nk)."""
    var lsum: Int
    var na: Int
    var x0: F
    var x1: F
    var x2: F
    var hA: F
    var hT: F
    var qB: F
    var aoS: F
    var aoT: F
    var ct: F
    var head: F
    var wo47: F
    var tabZ: I
    var tabA: I
    var map: I
    var srcid: I
    var tgt: I


struct Act(Movable):
    var mem: DeviceBuffer[DType.float32]
    var top: Int
    var mb: Int
    var nkmax: Int
    var rs: Int
    var rsf: Int
    var pzn: Int
    var lm: Int
    var u: F
    var mpre: F
    var m: F
    var rm: F
    var kk: F
    var vv: F
    var pz: F
    var dsz: F
    var zin: F
    var n1: F
    var r1: F
    var qz: F
    var oz: F
    var za: F
    var n2: F
    var r2: F
    var aa: F
    var bb: F
    var ss: F
    var kz: F
    var vz: F
    var zr: F
    var zn: F
    var rz: F
    var dz: F
    var dza: F
    var dn: F
    var dqz: F
    var doz: F
    var dkz: F
    var dvz: F
    var dff1: F
    var dff2: F
    var dff3: F
    var dkk: F
    var dvv: F
    var dm: F
    var du: F
    var dzn: F
    var hn: F
    var rh: F
    var qa: F
    var ka: F
    var va: F
    var oa: F
    var ua: F
    var hp: F
    var pa: F
    var logits: F
    var lrow: F
    var hit: F
    var ra: F
    var dhp: F
    var dua: F
    var doa: F
    var dqa: F
    var dka: F
    var dva: F
    var dsa: F
    var dhn: F
    var kb: F
    var vb: F
    var delta: F
    var dg: F
    var yb: F
    var rb: F
    var pb: F
    var ddg: F
    var ddelta: F
    var dqb: F
    var dkb: F
    var dvb: F
    var dsb: F
    var cc: F
    var rc: F
    var dcc: F
    var lc: F
    var part: F
    var tmp: F
    var sc: F

    def __init__(out self, ctx: DeviceContext, c: Cfg, mb: Int, maxlp: Int, maxla: Int) raises:
        var d = c.d
        var nz = mb * c.nz
        var lm = c.nsrc * mb * maxlp
        var na = mb * maxla
        var nkmax = max(c.nsrc * maxlp, c.nz)
        var pzn = mb * c.nh * NQ * nkmax
        var pan = mb * c.nh * NQ * c.nz
        var pbn = mb * c.nhb * NQ * c.nz
        var rs = nz * d
        var rsf = nz * c.ff
        var partn = ceildiv(max(lm, max(na, nz)), RC) * max(d, c.qw())
        var total = 9 * lm * d + 2 * lm + (c.R + 1) * pzn + c.R * (6 * rs + 2 * nz + 3 * rsf) + 2 * c.R * rs + 3 * rs + nz + 7 * rs + 3 * rsf
        total += 4 * (na * d) + na + 2 * rs + na * d + pan + na * c.vocab + 2 * na + 5 * na * d + 2 * rs + pan + na * d
        total += nz * (c.kbw() + c.vbw()) + 2 * na * c.qw() + 2 * na * d + pbn + 3 * na * c.qw() + nz * (c.kbw() + c.vbw()) + pbn
        total += 3 * rs + nz + partn + max(d, c.qw()) + NSC
        self.mem = ctx.enqueue_create_buffer[DType.float32](total)
        ctx.enqueue_memset(self.mem, 0)
        self.mb = mb
        self.nkmax = nkmax
        self.rs = rs
        self.rsf = rsf
        self.pzn = pzn
        self.lm = lm
        var b = F(unsafe_from_address=Int(self.mem.unsafe_ptr()))
        var top = 0
        self.u = take(b, top, lm * d)
        self.mpre = take(b, top, lm * d)
        self.m = take(b, top, lm * d)
        self.rm = take(b, top, lm)
        self.kk = take(b, top, lm * d)
        self.vv = take(b, top, lm * d)
        self.pz = take(b, top, c.R * pzn)
        self.dsz = take(b, top, pzn)
        self.zin = take(b, top, c.R * rs)
        self.n1 = take(b, top, c.R * rs)
        self.r1 = take(b, top, c.R * nz)
        self.qz = take(b, top, c.R * rs)
        self.oz = take(b, top, c.R * rs)
        self.za = take(b, top, c.R * rs)
        self.n2 = take(b, top, c.R * rs)
        self.r2 = take(b, top, c.R * nz)
        self.aa = take(b, top, c.R * rsf)
        self.bb = take(b, top, c.R * rsf)
        self.ss = take(b, top, c.R * rsf)
        self.kz = take(b, top, c.R * rs)
        self.vz = take(b, top, c.R * rs)
        self.zr = take(b, top, rs)
        self.zn = take(b, top, rs)
        self.rz = take(b, top, nz)
        self.dz = take(b, top, rs)
        self.dza = take(b, top, rs)
        self.dn = take(b, top, rs)
        self.dqz = take(b, top, rs)
        self.doz = take(b, top, rs)
        self.dkz = take(b, top, rs)
        self.dvz = take(b, top, rs)
        self.dff1 = take(b, top, rsf)
        self.dff2 = take(b, top, rsf)
        self.dff3 = take(b, top, rsf)
        self.dkk = take(b, top, lm * d)
        self.dvv = take(b, top, lm * d)
        self.dm = take(b, top, lm * d)
        self.du = take(b, top, lm * d)
        self.dzn = take(b, top, rs)
        self.hn = take(b, top, na * d)
        self.rh = take(b, top, na)
        self.qa = take(b, top, na * d)
        self.ka = take(b, top, rs)
        self.va = take(b, top, rs)
        self.oa = take(b, top, na * d)
        self.ua = take(b, top, na * d)
        self.hp = take(b, top, na * d)
        self.pa = take(b, top, pan)
        self.logits = take(b, top, na * c.vocab)
        self.lrow = take(b, top, na)
        self.hit = take(b, top, na)
        self.ra = take(b, top, na * d)
        self.dhp = take(b, top, na * d)
        self.dua = take(b, top, na * d)
        self.doa = take(b, top, na * d)
        self.dqa = take(b, top, na * d)
        self.dka = take(b, top, rs)
        self.dva = take(b, top, rs)
        self.dsa = take(b, top, pan)
        self.dhn = take(b, top, na * d)
        self.kb = take(b, top, nz * c.kbw())
        self.vb = take(b, top, nz * c.vbw())
        self.delta = take(b, top, na * c.qw())
        self.dg = take(b, top, na * c.qw())
        self.yb = take(b, top, na * d)
        self.rb = take(b, top, na * d)
        self.pb = take(b, top, pbn)
        self.ddg = take(b, top, na * c.qw())
        self.ddelta = take(b, top, na * c.qw())
        self.dqb = take(b, top, na * c.qw())
        self.dkb = take(b, top, nz * c.kbw())
        self.dvb = take(b, top, nz * c.vbw())
        self.dsb = take(b, top, pbn)
        self.cc = take(b, top, rs)
        self.rc = take(b, top, rs)
        self.dcc = take(b, top, rs)
        self.lc = take(b, top, nz)
        self.part = take(b, top, partn)
        self.tmp = take(b, top, max(d, c.qw()))
        self.sc = take(b, top, NSC)
        self.top = top

    def rnd(self, base: F, r: Int, ff: Bool = False) -> F:
        return base.unsafe_offset(r * (self.rsf if ff else self.rs))


def take(b: F, mut top: Int, n: Int) -> F:
    var p = b.unsafe_offset(top)
    top += n
    return p


def zero(ctx: DeviceContext, p: F, n: Int) raises:
    ctx.enqueue_memset(DeviceBuffer[DType.float32](ctx, p, n, owning=False), 0)


def rms(ctx: DeviceContext, x: F, w: F, y: F, r: F, n: Int, d: Int) raises:
    ctx.enqueue_function[rmsn_fwd_k](x, w, y, r, Int32(n), Int32(d), EPS, grid_dim=ceildiv(n, NW), block_dim=AT)


def rms_bwd(ctx: DeviceContext, a: Act, x: F, w: F, r: F, dy: F, dx: F, dw: F, n: Int, d: Int, acc: Bool) raises:
    ctx.enqueue_function[rmsn_dx_k](x, w, r, dy, dx, Int32(n), Int32(d), Int32(acc), grid_dim=ceildiv(n, NW), block_dim=AT)
    coldot(ctx, a, x, dy, r, dw, n, d, True)


def coldot(ctx: DeviceContext, a: Act, x: F, y: F, r: F, dst: F, n: Int, d: Int, user: Bool) raises:
    var nc = ceildiv(n, RC)
    ctx.enqueue_function[coldot_k](x, y, r, a.part, Int32(n), Int32(d), Int32(user), grid_dim=(nc, ceildiv(d, AT)), block_dim=AT)
    ctx.enqueue_function[colsum_k](a.part, dst, Int32(nc), Int32(d), grid_dim=grid(d), block_dim=THREADS)


def cols(ctx: DeviceContext, x: F, g: F, y: F, n: Int, d: Int, hs: Int) raises:
    ctx.enqueue_function[scale_cols_k](x, g, y, Int32(n), Int32(d), Int32(hs), grid_dim=grid(n * d), block_dim=THREADS)


def copy(ctx: DeviceContext, src: F, dst: F, n: Int, acc: Bool = False) raises:
    ctx.enqueue_function[copy_k](src, dst, Int32(n), Int32(acc), grid_dim=grid(n), block_dim=THREADS)


def diff(ctx: DeviceContext, x: F, t: F, r: F, n: Int) raises:
    ctx.enqueue_function[diff_k](x, t, r, Int32(n), grid_dim=grid(n), block_dim=THREADS)


def saxpy(ctx: DeviceContext, x: F, y: F, alpha: Float32, s: F, n: Int, acc: Bool) raises:
    ctx.enqueue_function[saxpy_k](x, y, alpha, s, Int32(n), Int32(acc), grid_dim=grid(n), block_dim=THREADS)


def sumsq(ctx: DeviceContext, a: Act, x: F, n: Int, res: F) raises:
    var np = ceildiv(n, CH)
    ctx.enqueue_function[sumsq_k](x, a.part, Int32(n), grid_dim=grid(np), block_dim=THREADS)
    ctx.enqueue_function[reduce_k](a.part, Int32(np), res, Int32(0), grid_dim=1, block_dim=AT)


def total(ctx: DeviceContext, x: F, n: Int, res: F) raises:
    ctx.enqueue_function[reduce_k](x, Int32(n), res, Int32(0), grid_dim=1, block_dim=AT)


@fieldwise_init
struct At(ImplicitlyCopyable):
    """Strides of one attention call: (row stride, head stride) for q, k, v, o; head dims; GQA group; heads; P stride."""
    var qrs: Int
    var qhs: Int
    var krs: Int
    var khs: Int
    var vrs: Int
    var vhs: Int
    var ors: Int
    var ohs: Int
    var hdk: Int
    var hdv: Int
    var grp: Int
    var nh: Int
    var nkmax: Int


def attn_fwd(ctx: DeviceContext, t: At, q: F, k: F, v: F, o: F, p: F, tab: I, mb: Int) raises:
    ctx.enqueue_function[attn_fwd_k](q, k, v, o, p, tab, Int32(t.qrs), Int32(t.qhs), Int32(t.krs), Int32(t.khs), Int32(t.vrs), Int32(t.vhs),
        Int32(t.ors), Int32(t.ohs), Int32(t.hdk), Int32(t.hdv), Int32(t.grp), Int32(t.nh), Int32(t.nkmax), Float32(t.hdk) ** -0.5,
        grid_dim=(mb, t.nh // t.grp), block_dim=AT)


def attn_bwd(ctx: DeviceContext, t: At, q: F, k: F, v: F, o: F, do: F, p: F, ds: F, dq: F, dk: F, dv: F, tab: I, mb: Int) raises:
    ctx.enqueue_function[attn_bwd_k](q, k, v, o, do, p, ds, dq, dk, dv, tab, Int32(t.qrs), Int32(t.qhs), Int32(t.krs), Int32(t.khs),
        Int32(t.vrs), Int32(t.vhs), Int32(t.ors), Int32(t.ohs), Int32(t.hdk), Int32(t.hdv), Int32(t.grp), Int32(t.nh), Int32(t.nkmax),
        Float32(t.hdk) ** -0.5, grid_dim=(mb, t.nh // t.grp), block_dim=AT)


def at_block(c: Cfg, a: Act) -> At:
    var d = c.d
    var hd = c.hd()
    return At(d, hd, d, hd, d, hd, d, hd, hd, hd, 1, c.nh, a.nkmax)


def at_a(c: Cfg) -> At:
    var d = c.d
    var hd = c.hd()
    return At(d, hd, d, hd, d, hd, d, hd, hd, hd, 1, c.nh, c.nz)


def at_b(c: Cfg) -> At:
    return At(c.qw(), c.hdb, c.kbw(), c.hdkb(), c.vbw(), c.hdb, c.qw(), c.hdb, c.hdkb(), c.hdb, c.nhb // c.nkvb, c.nhb, c.nz)


def block_fwd(ctx: DeviceContext, g: Gemm, c: Cfg, run: Run, p: List[F], a: Act, b: Bat) raises:
    var d = c.d
    var mb = a.mb
    var nz = mb * c.nz
    var lm = c.nsrc * b.lsum
    if not run.own:
        if run.single:
            zero(ctx, a.m, lm * d)
        else:
            copy(ctx, b.x0, a.u, b.lsum * d)
            if c.nsrc >= 2:
                g[False, False](ctx, b.x1, p[WP1], a.u.unsafe_offset(b.lsum * d), b.lsum, d, c.s1)
            if c.nsrc >= 3:
                g[False, False](ctx, b.x2, p[WP2], a.u.unsafe_offset(2 * b.lsum * d), b.lsum, d, c.s2)
            ctx.enqueue_function[rowmap_k](a.u, a.mpre, b.map, b.srcid, p[ESRC], Int32(lm), Int32(d), Int32(0), grid_dim=grid(lm * d), block_dim=THREADS)
            rms(ctx, a.mpre, p[GM], a.m, a.rm, lm, d)
        g[False, False](ctx, a.m, p[WK], a.kk, lm, d, d)
        g[False, False](ctx, a.m, p[WV], a.vv, lm, d, d)
    ctx.enqueue_function[bcast_k](p[Z0], a.zin, Int32(mb), Int32(c.nz * d), grid_dim=grid(nz * d), block_dim=THREADS)
    var t = at_block(c, a)
    for r in range(c.R):
        var zin = a.rnd(a.zin, r)
        var zout = a.rnd(a.zin, r + 1) if r + 1 < c.R else a.zr
        var n1 = a.rnd(a.n1, r)
        var n2 = a.rnd(a.n2, r)
        var za = a.rnd(a.za, r)
        rms(ctx, zin, p[G1], n1, a.r1.unsafe_offset(r * nz), nz, d)
        g[False, False](ctx, n1, p[WQ], a.rnd(a.qz, r), nz, d, d)
        var kk = a.kk
        var vv = a.vv
        if run.own:
            kk = a.rnd(a.kz, r)
            vv = a.rnd(a.vz, r)
            g[False, False](ctx, n1, p[WK], kk, nz, d, d)
            g[False, False](ctx, n1, p[WV], vv, nz, d, d)
        attn_fwd(ctx, t, a.rnd(a.qz, r), kk, vv, a.rnd(a.oz, r), a.pz.unsafe_offset(r * a.pzn), b.tabZ, mb)
        copy(ctx, zin, za, nz * d)
        g[False, False](ctx, a.rnd(a.oz, r), p[WO], za, nz, d, d, True)
        rms(ctx, za, p[G2], n2, a.r2.unsafe_offset(r * nz), nz, d)
        g[False, False](ctx, n2, p[W1], a.rnd(a.aa, r, True), nz, c.ff, d)
        g[False, False](ctx, n2, p[W3], a.rnd(a.bb, r, True), nz, c.ff, d)
        ctx.enqueue_function[swiglu_fwd_k](a.rnd(a.aa, r, True), a.rnd(a.bb, r, True), a.rnd(a.ss, r, True), Int32(nz * c.ff), grid_dim=grid(nz * c.ff), block_dim=THREADS)
        copy(ctx, za, zout, nz * d)
        g[False, False](ctx, a.rnd(a.ss, r, True), p[W2], zout, nz, d, c.ff, True)
    rms(ctx, a.zr, p[GZ], a.zn, a.rz, nz, d)


def block_bwd(ctx: DeviceContext, g: Gemm, c: Cfg, run: Run, p: List[F], gr: List[F], a: Act, b: Bat) raises:
    var d = c.d
    var mb = a.mb
    var nz = mb * c.nz
    var lm = c.nsrc * b.lsum
    var t = at_block(c, a)
    rms_bwd(ctx, a, a.zr, p[GZ], a.rz, a.dzn, a.dz, gr[GZ], nz, d, False)
    for r in reversed(range(c.R)):
        var zin = a.rnd(a.zin, r)
        var n1 = a.rnd(a.n1, r)
        var n2 = a.rnd(a.n2, r)
        var za = a.rnd(a.za, r)
        g[False, True](ctx, a.dz, p[W2], a.dff3, nz, c.ff, d)
        g[True, False](ctx, a.rnd(a.ss, r, True), a.dz, gr[W2], c.ff, d, nz, True)
        ctx.enqueue_function[swiglu_bwd_k](a.rnd(a.aa, r, True), a.rnd(a.bb, r, True), a.dff3, a.dff1, a.dff2, Int32(nz * c.ff), grid_dim=grid(nz * c.ff), block_dim=THREADS)
        g[True, False](ctx, n2, a.dff1, gr[W1], d, c.ff, nz, True)
        g[True, False](ctx, n2, a.dff2, gr[W3], d, c.ff, nz, True)
        g[False, True](ctx, a.dff1, p[W1], a.dn, nz, d, c.ff)
        g[False, True](ctx, a.dff2, p[W3], a.dn, nz, d, c.ff, True)
        copy(ctx, a.dz, a.dza, nz * d)
        rms_bwd(ctx, a, za, p[G2], a.r2.unsafe_offset(r * nz), a.dn, a.dza, gr[G2], nz, d, True)
        g[False, True](ctx, a.dza, p[WO], a.doz, nz, d, d)
        g[True, False](ctx, a.rnd(a.oz, r), a.dza, gr[WO], d, d, nz, True)
        var kk = a.kk
        var vv = a.vv
        var dk = a.dkk
        var dv = a.dvv
        if run.own:
            kk = a.rnd(a.kz, r)
            vv = a.rnd(a.vz, r)
            dk = a.dkz
            dv = a.dvz
            zero(ctx, dk, nz * d)
            zero(ctx, dv, nz * d)
        attn_bwd(ctx, t, a.rnd(a.qz, r), kk, vv, a.rnd(a.oz, r), a.doz, a.pz.unsafe_offset(r * a.pzn), a.dsz, a.dqz, dk, dv, b.tabZ, mb)
        g[True, False](ctx, n1, a.dqz, gr[WQ], d, d, nz, True)
        g[False, True](ctx, a.dqz, p[WQ], a.dn, nz, d, d)
        if run.own:
            g[True, False](ctx, n1, dk, gr[WK], d, d, nz, True)
            g[True, False](ctx, n1, dv, gr[WV], d, d, nz, True)
            g[False, True](ctx, dk, p[WK], a.dn, nz, d, d, True)
            g[False, True](ctx, dv, p[WV], a.dn, nz, d, d, True)
        copy(ctx, a.dza, a.dz, nz * d)
        rms_bwd(ctx, a, zin, p[G1], a.r1.unsafe_offset(r * nz), a.dn, a.dz, gr[G1], nz, d, True)
    ctx.enqueue_function[colsum_k](a.dz, gr[Z0], Int32(mb), Int32(c.nz * d), grid_dim=grid(c.nz * d), block_dim=THREADS)
    if run.own or run.single:
        return
    g[True, False](ctx, a.m, a.dkk, gr[WK], d, d, lm, True)
    g[True, False](ctx, a.m, a.dvv, gr[WV], d, d, lm, True)
    g[False, True](ctx, a.dkk, p[WK], a.dm, lm, d, d)
    g[False, True](ctx, a.dvv, p[WV], a.dm, lm, d, d, True)
    rms_bwd(ctx, a, a.mpre, p[GM], a.rm, a.dm, a.kk, gr[GM], lm, d, False)
    ctx.enqueue_function[rowmap_k](a.kk, a.du, b.map, b.srcid, p[ESRC], Int32(lm), Int32(d), Int32(1), grid_dim=grid(lm * d), block_dim=THREADS)
    for s in range(c.nsrc):
        ctx.enqueue_function[colsum_k](a.du.unsafe_offset(s * b.lsum * d), gr[ESRC].unsafe_offset(s * d), Int32(b.lsum), Int32(d), grid_dim=grid(d), block_dim=THREADS)
    if c.nsrc >= 2:
        g[True, False](ctx, b.x1, a.du.unsafe_offset(b.lsum * d), gr[WP1], c.s1, d, b.lsum, True)
    if c.nsrc >= 3:
        g[True, False](ctx, b.x2, a.du.unsafe_offset(2 * b.lsum * d), gr[WP2], c.s2, d, b.lsum, True)


def reader_a_fwd(ctx: DeviceContext, g: Gemm, c: Cfg, run: Run, p: List[F], a: Act, b: Bat, tab: I) raises:
    var d = c.d
    var mb = a.mb
    var nz = mb * c.nz
    var na = b.na
    rms(ctx, b.hA, p[GAN], a.hn, a.rh, na, d)
    g[False, False](ctx, a.hn, p[WQA], a.qa, na, d, d)
    g[False, False](ctx, a.zn, p[WKA], a.ka, nz, d, d)
    g[False, False](ctx, a.zn, p[WVA], a.va, nz, d, d)
    attn_fwd(ctx, at_a(c), a.qa, a.ka, a.va, a.oa, a.pa, tab, mb)
    g[False, False](ctx, a.oa, p[WOA], a.ua, na, d, d)
    cols(ctx, a.ua, p[GA], a.hp, na, d, 1)
    ctx.enqueue_function[add_k](b.hA, a.hp, Int32(na * d), grid_dim=grid(na * d), block_dim=THREADS)
    g[False, True](ctx, a.hp, b.head, a.logits, na, c.vocab, d)
    ctx.enqueue_function[ce_big_k](a.logits, b.tgt, a.lrow, a.hit, Int32(na), Int32(c.vocab), run.wa * run.wsc / Float32(na), grid_dim=na, block_dim=AT)
    total(ctx, a.lrow, na, a.sc.unsafe_offset(S_CE))
    total(ctx, a.hit, na, a.sc.unsafe_offset(S_HIT))
    diff(ctx, a.hp, b.hT, a.ra, na * d)
    sumsq(ctx, a, b.hT, na * d, a.sc.unsafe_offset(S_HT2))
    sumsq(ctx, a, a.ra, na * d, a.sc.unsafe_offset(S_RA2))


def reader_a_bwd(ctx: DeviceContext, g: Gemm, c: Cfg, run: Run, p: List[F], gr: List[F], a: Act, b: Bat, tab: I) raises:
    var d = c.d
    var mb = a.mb
    var nz = mb * c.nz
    var na = b.na
    g[False, False](ctx, a.logits, b.head, a.dhp, na, d, c.vocab)
    if run.lam > 0:
        saxpy(ctx, a.ra, a.dhp, 2 * run.lam * run.wa * run.wsc, a.sc.unsafe_offset(S_HT2), na * d, True)
    cols(ctx, a.dhp, p[GA], a.dua, na, d, 1)
    coldot(ctx, a, a.dhp, a.ua, a.rh, gr[GA], na, d, False)
    g[False, True](ctx, a.dua, p[WOA], a.doa, na, d, d)
    g[True, False](ctx, a.oa, a.dua, gr[WOA], d, d, na, True)
    zero(ctx, a.dka, nz * d)
    zero(ctx, a.dva, nz * d)
    attn_bwd(ctx, at_a(c), a.qa, a.ka, a.va, a.oa, a.doa, a.pa, a.dsa, a.dqa, a.dka, a.dva, tab, mb)
    g[True, False](ctx, a.hn, a.dqa, gr[WQA], d, d, na, True)
    g[False, True](ctx, a.dqa, p[WQA], a.dhn, na, d, d)
    coldot(ctx, a, b.hA, a.dhn, a.rh, gr[GAN], na, d, True)
    g[True, False](ctx, a.zn, a.dka, gr[WKA], d, d, nz, True)
    g[True, False](ctx, a.zn, a.dva, gr[WVA], d, d, nz, True)
    g[False, True](ctx, a.dka, p[WKA], a.dzn, nz, d, d, True)
    g[False, True](ctx, a.dva, p[WVA], a.dzn, nz, d, d, True)


def reader_b_fwd(ctx: DeviceContext, g: Gemm, c: Cfg, run: Run, p: List[F], a: Act, b: Bat, tab: I) raises:
    var d = c.d
    var mb = a.mb
    var nz = mb * c.nz
    var na = b.na
    var qw = c.qw()
    g[False, False](ctx, a.zn, p[WKB], a.kb, nz, c.kbw(), d)
    g[False, False](ctx, a.zn, p[WVB], a.vb, nz, c.vbw(), d)
    attn_fwd(ctx, at_b(c), b.qB.unsafe_offset(c.qlo), a.kb, a.vb, a.delta, a.pb, tab, mb)
    cols(ctx, a.delta, p[GB], a.dg, na, qw, c.hdb)
    g[False, True](ctx, a.dg, b.wo47, a.yb, na, d, qw)
    copy(ctx, b.aoS, a.rb, na * d)
    ctx.enqueue_function[add_k](a.yb, a.rb, Int32(na * d), grid_dim=grid(na * d), block_dim=THREADS)
    diff(ctx, a.rb, b.aoT, a.rb, na * d)
    sumsq(ctx, a, b.aoT, na * d, a.sc.unsafe_offset(S_AOT2))
    sumsq(ctx, a, a.rb, na * d, a.sc.unsafe_offset(S_RB2))


def reader_b_bwd(ctx: DeviceContext, g: Gemm, c: Cfg, run: Run, p: List[F], gr: List[F], a: Act, b: Bat, tab: I) raises:
    var d = c.d
    var mb = a.mb
    var nz = mb * c.nz
    var na = b.na
    var qw = c.qw()
    saxpy(ctx, a.rb, a.yb, 2 * run.wb * run.wsc, a.sc.unsafe_offset(S_AOT2), na * d, False)
    g[False, False](ctx, a.yb, b.wo47, a.ddg, na, qw, d)
    cols(ctx, a.ddg, p[GB], a.ddelta, na, qw, c.hdb)
    zero(ctx, a.tmp, qw)
    coldot(ctx, a, a.ddg, a.delta, a.rh, a.tmp, na, qw, False)
    ctx.enqueue_function[groupsum_k](a.tmp, gr[GB], Int32(c.nhb), Int32(c.hdb), grid_dim=grid(c.nhb), block_dim=THREADS)
    zero(ctx, a.dkb, nz * c.kbw())
    zero(ctx, a.dvb, nz * c.vbw())
    attn_bwd(ctx, at_b(c), b.qB.unsafe_offset(c.qlo), a.kb, a.vb, a.delta, a.ddelta, a.pb, a.dsb, a.dqb.unsafe_offset(c.qlo), a.dkb, a.dvb, tab, mb)
    g[True, False](ctx, a.zn, a.dkb, gr[WKB], d, c.kbw(), nz, True)
    g[True, False](ctx, a.zn, a.dvb, gr[WVB], d, c.vbw(), nz, True)
    g[False, True](ctx, a.dkb, p[WKB], a.dzn, nz, d, c.kbw(), True)
    g[False, True](ctx, a.dvb, p[WVB], a.dzn, nz, d, c.vbw(), True)


def reader_c_fwd(ctx: DeviceContext, g: Gemm, c: Cfg, run: Run, p: List[F], a: Act, b: Bat, ct: F) raises:
    var d = c.d
    var nz = a.mb * c.nz
    g[False, False](ctx, a.zn, p[WC], a.cc, nz, d, d)
    diff(ctx, a.cc, ct, a.rc, nz * d)
    sumsq(ctx, a, ct, nz * d, a.sc.unsafe_offset(S_CT2))
    sumsq(ctx, a, a.rc, nz * d, a.sc.unsafe_offset(S_RC2))
    ctx.enqueue_function[cos_k](a.cc, ct, a.dcc, a.lc, Int32(nz), Int32(d), run.wc * run.wsc / Float32(nz), grid_dim=ceildiv(nz, NW), block_dim=AT)
    total(ctx, a.lc, nz, a.sc.unsafe_offset(S_COS))


def reader_c_bwd(ctx: DeviceContext, g: Gemm, c: Cfg, run: Run, p: List[F], gr: List[F], a: Act, b: Bat) raises:
    var d = c.d
    var nz = a.mb * c.nz
    saxpy(ctx, a.rc, a.dcc, 2 * run.wc * run.wsc, a.sc.unsafe_offset(S_CT2), nz * d, True)
    g[True, False](ctx, a.zn, a.dcc, gr[WC], d, d, nz, True)
    g[False, True](ctx, a.dcc, p[WC], a.dzn, nz, d, d, True)


def forward(ctx: DeviceContext, g: Gemm, c: Cfg, run: Run, p: List[F], a: Act, b: Bat, tab: I, ct: F) raises:
    """Block then every active reader; tab/ct select matched (b.tabA, b.ct) or cross-item thoughts."""
    block_fwd(ctx, g, c, run, p, a, b)
    if run.ra:
        reader_a_fwd(ctx, g, c, run, p, a, b, tab)
    if run.rb:
        reader_b_fwd(ctx, g, c, run, p, a, b, tab)
    if run.rc:
        reader_c_fwd(ctx, g, c, run, p, a, b, ct)


def backward(ctx: DeviceContext, g: Gemm, c: Cfg, run: Run, p: List[F], gr: List[F], a: Act, b: Bat) raises:
    var d = c.d
    zero(ctx, a.dzn, a.mb * c.nz * d)
    if not run.own:
        zero(ctx, a.dkk, c.nsrc * b.lsum * d)
        zero(ctx, a.dvv, c.nsrc * b.lsum * d)
    if run.ra:
        reader_a_bwd(ctx, g, c, run, p, gr, a, b, b.tabA)
    if run.rb:
        reader_b_bwd(ctx, g, c, run, p, gr, a, b, b.tabA)
    if run.rc:
        reader_c_bwd(ctx, g, c, run, p, gr, a, b)
    block_bwd(ctx, g, c, run, p, gr, a, b)
