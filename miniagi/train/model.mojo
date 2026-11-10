from max.gpu.host import DeviceContext, DeviceBuffer
from ops import *
from fast import Wmma

comptime V = 256
comptime D = 256
comptime H = 4
comptime HD = D // H
comptime FF = 688
comptime L = 2
comptime DD = D * D
comptime DF = D * FF
comptime LAYER = 2 * D + 4 * DD + 3 * DF
comptime NP = V * D + L * LAYER + D
comptime EPS: Float32 = 1e-6


@fieldwise_init
struct W(ImplicitlyCopyable):
    var g1: F
    var wq: F
    var wk: F
    var wv: F
    var wo: F
    var g2: F
    var w1: F
    var w3: F
    var w2: F


def layer(p: F, l: Int) -> W:
    var b = p.unsafe_offset(V * D + l * LAYER)
    return W(b, b.unsafe_offset(D), b.unsafe_offset(D + DD), b.unsafe_offset(D + 2 * DD), b.unsafe_offset(D + 3 * DD),
             b.unsafe_offset(D + 4 * DD), b.unsafe_offset(2 * D + 4 * DD), b.unsafe_offset(2 * D + 4 * DD + DF),
             b.unsafe_offset(2 * D + 4 * DD + 2 * DF))


struct Arena(Movable):
    var buf: DeviceBuffer[DType.float32]
    var top: Int

    def __init__(out self, ctx: DeviceContext, n: Int) raises:
        self.buf = ctx.enqueue_create_buffer[DType.float32](n)
        self.top = 0

    def take(mut self, n: Int) -> F:
        var p = F(unsafe_from_address=Int(self.buf.unsafe_ptr())).unsafe_offset(self.top)
        self.top += n
        return p


@fieldwise_init
struct Acts(ImplicitlyCopyable):
    var x: F
    var n1: F
    var r1: F
    var qt: F
    var kt: F
    var vt: F
    var qh: F
    var kh: F
    var vh: F
    var p: F
    var oh: F
    var ot: F
    var h: F
    var n2: F
    var r2: F
    var a: F
    var b: F
    var s: F


struct Model[fast: Bool = False](Movable):
    var t: Int
    var w: Wmma
    var mem: Arena
    var acts: List[Acts]
    var xl: F
    var nf: F
    var rf: F
    var z: F
    var rows: F
    var dz: F
    var dx: F
    var dn: F
    var dt: F
    var d3: F
    var dh: F
    var ds: F
    var da: F
    var db: F
    var dff: F
    var rowsb: DeviceBuffer[DType.float32]
    var cs: DeviceBuffer[DType.float32]
    var sn: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, t: Int) raises:
        self.t = t
        self.w = Wmma(ctx, t * FF if Self.fast else 1)
        var per = 13 * t * D + 2 * t + H * t * t + 3 * t * FF
        self.mem = Arena(ctx, (L + 1) * t * D + L * per + 2 * t + t * V * 2 + 9 * t * D + 4 * t * D + H * t * t + 3 * t * FF)
        self.acts = List[Acts]()
        for _ in range(L):
            var m = self.mem.take(t * D)
            self.acts.append(Acts(m, self.mem.take(t * D), self.mem.take(t), self.mem.take(t * D), self.mem.take(t * D),
                self.mem.take(t * D), self.mem.take(t * D), self.mem.take(t * D), self.mem.take(t * D), self.mem.take(H * t * t),
                self.mem.take(t * D), self.mem.take(t * D), self.mem.take(t * D), self.mem.take(t * D), self.mem.take(t),
                self.mem.take(t * FF), self.mem.take(t * FF), self.mem.take(t * FF)))
        self.xl = self.mem.take(t * D)
        self.nf = self.mem.take(t * D)
        self.rf = self.mem.take(t)
        self.z = self.mem.take(t * V)
        self.rowsb = ctx.enqueue_create_buffer[DType.float32](t)
        self.rows = F(unsafe_from_address=Int(self.rowsb.unsafe_ptr()))
        self.dz = self.mem.take(t * V)
        self.dx = self.mem.take(t * D)
        self.dn = self.mem.take(t * D)
        self.dt = self.mem.take(t * D)
        self.d3 = self.mem.take(3 * t * D)
        self.dh = self.mem.take(4 * t * D)
        self.ds = self.mem.take(H * t * t)
        self.da = self.mem.take(t * FF)
        self.db = self.mem.take(t * FF)
        self.dff = self.mem.take(t * FF)
        var tab = rope_tables(t, HD, 10000)
        self.cs = ctx.enqueue_create_buffer[DType.float32](len(tab[0]))
        self.sn = ctx.enqueue_create_buffer[DType.float32](len(tab[1]))
        upload(ctx, self.cs, tab[0])
        upload(ctx, self.sn, tab[1])

    def mm[ta: Bool, tb: Bool](self, ctx: DeviceContext, a: F, b: F, c: F, m: Int, n: Int, k: Int) raises:
        comptime if Self.fast:
            self.w.gemm[ta, tb](ctx, a, b, c, m, n, k)
        else:
            gemm[ta, tb](ctx, a, b, c, m, n, k)

    def rope[inv: Bool](self, ctx: DeviceContext, x: F) raises:
        var c = F(unsafe_from_address=Int(self.cs.unsafe_ptr()))
        var s = F(unsafe_from_address=Int(self.sn.unsafe_ptr()))
        ctx.enqueue_function[rope_k[inv]](x, x, c, s, Int32(self.t), Int32(H), Int32(HD), Int32(0), grid_dim=grid(self.t * D // 2), block_dim=THREADS)

    def perm[inv: Bool](self, ctx: DeviceContext, x: F, y: F) raises:
        ctx.enqueue_function[perm_k[inv]](x, y, Int32(self.t), Int32(H), Int32(HD), grid_dim=grid(self.t * D), block_dim=THREADS)

    def rms(self, ctx: DeviceContext, x: F, g: F, y: F, r: F) raises:
        ctx.enqueue_function[rmsnorm_fwd_k](x, g, y, r, Int32(self.t), Int32(D), EPS, grid_dim=grid(self.t), block_dim=THREADS)

    def rms_bwd(self, ctx: DeviceContext, x: F, g: F, r: F, dy: F, dx: F, dg: F) raises:
        ctx.enqueue_function[rmsnorm_dx_k](x, g, r, dy, dx, Int32(self.t), Int32(D), grid_dim=grid(self.t), block_dim=THREADS)
        ctx.enqueue_function[rmsnorm_dw_k](x, r, dy, dg, Int32(self.t), Int32(D), Int32(1), grid_dim=grid(D), block_dim=THREADS)

    def add(self, ctx: DeviceContext, x: F, y: F, n: Int) raises:
        ctx.enqueue_function[add_k](x, y, Int32(n), grid_dim=grid(n), block_dim=THREADS)

    def forward(self, ctx: DeviceContext, p: F, tok: I, tgt: I) raises -> Float32:
        var t = self.t
        ctx.enqueue_function[embed_fwd_k](p, tok, self.acts[0].x, Int32(t), Int32(D), grid_dim=grid(t * D), block_dim=THREADS)
        for l in range(L):
            var a = self.acts[l]
            var w = layer(p, l)
            var out = self.acts[l + 1].x if l + 1 < L else self.xl
            self.rms(ctx, a.x, w.g1, a.n1, a.r1)
            self.mm[False, False](ctx, a.n1, w.wq, a.qt, t, D, D)
            self.mm[False, False](ctx, a.n1, w.wk, a.kt, t, D, D)
            self.mm[False, False](ctx, a.n1, w.wv, a.vt, t, D, D)
            self.rope[False](ctx, a.qt)
            self.rope[False](ctx, a.kt)
            self.perm[False](ctx, a.qt, a.qh)
            self.perm[False](ctx, a.kt, a.kh)
            self.perm[False](ctx, a.vt, a.vh)
            attn_fwd(ctx, a.qh, a.kh, a.vh, a.p, a.oh, H, t, HD)
            self.perm[True](ctx, a.oh, a.ot)
            self.mm[False, False](ctx, a.ot, w.wo, a.h, t, D, D)
            self.add(ctx, a.x, a.h, t * D)
            self.rms(ctx, a.h, w.g2, a.n2, a.r2)
            self.mm[False, False](ctx, a.n2, w.w1, a.a, t, FF, D)
            self.mm[False, False](ctx, a.n2, w.w3, a.b, t, FF, D)
            ctx.enqueue_function[swiglu_fwd_k](a.a, a.b, a.s, Int32(t * FF), grid_dim=grid(t * FF), block_dim=THREADS)
            self.mm[False, False](ctx, a.s, w.w2, out, t, D, FF)
            self.add(ctx, a.h, out, t * D)
        self.rms(ctx, self.xl, p.unsafe_offset(NP - D), self.nf, self.rf)
        self.mm[False, True](ctx, self.nf, p, self.z, t, V, D)
        ctx.enqueue_function[ce_k](self.z, tgt, self.rows, self.dz, Int32(t), Int32(V), grid_dim=grid(t), block_dim=THREADS)
        var rows = List[Float32](length=t, fill=0)
        ctx.enqueue_copy(dst_ptr=rows.unsafe_ptr(), src_buf=self.rowsb)
        ctx.synchronize()
        var loss: Float32 = 0
        for r in rows:
            loss += r
        return loss / Float32(t)

    def backward(self, ctx: DeviceContext, p: F, g: F, tok: I) raises:
        var t = self.t
        gemm[False, False](ctx, self.dz, p, self.dn, t, D, V)
        gemm[True, False](ctx, self.dz, self.nf, g, V, D, t, True)
        self.rms_bwd(ctx, self.xl, p.unsafe_offset(NP - D), self.rf, self.dn, self.dx, g.unsafe_offset(NP - D))
        for l in reversed(range(L)):
            var a = self.acts[l]
            var w = layer(p, l)
            var gw = layer(g, l)
            gemm[False, True](ctx, self.dx, w.w2, self.dff, t, FF, D)
            gemm[True, False](ctx, a.s, self.dx, gw.w2, FF, D, t, True)
            ctx.enqueue_function[swiglu_bwd_k](a.a, a.b, self.dff, self.da, self.db, Int32(t * FF), grid_dim=grid(t * FF), block_dim=THREADS)
            gemm[True, False](ctx, a.n2, self.da, gw.w1, D, FF, t, True)
            gemm[True, False](ctx, a.n2, self.db, gw.w3, D, FF, t, True)
            gemm[False, True](ctx, self.da, w.w1, self.dn, t, D, FF)
            gemm[False, True](ctx, self.db, w.w3, self.dn, t, D, FF, True)
            self.rms_bwd(ctx, a.h, w.g2, a.r2, self.dn, self.dt, gw.g2)
            self.add(ctx, self.dt, self.dx, t * D)
            gemm[False, True](ctx, self.dx, w.wo, self.dt, t, D, D)
            gemm[True, False](ctx, a.ot, self.dx, gw.wo, D, D, t, True)
            var doh = self.dh
            var dq = self.dh.unsafe_offset(t * D)
            var dk = self.dh.unsafe_offset(2 * t * D)
            var dv = self.dh.unsafe_offset(3 * t * D)
            self.perm[False](ctx, self.dt, doh)
            attn_bwd(ctx, a.qh, a.kh, a.vh, a.p, doh, self.ds, dq, dk, dv, H, t, HD)
            var qt = self.d3
            var kt = self.d3.unsafe_offset(t * D)
            var vt = self.d3.unsafe_offset(2 * t * D)
            self.perm[True](ctx, dq, qt)
            self.perm[True](ctx, dk, kt)
            self.perm[True](ctx, dv, vt)
            self.rope[True](ctx, qt)
            self.rope[True](ctx, kt)
            gemm[True, False](ctx, a.n1, qt, gw.wq, D, D, t, True)
            gemm[True, False](ctx, a.n1, kt, gw.wk, D, D, t, True)
            gemm[True, False](ctx, a.n1, vt, gw.wv, D, D, t, True)
            gemm[False, True](ctx, qt, w.wq, self.dn, t, D, D)
            gemm[False, True](ctx, kt, w.wk, self.dn, t, D, D, True)
            gemm[False, True](ctx, vt, w.wv, self.dn, t, D, D, True)
            self.rms_bwd(ctx, a.x, w.g1, a.r1, self.dn, self.dt, gw.g1)
            self.add(ctx, self.dt, self.dx, t * D)
        ctx.enqueue_function[embed_bwd_k](tok, self.dx, g, Int32(t), Int32(D), Int32(V), Int32(1), grid_dim=grid(V * D), block_dim=THREADS)
