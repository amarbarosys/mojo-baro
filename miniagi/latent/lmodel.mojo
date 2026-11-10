from std.math import ceildiv
from std.sys import get_defined_int
from max.gpu.host import DeviceContext, DeviceBuffer
from ops import F, I, THREADS, grid, upload, gemm, rope_tables, swiglu_fwd_k, swiglu_bwd_k, add_k
from lops import *
from layout import TileTensor, row_major
from linalg.matmul import matmul as lmatmul
from nlook import Look, EB

comptime V = 32
comptime D = DM
comptime H = get_defined_int["LAT_H", 4]()
comptime HD = D // H
comptime FF = D * 43 // 16
comptime L = 2
comptime NTH = 7
comptime DD = D * D
comptime DF = D * FF
comptime LAYER = 2 * D + 4 * DD + 3 * DF
comptime GF = V * D + L * LAYER
comptime PAUSE = GF + D
comptime NP = PAUSE + NTH * D
comptime EPS: Float32 = 1e-6
comptime TOK = 0
comptime LAT = 1
comptime PSE = 2
comptime GEN = 3
comptime SOFT = 4
comptime ENT = 5
comptime NEU = 6


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


@fieldwise_init
struct Acts(ImplicitlyCopyable):
    var x: F
    var n1: F
    var r1: F
    var q: F
    var k: F
    var v: F
    var p: F
    var o: F
    var h: F
    var n2: F
    var r2: F
    var a: F
    var b: F
    var s: F
    var dk: F
    var dv: F


struct Model[exact: Bool = False](Movable):
    """Position-major batch: row r*b + i is position r of sequence i. Segments are row ranges [r0, r1)."""
    var b: Int
    var sg: Bool
    var itemp: Float32
    var mem: DeviceBuffer[DType.float32]
    var top: Int
    var g: Gemm
    var acts: List[Acts]
    var xl: F
    var nf: F
    var rf: F
    var z: F
    var dz: F
    var lrow: F
    var dnf: F
    var dx: F
    var dn: F
    var dt: F
    var dq: F
    var da: F
    var db: F
    var dff: F
    var part: F
    var w16: DeviceBuffer[DType.bfloat16]
    var t16: DeviceBuffer[DType.bfloat16]
    var a16: DeviceBuffer[DType.bfloat16]
    var ctmp: F
    var sp: F
    var cs: DeviceBuffer[DType.float32]
    var sn: DeviceBuffer[DType.float32]
    var lk: Look

    def __init__(out self, ctx: DeviceContext, b: Int) raises:
        comptime assert D % 256 == 0 and D % H == 0 and D // H <= 64 and V == 32
        self.b = b
        self.lk = Look(ctx, b, 0, 0, 0, 0, 0)
        self.sg = False
        self.itemp = 1
        var R = SMAX * b
        var per = 12 * R * D + 2 * R + 3 * R * FF + b * H * SMAX * SMAX
        var total = L * per + (L + 1) * R * D + R + 2 * R * V + R + 7 * R * D + 3 * R * FF
        self.mem = ctx.enqueue_create_buffer[DType.float32](total)
        ctx.enqueue_memset(self.mem, 0)
        self.top = 0
        self.g = Gemm(ctx, 8 * 1024 * 1024)
        self.acts = List[Acts]()
        var base = F(unsafe_from_address=Int(self.mem.unsafe_ptr()))
        self.xl = base
        self.nf = base
        self.rf = base
        self.z = base
        self.dz = base
        self.lrow = base
        self.dnf = base
        self.dx = base
        self.dn = base
        self.dt = base
        self.dq = base
        self.da = base
        self.db = base
        self.dff = base
        self.part = base
        self.ctmp = base
        self.w16 = ctx.enqueue_create_buffer[DType.bfloat16](NP)
        self.t16 = ctx.enqueue_create_buffer[DType.bfloat16](NP)
        self.a16 = ctx.enqueue_create_buffer[DType.bfloat16](R * FF)
        self.sp = base
        var tab = rope_tables(SMAX, HD, 10000)
        self.cs = ctx.enqueue_create_buffer[DType.float32](len(tab[0]))
        self.sn = ctx.enqueue_create_buffer[DType.float32](len(tab[1]))
        upload(ctx, self.cs, tab[0])
        upload(ctx, self.sn, tab[1])
        for _ in range(L):
            var x = self.take(R * D)
            self.acts.append(Acts(x, self.take(R * D), self.take(R), self.take(R * D), self.take(R * D), self.take(R * D),
                self.take(b * H * SMAX * SMAX), self.take(R * D), self.take(R * D), self.take(R * D), self.take(R),
                self.take(R * FF), self.take(R * FF), self.take(R * FF), self.take(R * D), self.take(R * D)))
        self.xl = self.take(R * D)
        self.nf = self.take(R * D)
        self.rf = self.take(R)
        self.z = self.take(R * V)
        self.dz = self.take(R * V)
        self.lrow = self.take(R)
        self.dnf = self.take(R * D)
        self.dx = self.take(R * D)
        self.dn = self.take(R * D)
        self.dt = self.take(R * D)
        self.dq = self.take(R * D)
        self.da = self.take(R * FF)
        self.db = self.take(R * FF)
        self.dff = self.take(R * FF)
        self.part = self.take(max(ceildiv(R, RC) * D, ceildiv(R, EC) * V * D))
        self.sp = self.take(b * V)
        self.ctmp = self.take(R * D)

    def take(mut self, n: Int) -> F:
        var p = F(unsafe_from_address=Int(self.mem.unsafe_ptr())).unsafe_offset(self.top)
        self.top += n
        return p

    def mm[ta: Bool, tb: Bool](self, ctx: DeviceContext, p: F, a: F, b: F, c: F, m: Int, n: Int, k: Int, acc: Bool = False) raises:
        """b is a parameter matrix when not ta: its bf16 copy (tb) or bf16 transpose (not tb) feeds linalg."""
        comptime if Self.exact:
            gemm[ta, tb](ctx, a, b, c, m, n, k, acc)
        elif ta:
            self.g[ta, tb](ctx, a, b, c, m, n, k, acc)
        else:
            var off = (Int(b) - Int(p)) // 4
            var a16p = H16(unsafe_from_address=Int(self.a16.unsafe_ptr()))
            ctx.enqueue_function[cast16_k](a, a16p, Int32(m * k), grid_dim=grid(m * k), block_dim=THREADS)
            var bsrc = self.w16 if tb else self.t16
            var b16p = H16(unsafe_from_address=Int(bsrc.unsafe_ptr())).unsafe_offset(off)
            var out = self.ctmp if acc else c
            var cbuf = DeviceBuffer[DType.float32](ctx, out, m * n, owning=False)
            var abuf = DeviceBuffer[DType.bfloat16](ctx, a16p, m * k, owning=False)
            var bbuf = DeviceBuffer[DType.bfloat16](ctx, b16p, n * k, owning=False)
            var tc = TileTensor(cbuf, row_major(m, n))
            var tA = TileTensor(abuf, row_major(m, k))
            var tB = TileTensor(bbuf, row_major(n, k))
            lmatmul[transpose_b=True, target="gpu"](tc, tA, tB, ctx)
            if acc:
                self.add(ctx, self.ctmp, c, m * n)

    def refresh(self, ctx: DeviceContext, p: F) raises:
        """bf16 copy of all params and bf16 transposes of every matrix, from the current p."""
        comptime if not Self.exact:
            var w = H16(unsafe_from_address=Int(self.w16.unsafe_ptr()))
            var t = H16(unsafe_from_address=Int(self.t16.unsafe_ptr()))
            ctx.enqueue_function[cast16_k](p, w, Int32(NP), grid_dim=grid(NP), block_dim=THREADS)
            ctx.enqueue_function[tr16_k](p, t, Int32(V), Int32(D), grid_dim=grid(V * D), block_dim=THREADS)
            for l in range(L):
                var o = V * D + l * LAYER
                for j in range(4):
                    var q = o + D + j * DD
                    ctx.enqueue_function[tr16_k](p.unsafe_offset(q), t.unsafe_offset(q), Int32(D), Int32(D), grid_dim=grid(DD), block_dim=THREADS)
                for j in range(2):
                    var q = o + 2 * D + 4 * DD + j * DF
                    ctx.enqueue_function[tr16_k](p.unsafe_offset(q), t.unsafe_offset(q), Int32(D), Int32(FF), grid_dim=grid(DF), block_dim=THREADS)
                var q = o + 2 * D + 4 * DD + 2 * DF
                ctx.enqueue_function[tr16_k](p.unsafe_offset(q), t.unsafe_offset(q), Int32(FF), Int32(D), grid_dim=grid(DF), block_dim=THREADS)

    def rms(self, ctx: DeviceContext, x: F, g: F, y: F, r: F, n: Int) raises:
        ctx.enqueue_function[rms_fwd_k](x, g, y, r, Int32(n), EPS, grid_dim=ceildiv(n, NW), block_dim=AT)

    def rms_bwd(self, ctx: DeviceContext, x: F, g: F, r: F, dy: F, dx: F, dg: F, n: Int, acc: Bool) raises:
        ctx.enqueue_function[rms_dx_k](x, g, r, dy, dx, Int32(n), Int32(acc), grid_dim=ceildiv(n, NW), block_dim=AT)
        ctx.enqueue_function[rms_dwp_k](x, r, dy, self.part, Int32(n), grid_dim=ceildiv(n, RC), block_dim=D)
        ctx.enqueue_function[colsum_k](self.part, dg, Int32(ceildiv(n, RC)), Int32(D), grid_dim=grid(D), block_dim=THREADS)

    def add(self, ctx: DeviceContext, x: F, y: F, n: Int) raises:
        ctx.enqueue_function[add_k](x, y, Int32(n), grid_dim=grid(n), block_dim=THREADS)

    def copy(self, ctx: DeviceContext, src: F, dst: F, n: Int, acc: Bool) raises:
        ctx.enqueue_function[copy_k](src, dst, Int32(n), Int32(acc), grid_dim=grid(n), block_dim=THREADS)

    def rope[inv: Bool](self, ctx: DeviceContext, x: F, n: Int, r0: Int) raises:
        var c = F(unsafe_from_address=Int(self.cs.unsafe_ptr()))
        var s = F(unsafe_from_address=Int(self.sn.unsafe_ptr()))
        ctx.enqueue_function[rope_k[inv]](x, c, s, Int32(n), Int32(self.b), Int32(r0), Int32(H), Int32(HD), grid_dim=grid(n * D // 2), block_dim=THREADS)

    def embed(self, ctx: DeviceContext, p: F, ids: I, kinds: List[Int], S: Int) raises:
        """Token rows from the table; pause rows from the pause block; latent rows are filled by forward."""
        var b = self.b
        var x = self.acts[0].x
        ctx.enqueue_function[embed_k](p, ids, x, Int32(S * b), Int32(D), grid_dim=grid(S * b * D), block_dim=THREADS)
        var j = 0
        for r in range(S):
            if kinds[r] == PSE:
                ctx.enqueue_function[bcast_k](p.unsafe_offset(PAUSE + j * D), x.unsafe_offset(r * b * D), Int32(b), Int32(D), grid_dim=grid(b * D), block_dim=THREADS)
                j += 1

    def fwd_seg(self, ctx: DeviceContext, p: F, tgt: I, wr: F, r0: Int, r1: Int, head: Bool) raises:
        var b = self.b
        var n = (r1 - r0) * b
        var o = r0 * b * D
        for l in range(L):
            var a = self.acts[l]
            var w = layer(p, l)
            var out = self.acts[l + 1].x if l + 1 < L else self.xl
            self.rms(ctx, a.x.unsafe_offset(o), w.g1, a.n1.unsafe_offset(o), a.r1.unsafe_offset(r0 * b), n)
            self.mm[False, False](ctx, p, a.n1.unsafe_offset(o), w.wq, a.q.unsafe_offset(o), n, D, D)
            self.mm[False, False](ctx, p, a.n1.unsafe_offset(o), w.wk, a.k.unsafe_offset(o), n, D, D)
            self.mm[False, False](ctx, p, a.n1.unsafe_offset(o), w.wv, a.v.unsafe_offset(o), n, D, D)
            self.rope[False](ctx, a.q.unsafe_offset(o), n, r0)
            self.rope[False](ctx, a.k.unsafe_offset(o), n, r0)
            ctx.enqueue_function[attn_fwd_k](a.q, a.k, a.v, a.p, a.o, Int32(b), Int32(H), Int32(HD), Int32(r0), Int32(r1),
                Float32(HD) ** -0.5, grid_dim=b * H, block_dim=AT)
            self.mm[False, False](ctx, p, a.o.unsafe_offset(o), w.wo, a.h.unsafe_offset(o), n, D, D)
            self.add(ctx, a.x.unsafe_offset(o), a.h.unsafe_offset(o), n * D)
            self.rms(ctx, a.h.unsafe_offset(o), w.g2, a.n2.unsafe_offset(o), a.r2.unsafe_offset(r0 * b), n)
            self.mm[False, False](ctx, p, a.n2.unsafe_offset(o), w.w1, a.a.unsafe_offset(r0 * b * FF), n, FF, D)
            self.mm[False, False](ctx, p, a.n2.unsafe_offset(o), w.w3, a.b.unsafe_offset(r0 * b * FF), n, FF, D)
            ctx.enqueue_function[swiglu_fwd_k](a.a.unsafe_offset(r0 * b * FF), a.b.unsafe_offset(r0 * b * FF), a.s.unsafe_offset(r0 * b * FF),
                Int32(n * FF), grid_dim=grid(n * FF), block_dim=THREADS)
            self.mm[False, False](ctx, p, a.s.unsafe_offset(r0 * b * FF), w.w2, out.unsafe_offset(o), n, D, FF)
            self.add(ctx, a.h.unsafe_offset(o), out.unsafe_offset(o), n * D)
        self.rms(ctx, self.xl.unsafe_offset(o), p.unsafe_offset(GF), self.nf.unsafe_offset(o), self.rf.unsafe_offset(r0 * b), n)
        if head:
            self.mm[False, True](ctx, p, self.nf.unsafe_offset(o), p, self.z.unsafe_offset(r0 * b * V), n, V, D)
            ctx.enqueue_function[ce_k](self.z.unsafe_offset(r0 * b * V), tgt.unsafe_offset(r0 * b), wr.unsafe_offset(r0 * b), self.lrow.unsafe_offset(r0 * b),
                self.dz.unsafe_offset(r0 * b * V), Int32(n), Int32(V), grid_dim=grid(n), block_dim=THREADS)

    def bwd_seg(self, ctx: DeviceContext, p: F, g: F, r0: Int, r1: Int) raises:
        var b = self.b
        var n = (r1 - r0) * b
        var o = r0 * b * D
        var of = r0 * b * FF
        self.mm[False, False](ctx, p, self.dz.unsafe_offset(r0 * b * V), p, self.dnf.unsafe_offset(o), n, D, V, True)
        self.mm[True, False](ctx, p, self.dz.unsafe_offset(r0 * b * V), self.nf.unsafe_offset(o), g, V, D, n, True)
        self.rms_bwd(ctx, self.xl.unsafe_offset(o), p.unsafe_offset(GF), self.rf.unsafe_offset(r0 * b), self.dnf.unsafe_offset(o),
            self.dx.unsafe_offset(o), g.unsafe_offset(GF), n, False)
        var dx = self.dx.unsafe_offset(o)
        var dn = self.dn.unsafe_offset(o)
        var dt = self.dt.unsafe_offset(o)
        var dq = self.dq.unsafe_offset(o)
        for l in reversed(range(L)):
            var a = self.acts[l]
            var w = layer(p, l)
            var gw = layer(g, l)
            self.mm[False, True](ctx, p, dx, w.w2, self.dff.unsafe_offset(of), n, FF, D)
            self.mm[True, False](ctx, p, a.s.unsafe_offset(of), dx, gw.w2, FF, D, n, True)
            ctx.enqueue_function[swiglu_bwd_k](a.a.unsafe_offset(of), a.b.unsafe_offset(of), self.dff.unsafe_offset(of), self.da.unsafe_offset(of),
                self.db.unsafe_offset(of), Int32(n * FF), grid_dim=grid(n * FF), block_dim=THREADS)
            self.mm[True, False](ctx, p, a.n2.unsafe_offset(o), self.da.unsafe_offset(of), gw.w1, D, FF, n, True)
            self.mm[True, False](ctx, p, a.n2.unsafe_offset(o), self.db.unsafe_offset(of), gw.w3, D, FF, n, True)
            self.mm[False, True](ctx, p, self.da.unsafe_offset(of), w.w1, dn, n, D, FF)
            self.mm[False, True](ctx, p, self.db.unsafe_offset(of), w.w3, dn, n, D, FF, True)
            self.rms_bwd(ctx, a.h.unsafe_offset(o), w.g2, a.r2.unsafe_offset(r0 * b), dn, dx, gw.g2, n, True)
            self.mm[False, True](ctx, p, dx, w.wo, dt, n, D, D)
            self.mm[True, False](ctx, p, a.o.unsafe_offset(o), dx, gw.wo, D, D, n, True)
            ctx.enqueue_function[attn_bwd_k](a.q, a.k, a.v, a.p, self.dt, self.dq, a.dk, a.dv, Int32(b), Int32(H), Int32(HD), Int32(r0), Int32(r1),
                Float32(HD) ** -0.5, grid_dim=b * H, block_dim=AT)
            self.rope[True](ctx, dq, n, r0)
            self.rope[True](ctx, a.dk.unsafe_offset(o), n, r0)
            self.mm[True, False](ctx, p, a.n1.unsafe_offset(o), dq, gw.wq, D, D, n, True)
            self.mm[True, False](ctx, p, a.n1.unsafe_offset(o), a.dk.unsafe_offset(o), gw.wk, D, D, n, True)
            self.mm[True, False](ctx, p, a.n1.unsafe_offset(o), a.dv.unsafe_offset(o), gw.wv, D, D, n, True)
            self.mm[False, True](ctx, p, dq, w.wq, dn, n, D, D)
            self.mm[False, True](ctx, p, a.dk.unsafe_offset(o), w.wk, dn, n, D, D, True)
            self.mm[False, True](ctx, p, a.dv.unsafe_offset(o), w.wv, dn, n, D, D, True)
            self.rms_bwd(ctx, a.x.unsafe_offset(o), w.g1, a.r1.unsafe_offset(r0 * b), dn, dx, gw.g1, n, True)

    def step(self, ctx: DeviceContext, p: F, g: F, ids: I, tgt: I, wr: F, kinds: List[Int], S: Int, train: Bool) raises:
        """Forward (and backward when train) of one batch; S rows per sequence, kinds[r] in TOK/LAT/PSE/GEN; row loss weight wr."""
        var b = self.b
        var segs = List[Int]()
        segs.append(0)
        for r in range(1, S):
            if kinds[r] == LAT or kinds[r] == GEN or kinds[r] == SOFT or kinds[r] == NEU:
                segs.append(r)
        segs.append(S)
        self.refresh(ctx, p)
        self.embed(ctx, p, ids, kinds, S)
        var hop = List[Int](length=S, fill=-1)
        var nneu = 0
        for r in range(S):
            if kinds[r] == NEU:
                hop[r] = nneu
                nneu += 1
        if kinds[0] == ENT:
            self.lk.ent_fwd(ctx, p, self.acts[0].x)
        for i in range(len(segs) - 1):
            var r0 = segs[i]
            if kinds[r0] == NEU:
                self.lk.fwd[Self.exact](ctx, p, self.nf.unsafe_offset((r0 - 1) * b * D), self.acts[0].x.unsafe_offset(r0 * b * D), hop[r0])
            if kinds[r0] == LAT:
                self.copy(ctx, self.nf.unsafe_offset((r0 - 1) * b * D), self.acts[0].x.unsafe_offset(r0 * b * D), b * D, False)
            if kinds[r0] == SOFT:
                ctx.enqueue_function[soft_k](self.z.unsafe_offset((r0 - 1) * b * V), p, self.acts[0].x.unsafe_offset(r0 * b * D), Int32(b), Int32(V), Int32(D),
                    self.itemp, grid_dim=grid(b * D), block_dim=THREADS)
            if kinds[r0] == GEN:
                ctx.enqueue_function[argmax_k](self.z.unsafe_offset((r0 - 1) * b * V), ids.unsafe_offset(r0 * b), Int32(b), Int32(V), grid_dim=grid(b), block_dim=THREADS)
                ctx.enqueue_function[embed_k](p, ids.unsafe_offset(r0 * b), self.acts[0].x.unsafe_offset(r0 * b * D), Int32(b), Int32(D), grid_dim=grid(b * D), block_dim=THREADS)
            self.fwd_seg(ctx, p, tgt, wr, r0, segs[i + 1], True)
        if not train:
            return
        ctx.enqueue_memset(DeviceBuffer[DType.float32](ctx, self.dnf, S * b * D, owning=False), 0)
        for l in range(L):
            ctx.enqueue_memset(DeviceBuffer[DType.float32](ctx, self.acts[l].dk, S * b * D, owning=False), 0)
            ctx.enqueue_memset(DeviceBuffer[DType.float32](ctx, self.acts[l].dv, S * b * D, owning=False), 0)
        for i in reversed(range(len(segs) - 1)):
            var r0 = segs[i]
            var r1 = segs[i + 1]
            self.bwd_seg(ctx, p, g, r0, r1)
            var nr = (r1 - r0) * b
            ctx.enqueue_function[embed_bwd_k](ids.unsafe_offset(r0 * b), self.dx.unsafe_offset(r0 * b * D), self.part, Int32(nr),
                grid_dim=(ceildiv(nr, EC), D // 256), block_dim=256)
            ctx.enqueue_function[colsum_k](self.part, g, Int32(ceildiv(nr, EC)), Int32(V * D), grid_dim=grid(V * D), block_dim=THREADS)
            var j = 0
            for r in range(S):
                if kinds[r] == PSE:
                    if r >= r0 and r < r1:
                        ctx.enqueue_function[colsum_k](self.dx.unsafe_offset(r * b * D), g.unsafe_offset(PAUSE + j * D), Int32(b), Int32(D),
                            grid_dim=grid(D), block_dim=THREADS)
                    j += 1
            if kinds[r0] == LAT:
                self.copy(ctx, self.dx.unsafe_offset(r0 * b * D), self.dnf.unsafe_offset((r0 - 1) * b * D), b * D, True)
            if kinds[r0] == NEU:
                self.lk.bwd[Self.exact](ctx, p, g, self.dx.unsafe_offset(r0 * b * D), self.nf.unsafe_offset((r0 - 1) * b * D),
                    self.dnf.unsafe_offset((r0 - 1) * b * D), hop[r0])
            if kinds[0] == ENT and i == 0:
                self.lk.ent_bwd(ctx, g, self.dx)
            if kinds[r0] == SOFT and not self.sg:
                ctx.enqueue_function[soft_dz_k](self.z.unsafe_offset((r0 - 1) * b * V), p, self.dx.unsafe_offset(r0 * b * D),
                    self.dz.unsafe_offset((r0 - 1) * b * V), self.sp, Int32(b), self.itemp, grid_dim=grid(b), block_dim=THREADS)
                ctx.enqueue_function[soft_de_k](self.sp, self.dx.unsafe_offset(r0 * b * D), g, Int32(b), grid_dim=grid(V * D), block_dim=THREADS)


def row_weights(kinds: List[Int], tgt: List[Int32], b: Int, aux: Float64, rw: Bool = False, base: Int = 0) -> List[Float32]:
    """Per-row loss weights, rows r * b + i: rows feeding a LAT/SOFT row weigh aux / n_aux when aux > 0, other targeted rows
    1 / n_main (rw: row j = r - base weighs (1 + j/7) / (12 b)). Shared by run.weigh and test_latent, which checks it against the oracle."""
    var S = len(kinds)
    var w = List[Float32](length=S * b, fill=0)
    var nm = 0
    var na = 0
    for r in range(S):
        var feeds = aux > 0 and r + 1 < S and (kinds[r + 1] == LAT or kinds[r + 1] == SOFT)
        for i in range(b):
            if tgt[r * b + i] >= 0:
                if feeds:
                    na += 1
                else:
                    nm += 1
    for r in range(S):
        var feeds = aux > 0 and r + 1 < S and (kinds[r + 1] == LAT or kinds[r + 1] == SOFT)
        for i in range(b):
            if tgt[r * b + i] < 0:
                continue
            if feeds:
                w[r * b + i] = Float32(aux / Float64(na))
            elif rw:
                w[r * b + i] = Float32((1 + Float64(r - base) / 7) / (Float64(b) * 12))
            else:
                w[r * b + i] = Float32(1 / Float64(nm))
    return w^
