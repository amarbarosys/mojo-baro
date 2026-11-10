from std.math import exp, log
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from ops import F, I, THREADS, grid, gemm, upload
from lops import cast16_k, tr16_k, H16, DM
from layout import TileTensor, row_major
from linalg.matmul import matmul as lmatmul

comptime EB = 384


def lk_sm_k(l: F, a: F, mt: I, lo: F, b32: Int32, m32: Int32, mv32: Int32, ls: Float32):
    var i = global_idx.x
    var b = Int(b32)
    if i >= b:
        return
    var m = Int(m32)
    var mv = Int(mv32)
    var o = i * m
    var mx = l[unsafe_offset=o] * ls
    var arg = 0
    for j in range(1, mv):
        var x = l[unsafe_offset=o + j] * ls
        if x > mx:
            mx = x
            arg = j
    var s: Float32 = 0
    for j in range(mv):
        s += exp(l[unsafe_offset=o + j] * ls - mx)
    for j in range(mv):
        a[unsafe_offset=o + j] = exp(l[unsafe_offset=o + j] * ls - mx) / s
    for j in range(mv, m):
        a[unsafe_offset=o + j] = 0
    var t = Int(mt[unsafe_offset=i])
    lo[unsafe_offset=i] = mx + log(s) - l[unsafe_offset=o + t] * ls if t >= 0 else 0
    lo[unsafe_offset=b + i] = Float32(Int(t >= 0 and arg == t))


def lk_bwd_k(a: F, da: F, mt: I, b32: Int32, m32: Int32, mv32: Int32, ls: Float32, aw: Float32):
    var i = global_idx.x
    if i >= Int(b32):
        return
    var m = Int(m32)
    var mv = Int(mv32)
    var o = i * m
    var dot: Float32 = 0
    for j in range(mv):
        dot += a[unsafe_offset=o + j] * da[unsafe_offset=o + j]
    var t = Int(mt[unsafe_offset=i])
    for j in range(mv):
        var g = a[unsafe_offset=o + j] * (da[unsafe_offset=o + j] - dot)
        if t >= 0:
            g += aw * (a[unsafe_offset=o + j] - Float32(Int(j == t)))
        da[unsafe_offset=o + j] = g * ls
    for j in range(mv, m):
        da[unsafe_offset=o + j] = 0


def bmm(ctx: DeviceContext, a16: H16, b16: H16, c: F, m: Int, n: Int, k: Int) raises:
    var cbuf = DeviceBuffer[DType.float32](ctx, c, m * n, owning=False)
    var abuf = DeviceBuffer[DType.bfloat16](ctx, a16, m * k, owning=False)
    var bbuf = DeviceBuffer[DType.bfloat16](ctx, b16, n * k, owning=False)
    lmatmul[transpose_b=True, target="gpu"](TileTensor(cbuf, row_major(m, n)), TileTensor(abuf, row_major(m, k)), TileTensor(bbuf, row_major(n, k)), ctx)


struct Look(Movable):
    var b: Int
    var nh: Int
    var m: Int
    var mv: Int
    var wa: Int
    var wb: Int
    var ls: Float32
    var aw: Float32
    var kf: DeviceBuffer[DType.float32]
    var vf: DeviceBuffer[DType.float32]
    var k16: DeviceBuffer[DType.bfloat16]
    var v16: DeviceBuffer[DType.bfloat16]
    var kt16: DeviceBuffer[DType.bfloat16]
    var vt16: DeviceBuffer[DType.bfloat16]
    var t16: DeviceBuffer[DType.bfloat16]
    var u16: DeviceBuffer[DType.bfloat16]
    var logit: DeviceBuffer[DType.float32]
    var alpha: DeviceBuffer[DType.float32]
    var da: DeviceBuffer[DType.float32]
    var q: DeviceBuffer[DType.float32]
    var v: DeviceBuffer[DType.float32]
    var dv: DeviceBuffer[DType.float32]
    var dq: DeviceBuffer[DType.float32]
    var ent: DeviceBuffer[DType.float32]
    var mt: DeviceBuffer[DType.int32]
    var lo: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, b: Int, nh: Int, m: Int, mv: Int, wa: Int, wb: Int) raises:
        self.b = b
        self.nh = nh
        self.m = m
        self.mv = mv
        self.wa = wa
        self.wb = wb
        self.ls = 4
        self.aw = 0
        var mm = max(m, 1)
        self.kf = ctx.enqueue_create_buffer[DType.float32](mm * EB)
        self.vf = ctx.enqueue_create_buffer[DType.float32](mm * EB)
        self.k16 = ctx.enqueue_create_buffer[DType.bfloat16](mm * EB)
        self.v16 = ctx.enqueue_create_buffer[DType.bfloat16](mm * EB)
        self.kt16 = ctx.enqueue_create_buffer[DType.bfloat16](mm * EB)
        self.vt16 = ctx.enqueue_create_buffer[DType.bfloat16](mm * EB)
        self.t16 = ctx.enqueue_create_buffer[DType.bfloat16](max(b * mm, 1))
        self.u16 = ctx.enqueue_create_buffer[DType.bfloat16](max(b * EB, 1))
        self.logit = ctx.enqueue_create_buffer[DType.float32](max(b * mm, 1))
        self.alpha = ctx.enqueue_create_buffer[DType.float32](max(nh * b * mm, 1))
        self.da = ctx.enqueue_create_buffer[DType.float32](max(b * mm, 1))
        self.q = ctx.enqueue_create_buffer[DType.float32](max(nh * b * EB, 1))
        self.v = ctx.enqueue_create_buffer[DType.float32](max(nh * b * EB, 1))
        self.dv = ctx.enqueue_create_buffer[DType.float32](max(b * EB, 1))
        self.dq = ctx.enqueue_create_buffer[DType.float32](max(b * EB, 1))
        self.ent = ctx.enqueue_create_buffer[DType.float32](max(b * EB, 1))
        self.mt = ctx.enqueue_create_buffer[DType.int32](max(nh * b, 1))
        self.lo = ctx.enqueue_create_buffer[DType.float32](max(nh * 2 * b, 1))
        ctx.enqueue_memset(self.kf, 0)
        ctx.enqueue_memset(self.vf, 0)
        ctx.enqueue_memset(self.alpha, 0)
        ctx.enqueue_memset(self.ent, 0)
        ctx.enqueue_memset(self.lo, 0)
        ctx.enqueue_memset(self.mt, -1)

    def load(self, ctx: DeviceContext, keys: List[Float32], vals: List[Float32]) raises:
        upload(ctx, self.kf, keys)
        upload(ctx, self.vf, vals)
        var n = self.m * EB
        ctx.enqueue_function[cast16_k](F(unsafe_from_address=Int(self.kf.unsafe_ptr())), H16(unsafe_from_address=Int(self.k16.unsafe_ptr())), Int32(n), grid_dim=grid(n), block_dim=THREADS)
        ctx.enqueue_function[cast16_k](F(unsafe_from_address=Int(self.vf.unsafe_ptr())), H16(unsafe_from_address=Int(self.v16.unsafe_ptr())), Int32(n), grid_dim=grid(n), block_dim=THREADS)
        ctx.enqueue_function[tr16_k](F(unsafe_from_address=Int(self.kf.unsafe_ptr())), H16(unsafe_from_address=Int(self.kt16.unsafe_ptr())), Int32(self.m), Int32(EB), grid_dim=grid(n), block_dim=THREADS)
        ctx.enqueue_function[tr16_k](F(unsafe_from_address=Int(self.vf.unsafe_ptr())), H16(unsafe_from_address=Int(self.vt16.unsafe_ptr())), Int32(self.m), Int32(EB), grid_dim=grid(n), block_dim=THREADS)
        ctx.synchronize()

    def fp(self, buf: DeviceBuffer[DType.float32]) -> F:
        return F(unsafe_from_address=Int(buf.unsafe_ptr()))

    def hp(self, buf: DeviceBuffer[DType.bfloat16]) -> H16:
        return H16(unsafe_from_address=Int(buf.unsafe_ptr()))

    def ent_fwd(self, ctx: DeviceContext, p: F, x0: F) raises:
        gemm[False, False](ctx, self.fp(self.ent), p.unsafe_offset(self.wb), x0, self.b, DM, EB)

    def ent_bwd(self, ctx: DeviceContext, g: F, dx0: F) raises:
        gemm[True, False](ctx, self.fp(self.ent), dx0, g.unsafe_offset(self.wb), EB, DM, self.b, True)

    def fwd[exact: Bool](self, ctx: DeviceContext, p: F, nf: F, xrow: F, h: Int) raises:
        var b = self.b
        var m = self.m
        var q = self.fp(self.q).unsafe_offset(h * b * EB)
        var v = self.fp(self.v).unsafe_offset(h * b * EB)
        var a = self.fp(self.alpha).unsafe_offset(h * b * m)
        var lg = self.fp(self.logit)
        gemm[False, False](ctx, nf, p.unsafe_offset(self.wa), q, b, EB, DM)
        comptime if exact:
            gemm[False, True](ctx, q, self.fp(self.kf), lg, b, m, EB)
        else:
            ctx.enqueue_function[cast16_k](q, self.hp(self.u16), Int32(b * EB), grid_dim=grid(b * EB), block_dim=THREADS)
            bmm(ctx, self.hp(self.u16), self.hp(self.k16), lg, b, m, EB)
        ctx.enqueue_function[lk_sm_k](lg, a, I(unsafe_from_address=Int(self.mt.unsafe_ptr())).unsafe_offset(h * b), self.fp(self.lo).unsafe_offset(h * 2 * b),
            Int32(b), Int32(m), Int32(self.mv), self.ls, grid_dim=grid(b), block_dim=THREADS)
        comptime if exact:
            gemm[False, False](ctx, a, self.fp(self.vf), v, b, EB, m)
        else:
            ctx.enqueue_function[cast16_k](a, self.hp(self.t16), Int32(b * m), grid_dim=grid(b * m), block_dim=THREADS)
            bmm(ctx, self.hp(self.t16), self.hp(self.vt16), v, b, EB, m)
        gemm[False, False](ctx, v, p.unsafe_offset(self.wb), xrow, b, DM, EB)

    def bwd[exact: Bool](self, ctx: DeviceContext, p: F, g: F, dx: F, nf: F, dnf: F, h: Int) raises:
        var b = self.b
        var m = self.m
        var q = self.fp(self.q).unsafe_offset(h * b * EB)
        var v = self.fp(self.v).unsafe_offset(h * b * EB)
        var a = self.fp(self.alpha).unsafe_offset(h * b * m)
        var da = self.fp(self.da)
        var dv = self.fp(self.dv)
        var dq = self.fp(self.dq)
        gemm[True, False](ctx, v, dx, g.unsafe_offset(self.wb), EB, DM, b, True)
        gemm[False, True](ctx, dx, p.unsafe_offset(self.wb), dv, b, EB, DM)
        comptime if exact:
            gemm[False, True](ctx, dv, self.fp(self.vf), da, b, m, EB)
        else:
            ctx.enqueue_function[cast16_k](dv, self.hp(self.u16), Int32(b * EB), grid_dim=grid(b * EB), block_dim=THREADS)
            bmm(ctx, self.hp(self.u16), self.hp(self.v16), da, b, m, EB)
        ctx.enqueue_function[lk_bwd_k](a, da, I(unsafe_from_address=Int(self.mt.unsafe_ptr())).unsafe_offset(h * b), Int32(b), Int32(m), Int32(self.mv), self.ls,
            self.aw / Float32(b), grid_dim=grid(b), block_dim=THREADS)
        comptime if exact:
            gemm[False, False](ctx, da, self.fp(self.kf), dq, b, EB, m)
        else:
            ctx.enqueue_function[cast16_k](da, self.hp(self.t16), Int32(b * m), grid_dim=grid(b * m), block_dim=THREADS)
            bmm(ctx, self.hp(self.t16), self.hp(self.kt16), dq, b, EB, m)
        gemm[True, False](ctx, nf, dq, g.unsafe_offset(self.wa), DM, EB, b, True)
        gemm[False, True](ctx, dq, p.unsafe_offset(self.wa), dnf, b, DM, EB, True)
