"""One differential runner for miniagi/train/ ops.

gemm[ta, tb]: fp64 host reference, gate |err| <= k * 2^-24 * sum|a*b| (outputs near 0 defeat a relative gate).
Other ops: torch oracle cases from tools/train_oracle.py in .work/train/cases, gate max|err| <= TOL * max|want|.
"""
from std.random import seed, random_float64
from max.gpu.host import DeviceContext, DeviceBuffer
from ops import *
from model import *
from fast import Wmma

comptime CASES = ".work/train/cases/"
comptime TOL = 1e-5


struct Dev(Movable):
    """Host + device fp32 buffer pair."""
    var h: List[Float32]
    var d: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, n: Int, fill: Bool = True) raises:
        self.h = List[Float32](length=n, fill=0)
        if fill:
            for i in range(n):
                self.h[i] = Float32(random_float64(-1, 1))
        self.d = ctx.enqueue_create_buffer[DType.float32](n)
        upload(ctx, self.d, self.h)

    @staticmethod
    def of(ctx: DeviceContext, h: List[Float32]) raises -> Self:
        var d = Dev(ctx, len(h), fill=False)
        d.h = h.copy()
        upload(ctx, d.d, d.h)
        return d^

    @staticmethod
    def load(ctx: DeviceContext, path: String) raises -> Self:
        var raw = open(path, "r").read_bytes()
        var d = Dev(ctx, len(raw) // 4, fill=False)
        for i in range(len(d.h)):
            d.h[i] = raw.unsafe_ptr().unsafe_offset(4 * i).unsafe_bitcast[Float32]()[]
        upload(ctx, d.d, d.h)
        return d^

    def ints(self, ctx: DeviceContext) raises -> DeviceBuffer[DType.int32]:
        var h = List[Int32](length=len(self.h), fill=0)
        for i in range(len(h)):
            h[i] = Int32(self.h[i])
        var b = ctx.enqueue_create_buffer[DType.int32](len(h))
        upload(ctx, b, h)
        ctx.synchronize()
        return b^

    def p(self) -> F:
        return F(unsafe_from_address=Int(self.d.unsafe_ptr()))

    def pull(mut self, ctx: DeviceContext) raises:
        ctx.enqueue_copy(dst_ptr=self.h.unsafe_ptr(), src_buf=self.d)
        ctx.synchronize()


def check_gemm[ta: Bool, tb: Bool](ctx: DeviceContext, m: Int, n: Int, k: Int) raises -> Bool:
    var a = Dev(ctx, m * k)
    var b = Dev(ctx, k * n)
    var c = Dev(ctx, m * n)
    gemm[ta, tb](ctx, a.p(), b.p(), c.p(), m, n, k)
    c.pull(ctx)
    var worst = 0.0
    for i in range(m):
        for j in range(n):
            var s = 0.0
            var mag = 0.0
            for p in range(k):
                var t = Float64(a.h[p * m + i if ta else i * k + p]) * Float64(b.h[j * k + p if tb else p * n + j])
                s += t
                mag += abs(t)
            worst = max(worst, abs(Float64(c.h[i * n + j]) - s) / (Float64(k) * 2.0**-24 * mag))
    var ok = worst <= 1.0
    print("PASS" if ok else "FAIL", "gemm", ta, tb, m, n, k, "err/fp32_bound", worst)
    return ok


def ip(b: DeviceBuffer[DType.int32]) -> I:
    return I(unsafe_from_address=Int(b.unsafe_ptr()))


def same(ctx: DeviceContext, op: String, name: String, mut got: Dev, tol: Float64 = TOL) raises -> Bool:
    got.pull(ctx)
    var want = Dev.load(ctx, CASES + op + "/" + name + ".f32")
    var err: Float64 = 0
    var mag: Float64 = 0
    for i in range(len(want.h)):
        err = max(err, abs(Float64(got.h[i]) - Float64(want.h[i])))
        mag = max(mag, abs(Float64(want.h[i])))
    var ok = len(got.h) == len(want.h) and err <= tol * mag
    print("PASS" if ok else "FAIL", op, name, "err/max", err / mag)
    return ok


def in_(ctx: DeviceContext, op: String, name: String) raises -> Dev:
    return Dev.load(ctx, CASES + op + "/" + name + ".f32")


def check_wmma[ta: Bool, tb: Bool](ctx: DeviceContext, w: Wmma, m: Int, n: Int, k: Int) raises -> Bool:
    var a = Dev(ctx, m * k)
    var b = Dev(ctx, k * n)
    var c = Dev(ctx, m * n)
    w.gemm[ta, tb](ctx, a.p(), b.p(), c.p(), m, n, k)
    c.pull(ctx)
    var worst = 0.0
    for i in range(m):
        for j in range(n):
            var s = 0.0
            var mag = 0.0
            for p in range(k):
                var t = Float64(a.h[p * m + i if ta else i * k + p]) * Float64(b.h[j * k + p if tb else p * n + j])
                s += t
                mag += abs(t)
            worst = max(worst, abs(Float64(c.h[i * n + j]) - s) / ((2.0**-10 + Float64(k) * 2.0**-24) * mag))
    var ok = worst <= 1.0
    print("PASS" if ok else "FAIL", "wmma", ta, tb, m, n, k, "err/fp16_bound", worst)
    return ok


def check_ops(ctx: DeviceContext) raises -> Int:
    comptime T = 64
    comptime D = 96
    comptime H = 4
    comptime HD = 24
    comptime V = 256
    var fails = 0

    var x = in_(ctx, "rmsnorm", "x")
    var w = in_(ctx, "rmsnorm", "w")
    var dy = in_(ctx, "rmsnorm", "dy")
    var y = Dev(ctx, T * D, fill=False)
    var r = Dev(ctx, T, fill=False)
    var dx = Dev(ctx, T * D, fill=False)
    var dw = Dev(ctx, D, fill=False)
    ctx.enqueue_function[rmsnorm_fwd_k](x.p(), w.p(), y.p(), r.p(), Int32(T), Int32(D), Float32(1e-6), grid_dim=grid(T), block_dim=THREADS)
    ctx.enqueue_function[rmsnorm_dx_k](x.p(), w.p(), r.p(), dy.p(), dx.p(), Int32(T), Int32(D), grid_dim=grid(T), block_dim=THREADS)
    ctx.enqueue_function[rmsnorm_dw_k](x.p(), r.p(), dy.p(), dw.p(), Int32(T), Int32(D), Int32(0), grid_dim=grid(D), block_dim=THREADS)
    fails += Int(not same(ctx, "rmsnorm", "y", y)) + Int(not same(ctx, "rmsnorm", "dx", dx)) + Int(not same(ctx, "rmsnorm", "dw", dw))

    var a = in_(ctx, "swiglu", "a")
    var b = in_(ctx, "swiglu", "b")
    var gy = in_(ctx, "swiglu", "dy")
    var sy = Dev(ctx, T * D, fill=False)
    var da = Dev(ctx, T * D, fill=False)
    var db = Dev(ctx, T * D, fill=False)
    ctx.enqueue_function[swiglu_fwd_k](a.p(), b.p(), sy.p(), Int32(T * D), grid_dim=grid(T * D), block_dim=THREADS)
    ctx.enqueue_function[swiglu_bwd_k](a.p(), b.p(), gy.p(), da.p(), db.p(), Int32(T * D), grid_dim=grid(T * D), block_dim=THREADS)
    fails += Int(not same(ctx, "swiglu", "y", sy)) + Int(not same(ctx, "swiglu", "da", da)) + Int(not same(ctx, "swiglu", "db", db))

    var rx = in_(ctx, "rope", "x")
    var rdy = in_(ctx, "rope", "dy")
    var tab = rope_tables(T, HD, 10000)
    var cs = Dev.of(ctx, tab[0])
    var sn = Dev.of(ctx, tab[1])
    var ry = Dev(ctx, T * H * HD, fill=False)
    var rdx = Dev(ctx, T * H * HD, fill=False)
    ctx.enqueue_function[rope_k[False]](rx.p(), ry.p(), cs.p(), sn.p(), Int32(T), Int32(H), Int32(HD), Int32(0), grid_dim=grid(T * H * HD // 2), block_dim=THREADS)
    ctx.enqueue_function[rope_k[True]](rdy.p(), rdx.p(), cs.p(), sn.p(), Int32(T), Int32(H), Int32(HD), Int32(0), grid_dim=grid(T * H * HD // 2), block_dim=THREADS)
    fails += Int(not same(ctx, "rope", "y", ry)) + Int(not same(ctx, "rope", "dx", rdx))

    var e = in_(ctx, "embed", "e")
    var idx = in_(ctx, "embed", "idx").ints(ctx)
    var edy = in_(ctx, "embed", "dy")
    var ey = Dev(ctx, T * D, fill=False)
    var de = Dev(ctx, V * D, fill=False)
    ctx.enqueue_function[embed_fwd_k](e.p(), ip(idx), ey.p(), Int32(T), Int32(D), grid_dim=grid(T * D), block_dim=THREADS)
    ctx.enqueue_function[embed_bwd_k](ip(idx), edy.p(), de.p(), Int32(T), Int32(D), Int32(V), Int32(0), grid_dim=grid(V * D), block_dim=THREADS)
    fails += Int(not same(ctx, "embed", "y", ey)) + Int(not same(ctx, "embed", "de", de))

    var z = in_(ctx, "ce", "z")
    var tgt = in_(ctx, "ce", "tgt").ints(ctx)
    var rows = Dev(ctx, T, fill=False)
    var dz = Dev(ctx, T * V, fill=False)
    ctx.enqueue_function[ce_k](z.p(), ip(tgt), rows.p(), dz.p(), Int32(T), Int32(V), grid_dim=grid(T), block_dim=THREADS)
    rows.pull(ctx)
    var loss = Dev(ctx, 1, fill=False)
    for t in range(T):
        loss.h[0] += rows.h[t] / Float32(T)
    upload(ctx, loss.d, loss.h)
    fails += Int(not same(ctx, "ce", "loss", loss)) + Int(not same(ctx, "ce", "dz", dz))
    var q = in_(ctx, "attn", "q")
    var k = in_(ctx, "attn", "k")
    var v = in_(ctx, "attn", "v")
    var ady = in_(ctx, "attn", "dy")
    comptime N = T * H * HD
    var hq = Dev(ctx, N, fill=False)
    var hk = Dev(ctx, N, fill=False)
    var hv = Dev(ctx, N, fill=False)
    var hdo = Dev(ctx, N, fill=False)
    var pp = Dev(ctx, H * T * T, fill=False)
    var ds = Dev(ctx, H * T * T, fill=False)
    var ho = Dev(ctx, N, fill=False)
    var hdq = Dev(ctx, N, fill=False)
    var hdk = Dev(ctx, N, fill=False)
    var hdv = Dev(ctx, N, fill=False)
    var ay = Dev(ctx, N, fill=False)
    var adq = Dev(ctx, N, fill=False)
    var adk = Dev(ctx, N, fill=False)
    var adv = Dev(ctx, N, fill=False)
    for pr in [(q.p(), hq.p()), (k.p(), hk.p()), (v.p(), hv.p()), (ady.p(), hdo.p())]:
        ctx.enqueue_function[perm_k[False]](pr[0], pr[1], Int32(T), Int32(H), Int32(HD), grid_dim=grid(N), block_dim=THREADS)
    attn_fwd(ctx, hq.p(), hk.p(), hv.p(), pp.p(), ho.p(), H, T, HD)
    attn_bwd(ctx, hq.p(), hk.p(), hv.p(), pp.p(), hdo.p(), ds.p(), hdq.p(), hdk.p(), hdv.p(), H, T, HD)
    for pr in [(ho.p(), ay.p()), (hdq.p(), adq.p()), (hdk.p(), adk.p()), (hdv.p(), adv.p())]:
        ctx.enqueue_function[perm_k[True]](pr[0], pr[1], Int32(T), Int32(H), Int32(HD), grid_dim=grid(N), block_dim=THREADS)
    fails += Int(not same(ctx, "attn", "y", ay)) + Int(not same(ctx, "attn", "dq", adq))
    fails += Int(not same(ctx, "attn", "dk", adk)) + Int(not same(ctx, "attn", "dv", adv))
    return fails


def check_model(ctx: DeviceContext) raises -> Int:
    comptime T = 128
    var p = in_(ctx, "model", "init")
    var g = Dev(ctx, NP, fill=False)
    var toks = in_(ctx, "model", "tokens")
    var tok = Dev.of(ctx, List(toks.h[0:T])).ints(ctx)
    var tgt = Dev.of(ctx, List(toks.h[1 : T + 1])).ints(ctx)
    var m = Model(ctx, T)
    var loss = Dev(ctx, 1, fill=False)
    loss.h[0] = m.forward(ctx, p.p(), ip(tok), ip(tgt))
    upload(ctx, loss.d, loss.h)
    m.backward(ctx, p.p(), g.p(), ip(tok))
    var fails = Int(not same(ctx, "model", "loss", loss, 1e-5))
    var want = in_(ctx, "model", "grad")
    g.pull(ctx)
    var names = ["g1", "wq", "wk", "wv", "wo", "g2", "w1", "w3", "w2"]
    var sizes = [D, DD, DD, DD, DD, D, DF, DF, DF]
    var segs = List[Tuple[String, Int, Int]]()
    segs.append(("E", 0, V * D))
    for l in range(L):
        var o = V * D + l * LAYER
        for i in range(9):
            segs.append((String(names[i], l), o, sizes[i]))
            o += sizes[i]
    segs.append(("gf", NP - D, D))
    for sg in segs:
        var err: Float64 = 0
        var mag: Float64 = 0
        for i in range(sg[1], sg[1] + sg[2]):
            err = max(err, abs(Float64(g.h[i]) - Float64(want.h[i])))
            mag = max(mag, abs(Float64(want.h[i])))
        var ok = mag > 0 and err <= 1e-3 * mag
        fails += Int(not ok)
        print("PASS" if ok else "FAIL", "model grad", sg[0], "err/max", err / mag)
    return fails


def check_adamw(ctx: DeviceContext) raises -> Int:
    comptime N = 1000
    comptime STEPS = 10
    var p = in_(ctx, "adamw", "p0")
    var grads = in_(ctx, "adamw", "grads")
    var m = ctx.enqueue_create_buffer[DType.bfloat16](N)
    var v = ctx.enqueue_create_buffer[DType.bfloat16](N)
    ctx.enqueue_memset(m, 0)
    ctx.enqueue_memset(v, 0)
    var mp = B(unsafe_from_address=Int(m.unsafe_ptr()))
    var vp = B(unsafe_from_address=Int(v.unsafe_ptr()))
    for t in range(1, STEPS + 1):
        var c1 = Float32(1 / (1 - 0.9 ** Float64(t)))
        var c2 = Float32(1 / (1 - 0.95 ** Float64(t)))
        ctx.enqueue_function[adamw_k](p.p(), grads.p().unsafe_offset((t - 1) * N), mp, vp, Int32(N), Float32(3e-3), Float32(0.9),
            Float32(0.95), Float32(1e-8), Float32(0.1), c1, c2, grid_dim=grid(N), block_dim=THREADS)
    return Int(not same(ctx, "adamw", "p", p, 1e-5))


def main() raises:
    seed(0)
    var ctx = DeviceContext()
    var fails = 0
    comptime shapes = [(64, 64, 64), (37, 129, 70), (512, 256, 256)]
    comptime for s in range(3):
        comptime m = shapes[s][0]
        comptime n = shapes[s][1]
        comptime k = shapes[s][2]
        fails += Int(not check_gemm[False, False](ctx, m, n, k))
        fails += Int(not check_gemm[False, True](ctx, m, n, k))
        fails += Int(not check_gemm[True, False](ctx, m, n, k))
        fails += Int(not check_gemm[True, True](ctx, m, n, k))
    var w = Wmma(ctx, 512 * 688)
    comptime wshapes = [(64, 64, 64), (37, 129, 70), (512, 688, 256)]
    comptime for s in range(3):
        comptime m = wshapes[s][0]
        comptime n = wshapes[s][1]
        comptime k = wshapes[s][2]
        fails += Int(not check_wmma[False, False](ctx, w, m, n, k)) + Int(not check_wmma[False, True](ctx, w, m, n, k))
        fails += Int(not check_wmma[True, False](ctx, w, m, n, k)) + Int(not check_wmma[True, True](ctx, w, m, n, k))
    fails += check_ops(ctx)
    fails += check_model(ctx)
    fails += check_adamw(ctx)
    comptime N = 12 + 12 + 12 + 4 + 1 + 2 + 9 * 2 + 1
    if fails:
        raise Error("FAIL " + String(fails) + "/" + String(N) + " parity cases")
    print("PASS", N, "/", N)
