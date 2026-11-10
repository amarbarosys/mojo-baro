from std.math import ceildiv, cos, exp, log, rsqrt, sin, sqrt
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer

comptime F = MutPointer[Float32, MutUntrackedOrigin]
comptime THREADS = 256


def upload[T: DType](ctx: DeviceContext, dst: DeviceBuffer[T], src: List[Scalar[T]]) raises:
    ctx.enqueue_copy(dst_buf=dst, src_ptr=src.unsafe_ptr())
    ctx.synchronize()


def gemm_k[ta: Bool, tb: Bool](a: F, b: F, c: F, m32: Int32, n32: Int32, k32: Int32, acc: Int32):
    var m = Int(m32)
    var n = Int(n32)
    var k = Int(k32)
    var idx = global_idx.x
    if idx >= m * n:
        return
    var i = idx // n
    var j = idx % n
    var s: Float32 = 0
    for p in range(k):
        s += a[unsafe_offset=p * m + i if ta else i * k + p] * b[unsafe_offset=j * k + p if tb else p * n + j]
    c[unsafe_offset=idx] = c[unsafe_offset=idx] + s if acc != 0 else s


def gemm[ta: Bool, tb: Bool](ctx: DeviceContext, a: F, b: F, c: F, m: Int, n: Int, k: Int, acc: Bool = False) raises:
    ctx.enqueue_function[gemm_k[ta, tb]](a, b, c, Int32(m), Int32(n), Int32(k), Int32(acc), grid_dim=ceildiv(m * n, THREADS), block_dim=THREADS)


comptime I = MutPointer[Int32, MutUntrackedOrigin]


def grid(n: Int) -> Int:
    return ceildiv(n, THREADS)


def rmsnorm_fwd_k(x: F, w: F, y: F, r: F, t32: Int32, d32: Int32, eps: Float32):
    var t = global_idx.x
    var d = Int(d32)
    if t >= Int(t32):
        return
    var ss: Float32 = 0
    for j in range(d):
        ss += x[unsafe_offset=t * d + j] * x[unsafe_offset=t * d + j]
    var rr = rsqrt(ss / Float32(d) + eps)
    r[unsafe_offset=t] = rr
    for j in range(d):
        y[unsafe_offset=t * d + j] = x[unsafe_offset=t * d + j] * rr * w[unsafe_offset=j]


def rmsnorm_dx_k(x: F, w: F, r: F, dy: F, dx: F, t32: Int32, d32: Int32):
    var t = global_idx.x
    var d = Int(d32)
    if t >= Int(t32):
        return
    var rr = r[unsafe_offset=t]
    var s: Float32 = 0
    for j in range(d):
        s += dy[unsafe_offset=t * d + j] * w[unsafe_offset=j] * x[unsafe_offset=t * d + j]
    var c = rr * rr * rr * s / Float32(d)
    for j in range(d):
        var o = t * d + j
        dx[unsafe_offset=o] = rr * w[unsafe_offset=j] * dy[unsafe_offset=o] - x[unsafe_offset=o] * c


def rmsnorm_dw_k(x: F, r: F, dy: F, dw: F, t32: Int32, d32: Int32, acc: Int32):
    var j = global_idx.x
    var d = Int(d32)
    if j >= d:
        return
    var s: Float32 = 0
    for t in range(Int(t32)):
        s += dy[unsafe_offset=t * d + j] * x[unsafe_offset=t * d + j] * r[unsafe_offset=t]
    dw[unsafe_offset=j] = dw[unsafe_offset=j] + s if acc != 0 else s


def swiglu_fwd_k(a: F, b: F, y: F, n32: Int32):
    var i = global_idx.x
    if i >= Int(n32):
        return
    var g = a[unsafe_offset=i]
    y[unsafe_offset=i] = g / (1 + exp(-g)) * b[unsafe_offset=i]


def swiglu_bwd_k(a: F, b: F, dy: F, da: F, db: F, n32: Int32):
    var i = global_idx.x
    if i >= Int(n32):
        return
    var g = a[unsafe_offset=i]
    var s = 1 / (1 + exp(-g))
    var u = dy[unsafe_offset=i]
    db[unsafe_offset=i] = u * g * s
    da[unsafe_offset=i] = u * b[unsafe_offset=i] * s * (1 + g * (1 - s))


def rope_k[inv: Bool](x: F, y: F, cs: F, sn: F, t32: Int32, h32: Int32, hd32: Int32, pos0: Int32):
    var hd = Int(hd32)
    var half = hd // 2
    var idx = global_idx.x
    if idx >= Int(t32) * Int(h32) * half:
        return
    var i = idx % half
    var row = idx // half
    var a = (Int(pos0) + row // Int(h32)) * half + i
    var c = cs[unsafe_offset=a]
    var s = -sn[unsafe_offset=a] if inv else sn[unsafe_offset=a]
    var o = row * hd + i
    var x1 = x[unsafe_offset=o]
    var x2 = x[unsafe_offset=o + half]
    y[unsafe_offset=o] = x1 * c - x2 * s
    y[unsafe_offset=o + half] = x2 * c + x1 * s


def rope_tables(n_pos: Int, hd: Int, theta: Float64) -> Tuple[List[Float32], List[Float32]]:
    var half = hd // 2
    var cs = List[Float32](length=n_pos * half, fill=0)
    var sn = List[Float32](length=n_pos * half, fill=0)
    for t in range(n_pos):
        for i in range(half):
            var ang = Float64(t) * theta ** (-Float64(2 * i) / Float64(hd))
            cs[t * half + i] = Float32(cos(ang))
            sn[t * half + i] = Float32(sin(ang))
    return (cs^, sn^)


def embed_fwd_k(e: F, idx: I, y: F, t32: Int32, d32: Int32):
    var k = global_idx.x
    var d = Int(d32)
    if k >= Int(t32) * d:
        return
    y[unsafe_offset=k] = e[unsafe_offset=Int(idx[unsafe_offset=k // d]) * d + k % d]


def embed_bwd_k(idx: I, dy: F, de: F, t32: Int32, d32: Int32, v32: Int32, acc: Int32):
    var k = global_idx.x
    var d = Int(d32)
    if k >= Int(v32) * d:
        return
    var v = Int32(k // d)
    var s: Float32 = 0
    for t in range(Int(t32)):
        if idx[unsafe_offset=t] == v:
            s += dy[unsafe_offset=t * d + k % d]
    de[unsafe_offset=k] = de[unsafe_offset=k] + s if acc != 0 else s


def ce_k(z: F, tgt: I, loss: F, dz: F, t32: Int32, v32: Int32):
    var t = global_idx.x
    var v = Int(v32)
    if t >= Int(t32):
        return
    var m = z[unsafe_offset=t * v]
    for j in range(v):
        m = max(m, z[unsafe_offset=t * v + j])
    var s: Float32 = 0
    for j in range(v):
        s += exp(z[unsafe_offset=t * v + j] - m)
    var y = Int(tgt[unsafe_offset=t])
    loss[unsafe_offset=t] = log(s) + m - z[unsafe_offset=t * v + y]
    var inv_t = 1 / Float32(Int(t32))
    for j in range(v):
        var p = exp(z[unsafe_offset=t * v + j] - m) / s
        dz[unsafe_offset=t * v + j] = (p - Float32(Int(j == y))) * inv_t


def perm_k[inv: Bool](x: F, y: F, t32: Int32, h32: Int32, hd32: Int32):
    var hd = Int(hd32)
    var t = Int(t32)
    var h = Int(h32)
    var k = global_idx.x
    if k >= t * h * hd:
        return
    var e = k % hd
    var th = k // hd
    var a = (th // h) * h * hd + (th % h) * hd + e
    var b = (th % h) * t * hd + (th // h) * hd + e
    y[unsafe_offset=a if inv else b] = x[unsafe_offset=b if inv else a]


def softmax_causal_k(s: F, p: F, rows32: Int32, t32: Int32, scale: Float32):
    var r = global_idx.x
    var t = Int(t32)
    if r >= Int(rows32):
        return
    var q = r % t
    var o = r * t
    var m = s[unsafe_offset=o] * scale
    for j in range(q + 1):
        m = max(m, s[unsafe_offset=o + j] * scale)
    var z: Float32 = 0
    for j in range(q + 1):
        z += exp(s[unsafe_offset=o + j] * scale - m)
    for j in range(t):
        p[unsafe_offset=o + j] = exp(s[unsafe_offset=o + j] * scale - m) / z if j <= q else 0


def softmax_bwd_k(p: F, dp: F, rows32: Int32, t32: Int32, scale: Float32):
    var r = global_idx.x
    var t = Int(t32)
    if r >= Int(rows32):
        return
    var o = r * t
    var d: Float32 = 0
    for j in range(t):
        d += p[unsafe_offset=o + j] * dp[unsafe_offset=o + j]
    for j in range(t):
        dp[unsafe_offset=o + j] = p[unsafe_offset=o + j] * (dp[unsafe_offset=o + j] - d) * scale


def attn_fwd(ctx: DeviceContext, q: F, k: F, v: F, p: F, o: F, h: Int, t: Int, hd: Int) raises:
    for i in range(h):
        gemm[False, True](ctx, q.unsafe_offset(i * t * hd), k.unsafe_offset(i * t * hd), p.unsafe_offset(i * t * t), t, t, hd)
    ctx.enqueue_function[softmax_causal_k](p, p, Int32(h * t), Int32(t), Float32(hd) ** -0.5, grid_dim=grid(h * t), block_dim=THREADS)
    for i in range(h):
        gemm[False, False](ctx, p.unsafe_offset(i * t * t), v.unsafe_offset(i * t * hd), o.unsafe_offset(i * t * hd), t, hd, t)


def attn_bwd(ctx: DeviceContext, q: F, k: F, v: F, p: F, do: F, ds: F, dq: F, dk: F, dv: F, h: Int, t: Int, hd: Int) raises:
    for i in range(h):
        gemm[False, True](ctx, do.unsafe_offset(i * t * hd), v.unsafe_offset(i * t * hd), ds.unsafe_offset(i * t * t), t, t, hd)
        gemm[True, False](ctx, p.unsafe_offset(i * t * t), do.unsafe_offset(i * t * hd), dv.unsafe_offset(i * t * hd), t, hd, t)
    ctx.enqueue_function[softmax_bwd_k](p, ds, Int32(h * t), Int32(t), Float32(hd) ** -0.5, grid_dim=grid(h * t), block_dim=THREADS)
    for i in range(h):
        gemm[False, False](ctx, ds.unsafe_offset(i * t * t), k.unsafe_offset(i * t * hd), dq.unsafe_offset(i * t * hd), t, hd, t)
        gemm[True, False](ctx, ds.unsafe_offset(i * t * t), q.unsafe_offset(i * t * hd), dk.unsafe_offset(i * t * hd), t, hd, t)


def add_k(x: F, y: F, n32: Int32):
    var i = global_idx.x
    if i < Int(n32):
        y[unsafe_offset=i] = y[unsafe_offset=i] + x[unsafe_offset=i]


comptime B = MutPointer[BFloat16, MutUntrackedOrigin]


def adamw_k(p: F, g: F, m: B, v: B, n32: Int32, lr: Float32, b1: Float32, b2: Float32, eps: Float32, wd: Float32, c1: Float32, c2: Float32):
    var i = global_idx.x
    if i >= Int(n32):
        return
    var gi = g[unsafe_offset=i]
    var mi = (b1 * m[unsafe_offset=i].cast[DType.float32]() + (1 - b1) * gi).cast[DType.bfloat16]()
    var vi = (b2 * v[unsafe_offset=i].cast[DType.float32]() + (1 - b2) * gi * gi).cast[DType.bfloat16]()
    m[unsafe_offset=i] = mi
    v[unsafe_offset=i] = vi
    var pi = p[unsafe_offset=i] * (1 - lr * wd)
    p[unsafe_offset=i] = pi - lr * (mi.cast[DType.float32]() * c1) / (sqrt(vi.cast[DType.float32]() * c2) + eps)
