from std.math import ceildiv, exp, log, sqrt, rsqrt
from max.gpu import block_idx, thread_idx, global_idx
from max.gpu.primitives import warp
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import row_major, stack_allocation
from layout.tensor_core import mma
from std.sys import get_defined_int
from ops import F, I, THREADS, grid

comptime DM = get_defined_int["LAT_D", 256]()
comptime DL = DM // 32

comptime BK = 16
comptime bf = DType.bfloat16


def tgemm_k[ta: Bool, tb: Bool, WM: Int, WN: Int, TM: Int, TN: Int](a: F, b: F, c: F, m32: Int32, n32: Int32, k32: Int32, kl32: Int32, acc: Int32):
    comptime BM = WM * TM * 16
    comptime BN = WN * TN * 16
    comptime GT = WM * WN * 32
    var M = Int(m32)
    var N = Int(n32)
    var K = Int(k32)
    var k0 = Int(block_idx.z) * Int(kl32)
    var k1 = min(K, k0 + Int(kl32))
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var warp = tid // 32
    var wm = warp // WN
    var wn = warp % WN
    var h = lane % 16
    var half = lane // 16
    var r0 = Int(block_idx.y) * BM
    var c0 = Int(block_idx.x) * BN
    var sa = stack_allocation[bf, address_space=AddressSpace.SHARED](row_major[BM, BK]())
    var sb = stack_allocation[bf, address_space=AddressSpace.SHARED](row_major[BK, BN]())
    var accv = SIMD[DType.float32, TM * TN * 8](0)
    var kt = k0
    while kt < k1:
        comptime for s in range(BM * BK // GT):
            var e = tid + s * GT
            var r: Int
            var q: Int
            comptime if ta:
                r = e % BM
                q = e // BM
            else:
                r = e // BK
                q = e % BK
            var gr = r0 + r
            var gk = kt + q
            var v: Float32 = 0
            if gr < M and gk < k1:
                v = a[unsafe_offset=gk * M + gr if ta else gr * K + gk]
            sa[r, q] = rebind[sa.ElementType](v.cast[bf]())
        comptime for s in range(BK * BN // GT):
            var e = tid + s * GT
            var q: Int
            var cc: Int
            comptime if tb:
                q = e % BK
                cc = e // BK
            else:
                q = e // BN
                cc = e % BN
            var gk = kt + q
            var gc = c0 + cc
            var v: Float32 = 0
            if gk < k1 and gc < N:
                v = b[unsafe_offset=gc * K + gk if tb else gk * N + gc]
            sb[q, cc] = rebind[sb.ElementType](v.cast[bf]())
        barrier()
        comptime for tm in range(TM):
            var av = SIMD[bf, 16](0)
            var arow = (wm * TM + tm) * 16 + h
            comptime for i in range(16):
                av[i] = rebind[Scalar[bf]](sa[arow, i])
            comptime for tn in range(TN):
                var bv = SIMD[bf, 16](0)
                var bcol = (wn * TN + tn) * 16 + h
                comptime for i in range(16):
                    bv[i] = rebind[Scalar[bf]](sb[i, bcol])
                var cv = SIMD[DType.float32, 8](0)
                comptime for i in range(8):
                    cv[i] = accv[(tm * TN + tn) * 8 + i]
                var dv = SIMD[DType.float32, 8](0)
                mma(dv, av, bv, cv)
                comptime for i in range(8):
                    accv[(tm * TN + tn) * 8 + i] = dv[i]
        barrier()
        kt += BK
    var out = c.unsafe_offset(Int(block_idx.z) * M * N)
    comptime for tm in range(TM):
        comptime for tn in range(TN):
            comptime for i in range(8):
                var row = r0 + (wm * TM + tm) * 16 + 2 * i + half
                var col = c0 + (wn * TN + tn) * 16 + h
                if row < M and col < N:
                    var o = row * N + col
                    var v = accv[(tm * TN + tn) * 8 + i]
                    out[unsafe_offset=o] = out[unsafe_offset=o] + v if acc != 0 else v


def ksum_k(p: F, c: F, n32: Int32, s32: Int32, acc: Int32):
    var i = global_idx.x
    var n = Int(n32)
    if i >= n:
        return
    var v: Float32 = 0
    for s in range(Int(s32)):
        v += p[unsafe_offset=s * n + i]
    c[unsafe_offset=i] = c[unsafe_offset=i] + v if acc != 0 else v


struct Gemm(Movable):
    var part: DeviceBuffer[DType.float32]
    var cap: Int

    def __init__(out self, ctx: DeviceContext, cap: Int) raises:
        self.cap = cap
        self.part = ctx.enqueue_create_buffer[DType.float32](cap)

    def __call__[ta: Bool, tb: Bool](self, ctx: DeviceContext, a: F, b: F, c: F, m: Int, n: Int, k: Int, acc: Bool = False) raises:
        if k >= 2048 or ceildiv(m, 128) * ceildiv(n, 128) >= 48:
            self.run[ta, tb, 4, 2, 2, 4](ctx, a, b, c, m, n, k, acc)
        elif ceildiv(m, 64) * ceildiv(n, 64) >= 48:
            self.run[ta, tb, 2, 2, 2, 2](ctx, a, b, c, m, n, k, acc)
        else:
            self.run[ta, tb, 2, 2, 1, 1](ctx, a, b, c, m, n, k, acc)

    def run[ta: Bool, tb: Bool, WM: Int, WN: Int, TM: Int, TN: Int](self, ctx: DeviceContext, a: F, b: F, c: F, m: Int, n: Int, k: Int, acc: Bool) raises:
        comptime BM = WM * TM * 16
        comptime BN = WN * TN * 16
        comptime kern = tgemm_k[ta, tb, WM, WN, TM, TN]
        var tiles = ceildiv(m, BM) * ceildiv(n, BN)
        var splits = 1
        while tiles * splits < 192 and k // (splits * 2) >= 256 and (splits * 2) * m * n <= self.cap:
            splits *= 2
        var kl = ceildiv(ceildiv(k, splits), BK) * BK
        splits = ceildiv(k, kl)
        if splits == 1:
            ctx.enqueue_function[kern](a, b, c, Int32(m), Int32(n), Int32(k), Int32(kl), Int32(acc),
                grid_dim=(ceildiv(n, BN), ceildiv(m, BM), 1), block_dim=WM * WN * 32)
            return
        var p = F(unsafe_from_address=Int(self.part.unsafe_ptr()))
        ctx.enqueue_function[kern](a, b, p, Int32(m), Int32(n), Int32(k), Int32(kl), Int32(0),
            grid_dim=(ceildiv(n, BN), ceildiv(m, BM), splits), block_dim=WM * WN * 32)
        ctx.enqueue_function[ksum_k](p, c, Int32(m * n), Int32(splits), Int32(acc), grid_dim=grid(m * n), block_dim=THREADS)


comptime SMAX = get_defined_int["LAT_S", 48]()


def embed_k(e: F, idx: I, y: F, n32: Int32, d32: Int32):
    var k = global_idx.x
    var d = Int(d32)
    if k >= Int(n32) * d:
        return
    var t = Int(idx[unsafe_offset=k // d])
    y[unsafe_offset=k] = e[unsafe_offset=t * d + k % d] if t >= 0 else 0


comptime EC = 256


def embed_bwd_k(idx: I, dy: F, part: F, n32: Int32):
    var d = Int(block_idx.y) * 256 + Int(thread_idx.x)
    var c = Int(block_idx.x)
    var acc = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[32, 256]())
    var dl = Int(thread_idx.x)
    for v in range(32):
        acc[v, dl] = 0
    for t in range(c * EC, min(Int(n32), (c + 1) * EC)):
        var v = Int(idx[unsafe_offset=t])
        if v >= 0:
            acc[v, dl] = acc[v, dl] + rebind[acc.ElementType](dy[unsafe_offset=t * DM + d])
    for v in range(32):
        part[unsafe_offset=c * 32 * DM + v * DM + d] = rebind[Float32](acc[v, dl])


def embed_sort_k(perm: I, start: I, ids: I, dy: F, g: F):
    """Large-vocab embedding gradient: group x owns 32 ids, reads only its tokens through the host counting sort."""
    var c = Int(block_idx.x)
    var d = Int(block_idx.y) * 256 + Int(thread_idx.x)
    var dl = Int(thread_idx.x)
    var acc = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[32, 256]())
    for v in range(32):
        acc[v, dl] = 0
    for j in range(Int(start[unsafe_offset=c]), Int(start[unsafe_offset=c + 1])):
        var t = Int(perm[unsafe_offset=j])
        var v = Int(ids[unsafe_offset=t]) % 32
        acc[v, dl] = acc[v, dl] + rebind[acc.ElementType](dy[unsafe_offset=t * DM + d])
    for v in range(32):
        g[unsafe_offset=(c * 32 + v) * DM + d] = g[unsafe_offset=(c * 32 + v) * DM + d] + rebind[Float32](acc[v, dl])


def rope_k[inv: Bool](x: F, cs: F, sn: F, n32: Int32, b32: Int32, r032: Int32, h32: Int32, hd32: Int32):
    var hd = Int(hd32)
    var half = hd // 2
    var idx = global_idx.x
    if idx >= Int(n32) * Int(h32) * half:
        return
    var i = idx % half
    var row = idx // half
    var pos = Int(r032) + (row // Int(h32)) // Int(b32)
    var a = pos * half + i
    var c = cs[unsafe_offset=a]
    var s = -sn[unsafe_offset=a] if inv else sn[unsafe_offset=a]
    var o = row * hd + i
    var x1 = x[unsafe_offset=o]
    var x2 = x[unsafe_offset=o + half]
    x[unsafe_offset=o] = x1 * c - x2 * s
    x[unsafe_offset=o + half] = x2 * c + x1 * s


comptime AT = 256
comptime NW = AT // 32
comptime HP = 65
comptime SP = SMAX + 1


def attn_fwd_k(q: F, k: F, v: F, p: F, o: F, b32: Int32, h32: Int32, hd32: Int32, r032: Int32, r132: Int32, scale: Float32):
    var b = Int(b32)
    var h = Int(h32)
    var hd = Int(hd32)
    var r0 = Int(r032)
    var r1 = Int(r132)
    var nq = r1 - r0
    var blk = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var wid = tid // 32
    var ld = b * h * hd
    var base = (blk // h) * h * hd + (blk % h) * hd
    var ks = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[SMAX, HP]())
    var vs = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[SMAX, HP]())
    var qs = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[SMAX, HP]())
    var ps = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[SMAX, SP]())
    var idx = tid
    while idx < r1 * hd:
        var j = idx // hd
        var e = idx % hd
        ks[j, e] = rebind[ks.ElementType](k[unsafe_offset=j * ld + base + e])
        vs[j, e] = rebind[vs.ElementType](v[unsafe_offset=j * ld + base + e])
        if j < nq:
            qs[j, e] = rebind[qs.ElementType](q[unsafe_offset=(r0 + j) * ld + base + e])
        idx += AT
    barrier()
    idx = tid
    while idx < nq * r1:
        var i = idx // r1
        var j = idx % r1
        var s: Float32 = -3.0e38
        if j <= r0 + i:
            s = 0
            for e in range(hd):
                s += rebind[Float32](qs[i, e]) * rebind[Float32](ks[j, e])
            s *= scale
        ps[i, j] = rebind[ps.ElementType](s)
        idx += AT
    barrier()
    var i = wid
    while i < nq:
        var a0: Float32 = -3.0e38
        var a1: Float32 = -3.0e38
        if lane < r1:
            a0 = rebind[Float32](ps[i, lane])
        if lane + 32 < r1:
            a1 = rebind[Float32](ps[i, lane + 32])
        var m = warp.max(max(a0, a1))
        var e0: Float32 = exp(a0 - m) if lane <= r0 + i else 0
        var e1: Float32 = exp(a1 - m) if lane + 32 <= r0 + i else 0
        var iz = 1 / warp.sum(e0 + e1)
        var po = (blk * SMAX + r0 + i) * SMAX
        if lane < r1:
            ps[i, lane] = rebind[ps.ElementType](e0 * iz)
            p[unsafe_offset=po + lane] = e0 * iz
        if lane + 32 < r1:
            ps[i, lane + 32] = rebind[ps.ElementType](e1 * iz)
            p[unsafe_offset=po + lane + 32] = e1 * iz
        i += NW
    barrier()
    idx = tid
    while idx < nq * hd:
        var ii = idx // hd
        var e = idx % hd
        var acc: Float32 = 0
        for j in range(r0 + ii + 1):
            acc += rebind[Float32](ps[ii, j]) * rebind[Float32](vs[j, e])
        o[unsafe_offset=(r0 + ii) * ld + base + e] = acc
        idx += AT


def attn_bwd_k(q: F, k: F, v: F, p: F, do: F, dq: F, dk: F, dv: F, b32: Int32, h32: Int32, hd32: Int32, r032: Int32, r132: Int32, scale: Float32):
    var b = Int(b32)
    var h = Int(h32)
    var hd = Int(hd32)
    var r0 = Int(r032)
    var r1 = Int(r132)
    var nq = r1 - r0
    var blk = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var wid = tid // 32
    var ld = b * h * hd
    var base = (blk // h) * h * hd + (blk % h) * hd
    var ks = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[SMAX, HP]())
    var vs = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[SMAX, HP]())
    var dos = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[SMAX, HP]())
    var ps = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[SMAX, SP]())
    var ds = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[SMAX, SP]())
    var idx = tid
    while idx < r1 * hd:
        var j = idx // hd
        var e = idx % hd
        ks[j, e] = rebind[ks.ElementType](k[unsafe_offset=j * ld + base + e])
        vs[j, e] = rebind[vs.ElementType](v[unsafe_offset=j * ld + base + e])
        if j < nq:
            dos[j, e] = rebind[dos.ElementType](do[unsafe_offset=(r0 + j) * ld + base + e])
        idx += AT
    idx = tid
    while idx < nq * r1:
        var i = idx // r1
        var j = idx % r1
        ps[i, j] = rebind[ps.ElementType](p[unsafe_offset=(blk * SMAX + r0 + i) * SMAX + j] if j <= r0 + i else Float32(0))
        idx += AT
    barrier()
    idx = tid
    while idx < nq * r1:
        var i = idx // r1
        var j = idx % r1
        var s: Float32 = 0
        if j <= r0 + i:
            for e in range(hd):
                s += rebind[Float32](dos[i, e]) * rebind[Float32](vs[j, e])
        ds[i, j] = rebind[ds.ElementType](s)
        idx += AT
    barrier()
    var i = wid
    while i < nq:
        var p0: Float32 = 0
        var p1: Float32 = 0
        var d0: Float32 = 0
        var d1: Float32 = 0
        if lane < r1:
            p0 = rebind[Float32](ps[i, lane])
            d0 = rebind[Float32](ds[i, lane])
        if lane + 32 < r1:
            p1 = rebind[Float32](ps[i, lane + 32])
            d1 = rebind[Float32](ds[i, lane + 32])
        var dsum = warp.sum(p0 * d0 + p1 * d1)
        if lane < r1:
            ds[i, lane] = rebind[ds.ElementType](p0 * (d0 - dsum) * scale)
        if lane + 32 < r1:
            ds[i, lane + 32] = rebind[ds.ElementType](p1 * (d1 - dsum) * scale)
        i += NW
    barrier()
    idx = tid
    while idx < nq * hd:
        var ii = idx // hd
        var e = idx % hd
        var acc: Float32 = 0
        for j in range(r0 + ii + 1):
            acc += rebind[Float32](ds[ii, j]) * rebind[Float32](ks[j, e])
        dq[unsafe_offset=(r0 + ii) * ld + base + e] = acc
        idx += AT
    idx = tid
    while idx < r1 * hd:
        var j = idx // hd
        var e = idx % hd
        var ak: Float32 = 0
        var av: Float32 = 0
        for ii in range(max(0, j - r0), nq):
            ak += rebind[Float32](ds[ii, j]) * q[unsafe_offset=(r0 + ii) * ld + base + e]
            av += rebind[Float32](ps[ii, j]) * rebind[Float32](dos[ii, e])
        var o = j * ld + base + e
        dk[unsafe_offset=o] = dk[unsafe_offset=o] + ak
        dv[unsafe_offset=o] = dv[unsafe_offset=o] + av
        idx += AT


comptime RC = 32


def rms_fwd_k(x: F, w: F, y: F, r: F, n32: Int32, eps: Float32):
    var row = Int(block_idx.x) * NW + Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    if row >= Int(n32):
        return
    var o = row * DM
    var xv = SIMD[DType.float32, DL](0)
    comptime for t in range(DL):
        xv[t] = x[unsafe_offset=o + lane + 32 * t]
    var rr = rsqrt(warp.sum((xv * xv).reduce_add()) / Float32(DM) + eps)
    if lane == 0:
        r[unsafe_offset=row] = rr
    comptime for t in range(DL):
        y[unsafe_offset=o + lane + 32 * t] = xv[t] * rr * w[unsafe_offset=lane + 32 * t]


def rms_dx_k(x: F, w: F, r: F, dy: F, dx: F, n32: Int32, acc: Int32):
    var row = Int(block_idx.x) * NW + Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    if row >= Int(n32):
        return
    var o = row * DM
    var rr = r[unsafe_offset=row]
    var xv = SIMD[DType.float32, DL](0)
    var gv = SIMD[DType.float32, DL](0)
    comptime for t in range(DL):
        xv[t] = x[unsafe_offset=o + lane + 32 * t]
        gv[t] = dy[unsafe_offset=o + lane + 32 * t] * w[unsafe_offset=lane + 32 * t]
    var c = rr * rr * rr * warp.sum((gv * xv).reduce_add()) / Float32(DM)
    comptime for t in range(DL):
        var i = o + lane + 32 * t
        var val = rr * gv[t] - xv[t] * c
        dx[unsafe_offset=i] = dx[unsafe_offset=i] + val if acc != 0 else val


def rms_dwp_k(x: F, r: F, dy: F, part: F, n32: Int32):
    var j = Int(thread_idx.x)
    var c = Int(block_idx.x)
    var s: Float32 = 0
    for t in range(c * RC, min(Int(n32), (c + 1) * RC)):
        s += dy[unsafe_offset=t * DM + j] * x[unsafe_offset=t * DM + j] * r[unsafe_offset=t]
    part[unsafe_offset=c * DM + j] = s


def ce_k(z: F, tgt: I, wr: F, loss: F, dz: F, n32: Int32, v32: Int32):
    var t = global_idx.x
    var v = Int(v32)
    if t >= Int(n32):
        return
    var y = Int(tgt[unsafe_offset=t])
    var w = wr[unsafe_offset=t]
    if y < 0 or w == 0:
        loss[unsafe_offset=t] = 0
        for j in range(v):
            dz[unsafe_offset=t * v + j] = 0
        return
    var m = z[unsafe_offset=t * v]
    for j in range(v):
        m = max(m, z[unsafe_offset=t * v + j])
    var s: Float32 = 0
    for j in range(v):
        s += exp(z[unsafe_offset=t * v + j] - m)
    loss[unsafe_offset=t] = (log(s) + m - z[unsafe_offset=t * v + y]) * w
    for j in range(v):
        var pj = exp(z[unsafe_offset=t * v + j] - m) / s
        dz[unsafe_offset=t * v + j] = (pj - Float32(Int(j == y))) * w


def copy_k(src: F, dst: F, n32: Int32, acc: Int32):
    var i = global_idx.x
    if i < Int(n32):
        dst[unsafe_offset=i] = dst[unsafe_offset=i] + src[unsafe_offset=i] if acc != 0 else src[unsafe_offset=i]


def bcast_k(src: F, dst: F, b32: Int32, d32: Int32):
    var i = global_idx.x
    var d = Int(d32)
    if i < Int(b32) * d:
        dst[unsafe_offset=i] = src[unsafe_offset=i % d]


def colsum_k(src: F, dst: F, b32: Int32, d32: Int32):
    var j = global_idx.x
    var d = Int(d32)
    if j >= d:
        return
    var s: Float32 = 0
    for i in range(Int(b32)):
        s += src[unsafe_offset=i * d + j]
    dst[unsafe_offset=j] = dst[unsafe_offset=j] + s


comptime CH = 256


def zero_k(g: F, n32: Int32):
    var i = Int(global_idx.x)
    if i < Int(n32):
        g[unsafe_offset=i] = 0


def sumsq_k(g: F, part: F, n32: Int32):
    var c = global_idx.x
    var n = Int(n32)
    if c * CH >= n:
        return
    var s: Float32 = 0
    for i in range(c * CH, min(n, (c + 1) * CH)):
        var x = g[unsafe_offset=i]
        s += x * x
    part[unsafe_offset=c] = s


def clip_k(part: F, np32: Int32, res: F, maxn: Float32):
    var tid = Int(thread_idx.x)
    var red = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[256]())
    var s: Float32 = 0
    var i = tid
    while i < Int(np32):
        s += part[unsafe_offset=i]
        i += 256
    red[tid] = rebind[red.ElementType](s)
    barrier()
    var w = 128
    while w > 0:
        if tid < w:
            red[tid] = red[tid] + red[tid + w]
        barrier()
        w //= 2
    if tid == 0:
        var nrm = sqrt(rebind[Float32](red[0]))
        res[unsafe_offset=0] = nrm
        res[unsafe_offset=1] = min(Float32(1), maxn / (nrm + 1e-6))


def adamw_k(p: F, g: F, m: F, v: F, sc: F, n32: Int32, lr: Float32, b1: Float32, b2: Float32, eps: Float32, wd: Float32, c1: Float32, c2: Float32):
    var i = global_idx.x
    if i >= Int(n32):
        return
    var gi = g[unsafe_offset=i] * sc[unsafe_offset=1]
    var mi = b1 * m[unsafe_offset=i] + (1 - b1) * gi
    var vi = b2 * v[unsafe_offset=i] + (1 - b2) * gi * gi
    m[unsafe_offset=i] = mi
    v[unsafe_offset=i] = vi
    var pi = p[unsafe_offset=i] * (1 - lr * wd)
    p[unsafe_offset=i] = pi - lr * (mi * c1) / (sqrt(vi * c2) + eps)


def argmax_k(z: F, ids: I, b32: Int32, v32: Int32):
    var i = global_idx.x
    var v = Int(v32)
    if i >= Int(b32):
        return
    var best = 0
    for j in range(1, v):
        if z[unsafe_offset=i * v + j] > z[unsafe_offset=i * v + best]:
            best = j
    ids[unsafe_offset=i] = Int32(best)


def soft_k(z: F, e: F, x: F, b32: Int32, v32: Int32, d32: Int32, it: Float32):
    var k = global_idx.x
    var d = Int(d32)
    var v = Int(v32)
    if k >= Int(b32) * d:
        return
    var i = k // d
    var c = k % d
    var m = z[unsafe_offset=i * v] * it
    for j in range(v):
        m = max(m, z[unsafe_offset=i * v + j] * it)
    var s: Float32 = 0
    var acc: Float32 = 0
    for j in range(v):
        var w = exp(z[unsafe_offset=i * v + j] * it - m)
        s += w
        acc += w * e[unsafe_offset=j * d + c]
    x[unsafe_offset=k] = acc / s


def soft_dz_k(z: F, e: F, dx: F, dz: F, sp: F, b32: Int32, it: Float32):
    var i = global_idx.x
    if i >= Int(b32):
        return
    var m = z[unsafe_offset=i * 32] * it
    for j in range(32):
        m = max(m, z[unsafe_offset=i * 32 + j] * it)
    var p = SIMD[DType.float32, 32](0)
    var ed = SIMD[DType.float32, 32](0)
    comptime for j in range(32):
        p[j] = exp(z[unsafe_offset=i * 32 + j] * it - m)
    p = p / p.reduce_add()
    for c in range(DM):
        var g = dx[unsafe_offset=i * DM + c]
        comptime for j in range(32):
            ed[j] += e[unsafe_offset=j * DM + c] * g
    var dot = (p * ed).reduce_add()
    comptime for j in range(32):
        dz[unsafe_offset=i * 32 + j] = dz[unsafe_offset=i * 32 + j] + it * p[j] * (ed[j] - dot)
        sp[unsafe_offset=i * 32 + j] = p[j]


def soft_de_k(sp: F, dx: F, de: F, b32: Int32):
    var k = global_idx.x
    if k >= 32 * DM:
        return
    var v = k // DM
    var c = k % DM
    var s: Float32 = 0
    for i in range(Int(b32)):
        s += sp[unsafe_offset=i * 32 + v] * dx[unsafe_offset=i * DM + c]
    de[unsafe_offset=k] = de[unsafe_offset=k] + s


def ema_k(p: F, e: F, n32: Int32, d: Float32, init: Int32):
    var i = global_idx.x
    if i < Int(n32):
        e[unsafe_offset=i] = p[unsafe_offset=i] if init != 0 else d * e[unsafe_offset=i] + (1 - d) * p[unsafe_offset=i]


comptime H16 = MutPointer[BFloat16, MutUntrackedOrigin]


def cast16_k(x: F, y: H16, n32: Int32):
    var i = global_idx.x
    if i < Int(n32):
        y[unsafe_offset=i] = x[unsafe_offset=i].cast[DType.bfloat16]()


def tr16_k(x: F, y: H16, rows32: Int32, cols32: Int32):
    var i = global_idx.x
    var rows = Int(rows32)
    var cols = Int(cols32)
    if i < rows * cols:
        y[unsafe_offset=(i % cols) * rows + i // cols] = x[unsafe_offset=i].cast[DType.bfloat16]()
