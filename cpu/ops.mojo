from std.math import fma
from bexp import exp, log, frsq32, fsincos32

comptime f32 = DType.float32


def fold[n: Int](v: SIMD[f32, n], adj: Bool) -> Float32:
    comptime if n == 1:
        return v[0]
    else:
        comptime h = n // 2
        if adj:
            var d = v.deinterleave()
            return fold[h](rebind[SIMD[f32, h]](d[0] + d[1]), adj)
        return fold[h](v.slice[h, offset=0]() + v.slice[h, offset=h](), adj)


def q4_row_dot(
    a: Pointer[BFloat16, _], q: Pointer[UInt8, _], s: Pointer[Float16, _],
    nb: Int,
) -> Float32:
    var acc = Array[SIMD[f32, 16], 32](fill=SIMD[f32, 16](0))
    for blk in range(nb):
        var bytes = q.unsafe_offset(blk * 16).unsafe_load[width=16]()
        var d = s.unsafe_offset(blk).unsafe_load().cast[f32]()
        var dv = SIMD[f32, 16](d)
        var m8d = SIMD[f32, 16](d * -8)
        var wlo = fma((bytes & 0x0F).cast[f32](), dv, m8d)
        var whi = fma((bytes >> 4).cast[f32](), dv, m8d)
        var a_lo = a.unsafe_offset(blk * 32).unsafe_load[width=16]().cast[f32]()
        var a_hi = a.unsafe_offset(blk * 32 + 16).unsafe_load[width=16]().cast[f32]()
        var l = blk % 32
        acc[l] = fma(wlo, a_lo, fma(whi, a_hi, acc[l]))
    var p = SIMD[f32, 32](0)
    for l in range(32):
        p[l] = fold[16](acc[l], False)
    return fold[32](p, True)


def dpp_sum(v: SIMD[f32, 32], sq: Bool) -> Float32:
    var s = SIMD[f32, 32](0)
    for l in range(32):
        s[l] = fma(v[l], v[l], v[l ^ 1] * v[l ^ 1]) if sq else v[l] + v[l ^ 1]
    for m in [2, 7, 8, 16]:
        var t = s
        for l in range(32):
            t[l] = s[l] + s[l ^ m]
        s = t
    return s[0]


def blocksum(p: List[Float32], sq: Bool = False) -> Float32:
    var total = Float32(0)
    for w in range(len(p) // 32):
        var v = SIMD[f32, 32](0)
        for l in range(32):
            v[l] = p[w * 32 + l]
        total += dpp_sum(v, sq)
    return total


def rmsnorm_cast(x: List[Float32], g: List[Float32], eps: Float32) -> List[BFloat16]:
    var n = len(x)
    var part = List[Float32](length=256, fill=0)
    for t in range(256):
        var p = Float32(0)
        for i in range(t, n, 256):
            p = fma(x[i], x[i], p)
        part[t] = p
    var scale = frsq32(blocksum(part) / Float32(n) + eps)
    var out = List[BFloat16](length=n, fill=0)
    for i in range(n):
        out[i] = (x[i] * scale * g[i]).cast[DType.bfloat16]()
    return out^


def head_rmsnorm(mut x: List[Float32], head: Int, hd: Int, g: List[Float32], eps: Float32):
    var part = List[Float32](length=hd, fill=0)
    for d in range(hd):
        part[d] = x[head * hd + d]
    var scale = frsq32(blocksum(part, True) / Float32(hd) + eps)
    for d in range(hd):
        x[head * hd + d] = x[head * hd + d] * scale * g[d]


def rope_yarn(mut x: List[Float32], row: Int, hd: Int, pos: Int, nrot: Int, base: Float32, fscale: Float32, mscale: Float32, lo: Float32, hi: Float32):
    for j in range(nrot // 2):
        var te = Float32(pos) * exp(Float32(-2 * j) / Float32(nrot) * log(base))
        var ti = fscale * te
        var ramp = (Float32(j) - lo) / max(hi - lo, 0.001)
        ramp = min(max(ramp, 0), 1)
        var th = ti * (1 - ramp) + te * ramp
        var sc = fsincos32(th)
        var c = sc[1] * mscale
        var s = sc[0] * mscale
        var x0 = x[row * hd + j]
        var x1 = x[row * hd + j + nrot // 2]
        x[row * hd + j] = x0 * c - x1 * s
        x[row * hd + j + nrot // 2] = x0 * s + x1 * c


def attn_head(q: List[Float32], qoff: Int, kc: List[Float32], vc: List[Float32], kvh: Int, T: Int, hd: Int, scale: Float32) -> List[Float32]:
    var base = kvh * 128 * hd
    var m_run = Float32(-3.4e38)
    var l_run = Float32(0)
    var o = List[Float32](length=hd, fill=0)
    var sc = List[Float32](length=hd, fill=0)
    var t0 = 0
    while t0 < T:
        var cmax = Float32(-3.4e38)
        var s = List[Float32](length=hd, fill=0)
        for tid in range(hd):
            var t = t0 + tid
            var sv = Float32(-3.4e38)
            if t < T:
                var acc = Float32(0)
                var kb = base + (t & 127) * hd
                for d in range(hd):
                    acc = fma(q[qoff + d], kc[kb + d], acc)
                sv = acc * scale
            s[tid] = sv
            if sv > cmax:
                cmax = sv
        var m_new = max(m_run, cmax)
        var alpha = exp(m_run - m_new)
        var e = List[Float32](length=hd, fill=0)
        for tid in range(hd):
            e[tid] = exp(s[tid] - m_new)
        var csum = Float32(0)
        for w in range(hd // 32):
            var v = SIMD[f32, 32](0)
            for l in range(32):
                v[l] = e[w * 32 + l]
            csum += fold[32](v, True)
        l_run = fma(l_run, alpha, csum)
        m_run = m_new
        var n = min(T - t0, hd)
        for d in range(hd):
            var acc = o[d] * alpha
            for tt in range(n):
                var vv = vc[base + ((t0 + tt) & 127) * hd + d]
                acc = fma(e[tt], vv, acc)
            o[d] = acc
        t0 += hd
    var inv = Float32(1) / l_run
    for d in range(hd):
        o[d] = o[d] * inv
    return o^
