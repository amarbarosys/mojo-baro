from std.math import fma, log1p, sqrt
from bexp import fexp32 as exp, flog32 as log, frsq32, fsincos32

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


def wsum(p: Pointer[Float32, _], n: Int, sq: Bool) -> Float32:
    var total = Float32(0)
    for w in range(n // 32):
        total += dpp_sum(p.unsafe_offset(w * 32).unsafe_load[width=32](), sq)
    return total


def rmsnorm_cast[dt: DType = DType.bfloat16](x: Pointer[Float32, _], g: Pointer[Float32, _], dst: MutPointer[Scalar[dt], _], n: Int, eps: Float32):
    var part = Array[Float32, 256](fill=0)
    for t in range(256):
        var p = Float32(0)
        for i in range(t, n, 256):
            p = fma(x[unsafe_offset=i], x[unsafe_offset=i], p)
        part[t] = p
    var scale = frsq32(wsum(part.unsafe_ptr(), 256, False) / Float32(n) + eps)
    for i in range(n):
        dst[unsafe_offset=i] = (x[unsafe_offset=i] * scale * g[unsafe_offset=i]).cast[dt]()


def head_rmsnorm(x: MutPointer[Float32, _], hd: Int, g: Pointer[Float32, _], eps: Float32):
    var scale = frsq32(wsum(x, hd, True) / Float32(hd) + eps)
    for d in range(hd):
        x[unsafe_offset=d] = x[unsafe_offset=d] * scale * g[unsafe_offset=d]


def rope_yarn(x: MutPointer[Float32, _], pos: Int, nrot: Int, base: Float32, fscale: Float32, mscale: Float32, lo: Float32, hi: Float32):
    for j in range(nrot // 2):
        var te = Float32(pos) * exp(Float32(-2 * j) / Float32(nrot) * log(base))
        var ti = fscale * te
        var ramp = (Float32(j) - lo) / max(hi - lo, 0.001)
        ramp = min(max(ramp, 0), 1)
        var th = ti * (1 - ramp) + te * ramp
        var sc = fsincos32(th)
        var c = sc[1] * mscale
        var s = sc[0] * mscale
        var x0 = x[unsafe_offset=j]
        var x1 = x[unsafe_offset=j + nrot // 2]
        x[unsafe_offset=j] = x0 * c - x1 * s
        x[unsafe_offset=j + nrot // 2] = x0 * s + x1 * c


def attn_head(q: Pointer[Float32, _], kc: Pointer[Float32, _], vc: Pointer[Float32, _], T: Int, hd: Int, cap: Int, scale: Float32, o: MutPointer[Float32, _]):
    """One query head over T cached rows; row t of the head's cache is at (t % cap) * hd. hd <= 256."""
    var m_run = Float32(-3.4e38)
    var l_run = Float32(0)
    var s = Array[Float32, 256](fill=0)
    var e = Array[Float32, 256](fill=0)
    for d in range(hd):
        o[unsafe_offset=d] = 0
    var t0 = 0
    while t0 < T:
        var cmax = Float32(-3.4e38)
        for tid in range(hd):
            var t = t0 + tid
            var sv = Float32(-3.4e38)
            if t < T:
                var acc = Float32(0)
                var kb = (t % cap) * hd
                for d in range(hd):
                    acc = fma(q[unsafe_offset=d], kc[unsafe_offset=kb + d], acc)
                sv = acc * scale
            s[tid] = sv
            if sv > cmax:
                cmax = sv
        var m_new = max(m_run, cmax)
        var alpha = exp(m_run - m_new)
        for tid in range(hd):
            e[tid] = exp(s[tid] - m_new)
        var csum = Float32(0)
        for w in range(hd // 32):
            csum += fold[32](e.unsafe_ptr().unsafe_offset(w * 32).unsafe_load[width=32](), True)
        l_run = fma(l_run, alpha, csum)
        m_run = m_new
        var n = min(T - t0, hd)
        for d in range(hd):
            var acc = o[unsafe_offset=d] * alpha
            for tt in range(n):
                acc = fma(e[tt], vc[unsafe_offset=((t0 + tt) % cap) * hd + d], acc)
            o[unsafe_offset=d] = acc
        t0 += hd
    var inv = Float32(1) / l_run
    for d in range(hd):
        o[unsafe_offset=d] = o[unsafe_offset=d] * inv

def swiglu(g: Float32, u: Float32) -> BFloat16:
    var a = Float32(0) + g
    return ((a / (1 + exp(-a))) * (Float32(0) + u)).cast[DType.bfloat16]()


def ssm_gates(araw: Float32, braw: Float32, sa: Float32, db: Float32) -> SIMD[f32, 2]:
    var b = Float32(0) + braw
    var sp = log1p(exp((Float32(0) + araw) + db))
    return SIMD[f32, 2](exp(sp * sa), 1 / (1 + exp(-b)))


def ssm_conv(qkv: Pointer[Float32, _], st: MutPointer[Float32, _], cw: Pointer[Float32, _], dst: MutPointer[Float32, _], nc: Int, nk: Int, s: Int, eps: Float32):
    for c in range(nc):
        var w = SIMD[f32, 4](st[unsafe_offset=c], st[unsafe_offset=nc + c], st[unsafe_offset=2 * nc + c], qkv[unsafe_offset=c])
        var k = cw.unsafe_offset(4 * c).unsafe_load[width=4]()
        var acc = fma(w[3], k[3], fma(w[2], k[2], fma(w[0], k[0], w[1] * k[1])))
        dst[unsafe_offset=c] = acc / (1 + exp(-acc))
        for t in range(3):
            st[unsafe_offset=t * nc + c] = w[t + 1]
    for h in range(2 * nk):
        var p = dst.unsafe_offset(h * s)
        var total = Float32(0)
        for wv in range(s // 32):
            total += dpp_sum(p.unsafe_offset(wv * 32).unsafe_load[width=32](), True)
        var inv = frsq32(total + eps)
        if h < nk:
            inv = inv / sqrt(Float32(s))
        for i in range(s):
            p[unsafe_offset=i] = p[unsafe_offset=i] * inv


def ssm_delta(st: MutPointer[Float32, _], conv: Pointer[Float32, _], gb: Pointer[Float32, _], o: MutPointer[Float32, _], nv: Int, nk: Int, s: Int):
    for h in range(nv):
        var q = conv.unsafe_offset((h % nk) * s)
        var k = conv.unsafe_offset((nk + h % nk) * s)
        var S = st.unsafe_offset(h * s * s)
        for j in range(s):
            var sk = Float32(0)
            for i in range(s):
                sk = fma(S[unsafe_offset=i * s + j] * gb[unsafe_offset=h], k[unsafe_offset=i], sk)
            var d = (conv[unsafe_offset=(2 * nk + h) * s + j] - sk) * gb[unsafe_offset=nv + h]
            var acc = Float32(0)
            for i in range(s):
                var v = fma(k[unsafe_offset=i], d, S[unsafe_offset=i * s + j] * gb[unsafe_offset=h])
                S[unsafe_offset=i * s + j] = v
                acc = fma(v, q[unsafe_offset=i], acc)
            o[unsafe_offset=h * s + j] = acc


def ssm_gated(o: MutPointer[Float32, _], z: Pointer[Float32, _], nw: Pointer[Float32, _], dst: MutPointer[BFloat16, _], n: Int, s: Int, eps: Float32):
    """Gated rmsnorm per s-wide head; normalizes o in place."""
    for h in range(n // s):
        head_rmsnorm(o.unsafe_offset(h * s), s, nw, eps)
    for i in range(n):
        var zi = z[unsafe_offset=i]
        dst[unsafe_offset=i] = (o[unsafe_offset=i] * (zi / (1 + exp(-zi)))).cast[DType.bfloat16]()
