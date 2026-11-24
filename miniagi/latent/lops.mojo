from std.math import ceildiv, exp, log, sqrt, rsqrt, tanh
from max.gpu import block_idx, thread_idx, global_idx, MAX_THREADS_PER_BLOCK_METADATA
from std.utils import StaticTuple
from max.gpu.primitives import warp
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import row_major, stack_allocation
from layout.tensor_core import mma
from std.sys import get_defined_int, llvm_intrinsic
from std.memory import bitcast
from std.sys.info import _is_amd_rdna4
from ops import F, I, THREADS, grid


@always_inline
def _acc_row(base: Int, i: Int, half: Int) -> Int:
    comptime if _is_amd_rdna4():
        return base + i + 8 * half
    else:
        return base + 2 * i + half

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
                var row = _acc_row(r0 + (wm * TM + tm) * 16, i, half)
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


comptime EC = get_defined_int["LAT_EC", 256]()  # embed_bwd_k rows per chunk; large V: 2048 keeps the partials (R / EC x V x D) small


def embed_bwd_k(idx: I, dy: F, part: F, n32: Int32, nv32: Int32):
    """Embedding grad partials part[c][v][d] over row chunk c; block_idx.z owns vocab ids [32 z, 32 z + 32)
    (each block scans the chunk's ids in row order: deterministic, same sums as one 32-id block)."""
    var d = Int(block_idx.y) * 256 + Int(thread_idx.x)
    var c = Int(block_idx.x)
    var v0 = Int(block_idx.z) * 32
    var nv = Int(nv32)
    var acc = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[32, 256]())
    var dl = Int(thread_idx.x)
    for v in range(32):
        acc[v, dl] = 0
    for t in range(c * EC, min(Int(n32), (c + 1) * EC)):
        var v = Int(idx[unsafe_offset=t]) - v0
        if v >= 0 and v < 32:
            acc[v, dl] = acc[v, dl] + rebind[acc.ElementType](dy[unsafe_offset=t * DM + d])
    for v in range(min(32, nv - v0)):
        part[unsafe_offset=c * nv * DM + (v0 + v) * DM + d] = rebind[Float32](acc[v, dl])


def rope_k[inv: Bool](x: F, cs: F, sn: F, n32: Int32, b32: Int32, r032: Int32, h32: Int32, hd32: Int32):
    comptime hd = AHD  # comptime: the 64-long inner loops unroll, idx // hd becomes a shift
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
comptime OPT = get_defined_int["LAT_OPT", 3670015]()  # 8192: async double-buffered batch upload (run.mojo), 16384 rope folded into WMMA attention loads, 32768 rms_dxg_k, 65536 split-K wo dwg, 131072 rope3_k q/k/v grads, 262144 bf16 w1/w3/dff GEMM outputs, 1048576 bf16 attention output + bf16 dW operands + w16-sourced refresh, 2097152 rope3 folded into attn_bwd  # 1 cast/transpose dedupe, 2 bf16 rms + fused residual, 4 attn bwd q in LDS, 8 GEMM acc epilogue, 16 no dk/dv memset, 32 WMMA attention, 64 bf16 swiglu, 128 fused qkv GEMMs, 256 bf16-only swiglu outputs + bf16-source dW transposes, 512 adamw writes the bf16 weight copy, 1024 per-block grad-norm partials, 2048 rms keeps n1/n2 in bf16
comptime QLDS = (OPT & 4) != 0
comptime AHD = DM // get_defined_int["LAT_H", 4]()
comptime AW = (OPT & 32) != 0
comptime SW16 = (OPT & 64) != 0
comptime FQ = (OPT & 128) != 0  # q/k/v as one GEMM each way (fwd N=3D, dW N=3D scatter, dX K=3D)  # swiglu kernels write the bf16 GEMM operands; bwd runs dX before dW  # WMMA attention (bf16 tiles); exact mode keeps the fp32 kernels


def attn_fwd_k(q: F, k: F, v: F, p: F, o: F, b32: Int32, h32: Int32, hd32: Int32, r032: Int32, r132: Int32, scale: Float32):
    var b = Int(b32)
    var h = Int(h32)
    comptime hd = AHD  # comptime: the 64-long inner loops unroll, idx // hd becomes a shift
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


def attn_bwd_k(q: F, k: F, v: F, p: F, do: F, dq: F, dk: F, dv: F, b32: Int32, h32: Int32, hd32: Int32, r032: Int32, r132: Int32, scale: Float32, init: Int32):
    """init != 0: this is the first backward segment and covers rows [0, r1): write dk/dv, no prior memset."""
    var b = Int(b32)
    var h = Int(h32)
    comptime hd = AHD  # comptime: the 64-long inner loops unroll, idx // hd becomes a shift
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
    comptime if QLDS:
        # vs is dead after ds: hold q there so the dk loop reads LDS, not global memory nq times per element
        idx = tid
        while idx < nq * hd:
            var j = idx // hd
            var e = idx % hd
            vs[j, e] = rebind[vs.ElementType](q[unsafe_offset=(r0 + j) * ld + base + e])
            idx += AT
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
            comptime if QLDS:
                ak += rebind[Float32](ds[ii, j]) * rebind[Float32](vs[ii, e])
            else:
                ak += rebind[Float32](ds[ii, j]) * q[unsafe_offset=(r0 + ii) * ld + base + e]
            av += rebind[Float32](ps[ii, j]) * rebind[Float32](dos[ii, e])
        var o = j * ld + base + e
        if init != 0:
            dk[unsafe_offset=o] = ak
            dv[unsafe_offset=o] = av
        else:
            dk[unsafe_offset=o] = dk[unsafe_offset=o] + ak
            dv[unsafe_offset=o] = dv[unsafe_offset=o] + av
        idx += AT


comptime TQ = ceildiv(SMAX, 16)  # WMMA attention: 16-row tiles over the sequence
comptime SQ = TQ * 16
comptime KT = AHD // 16
comptime HW = AHD + 2  # bf16 row strides: odd word count, so 16 lanes reading 16 rows hit 16 banks
comptime PW = SQ + 2


@always_inline
def _z16() -> Scalar[bf]:
    return Float32(0).cast[bf]()


comptime OB16 = (OPT & 1048576) != 0 and DM * DM >= (1 << 20)  # attention output o stored bf16 (in o's bytes, bf16 index = fp32 index)
comptime AR3 = (OPT & 2097152) != 0  # attn_bwd_w_k writes the final rope'd bf16 q/k/v-grad rows (rope3_k folded in)
comptime RF = (OPT & 16384) != 0  # WMMA attention applies rope to q/k as it loads them (no rope_k launches forward)


@always_inline
def _rope_ld(x: F, off: Int, e: Int, pos: Int, cs: F, sn: F) -> Float32:
    """x[off] after rope at position pos, off = row base + e: the rope_k arithmetic on the fly."""
    comptime half = AHD // 2
    var i = e % half
    var o = off - e + i
    var x1 = x[unsafe_offset=o]
    var x2 = x[unsafe_offset=o + half]
    var c = cs[unsafe_offset=pos * half + i]
    var s = sn[unsafe_offset=pos * half + i]
    if e < half:
        return x1 * c - x2 * s
    return x2 * c + x1 * s


def attn_fwd_w_k(q: F, k: F, v: F, p: F, o: F, b32: Int32, h32: Int32, r032: Int32, r132: Int32, scale: Float32, cs: F, sn: F):
    """attn_fwd_k on WMMA: S = Q K^T and O = P V as 16x16 bf16 tiles (fp32 accumulate), one wave per tile; softmax in fp32."""
    var b = Int(b32)
    var h = Int(h32)
    var r0 = Int(r032)
    var r1 = Int(r132)
    var nq = r1 - r0
    var blk = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var wid = tid // 32
    var hl = lane % 16
    var half = lane // 16
    comptime hd = AHD
    var ld = b * h * hd
    var base = (blk // h) * h * hd + (blk % h) * hd
    var qs = stack_allocation[bf, address_space=AddressSpace.SHARED](row_major[SQ, HW]())
    var ks = stack_allocation[bf, address_space=AddressSpace.SHARED](row_major[SQ, HW]())
    var vs = stack_allocation[bf, address_space=AddressSpace.SHARED](row_major[SQ, HW]())
    var ps = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[SQ, SQ + 1]())
    var p16 = stack_allocation[bf, address_space=AddressSpace.SHARED](row_major[SQ, PW]())
    var idx = tid
    while idx < SQ * hd:
        var j = idx // hd
        var e = idx % hd
        var kv: Float32 = 0
        var vv: Float32 = 0
        var qv: Float32 = 0
        if j < r1:
            comptime if RF:
                kv = _rope_ld(k, j * ld + base + e, e, j, cs, sn)
            else:
                kv = k[unsafe_offset=j * ld + base + e]
            vv = v[unsafe_offset=j * ld + base + e]
        if j < nq:
            comptime if RF:
                qv = _rope_ld(q, (r0 + j) * ld + base + e, e, r0 + j, cs, sn)
            else:
                qv = q[unsafe_offset=(r0 + j) * ld + base + e]
        ks[j, e] = rebind[ks.ElementType](kv.cast[bf]())
        vs[j, e] = rebind[vs.ElementType](vv.cast[bf]())
        qs[j, e] = rebind[qs.ElementType](qv.cast[bf]())
        idx += AT
    idx = tid
    while idx < SQ * SQ:
        p16[idx // SQ, idx % SQ] = rebind[p16.ElementType](_z16())
        idx += AT
    barrier()
    var t = wid
    while t < TQ * TQ:
        var ti = t // TQ
        var tj = t % TQ
        var acc = SIMD[DType.float32, 8](0)
        comptime for kk in range(KT):
            var av = SIMD[bf, 16](0)
            var bv = SIMD[bf, 16](0)
            comptime for i in range(16):
                av[i] = rebind[Scalar[bf]](qs[ti * 16 + hl, kk * 16 + i])
                bv[i] = rebind[Scalar[bf]](ks[tj * 16 + hl, kk * 16 + i])
            var dv = SIMD[DType.float32, 8](0)
            mma(dv, av, bv, acc)
            acc = dv
        comptime for i in range(8):
            var row = _acc_row(ti * 16, i, half)
            var col = tj * 16 + hl
            var s: Float32 = acc[i] * scale if col <= r0 + row else -3.0e38
            ps[row, col] = rebind[ps.ElementType](s)
        t += NW
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
            p16[i, lane] = rebind[p16.ElementType]((e0 * iz).cast[bf]())
            p[unsafe_offset=po + lane] = e0 * iz
        if lane + 32 < r1:
            p16[i, lane + 32] = rebind[p16.ElementType]((e1 * iz).cast[bf]())
            p[unsafe_offset=po + lane + 32] = e1 * iz
        i += NW
    barrier()
    t = wid
    while t < TQ * KT:
        var ti = t // KT
        var te = t % KT
        var acc = SIMD[DType.float32, 8](0)
        comptime for kk in range(TQ):
            var av = SIMD[bf, 16](0)
            var bv = SIMD[bf, 16](0)
            comptime for i2 in range(16):
                av[i2] = rebind[Scalar[bf]](p16[ti * 16 + hl, kk * 16 + i2])
                bv[i2] = rebind[Scalar[bf]](vs[kk * 16 + i2, te * 16 + hl])
            var dv = SIMD[DType.float32, 8](0)
            mma(dv, av, bv, acc)
            acc = dv
        comptime for i2 in range(8):
            var row = _acc_row(ti * 16, i2, half)
            if row < nq:
                comptime if OB16:  # o's only readers take bf16 (wo GEMM, wo dW)
                    H16(unsafe_from_address=Int(o))[unsafe_offset=(r0 + row) * ld + base + te * 16 + hl] = acc[i2].cast[bf]()
                else:
                    o[unsafe_offset=(r0 + row) * ld + base + te * 16 + hl] = acc[i2]
        t += NW


def attn_bwd_w_k[ar3: Bool](q: F, k: F, v: F, p: F, do: F, dq: F, dk: F, dv: F, b32: Int32, h32: Int32, r032: Int32, r132: Int32,
                 scale: Float32, init: Int32, cs: F, sn: F, y16: H16):
    """attn_bwd_k on WMMA: dP = dO V^T, dQ = dS K, dK = dS^T Q, dV = P^T dO as 16x16 bf16 tiles; dS in fp32."""
    var b = Int(b32)
    var h = Int(h32)
    var r0 = Int(r032)
    var r1 = Int(r132)
    var nq = r1 - r0
    var blk = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var wid = tid // 32
    var hl = lane % 16
    var half = lane // 16
    comptime hd = AHD
    var ld = b * h * hd
    var base = (blk // h) * h * hd + (blk % h) * hd
    var qs = stack_allocation[bf, address_space=AddressSpace.SHARED](row_major[SQ, HW]())
    var ks = stack_allocation[bf, address_space=AddressSpace.SHARED](row_major[SQ, HW]())
    var vs = stack_allocation[bf, address_space=AddressSpace.SHARED](row_major[SQ, HW]())
    var dos = stack_allocation[bf, address_space=AddressSpace.SHARED](row_major[SQ, HW]())
    var ds = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[SQ, SQ + 1]())
    var p16 = stack_allocation[bf, address_space=AddressSpace.SHARED](row_major[SQ, PW]())
    var d16 = stack_allocation[bf, address_space=AddressSpace.SHARED](row_major[SQ, PW]())
    var idx = tid
    while idx < SQ * hd:
        var j = idx // hd
        var e = idx % hd
        var kv: Float32 = 0
        var vv: Float32 = 0
        var qv: Float32 = 0
        var dov: Float32 = 0
        if j < r1:
            comptime if RF:
                kv = _rope_ld(k, j * ld + base + e, e, j, cs, sn)
            else:
                kv = k[unsafe_offset=j * ld + base + e]
            vv = v[unsafe_offset=j * ld + base + e]
        if j < nq:
            comptime if RF:
                qv = _rope_ld(q, (r0 + j) * ld + base + e, e, r0 + j, cs, sn)
            else:
                qv = q[unsafe_offset=(r0 + j) * ld + base + e]
            dov = do[unsafe_offset=(r0 + j) * ld + base + e]
        ks[j, e] = rebind[ks.ElementType](kv.cast[bf]())
        vs[j, e] = rebind[vs.ElementType](vv.cast[bf]())
        qs[j, e] = rebind[qs.ElementType](qv.cast[bf]())
        dos[j, e] = rebind[dos.ElementType](dov.cast[bf]())
        idx += AT
    idx = tid
    while idx < SQ * SQ:
        p16[idx // SQ, idx % SQ] = rebind[p16.ElementType](_z16())
        d16[idx // SQ, idx % SQ] = rebind[d16.ElementType](_z16())
        idx += AT
    barrier()
    var t = wid
    while t < TQ * TQ:
        var ti = t // TQ
        var tj = t % TQ
        var acc = SIMD[DType.float32, 8](0)
        comptime for kk in range(KT):
            var av = SIMD[bf, 16](0)
            var bv = SIMD[bf, 16](0)
            comptime for i in range(16):
                av[i] = rebind[Scalar[bf]](dos[ti * 16 + hl, kk * 16 + i])
                bv[i] = rebind[Scalar[bf]](vs[tj * 16 + hl, kk * 16 + i])
            var dd = SIMD[DType.float32, 8](0)
            mma(dd, av, bv, acc)
            acc = dd
        comptime for i in range(8):
            ds[_acc_row(ti * 16, i, half), tj * 16 + hl] = rebind[ds.ElementType](acc[i])
        t += NW
    barrier()
    var i = wid
    while i < nq:
        var po = (blk * SMAX + r0 + i) * SMAX
        var p0: Float32 = 0
        var p1: Float32 = 0
        var d0: Float32 = 0
        var d1: Float32 = 0
        if lane < r1 and lane <= r0 + i:
            p0 = p[unsafe_offset=po + lane]
            d0 = rebind[Float32](ds[i, lane])
        if lane + 32 < r1 and lane + 32 <= r0 + i:
            p1 = p[unsafe_offset=po + lane + 32]
            d1 = rebind[Float32](ds[i, lane + 32])
        var dsum = warp.sum(p0 * d0 + p1 * d1)
        if lane < r1:
            p16[i, lane] = rebind[p16.ElementType](p0.cast[bf]())
            d16[i, lane] = rebind[d16.ElementType]((p0 * (d0 - dsum) * scale).cast[bf]())
        if lane + 32 < r1:
            p16[i, lane + 32] = rebind[p16.ElementType](p1.cast[bf]())
            d16[i, lane + 32] = rebind[d16.ElementType]((p1 * (d1 - dsum) * scale).cast[bf]())
        i += NW
    barrier()
    comptime if ar3:  # the caller passes AR3 and R3: it reads y16 only then
        # one wave per (row tile ti, column tiles te and te + KT / 2): the inverse-rope partners e and e + HD / 2 sit
        # in the same lane of the two accumulators, so rows [r0, r1) (final after this segment: later segments ran
        # first) are rotated here and written as the interleaved bf16 [n, 3D] q/k/v-grad rows (rope3_k's output).
        comptime HK = KT // 2
        comptime half_d = AHD // 2
        var bi = blk // h
        var hh = blk % h
        t = wid
        while t < TQ * HK:
            var ti = t // HK
            var tp = t % HK
            var aq = Array[SIMD[DType.float32, 8], 2](fill=SIMD[DType.float32, 8](0))
            var ak = Array[SIMD[DType.float32, 8], 2](fill=SIMD[DType.float32, 8](0))
            var avv = Array[SIMD[DType.float32, 8], 2](fill=SIMD[DType.float32, 8](0))
            comptime for kk in range(TQ):
                var a1 = SIMD[bf, 16](0)
                var a2 = SIMD[bf, 16](0)
                var a3 = SIMD[bf, 16](0)
                comptime for i2 in range(16):
                    a1[i2] = rebind[Scalar[bf]](d16[ti * 16 + hl, kk * 16 + i2])
                    a2[i2] = rebind[Scalar[bf]](d16[kk * 16 + i2, ti * 16 + hl])
                    a3[i2] = rebind[Scalar[bf]](p16[kk * 16 + i2, ti * 16 + hl])
                comptime for u in range(2):
                    var te = tp + u * HK
                    var b1 = SIMD[bf, 16](0)
                    var b2 = SIMD[bf, 16](0)
                    var b3 = SIMD[bf, 16](0)
                    comptime for i2 in range(16):
                        b1[i2] = rebind[Scalar[bf]](ks[kk * 16 + i2, te * 16 + hl])
                        b2[i2] = rebind[Scalar[bf]](qs[kk * 16 + i2, te * 16 + hl])
                        b3[i2] = rebind[Scalar[bf]](dos[kk * 16 + i2, te * 16 + hl])
                    mma(aq[u], a1, b1, aq[u])
                    mma(ak[u], a2, b2, ak[u])
                    mma(avv[u], a3, b3, avv[u])
            comptime for i2 in range(8):
                var row = _acc_row(ti * 16, i2, half)
                var e0 = tp * 16 + hl  # first half of the head dim; partner e0 + half_d
                if row < r1:
                    var o0 = row * ld + base + e0
                    var k0 = ak[0][i2]
                    var k1 = ak[1][i2]
                    var v0 = avv[0][i2]
                    var v1 = avv[1][i2]
                    if init == 0:
                        k0 += dk[unsafe_offset=o0]
                        k1 += dk[unsafe_offset=o0 + half_d]
                        v0 += dv[unsafe_offset=o0]
                        v1 += dv[unsafe_offset=o0 + half_d]
                    if row < r0:  # earlier segment's rows: keep accumulating in fp32
                        dk[unsafe_offset=o0] = k0
                        dk[unsafe_offset=o0 + half_d] = k1
                        dv[unsafe_offset=o0] = v0
                        dv[unsafe_offset=o0 + half_d] = v1
                    else:  # final: inverse rope (rope_k[True] arithmetic), bf16, interleaved [n, 3D]
                        var ci = (row - r0) * DM * 3 * b + (bi * 3 * DM) + hh * AHD + e0
                        var co = cs[unsafe_offset=row * half_d + e0]
                        var si = -sn[unsafe_offset=row * half_d + e0]
                        y16[unsafe_offset=ci + DM] = (k0 * co - k1 * si).cast[bf]()
                        y16[unsafe_offset=ci + DM + half_d] = (k1 * co + k0 * si).cast[bf]()
                        y16[unsafe_offset=ci + 2 * DM] = v0.cast[bf]()
                        y16[unsafe_offset=ci + 2 * DM + half_d] = v1.cast[bf]()
                if row < nq:
                    var q0 = aq[0][i2]
                    var q1 = aq[1][i2]
                    var pos = r0 + row
                    var cq = row * DM * 3 * b + (bi * 3 * DM) + hh * AHD + e0
                    var co = cs[unsafe_offset=pos * half_d + e0]
                    var si = -sn[unsafe_offset=pos * half_d + e0]
                    y16[unsafe_offset=cq] = (q0 * co - q1 * si).cast[bf]()
                    y16[unsafe_offset=cq + half_d] = (q1 * co + q0 * si).cast[bf]()
            t += NW
        return
    barrier()
    t = wid
    while t < TQ * KT:
        var ti = t // KT
        var te = t % KT
        var aq = SIMD[DType.float32, 8](0)
        var ak = SIMD[DType.float32, 8](0)
        var avv = SIMD[DType.float32, 8](0)
        comptime for kk in range(TQ):
            var a1 = SIMD[bf, 16](0)
            var b1 = SIMD[bf, 16](0)
            var a2 = SIMD[bf, 16](0)
            var a3 = SIMD[bf, 16](0)
            var b2 = SIMD[bf, 16](0)
            var b3 = SIMD[bf, 16](0)
            comptime for i2 in range(16):
                a1[i2] = rebind[Scalar[bf]](d16[ti * 16 + hl, kk * 16 + i2])
                b1[i2] = rebind[Scalar[bf]](ks[kk * 16 + i2, te * 16 + hl])
                a2[i2] = rebind[Scalar[bf]](d16[kk * 16 + i2, ti * 16 + hl])
                b2[i2] = rebind[Scalar[bf]](qs[kk * 16 + i2, te * 16 + hl])
                a3[i2] = rebind[Scalar[bf]](p16[kk * 16 + i2, ti * 16 + hl])
                b3[i2] = rebind[Scalar[bf]](dos[kk * 16 + i2, te * 16 + hl])
            var d1v = SIMD[DType.float32, 8](0)
            var d2v = SIMD[DType.float32, 8](0)
            var d3v = SIMD[DType.float32, 8](0)
            mma(d1v, a1, b1, aq)
            mma(d2v, a2, b2, ak)
            mma(d3v, a3, b3, avv)
            aq = d1v
            ak = d2v
            avv = d3v
        comptime for i2 in range(8):
            var row = _acc_row(ti * 16, i2, half)
            var col = te * 16 + hl
            if row < nq:
                dq[unsafe_offset=(r0 + row) * ld + base + col] = aq[i2]
            if row < r1:
                var o = row * ld + base + col
                if init != 0:
                    dk[unsafe_offset=o] = ak[i2]
                    dv[unsafe_offset=o] = avv[i2]
                else:
                    dk[unsafe_offset=o] = dk[unsafe_offset=o] + ak[i2]
                    dv[unsafe_offset=o] = dv[unsafe_offset=o] + avv[i2]
        t += NW


# Flash attention (causal, one sequence segment, any S): K/V streamed through LDS in FT-row tiles with an online
# softmax; forward keeps lse = max + log(sum) per query row instead of P. Backward is FA2 split in two kernels
# so nothing is accumulated across blocks (bit-identical reruns): flash_dkv_k (one block per key tile, dK/dV in
# registers over query tiles) and flash_dq_k (one block per query tile). Both recompute P from lse.
# Layout as attn_*_w_k: element (pos j, sequence s, head h, e) at j * b * D + (s * H + h) * HD + e; rope on load.
comptime FT = 64  # rows per flash tile: 4 waves x 16
comptime FAT = 128
comptime FW = AHD + 8  # bf16 LDS row stride: 36 words, 16 rows on 16 distinct banks
comptime FLASH = (OPT & 4194304) != 0


@always_inline
def _frag(x: H16, row: Int, ld: Int, base: Int, e0: Int, ok: Bool) -> SIMD[bf, 16]:
    """A operand straight from global bf16: row `row`, head columns e0..e0+15 (one 32 B load)."""
    if not ok:
        return SIMD[bf, 16](0)
    return rebind[SIMD[bf, 16]](x.unsafe_offset(row * ld + base + e0).load[width=16]())


comptime FTW = FT + 8  # transposed tile row stride (rows = head columns)


@always_inline
def _ld_tile(x: H16, t: Shared16, r0: Int, S: Int, ld: Int, base: Int):
    """FT rows from r0 of a bf16 [n, D] tensor into an [FT, FW] LDS tile, 8 values (16 B) per thread per pass;
    rows past S zero, all FAT threads."""
    var idx = Int(thread_idx.x) * 8
    while idx < FT * AHD:
        var j = idx // AHD
        var e = idx % AHD
        var r = r0 + j
        var val = SIMD[bf, 8](0)
        if r < S:
            val = rebind[SIMD[bf, 8]](x.unsafe_offset(r * ld + base + e).load[width=8]())
        t.unsafe_offset(j * FW + e).store(val)
        idx += FAT * 8


@always_inline
def _tr_lds(src: Shared16, dst: Shared16):
    """[FT, FW] LDS tile -> its transpose [HD, FTW] (no second global read); caller barriers before and after."""
    var idx = Int(thread_idx.x)
    while idx < FT * AHD:
        var j = idx % FT
        var e = idx // FT
        dst[e * FTW + j] = src[j * FW + e]
        idx += FAT


@always_inline
def _hsum(x: Float32) -> Float32:
    """Sum over the 16 lanes of a half wave (WMMA accumulator rows live in one half)."""
    var y = x + warp.shuffle_xor(x, 1)
    y += warp.shuffle_xor(y, 2)
    y += warp.shuffle_xor(y, 4)
    return y + warp.shuffle_xor(y, 8)


@always_inline
def _hmax(x: Float32) -> Float32:
    var y = max(x, warp.shuffle_xor(x, 1))
    y = max(y, warp.shuffle_xor(y, 2))
    y = max(y, warp.shuffle_xor(y, 4))
    return max(y, warp.shuffle_xor(y, 8))


comptime Shared16 = MutPointer[Scalar[bf], MutUntrackedOrigin, address_space=AddressSpace.SHARED]


@always_inline
def _shared16[n: Int]() -> Shared16:
    return rebind[Shared16](stack_allocation[bf, address_space=AddressSpace.SHARED](row_major[n]()).ptr)


def flash_fwd_k(q: H16, k: H16, v: H16, o: F, lse: F, b32: Int32, h32: Int32, s32: Int32, scale: Float32, cs: F, sn: F):
    """grid (b * H, ceil(S / FT)), FAT threads; block y counts down so the longest (last) query tiles start first."""
    var b = Int(b32)
    var S = Int(s32)
    var blk = Int(block_idx.x)
    var qt = ceildiv(S, FT) - 1 - Int(block_idx.y)
    var lane = Int(thread_idx.x) % 32
    var wid = Int(thread_idx.x) // 32
    var hl = lane % 16
    var half = lane // 16
    var ld = b * Int(h32) * AHD
    var base = blk * AHD
    var ks = _shared16[FT * FW]()
    var vs = _shared16[AHD * FTW]()
    var vn = _shared16[FT * FW]()
    var ps = _shared16[FT * (FT + 8)]()
    var q0 = qt * FT + wid * 16
    var qa = Array[SIMD[bf, 16], KT](fill=SIMD[bf, 16](0))
    comptime for kk in range(KT):
        qa[kk] = _frag(q, q0 + hl, ld, base, kk * 16, q0 + hl < S)
    var m = SIMD[DType.float32, 8](-3.0e38)
    var l = SIMD[DType.float32, 8](0)
    var acc = Array[SIMD[DType.float32, 8], KT](fill=SIMD[DType.float32, 8](0))
    for kt in range(qt + 1):
        barrier()
        _ld_tile(k, ks, kt * FT, S, ld, base)
        _ld_tile(v, vn, kt * FT, S, ld, base)
        barrier()
        _tr_lds(vn, vs)
        barrier()
        var s = Array[SIMD[DType.float32, 8], 4](fill=SIMD[DType.float32, 8](0))
        comptime for tj in range(4):
            comptime for kk in range(KT):
                var bv = SIMD[bf, 16](0)
                bv = rebind[SIMD[bf, 16]](ks.unsafe_offset((tj * 16 + hl) * FW + kk * 16).load[width=16, alignment=16]())
                mma(s[tj], qa[kk], bv, s[tj])
        var mn = m
        comptime for i in range(8):
            var row = q0 + _acc_row(0, i, half)
            var mx: Float32 = -3.0e38
            comptime for tj in range(4):
                var col = kt * FT + tj * 16 + hl
                var x: Float32 = s[tj][i] * scale if col <= row and col < S else -3.0e38
                s[tj][i] = x
                mx = max(mx, x)
            mn[i] = max(m[i], _hmax(mx))
        var alpha = exp(m - mn)
        var ls = SIMD[DType.float32, 8](0)
        comptime for i in range(8):
            var r = wid * 16 + _acc_row(0, i, half)
            var t: Float32 = 0
            comptime for tj in range(4):
                var pv = exp(s[tj][i] - mn[i])
                t += pv
                ps[r * (FT + 8) + tj * 16 + hl] = pv.cast[bf]()
            ls[i] = _hsum(t)
        l = l * alpha + ls
        m = mn
        comptime for te in range(KT):
            acc[te] = acc[te] * alpha
        barrier()
        comptime for te in range(KT):
            comptime for kk in range(4):
                var av = SIMD[bf, 16](0)
                var bv = SIMD[bf, 16](0)
                av = rebind[SIMD[bf, 16]](ps.unsafe_offset((wid * 16 + hl) * (FT + 8) + kk * 16).load[width=16, alignment=16]())
                bv = rebind[SIMD[bf, 16]](vs.unsafe_offset((te * 16 + hl) * FTW + kk * 16).load[width=16, alignment=16]())
                mma(acc[te], av, bv, acc[te])
    comptime for i in range(8):
        var row = q0 + _acc_row(0, i, half)
        if row < S:
            var il = 1 / l[i]
            comptime for te in range(KT):
                var off = row * ld + base + te * 16 + hl
                comptime if OB16:
                    H16(unsafe_from_address=Int(o))[unsafe_offset=off] = (acc[te][i] * il).cast[bf]()
                else:
                    o[unsafe_offset=off] = acc[te][i] * il
            if hl == 0:
                lse[unsafe_offset=blk * SMAX + row] = m[i] + log(l[i])


comptime FQR = 128  # flash_fwd8_k: query rows per block (8 waves x 16), sharing each FT-key K/V tile
comptime F8 = 256
comptime FX = get_defined_int["LAT_FX", 1]()  # 1: a wave skips a tile that is masked for all its rows (same bytes, flash_fwd8_k -6 to -9%); 0: the A/B base; 5: the gate's mutant


@always_inline
def _kv_regs(x: H16, r0: Int, S: Int, ld: Int, base: Int) -> Array[SIMD[bf, 8], 2]:
    """This thread's 2 x 8 values of the FT-row tile at r0 (rows past S zero): 16 B global loads, held in registers
    while the previous tile computes."""
    var out = Array[SIMD[bf, 8], 2](fill=SIMD[bf, 8](0))
    comptime for p in range(2):
        var idx = (Int(thread_idx.x) + p * F8) * 8
        var r = r0 + idx // AHD
        if r < S:
            out[p] = rebind[SIMD[bf, 8]](x.unsafe_offset(r * ld + base + idx % AHD).load[width=8]())
    return out^


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(F8)))
def flash_fwd8_k(q: H16, k: H16, v: H16, o: F, lse: F, b32: Int32, h32: Int32, s32: Int32, scale: Float32, cs: F, sn: F):
    """flash_fwd_k with 8 waves (FQR query rows) per block: grid (b * H, ceil(S / FQR)), F8 threads, longest tiles first.
    K/V tiles prefetched into registers during the previous tile; V transposed as it is stored; P rows wave-private."""
    var b = Int(b32)
    var S = Int(s32)
    var blk = Int(block_idx.x)
    var qt = ceildiv(S, FQR) - 1 - Int(block_idx.y)
    var lane = Int(thread_idx.x) % 32
    var wid = Int(thread_idx.x) // 32
    var hl = lane % 16
    var half = lane // 16
    var ld = b * Int(h32) * AHD
    var base = blk * AHD
    var ks = _shared16[FT * FW]()
    var vs = _shared16[AHD * FTW]()
    var ps = _shared16[FQR * (FT + 8)]()
    var q0 = qt * FQR + wid * 16
    var qa = Array[SIMD[bf, 16], KT](fill=SIMD[bf, 16](0))
    comptime for kk in range(KT):
        qa[kk] = _frag(q, q0 + hl, ld, base, kk * 16, q0 + hl < S)
    var m = SIMD[DType.float32, 8](-3.0e38)
    var l = SIMD[DType.float32, 8](0)
    var acc = Array[SIMD[DType.float32, 8], KT](fill=SIMD[DType.float32, 8](0))
    var nk = min(ceildiv(S, FT), (qt * FQR + FQR - 1) // FT + 1)
    var kr = _kv_regs(k, 0, S, ld, base)
    var vr = _kv_regs(v, 0, S, ld, base)
    for kt in range(nk):
        barrier()
        comptime for p in range(2):
            var idx = (Int(thread_idx.x) + p * F8) * 8
            var j = idx // AHD
            var e = idx % AHD
            ks.unsafe_offset(j * FW + e).store(kr[p])
            comptime for u in range(8):
                vs[(e + u) * FTW + j] = vr[p][u]
        barrier()
        if kt + 1 < nk:
            kr = _kv_regs(k, (kt + 1) * FT, S, ld, base)
            vr = _kv_regs(v, (kt + 1) * FT, S, ld, base)
        comptime if (FX & 1) != 0:
            comptime MUT = FT if (FX & 4) != 0 else 0  # 4: the gate's mutant, also skips the last tile a wave needs (never tile 0)
            if kt * FT > q0 + 15 - MUT and (MUT == 0 or kt > 0):  # every key of the tile is after every row of this wave
                continue
        var s = Array[SIMD[DType.float32, 8], 4](fill=SIMD[DType.float32, 8](0))
        comptime for tj in range(4):
            comptime for kk in range(KT):
                var bv = rebind[SIMD[bf, 16]](ks.unsafe_offset((tj * 16 + hl) * FW + kk * 16).load[width=16, alignment=16]())
                mma(s[tj], qa[kk], bv, s[tj])
        var mn = m
        comptime for i in range(8):
            var row = q0 + _acc_row(0, i, half)
            var mx: Float32 = -3.0e38
            comptime for tj in range(4):
                var col = kt * FT + tj * 16 + hl
                var x: Float32 = s[tj][i] * scale if col <= row and col < S else -3.0e38
                s[tj][i] = x
                mx = max(mx, x)
            mn[i] = max(m[i], _hmax(mx))
        var alpha = exp(m - mn)
        var ls = SIMD[DType.float32, 8](0)
        comptime for i in range(8):
            var r = wid * 16 + _acc_row(0, i, half)
            var t: Float32 = 0
            comptime for tj in range(4):
                var pv = exp(s[tj][i] - mn[i])
                t += pv
                ps[r * (FT + 8) + tj * 16 + hl] = pv.cast[bf]()
            ls[i] = _hsum(t)
        l = l * alpha + ls
        m = mn
        comptime for te in range(KT):
            acc[te] = acc[te] * alpha
        comptime for te in range(KT):
            comptime for kk in range(4):
                var av = rebind[SIMD[bf, 16]](ps.unsafe_offset((wid * 16 + hl) * (FT + 8) + kk * 16).load[width=16, alignment=16]())
                var bv = rebind[SIMD[bf, 16]](vs.unsafe_offset((te * 16 + hl) * FTW + kk * 16).load[width=16, alignment=16]())
                mma(acc[te], av, bv, acc[te])
    comptime for i in range(8):
        var row = q0 + _acc_row(0, i, half)
        if row < S:
            var il = 1 / l[i]
            comptime for te in range(KT):
                var off = row * ld + base + te * 16 + hl
                comptime if OB16:
                    H16(unsafe_from_address=Int(o))[unsafe_offset=off] = (acc[te][i] * il).cast[bf]()
                else:
                    o[unsafe_offset=off] = acc[te][i] * il
            if hl == 0:
                lse[unsafe_offset=blk * SMAX + row] = m[i] + log(l[i])


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(F8)))
def flash_dq8_k(q: H16, k: H16, v: H16, do: H16, lse: F, dl: F, y16: H16, b32: Int32, h32: Int32, s32: Int32, scale: Float32, cs: F, sn: F):
    """flash_dq_k with 8 waves (FQR query rows) per block: grid (b * H, ceil(S / FQR)), F8 threads; K/V prefetched
    into registers, K stored both ways (ks rows, kst transposed), dS rows wave-private (no inner barriers)."""
    var b = Int(b32)
    var h = Int(h32)
    var S = Int(s32)
    var blk = Int(block_idx.x)
    var qt = ceildiv(S, FQR) - 1 - Int(block_idx.y)
    var lane = Int(thread_idx.x) % 32
    var wid = Int(thread_idx.x) // 32
    var hl = lane % 16
    var half = lane // 16
    var ld = b * h * AHD
    var base = blk * AHD
    var ks = _shared16[FT * FW]()
    var vs = _shared16[FT * FW]()
    var kst = _shared16[AHD * FTW]()
    var dsw = _shared16[FQR * 24]()
    var q0 = qt * FQR + wid * 16
    var qa = Array[SIMD[bf, 16], KT](fill=SIMD[bf, 16](0))
    var da = Array[SIMD[bf, 16], KT](fill=SIMD[bf, 16](0))
    comptime for kk in range(KT):
        qa[kk] = _frag(q, q0 + hl, ld, base, kk * 16, q0 + hl < S)
        da[kk] = _frag(do, q0 + hl, ld, base, kk * 16, q0 + hl < S)
    var lq = SIMD[DType.float32, 8](0)
    var dq_ = SIMD[DType.float32, 8](0)
    comptime for i in range(8):
        var row = q0 + _acc_row(0, i, half)
        if row < S:
            lq[i] = lse[unsafe_offset=blk * SMAX + row]
            dq_[i] = dl[unsafe_offset=blk * SMAX + row]
    var dq = Array[SIMD[DType.float32, 8], KT](fill=SIMD[DType.float32, 8](0))
    var nk = min(ceildiv(S, FT), (qt * FQR + FQR - 1) // FT + 1)
    var kr = _kv_regs(k, 0, S, ld, base)
    var vr = _kv_regs(v, 0, S, ld, base)
    for kt in range(nk):
        barrier()
        comptime for p in range(2):
            var idx = (Int(thread_idx.x) + p * F8) * 8
            var j = idx // AHD
            var e = idx % AHD
            ks.unsafe_offset(j * FW + e).store(kr[p])
            vs.unsafe_offset(j * FW + e).store(vr[p])
            comptime for u in range(8):
                kst[(e + u) * FTW + j] = kr[p][u]
        barrier()
        if kt + 1 < nk:
            kr = _kv_regs(k, (kt + 1) * FT, S, ld, base)
            vr = _kv_regs(v, (kt + 1) * FT, S, ld, base)
        comptime if (FX & 1) != 0:
            if kt * FT > q0 + 15:
                continue
        comptime for k16 in range(4):
            var s = SIMD[DType.float32, 8](0)
            var dp = SIMD[DType.float32, 8](0)
            comptime for kk in range(KT):
                var bk = SIMD[bf, 16](0)
                var bv = SIMD[bf, 16](0)
                bk = rebind[SIMD[bf, 16]](ks.unsafe_offset((k16 * 16 + hl) * FW + kk * 16).load[width=16, alignment=16]())
                bv = rebind[SIMD[bf, 16]](vs.unsafe_offset((k16 * 16 + hl) * FW + kk * 16).load[width=16, alignment=16]())
                mma(s, qa[kk], bk, s)
                mma(dp, da[kk], bv, dp)
            var key = kt * FT + k16 * 16 + hl
            comptime for i in range(8):
                var row = q0 + _acc_row(0, i, half)
                var pv: Float32 = exp(s[i] * scale - lq[i]) if key <= row and key < S and row < S else 0
                dsw[(wid * 16 + _acc_row(0, i, half)) * 24 + hl] = (pv * (dp[i] - dq_[i]) * scale).cast[bf]()
            var ad = SIMD[bf, 16](0)
            ad = rebind[SIMD[bf, 16]](dsw.unsafe_offset((wid * 16 + hl) * 24).load[width=16, alignment=16]())
            comptime for te in range(KT):
                var bk = SIMD[bf, 16](0)
                bk = rebind[SIMD[bf, 16]](kst.unsafe_offset((te * 16 + hl) * FTW + k16 * 16).load[width=16, alignment=16]())
                mma(dq[te], ad, bk, dq[te])
    _put_y16(y16, dq, q0, half, hl, S, b, blk, Int(h32), 0, True, cs, sn)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(F8)))
def flash_dkv8_k(q: H16, k: H16, v: H16, do: H16, lse: F, dl: F, y16: H16, b32: Int32, h32: Int32, s32: Int32, scale: Float32, cs: F, sn: F):
    """flash_dkv_k with 8 waves (FQR keys) per block: grid (b * H, ceil(S / FQR)), F8 threads; Q/dO tiles prefetched
    into registers and stored both ways (rows and transposed), P^T / dS^T rows wave-private (no inner barriers)."""
    var b = Int(b32)
    var h = Int(h32)
    var S = Int(s32)
    var blk = Int(block_idx.x)
    var kt = Int(block_idx.y)
    var lane = Int(thread_idx.x) % 32
    var wid = Int(thread_idx.x) // 32
    var hl = lane % 16
    var half = lane // 16
    var ld = b * h * AHD
    var base = blk * AHD
    var qs = _shared16[FT * FW]()
    var dos = _shared16[FT * FW]()
    var qst = _shared16[AHD * FTW]()
    var dost = _shared16[AHD * FTW]()
    var pt = _shared16[FQR * 24]()
    var dt = _shared16[FQR * 24]()
    var k0 = kt * FQR + wid * 16
    var ka = Array[SIMD[bf, 16], KT](fill=SIMD[bf, 16](0))
    var va = Array[SIMD[bf, 16], KT](fill=SIMD[bf, 16](0))
    comptime for kk in range(KT):
        ka[kk] = _frag(k, k0 + hl, ld, base, kk * 16, k0 + hl < S)
        va[kk] = _frag(v, k0 + hl, ld, base, kk * 16, k0 + hl < S)
    var dk = Array[SIMD[DType.float32, 8], KT](fill=SIMD[DType.float32, 8](0))
    var dv = Array[SIMD[DType.float32, 8], KT](fill=SIMD[DType.float32, 8](0))
    var q_lo = kt * FQR // FT
    var nq = ceildiv(S, FT)
    var qr = _kv_regs(q, q_lo * FT, S, ld, base)
    var dr = _kv_regs(do, q_lo * FT, S, ld, base)
    for qt in range(q_lo, nq):
        barrier()
        comptime for p in range(2):
            var idx = (Int(thread_idx.x) + p * F8) * 8
            var j = idx // AHD
            var e = idx % AHD
            qs.unsafe_offset(j * FW + e).store(qr[p])
            dos.unsafe_offset(j * FW + e).store(dr[p])
            comptime for u in range(8):
                qst[(e + u) * FTW + j] = qr[p][u]
                dost[(e + u) * FTW + j] = dr[p][u]
        barrier()
        if qt + 1 < nq:
            qr = _kv_regs(q, (qt + 1) * FT, S, ld, base)
            dr = _kv_regs(do, (qt + 1) * FT, S, ld, base)
        comptime if (FX & 1) != 0:
            if qt * FT + FT - 1 < k0:  # every row of the tile is before every key of this wave
                continue
        comptime for q16 in range(4):
            var qq = qt * FT + q16 * 16 + hl
            var lq: Float32 = lse[unsafe_offset=blk * SMAX + qq] if qq < S else 0
            var dq_: Float32 = dl[unsafe_offset=blk * SMAX + qq] if qq < S else 0
            var st = SIMD[DType.float32, 8](0)
            var dpt = SIMD[DType.float32, 8](0)
            comptime for kk in range(KT):
                var bq = SIMD[bf, 16](0)
                var bd = SIMD[bf, 16](0)
                bq = rebind[SIMD[bf, 16]](qs.unsafe_offset((q16 * 16 + hl) * FW + kk * 16).load[width=16, alignment=16]())
                bd = rebind[SIMD[bf, 16]](dos.unsafe_offset((q16 * 16 + hl) * FW + kk * 16).load[width=16, alignment=16]())
                mma(st, ka[kk], bq, st)
                mma(dpt, va[kk], bd, dpt)
            comptime for i in range(8):
                var key = k0 + _acc_row(0, i, half)
                var pv: Float32 = exp(st[i] * scale - lq) if qq >= key and qq < S and key < S else 0
                var r = wid * 16 + _acc_row(0, i, half)
                pt[r * 24 + hl] = pv.cast[bf]()
                dt[r * 24 + hl] = (pv * (dpt[i] - dq_) * scale).cast[bf]()
            var ap = SIMD[bf, 16](0)
            var ad = SIMD[bf, 16](0)
            ap = rebind[SIMD[bf, 16]](pt.unsafe_offset((wid * 16 + hl) * 24).load[width=16, alignment=16]())
            ad = rebind[SIMD[bf, 16]](dt.unsafe_offset((wid * 16 + hl) * 24).load[width=16, alignment=16]())
            comptime for te in range(KT):
                var bd = SIMD[bf, 16](0)
                var bq = SIMD[bf, 16](0)
                bd = rebind[SIMD[bf, 16]](dost.unsafe_offset((te * 16 + hl) * FTW + q16 * 16).load[width=16, alignment=16]())
                bq = rebind[SIMD[bf, 16]](qst.unsafe_offset((te * 16 + hl) * FTW + q16 * 16).load[width=16, alignment=16]())
                mma(dv[te], ap, bd, dv[te])
                mma(dk[te], ad, bq, dk[te])
    _put_y16(y16, dk, k0, half, hl, S, b, blk, h, 1, True, cs, sn)
    _put_y16(y16, dv, k0, half, hl, S, b, blk, h, 2, False, cs, sn)


comptime FY = 0 if _is_amd_rdna4() else get_defined_int["LAT_FY", 11]()  # bits 1 / 2 / 4: flash_fwd8t_k / flash_dq8t_k / flash_dkv8t_k (4 measured no gain, off); 0: the A/B base


comptime FP = (FY & 8) != 0  # a tile thread takes 2 adjacent rows x 8 columns: the transposed copy is 8 4-byte LDS stores, not 16 of 2


@always_inline
def _tj(p: Int) -> Int:
    comptime if FP:
        comptime assert F8 * 16 == FT * AHD
        return (Int(thread_idx.x) * 8 // AHD) * 2 + p
    else:
        return (Int(thread_idx.x) + p * F8) * 8 // AHD


@always_inline
def _kv_regs2(x: H16, r0: Int, S: Int, ld: Int, base: Int) -> Array[SIMD[bf, 8], 2]:
    """_kv_regs for the rows _tj(0), _tj(1) of the tile at r0."""
    var out = Array[SIMD[bf, 8], 2](fill=SIMD[bf, 8](0))
    comptime for p in range(2):
        var r = r0 + _tj(p)
        if r < S:
            out[p] = rebind[SIMD[bf, 8]](x.unsafe_offset(r * ld + base + Int(thread_idx.x) * 8 % AHD).load[width=8]())
    return out^


@always_inline
def _st_r(dst: Shared16, r: Array[SIMD[bf, 8], 2]):
    comptime for p in range(2):
        dst.unsafe_offset(_tj(p) * FW + Int(thread_idx.x) * 8 % AHD).store(r[p])


@always_inline
def _st_t(dst: Shared16, r: Array[SIMD[bf, 8], 2]):
    var e = Int(thread_idx.x) * 8 % AHD
    comptime if FP:
        comptime for u in range(8):
            dst.unsafe_offset((e + u) * FTW + _tj(0)).store[alignment=4](SIMD[bf, 2](r[0][u], r[1][u]))
    else:
        comptime for p in range(2):
            comptime for u in range(8):
                dst[(e + u) * FTW + _tj(p)] = r[p][u]


@always_inline
def _x16i(x: Int32) -> Int32:
    """x of the lane 16 away (the other half wave): v_permlanex16_b32, a register move (shuffle_xor is ds_bpermute)."""
    return llvm_intrinsic["llvm.amdgcn.permlanex16", Int32, has_side_effect=False](
        Int32(0), x, Int32(0x76543210), Int32(-19088744), False, False)


@always_inline
def _x16(x: Float32) -> Float32:
    return bitcast[DType.float32](_x16i(bitcast[DType.int32](x)))


@always_inline
def _pack16[SH: Bool = True](x: SIMD[DType.float32, 8], half: Int) -> SIMD[bf, 16]:
    """The 16 accumulator rows of this lane's column as one bf16 WMMA operand (element = row): the transposed
    accumulator without an LDS round trip. Both half waves build the same operand. SH: shift the own half into place
    (fewer VALU and registers; flash_fwd8t_k keeps the select form, 191 VGPR against 193: one more workgroup per WGP)."""
    comptime if SH:
        var w = bitcast[DType.uint16, 8](x.cast[bf]()).cast[DType.int32]() << SIMD[DType.int32, 8](Int32(16 * half))
        comptime for i in range(8):
            w[i] = w[i] | _x16i(w[i])
        return bitcast[bf, 16](w)
    else:
        var out = SIMD[bf, 16](0)
        comptime for i in range(8):
            var own = bitcast[DType.uint16](x[i].cast[bf]()).cast[DType.int32]()
            var oth = _x16i(own)
            out[_acc_row(0, i, 0)] = bitcast[bf]((own if half == 0 else oth).cast[DType.uint16]())
            out[_acc_row(0, i, 1)] = bitcast[bf]((oth if half == 0 else own).cast[DType.uint16]())
        return out


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(F8)))
def flash_fwd8t_k(q: H16, k: H16, v: H16, o: F, lse: F, b32: Int32, h32: Int32, s32: Int32, scale: Float32, cs: F, sn: F):
    """flash_fwd8_k with the scores transposed: keys in the accumulator rows, the query row in the lane. Row max, row
    sum and P as a WMMA operand stay in registers (no P tile in LDS, no ds_bpermute); O is accumulated transposed.
    Every sum keeps flash_fwd8_k's order (byte-equal parameters, flash8-ab.sh)."""
    var b = Int(b32)
    var S = Int(s32)
    var blk = Int(block_idx.x)
    var qt = ceildiv(S, FQR) - 1 - Int(block_idx.y)
    var lane = Int(thread_idx.x) % 32
    var wid = Int(thread_idx.x) // 32
    var hl = lane % 16
    var half = lane // 16
    var ld = b * Int(h32) * AHD
    var base = blk * AHD
    var ks = _shared16[FT * FW]()
    var vs = _shared16[AHD * FTW]()
    var q0 = qt * FQR + wid * 16
    var row = q0 + hl
    var qa = Array[SIMD[bf, 16], KT](fill=SIMD[bf, 16](0))
    comptime for kk in range(KT):
        qa[kk] = _frag(q, row, ld, base, kk * 16, row < S)
    var m: Float32 = -3.0e38
    var l: Float32 = 0
    var acc = Array[SIMD[DType.float32, 8], KT](fill=SIMD[DType.float32, 8](0))
    var nk = min(ceildiv(S, FT), (qt * FQR + FQR - 1) // FT + 1)
    var kr = _kv_regs2(k, 0, S, ld, base)
    var vr = _kv_regs2(v, 0, S, ld, base)
    for kt in range(nk):
        barrier()
        _st_r(ks, kr)
        _st_t(vs, vr)
        barrier()
        if kt + 1 < nk:
            kr = _kv_regs2(k, (kt + 1) * FT, S, ld, base)
            vr = _kv_regs2(v, (kt + 1) * FT, S, ld, base)
        comptime if (FX & 1) != 0:
            comptime MUT = FT if (FX & 4) != 0 else 0
            if kt * FT > q0 + 15 - MUT and (MUT == 0 or kt > 0):
                continue
        var s = Array[SIMD[DType.float32, 8], 4](fill=SIMD[DType.float32, 8](0))
        comptime for tj in range(4):
            comptime for kk in range(KT):
                var av = rebind[SIMD[bf, 16]](ks.unsafe_offset((tj * 16 + hl) * FW + kk * 16).load[width=16, alignment=16]())
                mma(s[tj], av, qa[kk], s[tj])
        var mx: Float32 = -3.0e38
        comptime for i in range(8):
            comptime for tj in range(4):
                var col = kt * FT + tj * 16 + _acc_row(0, i, half)
                var x: Float32 = s[tj][i] * scale if col <= row and col < S else -3.0e38
                s[tj][i] = x
                mx = max(mx, x)
        var mn = max(m, max(mx, _x16(mx)))
        var alpha = exp(m - mn)
        var t = SIMD[DType.float32, 8](0)
        var pb = Array[SIMD[bf, 16], 4](fill=SIMD[bf, 16](0))
        comptime for tj in range(4):
            var p = exp(s[tj] - SIMD[DType.float32, 8](mn))
            t = t + p
            pb[tj] = _pack16[False](p, half)
        comptime for i in range(8):
            t[i] = t[i] + _x16(t[i])
        var ls = ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]))
        l = l * alpha + ls
        m = mn
        comptime for te in range(KT):
            acc[te] = acc[te] * SIMD[DType.float32, 8](alpha)
        comptime for te in range(KT):
            comptime for kk in range(4):
                var av = rebind[SIMD[bf, 16]](vs.unsafe_offset((te * 16 + hl) * FTW + kk * 16).load[width=16, alignment=16]())
                mma(acc[te], av, pb[kk], acc[te])
    if row < S:
        var il = 1 / l
        comptime for te in range(KT):
            comptime for i in range(8):
                var off = row * ld + base + te * 16 + _acc_row(0, i, half)
                comptime if OB16:
                    H16(unsafe_from_address=Int(o))[unsafe_offset=off] = (acc[te][i] * il).cast[bf]()
                else:
                    o[unsafe_offset=off] = acc[te][i] * il
        if half == 0:
            lse[unsafe_offset=blk * SMAX + row] = m + log(l)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(F8)))
def flash_dq8t_k(q: H16, k: H16, v: H16, do: H16, lse: F, dl: F, y16: H16, b32: Int32, h32: Int32, s32: Int32, scale: Float32, cs: F, sn: F):
    """flash_dq8_k with S and dP transposed (keys in the accumulator rows, the query row in the lane): dS becomes the
    dQ product's A operand in registers (no dS tile in LDS). Same sums, same order."""
    var b = Int(b32)
    var h = Int(h32)
    var S = Int(s32)
    var blk = Int(block_idx.x)
    var qt = ceildiv(S, FQR) - 1 - Int(block_idx.y)
    var lane = Int(thread_idx.x) % 32
    var wid = Int(thread_idx.x) // 32
    var hl = lane % 16
    var half = lane // 16
    var ld = b * h * AHD
    var base = blk * AHD
    var ks = _shared16[FT * FW]()
    var vs = _shared16[FT * FW]()
    var kst = _shared16[AHD * FTW]()
    var q0 = qt * FQR + wid * 16
    var row = q0 + hl
    var qa = Array[SIMD[bf, 16], KT](fill=SIMD[bf, 16](0))
    var da = Array[SIMD[bf, 16], KT](fill=SIMD[bf, 16](0))
    comptime for kk in range(KT):
        qa[kk] = _frag(q, row, ld, base, kk * 16, row < S)
        da[kk] = _frag(do, row, ld, base, kk * 16, row < S)
    var lq: Float32 = 0
    var dq_: Float32 = 0
    if row < S:
        lq = lse[unsafe_offset=blk * SMAX + row]
        dq_ = dl[unsafe_offset=blk * SMAX + row]
    var dq = Array[SIMD[DType.float32, 8], KT](fill=SIMD[DType.float32, 8](0))
    var nk = min(ceildiv(S, FT), (qt * FQR + FQR - 1) // FT + 1)
    var kr = _kv_regs2(k, 0, S, ld, base)
    var vr = _kv_regs2(v, 0, S, ld, base)
    for kt in range(nk):
        barrier()
        _st_r(ks, kr)
        _st_r(vs, vr)
        _st_t(kst, kr)
        barrier()
        if kt + 1 < nk:
            kr = _kv_regs2(k, (kt + 1) * FT, S, ld, base)
            vr = _kv_regs2(v, (kt + 1) * FT, S, ld, base)
        comptime if (FX & 1) != 0:
            if kt * FT > q0 + 15:
                continue
        comptime for k16 in range(4):
            var st = SIMD[DType.float32, 8](0)
            var dpt = SIMD[DType.float32, 8](0)
            comptime for kk in range(KT):
                var bk = rebind[SIMD[bf, 16]](ks.unsafe_offset((k16 * 16 + hl) * FW + kk * 16).load[width=16, alignment=16]())
                var bv = rebind[SIMD[bf, 16]](vs.unsafe_offset((k16 * 16 + hl) * FW + kk * 16).load[width=16, alignment=16]())
                mma(st, bk, qa[kk], st)
                mma(dpt, bv, da[kk], dpt)
            var ds = SIMD[DType.float32, 8](0)
            comptime for i in range(8):
                var key = kt * FT + k16 * 16 + _acc_row(0, i, half)
                var pv: Float32 = exp(st[i] * scale - lq) if key <= row and key < S and row < S else 0
                ds[i] = pv * (dpt[i] - dq_) * scale
            var ad = _pack16(ds, half)
            comptime for te in range(KT):
                var bk = rebind[SIMD[bf, 16]](kst.unsafe_offset((te * 16 + hl) * FTW + k16 * 16).load[width=16, alignment=16]())
                mma(dq[te], ad, bk, dq[te])
    _put_y16(y16, dq, q0, half, hl, S, b, blk, Int(h32), 0, True, cs, sn)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(F8)))
def flash_dkv8t_k(q: H16, k: H16, v: H16, do: H16, lse: F, dl: F, y16: H16, b32: Int32, h32: Int32, s32: Int32, scale: Float32, cs: F, sn: F):
    """flash_dkv8_k with S and dP the other way round (query rows in the accumulator rows, the key in the lane): P^T
    and dS^T become the dV / dK products' A operands in registers (no P^T / dS^T tiles in LDS); lse and dl are read
    per accumulator row. Same sums, same order."""
    var b = Int(b32)
    var h = Int(h32)
    var S = Int(s32)
    var blk = Int(block_idx.x)
    var kt = Int(block_idx.y)
    var lane = Int(thread_idx.x) % 32
    var wid = Int(thread_idx.x) // 32
    var hl = lane % 16
    var half = lane // 16
    var ld = b * h * AHD
    var base = blk * AHD
    var qs = _shared16[FT * FW]()
    var dos = _shared16[FT * FW]()
    var qst = _shared16[AHD * FTW]()
    var dost = _shared16[AHD * FTW]()
    var k0 = kt * FQR + wid * 16
    var key = k0 + hl
    var ka = Array[SIMD[bf, 16], KT](fill=SIMD[bf, 16](0))
    var va = Array[SIMD[bf, 16], KT](fill=SIMD[bf, 16](0))
    comptime for kk in range(KT):
        ka[kk] = _frag(k, key, ld, base, kk * 16, key < S)
        va[kk] = _frag(v, key, ld, base, kk * 16, key < S)
    var dk = Array[SIMD[DType.float32, 8], KT](fill=SIMD[DType.float32, 8](0))
    var dv = Array[SIMD[DType.float32, 8], KT](fill=SIMD[DType.float32, 8](0))
    var q_lo = kt * FQR // FT
    var nq = ceildiv(S, FT)
    var qr = _kv_regs2(q, q_lo * FT, S, ld, base)
    var dr = _kv_regs2(do, q_lo * FT, S, ld, base)
    for qt in range(q_lo, nq):
        barrier()
        _st_r(qs, qr)
        _st_r(dos, dr)
        _st_t(qst, qr)
        _st_t(dost, dr)
        barrier()
        if qt + 1 < nq:
            qr = _kv_regs2(q, (qt + 1) * FT, S, ld, base)
            dr = _kv_regs2(do, (qt + 1) * FT, S, ld, base)
        comptime if (FX & 1) != 0:
            if qt * FT + FT - 1 < k0:
                continue
        comptime for q16 in range(4):
            var s = SIMD[DType.float32, 8](0)
            var dp = SIMD[DType.float32, 8](0)
            comptime for kk in range(KT):
                var bq = rebind[SIMD[bf, 16]](qs.unsafe_offset((q16 * 16 + hl) * FW + kk * 16).load[width=16, alignment=16]())
                var bd = rebind[SIMD[bf, 16]](dos.unsafe_offset((q16 * 16 + hl) * FW + kk * 16).load[width=16, alignment=16]())
                mma(s, bq, ka[kk], s)
                mma(dp, bd, va[kk], dp)
            var pt = SIMD[DType.float32, 8](0)
            var dt = SIMD[DType.float32, 8](0)
            comptime for i in range(8):
                var qq = qt * FT + q16 * 16 + _acc_row(0, i, half)
                var lq: Float32 = lse[unsafe_offset=blk * SMAX + qq] if qq < S else 0
                var dq_: Float32 = dl[unsafe_offset=blk * SMAX + qq] if qq < S else 0
                var pv: Float32 = exp(s[i] * scale - lq) if qq >= key and qq < S and key < S else 0
                pt[i] = pv
                dt[i] = pv * (dp[i] - dq_) * scale
            var ap = _pack16(pt, half)
            var ad = _pack16(dt, half)
            comptime for te in range(KT):
                var bd = rebind[SIMD[bf, 16]](dost.unsafe_offset((te * 16 + hl) * FTW + q16 * 16).load[width=16, alignment=16]())
                var bq = rebind[SIMD[bf, 16]](qst.unsafe_offset((te * 16 + hl) * FTW + q16 * 16).load[width=16, alignment=16]())
                mma(dv[te], ap, bd, dv[te])
                mma(dk[te], ad, bq, dk[te])
    _put_y16(y16, dk, k0, half, hl, S, b, blk, h, 1, True, cs, sn)
    _put_y16(y16, dv, k0, half, hl, S, b, blk, h, 2, False, cs, sn)


comptime QKN =get_defined_int["LAT_QKN", 0]() == 1  # parameter-free per-head RMS norm of q and k (before rope)


def qkn_k(x: F, r: F, n32: Int32):
    """x [n, D] in place: each head's AHD slice / its RMS; r [n, H] keeps that RMS. One wave per (row, head)."""
    var t = global_idx.x // 32
    var lane = Int(thread_idx.x) % 32
    if t >= Int(n32) * (DM // AHD):
        return
    var row = x.unsafe_offset(t * AHD)  # heads are contiguous AHD slices, rows contiguous D: slice t starts at t * AHD
    var s: Float32 = 0
    for j in range(lane, AHD, 32):
        s += row[j] * row[j]
    s = warp.sum(s)
    var rms = sqrt(s / Float32(AHD) + 1e-6)
    for j in range(lane, AHD, 32):
        row[j] = row[j] / rms
    if lane == 0:
        r[unsafe_offset=t] = rms


def qknb_k[G16: Bool](g: F, g16: H16, gs32: Int32, xh: F, r: F, n32: Int32):
    """Norm backward in place: g <- (g - xh mean(g xh)) / r per (row, head); xh = the normalised x in g's frame.
    G16: g is bf16 with row stride gs (the flash backward's interleaved [n, 3D] rows), else fp32 stride D."""
    comptime NH = DM // AHD
    var t = global_idx.x // 32
    var lane = Int(thread_idx.x) % 32
    if t >= Int(n32) * NH:
        return
    var rw = t // NH
    var hh = t % NH
    var xr = xh.unsafe_offset(t * AHD)
    var go = rw * Int(gs32) + hh * AHD
    var d: Float32 = 0
    for j in range(lane, AHD, 32):
        comptime if G16:
            d += g16[unsafe_offset=go + j].cast[DType.float32]() * xr[j]
        else:
            d += g[unsafe_offset=go + j] * xr[j]
    d = warp.sum(d) / Float32(AHD)
    var iv = 1 / r[unsafe_offset=t]
    for j in range(lane, AHD, 32):
        comptime if G16:
            g16[unsafe_offset=go + j] = ((g16[unsafe_offset=go + j].cast[DType.float32]() - xr[j] * d) * iv).cast[bf]()
        else:
            g[unsafe_offset=go + j] = (g[unsafe_offset=go + j] - xr[j] * d) * iv


def rope16_k(x: F, y: H16, cs: F, sn: F, n32: Int32, b32: Int32, rope: Int32):
    """y = bf16(rope(x)) (rope != 0) or bf16(x) over n rows of D; row r is position r // b (flash inputs, once a layer).
    One thread per 8 rotation pairs (e, e + HD / 2) of one head: 32 B loads, 16 B bf16 stores."""
    comptime half = AHD // 2
    comptime G = half // 8
    var t = global_idx.x
    if t >= Int(n32) * (DM // AHD) * G:
        return
    var row = t // ((DM // AHD) * G)
    var hh = (t // G) % (DM // AHD)
    var i0 = (t % G) * 8
    var o = row * DM + hh * AHD + i0
    var x1 = x.unsafe_offset(o).load[width=8]()
    var x2 = x.unsafe_offset(o + half).load[width=8]()
    if rope != 0:
        var pos = row // Int(b32)
        var c = cs.unsafe_offset(pos * half + i0).load[width=8]()
        var sv = sn.unsafe_offset(pos * half + i0).load[width=8]()
        var r1 = x1 * c - x2 * sv
        var r2 = x2 * c + x1 * sv
        x1 = r1
        x2 = r2
    y.unsafe_offset(o).store(x1.cast[bf]())
    y.unsafe_offset(o + half).store(x2.cast[bf]())


def dlt_k(do: F, o: F, dl: F, b32: Int32, h32: Int32, s32: Int32):
    """dl[blk * SMAX + j] = sum_e dO * O (FA2's D): one wave per (position, sequence * head), lanes walk e (coalesced)."""
    var t = global_idx.x // 32
    var lane = Int(thread_idx.x) % 32
    var bh = Int(b32) * Int(h32)
    if t >= Int(s32) * bh:
        return
    var j = t // bh
    var blk = t % bh
    var off = j * bh * AHD + blk * AHD
    var acc: Float32 = 0
    for e in range(lane, AHD, 32):
        comptime if OB16:
            acc += do[unsafe_offset=off + e] * H16(unsafe_from_address=Int(o))[unsafe_offset=off + e].cast[DType.float32]()
        else:
            acc += do[unsafe_offset=off + e] * o[unsafe_offset=off + e]
    acc = warp.sum(acc)
    if lane == 0:
        dl[unsafe_offset=blk * SMAX + j] = acc


@always_inline
def _put_y16(y16: H16, a: Array[SIMD[DType.float32, 8], KT], row0: Int, half: Int, hl: Int, S: Int, b: Int, blk: Int, h: Int,
             part: Int, rope: Bool, cs: F, sn: F):
    """Rows row0 + acc rows of a 16 x HD accumulator into the interleaved bf16 [n, 3D] q/k/v-grad rows (part 0/1/2),
    inverse rope (rope_k[True]) when rope: partners e and e + HD / 2 are tiles te and te + KT / 2, same lane."""
    comptime HK = KT // 2
    comptime half_d = AHD // 2
    var bi = blk // h
    var hh = blk % h
    comptime for tp in range(HK):
        comptime for i in range(8):
            var row = row0 + _acc_row(0, i, half)
            if row < S:
                var e0 = tp * 16 + hl
                var x0 = a[tp][i]
                var x1 = a[tp + HK][i]
                var ci = row * DM * 3 * b + bi * 3 * DM + hh * AHD + e0 + part * DM
                if rope:
                    var co = cs[unsafe_offset=row * half_d + e0]
                    var si = -sn[unsafe_offset=row * half_d + e0]
                    y16[unsafe_offset=ci] = (x0 * co - x1 * si).cast[bf]()
                    y16[unsafe_offset=ci + half_d] = (x1 * co + x0 * si).cast[bf]()
                else:
                    y16[unsafe_offset=ci] = x0.cast[bf]()
                    y16[unsafe_offset=ci + half_d] = x1.cast[bf]()


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(FAT)))  # 128 threads: the 256-VGPR budget, no spills
def flash_dkv_k(q: H16, k: H16, v: H16, do: H16, lse: F, dl: F, y16: H16, b32: Int32, h32: Int32, s32: Int32, scale: Float32, cs: F, sn: F):
    """grid (b * H, ceil(S / FT)) key tiles, FAT threads: wave w owns keys k0..k0+15, loops query tiles from the
    diagonal: S^T = K Q^T, dP^T = V dO^T, P^T from lse, dS^T = P^T (dP^T - D) scale; dV += P^T dO, dK += dS^T Q."""
    var b = Int(b32)
    var h = Int(h32)
    var S = Int(s32)
    var blk = Int(block_idx.x)
    var kt = Int(block_idx.y)
    var lane = Int(thread_idx.x) % 32
    var wid = Int(thread_idx.x) // 32
    var hl = lane % 16
    var half = lane // 16
    var ld = b * h * AHD
    var base = blk * AHD
    var qs = _shared16[FT * FW]()
    var dos = _shared16[FT * FW]()
    var qst = _shared16[AHD * FTW]()
    var dost = _shared16[AHD * FTW]()
    var pt = _shared16[FT * 24]()
    var dt = _shared16[FT * 24]()
    var k0 = kt * FT + wid * 16
    var ka = Array[SIMD[bf, 16], KT](fill=SIMD[bf, 16](0))
    var va = Array[SIMD[bf, 16], KT](fill=SIMD[bf, 16](0))
    comptime for kk in range(KT):
        ka[kk] = _frag(k, k0 + hl, ld, base, kk * 16, k0 + hl < S)
        va[kk] = _frag(v, k0 + hl, ld, base, kk * 16, k0 + hl < S)
    var dk = Array[SIMD[DType.float32, 8], KT](fill=SIMD[DType.float32, 8](0))
    var dv = Array[SIMD[DType.float32, 8], KT](fill=SIMD[DType.float32, 8](0))
    for qt in range(kt, ceildiv(S, FT)):
        barrier()
        _ld_tile(q, qs, qt * FT, S, ld, base)
        _ld_tile(do, dos, qt * FT, S, ld, base)
        barrier()
        _tr_lds(qs, qst)
        _tr_lds(dos, dost)
        barrier()
        comptime for q16 in range(4):
            var qq = qt * FT + q16 * 16 + hl
            var lq: Float32 = lse[unsafe_offset=blk * SMAX + qq] if qq < S else 0
            var dq_: Float32 = dl[unsafe_offset=blk * SMAX + qq] if qq < S else 0
            var st = SIMD[DType.float32, 8](0)
            var dpt = SIMD[DType.float32, 8](0)
            comptime for kk in range(KT):
                var bq = SIMD[bf, 16](0)
                var bd = SIMD[bf, 16](0)
                bq = rebind[SIMD[bf, 16]](qs.unsafe_offset((q16 * 16 + hl) * FW + kk * 16).load[width=16, alignment=16]())
                bd = rebind[SIMD[bf, 16]](dos.unsafe_offset((q16 * 16 + hl) * FW + kk * 16).load[width=16, alignment=16]())
                mma(st, ka[kk], bq, st)
                mma(dpt, va[kk], bd, dpt)
            comptime for i in range(8):
                var key = k0 + _acc_row(0, i, half)
                var pv: Float32 = exp(st[i] * scale - lq) if qq >= key and qq < S and key < S else 0
                var r = wid * 16 + _acc_row(0, i, half)
                pt[r * 24 + hl] = pv.cast[bf]()
                dt[r * 24 + hl] = (pv * (dpt[i] - dq_) * scale).cast[bf]()
            barrier()
            var ap = SIMD[bf, 16](0)
            var ad = SIMD[bf, 16](0)
            ap = rebind[SIMD[bf, 16]](pt.unsafe_offset((wid * 16 + hl) * 24).load[width=16, alignment=16]())
            ad = rebind[SIMD[bf, 16]](dt.unsafe_offset((wid * 16 + hl) * 24).load[width=16, alignment=16]())
            comptime for te in range(KT):
                var bd = SIMD[bf, 16](0)
                var bq = SIMD[bf, 16](0)
                bd = rebind[SIMD[bf, 16]](dost.unsafe_offset((te * 16 + hl) * FTW + q16 * 16).load[width=16, alignment=16]())
                bq = rebind[SIMD[bf, 16]](qst.unsafe_offset((te * 16 + hl) * FTW + q16 * 16).load[width=16, alignment=16]())
                mma(dv[te], ap, bd, dv[te])
                mma(dk[te], ad, bq, dk[te])
            barrier()
    _put_y16(y16, dk, k0, half, hl, S, b, blk, h, 1, True, cs, sn)
    _put_y16(y16, dv, k0, half, hl, S, b, blk, h, 2, False, cs, sn)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(FAT)))  # 128 threads: the 256-VGPR budget, no spills
def flash_dq_k(q: H16, k: H16, v: H16, do: H16, lse: F, dl: F, y16: H16, b32: Int32, h32: Int32, s32: Int32, scale: Float32, cs: F, sn: F):
    """grid (b * H, ceil(S / FT)) query tiles, FAT threads: wave w owns queries q0..q0+15, loops key tiles up to the
    diagonal: S = Q K^T, dP = dO V^T, P from lse, dS = P (dP - D) scale; dQ += dS K."""
    var b = Int(b32)
    var h = Int(h32)
    var S = Int(s32)
    var blk = Int(block_idx.x)
    var qt = ceildiv(S, FT) - 1 - Int(block_idx.y)
    var lane = Int(thread_idx.x) % 32
    var wid = Int(thread_idx.x) // 32
    var hl = lane % 16
    var half = lane // 16
    var ld = b * h * AHD
    var base = blk * AHD
    var ks = _shared16[FT * FW]()
    var vs = _shared16[FT * FW]()
    var kst = _shared16[AHD * FTW]()
    var dsw = _shared16[FT * 24]()
    var q0 = qt * FT + wid * 16
    var qa = Array[SIMD[bf, 16], KT](fill=SIMD[bf, 16](0))
    var da = Array[SIMD[bf, 16], KT](fill=SIMD[bf, 16](0))
    comptime for kk in range(KT):
        qa[kk] = _frag(q, q0 + hl, ld, base, kk * 16, q0 + hl < S)
        da[kk] = _frag(do, q0 + hl, ld, base, kk * 16, q0 + hl < S)
    var lq = SIMD[DType.float32, 8](0)
    var dq_ = SIMD[DType.float32, 8](0)
    comptime for i in range(8):
        var row = q0 + _acc_row(0, i, half)
        if row < S:
            lq[i] = lse[unsafe_offset=blk * SMAX + row]
            dq_[i] = dl[unsafe_offset=blk * SMAX + row]
    var dq = Array[SIMD[DType.float32, 8], KT](fill=SIMD[DType.float32, 8](0))
    for kt in range(qt + 1):
        barrier()
        _ld_tile(k, ks, kt * FT, S, ld, base)
        _ld_tile(v, vs, kt * FT, S, ld, base)
        barrier()
        _tr_lds(ks, kst)
        barrier()
        comptime for k16 in range(4):
            var s = SIMD[DType.float32, 8](0)
            var dp = SIMD[DType.float32, 8](0)
            comptime for kk in range(KT):
                var bk = SIMD[bf, 16](0)
                var bv = SIMD[bf, 16](0)
                bk = rebind[SIMD[bf, 16]](ks.unsafe_offset((k16 * 16 + hl) * FW + kk * 16).load[width=16, alignment=16]())
                bv = rebind[SIMD[bf, 16]](vs.unsafe_offset((k16 * 16 + hl) * FW + kk * 16).load[width=16, alignment=16]())
                mma(s, qa[kk], bk, s)
                mma(dp, da[kk], bv, dp)
            var key = kt * FT + k16 * 16 + hl
            comptime for i in range(8):
                var row = q0 + _acc_row(0, i, half)
                var pv: Float32 = exp(s[i] * scale - lq[i]) if key <= row and key < S and row < S else 0
                dsw[(wid * 16 + _acc_row(0, i, half)) * 24 + hl] = (pv * (dp[i] - dq_[i]) * scale).cast[bf]()
            barrier()
            var ad = SIMD[bf, 16](0)
            ad = rebind[SIMD[bf, 16]](dsw.unsafe_offset((wid * 16 + hl) * 24).load[width=16, alignment=16]())
            comptime for te in range(KT):
                var bk = SIMD[bf, 16](0)
                bk = rebind[SIMD[bf, 16]](kst.unsafe_offset((te * 16 + hl) * FTW + k16 * 16).load[width=16, alignment=16]())
                mma(dq[te], ad, bk, dq[te])
            barrier()
    _put_y16(y16, dq, q0, half, hl, S, b, blk, Int(h32), 0, True, cs, sn)


# fp32 reference attention for the exact path when SMAX > 64 (the scalar kernels hold [S, S] in LDS): one thread per
# row, P recomputed in loops, q/k already roped (exact forward ropes them in place). Oracle only: slow by design.
def attn_ref_fwd_k(q: F, k: F, v: F, o: F, lse: F, b32: Int32, h32: Int32, s32: Int32, scale: Float32):
    var t = global_idx.x
    var bh = Int(b32) * Int(h32)
    if t >= Int(s32) * bh:
        return
    var j = t // bh
    var blk = t % bh
    var ld = bh * AHD
    var base = blk * AHD
    var m: Float32 = -3.0e38
    for c in range(j + 1):
        var s: Float32 = 0
        for e in range(AHD):
            s += q[unsafe_offset=j * ld + base + e] * k[unsafe_offset=c * ld + base + e]
        m = max(m, s * scale)
    var acc = SIMD[DType.float32, AHD](0)
    var l: Float32 = 0
    for c in range(j + 1):
        var s: Float32 = 0
        for e in range(AHD):
            s += q[unsafe_offset=j * ld + base + e] * k[unsafe_offset=c * ld + base + e]
        var pv = exp(s * scale - m)
        l += pv
        for e in range(AHD):
            acc[e] += pv * v[unsafe_offset=c * ld + base + e]
    for e in range(AHD):
        o[unsafe_offset=j * ld + base + e] = acc[e] / l
    lse[unsafe_offset=blk * SMAX + j] = m + log(l)


@always_inline
def _ref_p_ds(q: F, k: F, v: F, do: F, lse: F, dl: F, i: Int, c: Int, ld: Int, base: Int, blk: Int, scale: Float32) -> SIMD[DType.float32, 2]:
    """(P[i, c], dS[i, c]) of query i and key c, recomputed in fp32."""
    var s: Float32 = 0
    var dp: Float32 = 0
    for e in range(AHD):
        s += q[unsafe_offset=i * ld + base + e] * k[unsafe_offset=c * ld + base + e]
        dp += do[unsafe_offset=i * ld + base + e] * v[unsafe_offset=c * ld + base + e]
    var pv = exp(s * scale - lse[unsafe_offset=blk * SMAX + i])
    return SIMD[DType.float32, 2](pv, pv * (dp - dl[unsafe_offset=blk * SMAX + i]) * scale)


def attn_ref_bwd_k(q: F, k: F, v: F, o: F, do: F, lse: F, dl: F, dq: F, dk: F, dv: F, b32: Int32, h32: Int32, s32: Int32, scale: Float32, part: Int32):
    """part 0: D = rowsum(dO O) per row; part 1: dQ row j; part 2: dK, dV row j (keys). One thread per row."""
    var t = global_idx.x
    var bh = Int(b32) * Int(h32)
    var S = Int(s32)
    if t >= S * bh:
        return
    var j = t // bh
    var blk = t % bh
    var ld = bh * AHD
    var base = blk * AHD
    if part == 0:
        var acc: Float32 = 0
        for e in range(AHD):
            acc += do[unsafe_offset=j * ld + base + e] * o[unsafe_offset=j * ld + base + e]
        dl[unsafe_offset=blk * SMAX + j] = acc
        return
    var a1 = SIMD[DType.float32, AHD](0)
    var a2 = SIMD[DType.float32, AHD](0)
    if part == 1:
        for c in range(j + 1):
            var pd = _ref_p_ds(q, k, v, do, lse, dl, j, c, ld, base, blk, scale)
            for e in range(AHD):
                a1[e] += pd[1] * k[unsafe_offset=c * ld + base + e]
        for e in range(AHD):
            dq[unsafe_offset=j * ld + base + e] = a1[e]
        return
    for i in range(j, S):
        var pd = _ref_p_ds(q, k, v, do, lse, dl, i, j, ld, base, blk, scale)
        for e in range(AHD):
            a1[e] += pd[1] * q[unsafe_offset=i * ld + base + e]
            a2[e] += pd[0] * do[unsafe_offset=i * ld + base + e]
    for e in range(AHD):
        dk[unsafe_offset=j * ld + base + e] = a1[e]
        dv[unsafe_offset=j * ld + base + e] = a2[e]


def swiglu_fwd16_k(a: F, b: F, y: F, y16: H16, n32: Int32):
    """ops.swiglu_fwd_k plus the bf16 copy of y (the w2 GEMM's A operand)."""
    var i = global_idx.x
    if i >= Int(n32):
        return
    var g = a[unsafe_offset=i]
    var y0 = g / (1 + exp(-g)) * b[unsafe_offset=i]
    y[unsafe_offset=i] = y0
    y16[unsafe_offset=i] = y0.cast[bf]()


def swiglu_fwdh_k(a: F, b: F, y16: H16, n32: Int32):
    """swiglu_fwd16_k without the fp32 y: its only other reader was the dW transpose, which now reads y16."""
    var i = global_idx.x
    if i >= Int(n32):
        return
    var g = a[unsafe_offset=i]
    y16[unsafe_offset=i] = (g / (1 + exp(-g)) * b[unsafe_offset=i]).cast[bf]()


def swiglu_bwdh_k(a: F, b: F, dy: F, da16: H16, db16: H16, n32: Int32):
    """swiglu_bwd16_k without the fp32 da/db (read before only by the dW transposes)."""
    var i = global_idx.x
    if i >= Int(n32):
        return
    var g = a[unsafe_offset=i]
    var s = 1 / (1 + exp(-g))
    var u = dy[unsafe_offset=i]
    db16[unsafe_offset=i] = (u * g * s).cast[bf]()
    da16[unsafe_offset=i] = (u * b[unsafe_offset=i] * s * (1 + g * (1 - s))).cast[bf]()


def swiglu_fwdb_k(a: H16, b: H16, y16: H16, n32: Int32):
    """swiglu_fwdh_k on bf16 a/b (the w1/w3 GEMMs store bf16): half the read bytes."""
    var i = global_idx.x
    if i >= Int(n32):
        return
    var g = a[unsafe_offset=i].cast[DType.float32]()
    y16[unsafe_offset=i] = (g / (1 + exp(-g)) * b[unsafe_offset=i].cast[DType.float32]()).cast[bf]()


def swiglu_bwdb_k(a: H16, b: H16, dy: H16, da16: H16, db16: H16, n32: Int32):
    """swiglu_bwdh_k on bf16 a/b/dy."""
    var i = global_idx.x
    if i >= Int(n32):
        return
    var g = a[unsafe_offset=i].cast[DType.float32]()
    var s = 1 / (1 + exp(-g))
    var u = dy[unsafe_offset=i].cast[DType.float32]()
    db16[unsafe_offset=i] = (u * g * s).cast[bf]()
    da16[unsafe_offset=i] = (u * b[unsafe_offset=i].cast[DType.float32]() * s * (1 + g * (1 - s))).cast[bf]()


def swiglu_bwd16_k(a: F, b: F, dy: F, da: F, db: F, da16: H16, db16: H16, n32: Int32):
    """ops.swiglu_bwd_k plus bf16 copies of da and db (the dX GEMMs' A operands)."""
    var i = global_idx.x
    if i >= Int(n32):
        return
    var g = a[unsafe_offset=i]
    var s = 1 / (1 + exp(-g))
    var u = dy[unsafe_offset=i]
    var vb = u * g * s
    var va = u * b[unsafe_offset=i] * s * (1 + g * (1 - s))
    db[unsafe_offset=i] = vb
    da[unsafe_offset=i] = va
    db16[unsafe_offset=i] = vb.cast[bf]()
    da16[unsafe_offset=i] = va.cast[bf]()


def ileave_k(w: H16, y: H16, n32: Int32):
    """y [D, 3D] bf16, row i = (wq[i, :], wk[i, :], wv[i, :]) from the three contiguous [D, D] blocks at w."""
    var t = global_idx.x
    if t >= Int(n32):
        return
    var i = t // (3 * DM)
    var r = t % (3 * DM)
    y[unsafe_offset=t] = w[unsafe_offset=(r // DM) * DM * DM + i * DM + r % DM]


def rope3_k(x0: F, x1: F, x2: F, y: H16, cs: F, sn: F, n32: Int32, b32: Int32, r032: Int32):
    """cast3_k with the inverse rope of rope_k[True] applied to x0 (dq) and x1 (dk) on the fly: y [n, 3D] bf16 is
    the A operand of the fused q/k/v dX GEMM and the B operand of their dW (dwg_k, row stride 3D)."""
    comptime half = AHD // 2
    var t = global_idx.x
    if t >= Int(n32) * 3 * DM:
        return
    var r = t // (3 * DM)
    var c = t % (3 * DM)
    var j = c // DM
    var cc = c % DM
    var src = x0 if j == 0 else (x1 if j == 1 else x2)
    var off = r * DM + cc
    if j == 2:
        y[unsafe_offset=t] = src[unsafe_offset=off].cast[bf]()
        return
    var e = cc % AHD
    var i = e % half
    var o = off - e + i
    var pos = Int(r032) + r // Int(b32)
    var x1v = src[unsafe_offset=o]
    var x2v = src[unsafe_offset=o + half]
    var co = cs[unsafe_offset=pos * half + i]
    var si = -sn[unsafe_offset=pos * half + i]
    var v = x1v * co - x2v * si if e < half else x2v * co + x1v * si
    y[unsafe_offset=t] = v.cast[bf]()


def cast3_k(x0: F, x1: F, x2: F, y: H16, n32: Int32):
    """y [n, 3D] bf16, row r = (x0[r, :], x1[r, :], x2[r, :]): the A operand of the fused q/k/v dX GEMM."""
    var t = global_idx.x
    if t >= Int(n32) * 3 * DM:
        return
    var r = t // (3 * DM)
    var c = t % (3 * DM)
    var j = c // DM
    var src = x0 if j == 0 else (x1 if j == 1 else x2)
    y[unsafe_offset=t] = src[unsafe_offset=r * DM + c % DM].cast[bf]()


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


def rms_fwd16_k(x: F, hres: F, w: F, y: F, r: F, y16: H16, n32: Int32, eps: Float32, add: Int32):
    """rms_fwd_k plus a bf16 copy of y (the next GEMM's A operand); add != 0 first does x += hres (residual)."""
    var row = Int(block_idx.x) * NW + Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    if row >= Int(n32):
        return
    var o = row * DM
    var xv = SIMD[DType.float32, DL](0)
    comptime for t in range(DL):
        xv[t] = x[unsafe_offset=o + lane + 32 * t]
    if add != 0:
        comptime for t in range(DL):
            xv[t] += hres[unsafe_offset=o + lane + 32 * t]
            x[unsafe_offset=o + lane + 32 * t] = xv[t]
    var rr = rsqrt(warp.sum((xv * xv).reduce_add()) / Float32(DM) + eps)
    if lane == 0:
        r[unsafe_offset=row] = rr
    comptime for t in range(DL):
        var yv = xv[t] * rr * w[unsafe_offset=lane + 32 * t]
        y[unsafe_offset=o + lane + 32 * t] = yv
        y16[unsafe_offset=o + lane + 32 * t] = yv.cast[DType.bfloat16]()


def rms_fwdh_k[dup: Bool](x: F, hres: F, w: F, yk: H16, r: F, y16: H16, n32: Int32, eps: Float32, add: Int32):
    """rms_fwd16_k with the kept y in bf16 (yk, read later only by the dW transpose) instead of fp32."""
    var row = Int(block_idx.x) * NW + Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    if row >= Int(n32):
        return
    var o = row * DM
    var xv = SIMD[DType.float32, DL](0)
    comptime for t in range(DL):
        xv[t] = x[unsafe_offset=o + lane + 32 * t]
    if add != 0:
        comptime for t in range(DL):
            xv[t] += hres[unsafe_offset=o + lane + 32 * t]
            x[unsafe_offset=o + lane + 32 * t] = xv[t]
    var rr = rsqrt(warp.sum((xv * xv).reduce_add()) / Float32(DM) + eps)
    if lane == 0:
        r[unsafe_offset=row] = rr
    comptime for t in range(DL):
        var yv = (xv[t] * rr * w[unsafe_offset=lane + 32 * t]).cast[DType.bfloat16]()
        yk[unsafe_offset=o + lane + 32 * t] = yv
        comptime if dup:  # not dup: the consumers read yk (n1 / n2 bytes) directly
            y16[unsafe_offset=o + lane + 32 * t] = yv


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


def rms_dxg_k(x: F, w: F, r: F, dy: F, dx: F, part: F, n32: Int32, acc: Int32):
    """rms_dx_k plus the gain-gradient partials of rms_dwp_k in the same pass over x and dy: block = RC rows
    (each wave RC / NW of them), per-lane sums in registers, one LDS reduction over the waves, part[block, :]."""
    var wv = Int(thread_idx.x) // 32
    var lane = Int(thread_idx.x) % 32
    var n = Int(n32)
    var red = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[NW, DM]())
    var pg = SIMD[DType.float32, DL](0)
    for it in range(RC // NW):
        var row = Int(block_idx.x) * RC + it * NW + wv
        if row < n:
            var o = row * DM
            var rr = r[unsafe_offset=row]
            var xv = SIMD[DType.float32, DL](0)
            var gv = SIMD[DType.float32, DL](0)
            comptime for t in range(DL):
                var d = dy[unsafe_offset=o + lane + 32 * t]
                xv[t] = x[unsafe_offset=o + lane + 32 * t]
                gv[t] = d * w[unsafe_offset=lane + 32 * t]
                pg[t] += d * xv[t] * rr
            var c = rr * rr * rr * warp.sum((gv * xv).reduce_add()) / Float32(DM)
            comptime for t in range(DL):
                var i = o + lane + 32 * t
                var val = rr * gv[t] - xv[t] * c
                dx[unsafe_offset=i] = dx[unsafe_offset=i] + val if acc != 0 else val
    comptime for t in range(DL):
        red[wv, lane + 32 * t] = rebind[red.ElementType](pg[t])
    barrier()
    for j in range(Int(thread_idx.x), DM, AT):
        var sm: Float32 = 0
        comptime for q in range(NW):
            sm += rebind[Float32](red[q, j])
        part[unsafe_offset=Int(block_idx.x) * DM + j] = sm


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


def embed_srt_k(order: I, start: I, dy: F, g: F):
    """Embedding grad from rows sorted by token id (host counting sort, stable): block x = vocab id v, block y the
    256-column chunk; g[v] += sum of dy rows order[start[v]..start[v+1]) in row order. Deterministic, no partials."""
    var v = Int(block_idx.x)
    var d = Int(block_idx.y) * 256 + Int(thread_idx.x)
    var a = Int(start[unsafe_offset=v])
    var z = Int(start[unsafe_offset=v + 1])
    if a == z:
        return
    var acc: Float32 = 0
    for r in range(a, z):
        acc += dy[unsafe_offset=Int(order[unsafe_offset=r]) * DM + d]
    g[unsafe_offset=v * DM + d] = g[unsafe_offset=v * DM + d] + acc


comptime CAP = get_defined_int["LAT_CAP", 0]()  # >0: logits soft-capped to CAP tanh(z / CAP) in ce_w_k (loss and dz)


def ce_w_k[f32: Bool](z: F, tgt: I, wr: F, loss: F, dz: F, n32: Int32, v32: Int32, dz16: H16):
    """ce_k with one wave per row (large V), lanes stride the logits. f32: fp32 dz for the exact path, else bf16 dz16."""
    var t = global_idx.x // 32
    var lane = Int(thread_idx.x) % 32
    var v = Int(v32)
    if t >= Int(n32):
        return
    var y = Int(tgt[unsafe_offset=t])
    var w = wr[unsafe_offset=t]
    var zr = z.unsafe_offset(t * v)
    if y < 0 or w == 0:
        for j in range(lane, v, 32):
            comptime if f32:
                dz[unsafe_offset=t * v + j] = 0
            else:
                dz16[unsafe_offset=t * v + j] = Float32(0).cast[bf]()
        if lane == 0:
            loss[unsafe_offset=t] = 0
        return
    comptime if CAP > 0:  # z <- CAP tanh(z / CAP) in place; dz below gets the (1 - tanh^2) factor
        for j in range(lane, v, 32):
            zr[j] = Float32(CAP) * tanh(zr[j] / Float32(CAP))
    var m: Float32 = -3.0e38
    for j in range(lane, v, 32):
        m = max(m, zr[j])
    m = warp.max(m)
    var s: Float32 = 0
    for j in range(lane, v, 32):
        s += exp(zr[j] - m)
    s = warp.sum(s)
    var inv = 1 / s
    for j in range(lane, v, 32):  # bf16 only: both head GEMMs read bf16 (no cast16 / transpose of [rows, V])
        var gj = (exp(zr[j] - m) * inv - Float32(Int(j == y))) * w
        comptime if CAP > 0:
            var th = zr[j] / Float32(CAP)
            gj *= 1 - th * th
        comptime if f32:
            dz[unsafe_offset=t * v + j] = gj
        else:
            dz16[unsafe_offset=t * v + j] = gj.cast[bf]()
    if lane == 0:
        loss[unsafe_offset=t] = (log(s) + m - zr[y]) * w


def ce_kd_k[f32: Bool](z: F, zt: H16, tgt: I, wr: F, loss: F, dz: F, n32: Int32, v32: Int32, dz16: H16, lam: Float32):
    """ce_w_k plus in-place distillation (mdist=L): row loss w ((1 - L) CE + L KL(q || p)), q = softmax of the saved full-width
    logits zt (bf16, same [rows, V] layout, already soft-capped when CAP > 0), dz = w (p - (1 - L) onehot - L q). One wave per row."""
    var t = global_idx.x // 32
    var lane = Int(thread_idx.x) % 32
    var v = Int(v32)
    if t >= Int(n32):
        return
    var y = Int(tgt[unsafe_offset=t])
    var w = wr[unsafe_offset=t]
    var zr = z.unsafe_offset(t * v)
    var qr = zt.unsafe_offset(t * v)
    if y < 0 or w == 0:
        for j in range(lane, v, 32):
            comptime if f32:
                dz[unsafe_offset=t * v + j] = 0
            else:
                dz16[unsafe_offset=t * v + j] = Float32(0).cast[bf]()
        if lane == 0:
            loss[unsafe_offset=t] = 0
        return
    comptime if CAP > 0:
        for j in range(lane, v, 32):
            zr[j] = Float32(CAP) * tanh(zr[j] / Float32(CAP))
    var m: Float32 = -3.0e38
    var mq: Float32 = -3.0e38
    for j in range(lane, v, 32):
        m = max(m, zr[j])
        mq = max(mq, qr[j].cast[DType.float32]())
    m = warp.max(m)
    mq = warp.max(mq)
    var s: Float32 = 0
    var sq: Float32 = 0
    for j in range(lane, v, 32):
        s += exp(zr[j] - m)
        sq += exp(qr[j].cast[DType.float32]() - mq)
    s = warp.sum(s)
    sq = warp.sum(sq)
    var inv = 1 / s
    var iq = 1 / sq
    var ls = log(s) + m  # log-sum-exp of the student row
    var lq = log(sq) + mq  # and of the teacher row
    var kl: Float32 = 0
    for j in range(lane, v, 32):
        var lqj = qr[j].cast[DType.float32]() - lq
        kl += exp(lqj) * (lqj - (zr[j] - ls))
    kl = warp.sum(kl)
    for j in range(lane, v, 32):
        var qj = exp(qr[j].cast[DType.float32]() - mq) * iq
        var gj = (exp(zr[j] - m) * inv - (1 - lam) * Float32(Int(j == y)) - lam * qj) * w
        comptime if CAP > 0:
            var th = zr[j] / Float32(CAP)
            gj *= 1 - th * th
        comptime if f32:
            dz[unsafe_offset=t * v + j] = gj
        else:
            dz16[unsafe_offset=t * v + j] = gj.cast[bf]()
    if lane == 0:
        loss[unsafe_offset=t] = ((1 - lam) * (ls - zr[y]) + lam * kl) * w


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
    """dst[j] += sum_i src[i, j]; loads issue 8 rows at a time (one wait per 8, not per row), adds stay in row order."""
    var j = global_idx.x
    var d = Int(d32)
    if j >= d:
        return
    var nb = Int(b32)
    var s: Float32 = 0
    var i = 0
    while i + 8 <= nb:
        var v = SIMD[DType.float32, 8](0)
        comptime for q in range(8):
            v[q] = src[unsafe_offset=(i + q) * d + j]
        comptime for q in range(8):
            s += v[q]
        i += 8
    while i < nb:
        s += src[unsafe_offset=i * d + j]
        i += 1
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
    # grid-stride: thread c reads c, c + T, c + 2T ... (T = ceildiv(n, CH) threads), so a wave's loads coalesce.
    # Chunked c*CH.. made neighbouring threads CH floats apart: 9.4 ms for 63M params, ~3% of bandwidth.
    var t = (n + CH - 1) // CH
    var s: Float32 = 0
    for i in range(c, n, t):
        var x = g[unsafe_offset=i]
        s += x * x
    part[unsafe_offset=c] = s


def sumsqb_k(g: F, part: F, n32: Int32):
    """sumsq_k reduced per 256-thread block: clip_k then sums ceildiv(n / CH, 256) partials, not one per thread
    (one workgroup over 247k partials took 184 us at 63M)."""
    var c = global_idx.x
    var tid = Int(thread_idx.x)
    var n = Int(n32)
    var red = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[256]())
    var t = (n + CH - 1) // CH
    var s: Float32 = 0
    if c * CH < n:
        for i in range(c, n, t):
            var x = g[unsafe_offset=i]
            s += x * x
    red[tid] = rebind[red.ElementType](s)
    barrier()
    var w = 128
    while w > 0:
        if tid < w:
            red[tid] = red[tid] + red[tid + w]
        barrier()
        w //= 2
    if tid == 0:
        part[unsafe_offset=Int(block_idx.x)] = rebind[Float32](red[0])


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


def adamw16_k(p: F, g: F, m: F, v: F, sc: F, w16: H16, n32: Int32, lr: Float32, b1: Float32, b2: Float32, eps: Float32, wd: Float32, c1: Float32, c2: Float32):
    """adamw_k plus the bf16 copy of the new p (the GEMMs' weight operand): refresh then skips its full cast."""
    var i = global_idx.x
    if i >= Int(n32):
        return
    var gi = g[unsafe_offset=i] * sc[unsafe_offset=1]
    var mi = b1 * m[unsafe_offset=i] + (1 - b1) * gi
    var vi = b2 * v[unsafe_offset=i] + (1 - b2) * gi * gi
    m[unsafe_offset=i] = mi
    v[unsafe_offset=i] = vi
    var pi = p[unsafe_offset=i] * (1 - lr * wd)
    var pn = pi - lr * (mi * c1) / (sqrt(vi * c2) + eps)
    p[unsafe_offset=i] = pn
    w16[unsafe_offset=i] = pn.cast[bf]()


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


comptime TT = 32  # tr16t_k tile; block (TT, 8): each thread moves TT / 8 elements in and out


def tr16t_k(x: F, y: H16, rows32: Int32, cols32: Int32):
    """y[cols, rows] = bf16(x[rows, cols]^T) through a 32 x 33 LDS tile: reads and writes both coalesce.
    tr16_k's direct form writes rows apart (one cache line per element): 135.7 of 218 ms/step at 63M dW."""
    var rows = Int(rows32)
    var cols = Int(cols32)
    var tile = stack_allocation[DType.float32, address_space=AddressSpace.SHARED](row_major[TT, TT + 1]())
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var c0 = Int(block_idx.x) * TT
    var r0 = Int(block_idx.y) * TT
    for j in range(ty, TT, 8):
        var r = r0 + j
        var c = c0 + tx
        if r < rows and c < cols:
            tile[j, tx] = rebind[tile.ElementType](x[unsafe_offset=r * cols + c])
    barrier()
    for j in range(ty, TT, 8):
        var c = c0 + j
        var r = r0 + tx
        if c < cols and r < rows:
            y[unsafe_offset=c * rows + r] = rebind[Scalar[DType.float32]](tile[tx, j]).cast[DType.bfloat16]()


def tr16h_k(x: H16, y: H16, rows32: Int32, cols32: Int32):
    """tr16t_k for a bf16 source: half the read bytes when the producer already wrote bf16."""
    var rows = Int(rows32)
    var cols = Int(cols32)
    var tile = stack_allocation[DType.bfloat16, address_space=AddressSpace.SHARED](row_major[TT, TT + 2]())
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var c0 = Int(block_idx.x) * TT
    var r0 = Int(block_idx.y) * TT
    for j in range(ty, TT, 8):
        var r = r0 + j
        var c = c0 + tx
        if r < rows and c < cols:
            tile[j, tx] = rebind[tile.ElementType](x[unsafe_offset=r * cols + c])
    barrier()
    for j in range(ty, TT, 8):
        var c = c0 + j
        var r = r0 + tx
        if c < cols and r < rows:
            y[unsafe_offset=c * rows + r] = rebind[Scalar[DType.bfloat16]](tile[tx, j])


def tr16_k(x: F, y: H16, rows32: Int32, cols32: Int32):
    var i = global_idx.x
    var rows = Int(rows32)
    var cols = Int(cols32)
    if i < rows * cols:
        y[unsafe_offset=(i % cols) * rows + i // cols] = x[unsafe_offset=i].cast[DType.bfloat16]()


comptime WB = 128  # dwg_k block (M and N)
comptime WK = 64  # dwg_k K tile
comptime WS = WK + 8  # LDS row stride (bf16): spreads fragment rows over banks
comptime WT = 256  # dwg_k threads: 8 waves as 4 x 2, warp tile 32 x 64


@always_inline
def _ld8[dt: DType](p: MutPointer[Scalar[dt], MutUntrackedOrigin], off: Int, ok: Bool) -> SIMD[bf, 8]:
    if not ok:
        return SIMD[bf, 8](0)
    return p.unsafe_offset(off).unsafe_load[width=8]().cast[bf]()


def dwg_k[ta: DType, tb: DType, W4: Bool](a: MutPointer[Scalar[ta], MutUntrackedOrigin], b0: MutPointer[Scalar[tb], MutUntrackedOrigin],
        b1: MutPointer[Scalar[tb], MutUntrackedOrigin], b2: MutPointer[Scalar[tb], MutUntrackedOrigin], c0: F, c1: F, c2: F,
        m32: Int32, n32: Int32, k32: Int32, acc: Int32, ks: Int32, ldb32: Int32):
    """dW GEMM without transposes: c[M, N] (+)= a^T b with a [K, M] and b [K, N] row-major (activations and grads
    as the forward/backward wrote them, fp32 or bf16; cast to bf16 at the LDS store). block_idx.z picks (b, c), or
    with ks > 1 the K slice (c_z gets that slice's partial).
    Block 128 x 128. W4: 4 waves 2 x 2 with 64 x 64 warp tiles, K tile 32 (half the LDS fragment loads per WMMA);
    else 8 waves 4 x 2 with 32 x 64 warp tiles, K tile 64.
    Each thread loads rows k, k+1 of two 8-wide column chunks per operand and stores them as (k, k+1) pairs into
    [row][k] LDS tiles (lanes walk k: conflict-free; lanes walking columns put rows 8 apart on one bank, 2x slower),
    so the WMMA fragments are 16 contiguous k per row (the linalg transpose_b layout).
    The next K tile's global loads stay in registers during the current tile's WMMA. Needs M, N % 8 == 0, K % WK == 0."""
    comptime NT = 128 if W4 else 256
    comptime KT = 32 if W4 else 64
    comptime KS = KT + 8
    comptime KP = KT // 2
    comptime TM = 4 if W4 else 2
    var M = Int(m32)
    var N = Int(n32)
    var K = Int(k32)
    var z = Int(block_idx.z)
    var b = b0 if z == 0 else (b1 if z == 1 else b2)
    var c = c0 if z == 0 else (c1 if z == 1 else c2)
    var LB = Int(ldb32) if ldb32 > 0 else N  # b row stride (3D for the interleaved q/k/v grads)
    var ap = a
    if ks > 1:
        K = K // Int(ks)
        var kz = z * K
        ap = a.unsafe_offset(kz * M)
        b = b0.unsafe_offset(kz * LB)
    var tid = Int(thread_idx.x)
    var lane = tid % 32
    var wid = tid // 32
    var wm = wid // 2
    var wn = wid % 2
    var hl = lane % 16
    var half = lane // 16
    var m0 = Int(block_idx.y) * WB
    var n0 = Int(block_idx.x) * WB
    var sa = stack_allocation[bf, address_space=AddressSpace.SHARED](row_major[WB, KS]()).ptr
    var sb = stack_allocation[bf, address_space=AddressSpace.SHARED](row_major[WB, KS]()).ptr
    var kp = 2 * (tid % KP)
    var cc = (tid // KP) * 8
    var aok = m0 + cc < M
    var bok = n0 + cc < N
    var aok2 = m0 + cc + 64 < M
    var bok2 = n0 + cc + 64 < N
    var accv = Array[SIMD[DType.float32, 8], TM * 4](fill=SIMD[DType.float32, 8](0))
    var ra0 = _ld8[ta](ap, kp * M + m0 + cc, aok)
    var ra1 = _ld8[ta](ap, (kp + 1) * M + m0 + cc, aok)
    var rb0 = _ld8[tb](b, kp * LB + n0 + cc, bok)
    var rb1 = _ld8[tb](b, (kp + 1) * LB + n0 + cc, bok)
    var ra2 = _ld8[ta](ap, kp * M + m0 + cc + 64, aok2)
    var ra3 = _ld8[ta](ap, (kp + 1) * M + m0 + cc + 64, aok2)
    var rb2 = _ld8[tb](b, kp * LB + n0 + cc + 64, bok2)
    var rb3 = _ld8[tb](b, (kp + 1) * LB + n0 + cc + 64, bok2)
    var nk = K // KT
    for kt in range(nk):
        comptime for i in range(8):
            sa.unsafe_store((cc + i) * KS + kp, SIMD[bf, 2](ra0[i], ra1[i]))
            sb.unsafe_store((cc + i) * KS + kp, SIMD[bf, 2](rb0[i], rb1[i]))
            sa.unsafe_store((cc + 64 + i) * KS + kp, SIMD[bf, 2](ra2[i], ra3[i]))
            sb.unsafe_store((cc + 64 + i) * KS + kp, SIMD[bf, 2](rb2[i], rb3[i]))
        barrier()
        if kt + 1 < nk:
            var k1 = (kt + 1) * KT + kp
            ra0 = _ld8[ta](ap, k1 * M + m0 + cc, aok)
            ra1 = _ld8[ta](ap, (k1 + 1) * M + m0 + cc, aok)
            rb0 = _ld8[tb](b, k1 * LB + n0 + cc, bok)
            rb1 = _ld8[tb](b, (k1 + 1) * LB + n0 + cc, bok)
            ra2 = _ld8[ta](ap, k1 * M + m0 + cc + 64, aok2)
            ra3 = _ld8[ta](ap, (k1 + 1) * M + m0 + cc + 64, aok2)
            rb2 = _ld8[tb](b, k1 * LB + n0 + cc + 64, bok2)
            rb3 = _ld8[tb](b, (k1 + 1) * LB + n0 + cc + 64, bok2)
        comptime for kk in range(KT // 16):
            var af = Array[SIMD[bf, 16], TM](fill=SIMD[bf, 16](0))
            comptime for tm in range(TM):
                af[tm] = sa.unsafe_load[width=16]((wm * TM * 16 + tm * 16 + hl) * KS + kk * 16)
            comptime for tn in range(4):
                var bfr = sb.unsafe_load[width=16]((wn * 64 + tn * 16 + hl) * KS + kk * 16)
                comptime for tm in range(TM):
                    mma(accv[tm * 4 + tn], af[tm], bfr, accv[tm * 4 + tn])
        barrier()
    comptime for tm in range(TM):
        comptime for tn in range(4):
            comptime for i in range(8):
                var row = _acc_row(m0 + wm * TM * 16 + tm * 16, i, half)
                var col = n0 + wn * 64 + tn * 16 + hl
                if row < M and col < N:
                    var o = row * N + col
                    var v = accv[tm * 4 + tn][i]
                    c[unsafe_offset=o] = c[unsafe_offset=o] + v if acc != 0 else v

def tradd_k(src: F, dst: F, m32: Int32):
    """dst[:, :m] += src^T: src [m, DM] (a dW^T from dwg), dst a [DM, FF] weight gradient (Matryoshka prefix columns)."""
    comptime FFW = get_defined_int["LAT_FF", DM * 43 // 16]()
    var t = global_idx.x
    if t >= Int(m32) * DM:
        return
    var j = t // DM
    var i = t % DM
    dst[unsafe_offset=i * FFW + j] += src[unsafe_offset=t]


comptime BHK = get_defined_int["LAT_BH", 0]()  # block head: K - 1 extra heads predict t + 2 .. t + K (0 / 1: off)
comptime BHS = get_defined_int["LAT_BHS", 8]()  # block heads train on every BHS-th position (cost / BHS)


def bh_gather_k(nf: F, tgt: I, wr: F, x16: H16, tk: I, wk: F, nc32: Int32, b32: Int32, sn32: Int32, k32: Int32, lam: Float32):
    """Compact row c = j b + q <- row (j BHS) b + q of nf (bf16); its target is the token k - 1 rows of positions later."""
    var i = global_idx.x
    if i >= Int(nc32) * DM:
        return
    var c = i // DM
    var d = i % DM
    var b = Int(b32)
    var j = c // b
    var row = j * BHS * b + c % b
    x16[unsafe_offset=i] = nf[unsafe_offset=row * DM + d].cast[DType.bfloat16]()
    if d == 0:
        var pos = j * BHS + Int(k32) - 1
        var tr = row + (Int(k32) - 1) * b
        var ok = pos < Int(sn32)
        if ok:
            ok = tgt[unsafe_offset=tr] >= 0 and wr[unsafe_offset=tr] != 0
        tk[unsafe_offset=c] = tgt[unsafe_offset=tr] if ok else Int32(-1)
        wk[unsafe_offset=c] = lam * wr[unsafe_offset=tr] if ok else Float32(0)


def bh_act_k(nf: F, u: F, h16: H16, nc32: Int32, b32: Int32):
    """h = x + silu(u) (x the gathered nf row), bf16 for the head GEMMs."""
    var i = global_idx.x
    if i >= Int(nc32) * DM:
        return
    var c = i // DM
    var b = Int(b32)
    var row = (c // b) * BHS * b + c % b
    var uu = u[unsafe_offset=i]
    h16[unsafe_offset=i] = (nf[unsafe_offset=row * DM + i % DM] + uu / (1 + exp(-uu))).cast[DType.bfloat16]()


def bh_du_k(u: F, dh: F, du16: H16, n32: Int32):
    var i = global_idx.x
    if i >= Int(n32):
        return
    var uu = u[unsafe_offset=i]
    var s = 1 / (1 + exp(-uu))
    du16[unsafe_offset=i] = (dh[unsafe_offset=i] * s * (1 + uu * (1 - s))).cast[DType.bfloat16]()


def bh_scat_k(dh: F, dxu: F, dnf: F, nc32: Int32, b32: Int32):
    """dnf[row] += dh + du W (residual and MLP paths); rows are distinct per c, no race."""
    var i = global_idx.x
    if i >= Int(nc32) * DM:
        return
    var c = i // DM
    var b = Int(b32)
    var row = (c // b) * BHS * b + c % b
    dnf[unsafe_offset=row * DM + i % DM] += dh[unsafe_offset=i] + dxu[unsafe_offset=i]
