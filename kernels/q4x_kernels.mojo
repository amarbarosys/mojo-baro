from std.memory import bitcast
from std.math import fma, log1p, sqrt, cos, sin
from bexp import exp
from std.atomic import Atomic, Ordering, fence
from max.gpu import block_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.sys import get_defined_int

comptime FUSE_MUTANT = get_defined_int["Q4X_FUSE_MUTANT", 0]()  # planted bugs for bench/check-fuse-mutants.sh: 1 hcdown inject, 2 hcup cur, 3 gnorm_ffn combine
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from moe import (
    KLane, q4k_dot_blocks, q4k_value, q6k_row_dot, iq4nl_half,
    q5k_lane, iq2xxs_lane, iq1m_lane, q3k_lane, iq4xs_lane, Q5K_BYTES, IQ2XXS_BYTES, IQ1M_BYTES, Q3K_BYTES, IQ4XS_BYTES,
    q3k_scale, q4k_sm_w, iq4nl_i8, f16_at, Q4K_BYTES, Q6K_BYTES,
)

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime u8 = DType.uint8
comptime i8 = DType.int8
comptime i32 = DType.int32
comptime Q2_0_BYTES = 18
comptime FP = MutPointer[Scalar[f32], MutAnyOrigin]
comptime BP = MutPointer[Scalar[bf16], MutAnyOrigin]
comptime UP = MutPointer[Scalar[u8], MutAnyOrigin]
comptime IP = MutPointer[Scalar[i32], MutAnyOrigin]
comptime LP = MutPointer[Scalar[DType.int64], MutAnyOrigin]
comptime Q4X_H = 2560
comptime Q4X_HC = 4
comptime Q4X_HCD = Q4X_HC * Q4X_H
comptime Q4X_HD = 256
comptime Q4X_SD = 128
comptime Q4X_EPS = Float32(1e-6)
comptime Q4X_CONV = 10240
comptime Q4X_IO_DEC = 10240
comptime Q4X_IO_BETA = 10288
comptime Q4X_IO_O = 10336
comptime Q4X_SEL = 16


@always_inline
def xf[R: Int, XL: TensorLayout](X: TileTensor[bf16, XL, MutAnyOrigin], rows: SIMD[i32, R], c: Int, r: Int) -> SIMD[f32, 16]:
    return rebind[SIMD[bf16, 16]](X.vectorize[1, 16]()[Int(rows[r]), c]).cast[f32]()


@always_inline
def wsum[R: Int](acc: Array[SIMD[f32, 16], R]) -> SIMD[f32, R]:
    var out = SIMD[f32, R](0)
    comptime for r in range(R):
        out[r] = warp.sum(acc[r].reduce_add())
    return out


@always_inline
def q4_0_half(w: UP, base: Int, h: Int) -> SIMD[f32, 16]:
    var q = w.unsafe_offset(base + 2).load[width=16]()
    var n = (q >> 4) if h == 1 else (q & 15)
    return SIMD[f32, 16](f16_at(w, base)) * (n.cast[f32]() - 8)


@always_inline
def q5_0_half(w: UP, base: Int, h: Int) -> SIMD[f32, 16]:
    var qh = w.unsafe_offset(base + 2 + 2 * h).load[width=2]()
    var q = w.unsafe_offset(base + 6).load[width=16]()
    var n = (q >> 4) if h == 1 else (q & 15)
    var v = SIMD[f32, 16](0)
    comptime for j in range(16):
        v[j] = Scalar[f32](Int(n[j]) | (Int((qh[j // 8] >> UInt8(j % 8)) & 1) << 4)) - 16
    return SIMD[f32, 16](f16_at(w, base)) * v


@always_inline
def q8_0_half(w: UP, base: Int, h: Int) -> SIMD[f32, 16]:
    var q = w.unsafe_offset(base + 2 + h * 16).unsafe_bitcast[Scalar[i8]]().load[width=16]()
    return (SIMD[f32, 16](f16_at(w, base)) * q.cast[f32]()).cast[bf16]().cast[f32]()


@always_inline
def b32_rows_dot[half_fn: def(UP, Int, Int) thin -> SIMD[f32, 16], BYTES: Int, R: Int, XL: TensorLayout](
    X: TileTensor[bf16, XL, MutAnyOrigin], W: UP, rows: SIMD[i32, R], rb: Int, k: Int
) -> SIMD[f32, R]:
    var lane = Int(lane_id())
    var nb = k // 32
    var acc = Array[SIMD[f32, 16], R](fill=SIMD[f32, 16](0))
    var b = lane // 2
    var h = lane % 2
    while b < nb:
        var v = half_fn(W, rb + b * BYTES, h)
        comptime for r in range(R):
            acc[r] = fma(v, xf(X, rows, 2 * b + h, r), acc[r])
        b += 16
    return wsum(acc)


@always_inline
def bf16_rows_dot[R: Int, XL: TensorLayout](X: TileTensor[bf16, XL, MutAnyOrigin], W: UP, rows: SIMD[i32, R], rb: Int, k: Int) -> SIMD[f32, R]:
    var wp = W.unsafe_offset(rb).unsafe_bitcast[Scalar[bf16]]()
    var acc = Array[SIMD[f32, 16], R](fill=SIMD[f32, 16](0))
    var c = Int(lane_id())
    while c < k // 16:
        var v = wp.unsafe_offset(c * 16).load[width=16]().cast[f32]()
        comptime for r in range(R):
            acc[r] = fma(v, xf(X, rows, c, r), acc[r])
        c += WARP_SIZE
    return wsum(acc)


@always_inline
def kq_rows_dot[lane_fn: def(UP, Int, Int) thin -> KLane, BYTES: Int, R: Int, XL: TensorLayout](
    X: TileTensor[bf16, XL, MutAnyOrigin], W: UP, rows: SIMD[i32, R], rb: Int, k: Int
) -> SIMD[f32, R]:
    var lane = Int(lane_id())
    var nb = k // 256
    var acc = Array[SIMD[f32, 16], R](fill=SIMD[f32, 16](0))
    var b = lane // 8
    var s = lane % 8
    while b < nb:
        var r = lane_fn(W, rb + b * BYTES, s)
        var v0 = SIMD[f32, 16](r[2]) * r[0].cast[f32]() - SIMD[f32, 16](r[4])
        var v1 = SIMD[f32, 16](r[3]) * r[1].cast[f32]() - SIMD[f32, 16](r[5])
        var c0 = (b * 256 + r[6]) // 16
        var c1 = (b * 256 + r[7]) // 16
        comptime for t in range(R):
            acc[t] = fma(v0, xf(X, rows, c0, t), fma(v1, xf(X, rows, c1, t), acc[t]))
        b += 4
    return wsum(acc)


@always_inline
def q2_0_rows_dot[R: Int, XL: TensorLayout](X: TileTensor[bf16, XL, MutAnyOrigin], w: UP, rows: SIMD[i32, R], rb: Int, k: Int, L: Int) -> SIMD[f32, R]:
    var X64 = X.vectorize[1, 64]()
    var nb = k // 64
    var acc = Array[SIMD[f32, 16], R](fill=SIMD[f32, 16](0))
    var b = Int(lane_id()) % L
    while b < nb:
        var base = rb + b * Q2_0_BYTES
        var d = f16_at(w, base)
        var q = w.unsafe_offset(base + 2).load[width=16]()
        var v0 = (q & 3).cast[f32]() - 1
        var v1 = ((q >> 2) & 3).cast[f32]() - 1
        var v2 = ((q >> 4) & 3).cast[f32]() - 1
        var v3 = (q >> 6).cast[f32]() - 1
        comptime for r in range(R):
            var x = rebind[SIMD[bf16, 64]](X64[Int(rows[r]), b]).cast[f32]()
            var xe_xo = x.deinterleave()
            var x0_x2 = xe_xo[0].deinterleave()
            var x1_x3 = xe_xo[1].deinterleave()
            acc[r] = acc[r] + fma(v0, x0_x2[0], fma(v1, x1_x3[0], fma(v2, x0_x2[1], v3 * x1_x3[1]))) * d
        b += L
    var out = SIMD[f32, R](0)
    comptime for r in range(R):
        var v = acc[r].reduce_add()
        var off = 1
        while off < L:
            v += warp.shuffle_xor(v, UInt32(off))
            off *= 2
        out[r] = v
    return out


@always_inline
def q4x_dots[GT: Int, R: Int, XL: TensorLayout](X: TileTensor[bf16, XL, MutAnyOrigin], w: UP, rows: SIMD[i32, R], rb: Int, k: Int, L: Int) -> SIMD[f32, R]:
    comptime if GT == 42:
        return q2_0_rows_dot[R](X, w, rows, rb, k, L)
    elif GT == 30:
        return bf16_rows_dot[R](X, w, rows, rb, k)
    elif GT == 13:
        return kq_rows_dot[q5k_lane, Q5K_BYTES, R](X, w, rows, rb, k)
    elif GT == 16:
        return kq_rows_dot[iq2xxs_lane, IQ2XXS_BYTES, R](X, w, rows, rb, k)
    elif GT == 29:
        return kq_rows_dot[iq1m_lane, IQ1M_BYTES, R](X, w, rows, rb, k)
    elif GT == 11:
        return kq_rows_dot[q3k_lane, Q3K_BYTES, R](X, w, rows, rb, k)
    elif GT == 23:
        return kq_rows_dot[iq4xs_lane, IQ4XS_BYTES, R](X, w, rows, rb, k)
    elif GT == 20:
        return b32_rows_dot[iq4nl_half, 18, R](X, w, rows, rb, k)
    elif GT == 8:
        return b32_rows_dot[q8_0_half, 34, R](X, w, rows, rb, k)
    elif GT == 2:
        return b32_rows_dot[q4_0_half, 18, R](X, w, rows, rb, k)
    elif GT == 6:
        return b32_rows_dot[q5_0_half, 22, R](X, w, rows, rb, k)
    elif GT == 12:
        comptime assert R == 1
        return SIMD[f32, R](q4k_dot_blocks(X, w, Int(rows[0]), rb, k))
    else:
        comptime assert R == 1 and GT == 14
        return SIMD[f32, R](q6k_row_dot(X, w, Int(rows[0]), rb, k))


comptime KS16 = Tuple[SIMD[i8, 16], Scalar[f32], Scalar[f32], Int]


@always_inline
def q3k_s16(w: UP, base: Int, s: Int) -> KS16:
    var h = s // 8
    var j = (s % 8) // 2
    var t = s % 2
    var qs = w.unsafe_offset(base + 32 + h * 32 + 16 * t).load[width=16]()
    var hm = w.unsafe_offset(base + 16 * t).load[width=16]()
    var q = ((qs >> UInt8(2 * j)) & 3) | (((hm >> UInt8(4 * h + j)) & 1) << 2)
    var dsc = f16_at(w, base + 108) * Scalar[f32](q3k_scale(w, base, h * 8 + 2 * j + t))
    return (bitcast[i8, 16](q) - 4, dsc, Scalar[f32](0), h * 128 + j * 32 + 16 * t)


@always_inline
def q6k_s16(w: UP, base: Int, s: Int) -> KS16:
    var h = s // 8
    var g = (s % 8) // 2
    var t = s % 2
    var ql = w.unsafe_offset(base + h * 64 + (g % 2) * 32 + 16 * t).load[width=16]()
    var qh = w.unsafe_offset(base + 128 + h * 32 + 16 * t).load[width=16]()
    var nib = (ql & 0x0F) if g < 2 else (ql >> 4)
    var q = nib | (((qh >> UInt8(2 * g)) & 3) << 4)
    var sc = bitcast[i8, 1](SIMD[u8, 1](w[unsafe_offset=base + 192 + h * 8 + t + 2 * g]))[0]
    var dsc = f16_at(w, base + 208) * Scalar[f32](Int(sc))
    return (bitcast[i8, 16](q) - 32, dsc, Scalar[f32](0), h * 128 + g * 32 + 16 * t)


@always_inline
def q4k_s16(w: UP, base: Int, s: Int) -> KS16:
    var pair = s // 4
    var half = ((s % 4) // 2) * 16
    var nib = s % 2
    var hdr = w.unsafe_offset(base).load[width=16]()
    var hw = bitcast[DType.uint32, 4](hdr)
    var dd = bitcast[DType.float16, 2](hw[0])
    var sm = q4k_sm_w(hw, 2 * pair + nib)
    var qb = w.unsafe_offset(base + 16 + pair * 32 + half).load[width=16]()
    var q = (qb & 0x0F) if nib == 0 else (qb >> 4)
    return (bitcast[i8, 16](q), dd[0].cast[f32]() * Scalar[f32](sm[0]), dd[1].cast[f32]() * Scalar[f32](sm[1]), pair * 64 + nib * 32 + half)


@always_inline
def q5k_s16(w: UP, base: Int, s: Int) -> KS16:
    var pair = s // 4
    var half = ((s % 4) // 2) * 16
    var nib = s % 2
    var hw = bitcast[DType.uint32, 4](w.unsafe_offset(base).load[width=16]())
    var dd = bitcast[DType.float16, 2](hw[0])
    var sm = q4k_sm_w(hw, 2 * pair + nib)
    var qb = w.unsafe_offset(base + 48 + pair * 32 + half).load[width=16]()
    var qh = w.unsafe_offset(base + 16 + half).load[width=16]()
    var q = ((qb & 0x0F) | (((qh >> UInt8(2 * pair)) & 1) << 4)) if nib == 0 else ((qb >> 4) | (((qh >> UInt8(2 * pair + 1)) & 1) << 4))
    return (bitcast[i8, 16](q), dd[0].cast[f32]() * Scalar[f32](sm[0]), dd[1].cast[f32]() * Scalar[f32](sm[1]), pair * 64 + nib * 32 + half)


@always_inline
def iq4xs_s16(w: UP, base: Int, s: Int) -> KS16:
    var s8 = s // 2
    var nib = s % 2
    var sh = Int(w.unsafe_offset(base + 2).unsafe_bitcast[Scalar[DType.uint16]]()[])
    var ls = ((Int(w[unsafe_offset=base + 4 + s8 // 2]) >> (4 * (s8 % 2))) & 0xF) | (((sh >> (2 * s8)) & 3) << 4)
    var dl = f16_at(w, base) * Scalar[f32](ls - 32)
    var q = w.unsafe_offset(base + 8 + s8 * 16).load[width=16]()
    var qv = iq4nl_i8(q & 15) if nib == 0 else iq4nl_i8(q >> 4)
    return (qv, dl, Scalar[f32](0), s8 * 32 + nib * 16)


@always_inline
def kq2_rows_dot[lane16: def(UP, Int, Int) thin -> KS16, BYTES: Int, R: Int, XL: TensorLayout](
    X: TileTensor[bf16, XL, MutAnyOrigin], W: UP, rows: SIMD[i32, R], rb0: Int, rb1: Int, k: Int
) -> Tuple[SIMD[f32, R], SIMD[f32, R]]:
    var lane = Int(lane_id())
    var s = lane % 16
    var nb = k // 256
    var acc0 = Array[SIMD[f32, 16], R](fill=SIMD[f32, 16](0))
    var acc1 = Array[SIMD[f32, 16], R](fill=SIMD[f32, 16](0))
    var b = lane // 16
    while b < nb:
        var r0 = lane16(W, rb0 + b * BYTES, s)
        var r1 = lane16(W, rb1 + b * BYTES, s)
        var c = (b * 256 + r0[3]) // 16
        var v0 = r0[0].cast[f32]() * r0[1] - r0[2]
        var v1 = r1[0].cast[f32]() * r1[1] - r1[2]
        comptime for t in range(R):
            var xk = xf(X, rows, c, t)
            acc0[t] = fma(v0, xk, acc0[t])
            acc1[t] = fma(v1, xk, acc1[t])
        b += 2
    return (wsum(acc0), wsum(acc1))


@always_inline
def kq2_dots[GT: Int, R: Int, XL: TensorLayout](X: TileTensor[bf16, XL, MutAnyOrigin], W: UP, rows: SIMD[i32, R], rb0: Int, rb1: Int, k: Int) -> Tuple[SIMD[f32, R], SIMD[f32, R]]:
    comptime if GT == 11:
        return kq2_rows_dot[q3k_s16, Q3K_BYTES, R](X, W, rows, rb0, rb1, k)
    elif GT == 14:
        return kq2_rows_dot[q6k_s16, Q6K_BYTES, R](X, W, rows, rb0, rb1, k)
    elif GT == 12:
        return kq2_rows_dot[q4k_s16, Q4K_BYTES, R](X, W, rows, rb0, rb1, k)
    elif GT == 13:
        return kq2_rows_dot[q5k_s16, Q5K_BYTES, R](X, W, rows, rb0, rb1, k)
    else:
        return kq2_rows_dot[iq4xs_s16, IQ4XS_BYTES, R](X, W, rows, rb0, rb1, k)


@always_inline
def q4x_sigmoid(x: Float32) -> Float32:
    return 1 / (1 + exp(-x))


@always_inline
def q4x_silu(x: Float32) -> Float32:
    return x / (1 + exp(-x))


@always_inline
def q4x_emit(y: FP, xq: BP, o: Int, xo: Int, row: Int, s: Float32, act: Int32, scale: Float32):
    y[unsafe_offset=o + row] = s
    if act != 0:
        xq[unsafe_offset=xo + row] = q4x_silu(s * scale).cast[bf16]()


def amar_q4x_gemm[GT: Int, R: Int, KQ2: Bool, XL: TensorLayout](
    X: TileTensor[bf16, XL, MutAnyOrigin],
    w: UP,
    tab: LP,
    tiles: IP,
    xmap: IP,
    ymap: IP,
    y: FP,
    xq: BP,
    k: Int32,
    n: Int32,
    nsub: Int32,
    rowb: Int32,
    L: Int32,
    yst: Int32,
    xqst: Int32,
    act: Int32,
    scale: Float32,
    flags: Int32,
):
    var tl = Int(block_idx.y) * 4
    var grp = Int(tiles[unsafe_offset=tl])
    var r0 = Int(tiles[unsafe_offset=tl + 1])
    var cnt = Int(tiles[unsafe_offset=tl + 2])
    var lane = Int(lane_id())
    var pw = 2 if KQ2 else WARP_SIZE // Int(L)
    var row = (Int(block_idx.x) * 8 + Int(thread_idx.x) // WARP_SIZE) * pw + (0 if KQ2 else lane // Int(L))
    if row >= Int(n):
        return
    var tsub = Int(n) // Int(nsub)
    var rows = SIMD[i32, R](0)
    comptime for i in range(R):
        var g = r0 + min(i, cnt - 1)
        rows[i] = Int32(g) if (Int(flags) & 1) == 0 else xmap[unsafe_offset=g]
    var wb = Int(tab[unsafe_offset=grp * tsub + row // Int(nsub)]) + (row % Int(nsub)) * Int(rowb)
    comptime if KQ2:
        var row1 = row + 1 if row + 1 < Int(n) else row
        var wb1 = Int(tab[unsafe_offset=grp * tsub + row1 // Int(nsub)]) + (row1 % Int(nsub)) * Int(rowb)
        var s = kq2_dots[GT, R](X, w, rows, wb, wb1, Int(k))
        if lane == 0:
            comptime for i in range(R):
                if i < cnt:
                    var orow = r0 + i if (Int(flags) & 2) == 0 else Int(ymap[unsafe_offset=r0 + i])
                    q4x_emit(y, xq, orow * Int(yst), orow * Int(xqst), row, s[0][i], act, scale)
                    if row1 != row:
                        q4x_emit(y, xq, orow * Int(yst), orow * Int(xqst), row1, s[1][i], act, scale)
    else:
        var s = q4x_dots[GT, R](X, w, rows, wb, Int(k), Int(L))
        if lane % Int(L) == 0:
            comptime for i in range(R):
                if i < cnt:
                    var orow = r0 + i if (Int(flags) & 2) == 0 else Int(ymap[unsafe_offset=r0 + i])
                    q4x_emit(y, xq, orow * Int(yst), orow * Int(xqst), row, s[i], act, scale)


comptime WP16 = MutPointer[Scalar[DType.uint16], MutAnyOrigin]


@always_inline
def fmv_dot[R: Int](wr: WP16, x: FP, xst: Int, r0: Int, cnt: Int, k: Int) -> SIMD[f32, R]:
    var acc = Array[SIMD[f32, 16], R](fill=SIMD[f32, 16](0))
    for i in range(Int(lane_id()) * 16, k, WARP_SIZE * 16):
        var wv = bitcast[f32, 16](wr.unsafe_offset(i).load[width=16]().cast[DType.uint32]() << 16)
        comptime for r in range(R):
            acc[r] = wv.fma(x.unsafe_offset((r0 + min(r, cnt - 1)) * xst + i).load[width=16](), acc[r])
    return wsum(acc)


def amar_q4x_fmv[R: Int](w: WP16, w2: WP16, n1: Int32, x: FP, xst: Int32, y: FP, yst: Int32, k: Int32, n: Int32, T: Int32):
    var row = Int(block_idx.x) * 8 + Int(thread_idx.x) // WARP_SIZE
    if row >= Int(n):
        return
    var r0 = Int(block_idx.y) * R
    var cnt = min(R, Int(T) - r0)
    var wr = w.unsafe_offset(row * Int(k)) if row < Int(n1) else w2.unsafe_offset((row - Int(n1)) * Int(k))
    var s = fmv_dot[R](wr, x, Int(xst), r0, cnt, Int(k))
    if lane_id() == 0:
        comptime for r in range(R):
            if r < cnt:
                y[unsafe_offset=(r0 + r) * Int(yst) + row] = s[r]


def amar_q4x_hcdown[XL: TensorLayout](X: TileTensor[bf16, XL, MutAnyOrigin], w: UP, rowb: Int32, k: Int32, nd: Int32, y: FP, xq: BP, scale: Float32, wi: WP16, xi: FP, inj: FP, ni: Int32):
    """T = 1 hc down GEMV (BF16, silu epilogue, as amar_q4x_gemm[30]) and the inject fmv in one launch: waves past nd take the inject rows."""
    var row = Int(block_idx.x) * 8 + Int(thread_idx.x) // WARP_SIZE
    if row < Int(nd):
        var s = bf16_rows_dot[1](X, w, SIMD[i32, 1](0), row * Int(rowb), Int(k))
        if lane_id() == 0:
            q4x_emit(y, xq, 0, 0, row, s[0], 1, scale)
    elif row - Int(nd) < Int(ni):
        var s = fmv_dot[1](wi.unsafe_offset((row - Int(nd)) * Int(Q4X_HCD)), xi, 0, 0, 1, Q4X_HCD)
        if lane_id() == 0:
            inj[unsafe_offset=row - Int(nd)] = s[0] if FUSE_MUTANT != 1 else s[0] * 1.0000005


def amar_q4x_q4k_row(w: UP, y: FP, yst: Int32, toks: IP, rowb: Int32, k: Int32):
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    var t = Int(block_idx.y)
    if i < Int(k):
        y[unsafe_offset=t * Int(yst) + i] = q4k_value(w, Int(toks[unsafe_offset=t]) * Int(rowb), i)


def amar_q4x_q3k_row(w: UP, y: FP, yst: Int32, toks: IP, rowb: Int32, k: Int32):
    var tt = Int(block_idx.x) * 64 + Int(thread_idx.x)
    var t = Int(block_idx.y)
    if tt < Int(k) // 256 * 8:
        var b = tt // 8
        var r = q3k_lane(w, Int(toks[unsafe_offset=t]) * Int(rowb) + b * Q3K_BYTES, tt % 8)
        var yp = y.unsafe_offset(t * Int(yst))
        yp.unsafe_offset(b * 256 + r[6]).store(SIMD[f32, 16](r[2]) * r[0].cast[f32]() - SIMD[f32, 16](r[4]))
        yp.unsafe_offset(b * 256 + r[7]).store(SIMD[f32, 16](r[3]) * r[1].cast[f32]() - SIMD[f32, 16](r[5]))


@always_inline
def q4x_bsum[NT: Int](v: Float32) -> Float32:
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[NT // WARP_SIZE]())
    var w = warp.sum(v)
    if lane_id() == 0:
        sums[Int(thread_idx.x) // WARP_SIZE] = rebind[sums.ElementType](w)
    barrier()
    var t = Float32(0)
    comptime for i in range(NT // WARP_SIZE):
        t += rebind[Scalar[f32]](sums[i])
    barrier()
    return t


@always_inline
def q4x_bmax[NT: Int](v: Float32) -> Float32:
    var sums = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[NT // WARP_SIZE]())
    var w = warp.max(v)
    if lane_id() == 0:
        sums[Int(thread_idx.x) // WARP_SIZE] = rebind[sums.ElementType](w)
    barrier()
    var t = rebind[Scalar[f32]](sums[0])
    comptime for i in range(1, NT // WARP_SIZE):
        t = max(t, rebind[Scalar[f32]](sums[i]))
    barrier()
    return t


@always_inline
def gnorm_core(xp: FP, wp: FP, op: FP, xqp: BP, t: Int):
    var ss = Float32(0)
    for i in range(t, Q4X_H, 256):
        var v = xp[unsafe_offset=i]
        ss += v * v
    var s = 1 / sqrt(q4x_bsum[256](ss) / Q4X_H + Q4X_EPS)
    for i in range(t, Q4X_H, 256):
        var v = xp[unsafe_offset=i] * s * wp[unsafe_offset=i]
        op[unsafe_offset=i] = v
        xqp[unsafe_offset=i] = v.cast[bf16]()


def amar_q4x_gnorm(x: FP, xst: Int32, w: FP, o: FP, ost: Int32, xq: BP, xqst: Int32, cur: FP, curst: Int32, inj: FP, injst: Int32, add: Int32):
    var c = Int(block_idx.x)
    var tk = Int(block_idx.y)
    var t = Int(thread_idx.x)
    var xp = x.unsafe_offset(tk * Int(xst) + c * Q4X_H)
    if add != 0:
        var cp = cur.unsafe_offset(tk * Int(curst))
        var wc = q4x_sigmoid(inj[unsafe_offset=tk * Int(injst) + c] * (1.0 / Q4X_HC)) * 2
        for i in range(t, Q4X_H, 256):
            xp[unsafe_offset=i] = xp[unsafe_offset=i] + cp[unsafe_offset=i] * wc
    gnorm_core(xp, w.unsafe_offset(c * Q4X_H), o.unsafe_offset(tk * Int(ost) + c * Q4X_H), xq.unsafe_offset(tk * Int(xqst) + c * Q4X_H), t)


def amar_q4x_gnorm_ffn(x: FP, xst: Int32, w: FP, o: FP, ost: Int32, xq: BP, xqst: Int32, ye: FP, wts: FP, ysh: FP, yst: Int32, sg: FP, sgst: Int32, topk: Int32, inj: FP, injst: Int32):
    """amar_q4x_ffn_out folded into the pending-combine add of amar_q4x_gnorm: the combined branch output never goes to memory."""
    var c = Int(block_idx.x)
    var tk = Int(block_idx.y)
    var t = Int(thread_idx.x)
    var xp = x.unsafe_offset(tk * Int(xst) + c * Q4X_H)
    var wc = q4x_sigmoid(inj[unsafe_offset=tk * Int(injst) + c] * (1.0 / Q4X_HC)) * 2
    for i in range(t, Q4X_H, 256):
        xp[unsafe_offset=i] = xp[unsafe_offset=i] + ffn_cur(ye, wts, ysh, yst, sg, sgst, topk, tk, i) * (wc if FUSE_MUTANT != 3 else wc * 1.0000005)
    gnorm_core(xp, w.unsafe_offset(c * Q4X_H), o.unsafe_offset(tk * Int(ost) + c * Q4X_H), xq.unsafe_offset(tk * Int(xqst) + c * Q4X_H), t)


def amar_q4x_swiglu(y: FP, xq: BP, xqst: Int32, n: Int32, g2: FP, u2: FP, yst2: Int32, xq2: BP, xq2st: Int32, n2: Int32, ff: Int32):
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i < Int(n):
        var g = i // Int(ff)
        var c = i % Int(ff)
        var yp = y.unsafe_offset(g * 2 * Int(ff))
        xq[unsafe_offset=g * Int(xqst) + c] = (q4x_silu(yp[unsafe_offset=c]) * yp[unsafe_offset=Int(ff) + c]).cast[bf16]()
    elif i < Int(n) + Int(n2):
        var j = i - Int(n)
        var t = j // Int(ff)
        var c = j % Int(ff)
        xq2[unsafe_offset=t * Int(xq2st) + c] = (q4x_silu(g2[unsafe_offset=t * Int(yst2) + c]) * u2[unsafe_offset=t * Int(yst2) + c]).cast[bf16]()


def amar_q4x_hcmix(xn: FP, xnst: Int32, g: FP, gst: Int32, cur: FP, curst: Int32, xq: BP, xqst: Int32):
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    var t = Int(block_idx.y)
    var xp = xn.unsafe_offset(t * Int(xnst))
    var gp = g.unsafe_offset(t * Int(gst))
    var acc = xp[unsafe_offset=i] * q4x_sigmoid(gp[unsafe_offset=i])
    for c in range(1, Q4X_HC):
        acc = acc + xp[unsafe_offset=c * Q4X_H + i] * q4x_sigmoid(gp[unsafe_offset=c * Q4X_H + i])
    var v = acc * (1.0 / Q4X_HC)
    cur[unsafe_offset=t * Int(curst) + i] = v
    xq[unsafe_offset=t * Int(xqst) + i] = v.cast[bf16]()


def amar_q4x_hcup[XL: TensorLayout](X: TileTensor[bf16, XL, MutAnyOrigin], w: UP, rowb: Int32, k: Int32, xn: FP, cur: FP, xq: BP):
    """T = 1 hc up GEMV (BF16) and amar_q4x_hcmix in one launch: a wave dots the Q4X_HC gate rows of one output element."""
    var i = Int(block_idx.x) * 8 + Int(thread_idx.x) // WARP_SIZE
    if i >= Q4X_H:
        return
    var acc = Float32(0)
    comptime for c in range(Q4X_HC):
        var g = bf16_rows_dot[1](X, w, SIMD[i32, 1](0), (c * Q4X_H + i) * Int(rowb), Int(k))[0]
        var x = xn[unsafe_offset=c * Q4X_H + i]
        acc = x * q4x_sigmoid(g) if c == 0 else fma(x, q4x_sigmoid(g), acc)
    if lane_id() == 0:
        var v = acc * (1.0 / Q4X_HC)
        comptime if FUSE_MUTANT == 2:
            v *= 1.0000005
        cur[unsafe_offset=i] = v
        xq[unsafe_offset=i] = v.cast[bf16]()


def amar_q4x_bcast(y: FP, yst: Int32, res: FP, rst: Int32):
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    var t = Int(block_idx.y)
    res[unsafe_offset=t * Int(rst) + i] = y[unsafe_offset=t * Int(yst) + i % Q4X_H]


def amar_q4x_ssm_conv(y: FP, yst: Int32, st: FP, cw: FP, io: FP, iost: Int32, alpha: FP, beta: FP, dt: FP, a: FP, seqs: IP, rc: Int32, rw: Int32):
    var h = Int(block_idx.x)
    var j = Int(thread_idx.x)
    var sq = Int(block_idx.y)
    var t0 = Int(seqs[unsafe_offset=2 * sq])
    var t1 = Int(seqs[unsafe_offset=2 * sq + 1])
    var stp = st.unsafe_offset(sq * Int(rw) * 3 * Q4X_CONV)
    var keep = max(Int(rw) - 1, 1)
    if h < 80:
        var c = h * Q4X_SD + j
        var sp0 = stp.unsafe_offset((Int(rc) % Int(rw)) * 3 * Q4X_CONV)
        var s0 = sp0[unsafe_offset=c * 3]
        var s1 = sp0[unsafe_offset=c * 3 + 1]
        var s2 = sp0[unsafe_offset=c * 3 + 2]
        for t in range(t0, t1):
            var xi = y[unsafe_offset=t * Int(yst) + c]
            var acc = s0 * cw[unsafe_offset=c * 4] + s1 * cw[unsafe_offset=c * 4 + 1] + s2 * cw[unsafe_offset=c * 4 + 2] + xi * cw[unsafe_offset=c * 4 + 3]
            var v = q4x_silu(acc)
            s0 = s1
            s1 = s2
            s2 = xi
            if h < 32:
                var ss = q4x_bsum[128](v * v)
                var rs = 1 / max(sqrt(ss), Q4X_EPS)
                io[unsafe_offset=t * Int(iost) + c] = v * rs
            else:
                io[unsafe_offset=t * Int(iost) + c] = v
            if t1 - t <= keep:
                var dp = stp.unsafe_offset(((Int(rc) + t - t0 + 1) % Int(rw)) * 3 * Q4X_CONV)
                dp[unsafe_offset=c * 3] = s0
                dp[unsafe_offset=c * 3 + 1] = s1
                dp[unsafe_offset=c * 3 + 2] = s2
    elif j < 48:
        for t in range(t0, t1):
            var gl = alpha[unsafe_offset=t * Int(yst) + j] + dt[unsafe_offset=j]
            var sp = gl if gl > 20 else log1p(exp(gl))
            io[unsafe_offset=t * Int(iost) + Q4X_IO_DEC + j] = exp(sp * a[unsafe_offset=j])
            io[unsafe_offset=t * Int(iost) + Q4X_IO_BETA + j] = q4x_sigmoid(beta[unsafe_offset=t * Int(yst) + j])


def amar_q4x_delta(st: FP, io: FP, iost: Int32, scale: Float32, z: FP, zst: Int32, nw: FP, xq: BP, xqst: Int32, seqs: IP, rc: Int32, rw: Int32):
    var h = Int(block_idx.x)
    var sq = Int(block_idx.y)
    var t0 = Int(seqs[unsafe_offset=2 * sq])
    var t1 = Int(seqs[unsafe_offset=2 * sq + 1])
    var tt = Int(thread_idx.x)
    var j = tt % 128
    var g = tt // 128
    var kh = h % 16
    var sb = st.unsafe_offset(sq * Int(rw) * 48 * 16384 + h * 16384)
    var part = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[4, 128]())
    for t in range(t0, t1):
        var stp = sb.unsafe_offset(((Int(rc) + t - t0) % Int(rw)) * 48 * 16384)
        var dsp = sb.unsafe_offset(((Int(rc) + t - t0 + 1) % Int(rw)) * 48 * 16384)
        var iop = io.unsafe_offset(t * Int(iost))
        var qn = iop
        var kn = iop.unsafe_offset(2048)
        var v = iop.unsafe_offset(4096)
        var dec = iop[unsafe_offset=Q4X_IO_DEC + h]
        var beta = iop[unsafe_offset=Q4X_IO_BETA + h]
        var o = iop.unsafe_offset(Q4X_IO_O)
        var s = Float32(0)
        for i in range(g * 32, g * 32 + 32):
            s += stp[unsafe_offset=i * 128 + j] * dec * kn[unsafe_offset=kh * 128 + i]
        part[g, j] = rebind[part.ElementType](s)
        barrier()
        var ssum = Float32(0)
        comptime for gg in range(4):
            ssum += rebind[Scalar[f32]](part[gg, j])
        var dl = (v[unsafe_offset=h * 128 + j] - ssum) * beta
        barrier()
        var acc = Float32(0)
        for i in range(g * 32, g * 32 + 32):
            var w = stp[unsafe_offset=i * 128 + j] * dec + kn[unsafe_offset=kh * 128 + i] * dl
            dsp[unsafe_offset=i * 128 + j] = w
            acc += w * qn[unsafe_offset=kh * 128 + i]
        part[g, j] = rebind[part.ElementType](acc)
        barrier()
        var asum = Float32(0)
        comptime for gg in range(4):
            asum += rebind[Scalar[f32]](part[gg, j])
        var ov = asum * scale
        var ss = q4x_bsum[512]((ov * ov) if g == 0 else Float32(0))
        var rs = 1 / sqrt(ss / Q4X_SD + Q4X_EPS)
        if g == 0:
            var pv = ov * rs * nw[unsafe_offset=j] * q4x_sigmoid(z[unsafe_offset=t * Int(zst) + h * 128 + j])
            o[unsafe_offset=h * 128 + j] = pv
            xq[unsafe_offset=t * Int(xqst) + h * 128 + j] = pv.cast[bf16]()


@always_inline
def q4x_rope_pair(x0: Float32, x1: Float32, i: Int, pos: Int, ts: Float32) -> SIMD[f32, 2]:
    var theta = Float32(pos)
    for _ in range(i):
        theta *= ts
    var c = cos(theta)
    var s = sin(theta)
    return SIMD[f32, 2](x0 * c - x1 * s, x0 * s + x1 * c)


def amar_q4x_qkprep(y: FP, yst: Int32, qn: FP, kn: FP, q: FP, g: FP, qst: Int32, kc: FP, vc: FP, kv0: Int32, sst: IP, ts: Float32, nq: Int32, nkv: Int32):
    var h = Int(block_idx.x)
    var t = Int(block_idx.y)
    var i = Int(thread_idx.x)
    var yp = y.unsafe_offset(t * Int(yst))
    var kvr = Int(kv0) + t
    var pos = kvr - Int(sst[unsafe_offset=t])
    var sh = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[Q4X_HD]())
    var v: Float32
    var wn: Float32
    if h < Int(nq):
        v = yp[unsafe_offset=h * 2 * Q4X_HD + i]
        wn = qn[unsafe_offset=i]
    else:
        v = yp[unsafe_offset=Int(nq) * 2 * Q4X_HD + (h - Int(nq)) * Q4X_HD + i]
        wn = kn[unsafe_offset=i]
    var ss = q4x_bsum[256](v * v)
    var s = 1 / sqrt(ss / Q4X_HD + Q4X_EPS)
    sh[i] = rebind[sh.ElementType](v * s * wn)
    barrier()
    if i < 32:
        var r = q4x_rope_pair(rebind[Scalar[f32]](sh[i]), rebind[Scalar[f32]](sh[i + 32]), i, pos, ts)
        sh[i] = rebind[sh.ElementType](r[0])
        sh[i + 32] = rebind[sh.ElementType](r[1])
    barrier()
    var out = rebind[Scalar[f32]](sh[i])
    if h < Int(nq):
        q[unsafe_offset=t * Int(qst) + h * Q4X_HD + i] = out
        g[unsafe_offset=t * Int(qst) + h * Q4X_HD + i] = yp[unsafe_offset=h * 2 * Q4X_HD + Q4X_HD + i]
    else:
        var kh = h - Int(nq)
        kc[unsafe_offset=(kvr * Int(nkv) + kh) * Q4X_HD + i] = out
        vc[unsafe_offset=(kvr * Int(nkv) + kh) * Q4X_HD + i] = yp[unsafe_offset=Int(nq) * 2 * Q4X_HD + Int(nkv) * Q4X_HD + kh * Q4X_HD + i]


def amar_q4x_attn(q: FP, g: FP, qst: Int32, kc: FP, vc: FP, sc: FP, scst: Int32, vis: UP, xq: BP, xqst: Int32, kv0: Int32, sst: IP, masked: Int32, nq: Int32, nkv: Int32, scale: Float32):
    var h = Int(block_idx.x)
    var t = Int(block_idx.y)
    var i = Int(thread_idx.x)
    var kh = h // (Int(nq) // Int(nkv))
    var kvr = Int(kv0) + t
    var s0 = Int(sst[unsafe_offset=t])
    var T = kvr - s0 + 1
    var qs = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[Q4X_HD]())
    qs[i] = rebind[qs.ElementType](q[unsafe_offset=t * Int(qst) + h * Q4X_HD + i])
    barrier()
    var srow = sc.unsafe_offset((t * Int(nq) + h) * Int(scst))
    var mx = Float32(-3.0e38)
    for tt in range(i, T, 256):
        var s = Float32(0)
        var kr = kc.unsafe_offset(((s0 + tt) * Int(nkv) + kh) * Q4X_HD)
        for d in range(Q4X_HD):
            s += rebind[Scalar[f32]](qs[d]) * kr[unsafe_offset=d]
        s = s * scale
        if masked != 0 and vis[unsafe_offset=tt] == 0:
            s = Float32(-3.0e38)
        srow[unsafe_offset=tt] = s
        mx = max(mx, s)
    mx = q4x_bmax[256](mx)
    var den = Float32(0)
    for tt in range(i, T, 256):
        var keep = masked == 0 or vis[unsafe_offset=tt] != 0
        var p = exp(srow[unsafe_offset=tt] - mx) if keep else Float32(0)
        srow[unsafe_offset=tt] = p
        den += p
    den = q4x_bsum[256](den)
    var o = Float32(0)
    for tt in range(T):
        o += (srow[unsafe_offset=tt] / den) * vc[unsafe_offset=((s0 + tt) * Int(nkv) + kh) * Q4X_HD + i]
    xq[unsafe_offset=t * Int(xqst) + h * Q4X_HD + i] = (o * q4x_sigmoid(g[unsafe_offset=t * Int(qst) + h * Q4X_HD + i])).cast[bf16]()


def amar_q4x_attn_sc[G: Int](q: FP, qst: Int32, kc: FP, sc: FP, scst: Int32, kv0: Int32, sst: IP, nkv: Int32, scale: Float32):
    var kh = Int(block_idx.x)
    var i = Int(thread_idx.x)
    var s0 = Int(sst[unsafe_offset=0])
    var T = Int(kv0) - s0 + 1
    var k0 = Int(block_idx.y) * 256
    var qs = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[Q4X_HD, 16]())
    var ks = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[256, 33]())
    comptime for hh in range(16):
        comptime if hh < G:
            qs[i, hh] = rebind[qs.ElementType](q[unsafe_offset=(kh * G + hh) * Q4X_HD + i])
        else:
            qs[i, hh] = rebind[qs.ElementType](Float32(0))
    var s = SIMD[f32, 16](0)
    var qv = qs.vectorize[1, 16]()
    for d0 in range(0, Q4X_HD, 32):
        barrier()
        comptime for j in range(32):
            var key = i // 32 + 8 * j
            var v = Float32(0)
            if k0 + key < T:
                v = kc[unsafe_offset=((s0 + k0 + key) * Int(nkv) + kh) * Q4X_HD + d0 + i % 32]
            ks[key, i % 32] = rebind[ks.ElementType](v)
        barrier()
        comptime for dd in range(32):
            s += rebind[SIMD[f32, 16]](qv[d0 + dd, 0]) * SIMD[f32, 16](rebind[Scalar[f32]](ks[i, dd]))
    if k0 + i < T:
        comptime for hh in range(G):
            sc[unsafe_offset=(kh * G + hh) * Int(scst) + k0 + i] = s[hh] * scale


def amar_q4x_attn_sm(sc: FP, scst: Int32, kv0: Int32, sst: IP):
    var h = Int(block_idx.x)
    var i = Int(thread_idx.x)
    var T = Int(kv0) - Int(sst[unsafe_offset=0]) + 1
    var srow = sc.unsafe_offset(h * Int(scst))
    var mx = Float32(-3.0e38)
    for tt in range(i, T, 256):
        mx = max(mx, srow[unsafe_offset=tt])
    mx = q4x_bmax[256](mx)
    var den = Float32(0)
    for tt in range(i, T, 256):
        var p = exp(srow[unsafe_offset=tt] - mx)
        srow[unsafe_offset=tt] = p
        den += p
    den = q4x_bsum[256](den)
    for tt in range(i, T, 256):
        srow[unsafe_offset=tt] = srow[unsafe_offset=tt] / den


def amar_q4x_attn_out(sc: FP, scst: Int32, vc: FP, g: FP, xq: BP, kv0: Int32, sst: IP, nq: Int32, nkv: Int32):
    var h = Int(block_idx.x)
    var i = Int(block_idx.y) * 64 + Int(thread_idx.x)
    var kh = h // (Int(nq) // Int(nkv))
    var s0 = Int(sst[unsafe_offset=0])
    var T = Int(kv0) - s0 + 1
    var srow = sc.unsafe_offset(h * Int(scst))
    var vb = vc.unsafe_offset(s0 * Int(nkv) * Q4X_HD + kh * Q4X_HD + i)
    var stride = Int(nkv) * Q4X_HD
    var o = Float32(0)
    var tt = 0
    while tt + 32 <= T:
        var vs = SIMD[f32, 32](0)
        var ps = SIMD[f32, 32](0)
        comptime for u in range(32):
            vs[u] = vb[unsafe_offset=(tt + u) * stride]
            ps[u] = srow[unsafe_offset=tt + u]
        comptime for u in range(32):
            o += ps[u] * vs[u]
        tt += 32
    while tt < T:
        o += srow[unsafe_offset=tt] * vb[unsafe_offset=tt * stride]
        tt += 1
    xq[unsafe_offset=h * Q4X_HD + i] = (o * q4x_sigmoid(g[unsafe_offset=h * Q4X_HD + i])).cast[bf16]()


@always_inline
def ffn_cur(y: FP, wts: FP, ysh: FP, yst: Int32, sg: FP, sgst: Int32, topk: Int32, t: Int, i: Int) -> Float32:
    var yp = y.unsafe_offset(t * Int(topk) * Q4X_H)
    var acc = Float32(0)
    for j in range(Int(topk)):
        acc = acc + yp[unsafe_offset=j * Q4X_H + i] * wts[unsafe_offset=t * Int(topk) + j]
    return acc + ysh[unsafe_offset=t * Int(yst) + i] * q4x_sigmoid(sg[unsafe_offset=t * Int(sgst)])


def amar_q4x_ffn_out(y: FP, wts: FP, ysh: FP, yst: Int32, sg: FP, sgst: Int32, cur: FP, topk: Int32):
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    var t = Int(block_idx.y)
    cur[unsafe_offset=t * Q4X_H + i] = ffn_cur(y, wts, ysh, yst, sg, sgst, topk, t, i)


def amar_q4x_ple_gate(key: FP, qry: FP, val: FP, valst: Int32, gated: FP, gts: FP, gtsst: Int32):
    var c = Int(block_idx.x)
    var t = Int(block_idx.y)
    var i = Int(thread_idx.x)
    var kp = key.unsafe_offset(t * Q4X_HCD + c * Q4X_H)
    var qp = qry.unsafe_offset(t * Q4X_HCD + c * Q4X_H)
    var vp = val.unsafe_offset(t * Int(valst))
    var gp = gated.unsafe_offset(t * Q4X_HCD + c * Q4X_H)
    var part = Float32(0)
    for j in range(i, Q4X_H, 256):
        part += kp[unsafe_offset=j] * qp[unsafe_offset=j]
    var s = q4x_bsum[256](part) * (1 / sqrt(Float32(Q4X_H)))
    var mag = sqrt(max(abs(s), Float32(1e-6)))
    var sg = Float32(1) if s > 0 else (Float32(-1) if s < 0 else Float32(0))
    var gt = q4x_sigmoid(sg * mag)
    if i == 0:
        gts[unsafe_offset=t * Int(gtsst) + c] = gt
    for j in range(i, Q4X_H, 256):
        gp[unsafe_offset=j] = vp[unsafe_offset=j] * gt


def amar_q4x_ple_conv(nrm: FP, gated: FP, cw: FP, hist: FP, res: FP, seqs: IP, rc: Int32, rw: Int32):
    var c = Int(block_idx.x) * 256 + Int(thread_idx.x)
    var sq = Int(block_idx.y)
    var t0 = Int(seqs[unsafe_offset=2 * sq])
    var t1 = Int(seqs[unsafe_offset=2 * sq + 1])
    var hb = hist.unsafe_offset(sq * Int(rw) * 9 * Q4X_HCD + c * 9)
    for t in range(t0, t1):
        var hp = hb.unsafe_offset(((Int(rc) + t - t0) % Int(rw)) * 9 * Q4X_HCD)
        var hq = hb.unsafe_offset(((Int(rc) + t - t0 + 1) % Int(rw)) * 9 * Q4X_HCD)
        var x = nrm[unsafe_offset=t * Q4X_HCD + c]
        var acc = hp[unsafe_offset=0] * cw[unsafe_offset=c * 4] + hp[unsafe_offset=3] * cw[unsafe_offset=c * 4 + 1] + hp[unsafe_offset=6] * cw[unsafe_offset=c * 4 + 2] + x * cw[unsafe_offset=c * 4 + 3]
        var cv = q4x_silu(acc)
        for j in range(8):
            hq[unsafe_offset=j] = hp[unsafe_offset=j + 1]
        hq[unsafe_offset=8] = x
        res[unsafe_offset=t * Q4X_HCD + c] = res[unsafe_offset=t * Q4X_HCD + c] + (gated[unsafe_offset=t * Q4X_HCD + c] + cv)


@always_inline
def q4x_bmin_idx[NT: Int](v: Int32) -> Int32:
    var mins = stack_allocation[i32, address_space = AddressSpace.SHARED](row_major[NT // WARP_SIZE]())
    var w = warp.min(v)
    if lane_id() == 0:
        mins[Int(thread_idx.x) // WARP_SIZE] = rebind[mins.ElementType](w)
    barrier()
    var t = rebind[Scalar[i32]](mins[0])
    comptime for i in range(1, NT // WARP_SIZE):
        t = min(t, rebind[Scalar[i32]](mins[i]))
    barrier()
    return t


def amar_q4x_router(lg: FP, lgst: Int32, sel: IP, wts: FP, seq: Int32, topk: Int32, kb: FP, hk: Int32):
    var tk = Int(block_idx.x)
    var i = Int(thread_idx.x)
    var selp = sel.unsafe_offset(tk * Q4X_SEL)
    var ps = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[512]())
    var dens = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[1]())
    var sl = stack_allocation[i32, address_space = AddressSpace.SHARED](row_major[16]())
    var v = lg[unsafe_offset=tk * Int(lgst) + i]
    if hk != 0:
        v += kb[unsafe_offset=i]
    var mx = q4x_bmax[512](v)
    var e = exp(v - mx)
    ps[i] = rebind[ps.ElementType](e)
    barrier()
    if i == 0:
        var den = Float32(0)
        for j in range(512):
            den += rebind[Scalar[f32]](ps[j])
        dens[0] = rebind[dens.ElementType](den)
    barrier()
    var pv = e / rebind[Scalar[f32]](dens[0])
    var taken = False
    for r in range(Int(topk)):
        var best = q4x_bmax[512](pv if not taken else Float32(-1))
        var idx = q4x_bmin_idx[512](Int32(i) if (not taken and pv == best) else Int32(512))
        if i == Int(idx):
            taken = True
        if i == 0:
            sl[r] = rebind[sl.ElementType](idx)
    barrier()
    for r in range(Int(topk)):
        if i == Int(rebind[Scalar[i32]](sl[r])):
            ps[r] = rebind[ps.ElementType](pv)
    barrier()
    if i == 0:
        var ws = Float32(0)
        for r in range(Int(topk)):
            ws += rebind[Scalar[f32]](ps[r])
        ws = max(ws, Float32(6.103515625e-5))
        for r in range(Int(topk)):
            wts[unsafe_offset=tk * Int(topk) + r] = rebind[Scalar[f32]](ps[r]) / ws
            selp[unsafe_offset=1 + r] = rebind[Scalar[i32]](sl[r])
        fence[Ordering.RELEASE]()
        Atomic[Int32].store[ordering=Ordering.RELEASE](selp, seq)


def amar_q4x_mtp_in(e: FP, est: Int32, enw: FP, h: FP, hnw: FP, x: BP):
    var c = Int(block_idx.x)
    var t = Int(block_idx.y)
    var i = Int(thread_idx.x)
    var ep = e.unsafe_offset(t * Int(est))
    var hp = h.unsafe_offset(t * Q4X_HCD + c * Q4X_H)
    var xp = x.unsafe_offset((t * Q4X_HC + c) * 2 * Q4X_H)
    var se = Float32(0)
    var sh = Float32(0)
    for j in range(i, Q4X_H, 256):
        se += ep[unsafe_offset=j] * ep[unsafe_offset=j]
        sh += hp[unsafe_offset=j] * hp[unsafe_offset=j]
    var re = 1 / sqrt(q4x_bsum[256](se) / Q4X_H + Q4X_EPS)
    var rh = 1 / sqrt(q4x_bsum[256](sh) / Q4X_H + Q4X_EPS)
    for j in range(i, Q4X_H, 256):
        xp[unsafe_offset=j] = (ep[unsafe_offset=j] * re * enw[unsafe_offset=j]).cast[bf16]()
        xp[unsafe_offset=Q4X_H + j] = (hp[unsafe_offset=j] * rh * hnw[unsafe_offset=c * Q4X_H + j]).cast[bf16]()


def amar_q4x_mtp_tab(sel: IP, tab: LP, n: Int32, topk: Int32, ne: Int32, gs: Int64, us: Int64, ds: Int64, ub: Int64, db: Int64):
    var g = Int(block_idx.x) * 64 + Int(thread_idx.x)
    if g < Int(n) * Int(topk):
        var id = Int64(sel[unsafe_offset=(g // Int(topk)) * Q4X_SEL + 1 + g % Int(topk)])
        tab[unsafe_offset=2 * g] = id * gs
        tab[unsafe_offset=2 * g + 1] = ub + id * us
        tab[unsafe_offset=2 * Int(ne) + g] = db + id * ds


def amar_q4x_attn_g[G: Int](q: FP, g: FP, qst: Int32, kc: FP, vc: FP, sc: FP, scst: Int32, xq: BP, xqst: Int32, kv0: Int32, sst: IP, nkv: Int32, scale: Float32):
    var kh = Int(block_idx.x)
    var t = Int(block_idx.y)
    var i = Int(thread_idx.x)
    var nq = G * Int(nkv)
    var kvr = Int(kv0) + t
    var s0 = Int(sst[unsafe_offset=t])
    var T = kvr - s0 + 1
    var qs = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[G, Q4X_HD]())
    comptime for hh in range(G):
        qs[hh, i] = rebind[qs.ElementType](q[unsafe_offset=t * Int(qst) + (kh * G + hh) * Q4X_HD + i])
    barrier()
    var mx = Array[Float32, G](fill=Float32(-3.0e38))
    for tt in range(i, T, 256):
        var kr = kc.unsafe_offset(((s0 + tt) * Int(nkv) + kh) * Q4X_HD)
        var s = Array[Float32, G](fill=Float32(0))
        for d in range(Q4X_HD):
            var kv = kr[unsafe_offset=d]
            comptime for hh in range(G):
                s[hh] += rebind[Scalar[f32]](qs[hh, d]) * kv
        comptime for hh in range(G):
            var v = s[hh] * scale
            sc[unsafe_offset=(t * nq + kh * G + hh) * Int(scst) + tt] = v
            mx[hh] = max(mx[hh], v)
    var den = Array[Float32, G](fill=Float32(0))
    comptime for hh in range(G):
        mx[hh] = q4x_bmax[256](mx[hh])
    for tt in range(i, T, 256):
        comptime for hh in range(G):
            var sp = sc.unsafe_offset((t * nq + kh * G + hh) * Int(scst))
            var p = exp(sp[unsafe_offset=tt] - mx[hh])
            sp[unsafe_offset=tt] = p
            den[hh] += p
    comptime for hh in range(G):
        den[hh] = q4x_bsum[256](den[hh])
    var o = Array[Float32, G](fill=Float32(0))
    for tt in range(T):
        var vv = vc[unsafe_offset=((s0 + tt) * Int(nkv) + kh) * Q4X_HD + i]
        comptime for hh in range(G):
            o[hh] += (sc[unsafe_offset=(t * nq + kh * G + hh) * Int(scst) + tt] / den[hh]) * vv
    comptime for hh in range(G):
        var h = kh * G + hh
        xq[unsafe_offset=t * Int(xqst) + h * Q4X_HD + i] = (o[hh] * q4x_sigmoid(g[unsafe_offset=t * Int(qst) + h * Q4X_HD + i])).cast[bf16]()


def amar_q4x_argmax_p(lg: FP, lgst: Int32, n: Int32, nb: Int32, part: FP):
    var c = Int(block_idx.x)
    var t = Int(block_idx.y)
    var i = Int(thread_idx.x)
    var row = lg.unsafe_offset(t * Int(lgst))
    var lo = c * Int(n) // Int(nb)
    var hi = (c + 1) * Int(n) // Int(nb)
    var bv = Float32(-3.0e38)
    var bi = n
    for j in range(lo + i, hi, 256):
        var v = row[unsafe_offset=j]
        if v > bv:
            bv = v
            bi = Int32(j)
    var mx = q4x_bmax[256](bv)
    var idx = q4x_bmin_idx[256](bi if bv == mx else n)
    var s = Float32(0)
    for j in range(lo + i, hi, 256):
        s += exp(row[unsafe_offset=j] - mx)
    s = q4x_bsum[256](s)
    if i == 0:
        var pp = part.unsafe_offset((t * Int(nb) + c) * 3)
        pp[0] = mx
        pp[1] = bitcast[f32, 1](SIMD[i32, 1](idx))[0]
        pp[2] = s


def amar_q4x_argmax_r(part: FP, nb: Int32, n: Int32, ids: IP, prob: FP):
    var t = Int(block_idx.x)
    var i = Int(thread_idx.x)
    var mv = Float32(-3.0e38)
    if i < Int(nb):
        mv = part[unsafe_offset=(t * Int(nb) + i) * 3]
    var mx = q4x_bmax[256](mv)
    var bi = n
    var s = Float32(0)
    if i < Int(nb):
        var pp = part.unsafe_offset((t * Int(nb) + i) * 3)
        if pp[0] == mx:
            bi = bitcast[i32, 1](SIMD[f32, 1](pp[1]))[0]
        s = pp[2] * exp(pp[0] - mx)
    var idx = q4x_bmin_idx[256](bi)
    s = q4x_bsum[256](s)
    if i == 0:
        ids[unsafe_offset=t] = idx
        prob[unsafe_offset=t] = 1 / s
