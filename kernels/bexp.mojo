"""`exp` and `log` for every kernel and its CPU reference. Default: std.math.exp, unchanged code. With
`-D BARO_SHARED_EXP=1`, Float32 lanes use fexp32 instead, so the GPU kernel and the CPU path give the same bits
(std.math exp differs CPU vs GPU on 49% of Float32 inputs; fexp32 on 0 of 40017, newera tests/check_f32math.sh).
fexp32 costs 3.6x std.math exp on the 7900 XTX in an exp-bound kernel, ~0.1% of a decode token at 8k ctx.

fexp32 is newera lib/fexp.mojo: Cody-Waite reduction by ln 2 in two parts, the cephes expf polynomial, 2^k from
exponent bits, every multiply-add an explicit fma, within 1 ulp of the correctly rounded value on (-104, 89).
flog32 is the musl logf shape (fdlibm): x = 2^k (1 + f) with sqrt(2)/2 < 1 + f < sqrt(2), s = f / (2 + f), a degree-7
polynomial in s^2, ln 2 split in a high and a low part, every multiply-add an explicit fma."""
from std.math import fma, floor, sqrt
from std.math import exp as std_exp, log as std_log, rsqrt as std_rsqrt, sin as std_sin, cos as std_cos
from std.memory import bitcast
from std.sys import get_defined_bool

comptime SHARED_EXP = get_defined_bool["BARO_SHARED_EXP", False]()
comptime F = Float32
comptime P = SIMD[DType.float32, 8](
    F(1.9875691500e-4), F(1.3981999507e-3), F(8.3334519073e-3), F(4.1665795894e-2), F(1.6666665459e-1),
    F(5.0000001201e-1), 0, 0,
)
comptime L = SIMD[DType.float32, 4](F(0.66666662693), F(0.40000972152), F(0.28498786688), F(0.24279078841))


@always_inline
def pow2i(k: Int32) -> Float32:
    return bitcast[DType.float32](UInt32(k + 127) << 23)


@always_inline
def fexp32(x: Float32) -> Float32:
    if x != x:
        return x
    if x > 88.72283935546875:
        return Float32.MAX * 2
    if x < -103.97208:
        return 0
    var k = floor(fma(x, F(1.44269504088896341), F(0.5)))
    var r = fma(k, F(2.12194440e-4), fma(k, F(-0.693359375), x))
    var z = r * r
    var p = P[0]
    comptime for i in range(1, 6):
        p = fma(p, r, P[i])
    var y = fma(p, z, r) + 1
    var ki = Int32(k)
    var h = ki // 2
    return y * pow2i(h) * pow2i(ki - h)


@always_inline
def exp(x: SIMD) -> type_of(x) where x.dtype.is_floating_point():
    comptime if SHARED_EXP and x.dtype == DType.float32:
        var y = x
        comptime for i in range(x.length):
            y[i] = rebind[Scalar[x.dtype]](fexp32(rebind[Float32](x[i])))
        return y
    else:
        return std_exp(x)


@always_inline
def flog32(x_in: Float32) -> Float32:
    var x = x_in
    var u = bitcast[DType.uint32](x)
    var k = Int32(0)
    if u < 0x00800000 or (u >> 31) != 0:
        if (u << 1) == 0:
            return -Float32.MAX * 2
        if (u >> 31) != 0:
            return bitcast[DType.float32](UInt32(0x7FC00000))
        k = -25
        x *= 33554432.0
        u = bitcast[DType.uint32](x)
    elif u >= 0x7F800000:
        return x
    elif u == 0x3F800000:
        return 0
    u += 0x3F800000 - 0x3F3504F3
    k += Int32(u >> 23) - 127
    u = (u & 0x007FFFFF) + 0x3F3504F3
    var f = bitcast[DType.float32](u) - 1
    var s = f / (2 + f)
    var z = s * s
    var w = z * z
    var r = fma(z, fma(w, L[2], L[0]), w * fma(w, L[3], L[1]))
    var hfsq = fma(F(0.5) * f, f, 0)
    var dk = Float32(k)
    return fma(dk, F(6.9313812256e-01), fma(s, hfsq + r, dk * F(9.0580006145e-06)) - hfsq + f)


@always_inline
def log(x: SIMD) -> type_of(x) where x.dtype.is_floating_point():
    comptime if SHARED_EXP and x.dtype == DType.float32:
        var y = x
        comptime for i in range(x.length):
            y[i] = rebind[Scalar[x.dtype]](flog32(rebind[Float32](x[i])))
        return y
    else:
        return std_log(x)


comptime D = Float64
comptime S6 = SIMD[DType.float64, 8](
    D(-1.66666666666666324348e-01), D(8.33333333332248946124e-03), D(-1.98412698298579493134e-04),
    D(2.75573137070700676789e-06), D(-2.50507602534068634195e-08), D(1.58969099521155010221e-10), 0, 0,
)
comptime C6 = SIMD[DType.float64, 8](
    D(4.16666666666666019037e-02), D(-1.38888888888741095749e-03), D(2.48015872894767294178e-05),
    D(-2.75573143513906633035e-07), D(2.08757232129817482790e-09), D(-1.13596475577881948265e-11), 0, 0,
)


@always_inline
def fsincos32(x: Float32) -> SIMD[DType.float32, 2]:
    var xd = D(x)
    var k = floor(fma(xd, D(6.36619772367581382433e-01), D(0.5)))
    var r = fma(-k, D(1.57079632673412561417e+00), xd)
    r = fma(-k, D(6.07710050630396597660e-11), r)
    r = fma(-k, D(2.02226624871116645580e-21), r)
    r = fma(-k, D(8.47842766036889956997e-32), r)
    var z = r * r
    var ps = S6[5]
    var pc = C6[5]
    comptime for i in range(4, -1, -1):
        ps = fma(ps, z, S6[i])
        pc = fma(pc, z, C6[i])
    var sn = fma(r * z, ps, r)
    var cs = fma(z * z, pc, fma(z, D(-0.5), 1))
    var q = Int(k) & 3
    var a = sn
    var b = cs
    if q == 1:
        a = cs
        b = -sn
    elif q == 2:
        a = -sn
        b = -cs
    elif q == 3:
        a = -cs
        b = sn
    return SIMD[DType.float32, 2](Float32(a), Float32(b))


@always_inline
def frsq32(x: Float32) -> Float32:
    return Float32(1) / sqrt(x)


@always_inline
def rsqrt(x: SIMD) -> type_of(x) where x.dtype.is_floating_point():
    comptime if SHARED_EXP and x.dtype == DType.float32:
        var y = x
        comptime for i in range(x.length):
            y[i] = rebind[Scalar[x.dtype]](frsq32(rebind[Float32](x[i])))
        return y
    else:
        return std_rsqrt(x)


@always_inline
def sin(x: SIMD) -> type_of(x) where x.dtype.is_floating_point():
    comptime if SHARED_EXP and x.dtype == DType.float32:
        var y = x
        comptime for i in range(x.length):
            y[i] = rebind[Scalar[x.dtype]](fsincos32(rebind[Float32](x[i]))[0])
        return y
    else:
        return std_sin(x)


@always_inline
def cos(x: SIMD) -> type_of(x) where x.dtype.is_floating_point():
    comptime if SHARED_EXP and x.dtype == DType.float32:
        var y = x
        comptime for i in range(x.length):
            y[i] = rebind[Scalar[x.dtype]](fsincos32(rebind[Float32](x[i]))[1])
        return y
    else:
        return std_cos(x)
