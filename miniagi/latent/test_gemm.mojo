"""tgemm (bf16 WMMA, fp32 in/out, split-K) vs fp64 host reference, then TFLOPS at the training shapes.

Gate per element: |err| <= 2^-7 * sum_p |a*b| (bf16 inputs carry 8 mantissa bits; outputs near 0 defeat a relative gate).
Second gate, tgemm and the linalg forward path: vs an fp64 reference on bf16-ROUNDED inputs, |err| <= 2^-16 * sum_p |a*b|
(fp32 accumulation only); a GEMM that rounds anything besides its inputs fails it.
"""
from std.random import seed, random_float64
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer
from ops import *
from lops import Gemm
from layout import TileTensor, row_major
from linalg.matmul import matmul as lmatmul


def dev(ctx: DeviceContext, h: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    var d = ctx.enqueue_create_buffer[DType.float32](len(h))
    upload(ctx, d, h)
    return d


def rnd(n: Int) -> List[Float32]:
    var h = List[Float32](length=n, fill=0)
    for i in range(n):
        h[i] = Float32(random_float64(-1, 1))
    return h^


def r16(x: Float32) -> Float64:
    return Float64(x.cast[DType.bfloat16]().cast[DType.float32]())


def fp(b: DeviceBuffer[DType.float32]) -> F:
    return F(unsafe_from_address=Int(b.unsafe_ptr()))


def check[ta: Bool, tb: Bool](ctx: DeviceContext, g: Gemm, m: Int, n: Int, k: Int, acc: Bool) raises -> Bool:
    var a = rnd(m * k)
    var b = rnd(k * n)
    var c0 = rnd(m * n)
    var ad = dev(ctx, a)
    var bd = dev(ctx, b)
    var cd = dev(ctx, c0)
    g[ta, tb](ctx, fp(ad), fp(bd), fp(cd), m, n, k, acc)
    var got = List[Float32](length=m * n, fill=0)
    ctx.enqueue_copy(dst_ptr=got.unsafe_ptr(), src_buf=cd)
    ctx.synchronize()
    var worst: Float64 = 0
    var worst16: Float64 = 0
    for i in range(m):
        for j in range(n):
            var s: Float64 = Float64(c0[i * n + j]) if acc else 0
            var s16: Float64 = s
            var mag: Float64 = 0
            for p in range(k):
                var av = a[p * m + i] if ta else a[i * k + p]
                var bv = b[j * k + p] if tb else b[p * n + j]
                var x = Float64(av) * Float64(bv)
                s += x
                s16 += r16(av) * r16(bv)
                mag += abs(x)
            worst = max(worst, abs(Float64(got[i * n + j]) - s) / (mag * 2.0 ** -7))
            worst16 = max(worst16, abs(Float64(got[i * n + j]) - s16) / (mag * 2.0 ** -16))
    _ = (ad.unsafe_ptr(), bd.unsafe_ptr(), cd.unsafe_ptr())
    var ok = worst <= 1 and worst16 <= 1
    print("PASS" if ok else "FAIL", "tgemm", ta, tb, m, n, k, "acc", acc, "worst/gate", worst, "rounded-input worst/gate", worst16)
    return ok


def lcheck(ctx: DeviceContext, m: Int, n: Int, k: Int) raises -> Bool:
    """The Model.mm forward path: bf16 a (m x k) times bf16 b stored n x k, linalg matmul transpose_b, fp32 out."""
    var a = rnd(m * k)
    var b = rnd(n * k)
    var a16 = ctx.enqueue_create_buffer[DType.bfloat16](m * k)
    var b16 = ctx.enqueue_create_buffer[DType.bfloat16](n * k)
    var cd = ctx.enqueue_create_buffer[DType.float32](m * n)
    with a16.map_to_host() as h:
        for i in range(m * k):
            h[i] = a[i].cast[DType.bfloat16]()
    with b16.map_to_host() as h:
        for i in range(n * k):
            h[i] = b[i].cast[DType.bfloat16]()
    lmatmul[transpose_b=True, target="gpu"](TileTensor(cd, row_major(m, n)), TileTensor(a16, row_major(m, k)), TileTensor(b16, row_major(n, k)), ctx)
    var got = List[Float32](length=m * n, fill=0)
    ctx.enqueue_copy(dst_ptr=got.unsafe_ptr(), src_buf=cd)
    ctx.synchronize()
    var worst16: Float64 = 0
    for i in range(m):
        for j in range(n):
            var s16: Float64 = 0
            var mag: Float64 = 0
            for p in range(k):
                s16 += r16(a[i * k + p]) * r16(b[j * k + p])
                mag += abs(Float64(a[i * k + p]) * Float64(b[j * k + p]))
            worst16 = max(worst16, abs(Float64(got[i * n + j]) - s16) / (mag * 2.0 ** -16))
    var ok = worst16 <= 1
    print("PASS" if ok else "FAIL", "linalg bf16", m, n, k, "rounded-input worst/gate", worst16)
    return ok


def bench[ta: Bool, tb: Bool](ctx: DeviceContext, g: Gemm, m: Int, n: Int, k: Int) raises:
    var ad = dev(ctx, rnd(m * k))
    var bd = dev(ctx, rnd(k * n))
    var cd = dev(ctx, rnd(m * n))
    comptime R = 50
    for _ in range(3):
        g[ta, tb](ctx, fp(ad), fp(bd), fp(cd), m, n, k)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(R):
        g[ta, tb](ctx, fp(ad), fp(bd), fp(cd), m, n, k)
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / 1e3 / R
    var t1 = perf_counter_ns()
    for _ in range(5):
        gemm[ta, tb](ctx, fp(ad), fp(bd), fp(cd), m, n, k)
    ctx.synchronize()
    var nus = Float64(perf_counter_ns() - t1) / 1e3 / 5
    _ = (ad.unsafe_ptr(), bd.unsafe_ptr(), cd.unsafe_ptr())
    print("bench", ta, tb, m, n, k, "us", us, "TFLOPS", 2.0 * Float64(m * n * k) / us / 1e6, "naive_us", nus)


def main() raises:
    seed(1)
    var ctx = DeviceContext()
    var g = Gemm(ctx, 16 * 1024 * 1024)
    var fails = 0
    fails += Int(not check[False, False](ctx, g, 300, 200, 100, False))
    fails += Int(not check[False, True](ctx, g, 130, 260, 70, True))
    fails += Int(not check[True, False](ctx, g, 96, 112, 3000, True))
    fails += Int(not check[True, True](ctx, g, 40, 300, 50, False))
    fails += Int(not check[True, False](ctx, g, 256, 64, 2048, False))
    fails += Int(not lcheck(ctx, 492, 256, 256))
    fails += Int(not lcheck(ctx, 492, 688, 256))
    fails += Int(not lcheck(ctx, 12, 32, 256))
    comptime R = 11264
    bench[False, False](ctx, g, R, 256, 256)
    bench[False, False](ctx, g, R, 688, 256)
    bench[False, False](ctx, g, R, 256, 688)
    bench[False, True](ctx, g, R, 256, 256)
    bench[False, True](ctx, g, R, 256, 688)
    bench[False, True](ctx, g, R, 688, 256)
    bench[True, False](ctx, g, 256, 256, R)
    bench[True, False](ctx, g, 256, 688, R)
    bench[True, False](ctx, g, 688, 256, R)
    bench[False, False](ctx, g, 256, 256, 256)
    bench[False, False](ctx, g, 256, 688, 256)
    bench[False, False](ctx, g, 256, 256, 688)
    bench[False, True](ctx, g, 256, 256, 256)
    bench[True, False](ctx, g, 256, 256, 256)
    bench[False, False](ctx, g, 8960, 256, 256)
    bench[False, False](ctx, g, 768, 256, 256)
    if fails:
        raise Error("FAIL " + String(fails) + "/8 gemm cases")
    print("PASS 8/8")
