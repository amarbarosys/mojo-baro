from std.os import getenv
from std.sys import stderr
from std.memory import unsafe_memcpy
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from registry import *
from model import IS_MOE
from certify import Cert

comptime CERT_LAYER = 3
comptime CERT_KAPPA = 0.125
comptime bf16 = DType.bfloat16
comptime f32 = DType.float32


struct CertHook(Copyable, Movable):
    var on: Bool
    var seed: UInt64
    var certs: List[Cert]
    var hx: HostBuffer[bf16]
    var hy: HostBuffer[f32]
    var armed: Bool
    var ubytes: Int

    def __init__(out self, ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], off: List[Int], w: Int) raises:
        var env = getenv("BARO_CERT", "")
        self.on = env != "" and not IS_MOE
        self.seed = UInt64(atol(env)) if self.on else UInt64(0)
        self.certs = List[Cert]()
        self.armed = False
        self.ubytes = 0
        self.hx = ctx.enqueue_create_host_buffer[bf16](2 * H + ATT + FFN if self.on else 1)
        self.hy = ctx.enqueue_create_host_buffer[f32](QF + 2 * KV + 2 * H + 2 * FFN if self.on else 1)
        if not self.on:
            return
        var ws = [w + 1, w + 2, w + 3, w + 6, w + 8, w + 9, w + 10]
        var ns = [QF, KV, KV, H, FFN, FFN, H]
        var ks = [H, H, H, ATT, H, H, FFN]
        for m in range(7):
            var n = ns[m]
            var k = ks[m]
            var q = List[UInt8](length=n * k // 2, fill=0)
            var s = List[UInt8](length=n * k // 16, fill=0)
            ctx.enqueue_copy(q.unsafe_ptr(), DeviceBuffer[DType.uint8](ctx, wbuf.unsafe_ptr().unsafe_offset(off[ws[m]]), n * k // 2, owning=False))
            ctx.enqueue_copy(s.unsafe_ptr(), DeviceBuffer[DType.uint8](ctx, wbuf.unsafe_ptr().unsafe_offset(off[ws[m]] + n * k // 2), n * k // 16, owning=False))
            ctx.synchronize()
            self.certs.append(Cert(q.unsafe_ptr(), s.unsafe_ptr().unsafe_bitcast[Float16](), n, k, self.seed * 1000 + UInt64(m)))
            self.ubytes += k * 8
        print("cert on seed", self.seed, "layer", CERT_LAYER, "kappa", CERT_KAPPA, "u-vector bytes", self.ubytes, "matrices 7")

    def tap(mut self, ctx: DeviceContext, layer: Int, m: Int, site: Int, x: DeviceBuffer[bf16], y: DeviceBuffer[f32]) raises:
        if not self.on or layer != CERT_LAYER or m != 1:
            return
        var xo = [0, 0, 0, H, H + ATT, H + ATT, 2 * H + ATT]
        var xn = [H, H, H, ATT, H, H, FFN]
        var yo = [0, QF, QF + KV, QF + 2 * KV, QF + 2 * KV + H, QF + 2 * KV + H + FFN, QF + 2 * KV + H + 2 * FFN]
        var yn = [QF, KV, KV, H, FFN, FFN, H]
        if site == 0 or site == 3 or site == 4 or site == 6:
            ctx.enqueue_copy(dst_buf=self.hx.create_sub_buffer[bf16](xo[site], xn[site]), src_buf=DeviceBuffer[bf16](ctx, x.unsafe_ptr(), xn[site], owning=False))
        ctx.enqueue_copy(dst_buf=self.hy.create_sub_buffer[f32](yo[site], yn[site]), src_buf=DeviceBuffer[f32](ctx, y.unsafe_ptr(), yn[site], owning=False))
        self.armed = site == 6

    def verify(mut self, ctx: DeviceContext, tok: Int) raises:
        if not self.armed:
            return
        self.armed = False
        ctx.synchronize()
        var xo = [0, 0, 0, H, H + ATT, H + ATT, 2 * H + ATT]
        var yo = [0, QF, QF + KV, QF + 2 * KV, QF + 2 * KV + H, QF + 2 * KV + H + FFN, QF + 2 * KV + H + 2 * FFN]
        var passed = 0
        var worst = Float64(0)
        for m in range(7):
            var x = List[BFloat16](length=self.certs[m].k, fill=0)
            var y = List[Float32](length=self.certs[m].n, fill=0)
            unsafe_memcpy(dest=x.unsafe_ptr().unsafe_bitcast[UInt8](), src=self.hx.unsafe_ptr().unsafe_offset(xo[m]).unsafe_bitcast[UInt8](), count=2 * self.certs[m].k)
            unsafe_memcpy(dest=y.unsafe_ptr().unsafe_bitcast[UInt8](), src=self.hy.unsafe_ptr().unsafe_offset(yo[m]).unsafe_bitcast[UInt8](), count=4 * self.certs[m].n)
            var r = self.certs[m].check(x, y)
            var rho = r[0] / r[1]
            worst = max(worst, rho)
            passed += 1 if rho <= CERT_KAPPA else 0
        print("cert tok", tok, "seed", self.seed, "layer", CERT_LAYER, "pass", String(passed) + "/7", "maxrho", worst, file=stderr)
