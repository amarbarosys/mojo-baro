# wg_rope_norm against an fp64 host rope (same formula, adjacent pairs), at large positions where f32 angle error shows.
from std.math import cos, sin
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from rope import wg_rope_norm
from model import HD, NQH, FREQ_BASE

comptime T = 7
comptime P0 = 8000

def main() raises:
    var ctx = DeviceContext()
    comptime n = T * NQH * HD
    var xh = ctx.enqueue_create_host_buffer[DType.float32](n)
    var fh = ctx.enqueue_create_host_buffer[DType.float32](HD // 2)
    var oh = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n):
        xh[i] = Float32((i * 37) % 19 - 9) * 0.1
    for i in range(HD // 2):
        fh[i] = 1.0 if i < 20 else Float32(i - 18) # llama-3 style: high frequencies unscaled, low ones divided
    var xd = ctx.enqueue_create_buffer[DType.float32](n)
    var fd = ctx.enqueue_create_buffer[DType.float32](HD // 2)
    ctx.enqueue_copy(dst_buf=xd, src_buf=xh)
    ctx.enqueue_copy(dst_buf=fd, src_buf=fh)
    var X = TileTensor(xd, row_major[T * NQH, HD]())
    var F = TileTensor(fd, row_major[HD // 2]())
    ctx.enqueue_function[wg_rope_norm[type_of(X).LayoutType, type_of(F).LayoutType]](X, F, Int32(P0), Int32(NQH), grid_dim=(NQH, T), block_dim=HD // 2)
    ctx.enqueue_copy(dst_buf=oh, src_buf=xd)
    ctx.synchronize()
    var worst = 0.0
    for r in range(T):
        for h in range(NQH):
            for i in range(HD // 2):
                var b = (r * NQH + h) * HD + 2 * i
                var th = Float64(P0 + r) * Float64(FREQ_BASE) ** (-2.0 * Float64(i) / Float64(HD)) / Float64(fh[i])
                var x0 = Float64(xh[b])
                var x1 = Float64(xh[b + 1])
                worst = max(worst, abs(Float64(oh[b]) - (x0 * cos(th) - x1 * sin(th))), abs(Float64(oh[b + 1]) - (x0 * sin(th) + x1 * cos(th))))
    print("rope max abs err", worst)
    if worst > 2e-3:
        raise Error("FAIL rope: " + String(worst))
    print("PASS rope")
