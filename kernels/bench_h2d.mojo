from std.time import perf_counter_ns
from max.gpu.host import DeviceContext


def main() raises:
    var ctx = DeviceContext()
    var cap = 64 << 20
    var h = ctx.enqueue_create_host_buffer[DType.uint8](cap)
    var d = ctx.enqueue_create_buffer[DType.uint8](cap)
    h.enqueue_fill(1)
    ctx.synchronize()
    for sz in [460800, 4 << 20, 64 << 20]:
        var n = 200 if sz < (1 << 20) else 20
        for rep in range(2):
            var t0 = perf_counter_ns()
            for i in range(n):
                ctx.enqueue_copy(dst_buf=d.create_sub_buffer[DType.uint8](0, sz), src_buf=h.create_sub_buffer[DType.uint8](0, sz))
            ctx.synchronize()
            var dt = Float64(perf_counter_ns() - t0)
            if rep == 1:
                print("pinned H2D", sz, "B x", n, "queued:", dt / Float64(n) / 1e3, "us per copy,", Float64(sz) * Float64(n) / dt, "GB/s")
        var t1 = perf_counter_ns()
        for i in range(n):
            ctx.enqueue_copy(dst_buf=d.create_sub_buffer[DType.uint8](0, sz), src_buf=h.create_sub_buffer[DType.uint8](0, sz))
            ctx.synchronize()
        var dt1 = Float64(perf_counter_ns() - t1)
        print("pinned H2D", sz, "B x", n, "sync each:", dt1 / Float64(n) / 1e3, "us per copy,", Float64(sz) * Float64(n) / dt1, "GB/s")
