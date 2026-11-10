"""Gate for kernels/icflush.mojo: the cold-cache helper every cold-timing test imports.

1. Ruler units: D2D copy and read-sweep GB/s both within 0.3x..1.1x of the 960 GB/s card peak (FAIL units otherwise;
   the freivalds explore run once printed 654925 "GB/s" from a bytes-per-microsecond slip).
2. Flush works: a 48 MB buffer read right after a read of itself (Infinity Cache warm) vs read after ICFlush.flush;
   the flushed read must be slower by >= 5% (median of 11).
3. Read-only flush beats a copy flush: the next kernel after a 256 MB D2D copy pays the dirty-line write-back, so the
   48 MB read after a copy is slower than after the read sweep (the 258 vs 158 us effect, freivalds lane 2026-10-12).
Prints PASS 3/3 or FAIL k/3 names.
"""
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from icflush import ICFlush, ic_read_sweep, ic_median, IC_GRID, IC_THREADS, IC_PEAK_GBPS

comptime PROBE = 48 * 1024 * 1024


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var fl = ICFlush(ctx)
    var probe = ctx.enqueue_create_buffer[DType.uint64](PROBE // 8)
    var sums = ctx.enqueue_create_buffer[DType.uint64](IC_GRID * IC_THREADS)
    probe.enqueue_fill(3)
    ctx.synchronize()
    var r = fl.ruler(ctx)
    print("ruler copy GB/s", r.copy_gbps, "read GB/s", r.read_gbps, "peak", IC_PEAK_GBPS)
    var fails = 0
    var names = String()
    if not r.units_ok():
        fails += 1
        names += " units"
    var warm = List[Float64]()
    var cold = List[Float64]()
    var after_copy = List[Float64]()
    for it in range(14):
        ctx.enqueue_function[ic_read_sweep](probe.unsafe_ptr(), Int64(PROBE // 8), sums.unsafe_ptr(), grid_dim=IC_GRID, block_dim=IC_THREADS)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        ctx.enqueue_function[ic_read_sweep](probe.unsafe_ptr(), Int64(PROBE // 8), sums.unsafe_ptr(), grid_dim=IC_GRID, block_dim=IC_THREADS)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        fl.flush(ctx)
        ctx.synchronize()
        var t2 = perf_counter_ns()
        ctx.enqueue_function[ic_read_sweep](probe.unsafe_ptr(), Int64(PROBE // 8), sums.unsafe_ptr(), grid_dim=IC_GRID, block_dim=IC_THREADS)
        ctx.synchronize()
        var t3 = perf_counter_ns()
        ctx.enqueue_copy(dst_buf=fl.dst, src_buf=fl.src)
        ctx.synchronize()
        var t4 = perf_counter_ns()
        ctx.enqueue_function[ic_read_sweep](probe.unsafe_ptr(), Int64(PROBE // 8), sums.unsafe_ptr(), grid_dim=IC_GRID, block_dim=IC_THREADS)
        ctx.synchronize()
        var t5 = perf_counter_ns()
        if it >= 3:
            warm.append(Float64(t1 - t0) / 1e3)
            cold.append(Float64(t3 - t2) / 1e3)
            after_copy.append(Float64(t5 - t4) / 1e3)
    var w = ic_median(warm)
    var c = ic_median(cold)
    var a = ic_median(after_copy)
    print("48 MB read us: warm", w, "after read-sweep flush", c, "after D2D copy", a)
    if not (c >= 1.05 * w):
        fails += 1
        names += " flush_no_effect"
    if not (a > c):
        fails += 1
        names += " copy_flush_not_worse"
    if fails == 0:
        print("PASS 3/3")
    else:
        print("FAIL", fails, "/ 3", names)
        raise Error("icflush gate failed")
