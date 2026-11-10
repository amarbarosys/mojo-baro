from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns

comptime IC_BYTES = 256 * 1024 * 1024
comptime IC_THREADS = 256
comptime IC_GRID = 4096
comptime IC_PEAK_GBPS = 960.0
comptime ICP = MutPointer[Scalar[DType.uint64], MutAnyOrigin]


def ic_read_sweep(src: ICP, words: Int64, dst: ICP):
    var gid = Int(block_idx.x) * IC_THREADS + Int(thread_idx.x)
    var acc = SIMD[DType.uint64, 2](0)
    for j in range(gid * 2, Int(words), IC_GRID * IC_THREADS * 2):
        acc += src.unsafe_offset(j).unsafe_load[width=2]()
    dst[unsafe_offset=gid] = acc[0] + acc[1]


@fieldwise_init
struct ICRuler(Copyable, ImplicitlyCopyable, Movable):
    var copy_gbps: Float64
    var read_gbps: Float64

    def best(self) -> Float64:
        return max(self.copy_gbps, self.read_gbps)

    def units_ok(self) -> Bool:
        var lo = 0.3 * IC_PEAK_GBPS
        var hi = 1.1 * IC_PEAK_GBPS
        return self.copy_gbps >= lo and self.copy_gbps <= hi and self.read_gbps >= lo and self.read_gbps <= hi


def ic_median(v: List[Float64]) -> Float64:
    var s = v.copy()
    sort(s)
    return s[len(s) // 2]


struct ICFlush(Movable):
    var src: DeviceBuffer[DType.uint64]
    var dst: DeviceBuffer[DType.uint64]
    var sums: DeviceBuffer[DType.uint64]

    def __init__(out self, ctx: DeviceContext) raises:
        self.src = ctx.enqueue_create_buffer[DType.uint64](IC_BYTES // 8)
        self.dst = ctx.enqueue_create_buffer[DType.uint64](IC_BYTES // 8)
        self.sums = ctx.enqueue_create_buffer[DType.uint64](IC_GRID * IC_THREADS)
        self.src.enqueue_fill(1)
        ctx.synchronize()

    def flush(self, ctx: DeviceContext) raises:
        ctx.enqueue_function[ic_read_sweep](self.src.unsafe_ptr(), Int64(IC_BYTES // 8), self.sums.unsafe_ptr(), grid_dim=IC_GRID, block_dim=IC_THREADS)

    def ruler(self, ctx: DeviceContext, iters: Int = 14) raises -> ICRuler:
        var cp = List[Float64]()
        var rd = List[Float64]()
        for it in range(iters):
            ctx.synchronize()
            var t0 = perf_counter_ns()
            ctx.enqueue_copy(dst_buf=self.dst, src_buf=self.src)
            ctx.synchronize()
            var t1 = perf_counter_ns()
            self.flush(ctx)
            ctx.synchronize()
            var t2 = perf_counter_ns()
            if it >= 3:
                cp.append(2.0 * Float64(IC_BYTES) / Float64(t1 - t0))
                rd.append(Float64(IC_BYTES) / Float64(t2 - t1))
        return ICRuler(ic_median(cp), ic_median(rd))
