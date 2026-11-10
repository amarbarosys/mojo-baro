from max.gpu import block_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import row_major, stack_allocation
from std.math import isfinite

comptime NP_THREADS = 256


# Stage tracer (BARO_NANPROBE=1, serve/window.mojo trace_rows): one block per
# row of a [rows, stride] buffer, counts non-finite values in its first n
# columns into Out[row]. Device-side so a probe costs one launch and an m-int
# copy, not a host scan of the whole buffer.
def amar_nonfinite_rows[dt: DType](
    X: UnsafePointer[Scalar[dt], MutAnyOrigin],
    Out: UnsafePointer[Int32, MutAnyOrigin],
    stride: Int32, n: Int32,
):
    var r = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var cnt = stack_allocation[DType.int32, address_space = AddressSpace.SHARED](row_major[NP_THREADS]())
    var c: Int32 = 0
    var i = t
    while i < Int(n):
        if not isfinite(X[r * Int(stride) + i].cast[DType.float32]()):
            c += 1
        i += NP_THREADS
    cnt[t] = c
    barrier()
    if t == 0:
        var s: Int32 = 0
        for k in range(NP_THREADS):
            s += rebind[Int32](cnt[k])
        Out[r] = s
