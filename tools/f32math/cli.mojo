"""Line protocol for bench/check-shared-math.sh: stdin lines are Float32 bits in hex; each answer line is the bits of
fexp32(x) and flog32(x) from kernels/bexp.mojo, `nan` for any NaN. `cli gpu` runs the same calls in a kernel."""
from std.sys import argv
from std.math import isnan, ceildiv
from std.memory import bitcast
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from bexp import fexp32, flog32

comptime PF = MutPointer[Float32, MutAnyOrigin]


def show(y: Float32) -> String:
    return String("nan") if isnan(y) else hex(bitcast[DType.uint32](y))


def k_explog(x: PF, e: PF, l: PF, n: Int32):
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i < Int(n):
        e[i] = fexp32(x[i])
        l[i] = flog32(x[i])


def main() raises:
    var xs = List[Float32]()
    for l in open("/dev/stdin", "r").read().split("\n"):
        if l.byte_length() > 0:
            xs.append(bitcast[DType.float32](UInt32(atol(String(l), 16))))
    var n = len(xs)
    var es = List[Float32](length=n, fill=0)
    var ls = List[Float32](length=n, fill=0)
    if len(argv()) > 1 and String(argv()[1]) == "gpu":
        var ctx = DeviceContext()
        var dx = ctx.enqueue_create_buffer[DType.float32](n)
        var de = ctx.enqueue_create_buffer[DType.float32](n)
        var dl = ctx.enqueue_create_buffer[DType.float32](n)
        ctx.enqueue_copy(dx, xs.unsafe_ptr())
        ctx.enqueue_function[k_explog](dx.unsafe_ptr(), de.unsafe_ptr(), dl.unsafe_ptr(), Int32(n), grid_dim=ceildiv(n, 256), block_dim=256)
        ctx.enqueue_copy(es.unsafe_ptr(), de)
        ctx.enqueue_copy(ls.unsafe_ptr(), dl)
        ctx.synchronize()
    else:
        for i in range(n):
            es[i] = fexp32(xs[i])
            ls[i] = flog32(xs[i])
    var out = String()
    for i in range(n):
        out += show(es[i]) + " " + show(ls[i]) + "\n"
    print(out, end="")
