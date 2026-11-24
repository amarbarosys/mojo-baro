from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext

comptime FP = MutPointer[Scalar[DType.float32], MutAnyOrigin]


def vadd(a: FP, b: FP, c: FP, n: Int32):
    var i = Int(block_idx.x) * 64 + Int(thread_idx.x)
    if i < Int(n):
        c[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]


def main() raises:
    print("step 1 context")
    var ctx = DeviceContext()
    print("step 2 name", ctx.name())
    var a_h = ctx.enqueue_create_host_buffer[DType.float32](256)
    print("step 3 host buffer")
    var a_d = ctx.enqueue_create_buffer[DType.float32](256)
    var b_d = ctx.enqueue_create_buffer[DType.float32](256)
    var c_d = ctx.enqueue_create_buffer[DType.float32](256)
    print("step 4 device buffers")
    for i in range(256):
        a_h[i] = Float32(i)
    ctx.enqueue_copy(dst_buf=a_d, src_buf=a_h)
    ctx.enqueue_copy(dst_buf=b_d, src_buf=a_h)
    ctx.synchronize()
    print("step 5 copy h2d")
    var f = ctx.compile_function[vadd]()
    print("step 6 module load")
    ctx.enqueue_function(f, a_d.unsafe_ptr(), b_d.unsafe_ptr(), c_d.unsafe_ptr(), Int32(256), grid_dim=4, block_dim=64)
    ctx.synchronize()
    print("step 7 launch")
    ctx.enqueue_copy(dst_buf=a_h, src_buf=c_d)
    ctx.synchronize()
    print("step 8 result", a_h[3], "want 6")
