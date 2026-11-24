"""Vendor parity: arch.sdot4 / arch.sudot4 on the GPU (amdgcn sdot4/sudot4, NVIDIA idp4a)
against a byte-sum reference on the host. Runs on any GPU Mojo targets."""
from std.math import ceildiv
from std.sys import has_accelerator

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext

from arch import sdot4, sudot4

comptime i32 = DType.int32
comptime IP = MutPointer[Scalar[i32], MutAnyOrigin]
comptime N = 4096
comptime T = 256


def dots(a: IP, b: IP, c: IP, o: IP):
    var i = Int(block_idx.x) * T + Int(thread_idx.x)
    if i < N:
        var x = a[unsafe_offset=i]
        var y = b[unsafe_offset=i]
        var z = c[unsafe_offset=i]
        o[unsafe_offset=5 * i] = sdot4(x, y, z, False)
        o[unsafe_offset=5 * i + 1] = sudot4(True, x, True, y, z, False)
        o[unsafe_offset=5 * i + 2] = sudot4(True, x, False, y, z, False)
        o[unsafe_offset=5 * i + 3] = sudot4(False, x, True, y, z, False)
        o[unsafe_offset=5 * i + 4] = sudot4(False, x, False, y, z, False)


def want_of(a: Int32, sa: Bool, b: Int32, sb: Bool, c: Int32) -> Int32:
    var r = c
    for k in range(4):
        var x = (a >> Int32(8 * k)) & Int32(0xFF)
        var y = (b >> Int32(8 * k)) & Int32(0xFF)
        if sa and x >= 128:
            x -= 256
        if sb and y >= 128:
            y -= 256
        r += x * y
    return r


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var a_h = ctx.enqueue_create_host_buffer[i32](N)
    var b_h = ctx.enqueue_create_host_buffer[i32](N)
    var c_h = ctx.enqueue_create_host_buffer[i32](N)
    var o_h = ctx.enqueue_create_host_buffer[i32](5 * N)
    var a_d = ctx.enqueue_create_buffer[i32](N)
    var b_d = ctx.enqueue_create_buffer[i32](N)
    var c_d = ctx.enqueue_create_buffer[i32](N)
    var o_d = ctx.enqueue_create_buffer[i32](5 * N)
    ctx.synchronize()
    var s = UInt32(20261116)
    for i in range(N):
        s = s * 1664525 + 1013904223
        a_h[i] = Int32(Int(s))
        s = s * 1664525 + 1013904223
        b_h[i] = Int32(Int(s))
        s = s * 1664525 + 1013904223
        c_h[i] = Int32(Int(s >> 12)) - Int32(1 << 19)
    a_h[0] = Int32(-1)
    b_h[0] = Int32(-1)
    a_h[1] = Int32(0x7F7F7F7F)
    b_h[1] = Int32(Int(UInt32(0x80808080)))
    ctx.enqueue_copy(dst_buf=a_d, src_buf=a_h)
    ctx.enqueue_copy(dst_buf=b_d, src_buf=b_h)
    ctx.enqueue_copy(dst_buf=c_d, src_buf=c_h)
    ctx.enqueue_function[dots](
        a_d.unsafe_ptr(), b_d.unsafe_ptr(), c_d.unsafe_ptr(), o_d.unsafe_ptr(),
        grid_dim=ceildiv(N, T), block_dim=T,
    )
    ctx.enqueue_copy(dst_buf=o_h, src_buf=o_d)
    ctx.synchronize()
    var bad = 0
    for i in range(N):
        comptime for v in range(5):
            var want = want_of(a_h[i], v < 3, b_h[i], v != 2 and v != 4, c_h[i])
            var got = o_h[5 * i + v]
            if got != want:
                if bad < 5:
                    print("MISMATCH i", i, "variant", v, "got", got, "want", want)
                bad += 1
    if bad:
        raise Error("arch dot mismatch: ", bad, " of ", 5 * N)
    print("ARCH DOT OK", 5 * N, "dot4 results match the byte reference")
