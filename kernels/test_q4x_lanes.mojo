"""q4x GEMV lane (ggml type GT, -D GT=N) vs tools/iq-oracle.py on a real 2-D GGUF tensor in .work/p6/gt<GT>/.

Gate: GEMV within 1e-4 relative of the f64 reference over every row. GT=11 also decodes row 0 with the token_embd
kernel amar_q4x_q3k_row and requires it bit-equal to the oracle.
"""
from std.memory import bitcast
from std.sys import get_defined_int
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from std.os import getenv
from q4x_kernels import amar_q4x_gemm, amar_q4x_q3k_row

comptime Q4X_XR = 16
comptime Q4X_XK = 16384

comptime GT = get_defined_int["GT"]()
comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime u8 = DType.uint8


def load(name: String) raises -> List[UInt8]:
    with open(getenv("Q4X_FIX", ".work/p6/gt" + String(GT)) + "/" + name, "r") as f:
        return f.read_bytes()


def main() raises:
    var wb = load("w.bin")
    var deq = load("deq.f32")
    var xb = load("x.bf16")
    var yb = load("y.f64")
    var K = len(xb) // 2
    var N = len(yb) // 8
    var rowb = len(wb) // N
    var ctx = DeviceContext()
    var w = ctx.enqueue_create_buffer[u8](len(wb))
    var x = ctx.enqueue_create_buffer[bf16](Q4X_XR * Q4X_XK)
    var y = ctx.enqueue_create_buffer[f32](N)
    var r0 = ctx.enqueue_create_buffer[f32](K)
    with w.map_to_host() as h:
        for i in range(len(wb)):
            h[i] = wb[i]
    with x.map_to_host() as h:
        var p = xb.unsafe_ptr().unsafe_bitcast[Scalar[bf16]]()
        for i in range(K):
            h[i] = p[unsafe_offset=i]
    var X = TileTensor(x, row_major[Q4X_XR, Q4X_XK]())
    var xq = MutPointer[Scalar[bf16], MutAnyOrigin](unsafe_from_address=Int(x.unsafe_ptr()))
    var tiles = ctx.enqueue_create_buffer[DType.int32](4)
    with tiles.map_to_host() as h:
        h[0] = 0
        h[1] = 0
        h[2] = 1
        h[3] = 0
    var tp = MutPointer[Scalar[DType.int32], MutAnyOrigin](unsafe_from_address=Int(tiles.unsafe_ptr()))
    var tab = ctx.enqueue_create_buffer[DType.int64](1)
    tab.enqueue_fill(0)
    comptime KQ2 = GT == 11 or GT == 12 or GT == 14 or GT == 23
    var L = 32
    comptime if GT == 42:
        var nb = K // 64
        while L > 1 and nb % L != 0:
            L //= 2
    var pw = 2 if KQ2 else 32 // L
    print("lane: kq2" if KQ2 else "lane: generic", "lanes per row", L, "rows per wave", pw)
    ctx.enqueue_function[amar_q4x_gemm[GT, 1, KQ2, type_of(X).LayoutType]](X, w.unsafe_ptr(), tab.unsafe_ptr(), tp, tp, tp, y.unsafe_ptr(), xq, Int32(K), Int32(N), Int32(N), Int32(rowb), Int32(L), Int32(0), Int32(0), Int32(0), Float32(0), Int32(0), grid_dim=((N + 8 * pw - 1) // (8 * pw), 1), block_dim=256)
    comptime if GT == 11:
        ctx.enqueue_function[amar_q4x_q3k_row](w.unsafe_ptr(), r0.unsafe_ptr(), Int32(K), tiles.unsafe_ptr(), Int32(rowb), Int32(K), grid_dim=((K // 256 * 8 + 63) // 64, 1), block_dim=64)
    ctx.synchronize()
    var bad = 0
    comptime if GT == 11:
        var refv = deq.unsafe_ptr().unsafe_bitcast[Scalar[f32]]()
        with r0.map_to_host() as h:
            for i in range(K):
                if bitcast[DType.uint32, 1](SIMD[f32, 1](h[i])) != bitcast[DType.uint32, 1](SIMD[f32, 1](refv[unsafe_offset=i])):
                    if bad < 3:
                        print("  mismatch", i, h[i], refv[unsafe_offset=i])
                    bad += 1
    var yr = yb.unsafe_ptr().unsafe_bitcast[Float64]()
    var worst = 0.0
    var scale = 0.0
    with y.map_to_host() as h:
        for i in range(N):
            scale = max(scale, abs(yr[unsafe_offset=i]))
        for i in range(N):
            worst = max(worst, abs(Float64(h[i]) - yr[unsafe_offset=i]) / scale)
    print("gt", GT, "K", K, "N", N, "row0 decode mismatches", bad, "| gemv max rel err", worst)
    if bad != 0 or worst > 1e-4:
        raise Error("FAIL gt" + String(GT))
    print("PASS gt" + String(GT))
