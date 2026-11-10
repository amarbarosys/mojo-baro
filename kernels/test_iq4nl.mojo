"""IQ4_NL decode + GEMV vs gguf-py on a real Flash-Next expert (tools/iq-oracle.py).

Gate: decoded weights bit-equal to gguf-py dequant; GEMV within 1e-4 relative of the
f64 reference; prints GB/s of the GEMV over the expert bytes.
"""
from std.memory import bitcast
from std.time import perf_counter_ns
from max.gpu import block_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, row_major
from moe import iq4nl_half, iq4nl_row_dot, IQ4NL_BYTES

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime u8 = DType.uint8
comptime DIR = ".work/p6/iq4nl/"


def decode(w: MutPointer[Scalar[u8], MutAnyOrigin], dst: MutPointer[Scalar[f32], MutAnyOrigin], nblk: Int32):
    var t = block_idx.x * 64 + Int(thread_idx.x)
    if t < Int(nblk) * 2:
        var v = iq4nl_half(w, (t // 2) * IQ4NL_BYTES, t % 2)
        dst.unsafe_offset((t // 2) * 32 + (t % 2) * 16).store(v)


def gemv[
    XL: TensorLayout
](X: TileTensor[bf16, XL, MutAnyOrigin], w: MutPointer[Scalar[u8], MutAnyOrigin], y: MutPointer[Scalar[f32], MutAnyOrigin], k: Int32, n: Int32):
    var row = block_idx.x * 8 + Int(thread_idx.x) // WARP_SIZE
    if row < Int(n):
        var s = iq4nl_row_dot(X, w, 0, row * (Int(k) // 32) * IQ4NL_BYTES, Int(k))
        if lane_id() == 0:
            y[unsafe_offset=row] = s


def load(name: String) raises -> List[UInt8]:
    with open(DIR + name, "r") as f:
        return f.read_bytes()


def main() raises:
    comptime K = 640
    comptime N = 2560
    var wb = load("w.bin")
    var deq = load("deq.f32")
    var xb = load("x.bf16")
    var yb = load("y.f64")
    if len(wb) != N * (K // 32) * IQ4NL_BYTES:
        raise Error("FAIL size: w.bin " + String(len(wb)))
    var ctx = DeviceContext()
    var w = ctx.enqueue_create_buffer[u8](len(wb))
    var dq = ctx.enqueue_create_buffer[f32](N * K)
    var x = ctx.enqueue_create_buffer[bf16](K)
    var y = ctx.enqueue_create_buffer[f32](N)
    with w.map_to_host() as h:
        for i in range(len(wb)):
            h[i] = wb[i]
    with x.map_to_host() as h:
        var p = xb.unsafe_ptr().unsafe_bitcast[Scalar[bf16]]()
        for i in range(K):
            h[i] = p[unsafe_offset=i]
    var nblk = N * K // 32
    ctx.enqueue_function[decode](w.unsafe_ptr(), dq.unsafe_ptr(), Int32(nblk), grid_dim=(nblk * 2 + 63) // 64, block_dim=64)
    var X = TileTensor(x, row_major[1, K]())
    comptime G = gemv[type_of(X).LayoutType]
    ctx.enqueue_function[G](X, w.unsafe_ptr(), y.unsafe_ptr(), Int32(K), Int32(N), grid_dim=N // 8, block_dim=8 * WARP_SIZE)
    ctx.synchronize()
    var bad = 0
    var refv = deq.unsafe_ptr().unsafe_bitcast[Scalar[f32]]()
    with dq.map_to_host() as h:
        for i in range(N * K):
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
    var it = 200
    var t0 = perf_counter_ns()
    for _ in range(it):
        ctx.enqueue_function[G](X, w.unsafe_ptr(), y.unsafe_ptr(), Int32(K), Int32(N), grid_dim=N // 8, block_dim=8 * WARP_SIZE)
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / 1e3 / Float64(it)
    print("iq4nl decode mismatches", bad, "of", N * K, "| gemv max rel err", worst, "|", us, "us/call", Float64(len(wb)) / us / 1e3, "GB/s (L2-warm)")
    if bad != 0 or worst > 1e-4:
        raise Error("FAIL iq4nl")
    print("PASS iq4nl")
