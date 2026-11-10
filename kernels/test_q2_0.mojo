"""Q2_0 decode + GEMV vs tools/iq-oracle.py on a real GSQ-RCO Flash-Next expert.

Fixture: python tools/iq-oracle.py SHARD blk.0.ffn_down_exps.weight 0 .work/p6/q2_0  (Q2_0 decoded there from
llama.cpp's dequantize_row_q2_0). Gate: decoded weights bit-equal to the oracle; GEMV within 1e-4 relative of the
f64 reference. Prints GB/s of the GEMV over the expert bytes.
"""
from std.memory import bitcast
from std.time import perf_counter_ns
from max.gpu import block_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from q4x_kernels import q2_0_part, q2_0_row_dot, Q2_0_BYTES, Q4X_XR, Q4X_XK, Q4X_XL

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime u8 = DType.uint8
comptime DIR = ".work/p6/q2_0/"


def decode(w: MutPointer[Scalar[u8], MutAnyOrigin], dst: MutPointer[Scalar[f32], MutAnyOrigin], nblk: Int32):
    var t = block_idx.x * 64 + Int(thread_idx.x)
    if t < Int(nblk) * 4:
        dst.unsafe_offset((t // 4) * 64 + (t % 4) * 16).store(q2_0_part(w, (t // 4) * Q2_0_BYTES, t % 4))


def gemv(X: TileTensor[bf16, Q4X_XL, MutAnyOrigin], w: MutPointer[Scalar[u8], MutAnyOrigin], y: MutPointer[Scalar[f32], MutAnyOrigin], k: Int32, n: Int32):
    var row = block_idx.x * 8 + Int(thread_idx.x) // WARP_SIZE
    if row < Int(n):
        var s = q2_0_row_dot(X, w, 0, row * (Int(k) // 64) * Q2_0_BYTES, Int(k))
        if lane_id() == 0:
            y[unsafe_offset=row] = s


def load(name: String) raises -> List[UInt8]:
    with open(DIR + name, "r") as f:
        return f.read_bytes()


def main() raises:
    var wb = load("w.bin")
    var deq = load("deq.f32")
    var xb = load("x.bf16")
    var yb = load("y.f64")
    var K = len(xb) // 2
    var N = len(yb) // 8
    if len(wb) != N * (K // 64) * Q2_0_BYTES or len(deq) != N * K * 4:
        raise Error("FAIL size: w.bin " + String(len(wb)) + " for N " + String(N) + " K " + String(K))
    var ctx = DeviceContext()
    var w = ctx.enqueue_create_buffer[u8](len(wb))
    var dq = ctx.enqueue_create_buffer[f32](N * K)
    var x = ctx.enqueue_create_buffer[bf16](Q4X_XR * Q4X_XK)
    var y = ctx.enqueue_create_buffer[f32](N)
    with w.map_to_host() as h:
        for i in range(len(wb)):
            h[i] = wb[i]
    with x.map_to_host() as h:
        var p = xb.unsafe_ptr().unsafe_bitcast[Scalar[bf16]]()
        for i in range(K):
            h[i] = p[unsafe_offset=i]
    var nblk = N * K // 64
    ctx.enqueue_function[decode](w.unsafe_ptr(), dq.unsafe_ptr(), Int32(nblk), grid_dim=(nblk * 4 + 63) // 64, block_dim=64)
    var X = TileTensor(x, row_major[Q4X_XR, Q4X_XK]())
    ctx.enqueue_function[gemv](X, w.unsafe_ptr(), y.unsafe_ptr(), Int32(K), Int32(N), grid_dim=(N + 7) // 8, block_dim=8 * WARP_SIZE)
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
        ctx.enqueue_function[gemv](X, w.unsafe_ptr(), y.unsafe_ptr(), Int32(K), Int32(N), grid_dim=(N + 7) // 8, block_dim=8 * WARP_SIZE)
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / 1e3 / Float64(it)
    print("q2_0 K", K, "N", N, "decode mismatches", bad, "of", N * K, "| gemv max rel err", worst, "|", us, "us/call", Float64(len(wb)) / us / 1e3, "GB/s (L2-warm)")
    if bad != 0 or worst > 1e-4:
        raise Error("FAIL q2_0")
    print("PASS q2_0")
