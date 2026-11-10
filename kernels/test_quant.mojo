"""K-family / IQ4_XS row dots vs gguf-py on a real tensor (tools/iq-oracle.py dump).

usage: test_quant TYPE DIR     TYPE in Q4_K Q5_K IQ4_XS Q3_K Q2_K IQ2_XXS IQ1_M (Q4_K int8 = q4k_q8_row_dot[14], fixture k must be 3584)
Also the int8-dot4 path (x quantized per 32-group by amar_quantize_q8_rows, the engine's
activation quantizer): GEMV max rel err <= 1e-2 vs f64 (round 7 block-parity bar).
Gate: every decoded weight bit-equal to gguf-py; GEMV max rel err <= 1e-4 of max|y| vs f64.
Q4_K is the speed baseline (q4k_dot_blocks rounds weights to bf16 for engine identity:
no decode check, GEMV bar 5e-3); cold-cache GB/s
over >= 128 MB of rotated weight copies (CLAUDE.md: < 96 MB sits in Infinity Cache).
"""
from std.memory import bitcast
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu import block_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major
from elementwise import amar_quantize_q8_rows
from moe import (
    KLane, kq_row_dot, kq_q8_row_dot, q4k_dot_blocks, q4k_q8_row_dot, iq2xxs_lane, IQ2XXS_BYTES, iq1m_lane, IQ1M_BYTES, q5k_lane, iq4xs_lane, q3k_lane, q2k_lane,
    Q4K_BYTES, Q5K_BYTES, IQ4XS_BYTES, Q3K_BYTES, Q2K_BYTES,
)

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime u8 = DType.uint8
comptime f16 = DType.float16
comptime i8 = DType.int8
comptime MAXK = 32768


def decode[
    lane_fn: def(MutPointer[Scalar[u8], MutAnyOrigin], Int, Int) thin -> KLane, BYTES: Int
](w: MutPointer[Scalar[u8], MutAnyOrigin], dst: MutPointer[Scalar[f32], MutAnyOrigin], nblk: Int32):
    var t = block_idx.x * 64 + Int(thread_idx.x)
    if t < Int(nblk) * 8:
        var r = lane_fn(w, (t // 8) * BYTES, t % 8)
        dst.unsafe_offset((t // 8) * 256 + r[6]).store(SIMD[f32, 16](r[2]) * r[0].cast[f32]() - SIMD[f32, 16](r[4]))
        dst.unsafe_offset((t // 8) * 256 + r[7]).store(SIMD[f32, 16](r[3]) * r[1].cast[f32]() - SIMD[f32, 16](r[5]))


def gemv[
    lane_fn: def(MutPointer[Scalar[u8], MutAnyOrigin], Int, Int) thin -> KLane, BYTES: Int, Q4: Bool, XL: TensorLayout
](X: TileTensor[bf16, XL, MutAnyOrigin], w: MutPointer[Scalar[u8], MutAnyOrigin], y: MutPointer[Scalar[f32], MutAnyOrigin], k: Int32, n: Int32):
    var row = block_idx.x * 8 + Int(thread_idx.x) // WARP_SIZE
    if row < Int(n):
        var rb = row * (Int(k) // 256) * BYTES
        var s: Scalar[f32]
        comptime if Q4:
            s = q4k_dot_blocks(X, w, 0, rb, Int(k))
        else:
            s = kq_row_dot[lane_fn, BYTES](X, w, 0, rb, Int(k))
        if lane_id() == 0:
            y[unsafe_offset=row] = s


def gemv8[
    lane_fn: def(MutPointer[Scalar[u8], MutAnyOrigin], Int, Int) thin -> KLane, BYTES: Int, HAS_MIN: Bool, Q4: Bool
](xq: MutPointer[Scalar[i8], MutAnyOrigin], xs: MutPointer[Scalar[f16], MutAnyOrigin], w: MutPointer[Scalar[u8], MutAnyOrigin], y: MutPointer[Scalar[f32], MutAnyOrigin], k: Int32, n: Int32):
    var row = block_idx.x * 8 + Int(thread_idx.x) // WARP_SIZE
    if row < Int(n):
        var rb = row * (Int(k) // 256) * BYTES
        var s: Scalar[f32]
        comptime if Q4:
            s = q4k_q8_row_dot[14](xq, xs, 0, w, rb)
        else:
            s = kq_q8_row_dot[lane_fn, BYTES, HAS_MIN](xq, xs, 0, w, rb, Int(k))
        if lane_id() == 0:
            y[unsafe_offset=row] = s


def roof(w: MutPointer[Scalar[u8], MutAnyOrigin], y: MutPointer[Scalar[f32], MutAnyOrigin], rowb: Int32, n: Int32):
    var row = block_idx.x * 8 + Int(thread_idx.x) // WARP_SIZE
    if row < Int(n):
        var p = w.unsafe_offset(row * Int(rowb)).unsafe_bitcast[Scalar[DType.uint32]]()
        var acc = SIMD[DType.uint32, 4](0)
        for i in range(lane_id() * 4, Int(rowb) // 4, WARP_SIZE * 4):
            acc ^= p.unsafe_offset(i).unsafe_load[width=4]()
        if acc.reduce_add() == 0x9E3779B9:
            y[unsafe_offset=row] = 1.0


def load(path: String) raises -> List[UInt8]:
    with open(path, "r") as f:
        return f.read_bytes()


def run[lane_fn: def(MutPointer[Scalar[u8], MutAnyOrigin], Int, Int) thin -> KLane, BYTES: Int, Q4: Bool, HAS_MIN: Bool](name: String, dir: String) raises:
    var meta: String
    with open(dir + "/meta.txt", "r") as f:
        meta = f.read()
    var k = atol(meta.split("k=")[1].split(" ")[0])
    var n = atol(meta.split("n=")[1].split(" ")[0])
    var wb = load(dir + "/w.bin")
    var deq = load(dir + "/deq.f32")
    var xb = load(dir + "/x.bf16")
    var yb = load(dir + "/y.f64")
    if not meta.startswith(name) or len(wb) != n * (k // 256) * BYTES or k > MAXK:
        raise Error("FAIL fixture: " + meta + " bytes " + String(len(wb)))
    var ctx = DeviceContext()
    var copies = max(1, (128 << 20) // len(wb) + 1)
    var w = ctx.enqueue_create_buffer[u8](len(wb) * copies)
    var dq = ctx.enqueue_create_buffer[f32](n * k)
    var x = ctx.enqueue_create_buffer[bf16](MAXK)
    var y = ctx.enqueue_create_buffer[f32](n)
    with w.map_to_host() as h:
        for c in range(copies):
            for i in range(len(wb)):
                h[c * len(wb) + i] = wb[i]
    with x.map_to_host() as h:
        var p = xb.unsafe_ptr().unsafe_bitcast[Scalar[bf16]]()
        for i in range(k):
            h[i] = p[unsafe_offset=i]
    var nblk = n * k // 256
    var bad = 0
    comptime if not Q4:
        ctx.enqueue_function[decode[lane_fn, BYTES]](w.unsafe_ptr(), dq.unsafe_ptr(), Int32(nblk), grid_dim=(nblk * 8 + 63) // 64, block_dim=64)
        ctx.synchronize()
        var refv = deq.unsafe_ptr().unsafe_bitcast[Scalar[f32]]()
        with dq.map_to_host() as h:
            for i in range(n * k):
                if bitcast[DType.uint32, 1](SIMD[f32, 1](h[i])) != bitcast[DType.uint32, 1](SIMD[f32, 1](refv[unsafe_offset=i])):
                    if bad < 3:
                        print("  mismatch", i, h[i], refv[unsafe_offset=i])
                    bad += 1
    var X = TileTensor(x, row_major[1, MAXK]())
    comptime G = gemv[lane_fn, BYTES, Q4, type_of(X).LayoutType]
    ctx.enqueue_function[G](X, w.unsafe_ptr(), y.unsafe_ptr(), Int32(k), Int32(n), grid_dim=(n + 7) // 8, block_dim=8 * WARP_SIZE)
    ctx.synchronize()
    var yr = yb.unsafe_ptr().unsafe_bitcast[Float64]()
    var worst = 0.0
    var scale = 0.0
    with y.map_to_host() as h:
        for i in range(n):
            scale = max(scale, abs(yr[unsafe_offset=i]))
        for i in range(n):
            worst = max(worst, abs(Float64(h[i]) - yr[unsafe_offset=i]) / scale)
    var it = copies * 20
    var t0 = perf_counter_ns()
    for i in range(it):
        ctx.enqueue_function[G](X, w.unsafe_ptr() + (i % copies) * len(wb), y.unsafe_ptr(), Int32(k), Int32(n), grid_dim=(n + 7) // 8, block_dim=8 * WARP_SIZE)
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / 1e3 / Float64(it)
    var gbs = Float64(len(wb)) / us / 1e3
    print(name, "k", k, "n", n, "bytes/w", Float64(BYTES) * 8 / 256, "| decode mismatches", bad, "of", n * k if not Q4 else 0,
          "| gemv max rel err", worst, "|", us, "us", gbs, "GB/s cold", gbs * 8 / (Float64(BYTES) * 8 / 256), "Gw/s | copies", copies)
    var worst8 = 0.0
    if True:
        var xq = ctx.enqueue_create_buffer[DType.int8](MAXK)
        var xs = ctx.enqueue_create_buffer[DType.float16](MAXK // 32)
        comptime QK = amar_quantize_q8_rows[type_of(X).LayoutType, type_of(row_major[1, MAXK]()), type_of(row_major[1, MAXK // 32]())]
        ctx.enqueue_function[QK](X, TileTensor(xq, row_major[1, MAXK]()), TileTensor(xs, row_major[1, MAXK // 32]()), Int32(1), Int32(k), grid_dim=(1, k // 32), block_dim=WARP_SIZE)
        comptime G8 = gemv8[lane_fn, BYTES, HAS_MIN, Q4]
        ctx.enqueue_function[G8](xq.unsafe_ptr(), xs.unsafe_ptr(), w.unsafe_ptr(), y.unsafe_ptr(), Int32(k), Int32(n), grid_dim=(n + 7) // 8, block_dim=8 * WARP_SIZE)
        ctx.synchronize()
        with y.map_to_host() as h:
            for i in range(n):
                worst8 = max(worst8, abs(Float64(h[i]) - yr[unsafe_offset=i]) / scale)
        t0 = perf_counter_ns()
        for i in range(it):
            ctx.enqueue_function[G8](xq.unsafe_ptr(), xs.unsafe_ptr(), w.unsafe_ptr() + (i % copies) * len(wb), y.unsafe_ptr(), Int32(k), Int32(n), grid_dim=(n + 7) // 8, block_dim=8 * WARP_SIZE)
        ctx.synchronize()
        var us8 = Float64(perf_counter_ns() - t0) / 1e3 / Float64(it)
        t0 = perf_counter_ns()
        for i in range(it):
            ctx.enqueue_function[roof](w.unsafe_ptr() + (i % copies) * len(wb), y.unsafe_ptr(), Int32(len(wb) // n), Int32(n), grid_dim=(n + 7) // 8, block_dim=8 * WARP_SIZE)
        ctx.synchronize()
        var roofgb = Float64(len(wb)) / (Float64(perf_counter_ns() - t0) / 1e3 / Float64(it)) / 1e3
        print(name, "int8-dot4 | gemv max rel err", worst8, "|", us8, "us", Float64(len(wb)) / us8 / 1e3, "GB/s cold | speedup vs f32", us / us8)
        print(name, "roof (read-only, same launch)", roofgb, "GB/s | int8 at", 100 * Float64(len(wb)) / us8 / 1e3 / roofgb, "% of roof")
    if bad != 0 or worst > (5e-3 if Q4 else 1e-4) or worst8 > 1e-2:
        raise Error("FAIL " + name)
    print("PASS", name)


def main() raises:
    var a = argv()
    if len(a) != 3:
        raise Error("usage: test_quant TYPE DIR")
    var t = String(a[1])
    var d = String(a[2])
    if t == "Q4_K":
        run[q5k_lane, Q4K_BYTES, True, True](t, d)
    elif t == "Q5_K":
        run[q5k_lane, Q5K_BYTES, False, True](t, d)
    elif t == "IQ4_XS":
        run[iq4xs_lane, IQ4XS_BYTES, False, False](t, d)
    elif t == "Q3_K":
        run[q3k_lane, Q3K_BYTES, False, False](t, d)
    elif t == "IQ1_M":
        run[iq1m_lane, IQ1M_BYTES, False, False](t, d)
    elif t == "IQ2_XXS":
        run[iq2xxs_lane, IQ2XXS_BYTES, False, False](t, d)
    elif t == "Q2_K":
        run[q2k_lane, Q2K_BYTES, False, True](t, d)
    else:
        raise Error("FAIL type " + t)
