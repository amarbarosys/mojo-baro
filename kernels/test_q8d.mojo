"""Dense q8 (q8d) row dot on a real pack tensor: q8d_row_dot vs the unbatched reference loop, cold-cache GB/s and roof.

usage: test_q8d PACK_DIR TENSOR K
Reads TENSOR's bytes (quants n*k, then f16 scales per 32) from PACK_DIR/pack.bin via index.txt.
Gate: q8d_row_dot output bit-identical to ref_dot (same per-element fma order).
Cold cache: >= 128 MB of rotated copies; roof = read-only kernel with the same launch.
NOT predictive of the engine: the old loop read 318 GB/s here and ~470 in the engine trace; a
1.35-1.68x win here was 0.92x resident decode (7f0e48f reverted). Judge with bench/moe-token-profile.sh.
"""
from std.memory import bitcast
from std.sys import argv
from std.time import perf_counter_ns
from std.math import fma
from max.gpu import block_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, row_major
from moe import q8d_row_dot

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime u8 = DType.uint8
comptime u16 = DType.uint16
comptime f16 = DType.float16
comptime i8 = DType.int8
comptime MAXK = 8192


@always_inline
def ref_dot[
    XLayout: TensorLayout,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    x_row: Int,
    q_base: Int,
    s_base: Int,
    k_dim: Int,
) -> Scalar[f32]:
    comptime assert X.flat_rank == 2
    var lane = Int(lane_id())
    var Xv = X.vectorize[1, 16]()
    var nb = k_dim // 32
    var acc = SIMD[f32, 16](0)
    var b = lane // 2
    var h = lane % 2
    while b < nb:
        var raw = W.unsafe_offset(s_base + b * 2).unsafe_bitcast[Scalar[u16]]()[]
        var d = bitcast[f16, 1](SIMD[u16, 1](raw)).cast[f32]()[0]
        var q = W.unsafe_offset(q_base + b * 32 + h * 16).unsafe_bitcast[Scalar[i8]]().load[width=16]()
        var v = (SIMD[f32, 16](d) * q.cast[f32]()).cast[bf16]().cast[f32]()
        var a = rebind[SIMD[bf16, 16]](Xv[x_row, 2 * b + h]).cast[f32]()
        acc = fma(v, a, acc)
        b += 16
    return warp.sum(acc.reduce_add())


def gemv[U: Int, XL: TensorLayout](X: TileTensor[bf16, XL, MutAnyOrigin], w: MutPointer[Scalar[u8], MutAnyOrigin], y: MutPointer[Scalar[f32], MutAnyOrigin], n: Int32, k: Int32):
    var row = block_idx.x * 8 + Int(thread_idx.x) // WARP_SIZE
    if row < Int(n):
        var kk = Int(k)
        var s: Scalar[f32]
        comptime if U == 0:
            s = ref_dot(X, w, 0, row * kk, Int(n) * kk + row * (kk // 32) * 2, kk)
        else:
            s = q8d_row_dot(X, w, 0, row * kk, Int(n) * kk + row * (kk // 32) * 2, kk)
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


def main() raises:
    var a = argv()
    if len(a) != 4:
        raise Error("usage: test_q8d PACK_DIR TENSOR K")
    var dir = String(a[1])
    var name = String(a[2])
    var k = atol(String(a[3]))
    var idx: String
    with open(dir + "/index.txt", "r") as f:
        idx = f.read()
    var off = -1
    var elems = 0
    for line in idx.split("\n"):
        var c = line.split(" ")
        if len(c) == 4 and c[0] == name:
            off = atol(c[2])
            elems = atol(c[3])
    if off < 0 or elems % k != 0 or k > MAXK or k % 512 != 0:
        raise Error("FAIL tensor " + name + " k " + String(k))
    var n = elems // k
    var nbytes = elems + (elems // 32) * 2
    var wb: List[UInt8]
    with open(dir + "/pack.bin", "r") as f:
        _ = f.seek(off)
        wb = f.read_bytes(nbytes)
    var ctx = DeviceContext()
    var copies = max(1, (128 << 20) // nbytes + 1)
    var w = ctx.enqueue_create_buffer[u8](nbytes * copies)
    var x = ctx.enqueue_create_buffer[bf16](MAXK)
    var y = ctx.enqueue_create_buffer[f32](n)
    with w.map_to_host() as h:
        for c in range(copies):
            for i in range(nbytes):
                h[c * nbytes + i] = wb[i]
    with x.map_to_host() as h:
        for i in range(k):
            h[i] = Scalar[f32](Float32((i * 7919) % 1000) / 500.0 - 1.0).cast[bf16]()
    var X = TileTensor(x, row_major[1, MAXK]())
    var it = copies * 20
    var base = List[Float32]()
    var us0 = 0.0
    var bad = 0
    comptime for U in [0, 1]:
        comptime G = gemv[U, type_of(X).LayoutType]
        ctx.enqueue_function[G](X, w.unsafe_ptr(), y.unsafe_ptr(), Int32(n), Int32(k), grid_dim=(n + 7) // 8, block_dim=256)
        ctx.synchronize()
        var diff = 0
        with y.map_to_host() as h:
            for i in range(n):
                if U == 0:
                    base.append(h[i])
                elif bitcast[DType.uint32, 1](SIMD[f32, 1](h[i])) != bitcast[DType.uint32, 1](SIMD[f32, 1](base[i])):
                    diff += 1
        var t0 = perf_counter_ns()
        for i in range(it):
            ctx.enqueue_function[G](X, w.unsafe_ptr() + (i % copies) * nbytes, y.unsafe_ptr(), Int32(n), Int32(k), grid_dim=(n + 7) // 8, block_dim=256)
        ctx.synchronize()
        var us = Float64(perf_counter_ns() - t0) / 1e3 / Float64(it)
        if U == 0:
            us0 = us
        print(name, "n", n, "k", k, "U", U, "|", us, "us", Float64(nbytes) / us / 1e3, "GB/s cold | vs base", us0 / us, "| bit-diff", diff)
        bad += diff
    var t0 = perf_counter_ns()
    for i in range(it):
        ctx.enqueue_function[roof](w.unsafe_ptr() + (i % copies) * nbytes, y.unsafe_ptr(), Int32(k), Int32(n), grid_dim=(n + 7) // 8, block_dim=256)
    ctx.synchronize()
    var ur = Float64(perf_counter_ns() - t0) / 1e3 / Float64(it)
    print(name, "roof (read-only, same launch, quants only)", Float64(n * k) / ur / 1e3, "GB/s | base at", 100 * Float64(nbytes) / us0 / 1e3 / (Float64(n * k) / ur / 1e3), "% of roof")
    if bad != 0:
        raise Error("FAIL " + name)
    print("PASS", name)
