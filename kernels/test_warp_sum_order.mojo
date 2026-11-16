"""Step 0c (docs/design/CPU-BACKEND.md): the order of `warp.sum` on gfx1100 against CPU 32-lane trees, byte test, one dot.

Prediction, frozen before the first run: `warp.sum` on gfx1100 equals the xor butterfly the megakernel's `kq_wave_sum`
writes by hand (xor 1, 2, 4, 8, 16), so CPU tree "xor-asc" is bit-equal on every trial and every lane; the other orders
(xor-desc, sequential left fold, shuffle-down halving) differ on a large share of trials.

Each trial is one wave of 32 fp32 lane values (random sign, magnitude 2^-12 .. 2^12, so a different order changes bits).
Exit 0 only if xor-asc is bit-equal to warp.sum on lane 0 AND every lane of every trial. Prints, per CPU order, the
mismatching trials and the differing bits (the xor of the two results' bit patterns: count of set bits, highest set bit).
usage: test_warp_sum_order    (GPU)
"""
from std.memory import bitcast
from max.gpu import block_idx, thread_idx
from max.gpu.primitives import warp
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

comptime f32 = DType.float32
comptime T = 4096
comptime W = 32
comptime in_layout = row_major[T, W]()
comptime out_layout = row_major[T, W]()


def wave_sum_kernel(
    X: TileTensor[f32, type_of(in_layout), MutAnyOrigin],
    Y: TileTensor[f32, type_of(out_layout), MutAnyOrigin],
):
    var t = block_idx.x
    var l = thread_idx.x
    var v = rebind[Scalar[f32]](X[t, l])
    var s = warp.sum(v)
    Y[t, l] = rebind[Y.ElementType](s)


def xor_tree(v: List[Float32], ascending: Bool) -> Float32:
    var a = v.copy()
    var steps = [1, 2, 4, 8, 16] if ascending else [16, 8, 4, 2, 1]
    for s in steps:
        var b = a.copy()
        for l in range(W):
            b[l] = a[l] + a[l ^ s]
        a = b^
    return a[0]


def seq_fold(v: List[Float32]) -> Float32:
    var acc = v[0]
    for l in range(1, W):
        acc = acc + v[l]
    return acc


def down_halving(v: List[Float32]) -> Float32:
    var a = v.copy()
    var s = 16
    while s >= 1:
        for l in range(s):
            a[l] = a[l] + a[l + s]
        s //= 2
    return a[0]


def bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32, 1](x)


def main() raises:
    var ctx = DeviceContext()
    var xs = List[Float32](length=T * W, fill=0)
    var state = UInt64(88172645463325252)
    for i in range(T * W):
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        var m = Float32(Int((state >> 11) & 0xFFFFF)) / Float32(1048576.0) + Float32(0.5)
        var e = Int((state >> 40) % 25) - 12
        var sc = Float32(1.0)
        for _ in range(abs(e)):
            sc = sc * Float32(2.0) if e > 0 else sc * Float32(0.5)
        var sgn = Float32(-1.0) if ((state >> 33) & 1) == 1 else Float32(1.0)
        xs[i] = sgn * m * sc
    var x_d = ctx.enqueue_create_buffer[f32](T * W)
    var y_d = ctx.enqueue_create_buffer[f32](T * W)
    ctx.enqueue_copy(x_d, xs.unsafe_ptr())
    var X = TileTensor(x_d, in_layout)
    var Y = TileTensor(y_d, out_layout)
    ctx.enqueue_function[wave_sum_kernel](X, Y, grid_dim=T, block_dim=W)
    var got = List[Float32](length=T * W, fill=0)
    ctx.enqueue_copy(got.unsafe_ptr(), y_d)
    ctx.synchronize()
    print("arm gpu=warp.sum (max.gpu.primitives) cpu=32-lane trees, trials", T, "wave", W)
    var names = ["xor-asc", "xor-desc", "seq-fold", "down-halving"]
    var bad_lane0 = List[Int](length=4, fill=0)
    var bad_alllane = List[Int](length=4, fill=0)
    var maxbit = List[Int](length=4, fill=-1)
    var popsum = List[Int](length=4, fill=0)
    var lanes_equal = 0
    for t in range(T):
        var v = List[Float32](length=W, fill=0)
        for l in range(W):
            v[l] = xs[t * W + l]
        var refs = [xor_tree(v, True), xor_tree(v, False), seq_fold(v), down_halving(v)]
        var all_same = True
        for l in range(W):
            if bits(got[t * W + l]) != bits(got[t * W]):
                all_same = False
        if all_same:
            lanes_equal += 1
        for k in range(4):
            var d = bits(got[t * W]) ^ bits(refs[k])
            if d != 0:
                bad_lane0[k] += 1
                for b in range(32):
                    if (d >> UInt32(b)) & 1 == 1:
                        popsum[k] += 1
                        if b > maxbit[k]:
                            maxbit[k] = b
            for l in range(W):
                if bits(got[t * W + l]) != bits(refs[k]):
                    bad_alllane[k] += 1
                    break
    print("trials where all 32 lanes of warp.sum hold the same bits:", lanes_equal, "of", T)
    for k in range(4):
        print(names[k], ": lane0 mismatches", bad_lane0[k], "of", T, " any-lane mismatches", bad_alllane[k], " differing bits total", popsum[k], " highest differing bit", maxbit[k])
    if bad_lane0[0] != 0 or bad_alllane[0] != 0 or lanes_equal != T:
        raise Error("MISMATCH: xor-asc is not bit-equal to warp.sum; see the per-order lines above")
    print("PASS test_warp_sum_order: warp.sum on this device == xor-asc 32-lane tree, bit for bit, lane 0 and all lanes")
