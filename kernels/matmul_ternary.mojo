from max.gpu import block_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from layout import TileTensor, TensorLayout

from matmul import dtype
from matmul_skinny import ROW_WAVES

comptime B3_BLOCK = 128
comptime B3_BYTES = 26
comptime TQ_BLOCK = 256
comptime TQ1_BYTES = 52
comptime TQ2_BYTES = 64
comptime TV = 16

comptime B3LUT_X: Array[UInt32, 256] = [
    0x00000000, 0x00000001, 0x00000002, 0x00000100, 0x00000101, 0x00000102, 0x00000200, 0x00000201,
    0x00000202, 0x00010000, 0x00010001, 0x00010002, 0x00010100, 0x00010101, 0x00010102, 0x00010200,
    0x00010201, 0x00010202, 0x00020000, 0x00020001, 0x00020002, 0x00020100, 0x00020101, 0x00020102,
    0x00020200, 0x00020201, 0x00020202, 0x01000000, 0x01000001, 0x01000002, 0x01000100, 0x01000101,
    0x01000102, 0x01000200, 0x01000201, 0x01000202, 0x01010000, 0x01010001, 0x01010002, 0x01010100,
    0x01010101, 0x01010102, 0x01010200, 0x01010201, 0x01010202, 0x01020000, 0x01020001, 0x01020002,
    0x01020100, 0x01020101, 0x01020102, 0x01020200, 0x01020201, 0x01020202, 0x02000000, 0x02000001,
    0x02000002, 0x02000100, 0x02000101, 0x02000102, 0x02000200, 0x02000201, 0x02000202, 0x02010000,
    0x02010001, 0x02010002, 0x02010100, 0x02010101, 0x02010102, 0x02010200, 0x02010201, 0x02010202,
    0x02020000, 0x02020001, 0x02020002, 0x02020100, 0x02020101, 0x02020102, 0x02020200, 0x02020201,
    0x02020202, 0x40000000, 0x40000001, 0x40000002, 0x40000100, 0x40000101, 0x40000102, 0x40000200,
    0x40000201, 0x40000202, 0x40010000, 0x40010001, 0x40010002, 0x40010100, 0x40010101, 0x40010102,
    0x40010200, 0x40010201, 0x40010202, 0x40020000, 0x40020001, 0x40020002, 0x40020100, 0x40020101,
    0x40020102, 0x40020200, 0x40020201, 0x40020202, 0x41000000, 0x41000001, 0x41000002, 0x41000100,
    0x41000101, 0x41000102, 0x41000200, 0x41000201, 0x41000202, 0x41010000, 0x41010001, 0x41010002,
    0x41010100, 0x41010101, 0x41010102, 0x41010200, 0x41010201, 0x41010202, 0x41020000, 0x41020001,
    0x41020002, 0x41020100, 0x41020101, 0x41020102, 0x41020200, 0x41020201, 0x41020202, 0x42000000,
    0x42000001, 0x42000002, 0x42000100, 0x42000101, 0x42000102, 0x42000200, 0x42000201, 0x42000202,
    0x42010000, 0x42010001, 0x42010002, 0x42010100, 0x42010101, 0x42010102, 0x42010200, 0x42010201,
    0x42010202, 0x42020000, 0x42020001, 0x42020002, 0x42020100, 0x42020101, 0x42020102, 0x42020200,
    0x42020201, 0x42020202, 0x80000000, 0x80000001, 0x80000002, 0x80000100, 0x80000101, 0x80000102,
    0x80000200, 0x80000201, 0x80000202, 0x80010000, 0x80010001, 0x80010002, 0x80010100, 0x80010101,
    0x80010102, 0x80010200, 0x80010201, 0x80010202, 0x80020000, 0x80020001, 0x80020002, 0x80020100,
    0x80020101, 0x80020102, 0x80020200, 0x80020201, 0x80020202, 0x81000000, 0x81000001, 0x81000002,
    0x81000100, 0x81000101, 0x81000102, 0x81000200, 0x81000201, 0x81000202, 0x81010000, 0x81010001,
    0x81010002, 0x81010100, 0x81010101, 0x81010102, 0x81010200, 0x81010201, 0x81010202, 0x81020000,
    0x81020001, 0x81020002, 0x81020100, 0x81020101, 0x81020102, 0x81020200, 0x81020201, 0x81020202,
    0x82000000, 0x82000001, 0x82000002, 0x82000100, 0x82000101, 0x82000102, 0x82000200, 0x82000201,
    0x82000202, 0x82010000, 0x82010001, 0x82010002, 0x82010100, 0x82010101, 0x82010102, 0x82010200,
    0x82010201, 0x82010202, 0x82020000, 0x82020001, 0x82020002, 0x82020100, 0x82020101, 0x82020102,
    0x82020200, 0x82020201, 0x82020202, 0x00000000, 0x00000001, 0x00000002, 0x00000100, 0x00000101,
    0x00000102, 0x00000200, 0x00000201, 0x00000202, 0x00010000, 0x00010001, 0x00010002, 0x00010100,
]


@always_inline
def tq1_trits[W: Int](q: SIMD[DType.uint8, W]) -> SIMD[dtype, W]:
    return ((q.cast[DType.uint16]() * UInt16(3)) >> UInt16(8)).cast[dtype]() - 1


@always_inline
def tq2_trits[W: Int](q: SIMD[DType.uint8, W], shift: Int) -> SIMD[dtype, W]:
    return ((q >> UInt8(shift)) & UInt8(3)).cast[dtype]() - 1


def amar_matmul_skinny_q2b3row[
    MR: Int,
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout,
    PLayout: TensorLayout
](
    A: TileTensor[DType.bfloat16, ALayout, MutAnyOrigin],
    Q: TileTensor[DType.uint8, QLayout, MutAnyOrigin],
    S: TileTensor[DType.float16, SLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2
    comptime assert S.flat_rank == 2 and Cp.flat_rank == 3

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= N:
        return

    var Av = A.vectorize[1, TV]()
    var acc = Array[Scalar[dtype], MR](fill=Scalar[dtype](0))
    var lut = materialize[B3LUT_X]()

    var nb = K // B3_BLOCK
    var b = lane
    while b < nb:
        var qs = Array[UInt8, B3_BYTES](uninitialized=True)
        comptime for i in range(B3_BYTES):
            qs[i] = rebind[Scalar[DType.uint8]](Q[row, b * B3_BYTES + i])
        var d = rebind[Scalar[DType.float16]](S[row, b]).cast[dtype]()
        var blk = Array[SIMD[dtype, TV], MR](fill=SIMD[dtype, TV](0))

        comptime for c in range(4):
            comptime for h in range(2):
                var w = SIMD[dtype, TV](0)
                comptime for i in range(TV):
                    comptime t = h * TV + i
                    comptime byte = 6 * c + t // 5 if t < 30 else 24 + (c >> 1)
                    comptime digit = t % 5 if t < 30 else 2 * (c & 1) + (t - 30)
                    comptime shift = 8 * digit if digit < 4 else 30
                    w[i] = ((lut[Int(qs[byte])] >> UInt32(shift)) & UInt32(3)).cast[dtype]() - 1
                var e = b * B3_BLOCK + c * 32 + h * TV
                comptime for r in range(MR):
                    if r < M:
                        var a = rebind[SIMD[DType.bfloat16, TV]](Av[r, e // TV]).cast[dtype]()
                        blk[r] += w * a

        comptime for r in range(MR):
            if r < M:
                acc[r] += blk[r].reduce_add() * d
        b += WARP_SIZE

    comptime for r in range(MR):
        if r < M:
            var total = warp.sum(acc[r])
            if lane == 0:
                Cp[0, r, row] = rebind[Cp.ElementType](total)


def amar_matmul_skinny_tq1row[
    MR: Int,
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout,
    PLayout: TensorLayout
](
    A: TileTensor[DType.bfloat16, ALayout, MutAnyOrigin],
    Q: TileTensor[DType.uint8, QLayout, MutAnyOrigin],
    S: TileTensor[DType.float16, SLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2
    comptime assert S.flat_rank == 2 and Cp.flat_rank == 3

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= N:
        return

    var Av = A.vectorize[1, TV]()
    var Qv = Q.vectorize[1, 4]()
    var acc = Array[Scalar[dtype], MR](fill=Scalar[dtype](0))

    var nb = K // TQ_BLOCK
    var b = lane
    while b < nb:
        var raw = Array[SIMD[DType.uint8, 4], TQ1_BYTES // 4](uninitialized=True)
        comptime for i in range(TQ1_BYTES // 4):
            raw[i] = rebind[SIMD[DType.uint8, 4]](Qv[row, (b * TQ1_BYTES) // 4 + i])
        var qlo = raw[0].join(raw[1]).join(raw[2].join(raw[3]))
        var qhi = raw[4].join(raw[5]).join(raw[6].join(raw[7]))
        var q2 = raw[8].join(raw[9]).join(raw[10].join(raw[11]))
        var qh = raw[12]
        var d = rebind[Scalar[DType.float16]](S[row, b]).cast[dtype]()
        var blk = Array[SIMD[dtype, TV], MR](fill=SIMD[dtype, TV](0))
        var e0 = b * TQ_BLOCK

        comptime for n_ in range(5):
            comptime p = 3 ** n_
            var wlo = tq1_trits[TV](qlo * UInt8(p))
            var whi = tq1_trits[TV](qhi * UInt8(p))
            var e = e0 + n_ * 32
            comptime for r in range(MR):
                if r < M:
                    var alo = rebind[SIMD[DType.bfloat16, TV]](Av[r, e // TV]).cast[dtype]()
                    var ahi = rebind[SIMD[DType.bfloat16, TV]](Av[r, e // TV + 1]).cast[dtype]()
                    blk[r] += wlo * alo + whi * ahi

        comptime for n_ in range(5):
            comptime p = 3 ** n_
            var w = tq1_trits[TV](q2 * UInt8(p))
            var e = e0 + 160 + n_ * TV
            comptime for r in range(MR):
                if r < M:
                    var a = rebind[SIMD[DType.bfloat16, TV]](Av[r, e // TV]).cast[dtype]()
                    blk[r] += w * a

        var wq = SIMD[dtype, TV](0)
        comptime for n_ in range(4):
            comptime p = 3 ** n_
            var v = tq1_trits[4](qh * UInt8(p))
            comptime for j in range(4):
                wq[n_ * 4 + j] = v[j]
        comptime for r in range(MR):
            if r < M:
                var a = rebind[SIMD[DType.bfloat16, TV]](Av[r, (e0 + 240) // TV]).cast[dtype]()
                blk[r] += wq * a

        comptime for r in range(MR):
            if r < M:
                acc[r] += blk[r].reduce_add() * d
        b += WARP_SIZE

    comptime for r in range(MR):
        if r < M:
            var total = warp.sum(acc[r])
            if lane == 0:
                Cp[0, r, row] = rebind[Cp.ElementType](total)


def amar_matmul_skinny_tq2row[
    MR: Int,
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout,
    PLayout: TensorLayout
](
    A: TileTensor[DType.bfloat16, ALayout, MutAnyOrigin],
    Q: TileTensor[DType.uint8, QLayout, MutAnyOrigin],
    S: TileTensor[DType.float16, SLayout, MutAnyOrigin],
    Cp: TileTensor[dtype, PLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2
    comptime assert S.flat_rank == 2 and Cp.flat_rank == 3

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= N:
        return

    var Av = A.vectorize[1, TV]()
    var Qv = Q.vectorize[1, TV]()
    var acc = Array[Scalar[dtype], MR](fill=Scalar[dtype](0))

    var nb = K // TQ_BLOCK
    var b = lane
    while b < nb:
        var raw = Array[SIMD[DType.uint8, TV], TQ2_BYTES // TV](uninitialized=True)
        comptime for i in range(TQ2_BYTES // TV):
            raw[i] = rebind[SIMD[DType.uint8, TV]](Qv[row, (b * TQ2_BYTES) // TV + i])
        var d = rebind[Scalar[DType.float16]](S[row, b]).cast[dtype]()
        var blk = Array[SIMD[dtype, TV], MR](fill=SIMD[dtype, TV](0))
        var e0 = b * TQ_BLOCK

        comptime for jj in range(2):
            comptime for l in range(4):
                comptime for h in range(2):
                    var w = tq2_trits[TV](raw[jj * 2 + h], 2 * l)
                    var e = e0 + jj * 128 + l * 32 + h * TV
                    comptime for r in range(MR):
                        if r < M:
                            var a = rebind[SIMD[DType.bfloat16, TV]](Av[r, e // TV]).cast[dtype]()
                            blk[r] += w * a

        comptime for r in range(MR):
            if r < M:
                acc[r] += blk[r].reduce_add() * d
        b += WARP_SIZE

    comptime for r in range(MR):
        if r < M:
            var total = warp.sum(acc[r])
            if lane == 0:
                Cp[0, r, row] = rebind[Cp.ElementType](total)
