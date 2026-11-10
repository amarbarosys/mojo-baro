from std.math import fma, tanh
from bexp import exp
from max.gpu import block_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.primitives import warp
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from matmul_skinny import ROW_WAVES
from q4x_kernels import KS16, q4k_s16, q6k_s16, UP
from moe import Q4K_BYTES, Q6K_BYTES, q4k_sm_w, f16_at
from std.memory import bitcast
from std.sys import get_defined_int

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime GT_Q4K = 12
comptime GT_Q6K = 14


@always_inline
def kq_bytes[GT: Int]() -> Int:
    comptime assert GT == GT_Q4K or GT == GT_Q6K
    return Q4K_BYTES if GT == GT_Q4K else Q6K_BYTES


@always_inline
def kq_lane16[GT: Int](w: UP, base: Int, s: Int) -> KS16:
    comptime if GT == GT_Q4K:
        return q4k_s16(w, base, s)
    else:
        return q6k_s16(w, base, s)


@always_inline
def kq_lane_part[GT: Int, ADT: DType, XL: TensorLayout](X: TileTensor[ADT, XL, MutAnyOrigin], w: UP, rb: Int, k: Int, lane: Int) -> SIMD[f32, 16]:
    comptime assert X.flat_rank == 2
    var Xv = X.vectorize[1, 16]()
    var s = lane % 16
    var b = lane // 16
    var acc = SIMD[f32, 16](0)
    while b < k // 256:
        var r = kq_lane16[GT](w, rb + b * kq_bytes[GT](), s)
        var v = fma(r[0].cast[f32](), SIMD[f32, 16](r[1]), SIMD[f32, 16](-r[2]))
        var x = rebind[SIMD[ADT, 16]](Xv[0, (b * 256 + r[3]) // 16]).cast[f32]()
        acc = fma(v, x, acc)
        b += 2
    return acc


comptime KQ_D = get_defined_int["BARO_KQ_D", 2]()
comptime KQLD = Tuple[SIMD[DType.uint8, 16], SIMD[DType.uint8, 16], UInt8, Scalar[f32]]


@always_inline
def kq_sb_load[GT: Int](w: UP, base: Int, s: Int) -> KQLD:
    comptime if GT == GT_Q4K:
        var pair = s // 4
        var half = ((s % 4) // 2) * 16
        return (w.unsafe_offset(base).load[width=16](), w.unsafe_offset(base + 16 + pair * 32 + half).load[width=16](), UInt8(0), Scalar[f32](0))
    else:
        var h = s // 8
        var g = (s % 8) // 2
        var t = s % 2
        return (w.unsafe_offset(base + h * 64 + (g % 2) * 32 + 16 * t).load[width=16](), w.unsafe_offset(base + 128 + h * 32 + 16 * t).load[width=16](), w[unsafe_offset=base + 192 + h * 8 + t + 2 * g], f16_at(w, base + 208))


@always_inline
def kq_sb_x[GT: Int](s: Int) -> Int:
    comptime if GT == GT_Q4K:
        return (s // 4) * 64 + (s % 2) * 32 + ((s % 4) // 2) * 16
    else:
        return (s // 8) * 128 + ((s % 8) // 2) * 32 + 16 * (s % 2)


@always_inline
def kq_sb_fma[GT: Int](ld: KQLD, x: SIMD[f32, 16], s: Int, acc: SIMD[f32, 16]) -> SIMD[f32, 16]:
    comptime if GT == GT_Q4K:
        var hw = bitcast[DType.uint32, 4](ld[0])
        var dd = bitcast[DType.float16, 2](hw[0])
        var sm = q4k_sm_w(hw, 2 * (s // 4) + s % 2)
        var q = (ld[1] & 0x0F) if s % 2 == 0 else (ld[1] >> 4)
        var d1 = dd[0].cast[f32]() * Scalar[f32](sm[0])
        var d2 = dd[1].cast[f32]() * Scalar[f32](sm[1])
        return fma(fma(bitcast[DType.int8, 16](q).cast[f32](), SIMD[f32, 16](d1), SIMD[f32, 16](-d2)), x, acc)
    else:
        var g = (s % 8) // 2
        var nb = (ld[0] & 0x0F) if g < 2 else (ld[0] >> 4)
        var q = nb | (((ld[1] >> UInt8(2 * g)) & 3) << 4)
        var sc = bitcast[DType.int8, 1](SIMD[DType.uint8, 1](ld[2]))[0]
        var dsc = ld[3] * Scalar[f32](Int(sc))
        var zero = Scalar[f32](0)
        return fma(fma((bitcast[DType.int8, 16](q) - 32).cast[f32](), SIMD[f32, 16](dsc), SIMD[f32, 16](-zero)), x, acc)


@always_inline
def kq_lane_part_ld[XO: Origin, XS: AddressSpace, //, GT: Int, ADT: DType, D: Int = KQ_D](xp: Pointer[Scalar[ADT], XO, address_space=XS], w: UP, rb: Int, k: Int, lane: Int) -> SIMD[f32, 16]:
    """kq_lane_part with the loads of D superblocks issued before any decode: same arithmetic in the
    same order, same bytes; x through a pointer so the megakernel can read its LDS copy. At 256 VGPR
    (the megakernel) the scheduler keeps the source order, so q4k_s16 / q6k_s16 cost three serialized
    round trips per superblock there (header, quants, activations: 687 + 655 + 550 cycles); the megakernel
    runs 4 waves per SIMD, so the bytes in flight per wave set the bandwidth (wavewait 2026-10-26: one
    weight in flight 314 GB/s, two 550 GB/s)."""
    comptime assert GT == GT_Q4K or GT == GT_Q6K
    comptime assert D >= 1
    var s = lane % 16
    var b = lane // 16
    var nb = k // 256
    var xo = kq_sb_x[GT](s)
    var acc = SIMD[f32, 16](0)
    while b + 2 * (D - 1) < nb:
        var ld = Array[KQLD, D](uninitialized=True)
        var xs = Array[SIMD[ADT, 16], D](uninitialized=True)
        comptime for i in range(D):
            ld[i] = kq_sb_load[GT](w, rb + (b + 2 * i) * kq_bytes[GT](), s)
            xs[i] = xp.unsafe_offset((b + 2 * i) * 256 + xo).load[width=16]()
        comptime for i in range(D):
            acc = kq_sb_fma[GT](ld[i], xs[i].cast[f32](), s, acc)
        b += 2 * D
    while b < nb:
        var l0 = kq_sb_load[GT](w, rb + b * kq_bytes[GT](), s)
        var x0 = xp.unsafe_offset(b * 256 + xo).load[width=16]().cast[f32]()
        acc = kq_sb_fma[GT](l0, x0, s, acc)
        b += 2
    return acc


@always_inline
def kq_lane_part_ld2[XO: Origin, XS: AddressSpace, //, GT: Int, ADT: DType, D: Int = KQ_D](xp: Pointer[Scalar[ADT], XO, address_space=XS], wa: UP, wb: UP, rba: Int, rbb: Int, k: Int, lane: Int) -> Tuple[SIMD[f32, 16], SIMD[f32, 16]]:
    """Two rows (wa at rba, wb at rbb: ffn gate and up at one offset, or two rows of one weight) against one X:
    each row's sums are kq_lane_part_ld's, in the same order; the loads of both rows, D superblocks each, are in
    flight together (megakernel 2026-10-26: one row in flight 385 GB/s, two 660 GB/s)."""
    comptime assert GT == GT_Q4K or GT == GT_Q6K
    comptime assert D >= 1
    var s = lane % 16
    var b = lane // 16
    var nb = k // 256
    var xo = kq_sb_x[GT](s)
    var acc_a = SIMD[f32, 16](0)
    var acc_b = SIMD[f32, 16](0)
    while b + 2 * (D - 1) < nb:
        var la = Array[KQLD, D](uninitialized=True)
        var lb = Array[KQLD, D](uninitialized=True)
        var xs = Array[SIMD[ADT, 16], D](uninitialized=True)
        comptime for i in range(D):
            la[i] = kq_sb_load[GT](wa, rba + (b + 2 * i) * kq_bytes[GT](), s)
            lb[i] = kq_sb_load[GT](wb, rbb + (b + 2 * i) * kq_bytes[GT](), s)
            xs[i] = xp.unsafe_offset((b + 2 * i) * 256 + xo).load[width=16]()
        comptime for i in range(D):
            var x = xs[i].cast[f32]()
            acc_a = kq_sb_fma[GT](la[i], x, s, acc_a)
            acc_b = kq_sb_fma[GT](lb[i], x, s, acc_b)
        b += 2 * D
    while b < nb:
        var l0 = kq_sb_load[GT](wa, rba + b * kq_bytes[GT](), s)
        var l1 = kq_sb_load[GT](wb, rbb + b * kq_bytes[GT](), s)
        var x = xp.unsafe_offset(b * 256 + xo).load[width=16]().cast[f32]()
        acc_a = kq_sb_fma[GT](l0, x, s, acc_a)
        acc_b = kq_sb_fma[GT](l1, x, s, acc_b)
        b += 2
    return (acc_a, acc_b)


@always_inline
def kq_lane_part_s[GT: Int, S: Int, ADT: DType, XL: TensorLayout](X: TileTensor[ADT, XL, MutAnyOrigin], w: UP, rb: Int, k: Int, wave: Int, lane: Int) -> SIMD[f32, 16]:
    """kq_lane_part over every S-th superblock pair, starting at this wave's share (split-K)."""
    comptime assert X.flat_rank == 2
    var Xv = X.vectorize[1, 16]()
    var acc = SIMD[f32, 16](0)
    var b = (wave % S) * 2 + lane // 16
    while b < k // 256:
        var r = kq_lane16[GT](w, rb + b * kq_bytes[GT](), lane % 16)
        var v = fma(r[0].cast[f32](), SIMD[f32, 16](r[1]), SIMD[f32, 16](-r[2]))
        acc = fma(v, rebind[SIMD[ADT, 16]](Xv[0, (b * 256 + r[3]) // 16]).cast[f32](), acc)
        b += 2 * S
    return acc


@always_inline
def kq_tree16(acc: SIMD[f32, 16]) -> Float32:
    var a = acc
    comptime for i in range(4):
        comptime for j in range(8 >> i):
            a[j] = a[j] + a[j + (8 >> i)]
    return a[0]


@always_inline
def kq_wave_sum(v0: Float32) -> Float32:
    var v = v0
    comptime for i in range(5):
        v += warp.shuffle_xor(v, UInt32(1 << i))
    return v


@always_inline
def gemv_epi[EPI: Int, CLayout: TensorLayout, BLayout: TensorLayout, ODT: DType = bf16](
    C: TileTensor[f32, CLayout, MutAnyOrigin], Ob: TileTensor[ODT, BLayout, MutAnyOrigin], row: Int, total: Float32
):
    comptime assert C.flat_rank == 1 and Ob.flat_rank == 1
    comptime if EPI == 0:
        C[row] = rebind[C.ElementType](total)
    elif EPI == 1:
        C[row] = rebind[C.ElementType](rebind[Scalar[f32]](C[row]) + total)
    elif EPI == 2:
        var g = rebind[Scalar[f32]](C[row])
        var gelu = 0.5 * g * (1 + tanh(0.7978845608028654 * g * (1 + 0.044715 * g * g)))
        Ob[row] = rebind[Ob.ElementType]((gelu * total).cast[ODT]())
    else:
        var g = rebind[Scalar[f32]](C[row])
        var silu = g / (1 + exp(-g))
        Ob[row] = rebind[Ob.ElementType]((silu * total).cast[ODT]())


def amar_gemv_kq[
    GTA: Int, GTB: Int, EPI: Int, ALayout: TensorLayout, CLayout: TensorLayout, BLayout: TensorLayout, ADT: DType = bf16
](
    A: TileTensor[ADT, ALayout, MutAnyOrigin],
    W: UP,
    C: TileTensor[f32, CLayout, MutAnyOrigin],
    Ob: TileTensor[ADT, BLayout, MutAnyOrigin],
    n: Int32,
    k_dim: Int32,
    split: Int32,
    offb: Int32,
):
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= Int(n):
        return
    var acc: SIMD[f32, 16]
    if row < Int(split):
        acc = kq_lane_part[GTA, ADT](A, W, row * (K // 256) * kq_bytes[GTA](), K, lane)
    else:
        acc = kq_lane_part[GTB, ADT](A, W, Int(offb) + (row - Int(split)) * (K // 256) * kq_bytes[GTB](), K, lane)
    var total = kq_wave_sum(kq_tree16(acc))
    if lane == 0:
        gemv_epi[EPI, ODT=ADT](C, Ob, row, total)


def amar_gemv_kq_gateup[
    GT: Int, GELU: Bool, ALayout: TensorLayout, BLayout: TensorLayout, ADT: DType = bf16
](A: TileTensor[ADT, ALayout, MutAnyOrigin], WG: UP, WU: UP, Ob: TileTensor[ADT, BLayout, MutAnyOrigin], n: Int32, k_dim: Int32):
    """FFN gate and up in one launch (wavegroup's gateup): Ob = act(gate . a) * (up . a), one wave per row, one read of a."""
    comptime assert Ob.flat_rank == 1
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= Int(n):
        return
    var rb = row * (K // 256) * kq_bytes[GT]()
    var g = kq_wave_sum(kq_tree16(kq_lane_part[GT, ADT](A, WG, rb, K, lane)))
    var u = kq_wave_sum(kq_tree16(kq_lane_part[GT, ADT](A, WU, rb, K, lane)))
    if lane == 0:
        var a: Float32
        comptime if GELU:
            a = 0.5 * g * (1 + tanh(0.7978845608028654 * g * (1 + 0.044715 * g * g)))
        else:
            a = g / (1 + exp(-g))
        Ob[row] = rebind[Ob.ElementType]((a * u).cast[ADT]())



def amar_gemv_kq_splitk[
    GTA: Int, GTB: Int, S: Int, EPI: Int, ALayout: TensorLayout, CLayout: TensorLayout, BLayout: TensorLayout, ADT: DType = bf16
](
    A: TileTensor[ADT, ALayout, MutAnyOrigin], W: UP, C: TileTensor[f32, CLayout, MutAnyOrigin], Ob: TileTensor[ADT, BLayout, MutAnyOrigin],
    n: Int32, k_dim: Int32, split: Int32, offb: Int32,
):
    """amar_gemv_kq with S waves per row (wavegroup's gemv_split): more blocks for narrow outputs such as QKV.
    Block = ROW_WAVES waves = ROW_WAVES/S rows; partials meet in LDS in wave order, so the sum is deterministic."""
    comptime assert ROW_WAVES % S == 0
    var K = Int(k_dim)
    var wave = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * (ROW_WAVES // S) + wave // S
    var acc = SIMD[f32, 16](0)
    if row < Int(split):
        acc = kq_lane_part_s[GTA, S, ADT](A, W, row * (K // 256) * kq_bytes[GTA](), K, wave, lane)
    elif row < Int(n):
        acc = kq_lane_part_s[GTB, S, ADT](A, W, Int(offb) + (row - Int(split)) * (K // 256) * kq_bytes[GTB](), K, wave, lane)
    var part = stack_allocation[f32, address_space=AddressSpace.SHARED](row_major[ROW_WAVES]())
    var total = kq_wave_sum(kq_tree16(acc))
    if lane == 0:
        part[wave] = rebind[part.ElementType](total)
    barrier()
    if wave % S == 0 and lane == 0 and row < Int(n):
        var sum: Float32 = 0
        comptime for i in range(S):
            sum += rebind[Scalar[f32]](part[wave + i])
        gemv_epi[EPI, ODT=ADT](C, Ob, row, sum)


def amar_gemv_kq_rows[
    GT: Int, R: Int, ALayout: TensorLayout, CLayout: TensorLayout, ADT: DType = bf16
](
    A: TileTensor[ADT, ALayout, MutAnyOrigin],
    W: UP,
    C: TileTensor[f32, CLayout, MutAnyOrigin],
    n: Int32,
    k_dim: Int32,
    m: Int32,
):
    comptime assert A.flat_rank == 2
    var K = Int(k_dim)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * ROW_WAVES + Int(thread_idx.x) // WARP_SIZE
    if row >= Int(n):
        return
    var rb = row * (K // 256) * kq_bytes[GT]()
    var Xv = A.vectorize[1, 16]()
    var s = lane % 16
    var b = lane // 16
    var acc = Array[SIMD[f32, 16], R](fill=SIMD[f32, 16](0))
    while b < K // 256:
        var r = kq_lane16[GT](W, rb + b * kq_bytes[GT](), s)
        var v = fma(r[0].cast[f32](), SIMD[f32, 16](r[1]), SIMD[f32, 16](-r[2]))
        comptime for j in range(R):
            if j < Int(m):
                var x = rebind[SIMD[ADT, 16]](Xv[j, (b * 256 + r[3]) // 16]).cast[f32]()
                acc[j] = fma(v, x, acc[j])
        b += 2
    comptime for j in range(R):
        if j < Int(m):
            var total = kq_wave_sum(kq_tree16(acc[j]))
            if lane == 0:
                C[j * Int(n) + row] = rebind[C.ElementType](total)
