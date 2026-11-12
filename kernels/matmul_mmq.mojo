from arch import perm
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu import block_idx, global_idx, thread_idx, WARP_SIZE
from std.math import ceildiv, fma
from std.memory import bitcast
from std.sys.intrinsics import llvm_intrinsic
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from arch import mma

comptime f32 = DType.float32
comptime f16 = DType.float16
comptime bf16 = DType.bfloat16
comptime i8 = DType.int8
comptime i32 = DType.int32
comptime u8 = DType.uint8
comptime MMQ_WAVES = 8
comptime MMQ_THREADS = MMQ_WAVES * WARP_SIZE
comptime QT_THREADS = 256
comptime ABL = 0


@always_inline
def mmq_scale_slot(r: Int) -> Int:
    return (r // 16) * 16 + (r % 2) * 8 + (r % 16) // 2


def amar_quant_q8[
    ALayout: TensorLayout, QLayout: TensorLayout, DLayout: TensorLayout, NLayout: TensorLayout
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    Aq: TileTensor[i8, QLayout, MutAnyOrigin],
    Ad: TileTensor[f32, DLayout, MutAnyOrigin],
    An: TileTensor[i32, NLayout, MutAnyOrigin],
    m: Int32,
    k_dim: Int32,
    mpad: Int32,
):
    comptime assert A.flat_rank == 2 and Aq.flat_rank == 2
    comptime assert Ad.flat_rank == 2 and An.flat_rank == 2
    var K = Int(k_dim)
    var nb = K // 32
    var gid = Int(global_idx.x)
    if gid >= Int(mpad) * nb:
        return
    var r = gid // nb
    var kb = gid % nb
    var slot = mmq_scale_slot(r)
    if r >= Int(m):
        Ad[kb, slot] = rebind[Ad.ElementType](Scalar[f32](0))
        An[kb, slot] = rebind[An.ElementType](Scalar[i32](0))
        return
    var Av = A.vectorize[1, 16]()
    var lo = rebind[SIMD[bf16, 16]](Av[r, kb * 2]).cast[f32]()
    var hi = rebind[SIMD[bf16, 16]](Av[r, kb * 2 + 1]).cast[f32]()
    var amax = max(abs(lo).reduce_max(), abs(hi).reduce_max())
    var d = amax / 127
    var inv = Scalar[f32](0) if amax == 0 else 127 / amax
    var qlo = round(lo * inv)
    var qhi = round(hi * inv)
    var s = (qlo.reduce_add() + qhi.reduce_add()).cast[i32]()
    var q8lo = qlo.cast[i8]()
    var q8hi = qhi.cast[i8]()
    comptime for i in range(16):
        Aq[r, kb * 32 + i] = rebind[Aq.ElementType](q8lo[i])
        Aq[r, kb * 32 + 16 + i] = rebind[Aq.ElementType](q8hi[i])
    Ad[kb, slot] = rebind[Ad.ElementType](d)
    An[kb, slot] = rebind[An.ElementType](-8 * s)


@always_inline
def f32x8_to_bf16_trunc(x: SIMD[f32, 8]) -> SIMD[bf16, 8]:
    var u = bitcast[DType.uint32, 8](x)
    var packed = SIMD[DType.uint32, 4](0)
    comptime for i in range(4):
        packed[i] = perm(u[2 * i + 1], u[2 * i], UInt32(0x07060302))
    return bitcast[bf16, 8](packed)


@always_inline
def lds_chunk[KCH: Int](row: Int, c: Int) -> Int:
    comptime SH = 2 if KCH == 2 else 1
    return c ^ ((row >> SH) & (KCH - 1))


def amar_matmul_lds_q4[
    ADT: DType, WM: Int, WN: Int, TM: Int, TN: Int, ACC: Bool,
    ALayout: TensorLayout, DLayout: TensorLayout, NLayout: TensorLayout,
    QLayout: TensorLayout, SLayout: TensorLayout, CLayout: TensorLayout
](
    A: TileTensor[ADT, ALayout, MutAnyOrigin],
    Ad: TileTensor[f32, DLayout, MutAnyOrigin],
    An: TileTensor[i32, NLayout, MutAnyOrigin],
    Q: TileTensor[u8, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    C: TileTensor[f32, CLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert ADT == i8 or ADT == bf16
    comptime INT8 = ADT == i8
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2
    comptime assert Ad.flat_rank == 2 and An.flat_rank == 2
    comptime assert S.flat_rank == 2 and C.flat_rank == 2
    comptime assert WM * WN == MMQ_WAVES
    comptime NT = MMQ_THREADS
    comptime BM = WM * TM * 16
    comptime BN = WN * TN * 16
    comptime AVEC = 16 if INT8 else 8
    comptime KCH = 32 // AVEC
    comptime A_CHUNKS = BM * KCH
    comptime A_PER = ceildiv(A_CHUNKS, NT)
    comptime assert BN <= NT and BM <= NT and BM % 8 == 0

    var M = Int(m)
    var N = Int(n)
    var K = Int(k_dim)
    var tid = Int(thread_idx.x)
    var wave = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var h = lane % 16
    var half = lane // 16
    var wm = wave // WN
    var wn = wave % WN
    var block_row = Int(block_idx.y) * BM
    var block_col = Int(block_idx.x) * BN
    if block_row >= M or block_col >= N:
        return
    var row0 = block_row + wm * TM * 16
    var col0 = block_col + wn * TN * 16

    var sa = stack_allocation[ADT, address_space=AddressSpace.SHARED](row_major[2 * BM, 32]())
    var sb = stack_allocation[ADT, address_space=AddressSpace.SHARED](row_major[2 * BN, 32]())
    comptime SBM = BM if INT8 else 8
    var sd = stack_allocation[f32, address_space=AddressSpace.SHARED](row_major[2, SBM]())
    var sn = stack_allocation[i32, address_space=AddressSpace.SHARED](row_major[2, SBM]())
    var sav = sa.vectorize[1, AVEC]()
    var sbv = sb.vectorize[1, AVEC]()
    var sb8 = sb.vectorize[1, 8]()
    var sd8 = sd.vectorize[1, 8]()
    var sn8 = sn.vectorize[1, 8]()
    var Av = A.vectorize[1, AVEC]()
    var Q8 = Q.vectorize[1, 8]()

    var acc = Array[SIMD[f32, 8], TM * TN](fill=SIMD[f32, 8](0))
    var sta = Array[SIMD[ADT, AVEC], A_PER](fill=SIMD[ADT, AVEC](0))
    var stb = SIMD[u8, 8](0)
    var std8 = Scalar[f32](0)
    var stnu = Scalar[i32](0)
    var eight = SIMD[f32, 8](8)
    var nb = K // 32

    var b_c = tid // 2
    var b_half = tid % 2
    var b_col = min(block_col + b_c, N - 1)
    var s_row = block_row + tid if tid < BM else block_row + tid - BM

    for step in range(nb + 1):
        comptime if ABL != 4:
            barrier()
        if step < nb:
            comptime for s in range(A_PER):
                var v = tid + s * NT
                var r = v // KCH
                var c = v % KCH
                var gr = min(block_row + r, M - 1)
                comptime if ABL == 2:
                    sta[s] = SIMD[ADT, AVEC](Scalar[ADT](step % 3))
                else:
                    if A_CHUNKS % NT == 0 or v < A_CHUNKS:
                        sta[s] = rebind[SIMD[ADT, AVEC]](Av[gr, step * KCH + c])
            comptime if ABL == 1:
                stb = SIMD[u8, 8](UInt8(0x10 | (step & 0xF)))
            else:
                stb = rebind[SIMD[u8, 8]](Q8[b_col, step * 2 + b_half])
            comptime if INT8:
                comptime if ABL == 5:
                    std8 = Scalar[f32](step % 5) * 0.01
                    stnu = Scalar[i32](step % 7)
                else:
                    if tid < BM:
                        std8 = rebind[Scalar[f32]](Ad[step, s_row])
                    elif tid < 2 * BM:
                        stnu = rebind[Scalar[i32]](An[step, s_row])

        if step > 0:
            var kb = step - 1
            var cur = kb % 2
            var afr = Array[SIMD[ADT, 16], TM * 2](uninitialized=True)
            var d8 = Array[SIMD[f32, 8], TM](uninitialized=True)
            var nu = Array[SIMD[i32, 8], TM](uninitialized=True)
            comptime for tm in range(TM):
                var ar = wm * TM * 16 + tm * 16 + h
                var arow = cur * BM + ar
                comptime if INT8:
                    afr[2 * tm] = rebind[SIMD[ADT, 16]](sav[arow, lds_chunk[2](ar, 0)])
                    afr[2 * tm + 1] = rebind[SIMD[ADT, 16]](sav[arow, lds_chunk[2](ar, 1)])
                    var srow = (wm * TM * 16 + tm * 16 + half * 8) // 8
                    d8[tm] = rebind[SIMD[f32, 8]](sd8[cur, srow])
                    nu[tm] = rebind[SIMD[i32, 8]](sn8[cur, srow])
                else:
                    var a0 = rebind[SIMD[ADT, 8]](sav[arow, lds_chunk[4](ar, 0)])
                    var a1 = rebind[SIMD[ADT, 8]](sav[arow, lds_chunk[4](ar, 1)])
                    var a2 = rebind[SIMD[ADT, 8]](sav[arow, lds_chunk[4](ar, 2)])
                    var a3 = rebind[SIMD[ADT, 8]](sav[arow, lds_chunk[4](ar, 3)])
                    afr[2 * tm] = a0.join(a1)
                    afr[2 * tm + 1] = a2.join(a3)
            comptime for tn in range(TN):
                var bc = wn * TN * 16 + tn * 16 + h
                var brow = cur * BN + bc
                var c = min(block_col + bc, N - 1)
                var d4 = rebind[Scalar[f16]](S[c, kb]).cast[f32]()
                var blo: SIMD[ADT, 16]
                var bhi: SIMD[ADT, 16]
                comptime if INT8:
                    blo = rebind[SIMD[ADT, 16]](sbv[brow, lds_chunk[2](bc, 0)])
                    bhi = rebind[SIMD[ADT, 16]](sbv[brow, lds_chunk[2](bc, 1)])
                else:
                    var b0 = rebind[SIMD[ADT, 8]](sbv[brow, lds_chunk[4](bc, 0)])
                    var b1 = rebind[SIMD[ADT, 8]](sbv[brow, lds_chunk[4](bc, 1)])
                    var b2 = rebind[SIMD[ADT, 8]](sbv[brow, lds_chunk[4](bc, 2)])
                    var b3 = rebind[SIMD[ADT, 8]](sbv[brow, lds_chunk[4](bc, 3)])
                    blo = b0.join(b1)
                    bhi = b2.join(b3)
                comptime for tm in range(TM):
                    comptime if INT8:
                        var t = SIMD[i32, 8](0)
                        mma(t, afr[2 * tm], blo, nu[tm])
                        var t2 = SIMD[i32, 8](0)
                        mma(t2, afr[2 * tm + 1], bhi, t)
                        comptime if ABL == 3:
                            acc[tm * TN + tn] = acc[tm * TN + tn] + bitcast[f32, 8](t2)
                        else:
                            acc[tm * TN + tn] = fma(t2.cast[f32](), d8[tm] * d4, acc[tm * TN + tn])
                    else:
                        var t = SIMD[f32, 8](0)
                        mma(t, afr[2 * tm], blo, SIMD[f32, 8](0))
                        var t2 = SIMD[f32, 8](0)
                        mma(t2, afr[2 * tm + 1], bhi, t)
                        comptime if ABL == 3:
                            acc[tm * TN + tn] = acc[tm * TN + tn] + t2
                        else:
                            acc[tm * TN + tn] = fma(t2, SIMD[f32, 8](d4), acc[tm * TN + tn])

        if step < nb:
            var buf = step % 2
            comptime for s in range(A_PER):
                var v = tid + s * NT
                var r = v // KCH
                var c = v % KCH
                if A_CHUNKS % NT == 0 or v < A_CHUNKS:
                    sav[buf * BM + r, lds_chunk[KCH](r, c)] = rebind[sav.ElementType](sta[s])
            var brow = buf * BN + b_c
            comptime if INT8:
                var blo = bitcast[i8, 8](stb & UInt8(0xF))
                var bhi = bitcast[i8, 8](stb >> UInt8(4))
                sb8[brow, lds_chunk[2](b_c, 0) * 2 + b_half] = rebind[sb8.ElementType](blo)
                sb8[brow, lds_chunk[2](b_c, 1) * 2 + b_half] = rebind[sb8.ElementType](bhi)
                if tid < BM:
                    sd[buf, tid] = rebind[sd.ElementType](std8)
                elif tid < 2 * BM:
                    sn[buf, tid - BM] = rebind[sn.ElementType](stnu)
            else:
                var blo = f32x8_to_bf16_trunc((stb & UInt8(0xF)).cast[f32]() - eight)
                var bhi = f32x8_to_bf16_trunc((stb >> UInt8(4)).cast[f32]() - eight)
                sbv[brow, lds_chunk[4](b_c, b_half)] = rebind[sbv.ElementType](blo)
                sbv[brow, lds_chunk[4](b_c, 2 + b_half)] = rebind[sbv.ElementType](bhi)

    comptime for tm in range(TM):
        comptime for tn in range(TN):
            var c = col0 + tn * 16 + h
            if c < N:
                comptime for i in range(8):
                    var r = row0 + tm * 16 + 2 * i + half
                    if r < M:
                        comptime if ACC:
                            C[r, c] = rebind[C.ElementType](
                                rebind[Scalar[f32]](C[r, c]) + acc[tm * TN + tn][i]
                            )
                        else:
                            C[r, c] = rebind[C.ElementType](acc[tm * TN + tn][i])
