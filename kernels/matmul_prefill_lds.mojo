from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu import block_idx, thread_idx, WARP_SIZE
from std.math import ceildiv, fma
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from layout.tensor_core import mma

from matmul_prefill import f32x16_to_bf16_trunc

comptime f32 = DType.float32
comptime f16 = DType.float16
comptime bf16 = DType.bfloat16
comptime i8 = DType.int8
comptime u8 = DType.uint8
comptime LDS_WAVES = 8
comptime LDS_THREADS = LDS_WAVES * WARP_SIZE


@always_inline
def lds_chunk(row: Int, c: Int) -> Int:
    return c ^ ((row >> 1) & 3)


def amar_matmul_prefill_lds[
    WDT: DType, WM: Int, WN: Int, TM: Int, TN: Int, ACC: Bool,
    ALayout: TensorLayout, QLayout: TensorLayout, SLayout: TensorLayout, CLayout: TensorLayout
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    Q: TileTensor[WDT, QLayout, MutAnyOrigin],
    S: TileTensor[f16, SLayout, MutAnyOrigin],
    C: TileTensor[f32, CLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert WDT == u8 or WDT == i8
    comptime Q4 = WDT == u8
    comptime assert A.flat_rank == 2 and Q.flat_rank == 2
    comptime assert S.flat_rank == 2 and C.flat_rank == 2
    comptime assert WM * WN == LDS_WAVES
    comptime NT = LDS_THREADS
    comptime BM = WM * TM * 16
    comptime BN = WN * TN * 16
    comptime A_CHUNKS = BM * 4
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

    var sa = stack_allocation[bf16, address_space=AddressSpace.SHARED](row_major[2 * BM, 32]())
    var sb = stack_allocation[bf16, address_space=AddressSpace.SHARED](row_major[2 * BN, 32]())
    var sav = sa.vectorize[1, 8]()
    var sbv = sb.vectorize[1, 8]()
    var Av = A.vectorize[1, 8]()
    comptime QVEC = 16 if Q4 else 32
    var Qv = Q.vectorize[1, QVEC]()

    var acc = Array[SIMD[f32, 8], TM * TN](fill=SIMD[f32, 8](0))
    var sta = Array[SIMD[bf16, 8], A_PER](fill=SIMD[bf16, 8](0))
    var stb = SIMD[WDT, QVEC](0)
    var eight = SIMD[f32, 16](8)
    var nb = K // 32

    var b_ok = tid < BN
    var b_col = min(block_col + tid, N - 1)

    for step in range(nb + 1):
        barrier()
        if step < nb:
            comptime for s in range(A_PER):
                var v = tid + s * NT
                var r = v // 4
                var c = v % 4
                var gr = block_row + r
                if (A_CHUNKS % NT == 0 or v < A_CHUNKS) and gr < M:
                    sta[s] = rebind[SIMD[bf16, 8]](Av[gr, step * 4 + c])
                else:
                    sta[s] = SIMD[bf16, 8](0)
            if b_ok:
                stb = rebind[SIMD[WDT, QVEC]](Qv[b_col, step])

        if step > 0:
            var kb = step - 1
            var cur = kb % 2
            var afr = Array[SIMD[bf16, 16], TM * 2](uninitialized=True)
            comptime for tm in range(TM):
                var ar = wm * TM * 16 + tm * 16 + h
                var arow = cur * BM + ar
                var a0 = rebind[SIMD[bf16, 8]](sav[arow, lds_chunk(ar, 0)])
                var a1 = rebind[SIMD[bf16, 8]](sav[arow, lds_chunk(ar, 1)])
                var a2 = rebind[SIMD[bf16, 8]](sav[arow, lds_chunk(ar, 2)])
                var a3 = rebind[SIMD[bf16, 8]](sav[arow, lds_chunk(ar, 3)])
                afr[2 * tm] = a0.join(a1)
                afr[2 * tm + 1] = a2.join(a3)
            comptime for tn in range(TN):
                var bc = wn * TN * 16 + tn * 16 + h
                var brow = cur * BN + bc
                var c = min(block_col + bc, N - 1)
                var d4 = rebind[Scalar[f16]](S[c, kb]).cast[f32]()
                var b0 = rebind[SIMD[bf16, 8]](sbv[brow, lds_chunk(bc, 0)])
                var b1 = rebind[SIMD[bf16, 8]](sbv[brow, lds_chunk(bc, 1)])
                var b2 = rebind[SIMD[bf16, 8]](sbv[brow, lds_chunk(bc, 2)])
                var b3 = rebind[SIMD[bf16, 8]](sbv[brow, lds_chunk(bc, 3)])
                var blo = b0.join(b1)
                var bhi = b2.join(b3)
                comptime for tm in range(TM):
                    var t = SIMD[f32, 8](0)
                    mma(t, afr[2 * tm], blo, SIMD[f32, 8](0))
                    var t2 = SIMD[f32, 8](0)
                    mma(t2, afr[2 * tm + 1], bhi, t)
                    acc[tm * TN + tn] = fma(t2, SIMD[f32, 8](d4), acc[tm * TN + tn])

        if step < nb:
            var buf = step % 2
            comptime for s in range(A_PER):
                var v = tid + s * NT
                var r = v // 4
                var c = v % 4
                if A_CHUNKS % NT == 0 or v < A_CHUNKS:
                    sav[buf * BM + r, lds_chunk(r, c)] = rebind[sav.ElementType](sta[s])
            if b_ok:
                var brow = buf * BN + tid
                var blo: SIMD[bf16, 16]
                var bhi: SIMD[bf16, 16]
                comptime if Q4:
                    var q = rebind[SIMD[u8, 16]](stb)
                    blo = f32x16_to_bf16_trunc((q & UInt8(0xF)).cast[f32]() - eight)
                    bhi = f32x16_to_bf16_trunc((q >> UInt8(4)).cast[f32]() - eight)
                else:
                    var q = rebind[SIMD[i8, 32]](stb)
                    blo = f32x16_to_bf16_trunc(q.slice[16, offset=0]().cast[f32]())
                    bhi = f32x16_to_bf16_trunc(q.slice[16, offset=16]().cast[f32]())
                sbv[brow, lds_chunk(tid, 0)] = rebind[sbv.ElementType](blo.slice[8, offset=0]())
                sbv[brow, lds_chunk(tid, 1)] = rebind[sbv.ElementType](blo.slice[8, offset=8]())
                sbv[brow, lds_chunk(tid, 2)] = rebind[sbv.ElementType](bhi.slice[8, offset=0]())
                sbv[brow, lds_chunk(tid, 3)] = rebind[sbv.ElementType](bhi.slice[8, offset=8]())

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
