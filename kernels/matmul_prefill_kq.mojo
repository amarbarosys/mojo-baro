from max.gpu import MAX_THREADS_PER_BLOCK_METADATA
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu import block_idx, thread_idx, WARP_SIZE
from std.math import ceildiv, fma
from std.utils import StaticTuple
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from arch import mma

from gemv_kq import kq_bytes, GT_Q4K
from moe import q4k_sm_w, f16_at
from std.memory import bitcast
from matmul_prefill_lds import lds_chunk, LDS_WAVES, LDS_THREADS
from q4x_kernels import UP

comptime f32 = DType.float32
comptime f16 = DType.float16
comptime bf16 = DType.bfloat16


comptime u8 = DType.uint8
comptime B16 = SIMD[u8, 16]


@always_inline
def kq_k0[GT: Int](j: Int) -> Int:
    comptime if GT == GT_Q4K:
        return (j // 2) * 64 + (j % 2) * 16
    else:
        return (j // 4) * 128 + (j % 2) * 32 + ((j % 4) // 2) * 16


@always_inline
def kq_q_off[GT: Int](j: Int) -> Int:
    comptime if GT == GT_Q4K:
        return 16 + (j // 2) * 32 + (j % 2) * 16
    else:
        return (j // 4) * 64 + (j % 2) * 32 + ((j % 4) // 2) * 16


@always_inline
def kq_decode[GT: Int, run: Int](q: B16, qh: B16, sc: B16, d: Scalar[f32], dmin: Scalar[f32], j: Int) -> SIMD[f16, 16]:
    var v = (q & 0x0F) if run == 0 else (q >> 4)
    comptime if GT == GT_Q4K:
        var sm = q4k_sm_w(bitcast[DType.uint32, 4](sc), 2 * (j // 2) + run)
        return fma(v.cast[f32](), SIMD[f32, 16](d * Scalar[f32](sm[0])), SIMD[f32, 16](-dmin * Scalar[f32](sm[1]))).cast[f16]()
    else:
        var g = (j % 2) + 2 * run
        var h = j // 4
        var w = bitcast[DType.uint32, 4](sc)[2 * h + run]
        var sb = Int((w >> UInt32(8 * ((j % 4) // 2 + 2 * (j % 2)))) & 0xFF)
        var dsc = d * Scalar[f32](Int(bitcast[DType.int8, 1](SIMD[u8, 1](sb))[0]))
        var q6 = v | (((qh >> UInt8(2 * g)) & 3) << 4)
        return ((bitcast[DType.int8, 16](q6) - 32).cast[f32]() * SIMD[f32, 16](dsc)).cast[f16]()


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(LDS_THREADS)))
def amar_matmul_prefill_kq[
    GT: Int, WM: Int, WN: Int, TM: Int, TN: Int, ACC: Bool, U: Int,
    ALayout: TensorLayout, CLayout: TensorLayout
](
    A: TileTensor[bf16, ALayout, MutAnyOrigin],
    w: UP,
    C: TileTensor[f32, CLayout, MutAnyOrigin],
    m: Int32,
    n: Int32,
    k_dim: Int32,
):
    comptime assert A.flat_rank == 2 and C.flat_rank == 2
    comptime assert WM * WN == LDS_WAVES
    comptime NT = LDS_THREADS
    comptime BM = WM * TM * 16
    comptime BN = WN * TN * 16
    comptime A_CHUNKS = BM * 4
    comptime A_PER = ceildiv(A_CHUNKS, NT)
    comptime assert BN <= NT and BM <= NT and BM % 8 == 0
    comptime SBB = kq_bytes[GT]()
    comptime Q4 = GT == GT_Q4K
    comptime assert U == 1 or U == 2 or U == 4 or U == 8

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

    var sa = stack_allocation[f16, address_space=AddressSpace.SHARED](row_major[2 * BM, 32]())
    var sb = stack_allocation[f16, address_space=AddressSpace.SHARED](row_major[2 * BN, 32]())
    var sav = sa.vectorize[1, 8]()
    var sbv = sb.vectorize[1, 8]()
    var Av = A.vectorize[1, 8]()

    var acc = Array[SIMD[f32, 8], TM * TN](fill=SIMD[f32, 8](0))
    var sta = Array[SIMD[bf16, 8], A_PER](fill=SIMD[bf16, 8](0))
    var rq = B16(0)
    var rqh = B16(0)
    var rsc = B16(0)
    var rd = Scalar[f32](0)
    var rdm = Scalar[f32](0)
    var nsb = K // 256
    var rowb = nsb * SBB

    var b_ok = tid < BN
    var wb = min(block_col + tid, N - 1) * rowb

    var nb = nsb * 8
    for blk in range(nb // U + 1):
        comptime for jj in range(U):
            var step = blk * U + jj
            barrier()
            var j = step % 8
            if step < nb:
                var k0 = (step // 8) * 256 + kq_k0[GT](j)
                var k1 = k0 + (32 if Q4 else 64)
                comptime for s in range(A_PER):
                    var v = tid + s * NT
                    var r = v // 4
                    var c = v % 4
                    var gr = block_row + r
                    if (A_CHUNKS % NT == 0 or v < A_CHUNKS) and gr < M:
                        var kc = (k0 if c < 2 else k1) + (c % 2) * 8
                        sta[s] = rebind[SIMD[bf16, 8]](Av[gr, kc // 8])
                    else:
                        sta[s] = SIMD[bf16, 8](0)
                if b_ok:
                    var base = wb + (step // 8) * SBB
                    if j == 0:
                        comptime if Q4:
                            rsc = w.unsafe_offset(base).load[width=16]()
                            var dd = bitcast[DType.float16, 2](bitcast[DType.uint32, 4](rsc)[0])
                            rd = dd[0].cast[f32]()
                            rdm = dd[1].cast[f32]()
                        else:
                            rsc = w.unsafe_offset(base + 192).load[width=16]()
                            rd = f16_at(w, base + 208)
                    comptime if not Q4:
                        if j % 2 == 0:
                            rqh = w.unsafe_offset(base + 128 + (j // 4) * 32 + ((j % 4) // 2) * 16).load[width=16]()
                    rq = w.unsafe_offset(base + kq_q_off[GT](j)).load[width=16]()

            if step > 0 and step <= nb:
                var cur = (step - 1) % 2
                var afr = Array[SIMD[f16, 16], TM * 2](uninitialized=True)
                comptime for tm in range(TM):
                    var ar = wm * TM * 16 + tm * 16 + h
                    var arow = cur * BM + ar
                    var a0 = rebind[SIMD[f16, 8]](sav[arow, lds_chunk(ar, 0)])
                    var a1 = rebind[SIMD[f16, 8]](sav[arow, lds_chunk(ar, 1)])
                    var a2 = rebind[SIMD[f16, 8]](sav[arow, lds_chunk(ar, 2)])
                    var a3 = rebind[SIMD[f16, 8]](sav[arow, lds_chunk(ar, 3)])
                    afr[2 * tm] = a0.join(a1)
                    afr[2 * tm + 1] = a2.join(a3)
                comptime for tn in range(TN):
                    var bc = wn * TN * 16 + tn * 16 + h
                    var brow = cur * BN + bc
                    var b0 = rebind[SIMD[f16, 8]](sbv[brow, lds_chunk(bc, 0)])
                    var b1 = rebind[SIMD[f16, 8]](sbv[brow, lds_chunk(bc, 1)])
                    var b2 = rebind[SIMD[f16, 8]](sbv[brow, lds_chunk(bc, 2)])
                    var b3 = rebind[SIMD[f16, 8]](sbv[brow, lds_chunk(bc, 3)])
                    var blo = b0.join(b1)
                    var bhi = b2.join(b3)
                    comptime for tm in range(TM):
                        var t = SIMD[f32, 8](0)
                        mma(t, afr[2 * tm], blo, acc[tm * TN + tn])
                        var t2 = SIMD[f32, 8](0)
                        mma(t2, afr[2 * tm + 1], bhi, t)
                        acc[tm * TN + tn] = t2

            if step < nb:
                var buf = step % 2
                comptime for s in range(A_PER):
                    var v = tid + s * NT
                    var r = v // 4
                    var c = v % 4
                    if A_CHUNKS % NT == 0 or v < A_CHUNKS:
                        sav[buf * BM + r, lds_chunk(r, c)] = rebind[sav.ElementType](sta[s].cast[f16]())
                if b_ok:
                    var brow = buf * BN + tid
                    var vlo = kq_decode[GT, 0](rq, rqh, rsc, rd, rdm, j)
                    var vhi = kq_decode[GT, 1](rq, rqh, rsc, rd, rdm, j)
                    sbv[brow, lds_chunk(tid, 0)] = rebind[sbv.ElementType](vlo.slice[8, offset=0]())
                    sbv[brow, lds_chunk(tid, 1)] = rebind[sbv.ElementType](vlo.slice[8, offset=8]())
                    sbv[brow, lds_chunk(tid, 2)] = rebind[sbv.ElementType](vhi.slice[8, offset=0]())
                    sbv[brow, lds_chunk(tid, 3)] = rebind[sbv.ElementType](vhi.slice[8, offset=8]())

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
