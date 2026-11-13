from max.gpu import block_idx, thread_idx, lane_id, WARP_SIZE
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.math import ceildiv
from bexp import exp
from std.memory import bitcast
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from arch import mma, acc_row
from matmul_prefill_lds import lds_chunk
from moe import q4k_sm_w, TOPK, Q4K, Q4K_BYTES

comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime f16 = DType.float16
comptime i32 = DType.int32
comptime u8 = DType.uint8
comptime u16 = DType.uint16
comptime MW_WAVES = 8
comptime MW_THREADS = MW_WAVES * WARP_SIZE


@always_inline
def q4k_half16(W: MutPointer[Scalar[u8], MutAnyOrigin], base: Int, j: Int, h: Int) -> SIMD[bf16, 16]:
    var hdr = W.unsafe_offset(base).load[width=16]()
    var d = bitcast[f16, 1](SIMD[u16, 1](UInt16(Int(hdr[0]) | (Int(hdr[1]) << 8))))[0].cast[f32]()
    var dm = bitcast[f16, 1](SIMD[u16, 1](UInt16(Int(hdr[2]) | (Int(hdr[3]) << 8))))[0].cast[f32]()
    var p = q4k_sm_w(bitcast[DType.uint32, 4](hdr), j)
    var qb = W.unsafe_offset(base + 16 + (j // 2) * 32 + h * 16).load[width=16]()
    var q = (qb & 0x0F) if j % 2 == 0 else (qb >> 4)
    return (SIMD[f32, 16](d * Scalar[f32](p[0])) * q.cast[f32]() - SIMD[f32, 16](dm * Scalar[f32](p[1]))).cast[bf16]()


def moe_wmma_q4k_grouped[
    GU: Bool, WM: Int, WN: Int, TM: Int, TN: Int, N: Int, K: Int, ODT: DType,
    XLayout: TensorLayout, GLayout: TensorLayout, PLayout: TensorLayout, OLayout: TensorLayout,
](
    X: TileTensor[bf16, XLayout, MutAnyOrigin],
    W: MutPointer[Scalar[u8], MutAnyOrigin],
    GE: TileTensor[i32, GLayout, MutAnyOrigin],
    GP: TileTensor[i32, PLayout, MutAnyOrigin],
    O: TileTensor[ODT, OLayout, MutAnyOrigin],
    up_offset: Int32,
):
    comptime assert X.flat_rank == 2 and GE.flat_rank == 1 and GP.flat_rank == 1 and O.flat_rank == 2
    comptime assert WM * WN == MW_WAVES and (ODT == bf16) == GU
    comptime BM = WM * TM * 16
    comptime BN = WN * TN * 16
    comptime NB = 2 if GU else 1
    comptime NT = MW_THREADS
    comptime A_CHUNKS = BM * 4
    comptime A_PER = ceildiv(A_CHUNKS, NT)
    comptime B_HALVES = NB * BN * 2
    comptime B_PER = B_HALVES // NT
    comptime assert B_HALVES % NT == 0 and N % BN == 0 and K % Q4K == 0
    comptime NSTEP = K // 32
    comptime ROW_DIV = TOPK if GU else 1
    comptime ROW_BYTES = (K // Q4K) * Q4K_BYTES

    var g = Int(block_idx.y)
    var e = Int(rebind[Scalar[i32]](GE[g]))
    if e < 0:
        return
    var tid = Int(thread_idx.x)
    var wave = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var h = lane % 16
    var half = lane // 16
    var wm = wave // WN
    var wn = wave % WN
    var col_blk = Int(block_idx.x) * BN
    var ebase = e * N * ROW_BYTES

    var sa = stack_allocation[bf16, address_space=AddressSpace.SHARED](row_major[2 * BM, 32]())
    var sb = stack_allocation[bf16, address_space=AddressSpace.SHARED](row_major[2 * NB * BN, 32]())
    var sav = sa.vectorize[1, 8]()
    var sbv = sb.vectorize[1, 8]()
    var Xv = X.vectorize[1, 8]()

    var arow = Array[Int, A_PER](fill=-1)
    comptime for s in range(A_PER):
        var v = tid + s * NT
        if A_CHUNKS % NT == 0 or v < A_CHUNKS:
            var p = Int(rebind[Scalar[i32]](GP[g * BM + v // 4]))
            if p >= 0:
                arow[s] = p // ROW_DIV
    var bbase = Array[Int, B_PER](fill=0)
    comptime for s in range(B_PER):
        var v = tid + s * NT
        var hrow = v // 2
        var mat = hrow // BN
        bbase[s] = ebase + (col_blk + hrow % BN) * ROW_BYTES + mat * Int(up_offset)

    var acc = Array[SIMD[f32, 8], NB * TM * TN](fill=SIMD[f32, 8](0))
    var sta = Array[SIMD[bf16, 8], A_PER](fill=SIMD[bf16, 8](0))
    var stb = Array[SIMD[bf16, 16], B_PER](fill=SIMD[bf16, 16](0))

    for step in range(NSTEP + 1):
        barrier()
        if step < NSTEP:
            comptime for s in range(A_PER):
                var v = tid + s * NT
                if arow[s] >= 0:
                    sta[s] = rebind[SIMD[bf16, 8]](Xv[arow[s], step * 4 + v % 4])
                else:
                    sta[s] = SIMD[bf16, 8](0)
            comptime for s in range(B_PER):
                var v = tid + s * NT
                stb[s] = q4k_half16(W, bbase[s] + (step // 8) * Q4K_BYTES, step % 8, v % 2)

        if step > 0:
            var cur = (step - 1) % 2
            var afr = Array[SIMD[bf16, 16], TM * 2](uninitialized=True)
            comptime for tm in range(TM):
                var ar = wm * TM * 16 + tm * 16 + h
                var r = cur * BM + ar
                var a0 = rebind[SIMD[bf16, 8]](sav[r, lds_chunk(ar, 0)])
                var a1 = rebind[SIMD[bf16, 8]](sav[r, lds_chunk(ar, 1)])
                var a2 = rebind[SIMD[bf16, 8]](sav[r, lds_chunk(ar, 2)])
                var a3 = rebind[SIMD[bf16, 8]](sav[r, lds_chunk(ar, 3)])
                afr[2 * tm] = a0.join(a1)
                afr[2 * tm + 1] = a2.join(a3)
            comptime for mat in range(NB):
                comptime for tn in range(TN):
                    var bc = mat * BN + wn * TN * 16 + tn * 16 + h
                    var r = cur * NB * BN + bc
                    var b0 = rebind[SIMD[bf16, 8]](sbv[r, lds_chunk(bc, 0)])
                    var b1 = rebind[SIMD[bf16, 8]](sbv[r, lds_chunk(bc, 1)])
                    var b2 = rebind[SIMD[bf16, 8]](sbv[r, lds_chunk(bc, 2)])
                    var b3 = rebind[SIMD[bf16, 8]](sbv[r, lds_chunk(bc, 3)])
                    var blo = b0.join(b1)
                    var bhi = b2.join(b3)
                    comptime for tm in range(TM):
                        comptime ai = (mat * TM + tm) * TN + tn
                        var t = SIMD[f32, 8](0)
                        mma(t, afr[2 * tm], blo, acc[ai])
                        mma(acc[ai], afr[2 * tm + 1], bhi, t)

        if step < NSTEP:
            var buf = step % 2
            comptime for s in range(A_PER):
                var v = tid + s * NT
                if A_CHUNKS % NT == 0 or v < A_CHUNKS:
                    var r = v // 4
                    sav[buf * BM + r, lds_chunk(r, v % 4)] = rebind[sav.ElementType](sta[s])
            comptime for s in range(B_PER):
                var v = tid + s * NT
                var hrow = v // 2
                var c = 2 * (v % 2)
                sbv[buf * NB * BN + hrow, lds_chunk(hrow, c)] = rebind[sbv.ElementType](stb[s].slice[8, offset=0]())
                sbv[buf * NB * BN + hrow, lds_chunk(hrow, c + 1)] = rebind[sbv.ElementType](stb[s].slice[8, offset=8]())

    comptime for tm in range(TM):
        comptime for tn in range(TN):
            var c = col_blk + wn * TN * 16 + tn * 16 + h
            comptime for i in range(8):
                var lr = acc_row(wm * TM * 16 + tm * 16, i, half)
                var p = Int(rebind[Scalar[i32]](GP[g * BM + lr]))
                if p >= 0:
                    comptime if GU:
                        var gd = acc[tm * TN + tn][i]
                        var ud = acc[(TM + tm) * TN + tn][i]
                        O[p, c] = rebind[O.ElementType]((gd / (Scalar[f32](1) + exp(-gd)) * ud).cast[bf16]())
                    else:
                        O[p, c] = rebind[O.ElementType](acc[tm * TN + tn][i])
