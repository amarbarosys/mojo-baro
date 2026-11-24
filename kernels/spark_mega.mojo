from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, lane_id, thread_idx, WARP_SIZE
from max.gpu.memory import AddressSpace
from std.math import tanh
from std.sys import get_defined_int
from std.memory import bitcast
from std.utils import StaticTuple
from bexp import exp
from layout import TileTensor, TensorLayout, row_major, stack_allocation
from attn import KVT, attn_head_span
from elementwise import amar_rmsnorm_cast
from gemv_kq import gemv_epi
from mega import grid_barrier
from prefill_rows import head_rms_row
from spark_kernels import gemv_q8_row, rope_plain_body, rope_kv_body, amar_argmax_part, amar_argmax_final
from profile import (
    H, FFN, VOCAB, N_LAYERS, NQH, NKVH, HD, NORM_EPS, NROT_FULL, BASE_FULL, SWA_PERIOD, SWA_FULL_PHASE,
    ROPE_NEOX, QKV_BIAS, HAS_GATE, QK_NORM, ATTN_SCALE, ACTIVATION_GELU, TMAX, ROPE_FREQS,
)

comptime f32 = DType.float32
comptime u8 = DType.uint8
comptime u32 = DType.uint32
comptime i32 = DType.int32
comptime SM_G = 96
comptime SM_T = 256
comptime SM_W = SM_T // 32
comptime SM_TRACE = get_defined_int["BARO_TRACE_SUM", 0]() == 1
comptime SM_MUT = get_defined_int["BARO_SM_MUTANT", 0]() == 1
comptime SM_OK = not QKV_BIAS and not HAS_GATE and SWA_PERIOD == 1 and SWA_FULL_PHASE == 0
comptime SM_O = 2
comptime SM_LS = SM_O + 5 + (2 if QK_NORM else 0)
comptime SM_QD = NQH * HD
comptime SM_KD = NKVH * HD
comptime SM_QKV = SM_QD + 2 * SM_KD
comptime BP = MutPointer[Scalar[u8], MutAnyOrigin]
comptime FP = MutPointer[Scalar[f32], MutAnyOrigin]
comptime IP = MutPointer[Scalar[i32], MutAnyOrigin]
comptime UP32 = MutPointer[Scalar[u32], MutAnyOrigin]
comptime P32 = Pointer[Float32, MutAnyOrigin]


@always_inline
def sm_trace(src: FP, dst: UP32, n: Int, slot: Int):
    var acc = SIMD[u32, 8](2166136261)
    var i = 0
    while i + 8 <= n:
        acc = (acc ^ bitcast[u32, 8](src.load[width=8](i))) * 16777619
        i += 8
    var folded = UInt32(2166136261)
    comptime for j in range(8):
        folded = (folded ^ acc[j]) * 16777619
    dst[slot] = folded


@always_inline
def sm_bar(ctr: UP32) -> Bool:
    return grid_barrier(ctr, ctr.unsafe_offset(1), ctr.unsafe_offset(2))


@always_inline
def sm_gemv[N: Int, K: Int, EPI: Int, ADT: DType, CL: TensorLayout, BL: TensorLayout](
    a: MutPointer[Scalar[ADT], MutAnyOrigin], wbuf: BP, o: Int,
    C: TileTensor[f32, CL, MutAnyOrigin], Ob: TileTensor[ADT, BL, MutAnyOrigin], wave0: Int, lane: Int,
):
    var A = TileTensor(a, row_major[1, K]())
    var Q = TileTensor(wbuf.unsafe_offset(o).unsafe_bitcast[Scalar[DType.int8]](), row_major[N, K]())
    var S = TileTensor(wbuf.unsafe_offset(o + N * K).unsafe_bitcast[Scalar[DType.float16]](), row_major[N, K // 32]())
    var row = wave0
    while row < N:
        var total = gemv_q8_row(A, Q, S, row, lane, K)
        if lane == 0:
            gemv_epi[EPI, ODT=ADT](C, Ob, row, total)
        row += SM_G * SM_W


@always_inline
def sm_gateup[ADT: DType, BL: TensorLayout](
    a: MutPointer[Scalar[ADT], MutAnyOrigin], wbuf: BP, og: Int, ou: Int,
    Ob: TileTensor[ADT, BL, MutAnyOrigin], wave0: Int, lane: Int,
):
    var A = TileTensor(a, row_major[1, H]())
    var Qg = TileTensor(wbuf.unsafe_offset(og).unsafe_bitcast[Scalar[DType.int8]](), row_major[FFN, H]())
    var Sg = TileTensor(wbuf.unsafe_offset(og + FFN * H).unsafe_bitcast[Scalar[DType.float16]](), row_major[FFN, H // 32]())
    var Qu = TileTensor(wbuf.unsafe_offset(ou).unsafe_bitcast[Scalar[DType.int8]](), row_major[FFN, H]())
    var Su = TileTensor(wbuf.unsafe_offset(ou + FFN * H).unsafe_bitcast[Scalar[DType.float16]](), row_major[FFN, H // 32]())
    var row = wave0
    while row < FFN:
        var g = gemv_q8_row(A, Qg, Sg, row, lane, H)
        var total = gemv_q8_row(A, Qu, Su, row, lane, H)
        if lane == 0:
            comptime if ACTIVATION_GELU:
                var gelu = 0.5 * g * (1 + tanh(0.7978845608028654 * g * (1 + 0.044715 * g * g)))
                Ob[row] = rebind[Ob.ElementType]((gelu * total).cast[ADT]())
            else:
                comptime if SM_MUT:
                    var silu = total / (1 + exp(-total))
                    Ob[row] = rebind[Ob.ElementType]((silu * g).cast[ADT]())
                else:
                    var silu = g / (1 + exp(-g))
                    Ob[row] = rebind[Ob.ElementType]((silu * total).cast[ADT]())
        row += SM_G * SM_W


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(SM_T)))
def amar_spark_mega[ADT: DType](
    wbuf: BP,
    off: MutPointer[Scalar[DType.int64], MutAnyOrigin],
    x: FP,
    xb: MutPointer[Scalar[ADT], MutAnyOrigin],
    qkv: FP,
    kc: MutPointer[Scalar[KVT], MutAnyOrigin],
    vc: MutPointer[Scalar[KVT], MutAnyOrigin],
    tab: IP,
    aob: MutPointer[Scalar[ADT], MutAnyOrigin],
    fgb: MutPointer[Scalar[ADT], MutAnyOrigin],
    logits: FP,
    amv: FP,
    ami: IP,
    toks: IP,
    pred: IP,
    force: IP,
    ctr: UP32,
    ff: P32,
    trace: UP32,
    eps: Float32,
    base: Float32,
    ascale: Float32,
    hn: Int32,
    pos0: Int32,
    ntok: Int32,
):
    comptime assert SM_OK
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var lane = Int(lane_id())
    var wave0 = b * SM_W + tid // WARP_SIZE
    var qs = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD]())
    var scores = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD]())
    var red = stack_allocation[f32, address_space = AddressSpace.SHARED](row_major[HD // WARP_SIZE]())
    var X = TileTensor(x, row_major[1, H]())
    var X1 = TileTensor(x, row_major[H]())
    var Xb = TileTensor(xb, row_major[1, H]())
    var Xb1 = TileTensor(xb, row_major[H]())
    var Qkv1 = TileTensor(qkv, row_major[SM_QKV]())
    var Q = TileTensor(qkv, row_major[NQH, HD]())
    var K = TileTensor(qkv.unsafe_offset(SM_QD), row_major[NKVH, HD]())
    var V = TileTensor(qkv.unsafe_offset(SM_QD + SM_KD), row_major[NKVH, HD]())
    var Kc = TileTensor(kc, row_major[1]())
    var Vc = TileTensor(vc, row_major[1]())
    var AoB2 = TileTensor(aob, row_major[NQH, HD]())
    var Fgb1 = TileTensor(fgb, row_major[FFN]())
    var Logits1 = TileTensor(logits, row_major[VOCAB]())
    var Amv = TileTensor(amv, row_major[SM_G]())
    var Ami = TileTensor(ami, row_major[SM_G]())
    var Toks = TileTensor(toks, row_major[TMAX]())
    var Pred = TileTensor(pred, row_major[TMAX]())
    var emb = wbuf.unsafe_offset(Int(off[0])).unsafe_bitcast[Scalar[f32]]()
    var step = 0
    while step < Int(ntok):
        var pos = Int(pos0) + step
        var gi = b * SM_T + tid
        if gi < H:
            x[unsafe_offset=gi] = emb[unsafe_offset=Int(toks[pos]) * H + gi]
        if not sm_bar(ctr):
            return
        for i in range(N_LAYERS):
            var e = 1 + SM_LS * i
            if b == 0:
                amar_rmsnorm_cast(X, TileTensor(wbuf.unsafe_offset(Int(off[e])).unsafe_bitcast[Scalar[f32]](), row_major[H]()), Xb, hn, eps)
            if not sm_bar(ctr):
                return
            sm_gemv[SM_QKV, H, 0](xb, wbuf, Int(off[e + 1]), Qkv1, Xb1, wave0, lane)
            if not sm_bar(ctr):
                return
            comptime if QK_NORM:
                if b == 0 and tid < NQH + NKVH:
                    if tid < NQH:
                        head_rms_row[HD](P32(unsafe_from_address=Int(qkv)), P32(unsafe_from_address=Int(wbuf) + Int(off[e + SM_O + 5])), eps, tid)
                    else:
                        head_rms_row[HD](P32(unsafe_from_address=Int(qkv) + SM_QD * 4), P32(unsafe_from_address=Int(wbuf) + Int(off[e + SM_O + 6])), eps, tid - NQH)
                if not sm_bar(ctr):
                    return
            if b < NQH:
                if tid < NROT_FULL // 2:
                    rope_plain_body[NROT_FULL, NEOX=ROPE_NEOX, FF=ROPE_FREQS](Q, Int32(pos), Int32(NQH), base, ff, b, 0, tid)
            elif b < NQH + 2 * NKVH:
                if tid < HD:
                    rope_kv_body[NROT_FULL, N_LAYERS, HD_=HD, NKVH_=NKVH, NEOX=ROPE_NEOX, FF=ROPE_FREQS](
                        Kc, Vc, K, V, tab, Int32(pos), base, Int32(i), ff, (b - NQH) % NKVH, (b - NQH) // NKVH, tid, 0)
            if not sm_bar(ctr):
                return
            if b < NQH:
                var res = attn_head_span[NAT=N_LAYERS, HD_=HD, NKVH_=NKVH](
                    Q, Kc, Vc, tab, qs, scores, red, b, b // (NQH // NKVH), 0, pos + 1, tid, lane, ascale, i)
                if tid < HD:
                    var inv = 1 / res[1]
                    var o = res[2] * inv
                    AoB2[b, tid] = rebind[AoB2.ElementType](o.cast[ADT]())
            if not sm_bar(ctr):
                return
            sm_gemv[H, SM_QD, 1](aob, wbuf, Int(off[e + SM_O]), X1, Xb1, wave0, lane)
            if not sm_bar(ctr):
                return
            comptime if SM_TRACE:
                if b == 0 and tid == 0:
                    sm_trace(x, trace, H, pos * (3 * N_LAYERS + 1) + 3 * i + 1)
            if b == 0:
                amar_rmsnorm_cast(X, TileTensor(wbuf.unsafe_offset(Int(off[e + SM_O + 1])).unsafe_bitcast[Scalar[f32]](), row_major[H]()), Xb, hn, eps)
            if not sm_bar(ctr):
                return
            sm_gateup(xb, wbuf, Int(off[e + SM_O + 2]), Int(off[e + SM_O + 3]), Fgb1, wave0, lane)
            if not sm_bar(ctr):
                return
            comptime if SM_TRACE:
                if b == 0 and tid == 0:
                    sm_trace(rebind[FP](fgb), trace, FFN, pos * (3 * N_LAYERS + 1) + 3 * i)
            sm_gemv[H, FFN, 1](fgb, wbuf, Int(off[e + SM_O + 4]), X1, Fgb1, wave0, lane)
            if not sm_bar(ctr):
                return
            comptime if SM_TRACE:
                if b == 0 and tid == 0:
                    sm_trace(x, trace, H, pos * (3 * N_LAYERS + 1) + 3 * i + 2)
        if b == 0:
            amar_rmsnorm_cast(X, TileTensor(wbuf.unsafe_offset(Int(off[1 + SM_LS * N_LAYERS])).unsafe_bitcast[Scalar[f32]](), row_major[H]()), Xb, hn, eps)
        if not sm_bar(ctr):
            return
        sm_gemv[VOCAB, H, 0](xb, wbuf, Int(off[2 + SM_LS * N_LAYERS]), Logits1, Xb1, wave0, lane)
        if not sm_bar(ctr):
            return
        amar_argmax_part[SM_G](Logits1, Amv, Ami, Int32(VOCAB))
        if not sm_bar(ctr):
            return
        if b == 0:
            amar_argmax_final[SM_G](Amv, Ami, Toks, Pred, Int32(pos + 1), force[pos + 1])
        if not sm_bar(ctr):
            return
        step += 1
