"""Stage 1+2+3 (+ multi-row window) gate of bench/megakernel-protocol.md.

A synthetic 4-layer pack (ssm, ssm, ssm, attn; ffn after each; final norm +
head) in one device byte buffer with an engine-style offset table. The launch
path runs the engine's per-layer kernel sequence at m rows; amar_mega_token[MR]
runs the same four layers + head in one persistent launch at G=96. Gates:
residual X, conv windows, ssm states, KV cache, argmax token(s) and (m>1) the
f32 final-norm rows bit-identical; then per-token time of both. Run at m=1
and m=3.
"""
from std.math import ceildiv
from std.memory import bitcast
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, TensorLayout, row_major

from elementwise import amar_rmsnorm_cast, amar_rmsnorm, amar_argmax_pos
from matmul_skinny import amar_matmul_skinny_q8row, amar_matmul_skinny_q4row, amar_skinny_reduce, amar_skinny_reduce_add, amar_skinny_reduce_swiglu_bf16, ROW_WAVES, ROW_THREADS, SM, SPLITK
from ssm import amar_ssm_reduce_gates, amar_ssm_conv, amar_ssm_qk_l2norm, amar_ssm_delta_step, amar_ssm_gated_out_bf16, CONV, NH_V, SSTATE
from attn import amar_head_rmsnorm, amar_attn_decode, amar_gate_mul_cast, amar_qgate_split, amar_rope_yarn, amar_kv_append, HD, NQH, NKVH, KVT, TCAP, KVHSTR, KVPAGE
from mega import amar_mega_token, amar_mega_window, MEGA_G, MEGA_G_WIN, H, FFN, QF, KV, VOCAB

comptime NL = 4
comptime N_SSM_T = 3
comptime N_ATT_T = 1
comptime SLOTS = 4
comptime TM = 64
comptime POS = 9
comptime ATT_SPLIT = 0
comptime ITERS = 100
comptime u8 = DType.uint8
comptime u32 = DType.uint32
comptime i64 = DType.int64
comptime f32 = DType.float32
comptime f16 = DType.float16
comptime bf16 = DType.bfloat16
comptime i8 = DType.int8
comptime i32 = DType.int32

comptime h_layout = row_major[H]()
comptime hd_layout = row_major[HD]()
comptime g32_layout = row_major[NH_V]()
comptime cw_layout = row_major[CONV, 4]()
comptime n128_layout = row_major[SSTATE]()
comptime csall_layout = row_major[SLOTS, N_SSM_T, 3, CONV]()
comptime ssall_layout = row_major[SLOTS, N_SSM_T, NH_V, SSTATE, SSTATE]()
comptime cache_layout = row_major[TCAP]()
comptime ctr_layout = row_major[3]()
comptime toks_layout = row_major[64]()
comptime dtok_layout = row_major[9]()
comptime off_layout = row_major[64]()
comptime q_h_qf = row_major[QF, H]()
comptime s_h_qf = row_major[QF, H // 32]()
comptime q_h_h = row_major[H, H]()
comptime s_h_h = row_major[H, H // 32]()
comptime q_h_kv = row_major[KV, H]()
comptime s_h_kv = row_major[KV, H // 32]()
comptime q_h_32 = row_major[NH_V, H]()
comptime s_h_32 = row_major[NH_V, H // 32]()
comptime q_conv_h = row_major[CONV, H]()
comptime s_conv_h = row_major[CONV, H // 32]()
comptime q_h_ffn = row_major[FFN, H]()
comptime s_h_ffn = row_major[FFN, H // 32]()
comptime q_ffn_h = row_major[H, FFN]()
comptime s_ffn_h = row_major[H, FFN // 32]()
comptime q_h_v = row_major[VOCAB, H]()
comptime s_h_v = row_major[VOCAB, H // 32]()
comptime p_qf = row_major[SPLITK, SM, QF]()
comptime p_kv = row_major[SPLITK, SM, KV]()
comptime p_h = row_major[SPLITK, SM, H]()
comptime p_32 = row_major[SPLITK, SM, NH_V]()
comptime p_ffn = row_major[SPLITK, SM, FFN]()
comptime p_v = row_major[SPLITK, SM, VOCAB]()
comptime pf_sm = row_major[SM, FFN]()

def hash01(i: Int, salt: Int) -> Float32:
    var x = UInt32(i * 2654435761 + salt * 40503 + 12345)
    x ^= x >> 13
    x *= UInt32(0x5bd1e995)
    x ^= x >> 15
    return Float32(Float64(x & 0xFFFFFF) / 16777216.0)


def q8_bytes(n: Int, k: Int) -> Int:
    return n * k + (n * k // 32) * 2


def put_q8(h: HostBuffer[u8], o: Int, n: Int, k: Int, salt: Int):
    for i in range(n * k):
        h[o + i] = UInt8(Int(hash01(i, salt) * 255.0) & 0xFF)
    var so = o + n * k
    for i in range(n * k // 32):
        var v = Scalar[f16](0.002 + 0.004 * hash01(i, salt + 7))
        var b = bitcast[DType.uint16, 1](SIMD[f16, 1](v))[0]
        h[so + 2 * i] = UInt8(b & 0xFF)
        h[so + 2 * i + 1] = UInt8((b >> 8) & 0xFF)


def q4_bytes(n: Int, k: Int) -> Int:
    return n * k // 2 + (n * k // 32) * 2


def put_q4(h: HostBuffer[u8], o: Int, n: Int, k: Int, salt: Int):
    for i in range(n * k // 2):
        h[o + i] = UInt8(Int(hash01(i, salt) * 255.0) & 0xFF)
    var so = o + n * k // 2
    for i in range(n * k // 32):
        var v = Scalar[f16](0.004 + 0.008 * hash01(i, salt + 7))
        var b = bitcast[DType.uint16, 1](SIMD[f16, 1](v))[0]
        h[so + 2 * i] = UInt8(b & 0xFF)
        h[so + 2 * i + 1] = UInt8((b >> 8) & 0xFF)


def put_f32(h: HostBuffer[u8], o: Int, n: Int, salt: Int, lo: Float32, hi: Float32):
    for i in range(n):
        var v = lo + (hi - lo) * hash01(i, salt)
        var b = bitcast[DType.uint32, 1](SIMD[f32, 1](v))[0]
        h[o + 4 * i] = UInt8(b & 0xFF)
        h[o + 4 * i + 1] = UInt8((b >> 8) & 0xFF)
        h[o + 4 * i + 2] = UInt8((b >> 16) & 0xFF)
        h[o + 4 * i + 3] = UInt8((b >> 24) & 0xFF)


def fill_f32(ctx: DeviceContext, n: Int, salt: Int, lo: Float32, hi: Float32) raises -> DeviceBuffer[f32]:
    var h = ctx.enqueue_create_host_buffer[f32](n)
    ctx.synchronize()
    for i in range(n):
        h[i] = lo + (hi - lo) * hash01(i, salt)
    var d = ctx.enqueue_create_buffer[f32](n)
    ctx.enqueue_copy(dst_buf=d, src_buf=h)
    ctx.synchronize()
    return d^


def fill_kv(ctx: DeviceContext, n: Int, salt: Int, lo: Float32, hi: Float32) raises -> DeviceBuffer[KVT]:
    var h = ctx.enqueue_create_host_buffer[KVT](n)
    ctx.synchronize()
    for i in range(n):
        h[i] = (lo + (hi - lo) * hash01(i, salt)).cast[KVT]()
    var d = ctx.enqueue_create_buffer[KVT](n)
    ctx.enqueue_copy(dst_buf=d, src_buf=h)
    ctx.synchronize()
    return d^


def clone[dt: DType = f32](ctx: DeviceContext, src: DeviceBuffer[dt], n: Int) raises -> DeviceBuffer[dt]:
    var d = ctx.enqueue_create_buffer[dt](n)
    ctx.enqueue_copy(dst_buf=d, src_buf=src)
    return d^


def diff[dt: DType = f32](ctx: DeviceContext, name: String, a: DeviceBuffer[dt], b: DeviceBuffer[dt], n: Int) raises -> Int:
    var ha = ctx.enqueue_create_host_buffer[dt](n)
    var hb = ctx.enqueue_create_host_buffer[dt](n)
    ctx.enqueue_copy(dst_buf=ha, src_buf=a)
    ctx.enqueue_copy(dst_buf=hb, src_buf=b)
    ctx.synchronize()
    var bad = 0
    var maxd = Scalar[dt](0)
    for i in range(n):
        if ha[i] != hb[i]:
            bad += 1
            var d = abs(ha[i] - hb[i])
            if d > maxd:
                maxd = d
    print(name, " mismatches:", bad, "/", n, " max|d|=", maxd, " sample:", ha[0], ha[1], ha[n - 1])
    if bad > 0 and name == "ssm states":
        var big = 0
        var shown = 0
        var per_slot: List[Int] = [0, 0, 0, 0]
        var per_j_lo = 0
        var per_i_lo = 0
        for idx in range(n):
            var d = abs(ha[idx] - hb[idx])
            if d > 1.0:
                big += 1
                var j = idx % SSTATE
                var i = (idx // SSTATE) % SSTATE
                var h = (idx // (SSTATE * SSTATE)) % NH_V
                var si = (idx // (SSTATE * SSTATE * NH_V)) % N_SSM_T
                var slot = idx // (SSTATE * SSTATE * NH_V * N_SSM_T)
                per_slot[slot] += 1
                if j < 64:
                    per_j_lo += 1
                if i < 64:
                    per_i_lo += 1
                if shown < 6:
                    print("    big diff at slot", slot, "si", si, "h", h, "i", i, "j", j, " launch", ha[idx], " mega", hb[idx])
                    shown += 1
        print("    big (>1.0) diffs:", big, " per slot:", per_slot[0], per_slot[1], per_slot[2], per_slot[3], " j<64:", per_j_lo, " i<64:", per_i_lo)
    return bad


def tq[LT: TensorLayout](ctx: DeviceContext, w: DeviceBuffer[u8], o: Int, n: Int, lt: LT) -> TileTensor[i8, LT, MutAnyOrigin]:
    var b = DeviceBuffer[i8](ctx, (w.unsafe_ptr() + o).unsafe_bitcast[Scalar[i8]](), n, owning=False)
    return rebind[TileTensor[i8, LT, MutAnyOrigin]](TileTensor(b, lt))


def ts[LT: TensorLayout](ctx: DeviceContext, w: DeviceBuffer[u8], o: Int, n: Int, lt: LT) -> TileTensor[f16, LT, MutAnyOrigin]:
    var b = DeviceBuffer[f16](ctx, (w.unsafe_ptr() + o + n).unsafe_bitcast[Scalar[f16]](), n // 32, owning=False)
    return rebind[TileTensor[f16, LT, MutAnyOrigin]](TileTensor(b, lt))


def tq4[LT: TensorLayout](ctx: DeviceContext, w: DeviceBuffer[u8], o: Int, n: Int, lt: LT) -> TileTensor[u8, LT, MutAnyOrigin]:
    var b = DeviceBuffer[u8](ctx, w.unsafe_ptr() + o, n // 2, owning=False)
    return rebind[TileTensor[u8, LT, MutAnyOrigin]](TileTensor(b, lt))


def ts4[LT: TensorLayout](ctx: DeviceContext, w: DeviceBuffer[u8], o: Int, n: Int, lt: LT) -> TileTensor[f16, LT, MutAnyOrigin]:
    var b = DeviceBuffer[f16](ctx, (w.unsafe_ptr() + o + n // 2).unsafe_bitcast[Scalar[f16]](), n // 32, owning=False)
    return rebind[TileTensor[f16, LT, MutAnyOrigin]](TileTensor(b, lt))


def gl[
    Q4: Bool, MR: Int, N: Int, K: Int, AL: TensorLayout, PL: TensorLayout
](ctx: DeviceContext, mut w: DeviceBuffer[u8], A: TileTensor[bf16, AL, MutAnyOrigin], o: Int, P: TileTensor[f32, PL, MutAnyOrigin], M: Int) raises:
    comptime if Q4:
        comptime ql = row_major[N, K // 2]()
        comptime sl = row_major[N, K // 32]()
        ctx.enqueue_function[amar_matmul_skinny_q4row[2, MR, AL, type_of(ql), type_of(sl), PL]](A, tq4(ctx, w, o, N * K, ql), ts4(ctx, w, o, N * K, sl), P, Int32(M), Int32(N), Int32(K), grid_dim=ceildiv(N, ROW_WAVES), block_dim=ROW_THREADS)
    else:
        comptime ql = row_major[N, K]()
        comptime sl = row_major[N, K // 32]()
        ctx.enqueue_function[amar_matmul_skinny_q8row[4, MR, AL, type_of(ql), type_of(sl), PL]](A, tq(ctx, w, o, N * K, ql), ts(ctx, w, o, N * K, sl), P, Int32(M), Int32(N), Int32(K), grid_dim=ceildiv(N, ROW_WAVES), block_dim=ROW_THREADS)


def tf[LT: TensorLayout](ctx: DeviceContext, w: DeviceBuffer[u8], o: Int, n: Int, lt: LT) -> TileTensor[f32, LT, MutAnyOrigin]:
    var b = DeviceBuffer[f32](ctx, (w.unsafe_ptr() + o).unsafe_bitcast[Scalar[f32]](), n, owning=False)
    return rebind[TileTensor[f32, LT, MutAnyOrigin]](TileTensor(b, lt))



def run_case[MRT: Int, Q4: Bool](ctx: DeviceContext, mut wbuf: DeviceBuffer[u8], mut offd: DeviceBuffer[i64], off: List[Int]) raises -> Int:
    comptime xm_layout = row_major[MRT, H]()
    comptime qfm_layout = row_major[MRT, QF]()
    comptime g32m_layout = row_major[MRT, NH_V]()
    comptime convm_layout = row_major[MRT, CONV]()
    comptime om_layout = row_major[MRT, NH_V, SSTATE]()
    comptime kvm_flat = row_major[MRT, KV]()
    comptime kvm_layout = row_major[MRT * NKVH, HD]()
    comptime qm_layout = row_major[MRT * NQH, HD]()
    comptime xflat_layout = row_major[MRT * H]()
    comptime ffnm_layout = row_major[MRT, FFN]()
    comptime vm_layout = row_major[MRT, VOCAB]()
    comptime XL = type_of(xm_layout)
    comptime rmsc_k = amar_rmsnorm_cast[XL, type_of(h_layout), XL]
    comptime rms_m = amar_rmsnorm[XL, type_of(h_layout), XL]
    comptime r_conv = amar_skinny_reduce[type_of(p_qf), type_of(convm_layout), 1]
    comptime r_qf = amar_skinny_reduce[type_of(p_qf), type_of(qfm_layout), 1]
    comptime r_kv = amar_skinny_reduce[type_of(p_kv), type_of(kvm_flat), 1]
    comptime r_h = amar_skinny_reduce[type_of(p_h), XL, 1]
    comptime r_add = amar_skinny_reduce_add[type_of(p_h), XL, 1]
    comptime r_swiglu = amar_skinny_reduce_swiglu_bf16[type_of(p_ffn), type_of(ffnm_layout), 1]
    comptime r_head = amar_skinny_reduce[type_of(p_v), type_of(vm_layout), 1]
    comptime argmax_k = amar_argmax_pos[type_of(vm_layout), type_of(toks_layout)]
    comptime argmax_d = amar_argmax_pos[type_of(vm_layout), type_of(dtok_layout)]
    comptime rgates_k = amar_ssm_reduce_gates[type_of(p_32), type_of(g32m_layout), type_of(g32_layout)]
    comptime conv_k = amar_ssm_conv[type_of(convm_layout), type_of(csall_layout), type_of(cw_layout), type_of(convm_layout)]
    comptime l2_k = amar_ssm_qk_l2norm[type_of(convm_layout)]
    comptime delta_k = amar_ssm_delta_step[MRT, type_of(ssall_layout), type_of(convm_layout), type_of(g32m_layout), type_of(om_layout)]
    comptime gated_k = amar_ssm_gated_out_bf16[type_of(om_layout), XL, type_of(n128_layout), XL]
    comptime split_k = amar_qgate_split[type_of(qfm_layout), type_of(qm_layout), type_of(xflat_layout)]
    comptime hrms_q = amar_head_rmsnorm[type_of(qm_layout), type_of(hd_layout)]
    comptime hrms_kv = amar_head_rmsnorm[type_of(kvm_layout), type_of(hd_layout)]
    comptime rope_q = amar_rope_yarn[type_of(qm_layout)]
    comptime rope_k = amar_rope_yarn[type_of(kvm_layout)]
    comptime append_k = amar_kv_append[type_of(cache_layout), type_of(kvm_layout), N_ATT_T]
    comptime att_k = amar_attn_decode[type_of(qm_layout), type_of(cache_layout), type_of(qm_layout), N_ATT_T]
    comptime gmul_k = amar_gate_mul_cast[type_of(xflat_layout), type_of(xflat_layout), type_of(xflat_layout)]
    comptime GW = MEGA_G if MRT == 1 else MEGA_G_WIN
    comptime mega_k = amar_mega_token[
        MRT, False, Q4, XL, XL, XL, XL,
        type_of(convm_layout), type_of(g32m_layout), type_of(convm_layout), type_of(om_layout),
        type_of(csall_layout), type_of(ssall_layout),
        type_of(qfm_layout), type_of(kvm_flat), type_of(qm_layout), type_of(xflat_layout),
        type_of(pf_sm), type_of(ffnm_layout), type_of(off_layout), type_of(ctr_layout), type_of(toks_layout), type_of(dtok_layout),
        NL, N_ATT_T,
    ]
    comptime mega_w = amar_mega_window[
        MRT, True, Q4, XL, XL, XL, XL,
        type_of(convm_layout), type_of(g32m_layout), type_of(convm_layout), type_of(om_layout),
        type_of(csall_layout), type_of(ssall_layout),
        type_of(qfm_layout), type_of(kvm_flat), type_of(qm_layout), type_of(xflat_layout),
        type_of(pf_sm), type_of(ffnm_layout), type_of(off_layout), type_of(ctr_layout), type_of(toks_layout), type_of(dtok_layout),
        NL, N_ATT_T,
    ]
    comptime NCS = SLOTS * N_SSM_T * 3 * CONV
    comptime NSS = SLOTS * N_SSM_T * NH_V * SSTATE * SSTATE
    comptime NKC = ceildiv(TM, KVPAGE) * N_ATT_T * NKVH * KVHSTR
    var M = MRT
    print("=== case m =", M, " q4 =", Q4, "===")

    var x0 = fill_f32(ctx, MRT * H, 1, -1.0, 1.0)
    var cs0 = fill_f32(ctx, NCS, 12, -0.3, 0.3)
    var ss0 = fill_f32(ctx, NSS, 13, -0.1, 0.1)
    var kc0 = fill_kv(ctx, NKC, 14, -1.0, 1.0)
    var vc0 = fill_kv(ctx, NKC, 15, -1.0, 1.0)
    var xL = clone(ctx, x0, MRT * H)
    var xM = clone(ctx, x0, MRT * H)
    var csL = clone(ctx, cs0, NCS)
    var csM = clone(ctx, cs0, NCS)
    var ssL = clone(ctx, ss0, NSS)
    var ssM = clone(ctx, ss0, NSS)
    var kcL = clone(ctx, kc0, NKC)
    var kcM = clone(ctx, kc0, NKC)
    var tab_h = ctx.enqueue_create_host_buffer[DType.int32](4096)
    for i in range(4096):
        tab_h[i] = Int32(i)
    var tab_d = ctx.enqueue_create_buffer[DType.int32](4096)
    ctx.enqueue_copy(dst_buf=tab_d, src_buf=tab_h)
    var vcL = clone(ctx, vc0, NKC)
    var vcM = clone(ctx, vc0, NKC)
    var hnL = ctx.enqueue_create_buffer[f32](MRT * H)
    var hnM = ctx.enqueue_create_buffer[f32](MRT * H)
    ctx.enqueue_memset(hnL, 0)
    ctx.enqueue_memset(hnM, 0)

    var curb_d = ctx.enqueue_create_buffer[bf16](MRT * H)
    var resb_d = ctx.enqueue_create_buffer[bf16](MRT * H)
    var qkv_d = ctx.enqueue_create_buffer[f32](MRT * CONV)
    var z_d = ctx.enqueue_create_buffer[f32](MRT * H)
    var araw_d = ctx.enqueue_create_buffer[f32](MRT * NH_V)
    var braw_d = ctx.enqueue_create_buffer[f32](MRT * NH_V)
    var eg_d = ctx.enqueue_create_buffer[f32](MRT * NH_V)
    var beta_d = ctx.enqueue_create_buffer[f32](MRT * NH_V)
    var conv_d = ctx.enqueue_create_buffer[f32](MRT * CONV)
    var so_d = ctx.enqueue_create_buffer[f32](MRT * NH_V * SSTATE)
    var qf_d = ctx.enqueue_create_buffer[f32](MRT * QF)
    var k_d = ctx.enqueue_create_buffer[f32](MRT * KV)
    var v_d = ctx.enqueue_create_buffer[f32](MRT * KV)
    var q_d = ctx.enqueue_create_buffer[f32](MRT * H)
    var gate_d = ctx.enqueue_create_buffer[f32](MRT * H)
    var ao_d = ctx.enqueue_create_buffer[f32](MRT * H)
    var fgb_d = ctx.enqueue_create_buffer[bf16](MRT * FFN)
    var p_qf_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * QF)
    var p_kv_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * KV)
    var p_h_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * H)
    var p_32_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * NH_V)
    var p_32b_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * NH_V)
    var p_ffn_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * FFN)
    var p_ffn2_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * FFN)
    var p_v_d = ctx.enqueue_create_buffer[f32](SPLITK * SM * VOCAB)
    var logits_d = ctx.enqueue_create_buffer[f32](MRT * VOCAB)
    var ctr_d = ctx.enqueue_create_buffer[u32](3)
    var prof_d = ctx.enqueue_create_buffer[i64](16 * NL + 4)
    var dbg_d = ctx.enqueue_create_buffer[f32](2 * NL * H)
    var toksL_d = ctx.enqueue_create_buffer[i32](64)
    var toksM_d = ctx.enqueue_create_buffer[i32](64)
    var dtokL_d = ctx.enqueue_create_buffer[i32](9)
    var dtokM_d = ctx.enqueue_create_buffer[i32](9)
    var hmax_d = ctx.enqueue_create_buffer[f32](MRT * MEGA_G_WIN)
    var hidx_d = ctx.enqueue_create_buffer[i32](MRT * MEGA_G_WIN)
    ctx.enqueue_memset(p_qf_d, 0)
    ctx.enqueue_memset(p_kv_d, 0)
    ctx.enqueue_memset(p_h_d, 0)
    ctx.enqueue_memset(p_32_d, 0)
    ctx.enqueue_memset(p_32b_d, 0)
    ctx.enqueue_memset(p_ffn_d, 0)
    ctx.enqueue_memset(p_ffn2_d, 0)
    ctx.enqueue_memset(ctr_d, 0)
    ctx.enqueue_memset(prof_d, 0)
    ctx.enqueue_memset(toksL_d, 0)
    ctx.enqueue_memset(toksM_d, 0)
    ctx.enqueue_memset(dtokL_d, 0)
    ctx.enqueue_memset(dtokM_d, 0)
    ctx.synchronize()

    var CurB = TileTensor(curb_d, xm_layout)
    var ResB = TileTensor(resb_d, xm_layout)
    var Qkvm = TileTensor(qkv_d, convm_layout)
    var Zm = TileTensor(z_d, xm_layout)
    var Araw = TileTensor(araw_d, g32m_layout)
    var Braw = TileTensor(braw_d, g32m_layout)
    var Eg = TileTensor(eg_d, g32m_layout)
    var Beta = TileTensor(beta_d, g32m_layout)
    var Conv = TileTensor(conv_d, convm_layout)
    var So = TileTensor(so_d, om_layout)
    var Qfm = TileTensor(qf_d, qfm_layout)
    var Kflat = TileTensor(k_d, kvm_flat)
    var Khd = TileTensor(k_d, kvm_layout)
    var Vflat = TileTensor(v_d, kvm_flat)
    var Vhd = TileTensor(v_d, kvm_layout)
    var Q = TileTensor(q_d, qm_layout)
    var Gate = TileTensor(gate_d, xflat_layout)
    var Ao = TileTensor(ao_d, qm_layout)
    var Aoflat = TileTensor(ao_d, xflat_layout)
    var AoBflat = TileTensor(resb_d, xflat_layout)
    var FgB = TileTensor(fgb_d, ffnm_layout)
    var Pq = TileTensor(p_qf_d, p_qf)
    var Pkv = TileTensor(p_kv_d, p_kv)
    var Ph = TileTensor(p_h_d, p_h)
    var Pab = TileTensor(p_32_d, p_32)
    var Pab2 = TileTensor(p_32b_d, p_32)
    var Pg = TileTensor(p_ffn_d, p_ffn)
    var Pu = TileTensor(p_ffn2_d, p_ffn)
    var Pg1 = TileTensor(p_ffn_d, pf_sm)
    var Pu1 = TileTensor(p_ffn2_d, pf_sm)
    var Pv = TileTensor(p_v_d, p_v)
    var Logits = TileTensor(logits_d, vm_layout)
    var Ctr = TileTensor(ctr_d, ctr_layout)
    var Off = TileTensor(offd, off_layout)
    var ToksL = TileTensor(toksL_d, toks_layout)
    var ToksM = TileTensor(toksM_d, toks_layout)
    var DtokL = TileTensor(dtokL_d, dtok_layout)
    var DtokM = TileTensor(dtokM_d, dtok_layout)
    var HnL = TileTensor(hnL, xm_layout)
    var HnM = TileTensor(hnM, xm_layout)
    var XL_ = TileTensor(xL, xm_layout)
    var XM_ = TileTensor(xM, xm_layout)
    var CsL = TileTensor(csL, csall_layout)
    var CsM = TileTensor(csM, csall_layout)
    var SsL = TileTensor(ssL, ssall_layout)
    var SsM = TileTensor(ssM, ssall_layout)
    var KcL = TileTensor(kcL, cache_layout)
    var VcL = TileTensor(vcL, cache_layout)

    def launch_path(ring: Int) raises:
        var w = 1
        var ssm_i = 0
        for layer in range(NL):
            ctx.enqueue_function[rmsc_k](XL_, tf(ctx, wbuf, off[w], H, h_layout), CurB, Int32(H), Float32(1e-6), grid_dim=M, block_dim=256)
            if (layer + 1) % 4 == 0:
                gl[Q4, MRT, QF, H](ctx, wbuf, CurB, off[w + 1], Pq, M)
                ctx.enqueue_function[r_qf](Pq, Qfm, Int32(M), Int32(QF), grid_dim=ceildiv(M * QF, 256), block_dim=256)
                gl[Q4, MRT, KV, H](ctx, wbuf, CurB, off[w + 2], Pkv, M)
                ctx.enqueue_function[r_kv](Pkv, Kflat, Int32(M), Int32(KV), grid_dim=ceildiv(M * KV, 256), block_dim=256)
                gl[Q4, MRT, KV, H](ctx, wbuf, CurB, off[w + 3], Pkv, M)
                ctx.enqueue_function[r_kv](Pkv, Vflat, Int32(M), Int32(KV), grid_dim=ceildiv(M * KV, 256), block_dim=256)
                ctx.enqueue_function[split_k](Qfm, Q, Gate, grid_dim=(NQH, M), block_dim=HD)
                ctx.enqueue_function[hrms_q](Q, tf(ctx, wbuf, off[w + 4], HD, hd_layout), Float32(1e-6), grid_dim=M * NQH, block_dim=HD)
                ctx.enqueue_function[hrms_kv](Khd, tf(ctx, wbuf, off[w + 5], HD, hd_layout), Float32(1e-6), grid_dim=M * NKVH, block_dim=HD)
                ctx.enqueue_function[rope_q](Q, Int32(POS), Int32(NQH), grid_dim=(NQH, M), block_dim=32)
                ctx.enqueue_function[rope_k](Khd, Int32(POS), Int32(NKVH), grid_dim=(NKVH, M), block_dim=32)
                ctx.enqueue_function[append_k](KcL, Khd, tab_d.unsafe_ptr(), Int32(POS), Int32(layer // 4), grid_dim=(NKVH, M), block_dim=HD)
                ctx.enqueue_function[append_k](VcL, Vhd, tab_d.unsafe_ptr(), Int32(POS), Int32(layer // 4), grid_dim=(NKVH, M), block_dim=HD)
                ctx.enqueue_function[att_k](Q, KcL, VcL, Ao, tab_d.unsafe_ptr(), Int32(POS + 1), Float32(0.0625), Int32(layer // 4), grid_dim=(NQH, M), block_dim=HD)
                ctx.enqueue_function[gmul_k](Aoflat, Gate, AoBflat, Int32(M * H), grid_dim=ceildiv(M * H, 256), block_dim=256)
                gl[Q4, MRT, H, H](ctx, wbuf, ResB, off[w + 6], Ph, M)
                ctx.enqueue_function[r_add](Ph, XL_, Int32(M), Int32(H), grid_dim=ceildiv(M * H, 256), block_dim=256)
                w += 7
            else:
                gl[Q4, MRT, CONV, H](ctx, wbuf, CurB, off[w + 1], Pq, M)
                gl[Q4, MRT, H, H](ctx, wbuf, CurB, off[w + 2], Ph, M)
                gl[Q4, MRT, NH_V, H](ctx, wbuf, CurB, off[w + 3], Pab, M)
                gl[Q4, MRT, NH_V, H](ctx, wbuf, CurB, off[w + 4], Pab2, M)
                ctx.enqueue_function[r_conv](Pq, Qkvm, Int32(M), Int32(CONV), grid_dim=ceildiv(M * CONV, 256), block_dim=256)
                ctx.enqueue_function[r_h](Ph, Zm, Int32(M), Int32(H), grid_dim=ceildiv(M * H, 256), block_dim=256)
                ctx.enqueue_function[rgates_k](Pab, Pab2, Eg, Beta, tf(ctx, wbuf, off[w + 6], NH_V, g32_layout), tf(ctx, wbuf, off[w + 7], NH_V, g32_layout), Int32(M), grid_dim=1, block_dim=NH_V)
                ctx.enqueue_function[conv_k](Qkvm, CsL, tf(ctx, wbuf, off[w + 5], CONV * 4, cw_layout), Conv, Int32(ring), Int32(ssm_i), Int32(SLOTS), Int32(M), grid_dim=ceildiv(CONV, 256), block_dim=256)
                ctx.enqueue_function[l2_k](Conv, Int32(M), grid_dim=NH_V, block_dim=SSTATE)
                ctx.enqueue_function[delta_k](SsL, Conv, Eg, Beta, So, Int32(ring), Int32(ssm_i), Int32(SLOTS), grid_dim=NH_V, block_dim=SSTATE)
                ctx.enqueue_function[gated_k](So, Zm, tf(ctx, wbuf, off[w + 8], SSTATE, n128_layout), ResB, Int32(M), grid_dim=NH_V, block_dim=SSTATE)
                gl[Q4, MRT, H, H](ctx, wbuf, ResB, off[w + 9], Ph, M)
                ctx.enqueue_function[r_add](Ph, XL_, Int32(M), Int32(H), grid_dim=ceildiv(M * H, 256), block_dim=256)
                ssm_i += 1
                w += 10
            ctx.enqueue_function[rmsc_k](XL_, tf(ctx, wbuf, off[w], H, h_layout), CurB, Int32(H), Float32(1e-6), grid_dim=M, block_dim=256)
            gl[Q4, MRT, FFN, H](ctx, wbuf, CurB, off[w + 1], Pg, M)
            gl[Q4, MRT, FFN, H](ctx, wbuf, CurB, off[w + 2], Pu, M)
            ctx.enqueue_function[r_swiglu](Pg, Pu, FgB, Int32(M), Int32(FFN), grid_dim=ceildiv(M * FFN, 256), block_dim=256)
            gl[Q4, MRT, H, FFN](ctx, wbuf, FgB, off[w + 3], Ph, M)
            ctx.enqueue_function[r_add](Ph, XL_, Int32(M), Int32(H), grid_dim=ceildiv(M * H, 256), block_dim=256)
            w += 4
        ctx.enqueue_function[rms_m](XL_, tf(ctx, wbuf, off[w], H, h_layout), HnL, Int32(H), Float32(1e-6), grid_dim=M, block_dim=256)
        ctx.enqueue_function[rmsc_k](XL_, tf(ctx, wbuf, off[w], H, h_layout), CurB, Int32(H), Float32(1e-6), grid_dim=M, block_dim=256)
        gl[Q4, MRT, VOCAB, H](ctx, wbuf, CurB, off[w + 1], Pv, M)
        ctx.enqueue_function[r_head](Pv, Logits, Int32(M), Int32(VOCAB), grid_dim=ceildiv(M * VOCAB, 256), block_dim=256)
        if MRT == 1:
            ctx.enqueue_function[argmax_k](Logits, ToksL, Int32(VOCAB), Int32(POS + 1), grid_dim=M, block_dim=256)
        else:
            ctx.enqueue_function[argmax_d](Logits, DtokL, Int32(VOCAB), Int32(0), grid_dim=M, block_dim=256)

    def mega_path(ring: Int) raises:
        comptime if MRT == 1:
            ctx.enqueue_function[mega_k](
                wbuf.unsafe_ptr(), Off, XM_, CurB, ResB, Qkvm, Zm, Araw, Braw, Eg, Beta, Conv, So, CsM, SsM,
                Qfm, Kflat, Vflat, Q, Gate, Ao, kcM.unsafe_ptr(), vcM.unsafe_ptr(), tab_d.unsafe_ptr(), Pg1, Pu1, FgB, Ctr, prof_d.unsafe_ptr(), dbg_d.unsafe_ptr(),
                ToksM, DtokM, HnM, hmax_d.unsafe_ptr(), hidx_d.unsafe_ptr(),
                Int32(ring), Int32(SLOTS), Int32(POS), Int32(M), Int32(0), Int32(1), Int32(ATT_SPLIT), dbg_d.unsafe_ptr(), Int32(0),
                grid_dim=MEGA_G, block_dim=ROW_THREADS,
            )
        else:
            ctx.enqueue_function[mega_w](
                wbuf.unsafe_ptr(), Off, XM_, CurB, ResB, Qkvm, Zm, Araw, Braw, Eg, Beta, Conv, So, CsM, SsM,
                Qfm, Kflat, Vflat, Q, Gate, Ao, kcM.unsafe_ptr(), vcM.unsafe_ptr(), tab_d.unsafe_ptr(), Pg1, Pu1, FgB, Ctr, prof_d.unsafe_ptr(), dbg_d.unsafe_ptr(),
                ToksM, DtokM, HnM, hmax_d.unsafe_ptr(), hidx_d.unsafe_ptr(),
                Int32(ring), Int32(SLOTS), Int32(POS), Int32(M), Int32(0), Int32(2), Int32(ATT_SPLIT), dbg_d.unsafe_ptr(), Int32(0),
                grid_dim=MEGA_G_WIN, block_dim=ROW_THREADS,
            )

    def window_path(ring: Int, grid: Int) raises:
        ctx.enqueue_function[mega_w](
            wbuf.unsafe_ptr(), Off, XM_, CurB, ResB, Qkvm, Zm, Araw, Braw, Eg, Beta, Conv, So, CsM, SsM,
            Qfm, Kflat, Vflat, Q, Gate, Ao, kcM.unsafe_ptr(), vcM.unsafe_ptr(), tab_d.unsafe_ptr(), Pg1, Pu1, FgB, Ctr, prof_d.unsafe_ptr(), dbg_d.unsafe_ptr(),
            ToksM, DtokM, HnM, hmax_d.unsafe_ptr(), hidx_d.unsafe_ptr(),
            Int32(ring), Int32(SLOTS), Int32(POS), Int32(M), Int32(0), Int32(1 if MRT == 1 else 2), Int32(ATT_SPLIT), dbg_d.unsafe_ptr(), Int32(0),
            grid_dim=grid, block_dim=ROW_THREADS,
        )

    launch_path(0)
    ctx.synchronize()
    mega_path(0)
    ctx.synchronize()
    var flag = ctx.enqueue_create_host_buffer[u32](3)
    ctx.enqueue_copy(dst_buf=flag, src_buf=ctr_d)
    ctx.synchronize()
    if flag[2] != 0:
        print("FAIL: megakernel barrier NOT-RESIDENT at G=", MEGA_G)
        return 1
    var bad = diff(ctx, "residual X", xL, xM, MRT * H)
    bad += diff(ctx, "conv windows", csL, csM, NCS)
    bad += diff(ctx, "ssm states", ssL, ssM, NSS)
    bad += diff(ctx, "K cache", kcL, kcM, NKC)
    bad += diff(ctx, "V cache", vcL, vcM, NKC)
    var tl = ctx.enqueue_create_host_buffer[i32](64)
    var tm = ctx.enqueue_create_host_buffer[i32](64)
    var dl = ctx.enqueue_create_host_buffer[i32](9)
    var dm = ctx.enqueue_create_host_buffer[i32](9)
    ctx.enqueue_copy(dst_buf=tl, src_buf=toksL_d)
    ctx.enqueue_copy(dst_buf=tm, src_buf=toksM_d)
    ctx.enqueue_copy(dst_buf=dl, src_buf=dtokL_d)
    ctx.enqueue_copy(dst_buf=dm, src_buf=dtokM_d)
    ctx.synchronize()
    if MRT == 1:
        print("argmax token: launch", tl[POS + 1], " mega", tm[POS + 1])
        if tl[POS + 1] != tm[POS + 1]:
            bad += 1
    else:
        for r in range(M):
            print("argmax row", r, ": launch", dl[r], " mega", dm[r])
            if dl[r] != dm[r]:
                bad += 1
        bad += diff(ctx, "hn rows", hnL, hnM, MRT * H)
    if bad != 0:
        print("FAIL: megakernel token differs from launch path at m =", M)
        return 1
    print("PASS: per-token megakernel (4 layers + head) bit-identical to the launch path at m =", M)
    var wl = 1 + 10 + 4
    ctx.enqueue_function[rmsc_k](XL_, tf(ctx, wbuf, off[wl], H, h_layout), CurB, Int32(H), Float32(1e-6), grid_dim=M, block_dim=256)
    ctx.synchronize()
    var t_50_438 = perf_counter_ns()
    for _ in range(50):
        ctx.enqueue_function[rmsc_k](XL_, tf(ctx, wbuf, off[wl], H, h_layout), CurB, Int32(H), Float32(1e-6), grid_dim=M, block_dim=256)
    ctx.synchronize()
    print("  native rmsc m =", M, ":", Float64(perf_counter_ns() - t_50_438) / 1e3 / 50.0, "us")
    gl[Q4, MRT, CONV, H](ctx, wbuf, CurB, off[wl + 1], Pq, M)
    ctx.synchronize()
    var t_20_180 = perf_counter_ns()
    for _ in range(20):
        gl[Q4, MRT, CONV, H](ctx, wbuf, CurB, off[wl + 1], Pq, M)
    ctx.synchronize()
    print("  native qkv-gemm 8192x4096 m =", M, ":", Float64(perf_counter_ns() - t_20_180) / 1e3 / 20.0, "us")
    ctx.enqueue_function[delta_k](SsL, Conv, Eg, Beta, So, Int32(0), Int32(1), Int32(SLOTS), grid_dim=NH_V, block_dim=SSTATE)
    ctx.synchronize()
    var t_50_166 = perf_counter_ns()
    for _ in range(50):
        ctx.enqueue_function[delta_k](SsL, Conv, Eg, Beta, So, Int32(0), Int32(1), Int32(SLOTS), grid_dim=NH_V, block_dim=SSTATE)
    ctx.synchronize()
    print("  native delta m =", M, ":", Float64(perf_counter_ns() - t_50_166) / 1e3 / 50.0, "us")
    gl[Q4, MRT, FFN, H](ctx, wbuf, CurB, off[wl + 11], Pg, M)
    ctx.synchronize()
    var t_20_222 = perf_counter_ns()
    for _ in range(20):
        gl[Q4, MRT, FFN, H](ctx, wbuf, CurB, off[wl + 11], Pg, M)
    ctx.synchronize()
    print("  native ffn gate 12288x4096 m =", M, ":", Float64(perf_counter_ns() - t_20_222) / 1e3 / 20.0, "us")
    gl[Q4, MRT, H, FFN](ctx, wbuf, FgB, off[wl + 13], Ph, M)
    ctx.synchronize()
    var t_20_437 = perf_counter_ns()
    for _ in range(20):
        gl[Q4, MRT, H, FFN](ctx, wbuf, FgB, off[wl + 13], Ph, M)
    ctx.synchronize()
    print("  native ffn down 4096x12288 m =", M, ":", Float64(perf_counter_ns() - t_20_437) / 1e3 / 20.0, "us")
    gl[Q4, MRT, VOCAB, H](ctx, wbuf, CurB, off[len(off) - 1], Pv, M)
    ctx.synchronize()
    var t_10_148 = perf_counter_ns()
    for _ in range(10):
        gl[Q4, MRT, VOCAB, H](ctx, wbuf, CurB, off[len(off) - 1], Pv, M)
    ctx.synchronize()
    print("  native head 248320x4096 m =", M, ":", Float64(perf_counter_ns() - t_10_148) / 1e3 / 10.0, "us")

    for it in range(10):
        launch_path(it % SLOTS)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for it in range(ITERS):
        launch_path(it % SLOTS)
    ctx.synchronize()
    var us_launch = Float64(perf_counter_ns() - t0) / 1e3 / Float64(ITERS)
    for it in range(10):
        mega_path(it % SLOTS)
    ctx.synchronize()
    t0 = perf_counter_ns()
    for it in range(ITERS):
        mega_path(it % SLOTS)
    ctx.synchronize()
    var us_mega = Float64(perf_counter_ns() - t0) / 1e3 / Float64(ITERS)
    ctx.enqueue_copy(dst_buf=flag, src_buf=ctr_d)
    ctx.synchronize()
    var ph = ctx.enqueue_create_host_buffer[i64](16 * NL + 4)
    ctx.enqueue_copy(dst_buf=ph, src_buf=prof_d)
    ctx.synchronize()
    for layer in range(NL):
        var line = String("  layer ") + String(layer) + (" attn" if (layer + 1) % 4 == 0 else " ssm ") + " phases us:"
        var nph = 5 if (layer + 1) % 4 == 0 else 6
        for k in range(1, nph + 1):
            line += " " + String(Float64(ph[16 * layer + k] - ph[16 * layer + k - 1]) / 100.0)
        line += " | tail " + String(Float64(ph[16 * layer + 7] - ph[16 * layer + nph]) / 100.0)
        line += " | ffn:"
        for k in range(8, 11):
            line += " " + String(Float64(ph[16 * layer + k] - ph[16 * layer + k - 1]) / 100.0)
        line += " tail " + String(Float64(ph[16 * layer + 11] - ph[16 * layer + 10]) / 100.0)
        if layer == NL - 1:
            line += " | head: rmsc " + String(Float64(ph[16 * NL + 1] - ph[16 * NL]) / 100.0) + " gemm+argmax " + String(Float64(ph[16 * NL + 2] - ph[16 * NL + 1]) / 100.0) + " final " + String(Float64(ph[16 * NL + 3] - ph[16 * NL + 2]) / 100.0)
        line += " | sub " + String(Float64(ph[16 * layer + 7] - ph[16 * layer]) / 100.0) + " ffn " + String(Float64(ph[16 * layer + 11] - ph[16 * layer + 7]) / 100.0)
        print(line)
    print("m =", M, " 4-layer+head us/window: launch=", us_launch, " mega(G=", GW, ")=", us_mega, " ratio=", us_mega / us_launch, " fail=", flag[2])
    comptime if MRT == 1:
        for gi in range(2):
            var grid = 96 if gi == 0 else 192
            ctx.enqueue_memset(ctr_d, 0)
            for it in range(10):
                window_path(it % SLOTS, grid)
            ctx.synchronize()
            var t1 = perf_counter_ns()
            for it in range(ITERS):
                window_path(it % SLOTS, grid)
            ctx.synchronize()
            var us_w = Float64(perf_counter_ns() - t1) / 1e3 / Float64(ITERS)
            ctx.enqueue_copy(dst_buf=flag, src_buf=ctr_d)
            ctx.enqueue_copy(dst_buf=ph, src_buf=prof_d)
            ctx.synchronize()
            var l0 = String("    window kernel (192-cap, reload delta) G=") + String(grid) + ": us=" + String(us_w) + " fail=" + String(flag[2])
            l0 += " | layer0 ssm phases:"
            for k in range(1, 7):
                l0 += " " + String(Float64(ph[k] - ph[k - 1]) / 100.0)
            l0 += " tail " + String(Float64(ph[7] - ph[6]) / 100.0) + " | ffn: " + String(Float64(ph[9] - ph[8]) / 100.0) + " down-tail " + String(Float64(ph[11] - ph[10]) / 100.0)
            l0 += " | layer3 attn gemm " + String(Float64(ph[16 * 3 + 2] - ph[16 * 3 + 1]) / 100.0) + " tail " + String(Float64(ph[16 * 3 + 7] - ph[16 * 3 + 5]) / 100.0) + " | head " + String(Float64(ph[16 * NL + 2] - ph[16 * NL + 1]) / 100.0)
            print(l0)
    return 0


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()

    var bad = 0
    comptime for q4i in range(2):
        comptime Q4C = q4i == 1
        var off = List[Int]()
        var kinds = List[Int]()
        var nrows = List[Int]()
        var ncols = List[Int]()
        var cursor = 0
        off.append(0)
        kinds.append(-1)
        nrows.append(0)
        ncols.append(0)

        def add(kind: Int, n: Int, k: Int):
            off.append(cursor)
            kinds.append(kind)
            nrows.append(n)
            ncols.append(k)
            if kind == 1:
                cursor += q4_bytes(n, k) if Q4C else q8_bytes(n, k)
            else:
                cursor += n * 4

        for layer in range(NL):
            add(0, H, 0)
            if (layer + 1) % 4 == 0:
                add(1, QF, H)
                add(1, KV, H)
                add(1, KV, H)
                add(0, HD, 0)
                add(0, HD, 0)
                add(1, H, H)
            else:
                add(1, CONV, H)
                add(1, H, H)
                add(1, NH_V, H)
                add(1, NH_V, H)
                add(0, CONV * 4, 0)
                add(0, NH_V, 0)
                add(0, NH_V, 0)
                add(0, SSTATE, 0)
                add(1, H, H)
            add(0, H, 0)
            add(1, FFN, H)
            add(1, FFN, H)
            add(1, H, FFN)
        add(0, H, 0)
        add(1, VOCAB, H)
        print("synthetic pack q4 =", Q4C, ":", cursor, "bytes,", len(off), "entries")

        var wh = ctx.enqueue_create_host_buffer[u8](cursor)
        var offh = ctx.enqueue_create_host_buffer[i64](64)
        ctx.synchronize()
        for i in range(64):
            offh[i] = 0
        for e in range(len(off)):
            offh[e] = Int64(off[e])
            if kinds[e] == 1:
                comptime if Q4C:
                    put_q4(wh, off[e], nrows[e], ncols[e], 100 + e)
                else:
                    put_q8(wh, off[e], nrows[e], ncols[e], 100 + e)
            elif kinds[e] == 0:
                var n = nrows[e]
                if n == CONV * 4:
                    put_f32(wh, off[e], n, 100 + e, -0.5, 0.5)
                elif n == NH_V and e % 2 == 0:
                    put_f32(wh, off[e], n, 100 + e, -2.0, -0.1)
                elif n == NH_V:
                    put_f32(wh, off[e], n, 100 + e, -1.0, 1.0)
                else:
                    put_f32(wh, off[e], n, 100 + e, 0.5, 1.5)
        var wbuf = ctx.enqueue_create_buffer[u8](cursor)
        var offd = ctx.enqueue_create_buffer[i64](64)
        ctx.enqueue_copy(dst_buf=wbuf, src_buf=wh)
        ctx.enqueue_copy(dst_buf=offd, src_buf=offh)
        ctx.synchronize()
        bad += run_case[1, Q4C](ctx, wbuf, offd, off)
        comptime if not Q4C:
            bad += run_case[3, Q4C](ctx, wbuf, offd, off)
    if bad != 0:
        print("FAIL: megakernel gate")
        return
    print("PASS: megakernel gate at m = 1 and m = 3 (q8) and m = 1 (q4)")
