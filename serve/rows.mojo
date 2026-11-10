# One decode step for m distinct sequences on a dense pack (BARO_SEQS > 2,
# docs/design/multi-session-serve.md). Row r is the sequence whose slot view is
# views[r], at position pos[r] (its token at pos[r] in, its next token at
# pos[r] + 1 out). Kernels that only see a row's own activations run once over
# the m rows on b's scratch; the ones that read a sequence's position, KV table
# or recurrent state (embed, rope, KV append, attention, conv + gates, delta,
# argmax) run per row with m = 1, exactly as step_window's m = 1 decode does,
# so a row's ids do not depend on which rows share its step.
from std.math import ceildiv
from std.os import getenv
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major
from registry import *
from window import WindowBufs, gemm_w, row_f32, row_bf16, tens_f32, tens_bf16, tok_line, is_attn
from dattn import dattn_nsplit
from sample import SAMP_THREADS
from serve_proto import SampleParams
from gemv_kq import amar_gemv_kq_rows, GT_Q4K, GT_Q6K


def gemm_rows[
    N: Int, K: Int, AL: TensorLayout, PL: TensorLayout
](
    ctx: DeviceContext, A: TileTensor[bf16, AL, MutAnyOrigin], wbuf: DeviceBuffer[DType.uint8], o: Int, kind: Int,
    P: TileTensor[f32, PL, MutAnyOrigin], m: Int, rows_gemv: Bool,
) raises:
    # K-quant weights: one launch reads each weight row once for all m activation rows (amar_gemv_kq_rows,
    # byte-equal per row to the m = 1 kernel, kernels/test_gemv_kq_rows.mojo); P row r at r * N, as gemm_kq.
    if not rows_gemv or (kind != GT_Q4K and kind != GT_Q6K):
        gemm_w[N, K](ctx, A, wbuf, o, kind, P, m)
        return
    comptime a_l = row_major[MROWS, K]()
    comptime c_l = row_major[MROWS * N]()
    var a = TileTensor(DeviceBuffer[bf16](ctx, UnsafePointer[Scalar[bf16], MutAnyOrigin](unsafe_from_address=Int(A.ptr)), MROWS * K, owning=False), a_l)
    var c = TileTensor(DeviceBuffer[f32](ctx, UnsafePointer[Scalar[f32], MutAnyOrigin](unsafe_from_address=Int(P.ptr)), MROWS * N, owning=False), c_l)
    var w = wbuf.unsafe_ptr().unsafe_offset(o)
    if kind == GT_Q4K:
        ctx.enqueue_function[amar_gemv_kq_rows[GT_Q4K, MROWS, type_of(a_l), type_of(c_l)]](
            a, w, c, Int32(N), Int32(K), Int32(m), grid_dim=ceildiv(N, ROW_WAVES), block_dim=ROW_THREADS)
    else:
        ctx.enqueue_function[amar_gemv_kq_rows[GT_Q6K, MROWS, type_of(a_l), type_of(c_l)]](
            a, w, c, Int32(N), Int32(K), Int32(m), grid_dim=ceildiv(N, ROW_WAVES), block_dim=ROW_THREADS)


def step_rows(
    ctx: DeviceContext, mut b: WindowBufs, views: List[WindowBufs], pos: List[Int], ring: List[Int],
    ids: List[Int], att_split: Int, samples: List[SampleParams],
) raises:
    var m = len(views)
    var rows_gemv = getenv("BARO_ROWS_GEMV", "1") == "1"
    if m < 1 or m > MROWS:
        raise Error("step_rows: m = " + String(m) + " outside 1.." + String(MROWS))
    var Xm = row_f32(ctx, b.x_d, 0, MROWS * H, xm_layout)
    var CurBm = TileTensor(b.curb_d, xm_layout)
    var Embd = tens_bf16(ctx, b.wbuf, b.off[0], VOCAB * H, emb_layout)
    for r in range(m):
        ctx.enqueue_function[embed_k](
            Embd, row_f32(ctx, b.x_d, r * H, H, xm_layout), TileTensor(views[r].toks_d, toks_layout), Int32(pos[r]), Int32(H),
            grid_dim=(ceildiv(H, 256), 1), block_dim=256,
        )
    var w = 1
    var ssm_i = 0
    var att_i = 0
    for layer in range(N_LAYERS):
        ctx.enqueue_function[rmsc_k](Xm, tens_f32(ctx, b.wbuf, b.off[w], H, h_layout), CurBm, Int32(H), Float32(1e-6), grid_dim=m, block_dim=256)
        if is_attn(layer):
            var Qn = tens_f32(ctx, b.wbuf, b.off[w + 4], HD, hd_layout)
            var Kn = tens_f32(ctx, b.wbuf, b.off[w + 5], HD, hd_layout)
            var Pqf = TileTensor(b.p_qf_d, p_qf)
            var Pkv = TileTensor(b.p_kv_d, p_kv)
            var Ph = TileTensor(b.p_h_d, p_h)
            var Qfm = TileTensor(b.qf_d, qfm_layout)
            var Q = TileTensor(b.q_d, qm_layout)
            var Gate = TileTensor(b.gate_d, attflat_layout)
            var Kc = TileTensor(DeviceBuffer[KVT](ctx, b.kc_d.unsafe_ptr(), b.kvpool, owning=False), cache_layout)
            var Vc = TileTensor(DeviceBuffer[KVT](ctx, b.vc_d.unsafe_ptr(), b.kvpool, owning=False), cache_layout)
            gemm_rows[QF, H](ctx, CurBm, b.wbuf, b.off[w + 1], b.kind[w + 1], Pqf, m, rows_gemv)
            ctx.enqueue_function[r_qf](Pqf, Qfm, Int32(m), Int32(QF), grid_dim=ceildiv(m * QF, 256), block_dim=256)
            gemm_rows[KV, H](ctx, CurBm, b.wbuf, b.off[w + 2], b.kind[w + 2], Pkv, m, rows_gemv)
            ctx.enqueue_function[r_kv](Pkv, TileTensor(b.k_d, kvm_flat), Int32(m), Int32(KV), grid_dim=ceildiv(m * KV, 256), block_dim=256)
            gemm_rows[KV, H](ctx, CurBm, b.wbuf, b.off[w + 3], b.kind[w + 3], Pkv, m, rows_gemv)
            ctx.enqueue_function[r_kv](Pkv, TileTensor(b.v_d, kvm_flat), Int32(m), Int32(KV), grid_dim=ceildiv(m * KV, 256), block_dim=256)
            ctx.enqueue_function[split_k](Qfm, Q, Gate, grid_dim=(NQH, m), block_dim=HD)
            ctx.enqueue_function[hrms_q](Q, Qn, Float32(1e-6), grid_dim=m * NQH, block_dim=HD)
            ctx.enqueue_function[hrms_kv](TileTensor(b.k_d, kvm_layout), Kn, Float32(1e-6), grid_dim=m * NKVH, block_dim=HD)
            for r in range(m):
                var Qr = row_f32(ctx, b.q_d, r * ATT, ATT, qm_layout)
                var Kr = row_f32(ctx, b.k_d, r * KV, KV, kvm_layout)
                var Vr = row_f32(ctx, b.v_d, r * KV, KV, kvm_layout)
                var Aor = row_f32(ctx, b.ao_d, r * ATT, ATT, qm_layout)
                var tab = views[r].kvtab_d.unsafe_ptr()
                ctx.enqueue_function[rope_q](Qr, Int32(pos[r]), Int32(NQH), grid_dim=(NQH, 1), block_dim=32)
                ctx.enqueue_function[rope_k](Kr, Int32(pos[r]), Int32(NKVH), grid_dim=(NKVH, 1), block_dim=32)
                ctx.enqueue_function[append_k](Kc, Kr, tab, Int32(pos[r]), Int32(att_i), grid_dim=(NKVH, 1), block_dim=HD)
                ctx.enqueue_function[append_k](Vc, Vr, tab, Int32(pos[r]), Int32(att_i), grid_dim=(NKVH, 1), block_dim=HD)
                if pos[r] + 1 > att_split:
                    var dns = dattn_nsplit[HD, DATT_NLD, NKVH](pos[r] + 1, 1, MEGA_G)
                    var Pa = TileTensor(b.p_ffn_d, p_att_layout)
                    ctx.enqueue_function[datt_k](Qr, Kc, Vc, Aor, Pa, tab, Int32(pos[r] + 1), Int32(dns), Float32(0.0625), Int32(att_i), grid_dim=(NKVH, dns, 1), block_dim=ROW_THREADS)
                    if dns > 1:
                        ctx.enqueue_function[dcomb_k](Pa, Aor, Int32(dns), grid_dim=NQH, block_dim=ROW_THREADS)
                else:
                    ctx.enqueue_function[att_k](Qr, Kc, Vc, Aor, tab, Int32(pos[r] + 1), Float32(0.0625), Int32(att_i), grid_dim=(NQH, 1), block_dim=HD)
            ctx.enqueue_function[gmul_k](TileTensor(b.ao_d, attflat_layout), Gate, TileTensor(b.resb_d, attflat_layout), Int32(m * ATT), grid_dim=ceildiv(m * ATT, 256), block_dim=256)
            gemm_rows[H, ATT](ctx, TileTensor(b.resb_d, attm_layout), b.wbuf, b.off[w + 6], b.kind[w + 6], Ph, m, rows_gemv)
            ctx.enqueue_function[r_add](Ph, Xm, Int32(m), Int32(H), grid_dim=ceildiv(m * H, 256), block_dim=256)
            att_i += 1
            w += 7
        else:
            var Cw = tens_f32(ctx, b.wbuf, b.off[w + 5], CONV * 4, cw_layout)
            var SsmA = tens_f32(ctx, b.wbuf, b.off[w + 6], NH_V, g32_layout)
            var DtB = tens_f32(ctx, b.wbuf, b.off[w + 7], NH_V, g32_layout)
            var Nw = tens_f32(ctx, b.wbuf, b.off[w + 8], SSTATE, n128_layout)
            var Ph = TileTensor(b.p_h_d, p_h)
            var ZmOld = TileTensor(b.z_d, xmi_layout)
            var ResBmOld = TileTensor(b.resb_d, xmi_layout)
            gemm_rows[CONV, H](ctx, CurBm, b.wbuf, b.off[w + 1], b.kind[w + 1], TileTensor(b.p_qf_d, p_qf), m, rows_gemv)
            gemm_rows[INNER, H](ctx, CurBm, b.wbuf, b.off[w + 2], b.kind[w + 2], TileTensor(b.p_h_d, p_hi), m, rows_gemv)
            gemm_rows[NH_V, H](ctx, CurBm, b.wbuf, b.off[w + 3], b.kind[w + 3], TileTensor(b.p_32_d, p_32), m, rows_gemv)
            gemm_rows[NH_V, H](ctx, CurBm, b.wbuf, b.off[w + 4], b.kind[w + 4], TileTensor(b.p_32b_d, p_32), m, rows_gemv)
            ctx.enqueue_function[r_qf](TileTensor(b.p_qf_d, p_qf), TileTensor(b.qkv_d, qfm_layout), Int32(m), Int32(CONV), grid_dim=ceildiv(m * CONV, 256), block_dim=256)
            ctx.enqueue_function[r_hi](TileTensor(b.p_h_d, p_hi), ZmOld, Int32(m), Int32(INNER), grid_dim=ceildiv(m * INNER, 256), block_dim=256)
            for r in range(m):
                var Pab = row_f32(ctx, b.p_32_d, r * NH_V, SPLITK * SM * NH_V - r * NH_V, p_32)
                var Pab2 = row_f32(ctx, b.p_32b_d, r * NH_V, SPLITK * SM * NH_V - r * NH_V, p_32)
                var Qkvr = row_f32(ctx, b.qkv_d, r * QF, QF, qfm_layout)
                var Convr = row_f32(ctx, b.conv_d, r * CONV, CONV, convm_layout)
                var Egr = row_f32(ctx, b.eg_d, r * NH_V, NH_V, g32m_layout)
                var Betar = row_f32(ctx, b.beta_d, r * NH_V, NH_V, g32m_layout)
                var Sor = row_f32(ctx, b.so_d, r * INNER, INNER, om_layout)
                ctx.enqueue_function[convl2g_k](Qkvr, row_f32(ctx, views[r].convstate_d, 0, SLOTS * CONV_SLOT, csall_layout), Cw, Convr, Pab, Pab2, Egr, Betar, SsmA, DtB, Int32(ring[r]), Int32(ssm_i), Int32(SLOTS), Int32(1), grid_dim=CONV_GATE_BLOCKS, block_dim=256)
                delta_dispatch(ctx, row_f32(ctx, views[r].sstate_d, 0, SLOTS * SSM_SLOT, ssall_layout), Convr, Egr, Betar, Sor, Int32(ring[r]), Int32(ssm_i), Int32(SLOTS), 1)
            ctx.enqueue_function[gated_k](TileTensor(b.so_d, om_layout), ZmOld, Nw, ResBmOld, Int32(m), grid_dim=NH_V, block_dim=SSTATE)
            gemm_rows[H, INNER](ctx, ResBmOld, b.wbuf, b.off[w + 9], b.kind[w + 9], Ph, m, rows_gemv)
            ctx.enqueue_function[r_add](Ph, Xm, Int32(m), Int32(H), grid_dim=ceildiv(m * H, 256), block_dim=256)
            ssm_i += 1
            w += 10
        ctx.enqueue_function[rmsc_k](Xm, tens_f32(ctx, b.wbuf, b.off[w], H, h_layout), CurBm, Int32(H), Float32(1e-6), grid_dim=m, block_dim=256)
        var Pg = TileTensor(b.p_ffn_d, p_ffn)
        var Pu = TileTensor(b.p_ffn2_d, p_ffn)
        var Ph2 = TileTensor(b.p_h_d, p_h)
        var FgBm = TileTensor(b.fgb_d, ffnm_layout)
        gemm_rows[FFN, H](ctx, CurBm, b.wbuf, b.off[w + 1], b.kind[w + 1], Pg, m, rows_gemv)
        gemm_rows[FFN, H](ctx, CurBm, b.wbuf, b.off[w + 2], b.kind[w + 2], Pu, m, rows_gemv)
        ctx.enqueue_function[r_swiglu](Pg, Pu, FgBm, Int32(m), Int32(FFN), grid_dim=ceildiv(m * FFN, 256), block_dim=256)
        gemm_rows[H, FFN](ctx, FgBm, b.wbuf, b.off[w + 3], b.kind[w + 3], Ph2, m, rows_gemv)
        ctx.enqueue_function[r_add](Ph2, Xm, Int32(m), Int32(H), grid_dim=ceildiv(m * H, 256), block_dim=256)
        w += 4
    ctx.enqueue_function[rmsc_k](Xm, tens_f32(ctx, b.wbuf, b.off[w], H, h_layout), CurBm, Int32(H), Float32(1e-6), grid_dim=m, block_dim=256)
    var Pv = TileTensor(b.p_v_d, p_v)
    gemm_rows[VOCAB, H](ctx, CurBm, b.wbuf, b.off[w + 1], b.kind[w + 1], Pv, m, rows_gemv)
    ctx.enqueue_function[r_head](Pv, TileTensor(b.logits_d, vm_layout), Int32(m), Int32(VOCAB), grid_dim=ceildiv(m * VOCAB, 256), block_dim=256)
    var SampTok = TileTensor(b.dtok_d, dtok_layout)
    var SampProb = TileTensor(b.hmax_d, dtok_layout)
    for r in range(m):
        var Lr = row_f32(ctx, b.logits_d, r * VOCAB, VOCAB, vm_layout)
        var Tr = TileTensor(views[r].toks_d, toks_layout)
        ref sp = samples[r]
        if sp.temperature > 0:
            ctx.enqueue_function[sample_row_k](
                Lr, SampTok, SampProb, Int32(VOCAB), Float32(sp.temperature), Int32(sp.top_k), Float32(sp.top_p), Float32(sp.min_p),
                sp.seed, UInt64(pos[r]), grid_dim=1, block_dim=SAMP_THREADS)
            ctx.enqueue_function[tokcp_k](SampTok, Tr, Int32(0), Int32(pos[r] + 1), Int32(1), grid_dim=1, block_dim=32)
        else:
            ctx.enqueue_function[argmax_k](Lr, Tr, Int32(VOCAB), Int32(pos[r] + 1), grid_dim=1, block_dim=256)
        ctx.enqueue_copy(dst_buf=b.stream_h.create_sub_buffer[DType.int32](r, 1), src_buf=DeviceBuffer[DType.int32](ctx, views[r].toks_d.unsafe_ptr().unsafe_offset(pos[r] + 1), 1, owning=False))
    ctx.synchronize()
    for r in range(m):
        print(tok_line(ids[r], Int(b.stream_h[r])))
