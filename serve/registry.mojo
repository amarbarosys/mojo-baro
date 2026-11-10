from std.math import ceildiv
from std.sys import get_defined_string, get_defined_int
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from elementwise import (
    amar_rmsnorm, amar_rmsnorm_cast, amar_rmsnorm_cast2, amar_rmsnorm_cast2_q8, amar_rmsnorm_cast2_rows, amar_embed_lookup_pos, amar_argmax_pos, amar_tok_copy, amar_tok_remap, amar_graft_mean,
    amar_quantize_q8_rows,
)
from matmul_skinny import (
    amar_matmul_skinny_q8row, amar_matmul_skinny_q4rowb, amar_skinny_reduce, amar_skinny_reduce_add,
    amar_skinny_reduce_swiglu_bf16, amar_matmul_skinny_q8dot, amar_matmul_skinny_m1_row2, SM, SPLITK, ROW_WAVES, ROW_THREADS,
)
from matmul_ternary import (
    amar_matmul_skinny_q2b3row, amar_matmul_skinny_tq1row, amar_matmul_skinny_tq2row,
)
from ssm import (
    amar_ssm_reduce_gates, amar_ssm_conv, amar_ssm_qk_l2norm, amar_ssm_qk_l2norm_gates, amar_ssm_conv_l2g,
    amar_ssm_delta_step, amar_ssm_gated_out_bf16, amar_cast_bf16, amar_widen_bf16, CONV, CONV_GATE_BLOCKS, NH_V, SSTATE,
    amar_ssm_gates_rows, amar_ssm_conv_chunk, amar_ssm_qk_l2norm_rows, amar_ssm_delta_chunk,
    amar_ssm_gated_out_rows_bf16, amar_ssm_delta_chunk_w, amar_ssm_delta_rows, DC_BLOCKS,
)
from matmul_prefill import (
    amar_matmul_prefill_q4, amar_matmul_prefill_q8, amar_prefill_swiglu_bf16, PF_THREADS,
)
from matmul_prefill_lds import amar_matmul_prefill_lds, LDS_THREADS
from mega import amar_mega_token, amar_mega_window, MEGA_G, MEGA_G_WIN, DATT_NLD
from nanprobe import amar_nonfinite_rows, NP_THREADS
from mega_moe import amar_mega_moe_token, MOE_BARRIERS, MOE_BARRIERS_DOT, W_ATT as MOE_W_ATT, W_SSM as MOE_W_SSM
from mega_moe_rows import amar_mega_moe_rows
from sample import amar_sample_row, amar_sample_row_masked, amar_sample_probs, amar_spec_accept, amar_apply_penalties, amar_topn_probs, SAMP_THREADS, SAMP_CAP

from dattn import amar_dattn_split, amar_dattn_combine, dattn_nsplit
from vision import amar_rope_mrope, amar_head_rmsnorm_mrope, amar_inject_rows
from attn import (
    amar_head_rmsnorm_rope, amar_kv_append2,
    amar_head_rmsnorm, amar_attn_decode, amar_gate_mul_cast, amar_qgate_split, amar_rope_yarn, amar_kv_append,
    amar_attn_prefill, amar_attn_prefill_wmma, amar_attn_mla_decode, HD, NQH, NKVH, KVQ, KVQ8, KVT, TCAP, KVPAGE, KVHSTR, PA_ROWS, PW_ROWS, PW_THREADS,
)

from model import H, FFN, VOCAB, QF, KV, N_LAYERS, N_SSM, N_ATT, MEGA_ALLOWED, IS_MOE
from moe import amar_moe_down_q6k, amar_moe_router_top8_sig, amar_moe_router_top1, moe_down_q8_0_res, moe_matmul_q8_0_m1_add

comptime TMAX = 1088
comptime GEN_N = 64
# B4 stage 2 (bench/moe-locality-protocol.md): how many decoded tokens the
# expert-id trace plane holds. One GEN_N-length request fills it exactly; a
# longer request stops tracing rather than wrapping, so a trace is never a
# mixture of two positions.
comptime TRACE_TOK = 64
comptime PF_CAP_ROWS = 1024  # rows of one batched-prefill chunk the capture staging holds
comptime SPLICE_MAX = 1024  # positions a splice rows file may cover
# The trace plane's width is a literal 8 rather than the MoE profile's TOPK,
# because registry is shared with the dense profiles, which have no experts.
comptime TOPK_TRACE = get_defined_int["BARO_TOPK", 8]()
comptime CP = 1024
comptime PF_LDS_MIN = 128
comptime PF_MIN = 16

comptime bf16 = DType.bfloat16
comptime f32 = DType.float32
comptime f16 = DType.float16
comptime i8 = DType.int8

comptime MROWS = SM
comptime KMAX = SM
# Concurrent sequences the KV pool is sized for; each one costs a full
# tmax of KV, so a single-user long-context build uses -D BARO_SEQ_CAP=1.
comptime SEQ_CAP = get_defined_int["BARO_SEQ_CAP", 4]()
comptime SLOTS = get_defined_int["BARO_SLOTS", KMAX + 1]()
comptime CONV_SLOT = N_SSM * 3 * CONV
comptime SSM_SLOT = N_SSM * NH_V * SSTATE * SSTATE
comptime TPAGES = ceildiv(TMAX, KVPAGE)
comptime KVPOOL = TPAGES * N_ATT * NKVH * KVHSTR
comptime KVPOOL1 = TPAGES * NKVH * KVHSTR

comptime h_layout = row_major[H]()
comptime h2_layout = row_major[1, H]()
comptime xm_layout = row_major[MROWS, H]()
comptime xflat_layout = row_major[MROWS * H]()
comptime ATT = NQH * HD
# SSM value width; equals H only on Qwythos-9B (4096), so never use H for it.
comptime INNER = NH_V * SSTATE
comptime xmi_layout = row_major[MROWS, INNER]()
# The mega persistent kernel's call sites view z/resb/qkv with H/QF strides: Qwythos-9B shape only.
comptime MEGA_FITS = ATT <= INNER
comptime attm_layout = row_major[MROWS, ATT]()
comptime attflat_layout = row_major[MROWS * ATT]()
comptime qfm_layout = row_major[MROWS, QF]()
comptime convm_layout = row_major[MROWS, CONV]()
comptime qm_layout = row_major[MROWS * NQH, HD]()
comptime kvm_layout = row_major[MROWS * NKVH, HD]()
comptime kvm_flat = row_major[MROWS, KV]()
comptime ffnm_layout = row_major[MROWS, FFN]()
comptime aqm_h = row_major[MROWS, H]()
comptime asm_h = row_major[MROWS, H // 32]()
comptime aqm_ffn = row_major[MROWS, FFN]()
comptime asm_ffn = row_major[MROWS, FFN // 32]()
comptime vm_layout = row_major[MROWS, VOCAB]()
comptime qf_layout = row_major[QF]()
comptime q_layout = row_major[NQH, HD]()
comptime kvflat_layout = row_major[KV]()
comptime cache_layout = row_major[TCAP]()
comptime cache1_layout = row_major[TCAP]()
comptime hd_layout = row_major[HD]()
comptime conv_layout = row_major[CONV]()
comptime cw_layout = row_major[CONV, 4]()
comptime s_layout = row_major[NH_V, SSTATE, SSTATE]()
comptime o_layout = row_major[NH_V, SSTATE]()
comptime g32_layout = row_major[NH_V]()
comptime g32m_layout = row_major[MROWS, NH_V]()
comptime om_layout = row_major[MROWS, NH_V, SSTATE]()
comptime csall_layout = row_major[SLOTS, N_SSM, 3, CONV]()
comptime ssall_layout = row_major[SLOTS, N_SSM, NH_V, SSTATE, SSTATE]()
comptime n128_layout = row_major[SSTATE]()
comptime emb_layout = row_major[VOCAB, H]()
comptime vrow_layout = row_major[1, VOCAB]()
comptime toks_layout = row_major[TCAP]()
comptime dtok_layout = row_major[KMAX + 1]()
# Item 3-4, briefs/2026-09-16-sampling-all-models-lane.md: sparse per-row
# penalty lists (distinct generated ids + counts, kernels/sample.mojo's
# amar_apply_penalties) and the top-N probability row (amar_topn_probs).
# One row per window position, so a later spec+sample+penalties round can
# fill rows 1..MROWS-1 with drafts 0..j-1 without a layout change; only
# row 0 is filled today (plain non-spec sampled decode).
comptime pen_ids_layout = row_major[MROWS, SAMP_CAP]()
comptime pen_npen_layout = row_major[MROWS]()
comptime NTOPLP = 20
comptime topn_layout = row_major[MROWS, NTOPLP]()

comptime q_h_qf = row_major[QF, H]()
comptime s_h_qf = row_major[QF, H // 32]()
comptime q_h_h = row_major[H, H]()
comptime s_h_h = row_major[H, H // 32]()
comptime q_h_kv = row_major[KV, H]()
comptime s_h_kv = row_major[KV, H // 32]()
comptime q_h_32 = row_major[NH_V, H]()
comptime s_h_32 = row_major[NH_V, H // 32]()
comptime q_h_ffn = row_major[FFN, H]()
comptime s_h_ffn = row_major[FFN, H // 32]()
comptime q_ffn_h = row_major[H, FFN]()
comptime s_ffn_h = row_major[H, FFN // 32]()
comptime q_h_v = row_major[VOCAB, H]()
comptime s_h_v = row_major[VOCAB, H // 32]()
comptime q4_h_v = row_major[VOCAB, H // 2]()
comptime EH = 2 * H
comptime ccm_layout = row_major[MROWS, EH]()
comptime q_eh_h = row_major[H, EH]()
comptime s_eh_h = row_major[H, EH // 32]()

comptime p_qf = row_major[SPLITK, SM, QF]()
comptime p_h = row_major[SPLITK, SM, H]()
comptime p_hi = row_major[SPLITK, SM, INNER]()
comptime p_kv = row_major[SPLITK, SM, KV]()
comptime p_32 = row_major[SPLITK, SM, NH_V]()
comptime p_ffn = row_major[SPLITK, SM, FFN]()
comptime p_att_layout = row_major[SPLITK * SM * FFN]()
comptime p_v = row_major[SPLITK, SM, VOCAB]()
comptime c_qf = row_major[1, QF]()
comptime c_h = row_major[1, H]()
comptime c_kv = row_major[1, KV]()

comptime OFF_CAP = 1024
comptime off_layout = row_major[OFF_CAP]()
comptime pf_sm = row_major[SM, FFN]()
comptime ctr_layout = row_major[3]()
comptime mega_token_k = amar_mega_token[
    1, False, False, type_of(xm_layout), type_of(xm_layout), type_of(xmi_layout), type_of(xmi_layout),
    type_of(qfm_layout), type_of(g32m_layout), type_of(convm_layout), type_of(om_layout),
    type_of(csall_layout), type_of(ssall_layout),
    type_of(qfm_layout), type_of(kvm_flat), type_of(qm_layout), type_of(attflat_layout),
    type_of(pf_sm), type_of(ffnm_layout), type_of(off_layout), type_of(ctr_layout), type_of(toks_layout), type_of(dtok_layout),
    N_LAYERS, N_ATT,
]
comptime mega_token_q4_k = amar_mega_token[
    1, True, True, type_of(xm_layout), type_of(xm_layout), type_of(xmi_layout), type_of(xmi_layout),
    type_of(qfm_layout), type_of(g32m_layout), type_of(convm_layout), type_of(om_layout),
    type_of(csall_layout), type_of(ssall_layout),
    type_of(qfm_layout), type_of(kvm_flat), type_of(qm_layout), type_of(attflat_layout),
    type_of(pf_sm), type_of(ffnm_layout), type_of(off_layout), type_of(ctr_layout), type_of(toks_layout), type_of(dtok_layout),
    N_LAYERS, N_ATT,
]
comptime mega_token_kq_k = amar_mega_token[
    1, False, False, type_of(xm_layout), type_of(xm_layout), type_of(xmi_layout), type_of(xmi_layout),
    type_of(qfm_layout), type_of(g32m_layout), type_of(convm_layout), type_of(om_layout),
    type_of(csall_layout), type_of(ssall_layout),
    type_of(qfm_layout), type_of(kvm_flat), type_of(qm_layout), type_of(attflat_layout),
    type_of(pf_sm), type_of(ffnm_layout), type_of(off_layout), type_of(ctr_layout), type_of(toks_layout), type_of(dtok_layout),
    N_LAYERS, N_ATT, True,
]
comptime mega_moe_k = amar_mega_moe_token[type_of(csall_layout), type_of(ssall_layout), N_LAYERS, N_ATT]
comptime mega_moe2_k = amar_mega_moe_rows[type_of(csall_layout), type_of(ssall_layout), N_LAYERS, N_ATT, 2]
comptime MOE_G_STR = get_defined_string["BARO_MOE_G", "96"]()
comptime MOE_G = 192 if MOE_G_STR == "192" else (288 if MOE_G_STR == "288" else 96)
comptime MEGA_MR = 3
comptime mega_win_k = amar_mega_window[
    MEGA_MR, True, False, type_of(xm_layout), type_of(xm_layout), type_of(xmi_layout), type_of(xmi_layout),
    type_of(qfm_layout), type_of(g32m_layout), type_of(convm_layout), type_of(om_layout),
    type_of(csall_layout), type_of(ssall_layout),
    type_of(qfm_layout), type_of(kvm_flat), type_of(qm_layout), type_of(attflat_layout),
    type_of(pf_sm), type_of(ffnm_layout), type_of(off_layout), type_of(ctr_layout), type_of(toks_layout), type_of(dtok_layout),
    N_LAYERS, N_ATT,
]
comptime mega_win_q4_k = amar_mega_window[
    MEGA_MR, True, True, type_of(xm_layout), type_of(xm_layout), type_of(xmi_layout), type_of(xmi_layout),
    type_of(qfm_layout), type_of(g32m_layout), type_of(convm_layout), type_of(om_layout),
    type_of(csall_layout), type_of(ssall_layout),
    type_of(qfm_layout), type_of(kvm_flat), type_of(qm_layout), type_of(attflat_layout),
    type_of(pf_sm), type_of(ffnm_layout), type_of(off_layout), type_of(ctr_layout), type_of(toks_layout), type_of(dtok_layout),
    N_LAYERS, N_ATT,
]

comptime xp_layout = row_major[CP, H]()
comptime xpi_layout = row_major[CP, INNER]()
comptime attp_layout = row_major[CP, ATT]()
comptime attpflat_layout = row_major[CP * ATT]()
comptime qfp_layout = row_major[CP, QF]()
comptime g32p_layout = row_major[CP, NH_V]()
comptime convp_layout = row_major[CP, CONV]()
comptime op_layout = row_major[CP, NH_V, SSTATE]()
comptime qp_layout = row_major[CP * NQH, HD]()
comptime kvp_layout = row_major[CP * NKVH, HD]()
comptime kvp_flat = row_major[CP, KV]()
comptime ffnp_layout = row_major[CP, FFN]()

comptime B2 = 2
comptime B4 = 4

comptime rmsc_k = amar_rmsnorm_cast[type_of(xm_layout), type_of(h_layout), type_of(xm_layout)]
comptime rmsc2q_k = amar_rmsnorm_cast2_q8[H, type_of(xm_layout), type_of(h_layout), type_of(xm_layout), type_of(h2_layout)]
comptime embed_k = amar_embed_lookup_pos[type_of(emb_layout), type_of(xm_layout), type_of(toks_layout)]
comptime argmax_k = amar_argmax_pos[type_of(vm_layout), type_of(toks_layout)]
comptime argmax_d = amar_argmax_pos[type_of(vm_layout), type_of(dtok_layout)]
# M5 (briefs/2026-09-15-wiring-lane.md): temperature > 0 replaces argmax_k
# with this at the same call site (window.mojo). Out/Prob share dtok_layout
# (KMAX + 1 >= MROWS, and dtok_d is otherwise unused when spec is off, which
# temperature > 0 forces) -- reusing scratch rather than allocating new
# buffers, same as reusing hmax_d for Prob.
comptime sample_row_k = amar_sample_row[type_of(vm_layout), type_of(dtok_layout), type_of(dtok_layout)]
# JSON-enforcement item 1 (briefs/2026-09-16-json-enforcement-lane.md): same
# instantiation, masked, both temperature > 0 and masked greedy at <= 0.
# m == 1 only (grammar requests run with spec off) -- vrow_layout would also work
# for m == 1, but reusing vm_layout keeps one alias for both the plain and
# masked call sites at the window.mojo sampling branch.
comptime sample_row_masked_k = amar_sample_row_masked[type_of(vm_layout), type_of(dtok_layout), type_of(dtok_layout)]
# A1 (bench/spec-sample-protocol.md): sampled speculation needs the truncated
# probability row of the target and of the draft, and the accept plus residual
# draw. Both kernels already exist and are not touched by this round; these are
# the instantiations the window uses. MROWS == KMAX == SM, so one window's rows
# fit vm_layout on both sides.
comptime sample_probs_k = amar_sample_probs[type_of(vm_layout), type_of(vm_layout)]
# One-row views, for the draft head: it produces a single logits row per draft
# step, and its q row has to land in the draft plane's row j rather than row 0.
comptime sample_probs_1 = amar_sample_probs[type_of(vrow_layout), type_of(vrow_layout)]
comptime sample_row_1 = amar_sample_row[type_of(vrow_layout), type_of(dtok_layout), type_of(dtok_layout)]
comptime spec_accept_k = amar_spec_accept[type_of(vm_layout), type_of(dtok_layout)]
comptime apply_penalties_k = amar_apply_penalties[type_of(vm_layout), type_of(pen_ids_layout), type_of(pen_ids_layout), type_of(pen_npen_layout)]
comptime topn_probs_k = amar_topn_probs[type_of(vm_layout), type_of(topn_layout), type_of(topn_layout), CAP=SAMP_CAP]
comptime embed1_k = amar_embed_lookup_pos[type_of(emb_layout), type_of(h2_layout), type_of(dtok_layout)]
comptime rms_m = amar_rmsnorm[type_of(xm_layout), type_of(h_layout), type_of(xm_layout)]
comptime rms_h2 = amar_rmsnorm[type_of(h2_layout), type_of(h_layout), type_of(h2_layout)]
comptime rmsc_h2 = amar_rmsnorm_cast[type_of(h2_layout), type_of(h_layout), type_of(h2_layout)]
comptime cast_m = amar_cast_bf16[type_of(xflat_layout), type_of(xflat_layout)]
comptime cast_1 = amar_cast_bf16[type_of(h_layout), type_of(h_layout)]
# window.mojo widens bf16 activations back to f32 on the MoE path; the kernel
# census only scans registry/engine/spark/bench/test, so a kernel reached only
# from window.mojo reads as an orphan without an alias here.
comptime widen_1 = amar_widen_bf16[type_of(h_layout), type_of(h_layout)]
comptime tokcp_k = amar_tok_copy[type_of(dtok_layout), type_of(toks_layout)]
comptime tokcp_b = amar_tok_copy[type_of(toks_layout), type_of(dtok_layout)]
comptime graft_mean_k = amar_graft_mean[type_of(h_layout)]
comptime frmap_layout = row_major[VOCAB]()
comptime remap_d = amar_tok_remap[type_of(frmap_layout), type_of(dtok_layout)]
comptime r_qf = amar_skinny_reduce[type_of(p_qf), type_of(qfm_layout), 1]
comptime r_h = amar_skinny_reduce[type_of(p_h), type_of(xm_layout), 1]
comptime r_hi = amar_skinny_reduce[type_of(p_hi), type_of(xmi_layout), 1]
comptime r_kv = amar_skinny_reduce[type_of(p_kv), type_of(kvm_flat), 1]
comptime r_add = amar_skinny_reduce_add[type_of(p_h), type_of(xm_layout), 1]
comptime r_swiglu = amar_skinny_reduce_swiglu_bf16[type_of(p_ffn), type_of(ffnm_layout), 1]
comptime r_head = amar_skinny_reduce[type_of(p_v), type_of(vm_layout), 1]
comptime rgates_k = amar_ssm_reduce_gates[
    type_of(p_32), type_of(g32m_layout), type_of(g32_layout)
]
comptime conv_k = amar_ssm_conv[
    type_of(qfm_layout), type_of(csall_layout), type_of(cw_layout), type_of(convm_layout)
]
comptime l2_k = amar_ssm_qk_l2norm[type_of(convm_layout)]
comptime convl2g_k = amar_ssm_conv_l2g[type_of(qfm_layout), type_of(csall_layout), type_of(cw_layout), type_of(convm_layout), type_of(p_32), type_of(g32m_layout), type_of(g32_layout)]
comptime l2g_k = amar_ssm_qk_l2norm_gates[type_of(convm_layout), type_of(p_32), type_of(g32m_layout), type_of(g32_layout)]
comptime gated_k = amar_ssm_gated_out_bf16[
    type_of(om_layout), type_of(xmi_layout), type_of(n128_layout), type_of(xmi_layout)
]
comptime split_k = amar_qgate_split[type_of(qfm_layout), type_of(qm_layout), type_of(attflat_layout)]
comptime hrms_q = amar_head_rmsnorm[type_of(qm_layout), type_of(hd_layout)]
comptime hrms_kv = amar_head_rmsnorm[type_of(kvm_layout), type_of(hd_layout)]
comptime rope_q = amar_rope_yarn[type_of(qm_layout)]
comptime rope_k = amar_rope_yarn[type_of(kvm_layout)]
comptime append_k = amar_kv_append[type_of(cache_layout), type_of(kvm_layout), N_ATT]
comptime append2_k = amar_kv_append2[type_of(cache_layout), type_of(kvm_layout), N_ATT]
comptime hrr_q = amar_head_rmsnorm_rope[type_of(qm_layout), type_of(hd_layout)]
comptime hrr_kv = amar_head_rmsnorm_rope[type_of(kvm_layout), type_of(hd_layout)]
comptime IMG_CAP = get_defined_int["BARO_IMG_CAP", 4096]()
comptime img_layout = row_major[IMG_CAP, H]()
comptime rope_q_m = amar_rope_mrope[type_of(qm_layout)]
comptime rope_k_m = amar_rope_mrope[type_of(kvm_layout)]
comptime hrr_q_m = amar_head_rmsnorm_mrope[type_of(qm_layout), type_of(hd_layout)]
comptime hrr_kv_m = amar_head_rmsnorm_mrope[type_of(kvm_layout), type_of(hd_layout)]
comptime inject_p = amar_inject_rows[type_of(xp_layout), type_of(img_layout), type_of(toks_layout)]
comptime append_1 = amar_kv_append[type_of(cache1_layout), type_of(kvm_layout), 1]
comptime att_k = amar_attn_decode[type_of(qm_layout), type_of(cache_layout), type_of(qm_layout), N_ATT]
comptime att_1 = amar_attn_decode[type_of(qm_layout), type_of(cache1_layout), type_of(qm_layout), 1]
comptime datt_k = amar_dattn_split[HD, NQH, NKVH, KVT, N_ATT, DATT_NLD, False, type_of(qm_layout), type_of(cache_layout), type_of(qm_layout), type_of(p_att_layout)]
comptime dcomb_k = amar_dattn_combine[HD, MEGA_G, type_of(p_att_layout), type_of(qm_layout)]
comptime gmul_k = amar_gate_mul_cast[type_of(attflat_layout), type_of(attflat_layout), type_of(attflat_layout)]


comptime rmsc_p = amar_rmsnorm_cast[type_of(xp_layout), type_of(h_layout), type_of(xp_layout)]
comptime rmsc2_p = amar_rmsnorm_cast2_rows[type_of(xp_layout), type_of(h_layout), type_of(xp_layout), type_of(xp_layout)]
comptime embed_p = amar_embed_lookup_pos[type_of(emb_layout), type_of(xp_layout), type_of(toks_layout)]
comptime gates_p = amar_ssm_gates_rows[type_of(g32p_layout), type_of(g32p_layout), type_of(g32_layout)]
comptime conv_p = amar_ssm_conv_chunk[type_of(qfp_layout), type_of(csall_layout), type_of(cw_layout), type_of(convp_layout)]
comptime l2_p = amar_ssm_qk_l2norm_rows[type_of(convp_layout)]
comptime delta_p = amar_ssm_delta_chunk[type_of(ssall_layout), type_of(convp_layout), type_of(g32p_layout), type_of(op_layout)]
comptime deltar_p = amar_ssm_delta_rows[type_of(ssall_layout), type_of(convp_layout), type_of(g32p_layout), type_of(op_layout)]
comptime deltaw_p = amar_ssm_delta_chunk_w[type_of(ssall_layout), type_of(convp_layout), type_of(g32p_layout), type_of(op_layout)]
comptime gated_p = amar_ssm_gated_out_rows_bf16[type_of(op_layout), type_of(xpi_layout), type_of(n128_layout), type_of(xpi_layout)]
comptime split_p = amar_qgate_split[type_of(qfp_layout), type_of(qp_layout), type_of(attpflat_layout)]
comptime hrms_qp = amar_head_rmsnorm[type_of(qp_layout), type_of(hd_layout)]
comptime hrms_kvp = amar_head_rmsnorm[type_of(kvp_layout), type_of(hd_layout)]
comptime rope_qp = amar_rope_yarn[type_of(qp_layout)]
comptime rope_kp = amar_rope_yarn[type_of(kvp_layout)]
comptime append_p = amar_kv_append[type_of(cache_layout), type_of(kvp_layout), N_ATT]
comptime attp_k = amar_attn_prefill[type_of(qp_layout), type_of(cache_layout), type_of(qp_layout), N_ATT]
comptime attpw_k = amar_attn_prefill_wmma[type_of(qp_layout), type_of(cache_layout), type_of(qp_layout), N_ATT]
comptime gmul_p = amar_gate_mul_cast[type_of(attpflat_layout), type_of(attpflat_layout), type_of(attpflat_layout)]
comptime swiglu_p = amar_prefill_swiglu_bf16[type_of(ffnp_layout), type_of(ffnp_layout)]


def gemm_prefill_q4[
    ACC: Bool, AL: TensorLayout, QL: TensorLayout, SL: TensorLayout, CL: TensorLayout
](
    ctx: DeviceContext,
    A: TileTensor[bf16, AL, MutAnyOrigin],
    Wq: TileTensor[DType.uint8, QL, MutAnyOrigin],
    Ws: TileTensor[DType.float16, SL, MutAnyOrigin],
    C: TileTensor[f32, CL, MutAnyOrigin],
    m: Int, n: Int, k: Int, v: Int = -1,
) raises:
    if v == 0:
        ctx.enqueue_function[amar_matmul_prefill_q4[1, 2, 1, ACC, AL, QL, SL, CL]](
            A, Wq, Ws, C, Int32(m), Int32(n), Int32(k), grid_dim=(ceildiv(n, 256), ceildiv(m, 16)), block_dim=PF_THREADS)
    elif v == 1:
        ctx.enqueue_function[amar_matmul_prefill_q4[2, 2, 1, ACC, AL, QL, SL, CL]](
            A, Wq, Ws, C, Int32(m), Int32(n), Int32(k), grid_dim=(ceildiv(n, 256), ceildiv(m, 32)), block_dim=PF_THREADS)
    elif v == 4:
        ctx.enqueue_function[amar_matmul_prefill_lds[DType.uint8, 2, 4, 2, 2, ACC, AL, QL, SL, CL]](
            A, Wq, Ws, C, Int32(m), Int32(n), Int32(k), grid_dim=(ceildiv(n, 128), ceildiv(m, 64)), block_dim=LDS_THREADS)
    elif v == 2 or (v < 0 and m <= 64):
        ctx.enqueue_function[amar_matmul_prefill_q4[2, 2, 2, ACC, AL, QL, SL, CL]](
            A, Wq, Ws, C, Int32(m), Int32(n), Int32(k), grid_dim=(ceildiv(n, 128), ceildiv(m, 64)), block_dim=PF_THREADS)
    elif v == 3 or (v < 0 and m <= PF_LDS_MIN):
        ctx.enqueue_function[amar_matmul_prefill_q4[4, 2, 2, ACC, AL, QL, SL, CL]](
            A, Wq, Ws, C, Int32(m), Int32(n), Int32(k), grid_dim=(ceildiv(n, 128), ceildiv(m, 128)), block_dim=PF_THREADS)
    else:
        ctx.enqueue_function[amar_matmul_prefill_lds[DType.uint8, 4, 2, 2, 4, ACC, AL, QL, SL, CL]](
            A, Wq, Ws, C, Int32(m), Int32(n), Int32(k), grid_dim=(ceildiv(n, 128), ceildiv(m, 128)), block_dim=LDS_THREADS)


def gemm_prefill_q8[
    ACC: Bool, AL: TensorLayout, QL: TensorLayout, SL: TensorLayout, CL: TensorLayout
](
    ctx: DeviceContext,
    A: TileTensor[bf16, AL, MutAnyOrigin],
    Wq: TileTensor[DType.int8, QL, MutAnyOrigin],
    Ws: TileTensor[DType.float16, SL, MutAnyOrigin],
    C: TileTensor[f32, CL, MutAnyOrigin],
    m: Int, n: Int, k: Int,
) raises:
    if m <= 64:
        ctx.enqueue_function[amar_matmul_prefill_q8[2, 2, 2, ACC, AL, QL, SL, CL]](
            A, Wq, Ws, C, Int32(m), Int32(n), Int32(k), grid_dim=(ceildiv(n, 128), ceildiv(m, 64)), block_dim=PF_THREADS)
    elif m <= PF_LDS_MIN:
        ctx.enqueue_function[amar_matmul_prefill_q8[4, 2, 2, ACC, AL, QL, SL, CL]](
            A, Wq, Ws, C, Int32(m), Int32(n), Int32(k), grid_dim=(ceildiv(n, 128), ceildiv(m, 128)), block_dim=PF_THREADS)
    else:
        ctx.enqueue_function[amar_matmul_prefill_lds[DType.int8, 4, 2, 2, 4, ACC, AL, QL, SL, CL]](
            A, Wq, Ws, C, Int32(m), Int32(n), Int32(k), grid_dim=(ceildiv(n, 128), ceildiv(m, 128)), block_dim=LDS_THREADS)


def delta_dispatch(
    ctx: DeviceContext,
    SAll: TileTensor[f32, type_of(ssall_layout), MutAnyOrigin],
    ConvOut: TileTensor[f32, type_of(convm_layout), MutAnyOrigin],
    Eg: TileTensor[f32, type_of(g32m_layout), MutAnyOrigin],
    Beta: TileTensor[f32, type_of(g32m_layout), MutAnyOrigin],
    O: TileTensor[f32, type_of(om_layout), MutAnyOrigin],
    ring: Int32, ssm_i: Int32, slots: Int32, m: Int,
) raises:
    comptime DL = type_of(ssall_layout)
    comptime CL = type_of(convm_layout)
    comptime GL = type_of(g32m_layout)
    comptime OL = type_of(om_layout)
    if m == 1:
        ctx.enqueue_function[amar_ssm_delta_step[1, DL, CL, GL, OL]](
            SAll, ConvOut, Eg, Beta, O, ring, ssm_i, slots, grid_dim=NH_V, block_dim=SSTATE)
    elif m == 2:
        ctx.enqueue_function[amar_ssm_delta_step[2, DL, CL, GL, OL]](
            SAll, ConvOut, Eg, Beta, O, ring, ssm_i, slots, grid_dim=NH_V, block_dim=SSTATE)
    elif m == 3:
        ctx.enqueue_function[amar_ssm_delta_step[3, DL, CL, GL, OL]](
            SAll, ConvOut, Eg, Beta, O, ring, ssm_i, slots, grid_dim=NH_V, block_dim=SSTATE)
    elif m == 4:
        ctx.enqueue_function[amar_ssm_delta_step[4, DL, CL, GL, OL]](
            SAll, ConvOut, Eg, Beta, O, ring, ssm_i, slots, grid_dim=NH_V, block_dim=SSTATE)
    elif m == 5:
        ctx.enqueue_function[amar_ssm_delta_step[5, DL, CL, GL, OL]](
            SAll, ConvOut, Eg, Beta, O, ring, ssm_i, slots, grid_dim=NH_V, block_dim=SSTATE)
    elif m == 6:
        ctx.enqueue_function[amar_ssm_delta_step[6, DL, CL, GL, OL]](
            SAll, ConvOut, Eg, Beta, O, ring, ssm_i, slots, grid_dim=NH_V, block_dim=SSTATE)
    elif m == 7:
        ctx.enqueue_function[amar_ssm_delta_step[7, DL, CL, GL, OL]](
            SAll, ConvOut, Eg, Beta, O, ring, ssm_i, slots, grid_dim=NH_V, block_dim=SSTATE)
    else:
        ctx.enqueue_function[amar_ssm_delta_step[SM, DL, CL, GL, OL]](
            SAll, ConvOut, Eg, Beta, O, ring, ssm_i, slots, grid_dim=NH_V, block_dim=SSTATE)


def gemm_q8[
    AL: TensorLayout, QL: TensorLayout, SL: TensorLayout, PL: TensorLayout
](
    ctx: DeviceContext,
    A: TileTensor[bf16, AL, MutAnyOrigin],
    Wq: TileTensor[DType.int8, QL, MutAnyOrigin],
    Ws: TileTensor[DType.float16, SL, MutAnyOrigin],
    P: TileTensor[f32, PL, MutAnyOrigin],
    m: Int, n: Int, k: Int,
) raises:
    if m == 1:
        ctx.enqueue_function[amar_matmul_skinny_q8row[4, 1, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    elif m == 2:
        ctx.enqueue_function[amar_matmul_skinny_q8row[4, 2, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    elif m == 3:
        ctx.enqueue_function[amar_matmul_skinny_q8row[4, 3, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    elif m <= 5:
        ctx.enqueue_function[amar_matmul_skinny_q8row[4, 5, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    else:
        ctx.enqueue_function[amar_matmul_skinny_q8row[4, SM, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )


def gemm_q2b3[
    AL: TensorLayout, QL: TensorLayout, SL: TensorLayout, PL: TensorLayout
](
    ctx: DeviceContext,
    A: TileTensor[bf16, AL, MutAnyOrigin],
    Wq: TileTensor[DType.uint8, QL, MutAnyOrigin],
    Ws: TileTensor[DType.float16, SL, MutAnyOrigin],
    P: TileTensor[f32, PL, MutAnyOrigin],
    m: Int, n: Int, k: Int,
) raises:
    if m == 1:
        ctx.enqueue_function[amar_matmul_skinny_q2b3row[1, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    else:
        ctx.enqueue_function[amar_matmul_skinny_q2b3row[SM, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )


def gemm_tq1[
    AL: TensorLayout, QL: TensorLayout, SL: TensorLayout, PL: TensorLayout
](
    ctx: DeviceContext,
    A: TileTensor[bf16, AL, MutAnyOrigin],
    Wq: TileTensor[DType.uint8, QL, MutAnyOrigin],
    Ws: TileTensor[DType.float16, SL, MutAnyOrigin],
    P: TileTensor[f32, PL, MutAnyOrigin],
    m: Int, n: Int, k: Int,
) raises:
    if m == 1:
        ctx.enqueue_function[amar_matmul_skinny_tq1row[1, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    else:
        ctx.enqueue_function[amar_matmul_skinny_tq1row[SM, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )


def gemm_tq2[
    AL: TensorLayout, QL: TensorLayout, SL: TensorLayout, PL: TensorLayout
](
    ctx: DeviceContext,
    A: TileTensor[bf16, AL, MutAnyOrigin],
    Wq: TileTensor[DType.uint8, QL, MutAnyOrigin],
    Ws: TileTensor[DType.float16, SL, MutAnyOrigin],
    P: TileTensor[f32, PL, MutAnyOrigin],
    m: Int, n: Int, k: Int,
) raises:
    if m == 1:
        ctx.enqueue_function[amar_matmul_skinny_tq2row[1, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    else:
        ctx.enqueue_function[amar_matmul_skinny_tq2row[SM, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )


def quant_rows[
    AL: TensorLayout, AQL: TensorLayout, ASL: TensorLayout
](
    ctx: DeviceContext,
    A: TileTensor[bf16, AL, MutAnyOrigin],
    Aq: TileTensor[i8, AQL, MutAnyOrigin],
    As: TileTensor[f16, ASL, MutAnyOrigin],
    m: Int, k: Int,
) raises:
    ctx.enqueue_function[amar_quantize_q8_rows[AL, AQL, ASL]](
        A, Aq, As, Int32(m), Int32(k), grid_dim=(m, k // 32), block_dim=32,
    )


def gemm_q8dot[
    AQL: TensorLayout, ASL: TensorLayout,
    QL: TensorLayout, SL: TensorLayout, PL: TensorLayout
](
    ctx: DeviceContext,
    Aq: TileTensor[i8, AQL, MutAnyOrigin],
    As: TileTensor[f16, ASL, MutAnyOrigin],
    Wq: TileTensor[i8, QL, MutAnyOrigin],
    Ws: TileTensor[f16, SL, MutAnyOrigin],
    P: TileTensor[f32, PL, MutAnyOrigin],
    m: Int, n: Int, k: Int,
) raises:
    if m == 3:
        ctx.enqueue_function[amar_matmul_skinny_q8dot[4, 3, AQL, ASL, QL, SL, PL]](
            Aq, As, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    elif m <= 5:
        ctx.enqueue_function[amar_matmul_skinny_q8dot[4, 5, AQL, ASL, QL, SL, PL]](
            Aq, As, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    else:
        ctx.enqueue_function[amar_matmul_skinny_q8dot[4, SM, AQL, ASL, QL, SL, PL]](
            Aq, As, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )


def gemm_q4[
    AL: TensorLayout, QL: TensorLayout, SL: TensorLayout, PL: TensorLayout
](
    ctx: DeviceContext,
    A: TileTensor[bf16, AL, MutAnyOrigin],
    Wq: TileTensor[DType.uint8, QL, MutAnyOrigin],
    Ws: TileTensor[DType.float16, SL, MutAnyOrigin],
    P: TileTensor[f32, PL, MutAnyOrigin],
    m: Int, n: Int, k: Int,
) raises:
    if m == 1:
        ctx.enqueue_function[amar_matmul_skinny_q4rowb[2, 1, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    elif m == 2:
        ctx.enqueue_function[amar_matmul_skinny_q4rowb[2, 2, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    elif m == 3:
        ctx.enqueue_function[amar_matmul_skinny_q4rowb[2, 3, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    elif m == 4:
        ctx.enqueue_function[amar_matmul_skinny_q4rowb[2, 4, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    elif m <= 5:
        ctx.enqueue_function[amar_matmul_skinny_q4rowb[2, 5, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )
    else:
        ctx.enqueue_function[amar_matmul_skinny_q4rowb[2, SM, AL, QL, SL, PL]](
            A, Wq, Ws, P, Int32(m), Int32(n), Int32(k),
            grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS,
        )

