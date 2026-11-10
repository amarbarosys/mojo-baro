from std.ffi import c_ssize_t, external_call
from std.math import ceildiv
from std.math import log, exp
from std.os import getenv, setenv
from std.sys import has_accelerator, get_defined_int
from std.time import perf_counter_ns
from max.algorithm import parallelize
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import TileTensor, TensorLayout, row_major
from attn import KVQ, KVT, KVPAGE, KVPAD
from slens_kernels import amar_rope_micro, amar_attn_micro, amar_micro_init
from elementwise import amar_rmsnorm_cast, amar_rmsnorm_cast_steer, amar_scale_one, amar_tok_copy, amar_rmsnorm
from matmul_skinny import ROW_WAVES, ROW_THREADS
from tokenizer import Tokenizer
from minja import render_chat
from sample import amar_sample_row, amar_sample_row_masked, amar_topn_probs, SAMP_THREADS
from wire import err_line
from serve_proto import is_cancel_line, idle_cancel_note, read_line, parse_request, default_sample_params, json_key, json_int, json_str, json_float, parse_schema_field, parse_reasoning_field
from semantic_latent import SemanticInput, load_semantic_input, DirectionalPins
from latent_mlp import LatentMLPAdapter
from grammar_rt import GrammarRuntime, compile_schema_matcher, reasoning_boundary_observe
from grammar.automaton import Bitset
from grammar.matcher import Matcher
from matmul_wmma_pipe import amar_matmul_wmma_pipe
from q4x_kernels import UP
from prefill_rows import pf_dequant_kq, pf_dequant_q8, pf_cast16, pf_qkv_split, pf_add, pf_act_mul
from gemv_kq import amar_gemv_kq, amar_gemv_kq_gateup, amar_gemv_kq_splitk, GT_Q4K, GT_Q6K
from moe import Q4K_BYTES, Q6K_BYTES
from spark_kernels import (
    amar_embed_lookup_f32, amar_gemv_q8, amar_argmax_part, amar_argmax_final, amar_rope_kv_append, amar_attn_decode_swa_gated, amar_rope_plain, amar_bias_add,
    amar_trace_sum, amar_attn_split_part, amar_attn_split_merge, amar_rope_qkv, att_span_len, ATT_NSMAX,
)
from profile import (
    H, FFN, VOCAB, N_LAYERS, NQH, NKVH, HD, NORM_EPS, NROT_FULL, BASE_FULL, NROT_SWA, BASE_SWA,
    SWA_WIN, SWA_PERIOD, SWA_FULL_PHASE, ROPE_NEOX, QKV_BIAS, HAS_GATE, ATTN_SCALE,
    ACTIVATION_GELU, TMAX, ROPE_FREQS,
)
from gguf_reader import GGUFModel


def ablate_project(ctx: DeviceContext, x_d: DeviceBuffer[f32], h: HostBuffer[f32], mut prev: List[Float32], lam: Float32, v: Pointer[Float32, _]) raises:
    """Applies dW = -lam v v^T W to the output y just added to X; prev becomes the new X."""
    ctx.enqueue_copy(dst_buf=h, src_buf=DeviceBuffer[f32](ctx, x_d.unsafe_ptr(), H, owning=False))
    ctx.synchronize()
    var p = h.unsafe_ptr()
    var d: Float32 = 0
    for j in range(H):
        d += v[j] * (p[j] - prev[j])
    for j in range(H):
        p[j] -= lam * v[j] * d
        prev[j] = p[j]
    ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, x_d.unsafe_ptr(), H, owning=False), src_buf=h)
    ctx.synchronize()


def load_steer(ctx: DeviceContext, sv_d: DeviceBuffer[f32], sv_h: HostBuffer[f32], path: String, layer: Int) raises:
    """sv_d = f32[N_LAYERS * H] from path (layer < 0), or zeros with row `layer` = f32[H] from path."""
    var b = open(path, "r").read_bytes()
    if len(b) != (N_LAYERS * H if layer < 0 else H) * 4:
        raise Error("steer file " + path + ": wrong size " + String(len(b)))
    if layer >= N_LAYERS:
        raise Error("steer_layer out of range")
    var dst = sv_h.unsafe_ptr().unsafe_bitcast[UInt8]()
    for j in range(N_LAYERS * H * 4):
        dst[j] = 0
    var o = 0 if layer < 0 else layer * H * 4
    for j in range(len(b)):
        dst[o + j] = b[j]
    ctx.enqueue_copy(dst_buf=sv_d, src_buf=sv_h)
    ctx.synchronize()


def load_steer_row(ctx: DeviceContext, sv_d: DeviceBuffer[f32], sv_h: HostBuffer[f32], path: String, layer: Int, row: Int, scale: Float32) raises:
    """sv_d = zeros with row `layer` = scale * (row `row` of the f32[*, H] bank at path)."""
    if layer < 0 or layer >= N_LAYERS or row < 0:
        raise Error("steer_layer / steer_row out of range")
    var f = open(path, "r")
    _ = f.seek(row * H * 4)
    var b = f.read_bytes(H * 4)
    if len(b) != H * 4:
        raise Error("steer_bank " + path + ": row " + String(row) + " past the end")
    var hb = sv_h.unsafe_ptr().unsafe_bitcast[UInt8]()
    for j in range(N_LAYERS * H * 4):
        hb[j] = 0
    for j in range(H * 4):  # copy by index: a pointer taken from `b` would not keep it alive past its last named use
        hb[layer * H * 4 + j] = b[j]
    var dst = sv_h.unsafe_ptr()
    for j in range(H):
        dst[layer * H + j] = dst[layer * H + j] * scale
    ctx.enqueue_copy(dst_buf=sv_d, src_buf=sv_h)
    ctx.synchronize()


comptime f32 = DType.float32
comptime bf16 = DType.bfloat16
comptime f16 = DType.float16
comptime QDIM = NQH * HD
comptime KVDIM = NKVH * HD
comptime QKV = QDIM + 2 * KVDIM
comptime KVHSTR = KVPAGE * HD + KVPAD
comptime TPAGES = TMAX // KVPAGE
comptime KVPOOL = TPAGES * N_LAYERS * NKVH * KVHSTR
# Per-layer pack order: attn_norm, attn_qkv, [attn_qkv_bias], [attn_gate], attn_output,
# ffn_norm, ffn_gate, ffn_up, ffn_down -- the two bracketed tensors are present only when
# the model's recipe calls for them (spark2_5 has the gate, none of the others do).
comptime OFF_QKV_BIAS = 2 if QKV_BIAS else -1
comptime QKV_S = get_defined_int["BARO_SP_QKV_S", 4]()
comptime O_S = get_defined_int["BARO_SP_O_S", 1]()
comptime DOWN_S = get_defined_int["BARO_SP_DOWN_S", 1]()
comptime HEAD_S = get_defined_int["BARO_SP_HEAD_S", 1]()
comptime GATE_BASE = 2 + (1 if QKV_BIAS else 0)
comptime OFF_GATE = GATE_BASE if HAS_GATE else -1
comptime OFF_O = GATE_BASE + (1 if HAS_GATE else 0)
comptime OFF_FFN_NORM = OFF_O + 1
comptime OFF_FFN_GATE = OFF_O + 2
comptime OFF_FFN_UP = OFF_O + 3
comptime OFF_DOWN = OFF_O + 4
comptime LSTRIDE = OFF_O + 5
# -D BARO_TRACE_SUM=1 (diagnostic build, bench/p4-trace-soak.sh): one FNV checksum per
# (position, layer, stage) written on the device with no host sync, dumped per request to
# BARO_TRACE_SUM_DIR. Stages per layer: 0 = QKV after bias, 1 = residual after attention,
# 2 = residual after FFN; the last slot of a position is the logits row. Two runs of the same
# prompt must produce the same table, and the first differing cell names the kernel group
# that produced a transient wrong value. Off (the default) compiles to nothing.
comptime TRACE_SUM = get_defined_int["BARO_TRACE_SUM", 0]() == 1
comptime ADT = DType.bfloat16 if get_defined_int["BARO_ACT_BF16", 0]() == 1 else DType.float32
comptime TRACE_STRIDE = 3 * N_LAYERS + 1

comptime x_l = row_major[1, H]()
comptime xb_l = row_major[1, H]()
comptime h_l = row_major[H]()
comptime hf_l = row_major[1, H]()
comptime toks_l = row_major[TMAX]()
comptime emb_l = row_major[VOCAB, H]()
comptime qkv1_l = row_major[QKV]()
comptime h1_l = row_major[H]()
comptime ffn1_l = row_major[FFN]()
comptime v1_l = row_major[VOCAB]()
comptime dummy_l = row_major[1]()
comptime aob2_l = row_major[NQH, HD]()
comptime q_l = row_major[NQH, HD]()
comptime kv_l = row_major[NKVH, HD]()
comptime cache_l = row_major[KVPOOL]()
comptime aob_l = row_major[1, QDIM]()
comptime gate_l = row_major[NQH]()
comptime p_h_l = row_major[1, 1, H]()
comptime fgb_l = row_major[1, FFN]()
comptime AM_NB = 128
comptime amv_l = row_major[AM_NB]()

comptime q_qkv = row_major[QKV, H]()
comptime s_qkv = row_major[QKV, H // 32]()
comptime q_gate = row_major[NQH, H]()
comptime s_gate = row_major[NQH, H // 32]()
comptime q_o = row_major[H, QDIM]()
comptime s_o = row_major[H, QDIM // 32]()
comptime q_ffn = row_major[FFN, H]()
comptime s_ffn = row_major[FFN, H // 32]()
comptime q_down = row_major[H, FFN]()
comptime s_down = row_major[H, FFN // 32]()
comptime q_out = row_major[VOCAB, H]()
comptime s_out = row_major[VOCAB, H // 32]()

# Batched prefill (wavegroup's T-row forward): PT prompt rows per chunk, GEMMs on f16-dequantised weights.
# Needs -D WMMA_TB=1 -D WMMA_C_F16=0 (B as stored [N, K], f32 C). BARO_PREFILL_ROWS=0 keeps token-by-token prefill.
comptime PT = 128
comptime px_l = row_major[PT, H]()
comptime PTS = 1024  # rows of the steer lens arena (B micro-sequences of T tokens)
comptime px2_l = row_major[PTS, H]()
comptime pq_l = row_major[PT * NQH, HD]()
comptime pkv_l = row_major[PT * NKVH, HD]()
comptime pg_l = row_major[PT * NQH]()
comptime PF_K = max(H, max(QDIM, FFN))
comptime PF_N = max(QKV, max(FFN, H))
comptime PF_WS = max(QKV * H, max(FFN * H, H * QDIM))
comptime k_emb_p = amar_embed_lookup_f32[type_of(emb_l), type_of(px_l), type_of(toks_l)]
comptime k_rms_p = amar_rmsnorm_cast[type_of(px_l), type_of(h_l), type_of(px_l), ADT]
comptime sv_l = row_major[N_LAYERS * H]()
comptime k_rms_s = amar_rmsnorm_cast[type_of(px2_l), type_of(h_l), type_of(px2_l), ADT]
comptime k_emb_s = amar_embed_lookup_f32[type_of(emb_l), type_of(px2_l), type_of(toks_l)]
comptime k_rms_p_steer = amar_rmsnorm_cast_steer[type_of(px_l), type_of(h_l), type_of(px_l), type_of(sv_l), ADT]
comptime k_rope_full_p = amar_rope_plain[NROT_FULL, type_of(pq_l), NEOX=ROPE_NEOX, FF=ROPE_FREQS]
comptime k_rope_swa_p = amar_rope_plain[NROT_SWA, type_of(pq_l), NEOX=ROPE_NEOX]
comptime k_kv_full_p = amar_rope_kv_append[NROT_FULL, N_LAYERS, type_of(cache_l), type_of(pkv_l), HD, NKVH, NEOX=ROPE_NEOX, FF=ROPE_FREQS]
comptime k_kv_swa_p = amar_rope_kv_append[NROT_SWA, N_LAYERS, type_of(cache_l), type_of(pkv_l), HD, NKVH, NEOX=ROPE_NEOX]
comptime k_att_p = amar_attn_decode_swa_gated[type_of(pq_l), type_of(cache_l), type_of(pg_l), type_of(pq_l), N_LAYERS, HD, NQH, NKVH, HAS_GATE=HAS_GATE, ODT=ADT]

comptime k_emb = amar_embed_lookup_f32[type_of(emb_l), type_of(x_l), type_of(toks_l)]
comptime k_rms = amar_rmsnorm_cast[type_of(x_l), type_of(h_l), type_of(xb_l), ADT]
comptime k_rms_steer = amar_rmsnorm_cast_steer[type_of(x_l), type_of(h_l), type_of(xb_l), type_of(sv_l), ADT]
comptime k_rms_f32 = amar_rmsnorm[type_of(x_l), type_of(h_l), type_of(hf_l)]
comptime k_bias = amar_bias_add[type_of(qkv1_l), type_of(qkv1_l)]
comptime k_rope_full = amar_rope_plain[NROT_FULL, type_of(q_l), NEOX=ROPE_NEOX, FF=ROPE_FREQS]
comptime k_rope_swa = amar_rope_plain[NROT_SWA, type_of(q_l), NEOX=ROPE_NEOX]
comptime k_kv_full = amar_rope_kv_append[NROT_FULL, N_LAYERS, type_of(cache_l), type_of(kv_l), HD, NKVH, NEOX=ROPE_NEOX, FF=ROPE_FREQS]
comptime k_kv_swa = amar_rope_kv_append[NROT_SWA, N_LAYERS, type_of(cache_l), type_of(kv_l), HD, NKVH, NEOX=ROPE_NEOX]
comptime k_rqkv_full = amar_rope_qkv[NROT_FULL, N_LAYERS, type_of(q_l), type_of(cache_l), type_of(kv_l), HD, NQH, NKVH, NEOX=ROPE_NEOX, FF=ROPE_FREQS]
comptime k_rqkv_swa = amar_rope_qkv[NROT_SWA, N_LAYERS, type_of(q_l), type_of(cache_l), type_of(kv_l), HD, NQH, NKVH, NEOX=ROPE_NEOX]
comptime k_att = amar_attn_decode_swa_gated[type_of(q_l), type_of(cache_l), type_of(gate_l), type_of(aob2_l), N_LAYERS, HD, NQH, NKVH, HAS_GATE=HAS_GATE, ODT=ADT]
comptime k_att_sp = amar_attn_split_part[type_of(q_l), type_of(cache_l), N_LAYERS, HD, NQH, NKVH]
comptime k_att_mg = amar_attn_split_merge[type_of(gate_l), type_of(aob2_l), HD, HAS_GATE=HAS_GATE, ODT=ADT]
comptime k_argmax = amar_argmax_part[AM_NB, type_of(v1_l), type_of(amv_l), type_of(amv_l)]
comptime k_argmax_final = amar_argmax_final[AM_NB, type_of(amv_l), type_of(amv_l), type_of(toks_l)]
# Item 2, briefs/2026-09-16-sampling-all-models-lane.md: amar_sample_row at
# this profile's VOCAB, instantiated the same way registry.mojo's
# sample_row_1 already is for the dense/MoE engine. samp_x_l is a 2-D
# [1, VOCAB] view of the same v1_l-shaped logits_d buffer the argmax path
# reads (sample_row_body requires X.flat_rank == 2); samp_o_l is the
# 1-element scratch the token id and its probability land in before
# amar_tok_copy moves the id into Toks, the same indirection registry.mojo
# uses via amar_tok_copy for the megakernel window.
comptime samp_x_l = row_major[1, VOCAB]()
comptime samp_o_l = row_major[1]()
comptime k_sample = amar_sample_row[type_of(samp_x_l), type_of(samp_o_l), type_of(samp_o_l)]
comptime k_sample_masked = amar_sample_row_masked[type_of(samp_x_l), type_of(samp_o_l), type_of(samp_o_l)]
comptime k_tokcp = amar_tok_copy[type_of(samp_o_l), type_of(toks_l)]
comptime NTOPLP = 20
comptime topn_l = row_major[1, NTOPLP]()
comptime k_topn = amar_topn_probs[type_of(samp_x_l), type_of(topn_l), type_of(topn_l)]


def wq[LT: TensorLayout](ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT) -> TileTensor[DType.int8, LT, MutAnyOrigin]:
    var b = DeviceBuffer[DType.int8](ctx, (wbuf.unsafe_ptr() + o).unsafe_bitcast[Scalar[DType.int8]](), n, owning=False)
    return rebind[TileTensor[DType.int8, LT, MutAnyOrigin]](TileTensor(b, lt))


def ws[LT: TensorLayout](ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT) -> TileTensor[DType.float16, LT, MutAnyOrigin]:
    var b = DeviceBuffer[DType.float16](ctx, (wbuf.unsafe_ptr() + o + n).unsafe_bitcast[Scalar[DType.float16]](), n // 32, owning=False)
    return rebind[TileTensor[DType.float16, LT, MutAnyOrigin]](TileTensor(b, lt))


def wf[LT: TensorLayout](ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], o: Int, n: Int, lt: LT) -> TileTensor[f32, LT, MutAnyOrigin]:
    var b = DeviceBuffer[f32](ctx, (wbuf.unsafe_ptr() + o).unsafe_bitcast[Scalar[f32]](), n, owning=False)
    return rebind[TileTensor[f32, LT, MutAnyOrigin]](TileTensor(b, lt))


def sub_f32[LT: TensorLayout](ctx: DeviceContext, b: DeviceBuffer[f32], o: Int, n: Int, lt: LT) -> TileTensor[f32, LT, MutAnyOrigin]:
    var v = DeviceBuffer[f32](ctx, b.unsafe_ptr() + o, n, owning=False)
    return rebind[TileTensor[f32, LT, MutAnyOrigin]](TileTensor(v, lt))


def kq_code(s: String) raises -> Int:
    if s == "q4k":
        return GT_Q4K
    if s == "q6k":
        return GT_Q6K
    raise Error("bad dtype " + s)


def kq_bytes(t: Int, n: Int) -> Int:
    return n // 256 * (Q4K_BYTES if t == GT_Q4K else Q6K_BYTES)


# kq holds (type A, type B, elements of the A rows) per index entry, 0 for f32 / q8; a dense pack from
# tools/engine-pack.py --dense writes Q4_K / Q6_K matrices as "q4k", "q6k" or "q4k/q6k/<A elements>".
def load_pack(ctx: DeviceContext, packdir: String, mut off: List[Int], mut kq: List[Int]) raises -> DeviceBuffer[DType.uint8]:
    var total = 0
    var n_kq = 0
    var n_q8 = 0
    with open(packdir + "/index.txt", "r") as f:
        for line in f.read().splitlines():
            var parts = line.split(" ")
            if len(parts) < 4:
                continue
            var n = Int(parts[3])
            var dt = String(parts[1])
            off.append(Int(parts[2]))
            var ta = 0
            var tb = 0
            var ea = 0
            if dt == "f32":
                total += n * 4
            elif dt == "q8":
                total += n + (n // 32) * 2
            else:
                var kp = dt.split("/")
                ta = kq_code(String(kp[0]))
                tb = ta
                ea = n
                if len(kp) == 3:
                    tb = kq_code(String(kp[1]))
                    ea = Int(String(kp[2]))
                elif len(kp) != 1:
                    raise Error("bad dtype " + dt)
                total += kq_bytes(ta, ea) + kq_bytes(tb, n - ea)
            kq.append(ta)
            kq.append(tb)
            kq.append(ea)
            if ta != 0:
                n_kq += 1
            elif dt == "q8":
                n_q8 += 1
    var wbuf = ctx.enqueue_create_buffer[DType.uint8](total)
    comptime CHUNK = 1 << 28
    comptime RSPLIT = 4
    var stage0 = ctx.enqueue_create_host_buffer[DType.uint8](CHUNK)
    var stage1 = ctx.enqueue_create_host_buffer[DType.uint8](CHUNK)
    ctx.synchronize()
    var t_load = perf_counter_ns()
    with open(packdir + "/pack.bin", "r") as f:
        var fd = f._get_raw_fd()
        var done = 0
        var flip = False
        while done < total:
            var want = min(CHUNK, total - done)
            var stage = stage1 if flip else stage0
            var sptr = stage.unsafe_ptr()
            var rerr = List[Int64](unsafe_uninit_length=RSPLIT)
            var rerr_ptr = rerr.unsafe_ptr()
            def rchunk(t: Int) {imm fd, imm want, imm done, imm sptr, imm rerr_ptr}:
                var lo = want * t // RSPLIT
                var hi = want * (t + 1) // RSPLIT
                var got = lo
                while got < hi:
                    var n = external_call["pread", c_ssize_t](fd, sptr.unsafe_offset(got), hi - got, Int64(done + got))
                    if n <= 0:
                        rerr_ptr[t] = 1
                        return
                    got += Int(n)
                rerr_ptr[t] = 0
            parallelize(rchunk, RSPLIT)
            for t in range(RSPLIT):
                if rerr[t] != 0:
                    raise Error("short read")
            ctx.synchronize()
            var dslice = DeviceBuffer[DType.uint8](ctx, wbuf.unsafe_ptr() + done, want, owning=False)
            var hslice = stage.create_sub_buffer[DType.uint8](0, want) if want != CHUNK else stage
            ctx.enqueue_copy(dst_buf=dslice, src_buf=hslice)
            done += want
            flip = not flip
    ctx.synchronize()
    print("pack loaded", total, "bytes in", Float64(perf_counter_ns() - t_load) / 1e9, "s", " matrices: q8", n_q8, "kquant", n_kq, " act:", ADT)
    print("spark knobs: QKV_S", QKV_S, "O_S", O_S, "DOWN_S", DOWN_S, "HEAD_S", HEAD_S)
    return wbuf^


def gemv[
    EPI: Int, AL: TensorLayout, QL: TensorLayout, SL: TensorLayout, CL: TensorLayout, BL: TensorLayout, S: Int = 1
](
    ctx: DeviceContext, A: TileTensor[ADT, AL, MutAnyOrigin], wbuf: DeviceBuffer[DType.uint8], off: List[Int], kq: List[Int], e: Int,
    n: Int, k: Int, ql: QL, sl: SL, C: TileTensor[f32, CL, MutAnyOrigin], Ob: TileTensor[ADT, BL, MutAnyOrigin],
) raises:
    var g = ceildiv(n, ROW_WAVES)
    var ta = kq[3 * e]
    var tb = kq[3 * e + 1]
    if ta == 0:
        ctx.enqueue_function[amar_gemv_q8[EPI, AL, QL, SL, CL, BL, ADT]](A, wq(ctx, wbuf, off[e], n * k, ql), ws(ctx, wbuf, off[e], n * k, sl), C, Ob, Int32(n), Int32(k), grid_dim=g, block_dim=ROW_THREADS)
        return
    var w = wbuf.unsafe_ptr().unsafe_offset(off[e])
    var split = Int32(kq[3 * e + 2] // k)
    var offb = Int32(kq_bytes(ta, kq[3 * e + 2]))
    comptime if S > 1:
        var gs = ceildiv(n * S, ROW_WAVES)
        if ta == GT_Q4K and tb == GT_Q4K:
            ctx.enqueue_function[amar_gemv_kq_splitk[GT_Q4K, GT_Q4K, S, EPI, AL, CL, BL, ADT]](A, w, C, Ob, Int32(n), Int32(k), split, offb, grid_dim=gs, block_dim=ROW_THREADS)
            return
        if ta == GT_Q4K and tb == GT_Q6K:
            ctx.enqueue_function[amar_gemv_kq_splitk[GT_Q4K, GT_Q6K, S, EPI, AL, CL, BL, ADT]](A, w, C, Ob, Int32(n), Int32(k), split, offb, grid_dim=gs, block_dim=ROW_THREADS)
            return
        if ta == GT_Q6K and tb == GT_Q6K:
            ctx.enqueue_function[amar_gemv_kq_splitk[GT_Q6K, GT_Q6K, S, EPI, AL, CL, BL, ADT]](A, w, C, Ob, Int32(n), Int32(k), split, offb, grid_dim=gs, block_dim=ROW_THREADS)
            return
    if ta == GT_Q4K and tb == GT_Q4K:
        ctx.enqueue_function[amar_gemv_kq[GT_Q4K, GT_Q4K, EPI, AL, CL, BL, ADT]](A, w, C, Ob, Int32(n), Int32(k), split, offb, grid_dim=g, block_dim=ROW_THREADS)
    elif ta == GT_Q6K and tb == GT_Q6K:
        ctx.enqueue_function[amar_gemv_kq[GT_Q6K, GT_Q6K, EPI, AL, CL, BL, ADT]](A, w, C, Ob, Int32(n), Int32(k), split, offb, grid_dim=g, block_dim=ROW_THREADS)
    elif ta == GT_Q4K and tb == GT_Q6K:
        ctx.enqueue_function[amar_gemv_kq[GT_Q4K, GT_Q6K, EPI, AL, CL, BL, ADT]](A, w, C, Ob, Int32(n), Int32(k), split, offb, grid_dim=g, block_dim=ROW_THREADS)
    else:
        raise Error("unsupported K-quant pair " + String(ta) + "/" + String(tb) + " at index entry " + String(e))


def gateup[AL: TensorLayout, BL: TensorLayout](
    ctx: DeviceContext, A: TileTensor[ADT, AL, MutAnyOrigin], wbuf: DeviceBuffer[DType.uint8], off: List[Int], kq: List[Int], eg: Int, eu: Int,
    n: Int, k: Int, Ob: TileTensor[ADT, BL, MutAnyOrigin],
) raises -> Bool:
    """FFN gate and up as one fused K-quant GEMV when both are one plain Q4_K or Q6_K matrix; False leaves it to the two-GEMV path."""
    var t = kq[3 * eg]
    if t != kq[3 * eg + 1] or t != kq[3 * eu] or t != kq[3 * eu + 1] or (t != GT_Q4K and t != GT_Q6K):
        return False
    var wg = wbuf.unsafe_ptr().unsafe_offset(off[eg])
    var wu = wbuf.unsafe_ptr().unsafe_offset(off[eu])
    var g = ceildiv(n, ROW_WAVES)
    if t == GT_Q4K:
        ctx.enqueue_function[amar_gemv_kq_gateup[GT_Q4K, ACTIVATION_GELU, AL, BL, ADT]](A, wg, wu, Ob, Int32(n), Int32(k), grid_dim=g, block_dim=ROW_THREADS)
    else:
        ctx.enqueue_function[amar_gemv_kq_gateup[GT_Q6K, ACTIVATION_GELU, AL, BL, ADT]](A, wg, wu, Ob, Int32(n), Int32(k), grid_dim=g, block_dim=ROW_THREADS)
    return True


def pf_linear[N: Int, K: Int, M: Int = PT](
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], off: List[Int], kq: List[Int], e: Int, a: Int, T: Int, c: Int, x16: Int, wsc: Int,
) raises:
    """c[PT, N] (f32) = rows a[T, K] (ADT) times pack matrix e^T: cast rows to f16, dequantise e into the f16 scratch, WMMA GEMM."""
    var ws = Pointer[Float16, MutAnyOrigin](unsafe_from_address=wsc)
    ctx.enqueue_function[pf_cast16[ADT]](Pointer[Scalar[ADT], MutAnyOrigin](unsafe_from_address=a), Pointer[Float16, MutAnyOrigin](unsafe_from_address=x16),
        Int32(T * K), grid_dim=ceildiv(T * K, 256), block_dim=256)
    var ta = kq[3 * e]
    var tb = kq[3 * e + 1]
    var w = Int(wbuf.unsafe_ptr()) + off[e]
    if ta == 0:
        ctx.enqueue_function[pf_dequant_q8](Pointer[Int8, MutAnyOrigin](unsafe_from_address=w), Pointer[Float16, MutAnyOrigin](unsafe_from_address=w + N * K),
            ws, Int32(N * K), grid_dim=ceildiv(N * K, 256), block_dim=256)
    else:
        var split = kq[3 * e + 2] // K
        for part in range(2):
            var t = ta if part == 0 else tb
            var rows = split if part == 0 else N - split
            if rows == 0:
                continue
            var wp = UP(unsafe_from_address=w + (0 if part == 0 else kq_bytes(ta, kq[3 * e + 2])))
            var o = Pointer[Float16, MutAnyOrigin](unsafe_from_address=wsc + (0 if part == 0 else split * K * 2))
            if t == GT_Q4K:
                ctx.enqueue_function[pf_dequant_kq[GT_Q4K]](wp, o, Int32(rows), Int32(K), grid_dim=ceildiv(rows * K // 16, 256), block_dim=256)
            elif t == GT_Q6K:
                ctx.enqueue_function[pf_dequant_kq[GT_Q6K]](wp, o, Int32(rows), Int32(K), grid_dim=ceildiv(rows * K // 16, 256), block_dim=256)
            else:
                raise Error("prefill: unsupported K-quant " + String(t) + " at index entry " + String(e))
    comptime B128 = ceildiv(N, 128) * ceildiv(M, 128)
    comptime WM = 4 if B128 >= 64 else 2
    comptime WN = 4 if 64 <= B128 < 512 else 2
    comptime TN = 4 if B128 >= 512 else 2
    comptime assert N % (WN * TN * 16) == 0, "prefill GEMM: N must be a multiple of the WMMA tile (a narrow output takes the GEMV)"
    var A = TileTensor(DeviceBuffer[f16](ctx, Pointer[Float16, MutAnyOrigin](unsafe_from_address=x16), M * K, owning=False), row_major[M, K]())
    var Bm = TileTensor(DeviceBuffer[f16](ctx, ws, N * K, owning=False), row_major[N, K]())
    var C = TileTensor(DeviceBuffer[f32](ctx, Pointer[Float32, MutAnyOrigin](unsafe_from_address=c), M * N, owning=False), row_major[M, N]())
    ctx.enqueue_function[amar_matmul_wmma_pipe[type_of(A).LayoutType, type_of(Bm).LayoutType, type_of(C).LayoutType, WM, WN, 2, TN, TB_=1, CF16=0]](
        A, Bm, C, Int32(M), Int32(N), Int32(K), grid_dim=(ceildiv(N, WN * TN * 16), ceildiv(M, WM * 2 * 16)), block_dim=WM * WN * 32)


comptime P = Pointer[Float32, MutAnyOrigin]


def rope_ff(wbuf: DeviceBuffer[DType.uint8], off: List[Int]) -> P:
    """rope_freqs on the device (the pack's last entry) when the profile has it, else any valid address (unused)."""
    comptime if ROPE_FREQS:
        return P(unsafe_from_address=Int(wbuf.unsafe_ptr()) + off[3 + LSTRIDE * N_LAYERS])
    return P(unsafe_from_address=Int(wbuf.unsafe_ptr()))


def pf_t32[L: TensorLayout](ctx: DeviceContext, a: Int, l: L) -> TileTensor[f32, L, MutAnyOrigin]:
    var b = DeviceBuffer[f32](ctx, P(unsafe_from_address=a), l.size(), owning=False)
    return rebind[TileTensor[f32, L, MutAnyOrigin]](TileTensor(b, l))


def pf_tadt[L: TensorLayout](ctx: DeviceContext, a: Int, l: L) -> TileTensor[ADT, L, MutAnyOrigin]:
    var b = DeviceBuffer[ADT](ctx, Pointer[Scalar[ADT], MutAnyOrigin](unsafe_from_address=a), l.size(), owning=False)
    return rebind[TileTensor[ADT, L, MutAnyOrigin]](TileTensor(b, l))


def prefill_rows(
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], off: List[Int], kq: List[Int], Emb: TileTensor[f32, type_of(emb_l), MutAnyOrigin],
    Toks: TileTensor[DType.int32, type_of(toks_l), MutAnyOrigin], Kc: TileTensor[KVT, type_of(cache_l), MutAnyOrigin],
    Vc: TileTensor[KVT, type_of(cache_l), MutAnyOrigin], tab: Int, n: Int, arena: DeviceBuffer[DType.uint8],
    steer: Bool, SV: TileTensor[f32, type_of(sv_l), MutAnyOrigin],
) raises:
    """Rows 0..n-1 of the prompt through every layer, PT rows at a time; leaves their K/V in the cache (decode then starts at n)."""
    comptime A = 4 if ADT == f32 else 2
    var b = Int(arena.unsafe_ptr())
    var xp = b
    var xbp = xp + PT * H * 4
    var tmp = xbp + PT * PF_K * A
    var qp = tmp + PT * PF_N * 4
    var kp = qp + PT * QDIM * 4
    var vp = kp + PT * KVDIM * 4
    var gp = vp + PT * KVDIM * 4
    var gf = gp + PT * NQH * 4
    var x16 = gf + PT * FFN * 4
    var wsc = x16 + PT * PF_K * 2
    var Xp = pf_t32(ctx, xp, px_l)
    var Xbp = TileTensor(DeviceBuffer[ADT](ctx, Pointer[Scalar[ADT], MutAnyOrigin](unsafe_from_address=xbp), PT * H, owning=False), px_l)
    var Qp = pf_t32(ctx, qp, pq_l)
    var Kp = pf_t32(ctx, kp, pkv_l)
    var Vp = pf_t32(ctx, vp, pkv_l)
    var Gp = pf_t32(ctx, gp, pg_l)
    var tp = MutPointer[Scalar[DType.int32], MutAnyOrigin](unsafe_from_address=tab)
    var ffp = rope_ff(wbuf, off)
    var Aop = TileTensor(DeviceBuffer[ADT](ctx, Pointer[Scalar[ADT], MutAnyOrigin](unsafe_from_address=xbp), PT * QDIM, owning=False), pq_l)
    var c0 = 0
    while c0 < n:
        var T = min(PT, n - c0)
        ctx.enqueue_function[k_emb_p](Emb, Xp, Toks, Int32(c0), Int32(H), grid_dim=(ceildiv(H, 256), T), block_dim=256)
        for i in range(N_LAYERS):
            var e = 1 + LSTRIDE * i
            var swa = (i % SWA_PERIOD) != SWA_FULL_PHASE
            if steer and i > 0:
                ctx.enqueue_function[k_rms_p_steer](Xp, wf(ctx, wbuf, off[e], H, h_l), Xbp, Int32(H), NORM_EPS, SV, Int32((i - 1) * H), grid_dim=T, block_dim=256)
            else:
                ctx.enqueue_function[k_rms_p](Xp, wf(ctx, wbuf, off[e], H, h_l), Xbp, Int32(H), NORM_EPS, grid_dim=T, block_dim=256)
            pf_linear[QKV, H](ctx, wbuf, off, kq, e + 1, xbp, T, tmp, x16, wsc)
            var bias = P(unsafe_from_address=Int(wbuf.unsafe_ptr()) + off[e + OFF_QKV_BIAS]) if QKV_BIAS else P(unsafe_from_address=tmp)
            ctx.enqueue_function[pf_qkv_split[QKV_BIAS]](P(unsafe_from_address=tmp), bias, P(unsafe_from_address=qp), P(unsafe_from_address=kp),
                P(unsafe_from_address=vp), Int32(QDIM), Int32(KVDIM), Int32(T), grid_dim=ceildiv(T * QKV, 256), block_dim=256)
            if swa:
                ctx.enqueue_function[k_rope_swa_p](Qp, Int32(c0), Int32(NQH), BASE_SWA, ffp, grid_dim=(NQH, T), block_dim=NROT_SWA // 2)
                ctx.enqueue_function[k_kv_swa_p](Kc, Vc, Kp, Vp, tp, Int32(c0), BASE_SWA, Int32(i), ffp, grid_dim=(NKVH, 2, T), block_dim=HD)
            else:
                ctx.enqueue_function[k_rope_full_p](Qp, Int32(c0), Int32(NQH), BASE_FULL, ffp, grid_dim=(NQH, T), block_dim=NROT_FULL // 2)
                ctx.enqueue_function[k_kv_full_p](Kc, Vc, Kp, Vp, tp, Int32(c0), BASE_FULL, Int32(i), ffp, grid_dim=(NKVH, 2, T), block_dim=HD)
            comptime if HAS_GATE: # NQH columns: below the WMMA tile, so the decode GEMV per row
                for r in range(T):
                    gemv[0](ctx, pf_tadt(ctx, xbp + r * H * A, xb_l), wbuf, off, kq, e + OFF_GATE, NQH, H, q_gate, s_gate, pf_t32(ctx, gp + r * NQH * 4, gate_l), pf_tadt(ctx, tmp, dummy_l))
            ctx.enqueue_function[k_att_p](Qp, Kc, Vc, Gp, Aop, tp, Int32(c0 + 1), Int32(SWA_WIN if swa else 0), ATTN_SCALE, Int32(i), grid_dim=(NQH, T), block_dim=HD)
            pf_linear[H, QDIM](ctx, wbuf, off, kq, e + OFF_O, xbp, T, tmp, x16, wsc)
            ctx.enqueue_function[pf_add](P(unsafe_from_address=tmp), P(unsafe_from_address=xp), Int32(T * H), grid_dim=ceildiv(T * H, 256), block_dim=256)
            ctx.enqueue_function[k_rms_p](Xp, wf(ctx, wbuf, off[e + OFF_FFN_NORM], H, h_l), Xbp, Int32(H), NORM_EPS, grid_dim=T, block_dim=256)
            pf_linear[FFN, H](ctx, wbuf, off, kq, e + OFF_FFN_GATE, xbp, T, gf, x16, wsc)
            pf_linear[FFN, H](ctx, wbuf, off, kq, e + OFF_FFN_UP, xbp, T, tmp, x16, wsc)
            ctx.enqueue_function[pf_act_mul[ACTIVATION_GELU, ADT]](P(unsafe_from_address=gf), P(unsafe_from_address=tmp),
                Pointer[Scalar[ADT], MutAnyOrigin](unsafe_from_address=xbp), Int32(T * FFN), grid_dim=ceildiv(T * FFN, 256), block_dim=256)
            pf_linear[H, FFN](ctx, wbuf, off, kq, e + OFF_DOWN, xbp, T, tmp, x16, wsc)
            ctx.enqueue_function[pf_add](P(unsafe_from_address=tmp), P(unsafe_from_address=xp), Int32(T * H), grid_dim=ceildiv(T * H, 256), block_dim=256)
        c0 += T


def slens_pass[MG: Int](
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], off: List[Int], kq: List[Int], Emb: TileTensor[f32, type_of(emb_l), MutAnyOrigin],
    Toks: TileTensor[DType.int32, type_of(toks_l), MutAnyOrigin], arena: DeviceBuffer[DType.uint8], R: Int, T: Int, l_start: Int, cap_layer: Int, cap_addr: Int,
) raises:
    """Layers l_start.. on R = B * T rows of the arena (B micro-sequences of T tokens, no KV cache). l_start == 0 embeds Toks[0:R]
    first. cap_layer >= 0 copies the first T rows after that layer to cap_addr (the base residual a steer is added to)."""
    comptime if HAS_GATE or SWA_WIN > 0:
        raise Error("slens: needs a plain dense profile (no gate, no sliding window)")
    comptime A = 4 if ADT == f32 else 2
    var b = Int(arena.unsafe_ptr())
    var xp = b
    var xbp = xp + PTS * H * 4
    var tmp = xbp + PTS * PF_K * A
    var qp = tmp + PTS * PF_N * 4
    var kp = qp + PTS * QDIM * 4
    var vp = kp + PTS * KVDIM * 4
    var gf = vp + PTS * KVDIM * 4 + PTS * NQH * 4
    var x16 = gf + PTS * FFN * 4
    var wsc = x16 + PTS * PF_K * 2
    var Xp = pf_t32(ctx, xp, px2_l)
    var Xbp = TileTensor(DeviceBuffer[ADT](ctx, Pointer[Scalar[ADT], MutAnyOrigin](unsafe_from_address=xbp), PTS * H, owning=False), px2_l)
    var ffp = rope_ff(wbuf, off)
    var aop = Pointer[Scalar[ADT], MutAnyOrigin](unsafe_from_address=xbp)
    if l_start == 0:
        ctx.enqueue_function[k_emb_s](Emb, Xp, Toks, Int32(0), Int32(H), grid_dim=(ceildiv(H, 256), R), block_dim=256)
    for i in range(l_start, N_LAYERS):
        var e = 1 + LSTRIDE * i
        ctx.enqueue_function[k_rms_s](Xp, wf(ctx, wbuf, off[e], H, h_l), Xbp, Int32(H), NORM_EPS, grid_dim=R, block_dim=256)
        pf_linear[QKV, H, MG](ctx, wbuf, off, kq, e + 1, xbp, R, tmp, x16, wsc)
        var bias = P(unsafe_from_address=Int(wbuf.unsafe_ptr()) + off[e + OFF_QKV_BIAS]) if QKV_BIAS else P(unsafe_from_address=tmp)
        ctx.enqueue_function[pf_qkv_split[QKV_BIAS]](P(unsafe_from_address=tmp), bias, P(unsafe_from_address=qp), P(unsafe_from_address=kp),
            P(unsafe_from_address=vp), Int32(QDIM), Int32(KVDIM), Int32(R), grid_dim=ceildiv(R * QKV, 256), block_dim=256)
        ctx.enqueue_function[amar_rope_micro[HD, NROT_FULL, ROPE_NEOX, ROPE_FREQS]](P(unsafe_from_address=qp), Int32(T), Int32(NQH), BASE_FULL, ffp, grid_dim=(NQH, R), block_dim=NROT_FULL // 2)
        ctx.enqueue_function[amar_rope_micro[HD, NROT_FULL, ROPE_NEOX, ROPE_FREQS]](P(unsafe_from_address=kp), Int32(T), Int32(NKVH), BASE_FULL, ffp, grid_dim=(NKVH, R), block_dim=NROT_FULL // 2)
        ctx.enqueue_function[amar_attn_micro[HD, NQH, NKVH, ADT]](P(unsafe_from_address=qp), P(unsafe_from_address=kp), P(unsafe_from_address=vp), aop,
            Int32(T), ATTN_SCALE, grid_dim=(NQH, R), block_dim=HD)
        pf_linear[H, QDIM, MG](ctx, wbuf, off, kq, e + OFF_O, xbp, R, tmp, x16, wsc)
        ctx.enqueue_function[pf_add](P(unsafe_from_address=tmp), P(unsafe_from_address=xp), Int32(R * H), grid_dim=ceildiv(R * H, 256), block_dim=256)
        ctx.enqueue_function[k_rms_s](Xp, wf(ctx, wbuf, off[e + OFF_FFN_NORM], H, h_l), Xbp, Int32(H), NORM_EPS, grid_dim=R, block_dim=256)
        pf_linear[FFN, H, MG](ctx, wbuf, off, kq, e + OFF_FFN_GATE, xbp, R, gf, x16, wsc)
        pf_linear[FFN, H, MG](ctx, wbuf, off, kq, e + OFF_FFN_UP, xbp, R, tmp, x16, wsc)
        ctx.enqueue_function[pf_act_mul[ACTIVATION_GELU, ADT]](P(unsafe_from_address=gf), P(unsafe_from_address=tmp),
            Pointer[Scalar[ADT], MutAnyOrigin](unsafe_from_address=xbp), Int32(R * FFN), grid_dim=ceildiv(R * FFN, 256), block_dim=256)
        pf_linear[H, FFN, MG](ctx, wbuf, off, kq, e + OFF_DOWN, xbp, R, tmp, x16, wsc)
        ctx.enqueue_function[pf_add](P(unsafe_from_address=tmp), P(unsafe_from_address=xp), Int32(R * H), grid_dim=ceildiv(R * H, 256), block_dim=256)
        if i == cap_layer:
            ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, P(unsafe_from_address=cap_addr), T * H, owning=False), src_buf=DeviceBuffer[f32](ctx, P(unsafe_from_address=xp), T * H, owning=False))


def slens_head(
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], off: List[Int], kq: List[Int], arena: DeviceBuffer[DType.uint8], lg: DeviceBuffer[f32], lg_h: HostBuffer[f32], B: Int, T: Int,
) raises:
    """Logits of the last row of each of B micro-sequences (final norm, head GEMV per row) into lg_h [B, VOCAB]."""
    comptime A = 4 if ADT == f32 else 2
    var b = Int(arena.unsafe_ptr())
    var xp = b
    var xbp = xp + PTS * H * 4
    var tmp = xbp + PTS * PF_K * A
    for u in range(B):
        ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, P(unsafe_from_address=tmp + u * H * 4), H, owning=False),
            src_buf=DeviceBuffer[f32](ctx, P(unsafe_from_address=xp + (u * T + T - 1) * H * 4), H, owning=False))
    var Xl = pf_t32(ctx, tmp, px2_l)
    var Xbp = TileTensor(DeviceBuffer[ADT](ctx, Pointer[Scalar[ADT], MutAnyOrigin](unsafe_from_address=xbp), PTS * H, owning=False), px2_l)
    ctx.enqueue_function[k_rms_s](Xl, wf(ctx, wbuf, off[1 + LSTRIDE * N_LAYERS], H, h_l), Xbp, Int32(H), NORM_EPS, grid_dim=B, block_dim=256)
    for u in range(B):
        gemv[0, S=HEAD_S](ctx, pf_tadt(ctx, xbp + u * H * A, xb_l), wbuf, off, kq, 2 + LSTRIDE * N_LAYERS, VOCAB, H, q_out, s_out,
            pf_t32(ctx, Int(lg.unsafe_ptr()) + u * VOCAB * 4, v1_l), pf_tadt(ctx, tmp + PTS * PF_N * 4 - 4096, dummy_l))
    ctx.enqueue_copy(dst_buf=lg_h, src_buf=lg)
    ctx.synchronize()


def slens_acc_row(p: Pointer[Float32, MutAnyOrigin], base_lsm: Pointer[Float32, MutAnyOrigin], base_p: Pointer[Float32, MutAnyOrigin], acc: Pointer[Float32, MutAnyOrigin]) -> Float64:
    """acc += log_softmax(row) - base_lsm; returns the KL(base || row) term sum p_base (lsm_base - lsm_row)."""
    var m = p[0]
    for j in range(VOCAB):
        m = max(m, p[j])
    var z: Float64 = 0
    for j in range(VOCAB):
        z += Float64(exp(p[j] - m))
    var lse = m + Float32(log(z))
    var kc: Float64 = 0
    for j in range(VOCAB):
        var dl = p[j] - lse - base_lsm[j]
        acc[j] += dl
        kc -= Float64(base_p[j]) * Float64(dl)
    return kc


def slens_emit(acc: Pointer[Float32, MutAnyOrigin], inv: Float32, kl: Float64, k: Int, id: Int, unit: Int):
    """Print the k tokens whose mean log-prob change rose most (up) and fell most (dn) for one unit."""
    var uv = List[Float32](length=k, fill=-1e30)
    var ui = List[Int](length=k, fill=-1)
    var dv = List[Float32](length=k, fill=-1e30)
    var di = List[Int](length=k, fill=-1)
    for j in range(VOCAB):
        var d = acc[j] * inv
        for sgn in range(2):
            var x = d if sgn == 0 else -d
            if x <= (uv[k - 1] if sgn == 0 else dv[k - 1]):
                continue
            var q = k - 1
            while q > 0 and (uv[q - 1] if sgn == 0 else dv[q - 1]) < x:
                if sgn == 0:
                    uv[q] = uv[q - 1]
                    ui[q] = ui[q - 1]
                else:
                    dv[q] = dv[q - 1]
                    di[q] = di[q - 1]
                q -= 1
            if sgn == 0:
                uv[q] = x
                ui[q] = j
            else:
                dv[q] = x
                di[q] = j
    var out = String("{\"id\":") + String(id) + ",\"unit\":" + String(unit) + ",\"up\":["
    for q in range(k):
        out += ("," if q > 0 else "") + "[" + String(ui[q]) + "," + String(uv[q]) + "]"
    out += "],\"dn\":["
    for q in range(k):
        out += ("," if q > 0 else "") + "[" + String(di[q]) + "," + String(-dv[q]) + "]"
    print(out + "],\"kl\":" + String(kl * Float64(inv)) + "}")


def parse_ctxs(line: String, mut out: List[List[Int]]) -> Bool:
    """\"prompts\":[[id,...],[id,...]] into out."""
    var i = json_key(line, "prompts")
    var b = line.as_bytes()
    if i < 0 or i >= len(b) or b[i] != 91:
        return False
    i += 1
    while True:
        while i < len(b) and (b[i] == 32 or b[i] == 44):
            i += 1
        if i >= len(b):
            return False
        if b[i] == 93:
            return True
        if b[i] != 91:
            return False
        i += 1
        var one = List[Int]()
        while True:
            while i < len(b) and (b[i] == 32 or b[i] == 44):
                i += 1
            if i >= len(b):
                return False
            if b[i] == 93:
                i += 1
                break
            var v = 0
            if not json_int(line, i, v) or v < 0:
                return False
            one.append(v)
        out.append(one^)


def slens_op(
    ctx: DeviceContext, wbuf: DeviceBuffer[DType.uint8], off: List[Int], kq: List[Int], Emb: TileTensor[f32, type_of(emb_l), MutAnyOrigin],
    Toks: TileTensor[DType.int32, type_of(toks_l), MutAnyOrigin], toks_h: HostBuffer[DType.int32], toks_d: DeviceBuffer[DType.int32],
    line: String, id: Int,
) raises:
    """{"op":"slens","id":N,"layer":L,"bank":PATH,"row0":a,"rows":m,"scale":s,"k":K,"prompts":[[ids],..]}: for bank rows a..a+m-1
    (steering vectors added after layer L at every position of each prompt) print one line per unit with the top-K up / down
    tokens of the log-prob change at the last position, averaged over the prompts, and the mean KL(base || steered)."""
    var layer = -1
    var row0 = 0
    var nrows = 0
    var kk = 8
    var scale: Float64 = 1.0
    var bank = String("")
    var ctxs = List[List[Int]]()
    var i = json_key(line, "layer")
    if i < 0 or not json_int(line, i, layer) or layer < 0 or layer >= N_LAYERS:
        raise Error("slens: layer 0.." + String(N_LAYERS - 1))
    i = json_key(line, "row0")
    if i >= 0 and not json_int(line, i, row0):
        raise Error("slens: row0")
    i = json_key(line, "rows")
    if i < 0 or not json_int(line, i, nrows) or nrows < 1:
        raise Error("slens: rows >= 1")
    i = json_key(line, "k")
    if i >= 0 and not json_int(line, i, kk):
        raise Error("slens: k")
    if kk < 1 or kk > 64:
        raise Error("slens: k 1..64")
    i = json_key(line, "scale")
    if i >= 0 and not json_float(line, i, scale):
        raise Error("slens: scale")
    i = json_key(line, "bank")
    if i < 0 or not json_str(line, i, bank):
        raise Error("slens: bank path")
    if not parse_ctxs(line, ctxs) or len(ctxs) < 1:
        raise Error("slens: prompts [[ids],..]")
    var tmax = 0
    for c in range(len(ctxs)):
        if len(ctxs[c]) < 1:
            raise Error("slens: empty prompt")
        tmax = max(tmax, len(ctxs[c]))
    if tmax > 96:
        raise Error("slens: prompts up to 96 tokens")
    var nb = min(nrows, min(PTS // tmax, 64))
    var sar = ctx.enqueue_create_buffer[DType.uint8](slens_arena_bytes())
    var lg = ctx.enqueue_create_buffer[f32](nb * VOCAB)
    var lg_h = ctx.enqueue_create_host_buffer[f32](nb * VOCAB)
    var bank_d = ctx.enqueue_create_buffer[f32](nb * H)
    var bank_h = ctx.enqueue_create_host_buffer[f32](nb * H)
    var base_lsm = List[List[Float32]]()
    var base_p = List[List[Float32]]()
    var xc = List[DeviceBuffer[f32]]()
    var ids_h = ctx.enqueue_create_host_buffer[DType.int32](TMAX)
    # base: every prompt from the embeddings, unsteered; keep the residual after `layer` and the base log-softmax
    for c in range(len(ctxs)):
        var T = len(ctxs[c])
        for t in range(T):
            toks_h[t] = Int32(ctxs[c][t])
        ctx.enqueue_copy(dst_buf=toks_d, src_buf=toks_h)
        xc.append(ctx.enqueue_create_buffer[f32](T * H))
        slens_pass[128](ctx, wbuf, off, kq, Emb, Toks, sar, T, T, 0, layer, Int(xc[c].unsafe_ptr()))
        slens_head(ctx, wbuf, off, kq, sar, lg, lg_h, 1, T)
        var p = lg_h.unsafe_ptr()
        var m = p[0]
        for j in range(VOCAB):
            m = max(m, p[j])
        var z: Float64 = 0
        for j in range(VOCAB):
            z += Float64(exp(p[j] - m))
        var lse = m + Float32(log(z))
        base_lsm.append(List[Float32](length=VOCAB, fill=0))
        base_p.append(List[Float32](length=VOCAB, fill=0))
        for j in range(VOCAB):
            base_lsm[c][j] = p[j] - lse
            base_p[c][j] = exp(base_lsm[c][j])
    var acc = List[Float32](length=nb * VOCAB, fill=0)
    var klv = List[Float64](length=nb, fill=0)
    var u0 = 0
    while u0 < nrows:
        var B = min(nb, nrows - u0)
        var f = open(bank, "r")
        _ = f.seek((row0 + u0) * H * 4)
        var bb = f.read_bytes(B * H * 4)
        if len(bb) != B * H * 4:
            raise Error("slens: bank " + bank + " ends before row " + String(row0 + u0 + B - 1))
        for j in range(B * H * 4):
            bank_h.unsafe_ptr().unsafe_bitcast[UInt8]()[j] = bb[j]
        ctx.enqueue_copy(dst_buf=bank_d, src_buf=bank_h)
        for j in range(nb * VOCAB):
            acc[j] = 0
        for u in range(nb):
            klv[u] = 0
        for c in range(len(ctxs)):
            var T = len(ctxs[c])
            var R = B * T
            ctx.enqueue_function[amar_micro_init](P(unsafe_from_address=Int(sar.unsafe_ptr())), P(unsafe_from_address=Int(xc[c].unsafe_ptr())),
                P(unsafe_from_address=Int(bank_d.unsafe_ptr())), Float32(scale), Int32(T), Int32(H), Int32(R * H), grid_dim=ceildiv(R * H, 256), block_dim=256)
            if R <= 128:
                slens_pass[128](ctx, wbuf, off, kq, Emb, Toks, sar, R, T, layer + 1, -1, 0)
            else:
                slens_pass[PTS](ctx, wbuf, off, kq, Emb, Toks, sar, R, T, layer + 1, -1, 0)
            slens_head(ctx, wbuf, off, kq, sar, lg, lg_h, B, T)
            var lgp = lg_h.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            var accp = acc.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            var klp = klv.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            var bl = base_lsm[c].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            var bp = base_p[c].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            def post(u: Int) {imm lgp, imm accp, imm klp, imm bl, imm bp}:
                klp[u] += slens_acc_row(lgp + u * VOCAB, bl, bp, accp + u * VOCAB)
            parallelize(post, B)
        var inv = Float32(1.0) / Float32(len(ctxs))
        for u in range(B):
            slens_emit(acc.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]() + u * VOCAB, inv, klv[u], kk, id, u0 + u)
        u0 += B
    _ = (ids_h.unsafe_ptr(), len(xc))


def slens_arena_bytes() -> Int:
    comptime A = 4 if ADT == f32 else 2
    return PTS * H * 4 + PTS * PF_K * A + PTS * PF_N * 4 + PTS * (QDIM + 2 * KVDIM + NQH + FFN) * 4 + PTS * PF_K * 2 + PF_WS * 2


def pf_arena_bytes() -> Int:
    comptime A = 4 if ADT == f32 else 2
    return PT * H * 4 + PT * PF_K * A + PT * PF_N * 4 + PT * (QDIM + 2 * KVDIM + NQH + FFN) * 4 + PT * PF_K * 2 + PF_WS * 2


def read_prompt(path: String) raises -> List[Int]:
    var prompt = List[Int]()
    with open(path, "r") as f:
        var data = f.read_bytes()
        var val = 0
        var have = False
        for i in range(len(data)):
            var b = Int(data[i])
            if b >= 48 and b <= 57:
                val = val * 10 + (b - 48)
                have = True
            else:
                if have:
                    prompt.append(val)
                val = 0
                have = False
        if have:
            prompt.append(val)
    return prompt^


def tok_line(id: Int, tok: Int) -> String:
    return String("{\"id\":") + String(id) + ",\"tok\":" + String(tok) + "}"


# Export-only state file for the llama.cpp bridge (P1 item 4): serve/engine.mojo's
# BAROST01 with conv_n = ssm_n = 0 and a zero salt, the first ceil(pos/KVPAGE) pages
# of the page-major pool. Spark keeps the identity page table and has no state_load.
def save_kv_state(ctx: DeviceContext, kc_d: DeviceBuffer[KVT], vc_d: DeviceBuffer[KVT], path: String, prompt: List[Int], pos: Int) raises:
    comptime if KVT != DType.float32:
        raise Error("state_save: state files store f32 KV; this engine has BARO_KVQ=" + KVQ)
    else:
        var kvn = ceildiv(pos, KVPAGE) * N_LAYERS * NKVH * KVHSTR
        var kh = ctx.enqueue_create_host_buffer[KVT](kvn)
        var vh = ctx.enqueue_create_host_buffer[KVT](kvn)
        ctx.enqueue_copy(dst_buf=kh, src_buf=DeviceBuffer[KVT](ctx, kc_d.unsafe_ptr(), kvn, owning=False))
        ctx.enqueue_copy(dst_buf=vh, src_buf=DeviceBuffer[KVT](ctx, vc_d.unsafe_ptr(), kvn, owning=False))
        ctx.synchronize()
        var head = List[UInt8]()
        var magic = String("BAROST01")
        for i in range(8):
            head.append(magic.as_bytes()[i])
        for v in [pos, 0, 0, kvn]:
            for b in range(8):
                head.append(UInt8((v >> (8 * b)) & 0xFF))
        for _ in range(32):
            head.append(0)
        for t in range(pos):
            for b in range(4):
                head.append(UInt8((prompt[t] >> (8 * b)) & 0xFF))
        with open(path, "w") as f:
            f.write_bytes(Span(head))
            f.write_bytes(Span[UInt8](unsafe_ptr=kh.unsafe_ptr().unsafe_bitcast[UInt8](), length=kvn * 4))
            f.write_bytes(Span[UInt8](unsafe_ptr=vh.unsafe_ptr().unsafe_bitcast[UInt8](), length=kvn * 4))
        print("state saved:", path, " pos", pos, " kv pages", ceildiv(pos, KVPAGE), " format BAROST01 (kv only)")


def main() raises:
    comptime if KVQ != "f32" and KVQ != "int8":
        raise Error("the spark profile supports f32 or int8 KV; built with unsupported BARO_KVQ=" + KVQ)

    comptime if not get_defined_int["BARO_CPU", 0]():
        comptime assert has_accelerator(), "GPU required (or build with -D BARO_CPU=1)"
    var vram_pool_pct = getenv("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT", "")
    if vram_pool_pct == "":
        vram_pool_pct = "10"
        _ = setenv("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT", vram_pool_pct)
        print("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT: unset (defaulting to 10%)")
    else:
        print("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT:", vram_pool_pct)

    var ctx = DeviceContext()
    var packdir = getenv("BARO_PACK", ".work/spark/pack-q8")
    var serve = getenv("BARO_SERVE", "0") == "1"

    var gguf_path = packdir + "/pack.bin"
    try:
        var gguf = GGUFModel(gguf_path)
        print("GGUF metadata parsed dynamically: arch=" + gguf.arch + " layers=" + String(gguf.akv_int("block_count", N_LAYERS)) + " embed=" + String(gguf.akv_int("embedding_length", H)))
    except e:
        print("GGUF dynamic parse status: using comptime profile defaults (" + String(e) + ")")

    # Item 2, briefs/2026-09-16-sampling-all-models-lane.md: dumps the last

    # decode step's target row (same buffer amar_argmax_part / amar_sample_row
    # read from), one profile-agnostic capture path for the distribution test.
    # Off by default, no effect on any existing path.
    var dump_logits_path = getenv("BARO_DUMP_LOGITS", "")
    var dump_layers_path = getenv("BARO_DUMP_LAYERS", "")
    var dump_layers = dump_layers_path != ""
    # BARO_ABLATE file: f32 lambda[2 * N_LAYERS] (attn, ffn per layer), then v[N_LAYERS * H].
    var ablate_path = getenv("BARO_ABLATE", "")
    var ablate = ablate_path != ""
    var abl_b = List[UInt8]()
    if ablate:
        abl_b = open(ablate_path, "r").read_bytes()
        if len(abl_b) != (2 * N_LAYERS + N_LAYERS * H) * 4:
            raise Error("BARO_ABLATE: wrong file size")
    var abl_lam = abl_b.unsafe_ptr().unsafe_bitcast[Float32]()
    var abl_prev = List[Float32](length=H, fill=0.0)
    # BARO_STEER file: f32[N_LAYERS * H], row L is added to X after layer L (llama.cpp --control-vector
    # position). Fused into the next RMSNorm launch, so it adds no launch and no host sync.
    var steer_path = getenv("BARO_STEER", "")
    # BARO_STEER_SERVE=1: allocate the buffer so serve requests can carry "steer_file" (+ "steer_layer").
    var steer = steer_path != "" or getenv("BARO_STEER_SERVE", "0") == "1"
    var steer_now = steer_path != ""
    var gain_l = List[Int]()  # per request: neuron gains L:N:k applied to the FFN activation before the down gemv
    var gain_n = List[Int]()
    var gain_k = List[Float32]()
    var sv_d = ctx.enqueue_create_buffer[f32](N_LAYERS * H if steer else 1)
    var sv_h = ctx.enqueue_create_host_buffer[f32](N_LAYERS * H if steer else 1)
    if steer_now:
        load_steer(ctx, sv_d, sv_h, steer_path, -1)
    var SV = TileTensor(sv_d, sv_l)
    var abl_h = ctx.enqueue_create_host_buffer[f32](H)
    var force = List[Int]()
    var force_path = getenv("BARO_FORCE", "")
    if force_path != "":
        force = read_prompt(force_path)

    var off = List[Int]()
    var kq = List[Int]()
    var wbuf = load_pack(ctx, packdir, off, kq)

    # Buffers allocated once (serve/PROTOCOL.md); a served request refills
    # toks_h/toks_d from position 0 and every kernel below only ever reads a
    # position it has itself just written this request, so no explicit KV
    # cache clear is needed between requests -- position 0 always overwrites
    # whatever an earlier request left at that slot.
    var toks_h = ctx.enqueue_create_host_buffer[DType.int32](TMAX)
    var toks_d = ctx.enqueue_create_buffer[DType.int32](TMAX)
    var tok1_h = ctx.enqueue_create_host_buffer[DType.int32](1)

    var x_d = ctx.enqueue_create_buffer[f32](H)
    var xb_d = ctx.enqueue_create_buffer[ADT](H)
    var semantic_hidden_h = ctx.enqueue_create_host_buffer[f32](H)
    var semantic_hidden_d = ctx.enqueue_create_buffer[f32](H)
    var qkv_d = ctx.enqueue_create_buffer[f32](QKV)
    var semantic_input = SemanticInput()
    var semantic_req_loaded = False  # the live semantic_input came from a request line, not from the boot paths
    var semantic_pins = DirectionalPins()
    semantic_pins.sender_digest = getenv("BARO_SEMANTIC_SENDER_SHA256", "")
    semantic_pins.receiver_digest = getenv("BARO_SEMANTIC_RECEIVER_SHA256", "")
    semantic_pins.binding_digest = getenv("BARO_SEMANTIC_BINDING_SHA256", "")
    semantic_pins.receiver_runtime = getenv("BARO_SEMANTIC_RECEIVER_RUNTIME_SHA256", "")
    semantic_pins.receiver_tokenizer = getenv("BARO_SEMANTIC_RECEIVER_TOKENIZER_SHA256", "")
    semantic_pins.sender_runtime = getenv("BARO_SEMANTIC_SENDER_RUNTIME_SHA256", "")
    semantic_pins.sender_tokenizer = getenv("BARO_SEMANTIC_SENDER_TOKENIZER_SHA256", "")
    semantic_pins.scope_digest = getenv("BARO_SEMANTIC_SCOPE_SHA256", "")
    semantic_pins.source_layer = Int(getenv("BARO_SEMANTIC_SOURCE_LAYER", "0"))
    semantic_pins.injection_site = Int(getenv("BARO_SEMANTIC_INJECTION_SITE", "0"))
    semantic_pins.task_domain = Int(getenv("BARO_SEMANTIC_TASK_DOMAIN", "0"))
    semantic_pins.normalization_id = Int(getenv("BARO_SEMANTIC_NORMALIZATION_ID", "0"))
    semantic_pins.adapter_version = Int(getenv("BARO_SEMANTIC_ADAPTER_VERSION", "1"))
    semantic_pins.scope_revision = UInt64(Int(getenv("BARO_SEMANTIC_SCOPE_REVISION", "0")))
    semantic_pins.authority_epoch = UInt64(Int(getenv("BARO_SEMANTIC_AUTHORITY_EPOCH", "0")))
    var semantic_manifest_path = getenv("BARO_SEMANTIC_MANIFEST", "")
    var semantic_values_path = getenv("BARO_SEMANTIC_INPUT", "")
    var semantic_positions = Int(getenv("BARO_SEMANTIC_POSITIONS", "8"))
    var semantic_now_s = Int64(Int(getenv("BARO_SEMANTIC_NOW_S", "0")))
    var semantic_dump_path = getenv("BARO_SEMANTIC_DUMP", "")
    var adapter_dir = getenv("BARO_SEMANTIC_ADAPTER_DIR", "")
    var persistent_adapter = Optional[LatentMLPAdapter](None)
    if adapter_dir != "":
        persistent_adapter = Optional(LatentMLPAdapter(adapter_dir, H, 256, H))
        print("semantic adapter: persistent non-linear MLP (rank 256) loaded from", adapter_dir)

    # Per-request semantic latent (serve mode) overrides env var config
    var use_per_request = False
    if semantic_manifest_path != "" or semantic_values_path != "":
        if semantic_manifest_path == "" or semantic_values_path == "":
            raise Error("BARO_SEMANTIC_MANIFEST and BARO_SEMANTIC_INPUT must be set together")
        semantic_input = load_semantic_input(semantic_manifest_path, semantic_values_path, H, semantic_positions, semantic_now_s, semantic_pins)
        if persistent_adapter:
            persistent_adapter.value().forward_batch_in_place(semantic_input.values, semantic_input.positions)
            print("semantic adapter: applied non-linear MLP (rank 256) in-place from", adapter_dir)
        print("semantic binding:", semantic_input.binding_id(), "positions:", semantic_input.positions, "dim:", semantic_input.dim)
    var kc_d = ctx.enqueue_create_buffer[KVT](KVPOOL)
    var vc_d = ctx.enqueue_create_buffer[KVT](KVPOOL)
    var kvtab_h = ctx.enqueue_create_host_buffer[DType.int32](TPAGES)
    for i in range(TPAGES):
        kvtab_h[i] = Int32(i)
    var kvtab_d = ctx.enqueue_create_buffer[DType.int32](TPAGES)
    ctx.enqueue_copy(dst_buf=kvtab_d, src_buf=kvtab_h)
    var aob_d = ctx.enqueue_create_buffer[ADT](QDIM)
    var gate_d = ctx.enqueue_create_buffer[f32](NQH)
    var p_g_d = ctx.enqueue_create_buffer[f32](FFN)
    var fgb_d = ctx.enqueue_create_buffer[ADT](FFN)
    var dummy_d = ctx.enqueue_create_buffer[ADT](1)
    var logits_d = ctx.enqueue_create_buffer[f32](VOCAB)
    var pred_d = ctx.enqueue_create_buffer[DType.int32](TMAX)
    var trace_n = TMAX * TRACE_STRIDE if TRACE_SUM else 1
    var trace_d = ctx.enqueue_create_buffer[DType.uint32](trace_n)
    var trace_h = ctx.enqueue_create_host_buffer[DType.uint32](trace_n)
    var trace_dir = getenv("BARO_TRACE_SUM_DIR", "")
    var rope_fuse = getenv("BARO_ROPE_FUSE", "1") == "1"
    var att_split_min = 0 if getenv("BARO_ATT_SPLIT", "0") == "1" else atol(getenv("BARO_ATT_SPLIT_T", "128")) # engine.mojo's knobs
    var att_split = att_split_min < TMAX
    var att_pg = ctx.enqueue_create_buffer[DType.float32](NQH * ATT_NSMAX * (HD + 2) if att_split else 1)
    var att_pgp = Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(att_pg.unsafe_ptr()))
    var pf_on = not TRACE_SUM and getenv("BARO_PREFILL_ROWS", "1") != "0" and not ablate
    var pf_arena = ctx.enqueue_create_buffer[DType.uint8](pf_arena_bytes() if pf_on else 1)
    ctx.synchronize()

    var Emb = wf(ctx, wbuf, off[0], VOCAB * H, emb_l)
    var X = TileTensor(x_d, x_l)
    var Xb = TileTensor(xb_d, xb_l)
    var Hf = TileTensor(semantic_hidden_d, hf_l)
    var Toks = TileTensor(toks_d, toks_l)
    var Pred = TileTensor(pred_d, toks_l)
    var Qkv1 = TileTensor(qkv_d, qkv1_l)
    var X1 = TileTensor(x_d, h1_l)
    var Dummy = TileTensor(dummy_d, dummy_l)
    var Q = sub_f32(ctx, qkv_d, 0, QDIM, q_l)
    var K = sub_f32(ctx, qkv_d, QDIM, KVDIM, kv_l)
    var V = sub_f32(ctx, qkv_d, QDIM + KVDIM, KVDIM, kv_l)
    var Kc = TileTensor(kc_d, cache_l)
    var Vc = TileTensor(vc_d, cache_l)
    var AoB = TileTensor(aob_d, aob_l)
    var AoB2 = TileTensor(aob_d, aob2_l)
    var G1 = TileTensor(p_g_d, ffn1_l)
    var Fgb1 = TileTensor(fgb_d, ffn1_l)
    var Gate = TileTensor(gate_d, gate_l)
    var Fgb = TileTensor(fgb_d, fgb_l)
    var Logits1 = TileTensor(logits_d, v1_l)
    var LogitsSample = TileTensor(logits_d, samp_x_l)
    var amv_d = ctx.enqueue_create_buffer[f32](AM_NB)
    var ami_d = ctx.enqueue_create_buffer[DType.int32](AM_NB)
    var Amv = TileTensor(amv_d, amv_l)
    var Ami = TileTensor(ami_d, amv_l)
    var stok_d = ctx.enqueue_create_buffer[DType.int32](1)
    var sprob_d = ctx.enqueue_create_buffer[f32](1)
    var Stok = TileTensor(stok_d, samp_o_l)
    var Sprob = TileTensor(sprob_d, samp_o_l)
    var sprob_h = ctx.enqueue_create_host_buffer[f32](1)
    var topn_ids_d = ctx.enqueue_create_buffer[DType.int32](NTOPLP)
    var topn_probs_d = ctx.enqueue_create_buffer[f32](NTOPLP)
    var topn_ids_h = ctx.enqueue_create_host_buffer[DType.int32](NTOPLP)
    var topn_probs_h = ctx.enqueue_create_host_buffer[f32](NTOPLP)
    var gmask_h = ctx.enqueue_create_host_buffer[DType.uint64]((VOCAB + 63) // 64)
    var gmask_d = ctx.enqueue_create_buffer[DType.uint64]((VOCAB + 63) // 64)
    var row_h = ctx.enqueue_create_host_buffer[f32](VOCAB)
    var dump_dir = getenv("BARO_DUMP_LOGITS_DIR", "")
    var OutNorm = wf(ctx, wbuf, off[1 + LSTRIDE * N_LAYERS], H, h_l)
    var ffp = rope_ff(wbuf, off)
    var grt: Optional[GrammarRuntime] = None

    if serve:
        print("{\"ready\":true,\"tmax\":" + String(TMAX) + ",\"mrows\":1,\"kmax\":0,\"spec_k\":0,\"pack\":\"" + packdir + "\"}")

    # No cancel support this milestone (M4, briefs/2026-09-15-wiring-lane.md):
    # spark has no draft head and no spec decode, so a request always runs to
    # completion quickly; {"cancel":ID} lines are not sent by a client that
    # never streams past this engine's own doneness. A future round wanting
    # cancel reuses serve_proto.cancel_pending exactly as engine.mojo does.
    var req_id = 0
    while True:
        var prompt: List[Int]
        var gen_n = atol(getenv("BARO_GEN", "64"))
        var stop_seqs = List[List[Int]]()
        # Item 2, briefs/2026-09-16-sampling-all-models-lane.md: declared
        # outside the branch (engine.mojo's own one-shot path does the same)
        # so the decode loop below can read sample.temperature regardless of
        # mode; one-shot stays the default (temperature 0, greedy), matching
        # today's behaviour exactly.
        var sample = default_sample_params()
        var grammar: Optional[Matcher] = None
        var grammar_pending_think = False
        var grammar_think_buf = List[UInt8]()
        var grammar_stop = False
        var grammar_masked_draws = 0
        var grammar_accepted = 0
        var grammar_mask = Bitset(1)
        var sp_state_save = String("")
        var sp_semantic_manifest_path = String("")
        var sp_semantic_values_path = String("")
        var sp_semantic_positions = 0
        var sp_semantic_now_s = Int64(0)
        if serve:
            var line_in = read_line(0)
            if not line_in:
                break
            var idle_cancel = is_cancel_line(line_in.value())
            if idle_cancel[0]:
                print(idle_cancel_note(idle_cancel[1]))
                continue
            var opk = json_key(line_in.value(), "op")
            var opn = String("")
            if opk >= 0 and json_str(line_in.value(), opk, opn) and opn == "slens":
                var oid = 0
                var oi = json_key(line_in.value(), "id")
                if oi >= 0:
                    _ = json_int(line_in.value(), oi, oid)
                try:
                    slens_op(ctx, wbuf, off, kq, Emb, Toks, toks_h, toks_d, line_in.value(), oid)
                    print("{\"id\":" + String(oid) + ",\"done\":true}")
                except e:
                    print(err_line(oid, String(e)))
                continue
            prompt = List[Int]()
            var req_n = 0
            var req_spec = False
            var req_has_spec = False
            var ckpt_hints = List[Int]()
            var sp_state_load = String("")
            var sp_state_prefix = False
            var perr = parse_request(line_in.value(), req_id, prompt, req_n, req_spec, req_has_spec, stop_seqs, ckpt_hints, sample, sp_state_save, sp_state_load, sp_state_prefix, sp_semantic_manifest_path, sp_semantic_values_path, sp_semantic_positions, sp_semantic_now_s)
            if semantic_req_loaded or (semantic_manifest_path == "" and semantic_values_path == ""):
                # Per-request transfers must not poison the next plain request: a per-request input lives one
                # request (B-M14), and with boot paths set the boot input is loaded again from them.
                semantic_req_loaded = False
                semantic_input = SemanticInput()
                if semantic_manifest_path != "" or semantic_values_path != "":
                    semantic_input = load_semantic_input(semantic_manifest_path, semantic_values_path, H, semantic_positions, semantic_now_s, semantic_pins)
                    if persistent_adapter:
                        persistent_adapter.value().forward_batch_in_place(semantic_input.values, semantic_input.positions)
            if perr == "" and (sp_semantic_manifest_path != "" or sp_semantic_values_path != ""):
                if sp_semantic_manifest_path == "" or sp_semantic_values_path == "":
                    perr = "semantic_manifest and semantic_values must be set together"
                else:
                    var req_semantic_positions = sp_semantic_positions if sp_semantic_positions > 0 else semantic_positions
                    var req_semantic_now_s = sp_semantic_now_s if sp_semantic_now_s > 0 else semantic_now_s
                    semantic_input = load_semantic_input(sp_semantic_manifest_path, sp_semantic_values_path, H, req_semantic_positions, req_semantic_now_s, semantic_pins)
                    semantic_req_loaded = True
                    if persistent_adapter:
                        persistent_adapter.value().forward_batch_in_place(semantic_input.values, semantic_input.positions)
                        print("semantic adapter: applied non-linear MLP (rank 256) in-place per-request")
            steer_now = steer_path != ""
            if perr == "" and steer:
                var sf = String("")
                var sk = json_key(line_in.value(), "steer_file")
                var bank = String("")
                var bk = json_key(line_in.value(), "steer_bank")
                if bk >= 0 and json_str(line_in.value(), bk, bank):
                    var bl = -1
                    var br = -1
                    var bs: Float64 = 1.0
                    var k1 = json_key(line_in.value(), "steer_layer")
                    var k2 = json_key(line_in.value(), "steer_row")
                    var k3 = json_key(line_in.value(), "steer_scale")
                    if k1 < 0 or k2 < 0 or not json_int(line_in.value(), k1, bl) or not json_int(line_in.value(), k2, br):
                        perr = "steer_bank needs integer steer_layer and steer_row"
                    elif k3 >= 0 and not json_float(line_in.value(), k3, bs):
                        perr = "steer_scale must be a number"
                    if perr == "":
                        try:
                            load_steer_row(ctx, sv_d, sv_h, bank, bl, br, Float32(bs))
                            steer_now = True
                        except e:
                            perr = String(e)
                elif sk >= 0 and json_str(line_in.value(), sk, sf):
                    var sl = -1
                    var sk2 = json_key(line_in.value(), "steer_layer")
                    if sk2 >= 0 and not json_int(line_in.value(), sk2, sl):
                        perr = "steer_layer must be an integer"
                    if perr == "":
                        try:
                            load_steer(ctx, sv_d, sv_h, sf, sl)
                            steer_now = True
                        except e:
                            perr = String(e)
            gain_l = List[Int]()
            gain_n = List[Int]()
            gain_k = List[Float32]()
            var gk = json_key(line_in.value(), "gain")
            var gs = String("")
            if perr == "" and gk >= 0 and json_str(line_in.value(), gk, gs):
                comptime if ADT != f32:
                    perr = "gain needs f32 activations (build without BARO_ACT_BF16)"
                if not steer:
                    perr = "gain needs BARO_STEER_SERVE=1"
                try:
                    for u in gs.split(","):
                        var q = u.split(":")
                        if len(q) != 3 or Int(q[0]) < 0 or Int(q[0]) >= N_LAYERS or Int(q[1]) < 0 or Int(q[1]) >= FFN:
                            raise Error("gain wants L:N:k with L < " + String(N_LAYERS) + ", N < " + String(FFN))
                        gain_l.append(Int(q[0]))
                        gain_n.append(Int(q[1]))
                        gain_k.append(Float32(Float64(q[2])))
                except e:
                    perr = String(e)
            if perr == "" and len(prompt) < 1:
                perr = "empty prompt"
            if perr == "" and req_n < 1:
                perr = "n must be >= 1"
            if perr == "" and len(prompt) + req_n > TMAX:
                perr = "prompt+n exceeds TMAX " + String(TMAX)
            if perr == "" and semantic_input.active and len(prompt) < semantic_input.positions:
                perr = "semantic binding needs at least " + String(semantic_input.positions) + " prompt positions"
            if perr == "" and semantic_input.active and semantic_input.consumed > 0:
                perr = "semantic binding was already consumed; start a new engine process for a new transfer"
            if perr == "" and sp_state_load != "":
                perr = "state_load is not wired for this engine (spark); its state files are export-only (tools/state-to-llama-slot)"
            if perr == "" and sp_state_save != "" and len(prompt) < 2:
                perr = "state_save needs a prompt of at least 2 tokens"
            # Items 3-4, briefs/2026-09-16-sampling-all-models-lane.md: spark
            # parses presence_penalty/frequency_penalty/top_logprobs (M5/C3)
            # but has never acted on them -- silently ignoring a parameter
            # the caller asked for is the inert-parameter defect P1 forbids,
            # so refuse loudly rather than serve a request that looks
            # penalized/logprob'd and isn't, until this is wired here too.
            if perr == "" and (sample.presence_penalty != 0 or sample.frequency_penalty != 0):
                perr = "presence_penalty/frequency_penalty are not yet wired for this engine (spark); only the dense/MoE engine (serve/engine.mojo) supports them"
            if perr == "" and sample.embed == 1:
                perr = "embed is not wired for this engine (spark); only the dense engine (serve/engine.mojo) emits it"
            if perr == "" and (sample.hidden == 1 or sample.logits_topk > 0):
                perr = "hidden/logits_topk are not wired for this engine (spark); use the dense engine (serve/engine.mojo)"
            var schema_raw = parse_schema_field(line_in.value()) if perr == "" else String("")
            if perr == "" and schema_raw != "":
                if not grt:
                    try:
                        grt = Optional(GrammarRuntime(packdir))
                    except e:
                        perr = "response_format: grammar runtime failed to load: " + String(e)
                if perr == "" and grt.value().vocab_size != VOCAB:
                    perr = "response_format: grammar vocab_size " + String(grt.value().vocab_size) + " != engine VOCAB " + String(VOCAB)
                if perr == "":
                    try:
                        grammar = Optional(compile_schema_matcher(grt.value(), schema_raw))
                    except e:
                        perr = "response_format schema: " + String(e)
                    grammar_pending_think = parse_reasoning_field(line_in.value())
                    grammar_mask = Bitset(VOCAB)
            if perr != "":
                print(err_line(req_id, perr))
                continue
            gen_n = req_n
            # Per-request teacher forcing, the same "force":[ids] field
            # serve/engine.mojo reads; absent, the request inherits BARO_FORCE.
            force = read_prompt(force_path) if force_path != "" else List[Int]()
            var fi2 = json_key(line_in.value(), "force")
            if fi2 >= 0:
                var fline = line_in.value().copy()
                var fb = fline.as_bytes()
                if fi2 < len(fb) and fb[fi2] == 91:
                    fi2 += 1
                    while True:
                        while fi2 < len(fb) and (fb[fi2] == 32 or fb[fi2] == 44):
                            fi2 += 1
                        if fi2 >= len(fb) or fb[fi2] == 93:
                            break
                        var fv = 0
                        if not json_int(fline, fi2, fv):
                            break
                        force.append(fv)
        else:
            var text_path = getenv("BARO_PROMPT_TEXT", "")
            var chat_path = getenv("BARO_CHAT", "")
            if text_path != "" or chat_path != "":
                var gguf = getenv("BARO_GGUF", "")
                if gguf == "":
                    raise Error("BARO_PROMPT_TEXT / BARO_CHAT need BARO_GGUF (tokenizer + template source)")
                var t0 = perf_counter_ns()
                var tok = Tokenizer(gguf)
                var text: String
                if chat_path != "":
                    var case_json: String
                    with open(chat_path, "r") as f:
                        case_json = f.read()
                    text = render_chat(tok.chat_template, case_json, tok.token_str(tok.bos_id), tok.token_str(tok.eos_id), tok.token_str(tok.pad_id))
                    print("chat template rendered:", text.byte_length(), "bytes (mojo-minja)")
                else:
                    with open(text_path, "r") as f:
                        text = f.read()
                prompt = tok.encode(text)
                print("tokenized", len(prompt), "ids in", Float64(perf_counter_ns() - t0) / 1e9, "s (mojo tokenizer,", tok.pre, ")")
                var ps = String()
                for i in range(len(prompt)):
                    ps += String(prompt[i]) + " "
            else:
                prompt = read_prompt(getenv("BARO_PROMPT", packdir + "/prompt-tokens.txt"))
        var n_sem = semantic_input.positions if semantic_input.active else 0
        var n_prompt = len(prompt) + n_sem
        var n_total = n_prompt + gen_n
        if n_total > TMAX:
            raise Error("prompt + gen exceeds TMAX")
        print("prompt tokens:", n_prompt, "( sem:", n_sem, "text:", len(prompt), ") gen:", gen_n)

        for i in range(TMAX):
            if i < n_sem:
                toks_h[i] = Int32(0)
            elif i < n_prompt:
                toks_h[i] = Int32(prompt[i - n_sem])
            else:
                toks_h[i] = Int32(0)
        ctx.enqueue_copy(dst_buf=toks_d, src_buf=toks_h)
        comptime if TRACE_SUM:
            for i in range(trace_n):
                trace_h[i] = 0
            ctx.enqueue_copy(dst_buf=trace_d, src_buf=trace_h)

        var t_gen_start: Int = 0
        var generated_ids = List[Int]()
        var finish = String("length")
        ctx.synchronize()
        var t_pf_start = perf_counter_ns()
        var start = 0
        if n_prompt > 3 and not semantic_input.active and pf_on and len(gain_l) == 0:
            # the last prompt row stays token-by-token: it launches the decode kernels once outside the decode timer
            # (first launches after a batched prefill cost ~5 ms, measured 2026-10-26)
            prefill_rows(ctx, wbuf, off, kq, Emb, Toks, Kc, Vc, Int(kvtab_d.unsafe_ptr()), n_prompt - 2, pf_arena, steer_now, SV)
            start = n_prompt - 2
        for pos in range(start, n_total - 1):
            if pos == n_prompt - 1:
                ctx.synchronize()
                t_gen_start = perf_counter_ns()
                print("prefill_s:", Float64(t_gen_start - t_pf_start) / 1e9, " prefill rows:", n_prompt - 1, " chunk: 0")
            if semantic_input.active and pos < n_sem:
                for i in range(H):
                    semantic_hidden_h[i] = semantic_input.values[pos * H + i]
                ctx.enqueue_copy(
                    dst_buf=DeviceBuffer[f32](ctx, x_d.unsafe_ptr(), H, owning=False),
                    src_buf=semantic_hidden_h,
                )
                ctx.synchronize()
                semantic_input.mark_consumed()
            else:
                ctx.enqueue_function[k_emb](Emb, X, Toks, Int32(pos), Int32(H), grid_dim=(ceildiv(H, 256), 1), block_dim=256)
            for i in range(N_LAYERS):
                var e = 1 + LSTRIDE * i


                var swa = (i % SWA_PERIOD) != SWA_FULL_PHASE
                if ablate:
                    ctx.enqueue_copy(dst_buf=abl_h, src_buf=DeviceBuffer[f32](ctx, x_d.unsafe_ptr(), H, owning=False))
                    ctx.synchronize()
                    for j in range(H):
                        abl_prev[j] = abl_h.unsafe_ptr()[j]
                var AttnNorm = wf(ctx, wbuf, off[e], H, h_l)
                var FfnNorm = wf(ctx, wbuf, off[e + OFF_FFN_NORM], H, h_l)
                if steer_now and i > 0:
                    ctx.enqueue_function[k_rms_steer](X, AttnNorm, Xb, Int32(H), NORM_EPS, SV, Int32((i - 1) * H), grid_dim=1, block_dim=256)
                else:
                    ctx.enqueue_function[k_rms](X, AttnNorm, Xb, Int32(H), NORM_EPS, grid_dim=1, block_dim=256)
                gemv[0, S=QKV_S](ctx, Xb, wbuf, off, kq, e + 1, QKV, H, q_qkv, s_qkv, Qkv1, Dummy) # split-K: QKV is narrow
                comptime if QKV_BIAS:
                    var QkvBias = wf(ctx, wbuf, off[e + OFF_QKV_BIAS], QKV, qkv1_l)
                    ctx.enqueue_function[k_bias](Qkv1, QkvBias, Int32(QKV), grid_dim=ceildiv(QKV, 256), block_dim=256)
                comptime if TRACE_SUM:
                    ctx.enqueue_function[amar_trace_sum](qkv_d.unsafe_ptr(), trace_d.unsafe_ptr(), Int32(QKV), Int32(pos * TRACE_STRIDE + 3 * i), grid_dim=1, block_dim=1)
                if swa:
                    if rope_fuse:
                        ctx.enqueue_function[k_rqkv_swa](Q, Kc, Vc, K, V, kvtab_d.unsafe_ptr(), Int32(pos), BASE_SWA, Int32(i), ffp, grid_dim=NQH + 2 * NKVH, block_dim=HD)
                    else:
                        ctx.enqueue_function[k_rope_swa](Q, Int32(pos), Int32(NQH), BASE_SWA, ffp, grid_dim=(NQH, 1), block_dim=NROT_SWA // 2)
                        ctx.enqueue_function[k_kv_swa](Kc, Vc, K, V, kvtab_d.unsafe_ptr(), Int32(pos), BASE_SWA, Int32(i), ffp, grid_dim=(NKVH, 2), block_dim=HD)
                else:
                    if rope_fuse:
                        ctx.enqueue_function[k_rqkv_full](Q, Kc, Vc, K, V, kvtab_d.unsafe_ptr(), Int32(pos), BASE_FULL, Int32(i), ffp, grid_dim=NQH + 2 * NKVH, block_dim=HD)
                    else:
                        ctx.enqueue_function[k_rope_full](Q, Int32(pos), Int32(NQH), BASE_FULL, ffp, grid_dim=(NQH, 1), block_dim=NROT_FULL // 2)
                        ctx.enqueue_function[k_kv_full](Kc, Vc, K, V, kvtab_d.unsafe_ptr(), Int32(pos), BASE_FULL, Int32(i), ffp, grid_dim=(NKVH, 2), block_dim=HD)
                comptime if HAS_GATE:
                    gemv[0](ctx, Xb, wbuf, off, kq, e + OFF_GATE, NQH, H, q_gate, s_gate, Gate, Dummy)
                var win = SWA_WIN if swa else 0
                if att_split and pos + 1 >= att_split_min:
                    var span = att_span_len(pos + 1, win)
                    var ns = ceildiv(min(pos + 1, win) if win > 0 else pos + 1, span)
                    ctx.enqueue_function[k_att_sp](Q, Kc, Vc, att_pgp, kvtab_d.unsafe_ptr(), Int32(pos + 1), Int32(win), Int32(span), ATTN_SCALE, Int32(i), grid_dim=(NQH, ns), block_dim=HD)
                    ctx.enqueue_function[k_att_mg](att_pgp, Gate, AoB2, Int32(ns), grid_dim=NQH, block_dim=HD)
                else:
                    ctx.enqueue_function[k_att](Q, Kc, Vc, Gate, AoB2, kvtab_d.unsafe_ptr(), Int32(pos + 1), Int32(win), ATTN_SCALE, Int32(i), grid_dim=(NQH, 1), block_dim=HD)
                gemv[1, S=O_S](ctx, AoB, wbuf, off, kq, e + OFF_O, H, QDIM, q_o, s_o, X1, Dummy)
                if ablate:
                    ablate_project(ctx, x_d, abl_h, abl_prev, abl_lam[2 * i], abl_lam + 2 * N_LAYERS + i * H)
                comptime if TRACE_SUM:
                    ctx.enqueue_function[amar_trace_sum](x_d.unsafe_ptr(), trace_d.unsafe_ptr(), Int32(H), Int32(pos * TRACE_STRIDE + 3 * i + 1), grid_dim=1, block_dim=1)
                ctx.enqueue_function[k_rms](X, FfnNorm, Xb, Int32(H), NORM_EPS, grid_dim=1, block_dim=256)
                if not gateup(ctx, Xb, wbuf, off, kq, e + OFF_FFN_GATE, e + OFF_FFN_UP, FFN, H, Fgb1):
                    gemv[0](ctx, Xb, wbuf, off, kq, e + OFF_FFN_GATE, FFN, H, q_ffn, s_ffn, G1, Dummy)
                    gemv[2 if ACTIVATION_GELU else 3](ctx, Xb, wbuf, off, kq, e + OFF_FFN_UP, FFN, H, q_ffn, s_ffn, G1, Fgb1)
                comptime if ADT == f32:
                    for gi in range(len(gain_l)):
                        if gain_l[gi] == i:
                            ctx.enqueue_function[amar_scale_one](fgb_d.unsafe_ptr(), Int32(gain_n[gi]), gain_k[gi], grid_dim=1, block_dim=1)
                gemv[1, S=DOWN_S](ctx, Fgb, wbuf, off, kq, e + OFF_DOWN, H, FFN, q_down, s_down, X1, Dummy)
                if ablate:
                    ablate_project(ctx, x_d, abl_h, abl_prev, abl_lam[2 * i + 1], abl_lam + 2 * N_LAYERS + i * H)
                comptime if TRACE_SUM:
                    ctx.enqueue_function[amar_trace_sum](x_d.unsafe_ptr(), trace_d.unsafe_ptr(), Int32(H), Int32(pos * TRACE_STRIDE + 3 * i + 2), grid_dim=1, block_dim=1)
                if dump_layers and pos == n_prompt - 1:
                    ctx.enqueue_copy(dst_buf=semantic_hidden_h, src_buf=DeviceBuffer[f32](ctx, x_d.unsafe_ptr(), H, owning=False))
                    ctx.synchronize()
                    with open(dump_layers_path, "a") as f:
                        f.write_bytes(Span[UInt8](unsafe_ptr=semantic_hidden_h.unsafe_ptr().unsafe_bitcast[UInt8](), length=H * 4))
            if pos >= n_prompt - 1:
                if steer_now:
                    ctx.enqueue_function[k_rms_steer](X, OutNorm, Xb, Int32(H), NORM_EPS, SV, Int32((N_LAYERS - 1) * H), grid_dim=1, block_dim=256)
                else:
                    ctx.enqueue_function[k_rms](X, OutNorm, Xb, Int32(H), NORM_EPS, grid_dim=1, block_dim=256)
                if semantic_dump_path != "":
                    var append_mode = getenv("BARO_SEMANTIC_DUMP_APPEND", "0") == "1"
                    if (append_mode and pos < n_prompt) or (not append_mode and pos == n_total - 2):
                        ctx.enqueue_function[k_rms_f32](X, OutNorm, Hf, Int32(H), NORM_EPS, grid_dim=1, block_dim=256)
                        ctx.enqueue_copy(
                            dst_buf=semantic_hidden_h,
                            src_buf=DeviceBuffer[f32](ctx, semantic_hidden_d.unsafe_ptr(), H, owning=False),
                        )
                        ctx.synchronize()
                        var open_flag = "a" if (append_mode and pos > 0) else "w"
                        with open(semantic_dump_path, open_flag) as f:
                            f.write_bytes(Span[UInt8](unsafe_ptr=semantic_hidden_h.unsafe_ptr().unsafe_bitcast[UInt8](), length=H * 4))
                        print("semantic extract: pos=" + String(pos) + " dim=" + String(H) + " path=" + semantic_dump_path)
                gemv[0, S=HEAD_S](ctx, Xb, wbuf, off, kq, 2 + LSTRIDE * N_LAYERS, VOCAB, H, q_out, s_out, Logits1, Dummy)
                comptime if TRACE_SUM:
                    ctx.enqueue_function[amar_trace_sum](logits_d.unsafe_ptr(), trace_d.unsafe_ptr(), Int32(VOCAB), Int32(pos * TRACE_STRIDE + 3 * N_LAYERS), grid_dim=1, block_dim=1)
                if dump_logits_path != "" and pos == n_total - 2:
                    var lg_h = ctx.enqueue_create_host_buffer[f32](VOCAB)
                    ctx.enqueue_copy(
                        dst_buf=lg_h,
                        src_buf=DeviceBuffer[f32](ctx, logits_d.unsafe_ptr(), VOCAB, owning=False),
                    )
                    ctx.synchronize()
                    with open(dump_logits_path, "w") as f:
                        var p = lg_h.unsafe_ptr().unsafe_bitcast[UInt8]()
                        f.write_bytes(Span[UInt8](unsafe_ptr=p, length=VOCAB * 4))
                    print("dumped final logits row to", dump_logits_path)
                # Item 2, briefs/2026-09-16-sampling-all-models-lane.md: at
                # temperature > 0 amar_sample_row replaces the argmax pair;
                # at 0 (including one-shot, which never sets sample) the
                # path below is untouched, byte for byte.
                var want_lp = serve and sample.top_logprobs > 0
                if want_lp and dump_dir != "":
                    ctx.enqueue_copy(dst_buf=row_h, src_buf=DeviceBuffer[f32](ctx, logits_d.unsafe_ptr(), VOCAB, owning=False))
                    ctx.synchronize()
                    with open(dump_dir + "/row-" + String(pos + 1 - n_prompt) + ".bin", "w") as f:
                        f.write_bytes(Span[UInt8](unsafe_ptr=row_h.unsafe_ptr().unsafe_bitcast[UInt8](), length=VOCAB * 4))
                if want_lp:
                    ctx.enqueue_function[k_topn](
                        LogitsSample, TileTensor(topn_ids_d, topn_l), TileTensor(topn_probs_d, topn_l),
                        Int32(VOCAB), Int32(min(sample.top_logprobs, NTOPLP)),
                        Float32(sample.temperature), Int32(sample.top_k), Float32(sample.top_p), Float32(sample.min_p),
                        grid_dim=1, block_dim=SAMP_THREADS,
                    )
                    ctx.enqueue_copy(dst_buf=topn_ids_h, src_buf=topn_ids_d)
                    ctx.enqueue_copy(dst_buf=topn_probs_h, src_buf=topn_probs_d)
                var grammar_here = grammar.__bool__() and pos + 1 >= n_prompt and not grammar_pending_think
                if grammar_here:
                    ref mm = grammar.value()
                    mm.fill_mask(grammar_mask)
                    for i in range(len(grammar_mask.words)):
                        gmask_h[i] = grammar_mask.words[i]
                    ctx.enqueue_copy(dst_buf=gmask_d, src_buf=gmask_h)
                    grammar_masked_draws += 1
                    ctx.enqueue_function[k_sample_masked](
                        LogitsSample, Stok, Sprob, Int32(VOCAB), Float32(sample.temperature), Int32(sample.top_k),
                        Float32(sample.top_p), Float32(sample.min_p), sample.seed, UInt64(pos),
                        gmask_d.unsafe_ptr(), Int32((VOCAB + 63) // 64), grid_dim=1, block_dim=SAMP_THREADS,
                    )
                    ctx.enqueue_function[k_tokcp](Stok, Toks, Int32(0), Int32(pos + 1), Int32(1), grid_dim=1, block_dim=32)
                elif sample.temperature > 0:
                    ctx.enqueue_function[k_sample](
                        LogitsSample, Stok, Sprob, Int32(VOCAB), Float32(sample.temperature), Int32(sample.top_k),
                        Float32(sample.top_p), Float32(sample.min_p), sample.seed, UInt64(pos),
                        grid_dim=1, block_dim=SAMP_THREADS,
                    )
                    ctx.enqueue_function[k_tokcp](Stok, Toks, Int32(0), Int32(pos + 1), Int32(1), grid_dim=1, block_dim=32)
                else:
                    ctx.enqueue_function[k_argmax](Logits1, Amv, Ami, Int32(VOCAB), grid_dim=AM_NB, block_dim=256)
                    var fi = pos + 1 - n_prompt
                    var forced = Int32(force[fi]) if fi < len(force) else Int32(-1)
                    ctx.enqueue_function[k_argmax_final](Amv, Ami, Toks, Pred, Int32(pos + 1), forced, grid_dim=1, block_dim=32)
                if serve:
                    ctx.enqueue_copy(dst_buf=tok1_h, src_buf=DeviceBuffer[DType.int32](ctx, toks_d.unsafe_ptr().unsafe_offset(pos + 1), 1, owning=False))
                    ctx.synchronize()
                    var new_tok = Int(tok1_h[0])
                    if grammar.__bool__():
                        ref mm = grammar.value()
                        if new_tok < 0:
                            grammar_stop = True
                        elif grammar_pending_think:
                            if reasoning_boundary_observe(grammar_think_buf, mm.vocab[].token_bytes[new_tok]):
                                grammar_pending_think = False
                        else:
                            if mm.accept(new_tok):
                                grammar_accepted += 1
                            if mm.is_terminated():
                                grammar_stop = True
                    if grammar_stop and new_tok < 0:
                        finish = "stop"
                        break
                    if want_lp:
                        var lp = 0.0
                        if sample.temperature > 0:
                            ctx.enqueue_copy(dst_buf=sprob_h, src_buf=sprob_d)
                            ctx.synchronize()
                            lp = log(Float64(sprob_h[0])) if sprob_h[0] > 0 else -1e30
                        var tl = String("{\"id\":") + String(req_id) + ",\"tok\":" + String(new_tok) + ",\"logprob\":" + String(lp) + ",\"top_logprobs\":["
                        for i in range(min(sample.top_logprobs, NTOPLP)):
                            var tid = Int(topn_ids_h[i])
                            if tid < 0:
                                break
                            if i > 0:
                                tl += ","
                            var pr = topn_probs_h[i]
                            tl += "{\"id\":" + String(tid) + ",\"logprob\":" + String(log(Float64(pr)) if pr > 0 else -1e30) + "}"
                        tl += "]}"
                        print(tl)
                    else:
                        print(tok_line(req_id, new_tok))
                    generated_ids.append(new_tok)
                    # Checked once per generated token (spark decodes m=1 at a
                    # time, unlike engine.mojo's per-window check): the tail
                    # of what has been generated so far against every stop
                    # sequence; a match ends generation right here.
                    for seq in stop_seqs:
                        if len(seq) > 0 and len(seq) <= len(generated_ids):
                            var matched = True
                            for k in range(len(seq)):
                                if generated_ids[len(generated_ids) - len(seq) + k] != seq[k]:
                                    matched = False
                                    break
                            if matched:
                                finish = "stop"
                    if finish == "stop":
                        break
                    if grammar_stop:
                        finish = "stop"
                        break
        ctx.synchronize()
        var dt = Float64(perf_counter_ns() - t_gen_start) / 1e9
        var n_gen = len(generated_ids) if serve else gen_n
        ctx.enqueue_copy(dst_buf=toks_h, src_buf=toks_d)
        ctx.synchronize()
        var s = String("")
        for i in range(n_prompt, n_prompt + n_gen):
            s += String(Int(toks_h[i])) + " "
        print("generated:", s)
        comptime if TRACE_SUM:
            if trace_dir != "":
                ctx.enqueue_copy(dst_buf=trace_h, src_buf=trace_d)
                ctx.synchronize()
                with open(trace_dir + "/trace-" + String(req_id) + ".bin", "w") as f:
                    f.write_bytes(Span[UInt8](unsafe_ptr=trace_h.unsafe_ptr().unsafe_bitcast[UInt8](), length=(n_prompt + n_gen - 1) * TRACE_STRIDE * 4))
                print("trace sums:", trace_dir + "/trace-" + String(req_id) + ".bin", " positions", n_prompt + n_gen - 1, " stride", TRACE_STRIDE)
        if len(force) > 0:
            ctx.enqueue_copy(dst_buf=toks_h, src_buf=pred_d)
            ctx.synchronize()
            var ps = String("")
            var agree = 0
            for i in range(n_prompt, n_prompt + n_gen):
                ps += String(Int(toks_h[i])) + " "
                if i - n_prompt < len(force) and Int(toks_h[i]) == force[i - n_prompt]:
                    agree += 1
            print("predicted:", ps)
            print("forced agreement:", agree, "/", min(n_gen, len(force)))
        if semantic_input.active:
            print("semantic consume_count=" + String(semantic_input.consumed) + " binding=" + semantic_input.binding_id())
        print("tok/s_gen:", Float64(n_gen) / dt, "(", n_gen, "steps,", dt, "s )")
        if grammar.__bool__():
            # Gate 4 (bench/grammar-protocol.md), the dense engine's receipt line: every masked draw must be accepted.
            print("grammar masked draws:", grammar_masked_draws, " accepted:", grammar_accepted, " terminated:", grammar_stop,
                  "" if grammar_masked_draws == grammar_accepted else " MISMATCH: masked draws != accepted")
        if sp_state_save != "":
            save_kv_state(ctx, kc_d, vc_d, sp_state_save, prompt, n_prompt - 1)
        if serve:
            var tok_s = Float64(n_gen - 1) / dt if n_gen > 1 else 0.0
            print(
                "{\"id\":" + String(req_id) + ",\"done\":true,\"n\":" + String(n_gen)
                + ",\"prefill_s\":" + String(Float64(t_gen_start - t_pf_start) / 1e9)
                + ",\"decode_s\":" + String(dt) + ",\"tok_s\":" + String(tok_s)
                + ",\"finish\":\"" + finish + "\"}"
            )
        else:
            break
