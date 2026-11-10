# E8 HARNESS -- 5-arm dual-engine evaluator (exchange/e8-lane-plan-2026-09-09.md,
# item HARNESS). Two co-resident engine instances (A = producer, B = receiver)
# run each bench/data/e8_tasks.json item under one or more of:
#   0        B answers the prompt alone, greedy.
#   T        A emits <=300 tokens of CoT text; B answers prompt+text.
#   L8-raw   A feeds 8 raw h_L vectors back to itself and to B (bench_hidden_dtype
#            step_latent_raw, unmodified); B ingests them as a prefix, answers.
#   L8-soft  same, each vector passed through serve/realign.mojo first.
#   L32-soft same, 32 steps.
#   KV       A decodes as in T, stopping at T's trim point; its KV pages and SSM
#            checkpoint cross to B through sealed memfds (serve/latent.mojo); B
#            restores them and answers without re-prefilling (LatentOS E12).
# Soft arms call serve/realign.mojo (round 2 implementation, merged) which
# computes expected embedding e = softmax(logits) @ W_emb from b.x_d; errors
# are caught per-arm and recorded, never silently downgraded to the raw vector.
#
# Writes results/e8/<out-prefix>.raw.json: schema validity (json tasks) is
# computed here via grammar/'s Automaton/Matcher on B's decoded answer text;
# arithmetic extraction and JSON exact-match live in bench/e8_score.py, which
# turns the raw file into the final <out-prefix>.json / .md (bench/latent-handoff.sh).
from std.memory import ArcPointer
from std.os import getenv
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer

from registry import *
from window import *
from harness import load_pack, alloc_bufs, Pack
from realign import realign_expected_embedding, final_norm_hidden
from tokenizer import Tokenizer
from prefix import Chain, prefix_hash
from latent import mint_kv_latent, ingest_kv_latent, mint_chain_slot, ingest_into_chain
from e13_projector import load_projector, apply_projector_k
from serve_proto import default_sample_params
from identity import EngineIdentity, compute_engine_identity
import latentos.proto as proto
import latentos.sys as latentos_sys

from grammar.automaton import Automaton
from grammar.json_value import parse_json_file, parse_json_bytes, JSONDoc, JSONValue, JKindNull, JKindBool, JKindString, JKindNumber, JKindArray, JKindObject
from grammar.json_schema import compile_root_schema, str_bytes
from grammar.vocab import Vocab, load_vocab
from grammar.trie import TokenTrie, TrieNode, build_trie
from grammar.matcher import Matcher

comptime COT_MAX = 300
comptime K8 = 8
comptime K32 = 32


def get_str(doc: JSONDoc, idx: Int, key: String) -> String:
    var vi = doc.get_field(idx, key)
    if vi < 0:
        return String("")
    return doc.get(vi).s


def get_int_list(doc: JSONDoc, idx: Int, key: String) -> List[Int]:
    var out = List[Int]()
    var vi = doc.get_field(idx, key)
    if vi < 0:
        return out^
    var v = doc.get(vi)
    for j in range(len(v.arr)):
        out.append(Int(doc.get(v.arr[j]).n))
    return out^


def contains(xs: List[Int], v: Int) -> Bool:
    for i in range(len(xs)):
        if xs[i] == v:
            return True
    return False


def trim_at_stop(ids: List[Int], stops: List[Int]) -> List[Int]:
    var out = List[Int]()
    for i in range(len(ids)):
        if contains(stops, ids[i]):
            break
        out.append(ids[i])
    return out^


def strip_ws(s: String) -> String:
    var b = s.as_bytes()
    var lo = 0
    var hi = len(b)
    while lo < hi and (b[lo] == 32 or b[lo] == 9 or b[lo] == 10 or b[lo] == 13):
        lo += 1
    while hi > lo and (b[hi - 1] == 32 or b[hi - 1] == 9 or b[hi - 1] == 10 or b[hi - 1] == 13):
        hi -= 1
    return String(unsafe_from_utf8=Span(b)[lo:hi])


def extend_ids(mut dst: List[Int], src: List[Int]):
    for i in range(len(src)):
        dst.append(src[i])


def bytes_match_pat(b: Span[UInt8, _], i: Int, pat: String) -> Bool:
    var pb = pat.as_bytes()
    var m = len(pb)
    if i + m > len(b):
        return False
    for j in range(m):
        if b[i + j] != pb[j]:
            return False
    return True


def strip_think_and_fences(s: String) -> String:
    # round 2 defect 1: B wraps answers as "<think>...</think>\n```json\n{...}\n```".
    # Drop <think>...</think> spans (unterminated -> drop to end, an exhausted
    # ans_max budget) and every ``` fence marker before anything else looks at
    # the text.
    var b = s.as_bytes()
    var n = len(b)
    var out = List[UInt8]()
    var i = 0
    while i < n:
        if bytes_match_pat(b, i, "<think>"):
            var j = i + 7
            var found = False
            while j < n:
                if bytes_match_pat(b, j, "</think>"):
                    i = j + 8
                    found = True
                    break
                j += 1
            if found:
                continue
            break
        if bytes_match_pat(b, i, "```"):
            i += 3
            continue
        out.append(b[i])
        i += 1
    return String(unsafe_from_utf8=Span(out))


def extract_json_object(s: String) -> String:
    # First balanced {...} object in s (string-literal aware, so a brace
    # inside a JSON string value doesn't unbalance the scan). Falls back to
    # the whitespace-trimmed input if no complete object is found, so a bad
    # sample fails the schema/parse check downstream instead of raising here.
    var b = s.as_bytes()
    var n = len(b)
    var start = -1
    for i in range(n):
        if b[i] == 123:
            start = i
            break
    if start < 0:
        return strip_ws(s)
    var depth = 0
    var in_str = False
    var esc = False
    var end = -1
    var i = start
    while i < n:
        var c = b[i]
        if in_str:
            if esc:
                esc = False
            elif c == 92:
                esc = True
            elif c == 34:
                in_str = False
        else:
            if c == 34:
                in_str = True
            elif c == 123:
                depth += 1
            elif c == 125:
                depth -= 1
                if depth == 0:
                    end = i
                    break
        i += 1
    if end < 0:
        return strip_ws(s)
    return String(unsafe_from_utf8=Span(b)[start : end + 1])


def strip_for_math(text: String) -> String:
    return strip_think_and_fences(text)


def strip_for_json(text: String) -> String:
    return extract_json_object(strip_think_and_fences(text))


def json_value_to_string(doc: JSONDoc, idx: Int) -> String:
    if idx < 0:
        return String("null")
    var v = doc.get(idx)
    if v.kind == JKindNull:
        return String("null")
    elif v.kind == JKindBool:
        return String("true") if v.b else String("false")
    elif v.kind == JKindNumber:
        return v.s
    elif v.kind == JKindString:
        return String("\"") + json_escape(v.s) + "\""
    elif v.kind == JKindArray:
        var s = String("[")
        for i in range(len(v.arr)):
            if i > 0:
                s += ","
            s += json_value_to_string(doc, v.arr[i])
        s += "]"
        return s
    elif v.kind == JKindObject:
        var s = String("{")
        for i in range(len(v.obj_keys)):
            if i > 0:
                s += ","
            s += "\"" + json_escape(v.obj_keys[i]) + "\":" + json_value_to_string(doc, v.obj_vals[i])
        s += "}"
        return s
    return String("null")


def compact_json(sample: String) -> String:
    # Round 3 defect: grammar/'s Automaton.add_literal compiles each
    # structural literal with zero whitespace tolerance, so a schema-correct
    # but pretty-printed answer (the model's default style) always failed
    # the Matcher. Re-parse and re-serialize with json_value_to_string (no
    # spaces, same form grammar/test_accept_known_good.mojo's fixtures use)
    # before the schema check; scored_text in the output stays untouched.
    try:
        var doc = parse_json_bytes(str_bytes(sample))
        return json_value_to_string(doc, doc.root)
    except:
        return strip_ws(sample)


@fieldwise_init
struct GenResult(Copyable, Movable):
    var ids: List[Int]
    var elapsed_s: Float64


def reset_and_load(ctx: DeviceContext, mut b: WindowBufs, context: List[Int], tmax: Int) raises -> Tuple[Int, Int]:
    ctx.enqueue_memset(b.convstate_d, 0)
    ctx.enqueue_memset(b.sstate_d, 0)
    ctx.enqueue_memset(b.kc_d, 0)
    ctx.enqueue_memset(b.vc_d, 0)
    ctx.enqueue_memset(b.kc32_d, 0)
    ctx.enqueue_memset(b.vc32_d, 0)
    ctx.enqueue_memset(b.ctr_d, 0)
    ctx.enqueue_memset(b.prof_d, 0)
    var toks_h = ctx.enqueue_create_host_buffer[DType.int32](tmax)
    ctx.synchronize()
    for i in range(tmax):
        toks_h[i] = 0
    for i in range(len(context)):
        toks_h[i] = Int32(context[i])
    ctx.enqueue_copy(dst_buf=DeviceBuffer[DType.int32](ctx, b.toks_d.unsafe_ptr(), tmax, owning=False), src_buf=toks_h)
    ctx.synchronize()
    var pf_rows = 0
    var pf_tail = 0
    if len(context) - 1 >= PF_MIN:
        pf_rows = len(context) - 1
        pf_tail = pf_rows % MROWS
        if pf_tail == 0:
            pf_tail = MROWS
    return (pf_rows, pf_tail)


def make_cfg(pack_q4: Bool, q4_off: Int, e: Int, pf_rows: Int, pf_tail: Int, n_prompt: Int, n_total: Int) -> WindowCfg:
    return WindowCfg(
        pack_q4=pack_q4, draft_q4=False, q4_off=q4_off, e=e, kcfg=2,
        spec=False, expert_trace=False, spec_dbg=False, serve=False, req_id=0, prof=False, pf2=False, pf3=False, pf4=False,
        dump=False, dump4=False, dump_layer=0, mega=True, att_split=n_total, mega_win=False, dot3=False, pf_chunk=CP,
        pf_rows=pf_rows, pf_tail=pf_tail, n_total=n_total, fr_k=0, fr_off=0, fr_ids_off=0, replay_m=8, mrope=False, inject=False, n_prompt=n_prompt,
        sample=default_sample_params(), dump_pen=False,
    )


def read_toks(ctx: DeviceContext, b: WindowBufs, lo: Int, hi: Int, tmax: Int) raises -> List[Int]:
    var toks_h = ctx.enqueue_create_host_buffer[DType.int32](tmax)
    ctx.enqueue_copy(dst_buf=toks_h, src_buf=b.toks_d)
    ctx.synchronize()
    var out = List[Int]()
    for i in range(lo, hi):
        out.append(Int(toks_h[i]))
    return out^


def run_fresh_generate(
    ctx: DeviceContext, mut b: WindowBufs, mut wst: WindowState,
    pack_q4: Bool, q4_off: Int, e: Int, context: List[Int], gen_max: Int, tmax: Int,
) raises -> GenResult:
    var plen = len(context)
    var pf = reset_and_load(ctx, b, context, tmax)
    var cfg = make_cfg(pack_q4, q4_off, e, pf[0], pf[1], plen, plen + gen_max)
    var t0 = perf_counter_ns()
    wst.reset(t0)
    while wst.pos < plen + gen_max - 1:
        step_window(ctx, b, cfg, wst)
    ctx.synchronize()
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    var generated = read_toks(ctx, b, plen, plen + gen_max, tmax)
    return GenResult(generated^, dt)


def run_generate_to_stop(
    ctx: DeviceContext, mut b: WindowBufs, mut wst: WindowState,
    pack_q4: Bool, q4_off: Int, e: Int, context: List[Int], gen_max: Int, tmax: Int, stops: List[Int],
) raises -> GenResult:
    # Same cfg as run_fresh_generate, so the same greedy ids as arm T's
    # producer, but stops feeding at the first stop id: the SSM state cannot be
    # rewound, so this is the only way to hold the state after exactly
    # context + ids (T's trim point). Costs one 4-byte read-back per step.
    var plen = len(context)
    var pf = reset_and_load(ctx, b, context, tmax)
    var cfg = make_cfg(pack_q4, q4_off, e, pf[0], pf[1], plen, plen + gen_max)
    var tok_h = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.synchronize()
    var ids = List[Int]()
    var t0 = perf_counter_ns()
    wst.reset(t0)
    while wst.pos < plen:
        step_window(ctx, b, cfg, wst)
    while len(ids) < gen_max:
        var cur = DeviceBuffer[DType.int32](ctx, b.toks_d.unsafe_ptr() + wst.pos, 1, owning=False)
        ctx.enqueue_copy(dst_buf=tok_h, src_buf=cur)
        ctx.synchronize()
        var t = Int(tok_h[0])
        if contains(stops, t):
            break
        ids.append(t)
        step_window(ctx, b, cfg, wst)
        if wst.pos != plen + len(ids):
            raise Error("run_generate_to_stop: decode advanced to " + String(wst.pos) + ", expected " + String(plen + len(ids)))
    ctx.synchronize()
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    return GenResult(ids^, dt)


def run_to_prompt_end(
    ctx: DeviceContext, mut b: WindowBufs, mut wst: WindowState,
    pack_q4: Bool, q4_off: Int, e: Int, context: List[Int], tmax: Int,
) raises -> WindowCfg:
    var plen = len(context)
    var pf = reset_and_load(ctx, b, context, tmax)
    var cfg = make_cfg(pack_q4, q4_off, e, pf[0], pf[1], plen, plen + 1)
    wst.reset(perf_counter_ns())
    while wst.pos < plen:
        step_window(ctx, b, cfg, wst)
    return cfg^


def step_latent_raw(
    ctx: DeviceContext, mut b: WindowBufs, cfg: WindowCfg, mut st: WindowState,
    latent_dev_vec: DeviceBuffer[f32],
) raises:
    var x_slice = DeviceBuffer[f32](ctx, b.x_d.unsafe_ptr(), H, owning=False)
    ctx.enqueue_copy(dst_buf=x_slice, src_buf=latent_dev_vec)

    var Xm = TileTensor(b.x_d, xm_layout)
    var CurBm = TileTensor(b.curb_d, xm_layout)
    var Toks = TileTensor(b.toks_d, toks_layout)
    var ConvStateAll = TileTensor(b.convstate_d, csall_layout)
    var SStateAll = TileTensor(b.sstate_d, ssall_layout)
    var Hnm0 = TileTensor(b.hn_d, xm_layout)
    var Dtok0 = TileTensor(b.dtok_d, dtok_layout)

    ctx.enqueue_function[mega_token_q4_k](
        b.wbuf.unsafe_ptr(), TileTensor(b.off_d, off_layout), Xm, CurBm,
        TileTensor(b.resb_d, xm_layout), TileTensor(b.qkv_d, qfm_layout), TileTensor(b.z_d, xm_layout),
        TileTensor(b.araw_d, g32m_layout), TileTensor(b.braw_d, g32m_layout),
        TileTensor(b.eg_d, g32m_layout), TileTensor(b.beta_d, g32m_layout),
        TileTensor(b.conv_d, convm_layout), TileTensor(b.so_d, om_layout), ConvStateAll, SStateAll,
        TileTensor(b.qf_d, qfm_layout), TileTensor(b.k_d, kvm_flat), TileTensor(b.v_d, kvm_flat),
        TileTensor(b.q_d, qm_layout), TileTensor(b.gate_d, xflat_layout), TileTensor(b.ao_d, qm_layout),
        b.kc_d.unsafe_ptr(), b.vc_d.unsafe_ptr(), b.kvtab_d.unsafe_ptr(),
        TileTensor(b.p_ffn_d, pf_sm), TileTensor(b.p_ffn2_d, pf_sm), TileTensor(b.fgb_d, ffnm_layout),
        TileTensor(b.ctr_d, ctr_layout), b.prof_d.unsafe_ptr(), b.dbg_d.unsafe_ptr(),
        Toks, Dtok0, Hnm0, b.hmax_d.unsafe_ptr(), b.hidx_d.unsafe_ptr(),
        Int32(st.ring), Int32(SLOTS), Int32(st.pos), Int32(1), Int32(0), Int32(1), Int32(cfg.att_split),
        b.sv_d.unsafe_ptr(), Int32(0),
        grid_dim=MEGA_G, block_dim=ROW_THREADS,
    )
    st.ring = (st.ring + 1) % SLOTS
    st.pos_prev = st.pos
    st.pos += 1


def collect_latent_raw(
    ctx: DeviceContext, mut b: WindowBufs, cfg: WindowCfg, mut st: WindowState,
    latent_host: HostBuffer[f32], k: Int,
) raises:
    # Round 3: b.hn_d is never written under mega=True (kernels/mega.mojo
    # gates the write on fold_head==2; every launch here passes 1), so
    # reading it directly ships a stale earlier-chunk vector. final_norm_hidden
    # re-derives the current post-final-norm hidden from b.x_d instead.
    for s in range(k):
        var latent_dev = ctx.enqueue_create_buffer[f32](H)
        final_norm_hidden(ctx, b, latent_dev)
        var host_dst = latent_host.create_sub_buffer[f32](s * H, H)
        ctx.enqueue_copy(dst_buf=host_dst, src_buf=latent_dev)
        ctx.synchronize()
        step_latent_raw(ctx, b, cfg, st, latent_dev)


def collect_latent_soft(
    ctx: DeviceContext, mut b: WindowBufs, cfg: WindowCfg, mut st: WindowState,
    latent_host: HostBuffer[f32], k: Int, pack_q4: Bool,
) raises:
    # never falls back to raw hn_d on failure -- the caller catches any raise
    # from realign_expected_embedding and records the arm as errored.
    for s in range(k):
        var e_dev = ctx.enqueue_create_buffer[f32](H)
        realign_expected_embedding(ctx, b, e_dev, pack_q4)
        var host_dst = latent_host.create_sub_buffer[f32](s * H, H)
        ctx.enqueue_copy(dst_buf=host_dst, src_buf=e_dev)
        ctx.synchronize()
        step_latent_raw(ctx, b, cfg, st, e_dev)


def apply_latent_to_receiver(
    ctx: DeviceContext, mut b: WindowBufs, cfg: WindowCfg, mut st: WindowState,
    latent_host: HostBuffer[f32], k: Int,
) raises:
    for s in range(k):
        var latent_dev = ctx.enqueue_create_buffer[f32](H)
        var host_src = latent_host.create_sub_buffer[f32](s * H, H)
        ctx.enqueue_copy(dst_buf=latent_dev, src_buf=host_src)
        ctx.synchronize()
        step_latent_raw(ctx, b, cfg, st, latent_dev)


def append_known_tokens(
    ctx: DeviceContext, mut b: WindowBufs, mut wst: WindowState,
    pack_q4: Bool, q4_off: Int, e: Int, known_ids: List[Int], tmax: Int,
) raises:
    # Feeds real, already-known token ids (the turn-boundary reopener, the
    # no-think opener) through the SAME engine state a latent arm already
    # built via step_latent_raw, continuing from wst.pos rather than
    # resetting -- unlike run_fresh_generate, which restarts from scratch and
    # is used instead for arms 0/T (b_context is just a longer id list there).
    if len(known_ids) == 0:
        return
    var base_pos = wst.pos
    var target = base_pos + len(known_ids)
    var host = ctx.enqueue_create_host_buffer[DType.int32](len(known_ids))
    ctx.synchronize()
    for i in range(len(known_ids)):
        host[i] = Int32(known_ids[i])
    var dst = DeviceBuffer[DType.int32](ctx, b.toks_d.unsafe_ptr() + base_pos, len(known_ids), owning=False)
    ctx.enqueue_copy(dst_buf=dst, src_buf=host)
    ctx.synchronize()
    var cfg = make_cfg(pack_q4, q4_off, e, 0, 0, target, target + 1)
    while wst.pos < target:
        step_window(ctx, b, cfg, wst)


def greedy_tokenize(trie: TokenTrie, text_bytes: List[UInt8]) raises -> List[Int]:
    var out: List[Int] = []
    var pos = 0
    var n = len(text_bytes)
    while pos < n:
        var node_idx = trie.root_child(text_bytes[pos])
        if node_idx < 0:
            raise Error("no vocab token starts with byte at pos " + String(pos))
        var best_end = -1
        var best_token = -1
        if trie.nodes[Int(node_idx)].token_id >= 0:
            best_end = pos + 1
            best_token = Int(trie.nodes[Int(node_idx)].token_id)
        var cur = Int(node_idx)
        var i = pos + 1
        while i < n:
            var nxt = trie.nodes[cur].find_child(text_bytes[i])
            if nxt < 0:
                break
            cur = Int(nxt)
            i += 1
            if trie.nodes[cur].token_id >= 0:
                best_end = i
                best_token = Int(trie.nodes[cur].token_id)
        if best_token < 0:
            raise Error("cannot tokenize at pos " + String(pos))
        out.append(best_token)
        pos = best_end
    return out^


def check_schema_valid(schema_path: String, sample: String, vp: ArcPointer[Vocab], tp: ArcPointer[TokenTrie]) -> Bool:
    try:
        var doc = parse_json_file(schema_path)
        var a = Automaton()
        var rid = compile_root_schema(a, doc)
        var m = Matcher(a^, tp, vp)
        var toks = greedy_tokenize(tp[], str_bytes(compact_json(sample)))
        for i in range(len(toks)):
            if not m.accept(toks[i]):
                return False
        return m.is_terminated()
    except:
        return False


def json_escape(s: String) -> String:
    var out = List[UInt8]()
    for c in s.as_bytes():
        if c == 34:
            out.append(92); out.append(34)
        elif c == 92:
            out.append(92); out.append(92)
        elif c == 10:
            out.append(92); out.append(110)
        elif c == 13:
            out.append(92); out.append(114)
        elif c == 9:
            out.append(92); out.append(116)
        elif c < 32:
            pass
        else:
            out.append(c)
    return String(unsafe_from_utf8=Span(out))


def ids_to_json(ids: List[Int]) -> String:
    var s = String("[")
    for i in range(len(ids)):
        if i > 0:
            s += ","
        s += String(ids[i])
    s += "]"
    return s


def arm_json(name: String, producer_s: Float64, receiver_s: Float64, ids: List[Int], text: String, scored_text: String, schema_valid: Int, error: String, extra: String = "") -> String:
    var s = String("{\"arm\":\"") + name + "\",\"producer_s\":" + String(producer_s)
    s += ",\"receiver_s\":" + String(receiver_s)
    s += ",\"generated_ids\":" + ids_to_json(ids)
    s += ",\"answer_text\":\"" + json_escape(text) + "\""
    s += ",\"scored_text\":\"" + json_escape(scored_text) + "\""
    if schema_valid < 0:
        s += ",\"schema_valid\":null"
    else:
        s += ",\"schema_valid\":" + ("true" if schema_valid == 1 else "false")
    if error == "":
        s += ",\"error\":null"
    else:
        s += ",\"error\":\"" + json_escape(error) + "\""
    s += extra
    s += "}"
    return s


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var args = argv()
    var n_items = 120
    var ids_arg = String("")
    var arms_arg = String("0,T,L8-raw,L8-soft,L32-soft")
    var out_prefix = String("results/e8/topology1-q4")
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--items" and i + 1 < len(args):
            n_items = atol(String(args[i + 1]))
            i += 2
        elif a == "--ids" and i + 1 < len(args):
            # round 2: `--items N` can only take the first N items in file
            # order (all 20 json then all 20 math) -- this selects specific
            # ids by name, e.g. json_01,json_02,...,math_01,... for the
            # mixed-type smoke the coordinator asked for.
            ids_arg = String(args[i + 1])
            i += 2
        elif a == "--arms" and i + 1 < len(args):
            arms_arg = String(args[i + 1])
            i += 2
        elif a == "--out" and i + 1 < len(args):
            out_prefix = String(args[i + 1])
            i += 2
        else:
            i += 1
    var arms = arms_arg.split(",")
    var want_ids = ids_arg.split(",")
    var use_ids = ids_arg != ""
    var ans_max = atol(getenv("BARO_E8_ANS_MAX", "256"))
    # Budget-capped receiver (round 5). BARO_E8_RECV_MAX > 0 makes the handoff
    # load-bearing: B answers in a short fixed budget with no working, so arm 0
    # collapses toward guessing instead of silently re-deriving everything T
    # carried. The round-4 pilot was void because arm 0 could reason from the
    # prompt, giving T no private information and the gate no assay sensitivity
    # (runs/latent-os/E8-pilot-2026-09-09.md). 0 keeps the pilot's behaviour.
    var recv_max = atol(getenv("BARO_E8_RECV_MAX", "0"))
    var share_a = getenv("BARO_E8_SHARE_A", "0") == "1"
    var nothink = getenv("BARO_E8_NOTHINK", "1") == "1"
    var packdir = getenv("BARO_PACK", ".work/engine-pack-q4")
    var gguf_path = getenv(
        "BARO_E8_GGUF",
        getenv("HOME", "")
        + "/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf",
    )
    # longest arm needs prompt + max(COT_MAX, K32) + ans_max; round 4's
    # e8_tasks.json is GSM8K-hard + schema-in-prompt, so the longest prompt is
    # 228 tokens (measured, math_025) against round 3's 98: 228 + 300 + 256 =
    # 784, and 896 is the next whole KV page (tpages=ceildiv(tmax,128)=7).
    # Two co-resident engines at registry.TMAX=1088 measured 25.48/25.75 GB
    # used (bench/latent-handoff.sh smoke run, 2026-09-09); at 64 KiB/token
    # (E1) the two extra pages over round 3's 640 cost ~32 MiB across both
    # engines, without touching serve/window.mojo's own KVPAGE/TMAX.
    var tmax = atol(getenv("BARO_E8_TMAX", "896"))

    print("E8 HARNESS: items=", n_items, " arms=", arms_arg, " ans_max=", ans_max, " recv_max=", recv_max, " tmax=", tmax, " pack=", packdir)

    var ctx = DeviceContext()

    print("--- loading producer (A) pack ---")
    var packA = load_pack(ctx, packdir)
    var bufsA = alloc_bufs(ctx, packA, tmax)

    print("--- loading receiver (B) pack ---")
    var packB = load_pack(ctx, packdir)
    var bufsB = alloc_bufs(ctx, packB, tmax)

    # Marker for the wrapper script: both packs are resident now, so this is
    # the moment to read VRAM back -- a reading taken after this process exits
    # would just re-measure baseline (bench/latent-handoff.sh polls for this).
    with open(".work/e8-vram-ready.marker", "w") as f:
        f.write_bytes(String("1").as_bytes())

    var pack_q4 = packA.pack_q4
    var q4_off = packA.q4_off
    var eA = packA.e
    var eB = packB.e

    print("--- loading tokenizer + grammar vocab ---")
    var tok = Tokenizer(gguf_path)

    # U3 (docs/design/latent-os/09-roadmap.md "model universal" discussion):
    # both engines in this harness share one process, so B's own identity IS
    # A's; computed once here, never per handoff (identity.mojo docstring).
    var mojo_baro_dir = getenv("BARO_MOJO_BARO_DIR", getenv("HOME", "") + "/Projects/mojo/mojo-baro")
    var mojo_bin = getenv("BARO_MOJO_BIN", mojo_baro_dir + "/.venv/bin/mojo")
    var receiver_identity = compute_engine_identity(gguf_path, mojo_baro_dir, mojo_bin, tok)
    # KV-arm negative control (U3 goal 3's CHECK): flip one byte of one field
    # in the MINTED header before B's ingest check, simulating a producer on
    # a different model/tokenizer/build without needing two real GGUFs.
    var force_mismatch_field = getenv("BARO_E8_FORCE_IDENTITY_MISMATCH", "")

    var stops = List[Int]()
    if tok.eos_id >= 0:
        stops.append(tok.eos_id)
    var im_end = tok.tok2id.get("<|im_end|>", -1)
    if im_end >= 0 and not contains(stops, im_end):
        stops.append(im_end)

    var vocab = load_vocab(packdir)
    var trie = build_trie(vocab)
    var vp = ArcPointer(vocab^)
    var tp = ArcPointer(trie^)

    # round 2 defect 4/5: every arm hands B off at the same assistant-turn
    # boundary, and (BARO_E8_NOTHINK=1 default) with B's own CoT suppressed --
    # the only reasoning channel under test is the handoff, not B re-thinking
    # from scratch. Computed once via the real tokenizer/BPE so these match
    # however <|im_end|>/<|im_start|> and "assistant"/"<think>" actually tokenize.
    var turn_ids = tok.encode(String("<|im_end|>\n<|im_start|>assistant\n"), add_special=False)
    var nothink_ids = tok.encode(String("<think>\n\n</think>\n\n"), add_special=False)
    print("turn_ids:", ids_to_json(turn_ids), " nothink:", nothink, " nothink_ids:", ids_to_json(nothink_ids))

    # Same shape for every arm: [prompt] [payload or nothing] [recv turn] -> generate,
    # so the arms still differ only in what crossed the gap.
    var recv_math_ids = tok.encode(
        String("<|im_end|>\n<|im_start|>user\nGive only the final integer answer, as: Answer: <number>. No working.<|im_end|>\n<|im_start|>assistant\n"),
        add_special=False)
    var recv_json_ids = tok.encode(
        String("<|im_end|>\n<|im_start|>user\nGive only the JSON object. No explanation.<|im_end|>\n<|im_start|>assistant\n"),
        add_special=False)
    # E12-long: RULER items (bench/ruler/to_e8.py) answer with short strings.
    var recv_ruler_ids = tok.encode(
        String("<|im_end|>\n<|im_start|>user\nGive only the answer. No working.<|im_end|>\n<|im_start|>assistant\n"),
        add_special=False)
    if recv_max > 0:
        print("recv-capped: budget", recv_max, " math_ids", len(recv_math_ids), " json_ids", len(recv_json_ids), " ruler_ids", len(recv_ruler_ids))

    var wstA = WindowState(pos=0, pos_prev=0, ring=0, n_drafted=0, n_accepted=0, n_spec_windows=0, n_dumped=0, tp=0, tq=0, pf_att=0, pf_ssm=0, pf_ffn=0, pf_head=0, pf_proc=0, pf_draft=0, fc=[0, 0, 0, 0, 0, 0], pc=[0, 0, 0, 0, 0, 0, 0, 0], ac=[0, 0, 0, 0, 0, 0], p3=[0, 0, 0, 0], pfx=[0, 0, 0, 0], grammar=None, grammar_mask=Bitset(1), grammar_pending_think=False, grammar_think_buf=List[UInt8](), grammar_stop=False, grammar_masked_draws=0, grammar_accepted=0, tool_re="", tool_buf=List[UInt8](), grammar_tool=False)
    var wstB = WindowState(pos=0, pos_prev=0, ring=0, n_drafted=0, n_accepted=0, n_spec_windows=0, n_dumped=0, tp=0, tq=0, pf_att=0, pf_ssm=0, pf_ffn=0, pf_head=0, pf_proc=0, pf_draft=0, fc=[0, 0, 0, 0, 0, 0], pc=[0, 0, 0, 0, 0, 0, 0, 0], ac=[0, 0, 0, 0, 0, 0], p3=[0, 0, 0, 0], pfx=[0, 0, 0, 0], grammar=None, grammar_mask=Bitset(1), grammar_pending_think=False, grammar_think_buf=List[UInt8](), grammar_stop=False, grammar_masked_draws=0, grammar_accepted=0, tool_re="", tool_buf=List[UInt8](), grammar_tool=False)

    # KV arm (E12): one pinned 50.25 MiB SSM slot per side, allocated once.
    var kv_cap = 0
    for j in range(len(arms)):
        if String(arms[j]) == "KV":
            kv_cap = 1
    var chainA = Chain(ctx, kv_cap, packdir)
    var chainB = Chain(ctx, kv_cap, packdir)

    var doc = parse_json_file(getenv("BARO_E8_TASKS", "bench/data/e8_tasks.json"))
    var root = doc.get(doc.root)
    var n_avail = len(root.arr)
    var run_indices = List[Int]()
    if use_ids:
        for wi in range(len(want_ids)):
            var wid = String(want_ids[wi])
            var found = -1
            for j in range(n_avail):
                if get_str(doc, root.arr[j], "id") == wid:
                    found = root.arr[j]
                    break
            if found < 0:
                print("WARNING: --ids requested unknown id", wid, "-- skipping")
            else:
                run_indices.append(found)
    else:
        var n_run = n_items if n_items < n_avail else n_avail
        for j in range(n_run):
            run_indices.append(root.arr[j])

    var out = String("{\"topology\":\"topology1-q4\",\"pack\":\"") + packdir + "\""
    out += ",\"transport\":\"in-process host copy (memfd transport is E7's, not measured here)\""
    out += ",\"turn_boundary_note\":\"arm 0 already ends at the assistant turn boundary (full_prompt); arms T/L8-raw/L8-soft/L32-soft append <|im_end|>\\n<|im_start|>assistant\\n as real ids after the handoff (A's trimmed CoT ids for T, the k latent-injection steps for the L arms) before B generates, so every arm starts B at the same turn boundary\""
    out += ",\"nothink\":" + ("true" if nothink else "false")
    out += ",\"items\":["
    var first_item = True

    for it in range(len(run_indices)):
        var idx = run_indices[it]
        var task_id = get_str(doc, idx, "id")
        var task_type = get_str(doc, idx, "type")
        var schema_file = get_str(doc, idx, "schema_file")
        var tokens = get_int_list(doc, idx, "tokens")
        print("=== item", task_id, "(" + task_type + ") n_tok=" + String(len(tokens)), "===")

        var capped = recv_max > 0
        var hand_ids = turn_ids.copy()
        if capped:
            if task_type == "json":
                hand_ids = recv_json_ids.copy()
            elif task_type == "ruler":
                hand_ids = recv_ruler_ids.copy()
            else:
                hand_ids = recv_math_ids.copy()
        var gen_budget = recv_max if capped else ans_max

        if not first_item:
            out += ","
        first_item = False
        out += "{\"id\":\"" + json_escape(task_id) + "\",\"type\":\"" + json_escape(task_type) + "\""
        out += ",\"expected\":" + json_value_to_string(doc, doc.get_field(idx, "expected"))
        out += ",\"arms\":["
        var first_arm = True
        # BARO_E8_SHARE_A=1 (E12-long 32k amendment): KV reuses T's producer run
        # instead of re-decoding it. B never touches bufsA, so A's KV pages and
        # SSM state are still at T's trim point when the KV arm mints them.
        var share_ids = List[Int]()
        var share_s: Float64 = 0.0
        var have_share = False

        for arm_i in range(len(arms)):
            var arm = String(arms[arm_i])
            if not first_arm:
                out += ","
            first_arm = False
            print("  arm", arm)

            try:
                if arm == "0":
                    var b_context = tokens.copy()
                    if capped:
                        extend_ids(b_context, hand_ids)
                    if nothink:
                        extend_ids(b_context, nothink_ids)
                    var r = run_fresh_generate(ctx, bufsB, wstB, pack_q4, q4_off, eB, b_context, gen_budget, tmax)
                    var ans_ids = trim_at_stop(r.ids, stops)
                    var text = tok.decode(ans_ids)
                    var sv = -1
                    var scored = String("")
                    if task_type == "json":
                        scored = strip_for_json(text)
                        sv = 1 if check_schema_valid(schema_file, scored, vp, tp) else 0
                    else:
                        scored = strip_for_math(text)
                    out += arm_json(arm, 0.0, r.elapsed_s, r.ids, text, scored, sv, "")

                elif arm == "T":
                    # Stop-aware since E12: run_fresh_generate always decoded COT_MAX
                    # steps, so producer_s counted padding past the stop id
                    # (runs/latent-os/E12-smoke-2026-09-11.md). Same greedy ids.
                    var rA = run_generate_to_stop(ctx, bufsA, wstA, pack_q4, q4_off, eA, tokens, COT_MAX, tmax, stops)
                    var cot_ids = trim_at_stop(rA.ids, stops)
                    if share_a:
                        share_ids = rA.ids.copy()
                        share_s = rA.elapsed_s
                        have_share = True
                    var b_context = tokens.copy()
                    extend_ids(b_context, cot_ids)
                    extend_ids(b_context, hand_ids)
                    if nothink:
                        extend_ids(b_context, nothink_ids)
                    var rB = run_fresh_generate(ctx, bufsB, wstB, pack_q4, q4_off, eB, b_context, gen_budget, tmax)
                    var ans_ids = trim_at_stop(rB.ids, stops)
                    var text = tok.decode(ans_ids)
                    var sv = -1
                    var scored = String("")
                    if task_type == "json":
                        scored = strip_for_json(text)
                        sv = 1 if check_schema_valid(schema_file, scored, vp, tp) else 0
                    else:
                        scored = strip_for_math(text)
                    var t_ctx = tokens.copy()
                    extend_ids(t_ctx, cot_ids)
                    var t_extra = String(",\"handoff_pos\":") + String(len(t_ctx)) + ",\"handoff_hash\":\"" + String(prefix_hash(List[UInt8](), t_ctx, len(t_ctx))) + "\""
                    out += arm_json(arm, rA.elapsed_s, rB.elapsed_s, rB.ids, text, scored, sv, "", t_extra)

                elif arm == "L8-raw" or arm == "L8-soft" or arm == "L32-soft":
                    var k = K32 if arm == "L32-soft" else K8
                    var latent_hA = ctx.enqueue_create_host_buffer[f32](k * H)
                    ctx.synchronize()
                    var t0a = perf_counter_ns()
                    var cfgA = run_to_prompt_end(ctx, bufsA, wstA, pack_q4, q4_off, eA, tokens, tmax)
                    if arm == "L8-raw":
                        collect_latent_raw(ctx, bufsA, cfgA, wstA, latent_hA, k)
                    else:
                        collect_latent_soft(ctx, bufsA, cfgA, wstA, latent_hA, k, pack_q4)
                    ctx.synchronize()
                    var producer_s = Float64(perf_counter_ns() - t0a) / 1e9

                    var t0b = perf_counter_ns()
                    var cfgB = run_to_prompt_end(ctx, bufsB, wstB, pack_q4, q4_off, eB, tokens, tmax)
                    apply_latent_to_receiver(ctx, bufsB, cfgB, wstB, latent_hA, k)
                    append_known_tokens(ctx, bufsB, wstB, pack_q4, q4_off, eB, hand_ids, tmax)
                    if nothink:
                        append_known_tokens(ctx, bufsB, wstB, pack_q4, q4_off, eB, nothink_ids, tmax)
                    var start_pos = wstB.pos
                    var cfg_gen = make_cfg(pack_q4, q4_off, eB, 0, 0, start_pos, start_pos + gen_budget)
                    while wstB.pos < start_pos + gen_budget - 1:
                        step_window(ctx, bufsB, cfg_gen, wstB)
                    ctx.synchronize()
                    var receiver_s = Float64(perf_counter_ns() - t0b) / 1e9
                    var gen_ids = read_toks(ctx, bufsB, start_pos, start_pos + gen_budget, tmax)
                    var ans_ids = trim_at_stop(gen_ids, stops)
                    var text = tok.decode(ans_ids)
                    var sv = -1
                    var scored = String("")
                    if task_type == "json":
                        scored = strip_for_json(text)
                        sv = 1 if check_schema_valid(schema_file, scored, vp, tp) else 0
                    else:
                        scored = strip_for_math(text)
                    out += arm_json(arm, producer_s, receiver_s, gen_ids, text, scored, sv, "")

                elif arm == "L8-proj" or arm == "L32-proj":
                    # E13 (docs/design/latent-os/06-experiments.md): producer
                    # unchanged from L8-raw; a trained host-side projector
                    # (e13_projector.mojo, piece 2) maps each raw vector
                    # before apply_latent_to_receiver. BARO_E13_PROJ_K8 /
                    # BARO_E13_PROJ_K32 name the trained weights file
                    # (e13_projector.mojo's format); there is no built-in
                    # fallback -- an untrained/missing path is a hard error,
                    # never a silent downgrade to raw.
                    var k = K32 if arm == "L32-proj" else K8
                    var proj_path = getenv("BARO_E13_PROJ_K32", "") if arm == "L32-proj" else getenv("BARO_E13_PROJ_K8", "")
                    if proj_path == "":
                        raise Error(arm + ": set BARO_E13_PROJ_K8/BARO_E13_PROJ_K32 to a trained projector weights file")

                    var latent_hA = ctx.enqueue_create_host_buffer[f32](k * H)
                    ctx.synchronize()
                    var t0a = perf_counter_ns()
                    var cfgA = run_to_prompt_end(ctx, bufsA, wstA, pack_q4, q4_off, eA, tokens, tmax)
                    collect_latent_raw(ctx, bufsA, cfgA, wstA, latent_hA, k)
                    ctx.synchronize()

                    var proj = load_projector(proj_path, H)
                    var raw_flat = List[Float32](unsafe_uninit_length=k * H)
                    for s in range(k * H):
                        raw_flat[s] = latent_hA[s]
                    var proj_flat = apply_projector_k(proj, raw_flat, k)
                    var latent_hP = ctx.enqueue_create_host_buffer[f32](k * H)
                    ctx.synchronize()
                    for s in range(k * H):
                        latent_hP[s] = proj_flat[s]
                    var producer_s = Float64(perf_counter_ns() - t0a) / 1e9

                    var t0b = perf_counter_ns()
                    var cfgB = run_to_prompt_end(ctx, bufsB, wstB, pack_q4, q4_off, eB, tokens, tmax)
                    apply_latent_to_receiver(ctx, bufsB, cfgB, wstB, latent_hP, k)
                    append_known_tokens(ctx, bufsB, wstB, pack_q4, q4_off, eB, hand_ids, tmax)
                    if nothink:
                        append_known_tokens(ctx, bufsB, wstB, pack_q4, q4_off, eB, nothink_ids, tmax)
                    var start_pos = wstB.pos
                    var cfg_gen = make_cfg(pack_q4, q4_off, eB, 0, 0, start_pos, start_pos + gen_budget)
                    while wstB.pos < start_pos + gen_budget - 1:
                        step_window(ctx, bufsB, cfg_gen, wstB)
                    ctx.synchronize()
                    var receiver_s = Float64(perf_counter_ns() - t0b) / 1e9
                    var gen_ids = read_toks(ctx, bufsB, start_pos, start_pos + gen_budget, tmax)
                    var ans_ids = trim_at_stop(gen_ids, stops)
                    var text = tok.decode(ans_ids)
                    var sv = -1
                    var scored = String("")
                    if task_type == "json":
                        scored = strip_for_json(text)
                        sv = 1 if check_schema_valid(schema_file, scored, vp, tp) else 0
                    else:
                        scored = strip_for_math(text)
                    out += arm_json(arm, producer_s, receiver_s, gen_ids, text, scored, sv, "")

                elif arm == "KV":
                    var a_ids = List[Int]()
                    var a_s: Float64 = 0.0
                    if share_a and have_share:
                        a_ids = share_ids.copy()
                        a_s = share_s
                    else:
                        var rA = run_generate_to_stop(ctx, bufsA, wstA, pack_q4, q4_off, eA, tokens, COT_MAX, tmax, stops)
                        a_ids = rA.ids.copy()
                        a_s = rA.elapsed_s
                    var a_ctx = tokens.copy()
                    extend_ids(a_ctx, a_ids)
                    var hand_pos = len(a_ctx)
                    if wstA.pos != hand_pos:
                        raise Error("KV: producer at pos " + String(wstA.pos) + ", handoff at " + String(hand_pos))
                    var hand_hash = prefix_hash(List[UInt8](), a_ctx, hand_pos)
                    # KV pool is page-major (kernels/attn.mojo kv_off), so pages
                    # [0, ceil(hand_pos/128)) hold exactly the prefix.
                    var kv_pages = (hand_pos + KVPAGE - 1) // KVPAGE
                    var zero32 = Array[UInt8, 32](fill=0)
                    var t0m = perf_counter_ns()
                    var kv = mint_kv_latent(ctx, bufsA.kc_d, bufsA.vc_d, 0, kv_pages, UInt64(hand_pos), receiver_identity.weights_uuid, zero32, receiver_identity.runtime, receiver_identity.tokenizer_sha)
                    var kv_header = kv[0].copy()
                    var kv_fd = kv[1]
                    chainA.save(ctx, bufsA.convstate_d, bufsA.sstate_d, wstA.ring, hand_pos, a_ctx, True, True)
                    ctx.synchronize()
                    chainA.commit()
                    if not chainA.items[0].valid or chainA.items[0].pos != hand_pos:
                        raise Error("KV: SSM checkpoint not at the handoff position")
                    var ck = mint_chain_slot(chainA, 0, receiver_identity.weights_uuid, zero32, receiver_identity.runtime, receiver_identity.tokenizer_sha)
                    var ck_header = ck[0].copy()
                    var ck_fd = ck[1]
                    var mint_s = Float64(perf_counter_ns() - t0m) / 1e9

                    # U3 goal 3's negative-control knob: flip one byte of the
                    # MINTED header (never the receiver's own identity) so the
                    # check below refuses it exactly as a real cross-model
                    # producer would be refused.
                    if force_mismatch_field == "weights_uuid":
                        kv_header.weights_uuid[0] = kv_header.weights_uuid[0] ^ 0xFF
                    elif force_mismatch_field == "runtime":
                        kv_header.runtime[0] = kv_header.runtime[0] ^ 0xFF
                    elif force_mismatch_field == "sigma_id":
                        kv_header.sigma_id[0] = kv_header.sigma_id[0] ^ 0xFF
                    elif force_mismatch_field == "tokenizer_sha":
                        kv_header.tokenizer_sha[0] = kv_header.tokenizer_sha[0] ^ 0xFF

                    # Identity check BEFORE any B-side restore (U3 goal 2: "nothing
                    # is restored" on mismatch): check the minted header against B's
                    # own identity while B's buffers are still untouched.
                    var fallback_field = String("")
                    try:
                        proto.check_identity(kv_header, receiver_identity.weights_uuid, receiver_identity.runtime, receiver_identity.sigma_id, receiver_identity.tokenizer_sha)
                    except e:
                        fallback_field = String(String(e).removeprefix("IDENTITY_MISMATCH:"))
                        _ = latentos_sys.sys_close(kv_fd)
                        _ = latentos_sys.sys_close(ck_fd)

                    var b_ctx = a_ctx.copy()
                    extend_ids(b_ctx, hand_ids)
                    if nothink:
                        extend_ids(b_ctx, nothink_ids)
                    var total = len(b_ctx)

                    var receiver_s: Float64 = 0.0
                    var ingest_s: Float64 = 0.0
                    var gen_ids = List[Int]()
                    if fallback_field != "":
                        # Recipe fallback (U3 goal 3): B never touched the refused
                        # handle; it answers from A's text CoT instead, same
                        # context shape as arm T's producer-to-receiver handoff.
                        var t0b = perf_counter_ns()
                        var rB = run_fresh_generate(ctx, bufsB, wstB, pack_q4, q4_off, eB, b_ctx, gen_budget, tmax)
                        receiver_s = rB.elapsed_s
                        gen_ids = rB.ids.copy()
                    else:
                        if total - 1 - hand_pos < MROWS:
                            raise Error("KV: fewer than MROWS tokens after the handoff, the prefill tail would underflow")
                        var pfB = reset_and_load(ctx, bufsB, b_ctx, tmax)
                        var t0b = perf_counter_ns()
                        ingest_kv_latent(ctx, bufsB.kc_d, bufsB.vc_d, kv_header, kv_fd)
                        var slot = ingest_into_chain(chainB, ck_header, ck_fd)
                        chainB.restore(ctx, bufsB.convstate_d, bufsB.sstate_d, 0, slot)
                        ctx.synchronize()
                        ingest_s = Float64(perf_counter_ns() - t0b) / 1e9
                        # serve/engine.mojo's restore contract (slot 0, ring 0, pos = pos_prev
                        # = cached), then the rest of B's context in one batched prefill, the
                        # way arm T prefills it; the smoke fed it row by row and measured that.
                        wstB.reset(t0b)
                        wstB.pos = hand_pos
                        wstB.pos_prev = hand_pos
                        var cfgB = make_cfg(pack_q4, q4_off, eB, pfB[0], pfB[1], total, total + gen_budget)
                        while wstB.pos < total + gen_budget - 1:
                            step_window(ctx, bufsB, cfgB, wstB)
                        ctx.synchronize()
                        receiver_s = Float64(perf_counter_ns() - t0b) / 1e9
                        gen_ids = read_toks(ctx, bufsB, total, total + gen_budget, tmax)
                    var ans_ids = trim_at_stop(gen_ids, stops)
                    var text = tok.decode(ans_ids)
                    var sv = -1
                    var scored = String("")
                    if task_type == "json":
                        scored = strip_for_json(text)
                        sv = 1 if check_schema_valid(schema_file, scored, vp, tp) else 0
                    else:
                        scored = strip_for_math(text)
                    var kv_extra = String(",\"handoff_pos\":") + String(hand_pos) + ",\"handoff_hash\":\"" + String(hand_hash) + "\""
                    kv_extra += ",\"kv_pages\":" + String(kv_pages) + ",\"mint_s\":" + String(mint_s) + ",\"ingest_s\":" + String(ingest_s)
                    if fallback_field != "":
                        kv_extra += ",\"fallback\":\"" + fallback_field + "\""
                    else:
                        kv_extra += ",\"fallback\":null"
                    out += arm_json(arm, a_s, receiver_s, gen_ids, text, scored, sv, "", kv_extra)

                else:
                    out += arm_json(arm, 0.0, 0.0, List[Int](), "", "", -1, "unknown arm " + arm)
            except e:
                out += arm_json(arm, 0.0, 0.0, List[Int](), "", "", -1, String(e))

        out += "]}"

    out += "]}"

    with open(out_prefix + ".raw.json", "w") as f:
        f.write_bytes(out.as_bytes())
    print("wrote", out_prefix + ".raw.json")
