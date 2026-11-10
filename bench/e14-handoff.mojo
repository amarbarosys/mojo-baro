"""E14: one reader prefills a document once, N followers each ingest that
reader's KV+SSM state and answer a different question about it
(bench/e14-handoff.mojo).

briefs/2026-09-15-p3-b2-e14.md, ~/AMDHQ/docs/design/latent-os/06-experiments.md
E14. Not bench_latent_handoff.mojo's per-item loop: that harness always calls
reset_and_load on the producer with each item's full context, re-decoding the
document from position 0 on every item (BARO_E8_SHARE_A only reuses a
producer decode within one item's arms, never across items). E14's whole
point is that the reader decodes ONCE; this tool prefills the document a
single time, then loops the KV mint/ingest and the "Text" full re-prefill
baseline per follower, reusing the reader's unchanged buffers across all of
them. Built against the CURRENT engine API (serve/harness.mojo,
serve/prefix.mojo, serve/window.mojo's WindowCfg with dump4/dump_layer/sample)
rather than copied from bench_latent_handoff.mojo, which no longer compiles
against this tree (WindowCfg and Chain both gained fields since it was
written; confirmed by attempting its build before writing this file).

Arms, both run per follower in one process (one pack load pays once):
  KV    reader's KV pages + SSM checkpoint, minted once per follower from the
        reader's unchanged buffers, ingested into a fresh receiver, which then
        prefills only the follower's own question suffix and decodes.
  Text  a fresh receiver re-reads the ENTIRE document plus the follower's own
        question from position 0, no state shared with the reader.
Void check: the reader's wst.pos must still equal the document length
immediately before every mint (an assertion, not a soft check) -- if the
reader's buffers were ever touched between followers, its position would have
moved off the handoff point, since nothing else in this process alters it.
The llama.cpp arm is a separate tool (bench/e14-llama-run.sh); this binary
does not talk to llama.cpp.

Input: a JSON file from bench/e14-data.py, {"document_tokens": [...],
"questions": [{"id","key","expected","tokens": [...]}, ...]}. document_tokens
ends mid chat-turn (user message still open); each question's tokens are a
separately-encoded suffix (closes the user turn, opens the assistant turn),
concatenated onto document_tokens -- the same convention
bench_latent_handoff.mojo used for its hand_ids, so a per-follower context is
document_tokens + questions[i].tokens, never re-tokenized as one string.

Usage: bench/e14-handoff.mojo [--doc PATH] [--out PREFIX] [--gen-budget N]
[--tmax N] [--pack DIR] [--gguf PATH] (build then run under gpu-wait).
"""
from std.os import getenv
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer

from registry import *
from window import *
from harness import *
from prefix import Chain
from latent import mint_kv_latent, ingest_kv_latent, mint_chain_slot, ingest_into_chain
from serve_proto import default_sample_params
from tokenizer import Tokenizer

from grammar.json_value import parse_json_file, JSONDoc
from bench_latent_handoff import reset_and_load, get_int_list, read_toks, contains, json_escape, get_str, trim_at_stop, extend_ids, ids_to_json


def fresh_state() -> WindowState:
    return WindowState(pos=0, pos_prev=0, ring=0, n_drafted=0, n_accepted=0, n_spec_windows=0, n_dumped=0, tp=0, tq=0, pf_att=0, pf_ssm=0, pf_ffn=0, pf_head=0, pf_proc=0, pf_draft=0, fc=[0, 0, 0, 0, 0, 0], pc=[0, 0, 0, 0, 0, 0, 0, 0], ac=[0, 0, 0, 0, 0, 0], p3=[0, 0, 0, 0], pfx=[0, 0, 0, 0], grammar=None, grammar_mask=Bitset(1), grammar_pending_think=False, grammar_think_buf=List[UInt8](), grammar_stop=False, grammar_masked_draws=0, grammar_accepted=0, tool_re="", tool_buf=List[UInt8](), grammar_tool=False)


def make_cfg(pack_q4: Bool, q4_off: Int, e: Int, pf_rows: Int, pf_tail: Int, n_prompt: Int, n_total: Int) -> WindowCfg:
    return WindowCfg(
        pack_q4=pack_q4, draft_q4=False, q4_off=q4_off, fr_k=0, fr_off=0, fr_ids_off=0, e=e, kcfg=2,
        spec=False, spec_dbg=False, expert_trace=False, serve=False, req_id=0, prof=False, pf2=False, pf3=False, pf4=False,
        dump=False, dump4=False, dump_layer=0, mega=True, att_split=n_total, mega_win=False, dot3=False,
        pf_chunk=CP, pf_rows=pf_rows, pf_tail=pf_tail, n_total=n_total, replay_m=8, mrope=False, inject=False, n_prompt=n_prompt,
        sample=default_sample_params(), dump_pen=False,
    )


def run_fresh_generate(
    ctx: DeviceContext, mut b: WindowBufs, pack_q4: Bool, q4_off: Int, e: Int,
    context: List[Int], gen_max: Int, tmax: Int,
) raises -> Tuple[List[Int], Float64]:
    var plen = len(context)
    var pf = reset_and_load(ctx, b, context, tmax)
    var cfg = make_cfg(pack_q4, q4_off, e, pf[0], pf[1], plen, plen + gen_max)
    var wst = fresh_state()
    var t0 = perf_counter_ns()
    wst.reset(t0)
    while wst.pos < plen + gen_max - 1:
        step_window(ctx, b, cfg, wst)
    ctx.synchronize()
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    var generated = read_toks(ctx, b, plen, plen + gen_max, tmax)
    return (generated^, dt)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var args = argv()
    var doc_path = String(".work/e14/doc.json")
    var out_prefix = String("results/e14/e14")
    var gen_budget = 64
    var tmax = 33280
    var packdir = String(".work/engine-pack-q4")
    var gguf_path = getenv("HOME", "") + "/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf"
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--doc" and i + 1 < len(args):
            doc_path = String(args[i + 1]); i += 2
        elif a == "--out" and i + 1 < len(args):
            out_prefix = String(args[i + 1]); i += 2
        elif a == "--gen-budget" and i + 1 < len(args):
            gen_budget = atol(String(args[i + 1])); i += 2
        elif a == "--tmax" and i + 1 < len(args):
            tmax = atol(String(args[i + 1])); i += 2
        elif a == "--pack" and i + 1 < len(args):
            packdir = String(args[i + 1]); i += 2
        elif a == "--gguf" and i + 1 < len(args):
            gguf_path = String(args[i + 1]); i += 2
        else:
            i += 1

    print("E14 HARNESS: doc=", doc_path, " out=", out_prefix, " gen_budget=", gen_budget, " tmax=", tmax, " pack=", packdir)

    var ctx = DeviceContext()

    print("--- loading reader (A) pack ---")
    var packA = load_pack(ctx, packdir)
    var bufsA = alloc_bufs(ctx, packA, tmax)
    print("--- loading follower (B) pack ---")
    var packB = load_pack(ctx, packdir)
    var bufsB = alloc_bufs(ctx, packB, tmax)
    with open(".work/e14-vram-ready.marker", "w") as f:
        f.write_bytes(String("1").as_bytes())

    var pack_q4 = packA.pack_q4
    var q4_off = packA.q4_off
    var eA = packA.e
    var eB = packB.e

    var tok = Tokenizer(gguf_path)
    var stops = List[Int]()
    if tok.eos_id >= 0:
        stops.append(tok.eos_id)
    var im_end = tok.tok2id.get("<|im_end|>", -1)
    if im_end >= 0 and not contains(stops, im_end):
        stops.append(im_end)
    # Suppress reasoning so the answer fits the budget (bench_latent_handoff.mojo's
    # BARO_E8_NOTHINK=1 default, same precedent): a prefilled empty <think> block
    # closed before B generates, real ids like every other suffix in this file.
    var nothink_ids = tok.encode(String("<think>\n\n</think>\n\n"), add_special=False)

    var doc = parse_json_file(doc_path)
    var root = doc.root
    var document_tokens = get_int_list(doc, root, "document_tokens")
    var q_field = doc.get_field(root, "questions")
    var q_arr = doc.get(q_field)
    var n_q = len(q_arr.arr)
    print("document tokens:", len(document_tokens), " questions:", n_q)

    var chainA = Chain(ctx, 1, packdir)
    var chainB = Chain(ctx, 1, packdir)

    # ---- reader: prefill the document ONCE, timed ----
    var pfA = reset_and_load(ctx, bufsA, document_tokens, tmax)
    var doc_len = len(document_tokens)
    var cfgA = make_cfg(pack_q4, q4_off, eA, pfA[0], pfA[1], doc_len, doc_len + 1)
    var wstA = fresh_state()
    var t0r = perf_counter_ns()
    wstA.reset(t0r)
    while wstA.pos < doc_len:
        step_window(ctx, bufsA, cfgA, wstA)
    ctx.synchronize()
    var reader_prefill_s = Float64(perf_counter_ns() - t0r) / 1e9
    var hand_pos = wstA.pos
    if hand_pos != doc_len:
        raise Error("reader prefill ended at " + String(hand_pos) + ", expected " + String(doc_len))
    print("reader prefill:", reader_prefill_s, "s  hand_pos:", hand_pos)

    chainA.save(ctx, bufsA.convstate_d, bufsA.sstate_d, wstA.ring, hand_pos, document_tokens, True, True)
    ctx.synchronize()
    chainA.commit()
    if not chainA.items[0].valid or chainA.items[0].pos != hand_pos:
        raise Error("reader SSM checkpoint not saved at hand_pos")

    var kv_pages = (hand_pos + KVPAGE - 1) // KVPAGE

    var out = String("{\"topology\":\"e14\",\"pack\":\"") + packdir + "\",\"gguf\":\"" + gguf_path + "\""
    out += ",\"tmax\":" + String(tmax) + ",\"gen_budget\":" + String(gen_budget)
    out += ",\"reader_prefill_s\":" + String(reader_prefill_s) + ",\"hand_pos\":" + String(hand_pos)
    out += ",\"doc_tokens\":" + String(doc_len) + ",\"followers\":["

    for j in range(n_q):
        var qi = q_arr.arr[j]
        var q_id = get_str(doc, qi, "id")
        var q_key = get_str(doc, qi, "key")
        var q_expected = get_str(doc, qi, "expected")
        var q_tokens = get_int_list(doc, qi, "tokens")
        print("=== follower", q_id, "key=" + q_key, "===")

        var b_ctx = document_tokens.copy()
        extend_ids(b_ctx, q_tokens)
        extend_ids(b_ctx, nothink_ids)
        var total = len(b_ctx)
        if total - 1 - hand_pos < MROWS:
            raise Error("KV: fewer than MROWS tokens after the handoff for " + q_id)

        # ---- void check: the reader was never re-touched between followers ----
        if wstA.pos != hand_pos:
            raise Error("void: reader wst.pos drifted to " + String(wstA.pos) + " before follower " + q_id)

        if j > 0:
            out += ","
        out += "{\"id\":\"" + json_escape(q_id) + "\",\"key\":\"" + json_escape(q_key) + "\",\"expected\":\"" + json_escape(q_expected) + "\""

        # ---- KV arm: mint from the reader's unchanged buffers, ingest, decode only the tail ----
        var t0m = perf_counter_ns()
        var kv = mint_kv_latent(ctx, bufsA.kc_d, bufsA.vc_d, 0, kv_pages, UInt64(hand_pos))
        var kv_header = kv[0].copy()
        var kv_fd = kv[1]
        var ck = mint_chain_slot(chainA, 0)
        var ck_header = ck[0].copy()
        var ck_fd = ck[1]
        var mint_s = Float64(perf_counter_ns() - t0m) / 1e9

        var pfB = reset_and_load(ctx, bufsB, b_ctx, tmax)
        var t0b = perf_counter_ns()
        ingest_kv_latent(ctx, bufsB.kc_d, bufsB.vc_d, kv_header, kv_fd)
        var slot = ingest_into_chain(chainB, ck_header, ck_fd)
        chainB.restore(ctx, bufsB.convstate_d, bufsB.sstate_d, 0, slot)
        ctx.synchronize()
        var ingest_s = Float64(perf_counter_ns() - t0b) / 1e9
        var wstB = fresh_state()
        wstB.reset(t0b)
        wstB.pos = hand_pos
        wstB.pos_prev = hand_pos
        var cfgB = make_cfg(pack_q4, q4_off, eB, pfB[0], pfB[1], total, total + gen_budget)
        while wstB.pos < total + gen_budget - 1:
            step_window(ctx, bufsB, cfgB, wstB)
        ctx.synchronize()
        var kv_receiver_s = Float64(perf_counter_ns() - t0b) / 1e9
        var kv_gen_ids = read_toks(ctx, bufsB, total, total + gen_budget, tmax)
        var kv_ans = trim_at_stop(kv_gen_ids, stops)
        var kv_text = tok.decode(kv_ans)
        out += ",\"kv\":{\"mint_s\":" + String(mint_s) + ",\"ingest_s\":" + String(ingest_s)
        out += ",\"receiver_s\":" + String(kv_receiver_s) + ",\"generated_ids\":" + ids_to_json(kv_gen_ids)
        out += ",\"answer_text\":\"" + json_escape(kv_text) + "\"}"

        # ---- Text arm: fresh full re-prefill of document + this follower's question ----
        var tr = run_fresh_generate(ctx, bufsB, pack_q4, q4_off, eB, b_ctx, gen_budget, tmax)
        var text_gen_ids = tr[0].copy()
        var text_receiver_s = tr[1]
        var text_ans = trim_at_stop(text_gen_ids, stops)
        var text_text = tok.decode(text_ans)
        out += ",\"text\":{\"receiver_s\":" + String(text_receiver_s) + ",\"generated_ids\":" + ids_to_json(text_gen_ids)
        out += ",\"answer_text\":\"" + json_escape(text_text) + "\"}"

        out += "}"
        print("  KV receiver_s:", kv_receiver_s, " ingest_s:", ingest_s, " Text receiver_s:", text_receiver_s)

    out += "]}"

    with open(out_prefix + ".raw.json", "w") as f:
        f.write_bytes(out.as_bytes())
    print("wrote", out_prefix + ".raw.json")
