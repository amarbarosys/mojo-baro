# E11b HARNESS -- activation graft, one resident engine, two stages
# (protocol: ~/AMDHQ/docs/design/latent-os/06-experiments.md "## E11b:", frozen
# 2026-09-21; record schema: ~/Brain/AMDHQ/briefs/2026-09-21-e11b-contract.md).
#
#   E11B_STAGE=A  prefill every A prompt (items and twins) alone, tap the 32
#                 post-layer residuals at the last prompt token into the graft
#                 bank, then write the instructed T relay (<= 48 tokens) for
#                 every non-twin item.
#   E11B_STAGE=B  read the bank, run arms E11B_ARMS (0 T S G C X M I K) over
#                 E11B_LAYERS, greedy, stop-aware, <= 64 tokens -> records.jsonl.
#                 K (sensitivity arm, 06 amendment 2026-09-24 (3)): the vector is
#                 the residual of arm S's prompt, a sender that knows the answer.
#   E11B_STAGE=P  void check 2 on the first E11B_LIMIT (20) dev items at every
#                 layer of E11B_LAYERS; any mismatch prints FAIL and exits 1.
#
# Every arm runs B's final prompt token through the same m = 1 layer-loop step
# (mega off for that one step), with the tap on; arms differ only in the vector.
# The tap is the engine's existing cfg.dump plane: slot 2*l+1 of b.dump_h is the
# residual after layer l. The inject is step_window's graft_layer / graft_mean /
# graft_d. arm.txt is written before the first engine step of every stage.
#
# Build: mojo build bench/e11b/e11b_harness.mojo -I . -I kernels -I serve -I bench
from std.os import getenv, makedirs
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from registry import *
from window import *
from harness import load_pack, alloc_bufs, Pack
from tokenizer import Tokenizer
from serve_proto import default_sample_params
from sha256 import sha256
from minja import render_chat
from grammar.json_value import parse_json_bytes, JSONDoc

comptime RELAY_MAX = 48
comptime RECV_MAX = 64
comptime VEC_BYTES = H * 4
comptime REC_BYTES = N_LAYERS * VEC_BYTES
comptime RELAY_ASK = "Tell your partner what they need to know, in one sentence."


@fieldwise_init
struct Item(Copyable, Movable):
    var id: String
    var game: String
    var split: String
    var is_twin: Bool
    var twin_id: String
    var a_text: String
    var b_question: String
    var gold: String


@fieldwise_init
struct Run(Movable):
    var ids: List[Int]
    var prompt_back: List[Int]


def hex32(d: Array[UInt8, 32]) -> String:
    var digits = String("0123456789abcdef").as_bytes()
    var out = List[UInt8]()
    for i in range(32):
        out.append(digits[Int(d[i] >> 4)])
        out.append(digits[Int(d[i] & 15)])
    return String(unsafe_from_utf8=Span(out))


def sha_file(path: String) raises -> String:
    with open(path, "r") as f:
        var data = f.read_bytes()
        return hex32(sha256(Span(data)))


def json_escape(s: String) -> String:
    var hexd = String("0123456789abcdef").as_bytes()
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
            out.append(92); out.append(117); out.append(48); out.append(48)
            out.append(hexd[Int(c >> 4)]); out.append(hexd[Int(c & 15)])
        else:
            out.append(c)
    return String(unsafe_from_utf8=Span(out))


def field_str(doc: JSONDoc, key: String) raises -> String:
    var vi = doc.get_field(doc.root, key)
    if vi < 0:
        raise Error("items: missing field " + key)
    return doc.get(vi).s


def load_items(path: String) raises -> List[Item]:
    var text: String
    with open(path, "r") as f:
        text = f.read()
    var out = List[Item]()
    var lines = text.split("\n")
    for li in range(len(lines)):
        var line = String(lines[li])
        if line.byte_length() < 2:
            continue
        var doc = parse_json_bytes(List[UInt8](line.as_bytes()))
        var ti = doc.get_field(doc.root, "is_twin")
        if ti < 0:
            raise Error("items: missing field is_twin")
        out.append(Item(
            id=field_str(doc, "id"), game=field_str(doc, "game"), split=field_str(doc, "split"),
            is_twin=doc.get(ti).b, twin_id=field_str(doc, "twin_id"), a_text=field_str(doc, "a_text"),
            b_question=field_str(doc, "b_question"), gold=field_str(doc, "gold"),
        ))
    return out^


def find_item(items: List[Item], id: String) raises -> Int:
    for i in range(len(items)):
        if items[i].id == id:
            return i
    raise Error("items: no record with id " + id)


def chat_ids(tok: Tokenizer, content: String) raises -> List[Int]:
    var case_json = String("{\"messages\":[{\"role\":\"user\",\"content\":\"") + json_escape(content) + "\"}],\"enable_thinking\":false}"
    var text = render_chat(tok.chat_template, case_json, tok.token_str(tok.bos_id), tok.token_str(tok.eos_id), tok.token_str(tok.pad_id))
    return tok.encode(text)


def make_cfg(pack: Pack, pf_rows: Int, pf_tail: Int, n_prompt: Int, n_total: Int, mega: Bool, dump: Bool) -> WindowCfg:
    return WindowCfg(
        pack_q4=pack.pack_q4, draft_q4=False, q4_off=pack.q4_off, e=pack.e, kcfg=2,
        spec=False, expert_trace=False, spec_dbg=False, serve=False, req_id=0, prof=False, pf2=False, pf3=False, pf4=False,
        dump=dump, dump4=False, dump_layer=0, mega=mega, att_split=n_total, mega_win=False, dot3=False, pf_chunk=CP,
        pf_rows=pf_rows, pf_tail=pf_tail, n_total=n_total, fr_k=0, fr_off=0, fr_ids_off=0, replay_m=8, mrope=False, inject=False, n_prompt=n_prompt,
        sample=default_sample_params(), dump_pen=False,
    )


def contains(xs: List[Int], v: Int) -> Bool:
    for i in range(len(xs)):
        if xs[i] == v:
            return True
    return False


def run_arm(
    ctx: DeviceContext, mut b: WindowBufs, mut wst: WindowState, pack: Pack, toks_h: HostBuffer[DType.int32],
    prompt: List[Int], gen_max: Int, tmax: Int, stops: List[Int], mega: Bool, tap: Bool,
    graft_layer: Int, graft_mean: Bool, graft_d: DeviceBuffer[f32],
) raises -> Run:
    var plen = len(prompt)
    if plen + gen_max + 1 > tmax:
        raise Error("run_arm: prompt " + String(plen) + " + gen " + String(gen_max) + " exceeds tmax " + String(tmax))
    ctx.enqueue_memset(b.convstate_d, 0)
    ctx.enqueue_memset(b.sstate_d, 0)
    ctx.enqueue_memset(b.kc_d, 0)
    ctx.enqueue_memset(b.vc_d, 0)
    ctx.enqueue_memset(b.kc32_d, 0)
    ctx.enqueue_memset(b.vc32_d, 0)
    ctx.enqueue_memset(b.ctr_d, 0)
    ctx.enqueue_memset(b.prof_d, 0)
    for i in range(tmax):
        toks_h[i] = 0
    for i in range(plen):
        toks_h[i] = Int32(prompt[i])
    var toks_dev = DeviceBuffer[DType.int32](ctx, b.toks_d.unsafe_ptr(), tmax, owning=False)
    ctx.enqueue_copy(dst_buf=toks_dev, src_buf=toks_h)
    ctx.synchronize()
    var pf_rows = 0
    var pf_tail = 0
    if plen - 1 >= PF_MIN:
        pf_rows = plen - 1
        pf_tail = pf_rows % MROWS
        if pf_tail == 0:
            pf_tail = MROWS
    var n_total = plen + gen_max + 1
    var cfg_run = make_cfg(pack, pf_rows, pf_tail, plen, n_total, mega, False)
    var cfg_graft = make_cfg(pack, pf_rows, pf_tail, plen, n_total, False, tap)
    wst.reset(perf_counter_ns())
    while wst.pos < plen - 1:
        step_window(ctx, b, cfg_run, wst)
    if wst.pos != plen - 1:
        raise Error("run_arm: prompt replay stopped at " + String(wst.pos) + ", graft position is " + String(plen - 1))
    wst.n_dumped = 0
    if graft_layer >= 0:
        step_window(ctx, b, cfg_graft, wst, 0, graft_layer, graft_mean, graft_d)
    else:
        step_window(ctx, b, cfg_graft, wst)
    var tok_h = ctx.enqueue_create_host_buffer[DType.int32](1)
    var ids = List[Int]()
    while len(ids) < gen_max:
        var cur = DeviceBuffer[DType.int32](ctx, b.toks_d.unsafe_ptr() + wst.pos, 1, owning=False)
        ctx.enqueue_copy(dst_buf=tok_h, src_buf=cur)
        ctx.synchronize()
        var t = Int(tok_h[0])
        if contains(stops, t):
            break
        ids.append(t)
        if len(ids) == gen_max:
            break
        step_window(ctx, b, cfg_run, wst)
        if wst.pos != plen + len(ids):
            raise Error("run_arm: decode advanced to " + String(wst.pos) + ", expected " + String(plen + len(ids)))
    ctx.enqueue_copy(dst_buf=toks_h, src_buf=toks_dev)
    ctx.synchronize()
    var back = List[Int]()
    for i in range(plen):
        back.append(Int(toks_h[i]))
    return Run(ids^, back^)


def tapped_record(b: WindowBufs) -> List[UInt8]:
    # All 32 post-layer residuals of the last graft step, layer-major.
    var out = List[UInt8](capacity=REC_BYTES)
    var p = b.dump_h.unsafe_ptr().unsafe_bitcast[UInt8]()
    for l in range(N_LAYERS):
        var base = (2 * l + 1) * VEC_BYTES
        for i in range(VEC_BYTES):
            out.append(p[base + i])
    return out^


def fill_graft(ctx: DeviceContext, graft_h: HostBuffer[f32], graft_d: DeviceBuffer[f32], rec: List[UInt8], rec_off: Int, layer: Int) raises:
    var p = graft_h.unsafe_ptr().unsafe_bitcast[UInt8]()
    var base = rec_off + layer * VEC_BYTES
    for i in range(VEC_BYTES):
        p[i] = rec[base + i]
    ctx.enqueue_copy(dst_buf=graft_d, src_buf=graft_h)
    ctx.synchronize()


def readback_diff(b: WindowBufs, rec: List[UInt8], rec_off: Int, layer: Int) -> Int:
    # Bytes of the tapped residual after `layer` that differ from the record.
    var p = b.dump_h.unsafe_ptr().unsafe_bitcast[UInt8]()
    var src = (2 * layer + 1) * VEC_BYTES
    var base = rec_off + layer * VEC_BYTES
    var n = 0
    for i in range(VEC_BYTES):
        if p[src + i] != rec[base + i]:
            n += 1
    return n


def same_ids(a: List[Int], c: List[Int]) -> Bool:
    if len(a) != len(c):
        return False
    for i in range(len(a)):
        if a[i] != c[i]:
            return False
    return True


def parse_layers(s: String) raises -> List[Int]:
    var out = List[Int]()
    if s == "":
        return out^
    var parts = s.split(",")
    for i in range(len(parts)):
        var l = atol(String(parts[i]))
        if l < 0 or l >= N_LAYERS:
            raise Error("E11B_LAYERS: layer " + String(l) + " outside 0.." + String(N_LAYERS - 1))
        out.append(l)
    return out^


def record_line(item: String, arm: String, layer: Int, vec_src: String, out_text: String, n_tokens: Int, relay: String) -> String:
    var s = String("{\"item\":\"") + json_escape(item) + "\",\"arm\":\"" + arm + "\",\"L\":"
    s += String(layer) if layer >= 0 else String("null")
    s += ",\"vec_src\":"
    s += (String("\"") + json_escape(vec_src) + "\"") if vec_src != "" else String("null")
    s += ",\"out\":\"" + json_escape(out_text) + "\",\"n_tokens\":" + String(n_tokens)
    if arm == "T":
        s += ",\"relay\":\"" + json_escape(relay) + "\""
    s += "}\n"
    return s


def load_relays(path: String) raises -> Dict[String, String]:
    var out = Dict[String, String]()
    var text: String
    with open(path, "r") as f:
        text = f.read()
    var lines = text.split("\n")
    for li in range(len(lines)):
        var line = String(lines[li])
        if line.byte_length() < 2:
            continue
        var doc = parse_json_bytes(List[UInt8](line.as_bytes()))
        out[field_str(doc, "item")] = field_str(doc, "relay")
    return out^


def main() raises:
    var stage = getenv("E11B_STAGE", "")
    if stage != "A" and stage != "B" and stage != "P":
        raise Error("FAIL args: E11B_STAGE must be A, B or P")
    var items_path = getenv("E11B_ITEMS", "bench/e11b/fixtures/stub_items.jsonl")
    var out_dir = getenv("E11B_OUT", "")
    if out_dir == "":
        raise Error("FAIL args: E11B_OUT (output directory) is required")
    var bank_path = getenv("E11B_BANK", ".work/e11b/bank.f32")
    var relay_path = getenv("E11B_RELAYS", ".work/e11b/relays.jsonl")
    var split = getenv("E11B_SPLIT", "")
    var game = getenv("E11B_GAME", "")
    var limit = atol(getenv("E11B_LIMIT", "0"))
    var arms_arg = getenv("E11B_ARMS", "0,T,S,G,C,X,M,I")
    var layers = parse_layers(getenv("E11B_LAYERS", ""))
    var tmax = atol(getenv("E11B_TMAX", "512"))
    var mega = getenv("E11B_MEGA", "1") == "1"
    var packdir = getenv("BARO_PACK", ".work/engine-pack-q4")
    var gguf_path = getenv(
        "E11B_GGUF",
        getenv("HOME", "") + "/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf",
    )
    var arms = List[String]()
    var arm_parts = arms_arg.split(",")
    for i in range(len(arm_parts)):
        var a = String(arm_parts[i])
        if a != "0" and a != "T" and a != "S" and a != "G" and a != "C" and a != "X" and a != "M" and a != "I" and a != "K":
            raise Error("FAIL args: unknown arm " + a)
        arms.append(a)
    if (stage == "P" or stage == "B") and len(layers) == 0:
        var need = stage == "P"
        for i in range(len(arms)):
            if arms[i] != "0" and arms[i] != "T" and arms[i] != "S":
                need = True
        if need:
            raise Error("FAIL args: E11B_LAYERS is required for graft arms and for stage P")

    makedirs(out_dir, exist_ok=True)
    var items = load_items(items_path)
    var items_sha = sha_file(items_path)
    var tok = Tokenizer(gguf_path)
    var stops = List[Int]()
    if tok.eos_id >= 0:
        stops.append(tok.eos_id)
    var im_end = tok.tok2id.get("<|im_end|>", -1)
    if im_end >= 0 and not contains(stops, im_end):
        stops.append(im_end)

    # The scored set: non-twin records passing the split/game filter, in file
    # order, first `limit`. Stage A also takes each selected item's twin.
    var sel = List[Int]()
    for i in range(len(items)):
        if items[i].is_twin:
            continue
        if split != "" and items[i].split != split:
            continue
        if game != "" and items[i].game != game:
            continue
        if limit > 0 and len(sel) >= limit:
            break
        sel.append(i)
    if len(sel) == 0:
        raise Error("FAIL items: no record of " + items_path + " passes split='" + split + "' game='" + game + "'")

    var probe = chat_ids(tok, items[sel[0]].b_question)
    var last_tok = probe[len(probe) - 1]

    var bank = List[UInt8]()
    var bank_sha = String("none")
    var relays = Dict[String, String]()
    if stage == "B":
        with open(bank_path, "r") as f:
            bank = f.read_bytes()
        bank_sha = hex32(sha256(Span(bank)))
        if len(bank) != len(items) * REC_BYTES:
            raise Error("FAIL bank: " + bank_path + " holds " + String(len(bank)) + " bytes, items file needs " + String(len(items) * REC_BYTES))
        if contains_str(arms, "T"):
            relays = load_relays(relay_path)

    var layer_s = String("")
    for i in range(len(layers)):
        layer_s += ("," if i > 0 else "") + String(layers[i])
    var arm_txt = String("stage=") + stage + "\n"
    arm_txt += "arms=" + (arms_arg if stage == "B" else String("n/a")) + "\n"
    arm_txt += "layers=" + layer_s + "\n"
    arm_txt += "function=G,C,X,I:replace M:mean" + (" P:replace+mean" if stage == "P" else "") + "\n"
    arm_txt += "items=" + items_path + " sha256=" + items_sha + " records=" + String(len(items)) + "\n"
    arm_txt += "item_set=split:'" + split + "' game:'" + game + "' limit:" + String(limit) + " selected:" + String(len(sel)) + " first:" + items[sel[0]].id + " last:" + items[sel[len(sel) - 1]].id + "\n"
    arm_txt += "bank=" + bank_path + " sha256=" + bank_sha + "\n"
    arm_txt += "relays=" + relay_path + "\n"
    arm_txt += "recv_max=" + String(RECV_MAX) + " relay_max=" + String(RELAY_MAX) + " tmax=" + String(tmax) + " greedy=1 stop_ids=" + String(len(stops)) + "\n"
    arm_txt += "graft_step=layer-loop(mega off) tap=on-every-arm decode_mega=" + String(mega) + "\n"
    arm_txt += "pack=" + packdir + " index_sha256=" + sha_file(packdir + "/index.txt") + " pack_bin_sha256=" + getenv("E11B_PACK_SHA", "not-given") + "\n"
    arm_txt += "gguf=" + gguf_path + "\n"
    arm_txt += "engine_commit=" + getenv("E11B_COMMIT", "not-given") + " dirty='" + getenv("E11B_DIRTY", "not-given") + "'\n"
    arm_txt += "dev_receipt_commit=" + getenv("E11B_DEV_COMMIT", "n/a") + "\n"
    arm_txt += "graft_position=last prompt token id " + String(last_tok) + " '" + json_escape(tok.token_str(last_tok)) + "' probe_prompt_tokens=" + String(len(probe)) + "\n"
    arm_txt += "layers_total=" + String(N_LAYERS) + " hidden=" + String(H) + " vec_bytes=" + String(VEC_BYTES) + "\n"
    with open(out_dir + "/arm.txt", "w") as f:
        f.write_bytes(arm_txt.as_bytes())
    print(arm_txt)

    var ctx = DeviceContext()
    var pack = load_pack(ctx, packdir)
    var b = alloc_bufs(ctx, pack, tmax)
    var wst = WindowState(pos=0, pos_prev=0, ring=0, n_drafted=0, n_accepted=0, n_spec_windows=0, n_dumped=0, tp=0, tq=0, pf_att=0, pf_ssm=0, pf_ffn=0, pf_head=0, pf_proc=0, pf_draft=0, fc=[0, 0, 0, 0, 0, 0], pc=[0, 0, 0, 0, 0, 0, 0, 0], ac=[0, 0, 0, 0, 0, 0], p3=[0, 0, 0, 0], pfx=[0, 0, 0, 0], grammar=None, grammar_mask=Bitset(1), grammar_pending_think=False, grammar_think_buf=List[UInt8](), grammar_stop=False, grammar_masked_draws=0, grammar_accepted=0, tool_re="", tool_buf=List[UInt8](), grammar_tool=False)
    var toks_h = ctx.enqueue_create_host_buffer[DType.int32](tmax)
    var graft_h = ctx.enqueue_create_host_buffer[f32](H)
    var graft_d = ctx.enqueue_create_buffer[f32](H)
    ctx.synchronize()
    var t0 = perf_counter_ns()

    if stage == "A":
        # Bank layout: record i = items-file record i, 32 x 4096 f32. Records
        # outside the selected set (and their twins) stay zero.
        bank = List[UInt8](length=len(items) * REC_BYTES, fill=0)
        var todo = List[Int]()
        for s in range(len(sel)):
            todo.append(sel[s])
            if items[sel[s]].twin_id != "":
                todo.append(find_item(items, items[sel[s]].twin_id))
        var bad_last = 0
        for k in range(len(todo)):
            var it = items[todo[k]].copy()
            var prompt = chat_ids(tok, it.a_text)
            if prompt[len(prompt) - 1] != last_tok:
                bad_last += 1
            _ = run_arm(ctx, b, wst, pack, toks_h, prompt, 0, tmax, stops, mega, True, -1, False, graft_d)
            var rec = tapped_record(b)
            var base = todo[k] * REC_BYTES
            for i in range(REC_BYTES):
                bank[base + i] = rec[i]
        with open(bank_path, "w") as f:
            f.write_bytes(bank)
        bank_sha = hex32(sha256(Span(bank)))
        var relay_out = String("")
        for s in range(len(sel)):
            var it = items[sel[s]].copy()
            var prompt = chat_ids(tok, it.a_text + "\n\n" + RELAY_ASK)
            var r = run_arm(ctx, b, wst, pack, toks_h, prompt, RELAY_MAX, tmax, stops, mega, True, -1, False, graft_d)
            relay_out += String("{\"item\":\"") + json_escape(it.id) + "\",\"relay\":\"" + json_escape(tok.decode(r.ids)) + "\",\"n_tokens\":" + String(len(r.ids)) + "}\n"
        with open(relay_path, "w") as f:
            f.write_bytes(relay_out.as_bytes())
        with open(out_dir + "/relays.jsonl", "w") as f:
            f.write_bytes(relay_out.as_bytes())
        var summary = String("stage=A a_prompts=") + String(len(todo)) + " relays=" + String(len(sel)) + " bank_bytes=" + String(len(bank)) + " bank_sha256=" + bank_sha + " last_token_mismatch=" + String(bad_last) + " elapsed_s=" + String(Float64(perf_counter_ns() - t0) / 1e9) + "\n"
        with open(out_dir + "/summary.txt", "w") as f:
            f.write_bytes(summary.as_bytes())
        print(summary)
        if bad_last > 0:
            raise Error("FAIL stage A: " + String(bad_last) + " A prompts do not end in the graft token")
        print("E11B stage A OK")
        return

    if stage == "P":
        var n_checks = 0
        var n_bad = 0
        var log = String("")
        var prev_rec = List[UInt8]()
        var n_prefill = 0
        for s in range(len(sel)):
            var it = items[sel[s]].copy()
            var prompt = chat_ids(tok, it.b_question)
            if len(prompt) - 1 >= PF_MIN:
                n_prefill += 1
            var off = run_arm(ctx, b, wst, pack, toks_h, prompt, RECV_MAX, tmax, stops, mega, False, -1, False, graft_d)
            var on = run_arm(ctx, b, wst, pack, toks_h, prompt, RECV_MAX, tmax, stops, mega, True, -1, False, graft_d)
            var own = tapped_record(b)
            n_checks += 1
            if not same_ids(off.ids, on.ids):
                n_bad += 1
                log += "MISMATCH " + it.id + " tap-on ids != tap-off ids\n"
            for li in range(len(layers)):
                var L = layers[li]
                for fn_i in range(2):
                    var is_mean = fn_i == 1
                    fill_graft(ctx, graft_h, graft_d, own, 0, L)
                    var r = run_arm(ctx, b, wst, pack, toks_h, prompt, RECV_MAX, tmax, stops, mega, True, L, is_mean, graft_d)
                    var diff = readback_diff(b, own, 0, L)
                    n_checks += 2
                    var name = String("mean(v,v)") if is_mean else String("I")
                    if not same_ids(r.ids, off.ids):
                        n_bad += 1
                        log += "MISMATCH " + it.id + " " + name + "-L" + String(L) + " ids != arm 0 ids\n"
                    if diff != 0:
                        n_bad += 1
                        log += "MISMATCH " + it.id + " " + name + "-L" + String(L) + " readback differs in " + String(diff) + " bytes\n"
            # Falsifier: a foreign vector must show up in the readback, else
            # the inject is inert and every identity above is vacuous.
            if len(prev_rec) > 0:
                var L0 = layers[0]
                fill_graft(ctx, graft_h, graft_d, prev_rec, 0, L0)
                _ = run_arm(ctx, b, wst, pack, toks_h, prompt, 1, tmax, stops, mega, True, L0, False, graft_d)
                n_checks += 2
                if readback_diff(b, prev_rec, 0, L0) != 0:
                    n_bad += 1
                    log += "MISMATCH " + it.id + " foreign vector at L" + String(L0) + " not read back\n"
                if readback_diff(b, own, 0, L0) == 0:
                    n_bad += 1
                    log += "MISMATCH " + it.id + " foreign vector at L" + String(L0) + " equals own residual, falsifier has no power\n"
            prev_rec = own^
            print("P item", s + 1, "/", len(sel), it.id, "tokens", len(off.ids), "bad so far", n_bad)
        var summary = String("stage=P items=") + String(len(sel)) + " layers=" + layer_s + " prefill_path_items=" + String(n_prefill) + " replay_path_items=" + String(len(sel) - n_prefill) + " checks=" + String(n_checks) + " mismatches=" + String(n_bad) + " elapsed_s=" + String(Float64(perf_counter_ns() - t0) / 1e9) + "\n"
        with open(out_dir + "/summary.txt", "w") as f:
            f.write_bytes((summary + log).as_bytes())
        print(log)
        print(summary)
        if n_bad > 0:
            raise Error("FAIL plumbing: " + String(n_bad) + " of " + String(n_checks) + " checks mismatched")
        print("E11B PLUMBING IDENTITY OK checks=" + String(n_checks))
        return

    # stage B
    var prompt_mismatch = 0
    var readback_mismatch = 0
    var identity_mismatch = 0
    var k_first_mismatch = 0
    var n_records = 0
    with open(out_dir + "/records.jsonl", "w") as rf:
        for s in range(len(sel)):
            var idx = sel[s]
            var it = items[idx].copy()
            var prompt0 = chat_ids(tok, it.b_question)
            if prompt0[len(prompt0) - 1] != last_tok:
                raise Error("FAIL stage B: " + it.id + " does not end in the graft token")
            # Arm 0 always runs: it is the prompt-id and identity reference and
            # the source of B's own residuals for I and M.
            var r0 = run_arm(ctx, b, wst, pack, toks_h, prompt0, RECV_MAX, tmax, stops, mega, True, -1, False, graft_d)
            var own = tapped_record(b)
            for ai in range(len(arms)):
                var arm = arms[ai]
                if arm == "0":
                    rf.write_bytes(record_line(it.id, arm, -1, "", tok.decode(r0.ids), len(r0.ids), "").as_bytes())
                    n_records += 1
                elif arm == "T" or arm == "S":
                    var relay = String("")
                    var content = it.a_text + "\n\n" + it.b_question
                    if arm == "T":
                        relay = relays[it.id]
                        content = relay + "\n\n" + it.b_question
                    var pr = chat_ids(tok, content)
                    var r = run_arm(ctx, b, wst, pack, toks_h, pr, RECV_MAX, tmax, stops, mega, True, -1, False, graft_d)
                    rf.write_bytes(record_line(it.id, arm, -1, "", tok.decode(r.ids), len(r.ids), relay).as_bytes())
                    n_records += 1
                elif arm == "K":
                    var pk = chat_ids(tok, it.a_text + "\n\n" + it.b_question)
                    if pk[len(pk) - 1] != last_tok:
                        raise Error("FAIL stage B: K sender prompt of " + it.id + " does not end in the graft token")
                    var rk = run_arm(ctx, b, wst, pack, toks_h, pk, 1, tmax, stops, mega, True, -1, False, graft_d)
                    var krec = tapped_record(b)
                    for li in range(len(layers)):
                        var L = layers[li]
                        fill_graft(ctx, graft_h, graft_d, krec, 0, L)
                        var r = run_arm(ctx, b, wst, pack, toks_h, prompt0, RECV_MAX, tmax, stops, mega, True, L, False, graft_d)
                        if not same_ids(r.prompt_back, r0.prompt_back):
                            prompt_mismatch += 1
                        if readback_diff(b, krec, 0, L) != 0:
                            readback_mismatch += 1
                        if L == N_LAYERS - 1:
                            if len(r.ids) == 0 or len(rk.ids) == 0 or r.ids[0] != rk.ids[0]:
                                k_first_mismatch += 1
                        rf.write_bytes(record_line(it.id, arm, L, it.id + ":S", tok.decode(r.ids), len(r.ids), "").as_bytes())
                        n_records += 1
                else:
                    var src = idx
                    if arm == "C":
                        src = find_item(items, it.twin_id)
                    elif arm == "X":
                        src = -1
                        for d in range(1, len(sel)):
                            var cand = sel[(s + d) % len(sel)]
                            if items[cand].game == it.game and items[cand].gold != it.gold:
                                src = cand
                                break
                        if src < 0:
                            raise Error("FAIL stage B: no X source for " + it.id)
                    for li in range(len(layers)):
                        var L = layers[li]
                        if arm == "I":
                            fill_graft(ctx, graft_h, graft_d, own, 0, L)
                        else:
                            fill_graft(ctx, graft_h, graft_d, bank, src * REC_BYTES, L)
                        var r = run_arm(ctx, b, wst, pack, toks_h, prompt0, RECV_MAX, tmax, stops, mega, True, L, arm == "M", graft_d)
                        if not same_ids(r.prompt_back, r0.prompt_back):
                            prompt_mismatch += 1
                        if arm == "I":
                            if readback_diff(b, own, 0, L) != 0:
                                readback_mismatch += 1
                            if not same_ids(r.ids, r0.ids):
                                identity_mismatch += 1
                        elif arm != "M":
                            if readback_diff(b, bank, src * REC_BYTES, L) != 0:
                                readback_mismatch += 1
                        rf.write_bytes(record_line(it.id, arm, L, it.id if arm == "I" else items[src].id, tok.decode(r.ids), len(r.ids), "").as_bytes())
                        n_records += 1
            print("B item", s + 1, "/", len(sel), it.id)
    var summary = String("stage=B items=") + String(len(sel)) + " records=" + String(n_records) + " prompt_id_mismatch=" + String(prompt_mismatch) + " readback_mismatch=" + String(readback_mismatch) + " identity_mismatch=" + String(identity_mismatch) + " k_first_mismatch=" + String(k_first_mismatch) + " elapsed_s=" + String(Float64(perf_counter_ns() - t0) / 1e9) + "\n"
    with open(out_dir + "/summary.txt", "w") as f:
        f.write_bytes(summary.as_bytes())
    print(summary)
    if prompt_mismatch + readback_mismatch + identity_mismatch + k_first_mismatch > 0:
        raise Error("FAIL stage B void check: prompt " + String(prompt_mismatch) + " readback " + String(readback_mismatch) + " identity " + String(identity_mismatch) + " k_first " + String(k_first_mismatch))
    print("E11B stage B OK")


def contains_str(xs: List[String], v: String) -> Bool:
    for i in range(len(xs)):
        if xs[i] == v:
            return True
    return False
