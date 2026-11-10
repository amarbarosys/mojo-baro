# E13 piece 1: the engine dump mode (docs/design/latent-os/06-experiments.md
# section E13, "Built and checked before any E13 GPU run").
#
# dump_item_latents is the mechanism this file exists to reuse, not
# reimplement: it is exactly A's L8-raw producer path in
# bench/bench_latent_handoff.mojo (run_to_prompt_end then
# collect_latent_raw), imported unmodified. `--mode check` runs that path
# twice, independently, on 3 math items straight out of the harness's own
# bench/data/e8_tasks.json (same stored token ids the harness scores against,
# so tokenization is not a variable here) and asserts the two f32 vector sets
# are bit-identical. `--mode dump` is the real GSM8K train sweep (docs
# E13: >= 3 annotated `<<` steps, 4,799 items, k in {8, 32}); building it is
# in scope for this lane, running the full sweep is not (that is E13 data
# prep, which the lane brief excludes) -- it is never invoked below.
from std.memory import bitcast
from std.os import getenv
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, HostBuffer

from registry import H, f32
from window import WindowBufs, WindowState
from grammar.automaton import Bitset
from harness import load_pack, alloc_bufs, Pack
from bench_latent_handoff import (
    collect_latent_raw,
    run_to_prompt_end,
    get_str,
    get_int_list,
    ids_to_json,
)
from tokenizer import Tokenizer
from grammar.json_value import parse_json_file, parse_json_bytes, JSONDoc

comptime K8 = 8
comptime K32 = 32

comptime MATH_SYS = String(
    "You are a precise mathematical reasoning agent. Solve the problem and"
    " state the final integer answer at the end as: Answer: <number>"
)


def chat_prompt(sys_msg: String, user_msg: String) -> String:
    return (
        "<|im_start|>system\n" + sys_msg + "<|im_end|>\n"
        + "<|im_start|>user\n" + user_msg + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


def count_sub(hay: String, needle: String) -> Int:
    var b = hay.as_bytes()
    var t = needle.as_bytes()
    var n = 0
    var i = 0
    while i + len(t) <= len(b):
        var hit = True
        for j in range(len(t)):
            if b[i + j] != t[j]:
                hit = False
                break
        if hit:
            n += 1
            i += len(t)
        else:
            i += 1
    return n


def gsm_final_answer(ans: String) raises -> Int:
    """Parse the integer after the last '####' marker."""
    var b = ans.as_bytes()
    var at = -1
    var i = 0
    while i + 4 <= len(b):
        if (
            b[i] == UInt8(ord("#"))
            and b[i + 1] == UInt8(ord("#"))
            and b[i + 2] == UInt8(ord("#"))
            and b[i + 3] == UInt8(ord("#"))
        ):
            at = i
        i += 1
    if at < 0:
        raise Error("gsm8k answer has no #### marker")
    var neg = False
    var val = 0
    var seen = False
    for k in range(at + 4, len(b)):
        var c = b[k]
        if c == UInt8(ord("-")) and not seen:
            neg = True
        elif c >= UInt8(ord("0")) and c <= UInt8(ord("9")):
            val = val * 10 + (Int(c) - 48)
            seen = True
        elif seen:
            break
    if not seen:
        raise Error("gsm8k answer has no digits after ####")
    return -val if neg else val


def dump_item_latents(
    ctx: DeviceContext, mut b: WindowBufs, mut wst: WindowState,
    pack_q4: Bool, q4_off: Int, e: Int, tokens: List[Int], tmax: Int, k: Int,
) raises -> HostBuffer[f32]:
    """A's L8-raw producer path, unmodified: run to the end of the prompt,
    then collect k raw final-norm hidden vectors. Resets engine state itself
    (run_to_prompt_end -> reset_and_load), so callers need not reset between
    items."""
    var latent_h = ctx.enqueue_create_host_buffer[f32](k * H)
    ctx.synchronize()
    var cfg = run_to_prompt_end(ctx, b, wst, pack_q4, q4_off, e, tokens, tmax)
    collect_latent_raw(ctx, b, cfg, wst, latent_h, k)
    ctx.synchronize()
    return latent_h^


def write_dump_record(
    mut f: FileHandle, item_id: String, answer: Int, tokens: List[Int],
    latent_h: HostBuffer[f32], k: Int,
) raises:
    var idb = item_id.as_bytes()
    var header = List[UInt8]()
    var idlen = Int32(len(idb))
    header.append(UInt8(idlen & 0xFF)); header.append(UInt8((idlen >> 8) & 0xFF))
    header.append(UInt8((idlen >> 16) & 0xFF)); header.append(UInt8((idlen >> 24) & 0xFF))
    f.write_bytes(Span(header))
    f.write_bytes(idb)
    var ansb = List[UInt8]()
    var a32 = Int32(answer)
    ansb.append(UInt8(a32 & 0xFF)); ansb.append(UInt8((a32 >> 8) & 0xFF))
    ansb.append(UInt8((a32 >> 16) & 0xFF)); ansb.append(UInt8((a32 >> 24) & 0xFF))
    f.write_bytes(Span(ansb))
    var ntok = List[UInt8]()
    var nt32 = Int32(len(tokens))
    ntok.append(UInt8(nt32 & 0xFF)); ntok.append(UInt8((nt32 >> 8) & 0xFF))
    ntok.append(UInt8((nt32 >> 16) & 0xFF)); ntok.append(UInt8((nt32 >> 24) & 0xFF))
    f.write_bytes(Span(ntok))
    var tokb = List[UInt8](unsafe_uninit_length=len(tokens) * 4)
    for i in range(len(tokens)):
        var t32 = Int32(tokens[i])
        tokb[i * 4 + 0] = UInt8(t32 & 0xFF)
        tokb[i * 4 + 1] = UInt8((t32 >> 8) & 0xFF)
        tokb[i * 4 + 2] = UInt8((t32 >> 16) & 0xFF)
        tokb[i * 4 + 3] = UInt8((t32 >> 24) & 0xFF)
    f.write_bytes(Span(tokb))
    var vecb = List[UInt8](unsafe_uninit_length=k * H * 4)
    for i in range(k * H):
        var bits = bitcast[DType.uint32, 1](latent_h[i])[0]
        vecb[i * 4 + 0] = UInt8(bits & 0xFF)
        vecb[i * 4 + 1] = UInt8((bits >> 8) & 0xFF)
        vecb[i * 4 + 2] = UInt8((bits >> 16) & 0xFF)
        vecb[i * 4 + 3] = UInt8((bits >> 24) & 0xFF)
    f.write_bytes(Span(vecb))


def run_full_dump(
    ctx: DeviceContext, mut b: WindowBufs, mut wst: WindowState,
    pack_q4: Bool, q4_off: Int, e: Int, tok: Tokenizer,
    gsm_path: String, out_dir: String, tmax: Int, min_steps: Int, limit: Int,
) raises:
    """The real GSM8K train sweep -- built, only ever invoked at a small
    `limit` by this lane (E13-build-2026-09-11's 50-item smoke feeds piece
    3's trainer check; limit=0 means unlimited, the real run, never executed
    here). Filters train.jsonl to items with >= min_steps annotated `<<`
    steps (4,799 at min_steps=3, per the E13 design doc), builds each prompt
    exactly like tools/generate_e8_tasks.mojo's math items (MATH_SYS +
    chat_prompt), and for k in {8, 32} writes one binary record per item:
    [id_len u32][id bytes][answer i32][n_tok u32][tok ids u32*][k*H f32]."""
    var raw = String("")
    with open(gsm_path, "r") as gf:
        raw = gf.read()
    var f8_path = out_dir + "/train-k8.bin"
    var f32_path = out_dir + "/train-k32.bin"
    var n_written = 0
    var n_skipped = 0
    var skip_k32 = getenv("E13_SKIP_K32", "0") == "1"
    with open(f8_path, "w") as f8:
        with open(f32_path, "w") as f32f:
            for line in raw.split("\n"):
                var ln = String(line)
                var lb = ln.as_bytes()
                if len(lb) < 2:
                    continue
                var buf = List[UInt8]()
                for k in range(len(lb)):
                    buf.append(lb[k])
                var doc = parse_json_bytes(buf^)
                var qi = doc.get_field(doc.root, "question")
                var ai = doc.get_field(doc.root, "answer")
                if qi < 0 or ai < 0:
                    continue
                var atext = doc.get(ai).s
                if count_sub(atext, "<<") < min_steps:
                    continue
                var question = doc.get(qi).s
                var answer = gsm_final_answer(atext)
                var full = chat_prompt(MATH_SYS, question)
                var tokens = tok.encode(full, add_special=False)
                var item_id = String("gsm8k_train_") + String(n_written + n_skipped)

                if len(tokens) + K32 > tmax:
                    n_skipped += 1
                    continue

                var h8 = dump_item_latents(ctx, b, wst, pack_q4, q4_off, e, tokens, tmax, K8)
                write_dump_record(f8, item_id, answer, tokens, h8, K8)
                if not skip_k32:
                    var h32 = dump_item_latents(ctx, b, wst, pack_q4, q4_off, e, tokens, tmax, K32)
                    write_dump_record(f32f, item_id, answer, tokens, h32, K32)
                n_written += 1
                if limit > 0 and n_written >= limit:
                    break
    print("run_full_dump: wrote", n_written, "items, skipped", n_skipped, "(too long for tmax)")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var args = argv()
    var mode = String("check")
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--mode" and i + 1 < len(args):
            mode = String(args[i + 1])
            i += 2
        else:
            i += 1

    var packdir = getenv("BARO_PACK", ".work/engine-pack-q4")
    var gguf_path = getenv(
        "BARO_E8_GGUF",
        getenv("HOME", "")
        + "/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf",
    )
    var tmax = atol(getenv("BARO_E8_TMAX", "896"))

    var ctx = DeviceContext()
    print("--- loading pack ---")
    var pack = load_pack(ctx, packdir)
    var buf = alloc_bufs(ctx, pack, tmax)
    var pack_q4 = pack.pack_q4
    var q4_off = pack.q4_off
    var e = pack.e
    var wst = WindowState(pos=0, pos_prev=0, ring=0, n_drafted=0, n_accepted=0, n_spec_windows=0, n_dumped=0, tp=0, tq=0, pf_att=0, pf_ssm=0, pf_ffn=0, pf_head=0, pf_proc=0, pf_draft=0, fc=[0, 0, 0, 0, 0, 0], pc=[0, 0, 0, 0, 0, 0, 0, 0], ac=[0, 0, 0, 0, 0, 0], p3=[0, 0, 0, 0], pfx=[0, 0, 0, 0], grammar=None, grammar_mask=Bitset(1), grammar_pending_think=False, grammar_think_buf=List[UInt8](), grammar_stop=False, grammar_masked_draws=0, grammar_accepted=0, tool_re="", tool_buf=List[UInt8](), grammar_tool=False)

    if mode == "dump":
        var gsm_path = getenv("E8_GSM8K_TRAIN", getenv("HOME", "") + "/Models/datasets/gsm8k/main/train.jsonl")
        var out_dir = getenv("E13_DUMP_DIR", ".work/e13")
        var min_steps = atol(getenv("E13_MIN_STEPS", "3"))
        var limit = atol(getenv("E13_DUMP_LIMIT", "0"))
        var tok = Tokenizer(gguf_path)
        run_full_dump(ctx, buf, wst, pack_q4, q4_off, e, tok, gsm_path, out_dir, tmax, min_steps, limit)
        return

    # --mode check (default): 3 math items straight from the harness's own
    # eval set (bench/data/e8_tasks.json), using its stored token ids -- no
    # retokenization, so this isolates the engine-dump mechanism itself.
    # dump_item_latents (this file's whole reason to exist) is called twice,
    # independently, per item; CHECK is max abs diff 0 between the two calls.
    var doc = parse_json_file("bench/data/e8_tasks.json")
    var root = doc.get(doc.root)
    var n_avail = len(root.arr)
    var picked = List[Int]()
    for j in range(n_avail):
        if len(picked) >= 3:
            break
        if get_str(doc, root.arr[j], "type") == "math":
            picked.append(root.arr[j])
    if len(picked) < 3:
        raise Error("bench/data/e8_tasks.json has fewer than 3 math items")

    var overall_max_diff: Float64 = 0.0
    var overall_bad_bits = 0
    for pi in range(len(picked)):
        var idx = picked[pi]
        var task_id = get_str(doc, idx, "id")
        var tokens = get_int_list(doc, idx, "tokens")

        var vA = dump_item_latents(ctx, buf, wst, pack_q4, q4_off, e, tokens, tmax, K8)
        var vB = dump_item_latents(ctx, buf, wst, pack_q4, q4_off, e, tokens, tmax, K8)

        var item_max_diff: Float64 = 0.0
        var bad_bits = 0
        for s in range(K8 * H):
            var da = Float64(vA[s])
            var db = Float64(vB[s])
            var d = abs(da - db)
            if d > item_max_diff:
                item_max_diff = d
            if bitcast[DType.uint32, 1](vA[s])[0] != bitcast[DType.uint32, 1](vB[s])[0]:
                bad_bits += 1
        if item_max_diff > overall_max_diff:
            overall_max_diff = item_max_diff
        overall_bad_bits += bad_bits
        print("item", task_id, "n_tok", len(tokens), "max_abs_diff", item_max_diff, "bit_mismatches", bad_bits, "/", K8 * H)

    print("CHECK items=", len(picked), " k=", K8, " overall_max_abs_diff=", overall_max_diff, " overall_bit_mismatches=", overall_bad_bits)
    if overall_max_diff == 0.0 and overall_bad_bits == 0:
        print("CHECK: PASS")
    else:
        print("CHECK: FAIL")
        raise Error("dump_item_latents is not deterministic across two independent calls")
