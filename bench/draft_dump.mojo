# A4 draft-head data + parity dump (bench/draft-head-protocol.md).
#
# --mode dump: real held-out text, one token at a time through step_window
# (the same per-token loop bench_latent_handoff.run_to_prompt_end already
# runs), reading serve/realign.final_norm_hidden after each step. That is
# model.norm(residual) at the position just processed, i.e. h_i (row i of
# the dump = h AFTER processing tokens[i]): the same tensor blk32_forward
# consumes as hsrc, one step later (blk32's own call is pos==tok_pos==P,
# hsrc=h_{P-1}, predicting Toks[P+1] -- confirmed against both real call
# sites in serve/window.mojo and the self-chained draft loop's own
# `tokcp_k(..., st.pos+j+1, ...)` write-back). So for training: row h[i]
# pairs with INPUT token tokens[i+1] (P=i+1) and LABEL tokens[i+2] (P+1),
# valid for i in 0..n-3. No collect_latent_raw here: that function runs a
# k-step self-feeding latent rollout (E13's own out-of-manifold input), not
# a read of the trunk's real per-position hidden state, and does not apply
# to this lane (bench/draft-head-protocol.md).
#
# --mode parity: for every interior position P, calls blk32_forward
# directly (m=1, pos=tok_pos=P, do_head=True) with hsrc=h_{P-1}, the exact
# call shape step_window's own spec branch uses, and records the real
# engine's own drafted argmax (dtok_d[0]) alongside h_{P-1} and the true
# next token Toks[P+1]. tools/mtp_head.py --mode parity compares its torch
# replica's argmax against this file before any training gradient is
# trusted.
from std.math import exp
from std.memory import bitcast
from std.os import getenv
from std.sys import argv, has_accelerator

from max.gpu.host import DeviceContext, HostBuffer, DeviceBuffer

from registry import H, VOCAB, f32
from window import WindowBufs, WindowState, WindowCfg, blk32_forward, step_window
from realign import final_norm_hidden, head_logits
from bench_latent_handoff import run_to_prompt_end, reset_and_load, make_cfg, read_toks
from harness import load_pack, alloc_bufs, Pack
from grammar.automaton import Bitset
from grammar.json_value import parse_json_file, parse_json_bytes
from bench_latent_handoff import get_int_list, get_str
from tokenizer import Tokenizer


def write_u32(mut f: FileHandle, v: Int) raises:
    var x = Int32(v)
    var b = List[UInt8]()
    b.append(UInt8(x & 0xFF)); b.append(UInt8((x >> 8) & 0xFF))
    b.append(UInt8((x >> 16) & 0xFF)); b.append(UInt8((x >> 24) & 0xFF))
    f.write_bytes(Span(b))


def write_f32_vec(mut f: FileHandle, v: HostBuffer[f32], n: Int) raises:
    var b = List[UInt8](unsafe_uninit_length=n * 4)
    for i in range(n):
        var bits = bitcast[DType.uint32, 1](v[i])[0]
        b[i * 4 + 0] = UInt8(bits & 0xFF)
        b[i * 4 + 1] = UInt8((bits >> 8) & 0xFF)
        b[i * 4 + 2] = UInt8((bits >> 16) & 0xFF)
        b[i * 4 + 3] = UInt8((bits >> 24) & 0xFF)
    f.write_bytes(Span(b))


def write_f32(mut f: FileHandle, v: Float32) raises:
    var bits = bitcast[DType.uint32, 1](v)[0]
    var b = List[UInt8]()
    b.append(UInt8(bits & 0xFF)); b.append(UInt8((bits >> 8) & 0xFF))
    b.append(UInt8((bits >> 16) & 0xFF)); b.append(UInt8((bits >> 24) & 0xFF))
    f.write_bytes(Span(b))


def top8_from_logits(
    lg: HostBuffer[f32], n: Int
) -> Tuple[List[Int], List[Float32]]:
    """Top-8 by raw logit, descending, ties broken by the lower token id;
    `top8_probs` is softmax restricted to these eight logits (equal to a
    full-vocab softmax's top-8 renormalized over their own subsum -- the
    full-vocab denominator cancels between the two, so this is not an
    approximation, and it needs no VOCAB-wide softmax kernel at all)."""
    var ids = List[Int]()
    var vals = List[Float32]()
    for _ in range(8):
        ids.append(-1)
        vals.append(Float32(-3.4e38))
    for idx in range(n):
        var v = Float32(lg[idx])
        if v > vals[7] or (v == vals[7] and idx < ids[7]):
            var pos = 7
            while pos > 0 and (v > vals[pos - 1] or (v == vals[pos - 1] and idx < ids[pos - 1])):
                vals[pos] = vals[pos - 1]
                ids[pos] = ids[pos - 1]
                pos -= 1
            vals[pos] = v
            ids[pos] = idx
    var maxv = vals[0]
    var exps = List[Float32]()
    var sum_exp = Float32(0)
    for k in range(8):
        var e = exp(vals[k] - maxv)
        exps.append(e)
        sum_exp += e
    var probs = List[Float32]()
    for k in range(8):
        probs.append(exps[k] / sum_exp)
    return (ids^, probs^)


def dump_document_stepwise(
    ctx: DeviceContext, mut b: WindowBufs, mut wst: WindowState,
    pack_q4: Bool, q4_off: Int, e: Int, tokens: List[Int], tmax: Int,
    mut out: FileHandle,
) raises:
    var n = len(tokens)
    var pf = reset_and_load(ctx, b, tokens, tmax)
    var cfg = make_cfg(pack_q4, q4_off, e, pf[0], pf[1], n, n + 1)
    # Coordinator diagnosis, 2026-09-17, two causes: (1) make_cfg's mega=True
    # takes the m==1 megakernel path once st.pos+1 >= n_prompt
    # (window.mojo:971), fusing every layer into one kernel that never
    # writes the standalone hidden buffer final_norm_hidden reads
    # (window.mojo:1052 skips the host layer loop). (2) prefilled rows write
    # xp_d, not b.x_d, so a row processed under prefill stays zero even with
    # mega off. n_prompt=n and pf_rows/pf_tail from `pf` both push most of
    # the document through one or the other path. Force pure per-token
    # stepwise decode instead, the path final_norm_hidden actually reads.
    # Local override only; make_cfg (shared with run_parity, not broken) is
    # untouched.
    cfg.mega = False
    cfg.pf_rows = 0
    cfg.pf_tail = 0
    wst.reset(0)
    write_u32(out, n)
    for i in range(n):
        write_u32(out, tokens[i])
    # Records use 1 <= P <= n-2 (bench/p5a-smoke-protocol.md): P = i + 1, so
    # i ranges 0..n-3, n_records = n - 2 (one fewer than the old hidden-only
    # loop's n - 1, which wrote a last row with no valid tokens[P+1]).
    var n_records = n - 2
    if n_records < 0:
        n_records = 0
    write_u32(out, n_records)
    var hbuf = ctx.enqueue_create_host_buffer[f32](H)
    var lgbuf = ctx.enqueue_create_host_buffer[f32](VOCAB)
    for i in range(n_records):
        # Coordinator, 2026-09-17: step_window batches m = min(MROWS,
        # n_prompt-1-st.pos) rows per call whenever st.pos+1 < n_prompt,
        # independent of mega/pf_rows; with n_prompt=n that batched several
        # loop iterations into one call, and every one of them read the
        # SAME stale b.x_d row 0 from that call (rows 0 and 5 came back
        # byte-identical, row 20 in the next batch differed). n_prompt=i+2
        # forces m=1 for exactly this step while keeping st.pos+1 < n_prompt
        # true, so the teacher-forced path stays selected -- unlike
        # n_prompt=0, which was rejected: it takes the >=n_prompt branch
        # that writes the model's own greedy pick into Toks[pos+1] instead
        # of reading the true next token, corrupting the document under
        # teacher forcing.
        cfg.n_prompt = i + 2
        while wst.pos <= i:
            step_window(ctx, b, cfg, wst)
        var hdev = ctx.enqueue_create_buffer[f32](H)
        final_norm_hidden(ctx, b, hdev)  # h_i = h_(P-1), P = i+1
        ctx.enqueue_copy(dst_buf=hbuf, src_buf=hdev)
        # codex, 2026-09-17: head_logits from h_i predicts tokens[i+1] (the
        # standard next-token relationship), but the frozen v2 schema and
        # blk32_forward's own hsrc=h_{P-1} convention need the MTP-shifted
        # target, the trunk's prediction for tokens[P+1] -- one more
        # teacher-forced step ahead, from h_P. Validated: target_argmax
        # matched tokens[pos] (the P-1-shifted target) 87.76% of 98 records
        # before this fix, tokens[pos+1] only 1.02%.
        cfg.n_prompt = i + 3
        while wst.pos <= i + 1:
            step_window(ctx, b, cfg, wst)
        var lgdev = ctx.enqueue_create_buffer[f32](VOCAB)
        head_logits(ctx, b, pack_q4, lgdev)  # from h_P
        ctx.enqueue_copy(dst_buf=lgbuf, src_buf=lgdev)
        ctx.synchronize()
        var top8 = top8_from_logits(lgbuf, VOCAB)
        var top8_ids = top8[0].copy()
        var top8_probs = top8[1].copy()
        var pos = i + 1
        write_u32(out, pos)
        write_u32(out, tokens[pos])
        write_u32(out, top8_ids[0])
        write_u32(out, 8)
        for k in range(8):
            write_u32(out, top8_ids[k])
        for k in range(8):
            write_f32(out, top8_probs[k])
        write_f32_vec(out, hbuf, H)
    cfg.n_prompt = n
    while wst.pos < n:
        step_window(ctx, b, cfg, wst)
    # Invariant, not a diagnostic print: the n_prompt=i+2/i+3 teacher-forced
    # schedule must never let step_window take the >=n_prompt branch that
    # writes the model's own greedy pick into Toks[pos+1] -- that would
    # silently replace the true document with the model's own continuation.
    # Verified once already (0/100 mismatches); this stays as a hard check
    # on every future run, since a schedule mistake here corrupts training
    # data without any other visible symptom.
    var final_toks = read_toks(ctx, b, 0, n, tmax)
    for i in range(n):
        if final_toks[i] != tokens[i]:
            raise Error("draft_dump: teacher forcing broken, token " + String(i) + " is " + String(final_toks[i]) + ", expected " + String(tokens[i]))


def run_parity(
    ctx: DeviceContext, mut b: WindowBufs, mut wst: WindowState,
    pack_q4: Bool, q4_off: Int, e: Int, draft_q4: Bool, q4_off_draft: Int,
    tokens: List[Int], tmax: Int, mut out: FileHandle,
) raises:
    """For every interior position P (1 <= P <= n-2), calls blk32_forward
    (m=1, pos=P, tok_pos=P, do_head=True) with hsrc = h_{P-1} -- the exact
    call shape both real call sites in serve/window.mojo use (pos==tok_pos
    always; the output is copied into Toks[tok_pos+1], confirmed from the
    self-chained draft loop's own `tokcp_k(..., st.pos+j+1, ...)` right
    after a `blk32_forward(..., st.pos+j, st.pos+j, ...)` call) -- and
    records the real engine's own drafted argmax dtok_d[0] alongside
    h_{P-1} and the true next token. Format: [n_pairs u32] then per pair
    [pos u32][tok_input u32][true_next u32][draft_argmax i32][H f32
    h_{pos-1}]. tok_input = Toks[pos] (the token blk32 is given as its
    embedding input); true_next = Toks[pos+1] (what the draft's argmax
    predicts, mirroring the trunk's own h_pos -> lm_head -> Toks[pos+1])."""
    var n = len(tokens)
    var pf = reset_and_load(ctx, b, tokens, tmax)
    var cfg = make_cfg(pack_q4, q4_off, e, pf[0], pf[1], n, n + 1)
    wst.reset(0)
    var hbuf = ctx.enqueue_create_host_buffer[f32](H)
    var dtok_h = ctx.enqueue_create_host_buffer[DType.int32](1)
    var p3_scratch = List[Int]()
    var n_pairs = n - 2
    if n_pairs < 0:
        n_pairs = 0
    write_u32(out, n_pairs)
    for i in range(n_pairs):
        var pos = i + 1
        while wst.pos <= pos - 1:
            step_window(ctx, b, cfg, wst)
        var hdev = ctx.enqueue_create_buffer[f32](H)
        final_norm_hidden(ctx, b, hdev)
        ctx.enqueue_copy(dst_buf=hbuf, src_buf=hdev)
        ctx.synchronize()
        blk32_forward(
            ctx, b.wbuf, b.off, e, 1, pos, pos, True, hdev,
            b.x_d, b.curb_d, b.qf_d, b.q_d, b.k_d, b.v_d, b.gate_d, b.ao_d,
            b.resb_d, b.fgb_d, b.p_qf_d, b.p_kv_d, b.p_h_d, b.p_ffn_d,
            b.p_ffn2_d, b.p_v_d, b.logits_d, b.cc_d, b.de_d, b.hd_d,
            b.kc32_d, b.vc32_d, b.kvtab_d, b.toks_d, b.dtok_d,
            False, p3_scratch, draft_q4, q4_off_draft, pack_q4,
        )
        ctx.enqueue_copy(dst_buf=dtok_h, src_buf=DeviceBuffer[DType.int32](ctx, b.dtok_d.unsafe_ptr(), 1, owning=False))
        ctx.synchronize()
        write_u32(out, pos)
        write_u32(out, tokens[pos])
        write_u32(out, tokens[pos + 1])
        write_u32(out, Int(dtok_h[0]))
        write_f32_vec(out, hbuf, H)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var args = argv()
    var mode = String("dump")
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--mode" and i + 1 < len(args):
            mode = String(args[i + 1]); i += 2
        else:
            i += 1

    var packdir = getenv("BARO_PACK", ".work/engine-pack-q4")
    var tmax = atol(getenv("BARO_E8_TMAX", "896"))
    var ctx = DeviceContext()
    var pack = load_pack(ctx, packdir)
    var buf = alloc_bufs(ctx, pack, tmax)
    var wst = WindowState(pos=0, pos_prev=0, ring=0, n_drafted=0, n_accepted=0, n_spec_windows=0, n_dumped=0, tp=0, tq=0, pf_att=0, pf_ssm=0, pf_ffn=0, pf_head=0, pf_proc=0, pf_draft=0, fc=[0, 0, 0, 0, 0, 0], pc=[0, 0, 0, 0, 0, 0, 0, 0], ac=[0, 0, 0, 0, 0, 0], p3=[0, 0, 0, 0], pfx=[0, 0, 0, 0], grammar=None, grammar_mask=Bitset(1), grammar_pending_think=False, grammar_think_buf=List[UInt8](), grammar_stop=False, grammar_masked_draws=0, grammar_accepted=0, tool_re="", tool_buf=List[UInt8](), grammar_tool=False)

    var out_path = getenv("A4_OUT", ".work/a4/dump.bin")
    var limit = atol(getenv("A4_LIMIT", "0"))
    var draft_q4 = pack.have_q4_draft and getenv("BARO_DRAFT_Q4", "0") == "1"
    var n_written = 0
    var text_mode = getenv("A4_TEXT_MODE", "json")

    with open(out_path, "w") as out:
        if mode == "dump":
            # bench/p5a-smoke-protocol.md's frozen v2 dump format: the file
            # opens with this one version marker, then one [n, tokens,
            # n_records, records...] block per document. Parity mode's
            # format is untouched and carries no such marker.
            write_u32(out, 2)
        if text_mode == "gsm8k":
            # GSM8K train.jsonl, real natural-language text (question +
            # worked answer), genuinely held out: distinct from
            # bench/data/e8_tasks.json's round-5 items and from
            # bench/mtp-prompts/*.tokens' fixed 20-prompt gate set. Order
            # step 2 / the smoke's 2000-sequence data (bench/draft-head-protocol.md).
            var gsm_path = getenv("GSM8K_PATH", getenv("HOME", "") + "/Models/datasets/gsm8k/main/train.jsonl")
            var gguf_path = getenv(
                "BARO_E8_GGUF",
                getenv("HOME", "") + "/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf",
            )
            var tok = Tokenizer(gguf_path)
            var raw = String("")
            with open(gsm_path, "r") as gf:
                raw = gf.read()
            for line in raw.split("\n"):
                var ln = String(line)
                if ln.byte_length() < 2:
                    continue
                var buf2 = List[UInt8]()
                for bch in ln.as_bytes():
                    buf2.append(bch)
                var jdoc = parse_json_bytes(buf2^)
                var qi = jdoc.get_field(jdoc.root, "question")
                var ai = jdoc.get_field(jdoc.root, "answer")
                if qi < 0 or ai < 0:
                    continue
                var text = String("Question: ") + jdoc.get(qi).s + String("\nAnswer: ") + jdoc.get(ai).s
                var tokens = tok.encode(text, add_special=False)
                if len(tokens) < 4 or len(tokens) + 2 > tmax:
                    continue
                if mode == "dump":
                    dump_document_stepwise(ctx, buf, wst, pack.pack_q4, pack.q4_off, pack.e, tokens, tmax, out)
                else:
                    run_parity(ctx, buf, wst, pack.pack_q4, pack.q4_off, pack.e, draft_q4, pack.q4_off, tokens, tmax, out)
                n_written += 1
                if limit > 0 and n_written >= limit:
                    break
        elif text_mode == "tokens":
            # bench/mtp-prompts/*.tokens: the real 20-prompt gate set (space-
            # separated decimal ids), same files bench/mtp-prompts.sh reads,
            # comma-separated in A4_TOKENS_FILES (this session's real-gate-set
            # rerun, exchange/lane-A4-report.md; the gate set stays a gate,
            # never trained on).
            var files_env = getenv("A4_TOKENS_FILES", "")
            for fp in files_env.split(","):
                var fpath = String(fp)
                if fpath.byte_length() == 0:
                    continue
                var raw = String("")
                with open(fpath, "r") as tf:
                    raw = tf.read()
                var tokens = List[Int]()
                for piece in raw.split(" "):
                    var s = String(piece).strip()
                    if s.byte_length() > 0:
                        tokens.append(atol(s))
                if len(tokens) < 4 or len(tokens) + 2 > tmax:
                    continue
                if mode == "dump":
                    dump_document_stepwise(ctx, buf, wst, pack.pack_q4, pack.q4_off, pack.e, tokens, tmax, out)
                else:
                    run_parity(ctx, buf, wst, pack.pack_q4, pack.q4_off, pack.e, draft_q4, pack.q4_off, tokens, tmax, out)
                n_written += 1
                if limit > 0 and n_written >= limit:
                    break
        else:
            # bench/data/e8_tasks.json's own "tokens" field, same stored ids
            # the harness already scores against elsewhere: fine for a
            # structural check, NOT a held-out training corpus
            # (bench/draft-head-protocol.md: the gate set stays a gate).
            var text_path = getenv("A4_TEXT_PATH", "bench/data/e8_tasks.json")
            var doc = parse_json_file(text_path)
            var root = doc.get(doc.root)
            for j in range(len(root.arr)):
                var tokens = get_int_list(doc, root.arr[j], "tokens")
                if len(tokens) < 4 or len(tokens) + 2 > tmax:
                    continue
                if mode == "dump":
                    dump_document_stepwise(ctx, buf, wst, pack.pack_q4, pack.q4_off, pack.e, tokens, tmax, out)
                else:
                    run_parity(ctx, buf, wst, pack.pack_q4, pack.q4_off, pack.e, draft_q4, pack.q4_off, tokens, tmax, out)
                n_written += 1
                if limit > 0 and n_written >= limit:
                    break
    print("draft_dump: wrote", n_written, "documents, mode=", mode, "->", out_path)
