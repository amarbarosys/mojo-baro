"""Byte-exact prefix checkpoint restore (bench/chat-protocol.md M1a, P-F1).

Drives the real engine path (serve/harness.mojo buffers, serve/window.mojo
step_window, serve/prefix.mojo Chain) on the q4 pack over P = A||B with
A = p8192.tokens[0:1088], B = the next 64 tokens, n = 1, and compares the
conv slot, the delta slot, every KV entry in [0, n_total), the megakernel
head partials (hmax/hidx) or the full logits (window path), and the generated
token as raw bits:

  cold_u   one process, prefill chunks as the engine cuts them (1024, 127)
  cold_s   one process, no checkpoint, chunk boundary forced at 1087
           (separates chunk arithmetic from checkpoint bookkeeping)
  restore  request A (checkpoints at 1024 and 1087), then P restored at 1087
  repeat   P again, restored at its own prompt-end checkpoint (1151)
  ckpt1024 P' = P with token 1050 mutated, restored at 1024 (chunk-identical
           to cold_u(P')) vs cold_u(P')

Lookup checks (chain = {1024, 1087, 1151} after `restore`): mutations at
token 0 and 1023 miss everything, 1025 and 1086 fall back to 1024, 1087
(outside tokens[0:1087]) keeps 1087, and a corrupted hash on the 1087
checkpoint falls back to 1024. Both the megakernel and the window path run.

Build: ./.venv/bin/mojo build kernels/test_prefix.mojo -I kernels -I serve -o .work/test_prefix
Env: BARO_PACK (default .work/engine-pack-q4), BARO_PREFIX_TOKENS (default
bench/prefill-prompts/p8192.tokens).
"""
from std.os import getenv
from std.sys import exit, has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from registry import *
from window import *
from harness import *
from prefix import *
from attn import kv_off
from serve_proto import default_sample_params

comptime A_LEN = 1088
comptime B_LEN = 64
comptime T_MAX = 2048


def read_tokens(path: String, n: Int) raises -> List[Int]:
    var out = List[Int]()
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
                    out.append(val)
                    if len(out) == n:
                        return out^
                val = 0
                have = False
        if have:
            out.append(val)
    if len(out) < n:
        raise Error("prompt file holds " + String(len(out)) + " tokens, need " + String(n))
    return out^


struct Snap(Movable):
    var conv_h: HostBuffer[f32]
    var ssm_h: HostBuffer[f32]
    var kc_h: HostBuffer[KVT]
    var vc_h: HostBuffer[KVT]
    var hmax_h: HostBuffer[f32]
    var hidx_h: HostBuffer[DType.int32]
    var logits_h: HostBuffer[f32]
    var tok: Int
    var n_total: Int

    def __init__(out self, ctx: DeviceContext, bufs: WindowBufs, ring: Int, n_total: Int) raises:
        self.conv_h = ctx.enqueue_create_host_buffer[f32](CONV_SLOT)
        self.ssm_h = ctx.enqueue_create_host_buffer[f32](SSM_SLOT)
        self.kc_h = ctx.enqueue_create_host_buffer[KVT](bufs.kvpool)
        self.vc_h = ctx.enqueue_create_host_buffer[KVT](bufs.kvpool)
        self.hmax_h = ctx.enqueue_create_host_buffer[f32](MEGA_MR * MEGA_G_WIN)
        self.hidx_h = ctx.enqueue_create_host_buffer[DType.int32](MEGA_MR * MEGA_G_WIN)
        self.logits_h = ctx.enqueue_create_host_buffer[f32](VOCAB)
        var tok_h = ctx.enqueue_create_host_buffer[DType.int32](1)
        ctx.synchronize()
        ctx.enqueue_copy(dst_buf=self.conv_h, src_buf=DeviceBuffer[f32](ctx, bufs.convstate_d.unsafe_ptr() + ring * CONV_SLOT, CONV_SLOT, owning=False))
        ctx.enqueue_copy(dst_buf=self.ssm_h, src_buf=DeviceBuffer[f32](ctx, bufs.sstate_d.unsafe_ptr() + ring * SSM_SLOT, SSM_SLOT, owning=False))
        # Slot 0 only: since A3(b) the KV pool holds SEQ_CAP resident
        # sequences and this test drives one.
        ctx.enqueue_copy(dst_buf=self.kc_h, src_buf=DeviceBuffer[KVT](ctx, bufs.kc_d.unsafe_ptr(), bufs.kvpool, owning=False))
        ctx.enqueue_copy(dst_buf=self.vc_h, src_buf=DeviceBuffer[KVT](ctx, bufs.vc_d.unsafe_ptr(), bufs.kvpool, owning=False))
        ctx.enqueue_copy(dst_buf=self.hmax_h, src_buf=bufs.hmax_d)
        ctx.enqueue_copy(dst_buf=self.hidx_h, src_buf=bufs.hidx_d)
        ctx.enqueue_copy(dst_buf=self.logits_h, src_buf=DeviceBuffer[f32](ctx, bufs.logits_d.unsafe_ptr(), VOCAB, owning=False))
        ctx.enqueue_copy(dst_buf=tok_h, src_buf=DeviceBuffer[DType.int32](ctx, bufs.toks_d.unsafe_ptr() + n_total - 1, 1, owning=False))
        ctx.synchronize()
        self.tok = Int(tok_h[0])
        self.n_total = n_total


def check_bits[dt: DType](label: String, a: HostBuffer[dt], b: HostBuffer[dt], n: Int, mut fails: Int):
    var ap = a.unsafe_ptr().unsafe_bitcast[UInt32]()
    var bp = b.unsafe_ptr().unsafe_bitcast[UInt32]()
    var first = -1
    var bad = 0
    for i in range(n):
        if ap[i] != bp[i]:
            if first < 0:
                first = i
            bad += 1
    if bad == 0:
        print("  PASS", label, "(" + String(n) + " words)")
    else:
        print("  FAIL", label, ": " + String(bad) + " / " + String(n) + " words differ, first at", first)
        fails += 1


def kv_row_equal(a: HostBuffer[KVT], b: HostBuffer[KVT], t: Int) -> Bool:
    var ap = a.unsafe_ptr().unsafe_bitcast[UInt32]()
    var bp = b.unsafe_ptr().unsafe_bitcast[UInt32]()
    for att_i in range(N_ATT):
        for kvh in range(NKVH):
            var o = kv_off[N_ATT](t, att_i, kvh)
            for d in range(HD):
                if ap[o + d] != bp[o + d]:
                    return False
    return True


def compare(label: String, x: Snap, y: Snap, mega: Bool, mut fails: Int):
    print("==", label, "(" + ("megakernel" if mega else "window path") + ")")
    check_bits("conv slot", x.conv_h, y.conv_h, CONV_SLOT, fails)
    check_bits("delta slot", x.ssm_h, y.ssm_h, SSM_SLOT, fails)
    var n = x.n_total
    var bad_k = 0
    var bad_v = 0
    var first_k = -1
    var first_v = -1
    for t in range(n):
        if not kv_row_equal(x.kc_h, y.kc_h, t):
            if first_k < 0:
                first_k = t
            bad_k += 1
        if not kv_row_equal(x.vc_h, y.vc_h, t):
            if first_v < 0:
                first_v = t
            bad_v += 1
    if bad_k == 0 and bad_v == 0:
        print("  PASS KV entries [0," + String(n) + ") all layers/heads")
    else:
        print("  FAIL KV entries: K rows differing", bad_k, "(first", first_k, ") V rows differing", bad_v, "(first", first_v, ")")
        fails += 1
    var pos_list: List[Int] = [0, 1, 1023, 1024, 1025, 1087, 1088, 1151]
    var line = String("  KV at positions:")
    for i in range(len(pos_list)):
        var t = pos_list[i]
        if t < n:
            line += " " + String(t) + ("=ok" if (kv_row_equal(x.kc_h, y.kc_h, t) and kv_row_equal(x.vc_h, y.vc_h, t)) else "=DIFF")
    print(line)
    if mega:
        check_bits("head partial max (hmax)", x.hmax_h, y.hmax_h, MEGA_MR * MEGA_G_WIN, fails)
        check_bits("head partial idx (hidx)", x.hidx_h, y.hidx_h, MEGA_MR * MEGA_G_WIN, fails)
    else:
        check_bits("next-token logits", x.logits_h, y.logits_h, VOCAB, fails)
    if x.tok == y.tok:
        print("  PASS next token", x.tok)
    else:
        print("  FAIL next token", x.tok, "vs", y.tok)
        fails += 1


def make_cfg(pack: Pack, mega: Bool, pf_rows: Int, pf_tail: Int, n_total: Int, n_prompt: Int) -> WindowCfg:
    return WindowCfg(
        pack_q4=pack.pack_q4, draft_q4=False, q4_off=pack.q4_off, e=pack.e, kcfg=2, spec=False, expert_trace=False, spec_dbg=False,
        serve=False, req_id=0, prof=False, pf2=False, pf3=False, pf4=False, dump=False, dump4=False, dump_layer=0, mega=mega, att_split=TMAX,
        mega_win=False, dot3=False, pf_chunk=CP, pf_rows=pf_rows, pf_tail=pf_tail, n_total=n_total, fr_k=0, fr_off=0, fr_ids_off=0, replay_m=8, mrope=False, inject=False, n_prompt=n_prompt,
        sample=default_sample_params(), dump_pen=False,
    )


def load_prompt(ctx: DeviceContext, mut bufs: WindowBufs, prompt: List[Int]) raises:
    var toks_h = ctx.enqueue_create_host_buffer[DType.int32](T_MAX)
    ctx.synchronize()
    for i in range(T_MAX):
        toks_h[i] = 0
    for i in range(len(prompt)):
        toks_h[i] = Int32(prompt[i])
    ctx.enqueue_copy(dst_buf=DeviceBuffer[DType.int32](ctx, bufs.toks_d.unsafe_ptr(), len(toks_h), owning=False), src_buf=toks_h)
    ctx.synchronize()


def cold_reset(ctx: DeviceContext, mut bufs: WindowBufs) raises:
    ctx.enqueue_memset(bufs.convstate_d, 0)
    ctx.enqueue_memset(bufs.sstate_d, 0)
    ctx.enqueue_memset(bufs.kc_d, 0)
    ctx.enqueue_memset(bufs.vc_d, 0)
    ctx.enqueue_memset(bufs.kc32_d, 0)
    ctx.enqueue_memset(bufs.vc32_d, 0)
    ctx.enqueue_memset(bufs.ctr_d, 0)
    ctx.enqueue_memset(bufs.prof_d, 0)
    ctx.synchronize()


def prefill_to(ctx: DeviceContext, mut bufs: WindowBufs, pack: Pack, mega: Bool, mut wst: WindowState, end: Int, n_total: Int, n_prompt: Int, prompt: List[Int], mut chain: Chain, take: Bool, hints: List[Int]) raises:
    # prompt rows [wst.pos, end) through prefill_forward in chunks of CP, exactly
    # as the engine loop cuts them from wst.pos; checkpoints at the engine's
    # boundaries when `take`.
    if end - wst.pos < PF_MIN:
        raise Error("prefill_to: fewer than PF_MIN rows")
    var tail = (end - wst.pos) % MROWS
    if tail == 0:
        tail = MROWS
    var cfg = make_cfg(pack, mega, end, tail, n_total, n_prompt)
    while wst.pos < end:
        if take and len(hints) > 0:
            var step_cfg = cfg.copy()
            step_cfg.pf_chunk = min(cfg.pf_chunk, next_ckpt_stop(hints, wst.pos, end) - wst.pos)
            step_window(ctx, bufs, step_cfg, wst)
        else:
            step_window(ctx, bufs, cfg, wst)
        if take and wst.pos < n_prompt:
            if wst.pos == n_prompt - 1 or wst.pos % CKPT_PERIOD == 0:
                chain.save(ctx, bufs.convstate_d, bufs.sstate_d, wst.ring, wst.pos, prompt, False, False)
            else:
                var hi = hint_index(hints, wst.pos)
                if hi >= 0:
                    chain.save(ctx, bufs.convstate_d, bufs.sstate_d, wst.ring, wst.pos, prompt, hi == 0, True)


def finish(ctx: DeviceContext, mut bufs: WindowBufs, pack: Pack, mega: Bool, mut wst: WindowState, n_total: Int, n_prompt: Int) raises:
    var cfg = make_cfg(pack, mega, 0, 0, n_total, n_prompt)
    while wst.pos < n_total - 1:
        step_window(ctx, bufs, cfg, wst)
    ctx.synchronize()


def fresh_state() -> WindowState:
    return WindowState(pos=0, pos_prev=0, ring=0, n_drafted=0, n_accepted=0, n_spec_windows=0, n_dumped=0, tp=0, tq=0, pf_att=0, pf_ssm=0, pf_ffn=0, pf_head=0, pf_proc=0, pf_draft=0, fc=[0, 0, 0, 0, 0, 0], pc=[0, 0, 0, 0, 0, 0, 0, 0], ac=[0, 0, 0, 0, 0, 0], p3=[0, 0, 0, 0], pfx=[0, 0, 0, 0], grammar=None, grammar_mask=Bitset(1), grammar_pending_think=False, grammar_think_buf=List[UInt8](), grammar_stop=False, grammar_masked_draws=0, grammar_accepted=0, tool_re="", tool_buf=List[UInt8](), grammar_tool=False)


def run_cold(ctx: DeviceContext, mut bufs: WindowBufs, pack: Pack, mega: Bool, prompt: List[Int], split_at: Int, mut chain: Chain) raises -> Snap:
    var n_prompt = len(prompt)
    var n_total = n_prompt + 1
    cold_reset(ctx, bufs)
    load_prompt(ctx, bufs, prompt)
    var wst = fresh_state()
    wst.reset(perf_counter_ns())
    if split_at > 0:
        prefill_to(ctx, bufs, pack, mega, wst, split_at, n_total, n_prompt, prompt, chain, False, List[Int]())
    prefill_to(ctx, bufs, pack, mega, wst, n_prompt - 1, n_total, n_prompt, prompt, chain, False, List[Int]())
    finish(ctx, bufs, pack, mega, wst, n_total, n_prompt)
    return Snap(ctx, bufs, wst.ring, n_total)


def run_request(ctx: DeviceContext, mut bufs: WindowBufs, pack: Pack, mega: Bool, prompt: List[Int], mut chain: Chain, mut cached_out: Int, hints: List[Int]) raises -> Snap:
    # The engine serve path: lookup, restore into slot 0 or cold reset, replay,
    # checkpoints at the boundaries, commit after the final synchronize.
    var n_prompt = len(prompt)
    var n_total = n_prompt + 1
    var idx = chain.lookup(prompt, n_prompt)
    var cached = chain.pos_of(idx)
    chain.invalidate_above(cached)
    if idx >= 0:
        chain.restore(ctx, bufs.convstate_d, bufs.sstate_d, 0, idx)
        ctx.enqueue_memset(bufs.kc32_d, 0)
        ctx.enqueue_memset(bufs.vc32_d, 0)
        ctx.enqueue_memset(bufs.ctr_d, 0)
        ctx.enqueue_memset(bufs.prof_d, 0)
        ctx.synchronize()
    else:
        cold_reset(ctx, bufs)
    load_prompt(ctx, bufs, prompt)
    var wst = fresh_state()
    wst.reset(perf_counter_ns())
    wst.pos = cached
    wst.pos_prev = cached
    if n_prompt - 1 - cached >= PF_MIN:
        prefill_to(ctx, bufs, pack, mega, wst, n_prompt - 1, n_total, n_prompt, prompt, chain, True, hints)
    var cfg = make_cfg(pack, mega, 0, 0, n_total, n_prompt)
    while wst.pos < n_total - 1:
        step_window(ctx, bufs, cfg, wst)
        if wst.pos > cached and wst.pos < n_prompt:
            if wst.pos == n_prompt - 1 or wst.pos % CKPT_PERIOD == 0:
                chain.save(ctx, bufs.convstate_d, bufs.sstate_d, wst.ring, wst.pos, prompt, False, False)
            else:
                var hi = hint_index(hints, wst.pos)
                if hi >= 0:
                    chain.save(ctx, bufs.convstate_d, bufs.sstate_d, wst.ring, wst.pos, prompt, hi == 0, True)
    ctx.synchronize()
    chain.commit()
    cached_out = cached
    return Snap(ctx, bufs, wst.ring, n_total)


def expect_lookup(label: String, chain: Chain, tokens: List[Int], want: Int, mut fails: Int):
    var got = chain.pos_of(chain.lookup(tokens, len(tokens)))
    if got == want:
        print("  PASS lookup", label, "->", got)
    else:
        print("  FAIL lookup", label, "->", got, "want", want)
        fails += 1


def mutated(tokens: List[Int], at: Int) -> List[Int]:
    var out = tokens.copy()
    out[at] = out[at] + 1
    return out^


def test_retention(ctx: DeviceContext, bufs: WindowBufs, packdir: String, mut fails: Int) raises:
    # M1b retention (bench/chat-protocol.md M1b): pinned checkpoints are
    # never evicted; among the rest, a periodic-grid checkpoint (boundary =
    # False) is evicted before a role-boundary one, regardless of position.
    # Pure Chain.save bookkeeping -- no decode, so mega vs window is moot;
    # run once. The token lists are dummies (content doesn't matter, only
    # that each position gets a distinct hash), and the device buffers are
    # only a copy source (the bytes copied are never read back here).
    print("== M1b retention: pinned never evicted, periodic evicted before role-boundary")
    var c = Chain(ctx, 3, packdir)
    # prefix_hash reads tokens[0:pos], so this dummy list must cover the
    # largest pos used below (50); content doesn't matter otherwise.
    var toks = List[Int]()
    for i in range(64):
        toks.append(i)
    c.save(ctx, bufs.convstate_d, bufs.sstate_d, 0, 10, toks, True, True)
    c.commit()
    c.save(ctx, bufs.convstate_d, bufs.sstate_d, 0, 20, toks, False, True)
    c.commit()
    c.save(ctx, bufs.convstate_d, bufs.sstate_d, 0, 30, toks, False, False)
    c.commit()
    if c.count_valid() != 3:
        print("  FAIL expected 3 valid checkpoints before eviction, got", c.count_valid())
        fails += 1
    c.save(ctx, bufs.convstate_d, bufs.sstate_d, 0, 40, toks, False, False)
    c.commit()
    var have10 = False
    var have20 = False
    var have30 = False
    var have40 = False
    for i in range(len(c.items)):
        if c.items[i].valid:
            have10 = have10 or c.items[i].pos == 10
            have20 = have20 or c.items[i].pos == 20
            have30 = have30 or c.items[i].pos == 30
            have40 = have40 or c.items[i].pos == 40
    if have10 and have20 and have40 and not have30:
        print("  PASS periodic-grid checkpoint (30) evicted first; pinned (10) and role-boundary (20) kept")
    else:
        print("  FAIL retention order wrong: have10", have10, " have20", have20, " have30", have30, " have40", have40)
        fails += 1
    c.save(ctx, bufs.convstate_d, bufs.sstate_d, 0, 50, toks, False, False)
    c.commit()
    have10 = False
    have20 = False
    have40 = False
    var have50 = False
    for i in range(len(c.items)):
        if c.items[i].valid:
            have10 = have10 or c.items[i].pos == 10
            have20 = have20 or c.items[i].pos == 20
            have40 = have40 or c.items[i].pos == 40
            have50 = have50 or c.items[i].pos == 50
    if have10 and have20 and have50 and not have40:
        print("  PASS second periodic-grid checkpoint (40) evicted before the role-boundary one (20)")
    else:
        print("  FAIL second eviction wrong: have10", have10, " have20", have20, " have40", have40, " have50", have50)
        fails += 1


def test_kv_share(ctx: DeviceContext, mut bufs: WindowBufs, packdir: String, mut fails: Int) raises:
    # BARO_KV_SHARE: a request restored at a shared checkpoint keeps the other conversation's
    # checkpoints and pages, copies the page holding the restore point, and evicts only when the
    # pool runs out. Bookkeeping plus the copy's bytes; no decode.
    print("== BARO_KV_SHARE: shared pages kept, partial page copied, eviction when full")
    comptime PAGE = N_ATT * NKVH * KVHSTR
    var c = Chain(ctx, 4, packdir)
    c.npages = 6
    c.share_reset(ctx, bufs.kvtab_h, bufs.kvtab_d, bufs.tpages)
    var toks = List[Int]()
    for i in range(400):
        toks.append(i)
    c.map_pages(ctx, -1, 0, 3, bufs.kc_d, bufs.vc_d, bufs.kvtab_h, bufs.kvtab_d, bufs.tpages)
    c.save(ctx, bufs.convstate_d, bufs.sstate_d, 0, 200, toks, False, True)
    c.save(ctx, bufs.convstate_d, bufs.sstate_d, 0, 300, toks, False, True)
    c.commit()
    var mark = ctx.enqueue_create_host_buffer[KVT](PAGE)
    var back = ctx.enqueue_create_host_buffer[KVT](PAGE)
    ctx.synchronize()
    for i in range(PAGE):
        mark[i] = Scalar[KVT](i % 97)
    ctx.enqueue_copy(dst_buf=DeviceBuffer[KVT](ctx, bufs.kc_d.unsafe_ptr().unsafe_offset(Int(c.lives[0][1]) * PAGE), PAGE, owning=False), src_buf=mark)
    var ip = c.lookup(toks, 250)
    c.map_pages(ctx, ip, 200, 3, bufs.kc_d, bufs.vc_d, bufs.kvtab_h, bufs.kvtab_d, bufs.tpages)
    ctx.enqueue_copy(dst_buf=back, src_buf=DeviceBuffer[KVT](ctx, bufs.kc_d.unsafe_ptr().unsafe_offset(Int(c.lives[0][1]) * PAGE), PAGE, owning=False))
    ctx.synchronize()
    var same = True
    for i in range(PAGE):
        if back[i] != mark[i]:
            same = False
    var ia = c.lookup(toks, 400)
    if c.pos_of(ip) != 200 or c.pos_of(ia) != 300 or c.lives[0][0] != 0 or c.lives[0][1] < 3 or c.lives[0][2] < 3 or c.lives[0][1] == c.lives[0][2] or not same:
        print("  FAIL shared map: pos", c.pos_of(ip), c.pos_of(ia), " live", c.lives[0][0], c.lives[0][1], c.lives[0][2], " copy equal", same)
        fails += 1
    c.map_pages(ctx, -1, 0, 4, bufs.kc_d, bufs.vc_d, bufs.kvtab_h, bufs.kvtab_d, bufs.tpages)
    if c.count_valid() != 0 or len(c.lives[0]) != 4:
        print("  FAIL eviction: valid", c.count_valid(), " live", len(c.lives[0]))
        fails += 1


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var packdir = getenv("BARO_PACK", ".work/engine-pack-q4")
    var tokpath = getenv("BARO_PREFIX_TOKENS", "bench/prefill-prompts/p8192.tokens")
    var pack = load_pack(ctx, packdir)
    var bufs = alloc_bufs(ctx, pack, T_MAX)
    var P = read_tokens(tokpath, A_LEN + B_LEN)
    var A = List[Int]()
    for i in range(A_LEN):
        A.append(P[i])
    var fails = 0
    print("test_prefix: pack", packdir, " q4", pack.pack_q4, " tokens", tokpath, " A", len(A), " B", B_LEN, " tmax", T_MAX)
    print("checkpoints: cap 8, bytes", Float64(CKPT_BYTES) / 1e6, "MB each, period", CKPT_PERIOD)

    for arm in range(2):
        var mega = arm == 0
        var chain = Chain(ctx, 8, packdir)
        var t0 = perf_counter_ns()
        var cold_u = run_cold(ctx, bufs, pack, mega, P, 0, chain)
        var cold_s = run_cold(ctx, bufs, pack, mega, P, A_LEN - 1, chain)
        compare("cold_s vs cold_u (chunk split at 1087, no checkpoint)", cold_s, cold_u, mega, fails)
        var c0 = 0
        var snapA = run_request(ctx, bufs, pack, mega, A, chain, c0, List[Int]())
        print("  request A: cached", c0, " checkpoints", chain.count_valid(), " tok", snapA.tok)
        var c1 = 0
        var rest = run_request(ctx, bufs, pack, mega, P, chain, c1, List[Int]())
        print("  request P: cached", c1, " checkpoints", chain.count_valid())
        if c1 != A_LEN - 1:
            print("  FAIL restore point", c1, "want", A_LEN - 1)
            fails += 1
        compare("restore(A)+replay(B) vs cold_u", rest, cold_u, mega, fails)
        compare("restore(A)+replay(B) vs cold_s", rest, cold_s, mega, fails)

        print("== lookup (chain 1024 / 1087 / 1151)")
        expect_lookup("exact P", chain, P, A_LEN + B_LEN - 1, fails)
        expect_lookup("token 0 mutated", chain, mutated(P, 0), 0, fails)
        expect_lookup("token 1023 mutated", chain, mutated(P, 1023), 0, fails)
        expect_lookup("token 1025 mutated", chain, mutated(P, 1025), 1024, fails)
        expect_lookup("token 1086 mutated (last hashed token of A)", chain, mutated(P, 1086), 1024, fails)
        expect_lookup("token 1087 mutated (last token of A, outside tokens[0:1087])", chain, mutated(P, 1087), A_LEN - 1, fails)
        expect_lookup("token 1100 mutated", chain, mutated(P, 1100), A_LEN - 1, fails)
        var ci = -1
        for i in range(len(chain.items)):
            if chain.items[i].valid and chain.items[i].pos == A_LEN - 1:
                ci = i
        var saved_hash = chain.items[ci].hash.copy()
        chain.items[ci].hash[0] = chain.items[ci].hash[0] ^ UInt8(1)
        expect_lookup("corrupted hash on the 1087 checkpoint", chain, mutated(P, 1100), 1024, fails)
        chain.items[ci].hash = saved_hash^

        var c2 = 0
        var rep = run_request(ctx, bufs, pack, mega, P, chain, c2, List[Int]())
        print("  request P again: cached", c2, " checkpoints", chain.count_valid())
        if c2 != A_LEN + B_LEN - 1:
            print("  FAIL repeat restore point", c2, "want", A_LEN + B_LEN - 1)
            fails += 1
        compare("repeat(P) restored at 1151 vs cold_u", rep, cold_u, mega, fails)

        var Pm = mutated(P, 1050)
        var cold_m = run_cold(ctx, bufs, pack, mega, Pm, 0, chain)
        var c3 = 0
        var rest_m = run_request(ctx, bufs, pack, mega, Pm, chain, c3, List[Int]())
        print("  request P' (token 1050 mutated): cached", c3, " checkpoints", chain.count_valid())
        if c3 != 1024:
            print("  FAIL P' restore point", c3, "want 1024")
            fails += 1
        compare("restore(P') at 1024 vs cold_u(P') (chunk-identical)", rest_m, cold_m, mega, fails)

        # M1b role-boundary checkpoint (bench/chat-protocol.md M1b): a hint
        # position off the periodic-1024 grid, index 0 -> pinned+boundary.
        print("== M1b role-boundary checkpoint at 300 (fresh chain)")
        var rb_chain = Chain(ctx, 8, packdir)
        var A2 = List[Int]()
        for i in range(600):
            A2.append(P[i])
        var rb_hints: List[Int] = [300]
        var c4 = 0
        var _rbsnap = run_request(ctx, bufs, pack, mega, A2, rb_chain, c4, rb_hints)
        var rb_idx = -1
        for i in range(len(rb_chain.items)):
            if rb_chain.items[i].valid and rb_chain.items[i].pos == 300:
                rb_idx = i
        if rb_idx < 0 or not rb_chain.items[rb_idx].pinned or not rb_chain.items[rb_idx].boundary:
            print("  FAIL role-boundary hint at 300 not saved as pinned+boundary")
            fails += 1
        else:
            print("  PASS role-boundary hint at 300 saved via a real request (pinned, boundary)")
        var lookup_301 = rb_chain.pos_of(rb_chain.lookup(P, 301))
        if lookup_301 != 300:
            print("  FAIL lookup constrained to n=301 chose", lookup_301, "want 300")
            fails += 1
        else:
            print("  PASS lookup constrained to n=301 finds the role-boundary checkpoint at 300")
        # Force restore specifically from the role-boundary checkpoint: drop
        # the ordinary prompt-end checkpoint (599) that would otherwise win
        # (lookup prefers the larger matching position).
        for i in range(len(rb_chain.items)):
            if rb_chain.items[i].valid and rb_chain.items[i].pos == 599:
                rb_chain.items[i].valid = False
        var c5 = 0
        var rb_full = run_request(ctx, bufs, pack, mega, P, rb_chain, c5, List[Int]())
        if c5 != 300:
            print("  FAIL role-boundary restore point", c5, "want 300")
            fails += 1
        compare("restore(role-boundary@300)+replay vs cold_u", rb_full, cold_u, mega, fails)

        print("arm", "megakernel" if mega else "window", "done in", Float64(perf_counter_ns() - t0) / 1e9, "s")

    test_retention(ctx, bufs, packdir, fails)
    test_kv_share(ctx, bufs, packdir, fails)

    if fails == 0:
        print("PASS: prefix checkpoints byte-exact")
    else:
        print("FAIL: prefix checkpoints,", fails, "checks failed")
        exit(1)
