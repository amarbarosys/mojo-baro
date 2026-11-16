"""gemma4 (Forge-1 E2B) decode engine: one token per step, f16 weights read from the GGUF in place, f32 activations.

Pack: tools/gemma4-pack.mojo (index.txt, profile.txt, profile.mojo, pack.bin -> the GGUF). Spec of the step:
docs/provenance/gemma4.md. Attention: kernels/gemma4_attn.mojo with two KV pools, S (HD_S, sliding WIN) and F
(HD_F, full); a layer without own KV attends its source layer's slot (profile.txt).

Env (same CLI contract as serve/spark.mojo for bench/dense-run.sh):
  BARO_PACK=dir  BARO_PROMPT=ids file  BARO_GEN=n  BARO_FORCE=ids file (teacher-forced: feed these, count argmax hits)
  BARO_DUMP_DIR=dir  write l_out-<il> (every prompt position, [n_prompt, H] f32) in the tdiff layout
  BARO_SERVE=1  serve/PROTOCOL.md line loop on stdin (greedy; sampling fields parsed, not used)
  BARO_KV_ROUTE=own  reverse arm: a shared-KV layer attends slot 0 of its pool instead of its source's slot
Prints: generated: ..., predicted: ..., forced agreement: A / N, tok/s_gen: X (N steps, S s)

Build: ./.venv/bin/mojo build serve/gemma4.mojo -I PACKDIR -I . -I kernels -I serve -I tools -o .work/gemma4/engine
"""
from std.collections import Dict
from std.ffi import c_ssize_t, external_call
from std.math import ceildiv, sqrt
from std.os import getenv, setenv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from serve_proto import read_line, is_cancel_line, idle_cancel_note, parse_request, default_sample_params
from wire import err_line, window_exceeded, window_error
from layout import TileTensor, row_major
from matmul_skinny import ROW_WAVES, ROW_THREADS
from dattn import dkv_tab_off, KVPAGE
from gemma4_attn import amar_attn_decode_gemma4, amar_attn_merge_gemma4, g4_gb, g4_nld, g4_bt, g4_nsplit, G4_PHDR
from gemma4_ops import g4_gemv, g4_rmsnorm, g4_rope, g4_embed, g4_ple, NT, RMS_W, RMS_RES, RMS_SC, F16P, F32P, I32P
from spark_kernels import amar_argmax_part, amar_argmax_final
from profile import H, VOCAB, NL, NQH, HD_S, HD_F, WIN, PLE, FFN_MAX, NS_SLOTS, NF_SLOTS, EPS, BASE_F, BASE_S, TMAX

comptime f32 = DType.float32
comptime FL = row_major[1 << 27]()
comptime TPAGES = TMAX // KVPAGE
comptime AM_NB = 128
comptime am_l = row_major[AM_NB]()
comptime tok_l = row_major[TMAX]()
comptime v_l = row_major[VOCAB]()
comptime k_amax = amar_argmax_part[AM_NB, type_of(v_l), type_of(am_l), type_of(am_l)]
comptime k_afin = amar_argmax_final[AM_NB, type_of(am_l), type_of(am_l), type_of(tok_l)]
comptime QMAX = NQH * HD_F
comptime PGN = NQH * 8 * (HD_F + G4_PHDR)
# activation arena, f32 offsets
comptime O_X = 0
comptime O_HB = O_X + H
comptime O_AB = O_HB + H
comptime O_QB = O_AB + H
comptime O_OB = O_QB + QMAX
comptime O_KB = O_OB + QMAX
comptime O_VB = O_KB + HD_F
comptime O_GB = O_VB + HD_F
comptime O_PP = O_GB + FFN_MAX
comptime O_PLE = O_PP + NL * PLE
comptime O_PG = O_PLE + NL * PLE
comptime O_ATTP = O_PG + PLE
comptime O_LOG = O_ATTP + PGN
comptime O_AM = O_LOG + VOCAB
comptime ACT_N = O_AM + 2 * AM_NB

# tensors of a layer, in the order of Layer.w
comptime NAMES = ["attn_norm", "attn_q", "attn_q_norm", "attn_k", "attn_v", "attn_k_norm", "attn_output", "post_attention_norm",
    "ffn_norm", "ffn_gate", "ffn_up", "ffn_down", "post_ffw_norm", "inp_gate", "proj", "post_norm", "layer_output_scale"]
comptime W_AN = 0
comptime W_Q = 1
comptime W_QN = 2
comptime W_K = 3
comptime W_V = 4
comptime W_KN = 5
comptime W_O = 6
comptime W_PAN = 7
comptime W_FN = 8
comptime W_G = 9
comptime W_U = 10
comptime W_D = 11
comptime W_PFN = 12
comptime W_IG = 13
comptime W_PJ = 14
comptime W_PN = 15
comptime W_OS = 16


@fieldwise_init
struct Layer(Copyable, Movable):
    var full: Bool
    var hd: Int
    var kv: Bool
    var slot: Int
    var ffn: Int
    var w: List[Int]  # device address per NAMES entry, 0 for K/V tensors of a shared layer


@fieldwise_init
struct Eng(Movable):
    var layers: List[Layer]
    var emb: Int
    var ple_tab: Int
    var ple_proj: Int
    var ple_norm: Int
    var out_norm: Int
    var ff: Int
    var a: Int  # activation arena
    var ks: Int
    var vs: Int
    var kf: Int
    var vf: Int
    var tab: Int  # device page table
    var tabp: Int  # host page table (identity), for pool row offsets
    var toks: Int
    var pred: Int
    var dump: Int  # 0 = no layer dump
    var n_prompt: Int
    var own_route: Bool

    def f(self, o: Int) -> Int:
        return self.a + 4 * o


def read_ids(path: String) raises -> List[Int]:
    var ids = List[Int]()
    for w in open(path, "r").read().replace(",", " ").replace("\n", " ").split(" "):
        if w.byte_length() > 0:
            ids.append(Int(String(w)))
    return ids^


def load(ctx: DeviceContext, packdir: String, mut off: Dict[String, Int]) raises -> DeviceBuffer[DType.uint8]:
    """One device buffer holding the file span of every tensor in index.txt; off[name] = offset in that buffer."""
    var lo = Int.MAX
    var hi = 0
    for line in open(packdir + "/index.txt", "r").read().splitlines():
        var p = line.split(" ")
        var o = Int(String(p[2]))
        off[String(p[0])] = o
        lo = min(lo, o)
        hi = max(hi, o + Int(String(p[3])))
    for k in List(off.keys()):
        off[k] = off[k] - lo
    var total = hi - lo
    var wbuf = ctx.enqueue_create_buffer[DType.uint8](total)
    comptime CHUNK = 1 << 28
    var s0 = ctx.enqueue_create_host_buffer[DType.uint8](CHUNK)
    var s1 = ctx.enqueue_create_host_buffer[DType.uint8](CHUNK)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    with open(packdir + "/pack.bin", "r") as f:
        var fd = f._get_raw_fd()
        var done = 0
        var flip = False
        while done < total:
            var want = min(CHUNK, total - done)
            var sb = s1 if flip else s0
            ctx.synchronize()  # the copy out of this stage two chunks ago is done
            var got = 0
            while got < want:
                var r = external_call["pread", c_ssize_t](fd, sb.unsafe_ptr().unsafe_offset(got), want - got, Int64(lo + done + got))
                if r <= 0:
                    raise Error("short read of pack.bin")
                got += Int(r)
            var dslice = DeviceBuffer[DType.uint8](ctx, wbuf.unsafe_ptr().unsafe_offset(done), want, owning=False)
            ctx.enqueue_copy(dst_buf=dslice, src_buf=sb.create_sub_buffer[DType.uint8](0, want) if want != CHUNK else sb)
            done += want
            flip = not flip
    ctx.synchronize()
    print("pack loaded", total, "bytes in", Float64(perf_counter_ns() - t0) / 1e9, "s")
    return wbuf^


def ftt(ctx: DeviceContext, a: Int) -> TileTensor[f32, type_of(FL), MutAnyOrigin]:
    return rebind[TileTensor[f32, type_of(FL), MutAnyOrigin]](TileTensor(DeviceBuffer[f32](ctx, F32P(unsafe_from_address=a), 1 << 27, owning=False), FL))


def gemv[EPI: Int](ctx: DeviceContext, w: Int, x: Int, y: Int, aux: Int, n: Int, k: Int) raises:
    ctx.enqueue_function[g4_gemv[EPI]](F16P(unsafe_from_address=w), F32P(unsafe_from_address=x), F32P(unsafe_from_address=y),
        F32P(unsafe_from_address=aux), Int32(n), Int32(k), grid_dim=ceildiv(n, ROW_WAVES), block_dim=ROW_THREADS)


def rms(ctx: DeviceContext, x: Int, w: Int, res: Int, dst: Int, sc: Int, rows: Int, n: Int, eps: Float32, flags: Int) raises:
    ctx.enqueue_function[g4_rmsnorm](F32P(unsafe_from_address=x), F32P(unsafe_from_address=w), F32P(unsafe_from_address=res),
        F32P(unsafe_from_address=dst), F32P(unsafe_from_address=sc), Int32(n), eps, Int32(flags), grid_dim=rows, block_dim=NT)


def rope(ctx: DeviceContext, x: Int, dst: Int, rows: Int, hd: Int, pos: Int, base: Float32, ff: Int, use_ff: Bool) raises:
    ctx.enqueue_function[g4_rope](F32P(unsafe_from_address=x), F32P(unsafe_from_address=dst), Int32(hd), Int32(pos), base,
        F32P(unsafe_from_address=ff), Int32(Int(use_ff)), grid_dim=rows, block_dim=hd // 2)


def attend[HD_: Int, WINDOW: Int, NAT: Int](ctx: DeviceContext, e: Eng, kc: Int, vc: Int, t_len: Int, slot: Int) raises:
    comptime k_part = amar_attn_decode_gemma4[HD_, WINDOW, NAT, g4_gb(HD_, WINDOW), g4_nld(HD_, WINDOW), g4_bt(HD_, WINDOW),
        type_of(FL), type_of(FL), type_of(FL), type_of(FL)]
    comptime k_merge = amar_attn_merge_gemma4[HD_, type_of(FL), type_of(FL)]
    var ns = g4_nsplit(WINDOW, t_len)
    ctx.enqueue_function[k_part](ftt(ctx, e.f(O_QB)), ftt(ctx, kc), ftt(ctx, vc), ftt(ctx, e.f(O_OB)), ftt(ctx, e.f(O_ATTP)),
        I32P(unsafe_from_address=e.tab), Int32(t_len), Int32(ns), Int32(slot),
        grid_dim=(NQH // g4_gb(HD_, WINDOW), ns), block_dim=g4_bt(HD_, WINDOW))
    if ns > 1:
        ctx.enqueue_function[k_merge](ftt(ctx, e.f(O_ATTP)), ftt(ctx, e.f(O_OB)), Int32(ns), grid_dim=NQH, block_dim=HD_ // 8)


def step(ctx: DeviceContext, e: Eng, pos: Int) raises:
    """The forward of the token at toks[pos]; leaves the last hidden in x."""
    var x = e.f(O_X)
    var hb = e.f(O_HB)
    var ab = e.f(O_AB)
    var qb = e.f(O_QB)
    var kb = e.f(O_KB)
    var vb = e.f(O_VB)
    var gb = e.f(O_GB)
    var pp = e.f(O_PP)
    var ple = e.f(O_PLE)
    var pg = e.f(O_PG)
    var proj_s = Float32(1) / sqrt(Float32(H))
    var inv2 = Float32(1) / sqrt(Float32(2))
    ctx.enqueue_function[g4_embed](F16P(unsafe_from_address=e.emb), I32P(unsafe_from_address=e.toks), Int32(pos),
        F32P(unsafe_from_address=x), Int32(H), sqrt(Float32(H)), grid_dim=ceildiv(H, NT), block_dim=NT)
    gemv[0](ctx, e.ple_proj, x, pp, 0, NL * PLE, H)
    ctx.enqueue_function[g4_ple](F32P(unsafe_from_address=pp), F16P(unsafe_from_address=e.ple_tab), I32P(unsafe_from_address=e.toks),
        Int32(pos), F32P(unsafe_from_address=e.ple_norm), F32P(unsafe_from_address=ple), Int32(PLE), Int32(NL * PLE),
        EPS / (proj_s * proj_s), inv2, sqrt(Float32(PLE)) * inv2, grid_dim=NL, block_dim=NT)
    var tabp = Pointer[Int32, MutAnyOrigin](unsafe_from_address=e.tabp)
    for il in range(NL):
        ref L = e.layers[il]
        ref w = L.w
        var hd = L.hd
        var bse = BASE_F if L.full else BASE_S
        rms(ctx, x, w[W_AN], 0, hb, 0, 1, H, EPS, RMS_W)
        gemv[0](ctx, w[W_Q], hb, qb, 0, NQH * hd, H)
        rms(ctx, qb, w[W_QN], 0, qb, 0, NQH, hd, EPS, RMS_W)
        rope(ctx, qb, qb, NQH, hd, pos, bse, e.ff, L.full)
        var kc = e.kf if L.full else e.ks
        var vc = e.vf if L.full else e.vs
        var slot = L.slot if L.kv or not e.own_route else 0
        if L.kv:
            var row = 4 * (dkv_tab_off[HD_F, 1, NF_SLOTS](tabp, pos, L.slot, 0) if L.full else dkv_tab_off[HD_S, 1, NS_SLOTS](tabp, pos, L.slot, 0))
            gemv[0](ctx, w[W_K], hb, kb, 0, hd, H)
            gemv[0](ctx, w[W_V], hb, vb, 0, hd, H)
            rms(ctx, kb, w[W_KN], 0, kb, 0, 1, hd, EPS, RMS_W)
            rope(ctx, kb, kc + row, 1, hd, pos, bse, e.ff, L.full)
            rms(ctx, vb, 0, 0, vc + row, 0, 1, hd, EPS, 0)
        if L.full:
            attend[HD_F, 0, NF_SLOTS](ctx, e, kc, vc, pos + 1, slot)
        else:
            attend[HD_S, WIN, NS_SLOTS](ctx, e, kc, vc, pos + 1, slot)
        gemv[0](ctx, w[W_O], e.f(O_OB), ab, 0, H, NQH * hd)
        rms(ctx, ab, w[W_PAN], x, x, 0, 1, H, EPS, RMS_W | RMS_RES)
        rms(ctx, x, w[W_FN], 0, hb, 0, 1, H, EPS, RMS_W)
        gemv[0](ctx, w[W_G], hb, gb, 0, L.ffn, H)
        gemv[1](ctx, w[W_U], hb, gb, 0, L.ffn, H)
        gemv[0](ctx, w[W_D], gb, ab, 0, H, L.ffn)
        rms(ctx, ab, w[W_PFN], x, x, 0, 1, H, EPS, RMS_W | RMS_RES)
        gemv[2](ctx, w[W_IG], x, pg, ple + 4 * il * PLE, PLE, H)
        gemv[0](ctx, w[W_PJ], pg, ab, 0, H, PLE)
        rms(ctx, ab, w[W_PN], x, x, w[W_OS], 1, H, EPS, RMS_W | RMS_RES | RMS_SC)
        if e.dump != 0 and pos < e.n_prompt:
            ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, F32P(unsafe_from_address=e.dump + 4 * (il * e.n_prompt + pos) * H), H, owning=False),
                src_buf=DeviceBuffer[f32](ctx, F32P(unsafe_from_address=x), H, owning=False))


def head(ctx: DeviceContext, e: Eng, pos: Int, forced: Int) raises:
    """Argmax of the logits after position pos into toks[pos + 1] (the forced id there when forced >= 0) and pred[pos + 1].
    No softcap: 30 tanh(l / 30) is monotonic, so the argmax is the same."""
    var hb = e.f(O_HB)
    var lg = e.f(O_LOG)
    rms(ctx, e.f(O_X), e.out_norm, 0, hb, 0, 1, H, EPS, RMS_W)
    gemv[0](ctx, e.emb, hb, lg, 0, VOCAB, H)
    var amv = TileTensor(DeviceBuffer[f32](ctx, F32P(unsafe_from_address=e.f(O_AM)), AM_NB, owning=False), am_l)
    var ami = TileTensor(DeviceBuffer[DType.int32](ctx, I32P(unsafe_from_address=e.f(O_AM + AM_NB)), AM_NB, owning=False), am_l)
    ctx.enqueue_function[k_amax](TileTensor(DeviceBuffer[f32](ctx, F32P(unsafe_from_address=lg), VOCAB, owning=False), v_l),
        amv, ami, Int32(VOCAB), grid_dim=AM_NB, block_dim=256)
    ctx.enqueue_function[k_afin](amv, ami, TileTensor(DeviceBuffer[DType.int32](ctx, I32P(unsafe_from_address=e.toks), TMAX, owning=False), tok_l),
        TileTensor(DeviceBuffer[DType.int32](ctx, I32P(unsafe_from_address=e.pred), TMAX, owning=False), tok_l),
        Int32(pos + 1), Int32(forced), grid_dim=1, block_dim=32)


def serve(ctx: DeviceContext, e: Eng, mut toks_h: HostBuffer[DType.int32], toks_d: DeviceBuffer[DType.int32]) raises:
    """serve/PROTOCOL.md line loop, greedy (sampling fields are parsed and ignored), one request at a time.
    A request overwrites the KV from position 0 and attention reads only [t_lo, t_len), so nothing is cleared."""
    var tok1 = ctx.enqueue_create_host_buffer[DType.int32](1)
    print("{\"ready\":true,\"tmax\":" + String(TMAX) + ",\"mrows\":1,\"kmax\":0,\"spec_k\":0,\"pack\":\"gemma4\"}")
    while True:
        var line = read_line(0)
        if not line:
            break
        var l = line.value()
        var c = is_cancel_line(l)
        if c[0]:
            print(idle_cancel_note(c[1]))
            continue
        var id = 0
        var prompt = List[Int]()
        var n = 0
        var spec = False
        var has_spec = False
        var stop = List[List[Int]]()
        var ckpt = List[Int]()
        var sample = default_sample_params()
        var s1 = String()
        var s2 = String()
        var sp = False
        var m1 = String()
        var m2 = String()
        var mp = 0
        var now = Int64(0)
        var perr = parse_request(l, id, prompt, n, spec, has_spec, stop, ckpt, sample, s1, s2, sp, m1, m2, mp, now)
        if perr == "" and (len(prompt) == 0 or n < 1):
            perr = "empty prompt or n < 1"
        if perr == "" and window_exceeded(len(prompt), n, TMAX):
            perr = window_error(len(prompt), n, TMAX)
        if perr != "":
            print(err_line(id, perr))
            continue
        for i in range(len(prompt)):
            toks_h[i] = Int32(prompt[i])
        ctx.enqueue_copy(dst_buf=toks_d, src_buf=toks_h)
        var t0 = perf_counter_ns()
        for pos in range(len(prompt)):
            step(ctx, e, pos)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var gen = List[Int]()
        var finish = String("length")
        for i in range(n):
            var pos = len(prompt) - 1 + i
            head(ctx, e, pos, -1)
            ctx.enqueue_copy(dst_buf=tok1, src_buf=toks_d.create_sub_buffer[DType.int32](pos + 1, 1))
            if i + 1 < n:
                step(ctx, e, pos + 1)  # queued before the sync: the host prints while the GPU runs the next step
            ctx.synchronize()
            gen.append(Int(tok1[0]))
            print("{\"id\":" + String(id) + ",\"tok\":" + String(tok1[0]) + "}")
            var hit = False
            for s in stop:
                if len(s) > 0 and len(s) <= len(gen):
                    var eq = True
                    for k in range(len(s)):
                        if gen[len(gen) - len(s) + k] != s[k]:
                            eq = False
                    hit = hit or eq
            if hit:
                finish = "stop"
                break
        ctx.synchronize()
        var dec = Float64(perf_counter_ns() - t1) / 1e9
        var ng = len(gen)
        print("{\"id\":" + String(id) + ",\"done\":true,\"n\":" + String(ng) + ",\"prefill_s\":" + String(Float64(t1 - t0) / 1e9)
            + ",\"decode_s\":" + String(dec) + ",\"tok_s\":" + String(Float64(ng - 1) / dec if ng > 1 else 0.0) + ",\"finish\":\"" + finish + "\"}")
    _ = tok1.unsafe_ptr()


def main() raises:
    if getenv("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT", "") == "":
        _ = setenv("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT", "60")
    print("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT:", getenv("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT", ""))
    var ctx = DeviceContext()
    var packdir = getenv("BARO_PACK", ".work/gemma4/pack")
    var prompt = read_ids(getenv("BARO_PROMPT", "")) if getenv("BARO_PROMPT", "") != "" else List[Int]()
    var n_gen = Int(getenv("BARO_GEN", "64"))
    var force = List[Int]()
    if getenv("BARO_FORCE", "") != "":
        force = read_ids(getenv("BARO_FORCE", ""))
    var dump_dir = getenv("BARO_DUMP_DIR", "")
    var own_route = getenv("BARO_KV_ROUTE", "") == "own"
    var n_prompt = len(prompt)
    var serving = getenv("BARO_SERVE", "0") == "1"
    if not serving and (n_prompt == 0 or n_prompt + n_gen > TMAX):
        raise Error("prompt of " + String(n_prompt) + " ids + BARO_GEN " + String(n_gen) + " must fit TMAX " + String(TMAX))

    var off = Dict[String, Int]()
    var wbuf = load(ctx, packdir, off)
    var base = Int(wbuf.unsafe_ptr())
    var layers = List[Layer]()
    for line in open(packdir + "/profile.txt", "r").read().splitlines():
        var p = line.split(" ")
        var pre = "blk." + String(p[1]) + "."
        var kv = String(p[4]) == "1"
        var w = List[Int]()
        var names = materialize[NAMES]()
        for i in range(len(names)):
            w.append(base + off[pre + names[i] + ".weight"] if kv or (i != W_K and i != W_V and i != W_KN) else 0)
        layers.append(Layer(String(p[2]) == "F", Int(String(p[3])), kv, Int(String(p[6])), Int(String(p[7])), w^))
    if len(layers) != NL:
        raise Error("profile.txt has " + String(len(layers)) + " layers, profile.mojo " + String(NL))

    var act = ctx.enqueue_create_buffer[f32](ACT_N)
    var ks = ctx.enqueue_create_buffer[f32](TPAGES * NS_SLOTS * KVPAGE * HD_S)
    var vs = ctx.enqueue_create_buffer[f32](TPAGES * NS_SLOTS * KVPAGE * HD_S)
    var kf = ctx.enqueue_create_buffer[f32](TPAGES * NF_SLOTS * KVPAGE * HD_F)
    var vf = ctx.enqueue_create_buffer[f32](TPAGES * NF_SLOTS * KVPAGE * HD_F)
    var tab_h = ctx.enqueue_create_host_buffer[DType.int32](TPAGES)
    var tab_d = ctx.enqueue_create_buffer[DType.int32](TPAGES)
    var toks_h = ctx.enqueue_create_host_buffer[DType.int32](TMAX)
    var toks_d = ctx.enqueue_create_buffer[DType.int32](TMAX)
    var pred_h = ctx.enqueue_create_host_buffer[DType.int32](TMAX)
    var pred_d = ctx.enqueue_create_buffer[DType.int32](TMAX)
    var n_dump = NL * n_prompt * H if dump_dir != "" else 1
    var dump_d = ctx.enqueue_create_buffer[f32](n_dump)
    for i in range(TPAGES):
        tab_h[i] = Int32(i)
    for i in range(TMAX):
        toks_h[i] = Int32(prompt[i]) if i < n_prompt else 0
    ctx.enqueue_copy(dst_buf=tab_d, src_buf=tab_h)
    ctx.enqueue_copy(dst_buf=toks_d, src_buf=toks_h)
    ctx.synchronize()
    var e = Eng(layers^, base + off["token_embd.weight"], base + off["per_layer_token_embd.weight"], base + off["per_layer_model_proj.weight"],
        base + off["per_layer_proj_norm.weight"], base + off["output_norm.weight"], base + off["rope_freqs.weight"],
        Int(act.unsafe_ptr()), Int(ks.unsafe_ptr()), Int(vs.unsafe_ptr()), Int(kf.unsafe_ptr()), Int(vf.unsafe_ptr()),
        Int(tab_d.unsafe_ptr()), Int(tab_h.unsafe_ptr()), Int(toks_d.unsafe_ptr()), Int(pred_d.unsafe_ptr()),
        Int(dump_d.unsafe_ptr()) if dump_dir != "" else 0, n_prompt, own_route)
    print("gemma4 engine: H", H, "layers", NL, "TMAX", TMAX, "prompt", n_prompt, "gen", n_gen, "force", len(force), "kv_route", "own" if own_route else "source")

    if serving:
        serve(ctx, e, toks_h, toks_d)
        _ = (wbuf.unsafe_ptr(), act.unsafe_ptr(), ks.unsafe_ptr(), vs.unsafe_ptr(), kf.unsafe_ptr(), vf.unsafe_ptr(), dump_d.unsafe_ptr())
        _ = (tab_d.unsafe_ptr(), toks_d.unsafe_ptr(), pred_d.unsafe_ptr(), tab_h.unsafe_ptr())
        return
    var t_pf = perf_counter_ns()
    for pos in range(n_prompt):
        step(ctx, e, pos)
    ctx.synchronize()
    var t_gen = perf_counter_ns()
    for i in range(n_gen):
        var pos = n_prompt - 1 + i
        head(ctx, e, pos, force[i] if i < len(force) else -1)
        if i + 1 < n_gen:
            step(ctx, e, pos + 1)
    ctx.synchronize()
    var dt = Float64(perf_counter_ns() - t_gen) / 1e9
    print("prefill_s:", Float64(t_gen - t_pf) / 1e9, "(", n_prompt, "positions, one step each )")

    ctx.enqueue_copy(dst_buf=toks_h, src_buf=toks_d)
    ctx.enqueue_copy(dst_buf=pred_h, src_buf=pred_d)
    ctx.synchronize()
    var gs = String()
    var ps = String()
    var agree = 0
    for i in range(n_gen):
        gs += String(toks_h[n_prompt + i]) + " "
        ps += String(pred_h[n_prompt + i]) + " "
        if i < len(force) and Int(pred_h[n_prompt + i]) == force[i]:
            agree += 1
    print("generated:", gs)
    if len(force) > 0:
        print("predicted:", ps)
        print("forced agreement:", agree, "/", min(n_gen, len(force)))
    print("tok/s_gen:", Float64(n_gen) / dt, "(", n_gen, "steps,", dt, "s )")

    if dump_dir != "":
        var dh = ctx.enqueue_create_host_buffer[f32](n_dump)
        ctx.enqueue_copy(dst_buf=dh, src_buf=dump_d)
        ctx.synchronize()
        var idx = String()
        with open(dump_dir + "/data.bin", "w") as f:
            f.write_bytes(Span[UInt8](unsafe_ptr=dh.unsafe_ptr().unsafe_bitcast[UInt8](), length=n_dump * 4))
        for il in range(NL):
            idx += "l_out-" + String(il) + " f32 " + String(il * n_prompt * H * 4) + " " + String(n_prompt * H) + "\n"
        with open(dump_dir + "/index.txt", "w") as f:
            f.write(idx)
        print("dumped", NL, "layers x", n_prompt, "positions to", dump_dir)
    _ = (wbuf.unsafe_ptr(), act.unsafe_ptr(), ks.unsafe_ptr(), vs.unsafe_ptr(), kf.unsafe_ptr(), vf.unsafe_ptr(), dump_d.unsafe_ptr())
    _ = (tab_d.unsafe_ptr(), toks_d.unsafe_ptr(), pred_d.unsafe_ptr(), tab_h.unsafe_ptr())
