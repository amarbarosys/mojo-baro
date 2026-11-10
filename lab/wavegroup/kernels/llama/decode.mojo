# M4: Llama greedy decode. Prompt tokens 0..T-2 in one batched forward (unless "noprefill"), then one token per step;
# KV cache paged as mojo-baro's attention expects, argmax on the host.
# usage: decode MODEL.gguf IDS_CSV|TEXT N [noprefill][mega]   -> prints the N generated ids, comma separated (build -D WMMA_TB=1 -D WMMA_C_F16=0)
# TEXT (anything but digits and commas) goes through the GGUF's own tokenizer (BOS added) and the output is also printed as text;
# text mode stops at the GGUF's EOS id (checked every EOS_EVERY tokens: one copy + sync, ids stay on the device in between).
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, HostBuffer
from layout import TileTensor, row_major
from attn import KVT, KVPAGE, KVHSTR
from weights import Weights
from llama import Pass, forward, norm, linear, argmax_part, argmax_fin, AMB, P32
from mega import wg_decode_token, NW, MG, MT
from model import H, VOCAB, NKVH, N_LAYERS
from shard_index import TensorRef
from tokenizer import Tokenizer

comptime TMAX = 512
comptime PAGES = TMAX // KVPAGE
comptime EOS_EVERY = 32

def gen_ids(tok_h: HostBuffer[DType.int32], at: Int, n: Int) -> List[Int]:
    var ids = List[Int]()
    for i in range(n):
        ids.append(Int(tok_h[at + i]))
    return ids^

def eos_at(tok_h: HostBuffer[DType.int32], at: Int, n: Int, eos: Int) -> Int:
    """Index of the first EOS among the n generated ids, n if none."""
    for i in range(n):
        if Int(tok_h[at + i]) == eos:
            return i
    return n

def main() raises:
    var args = argv()
    var ctx = DeviceContext()
    var w = Weights(ctx, String(args[1]))
    var arg = String(args[2])
    var is_text = False
    for b in arg.as_bytes():
        is_text = is_text or not ((b >= 48 and b <= 57) or b == 44)
    var tk = Tokenizer(String(args[1]))
    var prompt = List[Int]()
    if is_text:
        prompt = tk.encode(arg)
    else:
        for s in arg.split(","):
            prompt.append(Int(s))
    var n_gen = Int(args[3])
    var eps = Float32(w.idx.gguf.akv_float("attention.layer_norm_rms_epsilon", 1e-5))
    if len(prompt) + n_gen > TMAX:
        raise Error("decode: context over " + String(TMAX))

    comptime NKVE = PAGES * N_LAYERS * NKVH * KVHSTR
    var kc = ctx.enqueue_create_buffer[KVT](NKVE)
    var vc = ctx.enqueue_create_buffer[KVT](NKVE)
    var Kc = TileTensor(kc, row_major[NKVE]())
    var Vc = TileTensor(vc, row_major[NKVE]())
    var tab_h = ctx.enqueue_create_host_buffer[DType.int32](PAGES)
    var tok_h = ctx.enqueue_create_host_buffer[DType.int32](TMAX) # one id slot per position: no host buffer is rewritten in flight
    ctx.synchronize()
    for i in range(PAGES):
        tab_h[i] = Int32(i)
    for i in range(len(prompt)):
        tok_h[i] = Int32(prompt[i])
    var tab_d = ctx.enqueue_create_buffer[DType.int32](PAGES)
    var tok_d = ctx.enqueue_create_buffer[DType.int32](TMAX)
    ctx.enqueue_copy(dst_buf=tab_d, src_buf=tab_h)
    ctx.enqueue_copy(dst_buf=tok_d, src_buf=tok_h)
    var tab = Pointer[Int32, MutAnyOrigin](unsafe_from_address=Int(tab_d.unsafe_ptr()))
    var toks = Pointer[Int32, MutAnyOrigin](unsafe_from_address=Int(tok_d.unsafe_ptr()))
    var one = Pass[1](ctx)

    var t0 = perf_counter_ns()
    var start = 0
    var head = len(prompt) - 1
    var opt = String(args[4]) if len(args) > 4 else String()
    if head > 0 and "noprefill" not in opt:
        if head <= 128:
            forward(ctx, w, Pass[128](ctx), toks, 0, head, Kc, Vc, tab, eps)
        else:
            forward(ctx, w, Pass[TMAX](ctx), toks, 0, head, Kc, Vc, tab, eps)
        ctx.synchronize()
        start = head
    var prefill_ms = Float64(perf_counter_ns() - t0) / 1e6
    var t_first = 0
    if "mega" in opt: # one dispatch per token, ids stay on the device, one sync at the end
        comptime NT = N_LAYERS * NW + 3
        var wt_h = ctx.enqueue_create_host_buffer[DType.int64](NT)
        var ty_h = ctx.enqueue_create_host_buffer[DType.int32](NT)
        ctx.synchronize()
        var names: List[String] = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down", "attn_norm", "ffn_norm"]
        for l in range(N_LAYERS):
            for j in range(NW):
                var n = "blk." + String(l) + "." + names[j] + ".weight"
                wt_h[l * NW + j] = Int64(w.addr(n))
                ty_h[l * NW + j] = Int32(w.ref(n).gtype)
        var tail: List[String] = ["output_norm.weight", "token_embd.weight", "rope_freqs.weight"]
        for j in range(3):
            wt_h[N_LAYERS * NW + j] = Int64(w.addr(tail[j]))
            ty_h[N_LAYERS * NW + j] = Int32(w.ref(tail[j]).gtype)
        var wt = ctx.enqueue_create_buffer[DType.int64](NT)
        var ty = ctx.enqueue_create_buffer[DType.int32](NT)
        var hv = ctx.enqueue_create_buffer[DType.float32](MG)
        var hi = ctx.enqueue_create_buffer[DType.int32](MG)
        var cw = ctx.enqueue_create_buffer[DType.uint32](3)
        ctx.enqueue_copy(dst_buf=wt, src_buf=wt_h)
        ctx.enqueue_copy(dst_buf=ty, src_buf=ty_h)
        ctx.enqueue_memset(cw, 0)
        def f32p(name: String) raises {one} -> Pointer[Float32, MutAnyOrigin]:
            return Pointer[Float32, MutAnyOrigin](unsafe_from_address=one.o[name])
        var t1 = perf_counter_ns()
        for pos in range(start, len(prompt) + n_gen - 1):
            ctx.enqueue_function[wg_decode_token[type_of(Kc).LayoutType]](
                Pointer[Int64, MutAnyOrigin](unsafe_from_address=Int(wt.unsafe_ptr())), Pointer[Int32, MutAnyOrigin](unsafe_from_address=Int(ty.unsafe_ptr())),
                f32p("x"), f32p("qkv"), f32p("att"), f32p("act"), f32p("pg"), Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(hv.unsafe_ptr())),
                Pointer[Int32, MutAnyOrigin](unsafe_from_address=Int(hi.unsafe_ptr())), Pointer[Scalar[DType.uint32], MutAnyOrigin](unsafe_from_address=Int(cw.unsafe_ptr())),
                toks, Kc, Vc, tab, Int32(pos), Int32(len(prompt)), eps, grid_dim=MG, block_dim=MT)
        ctx.enqueue_copy(dst_buf=tok_h, src_buf=tok_d)
        var cw_h = ctx.enqueue_create_host_buffer[DType.uint32](3)
        ctx.enqueue_copy(dst_buf=cw_h, src_buf=cw)
        ctx.synchronize()
        var ms = Float64(perf_counter_ns() - t1) / 1e6
        if cw_h[2] != 0:
            raise Error("decode: grid barrier failed (NOT-RESIDENT)")
        var s = String()
        for i in range(n_gen):
            s += (String(",") if i else String()) + String(Int(tok_h[len(prompt) + i]))
        print(s)
        if is_text:
            print(tk.decode(gen_ids(tok_h, len(prompt), n_gen)))
        print("prefill_ms", prefill_ms, "mega_ms", ms, "decode_tok_s", Float64(n_gen) / ms * 1e3)
        return
    for pos in range(start, len(prompt) + n_gen - 1): # ids stay on the device: argmax writes toks[pos + 1], no sync per token
        forward(ctx, w, one, toks + pos, pos, 1, Kc, Vc, tab, eps)
        if pos + 1 < len(prompt):
            continue
        norm(ctx, w, one, "output_norm.weight", 1, eps)
        linear[0, VOCAB, H](ctx, w, one, "token_embd.weight", 1, "xn", "logits", "logits")
        var am = one.o["am"]
        ctx.enqueue_function[argmax_part](P32(unsafe_from_address=one.o["logits"]), Int32(VOCAB), P32(unsafe_from_address=am),
            Pointer[Int32, MutAnyOrigin](unsafe_from_address=am + AMB * 4), grid_dim=AMB, block_dim=256)
        ctx.enqueue_function[argmax_fin](P32(unsafe_from_address=am), Pointer[Int32, MutAnyOrigin](unsafe_from_address=am + AMB * 4), toks + pos + 1, grid_dim=1, block_dim=1)
        if t_first == 0:
            ctx.synchronize()
            t_first = Int(perf_counter_ns() - t0)
        var done = pos + 2 - len(prompt) # tokens generated so far
        if is_text and tk.eos_id >= 0 and done % EOS_EVERY == 0:
            ctx.enqueue_copy(dst_buf=tok_h, src_buf=tok_d)
            ctx.synchronize()
            if eos_at(tok_h, len(prompt), done, tk.eos_id) < done:
                break
    ctx.enqueue_copy(dst_buf=tok_h, src_buf=tok_d)
    ctx.synchronize()
    var dec_ms = Float64(Int(perf_counter_ns() - t0) - t_first) / 1e6
    var n_run = n_gen
    if is_text and tk.eos_id >= 0:
        n_run = min(n_gen, ((eos_at(tok_h, len(prompt), n_gen, tk.eos_id) + EOS_EVERY) // EOS_EVERY) * EOS_EVERY) # tokens computed
        n_gen = eos_at(tok_h, len(prompt), n_gen, tk.eos_id)
    var s = String()
    for i in range(n_gen):
        s += (String(",") if i else String()) + String(Int(tok_h[len(prompt) + i]))
    print(s)
    if is_text:
        print(tk.decode(gen_ids(tok_h, len(prompt), n_gen)))
    print("prefill_ms", prefill_ms, "prompt_ms", Float64(t_first) / 1e6, "decode_tok_s", Float64(n_run - 1) / dec_ms * 1e3)
