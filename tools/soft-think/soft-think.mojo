"""Soft Thinking (arXiv 2505.15778, training-free) as a proposer backend for the self-optimising loop, arm (c).

During the thinking phase each step feeds back a concept token instead of a sampled one: the next input embedding is
the probability-weighted mix of the top-n token embeddings (eq. 5), after a temperature / top-k / top-p filter and
renormalisation. Thinking ends when the most probable token is </think>, or by Cold Stop (k consecutive steps with
entropy below tau insert </think>). The answer after </think> is sampled discretely. --mode discrete runs the same
driver with sampled thinking tokens: the control that isolates the soft feedback from the backend change.

Usage: .work/soft-think/soft-think MODEL.gguf REQUEST.json REPLY.json [--mode soft|discrete] [--n-ctx 32768]
         [--temp 0.6] [--top-k 30] [--top-p 0.95] [--top-n 15] [--tau 0.05] [--cold 256] [--seed 1] [--max-tokens N] [--ngl 99]
REQUEST is the OpenAI-shaped body tools/loop-propose.mojo writes; REPLY gets the same shape back (content,
reasoning_content = argmax trace of the thinking, finish_reason, usage) plus a "soft" stats object. Defaults are the
paper's (T 0.6, top-k 30, top-p 0.95, n 15 best for QwQ-32B; tau / k from its swept sets). Chat rendering follows the
27B's own template (system/user turns, earlier assistant turns as <think>\\n\\n</think>\\n\\n + content, generation
prompt <|im_start|>assistant\\n<think>\\n). libllama is reached through tools/soft-think/soft_shim.c.
Build: tools/soft-think/build.sh. Runs on the GPU (n_gpu_layers 99): under gpu-wait.
"""
from std.ffi import OwnedDLHandle
from std.sys import argv, exit, stderr
from std.math import exp, log
from std.random import seed, random_float64
from grammar.json_value import JSONDoc, parse_json_bytes


def die(msg: String):
    print("FAIL soft-think:", msg, file=stderr)
    exit(1)


def rd(p: String) raises -> String:
    with open(p, "r") as f:
        return f.read()


def wr(p: String, s: String) raises:
    with open(p, "w") as f:
        f.write(s)


def json_str(s: String) -> String:
    comptime HEX = "0123456789abcdef"
    var o = List[UInt8]()
    o.append(34)
    for b in s.as_bytes():
        if b == 34 or b == 92:
            o.append(92)
            o.append(b)
        elif b == 10:
            o.extend("\\n".as_bytes())
        elif b == 13:
            o.extend("\\r".as_bytes())
        elif b == 9:
            o.extend("\\t".as_bytes())
        elif b < 32:
            o.extend("\\u00".as_bytes())
            o.append(HEX.as_bytes()[Int(b >> 4)])
            o.append(HEX.as_bytes()[Int(b & 15)])
        else:
            o.append(b)
    o.append(34)
    return String(unsafe_from_utf8=o^)


@fieldwise_init
struct Pick(Copyable, Movable):
    var ids: List[Int]
    var w: List[Float64]
    var entropy: Float64
    var top: Int


def filter_step(lp: Int, nv: Int, temp: Float64, top_k: Int, top_p: Float64, top_n: Int) -> Pick:
    """Temperature softmax over the whole vocabulary (entropy H), top-k by logit, top-p within it, then the first
    top_n kept tokens renormalised (top_n <= 0 keeps every top-p survivor, the discrete sampler's set)."""
    var lg = Pointer[Float32, MutAnyOrigin](unsafe_from_address=lp)
    var m = Float64(lg[unsafe_offset=0])
    for i in range(1, nv):
        m = max(m, Float64(lg[unsafe_offset=i]))
    var z = 0.0
    var es = 0.0
    var kid = List[Int](length=top_k, fill=-1)
    var kv = List[Float64](length=top_k, fill=-1e300)
    for i in range(nv):
        var s = (Float64(lg[unsafe_offset=i]) - m) / temp
        var e = exp(s)
        z += e
        es += e * s
        if s > kv[top_k - 1]:
            var j = top_k - 1
            while j > 0 and kv[j - 1] < s:
                kv[j] = kv[j - 1]
                kid[j] = kid[j - 1]
                j -= 1
            kv[j] = s
            kid[j] = i
    var ids = List[Int]()
    var w = List[Float64]()
    var cum = 0.0
    var lim = top_n if top_n > 0 else top_k
    for j in range(top_k):
        if kid[j] < 0 or len(ids) >= lim:
            break
        var p = exp(kv[j]) / z
        ids.append(kid[j])
        w.append(p)
        cum += p
        if cum >= top_p:
            break
    var t = 0.0
    for x in w:
        t += x
    for j in range(len(w)):
        w[j] /= t
    return Pick(ids^, w^, log(z) - es / z, kid[0])


def sample(p: Pick) -> Int:
    var r = random_float64()
    var c = 0.0
    for j in range(len(p.ids)):
        c += p.w[j]
        if r <= c:
            return p.ids[j]
    return p.ids[len(p.ids) - 1]


def piece(lib: OwnedDLHandle, id: Int, buf: Int) raises -> String:
    var n = Int(lib.get_function[Int32]("st_piece")(Int32(id), buf, Int32(255)))
    var bp = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=buf)
    var o = List[UInt8]()
    for i in range(max(n, 0)):
        o.append(bp[unsafe_offset=i])
    return String(unsafe_from_utf8=o^)


def tokenize(lib: OwnedDLHandle, text: String) raises -> List[Int]:
    var cap = text.byte_length() + 16
    var buf = List[Int32](length=cap, fill=0)
    var t = text
    var n = Int(lib.get_function[Int32]("st_tokenize")(t.as_c_string_span().ptr(), Int32(text.byte_length()),
                                                       buf.unsafe_ptr(), Int32(cap), Int32(1)))
    var out = List[Int]()
    for i in range(n):
        out.append(Int(buf[i]))
    if n < 0:
        raise Error("tokenize failed")
    return out^


def render(doc: JSONDoc) raises -> String:
    var msgs = doc.get(doc.get_field(doc.root, "messages"))
    var out = String()
    for k in range(len(msgs.arr)):
        var m = msgs.arr[k]
        var role = doc.get(doc.get_field(m, "role")).s
        var content = doc.get(doc.get_field(m, "content")).s
        if role == "assistant":
            out += "<|im_start|>assistant\n<think>\n\n</think>\n\n" + content + "<|im_end|>\n"
        else:
            out += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n"
    return out + "<|im_start|>assistant\n<think>\n"


def main() raises:
    var av = argv()
    if len(av) < 4:
        die("usage: soft-think MODEL.gguf REQUEST.json REPLY.json [options], see tools/soft-think/soft-think.mojo")
    var model = String(av[1])
    var req = String(av[2])
    var rep = String(av[3])
    var mode = String("soft")
    var n_ctx = 32768
    var temp = 0.6
    var top_k = 30
    var top_p = 0.95
    var top_n = 15
    var tau = 0.05
    var cold = 256
    var sd = 1
    var max_tokens = -1
    var ngl = 99
    var i = 4
    while i + 1 < len(av):
        var a = String(av[i])
        var v = String(av[i + 1])
        if a == "--mode":
            mode = v
        elif a == "--n-ctx":
            n_ctx = Int(atol(v))
        elif a == "--temp":
            temp = atof(v)
        elif a == "--top-k":
            top_k = Int(atol(v))
        elif a == "--top-p":
            top_p = atof(v)
        elif a == "--top-n":
            top_n = Int(atol(v))
        elif a == "--tau":
            tau = atof(v)
        elif a == "--cold":
            cold = Int(atol(v))
        elif a == "--seed":
            sd = Int(atol(v))
        elif a == "--ngl":
            ngl = Int(atol(v))
        elif a == "--max-tokens":
            max_tokens = Int(atol(v))
        else:
            die("unknown argument " + a)
        i += 2
    if mode != "soft" and mode != "discrete":
        die("--mode must be soft or discrete")
    seed(sd)
    var doc = parse_json_bytes(List[UInt8](rd(req).as_bytes()))
    if max_tokens < 0:
        var mi = doc.get_field(doc.root, "max_tokens")
        max_tokens = Int(doc.get(mi).n) if mi >= 0 else 4096
    var prompt = render(doc)
    var lib = OwnedDLHandle(".work/soft-think/libsoft_shim.so")
    var mp = model
    var rc = Int(lib.get_function[Int32]("st_open")(mp.as_c_string_span().ptr(), Int32(n_ctx), Int32(ngl)))
    if rc != 0:
        die("st_open failed (" + String(rc) + ") for " + model)
    var nv = Int(lib.get_function[Int32]("st_n_vocab")())
    var ne = Int(lib.get_function[Int32]("st_n_embd")())
    var ptoks = tokenize(lib, prompt)
    var end_ids = tokenize(lib, "</think>")
    if len(end_ids) != 1:
        die("</think> is not one token (" + String(len(end_ids)) + ")")
    var think_end = end_ids[0]
    var tb = List[Int32](length=len(ptoks) + 1, fill=0)
    for k in range(len(ptoks)):
        tb[k] = Int32(ptoks[k])
    if Int(lib.get_function[Int32]("st_eval_tokens")(tb.unsafe_ptr(), Int32(len(ptoks)))) != 0:
        die("prompt decode failed (" + String(len(ptoks)) + " tokens)")
    var ebuf = List[Float32](length=ne, fill=0)
    var row = List[Float32](length=ne, fill=0)
    var pbuf = List[UInt8](length=256, fill=0)
    var trace = String()
    var answer = String()
    var thinking = True
    var cold_stopped = False
    var low = 0
    var steps = 0
    var think_steps = 0
    var hsum = 0.0
    var mix_sum = 0
    var finish = String("length")
    while steps < max_tokens:
        var lp = Int(lib.get_function[Int]("st_logits")())
        steps += 1
        if thinking:
            var pk = filter_step(lp, nv, temp, top_k, top_p, top_n if mode == "soft" else 0)
            hsum += pk.entropy
            think_steps += 1
            low = low + 1 if pk.entropy < tau else 0
            var end_now = pk.top == think_end or (mode == "soft" and low >= cold)
            if mode == "discrete":
                var t = sample(pk)
                end_now = t == think_end
                if not end_now:
                    trace += piece(lib, t, Int(pbuf.unsafe_ptr()))
                    tb[0] = Int32(t)
                    if Int(lib.get_function[Int32]("st_eval_tokens")(tb.unsafe_ptr(), Int32(1))) != 0:
                        die("decode failed")
                    continue
            if end_now:
                cold_stopped = pk.top != think_end
                thinking = False
                tb[0] = Int32(think_end)
                if Int(lib.get_function[Int32]("st_eval_tokens")(tb.unsafe_ptr(), Int32(1))) != 0:
                    die("decode failed")
                continue
            trace += piece(lib, pk.top, Int(pbuf.unsafe_ptr()))
            mix_sum += len(pk.ids)
            for d in range(ne):
                ebuf[d] = 0
            for j in range(len(pk.ids)):
                _ = lib.get_function[Int32]("st_embed_row")(Int32(pk.ids[j]), row.unsafe_ptr())
                var wj = Float32(pk.w[j])
                for d in range(ne):
                    ebuf[d] += wj * row[d]
            if Int(lib.get_function[Int32]("st_eval_embd")(ebuf.unsafe_ptr())) != 0:
                die("embedding decode failed")
            continue
        var pa = filter_step(lp, nv, temp, top_k, top_p, 0)
        var t = sample(pa)
        if Int(lib.get_function[Int32]("st_is_eog")(Int32(t))) != 0:
            finish = "stop"
            break
        answer += piece(lib, t, Int(pbuf.unsafe_ptr()))
        tb[0] = Int32(t)
        if Int(lib.get_function[Int32]("st_eval_tokens")(tb.unsafe_ptr(), Int32(1))) != 0:
            die("decode failed")
    var mean_h = hsum / Float64(max(think_steps, 1))
    var mean_mix = Float64(mix_sum) / Float64(max(think_steps, 1))
    wr(rep, '{"choices":[{"message":{"content":' + json_str(answer) + ',"reasoning_content":' + json_str(trace)
       + '},"finish_reason":"' + finish + '"}],"usage":{"completion_tokens":' + String(steps) + ',"prompt_tokens":'
       + String(len(ptoks)) + '},"soft":{"mode":"' + mode + '","think_steps":' + String(think_steps)
       + ',"thinking_ended":' + ("false" if thinking else "true") + ',"cold_stop":' + ("true" if cold_stopped else "false")
       + ',"mean_entropy":' + String(mean_h) + ',"mean_mix":' + String(mean_mix) + ',"top_n":' + String(top_n)
       + ',"tau":' + String(tau) + ',"cold":' + String(cold) + ',"seed":' + String(sd) + "}}")
    print("soft-think:", mode, "prompt", len(ptoks), "think", think_steps, "answer", steps - think_steps,
          "finish", finish, "cold_stop", cold_stopped, "mean_H", mean_h, "mean_mix", mean_mix)
