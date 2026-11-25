"""CPU cost of the Mojo front's text path on one long chat: render, encode, role_boundaries, Detok per token.

  mojo build bench/tok-front-bench.mojo -I . -I serve -I serve/front -o .work/tok-front-bench
  .work/tok-front-bench MODEL.gguf TEXT_FILE [MESSAGES=40] [REPS=11]

TEXT_FILE is cut into MESSAGES alternating user/assistant turns (bench/tok-corpus/whole/long-16k.txt gives a
16K-token prompt). Prints medians in ms; `detok_ns_per_token` is Detok.push over every prompt id.
"""
# ci-checks: needs -I serve/front (the front's `text` module; the bench step builds with -I . -I kernels -I serve -I bench)
from std.sys import argv
from std.time import perf_counter_ns
from json import Json
from text import Text


def med(mut v: List[Float64]) -> Float64:
    sort(v)
    return v[len(v) // 2]


def main() raises:
    var args = argv()
    if len(args) < 3:
        print("usage: tok-front-bench MODEL.gguf TEXT_FILE [MESSAGES] [REPS]")
        return
    var n_msg = atol(args[3]) if len(args) > 3 else 40
    var reps = atol(args[4]) if len(args) > 4 else 11
    var t0 = perf_counter_ns()
    var text = Text(String(args[1]))
    var load_ms = Float64(perf_counter_ns() - t0) / 1e6
    var doc: String
    with open(String(args[2]), "r") as f:
        doc = f.read()
    var b = doc.as_bytes()
    var msgs = Json.array()
    var at = 0
    for k in range(n_msg):
        var end = len(b) if k == n_msg - 1 else (k + 1) * len(b) // n_msg
        while end < len(b) and (Int(b[end]) & 0xC0) == 0x80:
            end += 1
        var m = Json.object()
        m.set("role", Json.of(String("user") if k % 2 == 0 else String("assistant")))
        m.set("content", Json.of(String(doc[byte=at:end])))
        msgs.push(m)
        at = end
    var render = List[Float64]()
    var encode = List[Float64]()
    var bounds = List[Float64]()
    var detok = List[Float64]()
    var n_ids = 0
    var n_bounds = 0
    var out_bytes = 0
    for _ in range(reps):
        var a = perf_counter_ns()
        var prompt = text.apply_chat_template(msgs, Json(), Json())
        var c = perf_counter_ns()
        var ids = text.encode(prompt, True)
        var d = perf_counter_ns()
        var rb = text.role_boundaries(msgs, Json(), Json())
        var e = perf_counter_ns()
        var dt = text.detok()
        out_bytes = 0
        for id in ids:
            out_bytes += dt.push(id).byte_length()
        out_bytes += dt.flush().byte_length()
        var g = perf_counter_ns()
        n_ids = len(ids)
        n_bounds = len(rb)
        render.append(Float64(c - a) / 1e6)
        encode.append(Float64(d - c) / 1e6)
        bounds.append(Float64(e - d) / 1e6)
        detok.append(Float64(g - e) / Float64(max(n_ids, 1)))
    print("messages:", n_msg, "ids:", n_ids, "load_ms:", load_ms, "render_ms:", med(render), "encode_ms:", med(encode),
          "role_boundaries_ms:", med(bounds), "boundaries:", n_bounds, "detok_ns_per_token:", med(detok),
          "detok_bytes:", out_bytes)
