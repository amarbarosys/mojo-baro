"""Per-tensor oracle for any GGUF llama.cpp can run, harvested from
llama-eval-callback. Mojo port of tools/llama-oracle.py, which stays as the
oracle for the port (bench/PROTOCOL-RULES.md, repo default is Mojo).

llama.cpp's eval-callback example prints every tensor in the compute graph
with its shape, the first and last three values of each row, and the sum of
the whole tensor. That is a complete per-layer reference for a model we
otherwise have no oracle for: reimplementing a 40-layer hybrid SSM+MoE stack
in numpy is ~300 lines of code that has to be right before it can find a bug
in someone else's code.

Usage:
  1. Produce the log (one token, zero states, is the cleanest arm -- with a
     multi-token prompt every sum is over all positions at once and cannot be
     compared against a single-token dump from our engine):

       llama-eval-callback -m MODEL.gguf -ngl 99 -c 512 -n 1 --temp 0 \
           -f one-token-prompt.txt > evalcb.log 2>&1

  2. llama_oracle evalcb.log [out.json]

Prints the per-layer table and writes {tensor: {layer: sum}} as JSON.

Names worth knowing for the qwen35moe stack (all suffixed -<layer>):
  attn_residual      x after the attention/SSM sub-block, before the FFN norm
  attn_post_norm     the value the MoE block consumes
  ffn_moe_logits/probs/topk/weights/weights_norm    routing, in order
  ffn_moe_gate/up/swiglu/down/weighted/out          the routed expert chain
  ffn_out            routed + gated shared expert
  l_out              the layer's output, = attn_residual + ffn_out
  conv_state_last/update, alpha, beta, beta_sigmoid, a_softplus,
  Qcur_normed, Kcur_normed, attn_gated                 SSM internals

Caveat that makes or breaks a comparison: a sum cancels. Layer 0 of
RegesCore matched llama to 1.2% on the sum of the post-SSM residual while
individual components were 6%, 12% and 78% out. Compare the printed
elements, not only the sums.
"""
from std.collections import Dict
from std.os import abort
from std.sys import argv, exit

comptime USAGE = "usage: llama_oracle <evalcb.log> [out.json]  (see the module docstring)"
comptime HEAD = "common_debug_cb_eval:"
comptime SUMP = "sum = "


def lstrip_ws(s: String) raises -> String:
    var b = s.as_bytes()
    var i = 0
    while i < len(b) and (b[i] == 32 or b[i] == 9):
        i += 1
    return String(from_utf8=b[i:])


def strip_ws(s: String) raises -> String:
    var b = s.as_bytes()
    var i = 0
    while i < len(b) and (b[i] == 32 or b[i] == 9 or b[i] == 13):
        i += 1
    var j = len(b)
    while j > i and (b[j - 1] == 32 or b[j - 1] == 9 or b[j - 1] == 13):
        j -= 1
    return String(from_utf8=b[i:j])


def head_name(line: String) raises -> String:
    """The tensor name in a `common_debug_cb_eval:   NAME = (f32) ...` line.

    Python used a non-greedy `(.+?) = `, i.e. the FIRST " = " after the
    prefix. `find` gives the same first occurrence; a later " = " inside the
    op arguments must not win.
    """
    if not line.startswith(HEAD):
        return String("")
    var rest = lstrip_ws(String(from_utf8=line.as_bytes()[HEAD.byte_length() :]))
    var eq = rest.find(" = ")
    if eq < 0:
        return String("")
    return strip_ws(String(from_utf8=rest.as_bytes()[:eq]))


def split_layer(name: String, mut base: String, mut layer: Int) raises -> Bool:
    """`foo-12` -> ("foo", 12). Trailing digits after the LAST '-', and the
    Python pattern was non-greedy on the left, so `a-b-3` splits at the last
    hyphen. Returns False when there is no `-<digits>` suffix."""
    var b = name.as_bytes()
    var i = len(b)
    while i > 0 and b[i - 1] >= 48 and b[i - 1] <= 57:
        i -= 1
    if i == len(b) or i == 0 or b[i - 1] != 45:
        return False
    var v = 0
    for k in range(i, len(b)):
        v = v * 10 + Int(b[k] - 48)
    base = String(from_utf8=b[: i - 1])
    layer = v
    return True


def fmt6(v: Float64) -> String:
    """Fixed 6 decimals, matching Python's f"{v:.6f}"."""
    var neg = v < 0
    var x = -v if neg else v
    var scaled = x * 1e6 + 0.5
    var whole = Int(scaled)
    var ip = whole // 1000000
    var fp = whole % 1000000
    var frac = String(fp)
    while frac.byte_length() < 6:
        frac = String("0") + frac
    var out = String(ip) + "." + frac
    return ("-" + out) if neg else out


def pad_left(s: String, width: Int) -> String:
    var out = s
    while out.byte_length() < width:
        out = String(" ") + out
    return out


def main() raises:
    var args = argv()
    if len(args) < 2:
        print(USAGE)
        exit(1)
    var path = String(args[1])

    # name -> (layer -> sum), first occurrence wins, exactly as the Python's
    # double setdefault does.
    var sums = Dict[String, Dict[Int, Float64]]()
    var order = List[String]()
    var n_tensors = 0

    var cur = String("")
    var have = False
    with open(path, "r") as f:
        for line in f.read().splitlines():
            var hn = head_name(String(line))
            if hn != "":
                cur = hn
                have = True
                continue
            if not have:
                continue
            var s = strip_ws(String(line))
            if s.startswith(SUMP):
                var total = Float64(
                    strip_ws(String(from_utf8=s.as_bytes()[SUMP.byte_length() :]))
                )
                var base = String("")
                var layer = -1
                if not split_layer(cur, base, layer):
                    base = cur
                    layer = -1
                if base not in sums:
                    sums[base] = Dict[Int, Float64]()
                    order.append(base)
                if layer not in sums[base]:
                    sums[base][layer] = total
                    n_tensors += 1
                have = False
                cur = String("")

    var interesting = [
        String("attn_residual"), String("attn_post_norm"), String("ffn_moe_out"),
        String("ffn_out"), String("l_out"), String("ffn_moe_weights_sum"),
        String("ffn_moe_weights_sum_clamped"),
    ]
    for name in interesting:
        if name not in sums:
            continue
        ref per = sums[name]
        var layers = List[Int]()
        for e in per.items():
            layers.append(e.key)
        sort(layers)
        print("---", name + ":", len(layers), "layers")
        var show = List[Int]()
        for i in range(min(4, len(layers))):
            show.append(layers[i])
        if len(layers) > 6:
            show.append(layers[len(layers) - 2])
            show.append(layers[len(layers) - 1])
        for l in show:
            var tag = String(l) if l >= 10 else ("0" + String(l))
            print("      L" + tag, "sum=" + pad_left(fmt6(per[l]), 16))

    if len(args) > 2:
        var out = String(args[2])
        # {tensor: {layer: sum}}, keys sorted, indent=0 -- json.dumps with
        # indent=0 puts every element on its own line with no leading spaces.
        var keys = List[String]()
        for e in sums.items():
            keys.append(e.key)
        sort(keys)
        var buf = String("{\n")
        for ki in range(len(keys)):
            var k = keys[ki]
            ref per = sums[k]
            var layers = List[Int]()
            for e in per.items():
                layers.append(e.key)
            sort(layers)
            buf += '"' + k + '": {\n'
            for li in range(len(layers)):
                var l = layers[li]
                buf += '"' + String(l) + '": ' + String(per[l])
                buf += ",\n" if li + 1 < len(layers) else "\n"
            buf += "}"
            buf += ",\n" if ki + 1 < len(keys) else "\n"
        buf += "}"
        with open(out, "w") as g:
            g.write(buf)
        print("wrote " + out + ":", len(keys), "tensors")
