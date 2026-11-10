#!/usr/bin/env python3
"""Per-tensor oracle for any GGUF llama.cpp can run, harvested from
llama-eval-callback.

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

  2. tools/llama-oracle.py evalcb.log [out.json]

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
import json
import re
import sys
from pathlib import Path

HEAD = re.compile(r"common_debug_cb_eval:\s+(.+?) = ")
LAYERED = re.compile(r"^(.+?)-(\d+)$")


def parse(path):
    """{tensor_name: {layer: sum}} plus {full_name: [first printed values]}."""
    sums, vals = {}, {}
    cur, rows = None, []
    for line in Path(path).read_text(errors="replace").splitlines():
        m = HEAD.match(line)
        if m:
            cur, rows = m.group(1).strip(), []
            continue
        if cur is None:
            continue
        s = line.strip()
        if s.startswith("[") and "," in s:
            rows.append(s)
        elif s.startswith("sum = "):
            total = float(s.split("=", 1)[1])
            lm = LAYERED.match(cur)
            name, layer = (lm.group(1), int(lm.group(2))) if lm else (cur, -1)
            # A name repeats across layers (norm-0 appears several times in one
            # layer); keep the first occurrence, which is the one the graph
            # reaches first.
            sums.setdefault(name, {}).setdefault(layer, total)
            vals.setdefault(cur, rows)
            cur, rows = None, []
    return sums, vals


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    sums, vals = parse(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else None
    interesting = [
        "attn_residual", "attn_post_norm", "ffn_moe_out", "ffn_out", "l_out",
        "ffn_moe_weights_sum", "ffn_moe_weights_sum_clamped",
    ]
    for name in interesting:
        per_layer = sums.get(name)
        if not per_layer:
            continue
        layers = sorted(per_layer)
        print(f"--- {name}: {len(layers)} layers")
        for layer in layers[:4] + (layers[-2:] if len(layers) > 6 else []):
            print(f"      L{layer:02d} sum={per_layer[layer]:>16.6f}")
    if out:
        Path(out).write_text(json.dumps(sums, indent=0, sort_keys=True))
        print(f"wrote {out}: {len(sums)} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
