#!/usr/bin/env python3
"""P4b oracle bisect: exact-ref.py (batched float64) against tools/model-ref.py (the decode reference that matches the engine), per layer and per
position, on the SAME weights: the first LAYERS layers of the GGUF dequantized once and written as an f32 pack that model-ref reads, so only the
layer maths differs between the two sides. model-ref runs token by token (state carried), exact-ref in one batched pass. Prints, for every layer, the
worst relative error (max |a-b| / max |b| over the position's hidden vector) across positions and the first position past 5%; the first layer
over 5% names the bug. model-ref rounds activations to bf16 (`rne`), so noise of a few 1e-3 is expected.

  oracle-bisect.py --gguf G --ids 760,... | --unit UNIT.ref.json --n 8 --layers 4 --pack OUTDIR
CPU only, through `gpu-wait run --room cpu`. A diagnostic, never a verdict."""
import argparse, importlib.util, json, os, sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--pack", required=True)
    a = ap.parse_args()
    ids = json.load(open(a.unit))["prompt_ids"][: a.n]
    ex = load(f"{HERE}/exact-ref.py", "exact_ref")
    M = ex.Model(a.gguf)
    os.makedirs(a.pack, exist_ok=True)
    names = [f"blk.{i}.{s}" for i in range(a.layers) for s in
             ("attn_norm.weight", "post_attention_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")]
    for i in range(a.layers):
        if (i + 1) % 4 == 0:
            names += [f"blk.{i}.{s}" for s in ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight")]
        else:
            names += [f"blk.{i}.{s}" for s in ("attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight", "ssm_out.weight",
                                               "ssm_conv1d.weight", "ssm_norm.weight", "ssm_dt.bias", "ssm_a")]
    idx, off = [], 0
    with open(f"{a.pack}/pack.bin", "wb") as f:
        for nm in names:
            w = M.w(nm).astype(np.float32)
            if w.ndim == 2 and not nm.endswith("ssm_conv1d.weight"):
                w = w.T  # GGUF numpy (out, in); model-ref reads (in, out)
            w = np.ascontiguousarray(w)
            f.write(w.tobytes())
            idx.append(f"{nm} f32 {off} {w.size}")
            off += w.size * 4
    open(f"{a.pack}/index.txt", "w").write("\n".join(idx) + "\n")
    os.environ["BARO_PACK"] = a.pack
    mr = load(f"{HERE}/../../tools/model-ref.py", "model_ref")
    x0 = M.w("token_embd.weight", rows=np.asarray(ids))
    # exact side: batched, capture x after every layer (same body as exact-ref.forward)
    X, E = x0.copy(), []
    for i in range(a.layers):
        X = ex.attn_layer(M, i, X) if (i + 1) % 4 == 0 else ex.ssm_layer(M, i, X)
        cur = ex.rms(X, M.w(f"blk.{i}.post_attention_norm.weight"))
        X = X + (ex.silu(cur @ M.w(f"blk.{i}.ffn_gate.weight").T) * (cur @ M.w(f"blk.{i}.ffn_up.weight").T)) @ M.w(f"blk.{i}.ffn_down.weight").T
        E.append(X.copy())
    # reference side: token by token
    R = [[] for _ in range(a.layers)]
    for pos in range(len(ids)):
        x = x0[pos].astype(np.float32).copy()
        for i in range(a.layers):
            x = mr.attn_layer(i, x, pos) if (i + 1) % 4 == 0 else mr.ssm_layer(i, x, pos)
            c = mr.rne(mr.amar_rmsnorm(x, mr.T(f"blk.{i}.post_attention_norm.weight", (mr.H,))))
            act = mr.rne(mr.silu(c @ mr.T(f"blk.{i}.ffn_gate.weight", (mr.H, mr.FFN))) * (c @ mr.T(f"blk.{i}.ffn_up.weight", (mr.H, mr.FFN))))
            x = x + act @ mr.T(f"blk.{i}.ffn_down.weight", (mr.FFN, mr.H))
            R[i].append(x.astype(np.float64))
    bad = None
    for i in range(a.layers):
        err = [float(np.max(np.abs(E[i][p] - R[i][p])) / (np.max(np.abs(R[i][p])) + 1e-12)) for p in range(len(ids))]
        first = next((p for p, e in enumerate(err) if e > 0.05), None)
        kind = "attn" if (i + 1) % 4 == 0 else "ssm"
        print(f"layer {i} ({kind}): worst rel err {max(err):.2e}, per position " + " ".join(f"{e:.1e}" for e in err) + (f"  FIRST>5% at position {first}" if first is not None else ""), flush=True)
        if first is not None and bad is None:
            bad = i
    print(f"oracle-bisect: first layer over 5% = {bad}" if bad is not None else f"oracle-bisect: all {a.layers} layers within 5% of model-ref on {len(ids)} positions")


if __name__ == "__main__":
    main()
