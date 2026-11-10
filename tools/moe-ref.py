#!/usr/bin/env python3
"""Numpy reference for one decode token through the qwen35moe sparse-MoE block.

Implements transformers `Qwen3_5MoeSparseMoeBlock` exactly:
  router_logits = x @ Wr.T                       Wr  [N_EXP, H], f32 in the GGUF
  probs         = softmax(router_logits, f32)    softmax over ALL experts first
  w, idx        = topk(probs, TOPK)              then take top-8
  w            /= w.sum()                        then renormalise the 8
  expert e      : silu(x @ Wg[e].T) * (x @ Wu[e].T) @ Wd[e].T, scaled by w
  shared        : silu(x @ Wgs.T) * (x @ Wus.T) @ Wds.T, scaled by sigmoid(x @ wsg)
  out           = sum(experts) + gated shared

Softmax-before-topk is the part worth getting right: topk-then-softmax is the
more common convention and gives different weights for the same logits.

Two weight sources. With --gguf MODEL.gguf it reads the real blk.<L> MoE
tensors and dequantises them (Q4_K experts, Q8_0 shared, F32 routers) -- that
needs the repo venv, which has gguf-py: ./.venv/bin/python3. Without it, the
weights are fixed-seed; the math is the same either way, and the synthetic path
keeps the test runnable with no 20 GB file present.

Weights are rounded to bf16 so the GPU consumes exactly what numpy did.

Writes inputs + expected outputs for kernels/test_moe_block.mojo.
"""
import argparse
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent.parent / ".work/gguf"

H = 2048          # qwen35moe hidden size
N_EXP = 256       # num_experts
TOPK = 8          # num_experts_per_tok
E_FFN = 512       # moe_intermediate_size
SH_FFN = 512      # shared_expert_intermediate_size
SEED = 20260908


def to_bf16(x):
    """Round-to-nearest-even f32 -> bf16, returned as f32 (the pack's rounding)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32)


def bf16_store(x):
    """The same value as the upper 16 bits, for writing a bf16 fixture."""
    return (np.ascontiguousarray(to_bf16(x), dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def from_gguf(path, layer):
    """Real blk.<layer> MoE tensors, dequantised. Needs the repo venv (gguf-py).

    gguf-py hands back C-order arrays whose shape is the GGUF dim list reversed,
    so ffn_gate_exps [2048, 512, 256] arrives as (expert, out, in) -- already the
    layout this reference and the kernel both index.
    """
    import gguf

    reader = gguf.GGUFReader(path)
    want = {}
    for t in reader.tensors:
        if t.name.startswith("blk.%d.ffn_" % layer):
            want[t.name.split(".", 2)[2]] = t

    def get(name):
        t = want[name]
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            return np.asarray(t.data, dtype=np.float32)
        return gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)

    return (
        get("ffn_gate_inp.weight").reshape(N_EXP, H),
        get("ffn_gate_inp_shexp.weight").reshape(H),
        to_bf16(get("ffn_gate_exps.weight").reshape(N_EXP, E_FFN, H)),
        to_bf16(get("ffn_up_exps.weight").reshape(N_EXP, E_FFN, H)),
        to_bf16(get("ffn_down_exps.weight").reshape(N_EXP, H, E_FFN)),
        to_bf16(get("ffn_gate_shexp.weight").reshape(SH_FFN, H)),
        to_bf16(get("ffn_up_shexp.weight").reshape(SH_FFN, H)),
        to_bf16(get("ffn_down_shexp.weight").reshape(H, SH_FFN)),
    )


def write_q4k_artifacts(path, layer, x, wg, wu, wd):
    """Write raw expert blocks plus a small decoder vector and offset semantics."""
    import gguf

    reader = gguf.GGUFReader(path)
    prefix = f"blk.{layer}."
    wanted = {
        "ffn_gate_exps.weight": ("gate", E_FFN, H),
        "ffn_up_exps.weight": ("up", E_FFN, H),
        "ffn_down_exps.weight": ("down", H, E_FFN),
    }
    entries = []
    raw_first = None
    for tensor in reader.tensors:
        short = tensor.name[len(prefix):] if tensor.name.startswith(prefix) else ""
        if short not in wanted:
            continue
        label, rows, k = wanted[short]
        raw = np.asarray(tensor.data, dtype=np.uint8)
        out = D / f"moe_q4k_{label}.bin"
        out.write_bytes(raw.tobytes())
        row_bytes = k // 256 * 144
        entries.append({
            "tensor": tensor.name,
            "projection": label,
            "expert_count": N_EXP,
            "rows_per_expert": rows,
            "k": k,
            "row_bytes": row_bytes,
            "source_shape": list(raw.shape),
            "source_bytes": int(raw.nbytes),
        })
        if label == "gate":
            raw_first = raw[0, 0].reshape(-1)[:144]

    if len(entries) != 3 or raw_first is None:
        raise RuntimeError("missing q4_k expert tensors")

    vals = wg[0, 0, :256].astype(np.float32)
    vector = {
        "tensor": f"blk.{layer}.ffn_gate_exps.weight",
        "expert": 0,
        "row": 0,
        "k": 256,
        "raw_block": raw_first.tolist(),
        "dequantized": vals.tolist(),
        "dot_input": x[:256].astype(np.float32).tolist(),
        "dot": float(np.dot(x[:256].astype(np.float32), vals)),
    }
    (D / "moe_q4k_vector.json").write_text(json.dumps(vector, indent=2) + "\n")
    vals.astype(np.float32).tofile(D / "moe_q4k_vector.bin")
    pack_index = Path(".work/moe-w1/pack/index.txt")
    pack_lines = {}
    if pack_index.exists():
        for line in pack_index.read_text().splitlines():
            parts = line.split()
            if len(parts) == 4:
                pack_lines[parts[0]] = {
                    "dtype": parts[1], "offset": int(parts[2]), "n_elem": int(parts[3])
                }
    for entry in entries:
        entry["pack"] = pack_lines.get(f"blk.{layer}.{entry['tensor'].split('.', 2)[-1]}")
    (D / "moe_q4k_map.json").write_text(json.dumps({"layer": layer, "tensors": entries}, indent=2) + "\n")


def synthetic(rng):
    return (
        (rng.standard_normal((N_EXP, H)) * 0.02).astype(np.float32),
        (rng.standard_normal(H) * 0.02).astype(np.float32),
        to_bf16(rng.standard_normal((N_EXP, E_FFN, H), dtype=np.float32) * 0.02),
        to_bf16(rng.standard_normal((N_EXP, E_FFN, H), dtype=np.float32) * 0.02),
        to_bf16(rng.standard_normal((N_EXP, H, E_FFN), dtype=np.float32) * 0.02),
        to_bf16(rng.standard_normal((SH_FFN, H), dtype=np.float32) * 0.02),
        to_bf16(rng.standard_normal((SH_FFN, H), dtype=np.float32) * 0.02),
        to_bf16(rng.standard_normal((H, SH_FFN), dtype=np.float32) * 0.02),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    D.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # The hidden state is fixed-seed either way: there is no real activation
    # without running the model, and the block does not care where x came from.
    x = to_bf16(rng.standard_normal(H, dtype=np.float32) * 0.5)

    if args.gguf:
        wr, wsg, wg, wu, wd, wgs, wus, wds = from_gguf(args.gguf, args.layer)
        write_q4k_artifacts(args.gguf, args.layer, x, wg, wu, wd)
    else:
        wr, wsg, wg, wu, wd, wgs, wus, wds = synthetic(rng)

    # --- router ---
    logits = (x @ wr.T).astype(np.float32)
    m = logits.max()
    probs = np.exp(logits - m)
    probs /= probs.sum()
    idx = np.argsort(-probs, kind="stable")[:TOPK].astype(np.int32)
    w = probs[idx]
    w = w / w.sum()

    # --- routed experts ---
    # h is rounded to bf16 before the down projection: the engine stores
    # activations bf16 between the two skinny GEMVs, and ssm-ref.py models the
    # same rounding on `gated`. Without it the reference measures a precision
    # choice rather than whether the kernel implements the block.
    out = np.zeros(H, dtype=np.float32)
    for j in range(TOPK):
        e = int(idx[j])
        h = to_bf16(silu(x @ wg[e].T) * (x @ wu[e].T))
        out += w[j] * (h @ wd[e].T)

    # --- shared expert, sigmoid-gated ---
    hs = to_bf16(silu(x @ wgs.T) * (x @ wus.T))
    shared = (hs @ wds.T) * sigmoid(np.float32(x @ wsg))
    y = out + shared

    x.tofile(D / "moe_x.bin")
    wr.tofile(D / "moe_wr.bin")
    wsg.tofile(D / "moe_wsg.bin")
    bf16_store(wg).tofile(D / "moe_wg.bin")
    bf16_store(wu).tofile(D / "moe_wu.bin")
    bf16_store(wd).tofile(D / "moe_wd.bin")
    bf16_store(wgs).tofile(D / "moe_wgs.bin")
    bf16_store(wus).tofile(D / "moe_wus.bin")
    bf16_store(wds).tofile(D / "moe_wds.bin")
    idx.tofile(D / "moe_idx_ref.bin")
    w.astype(np.float32).tofile(D / "moe_w_ref.bin")
    out.tofile(D / "moe_routed_ref.bin")
    shared.tofile(D / "moe_shared_ref.bin")
    y.tofile(D / "moe_y_ref.bin")

    print(json.dumps({
        "top1_expert": int(idx[0]),
        "top1_weight": float(w[0]),
        "topk_weight_sum": float(w.sum()),
        "routed_mean_abs": float(np.abs(out).mean()),
        "shared_mean_abs": float(np.abs(shared).mean()),
        "y_mean_abs": float(np.abs(y).mean()),
        "source": args.gguf or "synthetic",
        "layer": args.layer,
    }))


if __name__ == "__main__":
    main()
