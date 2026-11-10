#!/usr/bin/env python3
"""P5b: hand-written LoRA (no peft) on the 8 ffn_down Linear layers of
blk.24..31. Blocks 0..23 run under no_grad (frozen, no trainable params
there); a forward pre-hook on layer 24 re-enables grad so backprop reaches
exactly the LoRA A/B matrices in blocks 24..31. Merge: W + (alpha/r) B A,
computed in BF16, written as an NPZ keyed by GGUF tensor name for
tools/gguf-writeback.py."""
import os
import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

TARGET_LAYERS = list(range(24, 32))  # blk.24..31, not blk.32 (MTP head)


def load_corpus(manifest_path, gsm8k_path, limit):
    manifest = json.load(open(manifest_path))
    ids = sorted(row["id"] for row in manifest["docs"])[:limit] if limit else sorted(row["id"] for row in manifest["docs"])
    lines = open(gsm8k_path, encoding="utf-8").read().split("\n")
    docs = []
    for i in ids:
        rec = json.loads(lines[i])
        docs.append(f"Question: {rec['question']}\nAnswer: {rec['answer']}")
    return docs


def attach_lora(lm, rank, alpha, dtype, device):
    adapters = {}
    for i in TARGET_LAYERS:
        down = lm.layers[i].mlp.down_proj
        A = torch.nn.Parameter(torch.zeros(rank, down.in_features, dtype=dtype, device=device))
        B = torch.nn.Parameter(torch.zeros(down.out_features, rank, dtype=dtype, device=device))
        torch.nn.init.kaiming_uniform_(A, a=5**0.5)  # standard LoRA init; B stays zero (no-op at start)
        base_forward = down.forward

        def make_forward(base_forward, A, B):
            return lambda x: base_forward(x) + (alpha / rank) * (x @ A.t()) @ B.t()

        down.forward = make_forward(base_forward, A, B)
        adapters[i] = (A, B, down)
    return adapters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--corpus-manifest", required=True)
    ap.add_argument("--gsm8k", default=os.path.expanduser("~/Models/datasets/gsm8k/main/train.jsonl"))
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--gpu-hour-cap", type=float, default=1.0)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--out", required=True, help="merged-weight NPZ for gguf-writeback.py")
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(args.hf_dir)
    model = Qwen3_5ForCausalLM.from_pretrained(args.hf_dir, dtype=torch.bfloat16)
    model.to("cuda")
    for p in model.parameters():
        p.requires_grad_(False)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
    lm = model.model
    adapters = attach_lora(lm, args.rank, args.alpha, torch.bfloat16, "cuda")

    def wake(module, inputs):
        torch.set_grad_enabled(True)

    lm.layers[TARGET_LAYERS[0]].register_forward_pre_hook(wake)

    params = [p for a in adapters.values() for p in (a[0], a[1])]
    opt = torch.optim.AdamW(params, lr=args.lr)

    docs = load_corpus(args.corpus_manifest, args.gsm8k, args.limit_docs or args.accum * args.n_steps)
    n_needed = args.accum * args.n_steps
    if len(docs) < n_needed:
        raise SystemExit(f"need {n_needed} docs, corpus has {len(docs)}")

    step_losses = []
    di = 0
    for step in range(args.n_steps):
        if time.time() - t_start > args.gpu_hour_cap * 3600:
            print(f"gpu-hour-cap reached at step {step}, stopping early")
            break
        opt.zero_grad(set_to_none=True)
        window = []
        for _ in range(args.accum):
            ids = tok(docs[di], return_tensors="pt", truncation=True, max_length=args.max_seq).input_ids.to("cuda")
            di += 1
            with torch.no_grad():
                out = model(ids, labels=ids)
            (out.loss / args.accum).backward()
            window.append(float(out.loss.item()))
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step_losses.append(sum(window) / len(window))
        print(f"step {step + 1}/{args.n_steps} loss {step_losses[-1]:.4f}")

    merged = {}
    for i, (A, B, down) in adapters.items():
        delta = (args.alpha / args.rank) * (B.float() @ A.float())
        w = (down.weight.detach().float() + delta).to(torch.bfloat16)
        # NOT transposed: down_proj.weight's natural torch shape (out, in) is
        # already gguf-writeback.py's expected numpy-natural array shape for
        # this tensor (verified byte-for-byte against the source GGUF); a
        # .t() here writes a scrambled, transposed matrix into the file.
        merged[f"blk.{i}.ffn_down.weight"] = w.contiguous().view(torch.uint16).cpu().numpy()
    np.savez(args.out, **merged)

    report = {
        "target_layers": TARGET_LAYERS, "rank": args.rank, "alpha": args.alpha,
        "docs_used": di, "n_steps": len(step_losses), "step_losses": step_losses,
        "first_loss": step_losses[0] if step_losses else None,
        "last_loss": step_losses[-1] if step_losses else None,
        "wall_s": time.time() - t_start, "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
        "out": args.out,
    }
    json.dump(report, open(args.report, "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
