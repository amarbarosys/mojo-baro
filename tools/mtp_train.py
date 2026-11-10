#!/usr/bin/env python
"""P5a: the draft-head self-distillation training smoke
(bench/draft-head-protocol.md). Fine-tunes MTPHead (tools/mtp_head.py) on
dense real-text next-token supervision from bench/draft_dump.mojo --mode
dump --env A4_TEXT_MODE=gsm8k (GSM8K train.jsonl question+answer text,
genuinely held out from both bench/data/e8_tasks.json and
bench/mtp-prompts/*.tokens).

CORRECTED RECIPE (this session's second pass, after the first smoke's
lr=2e-4/no-accumulation/no-clip/no-warmup recipe damaged the head: 5-prompt
accepted/drafted went 67.78% -> 3.06%, exchange/lane-A4-report.md). the maintainer's
frozen correction: lr 2e-5, 16-document gradient accumulation per optimizer
step (E13's own tools/latent-os/e13_train.py setting, --accum 16), grad-norm
clip 1.0, 10% linear warmup, and a VOID check on the loss trend BEFORE the
real-engine gate runs at all: the mean per-step loss must be lower at the
last optimizer step than at the first, or the run is reported VOID and no
write-back/gate step should follow it.

One optimizer step per `--accum` DOCUMENTS, not per shuffled position:
blk.32's own attention accumulates a KV cache across a document's positions
(kc32_d / vc32_d in serve/window.mojo), so a micro-batch that mixed shuffled
positions from different documents would feed the wrong causal context.
Each document's whole position sequence goes through MTPHead in one causal
forward (exactly serve/window.mojo's own multi-row verify-window shape);
gradients from `--accum` documents are summed (each document's own loss
divided by `--accum` before backward) before one clipped, warmed-up
optimizer step.

Run only via gpu-wait:
  gpu-wait run --priority 10 --preemptible --vram 22 --timeout 900 -- \\
    PYTHONPATH=~/llama.cpp/gguf-py ~/AMDHQ/.venv/bin/python tools/mtp_train.py ...
"""
import os
import argparse
import json
import math
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "tools")
from mtp_head import MTPHead, build_causal_mask, load_trunk, read_v2_dump  # noqa: E402


def flatten_document(doc):
    """From read_dump's per-document {tokens, h[n-1,H]}: row h[i] pairs
    with INPUT token tokens[i+1] and LABEL tokens[i+2], valid i in
    0..n-3 (bench/draft_dump.mojo's own docstring)."""
    n = len(doc["tokens"])
    n_pairs = max(0, n - 2)
    if n_pairs <= 0:
        return None
    h = doc["h"][:n_pairs]
    input_ids = torch.tensor(doc["tokens"][1:1 + n_pairs], dtype=torch.long)
    labels = torch.tensor(doc["tokens"][2:2 + n_pairs], dtype=torch.long)
    return h, input_ids, labels


def flatten_v2_document(doc):
    """Make aligned CPU tensors from validated P5a records."""
    records = doc["records"]
    if not records:
        return None
    h = torch.stack([r["h"] for r in records])
    input_ids = torch.tensor([r["input_token"] for r in records], dtype=torch.long)
    # the maintainer's 2026-09-17 ruling: acceptance is agreement with the trunk's
    # greedy pick, so CE uses the recorded target argmax, not tokens[pos+1].
    labels = torch.tensor([r["target_argmax"] for r in records], dtype=torch.long)
    top8_ids = torch.stack([r["top8_ids"] for r in records])
    top8_probs = torch.stack([r["top8_probs"] for r in records])
    return h, input_ids, labels, top8_ids, top8_probs


def forward_loss(head, embed, lm_head, rotary, h, input_ids, labels, top8_ids, top8_probs, device):
    T = h.shape[0]
    h = h.to(device)
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    top8_ids = top8_ids.to(device)
    top8_probs = top8_probs.to(device)
    tok_embed_row = embed(input_ids)
    pos_ids = torch.arange(1, 1 + T, device=device).view(1, 1, -1).expand(4, 1, -1)
    mask = build_causal_mask(T, torch.float32, device)
    out = head(tok_embed_row, h, rotary, pos_ids, mask)
    logits = lm_head(out.to(lm_head.weight.dtype)).float()
    ce = F.cross_entropy(logits, labels)
    restricted = logits.gather(dim=-1, index=top8_ids)
    log_p8 = F.log_softmax(restricted, dim=-1)
    kl = F.kl_div(log_p8, top8_probs, reduction="batchmean")
    return ce + kl, ce.detach(), kl.detach(), T


def lr_multiplier(step_1indexed, n_steps, warmup_steps):
    """Linear warmup over the first warmup_steps optimizer steps: step i
    (1-indexed) gets i/(warmup_steps+1) of the target lr, reaching 1.0 once
    i > warmup_steps. 10% of n_steps, rounded up, at least 1 step."""
    if step_1indexed > warmup_steps:
        return 1.0
    return step_1indexed / (warmup_steps + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="bench/draft_dump.mojo P5a v2 output")
    ap.add_argument("--extracted", required=True, help="tools/mtp_head.py --mode extract output")
    ap.add_argument("--hf-dir", default=os.path.expanduser("~/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/hf"))
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--accum", type=int, default=16, help="documents per optimizer step (E13's own setting)")
    ap.add_argument("--n-steps", type=int, default=8, help="total optimizer steps")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", required=True, help="trained MTPHead state_dict checkpoint")
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    t_start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trunk(args.hf_dir)
    model.to(device)
    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    rotary = model.model.rotary_emb if hasattr(model, "model") else model.language_model.rotary_emb
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config

    head = MTPHead(cfg).to(device)
    sd = torch.load(args.extracted, map_location=device)
    head.load_blk32(sd)
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)

    docs = read_v2_dump(args.dump)
    flat_docs = [flatten_v2_document(d) for d in docs]
    flat_docs = [d for d in flat_docs if d is not None]
    n_needed = args.accum * args.n_steps
    if len(flat_docs) < n_needed:
        raise SystemExit(f"need {n_needed} usable documents ({args.accum}x{args.n_steps}), dump has {len(flat_docs)}")
    torch.manual_seed(args.seed)
    perm = torch.randperm(len(flat_docs)).tolist()[:n_needed]

    warmup_steps = max(1, math.ceil(args.warmup_ratio * args.n_steps))
    step_losses = []
    step_ce = []
    step_kl = []
    doc_losses = []
    n_docs_used = 0
    n_pairs_used = 0
    peak_vram = 0
    grad_norms = []

    for step in range(1, args.n_steps + 1):
        mult = lr_multiplier(step, args.n_steps, warmup_steps)
        for g in opt.param_groups:
            g["lr"] = args.lr * mult
        opt.zero_grad(set_to_none=True)
        window_losses = []
        window_ce = []
        window_kl = []
        for i in range(args.accum):
            idx = perm[(step - 1) * args.accum + i]
            h, input_ids, labels, top8_ids, top8_probs = flat_docs[idx]
            loss, ce, kl, t = forward_loss(
                head, embed, lm_head, rotary, h, input_ids, labels,
                top8_ids, top8_probs, device
            )
            (loss / args.accum).backward()
            window_losses.append(loss.item())
            window_ce.append(ce.item())
            window_kl.append(kl.item())
            n_docs_used += 1
            n_pairs_used += t
            if device == "cuda":
                peak_vram = max(peak_vram, torch.cuda.max_memory_allocated())
        gn = torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=args.grad_clip)
        grad_norms.append(float(gn))
        opt.step()
        doc_losses.extend(window_losses)
        step_losses.append(sum(window_losses) / len(window_losses))
        step_ce.append(sum(window_ce) / len(window_ce))
        step_kl.append(sum(window_kl) / len(window_kl))

    is_void = not (step_losses[-1] < step_losses[0])
    torch.save(head.state_dict(), args.out)
    report = {
        "recipe": "corrected: lr 2e-5, accum 16, grad-clip 1.0, 10pct warmup",
        "lr": args.lr, "accum": args.accum, "n_steps": args.n_steps,
        "grad_clip": args.grad_clip, "warmup_steps": warmup_steps,
        "n_docs_used": n_docs_used, "n_pairs_used": n_pairs_used,
        "step_losses": step_losses,
        "step_ce": step_ce,
        "step_kl": step_kl,
        "grad_norms": grad_norms,
        "first_step_loss": step_losses[0],
        "last_step_loss": step_losses[-1],
        "loss_went_down": step_losses[-1] < step_losses[0],
        "void": is_void,
        "min_doc_loss": min(doc_losses), "max_doc_loss": max(doc_losses),
        "wall_s": time.time() - t_start,
        "peak_vram_gb": peak_vram / 2**30 if peak_vram else None,
        "checkpoint": args.out,
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if is_void:
        raise SystemExit("VOID: loss did not go down across steps; do not proceed to the gate")


if __name__ == "__main__":
    main()
