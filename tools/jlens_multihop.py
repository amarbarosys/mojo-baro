"""Jacobian lens (Anthropic jlens) on Qwen2.5-0.5B: does the intermediate hop show up inside the model?

Usage (GPU, under gpu-wait): ~/Projects/AMD/AMDHQ/.venv/bin/python -P tools/jlens_multihop.py OUTDIR [N_FIT=100]
Fits J_l on N_FIT research-paper chunks (128 tokens, seed 0), caches the lens at OUTDIR/lens.pt, then reads the last prompt
position of all 93 items of lens-eval-multihop.json at every layer below the last.
Hit = the intermediate's first token (with or without a leading space) is in the top-5 at any of those layers.
Arms: jacobian lens; logit lens (same readout, no J_l); shuffled control (jacobian lens, item i scored against item i+1's
intermediates: the chance rate). Also: model's own top-1 equals the target's first token.

Predictions, frozen 2026-10-10 before any run: jacobian 30-60%, logit lens 10-30%, shuffled <= 10%; signal when
jacobian - shuffled >= 20 pp. Layer splits and the layer-0 token dump were added after the first run (exploratory, a23fb74). Metric v2 (frozen 2026-10-10 before its run, after 6625ba8): only candidates that are one Qwen token, layers >= 4; predictions: numeric items collapse to near chance, so the all-items rate lands within 5 pp of the word-only rate; word items stay 25-35% jacobian vs <= 8% shuffled. A 0.5B model is far below the paper's sizes, so a weak result is plausible.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "Projects/AiTrainer/imports/jacobian-lens-main"))
import jlens  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

MODEL = Path.home() / "Models/qwen2.5-0.5b-instruct-hf"
CORPUS = Path.home() / "Projects/AiTrainer/experiments/mojo-baro/work/research-paper-corpus/chunks-capped.jsonl"
EVAL = Path.home() / "Projects/AiTrainer/imports/jacobian-lens-main/data/evaluations/lens-eval-multihop.json"
K = 5

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
n_fit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).cuda().eval()
model = jlens.from_hf(hf, tok)
print("device", torch.cuda.get_device_name(0), "layers", model.n_layers, flush=True)

lp = out / "lens.pt"
if lp.exists():
    lens = jlens.JacobianLens.load(str(lp))
else:
    lines = CORPUS.read_text().splitlines()
    random.Random(0).shuffle(lines)
    prompts = [json.loads(x)["text"] for x in lines[:n_fit]]
    lens = jlens.fit(model, prompts=prompts, max_seq_len=128, checkpoint_path=str(out / "ckpt.pt"))
    lens.save(str(lp))
print("lens", lens, flush=True)

items = json.loads(EVAL.read_text())["items"]
ids = lambda words: {t for w in words for t in (tok.encode(" " + w)[:1] + tok.encode(w)[:1])}
layers = [l for l in lens.source_layers if l < model.n_layers - 1]
reads = {}
for it in items:
    jl, ml, _ = lens.apply(model, it["prompt"], layers=layers, positions=[-1])
    ll, _, _ = lens.apply(model, it["prompt"], layers=layers, positions=[-1], use_jacobian=False)
    reads[it["name"]] = (
        {l: set(jl[l][0].topk(K).indices.tolist()) for l in layers},
        {l: set(ll[l][0].topk(K).indices.tolist()) for l in layers},
        int(ml[0].argmax()),
    )
hit = lambda top, want: any(top[l] & want for l in layers)
first_layer = lambda top, want: next((l for l in layers if top[l] & want), None)
jac = log = shuf = tgt = 0
rows = []
for i, it in enumerate(items):
    jt, lt, top1 = reads[it["name"]]
    want = ids(it["intermediates"])
    other = ids(items[(i + 1) % len(items)]["intermediates"]) - want
    h_j, h_l, h_s = hit(jt, want), hit(lt, want), hit(jt, other)
    h_t = top1 in ids([it["target"]])
    jac, log, shuf, tgt = jac + h_j, log + h_l, shuf + h_s, tgt + h_t
    rows.append(f"{it['name']:28s} jac {int(h_j)} (layer {first_layer(jt, want)}) logit {int(h_l)} shuffled {int(h_s)} target {int(h_t)}")
n = len(items)
(out / "items.txt").write_text("\n".join(rows) + "\n")
summary = (f"items {n} layers {layers[0]}-{layers[-1]} top{K} fit_prompts {n_fit}\n"
           f"jacobian {jac}/{n} {100 * jac / n:.1f}%  logit_lens {log}/{n} {100 * log / n:.1f}%  "
           f"shuffled {shuf}/{n} {100 * shuf / n:.1f}%  model_top1_target {tgt}/{n} {100 * tgt / n:.1f}%\n"
           f"signal (jacobian - shuffled >= 20 pp): {'YES' if (jac - shuf) * 100 / n >= 20 else 'NO'}\n")
for lo in (1, 8):
    sub = [l for l in layers if l >= lo]
    hs = lambda top, want: any(top[l] & want for l in sub)
    j2 = sum(hs(reads[it["name"]][0], ids(it["intermediates"])) for it in items)
    l2 = sum(hs(reads[it["name"]][1], ids(it["intermediates"])) for it in items)
    s2 = sum(hs(reads[it["name"]][0], ids(items[(i + 1) % n]["intermediates"]) - ids(it["intermediates"])) for i, it in enumerate(items))
    summary += f"exploratory, layers {lo}-{layers[-1]}: jacobian {j2}/{n} {100 * j2 / n:.1f}%  logit_lens {l2}/{n} {100 * l2 / n:.1f}%  shuffled {s2}/{n} {100 * s2 / n:.1f}%\n"
one = lambda words: {e[0] for w in words for e in (tok.encode(" " + w), tok.encode(w)) if len(e) == 1}
v2 = [l for l in layers if l >= 4]
hv = lambda top, want: any(top[l] & want for l in v2)
for lab, sel in (("word", lambda it: not any(c.isdigit() for w in it["intermediates"] for c in w)), ("all", lambda it: True)):
    sc = [(i, it) for i, it in enumerate(items) if sel(it) and one(it["intermediates"])]
    m = len(sc)
    j3 = sum(hv(reads[it["name"]][0], one(it["intermediates"])) for _, it in sc)
    l3 = sum(hv(reads[it["name"]][1], one(it["intermediates"])) for _, it in sc)
    s3 = sum(hv(reads[it["name"]][0], one(items[(i + 1) % n]["intermediates"]) - one(it["intermediates"])) for i, it in sc)
    summary += (f"v2 ({lab} items, single-token candidates, layers 4-{layers[-1]}): scored {m}/{n}  jacobian {j3}/{m} {100 * j3 / max(m, 1):.1f}%  "
                f"logit_lens {l3}/{m} {100 * l3 / max(m, 1):.1f}%  shuffled {s3}/{m} {100 * s3 / max(m, 1):.1f}%\n")
(out / "summary.txt").write_text(summary)
print(summary, end="")
