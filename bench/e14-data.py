#!/usr/bin/env python3
"""bench/e14-data.py: E14's shared document, one 32k-token haystack with N
embedded magic-number needles, and N follower questions, each asking about a
different needle.

Adapted from bench/ruler/gen.py's niah_multikey (one haystack, 3 needles, 1
query) to N needles and N separate queries sharing ONE haystack, because E14
needs N followers asking DIFFERENT questions about the SAME document, not N
independent documents (briefs/2026-09-15-p3-b2-e14.md, 06-experiments.md E14).

Document tokens end mid-message (chat user turn still open); each question's
tokens are a separately-encoded suffix (closes the user turn, opens the
assistant turn), the same convention bench_latent_handoff.mojo uses for
hand_ids -- concatenation, not re-tokenizing the joined string.

Usage: bench/e14-data.py [--n 10] [--target-tokens 32000] [--seed 0] --out OUT.json
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench" / "ruler"))
import gen as genmod  # noqa: E402
import tok as tokmod  # noqa: E402

CLI = ROOT / ".work/baro-tokenize"
SYS = "You are a helpful assistant. Read the text carefully and answer the question."


def encode_all(texts, gguf, work):
    work.write_text("\0".join(texts))
    r = subprocess.run([str(CLI), "batch", str(work), str(gguf)],
                        capture_output=True, text=True, check=True)
    lines = r.stdout.splitlines()
    if len(lines) != len(texts):
        raise SystemExit(f"tokenizer returned {len(lines)} rows for {len(texts)} texts")
    return [[int(x) for x in ln.split()] for ln in lines]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--target-tokens", type=int, default=32000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf", default=str(Path.home() /
                     "Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf"))
    args = ap.parse_args()

    os.environ["BARO_GGUF"] = args.gguf
    _, _, count = tokmod.load(str(ROOT / ".work/engine-pack-q4"))

    import random
    rng = random.Random(f"e14:{args.seed}")
    keys = genmod.word_pool(rng, args.n)
    values = [genmod.rand_number(rng) for _ in keys]
    needles = [f"One of the special magic numbers for {k} is: {v}." for k, v in zip(keys, values)]
    depths = sorted(rng.random() for _ in range(args.n))
    sents = genmod.essay_sentences()

    intro = (f"There are {args.n} special magic numbers hidden within the following "
              "text, one per topic below. Make sure to memorize all of them; I will "
              "ask about each one afterwards.\n")

    def build_doc(n):
        chosen = genmod.tile(sents, n)
        positions = [int(len(chosen) * d) for d in depths]
        parts, last = [], 0
        for pos, needle in sorted(zip(positions, needles)):
            parts.append(" ".join(chosen[last:pos]))
            parts.append(needle)
            last = pos
        parts.append(" ".join(chosen[last:]))
        return intro + " ".join(p for p in parts if p)

    def chat_prefix(n):
        return f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{build_doc(n)}"

    n_sent = genmod.fit_units(chat_prefix, count, args.target_tokens)
    doc_text = chat_prefix(n_sent)

    questions = []
    for key, val in zip(keys, values):
        q = (f" What is the special magic number for {key} mentioned in the "
             "provided text? Answer with just the number.<|im_end|>\n<|im_start|>assistant\n")
        questions.append({"key": key, "expected": val, "text": q})

    work = ROOT / ".work/e14-data.tok-in"
    all_ids = encode_all([doc_text] + [q["text"] for q in questions], args.gguf, work)
    work.unlink(missing_ok=True)

    out_obj = {
        "seed": args.seed,
        "n": args.n,
        "gguf": args.gguf,
        "document_tokens": all_ids[0],
        "questions": [
            {"id": f"q{i + 1:02d}", "key": q["key"], "expected": q["expected"], "tokens": ids}
            for i, (q, ids) in enumerate(zip(questions, all_ids[1:]))
        ],
    }
    Path(args.out).write_text(json.dumps(out_obj))
    q_lens = [len(q["tokens"]) for q in out_obj["questions"]]
    print(f"document {len(all_ids[0])} tokens, {args.n} questions, "
          f"question tokens min/max: {min(q_lens)}/{max(q_lens)}")
    for q in out_obj["questions"]:
        print(f"  {q['id']}: key={q['key']} expected={q['expected']} tokens={len(q['tokens'])}")


if __name__ == "__main__":
    main()
