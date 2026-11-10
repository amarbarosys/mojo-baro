#!/usr/bin/env python3
"""bench/e14-llama-prep.py: write one token-id prompt file per E14 follower
(document_tokens + that follower's question tokens, concatenated, same
convention as bench/e14-handoff.mojo), for bench/e14-llama-run.sh to post to
llama-server one at a time. Token ids are OUR tokenizer's (bench/e14-data.py),
not llama.cpp's own -- llama-server accepts a prompt as token ids directly, so
no re-tokenization happens on that side either.

Usage: bench/e14-llama-prep.py --doc .work/e14/doc.json --out-dir .work/e14/llama
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    doc = json.loads(Path(args.doc).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same reasoning suppression as bench/e14-handoff.mojo's nothink_ids
    # (`<think>\n\n</think>\n\n`, our tokenizer): without it llama.cpp spends
    # the whole n_predict budget on an unsuppressed <think> preamble, which
    # is not a fair wall-clock comparison against the Mojo arms' answers.
    nothink_ids = [248068, 271, 248069, 271]

    for q in doc["questions"]:
        ids = doc["document_tokens"] + q["tokens"] + nothink_ids
        (out_dir / f"prompt-{q['id']}.txt").write_text(" ".join(str(t) for t in ids))
    print(f"wrote {len(doc['questions'])} prompt files to {out_dir}")


if __name__ == "__main__":
    main()
