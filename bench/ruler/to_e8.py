#!/usr/bin/env python3
"""bench/ruler/to_e8.py: turn bench/ruler/gen.py JSONL sets into an E8/E12 task file.

RULER prompts are completion-style: they end with an answer prefix (" The special
magic ... is", " Answer: ..."). The E8 harness is chat-shaped and hands B its own
"Give only the answer" turn, so the prefix is cut off and the rest becomes the
user message. Scoring is RULER's string_match_all (bench/e8_score.py, type "ruler").

Usage: bench/ruler/to_e8.py OUT.json IN1.jsonl [IN2.jsonl ...]
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / ".work/baro-tokenize"
GGUF = Path.home() / "Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf"
SYS = "You are a helpful assistant. Read the text carefully and answer the question."
PREFIXES = (" The special magic ", " Answer: ")


def strip_prefix(prompt):
    cut = max(prompt.rfind(p) for p in PREFIXES)
    if cut < 0:
        raise SystemExit(f"no answer prefix found: ...{prompt[-120:]!r}")
    return prompt[:cut]


def chat_prompt(user):
    # Same template as tools/generate_e8_tasks.mojo chat_prompt().
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")


def encode_all(texts, work):
    work.write_text("\0".join(texts))
    r = subprocess.run([str(CLI), "batch", str(work), str(GGUF)],
                       capture_output=True, text=True, check=True)
    lines = r.stdout.splitlines()
    if len(lines) != len(texts):
        raise SystemExit(f"tokenizer returned {len(lines)} rows for {len(texts)} texts")
    return [[int(x) for x in ln.split()] for ln in lines]


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])
    rows = []
    for p in sys.argv[2:]:
        rows += [json.loads(ln) for ln in Path(p).read_text().splitlines() if ln.strip()]
    fulls = [chat_prompt(strip_prefix(r["prompt"])) for r in rows]
    ids = encode_all(fulls, out.with_suffix(".tok-in"))
    tasks = [{"id": r["id"], "type": "ruler", "task": r["task"], "size": r["size"],
              "full_prompt": f, "tokens": t, "expected": r["answers"]}
             for r, f, t in zip(rows, fulls, ids)]
    out.write_text(json.dumps(tasks))
    out.with_suffix(".tok-in").unlink()
    lens = sorted(len(t["tokens"]) for t in tasks)
    print(f"{len(tasks)} tasks -> {out}; tokens min {lens[0]} max {lens[-1]}")


if __name__ == "__main__":
    main()
