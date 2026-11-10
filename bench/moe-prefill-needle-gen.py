#!/usr/bin/env python3
"""Build a ~N-word haystack chat request with a random needle placed in the
last 5%, shape of .work/profiles-exp/step0/needle.json. No haystack generator
sat next to that file (checked), so this one is new, not reused.
UNVERIFIED: word count is a proxy for token count (no tokenizer run here,
CPU-only task); "about 100k tokens" in the brief is approximated as N words
of common short vocabulary, which is not a 1:1 word:token ratio for a BPE
tokenizer.
"""
import argparse
import json
import random

VOCAB = ("bridge garden copper stone lamp window winter orange silver signal "
         "market paper station morning quiet river").split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, default=100000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-value", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    code = f"{rng.randint(1000, 9999)}-DELTA-{rng.randint(10, 99)}"
    needle_pos_words = int(args.words * 0.95)

    words = [rng.choice(VOCAB) for _ in range(needle_pos_words)]
    haystack = " ".join(words) + ("." if words else "")
    needle_sentence = f" The vault code for project Heron is {code}."
    tail_words = [rng.choice(VOCAB) for _ in range(args.words - needle_pos_words)]
    tail = " ".join(tail_words) + ("." if tail_words else "")
    question = "\n\nQuestion: what is the vault code for project Heron? Reply with the code only."

    content = haystack + needle_sentence + " " + tail + question
    req = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 20,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with open(args.out, "w") as f:
        json.dump(req, f)
    with open(args.out_value or (args.out + ".value"), "w") as f:
        f.write(code)
    print(f"wrote {args.out} ({len(content.split())} words, needle at "
          f"{needle_pos_words/args.words:.3f} of the prompt), code {code}")


if __name__ == "__main__":
    main()
