#!/usr/bin/env python3
"""bench/ruler/gen.py: generate the RULER subset (design docs/design/agent-engine-2026-09.md
sec.9) as JSONL prompt sets, sized in OUR tokenizer's tokens.

Tasks (RULER's own scripts/synthetic.yaml complexity space):
  niah_single    essay haystack, 1 key, needle value alternates numbers/uuids
  niah_multikey  essay haystack, 3 keys, 1 value, 1 query
  vt             noise haystack, 1 chain, hops rotate 3/4/5 across examples
  cwe            RULER's own cwe defaults (freq_cw=30, freq_ucw=3, num_cw=10)

Deviations from RULER's harness, not the task definitions (see bench/ruler-protocol.md):
own tokenizer, own adjective x noun word pool instead of wonderwords + a
465k-word dictionary, a regex sentence splitter instead of nltk.

Usage: bench/ruler/gen.py [--pack DIR] [--out DIR] [--tasks t1,t2,...]
       [--sizes 4096,8192,...] [-n N] [--seed SEED]

Writes OUTDIR/<task>_<size>.jsonl, one line per example:
  {"id","prompt","answers":[...],"task","size","seed"}
"""
import argparse
import functools
import hashlib
import json
import random
import re
import string
import sys
import uuid as uuidlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tok as tokmod

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"

SIZES = [4096, 8192, 16384, 32768, 65536, 131072]
TASKS = ["niah_single", "niah_multikey", "vt", "cwe"]
TOKENS_TO_GENERATE = {"niah_single": 128, "niah_multikey": 128, "vt": 30, "cwe": 120}

ADJS = ("brave calm dark eager fair glad happy icy jolly keen loud merry "
        "neat odd plain quiet rapid sad tall urban vast wide young zesty "
        "bold clever dry early fancy grand humble inner jagged kind lively "
        "mild noisy old proud rough silent tidy usual vivid warm").split()
NOUNS = ("apple bridge cloud desk engine forest garden harbor island jungle "
         "kettle ladder mirror needle ocean pencil quarry river stone table "
         "umbrella valley window anchor basket candle drum ember flute globe "
         "hammer ink jar knot lamp market nest oven path").split()

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."

NIAH_TEMPLATE = ("A special magic {v} is hidden within the following text. Make sure to "
                  "memorize it. I will quiz you about the {v} afterwards.\n{context}\nWhat "
                  "is the special magic {v} for {query} mentioned in the provided text?")
NIAH_PREFIX = " The special magic {v} for {query} mentioned in the provided text is"

VT_TEMPLATE = ("Memorize and track the chain(s) of variable assignment hidden in the "
               "following text.\n\n{context}\nQuestion: Find all variables that are "
               "assigned the value {query} in the text above.")
VT_PREFIX = (" Answer: According to the chain(s) of variable assignment in the text "
             "above, {num_v} variables are assigned the value {query}, they are: ")

CWE_TEMPLATE = ("Below is a numbered list of words. In these words, some appear more "
                 "often than others. Memorize the ones that appear most often.\n{context}\n"
                 "Question: What are the 10 most common words in the above list?")
CWE_PREFIX = " Answer: The top 10 words that appear most often in the list are:"


@functools.lru_cache(1)
def essay_sentences():
    f = DATA_DIR / "essays.txt"
    if not f.exists():  # third-party essays, not in the repo: see bench/ruler-protocol.md "Essay haystack"
        raise SystemExit(f"FAIL essays: {f} is missing; fetch it as bench/ruler-protocol.md says (sha256 b3a70ebc054f...)")
    text = re.sub(r"\s+", " ", f.read_text()).strip()
    return SENT_SPLIT.split(text)


def tile(units, n):
    if not units or n <= 0:
        return []
    reps = n // len(units) + 1
    return (units * reps)[:n]


def word_pool(rng, n):
    """n distinct adj-noun combos; extended with a numeric suffix past the base pool."""
    base = [f"{a}-{b}" for a in ADJS for b in NOUNS]
    rng.shuffle(base)
    if n <= len(base):
        return base[:n]
    out = list(base)
    k = 0
    while len(out) < n:
        out.append(f"{base[k % len(base)]}-{k // len(base)}")
        k += 1
    return out[:n]


def rand_number(rng):
    return str(rng.randint(10**6, 10**7 - 1))


def rand_uuid(rng):
    return str(uuidlib.UUID(int=rng.getrandbits(128), version=4))


def fit_units(build, count, target):
    """Largest n with count(build(n)) <= target, via exponential then binary search."""
    lo, hi = 0, 8
    while count(build(hi)) < target:
        lo, hi = hi, hi * 2
        if hi > 4_000_000:
            break
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if count(build(mid)) <= target:
            lo = mid
        else:
            hi = mid
    return lo


def gen_niah_single(rng, count, target, idx):
    key = word_pool(rng, 1)[0]
    use_uuid = idx % 2 == 1
    value = rand_uuid(rng) if use_uuid else rand_number(rng)
    vplural = "uuids" if use_uuid else "numbers"
    vsingular = "uuid" if use_uuid else "number"
    needle = f"One of the special magic {vplural} for {key} is: {value}."
    sents = essay_sentences()
    depth = rng.random()

    def build(n):
        chosen = tile(sents, n)
        pos = int(len(chosen) * depth)
        ctx = " ".join(chosen[:pos] + [needle] + chosen[pos:])
        return (NIAH_TEMPLATE.format(v=vsingular, context=ctx, query=key)
                + NIAH_PREFIX.format(v=vsingular, query=key))

    n = fit_units(build, count, target)
    return build(n), [value]


def gen_niah_multikey(rng, count, target):
    keys = word_pool(rng, 3)
    values = [rand_number(rng) for _ in keys]
    needles = [f"One of the special magic numbers for {k} is: {v}." for k, v in zip(keys, values)]
    order = list(range(3))
    rng.shuffle(order)
    depths = sorted(rng.random() for _ in range(3))
    query_idx = rng.randrange(3)
    sents = essay_sentences()

    def build(n):
        chosen = tile(sents, n)
        positions = [int(len(chosen) * d) for d in depths]
        parts, last = [], 0
        for pos, ni in zip(positions, [needles[i] for i in order]):
            parts.append(" ".join(chosen[last:pos]))
            parts.append(ni)
            last = pos
        parts.append(" ".join(chosen[last:]))
        ctx = " ".join(parts)
        return (NIAH_TEMPLATE.format(v="number", context=ctx, query=keys[query_idx])
                + NIAH_PREFIX.format(v="number", query=keys[query_idx]))

    n = fit_units(build, count, target)
    return build(n), [values[query_idx]]


def make_chain(rng, num_hops):
    names = set()
    while len(names) < num_hops + 1:
        names.add("".join(rng.choices(string.ascii_uppercase, k=5)))
    names = list(names)
    value = str(rng.randint(10000, 99999))
    lines = [f"VAR {names[0]} = {value}"]
    for j in range(num_hops):
        lines.append(f"VAR {names[j + 1]} = VAR {names[j]}")
    return names, value, lines


def gen_vt(rng, count, target, num_hops):
    names, value, lines = make_chain(rng, num_hops)
    demo_names, demo_value, demo_lines = make_chain(rng, num_hops)
    demo_sents = [NOISE] * 30
    for pos in sorted(rng.sample(range(30), len(demo_lines)), reverse=True):
        demo_sents.insert(pos, demo_lines.pop())
    demo = (VT_TEMPLATE.format(context="\n".join(demo_sents), query=demo_value)
            + VT_PREFIX.format(num_v=num_hops + 1, query=demo_value)
            + ", ".join(demo_names) + ".\n\n")
    depths = sorted(rng.random() for _ in lines)

    def build(n):
        sents = [NOISE] * n
        for d, line in sorted(zip(depths, lines), key=lambda dl: -dl[0]):
            sents.insert(int(n * d), line)
        return demo + VT_TEMPLATE.format(context="\n".join(sents), query=value) \
            + VT_PREFIX.format(num_v=num_hops + 1, query=value)

    n = fit_units(build, count, target)
    return build(n), names


def get_example(rng, num_words, common_repeat, uncommon_repeat, common_count, common=None):
    if common is None:
        pool = word_pool(rng, num_words)
        common, uncommon = pool[:common_count], pool[common_count:]
    else:
        common_set = set(common)
        pool = [w for w in word_pool(rng, num_words + common_count) if w not in common_set]
        uncommon = pool[: max(num_words - common_count, 0)]
    lst = common * common_repeat + uncommon * uncommon_repeat
    rng.shuffle(lst)
    ctx = " ".join(f"{i + 1}. {w}" for i, w in enumerate(lst))
    return ctx, common


def gen_cwe(rng, count, target, seed_tag):
    demo_ctx, demo_common = get_example(rng, 40, 10, 3, 10)
    demo = (CWE_TEMPLATE.format(context=demo_ctx) + " "
            + " ".join(f"{i + 1}. {w}" for i, w in enumerate(demo_common)) + "\n")
    common = word_pool(rng, 10)

    def build(n):
        local = random.Random(f"cwe-uncommon:{seed_tag}:{n}")
        ctx, _ = get_example(local, max(n - 10, 0), 30, 3, 10, common=common)
        return demo + CWE_TEMPLATE.format(context=ctx) + CWE_PREFIX

    n = fit_units(build, count, target)
    return build(n), common


def generate(task, size, num_samples, seed, count):
    rng = random.Random(f"{task}:{size}:{seed}")
    target = size - TOKENS_TO_GENERATE[task]
    rows = []
    for i in range(num_samples):
        if task == "niah_single":
            prompt, answers = gen_niah_single(rng, count, target, i)
        elif task == "niah_multikey":
            prompt, answers = gen_niah_multikey(rng, count, target)
        elif task == "vt":
            prompt, answers = gen_vt(rng, count, target, 3 + (i % 3))
        elif task == "cwe":
            prompt, answers = gen_cwe(rng, count, target, f"{task}:{size}:{seed}:{i}")
        else:
            raise SystemExit(f"unknown task {task!r}")
        rows.append({"id": f"{task}-{size}-{i:04d}", "prompt": prompt, "answers": answers,
                     "task": task, "size": size, "seed": seed})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", default=str(ROOT / ".work/engine-pack-q4"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "prompts"))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("-n", "--num-samples", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    _, _, count = tokmod.load(a.pack)
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tasks = a.tasks.split(",")
    sizes = [int(s) for s in a.sizes.split(",")]

    hashes = {}
    for task in tasks:
        for size in sizes:
            rows = generate(task, size, a.num_samples, a.seed, count)
            path = outdir / f"{task}_{size}.jsonl"
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[path.name] = h
            toks = sorted(count(r["prompt"]) for r in rows)
            mid = toks[len(toks) // 2]
            print(f"{path}: n={len(rows)} tok(min/med/max)={toks[0]}/{mid}/{toks[-1]} "
                  f"target={size} sha256={h[:16]}")
    print(json.dumps(hashes, indent=1))


if __name__ == "__main__":
    main()
