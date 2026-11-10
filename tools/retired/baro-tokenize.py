#!/usr/bin/env python3
"""baro-tokenize: text <-> token ids for an engine pack, bit-equal to llama.cpp.

  tools/baro-tokenize encode [--pack DIR] [--chat] [--system TEXT] [-o FILE] [TEXT | -]
  tools/baro-tokenize decode [--pack DIR] [--keep-special] [ID ... | -]
  tools/baro-tokenize prompt [--pack DIR] [--chat] [--system TEXT] [TEXT | -]
  tools/baro-tokenize info   [--pack DIR]

encode prints one id per line (the engine's prompt-tokens.txt layout); -o writes
that file instead. prompt = encode -o <pack>/prompt-tokens.txt, i.e. it sets the
prompt the engine reads on its next run (BARO_PROMPT overrides the path).
--chat wraps TEXT as one user turn in the pack's chat template with the
generation prompt appended. TEXT "-" or no TEXT reads stdin verbatim.
Uses <pack>/tokenizer.json and tokenizer-meta.json from tools/gguf-tokenizer.py.
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    import tokenizers  # noqa: F401
except ImportError:  # re-exec under the repo venv
    py = ROOT / ".venv/bin/python"
    if Path(sys.executable).resolve() != py.resolve() and py.exists():
        os.execv(str(py), [str(py), __file__, *sys.argv[1:]])
    raise

spec = importlib.util.spec_from_file_location("gt", ROOT / "tools/gguf-tokenizer.py")
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)


def read_text(arg):
    if arg is None or arg == "-":
        return sys.stdin.read()
    return arg


def add_common(p):
    p.add_argument("--pack", default=os.environ.get("BARO_PACK", str(ROOT / ".work/engine-pack-q4")))


def add_chat(p):
    p.add_argument("--chat", action="store_true", help="apply the pack's chat template (one user turn + generation prompt)")
    p.add_argument("--system", default=None, help="system message for --chat")


def to_ids(tok, meta, text, chat, system):
    if chat:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
        text = gt.render_chat(meta, msgs, add_generation_prompt=True)
    ids = tok.encode(text, add_special_tokens=False).ids
    if meta.get("add_bos") and meta.get("bos_token_id") is not None:
        ids = [meta["bos_token_id"]] + ids
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode"); add_common(e); add_chat(e)
    e.add_argument("-o", "--out", default=None); e.add_argument("text", nargs="?")
    d = sub.add_parser("decode"); add_common(d)
    d.add_argument("--keep-special", action="store_true"); d.add_argument("ids", nargs="*")
    p = sub.add_parser("prompt"); add_common(p); add_chat(p); p.add_argument("text", nargs="?")
    i = sub.add_parser("info"); add_common(i)
    a = ap.parse_args()

    meta = gt.load_meta(a.pack)
    if a.cmd == "info":
        for k in ("source_gguf", "model", "pre", "vocab_size", "n_merges", "bos_token_id", "eos_token_id",
                  "pad_token_id", "add_bos", "add_eos"):
            print(f"{k}: {meta[k]}")
        print("special_tokens:", " ".join(f"{t}={i}" for t, i in meta["special_tokens"].items()))
        print("added_tokens:", " ".join(f"{t}={i}" for t, i in meta["added_tokens"].items()))
        print("chat_template:", "yes" if meta.get("chat_template") else "no")
        return 0
    tok = gt.load_tokenizer(a.pack)

    if a.cmd == "decode":
        raw = " ".join(a.ids) if a.ids and a.ids != ["-"] else sys.stdin.read()
        ids = [int(x) for x in raw.replace(",", " ").split()]
        sys.stdout.write(tok.decode(ids, skip_special_tokens=not a.keep_special))
        return 0

    ids = to_ids(tok, meta, read_text(a.text), a.chat, a.system)
    out = a.out if a.cmd == "encode" else str(Path(a.pack) / "prompt-tokens.txt")
    body = "".join(f"{i}\n" for i in ids)
    if out:
        Path(out).write_text(body)
        print(f"{len(ids)} tokens -> {out}", file=sys.stderr)
    else:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
