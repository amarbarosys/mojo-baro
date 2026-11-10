#!/usr/bin/env python3
"""Gate for the Mojo tokenizer (serve/tokenizer.mojo via .work/baro-tokenize).

Same cases as tools/test_tokenizer.py (bench/mtp-prompts .tokens files + the
hard set + rendered chat template + empty string), reference = llama-tokenize on
the source GGUF; plus decode(encode(x)) == x. Exit 1 on any mismatch.
Usage: tools/test_tokenizer_mojo.py --gguf G [--llama-tokenize B] [--cli .work/baro-tokenize]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))

HARD_SET = {
    "ascii": "The quick brown fox jumps over the lazy dog.",
    "contractions": "I'm sure they've seen it, but we'll see; IT'S NOT what you'd think. Don't.",
    "leading-space": " hello world",
    # "empty" is covered by the dedicated ("empty", ...) case below, which
    # derives the right reference from add_bos; kept out of HARD_SET so it
    # isn't double-counted with two different (and, for add_bos=true models,
    # differently-correct) expectations.
    "trailing-space": "hello world ",
    "many-spaces": "   three   spaces   everywhere   ",
    "tabs-code": "def f(x):\n\tif x:\n\t\treturn [1, 2, 3]\n\treturn {}\n",
    "spaces-code": "class A:\n    def __init__(self, n=0):\n        self.n = n  # count\n",
    "newlines": "line one\n\nline three\n\n\n\nline seven\n",
    "crlf": "windows\r\nline endings\r\n",
    "mixed-ws": "  \n  \t \n x \t\t y\n",
    "nbsp-zwj": "a\u00a0b\u200bc\u200dd",
    "numbers": "1234567890 3.14159 -42 1,000,000 2026-09-06 0x1F 1e-9 ½ ²",
    "unicode-latin": "Ça va? Señor, façade, naïve, Æsir, Ørsted, Łódź, Ünïcödé",
    "unicode-nfd": "cafe\u0301 e\u0301 n\u0303 A\u030a",
    "cyrillic-greek": "Привет, мир! Καλημέρα κόσμε. Ёлка.",
    "arabic-hebrew": "مرحبا بالعالم שלום עולם",
    "devanagari-thai": "नमस्ते दुनिया สวัสดีชาวโลก",
    "cjk-zh": "你好，世界。今天天气很好。",
    "cjk-ja": "こんにちは世界。カタカナとひらがなと漢字。",
    "cjk-ko": "안녕하세요 세계. 한국어 문장입니다.",
    "cjk-mixed-ascii": "GPU显存24GB, 温度85°C。",
    "emoji": "hello 👋 world 🌍🚀 🇩🇪 👨‍👩‍👧‍👦 👍🏽 ❤️ 😂😂😂",
    "symbols": "∑ ∫ ≠ ≈ → ← ∞ © ® ™ € £ ¥ § ¶ • …",
    "punct-runs": "!!! ??? ... --- *** ((( ))) [[[ ]]] {{{ }}} <<< >>> ::: ;;; ,,,",
    "url-email": "See https://example.com/a/b?x=1&y=2#frag or mail root@example.org.",
    "markdown": "# Title\n\n- item **bold** _it_ `code`\n\n| a | b |\n|---|---|\n| 1 | 2 |\n",
    "json": '{"key": "value", "n": [1, 2.5, -3], "nested": {"ok": true, "x": null}}',
    "special-chat": "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
    "special-eot": "<|endoftext|>text after eot<|endoftext|>",
    "special-added": "<think>\nthinking\n</think>\n<tool_call>{}</tool_call><tool_response>r</tool_response>",
    "special-vision": "<|vision_start|><|image_pad|><|vision_end|> <|fim_prefix|>a<|fim_suffix|>b<|fim_middle|>",
    "special-fake": "<|not_a_token|> <| im_start |> <|im_start",
    "special-adjacent": "a<|im_end|>b <|im_end|> c\n<|im_end|>\n",
    "bytelevel-chars": "Ġ Ċ ĉ ā Ā",
    "long-repeat": "ab" * 200 + " " + "x" * 300,
    "long-ws": " " * 100 + "\n" * 50 + "\t" * 30,
    "apostrophes": "rock'n'roll 'quoted' don't ‘curly’ “double”",
    "hyphen-dash": "well-known — em-dash – en-dash ‐ hyphen",
    "control-chars": "bell\x07 esc\x1b[0m nul-free \x7f",
}


def encode_ref(llama_tokenize, gguf, text):
    if text == "":
        return []  # llama-tokenize refuses an empty prompt; add_bos=false means []
    r = subprocess.run([llama_tokenize, "-m", gguf, "--ids", "-p", text], capture_output=True, text=True, check=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("[")][-1]
    return json.loads(line)


def render_chat(gguf, messages, add_generation_prompt=True):
    """Oracle-side: apply tokenizer.chat_template from the GGUF the way transformers does."""
    import datetime
    import jinja2
    from gguf import GGUFReader
    rd = GGUFReader(gguf)

    def kv_str(key):
        f = rd.fields.get(key)
        return bytes(f.parts[f.data[0]]).decode("utf-8") if f else None

    def kv_int(key):
        f = rd.fields.get(key)
        return int(f.parts[f.data[0]][0]) if f else None

    tpl = kv_str("tokenizer.chat_template")
    if not tpl:
        return None
    toks = rd.fields["tokenizer.ggml.tokens"]
    tok = lambda i: bytes(toks.parts[toks.data[i]]).decode("utf-8") if i is not None else ""
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(jinja2.exceptions.TemplateError(msg))
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False, **kw)
    return env.from_string(tpl).render(messages=messages, add_generation_prompt=add_generation_prompt,
                                       bos_token=tok(kv_int("tokenizer.ggml.bos_token_id")),
                                       eos_token=tok(kv_int("tokenizer.ggml.eos_token_id")),
                                       pad_token=tok(kv_int("tokenizer.ggml.padding_token_id")))


def bos_add_bos_of(gguf):
    """(bos_id, add_bos), read from GGUF metadata -- never inferred from
    whether some case's ref happens to start with bos_id, since a model can
    have a hard-set case whose literal text IS its own bos/eot string (qwen2.5:
    bos_token_id == the <|endoftext|> id, and "special-eot" starts with that
    text -- a false "add_bos" positive if inferred from ref[0])."""
    from gguf import GGUFReader
    rd = GGUFReader(gguf)
    f = rd.fields.get("tokenizer.ggml.bos_token_id")
    bos = int(f.parts[f.data[0]][0]) if f else None
    f = rd.fields.get("tokenizer.ggml.add_bos_token")
    if f is not None:
        return bos, bool(f.parts[f.data[0]][0])
    f = rd.fields.get("tokenizer.ggml.pre")
    pre = bytes(f.parts[f.data[0]]).decode("utf-8") if f else None
    # llama.cpp llm_tokenizer_bpe ctor hardcodes add_bos=true for this pre
    # group when the GGUF carries no explicit add_bos_token key.
    return bos, pre in ("llama3", "llama-v3", "llama-bpe")


def ours_batch(cli, gguf, texts):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\0".join(texts))
        path = f.name
    r = subprocess.run([cli, "batch", path, gguf], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"cli failed: {r.stderr[-800:]}")
    return [[int(x) for x in line.split()] for line in r.stdout.split("\n")[: len(texts)]]


def ours_decode(cli, gguf, ids):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(" ".join(map(str, ids)))
        path = f.name
    r = subprocess.run([cli, "decode-keep", path, gguf], capture_output=True)
    return r.stdout.decode("utf-8", errors="surrogateescape")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="GGUF carrying the tokenizer (Qwythos family for the built-in cases)")
    ap.add_argument("--llama-tokenize", default=os.path.expanduser("~/llama.cpp/build/bin/llama-tokenize"))
    ap.add_argument("--cli", default=str(ROOT / ".work/baro-tokenize"))
    ap.add_argument("--extra", action="append", default=[], metavar="GGUF:TEXT_FILE:IDS_FILE",
                    help="fixed-id case for another model, e.g. Spark ids from llama-server /tokenize")
    a = ap.parse_args()
    gguf = a.gguf
    print(f"gguf {gguf}\nref  {a.llama_tokenize}\ncli  {a.cli}")

    cases = []
    prompts = json.loads((ROOT / "bench/mtp-prompts/prompts.json").read_text())
    for stem in prompts:
        txt = (ROOT / "bench/mtp-prompts" / f"{stem}.txt").read_text()
        cases.append((f"prompt:{stem}", txt, encode_ref(a.llama_tokenize, gguf, txt)))
    for name, text in HARD_SET.items():
        cases.append((f"hard:{name}", text, encode_ref(a.llama_tokenize, gguf, text)))
    chat = render_chat(gguf, [{"role": "user", "content": "What is 2+2?"}])
    if chat:
        cases.append(("chat", chat, encode_ref(a.llama_tokenize, gguf, chat)))
    # llama-tokenize refuses an empty prompt; "OG tokenizer behavior:
    # tokenizer.encode('', add_special_tokens=True) returns [bos] when the
    # model adds one" (llama.cpp llama-vocab.cpp), [] otherwise.
    bos, add_bos = bos_add_bos_of(gguf)
    cases.append(("empty", "", [bos] if add_bos else []))

    got = ours_batch(a.cli, gguf, [c[1] for c in cases])
    fails = 0
    for (name, text, ref), g in zip(cases, got):
        if g != ref:
            fails += 1
            i = next((k for k in range(min(len(g), len(ref))) if g[k] != ref[k]), min(len(g), len(ref)))
            print(f"FAIL {name}: first diff at {i}: ref {ref[i:i+5]} got {g[i:i+5]} (len {len(ref)} vs {len(g)})")
        else:
            # decode(encode(x)) == x is a content round trip; BOS is a
            # generation-context addition, not part of x, so strip it before
            # comparing (matches "add_special_tokens=False" semantics for the
            # purpose of this check only -- the id comparison above already
            # proved encode(add_special=True) agrees with llama-tokenize).
            g_rt = g[1:] if add_bos and g[:1] == [bos] else g
            rt = ours_decode(a.cli, gguf, g_rt)
            if rt != text:
                fails += 1
                print(f"FAIL decode {name}: {rt[:60]!r} != {text[:60]!r}")
    for spec in a.extra:
        g2, tf, idf = spec.split(":")
        text = Path(tf).read_text()
        ref = [int(x) for x in Path(idf).read_text().split()]
        got2 = ours_batch(a.cli, g2, [text])[0]
        ok = got2 == ref and ours_decode(a.cli, g2, got2) == text
        fails += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} extra {Path(g2).name}: {len(ref)} ids{'' if ok else f' got {got2[:8]}'}")
        cases.append(("extra", text, ref))
    print(f"{'PASS' if fails == 0 else 'FAIL'}: {len(cases)} cases, {fails} failures")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
