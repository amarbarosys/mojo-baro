#!/usr/bin/env python3
"""Oracle for serve/front/test_text.mojo: render the kq pack's chat template with real jinja2 (pinned) on the
chat rows of exchange/receipts/front/text-oracle.jsonl and write exchange/receipts/front/jinja2-oracle.jsonl.

Why: the Rust front renders with minijinja, which prints a float like 1e22 as 10000000000000000000000.0; jinja2
(what the models were trained on) prints 1e+22. The Mojo test pins the jinja2 text where the two differ.
Rendering rules are transformers' (and tools/minja-corpus.py): trim_blocks, lstrip_blocks, loopcontrols,
tojson = json.dumps(ensure_ascii=False), raise_exception, strftime_now on a fixed clock, kwargs merged over the context.
Edge rows (kind chat) are stored whole; p0 and gate rows as length + sha256 (their text is in the Rust oracle).
Extra row float-args-pyint: the same request with p16 a Python int 10**22 (no front can carry it: Rust's
serde_json and the Mojo Json both read an integer past 64 bits as a float).

Run: uv run --offline --no-project --with jinja2==3.1.6 python3 tools/front-text-jinja-oracle.py
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

import jinja2

PINNED = "3.1.6"
ROOT = Path(__file__).resolve().parent.parent
ORACLE = ROOT / "exchange/receipts/front/text-oracle.jsonl"
OUT = ROOT / "exchange/receipts/front/jinja2-oracle.jsonl"
NOW = 1700000000


def env():
    e = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    e.globals["raise_exception"] = raise_exception
    e.globals["strftime_now"] = lambda f: datetime.datetime.fromtimestamp(NOW, datetime.timezone.utc).strftime(f)
    e.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False, **kw)
    return e


def render(tpl, bos, eos, messages, tools, kwargs, gen=True):
    ctx = {"messages": messages, "add_generation_prompt": gen, "bos_token": bos or "", "eos_token": eos or ""}
    if tools is not None:
        ctx["tools"] = tools
    if isinstance(kwargs, dict):
        ctx.update(kwargs)
    return tpl.render(**ctx)


def p0_messages(body):
    out = []
    for m in body["messages"]:
        n = {"role": m["role"], "content": m["content"]}
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except ValueError:
                    args = {}
                calls.append({"function": {"name": c["function"]["name"], "arguments": args}})
            n["tool_calls"] = calls
        out.append(n)
    return out


def main():
    assert jinja2.__version__ == PINNED, f"jinja2 {jinja2.__version__}, pinned {PINNED}"
    rows = [json.loads(l) for l in ORACLE.read_text().split("\n") if l]
    head = next(r for r in rows if r["kind"] == "header" and r["pack"] == "kq")
    meta = json.loads(Path(head["meta"]).read_text())
    tpl = env().from_string(meta["chat_template"])
    bos, eos = meta.get("bos_token"), meta.get("eos_token")
    out = [{"kind": "header", "jinja2": jinja2.__version__, "python": sys.version.split()[0], "now": NOW,
            "template_sha256": hashlib.sha256(meta["chat_template"].encode()).hexdigest(), "oracle": str(ORACLE.name)}]
    n_same = n_diff = 0

    def put(kind, id, text, rust):
        nonlocal n_same, n_diff
        same = text == rust
        n_same += same
        n_diff += not same
        row = {"kind": kind, "id": id, "equals_rust": same}
        if kind == "chat":
            row["prompt"] = text
        else:
            row["len"] = len(text.encode())
            row["sha256"] = hashlib.sha256(text.encode()).hexdigest()
        out.append(row)

    for r in rows:
        if r.get("pack") != "kq" or r["kind"] in ("header", "raw"):
            continue
        k, id = r["kind"], r["id"]
        if k == "chat":
            q = r["req"]
            put(k, id, render(tpl, bos, eos, q["messages"], q["tools"], q["kwargs"]), r["prompt"])
            if id == "float-args":
                q = json.loads(json.dumps(q))
                q["messages"][1]["tool_calls"][0]["function"]["arguments"]["p16"] = 10**22
                put(k, "float-args-pyint", render(tpl, bos, eos, q["messages"], None, None), r["prompt"])
        elif k == "gate":
            text = (ROOT / "bench/mtp-prompts" / (id + ".txt")).read_text()
            put(k, id, render(tpl, bos, eos, [{"role": "user", "content": text}], None, None), r["prompt"])
        elif k == "p0":
            body = json.loads((ROOT / "bench/ailab" / (id + ".json")).read_bytes())
            put(k, id, render(tpl, bos, eos, p0_messages(body), body.get("tools"), body.get("chat_template_kwargs")), r["prompt"])
    OUT.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out))
    print(f"jinja2 {jinja2.__version__}: {len(out) - 1} rows, equal to Rust {n_same}, differing {n_diff}")
    for o in out[1:]:
        if not o["equals_rust"]:
            print("  differs from Rust:", o["kind"], o["id"])


main()
