#!/usr/bin/env python3
# Copyright 2026 amarbaro.org. Licensed under the Apache License, Version 2.0.
"""Dump every distinct tokenizer.chat_template from the GGUFs under ~/Models into
tests/minja-corpus/<name>.jinja, plus the fixed message cases (tests/minja-corpus/cases.json) and
the jinja2 oracle renders (tests/minja-corpus/expected/<template>/<case>.txt).

Oracle = transformers' rendering rules: jinja2 Environment(trim_blocks, lstrip_blocks,
loopcontrols), globals raise_exception + strftime_now (FIXED clock), filter tojson.
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

import jinja2

sys.path.insert(0, str(Path.home() / "llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests/minja-corpus"
MODELS = Path.home() / "Models"
FIXED_NOW = datetime.datetime(2026, 9, 8, 12, 0, 0)

CASES = {
    "user": {"messages": [{"role": "user", "content": "What is the capital of Anhui?"}]},
    "system_user": {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]},
    "three_turn": {"messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}, {"role": "user", "content": "and 3+3?"}]},
    "reasoning": {"messages": [{"role": "user", "content": "Think."}, {"role": "assistant", "content": "Done.", "reasoning_content": "step 1, step 2"}, {"role": "user", "content": "ok"}]},
    "tool_call": {"messages": [{"role": "user", "content": "weather in Hefei"},
                               {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Hefei", "unit": "c"}}}]},
                               {"role": "tool", "content": "{\"temp\": 31}"}],
                  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]},
    "two_tools": {"messages": [{"role": "user", "content": "x"},
                               {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "a", "arguments": {"k": 1}}}, {"type": "function", "function": {"name": "b", "arguments": {"k": [1, 2]}}}]},
                               {"role": "tool", "content": "r1"}, {"role": "tool", "content": "r2"}, {"role": "user", "content": "thanks"}]},
    "no_gen": {"messages": [{"role": "user", "content": "Hi"}], "add_generation_prompt": False},
    "think_off": {"messages": [{"role": "user", "content": "Hi"}], "enable_thinking": False},
    "think_on": {"messages": [{"role": "user", "content": "Hi"}], "enable_thinking": True},
    "blocks": {"messages": [{"role": "user", "content": [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]}]},
    "unicode": {"messages": [{"role": "system", "content": "你是助手。"}, {"role": "user", "content": "Ünïcödé 😀 \"quotes\" \\ back"}]},
    "empty_assistant": {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": ""}, {"role": "user", "content": "again"}]},
}


def kv_str(rd, key):
    f = rd.fields.get(key)
    return bytes(f.parts[f.data[0]]).decode("utf-8") if f else None


def kv_tok(rd, key):
    f = rd.fields.get(key)
    if not f:
        return ""
    toks = rd.fields["tokenizer.ggml.tokens"]
    return bytes(toks.parts[toks.data[int(f.parts[f.data[0]][0])]]).decode("utf-8")


def env():
    e = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)
    e.globals["raise_exception"] = raise_exception
    e.globals["strftime_now"] = lambda fmt: FIXED_NOW.strftime(fmt)
    e.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False, **kw)
    return e


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cases.json").write_text(json.dumps(CASES, ensure_ascii=False, indent=1))
    seen, templates = {}, {}
    for g in sorted(MODELS.glob("*/*.gguf")):
        if "BARO" in g.name or "mmproj" in g.name:
            continue
        rd = GGUFReader(g)
        tpl = kv_str(rd, "tokenizer.chat_template")
        if not tpl:
            continue
        h = hashlib.sha256(tpl.encode()).hexdigest()[:12]
        if h in seen:
            continue
        name = g.parent.name.split("-q")[0].split("-Q")[0]
        seen[h] = name
        special = {"bos_token": kv_tok(rd, "tokenizer.ggml.bos_token_id"), "eos_token": kv_tok(rd, "tokenizer.ggml.eos_token_id"), "pad_token": kv_tok(rd, "tokenizer.ggml.padding_token_id")}
        templates[name] = {"gguf": g.name, "sha": h, **special}
        (OUT / f"{name}.jinja").write_text(tpl)
    (OUT / "templates.json").write_text(json.dumps(templates, ensure_ascii=False, indent=1))
    e = env()
    n_ok = n_err = 0
    for name, info in templates.items():
        t = e.from_string((OUT / f"{name}.jinja").read_text())
        d = OUT / "expected" / name
        d.mkdir(parents=True, exist_ok=True)
        for cname, case in CASES.items():
            kw = {k: v for k, v in case.items() if k != "messages"}
            kw.setdefault("add_generation_prompt", True)
            try:
                out = t.render(messages=case["messages"], bos_token=info["bos_token"], eos_token=info["eos_token"], pad_token=info["pad_token"], **kw)
                (d / f"{cname}.txt").write_text(out)
                n_ok += 1
            except Exception as ex:  # template rejects the case: recorded, engine must raise too
                (d / f"{cname}.error").write_text(f"{type(ex).__name__}: {ex}")
                n_err += 1
    print(f"{len(templates)} templates, {len(CASES)} cases: {n_ok} renders, {n_err} template errors (recorded)")
    for name, info in templates.items():
        print(f"  {name:32s} {info['sha']}  {Path(info['gguf']).name}")


if __name__ == "__main__":
    main()
