#!/usr/bin/env python3
"""Writes bench/ailab/render-edge/: tools and history that stress `tojson` (P1, render-parity.sh vs llama-server).
Non-ASCII (BMP, emoji), HTML characters, quotes, backslash, control characters, key order that is not alphabetical,
empty object and array, int, bool, null, nested unicode in assistant tool_calls arguments; raw UTF-8 and \\u-escaped bodies.
Floats stay 0.1, 2.5, -2.5, 1e22, 1e-07 (llama.cpp prints the same as Python). NOT here: 1.0, 100.0, 123456789.0, 3.14159265358979:
llama.cpp's tojson prints them as 1, 100, 1.23457e+08, 3.14159 (6 digits), Python and HF print them exactly; those are pinned
against Python's strings by the cargo test text::tests::tojson_is_python_json_dumps.
Tool and function keys are in the canonical OpenAI order (type, function{name, description, parameters}): llama.cpp rebuilds
each tool in that order whatever the client sent, HF and baro-serve pass the client's JSON as sent; every real client
(the runner included) sends the canonical order. Inside `parameters` the order is the client's on both sides (tested)."""
import hashlib, json, os, sys

d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render-edge")
os.makedirs(d, exist_ok=True)
desc = 'Café → naïve 日本語 😀 <b>&amp;</b> it\'s "quoted" back\\slash /slash\ttab\nnewline \u0001 ctl'
tools = [
    {"type": "function", "function": {"name": "zeta", "description": desc, "parameters": {"type": "object", "properties": {
        "z_last": {"type": "string", "description": "é"}, "a_first": {"type": "number", "default": 2.5},
        "ratio": {"type": "number", "default": 0.1, "minimum": -2.5, "maximum": 1e22}, "tiny": {"type": "number", "default": 1e-07},
        "big": {"type": "integer", "default": 12345678901234567}, "flag": {"type": "boolean", "default": False}, "none": {"default": None},
        "empty_o": {"type": "object", "properties": {}}, "empty_a": {"type": "array", "items": {"type": "string"}, "enum": []},
        "mode": {"type": "string", "enum": ["b", "a", "ü"]}}, "required": ["z_last"]}}},
    {"type": "function", "function": {"name": "alpha", "description": "no args", "parameters": {"properties": {}, "type": "object"}}}]
args = {"z_last": "süß → 日本", "nested": {"k": [1, 2.5, {"é": None}], "b": True}, "n": 3}
user = {"role": "user", "content": "Wo bin ich? 😀 <tag> & 'q'"}
asst = {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "zeta", "arguments": json.dumps(args)}}]}
tool = {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"ok": True, "text": "é→😀"})}


def body(msgs):
    return {"model": "local", "messages": [{"role": "system", "content": "Sys é"}] + msgs, "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": False}, "tools": tools}


files = {"s01-r1": (body([user]), False), "s02-r1": (body([user]), True),
         "s03-r2": (body([user, asst, tool]), False), "s04-r2": (body([user, asst, tool]), True)}
rows = []
for k, (b, esc) in files.items():
    data = json.dumps(b, ensure_ascii=esc).encode()
    open(f"{d}/{k}.json", "wb").write(data)
    rows.append(f"{k}\t{k[-1]}\t{hashlib.md5(data).hexdigest()}\t{len(data)}")
man = "\n".join(rows) + "\n"
open(f"{d}/MANIFEST.tsv", "w").write(man)
open(f"{d}/SET.md5", "w").write(hashlib.md5(man.encode()).hexdigest() + "\n")
print(f"make-edge: {len(files)} bodies, set md5 {hashlib.md5(man.encode()).hexdigest()} -> {d}", file=sys.stderr)
