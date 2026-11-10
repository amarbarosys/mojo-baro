#!/usr/bin/env python3
"""A scripted engine for the Rust front: the line protocol of serve/PROTOCOL.md, nothing else.

FAKE_SCRIPT = JSON list of token ids to answer every request with; FAKE_MODE = normal | err_before | err_after.
serve/front/fakes.mojo `FakeEngine` emits the same events and the same done line, so the same request
gives the same answer through either front.
"""
import json
import os
import sys

script = json.loads(os.environ.get("FAKE_SCRIPT", "[]"))
mode = os.environ.get("FAKE_MODE", "normal")


def out(o):
    sys.stdout.write(json.dumps(o, separators=(",", ":")) + "\n")
    sys.stdout.flush()


out({"ready": True, "tmax": 16384, "mrows": 4, "kmax": 8, "spec_k": 0, "pack": "fake-pack", "kv": "f32"})
for line in sys.stdin:
    try:
        req = json.loads(line)
    except ValueError:
        continue
    if "cancel" in req or "id" not in req:
        continue
    rid = req["id"]
    if mode == "err_before":
        out({"id": rid, "error": "prompt rejected"})
        continue
    toks = script[:2] if mode == "err_after" else script
    for t in toks:
        out({"id": rid, "tok": t})
    if mode == "err_after":
        out({"id": rid, "error": "boom"})
        continue
    out({"id": rid, "done": True, "n": len(toks), "prefill_s": 0.5, "decode_s": 0.25, "tok_s": 4.0,
         "cached": 0, "prefill_rows": 7, "finish": "length"})
