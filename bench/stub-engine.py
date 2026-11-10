#!/usr/bin/env python3
"""CPU stand-in for serve/engine.mojo behind baro-serve, for dry-running a GPU job.

    ENGINE=bench/stub-engine.py FAKE_OUT=requests.jsonl bench/regescore-blender-ab.sh baro OUT

Prints the knob lines a job reads back (BARO_TIER lines only when BARO_TIER is
set), the ready line, and an empty done line per request; every request line
baro-serve sends is appended to FAKE_OUT, so the dry run can check what the
engine would have received (prompt length, sampling fields, checkpoint hints).
STUB_STUCK=1 keeps the stub alive after stdin EOF (lifecycle checks); BARO_SEQS=N echoes the multi-session line.
Knobs it does not print must fail the job's read-back: that proves the gate.
"""
import json
import os
import sys

out = open(os.environ["FAKE_OUT"], "a")
print("BARO_MEGA:", "True" if os.environ.get("BARO_MEGA") == "1" else "False")
if os.environ.get("BARO_TIER"):
    print("expert tier: cap", os.environ["BARO_TIER"], " cache 0 GB  host store 0 GB  mode pinned+zc  hot_bytes 0.0")
print("BARO_PREFILL: True  chunk", os.environ.get("BARO_PREFILL_C", "1024"), " min rows", 96 if os.environ.get("BARO_TIER") else 16)
print("BARO_SPEC: False")
print(json.dumps({"ready": True, "tmax": int(os.environ.get("BARO_TMAX", "8192")), "mrows": 8, "kmax": 8,
                  "spec_k": 0, "pack": os.environ["BARO_PACK"]}), flush=True)
if os.environ.get("BARO_SEQS"):  # multi-session echo, printed after the ready line like serve/engine.mojo
    print("BARO_SEQS:", os.environ["BARO_SEQS"], flush=True)
for line in sys.stdin:
    out.write(line)
    out.flush()
    try:
        rid = json.loads(line).get("id", 0)
    except ValueError:
        rid = 0
    print(json.dumps({"id": rid, "done": True, "n": 0, "prefill_s": 0.0, "decode_s": 0.0, "tok_s": 0.0, "cached": int(os.environ.get("STUB_CACHED", "0"))}), flush=True)
if os.environ.get("STUB_STUCK"):  # an engine that ignores stdin EOF (mid-decode hang): only PDEATHSIG or a kill ends it
    import time
    time.sleep(600)
