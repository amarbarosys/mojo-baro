#!/usr/bin/env python3
# Mock BARO_SERVE engine for bench/tests/a2-gate-selftest.sh: reference mode emits 64 ids per request,
# forced mode prints "forced agreement"; MOCK_BAD=<id> answers 50% on that request; "state_save"
# writes the file, "state_load" reports cached = its token count (error if missing); MOCK_DIE=<k>
# exits after k prompt requests (a killed job). Every prompt served is appended to $MOCK_LOG.
import json, os, sys
j = lambda o: json.dumps(o, separators=(",", ":"))
print(j({"ready": True, "tmax": 32768}), flush=True)
bad = int(os.environ.get("MOCK_BAD", "0")); log = os.environ.get("MOCK_LOG"); die = int(os.environ.get("MOCK_DIE", "0"))
served = 0
for line in sys.stdin:
    d = json.loads(line); i = d["id"]
    if "state_save" in d:
        open(d["state_save"], "w").write(str(len(d["prompt"]))); print(j({"id": i, "done": True, "cached": 0, "tok_s": 1.0}), flush=True); continue
    cached = 0
    if "state_load" in d:
        if not os.path.exists(d["state_load"]): print(j({"id": i, "error": "no state file"}), flush=True); continue
        cached = int(open(d["state_load"]).read())
    if log: open(log, "a").write(f"{i}\n")
    if "force" in d: print(f"forced agreement: {32 if i == bad else 64} / 64", flush=True)
    else:
        for t in range(64): print(j({"id": i, "tok": t}), flush=True)
    print(j({"id": i, "done": True, "cached": cached, "tok_s": 50.0}), flush=True)
    served += 1
    if die and served >= die: sys.exit(0)
