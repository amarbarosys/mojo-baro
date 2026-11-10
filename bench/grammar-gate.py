"""JSON enforcement gates 1, 2 and 4 (bench/grammar-protocol.md) against a
running baro-serve. Python is the oracle here: `json` parses, `jsonschema`
validates; neither shares code with grammar/.

  bench/grammar-gate.py URL SERVE_LOG OUT_DIR       QUICK=N: first N schemas at T=0 and 0.7 + one reasoning case
                                                    TRUNC=1: T>0 requests also send top_p 0.9, top_k 20, min_p 0.05
                                                    (the server leaves unset fields untruncated, so without it the
                                                    mask-before-truncation path is never exercised)

Requests go one at a time, so the Nth "grammar masked draws:" receipt in
SERVE_LOG (the engine's stdout, which baro-serve forwards to ITS STDERR with
an "engine:" prefix, so pass the server's stderr file) belongs to the
Nth request sent here. Gate 4 per request: masked draws == accepted, no
MISMATCH word, and masked draws == completion_tokens (no reasoning) or
<= completion_tokens (reasoning on, the think tokens are unmasked).
"""
import json, pathlib, re, sys, time, urllib.error, urllib.request

import jsonschema

url, serve_log, out = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
corpus = pathlib.Path("grammar/corpus")
names = [n for n in (corpus / "manifest.txt").read_text().split() if n]


def post(body):
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:  # keep the server's reason (2026-10-12: 66 bare "HTTP Error 400" twice)
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from None


def receipts():
    return re.findall(r"grammar masked draws: (\d+)\s+accepted: (\d+)\s+terminated: (\w+)(.*)", serve_log.read_text(errors="replace"))


quick = int(__import__("os").environ.get("QUICK", "0"))
trunc = __import__("os").environ.get("TRUNC", "0") == "1"
cases = [(n, t, False) for t in (0.0, 0.7) for n in (names[:quick] if quick else names)]
cases += [(names[0], 0.7, True)] + ([] if quick else [(names[20], 0.0, True)])
rows, fails = [], 0
for name, temp, think in cases:
    schema = json.loads((corpus / name).read_text())
    body = {
        "messages": [{"role": "user", "content": "Reply with one JSON value that matches this JSON schema, filled with realistic data: " + json.dumps(schema)}],
        "max_tokens": 800 if think else 400, "temperature": temp, "seed": 7,
        "response_format": {"type": "json_schema", "json_schema": {"name": name[:-5], "schema": schema}},
        "chat_template_kwargs": {"enable_thinking": think},
    }
    if trunc and temp > 0:
        body.update({"top_p": 0.9, "top_k": 20, "min_p": 0.05})
    before = len(receipts())
    t0 = time.time()
    try:
        resp = post(body)
    except Exception as e:
        resp = {"error": str(e)}
    time.sleep(0.2)
    rc = receipts()
    rec = rc[before] if len(rc) > before else None
    text = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if think and "</think>" in text:
        text = text.split("</think>", 1)[1]
    ntok = (resp.get("usage") or {}).get("completion_tokens")
    why = []
    try:
        jsonschema.validate(json.loads(text), schema)
    except Exception as e:
        why.append(f"invalid: {type(e).__name__}: {str(e)[:80]}")
    if rec is None:
        why.append("no masked-draw receipt")
    else:
        draws, acc, _, tail = int(rec[0]), int(rec[1]), rec[2], rec[3]
        if draws != acc or "MISMATCH" in tail:
            why.append(f"receipt desync {draws}/{acc}")
        if ntok is None or (draws != ntok if not think else not (0 < draws <= ntok)):
            why.append(f"masked draws {draws} vs completion_tokens {ntok}")
    ok = not why
    fails += not ok
    row = {"schema": name, "temperature": temp, "thinking": think, "ok": ok, "why": why, "receipt": rec,
           "completion_tokens": ntok, "finish": ((resp.get("choices") or [{}])[0]).get("finish_reason"),
           "seconds": round(time.time() - t0, 2), "text": text, "raw": resp}
    rows.append(row)
    print(("PASS" if ok else "FAIL"), name, f"T={temp}", "think" if think else "", f"tokens={ntok}", f"receipt={rec[:2] if rec else None}", "; ".join(why), flush=True)

(out / "results.json").write_text(json.dumps(rows, indent=1))
print(f"RESULT {'PASS' if fails == 0 else 'FAIL'}: {len(rows) - fails}/{len(rows)} requests valid with matching receipts")
sys.exit(1 if fails else 0)
