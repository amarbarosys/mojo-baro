"""JSON enforcement gate 5 (bench/grammar-protocol.md): decode cost of
response_format, same server, same stint.

  bench/grammar-cost.py URL OUT_DIR

For the first 20 corpus schemas: arm G sends the schema request (T=0,
reasoning off, spec forced off by the engine); arm U sends the same messages
with no response_format, spec false, max_tokens = G's completion_tokens, so
both arms decode the same number of tokens. Order alternates G,U / U,G per
prompt. Per request ms/token = decode_s / completion_tokens; the report is
the 20-prompt median per arm and the ratio.
"""
import json, pathlib, statistics, sys, urllib.request

url, out = sys.argv[1], pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
corpus = pathlib.Path("grammar/corpus")
names = [n for n in (corpus / "manifest.txt").read_text().split() if n][:20]


def post(body):
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def ms_per_tok(r):
    return 1000.0 * r["timings"]["decode_s"] / r["usage"]["completion_tokens"]


rows = []
for i, name in enumerate(names):
    schema = json.loads((corpus / name).read_text())
    base = {"messages": [{"role": "user", "content": "Reply with one JSON value that matches this JSON schema, filled with realistic data: " + json.dumps(schema)}],
            "temperature": 0, "seed": 7, "chat_template_kwargs": {"enable_thinking": False}}
    g_body = dict(base, max_tokens=400, response_format={"type": "json_schema", "json_schema": {"name": name[:-5], "schema": schema}})
    if i % 2 == 0:
        g = post(g_body)
        n = g["usage"]["completion_tokens"]
        u = post(dict(base, max_tokens=n, spec=False))
    else:
        g0 = post(g_body)
        n = g0["usage"]["completion_tokens"]
        u = post(dict(base, max_tokens=n, spec=False))
        g = post(g_body)
    ok = g["usage"]["completion_tokens"] == u["usage"]["completion_tokens"]
    rows.append({"schema": name, "tokens": n, "g_ms_tok": ms_per_tok(g), "u_ms_tok": ms_per_tok(u),
                 "g_drafted": g["timings"].get("drafted"), "u_drafted": u["timings"].get("drafted"), "same_len": ok})
    print(name, n, round(rows[-1]["g_ms_tok"], 3), round(rows[-1]["u_ms_tok"], 3), "same_len" if ok else "LEN_DIFF", flush=True)

gm = statistics.median(r["g_ms_tok"] for r in rows)
um = statistics.median(r["u_ms_tok"] for r in rows)
summary = {"n": len(rows), "median_ms_tok_grammar": gm, "median_ms_tok_plain": um, "cost_ratio": gm / um,
           "tok_s_grammar": 1000 / gm, "tok_s_plain": 1000 / um, "all_same_len": all(r["same_len"] for r in rows),
           "any_drafted": any(r["g_drafted"] or r["u_drafted"] for r in rows)}
(out / "cost.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
print("SUMMARY", json.dumps(summary))
