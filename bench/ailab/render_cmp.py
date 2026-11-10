#!/usr/bin/env python3
"""P1 comparator (called by render-parity.sh): prompt token ids, baro-serve vs llama-server, same request bytes.

  render_cmp.py --ours URL --fake-out FILE --llama URL --out DIR SET_DIR...
  render_cmp.py --selftest

ours  = POST the body (max_tokens floored to 16000) to baro-serve running on bench/stub-engine.py; the engine request line
        the stub appends to --fake-out carries `prompt` = the ids the real engine would prefill. No render logic is copied.
llama = POST /apply-template {messages, tools, chat_template_kwargs}, then /tokenize {add_special: true, parse_special: true}
        (what llama-server's own chat path does with the rendered text).
Writes DIR/ours/<id>.ids, DIR/llama/<id>.ids, DIR/llama/<id>.txt and DIR/verdict.tsv. Exit 0 only if every id is equal.
A DIFF prints the first differing token index and both decoded strings (llama /detokenize on both id lists).
"""
import glob, json, os, sys, urllib.request

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capture as C  # noqa: E402


def post(url, d, raw=False):
    data = d if raw else json.dumps(d).encode()
    r = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=300))


def first_diff(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def selftest():
    a = [5, 6, 7, 8]
    assert first_diff(a, a) == -1 and first_diff(a, [5, 6, 9, 8]) == 2 and first_diff(a, a[:3]) == 3 and first_diff([], a) == 0
    print("selftest: first_diff catches a flipped id, a short tail and an empty side")


def ours_ids(url, fake_out, body):
    n0 = sum(1 for _ in open(fake_out))
    post(url + "/v1/chat/completions", C.floor_body(body), raw=True)
    lines = open(fake_out).read().splitlines()
    if len(lines) != n0 + 1:
        raise SystemExit(f"FAIL ours: {len(lines) - n0} engine request lines for one POST (want 1)")
    ids = json.loads(lines[-1])["prompt"]
    if not ids or not all(isinstance(x, int) for x in ids):
        raise SystemExit("FAIL ours: engine request line has no id list in `prompt`")
    return ids


def llama_ids(url, body):
    b = json.loads(body)
    p = post(url + "/apply-template", {k: b[k] for k in ("messages", "tools", "chat_template_kwargs")})["prompt"]
    return p, post(url + "/tokenize", {"content": p, "add_special": True, "parse_special": True})["tokens"]


def show(llama, ids, i, w=8):
    return post(llama + "/detokenize", {"tokens": ids[max(0, i - w):i + w]})["content"]


def main():
    a = sys.argv[1:]
    if a == ["--selftest"]:
        return selftest()
    opt, sets, i = {}, [], 0
    while i < len(a):
        if a[i].startswith("--"):
            opt[a[i][2:]], i = a[i + 1], i + 2
        else:
            sets.append(a[i])
            i += 1
    for k in ("ours", "fake-out", "llama", "out"):
        if k not in opt:
            sys.exit(f"FAIL args: --{k} missing")
    if not sets:
        sys.exit("FAIL args: no body dirs")
    selftest()
    os.makedirs(f"{opt['out']}/ours", exist_ok=True)
    os.makedirs(f"{opt['out']}/llama", exist_ok=True)
    rows, bad = [], 0
    for d in sets:
        files = sorted(glob.glob(f"{d}/s*-r*.json"))
        if not files:
            sys.exit(f"FAIL input: no bodies in {d}")
        eq = 0
        for f in files:
            sid, body = os.path.basename(f)[:-5], open(f, "rb").read()
            o = ours_ids(opt["ours"], opt["fake-out"], body)
            text, l = llama_ids(opt["llama"], body)
            json.dump(o, open(f"{opt['out']}/ours/{sid}.ids", "w"))
            json.dump(l, open(f"{opt['out']}/llama/{sid}.ids", "w"))
            open(f"{opt['out']}/llama/{sid}.txt", "w").write(text)
            k = first_diff(o, l)
            rows.append(f"{os.path.basename(d)}\t{sid}\t{len(o)}\t{len(l)}\t{'EQ' if k < 0 else 'DIFF@' + str(k)}")
            if k < 0:
                eq += 1
                continue
            bad += 1
            if bad <= 5:
                print(f"DIFF {sid}: ours {len(o)} ids, llama {len(l)} ids, first differing token {k}")
                print(f"  ours : {show(opt['llama'], o, k)!r}")
                print(f"  llama: {show(opt['llama'], l, k)!r}")
        print(f"{os.path.basename(d)}: {eq}/{len(files)} equal")
    open(f"{opt['out']}/verdict.tsv", "w").write("\n".join(rows) + "\n")
    if bad:
        sys.exit(f"FAIL compare: {bad} of {len(rows)} requests differ, see {opt['out']}/verdict.tsv")
    print(f"PASS render parity: {len(rows)}/{len(rows)} requests, ids equal")


if __name__ == "__main__":
    main()
