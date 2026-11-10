#!/usr/bin/env python3
"""Task eval by identical token ids (bench/quality-protocol.md amendment 2).

  prep  --tok-url CPU_LLAMA --out DIR       render + tokenize all tasks, EOG ids
  ours  --engine E --pack P --out DIR       engine stdin protocol, one GPU arm
  llama --url GPU_LLAMA --out DIR           llama-server /completion, same ids
  score --tok-url CPU_LLAMA --out DIR       detokenize both arms, score, print

Scoring uses bench/e8_score.py's exact-match functions, unchanged from the first sweep.
"""
import argparse
import re
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from e8_score import extract_last_int, parse_json_obj, subset_match  # noqa: E402

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE_RE = re.compile(r"^```[a-zA-Z]*\n(.*)\n```$", re.DOTALL)
OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def sys_prompts(tasks):
    out = {}
    for t in tasks:
        m = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", t["full_prompt"], re.DOTALL)
        if m and t["type"] not in out:
            out[t["type"]] = m.group(1)
    return out


def score_one(task, text):
    t = THINK_RE.sub("", text).strip()
    m = FENCE_RE.match(t)
    if m:
        t = m.group(1).strip()
    if task["type"] == "math":
        got = extract_last_int(t)
        ok = got is not None and got == task["expected"]
        return ok, ok, "" if ok else f"got {got} want {task['expected']}"
    m2 = OBJ_RE.search(t)
    parsed = parse_json_obj(m2.group(0) if m2 else t)
    if parsed is None:
        return False, False, "did not parse as JSON"
    exact = parsed == task["expected"]
    return exact, subset_match(parsed, task["expected"]), "" if exact else f"parsed {parsed!r} vs {task['expected']!r}"


EOG_STRINGS = ["<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<|end_of_text|>", "</s>", "<|end|>"]
MAX_TOKENS = 300


def post(url, path, body, timeout=600):
    r = requests.post(url.rstrip("/") + path, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def prep(a):
    tasks = json.loads(Path(a.tasks).read_text())
    if a.limit > 0:
        tasks = tasks[:a.limit]
    sys_by_type = sys_prompts(tasks)
    items = []
    for t in tasks:
        msgs = [{"role": "system", "content": sys_by_type[t["type"]]}, {"role": "user", "content": t["prompt_text"]}]
        prompt = post(a.tok_url, "/apply-template", {"messages": msgs, "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
        ids = post(a.tok_url, "/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"]
        items.append({"id": t["id"], "prompt_ids": ids})
    props = requests.get(a.tok_url.rstrip("/") + "/props", timeout=60).json()
    eog = set()
    for s in EOG_STRINGS:
        toks = post(a.tok_url, "/tokenize", {"content": s, "add_special": False, "parse_special": True})["tokens"]
        if len(toks) == 1:
            eog.add(toks[0])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "prompts.json").write_text(json.dumps({"eog": sorted(eog), "items": items,
                                                          "chat_template_head": str(props.get("chat_template", ""))[:200]}))
    print(f"prep: thinking off, {len(items)} prompts, first prompt {len(items[0]['prompt_ids'])} ids, eog {sorted(eog)}")


def ours(a):
    p = json.loads((Path(a.out) / "prompts.json").read_text())
    env = dict(os.environ)
    env.update(BARO_SERVE="1", BARO_PACK=a.pack)
    for kv in (a.env or "").split():
        k, v = kv.split("=", 1)
        env[k] = v
    proc = subprocess.Popen([a.engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=open(Path(a.out) / "ours-engine.err", "w"), env=env, text=True, bufsize=1)
    while True:
        line = proc.stdout.readline()
        if not line:
            sys.exit("VOID: engine exited before ready")
        if line.startswith('{"ready":true'):
            ready = line.strip()
            break
    stop = [[e] for e in p["eog"]]
    out = []
    for i, it in enumerate(p["items"]):
        rid = i + 1
        proc.stdin.write(json.dumps({"id": rid, "prompt": it["prompt_ids"], "n": MAX_TOKENS, "spec": bool(a.spec), "stop": stop}) + "\n")
        proc.stdin.flush()
        toks, done = [], None
        while done is None:
            line = proc.stdout.readline()
            if not line:
                sys.exit(f"VOID: engine exited on item {it['id']}")
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("id") != rid:
                continue
            if "error" in d:
                done = d
            elif d.get("done"):
                done = d
            elif "tok" in d:
                toks.append(d["tok"])
        out.append({"id": it["id"], "tokens": toks, "done": done})
    proc.stdin.close()
    proc.wait(timeout=60)
    (Path(a.out) / "ours-ids.json").write_text(json.dumps({"ready": ready, "items": out}))
    print(f"ours: spec={bool(a.spec)} {len(out)} items, errors {sum(1 for o in out if 'error' in (o['done'] or {}))}")


def llama(a):
    p = json.loads((Path(a.out) / "prompts.json").read_text())
    from concurrent.futures import ThreadPoolExecutor

    def one(it):
        d = post(a.url, "/completion", {"prompt": it["prompt_ids"], "n_predict": MAX_TOKENS, "temperature": 0,
                                        "cache_prompt": False, "return_tokens": True, "samplers": ["top_k"], "top_k": 1})
        return {"id": it["id"], "tokens": d.get("tokens", []), "stop_type": d.get("stop_type"),
                "tokens_evaluated": d.get("tokens_evaluated")}

    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(one, p["items"]))
    (Path(a.out) / "llama-ids.json").write_text(json.dumps({"items": out}))
    print(f"llama: {len(out)} items, first tokens_evaluated {out[0]['tokens_evaluated']} vs prompt {len(p['items'][0]['prompt_ids'])}")


def score(a):
    tasks = {t["id"]: t for t in json.loads(Path(a.tasks).read_text())}
    p = json.loads((Path(a.out) / "prompts.json").read_text())
    eog = set(p["eog"])
    res = {}
    for arm in ("ours", "llama"):
        data = json.loads((Path(a.out) / f"{arm}-ids.json").read_text())
        exact = subset = 0
        items = []
        for it in data["items"]:
            toks = it["tokens"]
            cut = next((i for i, t in enumerate(toks) if t in eog), len(toks))
            toks = toks[:cut]
            text = post(a.tok_url, "/detokenize", {"tokens": toks})["content"] if toks else ""
            t = tasks[it["id"]]
            ok, sub, reason = score_one(t, text)
            exact += ok
            subset += sub
            items.append({"id": it["id"], "type": t["type"], "n_tokens": len(toks), "text": text,
                          "correct_exact": ok, "correct_subset": sub, "reason": reason})
        n = len(items)
        res[arm] = {"arm": arm, "n": n, "exact": exact, "subset": subset, "exact_pct": 100.0 * exact / n,
                    "by_type": {ty: sum(1 for x in items if x["type"] == ty and x["correct_exact"]) for ty in ("math", "json")},
                    "items": items}
        (Path(a.out) / f"{arm}.json").write_text(json.dumps(res[arm], indent=1))
    same = sum(1 for x, y in zip(res["ours"]["items"], res["llama"]["items"]) if x["text"] == y["text"])
    print(f"ours {res['ours']['exact']}/{res['ours']['n']} {res['ours']['by_type']}  llama {res['llama']['exact']}/{res['llama']['n']} {res['llama']['by_type']}  "
          f"delta_pp {res['ours']['exact_pct'] - res['llama']['exact_pct']:.1f}  identical outputs {same}/{res['ours']['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "ours", "llama", "score"])
    ap.add_argument("--tok-url")
    ap.add_argument("--url")
    ap.add_argument("--engine")
    ap.add_argument("--pack")
    ap.add_argument("--env", default="", help="the bake's baro.run.env, applied like tools/baro serve does")
    ap.add_argument("--tasks", default="bench/data/e8_tasks.json")
    ap.add_argument("--spec", type=int, default=0, help="1: our arm decodes with MTP spec (T=0 ids identical by the accept rule)")
    ap.add_argument("--limit", type=int, default=0, help="first N tasks only (quick gate)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"prep": prep, "ours": ours, "llama": llama, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
