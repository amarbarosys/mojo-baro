#!/usr/bin/env python3
"""bench/ruler/run.py: drive an OpenAI-compatible /v1/chat/completions endpoint
over bench/ruler/gen.py's prompt sets and save every raw response.

Both llama-server and our future baro-serve speak this surface
(serve/PROTOCOL.md "HTTP surface"). Concurrency --workers (default 1; match
the server's -np), temperature 0, resumable
(skips ids whose metrics record already exists -- rerun after a crash/kill
picks up where it left off).

Usage: bench/ruler/run.py --base-url http://127.0.0.1:PORT/v1 --model NAME
       [--prompts DIR] [--out DIR] [--tasks t1,t2] [--sizes 4096,...]
       [--max-tokens N] [--stream] [--workers K]

Writes OUTDIR/<task>_<size>/<id>.raw.json (full API response),
OUTDIR/<task>_<size>/<id>.response.txt (completion text used by score.py),
and appends one line per request to OUTDIR/<task>_<size>/metrics.jsonl:
  {"id","prompt_tokens","wall_ms","ttft_ms"}
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen import SIZES, TASKS, TOKENS_TO_GENERATE  # noqa: E402


def run_one(base_url, model, prompt, max_tokens, stream):
    url = f"{base_url.rstrip('/')}/chat/completions"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": stream}
    t0 = time.monotonic()
    ttft_ms = None
    if stream:
        text_parts, raw_chunks = [], []
        with requests.post(url, json=body, stream=True, timeout=3600) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                if ttft_ms is None:
                    ttft_ms = (time.monotonic() - t0) * 1000
                chunk = json.loads(payload)
                raw_chunks.append(chunk)
                delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
                if delta:
                    text_parts.append(delta)
        wall_ms = (time.monotonic() - t0) * 1000
        text = "".join(text_parts)
        usage = next((c["usage"] for c in reversed(raw_chunks) if c.get("usage")), None)
        prompt_tokens = usage.get("prompt_tokens") if usage else None
        raw = {"chunks": raw_chunks}
    else:
        r = requests.post(url, json=body, timeout=3600)
        r.raise_for_status()
        wall_ms = (time.monotonic() - t0) * 1000
        raw = r.json()
        message = (raw.get("choices") or [{}])[0].get("message", {})
        # reasoning models (this lane's baseline included) emit a separate
        # reasoning_content block before the answer; when max_tokens cuts
        # generation mid-thought, "content" is empty and the answer -- if
        # generated at all -- is inside reasoning_content. Score both.
        text = (message.get("reasoning_content") or "") + "\n" + (message.get("content") or "")
        usage = raw.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        timings = raw.get("timings")  # llama-server extension
        if timings and prompt_tokens is None:
            prompt_tokens = timings.get("prompt_n")
    return text, raw, prompt_tokens, wall_ms, ttft_ms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default=str(Path(__file__).resolve().parent / "prompts"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "responses"))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N examples per (task,size) file")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="concurrent requests (= server -np)")
    a = ap.parse_args()

    prompts_dir, out_dir = Path(a.prompts), Path(a.out)
    tasks = a.tasks.split(",")
    sizes = [int(s) for s in a.sizes.split(",")]

    for task in tasks:
        for size in sizes:
            src = prompts_dir / f"{task}_{size}.jsonl"
            if not src.exists():
                continue
            dst = out_dir / f"{task}_{size}"
            dst.mkdir(parents=True, exist_ok=True)
            metrics_path = dst / "metrics.jsonl"
            done = set()
            if metrics_path.exists():
                for line in metrics_path.read_text().splitlines():
                    if line.strip():
                        done.add(json.loads(line)["id"])
            max_tokens = a.max_tokens or (TOKENS_TO_GENERATE.get(task, 128) + 32)
            rows = []
            with open(src) as f:
                for i, line in enumerate(f):
                    if a.limit is not None and i >= a.limit:
                        break
                    row = json.loads(line)
                    if row["id"] not in done:
                        rows.append(row)

            def work(row):
                text, raw, prompt_tokens, wall_ms, ttft_ms = run_one(
                    a.base_url, a.model, row["prompt"], max_tokens, a.stream)
                (dst / f"{row['id']}.response.txt").write_text(text)
                (dst / f"{row['id']}.raw.json").write_text(json.dumps(raw, ensure_ascii=False))
                return {"id": row["id"], "prompt_tokens": prompt_tokens,
                        "wall_ms": round(wall_ms, 1),
                        "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None}

            # metrics written from the main thread only, in completion order
            with open(metrics_path, "a") as mf, ThreadPoolExecutor(a.workers) as ex:
                for rec in ex.map(work, rows):
                    mf.write(json.dumps(rec) + "\n")
                    mf.flush()
                    print(f"{rec['id']}: prompt_tokens={rec['prompt_tokens']} wall_ms={rec['wall_ms']}")

if __name__ == "__main__":
    main()
