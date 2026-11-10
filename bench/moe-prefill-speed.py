#!/usr/bin/env python3
"""Driver for bench/moe-prefill-speed.sh. Spawns one engine process per
(length, arm, rep), speaks its stdin/stdout JSONL protocol directly (no HTTP
server involved), and times prefill from the harness clock.

Why one process per (length, arm, rep), each starting with a warmup request:
BARO_PREFILL is read once at engine startup (serve/engine.mojo), so replay
and batched arms cannot share a process; restarting for every rep also means
every timed measurement starts from a cold, freshly constructed checkpoint
Chain (serve/prefix.mojo Chain.__init__ allocates `self.items` empty every
time; nothing is persisted to packdir across processes -- packdir there only
seeds a hash salt), so cached==0 by construction regardless of run order.
BARO_CKPT=0 is passed anyway, belt and suspenders, because p32768.tokens is
an exact prefix of p8192.tokens (verified once, off the record) and a future
change to this script that reused one process across lengths would otherwise
silently hit the prefix-checkpoint cache and go VOID quietly instead of loudly.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

FAIL_WORD_RE = re.compile(r"NOT-RESIDENT|^Error|error:|Unhandled exception", re.IGNORECASE | re.MULTILINE)


def fail(step, reason, log=None):
    msg = f"FAIL {step}: {reason}" + (f", see {log}" if log else "")
    print(msg, file=sys.stderr)
    sys.exit(1)


def load_ids(path):
    with open(path) as f:
        return [int(x) for x in f.read().split()]


def build_schedule(lens, reps, reps_long):
    sched = {}
    for L in lens:
        r_reps = reps_long if L == max(lens) and reps_long < reps else reps
        b_reps = reps
        order, ri, bi = [], 0, 0
        while ri < r_reps or bi < b_reps:
            if ri < r_reps:
                order.append(("replay", ri)); ri += 1
            if bi < b_reps:
                order.append(("batched", bi)); bi += 1
        sched[L] = order
    return sched


def mode_env(mode):
    if mode == "tier":
        return {"BARO_TIER": "64", "BARO_TIER_PINNED": "1", "BARO_TIER_ZC": "1"}
    return {}


def run_one(engine, pack, mode, tmax, arm, length, rep, warmup_ids, prompt_ids, out_dir, arm_file, n_gen):
    tag = f"{mode}-{length}-{arm}-rep{rep}"
    log_path = os.path.join(out_dir, f"{tag}.log")
    env = os.environ.copy()
    env.update({
        "BARO_SERVE": "1", "BARO_MEGA": "0", "BARO_SPEC": "0", "BARO_CKPT": "0",
        "BARO_PACK": pack, "BARO_TMAX": str(tmax),
        "BARO_PREFILL": "1" if arm == "batched" else "0",
    })
    env.update(mode_env(mode))

    proc = subprocess.Popen([engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    logf = open(log_path, "w")

    def readline_logged():
        line = proc.stdout.readline()
        if line:
            logf.write(line)
            if FAIL_WORD_RE.search(line):
                logf.flush()
                fail(tag, f"fail word in engine output: {line.strip()}", log_path)
        return line

    ready = None
    pf_readback = None
    while True:
        line = readline_logged()
        if line == "":
            logf.flush()
            fail(tag, "engine exited before printing a ready line", log_path)
        m = re.match(r"BARO_PREFILL:\s*(True|False)", line)
        if m:
            pf_readback = m.group(1) == "True"
        if line.startswith('{"ready":'):
            try:
                ready = json.loads(line)
            except json.JSONDecodeError:
                fail(tag, f"ready line did not parse as JSON: {line.strip()}", log_path)
            break

    with open(arm_file, "a") as af:
        af.write(f"{tag} tmax readback: {ready.get('tmax')} (expected {tmax})\n")
    if ready.get("tmax") != tmax:
        fail(tag, f"tmax readback {ready.get('tmax')} != expected {tmax}, VOID", log_path)
    if pf_readback is None:
        fail(tag, "no BARO_PREFILL: readback line before ready line", log_path)
    want_pf = (arm == "batched")
    if pf_readback != want_pf:
        fail(tag, f"BARO_PREFILL readback {pf_readback} != expected {want_pf}, VOID", log_path)

    def send_and_wait_done(req_id, ids, n, collect_toks):
        req = json.dumps({"id": req_id, "prompt": ids, "n": n, "spec": False})
        write_t = time.monotonic()
        proc.stdin.write(req + "\n")
        proc.stdin.flush()
        first_tok_t = None
        toks = []
        done = None
        while True:
            line = readline_logged()
            if line == "":
                logf.flush()
                fail(tag, f"engine exited before a done line for id {req_id}", log_path)
            if line.startswith('{"id":') and '"tok":' in line:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("id") == req_id:
                    if first_tok_t is None:
                        first_tok_t = time.monotonic()
                    if collect_toks:
                        toks.append(d["tok"])
            elif line.startswith('{"id":') and '"done":true' in line:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("id") == req_id:
                    done = d
                    break
        return write_t, first_tok_t, toks, done

    # Warmup: discarded by rule.
    send_and_wait_done(1, warmup_ids, n_gen, collect_toks=False)

    # Timed request.
    write_t, first_tok_t, toks, done = send_and_wait_done(2, prompt_ids, n_gen, collect_toks=True)
    if first_tok_t is None:
        fail(tag, "no tok line arrived for the timed request", log_path)
    if len(toks) != n_gen:
        fail(tag, f"got {len(toks)} tokens for the timed request, expected {n_gen}", log_path)

    proc.stdin.close()
    try:
        rc = proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        fail(tag, "engine did not exit within 120s of stdin EOF", log_path)
    logf.close()
    if rc != 0:
        fail(tag, f"engine exited {rc}", log_path)

    cached = done.get("cached")
    prefill_rows = done.get("prefill_rows")
    if cached != 0:
        fail(tag, f"cached={cached} on a timed request, VOID (prefix-checkpoint hit)", log_path)
    if arm == "batched" and not (prefill_rows and prefill_rows > 0):
        fail(tag, f"batched arm has prefill_rows={prefill_rows}, VOID", log_path)
    if arm == "replay" and prefill_rows != 0 and done.get("pf_rows", 0) != 0:
        fail(tag, f"replay arm has prefill_rows={prefill_rows} (expected 0), VOID", log_path)

    harness_tok_s = (len(prompt_ids) - 1) / (first_tok_t - write_t)
    return {
        "toks": toks, "harness_tok_s": harness_tok_s,
        "engine_prefill_s": done.get("prefill_s"), "prefill_rows": prefill_rows, "cached": cached,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", required=True, choices=["resident", "tier"])
    ap.add_argument("--tmax", type=int, required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--reps-replay-long", type=int, required=True)
    ap.add_argument("--warmup", required=True)
    ap.add_argument("--prompts-dir", required=True)
    ap.add_argument("--arm-file", required=True)
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()

    lens = [int(x) for x in args.lens.split()]
    warmup_ids = load_ids(args.warmup)
    prompt_ids = {L: load_ids(os.path.join(args.prompts_dir, f"p{L}.tokens")) for L in lens}
    for L in lens:
        if len(prompt_ids[L]) != L:
            fail("args", f"p{L}.tokens has {len(prompt_ids[L])} tokens, expected {L}")

    schedule = build_schedule(lens, args.reps, args.reps_replay_long)

    summary_path = os.path.join(args.out, "summary.tsv")
    cells = {}  # (len, arm) -> [harness_tok_s, ...]
    first_replay_toks = {}  # len -> [tok, ...]

    with open(summary_path, "w") as sf:
        sf.write("mode\tlen\tarm\trep\tharness_tok_s\tengine_prefill_s\tprefill_rows\tcached\n")
        for L in lens:
            for arm, rep in schedule[L]:
                r = run_one(args.engine, args.pack, args.mode, args.tmax, arm, L, rep,
                            warmup_ids, prompt_ids[L], args.out, args.arm_file, args.n)
                if arm == "replay" and L not in first_replay_toks:
                    first_replay_toks[L] = r["toks"]
                if arm == "batched":
                    ref = first_replay_toks.get(L)
                    if ref is None:
                        fail(f"{args.mode}-{L}-batched-rep{rep}",
                             "no replay run for this length ran before this batched run, cannot check identity")
                    if ref != r["toks"]:
                        div = next(i for i in range(len(ref)) if ref[i] != r["toks"][i])
                        fail(f"{args.mode}-{L}-batched-rep{rep}",
                             f"token identity broke at index {div}: replay={ref[div]} batched={r['toks'][div]}")
                sf.write(f"{args.mode}\t{L}\t{arm}\t{rep}\t{r['harness_tok_s']:.3f}\t"
                         f"{r['engine_prefill_s']}\t{r['prefill_rows']}\t{r['cached']}\n")
                sf.flush()
                cells.setdefault((L, arm), []).append(r["harness_tok_s"])
                print(f"{args.mode} len={L} arm={arm} rep={rep}: "
                      f"harness {r['harness_tok_s']:.1f} tok/s, engine prefill_s={r['engine_prefill_s']}")

    medians_path = os.path.join(args.out, "medians.tsv")
    with open(medians_path, "w") as mf:
        mf.write("len\tarm\tmedian_tok_s\tmin_tok_s\tmax_tok_s\treps\n")
        for (L, arm), vals in sorted(cells.items()):
            mf.write(f"{L}\t{arm}\t{statistics.median(vals):.3f}\t{min(vals):.3f}\t{max(vals):.3f}\t{len(vals)}\n")
        mf.write("len\tratio_batched_over_replay_median\n")
        for L in lens:
            rv = cells.get((L, "replay")); bv = cells.get((L, "batched"))
            if rv and bv:
                ratio = statistics.median(bv) / statistics.median(rv)
                mf.write(f"{L}\t{ratio:.3f}\n")
    print(open(medians_path).read())


if __name__ == "__main__":
    main()
