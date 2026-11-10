#!/usr/bin/env python3
"""P5/P6 oracle driver: N bodies per step through street-sim's own ask_openai, real HTTP to one server.

  bench/ailab/step.py URL N STEPS OUT [--states DIR] [--warm K] [--model NAME] [--pid ENGINE_PID] [--offset K] [--expect-cached C [--expect-from-step S]]

Each step starts N threads at once (lockstep, as ailab_run.py does); body b of step s takes state (K + s*N + b) % len(states), K = --offset (0)
from bench/ailab/states-tp (a fresh state per body per step), builds its prompt with the runner's own `ask`, and runs the
runner's `ask_openai` unmodified (street-sim imported read-only through capture.py). Two seams only: the game bridge is
capture.install_bridge (canned replies, no socket) and urllib.request.urlopen is wrapped so every POST (a) gets its
max_tokens raised to 16000 (repo floor; answers are 34 to 60 tokens, the cap does not bind) and (b) is recorded with its
HTTP status, seconds and usage. A non-200 the runner swallows (it re-asks WITHOUT tools on an HTTPError) still shows in
posts.tsv. Step wall = the slowest body's ask.
Files in OUT: posts.tsv (step body state round code secs prompt_tok completion_tok prefill_s decode_s cached prefill_rows err: the server's own timers, err = HTTP error body),
rounds.tsv (per round of the tool loop: mean secs, prefill_s, decode_s, tokens), asks.tsv (step body state rounds secs
err), steps.tsv (step wall_s asks posts errors warm sum_prefill_s sum_decode_s), vram.tsv (step engine_vram_bytes global_used_bytes; engine = sum of the drm-client VRAM in /proc/PID/fdinfo), summary.txt (one line, also printed).
The first --warm steps run but are left out of the summary. Env STUB_VRAM replaces the sysfs VRAM read (CPU dry runs), STUB_VRAM_GROW adds that many bytes per step.
Exit 3 = --expect-cached failed (first-round requests from step S restored fewer than C cached tokens). Exit 0 = ran every step; the verdict (errors, 429, p95) belongs to the caller (step.sh).
"""
import glob, io, json, os, sys, threading, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import capture as C  # noqa: E402  (also puts street-sim on sys.path, read only)

R = C.R
local = threading.local()
rec_lock = threading.Lock()
posts = []


def proc_vram(pid):
    """VRAM bytes the process holds on amdgpu: per drm-client-id (dup'd fds share one) the first of drm-resident-vram,
    drm-memory-vram, drm-total-vram found, summed over clients. -1 when the pid is gone or reports no VRAM key."""
    units = {"KiB": 1024, "MiB": 1 << 20, "GiB": 1 << 30, "B": 1}
    clients = {}
    try:
        names = os.listdir(f"/proc/{pid}/fdinfo")
    except OSError:
        return -1
    for n in names:
        try:
            txt = open(f"/proc/{pid}/fdinfo/{n}").read()
        except OSError:
            continue
        if "drm-driver:\tamdgpu" not in txt:
            continue
        kv = {}
        for line in txt.splitlines():
            k, _, v = line.partition(":")
            kv[k.strip()] = v.strip()
        for key in ("drm-resident-vram", "drm-memory-vram", "drm-total-vram"):
            if key in kv:
                num, _, unit = kv[key].partition(" ")
                clients[kv.get("drm-client-id", n)] = int(num) * units.get(unit or "B", 1)
                break
    return sum(clients.values()) if clients else -1


def vram(step=0, pid=0):
    """(engine bytes, global used bytes). STUB_VRAM replaces both reads (CPU dry runs): STUB_VRAM_GROW adds bytes per step to
    both, STUB_VRAM_JITTER +-bytes alternating per step, STUB_GLOBAL_GROW only to the global one (a desktop that grows while the engine does not): negative controls."""
    if os.environ.get("STUB_VRAM"):
        e = int(os.environ["STUB_VRAM"]) + step * int(os.environ.get("STUB_VRAM_GROW", "0"))
        e += int(os.environ.get("STUB_VRAM_JITTER", "0")) * (1 if step % 2 == 0 else -1)  # alternating noise: the resolution negative control
        return e, e + step * int(os.environ.get("STUB_GLOBAL_GROW", "0"))
    v = [int(open(p).read()) for p in glob.glob("/sys/class/drm/card*/device/mem_info_vram_used")]
    return (proc_vram(pid) if pid else -1), (max(v) if v else -1)


def note_post(code, secs, ptok, ctok, pre=0.0, dec=0.0, cached=0, rows=0, err=""):
    with rec_lock:
        posts.append((local.step, local.body, local.state, local.round, code, secs, ptok, ctok, pre, dec, cached, rows, err))
    local.round += 1


def shim(real):
    def urlopen(req, timeout=None):
        req.data = C.floor_body(req.data)
        t0 = time.time()
        try:
            raw = real(req, timeout=timeout).read()
        except urllib.error.HTTPError as e:
            try:  # the runner swallows this (it re-asks WITHOUT tools); the body says why (e.g. exceed_context)
                why = e.read().decode("utf-8", "replace")[:160].replace("\t", " ").replace("\n", " ")
            except Exception:
                why = ""
            note_post(e.code, round(time.time() - t0, 3), 0, 0, err=why)
            raise
        except Exception:
            note_post("EXC", round(time.time() - t0, 3), 0, 0)
            raise
        resp = json.loads(raw)
        use, tm = resp.get("usage") or {}, resp.get("timings") or {}
        if "prefill_s" in tm:  # baro-serve: the engine's own prefill / decode seconds for this request
            pre, dec, cached, rows = tm.get("prefill_s") or 0.0, tm.get("decode_s") or 0.0, tm.get("cached") or 0, tm.get("prefill_rows") or 0
        else:  # llama-server timings
            pre, dec, cached, rows = (tm.get("prompt_ms") or 0.0) / 1000, (tm.get("predicted_ms") or 0.0) / 1000, tm.get("cache_n") or 0, tm.get("prompt_n") or 0
        note_post(200, round(time.time() - t0, 3), use.get("prompt_tokens", 0), use.get("completion_tokens", 0), round(pre, 4), round(dec, 4), cached, rows)
        return io.BytesIO(raw)
    return urlopen


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else float("nan")


def main():
    a = sys.argv[1:]
    if len(a) < 4:
        sys.exit(__doc__)
    url, n, steps, out = a[0].rstrip("/"), int(a[1]), int(a[2]), a[3]
    sd, warm, model, pid, exp_cached, exp_from, off = f"{HERE}/states-tp", 0, "local", 0, 0, 0, 0
    i = 4
    while i < len(a):
        if a[i] == "--states":
            sd, i = a[i + 1], i + 2
        elif a[i] == "--warm":
            warm, i = int(a[i + 1]), i + 2
        elif a[i] == "--model":
            model, i = a[i + 1], i + 2
        elif a[i] == "--pid":
            pid, i = int(a[i + 1]), i + 2
        elif a[i] == "--offset":
            off, i = int(a[i + 1]), i + 2
        elif a[i] == "--expect-cached":
            exp_cached, i = int(a[i + 1]), i + 2
        elif a[i] == "--expect-from-step":
            exp_from, i = int(a[i + 1]), i + 2
        else:
            sys.exit(f"FAIL args: unknown {a[i]}")
    items = C.load_states(sd)
    os.makedirs(out, exist_ok=True)
    ports = [7910 + b for b in range(n)]
    state_of = {}
    C.install_bridge(state_of)
    urllib.request.urlopen = shim(urllib.request.urlopen)
    homes = f"{C.REPO}/.work/ailab/step-home"
    asks, stepfile, vfile = [], open(f"{out}/steps.tsv", "w"), open(f"{out}/vram.tsv", "w")
    for s in range(steps):
        res = [None] * n

        def body(b):
            item = items[(off + s * n + b) % len(items)]
            sid, name, back, st, diary = item
            k = int(sid[1:])
            state = C.runner_state(k, st, diary)
            h = f"{homes}/b{b}"
            C.fake_home(h, diary, [diary[0]] if k % 2 else [])
            state_of[ports[b]] = state
            local.step, local.body, local.state, local.round = s, b, sid, 1
            t0, err = time.time(), ""
            try:
                R.ask({"name": name, "backend": "openai", "url": url, "model": model, "backstory": back}, h, ports[b], state, False)
            except Exception as e:  # the runner logs a failed ask and idles the body
                err = f"{type(e).__name__}: {e}"[:120].replace("\t", " ").replace("\n", " ")
            res[b] = (s, b, sid, local.round - 1, round(time.time() - t0, 3), err)
        ts = [threading.Thread(target=body, args=(b,)) for b in range(n)]
        t0 = time.time()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        wall = round(time.time() - t0, 3)
        asks += res
        errs = sum(1 for r in res if r[5])
        sp = [p for p in posts if p[0] == s]
        stepfile.write(f"{s}\t{wall}\t{len(res)}\t{sum(r[3] for r in res)}\t{errs}\t{int(s < warm)}\t{sum(p[8] for p in sp):.3f}\t{sum(p[9] for p in sp):.3f}\n")
        stepfile.flush()
        vfile.write("%d\t%d\t%d\n" % ((s,) + vram(s, pid)))
        vfile.flush()
        print(f"step {s}: wall {wall} s, {n} asks, {errs} errors", flush=True)
    with open(f"{out}/posts.tsv", "w") as f:
        for p in posts:
            f.write("\t".join(str(x) for x in p) + "\n")
    with open(f"{out}/asks.tsv", "w") as f:
        for r in asks:
            f.write("\t".join(str(x) for x in r) + "\n")
    with open(f"{out}/rounds.tsv", "w") as f:  # where the time goes: per round of the tool loop, server-side timers
        f.write("round\tposts\tmean_secs\tmean_prefill_s\tmean_decode_s\tmean_prompt_tok\tmean_completion_tok\tmean_cached\tmean_prefill_rows\n")
        for k in sorted({p[3] for p in posts if p[0] >= warm}):
            g = [p for p in posts if p[0] >= warm and p[3] == k]
            m = lambda i: sum(p[i] for p in g) / len(g)  # noqa: E731
            f.write(f"{k}\t{len(g)}\t{m(5):.3f}\t{m(8):.3f}\t{m(9):.3f}\t{m(6):.0f}\t{m(7):.1f}\t{m(10):.0f}\t{m(11):.0f}\n")
    keep = [r for r in asks if r[0] >= warm]
    walls = [float(l.split("\t")[1]) for l in open(f"{out}/steps.tsv") if int(l.split("\t")[5]) == 0]
    pk = [p for p in posts if p[0] >= warm]
    code = lambda c: sum(1 for p in pk if str(p[4]) == c)  # noqa: E731
    n5 = sum(1 for p in pk if str(p[4]).isdigit() and 500 <= int(p[4]))
    n4 = sum(1 for p in pk if str(p[4]).isdigit() and 400 <= int(p[4]) < 500 and int(p[4]) != 429)
    summ = (f"N={n} steps={steps - warm} step_median={pct(walls, 0.5):.3f} step_max={max(walls):.3f} ask_p50={pct([r[4] for r in keep], 0.5):.3f} "
            f"ask_p95={pct([r[4] for r in keep], 0.95):.3f} asks={len(keep)} posts={len(pk)} rounds_mean={sum(r[3] for r in keep) / max(1, len(keep)):.2f} "
            f"errors={sum(1 for r in keep if r[5])} http429={code('429')} http4xx={n4} http5xx={n5} exc={code('EXC')} "
            f"completion_tok_mean={sum(p[7] for p in pk) / max(1, len(pk)):.1f}")
    open(f"{out}/summary.txt", "w").write(summ + "\n")
    print(summ)
    if exp_cached:  # the shared system+tools prefix must be restored from a checkpoint, not prefilled again: a cold burst overflows the KV pool
        low = [p for p in posts if p[0] >= exp_from and p[3] == 1 and p[4] == 200 and p[10] < exp_cached]
        if low:
            print(f"FAIL cached: {len(low)} of {len([p for p in posts if p[0] >= exp_from and p[3] == 1])} first-round requests from step {exp_from} restored fewer than {exp_cached} tokens "
                  f"(cached {sorted(p[10] for p in low)[:8]}); the shared prefix checkpoint is missing", flush=True)
            sys.exit(3)
        print(f"cached ok: every first-round request from step {exp_from} restored >= {exp_cached} tokens")


if __name__ == "__main__":
    main()
