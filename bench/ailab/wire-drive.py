#!/usr/bin/env python3
"""P4d harness: drive ONE engine process over its wire, one request in flight at a time, so a multi-session engine runs each request alone.

  wire-drive.py --engine BIN --units DIR --order FILE --creator UNIT [--hint 3569] --out DIR [--n 400] [--env K=V ...]

The creator request goes first with an explicit `ckpt` hint (default 3569, the shared system+tools prefix) and n = 1, then the units of --order
with `ckpt` []: both engines then restore the shared prefix from the checkpoint the SAME creator request made, instead of one made by whichever
burst request prefilled first with order-dependent hints. Writes OUT/<unit>.json ({"http","ids","cached","prefill_rows","call"}, the shape
parity_llama.py ownid-compare reads) and OUT/engine-echo.txt (the engine's own knob lines: the read-back). Greedy (no sampling fields), the
stop ids baro-serve sends. Exit 1 when the engine dies or a request times out."""
import argparse, hashlib, json, os, select, subprocess, sys, time

sys.dont_write_bytecode = True
STOP = [[248046], [248044]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--order", required=True)
    ap.add_argument("--creator", required=True)
    ap.add_argument("--hint", type=int, default=3569)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--timeout", type=int, default=240)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    env = dict(os.environ)
    env.update(kv.split("=", 1) for kv in a.env)
    p = subprocess.Popen([a.engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    fd, buf = p.stdout.fileno(), bytearray()
    echo = []

    def line(timeout):  # raw fd reads with our own line buffer: select on a buffered text stream blocks while whole lines sit in Python's buffer
        while b"\n" not in buf:
            if not select.select([fd], [], [], timeout)[0]:
                raise SystemExit(f"FAIL wire-drive: no output for {timeout} s from {a.engine}")
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                raise SystemExit(f"FAIL wire-drive: engine exited (rc {p.poll()}), last echo: {echo[-3:]}")
            buf.extend(chunk)
        i = buf.index(b"\n")
        s = bytes(buf[:i]).decode("utf-8", "replace").strip()
        del buf[: i + 1]
        return s

    def pending():
        return b"\n" in buf or bool(select.select([fd], [], [], 2)[0])
    try:
        while True:  # the ready line; every knob line before it is the engine's own read-back
            s = line(600)
            if s.startswith('{"ready"'):
                break
            if len(echo) < 400:
                echo.append(s[:200])
        ctrl = []
        try:  # the engine's real environment, read back from /proc (a daemon-leaked knob shows here, not in our --env list)
            raw = open(f"/proc/{p.pid}/environ", "rb").read().split(b"\0")
            keep = sorted(x.decode() for x in raw if x.split(b"=")[0].startswith((b"BARO_", b"MODULAR_", b"HSA", b"ROC", b"HIP")))
            open(f"{a.out}/engine-environ.txt", "w").write("\n".join(keep) + "\n")
        except OSError:
            open(f"{a.out}/engine-environ.txt", "w").write("unreadable\n")
        while len(ctrl) < 6 and pending():  # lines the engine prints right after ready (BARO_SEQS echo)
            ctrl.append(line(5)[:200])
        open(f"{a.out}/engine-echo.txt", "w").write("\n".join(echo + [s] + ctrl) + "\n")
        todo = [(a.creator, 1, [a.hint], True)] + [(u, a.n, [], False) for u in open(a.order).read().split()]
        wire = hashlib.sha256()
        for rid, (u, n, hints, creator) in enumerate(todo, 1):
            prompt = json.load(open(f"{a.units}/{u}.ref.json"))["prompt_ids"]
            req = {"id": rid, "prompt": prompt, "n": n, "spec": False, "stop": STOP, "ckpt": hints}
            wire.update((json.dumps(req) + "\n").encode())
            t0, ids, done, err = time.time(), [], None, None
            try:
                p.stdin.write((json.dumps(req) + "\n").encode())
                p.stdin.flush()
            except BrokenPipeError:
                pass  # the engine is gone: the read below reports it
            while done is None:
                s = line(a.timeout)
                if not s.startswith("{"):
                    continue
                try:
                    j = json.loads(s)
                except ValueError:
                    continue
                if j.get("id") != rid:
                    continue
                if "tok" in j:
                    ids.append(j["tok"])
                elif j.get("done"):
                    done = j
                elif "error" in j:
                    err, done = j["error"], {}
            rec = {"http": 200 if err is None else 500, "ids": ids, "cached": done.get("cached"), "prefill_rows": done.get("prefill_rows"),
                   "call": [], "secs": round(time.time() - t0, 2)}
            if err is not None:
                rec["error"] = str(err)[:200]
            if not creator:
                json.dump(rec, open(f"{a.out}/{u}.json", "w"))
            print(f"{'creator ' if creator else ''}{u}: {len(ids)} ids, cached {rec['cached']}, {rec['secs']} s" + (f", ERROR {err}" if err else ""), flush=True)
        open(f"{a.out}/wire.sha", "w").write(wire.hexdigest()[:16] + "\n")
    finally:
        try:
            p.stdin.close()
            p.wait(timeout=30)
        except Exception:
            p.kill()


if __name__ == "__main__":
    main()
