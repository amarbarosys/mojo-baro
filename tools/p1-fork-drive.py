#!/usr/bin/env python3
"""HTTP driver for bench/p1-fork-gate.sh (P1 gate 2, identity half). Subcommands:

  refs    OUT A_URL N                    cold single-node ids on A, then control S (A restores its own checkpoint)
  forks   OUT A_URL TARGET PROFILE N     N forks A -> TARGET, answered by node B
  falsify OUT A_URL FLIP_T SWAP_T N      N forks through the flip port (must 409) then the swapkv port
  rate    OUT SINK_IP SINK_PORT PROFILE  64 MiB into the forwarder's sink, timed at the receiver

Every call writes JSON under OUT and exits non-zero with a FAIL line on a transport error. It judges
nothing: tools/p1-fork-score.py applies the frozen rules."""
import glob
import json
import socket
import sys
import time
import urllib.error
import urllib.request

BRANCH = {"max_tokens": 32, "temperature": 0, "spec": False}


def fail(step, why):
    print(f"FAIL {step}: {why}")
    sys.exit(1)


def call(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw or b"{}")
        except ValueError:
            return e.code, {"raw": raw[:300].decode("latin1")}
    except OSError as e:
        fail("http", f"{base}{path}: {e}")


def prompts(n):
    out = []
    for f in sorted(glob.glob("bench/mtp-prompts/p*.tokens"))[:n]:
        out.append((f.split("/")[-1][:-7], [int(x) for x in open(f).read().split()]))
    if len(out) != n:
        fail("prompts", f"wanted {n}, found {len(out)}")
    return out


def branch_of(resp):
    b = resp["branches"][0]
    return b["tokens"], b.get("timings", {}).get("cached")


def main():
    cmd, out = sys.argv[1], sys.argv[2]
    if cmd == "refs":
        a, n, rows = sys.argv[3], int(sys.argv[4]), {}
        for name, ids in prompts(n):
            s, cold = call(a, "/v1/fork", {"prompt": ids, "branches": [BRANCH]})
            if s != 200:
                fail("refs", f"{name} cold HTTP {s}: {json.dumps(cold)[:200]}")
            s, again = call(a, "/v1/fork", {"prompt": ids, "branches": [BRANCH]})
            if s != 200:
                fail("refs", f"{name} control S HTTP {s}")
            ct, cc = branch_of(cold)
            st, sc = branch_of(again)
            rows[name] = {"n_prompt": len(ids), "cold": ct, "cold_cached": cc, "ctrlS": st, "ctrlS_cached": sc}
            print(f"ref {name}: |P|={len(ids)} cold_cached={cc} ctrlS_cached={sc} ctrlS_equals_cold={ct == st}", flush=True)
        json.dump(rows, open(f"{out}/refs.json", "w"))
    elif cmd == "forks":
        a, target, profile, n, rows = sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), {}
        for name, ids in prompts(n):
            t0 = time.monotonic()
            s, fk = call(a, "/v1/fork", {"prompt": ids, "target": target, "branches": [BRANCH]})
            wall = time.monotonic() - t0
            if s != 200:
                rows[name] = {"http": s, "body": fk}
                print(f"fork {profile} {name}: HTTP {s} {json.dumps(fk)[:200]}", flush=True)
                continue
            toks, cached = branch_of(fk)
            tg = fk.get("target", {})
            rows[name] = {"http": 200, "tokens": toks, "cached": cached, "pos": tg.get("pos"), "format": tg.get("format"),
                          "state_bytes": tg.get("state_bytes"), "export_s": tg.get("export_s"), "import_s": tg.get("import_s"),
                          "answer_s": tg.get("answer_s"), "wall_s": wall, "runtime_differs": tg.get("import", {}).get("runtime_differs")}
            print(f"fork {profile} {name}: pos={tg.get('pos')} B.cached={cached} format={tg.get('format')} bytes={tg.get('state_bytes')} import_s={tg.get('import_s', 0):.3f} wall_s={wall:.3f}", flush=True)
        json.dump(rows, open(f"{out}/forks-{profile}.json", "w"))
    elif cmd == "falsify":
        a, flip_t, swap_t, n, rows = sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), {}
        for name, ids in prompts(n):
            s, fl = call(a, "/v1/fork", {"prompt": ids, "target": flip_t, "branches": [BRANCH]})
            rows[name] = {"flip_http": s, "flip_body": fl}
            print(f"falsify flip {name}: HTTP {s} {json.dumps(fl)[:160]}", flush=True)
        # the flipped states restored nothing, so node B is still cold for these prompts
        for name, ids in prompts(n):
            s, sw = call(a, "/v1/fork", {"prompt": ids, "target": swap_t, "branches": [BRANCH]})
            if s == 200:
                toks, cached = branch_of(sw)
                rows[name].update({"swap_http": 200, "swap_tokens": toks, "swap_cached": cached, "swap_pos": sw.get("target", {}).get("pos")})
            else:
                rows[name].update({"swap_http": s, "swap_body": sw})
            print(f"falsify swapkv {name}: HTTP {s} cached={rows[name].get('swap_cached')}", flush=True)
        json.dump(rows, open(f"{out}/falsify.json", "w"))
    elif cmd == "rate":
        ip, port, profile = sys.argv[3], int(sys.argv[4]), sys.argv[5]
        n = 64 << 20
        try:
            s = socket.create_connection((ip, port), timeout=10)
            s.settimeout(300)
            s.sendall(bytes(n))
            s.shutdown(socket.SHUT_WR)
            got, secs = s.recv(200).decode().split()
        except OSError as e:
            fail("rate", f"{ip}:{port}: {e}")
        mbit = int(got) * 8 / float(secs) / 1e6
        json.dump({"profile": profile, "bytes": int(got), "seconds": float(secs), "mbit_s": mbit}, open(f"{out}/rate-{profile}.json", "w"))
        print(f"rate {profile}: {int(got)} bytes in {float(secs):.3f} s at the receiver = {mbit:.0f} Mbit/s", flush=True)
        if int(got) != n:
            fail("rate", f"sink saw {got} of {n} bytes")
    else:
        fail("usage", f"unknown subcommand {cmd}")


if __name__ == "__main__":
    main()
