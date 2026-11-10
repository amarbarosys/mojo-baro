#!/usr/bin/env python3
"""One q4x run with a 100 ms GTT sampler on the same clock as the engine's stdout lines.

usage: q4x-gtt-trace.py OUT.jsonl BIN GGUF N_TOK PROMPT.json [--tick-ms 100]
Q4X_* and MODULAR_* environment passes through to the engine unchanged. Exit code is the engine's.

Records (JSON lines, t = time.monotonic_ns of this process):
  {"k":"s","t":..,"tb":CLOCK_BOOTTIME ns,"tr":CLOCK_MONOTONIC_RAW ns,"gtt":bytes,"vram":bytes,"eng":{"gtt":KiB,"vram":KiB,"rss":kB,"lck":kB,"rd":bytes read from disk,"maj":major faults},
   "cl":[[pid,comm,gtt_KiB],..],"mem":{meminfo kB},"vm":{vmstat counters},"psi":[some_us,full_us],"nvme":sectors read}   one per tick
  {"k":"l","t":..,"line":".."}   one per engine output line
  {"k":"x","t":..,"rc":..}        engine exit
"cl" lists every other process holding an amdgpu fd with at least 64 MiB of GTT, so a spike gets an owner.
"""
import json, os, subprocess, sys, threading, time


def card():
    for d in sorted(os.listdir("/sys/class/drm")):
        f = f"/sys/class/drm/{d}/device/mem_info_vram_total"
        if d.startswith("card") and "-" not in d and os.path.exists(f) and int(open(f).read()) > 20_000_000_000:
            return f"/sys/class/drm/{d}/device/"
    sys.exit("FAIL gtt-trace: no card with more than 20 GB of VRAM")


MEM = ("MemAvailable", "Cached", "Mapped", "Active(file)", "Inactive(file)", "Dirty", "AnonPages", "Mlocked")
VM = ("pgmajfault", "pgfault", "pgscan_kswapd", "pgscan_direct", "pgsteal_kswapd", "pgsteal_direct", "pgmigrate_success", "pgmigrate_fail",
      "compact_stall", "compact_migrate_scanned", "numa_pages_migrated", "workingset_refault_file", "allocstall_normal", "thp_fault_alloc")


def kv(path, keys):
    d = {}
    for ln in open(path):
        w = ln.replace(":", " ").split()
        if w and w[0] in keys:
            d[w[0]] = int(w[1])
    return d


def psi():
    return [int(ln.split("total=")[1]) for ln in open("/proc/pressure/memory")]


def nvme_sectors():
    for ln in open("/proc/diskstats"):
        w = ln.split()
        if w[2] == "nvme0n1":
            return int(w[5])


def clients(pids):
    res = {}
    for pid in pids:
        seen = {}
        try:
            names = os.listdir(f"/proc/{pid}/fdinfo")
        except OSError:
            continue
        for n in names:
            try:
                txt = open(f"/proc/{pid}/fdinfo/{n}").read()
            except OSError:
                continue
            if "drm-driver:\tamdgpu" not in txt:
                continue
            f = {ln.split(":")[0]: ln.split()[1] for ln in txt.splitlines() if ln.startswith("drm-") and len(ln.split()) > 1}
            seen.setdefault(f.get("drm-client-id"), (int(f.get("drm-memory-gtt", 0)), int(f.get("drm-memory-vram", 0))))
        if seen:
            res[pid] = (sum(v[0] for v in seen.values()), sum(v[1] for v in seen.values()))
    return res


def main():
    out, binp, gguf, ntok, prompt = sys.argv[1:6]
    tick = float(sys.argv[sys.argv.index("--tick-ms") + 1]) / 1000.0 if "--tick-ms" in sys.argv else 0.1
    dev = card()
    rd = lambda n: int(open(dev + n).read())
    ids = [str(i) for i in json.load(open(prompt))]
    fh, lock, stop = open(out, "w"), threading.Lock(), threading.Event()

    def put(rec):
        with lock:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

    proc = subprocess.Popen([binp, gguf, ntok] + ids, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def reader():
        for line in proc.stdout:
            put({"k": "l", "t": time.monotonic_ns(), "line": line.rstrip("\n")})
        put({"k": "x", "t": time.monotonic_ns(), "rc": proc.wait()})
        stop.set()

    threading.Thread(target=reader, daemon=True).start()
    pids, rescan = [], 0.0
    while not stop.is_set():
        t0 = time.monotonic()
        if t0 >= rescan:
            pids, rescan = [int(p) for p in os.listdir("/proc") if p.isdigit()], t0 + 1.0
        cl, eng = clients(pids), {}
        if proc.pid in cl:
            eng = {"gtt": cl[proc.pid][0], "vram": cl[proc.pid][1]}
        try:
            st = {ln.split(":")[0]: int(ln.split()[1]) for ln in open(f"/proc/{proc.pid}/status") if ln.startswith(("VmRSS:", "VmLck:"))}
            eng["rss"], eng["lck"] = st.get("VmRSS", 0), st.get("VmLck", 0)
            io = kv(f"/proc/{proc.pid}/io", ("read_bytes",))
            eng["rd"], eng["maj"] = io.get("read_bytes", 0), int(open(f"/proc/{proc.pid}/stat").read().rsplit(")", 1)[1].split()[9])
        except (OSError, IndexError):
            pass
        oth = [[p, open(f"/proc/{p}/comm").read().strip(), v[0]] for p, v in cl.items() if p != proc.pid and v[0] >= 65536]
        put({"k": "s", "t": time.monotonic_ns(), "tb": time.clock_gettime_ns(time.CLOCK_BOOTTIME), "tr": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), "gtt": rd("mem_info_gtt_used"), "vram": rd("mem_info_vram_used"), "eng": eng, "cl": oth,
             "mem": kv("/proc/meminfo", MEM), "vm": kv("/proc/vmstat", VM), "psi": psi(), "nvme": nvme_sectors()})
        time.sleep(max(0.0, tick - (time.monotonic() - t0)))
    fh.close()
    sys.exit(proc.returncode or 0)


main()
