"""Per-arm table and excursion-versus-normal counters for a bench/q4x-gtt-trace.sh output directory.

usage: q4x-gtt-report OUT_DIR [--spike-gb 1]
build: mojo build tools/q4x-gtt-report.mojo -I . -I ~/iTools/lib/mojo -o .work/q4x-gtt-report
Reads OUT_DIR/run-NN[-ARM]-PROMPT.jsonl (the records of tools/q4x-gtt-trace.py). A spike is a run whose peak GTT exceeds its first sample by more than --spike-gb.
Prints one row per arm (runs, spikes, peak, cache slots, decode and prefill ms/token and expert misses, split into clean and spike runs), then per-second counter
rates on ticks inside an excursion against the other ticks, and a final line "spikes arm X: k of n" per arm.
"""
from std.math import nan
from std.os import listdir
from std.sys import argv
from grammar.json_value import JSONParser, JSONDoc
from pyformat import fmt_fixed, py_sum

comptime G = 1 << 30
comptime NK = 4
comptime KEY_NAMES = ["direct scan pages/s", "alloc stalls/s", "compaction scan/s", "page migrations/s", "memory PSI some ms/s", "MemAvailable GB"]


@fieldwise_init
struct Sample(Copyable, Movable):
    var t: Int
    var gtt: Int
    var avail_kb: Int
    var vm: List[Int]
    var psi: Int
    var has_vm: Bool


struct Run(Copyable, Movable):
    var arm: String
    var samples: List[Sample]
    var gen: List[Float64]
    var pf: List[Float64]
    var mis: List[Float64]
    var slots: List[Float64]
    var peak: Int
    var spike: Bool

    def __init__(out self, arm: String):
        self.arm = arm
        self.samples = []
        self.gen = []
        self.pf = []
        self.mis = []
        self.slots = []
        self.peak = 0
        self.spike = False


def ljust(s: String, w: Int) -> String:
    return s + String(" ") * max(0, w - s.byte_length())


def rjust(s: String, w: Int) -> String:
    return String(" ") * max(0, w - s.byte_length()) + s


def num(doc: JSONDoc, obj: Int, key: String) -> Int:
    var i = doc.get_field(obj, key)
    return Int(doc.pool[i].n) if i >= 0 else 0


def first_after(line: String, word: String) raises -> Float64:
    var parts = line.split()
    for i in range(len(parts) - 1):
        if parts[i] == word:
            return atof(String(parts[i + 1]))
    return -1.0


def load(path: String, arm: String, thr: Float64) raises -> Run:
    var r = Run(arm)
    var text: String
    with open(path, "r") as f:
        text = f.read()
    var vm_keys = ["pgscan_direct", "allocstall_normal", "compact_migrate_scanned", "pgmigrate_success"]
    var got_gen = False
    var got_pf = False
    var got_mis = False
    var got_slots = False
    for line in text.split("\n"):
        if line.byte_length() == 0:
            continue
        var data: List[UInt8] = []
        for b in line.as_bytes():
            data.append(b)
        var p = JSONParser(data^)
        var doc = JSONDoc()
        var root = p.parse_value(doc)
        var kind = doc.pool[doc.get_field(root, "k")].s
        if kind == "s":
            var vm = List[Int](length=NK, fill=0)
            var vmi = doc.get_field(root, "vm")
            if vmi >= 0:
                for k in range(NK):
                    vm[k] = num(doc, vmi, vm_keys[k])
            var psi = doc.get_field(root, "psi")
            var psi0 = 0
            if psi >= 0:
                psi0 = Int(doc.pool[doc.pool[psi].arr[0]].n)
            var mem = doc.get_field(root, "mem")
            r.samples.append(Sample(num(doc, root, "t"), num(doc, root, "gtt"), num(doc, mem, "MemAvailable"), vm^, psi0, vmi >= 0))
        elif kind == "l":
            var l = doc.pool[doc.get_field(root, "line")].s
            if not got_gen and l.startswith("gen ms/token"):
                var parts = l.split()
                r.gen.append(atof(String(parts[len(parts) - 1])))
                got_gen = True
            elif not got_pf and l.startswith("prompt prefill") and "ms/token" in l:
                r.pf.append(first_after(l, "ms/token"))
                got_pf = True
            elif not got_mis and l.startswith("expert misses"):
                r.mis.append(first_after(l, "misses"))
                got_mis = True
            elif not got_slots and l.startswith("q4x: expert cache"):
                r.slots.append(first_after(l, "cache"))
                got_slots = True
    if len(r.samples) < 2:
        raise Error("FAIL report: " + path + " has fewer than 2 samples")
    for s in r.samples:
        r.peak = max(r.peak, s.gtt)
    r.spike = Float64(r.peak - r.samples[0].gtt) > thr
    return r^


def median(v: List[Float64]) -> Float64:
    var s = v.copy()
    for i in range(1, len(s)):
        var x = s[i]
        var j = i - 1
        while j >= 0 and s[j] > x:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = x
    var n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2


def med(runs: List[Run], which: Int) -> String:
    var v: List[Float64] = []
    for r in runs:
        if which == 0 and len(r.gen) > 0:
            v.append(r.gen[0])
        elif which == 1 and len(r.pf) > 0:
            v.append(r.pf[0])
        elif which == 2 and len(r.mis) > 0:
            v.append(r.mis[0])
    return fmt_fixed(median(v), 2) if len(v) > 0 else String("-")


def mean(v: List[Float64]) -> Float64:
    return py_sum(v) / Float64(len(v))


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: q4x-gtt-report OUT_DIR [--spike-gb 1]")
        return
    var d = String(args[1])
    var thr = 1.0 * Float64(G)
    for i in range(2, len(args) - 1):
        if args[i] == "--spike-gb":
            thr = atof(String(args[i + 1])) * Float64(G)
    var names: List[String] = []
    for n in listdir(d):
        if n.startswith("run-") and n.endswith(".jsonl"):
            names.append(n)
    for i in range(1, len(names)):
        var x = names[i]
        var j = i - 1
        while j >= 0 and names[j] > x:
            names[j + 1] = names[j]
            j -= 1
        names[j + 1] = x
    var arm_names: List[String] = []
    var arms: List[List[Run]] = []
    var ticks_in: List[List[Float64]] = []
    var ticks_out: List[List[Float64]] = []
    for _ in range(NK + 2):
        ticks_in.append([])
        ticks_out.append([])
    for name in names:
        var rest = name.removeprefix("run-")
        var k = 0
        while k < rest.byte_length() and rest[byte=k] >= "0" and rest[byte=k] <= "9":
            k += 1
        var arm = String("-")
        var tail = rest[byte=k + 1 :]
        if tail.byte_length() > 2 and (tail[byte=0] == "A" or tail[byte=0] == "B") and tail[byte=1] == "-":
            arm = String(tail[byte=0])
        var r = load(d + "/" + name, arm, thr)
        var ai = -1
        for i in range(len(arm_names)):
            if arm_names[i] == arm:
                ai = i
        if ai < 0:
            arm_names.append(arm)
            arms.append([])
            ai = len(arms) - 1
        ref s = r.samples
        for i in range(1, len(s)):
            var dt = Float64(s[i].t - s[i - 1].t) / 1e9
            if dt > 0 and s[i].has_vm and s[i - 1].has_vm:
                var inx = Float64(s[i].gtt - s[0].gtt) > thr
                for c in range(NK):
                    var v = Float64(s[i].vm[c] - s[i - 1].vm[c]) / dt
                    if inx:
                        ticks_in[c].append(v)
                    else:
                        ticks_out[c].append(v)
                var psi = Float64(s[i].psi - s[i - 1].psi) / 1e3 / dt
                var avail = Float64(s[i].avail_kb) / Float64(G) * 1024
                if inx:
                    ticks_in[NK].append(psi)
                    ticks_in[NK + 1].append(avail)
                else:
                    ticks_out[NK].append(psi)
                    ticks_out[NK + 1].append(avail)
        arms[ai].append(r^)
    print(ljust("arm", 3), rjust("runs", 4), rjust("spikes", 6), rjust("peak GB", 8), rjust("slots", 13), "| clean:", rjust("decode", 7), rjust("prefill", 8),
          rjust("misses", 8), "| spike:", rjust("decode", 7), rjust("prefill", 8), "  (ms/token)")
    for ai in range(len(arms)):
        var clean: List[Run] = []
        var spk: List[Run] = []
        var peak = 0
        var lo = Float64.MAX
        var hi = -1.0
        var nsl = 0
        for r in arms[ai]:
            if r.spike:
                spk.append(r.copy())
            else:
                clean.append(r.copy())
            peak = max(peak, r.peak)
            if len(r.slots) > 0:
                lo = min(lo, r.slots[0])
                hi = max(hi, r.slots[0])
                nsl += 1
        if nsl > 0:
            print(ljust(arm_names[ai], 3), rjust(String(len(arms[ai])), 4), rjust(String(len(spk)), 6), fmt_fixed(Float64(peak) / Float64(G), 2, 8),
                  fmt_fixed(lo, 0, 6) + "-" + ljust(fmt_fixed(hi, 0), 6), "|", rjust(med(clean, 0), 14), rjust(med(clean, 1), 8), rjust(med(clean, 2), 8), "|",
                  rjust(med(spk, 0), 14), rjust(med(spk, 1), 8))
        else:
            print(arm_names[ai], len(arms[ai]), "runs without a cache line")
    if len(ticks_in[0]) > 0:
        print("\ncounters per second,", len(ticks_in[0]), "excursion ticks against", len(ticks_out[0]), "other ticks (mean)")
        for c in range(NK + 2):
            var mo = mean(ticks_out[c]) if len(ticks_out[c]) > 0 else nan[DType.float64]()
            print("  " + ljust(String(materialize[KEY_NAMES]()[c]), 22), fmt_fixed(mean(ticks_in[c]), 1, 12), fmt_fixed(mo, 1, 12))
    for ai in range(len(arms)):
        var n = 0
        for r in arms[ai]:
            if r.spike:
                n += 1
        print("spikes arm " + arm_names[ai] + ":", n, "of", len(arms[ai]))
