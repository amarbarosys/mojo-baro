"""Dsum-diff: compare two q4x `D` checksum streams (Q4X dump logs) and name the first (tensor, layer, pos) that differs.

  dsum-diff A.log B.log [--rtol 0]

A line is `D name layer sum` or `D name layer row sum`. The printed row is the row inside ONE call: decode prints
row 0 on every call, prefill prints 0..T-1 in one call. So the position is always the occurrence count of that
(name, layer) pair in the log, never the printed column; lines are keyed (name, layer, pos) that way and everything
else in the logs is ignored. One name can come from two call sites per token (hc_mixed): decode prints them
interleaved per token, prefill prints one site's rows 0..T-1 and then the next, so the key is (name, layer, site,
pos): prefill counts a new site at each row-0 reset, decode splits occurrences by the token count (model.input_embed
lines). Single-sequence dumps only (an S>1 batch interleaves sequences). Prints the first differing key in A's order
with both sums and the relative gap, then `match N/M keys` and the keys present on one side only.
Exit 0 when every shared key matches within --rtol (default 0 = exact) and no key is one-sided; 1 otherwise; 2 usage.
Why: the latent-flash P1 lane lost a turn to a "everything differs" misread (decode positions taken from the
printed column) and found a 1e-8 register-state drift by hand with awk.
"""
from std.collections import Dict
from std.sys import argv, exit, stderr


@fieldwise_init
struct Rec(Copyable, Movable):
    var nl: String
    var row: Int
    var v: Float64


def load(path: String, mut order: List[String]) raises -> Dict[String, Float64]:
    var recs = List[Rec]()
    var prefill = False
    with open(path, "r") as f:
        for line in f.read().split("\n"):
            var s = String(line)
            if not s.startswith("D "):
                continue
            var p = s.split(" ")
            if len(p) != 4 and len(p) != 5:
                continue
            var row = Int(String(p[3])) if len(p) == 5 else -1
            if row > 0:
                prefill = True
            recs.append(Rec(String(p[1]) + " " + String(p[2]), row, Float64(String(p[len(p) - 1]))))
    var total = Dict[String, Int]()
    for r in recs:
        total[r.nl] = total.get(r.nl, 0) + 1
    var tokens = total.get("model.input_embed -1", 0)
    var seen = Dict[String, Int]()
    var sites = Dict[String, Int]()
    var out = Dict[String, Float64]()
    for r in recs:
        var i = seen.get(r.nl, 0)
        seen[r.nl] = i + 1
        var site: Int
        var pos: Int
        if prefill and r.row >= 0:
            if r.row == 0:
                sites[r.nl] = sites.get(r.nl, -1) + 1
            site = sites.get(r.nl, 0)
            pos = r.row
        else:
            var per = total[r.nl] // tokens if tokens > 0 and total[r.nl] % tokens == 0 else 1
            site = i % per
            pos = i // per
        var key = r.nl + " s" + String(site) + " " + String(pos)
        if key not in out:
            order.append(key)
        out[key] = r.v
    return out^


def main() raises:
    var av = argv()
    var pos = List[String]()
    var rtol = 0.0
    var i = 1
    while i < len(av):
        var s = String(av[i])
        if s == "--rtol" and i + 1 < len(av):
            rtol = Float64(String(av[i + 1]))
            i += 2
        else:
            pos.append(s)
            i += 1
    if len(pos) != 2:
        print("usage: dsum-diff A.log B.log [--rtol 0]", file=stderr)
        exit(2)
    var oa = List[String]()
    var ob = List[String]()
    var a = load(pos[0], oa)
    var b = load(pos[1], ob)
    if len(oa) == 0 or len(ob) == 0:
        print("FAIL input: no D lines in " + (pos[0] if len(oa) == 0 else pos[1]), file=stderr)
        exit(1)
    var shared = 0
    var same = 0
    var first = String()
    for k in oa:
        if k not in b:
            continue
        shared += 1
        var x = a[k]
        var y = b[k]
        var gap = abs(x - y) / max(abs(x), abs(y), 1e-30)
        if gap <= rtol:
            same += 1
        elif first == "":
            first = k + "  A " + String(x) + "  B " + String(y) + "  rel " + String(gap)
    var only_a = 0
    var only_b = 0
    var ex_a = String()
    var ex_b = String()
    for k in oa:
        if k not in b:
            only_a += 1
            if ex_a == "":
                ex_a = k
    for k in ob:
        if k not in a:
            only_b += 1
            if ex_b == "":
                ex_b = k
    if first != "":
        print("first diff (name layer site pos):", first)
    print("match", same, "/", shared, "keys (rtol", rtol, ")  only-A", only_a, ex_a, " only-B", only_b, ex_b)
    if same != shared or only_a > 0 or only_b > 0:
        exit(1)
