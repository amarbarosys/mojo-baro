"""Warm list for serve/q4x.mojo from an expert trace (Q4X_EXPERTS=path, lines "il id0 .. id9").

usage: q4x-warm TRACE OUT [PER_LAYER]   writes "il id" lines, most used first within each layer,
at most PER_LAYER per layer (default 512 = all), layers interleaved by rank so a cut at any line
keeps every layer's top experts.
"""
from std.sys import argv

comptime NL = 48
comptime NE = 512


def main() raises:
    var a = argv()
    var per = Int(a[3]) if len(a) > 3 else NE
    var cnt = List[Int](length=NL * NE, fill=0)
    var rows = 0
    with open(String(a[1]), "r") as f:
        for line in f.read().split("\n"):
            var parts = String(line).split(" ")
            if len(parts) < 2:
                continue
            var il = atol(String(parts[0]))
            for j in range(1, len(parts)):
                cnt[il * NE + atol(String(parts[j]))] += 1
            rows += 1
    var order = List[List[Int]]()
    for il in range(NL):
        var ids = List[Int]()
        for e in range(NE):
            ids.append(e)
        var taken = List[Bool](length=NE, fill=False)
        var ranked = List[Int]()
        for _ in range(per):
            var best = -1
            for e in range(NE):
                if not taken[e] and cnt[il * NE + e] > 0 and (best < 0 or cnt[il * NE + e] > cnt[il * NE + best]):
                    best = e
            if best < 0:
                break
            taken[best] = True
            ranked.append(best)
        order.append(ranked^)
    var out = String()
    var n = 0
    for r in range(per):
        for il in range(NL):
            if r < len(order[il]):
                out += String(il) + " " + String(order[il][r]) + "\n"
                n += 1
    with open(String(a[2]), "w") as f:
        f.write(out)
    print("q4x-warm:", rows, "trace rows,", n, "entries ->", a[2])
