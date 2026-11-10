"""Bytes read per decode token from a pack's index.txt, split by tensor class, and the roof.

usage: bytes-per-token PACK_DIR [--topk 8] [--experts 256] [--gbs 800]
Tensor bytes = offset difference in index.txt (column 4 is an element count, not bytes).
`*_exps` tensors are routed experts: topk/experts of their bytes are read per token.
token_embd reads one row, so it is left out. Prints the top tensor kinds, the dense vs
expert split, and tok/s at --gbs GB/s (decide where decode time can go before any kernel work).
"""
from std.sys import argv
from std.os import stat


def kind(name: String) -> String:
    if name.startswith("blk."):
        var parts = name.split(".")
        var out = String("")
        for i in range(2, len(parts)):
            if i > 2:
                out += "."
            out += String(parts[i])
        return out
    return name


def pad(s: String, n: Int, right: Bool = False) -> String:
    var r = s
    var k = n - s.byte_length()
    for _ in range(k):
        r = (" " + r) if right else (r + " ")
    return r


def main() raises:
    var a = argv()
    if len(a) < 2:
        raise Error("FAIL usage: bytes-per-token PACK_DIR [--topk 8] [--experts 256] [--gbs 800]")
    var dir = String(a[1])
    var topk = 8.0
    var nexp = 256.0
    var gbs = 800.0
    var i = 2
    while i + 1 < len(a):
        var t = String(a[i])
        if t == "--topk":
            topk = atof(String(a[i + 1]))
        elif t == "--experts":
            nexp = atof(String(a[i + 1]))
        elif t == "--gbs":
            gbs = atof(String(a[i + 1]))
        i += 2
    var txt: String
    with open(dir + "/index.txt", "r") as f:
        txt = f.read()
    var names = List[String]()
    var offs = List[Int]()
    for ln in txt.split("\n"):
        var c = String(ln).split(" ")
        if len(c) == 4:
            names.append(String(c[0]))
            offs.append(atol(String(c[2])))
    var end = stat(dir + "/pack.bin").st_size
    var kinds = Dict[String, Int]()
    for j in range(len(names)):
        var nxt = end
        for q in range(len(offs)):
            if offs[q] > offs[j] and offs[q] < nxt:
                nxt = offs[q]
        var k = kind(names[j])
        kinds[k] = kinds.get(k, 0) + (nxt - offs[j])
    var dense = 0.0
    var exps = 0.0
    for e in kinds.items():
        if e.key.find("_exps") >= 0:
            exps += Float64(e.value)
        elif e.key != "token_embd.weight":
            dense += Float64(e.value)
    var act = exps * topk / nexp
    var tot = dense + act
    print("kind                              GB (whole pack)")
    for e in kinds.items():
        if Float64(e.value) > 0.02 * Float64(end):
            print(pad(e.key, 32), Float64(e.value) / 1e9)
    print("dense read/token", dense / 1e9, "GB  (", 100.0 * dense / tot, "% )")
    print("experts read/token", act / 1e9, "GB  (", 100.0 * act / tot, "%, topk", topk, "of", nexp, ")")
    print("bytes/token", tot / 1e9, "GB -> roof", gbs * 1e9 / tot, "tok/s at", gbs, "GB/s")
