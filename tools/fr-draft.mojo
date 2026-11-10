# Build an FR-Spec pack: a q4 engine pack plus a reduced MTP draft head.
#
# usage: mojo run tools/fr-draft.mojo SRC_PACK OUT_PACK IDS_FILE
#
# OUT_PACK/pack.bin is a reflink copy of SRC_PACK/pack.bin (on btrfs the two
# share extents, so the multi-GB trunk costs no extra disk), with two trailing
# entries appended after everything the source index lists:
#   output.weight.frdraft q4   rows of output.weight for the ids in IDS_FILE,
#                              in file order, byte-exact (nibbles, then scales)
#   frdraft.ids i32            the ids: draft row r -> vocabulary id
# serve/harness.mojo keeps both out of the blk.32 index math, as it does for
# output.weight.q4draft. Small side files (tokenizer, prompt, refs) are copied.
from std.os.path import exists
from std.subprocess import run
from std.sys import argv


def entry_bytes(dt: String, n: Int) raises -> Int:
    if dt == "bf16":
        return n * 2
    if dt == "f32" or dt == "i32":
        return n * 4
    if dt == "q8":
        return n + (n // 32) * 2
    if dt == "q4":
        return n // 2 + (n // 32) * 2
    raise Error("fr-draft: unknown pack dtype " + dt)


def main() raises:
    var a = argv()
    if len(a) != 4:
        print("usage: fr-draft SRC_PACK OUT_PACK IDS_FILE")
        return
    var src = String(a[1])
    var dst = String(a[2])
    if exists(dst):
        raise Error("fr-draft: " + dst + " exists, refusing to overwrite")
    var index: String
    with open(src + "/index.txt", "r") as f:
        index = f.read()
    var off = -1
    var n = 0
    var h = 0
    var end = 0
    for line in index.splitlines():
        var p = line.split(" ")
        if len(p) < 4:
            continue
        var name = String(p[0])
        var dt = String(p[1])
        var o = Int(String(p[2]))
        var ne = Int(String(p[3]))
        if name == "output.weight.frdraft":
            raise Error("fr-draft: source is already an FR pack")
        if name == "output.weight":
            if dt != "q4":
                raise Error("fr-draft: output.weight is " + dt + "; FR-Spec needs a --q4 pack")
            off = o
            n = ne
        if name == "output_norm.weight":
            h = ne
        end = max(end, o + entry_bytes(dt, ne))
    if off < 0 or h == 0:
        raise Error("fr-draft: output.weight or output_norm.weight missing")
    var vocab = n // h
    var ids = List[Int]()
    var seen = List[Bool](length=vocab, fill=False)
    with open(String(a[3]), "r") as f:
        for tok in f.read().split():
            var t = Int(String(tok))
            if t < 0 or t >= vocab or seen[t]:
                raise Error("fr-draft: id out of range or duplicate: " + String(t))
            seen[t] = True
            ids.append(t)
    var rq = h // 2
    var rs = (h // 32) * 2
    var q: List[UInt8]
    var d: List[UInt8]
    with open(src + "/pack.bin", "r") as f:
        _ = f.seek(off)
        q = f.read_bytes(vocab * rq)
        d = f.read_bytes(vocab * rs)
    var head = List[UInt8](capacity=len(ids) * (rq + rs) + len(ids) * 4)
    for t in ids:
        for j in range(rq):
            head.append(q[t * rq + j])
    for t in ids:
        for j in range(rs):
            head.append(d[t * rs + j])
    var head_bytes = len(head)
    for t in ids:
        comptime for b in range(4):
            head.append(UInt8((t >> (8 * b)) & 0xFF))
    _ = run("mkdir -p '" + dst + "' && cp --reflink=always '" + src + "/pack.bin' '" + dst + "/pack.bin'")
    if end % 4 != 0:
        raise Error("fr-draft: pack end not 4-byte aligned")
    with open(dst + "/pack.bin", "a") as f:
        f.write_bytes(Span(head))
    var k = len(ids)
    var lines = index
    if not lines.endswith("\n"):
        lines += "\n"
    lines += "output.weight.frdraft q4 " + String(end) + " " + String(k * h) + "\n"
    lines += "frdraft.ids i32 " + String(end + head_bytes) + " " + String(k) + "\n"
    with open(dst + "/index.txt", "w") as f:
        f.write(lines)
    _ = run("for f in '" + src + "'/*; do case \"${f##*/}\" in pack.bin|index.txt) ;; *) [ -f \"$f\" ] && cp -p \"$f\" '" + dst + "/';; esac; done")
    print("fr-draft:", k, "of", vocab, "rows, head", Float64(head_bytes) / 1048576.0, "MiB ->", dst)
