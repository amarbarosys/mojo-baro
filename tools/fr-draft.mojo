# Build an FR-Spec pack: an engine pack plus a reduced MTP draft head (q4; a q8 head's rows are requantized to q4
# with ggml quantize_row_q4_0_ref, which only changes drafts: the verify step keeps output identical).
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
from std.memory import bitcast
from std.math import trunc


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


def q4_from_q8(q: List[UInt8], d: List[UInt8], t: Int, h: Int, mut nib: List[UInt8], mut sc: List[UInt8]):
    """Row t of a q8 head (int8 values, f16 scale per 32) as ggml quantize_row_q4_0_ref (tools/engine-pack.mojo q4_row)."""
    for g in range(h // 32):
        var ds = Float32(bitcast[DType.float16](UInt16(d[t * (h // 16) + g * 2]) | (UInt16(d[t * (h // 16) + g * 2 + 1]) << 8)))
        var x = SIMD[DType.float32, 32](0)
        for l in range(32):
            x[l] = Float32(bitcast[DType.int8](q[t * h + g * 32 + l])) * ds
        var amax = Float32(-1)
        var at = 0
        for l in range(32):
            if abs(x[l]) > amax:
                amax = abs(x[l])
                at = l
        var dq = x[at] / Float32(-8)
        var id = Float32(1) / dq if dq != 0 else Float32(0)
        for j in range(16):
            var i0 = Int(min(max(trunc(x[j] * id + Float32(8.5)), Float32(0)), Float32(15)))
            var i1 = Int(min(max(trunc(x[16 + j] * id + Float32(8.5)), Float32(0)), Float32(15)))
            nib.append(UInt8(i0 | (i1 << 4)))
        var hb = bitcast[DType.uint16](Float16(dq))
        sc.append(UInt8(hb & 0xFF))
        sc.append(UInt8(hb >> 8))


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
    var hq8 = False
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
            if dt != "q4" and dt != "q8":
                raise Error("fr-draft: output.weight is " + dt + "; FR-Spec needs a q4 or q8 head")
            hq8 = dt == "q8"
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
        q = f.read_bytes(vocab * (h if hq8 else rq))
        d = f.read_bytes(vocab * rs)
    var head = List[UInt8](capacity=len(ids) * (rq + rs) + len(ids) * 4)
    if hq8:
        var sc = List[UInt8](capacity=len(ids) * rs)
        for t in ids:
            q4_from_q8(q, d, t, h, head, sc)
        head.extend(sc^)
    else:
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
