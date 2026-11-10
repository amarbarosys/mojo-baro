"""tdiff: navigate two tensor dumps without ever opening the bytes.

A dump is a directory holding `data.bin` plus `index.txt`, one line per
tensor, `name dtype byte_offset n_elem` -- the same convention the engine
packer already writes, so `.work/moe-w1/pack` and a dump read the same way.
Produced by llama.cpp with LLAMA_DUMP_DIR set (common/debug.cpp, patched
2026-09-12, env-gated so default behaviour is unchanged), and by our engine
via BARO_DUMP_DIR.

    tdiff summary OURS LLAMA [floor_tensor]
        One ranked row per tensor present in both, worst first. This is the
        whole workflow: it names the first divergence and says what KIND it
        is. Reads only what it needs; a 200 MB dump never enters memory.

    tdiff heads OURS LLAMA NAME NHEADS
        Per-head relative L2 for one tensor. Every MoE bug in this repo so
        far has been head-indexed (24 of 32 heads never written, a half-width
        gate, a key-head mapping), and a whole-tensor norm hides all three.

    tdiff show OURS LLAMA NAME [start] [count]
        Raw elements side by side, for when the table points somewhere.

    tdiff scan OURS
        Impossible-value report, NO reference needed: exact-constant runs,
        trailing all-zero regions, saturation at exactly 0.0/1.0, NaN/Inf.
        Both of the last two bugs announced themselves this way rather than
        by magnitude -- 24 beta gates at exactly 0.5000, which is sigmoid(0)
        on memory nobody had written.

Metrics, and why more than one:
    relL2   ||a-b|| / ||b||     overall wrongness, scale free
    cos     cosine similarity   direction wrong while magnitude is right,
                                which was this engine's signature twice
    maxrel  worst element + its index
    bad%    share of elements past tolerance -- THE DISCRIMINATOR: a few
            elements wrong means an indexing or layout bug, everything
            slightly wrong means numerics or quantisation

A sum is never a verdict (PROTOCOL-RULES P9): layer 0 once agreed with
llama to 1.2% on the sum while its components were 6%, 12% and 78% out.

Always establish the floor first. Our pack is Q4_K/Q8_0 and llama runs the
same GGUF, but the dequant paths are not bit-identical, so some divergence
is quantisation and not a defect. Pass a tensor you believe correct as the
third argument to `summary` and its relL2 is printed as the floor; rows
under it are noise.
"""
from std.collections import Dict
from std.math import sqrt, isnan, isinf
from std.sys import argv, exit

comptime USAGE = "usage: tdiff summary|heads|show|scan ... (see the module docstring)"


@fieldwise_init
struct Ent(Copyable, Movable):
    var off: Int
    var n: Int


def load_index(dir: String, mut names: List[String]) raises -> Dict[String, Ent]:
    var out = Dict[String, Ent]()
    with open(dir + "/index.txt", "r") as f:
        for line in f.read().splitlines():
            var p = String(line).split(" ")
            if len(p) < 4:
                continue
            # Parse from the END: llama tensor names contain spaces, e.g.
            # "cache_r_l0 (reshaped) (view) f32 24576 24576", so the name is
            # everything before the last three fields, never p[0].
            var nf = len(p)
            var nm = String(p[0])
            for k in range(1, nf - 3):
                nm += " " + String(p[k])
            if nm in out:
                # first occurrence wins, matching the oracle's setdefault
                continue
            out[nm] = Ent(Int(String(p[nf - 2])), Int(String(p[nf - 1])))
            names.append(nm)
    return out^


def read_f32(dir: String, e: Ent) raises -> List[Float32]:
    var out = List[Float32](unsafe_uninit_length=e.n)
    with open(dir + "/data.bin", "r") as f:
        _ = f.seek(e.off)
        var raw = f.read_bytes(e.n * 4)
        var src = raw.unsafe_ptr().unsafe_bitcast[Float32]()
        for i in range(e.n):
            out[i] = src[i]
    return out^


@fieldwise_init
struct Metrics(Copyable, ImplicitlyCopyable, Movable):
    var rel_l2: Float64
    var cos: Float64
    var maxrel: Float64
    var maxidx: Int
    var bad: Float64
    var nonfinite: Int


def compare(a: List[Float32], b: List[Float32], tol: Float64) -> Metrics:
    """Relative error is meaningless on elements that are ~0: a tiny absolute
    difference there is a huge ratio while contributing nothing to the norm.
    The first real run of this tool reported bad% 53.8 with relL2 0.019 and
    cos 0.9998, which was entirely that artifact. So elements are only judged
    where |reference| rises above a floor tied to the reference's own RMS,
    the same shape as numpy's atol + rtol*|b|."""
    var n = min(len(a), len(b))
    var rms = Float64(0)
    for i in range(n):
        var y = Float64(b[i])
        if not (isnan(y) or isinf(y)):
            rms += y * y
    rms = sqrt(rms / Float64(n)) if n > 0 else Float64(0)
    var mag_floor = rms * 0.01
    var num = Float64(0)
    var den = Float64(0)
    var dot = Float64(0)
    var na = Float64(0)
    var nb = Float64(0)
    var maxrel = Float64(0)
    var maxidx = -1
    var bad = 0
    var judged = 0
    var nonfinite = 0
    for i in range(n):
        var x = Float64(a[i])
        var y = Float64(b[i])
        if isnan(x) or isinf(x) or isnan(y) or isinf(y):
            nonfinite += 1
            continue
        var d = x - y
        num += d * d
        den += y * y
        dot += x * y
        na += x * x
        nb += y * y
        if abs(y) < mag_floor:
            continue
        judged += 1
        var r = abs(d) / abs(y)
        if r > maxrel:
            maxrel = r
            maxidx = i
        if r > tol:
            bad += 1
    var rel = sqrt(num) / sqrt(den) if den > 0 else Float64(0)
    var cos = dot / (sqrt(na) * sqrt(nb)) if na > 0 and nb > 0 else Float64(0)
    var badpct = Float64(bad) / Float64(judged) * 100.0 if judged > 0 else Float64(0)
    return Metrics(rel, cos, maxrel, maxidx, badpct, nonfinite)


def f(v: Float64, dec: Int) -> String:
    var neg = v < 0
    var x = -v if neg else v
    var m = 1.0
    for _ in range(dec):
        m *= 10.0
    var w = Int(x * m + 0.5)
    var ip = w // Int(m)
    var fp = w % Int(m)
    var fs = String(fp)
    while fs.byte_length() < dec:
        fs = String("0") + fs
    var s = String(ip) + "." + fs
    return ("-" + s) if neg else s


def pad(s: String, w: Int) -> String:
    var o = s
    while o.byte_length() < w:
        o += " "
    return o


def cmd_summary(ours: String, llama: String, floor_name: String) raises:
    var na = List[String]()
    var nb = List[String]()
    var ia = load_index(ours, na)
    var ib = load_index(llama, nb)
    var rows_name = List[String]()
    var rows_m = List[Metrics]()
    for nm in na:
        if nm not in ib:
            continue
        var m = compare(read_f32(ours, ia[nm]), read_f32(llama, ib[nm]), 0.02)
        rows_name.append(nm)
        rows_m.append(m)
    # worst first by relL2. Sort an index list rather than the rows: the row
    # count is the number of dumped tensors, not the element count, and the
    # payloads never move.
    var order = List[Int]()
    for i in range(len(rows_m)):
        order.append(i)
    for i in range(len(order)):
        var best = i
        for j in range(i + 1, len(order)):
            if rows_m[order[j]].rel_l2 > rows_m[order[best]].rel_l2:
                best = j
        if best != i:
            var t = order[i]
            order[i] = order[best]
            order[best] = t
    var floor = Float64(-1)
    for i in range(len(rows_name)):
        if rows_name[i] == floor_name:
            floor = rows_m[i].rel_l2
    print(pad("tensor", 28), pad("relL2", 10), pad("cos", 9), pad("maxrel", 11), pad("bad%", 8), "nonfinite")
    for oi in range(len(order)):
        var i = order[oi]
        ref m = rows_m[i]
        var mark = String("")
        if floor >= 0 and m.rel_l2 > floor * 3:
            mark = "  <-- above floor"
        print(
            pad(rows_name[i], 28), pad(f(m.rel_l2, 6), 10), pad(f(m.cos, 6), 9),
            pad(f(m.maxrel, 4) + "@" + String(m.maxidx), 11), pad(f(m.bad, 1), 8),
            String(m.nonfinite) + mark,
        )
    if floor >= 0:
        print("\nfloor (" + floor_name + "): relL2", f(floor, 6), "-- rows at or under this are quantisation, not defects")
    else:
        print("\nNO FLOOR GIVEN: pass a tensor you believe correct as arg 3, or every row is uninterpretable")


def cmd_heads(ours: String, llama: String, name: String, nheads: Int) raises:
    var na = List[String]()
    var nb = List[String]()
    var ia = load_index(ours, na)
    var ib = load_index(llama, nb)
    if name not in ia or name not in ib:
        print("tensor not in both dumps:", name)
        return
    var a = read_f32(ours, ia[name])
    var b = read_f32(llama, ib[name])
    var n = min(len(a), len(b))
    var per = n // nheads
    print("per-head relL2 for", name, "(", nheads, "heads x", per, ")")
    for h in range(nheads):
        var sa = List[Float32]()
        var sb = List[Float32]()
        for i in range(h * per, (h + 1) * per):
            sa.append(a[i])
            sb.append(b[i])
        var m = compare(sa, sb, 0.02)
        print("  h" + pad(String(h), 3), "relL2", pad(f(m.rel_l2, 6), 10), "cos", f(m.cos, 6))


def cmd_show(ours: String, llama: String, name: String, start: Int, count: Int) raises:
    var na = List[String]()
    var nb = List[String]()
    var ia = load_index(ours, na)
    var ib = load_index(llama, nb)
    if name not in ia or name not in ib:
        print("tensor not in both dumps:", name)
        return
    var a = read_f32(ours, ia[name])
    var b = read_f32(llama, ib[name])
    print(pad("idx", 8), pad("ours", 16), pad("llama", 16), "absdiff")
    for i in range(start, min(start + count, min(len(a), len(b)))):
        var d = abs(Float64(a[i]) - Float64(b[i]))
        print(pad(String(i), 8), pad(f(Float64(a[i]), 6), 16), pad(f(Float64(b[i]), 6), 16), f(d, 6))


def cmd_scan(dir: String) raises:
    """No reference needed. Flags what cannot occur naturally."""
    var names = List[String]()
    var idx = load_index(dir, names)
    print("scanning", len(names), "tensors for impossible values")
    var flagged = 0
    for nm in names:
        var v = read_f32(dir, idx[nm])
        var n = len(v)
        if n == 0:
            continue
        var nan = 0
        var zeros = 0
        var sat = 0
        var trailing_zero = 0
        var const_run = 1
        var best_run = 1
        var best_val = Float64(0)
        for i in range(n):
            var x = Float64(v[i])
            if isnan(x) or isinf(x):
                nan += 1
            if x == 0.0:
                zeros += 1
            if x == 0.0 or x == 1.0:
                sat += 1
            if i > 0 and v[i] == v[i - 1]:
                const_run += 1
                if const_run > best_run:
                    best_run = const_run
                    best_val = x
            else:
                const_run = 1
        var i = n - 1
        while i >= 0 and v[i] == 0.0:
            trailing_zero += 1
            i -= 1
        var msg = String("")
        if nan > 0:
            msg += " NONFINITE=" + String(nan)
        if trailing_zero > n // 8 and trailing_zero < n:
            msg += " TRAILING_ZEROS=" + String(trailing_zero) + "/" + String(n)
        if best_run > 8 and best_run < n:
            msg += " CONST_RUN=" + String(best_run) + "x" + f(best_val, 4)
        if sat == n and n > 4:
            msg += " ALL_SATURATED_0_OR_1"
        if zeros == n:
            msg += " ALL_ZERO"
        if msg != "":
            flagged += 1
            print(" ", pad(nm, 30), msg)
    print("flagged", flagged, "of", len(names))


def main() raises:
    var a = argv()
    if len(a) < 2:
        print(USAGE)
        exit(1)
    var cmd = String(a[1])
    if cmd == "summary" and len(a) >= 4:
        cmd_summary(String(a[2]), String(a[3]), String(a[4]) if len(a) > 4 else String(""))
    elif cmd == "heads" and len(a) >= 6:
        cmd_heads(String(a[2]), String(a[3]), String(a[4]), Int(String(a[5])))
    elif cmd == "show" and len(a) >= 5:
        cmd_show(
            String(a[2]), String(a[3]), String(a[4]),
            Int(String(a[5])) if len(a) > 5 else 0,
            Int(String(a[6])) if len(a) > 6 else 16,
        )
    elif cmd == "scan" and len(a) >= 3:
        cmd_scan(String(a[2]))
    else:
        print(USAGE)
        exit(1)
