from std.math import ceildiv, floor, log10, sqrt
from std.sys import argv, exit, stderr
from std.ffi import external_call

comptime KVPAGE = 128
comptime HD = 256
comptime USAGE = """Compare two engine state files section by section (tools/state_diff.py is the oracle, bench/tests/state-diff-test.sh the differential test).

Usage: .work/state_diff [--moe] [--kvg N] REFERENCE.state OTHER.state

Reads BAROST01 (f32 KV) and BAROST03 (the int8 KV pool verbatim: per page, layer and KV head a
128 x HD int8 block plus a tail of HD/KVG f32 scales per token row), dequantised to f32. --moe
uses the qwen35moe KV shape (10 attention layers, 2 KV heads), default the dense one (8, 4). --kvg N
reads an engine built with -D BARO_KVG=N (default 256). Prints, for conv, SSM state, K and V
(positions below pos only), the relative L2 error ||other - ref|| / ||ref||, the max absolute
difference and the reference norm; K and V also per attention layer beside quant-only (the
reference quantised the int8 pool's way, per KVG values, amax / 127, round half to even), so a
layer's own error separates from the error carried in. Sums in f64. Exit 1 if the headers
(pos, sizes, tokens, salt) disagree."""


struct State(Movable):
    var d: Pointer[UInt8, ImmUntrackedOrigin]
    var magic: String
    var pos: Int
    var conv_n: Int
    var ssm_n: Int
    var kvn: Int
    var f_off: Int
    var k_off: Int

    def __init__(out self, path: String) raises:
        var addr: Int
        with open(path, "r") as f:
            var fd = Int32(f._get_raw_fd())
            var size = external_call["lseek", Int](fd, Int64(0), Int32(2))
            addr = external_call["mmap", Int](Int(0), size, Int32(1), Int32(2), fd, Int64(0))
        if addr == -1:
            raise Error(path + ": mmap failed")
        self.d = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=addr)
        self.magic = String("")
        for i in range(8):
            self.magic += chr(Int(self.d[unsafe_offset=i]))
        if self.magic != "BAROST01" and self.magic != "BAROST03":
            raise Error(path + ": not a BAROST01/BAROST03 state (" + self.magic + ")")
        var h = self.d.unsafe_offset(8).unsafe_bitcast[Int64]()
        self.pos = Int(h[unsafe_offset=0])
        self.conv_n = Int(h[unsafe_offset=1])
        self.ssm_n = Int(h[unsafe_offset=2])
        self.kvn = Int(h[unsafe_offset=3])
        self.f_off = 72 + 4 * self.pos
        self.k_off = self.f_off + 4 * (self.conv_n + self.ssm_n)

    def f32(self, i: Int) -> Float64:
        return Float64(self.d.unsafe_offset(self.f_off).unsafe_bitcast[Float32]()[unsafe_offset=i])

    def kv_row(self, v: Bool, blk: Int, t: Int, kvg: Int, mut row: List[Float64]):
        if self.magic == "BAROST01":
            var p = self.d.unsafe_offset(self.k_off + (4 * self.kvn if v else 0)).unsafe_bitcast[Float32]()
            var o = blk * KVPAGE * HD + t * HD
            for e in range(HD):
                row[e] = Float64(p[unsafe_offset=o + e])
        else:
            var g = HD // kvg
            var bs = KVPAGE * HD + KVPAGE * 4 * g
            var b = self.k_off + (self.kvn if v else 0) + blk * bs
            var q = self.d.unsafe_offset(b + t * HD).unsafe_bitcast[Int8]()
            var sc = self.d.unsafe_offset(b + KVPAGE * HD + 4 * t * g).unsafe_bitcast[Float32]()
            for e in range(HD):
                row[e] = Float64(Float32(q[unsafe_offset=e]) * sc[unsafe_offset=e // kvg])


def round_even(x: Float32) -> Float32:
    var f = floor(x)
    var r = x - f
    if r > 0.5 or (r == 0.5 and Int(f) % 2 != 0):
        return f + 1
    return f


def e3(x: Float64) -> String:
    if x == 0:
        return "0.000e+00"
    var ex = Int(floor(log10(x)))
    var m = Int(x / 10.0 ** Float64(ex) * 1000 + 0.5)
    if m >= 10000:
        m = Int(Float64(m) / 10 + 0.5)
        ex += 1
    var fr = String(m % 1000)
    while fr.byte_length() < 3:
        fr = "0" + fr
    var es = String(abs(ex))
    if es.byte_length() < 2:
        es = "0" + es
    return String(m // 1000) + "." + fr + "e" + ("-" if ex < 0 else "+") + es


def pybool(b: Bool) -> String:
    return "True" if b else "False"


def main() raises:
    var args = List[String]()
    for a in argv()[1:]:
        args.append(String(a))
    var n_att = 8
    var nkvh = 4
    var kvg = 256
    var i = 0
    if len(args) > i and args[i] == "--moe":
        n_att = 10
        nkvh = 2
        i += 1
    if len(args) > i + 1 and args[i] == "--kvg":
        kvg = Int(atol(args[i + 1]))
        i += 2
    if len(args) - i != 2:
        print(USAGE, file=stderr)
        exit(1)
    var a = State(args[i])
    var b = State(args[i + 1])
    var toks_eq = a.pos == b.pos
    if toks_eq:
        for i in range(4 * a.pos):
            if a.d[unsafe_offset=72 + i] != b.d[unsafe_offset=72 + i]:
                toks_eq = False
                break
    var salt_eq = True
    for i in range(40, 72):
        if a.d[unsafe_offset=i] != b.d[unsafe_offset=i]:
            salt_eq = False
    print(a.magic + " vs " + b.magic + "; pos " + String(a.pos) + " vs " + String(b.pos) + "; tokens equal " + pybool(toks_eq) + "; salt equal " + pybool(salt_eq))
    if not (a.pos == b.pos and a.conv_n == b.conv_n and a.ssm_n == b.ssm_n and toks_eq and salt_eq):
        print("FAIL: headers disagree")
        exit(1)
    var lo = 0
    for name in ["conv", "ssm"]:
        var n = a.conv_n if name == "conv" else a.ssm_n
        var dd = 0.0
        var xx = 0.0
        var mx = 0.0
        for i in range(lo, lo + n):
            var x = a.f32(i)
            var dl = b.f32(i) - x
            dd += dl * dl
            xx += x * x
            mx = max(mx, abs(dl))
        lo += n
        print((" ssm" if name == "ssm" else "conv") + ": rel L2 " + e3(sqrt(dd) / max(sqrt(xx), 1e-30)) + "  max abs " + e3(mx) + "  ref norm " + e3(sqrt(xx)))
    var pages = ceildiv(a.pos, KVPAGE)
    var rx = List[Float64](length=HD, fill=0.0)
    var ry = List[Float64](length=HD, fill=0.0)
    var lines = List[String]()
    for vi in range(2):
        var v = vi == 1
        var name = "v" if v else "k"
        var tdd = 0.0
        var txx = 0.0
        var tmx = 0.0
        for li in range(n_att):
            var dd = 0.0
            var xx = 0.0
            var qq = 0.0
            var mx = 0.0
            for p in range(pages):
                for h in range(nkvh):
                    var blk = (p * n_att + li) * nkvh + h
                    for t in range(min(KVPAGE, a.pos - p * KVPAGE)):
                        a.kv_row(v, blk, t, kvg, rx)
                        b.kv_row(v, blk, t, kvg, ry)
                        for g0 in range(0, HD, kvg):
                            var am = Float32(0)
                            for e in range(g0, g0 + kvg):
                                am = max(am, abs(Float32(rx[e])))
                            var sc = am / 127 if am > 0 else Float32(1)
                            for e in range(g0, g0 + kvg):
                                var x = rx[e]
                                var dl = ry[e] - x
                                dd += dl * dl
                                xx += x * x
                                mx = max(mx, abs(dl))
                                var qd = Float64(min(max(round_even(Float32(x) / sc), Float32(-127)), Float32(127)) * sc) - x
                                qq += qd * qd
            var nx = max(sqrt(xx), 1e-30)
            lines.append("  " + name + " layer " + String(li) + ": rel L2 " + e3(sqrt(dd) / nx) + "  quant-only " + e3(sqrt(qq) / nx) + "  max abs " + e3(mx) + "  ref norm " + e3(sqrt(xx)))
            tdd += dd
            txx += xx
            tmx = max(tmx, mx)
        print("   " + name + ": rel L2 " + e3(sqrt(tdd) / max(sqrt(txx), 1e-30)) + "  max abs " + e3(tmx) + "  ref norm " + e3(sqrt(txx)))
    for l in lines:
        print(l)
