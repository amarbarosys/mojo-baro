"""q4x_cpu.cpu_experts (q4x round 3): scalar-reference parity on random Q2_0 experts at the real
qwen4exp shapes (H 2560, FF 640), then time per expert over a rotation past the 96 MB L3 (cold reads).
CPU only, no pack needed.

Build: ./.venv/bin/mojo build serve/test_q4x_cpu.mojo -I serve -o .work/test_q4x_cpu
"""
from std.memory import alloc, bitcast
from std.math import exp
from std.sys import exit
from std.time import perf_counter_ns
from q4x_cpu import cpu_experts, q2_row_ref, Q2B, FPtr, BPtr

comptime H = 2560
comptime FF = 640
comptime GB = FF * (H // 64) * Q2B
comptime DB = H * (FF // 64) * Q2B
comptime EB = 2 * GB + DB
comptime NROT = 96


def fill(p: BPtr, n: Int, mut s: UInt64):
    for i in range(n):
        s = s * 6364136223846793005 + 1442695040888963407
        p[i] = UInt8(s >> 56)
    for b in range(n // Q2B):
        p[b * Q2B] = 0x1F
        p[b * Q2B + 1] = 0x21


def main() raises:
    var s = UInt64(7)
    var w = alloc[UInt8](NROT * EB)
    var wp = BPtr(unsafe_from_address=Int(w))
    fill(wp, NROT * EB, s)
    var x = alloc[Float32](4 * H)
    for i in range(4 * H):
        s = s * 6364136223846793005 + 1442695040888963407
        x[i] = Float32(Int(s >> 40) % 2001 - 1000) / 1000
    var y = alloc[Float32](4 * H)
    var scratch = alloc[Float32](16 * (H + 2 * FF))
    var n = 4
    var gw = List[Int](); var uw = List[Int](); var dw = List[Int](); var xs = List[Int](); var ys = List[Int]()
    for i in range(n):
        var e = Int(w) + i * 7 * EB
        gw.append(e); uw.append(e + GB); dw.append(e + 2 * GB)
        xs.append(Int(x) + 4 * i * H); ys.append(Int(y) + 4 * i * H)
    cpu_experts(n, gw, uw, dw, xs, ys, FPtr(unsafe_from_address=Int(scratch)), H, FF)
    var hb = alloc[Float32](FF)
    var worst = Float32(0)
    for i in range(n):
        var xi = FPtr(unsafe_from_address=xs[i])
        for r in range(FF):
            var g = q2_row_ref(BPtr(unsafe_from_address=gw[i] + r * (H // 64) * Q2B), xi, H)
            var u = q2_row_ref(BPtr(unsafe_from_address=uw[i] + r * (H // 64) * Q2B), xi, H)
            hb[r] = (g / (1 + exp(-g))) * u
        var scale = Float32(0)
        var refs = List[Float32]()
        for r in range(H):
            var v = q2_row_ref(BPtr(unsafe_from_address=dw[i] + r * (FF // 64) * Q2B), FPtr(unsafe_from_address=Int(hb)), FF)
            refs.append(v)
            scale = max(scale, abs(v))
        var yi = FPtr(unsafe_from_address=ys[i])
        for r in range(H):
            worst = max(worst, abs(yi[r] - refs[r]) / scale)
    print("parity max |cpu - ref| / max|ref| =", worst)
    if not (worst < 2e-2):
        print("FAIL test_q4x_cpu: parity", worst)
        exit(1)
    for nn in [1, 4, 10]:
        var t0 = perf_counter_ns()
        var reps = 48
        for rep in range(reps):
            var g2 = List[Int](); var u2 = List[Int](); var d2 = List[Int](); var x2 = List[Int](); var y2 = List[Int]()
            for i in range(nn):
                var e = Int(w) + ((rep * nn + i) % NROT) * EB
                g2.append(e); u2.append(e + GB); d2.append(e + 2 * GB)
                x2.append(Int(x) + 4 * (i % 4) * H); y2.append(Int(y) + 4 * (i % 4) * H)
            cpu_experts(nn, g2, u2, d2, x2, y2, FPtr(unsafe_from_address=Int(scratch)), H, FF)
        var us = Float64(perf_counter_ns() - t0) / 1e3 / Float64(reps)
        print("n", nn, "us/call", us, "us/expert", us / Float64(nn), "GB/s", Float64(nn * EB) / (us * 1e3))
    print("PASS test_q4x_cpu")
