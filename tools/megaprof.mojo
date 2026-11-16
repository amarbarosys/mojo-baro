# megaprof: per-phase time of the persistent megakernel from BARO_MEGA_PROF=1 logs.
# usage: megaprof LOG [LOG_B]
# A MEGAPROF line is "MEGAPROF <layer> <16 realtime stamps>" (serve/engine.mojo::mega_prof, 100 MHz counter, 0 = not
# written). Slot k-1 -> k is a phase of kernels/mega.mojo (stamp(prof, pbase + k)). Layers with layer % 4 == 3 are
# full-attention layers, the rest SSM layers. Prints mean us per slot transition for both layer kinds and the mean
# layer span; with LOG_B, both columns side by side and the ratio B / A.
from std.sys import argv, exit


def parse(path: String) raises -> List[Float64]:
    # 4 x 12 sums: [kind][k] for kind 0 = ssm, 1 = att; index 0..11 = transition k-1 -> k (11 = span), plus counts.
    var acc = List[Float64](length=48, fill=0.0)
    var n_lines = 0
    with open(path, "r") as f:
        for line in f.read().splitlines():
            if not line.startswith("MEGAPROF "):
                continue
            var w = line.split(" ")
            if len(w) < 18:
                continue
            var layer = atol(w[1])
            var t = List[Int]()
            for i in range(16):
                t.append(atol(w[2 + i]))
            var kind = 1 if layer % 4 == 3 else 0
            for k in range(1, 12):
                if t[k - 1] > 0 and t[k] >= t[k - 1]:
                    acc[kind * 12 + k - 1] += Float64(t[k] - t[k - 1]) / 100.0
                    acc[24 + kind * 12 + k - 1] += 1.0
            if t[0] > 0 and t[11] >= t[0]:
                acc[kind * 12 + 11] += Float64(t[11] - t[0]) / 100.0
                acc[24 + kind * 12 + 11] += 1.0
            n_lines += 1
    if n_lines == 0:
        raise Error("FAIL parse: no MEGAPROF lines in " + path + " (run with BARO_MEGA_PROF=1 BARO_MEGA=1)")
    return acc^


def mean(a: List[Float64], kind: Int, k: Int) -> Float64:
    var n = a[24 + kind * 12 + k]
    return a[kind * 12 + k] / n if n > 0 else 0.0


def fmt(x: Float64) -> String:
    var s = String(Float64(Int(x * 10 + 0.5)) / 10.0)
    return " " * max(0, 8 - s.byte_length()) + s


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("FAIL usage: megaprof LOG [LOG_B]")
        exit(1)
    var a = parse(String(args[1]))
    var two = len(args) > 2
    var b = parse(String(args[2])) if two else a.copy()
    print("phase      ssm_us" + ("    ssm_B  B/A" if two else "") + "     att_us" + ("    att_B  B/A" if two else ""))
    for k in range(12):
        var name = "span    " if k == 11 else "slot " + String(k) + "->" + String(k + 1) + (" " if k < 9 else "")
        var line = name
        for kind in range(2):
            line += "  " + fmt(mean(a, kind, k))
            if two:
                var ma = mean(a, kind, k)
                var mb = mean(b, kind, k)
                line += " " + fmt(mb) + "  " + (String(Float64(Int(mb / ma * 100 + 0.5)) / 100.0) if ma > 0 else "-")
        print(line)
