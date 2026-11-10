# mspec_gaps: launch-gap share of the speculative verify window from rocprofv3
# kernel traces (bench/mtp-protocol.md, amendment MSPEC step 1).
# usage: mspec_gaps OUTDIR   (OUTDIR written by bench/mspec-trace.sh)
# Verify window = the dispatches strictly between two amar_tok_copy launches when
# there are more than 200 of them; it must open on amar_embed_lookup_p* (the trace mangles the name) and close
# on the last amar_argmax_pos before the result-copy blits (__amd_rocclr_copyBuffer,
# counted as host time), else it is counted as malformed. W = span, S = sum of
# durations, g = sum(W - S) / sum(W). Iteration = accept tok_copy to accept tok_copy.
from std.sys import argv


def split_csv(line: String) -> List[String]:
    var out = List[String]()
    var b = line.as_bytes()
    var n = len(b)
    var start = 0
    var inq = False
    for i in range(n + 1):
        if i < n and b[i] == UInt8(ord('"')):
            inq = not inq
        elif i == n or (b[i] == UInt8(ord(",")) and not inq):
            var a = start
            var e = i
            if e - a >= 2 and b[a] == UInt8(ord('"')) and b[e - 1] == UInt8(ord('"')):
                a += 1
                e -= 1
            out.append(String(line[byte=a:e]))
            start = i + 1
    return out^


def num_after(text: String, key: String) raises -> Float64:
    var i = text.find(key)
    if i < 0:
        raise Error("missing '" + key + "'")
    var b = text.as_bytes()
    var a = i + key.byte_length()
    while a < len(b) and b[a] == UInt8(ord(" ")):
        a += 1
    var e = a
    while e < len(b):
        var c = b[e]
        var ok = (c >= UInt8(ord("0")) and c <= UInt8(ord("9"))) or c == UInt8(ord(".")) or c == UInt8(ord("e")) or c == UInt8(ord("-")) or c == UInt8(ord("+"))
        if not ok:
            break
        e += 1
    return atof(String(text[byte=a:e]))


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def median(var v: List[Float64]) -> Float64:
    for i in range(1, len(v)):
        var x = v[i]
        var j = i - 1
        while j >= 0 and v[j] > x:
            v[j + 1] = v[j]
            j -= 1
        v[j + 1] = x
    var n = len(v)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return v[n // 2]
    return (v[n // 2 - 1] + v[n // 2]) / 2.0


def fmt2(x: Float64) -> String:
    var neg = x < 0.0
    var v = Int((-x if neg else x) * 100.0 + 0.5)
    var frac = v % 100
    return ("-" if neg else "") + String(v // 100) + "." + ("0" if frac < 10 else "") + String(frac)


@fieldwise_init
struct Stat(Copyable, Movable):
    var nwin: Int
    var bad: Int
    var gaps_ns: Int
    var wall_ns: Int
    var iter_span: Int
    var iter_ksum: Int
    var host_ns: Int
    var disp: List[Float64]
    var wms: List[Float64]


def analyze(path: String) raises -> Stat:
    var text = read_text(path)
    var lines = text.split("\n")
    var st = List[Int]()
    var en = List[Int]()
    var kind = List[Int]()
    var first = True
    for ln in lines:
        if first:
            first = False
            continue
        if ln.byte_length() == 0:
            continue
        var f = split_csv(String(ln))
        if len(f) < 22 or f[0] != "KERNEL_DISPATCH":
            continue
        var nm = f[7]
        var k = 0
        if "amar_tok_copy" in nm:
            k = 1
        elif "amar_embed_lookup_p" in nm:
            k = 2
        elif "amar_argmax_pos" in nm:
            k = 3
        elif "__amd_rocclr_copyBuffer" in nm:
            k = 4
        st.append(Int(f[9]))
        en.append(Int(f[10]))
        kind.append(k)
    var n = len(st)
    var ord_ = List[Int]()
    for i in range(n):
        ord_.append(i)
    for i in range(1, n):
        var x = ord_[i]
        var j = i - 1
        while j >= 0 and st[ord_[j]] > st[x]:
            ord_[j + 1] = ord_[j]
            j -= 1
        ord_[j + 1] = x
    var s = Stat(0, 0, 0, 0, 0, 0, 0, List[Float64](), List[Float64]())
    var prev_tc = -1
    var prev_acc_end = -1
    for p in range(n):
        var i = ord_[p]
        if kind[i] != 1:
            continue
        if prev_tc >= 0 and p - prev_tc - 1 > 200:
            var a = ord_[prev_tc + 1]
            var zq = p - 1
            while zq > prev_tc and kind[ord_[zq]] == 4:
                zq -= 1
            var z = ord_[zq]
            if kind[a] != 2 or kind[z] != 3:
                s.bad += 1
            else:
                var ksum = 0
                for q in range(prev_tc + 1, zq + 1):
                    ksum += en[ord_[q]] - st[ord_[q]]
                var w = en[z] - st[a]
                s.nwin += 1
                s.wall_ns += w
                s.gaps_ns += w - ksum
                s.host_ns += st[i] - en[z]
                s.disp.append(Float64(zq - prev_tc))
                s.wms.append(Float64(w) / 1e6)
                if prev_acc_end >= 0:
                    s.iter_span += en[i] - prev_acc_end
                    for q in range(p, -1, -1):
                        var r = ord_[q]
                        if en[r] <= prev_acc_end:
                            break
                        s.iter_ksum += en[r] - max(st[r], prev_acc_end)
                prev_acc_end = en[i]
        prev_tc = p
    return s^


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: mspec_gaps OUTDIR")
        return
    var out = String(args[1])
    var names = read_text(out + "/prompts.txt").split("\n")
    print("g_lo / ub_lo (not preregistered): the tracer's extra decode time D = decode_s(T) - decode_s(B) is charged")
    print("entirely to verify-window gaps, g_lo = (G - D) / (W - D); the untraced share lies in [g_lo, g].")
    print("")
    print("| prompt | k | windows | malformed | disp/win | W ms | g % | g_lo % | iter idle % | host us/win | B tok/s | T/B | ub tok/s | ub/B | ub_lo/B |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    var a_ts = List[Float64]()
    for nm in names:
        var p = String(nm)
        if p.byte_length() == 0:
            continue
        a_ts.append(num_after(read_text(out + "/" + p + ".A.log"), "tok/s_gen:"))
    for k in [2, 4]:
        var g_l = List[Float64]()
        var glo_l = List[Float64]()
        var ub_l = List[Float64]()
        var b_l = List[Float64]()
        var tb_l = List[Float64]()
        var d_l = List[Float64]()
        var w_l = List[Float64]()
        var it_l = List[Float64]()
        var h_l = List[Float64]()
        var urat_l = List[Float64]()
        var ulo_l = List[Float64]()
        var bad = 0
        for nm in names:
            var p = String(nm)
            if p.byte_length() == 0:
                continue
            var tag = String(k)
            var s = analyze(out + "/" + p + ".T" + tag + ".csv")
            var tlog = read_text(out + "/" + p + ".T" + tag + ".log")
            var blog = read_text(out + "/" + p + ".B" + tag + ".log")
            var b = num_after(blog, "tok/s_gen:")
            var t = num_after(tlog, "tok/s_gen:")
            var dec = num_after(tlog, "decode_s:")
            var dec_b = num_after(blog, "decode_s:")
            var gs = Float64(s.gaps_ns) / 1e9
            var ws = Float64(s.wall_ns) / 1e9
            var dx = max(dec - dec_b, 0.0)
            var g = 100.0 * gs / max(ws, 1e-12)
            var glo = 100.0 * max(gs - dx, 0.0) / max(ws - dx, 1e-12)
            var itl = 100.0 * (1.0 - Float64(s.iter_ksum) / Float64(max(s.iter_span, 1)))
            var ub = b / (1.0 - gs / dec)
            var ulo = b / (1.0 - max(gs - dx, 0.0) / dec_b)
            var host = Float64(s.host_ns) / 1e3 / Float64(max(s.nwin, 1))
            var dm = median(s.disp.copy())
            var wm = median(s.wms.copy())
            bad += s.bad
            g_l.append(g)
            glo_l.append(glo)
            ub_l.append(ub)
            b_l.append(b)
            tb_l.append(t / b)
            d_l.append(dm)
            w_l.append(wm)
            it_l.append(itl)
            h_l.append(host)
            urat_l.append(ub / b)
            ulo_l.append(ulo / b)
            print("| " + p + " | " + tag + " | " + String(s.nwin) + " | " + String(s.bad) + " | " + fmt2(dm) + " | " + fmt2(wm)
                  + " | " + fmt2(g) + " | " + fmt2(glo) + " | " + fmt2(itl) + " | " + fmt2(host) + " | " + fmt2(b)
                  + " | " + fmt2(t / b) + " | " + fmt2(ub) + " | " + fmt2(ub / b) + " | " + fmt2(ulo / b) + " |")
        var lo = 1e9
        var hi = -1e9
        for x in g_l:
            lo = min(lo, x)
            hi = max(hi, x)
        var mb = median(b_l^)
        var mu = median(ub_l^)
        print("| **median k=" + String(k) + "** | " + String(k) + " | | " + String(bad) + " | " + fmt2(median(d_l^)) + " | " + fmt2(median(w_l^))
              + " | **" + fmt2(median(g_l.copy())) + "** | " + fmt2(median(glo_l.copy())) + " | " + fmt2(median(it_l^)) + " | " + fmt2(median(h_l^)) + " | " + fmt2(mb)
              + " | " + fmt2(median(tb_l^)) + " | " + fmt2(mu) + " | " + fmt2(median(urat_l^)) + " | " + fmt2(median(ulo_l^)) + " |")
        print("MEDIAN_G k=" + String(k) + " " + fmt2(median(g_l^)) + " (min " + fmt2(lo) + " max " + fmt2(hi) + "), g_lo " + fmt2(median(glo_l^)))
    print("A (no spec) median tok/s_gen " + fmt2(median(a_ts^)))
