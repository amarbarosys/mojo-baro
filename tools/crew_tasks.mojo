from std.sys import argv


def rnd(mut st: UInt64, lo: Int, hi: Int) -> Int:
    st = st * 6364136223846793005 + 1442695040888963407
    return lo + Int((st >> 33) % UInt64(hi - lo + 1))


def main() raises:
    var args = argv()
    var L = Int(args[1])
    var N = Int(args[2])
    var st = UInt64(Int(args[3]))
    var names: List[String] = ["a", "b", "c", "d"]
    var hard = len(args) > 4
    var big = hard and String(args[4]) == "mod1000"
    var M = 1000 if big else 100
    var digits = String("three") if big else String("two")
    var out = String("[")
    for i in range(N):
        var v = List[Int]()
        for _ in range(4):
            v.append(rnd(st, 100, 999) if big else rnd(st, 2, 19))
        var p = String("Start: a=") + String(v[0]) + ", b=" + String(v[1]) + ", c=" + String(v[2]) + ", d=" + String(v[3]) + "."
        var note = String("")
        for k in range(L):
            var t = rnd(st, 0, 3)
            var s = rnd(st, 0, 2)
            if s >= t:
                s += 1
            var op = rnd(st, 0, 2)
            if hard:
                var tv = v[t]
                var sv = v[s]
                var raw = tv * sv if op == 2 else tv + sv
                if op == 2:
                    v[t] = (v[t] * v[s]) % M
                    p += " Set " + names[t] + " to " + names[t] + " times " + names[s] + ", keep the last " + digits + " digits."
                else:
                    v[t] = (v[t] + v[s]) % M
                    p += " Set " + names[t] + " to " + names[t] + " plus " + names[s] + ", keep the last " + digits + " digits."
                var sym = String("*") if op == 2 else String("+")
                note += String(k + 1) + ": " + names[t] + " = " + String(tv) + " " + sym + " " + String(sv) + " = " + String(raw) + ", keep " + String(v[t]) + ". State a=" + String(v[0]) + " b=" + String(v[1]) + " c=" + String(v[2]) + " d=" + String(v[3]) + "\\n"
            elif op == 0:
                v[t] += v[s]
                p += " Set " + names[t] + " to " + names[t] + " plus " + names[s] + "."
            elif op == 1:
                v[t] -= v[s]
                p += " Set " + names[t] + " to " + names[t] + " minus " + names[s] + "."
            else:
                var k = rnd(st, 2, 3)
                v[t] *= k
                p += " Set " + names[t] + " to " + names[t] + " times " + String(k) + "."
        p += " What is a+b+c+d? Give only the number after 'ANSWER:'."
        if i > 0:
            out += ","
        out += "{\"id\":\"L" + String(L) + "_" + String(i) + "\",\"L\":" + String(L) + ",\"prompt\":\"" + p + "\",\"note\":\"" + note + "Sum = " + String(v[0] + v[1] + v[2] + v[3]) + "\\nANSWER: " + String(v[0] + v[1] + v[2] + v[3]) + "\",\"expected\":" + String(v[0] + v[1] + v[2] + v[3]) + "}"
    print(out + "]")
