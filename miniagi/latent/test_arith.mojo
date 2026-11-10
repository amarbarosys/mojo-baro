"""CPU check of arithgen: hand cases, reduction chain vs the independent checker, answer parser, lengths. usage: test_arith [N]."""
from std.sys import argv
from arithgen import *



def fail(msg: String) raises:
    raise Error("FAIL " + msg)


def hand() raises:
    var P = OP + PLUS
    var M = OP + MINUS
    var T = OP + TIMES
    var L = OP + LP
    var R = OP + RP
    var e: List[Int] = [3, P, 4, T, 2]
    if value(e) != 11: fail("3+4*2")
    e = [L, 3, P, 4, R, T, 2]
    if value(e) != 14: fail("(3+4)*2")
    e = [9, M, 2, M, 3]
    if value(e) != 4: fail("9-2-3")
    e = [2, T, 3, T, 4, M, 5, T, L, 1, P, 1, R]
    if value(e) != 14: fail("2*3*4-5*(1+1)")
    var ne = List[Int]()
    e = [L, 3, P, 4, R, T, 2]
    if step(e, ne) != 1 or len(ne) != 3 or ne[0] != 7: fail("paren drop")
    e = [3, M, 5, P, 9]
    if step(e, ne) != -1: fail("negative must be refused")
    e = [9, T, 9, T, 9]
    if step(e, ne) != 1 or step(ne.copy(), ne) != -1: fail("243 must be refused")
    var t: List[Int] = [5]
    t = [0, 0, 15, 4, 16, 1, 7, 17]
    if answer(t, 3) != 17: fail("answer parse " + String(answer(t, 3)))
    t = [0, 15, 1, 2, 17]
    if answer(t, 2) != 12: fail("answer 2 digits")
    t = [0, 15, 1, 2, 3, 17]
    if answer(t, 2) != -1: fail("3 digits")
    t = [0, 15, 1, 2]
    if answer(t, 2) != -1: fail("no END")
    t = [0, 15, 1, -1, 17]
    if answer(t, 2) != -1: fail("pad inside")


def main() raises:
    var a = argv()
    var n_samples = Int(String(a[1])) if len(a) > 1 else 20000
    hand()
    var rng = Rng(7)
    for n in range(4, 6):
        for fam in range(3):
            var mx = 0
            var mxn = 0
            var tot = 0
            for _ in range(n_samples):
                var s = make(rng, fam, n, True)
                var z = make(rng, fam, n, False)
                if value(s.expr) != s.ans: fail("chain vs checker fam " + String(fam))
                if answer(s.toks, s.plen) != s.ans: fail("parser vs chain")
                if value(z.expr) != z.ans or answer(z.toks, z.plen) != z.ans: fail("nocot")
                mx = max(mx, len(s.toks))
                mxn = max(mxn, len(z.toks))
                tot += len(s.toks)
            print("n", n, "fam", fam, "samples", n_samples, "cot_len_max", mx, "cot_len_mean", Float64(tot) / Float64(n_samples), "nocot_len_max", mxn)
    var rr = Rng(11)
    for fam in range(3):
        var s = make(rr, fam, 4, True)
        var line = String("fam ") + String(fam) + ":"
        for t in s.toks:
            line += " " + String(t)
        print(line)
    print("OK")
