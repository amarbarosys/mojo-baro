"""Digit-level arithmetic expressions with a program checker. No GPU.

Items: a number is 0..99, an operator or paren is 100 + its token id. Tokens: digits 0-9, + - * ( ) = ; and END (V 24).
Families (shape, not variants): 0 flat + and - only; 1 flat with + - * and at least one *; 2 one parenthesised pair, ops + - *.
Chain (cot): EXPR = S1 ; S2 ; ... ; N END, each S is the whole expression after one reduction (innermost paren group first, then
the leftmost *, else the leftmost + or -). nocot: EXPR = N END. Accepted only if every value stays in 0..99.
The checker `value` is an independent shunting-yard evaluator; test_arith.mojo proves it equals the last reduction.
"""

comptime PLUS = 10
comptime MINUS = 11
comptime TIMES = 12
comptime LP = 13
comptime RP = 14
comptime EQ = 15
comptime SEMI = 16
comptime END = 17
comptime OP = 100


struct Rng(Movable):
    var s: UInt64

    def __init__(out self, seed: Int):
        self.s = UInt64(seed) * 0x9E3779B97F4A7C15 + 1

    def below(mut self, n: Int) -> Int:
        self.s ^= self.s << 13
        self.s ^= self.s >> 7
        self.s ^= self.s << 17
        return Int(self.s % UInt64(n))


def render(e: List[Int], mut out: List[Int]):
    for it in e:
        if it >= OP:
            out.append(it - OP)
        elif it >= 10:
            out.append(it // 10)
            out.append(it % 10)
        else:
            out.append(it)


def step(e: List[Int], mut ne: List[Int]) -> Int:
    """One reduction into ne. 1 stepped, 0 already a single number, -1 a value left 0..99."""
    var lo = 0
    var hi = len(e)
    for i in range(len(e)):
        if e[i] == OP + RP:
            hi = i
            lo = i
            while e[lo] != OP + LP:
                lo -= 1
            lo += 1
            break
    var k = -1
    for i in range(lo, hi):
        if e[i] == OP + TIMES:
            k = i
            break
    if k < 0:
        for i in range(lo, hi):
            if e[i] >= OP:
                k = i
                break
    if k < 0:
        return 0
    var a = e[k - 1]
    var b = e[k + 1]
    var v = a * b if e[k] == OP + TIMES else (a + b if e[k] == OP + PLUS else a - b)
    if v < 0 or v > 99:
        return -1
    var cut = hi < len(e) and hi - lo == 3
    ne.clear()
    for i in range(k - 1 - (1 if cut else 0)):
        ne.append(e[i])
    ne.append(v)
    for i in range(k + 2 + (1 if cut else 0), len(e)):
        ne.append(e[i])
    return 1


def value(e: List[Int]) -> Int:
    """Checker: shunting-yard over the items, standard precedence, left to right."""
    var nums = List[Int]()
    var ops = List[Int]()
    for it in e:
        if it < OP:
            nums.append(it)
        elif it == OP + LP:
            ops.append(it)
        else:
            var closing = it == OP + RP
            var pr = 2 if it == OP + TIMES else 1
            while len(ops) > 0 and ops[len(ops) - 1] != OP + LP and (closing or (2 if ops[len(ops) - 1] == OP + TIMES else 1) >= pr):
                var o = ops.pop()
                var b = nums.pop()
                var a = nums.pop()
                nums.append(a * b if o == OP + TIMES else (a + b if o == OP + PLUS else a - b))
            if closing:
                _ = ops.pop()
            else:
                ops.append(it)
    while len(ops) > 0:
        var o = ops.pop()
        var b = nums.pop()
        var a = nums.pop()
        nums.append(a * b if o == OP + TIMES else (a + b if o == OP + PLUS else a - b))
    return nums[0]


def draw(mut rng: Rng, fam: Int, n: Int) -> List[Int]:
    """Random expression of family fam with n operands; not yet filtered for range."""
    var ops = List[Int]()
    var has_t = False
    for _ in range(n - 1):
        var o = rng.below(2) if fam == 0 else rng.below(3)
        ops.append(OP + PLUS + o)
        has_t = has_t or o == 2
    if fam == 1 and not has_t:
        ops[rng.below(n - 1)] = OP + TIMES
    var g = rng.below(n - 1)
    var e = List[Int]()
    for i in range(n):
        if fam == 2 and i == g:
            e.append(OP + LP)
        e.append(rng.below(10))
        if fam == 2 and i == g + 1:
            e.append(OP + RP)
        if i < n - 1:
            e.append(ops[i])
    return e^


struct Sample(Movable):
    var toks: List[Int]
    var plen: Int
    var ans: Int
    var expr: List[Int]

    def __init__(out self):
        self.toks = List[Int]()
        self.plen = 0
        self.ans = 0
        self.expr = List[Int]()


def make(mut rng: Rng, fam: Int, n: Int, cot: Bool) -> Sample:
    """Rejection-sample an in-range expression and write its token sequence."""
    while True:
        var e = draw(rng, fam, n)
        var steps = List[List[Int]]()
        var cur = e.copy()
        var ok = True
        while True:
            var ne = List[Int]()
            var r = step(cur, ne)
            if r < 0:
                ok = False
                break
            if r == 0:
                break
            steps.append(ne.copy())
            cur = ne^
        if not ok:
            continue
        var s = Sample()
        render(e, s.toks)
        s.toks.append(EQ)
        s.plen = len(s.toks)
        s.ans = cur[0]
        if cot:
            for j in range(len(steps)):
                if j > 0:
                    s.toks.append(SEMI)
                render(steps[j], s.toks)
        else:
            render(cur, s.toks)
        s.toks.append(END)
        s.expr = e^
        return s^


def answer(toks: List[Int], plen: Int) -> Int:
    """Parse the generated tail: digits after the last ; (or =) up to END. -1 if there is no END or the tail is not 1-2 digits."""
    var end = -1
    for i in range(plen, len(toks)):
        if toks[i] == END:
            end = i
            break
        if toks[i] < 0:
            return -1
    if end < 0:
        return -1
    var st = plen
    for i in range(plen, end):
        if toks[i] == SEMI:
            st = i + 1
    if end - st < 1 or end - st > 2:
        return -1
    var v = 0
    for i in range(st, end):
        if toks[i] > 9:
            return -1
        v = v * 10 + toks[i]
    return v
