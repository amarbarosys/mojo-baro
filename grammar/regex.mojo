from std.collections import Dict
from grammar.automaton import Automaton, Rule


comptime RKByteSet: Int = 0
comptime RKConcat: Int = 1
comptime RKAlt: Int = 2
comptime RKStar: Int = 3
comptime RKPlus: Int = 4
comptime RKOpt: Int = 5


struct ReNode(Copyable, Movable):
    var kind: Int
    var lo: List[UInt8]
    var hi: List[UInt8]
    var children: List[Int]

    def __init__(out self):
        self.kind = RKByteSet
        self.lo = []
        self.hi = []
        self.children = []


struct ReDoc(Copyable, Movable):
    var pool: List[ReNode]
    var root: Int

    def __init__(out self):
        self.pool = []
        self.root = -1

    def push(mut self, var n: ReNode) -> Int:
        self.pool.append(n^)
        return len(self.pool) - 1


def expand_repeat(mut doc: ReDoc, atom: Int, mn: Int, mx: Int) raises -> Int:
    if mn > 200 or mx > 200:
        raise Error("regex: repeat count too large (max 200)")
    var n = ReNode()
    n.kind = RKConcat
    for _ in range(mn):
        n.children.append(atom)
    if mx < 0:
        var star = ReNode()
        star.kind = RKStar
        star.children.append(atom)
        n.children.append(doc.push(star^))
    else:
        for _ in range(mx - mn):
            var opt = ReNode()
            opt.kind = RKOpt
            opt.children.append(atom)
            n.children.append(doc.push(opt^))
    if len(n.children) == 0:
        var eps = ReNode()
        eps.kind = RKConcat
        return doc.push(eps^)
    if len(n.children) == 1:
        return n.children[0]
    return doc.push(n^)


struct ReParser:
    var b: List[UInt8]
    var pos: Int
    var doc: ReDoc

    def __init__(out self, pattern: String):
        var bs = pattern.as_bytes()
        self.b = []
        for i in range(len(bs)):
            self.b.append(bs[i])
        self.pos = 0
        self.doc = ReDoc()

    def peek(self) -> Int:
        if self.pos >= len(self.b):
            return -1
        return Int(self.b[self.pos])

    def parse(mut self) raises -> Int:
        var r = self.parse_alt()
        if self.pos != len(self.b):
            raise Error("regex: trailing input at " + String(self.pos))
        return r

    def parse_alt(mut self) raises -> Int:
        var first = self.parse_concat()
        if self.peek() != 124:
            return first
        var n = ReNode()
        n.kind = RKAlt
        n.children.append(first)
        while self.peek() == 124:
            self.pos += 1
            n.children.append(self.parse_concat())
        return self.doc.push(n^)

    def parse_concat(mut self) raises -> Int:
        var n = ReNode()
        n.kind = RKConcat
        while self.peek() >= 0 and self.peek() != 124 and self.peek() != 41:
            n.children.append(self.parse_repeat())
        if len(n.children) == 1:
            return n.children[0]
        return self.doc.push(n^)

    def parse_repeat(mut self) raises -> Int:
        var atom = self.parse_atom()
        while True:
            var c = self.peek()
            if c == 42:
                self.pos += 1
                var n = ReNode()
                n.kind = RKStar
                n.children.append(atom)
                atom = self.doc.push(n^)
            elif c == 43:
                self.pos += 1
                var n = ReNode()
                n.kind = RKPlus
                n.children.append(atom)
                atom = self.doc.push(n^)
            elif c == 63:
                self.pos += 1
                var n = ReNode()
                n.kind = RKOpt
                n.children.append(atom)
                atom = self.doc.push(n^)
            elif c == 123:
                atom = self.parse_brace(atom)
            else:
                break
        return atom

    def parse_brace(mut self, atom: Int) raises -> Int:
        self.pos += 1
        var mn = self._read_int()
        var mx = mn
        if self.peek() == 44:
            self.pos += 1
            if self.peek() == 125:
                mx = -1
            else:
                mx = self._read_int()
        if self.peek() != 125:
            raise Error("regex: expected '}' at " + String(self.pos))
        self.pos += 1
        return self._expand_repeat(atom, mn, mx)

    def _read_int(mut self) raises -> Int:
        var start = self.pos
        while self.peek() >= 48 and self.peek() <= 57:
            self.pos += 1
        if self.pos == start:
            raise Error("regex: expected digits at " + String(self.pos))
        var v = 0
        for i in range(start, self.pos):
            v = v * 10 + (Int(self.b[i]) - 48)
        return v

    def _expand_repeat(mut self, atom: Int, mn: Int, mx: Int) raises -> Int:
        return expand_repeat(self.doc, atom, mn, mx)

    def parse_atom(mut self) raises -> Int:
        var c = self.peek()
        if c < 0:
            raise Error("regex: unexpected end of pattern")
        if c == 40:
            self.pos += 1
            var inner = self.parse_alt()
            if self.peek() != 41:
                raise Error("regex: expected ')' at " + String(self.pos))
            self.pos += 1
            return inner
        if c == 91:
            return self.parse_class()
        if c == 46:
            self.pos += 1
            var n = ReNode()
            n.kind = RKByteSet
            n.lo.append(0)
            n.hi.append(255)
            return self.doc.push(n^)
        if c == 92:
            self.pos += 1
            return self.parse_escape_atom()
        self.pos += 1
        var n = ReNode()
        n.kind = RKByteSet
        n.lo.append(UInt8(c))
        n.hi.append(UInt8(c))
        return self.doc.push(n^)

    def parse_escape_atom(mut self) raises -> Int:
        var e = self.peek()
        if e < 0:
            raise Error("regex: dangling escape")
        self.pos += 1
        var n = ReNode()
        n.kind = RKByteSet
        if e == 100:
            n.lo.append(48); n.hi.append(57)
        elif e == 68:
            n.lo.append(0); n.hi.append(47)
            n.lo.append(58); n.hi.append(255)
        elif e == 119:
            n.lo.append(48); n.hi.append(57)
            n.lo.append(65); n.hi.append(90)
            n.lo.append(95); n.hi.append(95)
            n.lo.append(97); n.hi.append(122)
        elif e == 115:
            n.lo.append(9); n.hi.append(10)
            n.lo.append(13); n.hi.append(13)
            n.lo.append(32); n.hi.append(32)
        elif e == 110:
            n.lo.append(10); n.hi.append(10)
        elif e == 116:
            n.lo.append(9); n.hi.append(9)
        elif e == 114:
            n.lo.append(13); n.hi.append(13)
        else:
            n.lo.append(UInt8(e)); n.hi.append(UInt8(e))
        return self.doc.push(n^)

    def parse_class(mut self) raises -> Int:
        self.pos += 1
        var negate = False
        if self.peek() == 94:
            negate = True
            self.pos += 1
        var lo: List[UInt8] = []
        var hi: List[UInt8] = []
        var first = True
        while self.peek() != 93 or first:
            first = False
            if self.peek() < 0:
                raise Error("regex: unterminated class")
            if self.peek() == 92 and self._is_shorthand_next():
                self.pos += 1
                _ = self._class_escape(lo, hi)
                continue
            var lob: UInt8
            if self.peek() == 92:
                self.pos += 1
                lob = UInt8(self._escape_byte())
            else:
                lob = UInt8(self.peek())
                self.pos += 1
            if self.peek() == 45 and self.pos + 1 < len(self.b) and self.b[self.pos + 1] != 93:
                self.pos += 1
                var hib: UInt8
                if self.peek() == 92:
                    self.pos += 1
                    hib = UInt8(self._escape_byte())
                else:
                    hib = UInt8(self.peek())
                    self.pos += 1
                lo.append(lob)
                hi.append(hib)
            else:
                lo.append(lob)
                hi.append(lob)
        self.pos += 1
        var n = ReNode()
        if negate:
            n.kind = RKByteSet
            var covered: List[Bool] = []
            for _ in range(256):
                covered.append(False)
            for i in range(len(lo)):
                for v in range(Int(lo[i]), Int(hi[i]) + 1):
                    covered[v] = True
            var start = -1
            for v in range(256):
                if not covered[v]:
                    if start < 0:
                        start = v
                elif start >= 0:
                    n.lo.append(UInt8(start)); n.hi.append(UInt8(v - 1))
                    start = -1
            if start >= 0:
                n.lo.append(UInt8(start)); n.hi.append(255)
        else:
            n.kind = RKByteSet
            n.lo = lo^
            n.hi = hi^
        return self.doc.push(n^)

    def _escape_byte(mut self) raises -> Int:
        var e = self.peek()
        self.pos += 1
        if e == 110:
            return 10
        if e == 116:
            return 9
        if e == 114:
            return 13
        return e

    def _is_shorthand_next(self) -> Bool:
        if self.pos + 1 >= len(self.b):
            return False
        var e = Int(self.b[self.pos + 1])
        return e == 100 or e == 119 or e == 115

    def _class_escape(mut self, mut lo: List[UInt8], mut hi: List[UInt8]) raises -> UInt8:
        var e = self.peek()
        self.pos += 1
        if e == 100:
            lo.append(48); hi.append(57)
        elif e == 119:
            lo.append(48); hi.append(57)
            lo.append(65); hi.append(90)
            lo.append(95); hi.append(95)
            lo.append(97); hi.append(122)
        elif e == 115:
            lo.append(9); hi.append(10)
            lo.append(13); hi.append(13)
            lo.append(32); hi.append(32)
        return UInt8(e)


@fieldwise_init
struct Frag(Copyable, Movable):
    var start: Int32
    var accept: Int32


struct NState(Copyable, Movable):
    var eps: List[Int32]
    var blo: List[UInt8]
    var bhi: List[UInt8]
    var btarget: List[Int32]

    def __init__(out self):
        self.eps = []
        self.blo = []
        self.bhi = []
        self.btarget = []


struct NFA(Copyable, Movable):
    var states: List[NState]

    def __init__(out self):
        self.states = []

    def new_state(mut self) -> Int32:
        self.states.append(NState())
        return Int32(len(self.states) - 1)

    def add_eps(mut self, src: Int32, dst: Int32):
        self.states[Int(src)].eps.append(dst)

    def add_byte(mut self, src: Int32, lo: UInt8, hi: UInt8, dst: Int32):
        self.states[Int(src)].blo.append(lo)
        self.states[Int(src)].bhi.append(hi)
        self.states[Int(src)].btarget.append(dst)


def build_frag(doc: ReDoc, node_idx: Int, mut nfa: NFA) -> Frag:
    ref node = doc.pool[node_idx]
    if node.kind == RKByteSet:
        var s = nfa.new_state()
        var e = nfa.new_state()
        for i in range(len(node.lo)):
            nfa.add_byte(s, node.lo[i], node.hi[i], e)
        return Frag(s, e)
    if node.kind == RKConcat:
        if len(node.children) == 0:
            var s = nfa.new_state()
            return Frag(s, s)
        var first = build_frag(doc, node.children[0], nfa)
        var prev_accept = first.accept
        for i in range(1, len(node.children)):
            var f = build_frag(doc, node.children[i], nfa)
            nfa.add_eps(prev_accept, f.start)
            prev_accept = f.accept
        return Frag(first.start, prev_accept)
    if node.kind == RKAlt:
        var s = nfa.new_state()
        var e = nfa.new_state()
        for i in range(len(node.children)):
            var f = build_frag(doc, node.children[i], nfa)
            nfa.add_eps(s, f.start)
            nfa.add_eps(f.accept, e)
        return Frag(s, e)
    if node.kind == RKStar:
        var s = nfa.new_state()
        var e = nfa.new_state()
        var f = build_frag(doc, node.children[0], nfa)
        nfa.add_eps(s, f.start)
        nfa.add_eps(s, e)
        nfa.add_eps(f.accept, f.start)
        nfa.add_eps(f.accept, e)
        return Frag(s, e)
    if node.kind == RKPlus:
        var f = build_frag(doc, node.children[0], nfa)
        var e = nfa.new_state()
        nfa.add_eps(f.accept, f.start)
        nfa.add_eps(f.accept, e)
        return Frag(f.start, e)
    var s = nfa.new_state()
    var e = nfa.new_state()
    var f = build_frag(doc, node.children[0], nfa)
    nfa.add_eps(s, f.start)
    nfa.add_eps(s, e)
    nfa.add_eps(f.accept, e)
    return Frag(s, e)


def eps_closure(nfa: NFA, seed: List[Int32]) -> List[Int32]:
    var seen: List[Bool] = []
    for _ in range(len(nfa.states)):
        seen.append(False)
    var stack: List[Int32] = []
    for i in range(len(seed)):
        var s = seed[i]
        if not seen[Int(s)]:
            seen[Int(s)] = True
            stack.append(s)
    while len(stack) > 0:
        var cur = stack.pop()
        ref st = nfa.states[Int(cur)]
        for i in range(len(st.eps)):
            var t = st.eps[i]
            if not seen[Int(t)]:
                seen[Int(t)] = True
                stack.append(t)
    var out: List[Int32] = []
    for i in range(len(nfa.states)):
        if seen[i]:
            out.append(Int32(i))
    return out^


def closure_key(ids: List[Int32]) -> String:
    var s = String()
    for i in range(len(ids)):
        if i > 0:
            s += ","
        s += String(Int(ids[i]))
    return s


def _contains32(items: List[Int32], v: Int32) -> Bool:
    for i in range(len(items)):
        if items[i] == v:
            return True
    return False


def _byte_targets(nfa: NFA, state_set: List[Int32], b: UInt8) -> List[Int32]:
    var out: List[Int32] = []
    for i in range(len(state_set)):
        ref st = nfa.states[Int(state_set[i])]
        for j in range(len(st.blo)):
            if b >= st.blo[j] and b <= st.bhi[j]:
                out.append(st.btarget[j])
    return out^


def _strip_anchors(pattern: String) raises -> String:
    var b = pattern.as_bytes()
    var n = len(b)
    var lo = 0
    var hi = n
    if n > 0 and b[0] == UInt8(ord("^")):
        lo = 1
    if hi > lo and b[hi - 1] == UInt8(ord("$")) and not (hi - 1 > lo and b[hi - 2] == UInt8(ord("\\"))):
        hi -= 1
    var out: List[UInt8] = []
    for i in range(lo, hi):
        out.append(b[i])
    return String(from_utf8=Span(out))


def compile_regex_to_rule(mut automaton: Automaton, pattern: String, name: String) raises -> Int32:
    # The compiled automaton always matches the pattern against the whole
    # string (no ECMA-262 partial/search semantics this round), so leading
    # `^` / trailing `$` anchors are redundant -- strip them rather than
    # treat them as literal characters.
    var p = ReParser(_strip_anchors(pattern))
    var root = p.parse()
    return compile_ast_to_rule(automaton, p.doc, root, name)


def compile_ast_to_rule(mut automaton: Automaton, doc: ReDoc, root: Int, name: String) raises -> Int32:
    var nfa = NFA()
    var frag = build_frag(doc, root, nfa)
    var accept_state = frag.accept

    var rid = automaton.add_rule(name)

    var dfa_sets: List[List[Int32]] = []
    var dfa_keys: Dict[String, Int] = Dict[String, Int]()

    var start_seed: List[Int32] = [frag.start]
    var start_closure = eps_closure(nfa, start_seed)
    var start_key = closure_key(start_closure)
    dfa_keys[start_key] = 0
    if _contains32(start_closure, accept_state):
        automaton.rules[Int(rid)].set_accept(0)
    dfa_sets.append(start_closure^)

    var queue: List[Int] = [0]
    var qi = 0
    while qi < len(queue):
        var didx = queue[qi]
        qi += 1
        var cur_set = dfa_sets[didx].copy()
        var b = 0
        while b < 256:
            var target = _byte_targets(nfa, cur_set, UInt8(b))
            if len(target) == 0:
                b += 1
                continue
            var tclosure = eps_closure(nfa, target)
            var tkey = closure_key(tclosure)
            var b_start = b
            b += 1
            while b < 256:
                var t2 = _byte_targets(nfa, cur_set, UInt8(b))
                if len(t2) == 0:
                    break
                var c2 = eps_closure(nfa, t2)
                var k2 = closure_key(c2)
                if k2 != tkey:
                    break
                b += 1
            var b_end = b - 1
            var tidx: Int
            if tkey in dfa_keys:
                tidx = dfa_keys[tkey]
            else:
                dfa_sets.append(tclosure^)
                tidx = len(dfa_sets) - 1
                dfa_keys[tkey] = tidx
                _ = automaton.rules[Int(rid)].new_state()
                if _contains32(dfa_sets[tidx], accept_state):
                    automaton.rules[Int(rid)].set_accept(Int32(tidx))
                queue.append(tidx)
            automaton.rules[Int(rid)].add_byte_trans(Int32(didx), UInt8(b_start), UInt8(b_end), Int32(tidx))
    return rid
