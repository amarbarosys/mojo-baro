# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-uregex/src/uregex/vm.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
# Copyright 2026 amarbaro.org. Licensed under the Apache License, Version 2.0.
from .parser import (
    Ast, Node, N_EMPTY, N_LIT, N_CLASS, N_ANY, N_CAT, N_ALT, N_REPEAT, N_LOOK, N_BOL, N_EOL,
    N_WORDB, N_GROUP,
)
from .classes import CharClass, CaseMap

comptime I_CHAR = 0
comptime I_CLASS = 1
comptime I_ANY = 2
comptime I_SPLIT = 3
comptime I_JMP = 4
comptime I_MATCH = 5
comptime I_LOOK = 6
comptime I_LOOK_END = 7
comptime I_BOL = 8
comptime I_EOL = 9
comptime I_WORDB = 10
comptime I_SAVE = 11
comptime I_LOOKB = 12


@fieldwise_init
struct Inst(Copyable, Movable):
    var op: UInt8
    var ci: Bool
    var neg: Bool
    var a: Int32
    var b: Int32


comptime MAX_PROGRAM = 1 << 20
"""Counted repeats are unrolled; a pattern that compiles to more instructions than this raises (Inst is 12 B)."""


struct Program(Copyable, Movable):
    var code: List[Inst]
    var too_big: Bool
    var classes: List[CharClass]
    var cases: CaseMap
    var groups: Int
    var names: List[String]
    var full: Bool
    """Uses groups, lookbehind or word boundaries: needs exec[True]."""
    var has_first: Bool
    """True when every match starts with a char in the first set below."""
    var first_lo: UInt64
    var first_hi: UInt64
    var first_nonascii: Bool

    def __init__(out self, ast: Ast) raises:
        self.code = List[Inst]()
        self.too_big = False
        self.classes = ast.classes.copy()
        for ref c in self.classes:
            c.build_ascii()
        self.cases = CaseMap()
        self.groups = ast.groups
        self.full = False
        self.names = ast.names.copy()
        self.has_first = False
        self.first_lo = 0
        self.first_hi = 0
        self.first_nonascii = False
        self.emit_node(ast, ast.root)
        if self.too_big:
            raise Error("uregex: pattern too large (counted repeats compile to more than " + String(MAX_PROGRAM) + " instructions)")
        self.code.append(Inst(UInt8(I_MATCH), False, False, 0, 0))
        self.build_first()
        self.full = self.groups > 0
        for ref i in self.code:
            if Int(i.op) == I_LOOKB or Int(i.op) == I_WORDB:
                self.full = True

    def add_first(mut self, cp: Int):
        if cp < 64:
            self.first_lo |= UInt64(1) << UInt64(cp)
        elif cp < 128:
            self.first_hi |= UInt64(1) << UInt64(cp - 64)
        else:
            self.first_nonascii = True

    def build_first(mut self):
        """Set of chars a match can start with. Left disabled when the pattern can match empty
        or starts with `.` or a case-insensitive class."""
        var seen = List[Bool](length=len(self.code), fill=False)
        var work = List[Int]()
        work.append(0)
        while len(work) > 0:
            var pc = work.pop()
            if seen[pc]:
                continue
            seen[pc] = True
            var op = Int(self.code[pc].op)
            var a = Int(self.code[pc].a)
            var ci = self.code[pc].ci
            if op == I_CHAR:
                self.add_first(a)
                if ci:
                    self.add_first(self.cases.fold(a))
                    self.add_first(self.cases.up(a))
                    self.first_nonascii = True
            elif op == I_CLASS:
                if ci:
                    return
                self.first_lo |= self.classes[a].ascii_lo
                self.first_hi |= self.classes[a].ascii_hi
                self.first_nonascii = True
            elif op == I_SPLIT:
                work.append(a)
                work.append(Int(self.code[pc].b))
            elif op == I_JMP or op == I_LOOK or op == I_LOOKB:
                work.append(a)
            elif op == I_BOL or op == I_EOL or op == I_WORDB or op == I_SAVE:
                work.append(pc + 1)
            else:
                return
        self.has_first = True

    @always_inline
    def can_start(self, b0: Byte) -> Bool:
        if b0 >= 128:
            return self.first_nonascii
        if b0 < 64:
            return (self.first_lo >> UInt64(b0)) & 1 == 1
        return (self.first_hi >> UInt64(b0 - 64)) & 1 == 1

    def emit(mut self, op: Int, a: Int = 0, b: Int = 0, ci: Bool = False, neg: Bool = False) -> Int:
        if len(self.code) >= MAX_PROGRAM:
            self.too_big = True
            return 0
        self.code.append(Inst(UInt8(op), ci, neg, Int32(a), Int32(b)))
        return len(self.code) - 1

    def emit_node(mut self, ast: Ast, i: Int):
        if self.too_big:
            return
        var n = ast.nodes[i].copy()
        if n.kind == N_EMPTY:
            return
        if n.kind == N_LIT:
            # No codepoint case-maps to or from an ASCII non-letter, so ci is a no-op there.
            var letter = (n.cp >= 65 and n.cp <= 90) or (n.cp >= 97 and n.cp <= 122)
            _ = self.emit(I_CHAR, n.cp, 0, n.ci and (letter or n.cp >= 128))
        elif n.kind == N_CLASS:
            _ = self.emit(I_CLASS, n.cls, 0, n.ci)
        elif n.kind == N_ANY:
            _ = self.emit(I_ANY)
        elif n.kind == N_BOL:
            _ = self.emit(I_BOL)
        elif n.kind == N_EOL:
            _ = self.emit(I_EOL, 0, 0, False, n.neg)
        elif n.kind == N_CAT:
            for k in n.kids:
                self.emit_node(ast, k)
        elif n.kind == N_ALT:
            var jumps = List[Int]()
            for j in range(len(n.kids)):
                if j < len(n.kids) - 1:
                    var split = self.emit(I_SPLIT)
                    self.code[split].a = Int32(split + 1)
                    self.emit_node(ast, n.kids[j])
                    jumps.append(self.emit(I_JMP))
                    self.code[split].b = Int32(len(self.code))
                else:
                    self.emit_node(ast, n.kids[j])
            for jp in jumps:
                self.code[jp].a = Int32(len(self.code))
        elif n.kind == N_REPEAT:
            for _ in range(n.lo):
                if self.too_big:
                    return
                self.emit_node(ast, n.kids[0])
            # SPLIT tries `a` first: the body when greedy, the exit when lazy.
            if n.hi == -1:
                var split = self.emit(I_SPLIT)
                self.emit_node(ast, n.kids[0])
                _ = self.emit(I_JMP, split)
                self.set_split(split, split + 1, len(self.code), n.neg)
            else:
                var splits = List[Int]()
                for _ in range(n.hi - n.lo):
                    if self.too_big:
                        return
                    splits.append(self.emit(I_SPLIT))
                    self.emit_node(ast, n.kids[0])
                for sp in splits:
                    self.set_split(sp, sp + 1, len(self.code), n.neg)
        elif n.kind == N_LOOK:
            # I_LOOKB: b is the lookbehind width in codepoints.
            var look = self.emit(I_LOOKB if n.lo == 1 else I_LOOK, 0, n.hi, False, n.neg)
            self.emit_node(ast, n.kids[0])
            _ = self.emit(I_LOOK_END)
            self.code[look].a = Int32(len(self.code))
        elif n.kind == N_WORDB:
            _ = self.emit(I_WORDB, n.cls, 0, False, n.neg)
        elif n.kind == N_GROUP:
            _ = self.emit(I_SAVE, 2 * (n.cp - 1))
            self.emit_node(ast, n.kids[0])
            _ = self.emit(I_SAVE, 2 * (n.cp - 1) + 1)

    def set_split(mut self, split: Int, body: Int, exit: Int, lazy: Bool):
        self.code[split].a = Int32(exit if lazy else body)
        self.code[split].b = Int32(body if lazy else exit)

    def char_eq(self, a: Int, b: Int, ci: Bool) -> Bool:
        if a == b:
            return True
        if not ci:
            return False
        return self.cases.fold(a) == self.cases.fold(b) or self.cases.up(a) == self.cases.up(b)

    def class_has(self, cls: Int, cp: Int, ci: Bool) -> Bool:
        if self.classes[cls].contains(cp):
            return True
        if not ci:
            return False
        return self.classes[cls].contains(self.cases.fold(cp)) or self.classes[cls].contains(self.cases.up(cp))

    @no_inline
    def rare_op(self, text: Span[Byte, _], mut stack: List[Int32], pc: Int, pos: Int) -> Int:
        """Lookbehind, word boundary and capture save, kept out of the hot loop. Next pc, or -1 to backtrack."""
        ref ins = self.code[pc]
        var op = Int(ins.op)
        var n = len(text)
        if op == I_LOOKB:
            var mark = len(stack)
            var r = -1
            var from_pos = pos
            for _ in range(Int(ins.b)):
                if from_pos == 0:
                    from_pos = -1
                    break
                from_pos = cp_start_before(text, from_pos)
            if from_pos >= 0:
                r = self.exec[True](text, stack, pc + 1, from_pos, True)
                if r >= 0 and r != pos:
                    undo_to(stack, mark)
                    r = -1
            if (r >= 0) == ins.neg:
                if r >= 0:
                    undo_to(stack, mark)
                return -1
            return Int(ins.a)
        if op == I_WORDB:
            var before = pos > 0 and self.classes[Int(ins.a)].contains(decode_utf8(text, cp_start_before(text, pos))[0])
            var after = pos < n and self.classes[Int(ins.a)].contains(decode_utf8(text, pos)[0])
            return pc + 1 if (before != after) != ins.neg else -1
        # I_SAVE
        var slot = Int(ins.a)
        stack.append(Int32(-1 - slot))
        stack.append(stack[slot])
        stack[slot] = Int32(pos)
        return pc + 1

    def exec[full: Bool](
        self, text: Span[Byte, _], mut stack: List[Int32], start_pc: Int, start_pos: Int, sub: Bool
    ) -> Int:
        """Backtracking run. `stack` is caller-owned scratch. Its first 2 * groups entries are the
        capture slots (group g at 2(g-1), 2(g-1)+1); above them are (pc, pos) choice points, or
        (-1 - slot, old value) entries that restore a slot when popped. Returns the end position or -1.
        A failed run leaves `stack` as found; a successful lookaround run leaves only its restore
        entries, so outer backtracking still undoes its captures. `full=False` (no groups, lookbehind
        or word boundary) compiles all of that out of the loop."""
        var n = len(text)
        var base = len(stack)
        var pc = start_pc
        var pos = start_pos
        while True:
            ref ins = self.code[pc]
            var op = Int(ins.op)
            var ok = True
            if op == I_MATCH:
                if not sub:
                    stack.shrink(base)
                    return pos
                ok = False
            elif op == I_LOOK_END:
                if sub:
                    comptime if full:
                        keep_restores(stack, base)
                    else:
                        stack.shrink(base)
                    return pos
                pc += 1
                continue
            elif op == I_CHAR:
                if pos < n:
                    var d = decode_utf8(text, pos)
                    if self.char_eq(d[0], Int(ins.a), ins.ci):
                        pos += d[1]
                        pc += 1
                    else:
                        ok = False
                else:
                    ok = False
            elif op == I_CLASS:
                if pos < n:
                    var d = decode_utf8(text, pos)
                    if self.class_has(Int(ins.a), d[0], ins.ci):
                        pos += d[1]
                        pc += 1
                    else:
                        ok = False
                else:
                    ok = False
            elif op == I_ANY:
                if pos < n and text[pos] != 10:
                    pos += utf8_len(text[pos])
                    pc += 1
                else:
                    ok = False
            elif op == I_BOL:
                if pos == 0:
                    pc += 1
                else:
                    ok = False
            elif op == I_EOL:   # Python's `$`: the end, or just before a final "\n"; neg (\Z, \z): the end only
                if pos == n or (not ins.neg and pos == n - 1 and text[pos] == 10):
                    pc += 1
                else:
                    ok = False
            elif op == I_JMP:
                pc = Int(ins.a)
            elif op == I_SPLIT:
                stack.append(ins.b)
                stack.append(Int32(pos))
                pc = Int(ins.a)
            elif op == I_LOOK:
                comptime if full:
                    var mark = len(stack)
                    var r = self.exec[full](text, stack, pc + 1, pos, True)
                    if (r >= 0) == ins.neg:
                        if r >= 0:
                            undo_to(stack, mark)
                        ok = False
                    else:
                        pc = Int(ins.a)
                else:
                    var r = self.exec[full](text, stack, pc + 1, pos, True)
                    if (r >= 0) == ins.neg:
                        ok = False
                    else:
                        pc = Int(ins.a)
            elif full:
                var next = self.rare_op(text, stack, pc, pos)
                if next < 0:
                    ok = False
                else:
                    pc = next
            if not ok:
                if len(stack) == base:
                    return -1
                pos = Int(stack.pop())
                pc = Int(stack.pop())
                comptime if full:
                    if pc < 0:
                        var to = unwind(stack, pc, pos, base)
                        if to < 0:
                            return -1
                        pc = Int(stack.pop())
                        pos = to


@no_inline
def unwind(mut stack: List[Int32], pc: Int, pos: Int, base: Int) -> Int:
    """Apply the restore entry (pc, pos) just popped and any below it down to the next choice point.
    Returns that choice point's pos with its pc left on top of `stack`, or -1 at `base`."""
    stack[-1 - pc] = Int32(pos)
    while len(stack) > base:
        var v = stack.pop()
        if stack[len(stack) - 1] >= 0:
            return Int(v)
        var p = Int(stack.pop())
        stack[-1 - p] = v
    return -1


def keep_restores(mut stack: List[Int32], base: Int):
    """Drop the choice points above `base`, keep the capture-restore entries in order."""
    var w = base
    for r in range(base, len(stack), 2):
        if stack[r] < 0:
            stack[w] = stack[r]
            stack[w + 1] = stack[r + 1]
            w += 2
    stack.shrink(w)


def undo_to(mut stack: List[Int32], mark: Int):
    while len(stack) > mark:
        var v = stack.pop()
        var p = Int(stack.pop())
        if p < 0:
            stack[-1 - p] = v


@always_inline
def cp_start_before(text: Span[Byte, _], pos: Int) -> Int:
    var q = pos - 1
    while q > 0 and (text[q] & 0xC0) == 0x80:
        q -= 1
    return q


@always_inline
def utf8_len(b0: Byte) -> Int:
    if b0 < 0x80:
        return 1
    if b0 < 0xE0:
        return 2
    if b0 < 0xF0:
        return 3
    return 4


@always_inline
def decode_utf8(text: Span[Byte, _], pos: Int) -> Tuple[Int, Int]:
    """Codepoint at byte `pos` and its byte length. `text` is valid UTF-8."""
    var b0 = Int(text[pos])
    if b0 < 0x80:
        return (b0, 1)
    if b0 < 0xE0:
        return (((b0 & 0x1F) << 6) | (Int(text[pos + 1]) & 0x3F), 2)
    if b0 < 0xF0:
        return (((b0 & 0x0F) << 12) | ((Int(text[pos + 1]) & 0x3F) << 6) | (Int(text[pos + 2]) & 0x3F), 3)
    return (
        ((b0 & 0x07) << 18) | ((Int(text[pos + 1]) & 0x3F) << 12)
        | ((Int(text[pos + 2]) & 0x3F) << 6) | (Int(text[pos + 3]) & 0x3F),
        4,
    )
