# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-uregex/src/uregex/parser.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
# Copyright 2026 amarbaro.org. Licensed under the Apache License, Version 2.0.
from .classes import CharClass, category_class, escape_class

comptime N_EMPTY = 0
comptime N_LIT = 1
comptime N_CLASS = 2
comptime N_ANY = 3
comptime N_CAT = 4
comptime N_ALT = 5
comptime N_REPEAT = 6
comptime N_LOOK = 7
comptime N_BOL = 8
comptime N_EOL = 9
comptime N_WORDB = 10
comptime N_GROUP = 11


struct Node(Copyable, Movable):
    var kind: Int
    var cp: Int
    var cls: Int
    var ci: Bool
    var neg: Bool
    var lo: Int
    var hi: Int
    var kids: List[Int]

    def __init__(out self, kind: Int):
        self.kind = kind
        self.cp = 0
        self.cls = -1
        self.ci = False
        self.neg = False
        self.lo = 0
        self.hi = 0
        self.kids = List[Int]()


struct Ast(Copyable, Movable):
    var nodes: List[Node]
    var classes: List[CharClass]
    var root: Int
    var groups: Int
    var names: List[String]
    """Group names by index (index 0 unused); "" for unnamed groups."""

    def __init__(out self):
        self.nodes = List[Node]()
        self.classes = List[CharClass]()
        self.root = -1
        self.groups = 0
        self.names = List[String]()
        self.names.append("")

    def add(mut self, var n: Node) -> Int:
        self.nodes.append(n^)
        return len(self.nodes) - 1

    def nullable(self, i: Int) -> Bool:
        var n = self.nodes[i].copy()
        if n.kind == N_EMPTY or n.kind == N_LOOK or n.kind == N_BOL or n.kind == N_EOL or n.kind == N_WORDB:
            return True
        if n.kind == N_GROUP:
            return self.nullable(n.kids[0])
        if n.kind == N_LIT or n.kind == N_CLASS or n.kind == N_ANY:
            return False
        if n.kind == N_CAT:
            for k in n.kids:
                if not self.nullable(k):
                    return False
            return True
        if n.kind == N_ALT:
            for k in n.kids:
                if self.nullable(k):
                    return True
            return False
        return n.lo == 0 or self.nullable(n.kids[0])

    def width(self, i: Int) -> Int:
        """Match length in codepoints if every match has the same length, else -1."""
        ref n = self.nodes[i]
        if n.kind == N_LIT or n.kind == N_CLASS or n.kind == N_ANY:
            return 1
        if n.kind == N_GROUP:
            return self.width(n.kids[0])
        if n.kind == N_CAT:
            var w = 0
            for k in n.kids:
                var kw = self.width(k)
                if kw < 0:
                    return -1
                w += kw
            return w
        if n.kind == N_ALT:
            var w = self.width(n.kids[0])
            for k in n.kids:
                if self.width(k) != w:
                    return -1
            return w
        if n.kind == N_REPEAT:
            var kw = self.width(n.kids[0])
            if n.lo != n.hi or kw < 0:
                return -1
            return n.lo * kw
        return 0


struct Parser:
    var p: List[Int]
    var i: Int
    var ci: Bool
    var ast: Ast
    var captures: Bool
    """False: `(...)` is non-capturing and named groups are rejected."""

    def __init__(out self, pattern: String, captures: Bool = True):
        self.p = List[Int]()
        for c in pattern.codepoints():
            self.p.append(Int(c.to_u32()))
        self.i = 0
        self.ci = False
        self.ast = Ast()
        self.captures = captures

    def peek(self) -> Int:
        return self.p[self.i] if self.i < len(self.p) else -1

    def eat(mut self, cp: Int) -> Bool:
        if self.peek() == cp:
            self.i += 1
            return True
        return False

    def expect(mut self, cp: Int) raises:
        if not self.eat(cp):
            raise Error("uregex: expected '" + String(Codepoint.from_u32(UInt32(cp)).value()) + "' at " + String(self.i))

    def parse(mut self) raises -> Ast:
        var r = self.alternation()
        if self.i != len(self.p):
            raise Error("uregex: unexpected ')' at " + String(self.i))
        self.ast.root = r
        return self.ast.copy()

    def alternation(mut self) raises -> Int:
        var branches = List[Int]()
        branches.append(self.concat())
        while self.eat(124):
            branches.append(self.concat())
        if len(branches) == 1:
            return branches[0]
        var n = Node(N_ALT)
        n.kids = branches^
        return self.ast.add(n^)

    def concat(mut self) raises -> Int:
        var items = List[Int]()
        while True:
            var c = self.peek()
            if c == -1 or c == 124 or c == 41:
                break
            items.append(self.repeat())
        if len(items) == 0:
            return self.ast.add(Node(N_EMPTY))
        if len(items) == 1:
            return items[0]
        var n = Node(N_CAT)
        n.kids = items^
        return self.ast.add(n^)

    def number(mut self) -> Int:
        var v = -1
        while self.peek() >= 48 and self.peek() <= 57:
            v = (0 if v < 0 else v) * 10 + (self.peek() - 48)
            self.i += 1
        return v

    def repeat(mut self) raises -> Int:
        var a = self.atom()
        while True:
            var c = self.peek()
            var lo = 0
            var hi = 0
            if c == 42:
                lo = 0
                hi = -1
            elif c == 43:
                lo = 1
                hi = -1
            elif c == 63:
                lo = 0
                hi = 1
            elif c == 123:
                var save = self.i
                self.i += 1
                lo = self.number()
                if lo < 0:
                    self.i = save
                    return a
                if self.eat(44):
                    hi = self.number()
                else:
                    hi = lo
                if not self.eat(125):
                    self.i = save
                    return a
                if hi != -1 and hi < lo:
                    raise Error("uregex: bad repeat, {" + String(lo) + "," + String(hi) + "} has max below min")
                self.i -= 1
            else:
                return a
            self.i += 1
            if self.peek() == 43:
                raise Error("uregex: possessive quantifiers are not supported")
            var lazy = self.eat(63)
            if hi == -1 and self.ast.nullable(a):
                raise Error("uregex: unbounded repeat of a nullable expression is not supported")
            var n = Node(N_REPEAT)
            n.lo = lo
            n.hi = hi
            n.neg = lazy
            n.kids.append(a)
            a = self.ast.add(n^)

    def atom(mut self) raises -> Int:
        var c = self.peek()
        if c == 40:
            self.i += 1
            var saved_ci = self.ci
            var neg = False
            var look = False
            var behind = False
            var group = -1
            if self.eat(63):
                var f = self.peek()
                if f == 58:
                    self.i += 1
                elif f == 33 or f == 61:
                    self.i += 1
                    look = True
                    neg = f == 33
                elif f == 60 and self.i + 1 < len(self.p) and (self.p[self.i + 1] == 61 or self.p[self.i + 1] == 33):
                    neg = self.p[self.i + 1] == 33
                    self.i += 2
                    look = True
                    behind = True
                elif f == 80 or f == 60:
                    if not self.captures:
                        raise Error("uregex: named groups need FullPattern")
                    self.i += 1
                    if f == 80:
                        self.expect(60)
                    var name = String("")
                    while self.peek() != 62 and self.peek() != -1:
                        name += String(Codepoint.from_u32(UInt32(self.peek())).value())
                        self.i += 1
                    self.expect(62)
                    if name.byte_length() == 0 or name in self.ast.names:
                        raise Error("uregex: bad or duplicate group name at " + String(self.i))
                    self.ast.groups += 1
                    group = self.ast.groups
                    self.ast.names.append(name)
                elif f == 105:
                    self.i += 1
                    self.expect(58)
                    self.ci = True
                else:
                    raise Error("uregex: unsupported group syntax at " + String(self.i))
            elif self.captures:
                self.ast.groups += 1
                group = self.ast.groups
                self.ast.names.append("")
            var inner = self.alternation()
            self.expect(41)
            self.ci = saved_ci
            if look:
                var n = Node(N_LOOK)
                n.neg = neg
                if behind:
                    n.lo = 1
                    n.hi = self.ast.width(inner)
                    if n.hi < 0:
                        raise Error("uregex: lookbehind must have a fixed length")
                n.kids.append(inner)
                return self.ast.add(n^)
            if group > 0:
                var n = Node(N_GROUP)
                n.cp = group
                n.kids.append(inner)
                return self.ast.add(n^)
            return inner
        if c == 91:
            self.i += 1
            return self.char_class()
        if c == 46:
            self.i += 1
            return self.ast.add(Node(N_ANY))
        if c == 94:
            self.i += 1
            return self.ast.add(Node(N_BOL))
        if c == 36:
            self.i += 1
            return self.ast.add(Node(N_EOL))
        if c == 92:
            self.i += 1
            return self.escape_atom()
        if c == 42 or c == 43 or c == 63:
            raise Error("uregex: nothing to repeat at " + String(self.i))
        self.i += 1
        return self.lit(c)

    def lit(mut self, cp: Int) -> Int:
        var n = Node(N_LIT)
        n.cp = cp
        n.ci = self.ci
        return self.ast.add(n^)

    def class_node(mut self, var cc: CharClass) -> Int:
        cc.normalize()
        self.ast.classes.append(cc^)
        var n = Node(N_CLASS)
        n.cls = len(self.ast.classes) - 1
        n.ci = self.ci
        return self.ast.add(n^)

    def property_name(mut self) raises -> String:
        var name = String("")
        if self.eat(123):
            while self.peek() != 125 and self.peek() != -1:
                name += String(Codepoint.from_u32(UInt32(self.peek())).value())
                self.i += 1
            self.expect(125)
        else:
            if self.peek() == -1:
                raise Error("uregex: \\p needs a property name")
            name = String(Codepoint.from_u32(UInt32(self.peek())).value())
            self.i += 1
        return name

    def hex_escape(mut self, digits: Int) raises -> Int:
        var v = 0
        var n = 0
        var braced = False
        if digits == 0:
            self.expect(123)
            braced = True
        while braced or n < digits:
            var c = self.peek()
            if braced and c == 125:
                self.i += 1
                break
            var d = -1
            if c >= 48 and c <= 57:
                d = c - 48
            elif c >= 97 and c <= 102:
                d = c - 87
            elif c >= 65 and c <= 70:
                d = c - 55
            if d < 0:
                raise Error("uregex: bad hex escape at " + String(self.i))
            v = v * 16 + d
            n += 1
            self.i += 1
        return v

    def simple_escape(mut self, c: Int) raises -> Int:
        if c == 110:
            return 10
        if c == 114:
            return 13
        if c == 116:
            return 9
        if c == 102:
            return 12
        if c == 118:
            return 11
        if c == 97:
            return 7
        if c == 101:
            return 27
        if c == 48:
            return 0
        if c == 120:
            return self.hex_escape(0 if self.peek() == 123 else 2)
        if c == 117:
            return self.hex_escape(4)
        if c == 85:
            return self.hex_escape(8)
        return c

    def escape_atom(mut self) raises -> Int:
        var c = self.peek()
        if c == -1:
            raise Error("uregex: trailing backslash")
        self.i += 1
        if c == 112 or c == 80:
            var cc = CharClass()
            cc.add_ranges(category_class(self.property_name()))
            cc.negate = c == 80
            return self.class_node(cc^)
        if c == 100 or c == 119 or c == 115 or c == 68 or c == 87 or c == 83:
            var lower = c if c >= 97 else c + 32
            var cc = CharClass()
            cc.add_ranges(escape_class(String(Codepoint.from_u32(UInt32(lower)).value())))
            cc.negate = c < 97
            return self.class_node(cc^)
        if c == 98 or c == 66:
            var cc = CharClass()
            cc.add_ranges(escape_class("w"))
            cc.normalize()
            self.ast.classes.append(cc^)
            var n = Node(N_WORDB)
            n.cls = len(self.ast.classes) - 1
            n.neg = c == 66
            return self.ast.add(n^)
        if c == 90 or c == 122:   # \Z and \z: the very end, not before a final newline
            var n = Node(N_EOL)
            n.neg = True
            return self.ast.add(n^)
        if (c >= 49 and c <= 57) or (c == 107 and self.peek() == 60):
            raise Error("uregex: backreferences are not supported")
        return self.lit(self.simple_escape(c))

    def char_class(mut self) raises -> Int:
        var cc = CharClass()
        if self.eat(94):
            cc.negate = True
        var first = True
        while True:
            var c = self.peek()
            if c == -1:
                raise Error("uregex: unterminated character class")
            if c == 93 and not first:
                self.i += 1
                break
            first = False
            self.i += 1
            var lo = c
            if c == 92:
                var e = self.peek()
                self.i += 1
                if e == 112 or e == 80:
                    var rs = category_class(self.property_name())
                    if e == 80:
                        var inv = CharClass()
                        inv.add_ranges(rs)
                        inv.normalize()
                        var prev = 0
                        for k in range(0, len(inv.ranges), 2):
                            if Int(inv.ranges[k]) > prev:
                                cc.add(prev, Int(inv.ranges[k]) - 1)
                            prev = Int(inv.ranges[k + 1]) + 1
                        if prev <= 0x10FFFF:
                            cc.add(prev, 0x10FFFF)
                    else:
                        cc.add_ranges(rs)
                    continue
                if e == 100 or e == 119 or e == 115:
                    cc.add_ranges(escape_class(String(Codepoint.from_u32(UInt32(e)).value())))
                    continue
                if e == 68 or e == 87 or e == 83:
                    raise Error("uregex: negated class escapes inside [] are not supported")
                lo = self.simple_escape(e)
            var hi = lo
            if self.peek() == 45 and self.i + 1 < len(self.p) and self.p[self.i + 1] != 93:
                self.i += 1
                var h = self.peek()
                self.i += 1
                if h == 92:
                    var e2 = self.peek()
                    self.i += 1
                    h = self.simple_escape(e2)
                hi = h
                if hi < lo:
                    raise Error("uregex: bad range in character class")
            cc.add(lo, hi)
        return self.class_node(cc^)
