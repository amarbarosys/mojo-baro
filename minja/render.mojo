# Copyright 2026 amarbaro.org. Licensed under the Apache License, Version 2.0.
from std.ffi import external_call
from std.os import getenv
from .value import (
    Heap, parse_json, V_NONE, V_BOOL, V_INT, V_FLOAT, V_STR, V_LIST, V_DICT, V_UNDEF,
)


struct Macro(Copyable, Movable):
    var params: List[String]
    var defaults: List[Int]
    var start: Int
    var end: Int

    def __init__(out self):
        self.params = List[String]()
        self.defaults = List[Int]()
        self.start = 0
        self.end = 0


struct Renderer:
    var src: List[Int]
    var pos: Int
    var h: Heap
    var scopes: List[Int]
    var macro_names: List[String]
    var macros: List[Macro]
    var out: String
    var now: Int
    var eval_off: Int

    def __init__(out self, template: String) raises:
        self.src = List[Int]()
        for c in template.codepoints():
            self.src.append(Int(c.to_u32()))
        self.pos = 0
        self.h = Heap()
        self.scopes = List[Int]()
        self.scopes.append(self.h.new(V_DICT))
        self.macro_names = List[String]()
        self.macros = List[Macro]()
        self.out = String("")
        self.eval_off = 0
        var env_now = getenv("MINJA_NOW", "")
        # strftime_now is a calendar date: wall-clock seconds (CLOCK_REALTIME). perf_counter_ns is time since boot and
        # printed a 1970 date.
        var ts = List[Int64](length=2, fill=0)
        _ = external_call["clock_gettime", Int32](Int32(0), ts.unsafe_ptr())
        self.now = atol(env_now) if env_now != "" else Int(ts[0])

    def set_global(mut self, name: String, val: Int):
        self.h.set(self.scopes[0], name, val)

    # ---------- source helpers ----------
    def at(self, p: Int) -> Int:
        return self.src[p] if p < len(self.src) else -1

    def starts(self, p: Int, lit: String) -> Bool:
        var k = 0
        for c in lit.codepoints():
            if self.at(p + k) != Int(c.to_u32()):
                return False
            k += 1
        return True

    def slice(self, a: Int, b: Int) -> String:
        var s = String("")
        for k in range(a, b):
            s += String(Codepoint.from_u32(UInt32(self.src[k])).value())
        return s^

    def is_space(self, c: Int) -> Bool:
        return c == 32 or c == 9 or c == 10 or c == 13

    def skip_ws(mut self):
        while self.is_space(self.at(self.pos)):
            self.pos += 1

    def is_name_char(self, c: Int, first: Bool) -> Bool:
        if (c >= 97 and c <= 122) or (c >= 65 and c <= 90) or c == 95:
            return True
        return (not first) and c >= 48 and c <= 57

    def read_name(mut self) -> String:
        var a = self.pos
        while self.is_name_char(self.at(self.pos), self.pos == a):
            self.pos += 1
        return self.slice(a, self.pos)

    def peek_name(self) -> String:
        var p = self.pos
        while self.is_name_char(self.at(p), p == self.pos):
            p += 1
        return self.slice(self.pos, p)

    def expect(mut self, lit: String) raises:
        self.skip_ws()
        if not self.starts(self.pos, lit):
            raise Error("minja: expected '" + lit + "' at " + String(self.pos) + " near '" + self.slice(self.pos, min(self.pos + 20, len(self.src))) + "'")
        self.pos += lit.count_codepoints()

    def accept(mut self, lit: String) -> Bool:
        self.skip_ws()
        if self.starts(self.pos, lit):
            self.pos += lit.count_codepoints()
            return True
        return False

    def accept_word(mut self, w: String) -> Bool:
        self.skip_ws()
        if self.peek_name() == w:
            self.pos += w.count_codepoints()
            return True
        return False

    # ---------- scopes ----------
    def lookup(self, name: String) -> Int:
        var i = len(self.scopes) - 1
        while i >= 0:
            var v = self.h.get(self.scopes[i], name)
            if not self.h.is_undef(v):
                return v
            i -= 1
        return self.h.undef()

    def assign(mut self, name: String, val: Int):
        self.h.set(self.scopes[len(self.scopes) - 1], name, val)

    # ---------- top level ----------
    def render(mut self) raises -> String:
        self.run_block(False)
        if self.pos < len(self.src):
            raise Error("minja: unexpected block end at " + String(self.pos))
        return self.out

    def emit_text(mut self, a: Int, b: Int, lstrip_block: Bool, rstrip_all: Bool, skip: Bool):
        if skip:
            return
        var e = b
        if rstrip_all:
            while e > a and self.is_space(self.src[e - 1]):
                e -= 1
        elif lstrip_block:
            var k = e
            while k > a and (self.src[k - 1] == 32 or self.src[k - 1] == 9):
                k -= 1
            if k == a or self.src[k - 1] == 10:
                e = k
        for k in range(a, e):
            self.out += String(Codepoint.from_u32(UInt32(self.src[k])).value())

    def after_tag(mut self, strip_all: Bool, block: Bool):
        if strip_all:
            while self.is_space(self.at(self.pos)):
                self.pos += 1
        elif block and self.at(self.pos) == 10:
            self.pos += 1

    def close_tag(mut self, close: String) raises -> Bool:
        self.skip_ws()
        var strip = False
        if self.at(self.pos) == 45 and self.starts(self.pos + 1, close):
            strip = True
            self.pos += 1
        self.expect(close)
        return strip

    def scan_to(mut self, close: String) -> Bool:
        while self.pos < len(self.src):
            if self.starts(self.pos, close):
                var strip = self.src[self.pos - 1] == 45
                self.pos += close.count_codepoints()
                return strip
            self.pos += 1
        return False

    def run_block(mut self, skip: Bool) raises -> String:
        while self.pos < len(self.src):
            var a = self.pos
            while self.pos < len(self.src) and not (self.src[self.pos] == 123 and (self.at(self.pos + 1) == 123 or self.at(self.pos + 1) == 37 or self.at(self.pos + 1) == 35)):
                self.pos += 1
            var kind = self.at(self.pos + 1)
            var strip_before = self.at(self.pos + 2) == 45
            self.emit_text(a, self.pos, kind != 123, strip_before, skip)
            if self.pos >= len(self.src):
                return String("")
            self.pos += 2
            if strip_before:
                self.pos += 1
            if kind == 35:
                var strip_after = self.scan_to("#}")
                self.after_tag(strip_after, True)
            elif kind == 123:
                if skip:
                    var strip_after = self.scan_to("}}")
                    self.after_tag(strip_after, False)
                else:
                    var v = self.expr()
                    var strip_after = self.close_tag("}}")
                    self.out += self.h.to_str(v)
                    self.after_tag(strip_after, False)
            else:
                self.skip_ws()
                var kw = self.read_name()
                if kw == "endif" or kw == "endfor" or kw == "endmacro" or kw == "else" or kw == "elif":
                    return kw
                self.statement(kw, skip)
        return String("")

    def finish_tag(mut self) raises:
        var strip_after = self.close_tag("%}")
        self.after_tag(strip_after, True)

    def skip_tag_body(mut self):
        _ = self.scan_to("%}")
        var strip_after = self.src[self.pos - 3] == 45
        self.after_tag(strip_after, True)

    def statement(mut self, kw: String, skip: Bool) raises:
        if kw == "if":
            var taken = False
            if skip:
                self.skip_tag_body()
            else:
                taken = self.h.truthy(self.expr())
                self.finish_tag()
            var done = taken
            var end = self.run_block(skip or not taken)
            while end == "elif" or end == "else":
                var this_taken = False
                if end == "elif":
                    if skip or done:
                        self.skip_tag_body()
                    else:
                        this_taken = self.h.truthy(self.expr())
                        self.finish_tag()
                else:
                    this_taken = not done
                    self.finish_tag()
                end = self.run_block(skip or done or not this_taken)
                done = done or this_taken
            if end != "endif":
                raise Error("minja: expected endif")
            self.finish_tag()
        elif kw == "for":
            if skip:
                self.skip_tag_body()
                var end = self.run_block(True)
                if end != "endfor":
                    raise Error("minja: expected endfor")
                self.finish_tag()
                return
            self.skip_ws()
            var n1 = self.read_name()
            var n2 = String("")
            if self.accept(","):
                self.skip_ws()
                n2 = self.read_name()
            if not self.accept_word("in"):
                raise Error("minja: expected 'in' in for")
            var seq = self.expr()
            self.finish_tag()
            var body_start = self.pos
            var items = List[Int]()
            var t = self.h.tag(seq)
            if t == V_LIST:
                items = self.h.v[seq].kids.copy()
            elif t == V_DICT:
                var keys = self.h.v[seq].keys.copy()
                for k in keys:
                    items.append(self.h.string(k))
            elif t == V_STR:
                var src = self.h.v[seq].s.copy()
                for c in src.codepoints():
                    items.append(self.h.string(String(c)))
            elif t != V_NONE and t != V_UNDEF:
                raise Error("minja: for over non-iterable")
            var n = len(items)
            if n == 0:
                var end = self.run_block(True)
                if end != "endfor":
                    raise Error("minja: expected endfor")
                self.finish_tag()
                return
            for i in range(n):
                self.pos = body_start
                var scope = self.h.new(V_DICT)
                self.scopes.append(scope)
                if n2 != "":
                    var pair = items[i]
                    self.assign(n1, self.h.v[pair].kids[0])
                    self.assign(n2, self.h.v[pair].kids[1])
                else:
                    self.assign(n1, items[i])
                var loop = self.h.new(V_DICT)
                self.h.set(loop, "index0", self.h.integer(i))
                self.h.set(loop, "index", self.h.integer(i + 1))
                self.h.set(loop, "first", self.h.boolean(i == 0))
                self.h.set(loop, "last", self.h.boolean(i == n - 1))
                self.h.set(loop, "length", self.h.integer(n))
                self.h.set(loop, "previtem", items[i - 1] if i > 0 else self.h.undef())
                self.h.set(loop, "nextitem", items[i + 1] if i < n - 1 else self.h.undef())
                self.assign("loop", loop)
                var end = self.run_block(False)
                if end != "endfor":
                    raise Error("minja: expected endfor")
                _ = self.scopes.pop()
            self.finish_tag()
        elif kw == "set":
            if skip:
                self.skip_tag_body()
                return
            self.skip_ws()
            var name = self.read_name()
            var attr = String("")
            if self.accept("."):
                attr = self.read_name()
            self.expect("=")
            var v = self.expr()
            self.finish_tag()
            if attr != "":
                var target = self.lookup(name)
                if self.h.tag(target) != V_DICT:
                    raise Error("minja: set on non-namespace " + name)
                self.h.set(target, attr, v)
            else:
                self.assign(name, v)
        elif kw == "macro":
            self.skip_ws()
            var name = self.read_name()
            var m = Macro()
            self.expect("(")
            while not self.accept(")"):
                self.skip_ws()
                m.params.append(self.read_name())
                if self.accept("="):
                    m.defaults.append(self.expr())
                else:
                    m.defaults.append(-1)
                _ = self.accept(",")
            self.finish_tag()
            m.start = self.pos
            var end = self.run_block(True)
            if end != "endmacro":
                raise Error("minja: expected endmacro")
            m.end = self.pos - 8
            self.finish_tag()
            if not skip:
                self.macro_names.append(name)
                self.macros.append(m^)
        else:
            raise Error("minja: unsupported statement '" + kw + "' at " + String(self.pos))

    def call_macro(mut self, idx: Int, args: List[Int]) raises -> Int:
        var m = self.macros[idx].copy()
        var saved_pos = self.pos
        var saved_out = self.out
        var saved_depth = len(self.scopes)
        self.out = String("")
        var scope = self.h.new(V_DICT)
        self.scopes.append(scope)
        for k in range(len(m.params)):
            var v = args[k] if k < len(args) else (m.defaults[k] if m.defaults[k] >= 0 else self.h.undef())
            self.assign(m.params[k], v)
        self.pos = m.start
        var end = self.run_block(False)
        if end != "endmacro":
            raise Error("minja: macro body did not end at endmacro")
        while len(self.scopes) > saved_depth:
            _ = self.scopes.pop()
        var result = self.out
        self.out = saved_out^
        self.pos = saved_pos
        return self.h.string(result)

    # ---------- expressions ----------
    def expr(mut self) raises -> Int:
        var v = self.or_expr()
        if self.accept_word("if"):
            var cond = self.or_expr()
            var other = self.h.undef()
            if self.accept_word("else"):
                var taken = self.h.truthy(cond)
                if taken:
                    self.eval_off += 1
                other = self.expr()
                if taken:
                    self.eval_off -= 1
            return v if self.h.truthy(cond) else other
        return v

    def or_expr(mut self) raises -> Int:
        var v = self.and_expr()
        while self.accept_word("or"):
            var done = self.h.truthy(v)
            if done:
                self.eval_off += 1
            var r = self.and_expr()
            if done:
                self.eval_off -= 1
            else:
                v = r
        return v

    def and_expr(mut self) raises -> Int:
        var v = self.not_expr()
        while self.accept_word("and"):
            var done = not self.h.truthy(v)
            if done:
                self.eval_off += 1
            var r = self.not_expr()
            if done:
                self.eval_off -= 1
            else:
                v = r
        return v

    def not_expr(mut self) raises -> Int:
        if self.accept_word("not"):
            return self.h.boolean(not self.h.truthy(self.not_expr()))
        return self.comparison()

    def comparison(mut self) raises -> Int:
        var v = self.concat()
        while True:
            self.skip_ws()
            if self.accept("=="):
                v = self.h.boolean(self.h.equals(v, self.concat()))
            elif self.accept("!="):
                v = self.h.boolean(not self.h.equals(v, self.concat()))
            elif self.accept("<="):
                v = self.h.boolean(self.num(v) <= self.num(self.concat()))
            elif self.accept(">="):
                v = self.h.boolean(self.num(v) >= self.num(self.concat()))
            elif self.accept("<"):
                v = self.h.boolean(self.num(v) < self.num(self.concat()))
            elif self.accept(">"):
                v = self.h.boolean(self.num(v) > self.num(self.concat()))
            elif self.peek_name() == "in":
                _ = self.accept_word("in")
                v = self.h.boolean(self.contains(self.concat(), v))
            elif self.peek_name() == "not":
                var save = self.pos
                _ = self.accept_word("not")
                if self.accept_word("in"):
                    v = self.h.boolean(not self.contains(self.concat(), v))
                else:
                    self.pos = save
                    return v
            elif self.peek_name() == "is":
                _ = self.accept_word("is")
                var neg = self.accept_word("not")
                self.skip_ws()
                var test = self.read_name()
                var r = self.run_test(test, v)
                v = self.h.boolean(r != neg)
            else:
                return v

    def num(self, v: Int) -> Int:
        var t = self.h.tag(v)
        if t == V_INT or t == V_BOOL:
            return self.h.v[v].n
        if t == V_FLOAT:
            return Int(self.h.v[v].f)
        return 0

    def contains(self, container: Int, item: Int) -> Bool:
        var t = self.h.tag(container)
        if t == V_STR:
            return self.h.v[container].s.find(self.h.to_str(item)) >= 0
        if t == V_DICT:
            var it1 = self.h.v[container].keys.copy()
            for k in it1:
                if k == self.h.to_str(item):
                    return True
            return False
        if t == V_LIST:
            var it2 = self.h.v[container].kids.copy()
            for k in it2:
                if self.h.equals(k, item):
                    return True
        return False

    def run_test(mut self, name: String, v: Int) raises -> Bool:
        var t = self.h.tag(v)
        if name == "defined":
            return t != V_UNDEF
        if name == "undefined":
            return t == V_UNDEF
        if name == "none":
            return t == V_NONE
        if name == "string":
            return t == V_STR
        if name == "mapping":
            return t == V_DICT
        if name == "iterable":
            return t == V_STR or t == V_LIST or t == V_DICT
        if name == "sequence":
            return t == V_STR or t == V_LIST or t == V_DICT
        if name == "number":
            return t == V_INT or t == V_FLOAT or t == V_BOOL
        if name == "boolean":
            return t == V_BOOL
        if name == "true":
            return t == V_BOOL and self.h.v[v].n == 1
        if name == "false":
            return t == V_BOOL and self.h.v[v].n == 0
        if name == "sameas" or name == "eq" or name == "equalto":
            self.expect("(")
            var o = self.expr()
            self.expect(")")
            return self.h.equals(v, o)
        raise Error("minja: unsupported test '" + name + "'")

    def concat(mut self) raises -> Int:
        var v = self.additive()
        while self.accept("~"):
            var r = self.additive()
            v = self.h.string(self.h.to_str(v) + self.h.to_str(r))
        return v

    def additive(mut self) raises -> Int:
        var v = self.term()
        while True:
            self.skip_ws()
            if self.accept("+"):
                var r = self.term()
                if self.h.tag(v) == V_STR or self.h.tag(r) == V_STR:
                    if self.eval_off == 0 and (self.h.tag(v) == V_LIST or self.h.tag(r) == V_LIST or self.h.tag(v) == V_DICT or self.h.tag(r) == V_DICT):
                        raise Error("TypeError: can only concatenate str to str")
                    v = self.h.string(self.h.to_str(v) + self.h.to_str(r))
                elif self.h.tag(v) == V_LIST and self.h.tag(r) == V_LIST:
                    var l = self.h.copy_shallow(v)
                    var it3 = self.h.v[r].kids.copy()
                    for k in it3:
                        self.h.push(l, k)
                    v = l
                else:
                    v = self.h.integer(self.num(v) + self.num(r))
            elif self.at(self.pos) == 45 and self.at(self.pos + 1) != 37 and self.at(self.pos + 1) != 125:
                self.pos += 1
                v = self.h.integer(self.num(v) - self.num(self.term()))
            else:
                return v

    def term(mut self) raises -> Int:
        var v = self.unary()
        while True:
            self.skip_ws()
            if self.accept("*"):
                v = self.h.integer(self.num(v) * self.num(self.unary()))
            elif self.accept("//"):
                v = self.h.integer(self.num(v) // self.num(self.unary()))
            elif self.accept("/"):
                v = self.h.floating(Float64(self.num(v)) / Float64(self.num(self.unary())))
            elif self.at(self.pos) == 37 and self.at(self.pos + 1) != 125:
                self.pos += 1
                v = self.h.integer(self.num(v) % self.num(self.unary()))
            else:
                return v

    def unary(mut self) raises -> Int:
        self.skip_ws()
        if self.at(self.pos) == 45:
            self.pos += 1
            return self.h.integer(-self.num(self.unary()))
        return self.postfix(self.primary())

    def postfix(mut self, base: Int) raises -> Int:
        var v = base
        while True:
            if self.at(self.pos) == 46 and self.is_name_char(self.at(self.pos + 1), True):
                self.pos += 1
                var name = self.read_name()
                if self.at(self.pos) == 40:
                    self.pos += 1
                    var args = self.arg_list()
                    v = self.call_method(v, name, args)
                else:
                    v = self.h.get(v, name)
            elif self.at(self.pos) == 91:
                self.pos += 1
                v = self.subscript(v)
            elif self.at(self.pos) == 40:
                self.pos += 1
                var args = self.arg_list()
                v = self.call_value(v, args)
            elif self.at(self.pos) == 124 or (self.at(self.pos) == 32 and self.starts(self.pos + 1, "|")):
                self.skip_ws()
                self.pos += 1
                self.skip_ws()
                var name = self.read_name()
                var args = List[Int]()
                var kwnames = List[String]()
                if self.at(self.pos) == 40:
                    self.pos += 1
                    args = self.arg_list_kw(kwnames)
                v = self.run_filter(name, v, args, kwnames)
            else:
                return v

    def subscript(mut self, v: Int) raises -> Int:
        self.skip_ws()
        var lo = 0
        var has_lo = False
        var hi = 0
        var has_hi = False
        var step = 1
        var is_slice = False
        if self.at(self.pos) != 58:
            var idx = self.expr()
            self.skip_ws()
            if self.at(self.pos) != 58:
                self.expect("]")
                if self.eval_off > 0:
                    return self.h.undef()
                if self.h.tag(idx) == V_STR:
                    return self.h.get(v, self.h.v[idx].s)
                var i = self.num(idx)
                var n = self.h.length(v)
                if i < 0:
                    i += n
                if i < 0 or i >= n:
                    raise Error("minja: index out of range")
                if self.h.tag(v) == V_STR:
                    var k = 0
                    var it4 = self.h.v[v].s.copy()
                    for c in it4.codepoints():
                        if k == i:
                            return self.h.string(String(c))
                        k += 1
                return self.h.v[v].kids[i]
            lo = self.num(idx)
            has_lo = True
        is_slice = True
        self.expect(":")
        self.skip_ws()
        if self.at(self.pos) != 58 and self.at(self.pos) != 93:
            hi = self.num(self.expr())
            has_hi = True
        self.skip_ws()
        if self.accept(":"):
            self.skip_ws()
            if self.at(self.pos) != 93:
                step = self.num(self.expr())
        self.expect("]")
        if self.eval_off > 0:
            return self.h.undef()
        var n = self.h.length(v)
        var result = self.h.new(self.h.tag(v))
        var idxs = List[Int]()
        if step > 0:
            var a = lo if has_lo else 0
            var b = hi if has_hi else n
            if a < 0:
                a += n
            if b < 0:
                b += n
            var k = max(a, 0)
            while k < min(b, n):
                idxs.append(k)
                k += step
        else:
            var a = lo if has_lo else n - 1
            var b = hi if has_hi else -n - 1
            if a < 0:
                a += n
            if has_hi and b < 0:
                b += n
            var k = min(a, n - 1)
            while k > b and k >= 0:
                idxs.append(k)
                k += step
        if self.h.tag(v) == V_STR:
            var cps = List[String]()
            var it5 = self.h.v[v].s.copy()
            for c in it5.codepoints():
                cps.append(String(c))
            var s = String("")
            for k in idxs:
                s += cps[k]
            self.h.v[result].s = s
        else:
            for k in idxs:
                self.h.push(result, self.h.v[v].kids[k])
        return result

    def arg_list(mut self) raises -> List[Int]:
        var names = List[String]()
        return self.arg_list_kw(names)

    def arg_list_kw(mut self, mut kwnames: List[String]) raises -> List[Int]:
        var args = List[Int]()
        while True:
            self.skip_ws()
            if self.accept(")"):
                return args^
            var save = self.pos
            var name = self.read_name()
            self.skip_ws()
            if name != "" and self.at(self.pos) == 61 and self.at(self.pos + 1) != 61:
                self.pos += 1
                kwnames.append(name)
                args.append(self.expr())
            else:
                self.pos = save
                kwnames.append(String(""))
                args.append(self.expr())
            _ = self.accept(",")

    def call_value(mut self, callee: Int, args: List[Int]) raises -> Int:
        raise Error("minja: value is not callable")

    def call_name(mut self, name: String, args: List[Int], kwnames: List[String]) raises -> Int:
        if self.eval_off > 0:
            return self.h.undef()
        if name == "namespace":
            var d = self.h.new(V_DICT)
            for k in range(len(args)):
                self.h.set(d, kwnames[k], args[k])
            return d
        if name == "raise_exception":
            raise Error("TemplateError: " + self.h.to_str(args[0]))
        if name == "strftime_now":
            return self.h.string(strftime(self.h.to_str(args[0]), self.now))
        if name == "range":
            var l = self.h.new(V_LIST)
            var a = 0
            var b = self.num(args[0])
            if len(args) > 1:
                a = b
                b = self.num(args[1])
            for k in range(a, b):
                self.h.push(l, self.h.integer(k))
            return l
        for k in range(len(self.macro_names)):
            if self.macro_names[k] == name:
                return self.call_macro(k, args)
        raise Error("minja: unknown function '" + name + "'")

    def call_method(mut self, v: Int, name: String, args: List[Int]) raises -> Int:
        if self.eval_off > 0:
            return self.h.undef()
        if self.h.tag(v) == V_DICT:
            if name == "items":
                return self.run_filter("items", v, List[Int](), List[String]())
            if name == "keys":
                var l = self.h.new(V_LIST)
                var keys = self.h.v[v].keys.copy()
                for k in keys:
                    self.h.push(l, self.h.string(k))
                return l
            if name == "get":
                var r = self.h.get(v, self.h.to_str(args[0]))
                if self.h.is_undef(r):
                    return args[1] if len(args) > 1 else self.h.none()
                return r
        if self.h.tag(v) == V_STR:
            var s = self.h.v[v].s
            if name == "strip" or name == "lstrip" or name == "rstrip":
                var chars = self.h.to_str(args[0]) if len(args) > 0 else String(" \t\n\r")
                var cps = List[Int]()
                for c in s.codepoints():
                    cps.append(Int(c.to_u32()))
                var cset = List[Int]()
                for c in chars.codepoints():
                    cset.append(Int(c.to_u32()))
                var a = 0
                var b = len(cps)
                if name != "rstrip":
                    while a < b and _in_set(cps[a], cset):
                        a += 1
                if name != "lstrip":
                    while b > a and _in_set(cps[b - 1], cset):
                        b -= 1
                var r = String("")
                for k in range(a, b):
                    r += String(Codepoint.from_u32(UInt32(cps[k])).value())
                return self.h.string(r)
            if name == "startswith":
                return self.h.boolean(s.startswith(self.h.to_str(args[0])))
            if name == "endswith":
                return self.h.boolean(s.endswith(self.h.to_str(args[0])))
            if name == "split":
                var l = self.h.new(V_LIST)
                if len(args) == 0:
                    for part in s.split():
                        self.h.push(l, self.h.string(String(part)))
                else:
                    for part in s.split(self.h.to_str(args[0])):
                        self.h.push(l, self.h.string(String(part)))
                return l
            if name == "lower":
                return self.h.string(s.lower())
            if name == "upper":
                return self.h.string(s.upper())
            if name == "replace":
                return self.h.string(s.replace(self.h.to_str(args[0]), self.h.to_str(args[1])))
        raise Error("minja: unsupported method ." + name + "()")

    def run_filter(mut self, name: String, v: Int, args: List[Int], kwnames: List[String]) raises -> Int:
        if self.eval_off > 0:
            return self.h.undef()
        if name == "trim":
            if self.h.tag(v) != V_STR:
                return self.call_method(self.h.string(self.h.to_str(v)), "strip", List[Int]())
            return self.call_method(v, "strip", List[Int]())
        if name == "safe" or name == "string" and self.h.tag(v) == V_STR:
            return v
        if name == "string":
            return self.h.string(self.h.to_str(v))
        if name == "length" or name == "count":
            return self.h.integer(self.h.length(v))
        if name == "lower" or name == "upper" or name == "replace":
            return self.call_method(v, name, args)
        if name == "tojson":
            var indent = -1
            for k in range(len(args)):
                if kwnames[k] == "indent" or (kwnames[k] == "" and k == 0):
                    indent = self.num(args[k])
            if indent < 0:
                return self.h.string(self.h.to_json(v))
            return self.h.string(to_json_indent(self.h, v, indent, 0))
        if name == "items":
            var l = self.h.new(V_LIST)
            var keys = self.h.v[v].keys.copy()
            var kids = self.h.v[v].kids.copy()
            for k in range(len(keys)):
                var pair = self.h.new(V_LIST)
                self.h.push(pair, self.h.string(keys[k]))
                self.h.push(pair, kids[k])
                self.h.push(l, pair)
            return l
        if name == "default" or name == "d":
            var dflt = args[0] if len(args) > 0 else self.h.string(String(""))
            var boolean = len(args) > 1 and self.h.truthy(args[1])
            if self.h.is_undef(v) or (boolean and not self.h.truthy(v)):
                return dflt
            return v
        if name == "join":
            var sep = self.h.to_str(args[0]) if len(args) > 0 else String("")
            var s = String("")
            for k in range(len(self.h.v[v].kids)):
                if k > 0:
                    s += sep
                s += self.h.to_str(self.h.v[v].kids[k])
            return self.h.string(s)
        if name == "list":
            return v
        if name == "int":
            return self.h.integer(self.num(v))
        raise Error("minja: unsupported filter '" + name + "'")

    def primary(mut self) raises -> Int:
        self.skip_ws()
        var c = self.at(self.pos)
        if c == 39 or c == 34:
            return self.h.string(self.string_lit())
        if c >= 48 and c <= 57:
            var a = self.pos
            var is_float = False
            while (self.at(self.pos) >= 48 and self.at(self.pos) <= 57) or self.at(self.pos) == 46:
                if self.at(self.pos) == 46:
                    is_float = True
                self.pos += 1
            var txt = self.slice(a, self.pos)
            return self.h.floating(atof(txt)) if is_float else self.h.integer(atol(txt))
        if c == 40:
            self.pos += 1
            var v = self.expr()
            if self.accept(","):
                var l = self.h.new(V_LIST)
                self.h.push(l, v)
                while not self.accept(")"):
                    self.h.push(l, self.expr())
                    _ = self.accept(",")
                return l
            self.expect(")")
            return v
        if c == 91:
            self.pos += 1
            var l = self.h.new(V_LIST)
            while not self.accept("]"):
                self.h.push(l, self.expr())
                _ = self.accept(",")
            return l
        if c == 123:
            self.pos += 1
            var d = self.h.new(V_DICT)
            while not self.accept("}"):
                var k = self.expr()
                self.expect(":")
                var val = self.expr()
                self.h.set(d, self.h.to_str(k), val)
                _ = self.accept(",")
            return d
        var name = self.read_name()
        if name == "":
            raise Error("minja: unexpected character at " + String(self.pos) + " near '" + self.slice(self.pos, min(self.pos + 20, len(self.src))) + "'")
        if name == "true" or name == "True":
            return self.h.boolean(True)
        if name == "false" or name == "False":
            return self.h.boolean(False)
        if name == "none" or name == "None":
            return self.h.none()
        if self.at(self.pos) == 40:
            self.pos += 1
            var kwnames = List[String]()
            var args = self.arg_list_kw(kwnames)
            return self.call_name(name, args, kwnames)
        var v = self.lookup(name)
        if self.h.is_undef(v):
            if name == "strftime_now" or name == "raise_exception" or name == "namespace" or name == "range":
                return self.h.string("<function " + name + ">")
            for k in range(len(self.macro_names)):
                if self.macro_names[k] == name:
                    return self.h.string("<macro " + name + ">")
        return v

    def string_lit(mut self) raises -> String:
        var q = self.at(self.pos)
        self.pos += 1
        var s = String("")
        while True:
            var c = self.at(self.pos)
            if c < 0:
                raise Error("minja: unterminated string")
            self.pos += 1
            if c == q:
                break
            if c == 92:
                var e = self.at(self.pos)
                self.pos += 1
                if e == 110:
                    s += "\n"
                elif e == 116:
                    s += "\t"
                elif e == 114:
                    s += "\r"
                else:
                    s += String(Codepoint.from_u32(UInt32(e)).value())
            else:
                s += String(Codepoint.from_u32(UInt32(c)).value())
        return s^


def _in_set(c: Int, cset: List[Int]) -> Bool:
    for k in cset:
        if k == c:
            return True
    return False


def to_json_indent(h: Heap, v: Int, indent: Int, depth: Int) -> String:
    var t = h.tag(v)
    if (t != V_LIST and t != V_DICT) or len(h.v[v].kids) == 0:
        return h.to_json(v)
    var pad = String("")
    for _ in range((depth + 1) * indent):
        pad += " "
    var pad_close = String("")
    for _ in range(depth * indent):
        pad_close += " "
    var out = String("[\n") if t == V_LIST else String("{\n")
    for k in range(len(h.v[v].kids)):
        if k > 0:
            out += ",\n"
        out += pad
        if t == V_DICT:
            out += _quote_key(h.v[v].keys[k]) + ": "
        out += to_json_indent(h, h.v[v].kids[k], indent, depth + 1)
    out += "\n" + pad_close + ("]" if t == V_LIST else "}")
    return out^


def strftime(fmt: String, epoch: Int) -> String:
    var days = epoch // 86400
    var secs = epoch % 86400
    var z = days + 719468
    var era = z // 146097
    var doe = z - era * 146097
    var yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    var y = yoe + era * 400
    var doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    var mp = (5 * doy + 2) // 153
    var d = doy - (153 * mp + 2) // 5 + 1
    var m = mp + 3 if mp < 10 else mp - 9
    if m <= 2:
        y += 1
    comptime MONTHS = "JanFebMarAprMayJunJulAugSepOctNovDec"
    var out = String("")
    var cps = List[Int]()
    for c in fmt.codepoints():
        cps.append(Int(c.to_u32()))
    var i = 0
    while i < len(cps):
        if cps[i] == 37 and i + 1 < len(cps):
            var f = cps[i + 1]
            i += 2
            if f == 100:
                out += _pad2(d)
            elif f == 109:
                out += _pad2(m)
            elif f == 89:
                out += String(y)
            elif f == 121:
                out += _pad2(y % 100)
            elif f == 98:
                out += String(MONTHS[byte=(m - 1) * 3]) + String(MONTHS[byte=(m - 1) * 3 + 1]) + String(MONTHS[byte=(m - 1) * 3 + 2])
            elif f == 72:
                out += _pad2(secs // 3600)
            elif f == 77:
                out += _pad2((secs // 60) % 60)
            elif f == 83:
                out += _pad2(secs % 60)
            elif f == 37:
                out += "%"
            else:
                out += "%" + String(Codepoint.from_u32(UInt32(f)).value())
        else:
            out += String(Codepoint.from_u32(UInt32(cps[i])).value())
            i += 1
    return out^


def _pad2(n: Int) -> String:
    return ("0" + String(n)) if n < 10 else String(n)


def render_chat(template: String, case_json: String, bos: String, eos: String, pad: String) raises -> String:
    var r = Renderer(template)
    r.set_global("bos_token", r.h.string(bos))
    r.set_global("eos_token", r.h.string(eos))
    r.set_global("pad_token", r.h.string(pad))
    r.set_global("add_generation_prompt", r.h.boolean(True))
    var kw = parse_json(r.h, case_json)
    var keys = r.h.v[kw].keys.copy()
    var kids = r.h.v[kw].kids.copy()
    for k in range(len(keys)):
        r.set_global(keys[k], kids[k])
    return r.render()


def _quote_key(k: String) -> String:
    var h = Heap()
    return h.to_json(h.string(k))
