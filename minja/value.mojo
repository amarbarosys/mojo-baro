# Copyright 2026 amarbaro.org. Licensed under the Apache License, Version 2.0.
comptime V_NONE = 0
comptime V_BOOL = 1
comptime V_INT = 2
comptime V_FLOAT = 3
comptime V_STR = 4
comptime V_LIST = 5
comptime V_DICT = 6
comptime V_UNDEF = 7


struct Val(Copyable, Movable):
    var tag: Int
    var n: Int
    var f: Float64
    var s: String
    var kids: List[Int]
    var keys: List[String]

    def __init__(out self, tag: Int):
        self.tag = tag
        self.n = 0
        self.f = 0
        self.s = String("")
        self.kids = List[Int]()
        self.keys = List[String]()


struct Heap(Movable):
    var v: List[Val]

    def __init__(out self):
        self.v = List[Val]()
        _ = self.new(V_NONE)
        _ = self.new(V_UNDEF)
        _ = self.new(V_BOOL)
        _ = self.new(V_BOOL)
        self.v[3].n = 1

    def new(mut self, tag: Int) -> Int:
        self.v.append(Val(tag))
        return len(self.v) - 1

    def none(self) -> Int:
        return 0

    def undef(self) -> Int:
        return 1

    def boolean(self, b: Bool) -> Int:
        return 3 if b else 2

    def string(mut self, s: String) -> Int:
        var i = self.new(V_STR)
        self.v[i].s = s
        return i

    def integer(mut self, n: Int) -> Int:
        var i = self.new(V_INT)
        self.v[i].n = n
        return i

    def floating(mut self, f: Float64) -> Int:
        var i = self.new(V_FLOAT)
        self.v[i].f = f
        return i

    def tag(self, i: Int) -> Int:
        return self.v[i].tag

    def is_undef(self, i: Int) -> Bool:
        return self.v[i].tag == V_UNDEF

    def truthy(self, i: Int) -> Bool:
        var t = self.v[i].tag
        if t == V_NONE or t == V_UNDEF:
            return False
        if t == V_BOOL or t == V_INT:
            return self.v[i].n != 0
        if t == V_FLOAT:
            return self.v[i].f != 0
        if t == V_STR:
            return self.v[i].s.byte_length() > 0
        return len(self.v[i].kids) > 0

    def get(self, i: Int, key: String) -> Int:
        if self.v[i].tag != V_DICT:
            return 1
        for k in range(len(self.v[i].keys)):
            if self.v[i].keys[k] == key:
                return self.v[i].kids[k]
        return 1

    def set(mut self, i: Int, key: String, val: Int):
        for k in range(len(self.v[i].keys)):
            if self.v[i].keys[k] == key:
                self.v[i].kids[k] = val
                return
        self.v[i].keys.append(key)
        self.v[i].kids.append(val)

    def push(mut self, i: Int, val: Int):
        self.v[i].kids.append(val)

    def length(self, i: Int) -> Int:
        if self.v[i].tag == V_STR:
            return self.v[i].s.count_codepoints()
        return len(self.v[i].kids)

    def copy_shallow(mut self, i: Int) -> Int:
        var j = self.new(self.v[i].tag)
        self.v[j].n = self.v[i].n
        self.v[j].f = self.v[i].f
        self.v[j].s = self.v[i].s
        self.v[j].kids = self.v[i].kids.copy()
        self.v[j].keys = self.v[i].keys.copy()
        return j

    def equals(self, a: Int, b: Int) -> Bool:
        var ta = self.v[a].tag
        var tb = self.v[b].tag
        if (ta == V_INT or ta == V_BOOL) and (tb == V_INT or tb == V_BOOL):
            return self.v[a].n == self.v[b].n
        if ta != tb:
            return False
        if ta == V_STR:
            return self.v[a].s == self.v[b].s
        if ta == V_NONE or ta == V_UNDEF:
            return True
        if ta == V_FLOAT:
            return self.v[a].f == self.v[b].f
        if len(self.v[a].kids) != len(self.v[b].kids):
            return False
        for k in range(len(self.v[a].kids)):
            if ta == V_DICT and self.v[a].keys[k] != self.v[b].keys[k]:
                return False
            if not self.equals(self.v[a].kids[k], self.v[b].kids[k]):
                return False
        return True

    def to_str(self, i: Int) -> String:
        var t = self.v[i].tag
        if t == V_STR:
            return self.v[i].s
        if t == V_NONE:
            return String("None")
        if t == V_UNDEF:
            return String("")
        if t == V_BOOL:
            return String("True") if self.v[i].n != 0 else String("False")
        if t == V_INT:
            return String(self.v[i].n)
        if t == V_FLOAT:
            return String(self.v[i].f)
        return self.to_json(i, True)

    def to_json(self, i: Int, py_repr: Bool = False) -> String:
        var t = self.v[i].tag
        if t == V_STR:
            return _quote(self.v[i].s, py_repr)
        if t == V_NONE or t == V_UNDEF:
            return String("None") if py_repr else String("null")
        if t == V_BOOL:
            if py_repr:
                return String("True") if self.v[i].n != 0 else String("False")
            return String("true") if self.v[i].n != 0 else String("false")
        if t == V_INT:
            return String(self.v[i].n)
        if t == V_FLOAT:
            return String(self.v[i].f)
        var out = String("[") if t == V_LIST else String("{")
        for k in range(len(self.v[i].kids)):
            if k > 0:
                out += ", "
            if t == V_DICT:
                out += _quote(self.v[i].keys[k], py_repr) + ": "
            out += self.to_json(self.v[i].kids[k], py_repr)
        out += "]" if t == V_LIST else "}"
        return out^


def _quote(s: String, py_repr: Bool) -> String:
    var q = String("'") if py_repr and s.find("'") < 0 else String('"')
    var out = q.copy()
    for c in s.codepoints():
        var cp = Int(c.to_u32())
        if cp == 34 and q == '"':
            out += '\\"'
        elif cp == 92:
            out += "\\\\"
        elif cp == 10:
            out += "\\n"
        elif cp == 13:
            out += "\\r"
        elif cp == 9:
            out += "\\t"
        elif cp == 8 and not py_repr:
            out += "\\b"
        elif cp == 12 and not py_repr:
            out += "\\f"
        elif cp < 32:
            out += "\\u00" + _hex2(cp)
        else:
            out += String(c)
    out += q
    return out^


def _hex2(n: Int) -> String:
    comptime digits = "0123456789abcdef"
    return String(digits[byte=n >> 4]) + String(digits[byte=n & 15])


def _hexval(h: Int) -> Int:
    return h - 48 if h <= 57 else (h | 32) - 87


struct JsonParser:
    var s: List[Int]
    var i: Int

    def __init__(out self, text: String):
        self.s = List[Int]()
        for c in text.codepoints():
            self.s.append(Int(c.to_u32()))
        self.i = 0

    def ws(mut self):
        while self.i < len(self.s) and (self.s[self.i] == 32 or self.s[self.i] == 10 or self.s[self.i] == 13 or self.s[self.i] == 9):
            self.i += 1

    def peek(self) -> Int:
        return self.s[self.i] if self.i < len(self.s) else -1

    def lit(mut self, word: String) -> Bool:
        var n = 0
        for c in word.codepoints():
            if self.i + n >= len(self.s) or self.s[self.i + n] != Int(c.to_u32()):
                return False
            n += 1
        self.i += n
        return True

    def string(mut self) raises -> String:
        if self.peek() != 34:
            raise Error("json: expected string at " + String(self.i))
        self.i += 1
        var out = String("")
        while True:
            var c = self.peek()
            if c < 0:
                raise Error("json: unterminated string")
            self.i += 1
            if c == 34:
                break
            if c == 92:
                var e = self.peek()
                self.i += 1
                if e == 110:
                    out += "\n"
                elif e == 116:
                    out += "\t"
                elif e == 114:
                    out += "\r"
                elif e == 98:
                    out += "\x08"
                elif e == 102:
                    out += "\x0c"
                elif e == 117:
                    var cp = 0
                    for _ in range(4):
                        cp = cp * 16 + _hexval(self.peek())
                        self.i += 1
                    if cp >= 0xD800 and cp < 0xDC00 and self.peek() == 92:
                        self.i += 2
                        var lo = 0
                        for _ in range(4):
                            lo = lo * 16 + _hexval(self.peek())
                            self.i += 1
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                    out += String(Codepoint.from_u32(UInt32(cp)).value())
                else:
                    out += String(Codepoint.from_u32(UInt32(e)).value())
            else:
                out += String(Codepoint.from_u32(UInt32(c)).value())
        return out^

    def value(mut self, mut h: Heap) raises -> Int:
        self.ws()
        var c = self.peek()
        if c == 123:
            self.i += 1
            var d = h.new(V_DICT)
            self.ws()
            if self.peek() == 125:
                self.i += 1
                return d
            while True:
                self.ws()
                var k = self.string()
                self.ws()
                if self.peek() != 58:
                    raise Error("json: expected ':'")
                self.i += 1
                var val = self.value(h)
                h.set(d, k, val)
                self.ws()
                if self.peek() == 44:
                    self.i += 1
                    continue
                if self.peek() == 125:
                    self.i += 1
                    return d
                raise Error("json: expected ',' or '}'")
        if c == 91:
            self.i += 1
            var l = h.new(V_LIST)
            self.ws()
            if self.peek() == 93:
                self.i += 1
                return l
            while True:
                var val = self.value(h)
                h.push(l, val)
                self.ws()
                if self.peek() == 44:
                    self.i += 1
                    continue
                if self.peek() == 93:
                    self.i += 1
                    return l
                raise Error("json: expected ',' or ']'")
        if c == 34:
            return h.string(self.string())
        if self.lit("true"):
            return h.boolean(True)
        if self.lit("false"):
            return h.boolean(False)
        if self.lit("null"):
            return h.none()
        var start = self.i
        var is_float = False
        while self.i < len(self.s):
            var d = self.s[self.i]
            if (d >= 48 and d <= 57) or d == 45 or d == 43:
                self.i += 1
            elif d == 46 or d == 101 or d == 69:
                is_float = True
                self.i += 1
            else:
                break
        if self.i == start:
            raise Error("json: unexpected char at " + String(self.i))
        var txt = String("")
        for k in range(start, self.i):
            txt += String(Codepoint.from_u32(UInt32(self.s[k])).value())
        if is_float:
            return h.floating(atof(txt))
        return h.integer(atol(txt))


def parse_json(mut h: Heap, text: String) raises -> Int:
    var p = JsonParser(text)
    var v = p.value(h)
    p.ws()
    if p.i != len(p.s):
        raise Error("json: trailing data")
    return v
