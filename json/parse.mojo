# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-json/src/json/parse.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
"""RFC 8259 parser: strict (no NaN, no trailing commas, no lone surrogates, UTF-8 checked)."""

from std.ffi import external_call
from json.value import Json, Node, _bits, NULL, BOOL, INT, FLOAT, STR, ARR, OBJ

comptime MAX_DEPTH = 1000  # nested arrays and objects; Python's json stops near here too
comptime _Bytes = Pointer[UInt8, ImmUntrackedOrigin]


struct _Parser:
    var src: _Bytes
    var n: Int
    var p: Int
    var out: Json
    var tmp: List[UInt8]  # a NUL-terminated copy of a float for strtod

    def __init__(out self, data: Span[UInt8, _]):
        self.src = _Bytes(unsafe_from_address=Int(data.unsafe_ptr()))
        self.n = len(data)
        self.p = 0
        self.out = Json()
        self.out.n.clear()
        self.tmp = List[UInt8]()
        self.out.n.reserve(len(data) // 8 + 4)  # a guess; the list grows past it

    @always_inline
    def at(self, i: Int) -> UInt8:
        return self.src.unsafe_offset(i)[]

    def finish(deinit self) -> Json:
        return self.out^

    def fail(self, what: String) -> Error:
        return Error("json: ", what, " at byte ", self.p)

    def ws(mut self):
        while self.p < self.n:
            var c = self.at(self.p)
            if c != 32 and c != 10 and c != 13 and c != 9:
                return
            self.p += 1

    def add(mut self, kind: UInt8, koff: Int, klen: Int) -> Int:
        self.out.n.append(Node(0, UInt32(koff), UInt32(klen), 0, 0, 1, kind))
        return len(self.out.n) - 1

    def value(mut self, koff: Int, klen: Int, depth: Int) raises:
        """Parse one value, appending its nodes under the name (koff, klen) in the text buffer."""
        self.ws()
        if self.p >= self.n:
            raise self.fail("unexpected end")
        var c = self.at(self.p)
        if c == 123 or c == 91:
            if depth >= MAX_DEPTH:
                raise self.fail("nesting deeper than " + String(MAX_DEPTH))
            var at = self.add(OBJ if c == 123 else ARR, koff, klen)
            var close = UInt8(125 if c == 123 else 93)
            self.p += 1
            self.ws()
            if self.p < self.n and self.at(self.p) == close:
                self.p += 1
                return
            while True:
                self.ws()
                var ko = 0
                var kl = 0
                if c == 123:
                    if self.p >= self.n or self.at(self.p) != 34:
                        raise self.fail("object key expected")
                    ko = len(self.out.buf)
                    self.string()
                    kl = len(self.out.buf) - ko
                    self.ws()
                    if self.p >= self.n or self.at(self.p) != 58:
                        raise self.fail("':' expected")
                    self.p += 1
                self.value(ko, kl, depth + 1)
                self.ws()
                if self.p < self.n and self.at(self.p) == 44:
                    self.p += 1
                    continue
                if self.p < self.n and self.at(self.p) == close:
                    self.p += 1
                    self.out.n[at].size = UInt32(len(self.out.n) - at)
                    return
                raise self.fail("',' or closing bracket expected")
        elif c == 34:
            var at = self.add(STR, koff, klen)
            var so = len(self.out.buf)
            self.string()
            self.out.n[at].soff = UInt32(so)
            self.out.n[at].slen = UInt32(len(self.out.buf) - so)
        elif self.lit("true"):
            self.out.n[self.add(BOOL, koff, klen)].num = 1
        elif self.lit("false"):
            _ = self.add(BOOL, koff, klen)
        elif self.lit("null"):
            _ = self.add(NULL, koff, klen)
        else:
            self.number(koff, klen)

    def lit(mut self, w: StaticString) -> Bool:
        var wb = w.as_bytes()
        if self.p + len(wb) > self.n:
            return False
        for i in range(len(wb)):
            if self.at(self.p + i) != wb[i]:
                return False
        self.p += len(wb)
        return True

    def digits(mut self) -> Int:
        var st = self.p
        while self.p < self.n and self.at(self.p) >= 48 and self.at(self.p) <= 57:
            self.p += 1
        return self.p - st

    def number(mut self, koff: Int, klen: Int) raises:
        var st = self.p
        var neg = self.p < self.n and self.at(self.p) == 45
        self.p += 1 if neg else 0
        var first = self.p
        if self.p < self.n and self.at(self.p) == 48:
            self.p += 1
        elif self.digits() == 0:
            raise self.fail("value expected")
        var is_float = False
        if self.p < self.n and self.at(self.p) == 46:
            self.p += 1
            is_float = True
            if self.digits() == 0:
                raise self.fail("digit expected after '.'")
        if self.p < self.n and (self.at(self.p) == 101 or self.at(self.p) == 69):
            self.p += 1
            is_float = True
            if self.p < self.n and (self.at(self.p) == 43 or self.at(self.p) == 45):
                self.p += 1
            if self.digits() == 0:
                raise self.fail("digit expected in exponent")
        if not is_float:  # an integer that fits Int64 stays one, as in Python; wider ones become floats
            var acc = UInt64(0)
            var fits = self.p - first <= 19
            if fits:
                for i in range(first, self.p):
                    acc = acc * 10 + UInt64(self.at(i) - 48)
                fits = acc <= (UInt64(1) << 63) - (UInt64(0) if neg else UInt64(1))
            if fits:
                var at = self.add(INT, koff, klen)
                self.out.n[at].num = -Int64(acc) if neg else Int64(acc)
                return
        self.tmp.clear()
        for i in range(st, self.p):
            self.tmp.append(self.at(i))
        self.tmp.append(0)
        var f = external_call["strtod", Float64](self.tmp.unsafe_ptr(), Int(0))
        if f > 1.7976931348623157e308 or f < -1.7976931348623157e308:
            raise self.fail("number out of range")
        self.out.n[self.add(FLOAT, koff, klen)].num = _bits(f)

    def hex4(mut self) raises -> Int:
        if self.p + 4 > self.n:
            raise self.fail("bad \\u escape")
        var v = 0
        for _ in range(4):
            var c = Int(self.at(self.p))
            var d = c - 48 if c >= 48 and c <= 57 else (c - 87 if c >= 97 and c <= 102 else (c - 55 if c >= 65 and c <= 70 else -1))
            if d < 0:
                raise self.fail("bad \\u escape")
            v = v * 16 + d
            self.p += 1
        return v

    def utf8(mut self) raises:
        """Skip one multi-byte UTF-8 sequence at p (RFC 3629: no overlongs, surrogates or > U+10FFFF)."""
        var c = Int(self.at(self.p))
        var extra = 1 if c >= 0xC2 and c <= 0xDF else (2 if c >= 0xE0 and c <= 0xEF else (3 if c >= 0xF0 and c <= 0xF4 else 0))
        if extra == 0 or self.p + extra >= self.n:
            raise self.fail("invalid UTF-8")
        var b1 = Int(self.at(self.p + 1))
        var lo = 0x80
        var hi = 0xBF
        if c == 0xE0: lo = 0xA0
        elif c == 0xED: hi = 0x9F
        elif c == 0xF0: lo = 0x90
        elif c == 0xF4: hi = 0x8F
        if b1 < lo or b1 > hi:
            raise self.fail("invalid UTF-8")
        for i in range(2, extra + 1):
            if (Int(self.at(self.p + i)) & 0xC0) != 0x80:
                raise self.fail("invalid UTF-8")
        self.p += extra + 1

    def push_cp(mut self, cp: Int):
        if cp < 0x80:
            self.out.buf.append(UInt8(cp))
        elif cp < 0x800:
            self.out.buf.append(UInt8(0xC0 | (cp >> 6)))
            self.out.buf.append(UInt8(0x80 | (cp & 0x3F)))
        elif cp < 0x10000:
            self.out.buf.append(UInt8(0xE0 | (cp >> 12)))
            self.out.buf.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
            self.out.buf.append(UInt8(0x80 | (cp & 0x3F)))
        else:
            self.out.buf.append(UInt8(0xF0 | (cp >> 18)))
            self.out.buf.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
            self.out.buf.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
            self.out.buf.append(UInt8(0x80 | (cp & 0x3F)))

    def string(mut self) raises:
        """Decode the string at p (the opening quote) onto the end of the text buffer."""
        self.p += 1
        var run = self.p  # bytes from here to p are copied as they are
        while True:
            if self.p >= self.n:
                raise self.fail("unterminated string")
            var c = self.at(self.p)
            if c == 34:
                break
            if c >= 0x20 and c != 92 and c < 0x80:
                self.p += 1
            elif c < 0x20:
                raise self.fail("control character in string")
            elif c >= 0x80:
                self.utf8()
            else:
                self.out.buf.extend(Span(unsafe_ptr=self.src.unsafe_offset(run), length=self.p - run))
                self.p += 1
                if self.p >= self.n:
                    raise self.fail("unterminated string")
                var e = Int(self.at(self.p))
                self.p += 1
                if e == 117:
                    var cp = self.hex4()
                    if cp >= 0xD800 and cp < 0xDC00:
                        if not (self.p + 1 < self.n and self.at(self.p) == 92 and self.at(self.p + 1) == 117):
                            raise self.fail("lone surrogate")
                        self.p += 2
                        var lo = self.hex4()
                        if lo < 0xDC00 or lo > 0xDFFF:
                            raise self.fail("lone surrogate")
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                    elif cp >= 0xDC00 and cp <= 0xDFFF:
                        raise self.fail("lone surrogate")
                    self.push_cp(cp)
                elif e == 34 or e == 92 or e == 47:
                    self.out.buf.append(UInt8(e))
                elif e == 98 or e == 102 or e == 110 or e == 114 or e == 116:
                    self.out.buf.append(UInt8(8 if e == 98 else (12 if e == 102 else (10 if e == 110 else (13 if e == 114 else 9)))))
                else:
                    self.p -= 1
                    raise self.fail("bad escape")
                run = self.p
        self.out.buf.extend(Span(unsafe_ptr=self.src.unsafe_offset(run), length=self.p - run))
        self.p += 1


def parse(data: Span[UInt8, _]) raises -> Json:
    """Parse one JSON document from bytes. Raises `json: <what> at byte N` on anything RFC 8259
    rejects, on invalid UTF-8, on a number that overflows a float and on nesting past MAX_DEPTH."""
    var p = _Parser(data)
    p.value(0, 0, 0)
    p.ws()
    if p.p != p.n:
        raise p.fail("extra data")
    _ = data[0] if len(data) > 0 else 0  # the bytes stay alive until here
    return p^.finish()


def parse(text: String) raises -> Json:
    return parse(text.as_bytes())
