# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-json/src/json/value.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
"""A JSON value with Python json's int/float distinction, key order and ensure_ascii output."""

from std.ffi import external_call
from std.memory import bitcast

comptime NULL: UInt8 = 0
comptime BOOL: UInt8 = 1
comptime INT: UInt8 = 2
comptime FLOAT: UInt8 = 3
comptime STR: UInt8 = 4
comptime ARR: UInt8 = 5
comptime OBJ: UInt8 = 6


def _bits(f: Float64) -> Int64:
    return bitcast[DType.int64](f)


def _float(n: Int64) -> Float64:
    return bitcast[DType.float64](n)


@fieldwise_init
struct Node(TrivialRegisterPassable):
    """32 bytes. Text (the member name under an object parent, a string value) lives in the
    owner's byte buffer as an offset and a length."""

    var num: Int64  # an INT, 1/0 for a BOOL, the bits of a FLOAT
    var koff: UInt32
    var klen: UInt32
    var soff: UInt32
    var slen: UInt32
    var size: UInt32  # nodes in this subtree, itself included
    var kind: UInt8


struct Json(Copyable, Movable):
    """A value stored as its nodes in preorder, all text in one byte buffer. `get`, `at` and
    `push` copy a subtree; `parse` makes two allocations however large the document."""

    var n: List[Node]
    var buf: List[UInt8]

    def __init__(out self, kind: UInt8 = NULL):
        self.n = [Node(0, 0, 0, 0, 0, 1, kind)]
        self.buf = List[UInt8]()

    @staticmethod
    def of(v: String) -> Json:
        var j = Json(STR)
        j.buf.extend(v.as_bytes())
        j.n[0].slen = UInt32(v.byte_length())
        return j^

    @staticmethod
    def of(v: Int) -> Json:
        var j = Json(INT)
        j.n[0].num = Int64(v)
        return j^

    @staticmethod
    def of(v: Float64) -> Json:
        var j = Json(FLOAT)
        j.n[0].num = _bits(v)
        return j^

    @staticmethod
    def of(v: Bool) -> Json:
        var j = Json(BOOL)
        j.n[0].num = 1 if v else 0
        return j^

    @staticmethod
    def of(v: Optional[Float64]) -> Json:
        return Json.of(v.value()) if v else Json()

    @staticmethod
    def of(v: Optional[Int]) -> Json:
        return Json.of(v.value()) if v else Json()

    @staticmethod
    def of(v: Optional[String]) -> Json:
        return Json.of(v.value()) if v else Json()

    @staticmethod
    def array() -> Json:
        return Json(ARR)

    @staticmethod
    def object() -> Json:
        return Json(OBJ)

    def kind(self) -> UInt8:
        return self.n[0].kind

    def is_null(self) -> Bool:
        return self.n[0].kind == NULL

    def is_num(self) -> Bool:
        return self.n[0].kind == INT or self.n[0].kind == FLOAT

    def num(self) -> Float64:
        """An INT or FLOAT as a float; 0.0 for any other kind."""
        ref r = self.n[0]
        return Float64(r.num) if r.kind == INT else (_float(r.num) if r.kind == FLOAT else 0.0)

    def int(self) -> Int:
        """An INT or BOOL as is, a FLOAT truncated; 0 for any other kind."""
        ref r = self.n[0]
        return Int(r.num) if r.kind == INT or r.kind == BOOL else (Int(_float(r.num)) if r.kind == FLOAT else 0)

    def str(self) -> String:
        """The text of a STR; empty for any other kind."""
        return self._text(self.n[0].soff, self.n[0].slen)

    def _text(self, off: UInt32, ln: UInt32) -> String:
        return String(unsafe_from_utf8=Span(self.buf)[Int(off) : Int(off) + Int(ln)])

    def len(self) -> Int:
        """Elements of an array, members of an object."""
        var c = 0
        var k = 1
        while k < Int(self.n[0].size):
            c += 1
            k += Int(self.n[k].size)
        return c

    def _adopt(mut self, src: Json, start: Int):
        """Append the subtree of src at node `start`, its text copied into this buffer. The root loses its name."""
        for k in range(start, start + Int(src.n[start].size)):
            var nd = src.n[k]
            var at = len(self.buf)
            self.buf.extend(Span(src.buf)[Int(nd.soff) : Int(nd.soff) + Int(nd.slen)])
            nd.soff = UInt32(at)
            if k == start:
                nd.koff = 0
                nd.klen = 0
            else:
                at = len(self.buf)
                self.buf.extend(Span(src.buf)[Int(nd.koff) : Int(nd.koff) + Int(nd.klen)])
                nd.koff = UInt32(at)
            self.n.append(nd)

    def _sub(self, start: Int) -> Json:
        var j = Json()
        j.n.clear()
        j._adopt(self, start)
        return j^

    def at(self, idx: Int) -> Json:
        """Element or member number idx (a null past the end)."""
        var c = 0
        var k = 1
        while k < Int(self.n[0].size):
            if c == idx:
                return self._sub(k)
            c += 1
            k += Int(self.n[k].size)
        return Json()

    def items(self) -> List[Json]:
        """The elements of an array or the member values of an object, in one pass: O(n) where
        `at(i)` in a loop is O(n^2). Empty for any other kind."""
        var out = List[Json]()
        var k = 1
        while k < Int(self.n[0].size):
            out.append(self._sub(k))
            k += Int(self.n[k].size)
        return out^

    def key_at(self, idx: Int) -> String:
        var c = 0
        var k = 1
        while k < Int(self.n[0].size):
            if c == idx:
                return self._text(self.n[k].koff, self.n[k].klen)
            c += 1
            k += Int(self.n[k].size)
        return String()

    def push(mut self, v: Json):
        self._adopt(v, 0)
        self.n[0].size += v.n[0].size

    def _find(self, key: String) -> Int:
        var kb = key.as_bytes()
        var k = 1
        while k < Int(self.n[0].size):
            ref nd = self.n[k]
            if Int(nd.klen) == len(kb):
                var a = Span(self.buf)[Int(nd.koff) : Int(nd.koff) + len(kb)]
                var same = True
                for i in range(len(kb)):
                    same = same and a[i] == kb[i]
                if same:
                    return k
            k += Int(nd.size)
        return -1

    def set(mut self, key: String, v: Json):
        var at = self._find(key)
        var tail = List[Node]()
        var old = 1
        if at >= 0:
            old = Int(self.n[at].size)
            tail.extend(Span(self.n)[at + old :])
            self.n.shrink(at)
        var first = len(self.n)
        self._adopt(v, 0)
        self.n[first].koff = UInt32(len(self.buf))
        self.n[first].klen = UInt32(key.byte_length())
        self.buf.extend(key.as_bytes())
        self.n.extend(tail^)
        self.n[0].size = UInt32(Int(self.n[0].size) + Int(v.n[0].size) - (old if at >= 0 else 0))

    def has(self, key: String) -> Bool:
        return self._find(key) >= 0

    def get(self, key: String) -> Json:
        """The member, or null (also on a non-object)."""
        var k = self._find(key) if self.n[0].kind == OBJ else -1
        return self._sub(k) if k >= 0 else Json()

    def text(self, key: String, default: String = "") -> String:
        var k = self._find(key)
        return self._text(self.n[k].soff, self.n[k].slen) if k >= 0 and self.n[k].kind == STR else default

    def opt_f(self, key: String) -> Optional[Float64]:
        var k = self._find(key) if self.n[0].kind == OBJ else -1
        if k < 0 or (self.n[k].kind != INT and self.n[k].kind != FLOAT):
            return Optional[Float64]()
        return Optional[Float64](Float64(self.n[k].num) if self.n[k].kind == INT else _float(self.n[k].num))

    def opt_s(self, key: String) -> Optional[String]:
        var k = self._find(key)
        return Optional[String](self._text(self.n[k].soff, self.n[k].slen)) if k >= 0 and self.n[k].kind == STR else Optional[String]()

    def truthy(self) -> Bool:
        """Python truthiness."""
        ref r = self.n[0]
        if r.kind == NULL:
            return False
        if r.kind == BOOL or r.kind == INT:
            return r.num != 0
        if r.kind == FLOAT:
            return _float(r.num) != 0.0
        if r.kind == STR:
            return r.slen > 0
        return r.size > 1

    def dumps(self, compact: Bool = True, indent: Int = -1) -> String:
        """Python's json.dumps(ensure_ascii=True). compact: ',' and ':' separators (the wire); else
        Python's default ', ' and ': ' (the DB). indent >= 0 breaks lines like json.dumps(indent=)."""
        var out = List[UInt8](capacity=len(self.buf) + 16 * len(self.n))
        _ = self._dump(0, out, compact, indent, 0)
        return String(unsafe_from_utf8=Span(out))

    def _dump(self, at: Int, mut out: List[UInt8], compact: Bool, indent: Int, depth: Int) -> Int:
        """Write the value at node `at`; return the index after its subtree."""
        ref nd = self.n[at]
        if nd.kind == NULL:
            out.extend("null".as_bytes())
        elif nd.kind == BOOL:
            out.extend(("true" if nd.num != 0 else "false").as_bytes())
        elif nd.kind == INT:
            out.extend(String(nd.num).as_bytes())
        elif nd.kind == FLOAT:
            _repr(_float(nd.num), out)
        elif nd.kind == STR:
            _quote(Span(self.buf)[Int(nd.soff) : Int(nd.soff) + Int(nd.slen)], out)
        else:
            var is_obj = nd.kind == OBJ
            out.append(UInt8(123 if is_obj else 91))
            var k = at + 1
            var end = at + Int(nd.size)
            while k < end:
                if k > at + 1:
                    out.append(44)
                    if not (compact or indent >= 0):
                        out.append(32)
                if indent >= 0:
                    _newline(out, indent * (depth + 1))
                if is_obj:
                    _quote(Span(self.buf)[Int(self.n[k].koff) : Int(self.n[k].koff) + Int(self.n[k].klen)], out)
                    out.append(58)
                    if not (compact and indent < 0):
                        out.append(32)
                k = self._dump(k, out, compact, indent, depth + 1)
            if indent >= 0 and nd.size > 1:
                _newline(out, indent * depth)
            out.append(UInt8(125 if is_obj else 93))
        return at + Int(nd.size)


def _newline(mut out: List[UInt8], spaces: Int):
    out.append(10)
    for _ in range(spaces):
        out.append(32)


def _repr(f: Float64, mut out: List[UInt8]):
    """Python's repr(float), and json.dumps' NaN / Infinity / -Infinity.

    Mojo's own printing gives the shortest digit count but rounds a tie up where Python rounds it
    to even (25110357248038.5625 prints ...563, Python ...562), so the digits come from printf
    at that count, which rounds the exact binary value half-even. If printf's digits do not read
    back as f (a power of two, whose interval is narrower below), Mojo's text stands."""
    if f != f:
        out.extend("NaN".as_bytes())
        return
    if f > 1.7976931348623157e308:
        out.extend("Infinity".as_bytes())
        return
    if f < -1.7976931348623157e308:
        out.extend("-Infinity".as_bytes())
        return
    var s = String(f)
    var digits = 0  # significant digits of s, trailing zeros not counted
    var lead = True
    var seen = 0
    for c in s.as_bytes():
        if c == 101:
            break
        if c >= 48 and c <= 57 and not (lead and c == 48):
            lead = False
            seen += 1
            if c != 48:
                digits = seen
    if digits == 0:  # 0.0 and -0.0
        out.extend(s.as_bytes())
        return
    var txt = List[UInt8](length=40, fill=0)
    var fmt: List[UInt8] = [37, 46, 42, 101, 0]  # "%.*e"
    _ = external_call["snprintf", Int32, num_fixed_args=3](txt.unsafe_ptr(), Int(40), fmt.unsafe_ptr(), Int32(digits - 1), f)
    _ = fmt[0]
    if external_call["strtod", Float64](txt.unsafe_ptr(), Int(0)) != f:
        out.extend(s.as_bytes())
        return
    # txt is [-]d[.ddd]e[+-]XX
    var i = 0
    if txt[0] == 45:
        out.append(45)
        i = 1
    var mant = List[UInt8](capacity=20)
    while txt[i] != 101:
        if txt[i] != 46:
            mant.append(txt[i])
        i += 1
    var e = 0
    var neg = txt[i + 1] == 45
    i += 2
    while txt[i] != 0:
        e = e * 10 + Int(txt[i] - 48)
        i += 1
    e = -e if neg else e
    if e < -4 or e >= 16:  # scientific: d[.ddd]e+XX
        out.append(mant[0])
        if len(mant) > 1:
            out.append(46)
            out.extend(Span(mant)[1:])
        out.append(101)
        out.append(UInt8(45 if e < 0 else 43))
        if abs(e) < 10:
            out.append(48)
        out.extend(String(abs(e)).as_bytes())
    elif e >= 0:  # digits, the point after e + 1 of them (zeros fill), and at least .0
        for k in range(e + 1):
            out.append(mant[k] if k < len(mant) else 48)
        out.append(46)
        if len(mant) > e + 1:
            out.extend(Span(mant)[e + 1 :])
        else:
            out.append(48)
    else:
        out.extend("0.".as_bytes())
        for _ in range(-e - 1):
            out.append(48)
        out.extend(Span(mant))


def _quote(s: Span[UInt8, _], mut out: List[UInt8]):
    """A JSON string literal: ASCII only, \\uXXXX (surrogate pairs above U+FFFF) for the rest,
    exactly as Python's ensure_ascii; s is valid UTF-8."""
    comptime HEX = "0123456789abcdef".as_bytes()
    out.append(34)
    var i = 0
    while i < len(s):
        var c = Int(s[i])
        i += 1
        if c >= 0x80:  # decode the sequence
            var extra = 1 if c < 0xE0 else (2 if c < 0xF0 else 3)
            c &= 0x3F >> extra
            for _ in range(extra):
                c = (c << 6) | (Int(s[i]) & 0x3F)
                i += 1
        if c == 34 or c == 92:
            out.append(92)
            out.append(UInt8(c))
        elif c >= 32 and c < 127:
            out.append(UInt8(c))
        elif c == 10 or c == 13 or c == 9 or c == 8 or c == 12:
            out.append(92)
            out.append(UInt8(110 if c == 10 else (114 if c == 13 else (116 if c == 9 else (98 if c == 8 else 102)))))
        else:
            var units = 1
            if c > 0xFFFF:
                c -= 0x10000
                units = 2
            for u in range(units):
                var cu = c if units == 1 else (0xD800 + (c >> 10) if u == 0 else 0xDC00 + (c & 0x3FF))
                out.append(92)
                out.append(117)
                for sh in range(4):
                    out.append(HEX[(cu >> (12 - 4 * sh)) & 15])
    out.append(34)
