# Copyright (c) 2026 AMARBARO · amarbaro.org labs. All rights reserved.
from std.collections import Dict


comptime JKindNull: Int = 0
comptime JKindBool: Int = 1
comptime JKindNumber: Int = 2
comptime JKindString: Int = 3
comptime JKindArray: Int = 4
comptime JKindObject: Int = 5


struct JSONValue(Copyable, Movable):
    var kind: Int
    var b: Bool
    var n: Float64
    var s: String
    var arr: List[Int]
    var obj_keys: List[String]
    var obj_vals: List[Int]

    def __init__(out self):
        self.kind = JKindNull
        self.b = False
        self.n = 0.0
        self.s = ""
        self.arr = []
        self.obj_keys = []
        self.obj_vals = []

    def is_null(self) -> Bool:
        return self.kind == JKindNull

    def is_object(self) -> Bool:
        return self.kind == JKindObject

    def is_array(self) -> Bool:
        return self.kind == JKindArray

    def is_string(self) -> Bool:
        return self.kind == JKindString

    def is_number(self) -> Bool:
        return self.kind == JKindNumber

    def is_bool(self) -> Bool:
        return self.kind == JKindBool

    def find_key(self, key: String) -> Int:
        for i in range(len(self.obj_keys)):
            if self.obj_keys[i] == key:
                return self.obj_vals[i]
        return -1


struct JSONDoc(Copyable, Movable):
    var pool: List[JSONValue]
    var root: Int

    def __init__(out self):
        self.pool = []
        self.root = -1

    def push(mut self, var v: JSONValue) -> Int:
        self.pool.append(v^)
        return len(self.pool) - 1

    def get(self, idx: Int) -> JSONValue:
        return self.pool[idx].copy()

    def get_field(self, idx: Int, key: String) -> Int:
        if idx < 0:
            return -1
        return self.pool[idx].find_key(key)

    def has_field(self, idx: Int, key: String) -> Bool:
        return self.get_field(idx, key) >= 0


struct JSONParser:
    var data: List[UInt8]
    var pos: Int

    def __init__(out self, var data: List[UInt8]):
        self.data = data^
        self.pos = 0

    def skip_ws(mut self):
        while self.pos < len(self.data):
            var c = self.data[self.pos]
            if c == 32 or c == 9 or c == 10 or c == 13:
                self.pos += 1
            else:
                break

    def expect(mut self, c: UInt8) raises:
        self.skip_ws()
        if self.pos >= len(self.data) or self.data[self.pos] != c:
            raise Error("json: expected byte " + String(Int(c)) + " at " + String(self.pos))
        self.pos += 1

    def parse_value(mut self, mut doc: JSONDoc) raises -> Int:
        self.skip_ws()
        if self.pos >= len(self.data):
            raise Error("json: unexpected eof")
        var c = self.data[self.pos]
        if c == 123:
            return self.parse_object(doc)
        if c == 91:
            return self.parse_array(doc)
        if c == 34:
            var v = JSONValue()
            v.kind = JKindString
            v.s = self.parse_string_raw()
            return doc.push(v^)
        if c == 116:
            self._expect_lit("true")
            var v = JSONValue()
            v.kind = JKindBool
            v.b = True
            return doc.push(v^)
        if c == 102:
            self._expect_lit("false")
            var v = JSONValue()
            v.kind = JKindBool
            v.b = False
            return doc.push(v^)
        if c == 110:
            self._expect_lit("null")
            return doc.push(JSONValue())
        return self.parse_number(doc)

    def _expect_lit(mut self, lit: String) raises:
        var lb = lit.as_bytes()
        for i in range(len(lb)):
            if self.pos >= len(self.data) or self.data[self.pos] != lb[i]:
                raise Error("json: expected literal " + lit)
            self.pos += 1

    def parse_number(mut self, mut doc: JSONDoc) raises -> Int:
        var start = self.pos
        if self.pos < len(self.data) and self.data[self.pos] == 45:
            self.pos += 1
        while self.pos < len(self.data):
            var c = self.data[self.pos]
            if (c >= 48 and c <= 57) or c == 46 or c == 101 or c == 69 or c == 43 or c == 45:
                self.pos += 1
            else:
                break
        if self.pos == start:
            raise Error("json: bad number at " + String(self.pos))
        var text = String()
        for i in range(start, self.pos):
            text += String(chr(Int(self.data[i])))
        var v = JSONValue()
        v.kind = JKindNumber
        v.n = Float64(text)
        v.s = text
        return doc.push(v^)

    def parse_string_raw(mut self) raises -> String:
        self.expect(34)
        var out: List[UInt8] = []
        while True:
            if self.pos >= len(self.data):
                raise Error("json: unterminated string")
            var c = self.data[self.pos]
            if c == 34:
                self.pos += 1
                break
            if c == 92:
                self.pos += 1
                if self.pos >= len(self.data):
                    raise Error("json: bad escape")
                var e = self.data[self.pos]
                if e == 110:
                    out.append(10)
                elif e == 116:
                    out.append(9)
                elif e == 114:
                    out.append(13)
                elif e == 98:
                    out.append(8)
                elif e == 102:
                    out.append(12)
                elif e == 34:
                    out.append(34)
                elif e == 92:
                    out.append(92)
                elif e == 47:
                    out.append(47)
                elif e == 117:
                    var cp = 0
                    for _ in range(4):
                        self.pos += 1
                        cp = cp * 16 + _hex_val(self.data[self.pos])
                    _append_utf8(out, cp)
                else:
                    out.append(e)
                self.pos += 1
            else:
                out.append(c)
                self.pos += 1
        return String(from_utf8=Span(out))

    def parse_array(mut self, mut doc: JSONDoc) raises -> Int:
        self.expect(91)
        var v = JSONValue()
        v.kind = JKindArray
        self.skip_ws()
        if self.pos < len(self.data) and self.data[self.pos] == 93:
            self.pos += 1
            return doc.push(v^)
        while True:
            var item = self.parse_value(doc)
            v.arr.append(item)
            self.skip_ws()
            if self.pos >= len(self.data):
                raise Error("json: unterminated array")
            if self.data[self.pos] == 44:
                self.pos += 1
                continue
            self.expect(93)
            break
        return doc.push(v^)

    def parse_object(mut self, mut doc: JSONDoc) raises -> Int:
        self.expect(123)
        var v = JSONValue()
        v.kind = JKindObject
        self.skip_ws()
        if self.pos < len(self.data) and self.data[self.pos] == 125:
            self.pos += 1
            return doc.push(v^)
        while True:
            self.skip_ws()
            var key = self.parse_string_raw()
            self.expect(58)
            var val = self.parse_value(doc)
            v.obj_keys.append(key)
            v.obj_vals.append(val)
            self.skip_ws()
            if self.pos >= len(self.data):
                raise Error("json: unterminated object")
            if self.data[self.pos] == 44:
                self.pos += 1
                continue
            self.expect(125)
            break
        return doc.push(v^)


def _append_utf8(mut buf: List[UInt8], cp: Int):
    if cp < 0x80:
        buf.append(UInt8(cp))
    elif cp < 0x800:
        buf.append(UInt8(0xC0 | (cp >> 6)))
        buf.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        buf.append(UInt8(0xE0 | (cp >> 12)))
        buf.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        buf.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        buf.append(UInt8(0xF0 | (cp >> 18)))
        buf.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        buf.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        buf.append(UInt8(0x80 | (cp & 0x3F)))


def _hex_val(c: UInt8) -> Int:
    if c >= 48 and c <= 57:
        return Int(c) - 48
    if c >= 97 and c <= 102:
        return Int(c) - 97 + 10
    if c >= 65 and c <= 70:
        return Int(c) - 65 + 10
    return 0


def parse_json_bytes(var data: List[UInt8]) raises -> JSONDoc:
    var p = JSONParser(data^)
    var doc = JSONDoc()
    var root = p.parse_value(doc)
    doc.root = root
    return doc^


def read_file_bytes(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var text = f.read()
    f.close()
    var b = text.as_bytes()
    var data: List[UInt8] = []
    for i in range(len(b)):
        data.append(b[i])
    return data^


def parse_json_file(path: String) raises -> JSONDoc:
    var data = read_file_bytes(path)
    return parse_json_bytes(data^)
