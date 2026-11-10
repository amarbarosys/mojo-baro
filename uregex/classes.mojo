# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-uregex/src/uregex/classes.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
# Copyright 2026 amarbaro.org. Licensed under the Apache License, Version 2.0.
from .unicode_tables import category_table, escape_table, lower_table, upper_table


def _hex(s: StringSlice) -> Int:
    var v = 0
    for b in s.as_bytes():
        var c = Int(b)
        if c >= 48 and c <= 57:
            v = v * 16 + (c - 48)
        elif c >= 97 and c <= 102:
            v = v * 16 + (c - 87)
        elif c >= 65 and c <= 70:
            v = v * 16 + (c - 55)
    return v


def parse_ranges(table: StaticString) -> List[Int]:
    var out = List[Int]()
    if table.byte_length() == 0:
        return out^
    for item in table.split(";"):
        var ab = item.split("-")
        out.append(_hex(ab[0]))
        out.append(_hex(ab[1]))
    return out^


struct CharClass(Copyable, Movable):
    var ranges: List[UInt32]
    var negate: Bool
    var ascii_lo: UInt64
    var ascii_hi: UInt64

    def __init__(out self):
        self.ranges = List[UInt32]()
        self.negate = False
        self.ascii_lo = 0
        self.ascii_hi = 0

    def build_ascii(mut self):
        """Cache membership of codepoints 0-127 (negation applied) as a 128-bit map."""
        self.ascii_lo = 0
        self.ascii_hi = 0
        for cp in range(128):
            if self._in(cp) != self.negate:
                if cp < 64:
                    self.ascii_lo |= UInt64(1) << UInt64(cp)
                else:
                    self.ascii_hi |= UInt64(1) << UInt64(cp - 64)

    def add(mut self, lo: Int, hi: Int):
        self.ranges.append(UInt32(lo))
        self.ranges.append(UInt32(hi))

    def add_ranges(mut self, rs: List[Int]):
        for i in range(0, len(rs), 2):
            self.ranges.append(UInt32(rs[i]))
            self.ranges.append(UInt32(rs[i + 1]))

    def normalize(mut self):
        var n = len(self.ranges) // 2
        var idx = List[Int](capacity=n)
        for i in range(n):
            idx.append(i)
        for i in range(1, n):
            var j = i
            while j > 0 and self.ranges[2 * idx[j - 1]] > self.ranges[2 * idx[j]]:
                var t = idx[j]
                idx[j] = idx[j - 1]
                idx[j - 1] = t
                j -= 1
        var merged = List[UInt32]()
        for k in range(n):
            var lo = self.ranges[2 * idx[k]]
            var hi = self.ranges[2 * idx[k] + 1]
            if len(merged) > 0 and lo <= merged[len(merged) - 1] + 1:
                if hi > merged[len(merged) - 1]:
                    merged[len(merged) - 1] = hi
            else:
                merged.append(lo)
                merged.append(hi)
        self.ranges = merged^

    def _in(self, cp: Int) -> Bool:
        var lo = 0
        var hi = len(self.ranges) // 2
        while lo < hi:
            var mid = (lo + hi) // 2
            if cp < Int(self.ranges[2 * mid]):
                hi = mid
            elif cp > Int(self.ranges[2 * mid + 1]):
                lo = mid + 1
            else:
                return True
        return False

    def contains(self, cp: Int) -> Bool:
        if cp < 64:
            return (self.ascii_lo >> UInt64(cp)) & 1 == 1
        if cp < 128:
            return (self.ascii_hi >> UInt64(cp - 64)) & 1 == 1
        return self._in(cp) != self.negate


def _hex_at(t: Span[Byte, _], i: Int, w: Int) -> Int:
    var v = 0
    for k in range(i, i + w):
        var c = Int(t[k])
        v = v * 16 + (c - 48 if c <= 57 else c - 87)
    return v


def _pair_lookup(table: StaticString, cp: Int) -> Int:
    """`table` is sorted fixed-width hex pairs (6+6 chars); value for key `cp`, else `cp`."""
    var t = table.as_bytes()
    var lo = 0
    var hi = len(t) // 12
    while lo < hi:
        var mid = (lo + hi) // 2
        var k = _hex_at(t, mid * 12, 6)
        if cp < k:
            hi = mid
        elif cp > k:
            lo = mid + 1
        else:
            return _hex_at(t, mid * 12 + 6, 6)
    return cp


@fieldwise_init
struct CaseMap(Copyable, Movable):
    """Simple case maps, searched in the static generated tables (no per-instance data)."""

    def fold(self, cp: Int) -> Int:
        if cp < 128:
            return cp + 32 if cp >= 65 and cp <= 90 else cp
        return _pair_lookup(lower_table(), cp)

    def up(self, cp: Int) -> Int:
        if cp < 128:
            return cp - 32 if cp >= 97 and cp <= 122 else cp
        return _pair_lookup(upper_table(), cp)


def category_class(name: String) raises -> List[Int]:
    var t = category_table(name)
    if t.byte_length() == 0:
        raise Error("uregex: unknown Unicode category \\p{" + name + "}")
    return parse_ranges(t)


def escape_class(c: String) raises -> List[Int]:
    var t = escape_table(c)
    if t.byte_length() == 0:
        raise Error("uregex: unknown class escape \\" + c)
    return parse_ranges(t)
