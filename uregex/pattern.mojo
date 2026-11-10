# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-uregex/src/uregex/pattern.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
# Copyright 2026 amarbaro.org. Licensed under the Apache License, Version 2.0.
from .parser import Parser
from .vm import Program, utf8_len


@fieldwise_init
struct Match(Copyable, Movable, ImplicitlyCopyable):
    var start: Int
    var end: Int


def from_codepoints(cps: List[Int], start: Int, end: Int) -> String:
    var s = String("")
    for i in range(start, end):
        s += String(Codepoint.from_u32(UInt32(cps[i])).value())
    return s^


def _slice(text: String, start: Int, end: Int) -> String:
    return String(unsafe_from_utf8=text.as_bytes()[start:end])


struct PatternT[full: Bool](Copyable, Movable):
    """Matches on UTF-8 bytes. `Match` offsets are byte offsets into the input String.

    Use the aliases. `Pattern` is the tokenizer engine: `(...)` does not capture, and named groups,
    lookbehind and word boundaries raise. `FullPattern` supports them; that machinery is compiled
    only into that type, so `Pattern` pays nothing for it."""

    var prog: Program
    var source: String

    def __init__(out self, pattern: String) raises:
        var p = Parser(pattern, Self.full)
        var ast = p.parse()
        self.prog = Program(ast)
        self.source = pattern
        comptime if not Self.full:
            if self.prog.full:
                raise Error("uregex: lookbehind and word boundaries need FullPattern")

    def groups(self) -> Int:
        """Number of capture groups."""
        return self.prog.groups

    def group_index(self, name: String) raises -> Int:
        for i in range(1, len(self.prog.names)):
            if self.prog.names[i] == name:
                return i
        raise Error("uregex: no group named " + name)

    def new_stack(self) -> List[Int32]:
        """Backtrack scratch with the capture slots at the bottom, all unset (-1)."""
        return List[Int32](length=2 * self.prog.groups, fill=-1)

    def search_from(self, text: Span[Byte, _], mut stack: List[Int32], start: Int) -> Optional[Match]:
        """`stack` comes from `new_stack`; on a match its slots hold the groups."""
        var n = len(text)
        var s = start
        if self.prog.has_first:
            while s < n:
                if self.prog.can_start(text[s]):
                    var e = self.prog.exec[Self.full](text, stack, 0, s, False)
                    if e >= 0:
                        return Match(s, e)
                s += utf8_len(text[s])
            return None
        while True:
            var e = self.prog.exec[Self.full](text, stack, 0, s, False)
            if e >= 0:
                return Match(s, e)
            if s >= n:
                return None
            s += utf8_len(text[s])

    def search(self, text: String) -> Optional[Match]:
        var stack = self.new_stack()
        return self.search_from(text.as_bytes(), stack, 0)

    def findall_spans(self, text: String) -> List[Match]:
        var bytes = text.as_bytes()
        var n = len(bytes)
        var out = List[Match]()
        var stack = self.new_stack()
        var pos = 0
        while pos <= n:
            var m = self.search_from(bytes, stack, pos)
            if not m:
                break
            var mm = m.value()
            out.append(mm)
            if mm.end > mm.start:
                pos = mm.end
            elif mm.end < n:
                pos = mm.end + utf8_len(bytes[mm.end])
            else:
                break
        return out^

    def findall_groups(self, text: String) -> List[List[Match]]:
        """Every match as byte spans: [0] the whole match, [i] group i, (-1, -1) when unset."""
        var bytes = text.as_bytes()
        var n = len(bytes)
        var out = List[List[Match]]()
        var stack = self.new_stack()
        var pos = 0
        while pos <= n:
            for i in range(len(stack)):
                stack[i] = -1
            var m = self.search_from(bytes, stack, pos)
            if not m:
                break
            var mm = m.value()
            var row = List[Match](capacity=self.prog.groups + 1)
            row.append(mm)
            for g in range(1, self.prog.groups + 1):
                row.append(Match(Int(stack[2 * g - 2]), Int(stack[2 * g - 1])))
            out.append(row^)
            if mm.end > mm.start:
                pos = mm.end
            elif mm.end < n:
                pos = mm.end + utf8_len(bytes[mm.end])
            else:
                break
        return out^

    def findall_cps_offsets(self, text: String) -> List[Match]:
        """Like `findall_spans`, with codepoint offsets."""
        var bytes = text.as_bytes()
        var out = List[Match]()
        var b = 0
        var cp = 0
        for m in self.findall_spans(text):
            while b < m.start:
                b += utf8_len(bytes[b])
                cp += 1
            var s = cp
            while b < m.end:
                b += utf8_len(bytes[b])
                cp += 1
            out.append(Match(s, cp))
        return out^

    def findall_cps(self, text: List[Int]) -> List[Match]:
        """Compatibility: codepoint input, codepoint offsets."""
        return self.findall_cps_offsets(from_codepoints(text, 0, len(text)))

    def findall(self, text: String) -> List[String]:
        var out = List[String]()
        for m in self.findall_spans(text):
            out.append(_slice(text, m.start, m.end))
        return out^

    def split_keep(self, text: String) -> List[String]:
        var out = List[String]()
        var last = 0
        for m in self.findall_spans(text):
            if m.start > last:
                out.append(_slice(text, last, m.start))
            if m.end > m.start:
                out.append(_slice(text, m.start, m.end))
            last = m.end
        if last < text.byte_length():
            out.append(_slice(text, last, text.byte_length()))
        return out^


comptime Pattern = PatternT[False]
comptime FullPattern = PatternT[True]


# One-shot helpers accept the full syntax.
def compile(pattern: String) raises -> FullPattern:
    return FullPattern(pattern)


def search(pattern: String, text: String) raises -> Optional[Match]:
    return FullPattern(pattern).search(text)


def findall(pattern: String, text: String) raises -> List[String]:
    return FullPattern(pattern).findall(text)


def split_keep(pattern: String, text: String) raises -> List[String]:
    return FullPattern(pattern).split_keep(text)
