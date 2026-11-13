# Copyright (c) 2026 AMARBARO · amarbaro.org labs. All rights reserved. Vendored from newera tools/ (see tools/ci-checks.sh).
"""bge-small WordPiece in Mojo, token-exact against HF tokenizers (BertNormalizer + BertPreTokenizer + WordPiece).
The per-codepoint normalizer table comes from tools/bge_norm_oracle.py (unicodedata); CJK and private-use ranges are
tested here so the table stays small.

    wordpiece <vocab.txt> <norm.tsv> <leads.tsv> <rows>
"""
from std.collections import Dict, Set
from std.sys import argv, exit, stderr


def utf8(c: Int, mut out: List[UInt8]):
    if c < 0x80:
        out.append(UInt8(c))
    elif c < 0x800:
        out.append(UInt8(0xC0 | (c >> 6)))
        out.append(UInt8(0x80 | (c & 0x3F)))
    elif c < 0x10000:
        out.append(UInt8(0xE0 | (c >> 12)))
        out.append(UInt8(0x80 | ((c >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (c & 0x3F)))
    else:
        out.append(UInt8(0xF0 | (c >> 18)))
        out.append(UInt8(0x80 | ((c >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((c >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (c & 0x3F)))


def decode(b: Span[UInt8, _]) -> List[Int]:
    var cps = List[Int]()
    var i = 0
    var n = len(b)
    while i < n:
        var c = Int(b[i])
        var k = 0 if c < 0x80 else (1 if c < 0xE0 else (2 if c < 0xF0 else 3))
        var v = c if k == 0 else (c & (0x3F >> k))
        for j in range(1, k + 1):
            v = (v << 6) | (Int(b[i + j]) & 0x3F) if i + j < n else v
        cps.append(v)
        i += k + 1
    return cps^


def cjk(c: Int) -> Bool:
    return (0x4E00 <= c <= 0x9FFF) or (0x3400 <= c <= 0x4DBF) or (0x20000 <= c <= 0x2CEAF) or (0xF900 <= c <= 0xFAFF) or (0x2F800 <= c <= 0x2FA1F)


def ascii_punct(c: Int) -> Bool:
    return (33 <= c <= 47) or (58 <= c <= 64) or (91 <= c <= 96) or (123 <= c <= 126)


def piece(word: List[Int], vocab: Dict[String, Int]) raises -> List[Int]:
    var out = List[Int]()
    var n = len(word)
    if n > 100:  # max_input_chars_per_word counts codepoints
        out.append(vocab["[UNK]"])
        return out^
    var start = 0
    while start < n:
        var end = n
        var found = False
        while end > start:
            var b = List[UInt8]()
            if start > 0:
                b.append(35)
                b.append(35)
            for i in range(start, end):
                utf8(word[i], b)
            var cand = String(StringSlice(unsafe_from_utf8=Span(b)))
            if cand in vocab:
                out.append(vocab[cand])
                start = end
                found = True
                break
            end -= 1
        if not found:
            out.clear()
            out.append(vocab["[UNK]"])
            return out^
    return out^


struct Tok(Movable):
    var vocab: Dict[String, Int]
    var table: Dict[Int, List[Int]]
    var punct: Set[Int]

    def __init__(out self, vocab_path: String, norm_path: String) raises:
        self.vocab = Dict[String, Int]()
        self.table = Dict[Int, List[Int]]()
        self.punct = Set[Int]()
        var id = 0
        for line in String(open(vocab_path, "r").read()).split("\n"):
            self.vocab[String(line)] = id
            id += 1
        for line in String(open(norm_path, "r").read()).split("\n"):
            var f = line.split("\t")
            if len(f) < 3:
                continue
            var rep = List[Int]()
            for h in f[2].split(" "):
                if h.byte_length() > 0:
                    rep.append(atol(h, 16))
            if f[1] == "P":
                self.punct.add(rep[0])
            self.table[atol(f[0], 16)] = rep^

    def encode(self, text: String, max_len: Int = 256) raises -> List[Int]:
        """[CLS] ids [SEP], truncated to max_len like tokenizer.enable_truncation(max_len)."""
        var norm = List[Int]()
        for c in decode(text.as_bytes()):
            if (0xE000 <= c <= 0xF8FF) or c >= 0xF0000:
                continue
            if cjk(c):
                norm.append(32)
                norm.append(c)
                norm.append(32)
            elif c in self.table:
                norm.extend(self.table[c].copy())
            else:
                norm.append(c)
        var ids = List[Int]()
        ids.append(self.vocab["[CLS]"])
        var word = List[Int]()
        for i in range(len(norm) + 1):
            var c = norm[i] if i < len(norm) else 32
            var p = ascii_punct(c) or c in self.punct
            if c == 32 or p:
                if len(word) > 0:
                    ids.extend(piece(word, self.vocab))
                    word.clear()
                if p:
                    var one = List[Int]()
                    one.append(c)
                    ids.extend(piece(one, self.vocab))
            else:
                word.append(c)
        if len(ids) > max_len - 1:
            ids.resize(max_len - 1, 0)
        ids.append(self.vocab["[SEP]"])
        return ids^


def main() raises:
    var args = argv()
    if len(args) != 5:
        print("usage: wordpiece <vocab.txt> <norm.tsv> <leads.tsv> <rows>", file=stderr)
        exit(2)
    var tok = Tok(String(args[1]), String(args[2]))
    var rows = Int(String(args[4]))
    var count = 0
    for line in String(open(String(args[3]), "r").read()).split("\n"):
        if count == rows:
            break
        var cols = line.split("\t")
        if len(cols) < 3:
            continue
        var ids = tok.encode(String(cols[1]) + "\n" + String(cols[2]))
        for i in range(len(ids)):
            if i > 0:
                print(" ", end="")
            print(ids[i], end="")
        print()
        count += 1
