"""Byte-level BPE tokenizer read straight from a GGUF header (no tensors touched).

Mirrors tools/gguf-tokenizer.py: gpt2-model vocab + merges + token_type from the
tokenizer.ggml.* keys, pre-tokenizer regexes per tokenizer.ggml.pre (llama.cpp's
tables), control/user_defined tokens matched in the text before BPE. Regexes run
on the vendored uregex/ (repo root, -I ., upstream ~/Projects/mojo/mojo-libs/libs/uregex).
"""
from std.collections import Dict, Set
from std.memory import bitcast
from uregex import Pattern


def to_codepoints(s: StringSlice) -> List[Int]:
    var out = List[Int]()
    for c in s.codepoints():
        out.append(Int(c.to_u32()))
    return out^


def from_codepoints(cps: List[Int], start: Int, end: Int) -> String:
    var s = String("")
    for i in range(start, end):
        s += String(Codepoint.from_u32(UInt32(cps[i])).value())
    return s^


comptime T_NORMAL = 1
comptime T_UNKNOWN = 2
comptime T_CONTROL = 3
comptime T_USER = 4
comptime T_UNUSED = 5
comptime T_BYTE = 6

comptime HEADER_MAX = 96 << 20
comptime HEADER_FIRST = 16 << 20


struct Reader:
    var buf: List[UInt8]
    var pos: Int
    var path: String

    def __init__(out self, var buf: List[UInt8], path: String = ""):
        self.buf = buf^
        self.pos = 0
        self.path = path

    def need(mut self, n: Int) raises:
        while self.pos + n > len(self.buf):
            if self.path == "" or len(self.buf) >= HEADER_MAX:
                raise Error("gguf: header longer than " + String(HEADER_MAX) + " bytes")
            with open(self.path, "r") as f:
                _ = f.seek(len(self.buf))
                var more = f.read_bytes(min(max(len(self.buf), self.pos + n - len(self.buf)), HEADER_MAX - len(self.buf)))
                if len(more) == 0:
                    raise Error("gguf: file ends inside the header")
                self.buf.extend(Span(more))

    def u32(mut self) raises -> Int:
        self.need(4)
        var v = 0
        for i in range(4):
            v |= Int(self.buf[self.pos + i]) << (8 * i)
        self.pos += 4
        return v

    def u64(mut self) raises -> Int:
        self.need(8)
        var v = 0
        for i in range(8):
            v |= Int(self.buf[self.pos + i]) << (8 * i)
        self.pos += 8
        return v

    def f32(mut self) raises -> Float32:
        self.need(4)
        var b0 = UInt32(self.buf[self.pos])
        var b1 = UInt32(self.buf[self.pos + 1])
        var b2 = UInt32(self.buf[self.pos + 2])
        var b3 = UInt32(self.buf[self.pos + 3])
        var bits = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        self.pos += 4
        return bitcast[DType.float32, 1](bits)[0]

    def string(mut self) raises -> String:
        var n = self.u64()
        self.need(n)
        var s = String(unsafe_from_utf8=Span(self.buf)[self.pos : self.pos + n])
        self.pos += n
        return s^

    def skip_scalar(mut self, vtype: Int) raises:
        var n = 0
        if vtype == 0 or vtype == 1 or vtype == 7:
            n = 1
        elif vtype == 2 or vtype == 3:
            n = 2
        elif vtype == 4 or vtype == 5 or vtype == 6:
            n = 4
        elif vtype == 10 or vtype == 11 or vtype == 12:
            n = 8
        else:
            raise Error("gguf: unknown scalar type " + String(vtype))
        self.need(n)
        self.pos += n

    def scalar_int(mut self, vtype: Int) raises -> Int:
        var n = 0
        if vtype == 0 or vtype == 1 or vtype == 7:
            n = 1
        elif vtype == 2 or vtype == 3:
            n = 2
        elif vtype == 4 or vtype == 5:
            n = 4
        elif vtype == 10 or vtype == 11:
            n = 8
        else:
            self.skip_scalar(vtype)
            return -1
        self.need(n)
        var v = 0
        for i in range(n):
            v |= Int(self.buf[self.pos + i]) << (8 * i)
        self.pos += n
        return v


struct _PairHeap:
    """Max-heap of SPM merge candidates: highest score first, then the lower left symbol."""
    var score: List[Float32]
    var left: List[Int]
    var right: List[Int]
    var size: List[Int]
    var h: List[Int]

    def __init__(out self):
        self.score = List[Float32]()
        self.left = List[Int]()
        self.right = List[Int]()
        self.size = List[Int]()
        self.h = List[Int]()

    def before(self, a: Int, b: Int) -> Bool:
        return self.score[a] > self.score[b] or (self.score[a] == self.score[b] and self.left[a] < self.left[b])

    def push(mut self, sc: Float32, l: Int, r: Int, z: Int):
        self.score.append(sc)
        self.left.append(l)
        self.right.append(r)
        self.size.append(z)
        var k = len(self.h)
        self.h.append(len(self.score) - 1)
        while k > 0:
            var p = (k - 1) // 2
            if not self.before(self.h[k], self.h[p]):
                break
            var t = self.h[k]
            self.h[k] = self.h[p]
            self.h[p] = t
            k = p

    def pop(mut self) -> Int:
        var top = self.h[0]
        var last = self.h.pop()
        var m = len(self.h)
        if m > 0:
            self.h[0] = last
            var k = 0
            while 2 * k + 1 < m:
                var c = 2 * k + 1
                if c + 1 < m and self.before(self.h[c + 1], self.h[c]):
                    c += 1
                if not self.before(self.h[c], self.h[k]):
                    break
                var t = self.h[k]
                self.h[k] = self.h[c]
                self.h[c] = t
                k = c
        return top


def _pos(r: Int) -> Int:
    return r


def _pos(r: Optional[Int]) -> Int:
    return r.or_else(-1)


struct Tokenizer(Movable):
    var tokens: List[String]
    var types: List[UInt8]
    var tok2id: Dict[String, Int]
    var merge_of: Dict[Int, Int]
    var merges: List[Int]
    var byte_id: List[Int]
    var special_first: List[Bool]  # by first byte: can a special token start here
    var scores: List[Float32]
    var pre: String
    var patterns: List[Pattern]
    var specials: List[List[UInt8]]  # UTF-8 bytes of each control, user-defined or unknown token
    var special_ids: List[Int]
    var bos_id: Int
    var eos_id: Int
    var pad_id: Int
    var add_bos: Bool
    var chat_template: String
    var b2u: List[Int]
    var u2b: Dict[Int, Int]
    var is_spm: Bool
    var add_space_prefix: Bool
    var ignore_merges: Bool
    var is_g4: Bool  # tokenizer.ggml.model gemma4: BPE on raw UTF-8, spaces as U+2581, newline runs split

    def __init__(out self, gguf_path: String) raises:
        self.tokens = List[String]()
        self.types = List[UInt8]()
        self.tok2id = Dict[String, Int]()
        self.merge_of = Dict[Int, Int]()
        self.merges = List[Int]()
        self.byte_id = List[Int]()
        self.special_first = List[Bool](length=256, fill=False)
        self.scores = List[Float32]()
        self.pre = String("default")
        self.patterns = List[Pattern]()
        self.specials = List[List[UInt8]]()
        self.special_ids = List[Int]()
        self.bos_id = -1
        self.eos_id = -1
        self.pad_id = -1
        self.add_bos = False
        self.chat_template = String("")
        self.b2u = List[Int]()
        self.u2b = Dict[Int, Int]()
        self.is_spm = False
        self.add_space_prefix = False
        self.ignore_merges = False
        self.is_g4 = False
        self._byte_maps()
        self._read_gguf(gguf_path)
        if not self.is_spm:
            self._compile_pre()
            for b in range(256):
                self.byte_id.append(self.tok2id.get(String(Codepoint.from_u32(UInt32(self.b2u[b])).value()), -1))
        for i in range(len(self.tokens)):
            if self.types[i] == T_CONTROL or self.types[i] == T_USER or self.types[i] == T_UNKNOWN:
                var sb = List[UInt8]()
                for b in self.tokens[i].as_bytes():
                    sb.append(b)
                if len(sb) == 0:
                    continue
                self.special_first[Int(sb[0])] = True
                self.specials.append(sb^)
                self.special_ids.append(i)

    def _byte_maps(mut self):
        var n = 0
        for b in range(256):
            var printable = (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255)
            var u = b
            if not printable:
                u = 256 + n
                n += 1
            self.b2u.append(u)
            self.u2b[u] = b

    def _read_gguf(mut self, path: String) raises:
        var buf: List[UInt8]
        with open(path, "r") as f:
            buf = f.read_bytes(HEADER_FIRST)
        var r = Reader(buf^, path)
        var magic = r.u32()
        if magic != 0x46554747:
            raise Error("gguf: bad magic")
        var version = r.u32()
        if version != 3 and version != 2:  # v2 and v3 share the little-endian layout read here
            raise Error("gguf: unsupported version " + String(version))
        _ = r.u64()
        var n_kv = r.u64()
        var model = String("")
        var saw_add_bos_key = False
        var saw_add_space_prefix_key = False
        for _ in range(n_kv):
            var key = r.string()
            var vtype = r.u32()
            if vtype == 8:
                var s = r.string()
                if key == "tokenizer.ggml.model":
                    model = s
                elif key == "tokenizer.ggml.pre":
                    self.pre = s
                elif key == "tokenizer.chat_template":
                    self.chat_template = s
            elif vtype == 9:
                var etype = r.u32()
                var n = r.u64()
                if key == "tokenizer.ggml.tokens" and etype == 8:
                    self.tokens.reserve(n)
                    for i in range(n):
                        self.tokens.append(r.string())
                        self.tok2id[self.tokens[i]] = i
                elif key == "tokenizer.ggml.merges" and etype == 8:
                    if len(self.tok2id) == 0:
                        raise Error("gguf: tokenizer.ggml.merges before tokenizer.ggml.tokens")
                    self.merges.reserve(n)
                    for i in range(n):
                        var m = r.string()
                        var sp = _pos(m.find(" ", 1))  # llama.cpp: the first space after the first byte
                        var left = String(m[byte=:sp])
                        var a = self.tok2id.get(left, -1)
                        var right = String(m[byte=sp + 1:])
                        var b = self.tok2id.get(right, -1)
                        left += right
                        var ab = self.tok2id.get(left, -1)
                        if model == "gemma4" and (sp < 0 or a < 0 or b < 0 or ab < 0):
                            continue  # a merge of strings no symbol can reach; llama.cpp keeps it as an unused rank
                        if sp < 0 or a < 0 or b < 0 or ab < 0:
                            raise Error("gguf: merge outside the vocab: " + m)
                        self.merges.append((a << 32) | b)
                        self.merge_of[(a << 32) | b] = (i << 32) | ab
                elif key == "tokenizer.ggml.token_type":
                    self.types.reserve(n)
                    for _ in range(n):
                        self.types.append(UInt8(r.scalar_int(etype)))
                elif key == "tokenizer.ggml.scores" and etype == 6:
                    self.scores.reserve(n)
                    for _ in range(n):
                        self.scores.append(r.f32())
                elif etype == 8:
                    for _ in range(n):
                        _ = r.string()
                elif etype == 9:
                    raise Error("gguf: nested arrays unsupported")
                else:
                    for _ in range(n):
                        r.skip_scalar(etype)
            else:
                var v = r.scalar_int(vtype)
                if key == "tokenizer.ggml.bos_token_id":
                    self.bos_id = v
                elif key == "tokenizer.ggml.eos_token_id":
                    self.eos_id = v
                elif key == "tokenizer.ggml.padding_token_id":
                    self.pad_id = v
                elif key == "tokenizer.ggml.add_bos_token":
                    self.add_bos = v == 1
                    saw_add_bos_key = True
                elif key == "tokenizer.ggml.add_space_prefix":
                    self.add_space_prefix = v == 1
                    saw_add_space_prefix_key = True
        if model == "gpt2":
            self.is_spm = False
        elif model == "llama":
            self.is_spm = True
        elif model == "gemma4":
            self.is_g4 = True
        else:
            raise Error("tokenizer.ggml.model=" + model + ": only gpt2 (BPE) or llama (SPM) is supported")
        if len(self.tokens) == 0 or len(self.tokens) != len(self.types):
            raise Error("gguf: tokens/token_type missing or mismatched")
        if self.is_spm and len(self.scores) != len(self.tokens):
            raise Error("gguf: tokenizer.ggml.scores missing or mismatched for SPM model")
        if not saw_add_space_prefix_key:
            self.add_space_prefix = self.is_spm
        self.ignore_merges = self.pre == "llama3" or self.pre == "llama-bpe" or self.pre == "minicpm5"
        if not saw_add_bos_key and (self.ignore_merges or self.is_spm):
            self.add_bos = True  # llama.cpp defaults: SPM adds BOS, BPE does not
        if self.is_g4:
            self.add_bos = True  # llama.cpp overrides add_bos to true for the gemma4 pre-tokenizer

    def _compile_pre(mut self) raises:
        comptime CONTR = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        if self.pre == "qwen2" or self.pre == "deepseek-r1-qwen" or self.pre == "f2llmv2" or self.pre == "grok-2" or self.pre == "hunyuan" or self.pre == "kormo" or self.pre == "megrez" or self.pre == "solar-open" or self.pre == "stablelm2":
            self.patterns.append(Pattern(CONTR + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"))
        elif self.pre == "qwen35":
            self.patterns.append(Pattern(CONTR + r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"))
        elif self.pre == "llama3" or self.pre == "llama-bpe" or self.pre == "dbrx" or self.pre == "falcon-h1" or self.pre == "falcon3" or self.pre == "chatglm-bpe" or self.pre == "glm4" or self.pre == "jina-v5-nano" or self.pre == "lfm2" or self.pre == "llama-v3" or self.pre == "midm-2.0" or self.pre == "pixtral" or self.pre == "smaug-bpe":
            self.patterns.append(Pattern(CONTR + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"))
        elif self.pre == "spark2_5":
            self.patterns.append(Pattern(r"\p{N}{1,3}"))
            self.patterns.append(Pattern(r"[一-龥぀-ゟ゠-ヿ]+"))
            self.patterns.append(Pattern(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+|[\r\n]|\s+(?!\S)|\s+"))
            self.patterns.append(Pattern(r"\p{N}"))
        elif self.pre == "minicpm5":
            self.patterns.append(Pattern(r"\p{N}{1,3}"))
            self.patterns.append(Pattern(CONTR + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"))
        elif self.pre == "deepseek-v3" or self.pre == "hunyuan-dense" or self.pre == "hy_v4" or self.pre == "joyai-llm":
            self.patterns.append(Pattern(r"\p{N}{1,3}"))
            self.patterns.append(Pattern(r"[一-龥぀-ゟ゠-ヿ]+"))
            self.patterns.append(Pattern(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"))
        elif self.pre == "bailingmoe" or self.pre == "bailingmoe2" or self.pre == "llada-moe":
            self.patterns.append(Pattern(CONTR + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"))
        elif self.pre == "exaone" or self.pre == "command-r" or self.pre == "codeshell" or self.pre == "mellum2" or self.pre == "minerva-7b" or self.pre == "refact" or self.pre == "smollm" or self.pre == "starcoder":
            self.patterns.append(Pattern(r"\p{N}"))
            self.patterns.append(Pattern(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"))
        elif self.pre == "cohere2moe" or self.pre == "tiny_aya":
            self.patterns.append(Pattern(r"\d{1,3}(?=(?:\d{3})*\b)"))
            self.patterns.append(Pattern(r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"))
        elif self.pre == "default":  # also a file with no tokenizer.ggml.pre
            self.patterns.append(Pattern(r"[\p{P}\$\+<=>\^~\|]+"))
            self.patterns.append(Pattern(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"))
            self.patterns.append(Pattern(r"\p{N}+"))
            self.patterns.append(Pattern(r"[0-9][0-9][0-9]"))
        elif self.pre == "gpt-2" or self.pre == "granite-docling" or self.pre == "exaone4":
            self.patterns.append(Pattern(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"))
        elif self.pre == "a.x-4.0" or self.pre == "gigachat" or self.pre == "jais" or self.pre == "jina-de" or self.pre == "jina-es" or self.pre == "jina-v1-en" or self.pre == "jina-v2-code" or self.pre == "jina-v2-de" or self.pre == "jina-v2-es" or self.pre == "mellum" or self.pre == "modern-bert" or self.pre == "mpt" or self.pre == "olmo" or self.pre == "phi-2" or self.pre == "roberta-bpe" or self.pre == "trillion":
            self.patterns.append(Pattern(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"))
        elif self.pre == "chameleon" or self.pre == "gemma4" or self.pre == "granite-embed-multi-311m" or self.pre == "granite-embed-multi-97m" or self.pre == "sarvam-moe" or self.pre == "tekken" or self.pre == "whitespace":
            pass
        elif self.pre == "bloom" or self.pre == "gpt3-finnish" or self.pre == "poro-chat":
            self.patterns.append(Pattern(r" ?[^(\s|.,!?…。，、।۔،)]+"))
        elif self.pre == "gpt-4o" or self.pre == "kanana2" or self.pre == "llama4" or self.pre == "minimax-m2" or self.pre == "talkie":
            self.patterns.append(Pattern(r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"))
        elif self.pre == "deepseek-coder":
            self.patterns.append(Pattern(r"[\r\n]"))
            self.patterns.append(Pattern(r"\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+"))
            self.patterns.append(Pattern(r"\s?[!-/:-~！-／：-～‘-‟　-。]+"))
            self.patterns.append(Pattern(r"\s+$"))
            self.patterns.append(Pattern(r"[一-龥ࠀ-一가-퟿]+"))
            self.patterns.append(Pattern(r"\p{N}"))
        elif self.pre == "deepseek-llm":
            self.patterns.append(Pattern(r"[\r\n]"))
            self.patterns.append(Pattern(r"\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+"))
            self.patterns.append(Pattern(r"\s?[!-/:-~！-／：-～‘-‟　-。]+"))
            self.patterns.append(Pattern(r"\s+$"))
            self.patterns.append(Pattern(r"[一-龥ࠀ-一가-퟿]+"))
            self.patterns.append(Pattern(r"\p{N}+"))
        elif self.pre == "exaone-moe":
            self.patterns.append(Pattern(CONTR + r"|[^\r\n\p{L}\p{N}]?(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]?|\s*[\r\n]|\s+(?!\S)|\s+"))
        elif self.pre == "falcon":
            self.patterns.append(Pattern(r"[\p{P}\$\+<=>\^~\|`]+"))
            self.patterns.append(Pattern(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"))
            self.patterns.append(Pattern(r"[0-9][0-9][0-9]"))
        elif self.pre == "jais-2":
            self.patterns.append(Pattern(CONTR + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s{512}(?!\S)|\s{256}(?!\S)|\s{128}(?!\S)|\s{64}(?!\S)|\s{32}(?!\S)|\s{16}(?!\S)|\s{8}(?!\S)|\s{4}(?!\S)|\s{1,2}(?!\S)|\s{1}"))
        elif self.pre == "kimi-k2":
            self.patterns.append(Pattern(r"\p{Han}+"))
        elif self.pre == "laguna":
            self.patterns.append(Pattern(r"[^\n]+|[\n]+"))
            self.patterns.append(Pattern(CONTR + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"))
        elif self.pre == "seed-coder":
            self.patterns.append(Pattern(CONTR + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}\r\n]+|\s*[\r\n]+|\s+(?!\S)|\s+"))
        elif self.pre == "superbpe":
            self.patterns.append(Pattern(r"\p{N}+"))
            self.patterns.append(Pattern(r"(?=(\d{3})+(?!\d))"))
        elif self.pre == "viking":
            self.patterns.append(Pattern(r" ?[^(\s|.,!?…。，、।۔،)]+"))
            self.patterns.append(Pattern(r"\p{N}"))
        elif self.pre == "youtu":
            self.patterns.append(Pattern(r"[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥぀-ゟ゠-ヿ]+"))
            self.patterns.append(Pattern(r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"))

        else:
            raise Error("tokenizer.ggml.pre=" + self.pre + ": no pre-tokenizer regex on file")


    def _pieces(self, chunk: String) -> List[String]:
        var pieces = List[String]()
        pieces.append(chunk)
        for p in self.patterns:
            var next = List[String]()
            for piece in pieces:
                for q in p.split_keep(piece):
                    next.append(q)
            pieces = next^
        return pieces^

    def _bpe(self, piece: String, mut out: List[Int]) raises:
        if self.ignore_merges and len(piece.as_bytes()) > 0:
            var whole = String("")
            for b in piece.as_bytes():
                whole += String(Codepoint.from_u32(UInt32(self.b2u[Int(b)])).value())
            var wid = self.tok2id.get(whole, -1)
            if wid >= 0:
                out.append(wid)
                return
        var syms = List[Int](capacity=len(piece.as_bytes()))
        for b in piece.as_bytes():
            syms.append(self.byte_id[Int(b)])
        while len(syms) > 1:
            var best = -1
            var best_m = 1 << 62
            for i in range(len(syms) - 1):
                var m = self.merge_of.get((syms[i] << 32) | syms[i + 1], -1)
                if m >= 0 and m < best_m:
                    best_m = m
                    best = i
            if best < 0:
                break
            syms[best] = best_m & 0xFFFFFFFF
            _ = syms.pop(best + 1)
        for id in syms:
            if id >= 0:  # a byte with no token is dropped, as llama.cpp does (falcon has no token for some bytes)
                out.append(id)

    def _hex_digit(self, n: Int) -> String:
        var code = UInt32(48 + n) if n < 10 else UInt32(55 + n)
        return String(Codepoint.from_u32(code).value())

    def _byte_tok(self, b: UInt8) raises -> Int:
        var key = String("<0x")
        key += self._hex_digit(Int(b) >> 4)
        key += self._hex_digit(Int(b) & 15)
        key += ">"
        var id = self.tok2id.get(key, -1)
        if id < 0:
            raise Error("tokenizer: no byte token for " + key)
        return id

    def _hex_val(self, c: UInt8) -> Int:
        return Int(c) - 48 if c <= 57 else Int(c) - 65 + 10

    def _byte_from_tok(self, id: Int) -> UInt8:
        var raw = List[UInt8]()
        for b in self.tokens[id].as_bytes():
            raw.append(b)
        return UInt8(self._hex_val(raw[3]) * 16 + self._hex_val(raw[4]))

    def _spm_pair(self, text: String, st: List[Int], ln: List[Int], l: Int, r: Int, mut heap: _PairHeap):
        var id = self.tok2id.get(String(text[byte=st[l] : st[l] + ln[l] + ln[r]]), -1)
        if id >= 0:
            heap.push(self.scores[id], l, r, ln[l] + ln[r])

    def _spm(self, text: String, mut out: List[Int]) raises:
        """Merge the adjacent pair with the highest score first, the leftmost on a tie, until no pair is a token.
        Symbols are byte ranges of text in a linked list; candidate pairs wait in a max-heap, and a popped pair
        whose sides changed since it was pushed is dropped. n log n, where the full rescan per merge was n^2."""
        var tb = text.as_bytes()
        var n = len(tb)
        var st = List[Int]()
        var ln = List[Int]()
        var i = 0
        while i < n:
            var c = Int(tb[i])
            var l = min(1 if c < 0x80 else (2 if c < 0xE0 else (3 if c < 0xF0 else 4)), n - i)
            st.append(i)
            ln.append(l)
            i += l
        var ns = len(st)
        if ns == 0:
            return
        var nx = List[Int](capacity=ns)
        var pv = List[Int](capacity=ns)
        for k in range(ns):
            pv.append(k - 1)
            nx.append(k + 1 if k + 1 < ns else -1)
        var heap = _PairHeap()
        for k in range(ns - 1):
            self._spm_pair(text, st, ln, k, k + 1, heap)
        while len(heap.h) > 0:
            var e = heap.pop()
            var l = heap.left[e]
            var r = heap.right[e]
            if ln[l] == 0 or ln[r] == 0 or ln[l] + ln[r] != heap.size[e]:
                continue
            ln[l] += ln[r]
            ln[r] = 0
            nx[l] = nx[r]
            if nx[l] >= 0:
                pv[nx[l]] = l
                self._spm_pair(text, st, ln, l, nx[l], heap)
            if pv[l] >= 0:
                self._spm_pair(text, st, ln, pv[l], l, heap)
        var k = 0
        while k >= 0:
            var id = self.tok2id.get(String(text[byte=st[k] : st[k] + ln[k]]), -1)
            if id >= 0:
                out.append(id)
            else:
                for j in range(st[k], st[k] + ln[k]):
                    out.append(self._byte_tok(tb[j]))
            k = nx[k]

    def _spm_encode(self, chunk: String, is_prev_special: Bool, mut out: List[Int]) raises:
        var text = String(" ") if (self.add_space_prefix and is_prev_special) else String("")
        text += chunk
        var raw = List[UInt8]()
        for b in text.as_bytes():
            if b == 32:
                raw.append(0xE2)
                raw.append(0x96)
                raw.append(0x81)
            else:
                raw.append(b)
        var escaped = String(unsafe_from_utf8=Span(raw))
        self._spm(escaped, out)

    def _g4_encode(self, cps: List[Int], start: Int, end: Int, mut out: List[Int]) raises:
        """gemma4: spaces become U+2581, the text splits into runs of newlines and runs of anything else, a run that is
        one token is that token, else BPE merges (lowest rank first) over its codepoints; a codepoint outside the
        vocab is emitted as <0xNN> byte tokens."""
        var i = start
        while i < end:
            var nl = cps[i] == 10
            var j = i
            while j < end and (cps[j] == 10) == nl:
                j += 1
            var syms = List[Int]()
            var sc = List[Int]()
            for k in range(i, j):
                var c = 0x2581 if cps[k] == 32 else cps[k]
                sc.append(c)
                syms.append(self.tok2id.get(String(Codepoint.from_u32(UInt32(c)).value()), -1))
            var whole = -1
            if nl:
                whole = self.tok2id.get(from_codepoints(cps, i, j), -1)
            if whole >= 0:
                out.append(whole)
                i = j
                continue
            while len(syms) > 1:
                var best = -1
                var best_m = 1 << 62
                for k in range(len(syms) - 1):
                    if syms[k] < 0 or syms[k + 1] < 0:
                        continue
                    var m = self.merge_of.get((syms[k] << 32) | syms[k + 1], -1)
                    if m >= 0 and m < best_m:
                        best_m = m
                        best = k
                if best < 0:
                    break
                syms[best] = best_m & 0xFFFFFFFF
                _ = syms.pop(best + 1)
                _ = sc.pop(best + 1)
            for k in range(len(syms)):
                if syms[k] >= 0:
                    out.append(syms[k])
                else:
                    for b in String(Codepoint.from_u32(UInt32(sc[k])).value()).as_bytes():
                        out.append(self._byte_tok(b))
            i = j

    def _match_special(self, tb: Span[UInt8, _], at: Int) -> Int:
        """Index of the longest special token whose bytes start at byte at, or -1. A special starts with a UTF-8
        lead byte, so it never matches inside a character."""
        var best = -1
        var best_len = 0
        if not self.special_first[Int(tb[at])]:
            return -1
        for k in range(len(self.specials)):
            var n = len(self.specials[k])
            if n <= best_len or at + n > len(tb):
                continue
            var ok = True
            for j in range(n):
                if tb[at + j] != self.specials[k][j]:
                    ok = False
                    break
            if ok:
                best = k
                best_len = n
        return best

    def token_str(self, id: Int) -> String:
        return self.tokens[id] if id >= 0 and id < len(self.tokens) else String("")

    def encode(self, text: String, add_special: Bool = True) raises -> List[Int]:
        var pos = List[Int]()
        var cnt = List[Int]()
        return self.encode_cuts(text, add_special, False, pos, cnt)

    def encode_cuts(self, text: String, add_special: Bool, record: Bool, mut pos: List[Int], mut cnt: List[Int]) raises -> List[Int]:
        """encode; with record, also every byte position at an edge of a special token (and the end of the text) in
        pos, with the number of ids before it in cnt. The text between two specials is encoded on its own, so
        encode(text[:pos[k]]) is exactly the first cnt[k] ids."""
        var out = List[Int]()
        if add_special and self.add_bos and self.bos_id >= 0:
            out.append(self.bos_id)
        var tb = text.as_bytes()
        var n = len(tb)
        var start = 0
        var i = 0
        var is_prev_special = True
        var memo = Dict[String, Int]()  # this call only: a short piece seen before -> (where its ids start in out) << 16 | count
        while i <= n:
            var k = -1 if i == n else self._match_special(tb, i)
            if k >= 0 or i == n:
                if i > start:
                    var chunk = String(text[byte=start:i])
                    if self.is_spm:
                        self._spm_encode(chunk, is_prev_special, out)
                    elif self.is_g4:
                        var cps = to_codepoints(chunk)
                        self._g4_encode(cps, 0, len(cps), out)
                    else:
                        for piece in self._pieces(chunk):
                            var hit = memo.get(piece, -1)
                            if hit >= 0:
                                for j in range(hit & 0xFFFF):
                                    var id = out[(hit >> 16) + j]
                                    out.append(id)
                                continue
                            var p = len(out)
                            self._bpe(piece, out)
                            if piece.byte_length() <= 32:
                                memo[String(piece)] = (p << 16) | (len(out) - p)
                    is_prev_special = False
                if record:
                    pos.append(i)
                    cnt.append(len(out))
                if k >= 0:
                    out.append(self.special_ids[k])
                    i += len(self.specials[k])
                    start = i
                    is_prev_special = True
                    if record:
                        pos.append(i)
                        cnt.append(len(out))
                else:
                    i += 1
            else:
                i += 1
        return out^

    def decode(self, ids: List[Int], keep_special: Bool = False, strip_space: Bool = True) -> String:
        """strip_space=False keeps the SPM prefix space of a piece (the raw token_to_piece bytes)."""
        var bytes = List[UInt8]()
        var is_prev_special = strip_space
        for id in ids:
            if id < 0 or id >= len(self.tokens):
                continue
            var ty = self.types[id]
            if ty == T_CONTROL and not keep_special:
                is_prev_special = True
                continue
            if ty == T_CONTROL or ty == T_USER or ty == T_UNKNOWN:
                for b in self.tokens[id].as_bytes():
                    bytes.append(b)
                is_prev_special = True
                continue
            if self.is_spm or self.is_g4:
                if ty == T_BYTE:
                    bytes.append(self._byte_from_tok(id))
                else:
                    var raw = List[UInt8]()
                    for b in self.tokens[id].as_bytes():
                        raw.append(b)
                    var piece = List[UInt8]()
                    var j = 0
                    var n = len(raw)
                    while j < n:
                        if j + 2 < n and raw[j] == 0xE2 and raw[j + 1] == 0x96 and raw[j + 2] == 0x81:
                            piece.append(32)
                            j += 3
                        else:
                            piece.append(raw[j])
                            j += 1
                    var off = 0
                    if self.add_space_prefix and is_prev_special and len(piece) > 0 and piece[0] == 32:
                        off = 1
                    for k in range(off, len(piece)):
                        bytes.append(piece[k])
            else:
                for c in self.tokens[id].codepoints():
                    var b = self.u2b.get(Int(c.to_u32()), -1)
                    if b >= 0:
                        bytes.append(UInt8(b))
            is_prev_special = False
        return String(unsafe_from_utf8=Span(bytes))
