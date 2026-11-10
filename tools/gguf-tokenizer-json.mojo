"""Write an HF tokenizers `tokenizer.json` and `tokenizer-meta.json` for
baro-serve, from a GGUF's own tokenizer metadata (serve/tokenizer.mojo reads it).

  gguf-tokenizer-json MODEL.gguf OUTDIR

tokenizer-meta.json (bos/eos ids, add_bos, chat template) is always written.
tokenizer.json is written for byte-level BPE with a single pre-tokenizer
regex (qwen2, qwen35, llama-bpe/llama3, gpt-2/default); any other tokenizer
(SPM, multi-pattern spark2_5) exits 2 after the meta, so the caller can fall
back to a shipped tokenizer.json. Used by tools/baro serve.
"""
from std.sys import argv
from tokenizer import Tokenizer, T_CONTROL, T_USER


def put(mut o: List[UInt8], s: String):
    for b in s.as_bytes():
        o.append(b)


def jstr(mut o: List[UInt8], s: String):
    comptime HEX = "0123456789abcdef"
    o.append(34)
    for b in s.as_bytes():
        if b == 34 or b == 92:
            o.append(92)
            o.append(b)
        elif b < 32:
            put(o, "\\u00")
            o.append(HEX.as_bytes()[Int(b >> 4)])
            o.append(HEX.as_bytes()[Int(b & 15)])
        else:
            o.append(b)
    o.append(34)


def pre_regex(pre: String) -> String:
    comptime CONTR = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    if pre == "qwen2" or pre == "deepseek-r1-qwen":
        return CONTR + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    if pre == "qwen35":
        return CONTR + r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    if pre == "llama3" or pre == "llama-bpe":
        return CONTR + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    if pre == "gpt-2" or pre == "default":
        return r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    return ""


def opt_id(v: Int) -> String:
    return String(v) if v >= 0 else "null"


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error("usage: gguf-tokenizer-json MODEL.gguf OUTDIR")
    var gguf = String(args[1])
    var out = String(args[2])
    var t = Tokenizer(gguf)

    var m = List[UInt8]()
    put(m, "{\"source_gguf\":")
    jstr(m, gguf)
    put(m, ",\"pre\":")
    jstr(m, t.pre)
    put(m, ",\"vocab_size\":" + String(len(t.tokens)))
    put(m, ",\"bos_token_id\":" + opt_id(t.bos_id) + ",\"eos_token_id\":" + opt_id(t.eos_id) + ",\"pad_token_id\":" + opt_id(t.pad_id))
    put(m, ",\"bos_token\":")
    if t.bos_id >= 0:
        jstr(m, t.tokens[t.bos_id])
    else:
        put(m, "null")
    put(m, ",\"eos_token\":")
    if t.eos_id >= 0:
        jstr(m, t.tokens[t.eos_id])
    else:
        put(m, "null")
    put(m, ",\"add_bos\":" + ("true" if t.add_bos else "false") + ",\"chat_template\":")
    if t.chat_template != "":
        jstr(m, t.chat_template)
    else:
        put(m, "null")
    put(m, "}")
    with open(out + "/tokenizer-meta.json", "w") as f:
        f.write_bytes(Span(m))

    var rx = pre_regex(t.pre)
    if t.is_spm or rx == "":
        print("tokenizer-meta.json written; no tokenizer.json for pre=" + t.pre + (" (SPM)" if t.is_spm else "") + ", unsupported here")
        raise Error("unsupported tokenizer for tokenizer.json")

    var merges = List[String](capacity=len(t.merges))
    for ab in t.merges:
        merges.append(t.tokens[ab >> 32] + " " + t.tokens[ab & 0xFFFFFFFF])

    var o = List[UInt8]()
    put(o, "{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[")
    var first = True
    for i in range(len(t.tokens)):
        if t.types[i] == T_CONTROL or t.types[i] == T_USER:
            if not first:
                put(o, ",")
            first = False
            put(o, "{\"id\":" + String(i) + ",\"content\":")
            jstr(o, t.tokens[i])
            put(o, ",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":")
            put(o, ("true" if t.types[i] == T_CONTROL else "false") + "}")
    put(o, "],\"normalizer\":null,\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Split\",\"pattern\":{\"Regex\":")
    jstr(o, rx)
    put(o, "},\"behavior\":\"Isolated\",\"invert\":false},{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":false,\"use_regex\":false}]},")
    put(o, "\"post_processor\":null,\"decoder\":{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":true,\"use_regex\":true},")
    put(o, "\"model\":{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,\"continuing_subword_prefix\":\"\",\"end_of_word_suffix\":\"\",")
    put(o, "\"fuse_unk\":false,\"byte_fallback\":false,\"ignore_merges\":" + ("true" if t.ignore_merges else "false") + ",\"vocab\":{")
    for i in range(len(t.tokens)):
        if i > 0:
            put(o, ",")
        jstr(o, t.tokens[i])
        put(o, ":" + String(i))
    put(o, "},\"merges\":[")
    for i in range(len(merges)):
        if i > 0:
            put(o, ",")
        jstr(o, merges[i])
    put(o, "]}}")
    with open(out + "/tokenizer.json", "w") as f:
        f.write_bytes(Span(o))
    print("tokenizer.json: pre=" + t.pre + " vocab=" + String(len(t.tokens)) + " merges=" + String(len(merges)) + " add_bos=" + ("true" if t.add_bos else "false"))
