from std.collections import Dict
from grammar.json_value import parse_json_file, JSONDoc, JKindObject


struct Vocab(Copyable, Movable):
    var token_bytes: List[List[UInt8]]
    var is_special: List[Bool]
    var vocab_size: Int
    var eos_id: Int
    var pad_id: Int

    def __init__(out self):
        self.token_bytes = []
        self.is_special = []
        self.vocab_size = 0
        self.eos_id = -1
        self.pad_id = -1

    def bytes_of(self, token_id: Int) -> List[UInt8]:
        return self.token_bytes[token_id].copy()


def _gpt2_byte_to_cp() -> List[Int]:
    var printable: List[Int] = []
    for b in range(33, 127):
        printable.append(b)
    for b in range(161, 173):
        printable.append(b)
    for b in range(174, 256):
        printable.append(b)
    var is_printable: List[Bool] = []
    for _ in range(256):
        is_printable.append(False)
    for b in printable:
        is_printable[b] = True
    var table: List[Int] = []
    for _ in range(256):
        table.append(0)
    var n = 0
    for b in range(256):
        if is_printable[b]:
            table[b] = b
        else:
            table[b] = 256 + n
            n += 1
    return table^


def _gpt2_cp_to_byte() -> Dict[Int, UInt8]:
    var fwd = _gpt2_byte_to_cp()
    var rev: Dict[Int, UInt8] = Dict[Int, UInt8]()
    for b in range(256):
        rev[fwd[b]] = UInt8(b)
    return rev^


def decode_gpt2_token(tok: String, rev: Dict[Int, UInt8]) raises -> List[UInt8]:
    var out: List[UInt8] = []
    for cp in tok.codepoints():
        var c = Int(cp)
        if c in rev:
            out.append(rev[c])
        else:
            raise Error("vocab: codepoint " + String(c) + " not in gpt2 byte map")
    return out^


def load_vocab(pack_dir: String) raises -> Vocab:
    var tok_doc = parse_json_file(pack_dir + "/tokenizer.json")
    var meta_doc = parse_json_file(pack_dir + "/tokenizer-meta.json")

    var model_idx = tok_doc.get_field(tok_doc.root, "model")
    var vocab_idx = tok_doc.get_field(model_idx, "vocab")
    ref vocab_val = tok_doc.get(vocab_idx)
    if vocab_val.kind != JKindObject:
        raise Error("vocab: model.vocab is not an object")

    var n = len(vocab_val.obj_keys)
    var rev = _gpt2_cp_to_byte()

    var out = Vocab()
    var undecodable = List[Int]()
    out.vocab_size = n
    for _ in range(n):
        out.token_bytes.append([])
        out.is_special.append(False)

    for i in range(n):
        ref key = vocab_val.obj_keys[i]
        var id_val = tok_doc.get(vocab_val.obj_vals[i])
        var tid = Int(id_val.n)
        if tid >= len(out.token_bytes):
            while len(out.token_bytes) <= tid:
                out.token_bytes.append([])
                out.is_special.append(False)
        try:
            out.token_bytes[tid] = decode_gpt2_token(key, rev)
        except e:
            # Not byte-level: allowed only for an added (special) token, checked below. spark2_5's
            # <｜end▁of▁sentence｜> and 7 others carry U+FF5C; every ordinary token decodes (2026-10-12).
            undecodable.append(tid)

    if len(out.token_bytes) > out.vocab_size:
        out.vocab_size = len(out.token_bytes)

    var added_idx = tok_doc.get_field(tok_doc.root, "added_tokens")
    if added_idx >= 0:
        ref added = tok_doc.get(added_idx)
        for i in range(len(added.arr)):
            ref at = tok_doc.get(added.arr[i])
            var id_idx = at.find_key("id")
            if id_idx >= 0:
                var tid = Int(tok_doc.get(id_idx).n)
                if tid >= 0 and tid < len(out.is_special):
                    out.is_special[tid] = True

    var special_idx = meta_doc.get_field(meta_doc.root, "special_tokens")
    if special_idx >= 0:
        ref sp = meta_doc.get(special_idx)
        for i in range(len(sp.obj_keys)):
            var tid = Int(meta_doc.get(sp.obj_vals[i]).n)
            if tid >= 0 and tid < len(out.is_special):
                out.is_special[tid] = True

    var added2_idx = meta_doc.get_field(meta_doc.root, "added_tokens")
    if added2_idx >= 0:
        ref ad2 = meta_doc.get(added2_idx)
        for i in range(len(ad2.obj_keys)):
            var tid = Int(meta_doc.get(ad2.obj_vals[i]).n)
            if tid >= 0 and tid < len(out.is_special):
                out.is_special[tid] = True

    for tid in undecodable:
        if not out.is_special[tid]:
            raise Error("vocab: token " + String(tid) + " is not byte-level BPE and not an added token")
        out.token_bytes[tid] = []

    var eos_idx = meta_doc.get_field(meta_doc.root, "eos_token_id")
    if eos_idx >= 0 and not meta_doc.get(eos_idx).is_null():
        out.eos_id = Int(meta_doc.get(eos_idx).n)
    var pad_idx = meta_doc.get_field(meta_doc.root, "pad_token_id")
    if pad_idx >= 0 and not meta_doc.get(pad_idx).is_null():
        out.pad_id = Int(meta_doc.get(pad_idx).n)

    return out^
