#!/usr/bin/env python3
"""Oracle side of bench/tok-gate.sh. Never the judge: it prints, the gate diffs.

  fp GGUF                 one line: FP16 <tab> tokenizer.ggml.model <tab> pre <tab> n_tokens <tab> VFP16 (FP16 = sha256
                          of every tokenizer.* key and value except the chat template, VFP16 = sha256 of the token
                          strings in id order; `-` when the file has no tokenizer)
  pieces GGUF             one hex line per token id: the bytes llama.cpp's token_to_piece(special=true) gives
  hf TOKENIZER_JSON LIST  one line of ids per case file named in LIST (HF tokenizers, add_special_tokens=False)
  jsonfp JSON...          VFP16 <tab> path <tab> n per tokenizer.json (vocab + added tokens, id order): equal VFP16 = same vocab
  prefixfp GGUF N         VFP16 of the first N GGUF tokens <tab> unused-tail|used-tail (are all later tokens type UNUSED)

Own GGUF header parser (struct), independent of serve/tokenizer.mojo. HF `tokenizers` only in `hf` and `match`.
"""
import hashlib, json, struct, sys

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_kv(path):
    """tokenizer.* keys of a GGUF v2/v3 header as {key: value}; strings stay bytes."""
    out = {}
    with open(path, "rb") as f:
        buf = f.read(16 << 20)
        magic, version = struct.unpack_from("<II", buf, 0)
        if magic != 0x46554747 or version not in (2, 3):
            raise SystemExit(f"FAIL gguf: bad magic or version in {path}")
        n_kv = struct.unpack_from("<Q", buf, 16)[0]
        pos = 24

        def need(n):
            nonlocal buf
            while pos + n > len(buf):
                more = f.read(max(len(buf), n))
                if not more:
                    raise SystemExit(f"FAIL gguf: file ends inside the header: {path}")
                buf += more

        def string():
            nonlocal pos
            need(8)
            n = struct.unpack_from("<Q", buf, pos)[0]
            pos += 8
            need(n)
            s = buf[pos:pos + n]
            pos += n
            return s

        def value(t):
            nonlocal pos
            if t == 8:
                return string()
            if t == 9:
                need(12)
                et, n = struct.unpack_from("<IQ", buf, pos)
                pos += 12
                if et == 8:
                    return [string() for _ in range(n)]
                size = struct.calcsize(SCALAR[et])
                need(size * n)
                v = list(struct.unpack_from("<" + str(n) + SCALAR[et][1], buf, pos))
                pos += size * n
                return v
            size = struct.calcsize(SCALAR[t])
            need(size)
            v = struct.unpack_from(SCALAR[t], buf, pos)[0]
            pos += size
            return v

        for _ in range(n_kv):
            key = string().decode()
            need(4)
            t = struct.unpack_from("<I", buf, pos)[0]
            pos += 4
            v = value(t)
            if key.startswith("tokenizer."):
                out[key] = v
    return out


def fingerprint(kv):
    h = hashlib.sha256()
    for k in sorted(kv):
        if k == "tokenizer.chat_template" or k.startswith("tokenizer.chat_template."):
            continue
        h.update(k.encode() + b"\0")
        v = kv[k]
        if isinstance(v, list):
            for x in v:
                h.update((x if isinstance(x, bytes) else repr(x).encode()) + b"\1")
        else:
            h.update(v if isinstance(v, bytes) else repr(v).encode())
        h.update(b"\2")
    return h.hexdigest()[:16]


def gpt2_unicode_to_byte():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def pieces(path):
    kv = read_kv(path)
    model = kv["tokenizer.ggml.model"].decode()
    toks = kv["tokenizer.ggml.tokens"]
    types = kv.get("tokenizer.ggml.token_type", [1] * len(toks))
    u2b = gpt2_unicode_to_byte()
    for tok, ty in zip(toks, types):
        if ty in (2, 3, 4, 5):  # unknown, control, user_defined, unused: the text as stored
            b = tok
        elif model == "gpt2":
            b = bytes(u2b[c] for c in tok.decode() if c in u2b)
        elif ty == 6:  # <0xNN>
            b = bytes([int(tok[3:5], 16)])
        else:
            b = tok.replace(b"\xe2\x96\x81", b" ")
        print(b.hex())


def hf(json_path, list_path):
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(json_path)
    for case in open(list_path).read().split("\n"):
        if case:
            text = open(case, "rb").read().decode("utf-8")
            print(" ".join(str(i) for i in t.encode(text, add_special_tokens=False).ids))


def vocab_fp(strings):
    return hashlib.sha256("\n".join(strings).encode("utf-8", "replace")).hexdigest()[:16]


def jsonfp(jsons):
    for p in jsons:
        try:
            j = json.load(open(p, encoding="utf-8"))
            vocab = j["model"]["vocab"]
            if isinstance(vocab, list):  # Unigram: [[piece, score], ...]
                ids = {i: v[0] for i, v in enumerate(vocab)}
            else:
                ids = {i: s for s, i in vocab.items()}
            for a in j.get("added_tokens", []):
                ids[a["id"]] = a["content"]
            print(vocab_fp([ids[i] for i in range(len(ids))]) + "\t" + p + "\t" + str(len(ids)))
        except Exception:
            print("-\t" + p + "\t0")


def prefixfp(gguf, n):
    kv = read_kv(gguf)
    toks = [t.decode("utf-8", "replace") for t in kv["tokenizer.ggml.tokens"]]
    tail = set(kv.get("tokenizer.ggml.token_type", [])[n:])
    print(vocab_fp(toks[:n]) + "\t" + ("unused-tail" if tail <= {5} else "used-tail"))


def main():
    cmd = sys.argv[1]
    if cmd == "fp":
        kv = read_kv(sys.argv[2])
        if "tokenizer.ggml.tokens" not in kv:
            print("-\t-\t-\t0\t-")
        else:
            print("\t".join([fingerprint(kv), kv.get("tokenizer.ggml.model", b"-").decode(),
                             kv.get("tokenizer.ggml.pre", b"-").decode() or "-", str(len(kv["tokenizer.ggml.tokens"])),
                             vocab_fp([t.decode("utf-8", "replace") for t in kv["tokenizer.ggml.tokens"]])]))
    elif cmd == "pieces":
        pieces(sys.argv[2])
    elif cmd == "hf":
        hf(sys.argv[2], sys.argv[3])
    elif cmd == "jsonfp":
        jsonfp(sys.argv[2:])
    elif cmd == "prefixfp":
        prefixfp(sys.argv[2], int(sys.argv[3]))
    else:
        raise SystemExit("FAIL usage: tok-gate-oracle.py fp|pieces|hf|jsonfp ...")


if __name__ == "__main__":
    main()
