# latent-bridge P2 data: GSM8K + synthetic multi-hop items, frozen splits, Flash token ids, and the scorer.
# One row per item in items.tsv:
#   id  task  split  rank  answer  pre  rat  post  ans      (ids space-separated)
# teacher = pre rat post ans, student = pre post ans, eval = pre [payload] post -> generate.
# usage: latent-data build GSM8K_DIR GGUF OUT_DIR | score ITEMS GEN | check ITEMS GGUF | emb JSON OUT.f32 WIDTH
#   emb: appends the rows of a llama-server /embedding reply (pooling none) as f32, prints the row count
# build: ./.venv/bin/mojo build tools/latent-data.mojo -I . -I serve -o .work/latent-data
from std.sys import argv
from tokenizer import Tokenizer
from grammar.json_value import parse_json_bytes

comptime PRE_A = "<|im_start|>user\n"
comptime PRE_B = "\nWork through it.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
comptime POST = "<|im_end|>\n<|im_start|>user\nGive only the final answer, as: Answer: <answer>.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:"
comptime ANS_END = "<|im_end|>"
comptime N_VAL = 200
comptime N_TEST = 300
comptime MH_TRAIN = 7500


struct Rng:
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019

    def next(mut self) -> UInt64:
        self.s ^= self.s << 13
        self.s ^= self.s >> 7
        self.s ^= self.s << 17
        return self.s

    def below(mut self, n: Int) -> Int:
        return Int(self.next() % UInt64(n))


def sub(s: String, lo: Int, hi: Int) -> String:
    return String(s[byte=lo:hi])


def strip(s: String) -> String:
    return String(s.strip())


def permutation(n: Int, seed: UInt64) -> List[Int]:
    var r = Rng(seed)
    var xs = List[Int]()
    for i in range(n):
        xs.append(i)
    for i in range(n - 1, 0, -1):
        var j = r.below(i + 1)
        var t = xs[i]
        xs[i] = xs[j]
        xs[j] = t
    return xs^


def fnv(s: String) -> UInt64:
    var h: UInt64 = 0xCBF29CE484222325
    for b in s.as_bytes():
        h = (h ^ UInt64(b)) * 0x100000001B3
    return h


def drop_calc(s: String) -> String:
    var out = String()
    var i = 0
    while True:
        var a = s.find("<<", i)
        var b = s.find(">>", a) if a >= 0 else -1
        if b < 0:
            out += sub(s, i, s.byte_length())
            return out^
        out += sub(s, i, a)
        i = b + 2


def norm_answer(s: String) -> String:
    var t = strip(s)
    var out = String()
    for b in t.as_bytes():
        if b != UInt8(ord(",")) and b != UInt8(ord("$")):
            out += chr(Int(b))
    while out.byte_length() > 0 and out.as_bytes()[out.byte_length() - 1] == UInt8(ord(".")):
        out = sub(out, 0, out.byte_length() - 1)
    return out^


def ids_str(ids: List[Int]) -> String:
    var out = String()
    for i in range(len(ids)):
        out += (" " if i > 0 else "") + String(ids[i])
    return out^


@fieldwise_init
struct Item(Copyable, Movable):
    var id: String
    var task: String
    var split: String
    var rank: Int
    var question: String
    var rationale: String
    var answer: String


def gsm8k(path: String, prefix: String, n_first: Int, first: String, rest: String, seed: UInt64) raises -> List[Item]:
    var rows = List[String]()
    with open(path, "r") as f:
        for line in f.read().split("\n"):
            if line.byte_length() > 0:
                rows.append(String(line))
    var perm = permutation(len(rows), seed)
    var out = List[Item]()
    for rank in range(len(perm)):
        var k = perm[rank]
        var bytes = List[UInt8]()
        for b in rows[k].as_bytes():
            bytes.append(b)
        var doc = parse_json_bytes(bytes^)
        var q = doc.get(doc.get_field(doc.root, "question")).s
        var a = doc.get(doc.get_field(doc.root, "answer")).s
        var cut = a.find("#### ")
        if cut < 0:
            raise Error("gsm8k row " + String(k) + " has no #### answer")
        var split = first if rank < n_first else rest
        out.append(Item(prefix + String(k), "gsm8k", split, rank, q, strip(drop_calc(sub(a, 0, cut))), norm_answer(sub(a, cut + 5, a.byte_length()))))
    return out^


comptime CONS = "bcdfghklmnprstvz"
comptime VOWS = "aeiou"


def word(mut r: Rng, pool: Int, suffix: String) -> String:
    while True:
        var n = 2 + r.below(2)
        var w = String()
        for i in range(n):
            var c = chr(Int(CONS.as_bytes()[r.below(16)]))
            w += c.upper() if i == 0 else c
            w += chr(Int(VOWS.as_bytes()[r.below(5)]))
        w += suffix
        if Int(fnv(w) & 1) == pool:
            return w^


def multihop(n: Int, pool: Int, split: String, seed: UInt64) -> List[Item]:
    var r = Rng(seed)
    var rels: List[String] = ["mentor", "sister", "neighbor", "boss", "friend", "cousin"]
    var out = List[Item]()
    for rank in range(n):
        var hops = 2 + r.below(3)
        var facts = List[String]()
        var chain = List[String]()
        var q = String()
        var answer = String()
        for c in range(3):
            var who = word(r, pool, "")
            var path = who + "'s"
            for h in range(hops - 1):
                var rel = rels[r.below(len(rels))]
                var nxt = word(r, pool, "")
                var fact = who + "'s " + rel + " is " + nxt + "."
                facts.append(fact)
                if c == 0:
                    chain.append(fact)
                    path += " " + rel + ("'s" if h < hops - 2 else "")
                who = nxt
            var city = word(r, pool, "ton")
            var last = who + " lives in " + city + "."
            facts.append(last)
            if c == 0:
                chain.append(last)
                q = "Where does " + path + " live?"
                answer = city
        var order = permutation(len(facts), r.next())
        var ctx = String()
        for i in range(len(order)):
            ctx += (" " if i > 0 else "") + facts[order[i]]
        var rat = String()
        for i in range(len(chain)):
            rat += (" " if i > 0 else "") + chain[i]
        rat += " So the answer is " + answer + "."
        out.append(Item("mh-" + split + "-" + String(rank), "multihop", split, rank, ctx + "\n" + q, rat, answer))
    return out^


def build(gdir: String, gguf: String, odir: String) raises:
    var tok = Tokenizer(gguf)
    var items = gsm8k(gdir + "/train.jsonl", "gt", N_VAL, "val", "train", 20261012)
    items.extend(gsm8k(gdir + "/test.jsonl", "ge", N_TEST, "test", "unused", 20261013))
    items.extend(multihop(MH_TRAIN, 0, "train", 1))
    items.extend(multihop(N_VAL, 0, "val", 2))
    items.extend(multihop(N_TEST, 1, "test", 3))
    var post = tok.encode(String(POST), add_special=False)
    var end = tok.encode(String(ANS_END), add_special=False)
    var tsv = String()
    var n = 0
    for it in items:
        if it.split == "unused":
            continue
        var pre = tok.encode(String(PRE_A) + it.question + String(PRE_B), add_special=False)
        var rat = tok.encode(it.rationale, add_special=False)
        var ans = tok.encode(" " + it.answer, add_special=False)
        ans.extend(end.copy())
        tsv += it.id + "\t" + it.task + "\t" + it.split + "\t" + String(it.rank) + "\t" + it.answer + "\t"
        tsv += ids_str(pre) + "\t" + ids_str(rat) + "\t" + ids_str(post) + "\t" + ids_str(ans) + "\n"
        n += 1
    with open(odir + "/items.tsv", "w") as f:
        f.write(tsv)
    print("PASS build", n, "items to", odir + "/items.tsv")


def parse_ids(s: String) raises -> List[Int]:
    var ids = List[Int]()
    for p in s.split(" "):
        if p.byte_length() > 0:
            ids.append(atol(String(p)))
    return ids^


def read_lines(path: String) raises -> List[String]:
    var out = List[String]()
    with open(path, "r") as f:
        for line in f.read().split("\n"):
            if line.byte_length() > 0:
                out.append(String(line))
    return out^


def extract(gen: String) -> String:
    var t = gen
    var e = t.find("<|im_end|>")
    if e >= 0:
        t = sub(t, 0, e)
    var nl = t.find("\n")
    if nl >= 0:
        t = sub(t, 0, nl)
    return norm_answer(t)


def score(items_path: String, gen_path: String) raises:
    var gold = Dict[String, String]()
    var task = Dict[String, String]()
    for line in read_lines(items_path):
        var f = line.split("\t")
        gold[String(f[0])] = String(f[4])
        task[String(f[0])] = String(f[1])
    var hit = Dict[String, Int]()
    var tot = Dict[String, Int]()
    for line in read_lines(gen_path):
        var f = line.split("\t")
        var id = String(f[0])
        if id not in gold:
            raise Error("score: unknown id " + id)
        var t = task[id]
        var ok = extract(String(f[1]) if len(f) > 1 else String()) == gold[id]
        print("item", id, t, "ok" if ok else "miss")
        hit[t] = hit.get(t, 0) + (1 if ok else 0)
        tot[t] = tot.get(t, 0) + 1
    for e in tot.items():
        print("score", e.key, hit.get(e.key, 0), "/", e.value)


def check(items_path: String, gguf: String) raises:
    var tok = Tokenizer(gguf)
    var seen = Dict[String, String]()
    var count = Dict[String, Int]()
    var fails = 0
    var gen = String()
    for line in read_lines(items_path):
        var f = line.split("\t")
        var split = String(f[2])
        var key = String(f[1]) + "/" + split
        count[key] = count.get(key, 0) + 1
        var h = String(fnv(tok.decode(parse_ids(String(f[5])), keep_special=True)))
        if h in seen and seen[h] != split:
            print("FAIL check: prompt of", f[0], "is in", seen[h], "and", split)
            fails += 1
        seen[h] = split
        if split == "test":
            gen += String(f[0]) + "\t" + tok.decode(parse_ids(String(f[8])), keep_special=True) + "\n"
    for e in count.items():
        print("count", e.key, e.value)
    with open(items_path + ".gold-gen.tsv", "w") as f:
        f.write(gen)
    if fails > 0:
        raise Error("FAIL check: " + String(fails) + " prompts shared across splits")
    print("PASS check: no prompt shared across splits; gold generations in", items_path + ".gold-gen.tsv")


def emb(json_path: String, out_path: String, width: Int) raises:
    var data: String
    with open(json_path, "r") as f:
        data = f.read()
    var at = data.find("\"embedding\"")
    if at < 0:
        raise Error("emb: no embedding key in " + json_path)
    var b = data.as_bytes()
    var vals = List[Float32]()
    var i = at + 11
    var n = len(b)
    while i < n:
        var c = b[i]
        if c == UInt8(ord("-")) or (c >= UInt8(ord("0")) and c <= UInt8(ord("9"))):
            var j = i + 1
            while j < n and b[j] != UInt8(ord(",")) and b[j] != UInt8(ord("]")):
                j += 1
            vals.append(Float32(atof(String(StringSlice(unsafe_from_utf8=b[i:j])))))
            i = j
        elif c == UInt8(ord("}")):
            break
        else:
            i += 1
    if len(vals) == 0 or len(vals) % width != 0:
        raise Error("emb: " + String(len(vals)) + " values is not a positive multiple of width " + String(width))
    with open(out_path, "a") as f:
        f.write_bytes(Span[UInt8](unsafe_ptr=vals.unsafe_ptr().unsafe_bitcast[UInt8](), length=len(vals) * 4))
    print(len(vals) // width)


def main() raises:
    var a = argv()
    var cmd = String(a[1]) if len(a) > 1 else String()
    if cmd == "emb" and len(a) == 5:
        emb(String(a[2]), String(a[3]), Int(String(a[4])))
        return
    if cmd == "build" and len(a) == 5:
        build(String(a[2]), String(a[3]), String(a[4]))
    elif cmd == "score" and len(a) == 4:
        score(String(a[2]), String(a[3]))
    elif cmd == "check" and len(a) == 4:
        check(String(a[2]), String(a[3]))
    else:
        raise Error("usage: latent-data build GSM8K_DIR GGUF OUT_DIR | score ITEMS GEN | check ITEMS GGUF")
