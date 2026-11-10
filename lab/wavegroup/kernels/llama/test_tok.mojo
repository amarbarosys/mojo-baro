# Tokenizer gate: mojo-baro's GGUF BPE tokenizer gives llama.cpp's ids on every line of tests/tok_corpus.txt, and
# decode(encode(line)) gives the line back. CPU only.
# usage: test_tok MODEL.gguf tests/tok_corpus.txt tests/tok_ref.txt
from std.sys import argv
from std.time import perf_counter_ns
from tokenizer import Tokenizer

def main() raises:
    var args = argv()
    var t0 = perf_counter_ns()
    var tk = Tokenizer(String(args[1]))
    var load_ms = Float64(perf_counter_ns() - t0) / 1e6
    var lines = open(String(args[2]), "r").read().split("\n")
    var refs = open(String(args[3]), "r").read().split("\n")
    var ok = 0
    var n = 0
    for i in range(len(refs)):
        if refs[i] == "":
            continue
        n += 1
        var line = String(lines[i])
        var ids = tk.encode(line)
        var got = String()
        for j in range(len(ids)):
            got += (String(",") if j else String()) + String(ids[j])
        var back = tk.decode(List[Int](ids[1:]) if tk.add_bos and tk.bos_id >= 0 else ids.copy(), keep_special=True) # drop the BOS encode adds
        if got == String(refs[i]) and back == line:
            ok += 1
        else:
            print("FAIL line", i + 1, "ids", got, "ref", refs[i], "back", back)
    print("tok", ok, "/", n, "load_ms", load_ms)
    if ok != n:
        raise Error("FAIL test_tok: " + String(n - ok) + " lines differ")
