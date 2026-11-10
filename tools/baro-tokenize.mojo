from std.sys import argv
from std.time import perf_counter_ns
from tokenizer import Tokenizer


def main() raises:
    var args = argv()
    if len(args) < 3:
        print("usage: baro-tokenize (encode TEXT_FILE | decode IDS_FILE | decode-keep IDS_FILE | batch NUL_TEXTS_FILE | count NUL_TEXTS_FILE | info) MODEL.gguf")
        return
    var t0 = perf_counter_ns()
    var tok = Tokenizer(String(args[len(args) - 1]))
    var t_load = Float64(perf_counter_ns() - t0) / 1e9
    var cmd = String(args[1])
    if cmd == "info":
        print("pre:", tok.pre, "vocab:", len(tok.tokens), "merges:", len(tok.merges), "specials:", len(tok.specials),
              "bos:", tok.bos_id, "eos:", tok.eos_id, "add_bos:", tok.add_bos, "load_s:", t_load)
        return
    var data: String
    with open(String(args[2]), "r") as f:
        data = f.read()
    if cmd == "encode":
        for id in tok.encode(data):
            print(id)
    elif cmd == "decode" or cmd == "decode-keep":
        var ids = List[Int]()
        for part in data.split():
            ids.append(atol(part))
        print(tok.decode(ids, keep_special=cmd == "decode-keep"), end="")
    elif cmd == "count":
        for t in data.split("\0"):
            print(len(tok.encode(String(t))))
    elif cmd == "batch":
        for t in data.split("\0"):
            var line = String("")
            for id in tok.encode(String(t)):
                line += String(id) + " "
            print(line)
    else:
        raise Error("unknown command " + cmd)
