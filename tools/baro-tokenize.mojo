from std.sys import argv
from std.time import perf_counter_ns
from tokenizer import Tokenizer


def hex_of(s: String) -> String:
    comptime HEX = "0123456789abcdef"
    var o = List[UInt8](capacity=2 * len(s.as_bytes()))
    for b in s.as_bytes():
        o.append(HEX.as_bytes()[Int(b >> 4)])
        o.append(HEX.as_bytes()[Int(b & 15)])
    return String(unsafe_from_utf8=Span(o))


def main() raises:
    var args = argv()
    if len(args) < 3:
        print("usage: baro-tokenize (encode TEXT_FILE | decode IDS_FILE | decode-keep IDS_FILE | batch NUL_TEXTS_FILE | count NUL_TEXTS_FILE | ids CASE_LIST | ids-raw CASE_LIST | roundtrip CASE_LIST | bench TEXT_FILE | pieces X | info X) MODEL.gguf")
        return
    var t0 = perf_counter_ns()
    var tok = Tokenizer(String(args[len(args) - 1]))
    var t_load = Float64(perf_counter_ns() - t0) / 1e9
    var cmd = String(args[1])
    if cmd == "info":
        print("pre:", tok.pre, "vocab:", len(tok.tokens), "merges:", len(tok.merges), "specials:", len(tok.specials),
              "bos:", tok.bos_id, "eos:", tok.eos_id, "add_bos:", tok.add_bos, "load_s:", t_load)
        return
    if cmd == "pieces":  # one hex line per id: decode([id], keep_special) bytes (bench/tok-gate.sh)
        for id in range(len(tok.tokens)):
            var one = List[Int]()
            one.append(id)
            print(hex_of(tok.decode(one, keep_special=True, strip_space=False)))
        return
    var data: String
    with open(String(args[2]), "r") as f:
        data = f.read()
    if cmd == "ids" or cmd == "ids-raw" or cmd == "roundtrip":  # data = one case path per line
        for path in data.split("\n"):
            if path.byte_length() == 0:
                continue
            var text: String
            with open(String(path), "r") as f:
                text = f.read()
            var ids = tok.encode(text, add_special=cmd == "ids")
            if cmd == "roundtrip":
                var back = tok.decode(ids, keep_special=True)
                print("OK" if back == text else "DIFF " + hex_of(back))
                continue
            var line = String("")
            for k in range(len(ids)):
                if k > 0:
                    line += " "
                line += String(ids[k])
            print(line)
        return
    if cmd == "bench":  # load, first encode, then medians of BENCH_REPS (default 21) encodes and decodes of one file
        var reps = 21
        var t1 = perf_counter_ns()
        var ids = tok.encode(data)
        var first_ms = Float64(perf_counter_ns() - t1) / 1e6
        var enc = List[Float64]()
        var dec = List[Float64]()
        var nbytes = 0
        for _ in range(reps):
            var a = perf_counter_ns()
            var again = tok.encode(data)
            var b = perf_counter_ns()
            var text = tok.decode(again, keep_special=True)
            var c = perf_counter_ns()
            nbytes = text.byte_length()
            enc.append(Float64(b - a) / 1e6)
            dec.append(Float64(c - b) / 1e6)
        sort(enc)
        sort(dec)
        var n = Float64(len(ids))
        print("ids:", len(ids), "bytes:", data.byte_length(), "load_ms:", t_load * 1e3, "first_encode_ms:", first_ms,
              "first_token_ms:", t_load * 1e3 + first_ms, "encode_ms_med:", enc[reps // 2], "encode_ms_min:", enc[0],
              "encode_tok_s:", n / enc[reps // 2] * 1e3, "decode_ms_med:", dec[reps // 2], "decode_tok_s:", n / dec[reps // 2] * 1e3,
              "decoded_bytes:", nbytes)
        return
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
