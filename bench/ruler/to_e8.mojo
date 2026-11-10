# Turn bench/ruler/gen.py JSONL sets into an E8/E12 task file.
# Mojo port of bench/ruler/to_e8.py (kept as the oracle): same tasks, same token ids,
# tokenized in-process with serve/tokenizer.mojo instead of through .work/baro-tokenize.
#
# RULER prompts are completion-style: they end with an answer prefix (" The special
# magic ... is", " Answer: ..."). The E8 harness is chat-shaped and hands B its own
# "Give only the answer" turn, so the prefix is cut off and the rest becomes the user
# message. Scoring is RULER's string_match_all (bench/e8_score.py, type "ruler").
#
# build: ./.venv/bin/mojo build bench/ruler/to_e8.mojo -I . -I serve -o .work/to_e8
# usage: .work/to_e8 OUT.json IN1.jsonl [IN2.jsonl ...]

from std.os import getenv
from std.sys import argv

from grammar.json_value import JSONDoc, parse_json_bytes
from tokenizer import Tokenizer

comptime SYS = String("You are a helpful assistant. Read the text carefully and answer the question.")


def json_escape(s: String) -> String:
    var out = String("")
    for cp in s.codepoint_slices():
        var c = String(cp)
        if c == "\"":
            out += "\\\""
        elif c == "\\":
            out += "\\\\"
        elif c == "\n":
            out += "\\n"
        elif c == "\r":
            out += "\\r"
        elif c == "\t":
            out += "\\t"
        elif c.byte_length() == 1 and Int(c.as_bytes()[0]) < 0x20:
            var h = String(hex(Int(c.as_bytes()[0])))  # "0x1f"
            var digits = String(h.removeprefix("0x"))
            while digits.byte_length() < 4:
                digits = "0" + digits
            out += "\\u" + digits
        else:
            out += c
    return out


def rfind(s: String, sub: String) -> Int:
    var best = -1
    var start = 0
    while True:
        var i = s.find(sub, start)
        if i < 0:
            return best
        best = i
        start = i + 1


def strip_prefix(prompt: String) raises -> String:
    var cut = max(rfind(prompt, " The special magic "), rfind(prompt, " Answer: "))
    if cut < 0:
        raise Error("no answer prefix found in a RULER prompt")
    return String(unsafe_from_utf8=prompt.as_bytes()[:cut])


def chat_prompt(user: String) -> String:
    # Same template as tools/generate_e8_tasks.mojo chat_prompt().
    return (
        "<|im_start|>system\n" + SYS + "<|im_end|>\n"
        + "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"
    )


def main() raises:
    var args = argv()
    if len(args) < 3:
        print("usage: to_e8 OUT.json IN1.jsonl [IN2.jsonl ...]")
        return
    var gguf = getenv(
        "BARO_GGUF",
        getenv("HOME", "")
        + "/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf",
    )
    var tok = Tokenizer(gguf)
    var out = String("[")
    var n = 0
    var lo = -1
    var hi = 0
    for ai in range(2, len(args)):
        var text: String
        with open(String(args[ai]), "r") as f:
            text = f.read()
        for line in text.split("\n"):
            if String(line).strip() == "":
                continue
            var doc = parse_json_bytes(List[UInt8](String(line).as_bytes()))
            var r = doc.root
            var full = chat_prompt(strip_prefix(doc.get(doc.get_field(r, "prompt")).s))
            var ids = tok.encode(full, add_special=False)
            if n > 0:
                out += ", "
            n += 1
            lo = len(ids) if lo < 0 or len(ids) < lo else lo
            hi = len(ids) if len(ids) > hi else hi
            out += "{\"id\": \"" + json_escape(doc.get(doc.get_field(r, "id")).s) + "\""
            out += ", \"type\": \"ruler\""
            out += ", \"task\": \"" + json_escape(doc.get(doc.get_field(r, "task")).s) + "\""
            out += ", \"size\": " + String(Int(doc.get(doc.get_field(r, "size")).n))
            out += ", \"full_prompt\": \"" + json_escape(full) + "\""
            out += ", \"tokens\": ["
            for k in range(len(ids)):
                if k > 0:
                    out += ", "
                out += String(ids[k])
            out += "], \"expected\": ["
            var ans = doc.get(doc.get_field(r, "answers")).arr.copy()
            for k in range(len(ans)):
                if k > 0:
                    out += ", "
                out += "\"" + json_escape(doc.get(ans[k]).s) + "\""
            out += "]}"
    out += "]"
    with open(String(args[1]), "w") as f:
        f.write(out)
    print(n, "tasks ->", String(args[1]) + "; tokens min", lo, "max", hi)
