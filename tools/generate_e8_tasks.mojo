# Generate bench/data/e8_tasks.json for the E8 latent-handoff evaluator.
#
# Round 4 (2026-09-09), rebuilt after the 2026-09-09 E8 run came back
# INCONCLUSIVE. That run could not discriminate because of two defects in the
# task set, both fixed here:
#
#   1. Math was saturated. The 20 hand-written 2-to-3 step problems scored
#      20/20 in every arm including arm0 (no handoff), so the set could not
#      tell a working handoff from one that does nothing. Replaced with
#      N_MATH GSM8K test items filtered to >= MIN_STEPS annotated steps.
#
#   2. The schema was never shown to the model. The system prompt said
#      "adhering strictly to the schema" while no schema appeared in the
#      prompt, and 15 of 20 rows named keys absent from the schema they
#      referenced (picked by filename, not content). Schemas are now
#      interpolated into the prompt, rows are rebuilt against real
#      properties, and validate_specs() fails the build on any future drift.
#
# usage:
#   mojo run tools/generate_e8_tasks.mojo -I . -I serve
#   mojo run tools/generate_e8_tasks.mojo --check-tokenizer .work/e8_tasks.round3.json
#
# The GSM8K jsonl comes from tools/gsm8k-parquet-to-jsonl.py (parquet needs
# pyarrow, which has no Mojo reader; that is the one Python step here).

from std.os import getenv
from std.sys import argv

from grammar.json_value import (
    JSONDoc,
    JSONValue,
    JKindArray,
    JKindBool,
    JKindNull,
    JKindNumber,
    JKindObject,
    JKindString,
    parse_json_bytes,
    parse_json_file,
    read_file_bytes,
)
from tokenizer import Tokenizer

comptime CORPUS = String("grammar/corpus/")
comptime OUT_PATH = String("bench/data/e8_tasks.json")

comptime MATH_SYS = String(
    "You are a precise mathematical reasoning agent. Solve the problem and"
    " state the final integer answer at the end as: Answer: <number>"
)
comptime JSON_SYS_HEAD = String(
    "You are a structured extraction agent. Output JSON adhering strictly to"
    " this schema:\n"
)


# ---------------------------------------------------------------------------
# byte helpers
# ---------------------------------------------------------------------------

def bytes_to_string(b: List[UInt8]) -> String:
    return String(unsafe_from_utf8=Span(b))


def hex_digit(v: Int) -> UInt8:
    return UInt8(ord("0") + v) if v < 10 else UInt8(ord("a") + v - 10)


def json_escape(s: String) -> String:
    var b = s.as_bytes()
    var out = List[UInt8]()
    for i in range(len(b)):
        var c = b[i]
        if c == UInt8(ord('"')):
            out.append(UInt8(ord("\\")))
            out.append(UInt8(ord('"')))
        elif c == UInt8(ord("\\")):
            out.append(UInt8(ord("\\")))
            out.append(UInt8(ord("\\")))
        elif c == UInt8(10):
            out.append(UInt8(ord("\\")))
            out.append(UInt8(ord("n")))
        elif c == UInt8(13):
            out.append(UInt8(ord("\\")))
            out.append(UInt8(ord("r")))
        elif c == UInt8(9):
            out.append(UInt8(ord("\\")))
            out.append(UInt8(ord("t")))
        elif c < UInt8(0x20):
            out.append(UInt8(ord("\\")))
            out.append(UInt8(ord("u")))
            out.append(UInt8(ord("0")))
            out.append(UInt8(ord("0")))
            out.append(hex_digit(Int(c) >> 4))
            out.append(hex_digit(Int(c) & 0xF))
        else:
            out.append(c)
    return bytes_to_string(out)


def compact_json(s: String) -> String:
    """Strip whitespace outside string literals so the schema costs fewer tokens."""
    var b = s.as_bytes()
    var out = List[UInt8]()
    var in_str = False
    var esc = False
    for i in range(len(b)):
        var c = b[i]
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == UInt8(ord("\\")):
                esc = True
            elif c == UInt8(ord('"')):
                in_str = False
        elif c == UInt8(ord('"')):
            in_str = True
            out.append(c)
        elif c == UInt8(32) or c == UInt8(9) or c == UInt8(10) or c == UInt8(13):
            continue
        else:
            out.append(c)
    return bytes_to_string(out)


def count_sub(hay: String, needle: String) -> Int:
    var b = hay.as_bytes()
    var t = needle.as_bytes()
    var n = 0
    var i = 0
    while i + len(t) <= len(b):
        var hit = True
        for j in range(len(t)):
            if b[i + j] != t[j]:
                hit = False
                break
        if hit:
            n += 1
            i += len(t)
        else:
            i += 1
    return n


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


# ---------------------------------------------------------------------------
# JSON value comparison, for the enum/const checks
# ---------------------------------------------------------------------------

def jval_equal(a: JSONDoc, ia: Int, b: JSONDoc, ib: Int) -> Bool:
    var va = a.get(ia)
    var vb = b.get(ib)
    if va.kind != vb.kind:
        return False
    if va.kind == JKindString:
        return va.s == vb.s
    if va.kind == JKindNumber:
        return va.n == vb.n
    if va.kind == JKindBool:
        return va.b == vb.b
    if va.kind == JKindNull:
        return True
    return False


# ---------------------------------------------------------------------------
# the JSON task rows: (schema file, prompt, expected JSON)
# every key in `expected` must exist in the named schema's `properties`, and
# every const/enum value must be legal -- validate_specs enforces both
# ---------------------------------------------------------------------------

def json_specs(
    mut files: List[String], mut prompts: List[String], mut expects: List[String]
):
    def add(f: String, p: String, e: String) {mut files, mut prompts, mut expects}:
        files.append(f)
        prompts.append(p)
        expects.append(e)

    add("01_simple_required.json",
        "Extract the user details: Alice is a 29 year old engineer.",
        '{"name":"Alice","age":29}')
    add("01_simple_required.json",
        "Extract person info: Bob is 42 years old.",
        '{"name":"Bob","age":42}')
    add("02_optional_params.json",
        "Search request: user searched for 'rocm benchmarks', showing 20 results starting at result 60.",
        '{"query":"rocm benchmarks","limit":20,"offset":60}')
    add("02_optional_params.json",
        "Search request: query 'Mojo standard library' with a limit of 5 results.",
        '{"query":"Mojo standard library","limit":5}')
    add("10_bool_field.json",
        "Feature flag state: the background compiler is switched on.",
        '{"enabled":true}')
    add("10_bool_field.json",
        "Feature flag state: telemetry reporting is switched off.",
        '{"enabled":false}')
    add("12_const_field.json",
        "Tool invocation: get the weather for Tokyo.",
        '{"kind":"weather.get","city":"Tokyo"}')
    add("12_const_field.json",
        "Tool invocation: get the weather for Boston.",
        '{"kind":"weather.get","city":"Boston"}')
    add("15_min_max_length_string.json",
        "Record the airport code for Frankfurt, which is FRA.",
        '{"code":"FRA"}')
    add("19_all_types_object.json",
        "Log entry: label is 'build', count is 7, ratio is 0.5, passing is true, and the error slot is empty.",
        '{"s":"build","i":7,"n":0.5,"b":true,"z":null}')
    add("20_multi_optional.json",
        "Counters: a is 3, b is 8, d is 12. c was not reported.",
        '{"a":3,"b":8,"d":12}')
    add("21_weather_tool_call.json",
        "Tool call: weather for Tokyo in celsius.",
        '{"location":"Tokyo","unit":"celsius"}')
    add("21_weather_tool_call.json",
        "Tool call: weather for Boston in fahrenheit.",
        '{"location":"Boston","unit":"fahrenheit"}')
    add("22_search_tool_call.json",
        "Search request: find documentation on 'hugetlbfs'.",
        '{"query":"hugetlbfs"}')
    add("24_negative_number_field.json",
        "Temperature change: the reading fell by 4.5 degrees.",
        '{"delta":-4.5}')
    add("25_const_and_enum_mixed.json",
        "Style operation: add 16 px.",
        '{"op":"add","unit":"px","value":16}')
    add("25_const_and_enum_mixed.json",
        "Style operation: add 2 rem.",
        '{"op":"add","unit":"rem","value":2}')
    add("28_boolean_enum_like.json",
        "Open the file for reading only, with strict checking on.",
        '{"mode":"read","strict":true}')
    add("28_boolean_enum_like.json",
        "Open the file for reading and writing, with strict checking off.",
        '{"mode":"read_write","strict":false}')
    add("30_percentage_number.json",
        "Metrics report: memory utilization is at 85 percent.",
        '{"pct":85,"label":"memory utilization"}')


def validate_specs(
    files: List[String], expects: List[String]
) raises -> Int:
    """Fail the build if a row names keys or values its schema rejects.

    This is the check that was missing when the 2026-09-09 run shipped 15 of
    20 rows whose expected keys did not exist in the schema they named.
    """
    var errors = List[String]()
    for i in range(len(files)):
        var tag = String("json_") + String(i + 1)
        var sdoc = parse_json_file(CORPUS + files[i])
        var sroot = sdoc.root
        var props_i = sdoc.get_field(sroot, "properties")
        if props_i < 0:
            errors.append(tag + ": " + files[i] + " has no `properties`, unusable for key extraction")
            continue
        var props = sdoc.get(props_i)

        var ebytes = expects[i].as_bytes()
        var eb = List[UInt8]()
        for k in range(len(ebytes)):
            eb.append(ebytes[k])
        var edoc = parse_json_bytes(eb^)
        var eroot = edoc.get(edoc.root)

        for k in range(len(eroot.obj_keys)):
            var key = eroot.obj_keys[k]
            var pi = props.find_key(key)
            if pi < 0:
                var have = String("")
                for pk in range(len(props.obj_keys)):
                    if pk > 0:
                        have += ","
                    have += props.obj_keys[pk]
                errors.append(tag + ": key '" + key + "' absent from " + files[i] + " (has " + have + ")")
                continue
            var ci = sdoc.get_field(pi, "const")
            if ci >= 0 and not jval_equal(sdoc, ci, edoc, eroot.obj_vals[k]):
                errors.append(tag + ": '" + key + "' violates const in " + files[i])
            var ei = sdoc.get_field(pi, "enum")
            if ei >= 0:
                var opts = sdoc.get(ei)
                var ok = False
                for oi in range(len(opts.arr)):
                    if jval_equal(sdoc, opts.arr[oi], edoc, eroot.obj_vals[k]):
                        ok = True
                        break
                if not ok:
                    errors.append(tag + ": '" + key + "' not in enum of " + files[i])

        var ri = sdoc.get_field(sroot, "required")
        if ri >= 0:
            var req = sdoc.get(ri)
            for rq in range(len(req.arr)):
                var rkey = sdoc.get(req.arr[rq]).s
                if eroot.find_key(rkey) < 0:
                    errors.append(tag + ": required key '" + rkey + "' missing from expected")

    if len(errors) > 0:
        print("json spec validation FAILED:")
        for i in range(len(errors)):
            print("  " + errors[i])
        raise Error("json specs inconsistent with their schemas")
    print("json spec validation OK (", len(files), "rows consistent with their schemas )")
    return len(files)


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def gsm_final_answer(ans: String) raises -> Int:
    """Parse the integer after the last '####' marker."""
    var b = ans.as_bytes()
    var at = -1
    var i = 0
    while i + 4 <= len(b):
        if (
            b[i] == UInt8(ord("#"))
            and b[i + 1] == UInt8(ord("#"))
            and b[i + 2] == UInt8(ord("#"))
            and b[i + 3] == UInt8(ord("#"))
        ):
            at = i
        i += 1
    if at < 0:
        raise Error("gsm8k answer has no #### marker")
    var neg = False
    var val = 0
    var seen = False
    for k in range(at + 4, len(b)):
        var c = b[k]
        if c == UInt8(ord("-")) and not seen:
            neg = True
        elif c >= UInt8(ord("0")) and c <= UInt8(ord("9")):
            val = val * 10 + (Int(c) - 48)
            seen = True
        elif c == UInt8(ord(",")) or c == UInt8(32):
            continue
        elif seen:
            break
    if not seen:
        raise Error("gsm8k answer has no integer after ####")
    return -val if neg else val


def load_gsm8k_hard(
    path: String, want: Int, min_steps: Int,
    mut out_q: List[String], mut out_a: List[Int],
) raises:
    """Pick the `want` items with the most annotated reasoning steps.

    GSM8K annotates each arithmetic step as <<a*b=c>>, so counting those is a
    usable difficulty proxy. Selection is deterministic: descending step
    count, original file order within a count, which is what the bucket walk
    below produces without needing a comparator.
    """
    var raw = read_text(path)
    var questions = List[String]()
    var answers = List[Int]()
    var steps = List[Int]()
    var max_steps = 0

    for line in raw.split("\n"):
        var ln = String(line)
        if len(ln.as_bytes()) < 2:
            continue
        var lb = ln.as_bytes()
        var buf = List[UInt8]()
        for k in range(len(lb)):
            buf.append(lb[k])
        var doc = parse_json_bytes(buf^)
        var qi = doc.get_field(doc.root, "question")
        var ai = doc.get_field(doc.root, "answer")
        if qi < 0 or ai < 0:
            continue
        var atext = doc.get(ai).s
        var st = count_sub(atext, "<<")
        if st < min_steps:
            continue
        questions.append(doc.get(qi).s)
        answers.append(gsm_final_answer(atext))
        steps.append(st)
        if st > max_steps:
            max_steps = st

    var total = len(questions)
    if total < want:
        raise Error(
            String("only ") + String(total) + " GSM8K items at >= "
            + String(min_steps) + " steps, need " + String(want)
        )

    var taken = 0
    var lo_used = max_steps
    var s = max_steps
    while s >= min_steps and taken < want:
        for i in range(total):
            if steps[i] == s:
                out_q.append(questions[i])
                out_a.append(answers[i])
                taken += 1
                lo_used = s
                if taken == want:
                    break
        s -= 1
    print(
        "gsm8k:", total, "items at >=", min_steps, "steps, took hardest", want,
        "(", max_steps, "down to", lo_used, "steps )",
    )


# ---------------------------------------------------------------------------
# emit
# ---------------------------------------------------------------------------

def chat_prompt(sys_msg: String, user_msg: String) -> String:
    return (
        "<|im_start|>system\n" + sys_msg + "<|im_end|>\n"
        + "<|im_start|>user\n" + user_msg + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


def pad3(n: Int) -> String:
    if n < 10:
        return String("00") + String(n)
    if n < 100:
        return String("0") + String(n)
    return String(n)


def pad2(n: Int) -> String:
    return (String("0") + String(n)) if n < 10 else String(n)


def check_tokenizer(tok: Tokenizer, path: String) raises:
    """Re-tokenize every prompt in an existing task file, compare to its ids.

    Proves the Mojo tokenizer agrees with whatever produced that file before
    the generated ids are trusted.
    """
    var doc = parse_json_file(path)
    var root = doc.get(doc.root)
    var n_ok = 0
    var n_bad = 0
    for j in range(len(root.arr)):
        var item = root.arr[j]
        var pi = doc.get_field(item, "full_prompt")
        var ti = doc.get_field(item, "tokens")
        var idi = doc.get_field(item, "id")
        if pi < 0 or ti < 0:
            continue
        var want = doc.get(ti)
        var got = tok.encode(doc.get(pi).s, add_special=False)
        var same = len(got) == len(want.arr)
        if same:
            for k in range(len(got)):
                if got[k] != Int(doc.get(want.arr[k]).n):
                    same = False
                    break
        if same:
            n_ok += 1
        else:
            n_bad += 1
            if n_bad <= 3:
                print("  MISMATCH", doc.get(idi).s, "mojo len", len(got), "stored len", len(want.arr))
    print("tokenizer check:", n_ok, "match,", n_bad, "mismatch, against", path)
    if n_bad > 0:
        raise Error("mojo tokenizer disagrees with the stored ids")


def main() raises:
    var args = argv()
    var gguf = getenv(
        "BARO_E8_GGUF",
        getenv("HOME", "")
        + "/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_0-pure.gguf",
    )
    var gsm_path = getenv(
        "E8_GSM8K_JSONL", getenv("HOME", "") + "/Models/datasets/gsm8k/main/test.jsonl"
    )
    var n_math = atol(getenv("E8_N_MATH", "100"))
    var min_steps = atol(getenv("E8_MIN_STEPS", "5"))

    var tok = Tokenizer(gguf)

    for i in range(1, len(args)):
        if String(args[i]) == "--check-tokenizer" and i + 1 < len(args):
            check_tokenizer(tok, String(args[i + 1]))
            return

    var files = List[String]()
    var prompts = List[String]()
    var expects = List[String]()
    json_specs(files, prompts, expects)
    var n_json = validate_specs(files, expects)

    var gq = List[String]()
    var ga = List[Int]()
    load_gsm8k_hard(gsm_path, n_math, min_steps, gq, ga)

    var out = String("[\n")
    var longest = 0
    var longest_id = String("")

    for i in range(n_json):
        var schema_text = compact_json(read_text(CORPUS + files[i]))
        var sys_msg = JSON_SYS_HEAD + schema_text
        var full = chat_prompt(sys_msg, prompts[i])
        var ids = tok.encode(full, add_special=False)
        var tid = String("json_") + pad2(i + 1)
        if len(ids) > longest:
            longest = len(ids)
            longest_id = tid
        out += '  {\n    "id": "' + tid + '",\n'
        out += '    "type": "json",\n'
        out += '    "schema_file": "' + json_escape(CORPUS + files[i]) + '",\n'
        out += '    "prompt_text": "' + json_escape(prompts[i]) + '",\n'
        out += '    "full_prompt": "' + json_escape(full) + '",\n'
        out += '    "tokens": ['
        for k in range(len(ids)):
            if k > 0:
                out += ","
            out += String(ids[k])
        out += "],\n"
        out += '    "expected": ' + expects[i] + "\n  },\n"

    for i in range(len(gq)):
        var full = chat_prompt(MATH_SYS, gq[i])
        var ids = tok.encode(full, add_special=False)
        var tid = String("math_") + pad3(i + 1)
        if len(ids) > longest:
            longest = len(ids)
            longest_id = tid
        out += '  {\n    "id": "' + tid + '",\n'
        out += '    "type": "math",\n'
        out += '    "prompt_text": "' + json_escape(gq[i]) + '",\n'
        out += '    "full_prompt": "' + json_escape(full) + '",\n'
        out += '    "tokens": ['
        for k in range(len(ids)):
            if k > 0:
                out += ","
            out += String(ids[k])
        out += "],\n"
        out += '    "expected": ' + String(ga[i]) + "\n  }"
        out += ",\n" if i + 1 < len(gq) else "\n"

    out += "]\n"

    with open(OUT_PATH, "w") as f:
        f.write(out)

    print("Generated", n_json + len(gq), "tasks (", n_json, "JSON,", len(gq), "math ) to", OUT_PATH)
    print("longest prompt:", longest, "tokens (", longest_id, ") -- BARO_E8_TMAX must exceed longest + max(COT_MAX,K32) + ans_max")
