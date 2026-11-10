# Per-item E12-long batch check: T vs KV accuracy, handoff hash, receiver ids, timing.
# Mojo port of bench/e12long_check.py, which stays as the oracle (same output, line for line).
# Scores type "ruler" only (RULER string_match_all), the one type E12-long uses.
#
# build: ./.venv/bin/mojo build bench/e12long_check.mojo -I . -o .work/e12long_check
# usage: .work/e12long_check TASKS.json RAW1.raw.json [RAW2.raw.json ...]

from std.math import sqrt
from std.sys import argv

from grammar.json_value import JSONDoc, parse_json_file


def field_str(doc: JSONDoc, idx: Int, key: String) -> String:
    var vi = doc.get_field(idx, key)
    return doc.get(vi).s if vi >= 0 else String("")


def field_num(doc: JSONDoc, idx: Int, key: String) -> Float64:
    var vi = doc.get_field(idx, key)
    return doc.get(vi).n if vi >= 0 else 0.0


def same_int_array(a: JSONDoc, ai: Int, b: JSONDoc, bi: Int) -> Bool:
    var va = a.get(ai).arr.copy()
    var vb = b.get(bi).arr.copy()
    if len(va) != len(vb):
        return False
    for j in range(len(va)):
        if a.get(va[j]).n != b.get(vb[j]).n:
            return False
    return True


def ruler_correct(tasks: JSONDoc, task_idx: Int, raw: JSONDoc, arm_idx: Int) -> Bool:
    """RULER string_match_all: every expected answer is a case-insensitive substring."""
    if field_str(raw, arm_idx, "error") != "":
        return False
    var scored = field_str(raw, arm_idx, "scored_text").lower()
    var exp = tasks.get(tasks.get_field(task_idx, "expected")).arr.copy()
    for j in range(len(exp)):
        if not (tasks.get(exp[j]).s.lower() in scored):
            return False
    return True


def fixed(x: Float64, digits: Int) -> String:
    """Round half to even at `digits` decimals, like Python's f"{x:.Nf}" for these values."""
    var scale = 1
    for _ in range(digits):
        scale *= 10
    var v = x * Float64(scale)
    var n = Int(v)
    var frac = v - Float64(n)
    if frac > 0.5 or (frac == 0.5 and n % 2 == 1):
        n += 1
    var ip = String(n // scale)
    if digits == 0:
        return ip
    var fp = String(n % scale)
    while fp.byte_length() < digits:
        fp = "0" + fp
    return ip + "." + fp


def pad_right(s: String, width: Int) -> String:
    var out = s
    while out.byte_length() < width:
        out += " "
    return out


def median(var xs: List[Float64]) -> Float64:
    for i in range(1, len(xs)):
        var v = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > v:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = v
    var n = len(xs)
    if n % 2 == 1:
        return xs[n // 2]
    return (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def main() raises:
    var args = argv()
    if len(args) < 3:
        print("usage: e12long_check TASKS.json RAW1.raw.json [RAW2.raw.json ...]")
        return
    var tasks = parse_json_file(String(args[1]))
    var troot = tasks.get(tasks.root).arr.copy()

    var ok_t = 0
    var ok_k = 0
    var disc_b = 0  # KV right, T wrong
    var disc_c = 0  # T right, KV wrong
    var n = 0
    var bad = List[String]()
    var r_t = List[Float64]()
    var r_k = List[Float64]()
    var mi = List[Float64]()

    for ri in range(2, len(args)):
        var raw = parse_json_file(String(args[ri]))
        var items = raw.get(raw.get_field(raw.root, "items")).arr.copy()
        for ii in range(len(items)):
            var it = items[ii]
            var id = field_str(raw, it, "id")
            var task_idx = -1
            for j in range(len(troot)):
                if field_str(tasks, troot[j], "id") == id:
                    task_idx = troot[j]
                    break
            if task_idx < 0:
                raise Error("task id not in tasks file: " + id)
            var arms = raw.get(raw.get_field(it, "arms")).arr.copy()
            var t = -1
            var k = -1
            for j in range(len(arms)):
                var name = field_str(raw, arms[j], "arm")
                if name == "T":
                    t = arms[j]
                elif name == "KV":
                    k = arms[j]
            if t < 0 or k < 0:
                raise Error("item " + id + " lacks a T or KV arm")

            var s_t = ruler_correct(tasks, task_idx, raw, t)
            var s_k = ruler_correct(tasks, task_idx, raw, k)
            if s_t:
                ok_t += 1
            if s_k:
                ok_k += 1
            if s_k and not s_t:
                disc_b += 1
            if s_t and not s_k:
                disc_c += 1
            var hash_eq = field_str(raw, t, "handoff_hash") == field_str(raw, k, "handoff_hash")
            var ids_eq = same_int_array(
                raw, raw.get_field(t, "generated_ids"), raw, raw.get_field(k, "generated_ids")
            )
            if not (hash_eq and ids_eq) or s_t != s_k:
                bad.append(id)
            var rt = field_num(raw, t, "receiver_s")
            var rk = field_num(raw, k, "receiver_s")
            var m = 1000.0 * (field_num(raw, k, "mint_s") + field_num(raw, k, "ingest_s"))
            r_t.append(rt)
            r_k.append(rk)
            mi.append(m)
            n += 1
            print(
                pad_right(id, 28), " T ", 1 if s_t else 0, " KV ", 1 if s_k else 0,
                " hash ", "=" if hash_eq else "X", " ids ", "=" if ids_eq else "X",
                " recv ", fixed(rt, 2), "/", fixed(rk, 2), " s mint+ingest ", fixed(m, 0), " ms",
                sep="",
            )

    var bad_s = String("none")
    if len(bad) > 0:
        bad_s = "["
        for j in range(len(bad)):
            if j > 0:
                bad_s += ", "
            bad_s += "'" + bad[j] + "'"
        bad_s += "]"
    print(
        "n=", n, "  T ", ok_t, "/", n, "  KV ", ok_k, "/", n,
        "  median recv T ", fixed(median(r_t^), 2), " s KV ", fixed(median(r_k^), 2), " s",
        "  median mint+ingest ", fixed(median(mi^), 0), " ms  mismatched: ", bad_s,
        sep="",
    )
    # E12 paired Wald CI on KV - T (06-experiments.md, E12 and E12-long thresholds).
    var nf = Float64(n)
    var d = Float64(disc_b - disc_c) / nf
    var se = sqrt(Float64(disc_b + disc_c) - Float64((disc_b - disc_c) * (disc_b - disc_c)) / nf) / nf
    var lo = 100.0 * (d - 1.645 * se)
    var hi = 100.0 * (d + 1.645 * se)
    print(
        "gate: n=", n, " b=", disc_b, " c=", disc_c, " d=", pp(100.0 * d), " pp  90% CI [",
        pp(lo), ", ", pp(hi), "] pp  inside +/-5 pp: ", "yes" if lo > -5.0 and hi < 5.0 else "no",
        sep="",
    )


def pp(x: Float64) -> String:
    return ("-" if x < 0.0 else "+") + fixed(abs(x), 1)
