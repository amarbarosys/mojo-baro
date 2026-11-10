from std.collections import Dict
from std.math import sqrt
from std.sys import argv

from grammar.json_value import JSONDoc, JKindString, parse_json_bytes

# Mojo port of tools/e11b_items_oracle.py score(): per game/arm correct/n,
# paired delta vs arm 0 with a one-sided 95% lower bound, multi-candidate
# count, and arm C's "follows the vector" rate. Contract:
# ~/Brain/AMDHQ/briefs/2026-09-21-e11b-contract.md.
#
# usage: mojo run score.mojo -I . RECORDS_PATH ITEMS_PATH OUT_DIR


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def bytes_to_string(b: List[UInt8]) -> String:
    return String(unsafe_from_utf8=Span(b))


def to_uint8_list(s: String) -> List[UInt8]:
    var b = s.as_bytes()
    var out = List[UInt8]()
    for i in range(len(b)):
        out.append(b[i])
    return out^


def is_alnum_lower(c: UInt8) -> Bool:
    return (c >= UInt8(ord("a")) and c <= UInt8(ord("z"))) or (
        c >= UInt8(ord("0")) and c <= UInt8(ord("9"))
    )


def norm(s: String) -> String:
    """Lowercase, strip whole-word the/a/an, non-alnum -> single space, strip."""
    var b = s.as_bytes()
    var tokens = List[String]()
    var cur = List[UInt8]()
    for i in range(len(b)):
        var c = b[i]
        var lc = c
        if c >= UInt8(ord("A")) and c <= UInt8(ord("Z")):
            lc = c + UInt8(32)
        if is_alnum_lower(lc):
            cur.append(lc)
        elif len(cur) > 0:
            tokens.append(bytes_to_string(cur))
            cur = List[UInt8]()
    if len(cur) > 0:
        tokens.append(bytes_to_string(cur))
    var out = String("")
    for i in range(len(tokens)):
        var t = tokens[i]
        if t == "the" or t == "a" or t == "an":
            continue
        if out.byte_length() > 0:
            out += " "
        out += t
    return out


def contains_whole_word(haystack: String, needle: String) -> Bool:
    var h = haystack.as_bytes()
    var n = needle.as_bytes()
    if len(n) == 0 or len(n) > len(h):
        return False
    var i = 0
    while i + len(n) <= len(h):
        var matched = True
        for k in range(len(n)):
            if h[i + k] != n[k]:
                matched = False
                break
        if matched:
            var left_ok = (i == 0) or (not is_alnum_lower(h[i - 1]))
            var right_end = i + len(n)
            var right_ok = (right_end == len(h)) or (not is_alnum_lower(h[right_end]))
            if left_ok and right_ok:
                return True
        i += 1
    return False


def str_lt(a: String, b: String) -> Bool:
    var ab = a.as_bytes()
    var bb = b.as_bytes()
    var n = len(ab) if len(ab) < len(bb) else len(bb)
    for i in range(n):
        if ab[i] != bb[i]:
            return ab[i] < bb[i]
    return len(ab) < len(bb)


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def split_lines(s: String) -> List[String]:
    var out = List[String]()
    for line in s.split("\n"):
        out.append(String(line))
    return out^


def fixed(x: Float64, digits: Int) -> String:
    """Round half to even at `digits` decimals, fixed-point, e.g. "-0.030"."""
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


def fixed_signed(x: Float64, digits: Int) -> String:
    if x < 0.0:
        return "-" + fixed(-x, digits)
    return fixed(x, digits)


# ---------------------------------------------------------------------------
# items.jsonl
# ---------------------------------------------------------------------------


struct ItemInfo(Copyable, Movable):
    var id: String
    var game: String
    var is_twin: Bool
    var gold_norm: String
    var twin_gold_norm: String
    var answers: List[List[String]]

    def __init__(out self):
        self.id = String("")
        self.game = String("")
        self.is_twin = False
        self.gold_norm = String("")
        self.twin_gold_norm = String("")
        self.answers = List[List[String]]()


def parse_item_info(line: String) raises -> ItemInfo:
    var doc = parse_json_bytes(to_uint8_list(line))
    var root = doc.root
    var r = ItemInfo()
    r.id = doc.get(doc.get_field(root, "id")).s
    r.game = doc.get(doc.get_field(root, "game")).s
    r.is_twin = doc.get(doc.get_field(root, "is_twin")).b
    r.gold_norm = norm(doc.get(doc.get_field(root, "gold")).s)
    r.twin_gold_norm = norm(doc.get(doc.get_field(root, "twin_gold")).s)
    var ans_val = doc.get(doc.get_field(root, "answers"))
    for gi in range(len(ans_val.arr)):
        var group_val = doc.get(ans_val.arr[gi])
        var group_list = List[String]()
        for ai in range(len(group_val.arr)):
            group_list.append(doc.get(group_val.arr[ai]).s)
        r.answers.append(group_list^)
    return r^


def load_items(
    path: String,
    mut items: List[ItemInfo],
    mut item_idx: Dict[String, Int],
    mut countries_ids: List[String],
    mut tips_ids: List[String],
    mut relations_ids: List[String],
) raises:
    var lines = split_lines(read_text(path))
    for i in range(len(lines)):
        var ln = lines[i]
        if ln.byte_length() == 0:
            continue
        var it = parse_item_info(ln)
        item_idx[it.id] = len(items)
        if not it.is_twin:
            if it.game == "countries":
                countries_ids.append(it.id)
            elif it.game == "tips":
                tips_ids.append(it.id)
            else:
                relations_ids.append(it.id)
        items.append(it^)


# ---------------------------------------------------------------------------
# records.jsonl
# ---------------------------------------------------------------------------


def load_records(
    path: String,
    mut rec_item: List[String],
    mut rec_arm: List[String],
    mut rec_out: List[String],
) raises:
    var lines = split_lines(read_text(path))
    for i in range(len(lines)):
        var ln = lines[i]
        if ln.byte_length() == 0:
            continue
        var doc = parse_json_bytes(to_uint8_list(ln))
        var root = doc.root
        rec_item.append(doc.get(doc.get_field(root, "item")).s)
        rec_arm.append(doc.get(doc.get_field(root, "arm")).s)
        var out_idx = doc.get_field(root, "out")
        if out_idx < 0:
            rec_out.append(String(""))
        else:
            var ov = doc.get(out_idx)
            rec_out.append(ov.s if ov.kind == JKindString else String(""))


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def score_one(padded_text: String, gold_norm: String, answers: List[List[String]]) -> Tuple[Bool, Bool]:
    var cand_set = Dict[String, Bool]()
    for gi in range(len(answers)):
        for ai in range(len(answers[gi])):
            var a = answers[gi][ai]
            if contains_whole_word(padded_text, a):
                cand_set[a] = True
    var n_distinct = len(cand_set)
    var multi = n_distinct > 1
    var ok = False
    if n_distinct == 1:
        for key in cand_set.keys():
            if key == gold_norm:
                ok = True
    return Tuple(ok, multi)


def pair_less(g1: String, a1: String, g2: String, a2: String) -> Bool:
    if g1 != g2:
        return str_lt(g1, g2)
    return str_lt(a1, a2)


def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error("usage: mojo run score.mojo -I . RECORDS_PATH ITEMS_PATH OUT_DIR")
    var records_path = String(args[1])
    var items_path = String(args[2])
    var out_dir = String(args[3])

    var items = List[ItemInfo]()
    var item_idx = Dict[String, Int]()
    var countries_ids = List[String]()
    var tips_ids = List[String]()
    var relations_ids = List[String]()
    load_items(items_path, items, item_idx, countries_ids, tips_ids, relations_ids)

    var rec_item = List[String]()
    var rec_arm = List[String]()
    var rec_out = List[String]()
    load_records(records_path, rec_item, rec_arm, rec_out)

    var tab_seen = Dict[String, Bool]()
    var tab_game = List[String]()
    var tab_arm = List[String]()
    var tab_c = Dict[String, Int]()
    var tab_n = Dict[String, Int]()
    var tab_m = Dict[String, Int]()
    var per_flat = Dict[String, Int]()

    for i in range(len(rec_item)):
        var iid = rec_item[i]
        if iid not in item_idx:
            continue
        var it = items[item_idx[iid]].copy()
        if it.is_twin:
            continue
        var g = it.game
        var a = rec_arm[i]
        var padded = String(" ") + norm(rec_out[i]) + " "
        var res = score_one(padded, it.gold_norm, it.answers)
        var ok = res[0]
        var multi = res[1]

        var tkey = g + "\x01" + a
        if tkey not in tab_seen:
            tab_seen[tkey] = True
            tab_game.append(g)
            tab_arm.append(a)
            tab_c[tkey] = 0
            tab_n[tkey] = 0
            tab_m[tkey] = 0
        tab_c[tkey] = tab_c[tkey] + (1 if ok else 0)
        tab_n[tkey] = tab_n[tkey] + 1
        tab_m[tkey] = tab_m[tkey] + (1 if multi else 0)

        var pkey = g + "\x01" + iid + "\x01" + a
        per_flat[pkey] = 1 if ok else 0

    var order = List[Int]()
    for i in range(len(tab_game)):
        order.append(i)
    for i in range(1, len(order)):
        var key = order[i]
        var kg = tab_game[key]
        var ka = tab_arm[key]
        var j = i - 1
        while j >= 0 and pair_less(kg, ka, tab_game[order[j]], tab_arm[order[j]]):
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key

    var md = String(
        "| game | arm | correct/n | delta-vs-0 | lower95 | multi-candidate | follows-vector |\n"
        "|---|---|---:|---:|---:|---:|---:|\n"
    )
    var js = String("[\n")

    for oi in range(len(order)):
        var idx = order[oi]
        var g = tab_game[idx]
        var a = tab_arm[idx]
        var tkey = g + "\x01" + a
        var c = tab_c[tkey]
        var n = tab_n[tkey]
        var m = tab_m[tkey]

        var ids: List[String]
        if g == "countries":
            ids = countries_ids.copy()
        elif g == "tips":
            ids = tips_ids.copy()
        else:
            ids = relations_ids.copy()

        var dif = List[Int]()
        if a != "0":
            for k in range(len(ids)):
                var id_ = ids[k]
                var key0 = g + "\x01" + id_ + "\x01" + "0"
                var keya = g + "\x01" + id_ + "\x01" + a
                if key0 in per_flat and keya in per_flat:
                    dif.append(per_flat[keya] - per_flat[key0])

        var d: Float64 = 0.0
        var se: Float64 = 0.0
        if len(dif) > 0:
            var s: Float64 = 0.0
            var ss: Float64 = 0.0
            for k in range(len(dif)):
                var x = Float64(dif[k])
                s += x
                ss += x * x
            d = s / Float64(len(dif))
            if len(dif) > 1:
                var num = ss - (s * s) / Float64(len(dif))
                se = sqrt(num) / Float64(len(dif))
        var lower = d - 1.645 * se

        var follows = 0
        var follows_n = 0
        if a == "C":
            for k in range(len(rec_item)):
                var iid2 = rec_item[k]
                if iid2 not in item_idx:
                    continue
                var it2 = items[item_idx[iid2]].copy()
                if it2.is_twin or it2.game != g:
                    continue
                # Matches the oracle exactly: the numerator is NOT gated on
                # this record's own arm, only the denominator is (score() in
                # tools/e11b_items_oracle.py, same asymmetry).
                if norm(rec_out[k]) == it2.twin_gold_norm:
                    follows += 1
                if rec_arm[k] == "C":
                    follows_n += 1

        md += (
            "| " + g + " | " + a + " | " + String(c) + "/" + String(n) + " | "
            + fixed_signed(d, 3) + " | " + fixed_signed(lower, 3) + " | "
            + String(m) + " | " + String(follows) + "/" + String(follows_n) + " |\n"
        )

        if oi > 0:
            js += ",\n"
        js += (
            '  {"game":"' + g + '","arm":"' + a + '","correct":' + String(c)
            + ',"n":' + String(n) + ',"delta_vs_0":' + fixed_signed(d, 9)
            + ',"lower95":' + fixed_signed(lower, 9) + ',"multi_candidate":' + String(m)
            + ',"follows_vector":' + String(follows) + ',"follows_vector_n":' + String(follows_n)
            + "}"
        )

    js += "\n]\n"

    with open(out_dir + "/table.md", "w") as f:
        f.write(md)
    with open(out_dir + "/table.json", "w") as f:
        f.write(js)

    print(md, end="")
