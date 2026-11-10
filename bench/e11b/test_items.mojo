from std.collections import Dict
from std.subprocess import run

from grammar.json_value import (
    JSONDoc,
    JSONValue,
    JKindNull,
    JKindBool,
    JKindArray,
    parse_json_bytes,
)

# Independent verifier: parses bench/e11b/items.jsonl itself (does not call
# the builder or the oracle) and checks every property from the ITEMS-MOJO
# brief plus the contract's item-list invariants.

comptime ITEMS_PATH = String("bench/e11b/items.jsonl")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def bytes_to_string(b: List[UInt8]) -> String:
    return String(unsafe_from_utf8=Span(b))


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def split_lines(s: String) -> List[String]:
    var out = List[String]()
    for line in s.split("\n"):
        out.append(String(line))
    return out^


def split_space(s: String) -> List[String]:
    var out = List[String]()
    for part in s.split(" "):
        out.append(String(part))
    return out^


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


def has_digit(s: String) -> Bool:
    var b = s.as_bytes()
    for i in range(len(b)):
        if b[i] >= UInt8(ord("0")) and b[i] <= UInt8(ord("9")):
            return True
    return False


def norm(s: String) -> String:
    """Lowercase, strip whole-word the/a/an, non-alnum -> single space, strip.

    Kept identical to build_items.mojo's norm() (small enough that a shared
    import isn't worth the extra file / ownership question).
    """
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


# ---------------------------------------------------------------------------
# item row
# ---------------------------------------------------------------------------


struct ItemRow(Copyable, Movable):
    var id: String
    var game: String
    var split: String
    var is_twin: Bool
    var twin_id: String
    var name: String
    var a_text: String
    var b_question: String
    var gold: String
    var twin_gold: String
    var has_relation: Bool
    var relation: String
    var answers: List[List[String]]

    def __init__(out self):
        self.id = String("")
        self.game = String("")
        self.split = String("")
        self.is_twin = False
        self.twin_id = String("")
        self.name = String("")
        self.a_text = String("")
        self.b_question = String("")
        self.gold = String("")
        self.twin_gold = String("")
        self.has_relation = False
        self.relation = String("")
        self.answers = List[List[String]]()


def parse_item(line: String) raises -> ItemRow:
    var doc = parse_json_bytes(to_uint8_list(line))
    var root = doc.root
    var r = ItemRow()
    r.id = doc.get(doc.get_field(root, "id")).s
    r.game = doc.get(doc.get_field(root, "game")).s
    r.split = doc.get(doc.get_field(root, "split")).s
    r.is_twin = doc.get(doc.get_field(root, "is_twin")).b
    r.twin_id = doc.get(doc.get_field(root, "twin_id")).s
    r.name = doc.get(doc.get_field(root, "name")).s
    r.a_text = doc.get(doc.get_field(root, "a_text")).s
    r.b_question = doc.get(doc.get_field(root, "b_question")).s
    r.gold = doc.get(doc.get_field(root, "gold")).s
    r.twin_gold = doc.get(doc.get_field(root, "twin_gold")).s

    var rel_val = doc.get(doc.get_field(root, "relation"))
    if rel_val.kind == JKindNull:
        r.has_relation = False
    else:
        r.has_relation = True
        r.relation = rel_val.s

    var ans_val = doc.get(doc.get_field(root, "answers"))
    for gi in range(len(ans_val.arr)):
        var group_val = doc.get(ans_val.arr[gi])
        var group_list = List[String]()
        for ai in range(len(group_val.arr)):
            group_list.append(doc.get(group_val.arr[ai]).s)
        r.answers.append(group_list^)
    return r^


def flat_contains(row: ItemRow, target: String) -> Bool:
    for gi in range(len(row.answers)):
        for ai in range(len(row.answers[gi])):
            if row.answers[gi][ai] == target:
                return True
    return False


def check_no_cross_alias(row: ItemRow) raises:
    for gi in range(len(row.answers)):
        for gj in range(len(row.answers)):
            if gi == gj:
                continue
            for ai in range(len(row.answers[gi])):
                var a = row.answers[gi][ai]
                for bi in range(len(row.answers[gj])):
                    var b = row.answers[gj][bi]
                    if contains_whole_word(b, a):
                        raise Error(
                            "alias collision in " + row.id + ": '" + a
                            + "' (answer group " + String(gi)
                            + ") is a whole word inside '" + b
                            + "' (answer group " + String(gj) + ")"
                        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() raises:
    var lines = split_lines(read_text(ITEMS_PATH))
    var rows = List[ItemRow]()
    var id_index = Dict[String, Int]()
    for i in range(len(lines)):
        var ln = lines[i]
        if ln.byte_length() == 0:
            continue
        var row = parse_item(ln)
        id_index[row.id] = len(rows)
        rows.append(row^)

    if len(rows) != 1000:
        raise Error("expected 1000 rows (500 base + 500 twin), got " + String(len(rows)))

    var base = List[Int]()
    for i in range(len(rows)):
        if not rows[i].is_twin:
            base.append(i)
    if len(base) != 500:
        raise Error("expected 500 base (non-twin) items, got " + String(len(base)))

    # ---- gold in answers, per-item alias collision, no-digit names ----
    for bi in range(len(rows)):
        var row = rows[bi].copy()
        if not flat_contains(row, norm(row.gold)):
            raise Error(row.id + ": gold '" + row.gold + "' not found in its own answers list")
        check_no_cross_alias(row)
    for bi in range(len(base)):
        if has_digit(rows[base[bi]].name):
            raise Error(rows[base[bi]].id + ": name '" + rows[base[bi]].name + "' contains a digit")

    # ---- per (split, game) bucket: unique names, first/last balance, pool size ----
    var bucket_split = List[String]()
    var bucket_game = List[String]()
    bucket_split.append(String("dev")); bucket_game.append(String("countries"))
    bucket_split.append(String("test")); bucket_game.append(String("countries"))
    bucket_split.append(String("dev")); bucket_game.append(String("tips"))
    bucket_split.append(String("test")); bucket_game.append(String("tips"))
    bucket_split.append(String("test")); bucket_game.append(String("relations"))

    for bk in range(len(bucket_split)):
        var sp = bucket_split[bk]
        var gm = bucket_game[bk]
        var seen_names = Dict[String, Bool]()
        var first_counts = Dict[String, Int]()
        var last_counts = Dict[String, Int]()
        var n_items = 0
        for bi in range(len(base)):
            var row = rows[base[bi]].copy()
            if row.split != sp or row.game != gm:
                continue
            n_items += 1
            if row.name in seen_names:
                raise Error(
                    "duplicate name '" + row.name + "' within (" + sp + "," + gm + ")"
                )
            seen_names[row.name] = True
            var toks = split_space(row.name)
            if len(toks) != 2:
                raise Error(row.id + ": name '" + row.name + "' is not two tokens")
            var f = toks[0]
            var l = toks[1]
            first_counts[f] = (first_counts[f] if f in first_counts else 0) + 1
            last_counts[l] = (last_counts[l] if l in last_counts else 0) + 1
        if n_items == 0:
            raise Error("bucket (" + sp + "," + gm + ") is empty")
        for key in first_counts.keys():
            if first_counts[key] > 4:
                raise Error(
                    "first name '" + key + "' used " + String(first_counts[key])
                    + "x in (" + sp + "," + gm + "), cap is 4"
                )
        for key in last_counts.keys():
            if last_counts[key] > 4:
                raise Error(
                    "last name '" + key + "' used " + String(last_counts[key])
                    + "x in (" + sp + "," + gm + "), cap is 4"
                )
        if len(first_counts) < 40:
            raise Error(
                "(" + sp + "," + gm + ") first-name pool only " + String(len(first_counts))
                + " distinct, want >=40"
            )

    # ---- countries: <=4 gold per country, no landmark (a_text) shared dev/test ----
    var cty_gold_counts = Dict[String, Int]()
    var cty_dev_atext = Dict[String, Bool]()
    var cty_test_atext = Dict[String, Bool]()
    for bi in range(len(base)):
        var row = rows[base[bi]].copy()
        if row.game != "countries":
            continue
        cty_gold_counts[row.gold] = (
            cty_gold_counts[row.gold] if row.gold in cty_gold_counts else 0
        ) + 1
        if row.split == "dev":
            cty_dev_atext[row.a_text] = True
        else:
            cty_test_atext[row.a_text] = True
    for key in cty_gold_counts.keys():
        if cty_gold_counts[key] > 4:
            raise Error(
                "country '" + key + "' is gold " + String(cty_gold_counts[key])
                + "x, cap is 4"
            )
    for key in cty_dev_atext.keys():
        if key in cty_test_atext:
            raise Error("a_text (landmark) shared between dev and test: '" + key + "'")

    # ---- tips: 60+ company pool, gold balanced per company (within 2 of avg) ----
    var tip_company_pool = Dict[String, Bool]()
    var tip_gold_counts = Dict[String, Int]()
    var tip_slot_counts = Dict[Int, Int]()
    var n_tips = 0
    for bi in range(len(base)):
        var row = rows[base[bi]].copy()
        if row.game != "tips":
            continue
        n_tips += 1
        var slot = -1
        for gi in range(len(row.answers)):
            var al = row.answers[gi][0]
            tip_company_pool[al] = True
            if al == norm(row.gold):
                slot = gi
        tip_gold_counts[row.gold] = (
            tip_gold_counts[row.gold] if row.gold in tip_gold_counts else 0
        ) + 1
        if slot < 0:
            raise Error(row.id + ": gold slot not found among its own answer groups")
        tip_slot_counts[slot] = (tip_slot_counts[slot] if slot in tip_slot_counts else 0) + 1
    if len(tip_company_pool) < 60:
        raise Error("tips company pool only " + String(len(tip_company_pool)) + " distinct, want >=60")
    var avg = Float64(n_tips) / Float64(len(tip_gold_counts))
    for key in tip_gold_counts.keys():
        var d = Float64(tip_gold_counts[key]) - avg
        if d < 0:
            d = -d
        if d > 2.0:
            raise Error(
                "tips gold for '" + key + "' is " + String(tip_gold_counts[key])
                + ", more than 2 off the " + String(avg) + " average"
            )
    var slot_lo = -1
    var slot_hi = -1
    for k in range(3):
        var c = tip_slot_counts[k] if k in tip_slot_counts else 0
        if slot_lo < 0 or c < slot_lo:
            slot_lo = c
        if c > slot_hi:
            slot_hi = c
    if slot_hi - slot_lo > 5:
        raise Error(
            "tips gold slot balance spread " + String(slot_hi - slot_lo) + " exceeds 5"
        )

    # ---- twins: back-reference, and "changes only the fact" ----
    for i in range(len(rows)):
        var r = rows[i].copy()
        if not r.is_twin:
            continue
        if r.twin_id not in id_index:
            raise Error(r.id + ": twin_id '" + r.twin_id + "' does not exist")
        var q = rows[id_index[r.twin_id]].copy()
        if q.is_twin:
            raise Error(r.id + ": twin_id '" + r.twin_id + "' also marked is_twin")
        if q.twin_id != r.id:
            raise Error(r.id + ": twin '" + q.id + "' does not point back (has twin_id '" + q.twin_id + "')")
        if r.game != q.game or r.split != q.split or r.name != q.name:
            raise Error(r.id + ": twin pair differs in game/split/name")
        if r.has_relation != q.has_relation or r.relation != q.relation:
            raise Error(r.id + ": twin pair differs in relation")
        if r.b_question != q.b_question:
            raise Error(r.id + ": twin pair differs in b_question (should only change the fact)")
        if r.a_text == q.a_text:
            raise Error(r.id + ": twin pair has identical a_text (the fact did not change)")

    print("PASS test_items: 1000 rows, 500 base, all invariants held")
    var digest = run("sha256sum " + ITEMS_PATH)
    print("sha256:", String(digest).strip())
