from std.collections import Dict

# Builds bench/e11b/items.jsonl from bench/e11b/data/*.tsv, port of
# tools/e11b_items_oracle.py build(). Both sides draw from the same
# splitmix64 PRNG (see SplitMix64 below) so a separate-process run of each
# produces byte-identical output; verified by check.sh via cmp.

comptime SEED: UInt64 = 110921
comptime OUT_PATH = String("bench/e11b/items.jsonl")
comptime COUNTRIES_TSV = String("bench/e11b/data/countries.tsv")
comptime LANDMARKS_TSV = String("bench/e11b/data/landmarks.tsv")


# ---------------------------------------------------------------------------
# byte / string helpers
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


def pad3(n: Int) -> String:
    if n < 10:
        return String("00") + String(n)
    if n < 100:
        return String("0") + String(n)
    return String(n)


def strip_last_char(s: String) -> String:
    var b = s.as_bytes()
    return String(unsafe_from_utf8=b[: len(b) - 1])


def str_lt(a: String, b: String) -> Bool:
    var ab = a.as_bytes()
    var bb = b.as_bytes()
    var n = len(ab) if len(ab) < len(bb) else len(bb)
    for i in range(n):
        if ab[i] != bb[i]:
            return ab[i] < bb[i]
    return len(ab) < len(bb)


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
        var is_alnum = (lc >= UInt8(ord("a")) and lc <= UInt8(ord("z"))) or (
            lc >= UInt8(ord("0")) and lc <= UInt8(ord("9"))
        )
        if is_alnum:
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


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def split_lines(s: String) -> List[String]:
    var out = List[String]()
    for line in s.split("\n"):
        out.append(String(line))
    return out^


def split_tab(s: String) -> List[String]:
    var out = List[String]()
    for part in s.split("\t"):
        out.append(String(part))
    return out^


def first_pipe_field(s: String) -> String:
    for p in s.split("|"):
        return String(p)
    return s


# ---------------------------------------------------------------------------
# splitmix64: integer-only PRNG shared with tools/e11b_items_oracle.py
# ---------------------------------------------------------------------------

struct SplitMix64:
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next_u64(mut self) -> UInt64:
        self.state = self.state + UInt64(0x9E3779B97F4A7C15)
        var z = self.state
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        return z ^ (z >> 31)

    def below(mut self, n: Int) -> Int:
        """Uniform in [0, n), rejection-sampled: no modulo bias.

        threshold = 2**64 mod n, computed via unsigned wraparound as
        (0 - n) mod n so the arithmetic never needs an unrepresentable
        2**64 literal.
        """
        var nu = UInt64(n)
        var threshold = (UInt64(0) - nu) % nu
        while True:
            var x = self.next_u64()
            if x >= threshold:
                return Int(x % nu)

    def shuffle_str(mut self, mut lst: List[String]):
        var i = len(lst) - 1
        while i > 0:
            var j = self.below(i + 1)
            if j != i:
                var tmp = lst[i]
                lst[i] = lst[j]
                lst[j] = tmp
            i -= 1

    def shuffle_pair(mut self, mut a: List[String], mut b: List[String]):
        var i = len(a) - 1
        while i > 0:
            var j = self.below(i + 1)
            if j != i:
                var ta = a[i]
                a[i] = a[j]
                a[j] = ta
                var tb = b[i]
                b[i] = b[j]
                b[j] = tb
            i -= 1


# ---------------------------------------------------------------------------
# literal tables (identical lists in tools/e11b_items_oracle.py)
# ---------------------------------------------------------------------------

def first_names() -> List[String]:
    var FIRST = List[String]()
    FIRST.append(String("Ari"))
    FIRST.append(String("Bela"))
    FIRST.append(String("Cora"))
    FIRST.append(String("Dara"))
    FIRST.append(String("Elio"))
    FIRST.append(String("Faye"))
    FIRST.append(String("Galen"))
    FIRST.append(String("Hana"))
    FIRST.append(String("Iris"))
    FIRST.append(String("Juno"))
    FIRST.append(String("Kian"))
    FIRST.append(String("Lara"))
    FIRST.append(String("Milo"))
    FIRST.append(String("Nia"))
    FIRST.append(String("Orin"))
    FIRST.append(String("Pia"))
    FIRST.append(String("Quin"))
    FIRST.append(String("Rhea"))
    FIRST.append(String("Soren"))
    FIRST.append(String("Tala"))
    FIRST.append(String("Uma"))
    FIRST.append(String("Vera"))
    FIRST.append(String("Wren"))
    FIRST.append(String("Xara"))
    FIRST.append(String("Yuri"))
    FIRST.append(String("Zane"))
    FIRST.append(String("Ada"))
    FIRST.append(String("Bram"))
    FIRST.append(String("Cleo"))
    FIRST.append(String("Dion"))
    FIRST.append(String("Esme"))
    FIRST.append(String("Finn"))
    FIRST.append(String("Gia"))
    FIRST.append(String("Hugo"))
    FIRST.append(String("Isla"))
    FIRST.append(String("Jace"))
    FIRST.append(String("Kira"))
    FIRST.append(String("Levi"))
    FIRST.append(String("Maya"))
    FIRST.append(String("Niko"))
    FIRST.append(String("Opal"))
    FIRST.append(String("Pax"))
    FIRST.append(String("Romy"))
    FIRST.append(String("Sage"))
    FIRST.append(String("Tess"))
    FIRST.append(String("Ugo"))
    FIRST.append(String("Vivi"))
    FIRST.append(String("Wade"))
    FIRST.append(String("Xena"))
    FIRST.append(String("Yara"))
    FIRST.append(String("Zuri"))
    FIRST.append(String("Anya"))
    FIRST.append(String("Bryn"))
    FIRST.append(String("Calla"))
    FIRST.append(String("Demi"))
    FIRST.append(String("Enzo"))
    FIRST.append(String("Freya"))
    FIRST.append(String("Gio"))
    FIRST.append(String("Holly"))
    FIRST.append(String("Ivan"))
    FIRST.append(String("Jade"))
    return FIRST^


def last_names() -> List[String]:
    var LAST = List[String]()
    LAST.append(String("Alden"))
    LAST.append(String("Briar"))
    LAST.append(String("Cove"))
    LAST.append(String("Dover"))
    LAST.append(String("Ember"))
    LAST.append(String("Frost"))
    LAST.append(String("Grove"))
    LAST.append(String("Hart"))
    LAST.append(String("Ives"))
    LAST.append(String("Jett"))
    LAST.append(String("Kline"))
    LAST.append(String("Lumen"))
    LAST.append(String("Marlow"))
    LAST.append(String("Nash"))
    LAST.append(String("Orwell"))
    LAST.append(String("Pike"))
    LAST.append(String("Quill"))
    LAST.append(String("Reed"))
    LAST.append(String("Stone"))
    LAST.append(String("Thorne"))
    LAST.append(String("Vale"))
    LAST.append(String("Wells"))
    LAST.append(String("York"))
    LAST.append(String("Zephyr"))
    LAST.append(String("Arden"))
    LAST.append(String("Banks"))
    LAST.append(String("Cairn"))
    LAST.append(String("Dale"))
    LAST.append(String("Ellis"))
    LAST.append(String("Field"))
    LAST.append(String("Gray"))
    LAST.append(String("Haven"))
    LAST.append(String("Irwin"))
    LAST.append(String("James"))
    LAST.append(String("Kerr"))
    LAST.append(String("Lane"))
    LAST.append(String("Morse"))
    LAST.append(String("North"))
    LAST.append(String("Oakes"))
    LAST.append(String("Page"))
    LAST.append(String("Quade"))
    LAST.append(String("Rowan"))
    LAST.append(String("Shaw"))
    LAST.append(String("Tate"))
    LAST.append(String("Ulmer"))
    LAST.append(String("Voss"))
    LAST.append(String("Wright"))
    LAST.append(String("Yates"))
    LAST.append(String("Zeller"))
    LAST.append(String("Ashby"))
    LAST.append(String("Brooks"))
    LAST.append(String("Cruz"))
    LAST.append(String("Dune"))
    LAST.append(String("Everest"))
    LAST.append(String("Flynn"))
    LAST.append(String("Grant"))
    LAST.append(String("Hale"))
    LAST.append(String("Irons"))
    LAST.append(String("Keene"))
    LAST.append(String("Lark"))
    LAST.append(String("Moss"))
    return LAST^


def companies() -> List[String]:
    var COMPANIES = List[String]()
    COMPANIES.append(String("Amber Arc"))
    COMPANIES.append(String("Birch Byte"))
    COMPANIES.append(String("Cobalt Crest"))
    COMPANIES.append(String("Dawn Delta"))
    COMPANIES.append(String("Elm Engine"))
    COMPANIES.append(String("Fable Forge"))
    COMPANIES.append(String("Granite Grid"))
    COMPANIES.append(String("Harbor Hex"))
    COMPANIES.append(String("Ivory Index"))
    COMPANIES.append(String("Jasper Junction"))
    COMPANIES.append(String("Kite Kernel"))
    COMPANIES.append(String("Lumen Loop"))
    COMPANIES.append(String("Maple Mesh"))
    COMPANIES.append(String("Nimbus Node"))
    COMPANIES.append(String("Olive Orbit"))
    COMPANIES.append(String("Pebble Pulse"))
    COMPANIES.append(String("Quartz Query"))
    COMPANIES.append(String("River Relay"))
    COMPANIES.append(String("Solar Span"))
    COMPANIES.append(String("Timber Thread"))
    COMPANIES.append(String("Umber Unit"))
    COMPANIES.append(String("Violet Vector"))
    COMPANIES.append(String("Willow Wire"))
    COMPANIES.append(String("Xenon Xchange"))
    COMPANIES.append(String("Yarrow Yard"))
    COMPANIES.append(String("Zenith Zone"))
    COMPANIES.append(String("Aster Axis"))
    COMPANIES.append(String("Bracken Beam"))
    COMPANIES.append(String("Cinder Circuit"))
    COMPANIES.append(String("Drift Domain"))
    COMPANIES.append(String("Echo Edge"))
    COMPANIES.append(String("Flint Field"))
    COMPANIES.append(String("Garnet Gate"))
    COMPANIES.append(String("Horizon Hub"))
    COMPANIES.append(String("Indigo Inlet"))
    COMPANIES.append(String("Juniper Jet"))
    COMPANIES.append(String("Kestrel Key"))
    COMPANIES.append(String("Laurel Link"))
    COMPANIES.append(String("Meadow Matrix"))
    COMPANIES.append(String("Noble Nexus"))
    COMPANIES.append(String("Opal Outlet"))
    COMPANIES.append(String("Prism Port"))
    COMPANIES.append(String("Quarry Queue"))
    COMPANIES.append(String("Raven Route"))
    COMPANIES.append(String("Sable Signal"))
    COMPANIES.append(String("Topaz Tensor"))
    COMPANIES.append(String("Uplink Union"))
    COMPANIES.append(String("Vertex Vault"))
    COMPANIES.append(String("Warden Wave"))
    COMPANIES.append(String("Xylo Xenial"))
    COMPANIES.append(String("Yonder Yield"))
    COMPANIES.append(String("Zephyr Zenith"))
    COMPANIES.append(String("Acorn Array"))
    COMPANIES.append(String("Beacon Bloom"))
    COMPANIES.append(String("Canyon Cache"))
    COMPANIES.append(String("Dune Data"))
    COMPANIES.append(String("Estuary Element"))
    COMPANIES.append(String("Fjord Flux"))
    COMPANIES.append(String("Grove Graph"))
    COMPANIES.append(String("Hearth Hash"))
    COMPANIES.append(String("Island Index"))
    COMPANIES.append(String("Joule Junction"))
    return COMPANIES^


# ---------------------------------------------------------------------------
# data tables: countries.tsv / landmarks.tsv
# ---------------------------------------------------------------------------

def load_countries(
    path: String,
    mut names: List[String],
    mut capital: List[String],
    mut currency: List[String],
    mut continent: List[String],
) raises:
    var lines = split_lines(read_text(path))
    for li in range(len(lines)):
        if li == 0:
            continue
        var ln = lines[li]
        if len(ln.as_bytes()) == 0:
            continue
        var f = split_tab(ln)
        names.append(f[0])
        capital.append(first_pipe_field(f[2]))
        currency.append(first_pipe_field(f[3]))
        continent.append(f[4])


def load_landmarks(
    path: String, mut lm_name: List[String], mut lm_country: List[String]
) raises:
    var lines = split_lines(read_text(path))
    for li in range(len(lines)):
        if li == 0:
            continue
        var ln = lines[li]
        if len(ln.as_bytes()) == 0:
            continue
        var f = split_tab(ln)
        lm_name.append(f[0])
        lm_country.append(f[1])


def filter_landmarks(
    all_name: List[String],
    all_country: List[String],
    mut lm_name: List[String],
    mut lm_country: List[String],
) raises:
    var counts = Dict[String, Int]()
    for i in range(len(all_name)):
        var c = all_country[i]
        var cnt = counts[c] if c in counts else 0
        if cnt >= 4:
            continue
        lm_name.append(all_name[i])
        lm_country.append(c)
        counts[c] = cnt + 1
        if len(lm_name) == 200:
            break
    if len(lm_name) != 200:
        raise Error("landmark filter did not reach 200 (<=4/country cap)")


# ---------------------------------------------------------------------------
# name draws
# ---------------------------------------------------------------------------

def draw_name(
    mut rng: SplitMix64,
    mut first_counts: Dict[String, Int],
    mut last_counts: Dict[String, Int],
    mut used: Dict[String, Bool],
    first_pool: List[String],
    last_pool: List[String],
) raises -> String:
    var fs = List[String]()
    for i in range(len(first_pool)):
        var f = first_pool[i]
        var c = first_counts[f] if f in first_counts else 0
        if c < 4:
            fs.append(f)
    var ls = List[String]()
    for i in range(len(last_pool)):
        var l = last_pool[i]
        var c = last_counts[l] if l in last_counts else 0
        if c < 4:
            ls.append(l)
    rng.shuffle_str(fs)
    rng.shuffle_str(ls)
    for fi in range(len(fs)):
        var f = fs[fi]
        for li in range(len(ls)):
            var l = ls[li]
            var n = f + " " + l
            if n in used:
                continue
            first_counts[f] = (first_counts[f] if f in first_counts else 0) + 1
            last_counts[l] = (last_counts[l] if l in last_counts else 0) + 1
            used[n] = True
            return n
    raise Error("name pool exhausted")


# ---------------------------------------------------------------------------
# item text
# ---------------------------------------------------------------------------

def ph_country(pidx: Int, name: String, landmark: String) -> String:
    if pidx == 0:
        return name + " is at the " + landmark + "."
    elif pidx == 1:
        return name + " can be found at the " + landmark + "."
    else:
        return "The traveler " + name + " is visiting the " + landmark + "."


def ph_question(pidx: Int, name: String) -> String:
    if pidx == 0:
        return "Which country is " + name + " located in?"
    elif pidx == 1:
        return "What country is " + name + " in?"
    else:
        return "Name the country where " + name + " is located."


def item_json(
    id_: String,
    game: String,
    split: String,
    is_twin: Bool,
    twin_id: String,
    name: String,
    a_text: String,
    b_question: String,
    gold: String,
    twin_gold: String,
    has_relation: Bool,
    relation: String,
    answers_json: String,
) -> String:
    var out = String('{"id":"') + json_escape(id_)
    out += '","game":"' + json_escape(game)
    out += '","split":"' + json_escape(split)
    out += '","is_twin":' + (String("true") if is_twin else String("false"))
    out += ',"twin_id":"' + json_escape(twin_id)
    out += '","name":"' + json_escape(name)
    out += '","a_text":"' + json_escape(a_text)
    out += '","b_question":"' + json_escape(b_question)
    out += '","gold":"' + json_escape(gold)
    out += '","twin_gold":"' + json_escape(twin_gold)
    out += '","relation":' + (
        ('"' + json_escape(relation) + '"') if has_relation else String("null")
    )
    out += ',"answers":' + answers_json + "}"
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() raises:
    var FIRST = first_names()
    var LAST = last_names()
    var COMPANIES = companies()

    var country_names = List[String]()
    var country_capital = List[String]()
    var country_currency = List[String]()
    var country_continent = List[String]()
    load_countries(
        COUNTRIES_TSV, country_names, country_capital, country_currency, country_continent
    )

    var all_lm_name = List[String]()
    var all_lm_country = List[String]()
    load_landmarks(LANDMARKS_TSV, all_lm_name, all_lm_country)

    var lm_name = List[String]()
    var lm_country = List[String]()
    filter_landmarks(all_lm_name, all_lm_country, lm_name, lm_country)

    var rng = SplitMix64(SEED)
    rng.shuffle_pair(lm_name, lm_country)

    var country_idx = Dict[String, Int]()
    for i in range(len(country_names)):
        country_idx[country_names[i]] = i

    var sorted_countries = List[String]()
    for i in range(len(country_names)):
        sorted_countries.append(country_names[i])
    for i in range(1, len(sorted_countries)):
        var key = sorted_countries[i]
        var j = i - 1
        while j >= 0 and str_lt(key, sorted_countries[j]):
            sorted_countries[j + 1] = sorted_countries[j]
            j -= 1
        sorted_countries[j + 1] = key

    var countries_answers_json = String("[")
    for i in range(len(country_names)):
        if i > 0:
            countries_answers_json += ","
        countries_answers_json += '["' + json_escape(norm(country_names[i])) + '"]'
    countries_answers_json += "]"

    var out = String("")

    # ---- countries game ----
    var cty_dev_first = Dict[String, Int]()
    var cty_dev_last = Dict[String, Int]()
    var cty_dev_used = Dict[String, Bool]()
    var cty_test_first = Dict[String, Int]()
    var cty_test_last = Dict[String, Int]()
    var cty_test_used = Dict[String, Bool]()

    for i in range(200):
        var land = lm_name[i]
        var c = lm_country[i]
        var sp = String("dev") if i < 100 else String("test")
        var n: String
        if i < 100:
            n = draw_name(rng, cty_dev_first, cty_dev_last, cty_dev_used, FIRST, LAST)
        else:
            n = draw_name(rng, cty_test_first, cty_test_last, cty_test_used, FIRST, LAST)
        var p = i % 3
        var base_id = String("cty-") + sp + String("-") + pad3(i)
        var at = ph_country(p, n, land)
        var q = ph_question(p, n)

        var twin_c = String("")
        for k in range(200):
            if lm_country[k] != c:
                twin_c = lm_country[k]
                break
        var twin_l = String("")
        for k in range(200):
            if lm_country[k] == twin_c:
                twin_l = lm_name[k]
                break
        var t_at = ph_country(p, n, twin_l)

        out += (
            item_json(
                base_id, "countries", sp, False, base_id + "t", n, at, q,
                c, twin_c, False, "", countries_answers_json,
            ) + "\n"
        )
        out += (
            item_json(
                base_id + "t", "countries", sp, True, base_id, n, t_at, q,
                twin_c, c, False, "", countries_answers_json,
            ) + "\n"
        )

    # ---- tips game ----
    var SITU0 = String("earnings dipped")
    var SITU1 = String("earnings reached a record high")
    var SITU2 = String("earnings reached a record high while facing a lawsuit")

    var tip_dev_first = Dict[String, Int]()
    var tip_dev_last = Dict[String, Int]()
    var tip_dev_used = Dict[String, Bool]()
    var tip_test_first = Dict[String, Int]()
    var tip_test_last = Dict[String, Int]()
    var tip_test_used = Dict[String, Bool]()

    for i in range(200):
        var sp = String("dev") if i < 100 else String("test")
        var n: String
        if i < 100:
            n = draw_name(rng, tip_dev_first, tip_dev_last, tip_dev_used, FIRST, LAST)
        else:
            n = draw_name(rng, tip_test_first, tip_test_last, tip_test_used, FIRST, LAST)
        var start = (i * 3) % len(COMPANIES)
        var names = List[String]()
        names.append(COMPANIES[(start + 0) % len(COMPANIES)])
        names.append(COMPANIES[(start + 1) % len(COMPANIES)])
        names.append(COMPANIES[(start + 2) % len(COMPANIES)])
        var clean_i = i % 3

        var sts = List[String]()
        sts.append(SITU0)
        sts.append(SITU1)
        sts.append(SITU0)
        sts[clean_i] = SITU1
        sts[(clean_i + 1) % 3] = SITU2
        var clean = names[clean_i]

        var base_id = String("tip-") + sp + String("-") + pad3(i)
        var at = (
            names[0] + ": " + sts[0] + ". " + names[1] + ": " + sts[1]
            + ". " + names[2] + ": " + sts[2] + "."
        )
        var q = (
            String("You must invest in only one company out of ") + names[0]
            + ", " + names[1] + ", " + names[2] + ". Which do you invest in?"
        )
        var answers_json = (
            '[["' + json_escape(norm(names[0])) + '"],["' + json_escape(norm(names[1]))
            + '"],["' + json_escape(norm(names[2])) + '"]]'
        )

        var twin_sts = List[String]()
        twin_sts.append(sts[0])
        twin_sts.append(sts[1])
        twin_sts.append(sts[2])
        var j = -1
        for k in range(3):
            if twin_sts[k] == SITU1:
                j = k
                break
        twin_sts[j] = SITU2
        var kk = -1
        for k in range(3):
            if twin_sts[k] == SITU2 and k != j:
                kk = k
                break
        twin_sts[kk] = SITU1
        var twin_clean = names[kk]

        var t_at = (
            names[0] + ": " + twin_sts[0] + ". " + names[1] + ": " + twin_sts[1]
            + ". " + names[2] + ": " + twin_sts[2] + "."
        )

        out += (
            item_json(
                base_id, "tips", sp, False, base_id + "t", n, at, q,
                clean, twin_clean, False, "", answers_json,
            ) + "\n"
        )
        out += (
            item_json(
                base_id + "t", "tips", sp, True, base_id, n, t_at, q,
                twin_clean, clean, False, "", answers_json,
            ) + "\n"
        )

    # ---- relations game ----
    var REL_NAMES = List[String]()
    REL_NAMES.append(String("capital"))
    REL_NAMES.append(String("currency"))
    REL_NAMES.append(String("continent"))

    var rel_first = Dict[String, Int]()
    var rel_last = Dict[String, Int]()
    var rel_used = Dict[String, Bool]()

    for i in range(100):
        var c = sorted_countries[i % len(sorted_countries)]
        var land = String("")
        for k in range(200):
            if lm_country[k] == c:
                land = lm_name[k]
                break
        var rel = REL_NAMES[i % 3]
        var n = draw_name(rng, rel_first, rel_last, rel_used, FIRST, LAST)
        var p = i % 3
        var base_id = String("rel-test-") + pad3(i)
        var at = ph_country(p, n, land)
        var q = strip_last_char(ph_question(p, n)) + " " + rel + "?"

        var c_idx = country_idx[c]
        var gold: String
        if rel == "capital":
            gold = country_capital[c_idx]
        elif rel == "currency":
            gold = country_currency[c_idx]
        else:
            gold = country_continent[c_idx]
        var answers_json = '[["' + json_escape(norm(gold)) + '"]]'

        var c_sorted_idx = -1
        for k in range(len(sorted_countries)):
            if sorted_countries[k] == c:
                c_sorted_idx = k
                break
        var twin_c = sorted_countries[(c_sorted_idx + 1) % len(sorted_countries)]
        var twin_land = String("")
        for k in range(200):
            if lm_country[k] == twin_c:
                twin_land = lm_name[k]
                break
        var t_at = ph_country(p, n, twin_land)

        var twin_c_idx = country_idx[twin_c]
        var t_gold: String
        if rel == "capital":
            t_gold = country_capital[twin_c_idx]
        elif rel == "currency":
            t_gold = country_currency[twin_c_idx]
        else:
            t_gold = country_continent[twin_c_idx]
        var t_answers_json = '[["' + json_escape(norm(t_gold)) + '"]]'

        out += (
            item_json(
                base_id, "relations", "test", False, base_id + "t", n, at, q,
                gold, t_gold, True, rel, answers_json,
            ) + "\n"
        )
        out += (
            item_json(
                base_id + "t", "relations", "test", True, base_id, n, t_at, q,
                t_gold, gold, True, rel, t_answers_json,
            ) + "\n"
        )

    with open(OUT_PATH, "w") as f:
        f.write(out)

    print("PASS Mojo builder wrote", OUT_PATH, "(", out.byte_length(), "bytes )")
