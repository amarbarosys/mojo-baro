from grammar.automaton import Automaton
from grammar.json_value import JSONDoc, JKindString, JKindNumber, JKindBool, JKindNull, JKindArray, JKindObject
from grammar.regex import ReDoc, ReNode, RKByteSet, RKConcat, RKAlt, compile_regex_to_rule, compile_ast_to_rule, expand_repeat


def str_bytes(s: String) -> List[UInt8]:
    var b = s.as_bytes()
    var out: List[UInt8] = []
    for i in range(len(b)):
        out.append(b[i])
    return out^


def _contains_str(items: List[String], v: String) -> Bool:
    for i in range(len(items)):
        if items[i] == v:
            return True
    return False


def add_literal(mut automaton: Automaton, rid: Int32, cur: Int32, lit: List[UInt8]) -> Int32:
    var c = cur
    for i in range(len(lit)):
        var nxt = automaton.rules[Int(rid)].new_state()
        automaton.rules[Int(rid)].add_byte_trans(c, lit[i], lit[i], nxt)
        c = nxt
    return c


def add_literal_edge(mut automaton: Automaton, rid: Int32, src: Int32, byte: UInt8, target: Int32):
    automaton.rules[Int(rid)].add_byte_trans(src, byte, byte, target)


def add_call(mut automaton: Automaton, rid: Int32, cur: Int32, call_rule: Int32) -> Int32:
    var ret = automaton.rules[Int(rid)].new_state()
    automaton.rules[Int(rid)].add_call_trans(cur, call_rule, ret)
    return ret


@fieldwise_init
struct SeqStep(Copyable, Movable):
    var required: Bool
    var prefix: List[UInt8]
    var call_rule: Int32


def build_sequence(mut automaton: Automaton, rid: Int32, entry_state: Int32, steps: List[SeqStep], closer: UInt8) -> Int32:
    var accept_state = automaton.rules[Int(rid)].new_state()
    automaton.rules[Int(rid)].set_accept(accept_state)
    var n = len(steps)
    # suffix_all_optional[i] = True iff steps[i:] contains no required step.
    # An optional step's early-exit-to-closer shortcut is only sound when
    # every step after it is also skippable -- otherwise the shortcut would
    # let the automaton jump straight past a still-mandatory later step
    # (caught via the corpus test: 25_const_and_enum_mixed's
    # required=["op","value"] with "unit" optional in between was producing
    # {"op":"add"}, silently dropping the required "value" field). When a
    # required step follows, the optional step in between is conservatively
    # always emitted rather than reconstructing full subset generation.
    var suffix_all_optional: List[Bool] = []
    for _ in range(n + 1):
        suffix_all_optional.append(True)
    for i in range(n - 1, -1, -1):
        suffix_all_optional[i] = (not steps[i].required) and suffix_all_optional[i + 1]

    var cur = entry_state
    var first = True
    for i in range(n):
        ref step = steps[i]
        if not step.required and suffix_all_optional[i + 1]:
            add_literal_edge(automaton, rid, cur, closer, accept_state)
        if not first:
            cur = add_literal(automaton, rid, cur, [UInt8(44)])
        if len(step.prefix) > 0:
            cur = add_literal(automaton, rid, cur, step.prefix)
        cur = add_call(automaton, rid, cur, step.call_rule)
        first = False
    add_literal_edge(automaton, rid, cur, closer, accept_state)
    return accept_state


def _json_encode_string(s: String) -> List[UInt8]:
    var out: List[UInt8] = [UInt8(34)]
    var b = s.as_bytes()
    for i in range(len(b)):
        var c = b[i]
        if c == 34:
            out.append(92); out.append(34)
        elif c == 92:
            out.append(92); out.append(92)
        elif c == 10:
            out.append(92); out.append(110)
        elif c == 13:
            out.append(92); out.append(114)
        elif c == 9:
            out.append(92); out.append(116)
        else:
            out.append(c)
    out.append(34)
    return out^


def _json_literal_bytes(doc: JSONDoc, idx: Int) raises -> List[UInt8]:
    var v = doc.get(idx)
    if v.kind == JKindString:
        return _json_encode_string(v.s)
    if v.kind == JKindNumber:
        return str_bytes(v.s)
    if v.kind == JKindBool:
        if v.b:
            return str_bytes("true")
        return str_bytes("false")
    if v.kind == JKindNull:
        return str_bytes("null")
    raise Error("json_schema: enum/const value must be string, number, bool, or null")


def _literal_ast(mut doc: ReDoc, lit: List[UInt8]) -> Int:
    var n = ReNode()
    n.kind = RKConcat
    for i in range(len(lit)):
        var bn = ReNode()
        bn.kind = RKByteSet
        bn.lo.append(lit[i])
        bn.hi.append(lit[i])
        n.children.append(doc.push(bn^))
    return doc.push(n^)


def _bs1(mut doc: ReDoc, lo: UInt8, hi: UInt8) -> Int:
    var n = ReNode()
    n.kind = RKByteSet
    n.lo.append(lo)
    n.hi.append(hi)
    return doc.push(n^)


def _cat(mut doc: ReDoc, parts: List[Int]) -> Int:
    var n = ReNode()
    n.kind = RKConcat
    for i in range(len(parts)):
        n.children.append(parts[i])
    return doc.push(n^)


def _build_utf8_char_ast(mut doc: ReDoc) -> Int:
    var cont = _bs1(doc, 128, 191)
    var ascii1 = _bs1(doc, 32, 33)
    var ascii2 = _bs1(doc, 35, 91)
    var ascii3 = _bs1(doc, 93, 127)

    var b2_lead = _bs1(doc, 0xC2, 0xDF)
    var utf8_2 = _cat(doc, [b2_lead, cont])

    var e0_lead = _bs1(doc, 0xE0, 0xE0)
    var e0_2nd = _bs1(doc, 0xA0, 0xBF)
    var utf8_3a = _cat(doc, [e0_lead, e0_2nd, cont])

    var e1ec_lead = _bs1(doc, 0xE1, 0xEC)
    var utf8_3b = _cat(doc, [e1ec_lead, cont, cont])

    var ed_lead = _bs1(doc, 0xED, 0xED)
    var ed_2nd = _bs1(doc, 0x80, 0x9F)
    var utf8_3c = _cat(doc, [ed_lead, ed_2nd, cont])

    var eeef_lead = _bs1(doc, 0xEE, 0xEF)
    var utf8_3d = _cat(doc, [eeef_lead, cont, cont])

    var f0_lead = _bs1(doc, 0xF0, 0xF0)
    var f0_2nd = _bs1(doc, 0x90, 0xBF)
    var utf8_4a = _cat(doc, [f0_lead, f0_2nd, cont, cont])

    var f13_lead = _bs1(doc, 0xF1, 0xF3)
    var utf8_4b = _cat(doc, [f13_lead, cont, cont, cont])

    var f4_lead = _bs1(doc, 0xF4, 0xF4)
    var f4_2nd = _bs1(doc, 0x80, 0x8F)
    var utf8_4c = _cat(doc, [f4_lead, f4_2nd, cont, cont])

    var alt = ReNode()
    alt.kind = RKAlt
    alt.children = [ascii1, ascii2, ascii3, utf8_2, utf8_3a, utf8_3b, utf8_3c, utf8_3d, utf8_4a, utf8_4b, utf8_4c]
    return doc.push(alt^)


def compile_json_string_body_rule(mut automaton: Automaton, min_len: Int, max_len: Int, name: String) raises -> Int32:
    var doc = ReDoc()
    var unescaped_idx = _build_utf8_char_ast(doc)

    var bs = ReNode()
    bs.kind = RKByteSet
    bs.lo.append(92); bs.hi.append(92)
    var bs_idx = doc.push(bs^)

    var simple = ReNode()
    simple.kind = RKByteSet
    simple.lo = [34, 92, 47, 98, 102, 110, 114, 116]
    simple.hi = [34, 92, 47, 98, 102, 110, 114, 116]
    var simple_idx = doc.push(simple^)

    var hexdigit = ReNode()
    hexdigit.kind = RKByteSet
    hexdigit.lo = [48, 65, 97]
    hexdigit.hi = [57, 70, 102]
    var hex_idx = doc.push(hexdigit^)

    var u_lit = ReNode()
    u_lit.kind = RKByteSet
    u_lit.lo.append(117); u_lit.hi.append(117)
    var u_idx = doc.push(u_lit^)

    var u_seq = ReNode()
    u_seq.kind = RKConcat
    u_seq.children.append(u_idx)
    for _ in range(4):
        u_seq.children.append(hex_idx)
    var u_seq_idx = doc.push(u_seq^)

    var esc_body = ReNode()
    esc_body.kind = RKAlt
    esc_body.children.append(simple_idx)
    esc_body.children.append(u_seq_idx)
    var esc_body_idx = doc.push(esc_body^)

    var escape = ReNode()
    escape.kind = RKConcat
    escape.children.append(bs_idx)
    escape.children.append(esc_body_idx)
    var escape_idx = doc.push(escape^)

    var char_node = ReNode()
    char_node.kind = RKAlt
    char_node.children.append(unescaped_idx)
    char_node.children.append(escape_idx)
    var char_idx = doc.push(char_node^)

    var body_idx = expand_repeat(doc, char_idx, min_len, max_len)
    return compile_ast_to_rule(automaton, doc, body_idx, name)


def _reject_unsupported(doc: JSONDoc, schema_idx: Int) raises:
    var forbidden: List[String] = ["$ref", "$defs", "definitions", "allOf", "not", "if", "then", "else", "patternProperties"]
    for i in range(len(forbidden)):
        if doc.has_field(schema_idx, forbidden[i]):
            raise Error("json_schema: unsupported keyword '" + forbidden[i] + "'")


def compile_boolean(mut automaton: Automaton, name: String) -> Int32:
    var rid = automaton.add_rule(name)
    var t_end = add_literal(automaton, rid, 0, str_bytes("true"))
    automaton.rules[Int(rid)].set_accept(t_end)
    var f_end = add_literal(automaton, rid, 0, str_bytes("false"))
    automaton.rules[Int(rid)].set_accept(f_end)
    return rid


def compile_const(mut automaton: Automaton, doc: JSONDoc, schema_idx: Int, name: String) raises -> Int32:
    var const_idx = doc.get_field(schema_idx, "const")
    var lit = _json_literal_bytes(doc, const_idx)
    var rid = automaton.add_rule(name)
    var cur = add_literal(automaton, rid, 0, lit)
    automaton.rules[Int(rid)].set_accept(cur)
    return rid


def compile_enum(mut automaton: Automaton, doc: JSONDoc, schema_idx: Int, name: String) raises -> Int32:
    var enum_idx = doc.get_field(schema_idx, "enum")
    var items = doc.get(enum_idx)
    var redoc = ReDoc()
    var alt = ReNode()
    alt.kind = RKAlt
    for i in range(len(items.arr)):
        var lit = _json_literal_bytes(doc, items.arr[i])
        alt.children.append(_literal_ast(redoc, lit))
    var root = redoc.push(alt^)
    return compile_ast_to_rule(automaton, redoc, root, name)


def compile_string(mut automaton: Automaton, doc: JSONDoc, schema_idx: Int, name: String) raises -> Int32:
    var rejected: List[String] = ["contentEncoding", "contentMediaType", "format"]
    for i in range(len(rejected)):
        if doc.has_field(schema_idx, rejected[i]):
            raise Error("json_schema: unsupported keyword '" + rejected[i] + "'")
    var pattern_idx = doc.get_field(schema_idx, "pattern")
    var rid = automaton.add_rule(name)
    var cur = add_literal(automaton, rid, 0, [UInt8(34)])
    var body_rid: Int32
    if pattern_idx >= 0:
        var pat = doc.get(pattern_idx).s
        body_rid = compile_regex_to_rule(automaton, pat, name + ".pattern")
    else:
        var mn = 0
        var mx = 24
        var mn_idx = doc.get_field(schema_idx, "minLength")
        if mn_idx >= 0:
            mn = Int(doc.get(mn_idx).n)
        var mx_idx = doc.get_field(schema_idx, "maxLength")
        if mx_idx >= 0:
            mx = Int(doc.get(mx_idx).n)
        body_rid = compile_json_string_body_rule(automaton, mn, mx, name + ".body")
    cur = add_call(automaton, rid, cur, body_rid)
    var accept = automaton.rules[Int(rid)].new_state()
    automaton.rules[Int(rid)].set_accept(accept)
    add_literal_edge(automaton, rid, cur, UInt8(34), accept)
    return rid


def _reject_numeric_constraints(doc: JSONDoc, schema_idx: Int) raises:
    var rejected: List[String] = ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"]
    for i in range(len(rejected)):
        if doc.has_field(schema_idx, rejected[i]):
            raise Error("json_schema: unsupported keyword '" + rejected[i] + "' (numeric ranges not enforced this round)")


def compile_integer(mut automaton: Automaton, doc: JSONDoc, schema_idx: Int, name: String) raises -> Int32:
    _reject_numeric_constraints(doc, schema_idx)
    return compile_regex_to_rule(automaton, "-?(0|[1-9][0-9]*)", name)


def compile_number(mut automaton: Automaton, doc: JSONDoc, schema_idx: Int, name: String) raises -> Int32:
    _reject_numeric_constraints(doc, schema_idx)
    return compile_regex_to_rule(automaton, "-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?", name)


def compile_object(mut automaton: Automaton, doc: JSONDoc, schema_idx: Int, name: String) raises -> Int32:
    if doc.has_field(schema_idx, "additionalProperties"):
        var ap_idx = doc.get_field(schema_idx, "additionalProperties")
        var ap = doc.get(ap_idx)
        if not (ap.is_bool() and ap.b == False):
            raise Error("json_schema: unsupported keyword 'additionalProperties' (only 'false' supported)")

    var props_idx = doc.get_field(schema_idx, "properties")
    var required_idx = doc.get_field(schema_idx, "required")
    var required_names: List[String] = []
    if required_idx >= 0:
        var req = doc.get(required_idx)
        for i in range(len(req.arr)):
            required_names.append(doc.get(req.arr[i]).s)

    var rid = automaton.add_rule(name)
    var cur = add_literal(automaton, rid, 0, [UInt8(123)])

    var steps: List[SeqStep] = []
    if props_idx >= 0:
        var props = doc.get(props_idx)
        for i in range(len(props.obj_keys)):
            var key = props.obj_keys[i]
            var val_idx = props.obj_vals[i]
            var is_req = _contains_str(required_names, key)
            var sub_rid = compile_schema(automaton, doc, val_idx, name + "." + key)
            var prefix = _json_encode_string(key)
            prefix.append(58)
            steps.append(SeqStep(is_req, prefix^, sub_rid))

    _ = build_sequence(automaton, rid, cur, steps, UInt8(125))
    return rid


def compile_array(mut automaton: Automaton, doc: JSONDoc, schema_idx: Int, name: String) raises -> Int32:
    var items_idx = doc.get_field(schema_idx, "items")
    if items_idx < 0:
        raise Error("json_schema: array without 'items' unsupported (tuple validation not supported)")
    var items_rid = compile_schema(automaton, doc, items_idx, name + "[]")

    var mn = 0
    var mx = 10
    var mn_idx = doc.get_field(schema_idx, "minItems")
    if mn_idx >= 0:
        mn = Int(doc.get(mn_idx).n)
    var mx_idx = doc.get_field(schema_idx, "maxItems")
    if mx_idx >= 0:
        mx = Int(doc.get(mx_idx).n)
    if mx < mn:
        raise Error("json_schema: maxItems < minItems")

    var rid = automaton.add_rule(name)
    var cur = add_literal(automaton, rid, 0, [UInt8(91)])
    var steps: List[SeqStep] = []
    for _ in range(mn):
        steps.append(SeqStep(True, [], items_rid))
    for _ in range(mx - mn):
        steps.append(SeqStep(False, [], items_rid))
    _ = build_sequence(automaton, rid, cur, steps, UInt8(93))
    return rid


def compile_schema(mut automaton: Automaton, doc: JSONDoc, schema_idx: Int, name: String) raises -> Int32:
    _reject_unsupported(doc, schema_idx)
    if doc.has_field(schema_idx, "enum"):
        return compile_enum(automaton, doc, schema_idx, name)
    if doc.has_field(schema_idx, "const"):
        return compile_const(automaton, doc, schema_idx, name)
    if doc.has_field(schema_idx, "anyOf"):
        raise Error("json_schema: unsupported keyword 'anyOf'")
    if doc.has_field(schema_idx, "oneOf"):
        raise Error("json_schema: unsupported keyword 'oneOf'")

    var type_idx = doc.get_field(schema_idx, "type")
    if type_idx < 0:
        raise Error("json_schema: missing 'type' (and no enum/const)")
    var type_val = doc.get(type_idx)
    if not type_val.is_string():
        raise Error("json_schema: 'type' must be a string (union types not supported)")
    var t = type_val.s
    if t == "object":
        return compile_object(automaton, doc, schema_idx, name)
    if t == "array":
        return compile_array(automaton, doc, schema_idx, name)
    if t == "string":
        return compile_string(automaton, doc, schema_idx, name)
    if t == "integer":
        return compile_integer(automaton, doc, schema_idx, name)
    if t == "number":
        return compile_number(automaton, doc, schema_idx, name)
    if t == "boolean":
        return compile_boolean(automaton, name)
    if t == "null":
        var rid = automaton.add_rule(name)
        var cur = add_literal(automaton, rid, 0, str_bytes("null"))
        automaton.rules[Int(rid)].set_accept(cur)
        return rid
    raise Error("json_schema: unsupported 'type' value '" + t + "'")


def compile_root_schema(mut automaton: Automaton, doc: JSONDoc) raises -> Int32:
    var rid = compile_schema(automaton, doc, doc.root, "root")
    automaton.start_rule = rid
    return rid
