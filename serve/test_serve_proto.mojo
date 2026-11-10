"""`parse_schema_field`: the request-line schema slice used by the
JSON-enforcement lane (briefs/2026-09-16-json-enforcement-lane.md item 1).
Pure string handling, no accelerator, no pack needed.

Build: ./.venv/bin/mojo build serve/test_serve_proto.mojo -I serve -o .work/test_serve_proto
"""
from std.testing import assert_equal
from grammar.json_value import parse_json_bytes
from serve_proto import parse_schema_field, parse_reasoning_field, parse_request, default_sample_params, SampleParams
from grammar_rt import reasoning_boundary_observe


def main() raises:
    # Absent: the common case, every existing request line before this lane.
    assert_equal(parse_schema_field(String('{"id":1,"prompt":[1,2],"n":8}')), String(""))

    # Present, nested, alongside other fields on both sides.
    var line = String('{"id":1,"prompt":[1,2],"n":8,"schema":{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"array","items":{"type":"integer"}}}},"temperature":0.7}')
    var raw = parse_schema_field(line)
    assert_equal(raw, String('{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"array","items":{"type":"integer"}}}}'))
    # And it must actually compile as a schema (json_schema is the consumer).
    var rb = raw.as_bytes()
    var owned = List[UInt8]()
    for i in range(len(rb)):
        owned.append(rb[i])
    var doc = parse_json_bytes(owned^)
    assert_equal(doc.get(doc.root).obj_keys[0], String("type"))

    # A string inside the schema containing an escaped brace/quote must not
    # desync the depth counter.
    var line2 = String('{"id":2,"prompt":[1],"n":1,"schema":{"type":"string","pattern":"\\\\{a\\\\}"},"n2":9}')
    var raw2 = parse_schema_field(line2)
    assert_equal(raw2, String('{"type":"string","pattern":"\\\\{a\\\\}"}'))

    # Malformed (unterminated): returns "", never crashes.
    assert_equal(parse_schema_field(String('{"id":1,"prompt":[1],"n":1,"schema":{"type":"object"')), String(""))

    # parse_request itself is untouched by the new field's presence.
    var id = 0
    var prompt = List[Int]()
    var n = 0
    var spec = False
    var has_spec = False
    var stop = List[List[Int]]()
    var ckpt = List[Int]()
    var sample = default_sample_params()
    var state_save = String("")
    var state_load = String("")
    var state_prefix = False
    var sm_path = String("")
    var sv_path = String("")
    var s_positions = 0
    var s_now_s = Int64(0)
    var err = parse_request(line, id, prompt, n, spec, has_spec, stop, ckpt, sample, state_save, state_load, state_prefix, sm_path, sv_path, s_positions, s_now_s)
    assert_equal(err, String(""))
    assert_equal(id, 1)
    assert_equal(n, 8)

    # reasoning field: default true (matches Qwythos's own template default),
    # explicit false, explicit true.
    assert_equal(parse_reasoning_field(String('{"id":1,"prompt":[1],"n":1}')), True)
    assert_equal(parse_reasoning_field(String('{"id":1,"prompt":[1],"n":1,"reasoning":false}')), False)
    assert_equal(parse_reasoning_field(String('{"id":1,"prompt":[1],"n":1,"reasoning":true}')), True)

    # Reasoning-boundary scan: tokens before "</think>" never trip it; the
    # token containing the close does, on the first call after it lands
    # (grammar/test_reasoning_boundary.mojo's own boundary convention).
    var buf = List[UInt8]()
    var b1 = List[UInt8]()
    for c in String("<think>\nreasoning goes here").as_bytes():
        b1.append(c)
    assert_equal(reasoning_boundary_observe(buf, b1), False)
    var b2 = List[UInt8]()
    for c in String("</think>\n").as_bytes():
        b2.append(c)
    assert_equal(reasoning_boundary_observe(buf, b2), True)
    var b3 = List[UInt8]()
    for c in String("{").as_bytes():
        b3.append(c)
    assert_equal(reasoning_boundary_observe(buf, b3), False)

    # A boundary split across two small token appends still fires.
    var buf2 = List[UInt8]()
    var s1 = List[UInt8]()
    for c in String("</thi").as_bytes():
        s1.append(c)
    var s2 = List[UInt8]()
    for c in String("nk>").as_bytes():
        s2.append(c)
    assert_equal(reasoning_boundary_observe(buf2, s1), False)
    assert_equal(reasoning_boundary_observe(buf2, s2), True)

    print("PASS")
