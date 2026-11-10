"""Tool-name constraint (briefs/2026-11-07-ailab-tools-and-cast.md item 1), CPU only:
the regex main.rs sends after `<tool_call>` admits only the request's tool names, the
token that completes the name may carry trailing bytes (release mode), and the opening
scan fires on the special <tool_call> id and on a spelled-out opening.

Build: ./.venv/bin/mojo build serve/test_tool_grammar.mojo -I serve -I . -o .work/test_tool_grammar
Run:   .work/test_tool_grammar [PACK_DIR]   (default .work/ailab/pack-kq)
"""
from std.sys import argv
from std.testing import assert_true, assert_equal
from grammar.automaton import Bitset
from grammar.vocab import Vocab
from grammar.matcher import Matcher
from grammar_rt import GrammarRuntime, compile_tool_matcher, tool_open_observe


def tok_of(v: Vocab, text: String) -> Int:
    # Exact single token whose bytes are text, or -1.
    var tb = text.as_bytes()
    for tid in range(v.vocab_size):
        ref b = v.token_bytes[tid]
        if len(b) != len(tb):
            continue
        var ok = True
        for i in range(len(b)):
            if b[i] != tb[i]:
                ok = False
                break
        if ok:
            return tid
    return -1


def accept_text(mut m: Matcher, v: Vocab, pieces: List[String]) raises:
    for p in pieces:
        var t = tok_of(v, p)
        assert_true(t >= 0, "no single token for " + p)
        var mask = Bitset(v.vocab_size)
        m.fill_mask(mask)
        assert_true(mask.get_bit(t), "mask refuses " + p)
        assert_true(m.accept(t), "accept refuses " + p)


def main() raises:
    var pack = String(".work/ailab/pack-kq")
    var args = argv()
    if len(args) > 1:
        pack = String(args[1])
    var rt = GrammarRuntime(pack)
    ref v = rt.vocab[]

    # XML wire (Qwythos family): `<tool_call>\n<function=NAME>`.
    var m = compile_tool_matcher(rt, "\\s*<function=(hold|turn)>")
    accept_text(m, v, ["\n", "<", "function", "="])
    var mask = Bitset(v.vocab_size)
    m.fill_mask(mask)
    var t_beg = tok_of(v, "beg")
    var t_hold = tok_of(v, "hold")
    var t_h = tok_of(v, "h")
    var t_b = tok_of(v, "b")
    assert_true(t_beg >= 0 and t_hold >= 0 and t_h >= 0 and t_b >= 0, "probe tokens missing")
    assert_true(not mask.get_bit(t_beg), "beg allowed after <function=")
    assert_true(not mask.get_bit(t_b), "b allowed after <function=")
    assert_true(mask.get_bit(t_hold) and mask.get_bit(t_h), "hold refused")
    var n_allowed = mask.count()
    assert_true(n_allowed < 64, "name mask too wide: " + String(n_allowed))
    accept_text(m, v, ["hold"])
    assert_true(not m.is_terminated(), "terminated before >")
    accept_text(m, v, [">"])
    assert_true(m.is_terminated(), "not terminated after hold>")

    # JSON wire: the name's closing quote may share a token with the comma (`",`).
    var j = compile_tool_matcher(rt, "\\s*\\{\\s*\"name\"\\s*:\\s*\"(hold|turn)\"")
    accept_text(j, v, ["\n", "{\"", "name", "\":", " \"", "turn"])
    var jm = Bitset(v.vocab_size)
    j.fill_mask(jm)
    var t_qc = tok_of(v, "\",")
    assert_true(t_qc >= 0, "no token for \",")
    assert_true(jm.get_bit(t_qc), "release: \", refused after the name")
    assert_true(not jm.get_bit(tok_of(v, "_left")), "a longer name allowed")
    assert_true(j.accept(t_qc) and j.is_terminated(), "\", did not end the name")

    # tool_choice required: the prompt already ends with `<function=`.
    var r = compile_tool_matcher(rt, "(hold|turn)>")
    var rm = Bitset(v.vocab_size)
    r.fill_mask(rm)
    assert_true(not rm.get_bit(t_beg) and rm.get_bit(t_hold), "required: name mask wrong")

    # Opening scan: the special id, then a spelled-out opening across pieces.
    var t_open = tok_of(v, "<tool_call>")
    assert_true(t_open >= 0 and v.is_special[t_open], "<tool_call> is not a special token with text bytes")
    var buf = List[UInt8]()
    assert_true(not tool_open_observe(buf, v.token_bytes[tok_of(v, "hello")]), "fired on hello")
    assert_true(tool_open_observe(buf, v.token_bytes[t_open]), "missed the special id")
    var buf2 = List[UInt8]()
    var fired = False
    for p in [String("<"), String("tool"), String("_call"), String(">")]:
        var t = tok_of(v, p)
        assert_true(t >= 0, "no token " + p)
        fired = tool_open_observe(buf2, v.token_bytes[t])
    assert_true(fired, "missed the spelled-out opening")
    print("test_tool_grammar: PASS (name mask", n_allowed, "tokens)")
