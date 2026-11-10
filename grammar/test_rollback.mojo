from std.memory import ArcPointer
from std.testing import assert_true, assert_equal
from grammar.automaton import Automaton, Bitset
from grammar.json_value import parse_json_file
from grammar.json_schema import compile_root_schema, str_bytes
from grammar.vocab import Vocab, load_vocab
from grammar.trie import TokenTrie, build_trie
from grammar.matcher import Matcher

comptime PACK_DIR = ".work/engine-pack-q4"


def mask_signature(mut m: Matcher, mut mask: Bitset) -> List[Int]:
    m.fill_mask(mask)
    var out: List[Int] = []
    for i in range(mask.n):
        if mask.get_bit(i):
            out.append(i)
    return out^


def greedy_tokenize(trie: TokenTrie, text_bytes: List[UInt8]) raises -> List[Int]:
    var out: List[Int] = []
    var pos = 0
    var n = len(text_bytes)
    while pos < n:
        var node_idx = trie.root_child(text_bytes[pos])
        if node_idx < 0:
            raise Error("no vocab token starts with byte at pos " + String(pos))
        var best_end = -1
        var best_token = -1
        if trie.nodes[Int(node_idx)].token_id >= 0:
            best_end = pos + 1
            best_token = Int(trie.nodes[Int(node_idx)].token_id)
        var cur = Int(node_idx)
        var i = pos + 1
        while i < n:
            var nxt = trie.nodes[cur].find_child(text_bytes[i])
            if nxt < 0:
                break
            cur = Int(nxt)
            i += 1
            if trie.nodes[cur].token_id >= 0:
                best_end = i
                best_token = Int(trie.nodes[cur].token_id)
        if best_token < 0:
            raise Error("cannot tokenize at pos " + String(pos))
        out.append(best_token)
        pos = best_end
    return out^


def check_rollback(schema_path: String, sample: String, k: Int, j: Int, vsize: Int, vp: ArcPointer[Vocab], tp: ArcPointer[TokenTrie]) raises:
    var doc = parse_json_file(schema_path)
    var a = Automaton()
    var rid = compile_root_schema(a, doc)
    var m = Matcher(a^, tp, vp)
    var toks = greedy_tokenize(tp[], str_bytes(sample))
    assert_true(k + j <= len(toks), "not enough tokens in sample for k+j in " + schema_path)

    var mask = Bitset(vsize)
    for i in range(k):
        assert_true(m.accept(toks[i]), "prefix token rejected in " + schema_path)

    var snap = m.snapshot()

    var masks_first: List[List[Int]] = []
    for i in range(j):
        masks_first.append(mask_signature(m, mask))
        assert_true(m.accept(toks[k + i]), "first pass token rejected in " + schema_path)
    var term_first = m.is_terminated()
    var rule_first = m.ms.cur_rule
    var state_first = m.ms.cur_state

    m.rollback(snap)

    var masks_second: List[List[Int]] = []
    for i in range(j):
        masks_second.append(mask_signature(m, mask))
        assert_true(m.accept(toks[k + i]), "second pass token rejected in " + schema_path)
    var term_second = m.is_terminated()
    var rule_second = m.ms.cur_rule
    var state_second = m.ms.cur_state

    assert_equal(len(masks_first), len(masks_second))
    for i in range(j):
        assert_equal(len(masks_first[i]), len(masks_second[i]), "mask size differs at step " + String(i) + " in " + schema_path)
        for b in range(len(masks_first[i])):
            assert_equal(masks_first[i][b], masks_second[i][b], "mask bit differs at step " + String(i) + " in " + schema_path)
    assert_equal(term_first, term_second, "terminated differs after replay in " + schema_path)
    assert_equal(rule_first, rule_second, "cur_rule differs after replay in " + schema_path)
    assert_equal(state_first, state_second, "cur_state differs after replay in " + schema_path)


def main() raises:
    var vocab = load_vocab(PACK_DIR)
    var vsize = vocab.vocab_size
    var trie = build_trie(vocab)
    var vp = ArcPointer(vocab^)
    var tp = ArcPointer(trie^)

    check_rollback("grammar/corpus/01_simple_required.json", String('{"name":"Alice","age":30}'), 2, 3, vsize, vp, tp)
    check_rollback("grammar/corpus/03_nested_object.json", String('{"user":{"name":"Bob","email":"b@x.com"},"active":true}'), 3, 4, vsize, vp, tp)
    print("PASS")
