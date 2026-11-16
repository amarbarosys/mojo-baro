from std.memory import ArcPointer
from std.testing import assert_true
from grammar.automaton import Automaton
from grammar.json_value import parse_json_file
from grammar.json_schema import compile_root_schema, str_bytes
from grammar.vocab import Vocab, load_vocab
from grammar.trie import TokenTrie, build_trie, greedy_tokenize
from grammar.matcher import Matcher

comptime PACK_DIR = ".work/engine-pack-q4"


def check_known_good(schema_path: String, sample: String, vp: ArcPointer[Vocab], tp: ArcPointer[TokenTrie]) raises:
    var doc = parse_json_file(schema_path)
    var a = Automaton()
    var rid = compile_root_schema(a, doc)
    var m = Matcher(a^, tp, vp)
    var toks = greedy_tokenize(tp[], str_bytes(sample))
    for i in range(len(toks)):
        assert_true(m.accept(toks[i]), "token rejected at index " + String(i) + " in " + schema_path)
    assert_true(m.is_terminated(), "final state not terminated for " + schema_path)


def main() raises:
    var vocab = load_vocab(PACK_DIR)
    var trie = build_trie(vocab)
    var vp = ArcPointer(vocab^)
    var tp = ArcPointer(trie^)

    check_known_good("grammar/corpus/01_simple_required.json", String('{"name":"Alice","age":30}'), vp, tp)
    check_known_good("grammar/corpus/03_nested_object.json", String('{"user":{"name":"Bob","email":"b@x.com"},"active":true}'), vp, tp)
    check_known_good("grammar/corpus/17_array_of_enums.json", String('["north","south"]'), vp, tp)
    check_known_good("grammar/corpus/12_const_field.json", String('{"kind":"weather.get","city":"Paris"}'), vp, tp)
    print("PASS")
