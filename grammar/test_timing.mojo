from std.memory import ArcPointer
import std.random as random
import std.time as time
from grammar.automaton import Automaton, Bitset
from grammar.json_value import parse_json_file
from grammar.json_schema import compile_root_schema
from grammar.vocab import Vocab, load_vocab
from grammar.trie import TokenTrie, build_trie
from grammar.matcher import Matcher

comptime PACK_DIR = ".work/engine-pack-q4"
comptime LARGEST_SCHEMA = "grammar/corpus/27_nested_array_of_objects_optional.json"
comptime WARMUP_TOKENS = 50
comptime TIMED_TOKENS = 1000
comptime BUDGET_NS = 500_000


def fresh_matcher(schema_path: String, tp: ArcPointer[TokenTrie], vp: ArcPointer[Vocab]) raises -> Matcher:
    var doc = parse_json_file(schema_path)
    var a = Automaton()
    var rid = compile_root_schema(a, doc)
    return Matcher(a^, tp, vp)


def step(mut m: Matcher, mut mask: Bitset) raises -> Bool:
    m.fill_mask(mask)
    var choices: List[Int] = []
    for i in range(mask.n):
        if mask.get_bit(i):
            choices.append(i)
    if len(choices) == 0:
        return False
    var pick = choices[Int(random.random_ui64(0, UInt64(len(choices) - 1)))]
    return m.accept(pick)


def main() raises:
    random.seed(7)
    var vocab = load_vocab(PACK_DIR)
    var vsize = vocab.vocab_size
    var trie = build_trie(vocab)
    var vp = ArcPointer(vocab^)
    var tp = ArcPointer(trie^)

    var total_states = 0
    var probe_doc = parse_json_file(LARGEST_SCHEMA)
    var probe_a = Automaton()
    var probe_rid = compile_root_schema(probe_a, probe_doc)
    for r in range(len(probe_a.rules)):
        total_states += len(probe_a.rules[r].states)
    print("schema:", LARGEST_SCHEMA, "total_states:", total_states)

    var m = fresh_matcher(LARGEST_SCHEMA, tp, vp)
    var mask = Bitset(vsize)

    for _ in range(WARMUP_TOKENS):
        if m.is_terminated() and random.random_ui64(0, 3) == 0:
            m = fresh_matcher(LARGEST_SCHEMA, tp, vp)
            continue
        if not step(m, mask):
            m = fresh_matcher(LARGEST_SCHEMA, tp, vp)

    var samples: List[Int] = []
    for _ in range(TIMED_TOKENS):
        if m.is_terminated() and random.random_ui64(0, 3) == 0:
            m = fresh_matcher(LARGEST_SCHEMA, tp, vp)

        var t0 = time.perf_counter_ns()
        m.fill_mask(mask)
        var t1 = time.perf_counter_ns()
        samples.append(t1 - t0)

        var choices: List[Int] = []
        for i in range(mask.n):
            if mask.get_bit(i):
                choices.append(i)
        if len(choices) == 0:
            m = fresh_matcher(LARGEST_SCHEMA, tp, vp)
            continue
        var pick = choices[Int(random.random_ui64(0, UInt64(len(choices) - 1)))]
        if not m.accept(pick):
            m = fresh_matcher(LARGEST_SCHEMA, tp, vp)

    var sorted_samples = samples.copy()
    for i in range(len(sorted_samples)):
        for j in range(i + 1, len(sorted_samples)):
            if sorted_samples[j] < sorted_samples[i]:
                var tmp = sorted_samples[i]
                sorted_samples[i] = sorted_samples[j]
                sorted_samples[j] = tmp

    var n = len(sorted_samples)
    var median = sorted_samples[n // 2]
    var p90 = sorted_samples[(n * 90) // 100]
    var p99 = sorted_samples[(n * 99) // 100]
    var worst = sorted_samples[n - 1]

    print("samples:", n)
    print("median_ns:", median)
    print("p90_ns:", p90)
    print("p99_ns:", p99)
    print("worst_ns:", worst)
    print("budget_ns:", BUDGET_NS)
    if median <= BUDGET_NS:
        print("PASS: median within budget")
    else:
        print("MISS: median over budget, report where time goes")
