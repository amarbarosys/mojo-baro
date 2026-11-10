from std.memory import ArcPointer
from std.python import Python, PythonObject
import std.random as random
from grammar.automaton import Automaton, Bitset
from grammar.json_value import parse_json_file
from grammar.json_schema import compile_root_schema
from grammar.vocab import Vocab, load_vocab
from grammar.trie import TokenTrie, build_trie
from grammar.matcher import Matcher

comptime PACK_DIR = ".work/engine-pack-q4"
comptime SAMPLES_PER_SCHEMA = 100
comptime MAX_TOKENS = 200


def sample_one(mut m: Matcher, vocab: Vocab, mut mask: Bitset) raises -> String:
    var out: List[UInt8] = []
    for _ in range(MAX_TOKENS):
        var can_stop = m.is_terminated()
        m.fill_mask(mask)
        var choices: List[Int] = []
        for i in range(mask.n):
            if mask.get_bit(i):
                choices.append(i)
        if len(choices) == 0:
            break
        if can_stop and random.random_ui64(0, 1) == 0:
            break
        var pick = choices[Int(random.random_ui64(0, UInt64(len(choices) - 1)))]
        _ = m.accept(pick)
        ref tb = vocab.token_bytes[pick]
        for i in range(len(tb)):
            out.append(tb[i])
    return String(from_utf8=Span(out))


def validate_json(subprocess: PythonObject, env: PythonObject, schema_path: String, sample_text: String) raises -> Bool:
    var args = Python.list(
        ".venv/bin/python3",
        "grammar/tools/validate_json.py",
        schema_path,
    )
    var r = subprocess.run(args, input=PythonObject(sample_text), capture_output=PythonObject(True), text=PythonObject(True), env=env)
    return Int(py=r.returncode) == 0


def main() raises:
    random.seed(1234)
    var subprocess = Python.import_module("subprocess")
    var os_mod = Python.import_module("os")
    var env = os_mod.environ.copy()
    for k in ["PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE"]:
        if Bool(py=env.__contains__(k)):
            _ = env.pop(k)

    var vocab = load_vocab(PACK_DIR)
    var vsize = vocab.vocab_size
    var trie = build_trie(vocab)
    var vp = ArcPointer(vocab^)
    var tp = ArcPointer(trie^)

    var f = open("grammar/corpus/manifest.txt", "r")
    var text = f.read()
    f.close()
    var lines = text.split("\n")

    var total_samples = 0
    var total_failures = 0
    var schema_count = 0

    for i in range(len(lines)):
        var name = String(lines[i])
        if name.byte_length() == 0:
            continue
        schema_count += 1
        var schema_path = "grammar/corpus/" + name
        var doc = parse_json_file(schema_path)
        var automaton = Automaton()
        var rid = compile_root_schema(automaton, doc)

        var mask = Bitset(vsize)
        var schema_failures = 0
        for _ in range(SAMPLES_PER_SCHEMA):
            var m = Matcher(automaton.copy(), tp, vp)
            var sample = sample_one(m, vp[], mask)
            total_samples += 1
            if not m.is_terminated():
                schema_failures += 1
                total_failures += 1
                continue
            if not validate_json(subprocess, env, schema_path, sample):
                schema_failures += 1
                total_failures += 1
                print("INVALID", name, sample)
        if schema_failures > 0:
            print(name, "failures:", schema_failures, "/", SAMPLES_PER_SCHEMA)

    print("schemas:", schema_count, "samples:", total_samples, "failures:", total_failures)
    if total_failures > 0:
        raise Error("corpus test: " + String(total_failures) + " sample failures")
    print("PASS")
