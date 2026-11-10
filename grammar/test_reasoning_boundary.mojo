from std.memory import ArcPointer
from std.testing import assert_true, assert_equal
from grammar.automaton import Automaton, Bitset
from grammar.json_value import parse_json_file
from grammar.json_schema import compile_root_schema, str_bytes
from grammar.vocab import Vocab, load_vocab
from grammar.trie import TokenTrie, build_trie
from grammar.matcher import Matcher

comptime PACK_DIR = ".work/engine-pack-q4"


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


def find_other_token(trie: TokenTrie, vocab: Vocab, avoid: Int) -> Int:
    for tid in range(vocab.vocab_size):
        if vocab.is_special[tid]:
            continue
        if tid != avoid and len(vocab.token_bytes[tid]) > 0:
            return tid
    return -1


def main() raises:
    var vocab = load_vocab(PACK_DIR)
    var vsize = vocab.vocab_size
    var trie = build_trie(vocab)
    var vp = ArcPointer(vocab^)
    var tp = ArcPointer(trie^)

    # Simulate: <think>...reasoning tokens, never fed to the grammar...</think>
    # then the grammar is attached fresh at the boundary. The reasoning tokens
    # themselves are irrelevant to the grammar under test -- the caller never
    # calls accept() for them; the matcher below starts life already
    # positioned exactly at the post-</think> boundary.
    var doc = parse_json_file("grammar/corpus/01_simple_required.json")
    var a = Automaton()
    var rid = compile_root_schema(a, doc)
    var m = Matcher(a^, tp, vp)

    var json_sample = String('{"name":"Alice","age":30}')
    var toks = greedy_tokenize(tp[], str_bytes(json_sample))
    assert_true(len(toks) >= 3, "sample too short to exercise draft retry")

    var mask = Bitset(vsize)

    # The first post-marker token sits in a draft window: a speculative
    # decode step may propose a token that the real (larger) model later
    # rejects, forcing a rollback and a retry with the correct token. The
    # grammar must observe the *correct* first token exactly once, with no
    # residual effect from the rejected speculative attempt.
    var real_first = toks[0]
    var wrong_first = find_other_token(tp[], vp[], real_first)
    assert_true(wrong_first >= 0, "could not find a distinct probe token")

    var snap = m.snapshot()
    m.fill_mask(mask)
    var mask_before = mask.count()

    # Speculative draft proposes the wrong token -- if the grammar happens to
    # accept it (not guaranteed rejected, since mask validity is independent
    # of "real" vs "draft"), roll back regardless: the real model rejected
    # this draft, so the grammar state must not advance past this point.
    _ = m.accept(wrong_first)
    m.rollback(snap)

    m.fill_mask(mask)
    var mask_after_rollback = mask.count()
    assert_equal(mask_before, mask_after_rollback, "mask changed after snapshot/rollback around the boundary")

    # Now commit the real first token -- the FSM sees it exactly once.
    assert_true(m.accept(real_first), "real first post-marker token rejected")

    for i in range(1, len(toks)):
        assert_true(m.accept(toks[i]), "post-marker token " + String(i) + " rejected")

    assert_true(m.is_terminated(), "final state not terminated after reasoning-boundary sequence")
    print("PASS")
