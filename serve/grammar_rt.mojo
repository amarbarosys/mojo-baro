"""Host glue between the grammar engine (grammar/) and the decode loop
(serve/window.mojo, serve/engine.mojo): loads the vocab/trie once per
process (lazily, on the first request that carries a schema, so every
existing BARO_SERVE=0 one-shot run and every request that never sets
response_format pays nothing), compiles a response_format schema into a
Matcher per request, and tracks the `</think>` reasoning boundary so
reasoning tokens are never fed to the matcher
(grammar/test_reasoning_boundary.mojo's own contract).

briefs/2026-09-16-json-enforcement-lane.md items 1-2. Grammar requests run
with speculation and the megakernel off (serve/engine.mojo); the masked
kernel covers temperature > 0 and masked greedy at temperature <= 0.
"""
from std.memory import ArcPointer
from grammar.automaton import Automaton, Bitset
from grammar.json_value import parse_json_bytes
from grammar.json_schema import compile_root_schema
from grammar.regex import compile_regex_to_rule
from grammar.vocab import Vocab, load_vocab
from grammar.trie import TokenTrie, build_trie
from grammar.matcher import Matcher


struct GrammarRuntime(Copyable, Movable):
    var vocab: ArcPointer[Vocab]
    var trie: ArcPointer[TokenTrie]
    var vocab_size: Int

    def __init__(out self, pack_dir: String) raises:
        var v = load_vocab(pack_dir)
        self.vocab_size = v.vocab_size
        var t = build_trie(v)
        self.vocab = ArcPointer(v^)
        self.trie = ArcPointer(t^)


def _owned_bytes(s: String) -> List[UInt8]:
    var b = s.as_bytes()
    var out = List[UInt8]()
    for i in range(len(b)):
        out.append(b[i])
    return out^


def compile_schema_matcher(rt: GrammarRuntime, schema_json: String) raises -> Matcher:
    var doc = parse_json_bytes(_owned_bytes(schema_json))
    var a = Automaton()
    _ = compile_root_schema(a, doc)
    return Matcher(a^, rt.trie, rt.vocab)


def compile_tool_matcher(rt: GrammarRuntime, pattern: String) raises -> Matcher:
    # Tool-name constraint (briefs/2026-11-07-ailab-tools-and-cast.md item 1):
    # main.rs sends the regex for the text after `<tool_call>` up to the end
    # of the name (`\s*<function=(hold|turn)>`). release=True: the matcher
    # governs only that prefix; the token that completes it may carry more
    # bytes, and the arguments after it stay free.
    var a = Automaton()
    a.start_rule = compile_regex_to_rule(a, pattern, "tool_name")
    return Matcher(a^, rt.trie, rt.vocab, release=True)


def tool_open_observe(mut buf: List[UInt8], tok_bytes: List[UInt8]) -> Bool:
    # The same substring scan as reasoning_boundary_observe, for the
    # tool-call opening; the special <tool_call> token decodes to its own
    # text in the pack vocab, so one scan covers the special id and a
    # spelled-out opening.
    return _needle_observe(buf, tok_bytes, "<tool_call>")


def reasoning_boundary_observe(mut buf: List[UInt8], tok_bytes: List[UInt8]) -> Bool:
    # Appends tok_bytes to buf and reports whether "</think>" has now been
    # seen anywhere in it (a token's decoded bytes are not guaranteed to
    # align with the tag -- BPE can split or merge it with neighbors -- so
    # this is a substring scan, not a tail check). Once True the caller
    # stops calling this; buf is cleared here so a model that reasons for
    # thousands of tokens does not grow buf without bound before the
    # boundary, and so nothing lingers after it.
    return _needle_observe(buf, tok_bytes, "</think>")


def _needle_observe(mut buf: List[UInt8], tok_bytes: List[UInt8], needle: String) -> Bool:
    var nb = needle.as_bytes()
    for i in range(len(tok_bytes)):
        buf.append(tok_bytes[i])
    var n = len(nb)
    var found = False
    if len(buf) >= n:
        var limit = len(buf) - n
        for start in range(limit + 1):
            var ok = True
            for i in range(n):
                if buf[start + i] != nb[i]:
                    ok = False
                    break
            if ok:
                found = True
                break
    if found:
        buf = List[UInt8]()
        return True
    # Bound growth: only the trailing n-1 bytes can ever complete a future
    # match, so once buf is comfortably larger than that, drop the front.
    if len(buf) > 4 * n:
        var trimmed = List[UInt8]()
        var start2 = len(buf) - (n - 1)
        for j in range(start2, len(buf)):
            trimmed.append(buf[j])
        buf = trimmed^
    return False
