# grammar — guided-decoding grammar engine

CPU-only library: JSON-schema / regex -> byte-level pushdown automaton (PDA),
token-mask fill against a real BPE vocab, accept/snapshot/rollback for
speculative decoding. Built off the GPU per the M6 design (see
`~/Brain/mojo-baro/briefs/2026-09-08-lane-grammar.md`).

## Build / run

```
./.venv/bin/mojo run -I . grammar/test_corpus.mojo
./.venv/bin/mojo run -I . grammar/test_accept_known_good.mojo
./.venv/bin/mojo run -I . grammar/test_rollback.mojo
./.venv/bin/mojo run -I . grammar/test_reasoning_boundary.mojo
./.venv/bin/mojo run -I . grammar/test_timing.mojo
```

All internal `grammar/*.mojo` files use package-qualified imports
(`from grammar.X import ...`) and must be built with `-I .` from the repo
root — `-I grammar` with flat imports does not work once any file uses a
qualified import.

## API (frozen)

```mojo
from grammar.automaton import Automaton, Bitset
from grammar.json_schema import compile_root_schema
from grammar.vocab import load_vocab
from grammar.trie import build_trie
from grammar.matcher import Matcher

var doc = parse_json_file("schema.json")
var automaton = Automaton()
var rid = compile_root_schema(automaton, doc)

var m = Matcher(automaton, trie_ptr, vocab_ptr)   # ArcPointer[TokenTrie], ArcPointer[Vocab]
m.fill_mask(mut mask: Bitset)     # one bit per vocab id
m.accept(token_id: Int) -> Bool   # False = rejected, state unchanged
m.snapshot() -> Int
m.rollback(to: Int)
m.is_terminated() -> Bool
```

`Vocab`/`TokenTrie` are built once per process (`load_vocab` + `build_trie`)
and shared across every `Matcher` via `ArcPointer` — `Automaton` is cheap to
`.copy()` per request (tens to thousands of states, no vocab-sized data).

## Automaton representation

A byte-level PDA, not a general Earley/CFG parser: JSON-schema and the regex
subset are both unambiguous by construction (no left recursion, no
`$ref`/recursive schema support this round), so the schema tree compiles
directly into `Rule`s (each a byte-DFA over `AutoState`/`Transition`) linked
by `Call(rule_id) -> return_state` transitions. `MatcherState` holds
`(cur_rule, cur_state, call_stack: List[Frame], terminated)`; `call_stack` is
the explicit PDA stack.

**Accept-with-more-transitions.** A DFA state can be both `is_accept` and
have further outgoing byte transitions at once (`a*`, `\d{2,4}` mid-match).
`step_byte` therefore tries a direct byte match first and only pops the call
stack on failure, retrying after each pop ("try direct, pop-on-demand,
never eager-pop"). `is_terminated()` is a separate read-only walk
(`check_terminated`) that asks "can I legally stop here", not "did I just
pop" — an accept-but-extendable state must not be treated as forced-stop.

**Snapshot/rollback** is a full copy of `(cur_rule, cur_state, call_stack,
terminated)` — O(depth), not O(1), since `call_stack` can be mutated by
`step_byte`'s pop-on-demand as a normal part of forward progress (frames are
not just truncated, they can be permanently removed), so a length-only
restore is not sufficient. Depth is bounded by JSON nesting depth in
practice (small), so this is a documented deviation from the brief's O(1)
ask rather than a correctness gap.

**Mask fill** (`fill_mask`) is a DFS over the token trie, replaying
`step_byte` + `resolve_epsilon` per byte and undoing per child. The undo
avoids a `List[Frame]` copy per trie node: `step_byte_tracked` records
exactly the frames it pops into a caller-owned scratch buffer (reused across
the whole walk), and pushes are undone by length-truncation (pushes are pure
appends, no value to preserve) — so the common case (deep inside a regex
body, no call-stack mutation at all) touches the scratch buffer zero times.

## What this round does NOT support

- General CFG / non-JSON grammars.
- `$ref`/`$defs`/`definitions`, or any recursive schema.
- `patternProperties`; `additionalProperties` as a schema (only `true`/`false`).
- `allOf`, `not`, `if`/`then`/`else`, `anyOf`, `oneOf` — all raise a clear
  error naming the keyword.
- Numeric range keywords (`minimum`, `maximum`, `exclusiveMinimum`,
  `exclusiveMaximum`, `multipleOf`) — raise rather than silently ignore.
- `contentEncoding`, `contentMediaType`, `format` on strings.
- Regex backreferences, lookaround, non-greedy quantifiers (POSIX ERE has
  none of these).
- `pattern` strings whose match would itself need to emit `"` or `\` inside
  the JSON string body (pattern bytes are treated as literal JSON-string
  content, no escape-awareness) — common id/date/enum-like patterns never
  hit this.
- Object property order is fixed to schema-declaration order, not full
  JSON-Schema arbitrary-order semantics.
- `pattern`-given strings do not also enforce `minLength`/`maxLength`.
- String default `maxLength` when unspecified (no pattern): 24 characters.
  Array default `maxItems` when unspecified: 10.

## Vocab

Real pack `.work/engine-pack-q4`: `vocab_size = 248320` (extended
Qwythos/Qwen3 vocab, not ~151k), `eos_id=248046`, `pad_id=248044`, 33
special tokens (never matched by the grammar). GPT-2 byte-map decode reads
every token id to its raw byte string once at load; the token trie (~592k
nodes) is built once and shared.

## Timing (item 5e)

`grammar/test_timing.mojo` measures median `fill_mask` over 1000 tokens
after warmup on the largest corpus schema
(`grammar/corpus/27_nested_array_of_objects_optional.json`, 2667 automaton
states). Budget: 0.5ms. Numbers recorded in the status file and the report.

A first isolated measurement showed ~20ms (40x over budget) on
`07_plain_string`. The cause was not algorithmic: it was scheduling noise
from other lanes' concurrent CPU-bound jobs on the same 16-core machine
(load average ~8 at measurement time). Re-measured at steady state,
`fill_mask` costs ~28-30us — the trie DFS visits only a few hundred nodes
per call (455 visits producing 272 accepted tokens on a 577-state
automaton), far short of the ~2x-vocab-size naive-walk worst case the
design doc predicted. The adaptive context-independent mask cache from the
original PLAN was not needed this round; `fill_mask` also stopped taking a
full `List[Frame]` copy per trie node (see Automaton representation above)
as a real, if secondary, improvement.
