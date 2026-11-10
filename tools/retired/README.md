# retired

Python tools superseded by Mojo implementations. Kept for history and as oracle
material; not on any path, not maintained.

| retired | replaced by | since |
|---|---|---|
| `baro-tokenize.py` (Python CLI over `tokenizers`) | `tools/baro-tokenize.mojo` → `.work/baro-tokenize` | 2026-09-08 |

Not carried over yet: `--chat` rendering of `tokenizer.chat_template` (jinja2)
in the CLI; the gate renders the template in Python as oracle-side input.

`gguf-tokenizer.py` (replaced by `tools/gguf-tokenizer-json.mojo`) and `test_tokenizer.py` (replaced by
`tools/test_tokenizer_mojo.py`) were deleted 2026-09-16; git history has them. `baro-tokenize.py` stays
while its `--chat` rendering has no Mojo equivalent.
