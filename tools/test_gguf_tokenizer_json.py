"""Oracle check for tools/gguf-tokenizer-json.mojo: generated tokenizer.json vs a
shipped HF tokenizer.json (qwen2, qwen35) and vs llama-tokenize (llama-bpe).
  .venv/bin/python3 tools/test_gguf_tokenizer_json.py QWEN25_HF_JSON QWYTHOS_HF_JSON LLAMA_GGUF
expects .work/tokjson/{qwen25,qwythos,llama1b}/ written by the tool first."""
import os
import json, subprocess, sys
from tokenizers import Tokenizer
cases = ["Hello world", "  leading spaces and\ttabs\n\nnewlines", "123456789 3.14159 1,000,000", "naïve café résumé 日本語のテキスト 中文", "<|im_start|>user\nhi<|im_end|>", "don't won't I'll we've", "emoji 🙂👍🏽 and symbols ©®™ ∑∫", "def f(x):\n    return x**2  # comment", "URL https://example.com/a?b=c&d=e", "   \n  \t "]
def ids_hf(path, s): return Tokenizer.from_file(path).encode(s, add_special_tokens=False).ids
bad = 0
for name, ref in [("qwen25", sys.argv[1]), ("qwythos", sys.argv[2])]:
    for s in cases:
        a = ids_hf(f".work/tokjson/{name}/tokenizer.json", s); b = ids_hf(ref, s)
        if a != b: bad += 1; print("DIFF", name, repr(s), a[:10], b[:10])
    print(name, "vs shipped tokenizer.json checked", len(cases))
g = sys.argv[3]
for s in cases:
    a = ids_hf(".work/tokjson/llama1b/tokenizer.json", s)
    out = subprocess.run([os.path.expanduser("~/llama.cpp/build/bin/llama-tokenize"), "-m", g, "-p", s, "--ids", "--no-bos", "--log-disable"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    b = json.loads(out)
    if a != b: bad += 1; print("DIFF llama1b", repr(s), a[:10], b[:10])
print("llama1b vs llama-tokenize checked", len(cases))
print("RESULT", "PASS" if bad == 0 else f"FAIL {bad}")
