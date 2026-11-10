"""Token counts for RULER sizing via the Mojo tokenizer (.work/baro-tokenize).

Not llama.cpp's tokenizer, not nemo/hf/openai -- ours, per the lane brief:
context size is measured in OUR tokenizer's tokens. The GGUF that carries the
tokenizer comes from $BARO_GGUF, else <pack>/tokenizer-meta.json "source_gguf".
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / ".work/baro-tokenize"


def load(pack):
    meta_path = Path(pack) / "tokenizer-meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    gguf = os.environ.get("BARO_GGUF")
    if not gguf:
        gguf = meta.get("source_gguf")
        if gguf and not Path(gguf).exists():
            pure = Path(gguf).with_name(Path(gguf).stem + "-pure" + Path(gguf).suffix)
            if pure.exists():
                gguf = str(pure)
    if not gguf or not Path(gguf).exists():
        raise SystemExit(f"tok.load: set BARO_GGUF to the tokenizer's GGUF (got {gguf!r})")
    if not CLI.exists():
        raise SystemExit(f"tok.load: build the CLI first: ./.venv/bin/mojo build tools/baro-tokenize.mojo "
                         f"-I . -I serve -o {CLI}")

    def count_many(texts):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("\0".join(texts))
            path = f.name
        try:
            r = subprocess.run([str(CLI), "count", path, gguf], capture_output=True, text=True, check=True)
        finally:
            os.unlink(path)
        return [int(x) for x in r.stdout.split()][: len(texts)]

    def count(text):
        return count_many([text])[0]

    count.many = count_many
    return gguf, meta, count
