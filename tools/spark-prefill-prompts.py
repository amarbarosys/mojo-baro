#!/usr/bin/env python3
"""Write bench/spark-prefill-prompts/p{0064,0256,1024,2048}.txt: prefixes of the
concatenation of bench/mtp-prompts/*.txt + docs/*.md sliced to the target id
count by binary search over characters, counted with the Mojo tokenizer CLI
(.work/baro-tokenize encode FILE MODEL.gguf). Deterministic, re-runnable."""
import subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / ".work/baro-tokenize"
MODEL = Path("~/Models/spark-x2.5-4b/Spark-X2.5-4B-Q8_0-requant.gguf").expanduser()
OUT = ROOT / "bench/spark-prefill-prompts"
TARGETS = [int(x) for x in sys.argv[1:]] or [64, 256, 1024, 2048]


def count_ids(text):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text); name = f.name
    r = subprocess.run([str(CLI), "encode", name, str(MODEL)], capture_output=True, text=True, check=True)
    Path(name).unlink()
    return len(r.stdout.split())


def main():
    parts = [p.read_text() for p in sorted((ROOT / "bench/mtp-prompts").glob("*.txt"))]
    parts += [p.read_text() for p in sorted((ROOT / "docs").glob("*.md"))]
    text = "\n".join(parts)
    OUT.mkdir(exist_ok=True)
    for t in TARGETS:
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count_ids(text[:mid]) <= t:
                lo = mid
            else:
                hi = mid - 1
        n = count_ids(text[:lo])
        (OUT / f"p{t:04d}.txt").write_text(text[:lo])
        print(f"p{t:04d}.txt {n} ids {lo} chars {'OK' if abs(n - t) / t <= 0.03 else 'OFF'}")


if __name__ == "__main__":
    main()
