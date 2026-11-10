"""bench/ruler/test_gen.py: floor tests for gen.py (lane-ruler brief).

- needle present exactly once per example (niah_single/niah_multikey)
- generated sizes within 2% of the target size, in our tokenizer's tokens
- deterministic for a fixed seed

Run: ./.venv/bin/python3 -m pytest bench/ruler/test_gen.py -v
(or plain: ./.venv/bin/python3 bench/ruler/test_gen.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen
import tok as tokmod

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / ".work/engine-pack-q4"
SIZES = [4096, 8192, 32768]
TOL = 0.02


def _count():
    _, _, count = tokmod.load(PACK)
    return count


def test_niah_single_needle_count():
    count = _count()
    for size in SIZES:
        rows = gen.generate("niah_single", size, 4, seed=1, count=count)
        for row in rows:
            value = row["answers"][0]
            assert row["prompt"].count(value) == 1, (row["id"], value)


def test_niah_multikey_needle_count():
    count = _count()
    for size in SIZES:
        rows = gen.generate("niah_multikey", size, 4, seed=1, count=count)
        for row in rows:
            value = row["answers"][0]
            assert row["prompt"].count(value) == 1, (row["id"], value)
            # 3 keys -> 3 needle sentences in the haystack
            assert row["prompt"].count("special magic numbers for") == 3


def test_sizes_within_tolerance():
    count = _count()
    for task in gen.TASKS:
        for size in SIZES:
            rows = gen.generate(task, size, 3, seed=2, count=count)
            target = size - gen.TOKENS_TO_GENERATE[task]
            for row in rows:
                n = count(row["prompt"])
                rel = abs(n - target) / size
                assert rel <= TOL, (task, size, row["id"], n, target, rel)


def test_deterministic_for_seed():
    count = _count()
    for task in gen.TASKS:
        a = gen.generate(task, 4096, 3, seed=7, count=count)
        b = gen.generate(task, 4096, 3, seed=7, count=count)
        assert a == b, task


def test_vt_answers_all_present():
    count = _count()
    for size in SIZES:
        rows = gen.generate("vt", size, 3, seed=3, count=count)
        for row in rows:
            for var in row["answers"]:
                assert row["prompt"].count(f"VAR {var}") >= 1, (row["id"], var)


def test_cwe_common_words_present():
    count = _count()
    for size in SIZES:
        rows = gen.generate("cwe", size, 3, seed=4, count=count)
        for row in rows:
            assert len(row["answers"]) == 10
            for w in row["answers"]:
                assert w in row["prompt"], (row["id"], w)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
