"""bench/ruler/test_score.py: canned-response tests for score.py's scoring
and effective-length logic (lane-ruler brief floor)."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score


def test_string_match_all_full():
    assert score.string_match_all("the value is 42", ["42"]) == 1.0


def test_string_match_all_partial():
    assert score.string_match_all("the value is 42", ["42", "99"]) == 0.5


def test_string_match_all_case_insensitive():
    assert score.string_match_all("Answer: ABCDE, FGHIJ", ["abcde", "fghij"]) == 1.0


def test_string_match_all_none():
    assert score.string_match_all("no match here", ["42"]) == 0.0


def test_string_match_all_empty_answers():
    assert score.string_match_all("anything", []) == 0.0


def test_score_file_and_effective_length():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        prompts_dir, responses_dir = d / "prompts", d / "responses"
        prompts_dir.mkdir()
        for size, correct in [(4096, 4), (8192, 4), (16384, 1)]:
            (responses_dir / f"niah_single_{size}").mkdir(parents=True)
            with open(prompts_dir / f"niah_single_{size}.jsonl", "w") as f:
                for i in range(4):
                    row = {"id": f"niah_single-{size}-{i:04d}", "prompt": "x",
                           "answers": ["needle42"], "task": "niah_single", "size": size, "seed": 0}
                    f.write(json.dumps(row) + "\n")
                    text = "needle42 found" if i < correct else "nothing here"
                    (responses_dir / f"niah_single_{size}" / f"{row['id']}.response.txt").write_text(text)

        table = {}
        for size in (4096, 8192, 16384):
            scores = score.score_file(prompts_dir / f"niah_single_{size}.jsonl", responses_dir / f"niah_single_{size}")
            table[size] = 100 * sum(scores) / len(scores)

        assert table[4096] == 100.0
        assert table[8192] == 100.0
        assert table[16384] == 25.0
        # effective length: largest size with avg >= 0.85 * 4k avg (100) -> 8192
        threshold = 0.85 * table[4096]
        eff = max((s for s in (4096, 8192, 16384) if table[s] >= threshold), default=None)
        assert eff == 8192


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
