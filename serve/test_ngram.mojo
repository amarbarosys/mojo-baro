from std.testing import assert_equal
from ngram import ngram_propose


def main() raises:
    assert_equal(len(ngram_propose(List[Int](), 2)), 0)
    assert_equal(len(ngram_propose([1, 2, 3], 2)), 0)
    assert_equal(ngram_propose([1, 2, 3, 4, 1, 2], 2), [3, 4])
    assert_equal(ngram_propose([1, 2, 3, 4, 1, 2], 1), [3])
    assert_equal(ngram_propose([1, 2, 3, 4, 1, 2], 0), List[Int]())
    assert_equal(ngram_propose([1, 2, 3, 1, 2, 4, 1, 2], 2), [4, 1])
    assert_equal(ngram_propose([7, 7, 7, 7], 3), [7])
    assert_equal(ngram_propose([1, 2, 3, 4, 1, 2], 2, 2), [3, 4])
    assert_equal(ngram_propose([1, 2, 3, 4, 1, 2], 2, 3), List[Int]())
    print("PASS ngram: empty, no match, longest, newest, bounded, overlap, min width")
