"""Bounded prompt lookup proposals; target verification decides acceptance."""


def ngram_propose(history: List[Int], limit: Int, min_width: Int = 1) -> List[Int]:
    var n = len(history)
    # Longest suffix first, latest completed occurrence first. Proposals only
    # read committed history, never rejected tokens left in device storage.
    # min_width gates weak matches: a short suffix mostly drafts a token the
    # target rejects, and a rejected window costs a full row per draft.
    for width in range(min(8, n - 1), max(min_width, 1) - 1, -1):
        for start in range(n - width - 1, max(-1, n - 4096), -1):
            var same = True
            for j in range(width):
                if history[start + j] != history[n - width + j]:
                    same = False
                    break
            if same:
                var result = List[Int]()
                for j in range(min(limit, n - start - width)):
                    result.append(history[start + width + j])
                return result^
    return List[Int]()


def ngram_propose_tree(history: List[Int], limit: Int, branches: Int = 3, min_width: Int = 1) -> List[List[Int]]:
    var n = len(history)
    var results = List[List[Int]]()
    for width in range(min(8, n - 1), max(min_width, 1) - 1, -1):
        for start in range(n - width - 1, max(-1, n - 4096), -1):
            var same = True
            for j in range(width):
                if history[start + j] != history[n - width + j]:
                    same = False
                    break
            if same:
                var branch = List[Int]()
                for j in range(min(limit, n - start - width)):
                    branch.append(history[start + width + j])
                results.append(branch^)
                if len(results) >= branches:
                    return results^
    return results^

