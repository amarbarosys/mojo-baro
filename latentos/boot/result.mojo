# result.mojo: the one machine-readable last line of the S0 and S1 boot actors, and the exit rule.
# `S1: RESULT mounts=PASS|FAIL|SKIPPED reservation=PASS|FAIL|SKIPPED status=complete|incomplete|error`
# (S0 has no mounts key). A harness greps this line instead of the prose, and the process exits non-zero unless the
# pass completed and nothing failed to mount.


def result_line(actor: String, mounts: String, reservation: String, status: String) -> String:
    var out = actor + ": RESULT"
    if mounts != "":
        out += " mounts=" + mounts
    return out + " reservation=" + reservation + " status=" + status


def must_fail(mounts: String, status: String) -> Bool:
    """True when the process must exit non-zero: the pass did not complete, or a mount failed."""
    return status != "complete" or mounts == "FAIL"


def parse_count(s: String) -> Int:
    """A non-negative decimal, or -1 for anything else (empty, a sign, a letter): `Int(s)` raised and aborted the
    actor with no result line at all."""
    var b = s.as_bytes()
    if len(b) == 0 or len(b) > 12:
        return -1
    var n = 0
    for i in range(len(b)):
        if b[i] < 48 or b[i] > 57:
            return -1
        n = n * 10 + Int(b[i] - 48)
    return n
