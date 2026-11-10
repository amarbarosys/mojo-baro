# Audit row C8: the boot actors' RESULT line and exit rule (the actors themselves mount disks and cannot run here;
# these are the pure functions they end through).
import latentos.boot.result as result


def main() raises:
    var fails = 0
    var l1 = result.result_line("S1", "FAIL", "PASS", "complete")
    if l1 != "S1: RESULT mounts=FAIL reservation=PASS status=complete":
        print("FAIL line: " + l1)
        fails += 1
    var l0 = result.result_line("S0", "", "SKIPPED", "incomplete")
    if l0 != "S0: RESULT reservation=SKIPPED status=incomplete":
        print("FAIL S0 line: " + l0)
        fails += 1
    # exit code: a failed mount or an incomplete/error pass fails the process; "complete" with nothing failed does not
    if result.must_fail("PASS", "complete") or result.must_fail("SKIPPED", "complete") or result.must_fail("", "complete"):
        print("FAIL: a complete pass must exit 0")
        fails += 1
    if not (result.must_fail("FAIL", "complete") and result.must_fail("PASS", "incomplete") and result.must_fail("PASS", "error")):
        print("FAIL: a failed mount or an incomplete/error pass must exit non-zero")
        fails += 1
    # hugepages=: a number parses, anything else is -1 (Int() raised and aborted with no result line)
    if result.parse_count("1024") != 1024 or result.parse_count("0") != 0:
        print("FAIL parse_count numbers")
        fails += 1
    if result.parse_count("") != -1 or result.parse_count("12x") != -1 or result.parse_count("-3") != -1 or result.parse_count("1" * 40) != -1:
        print("FAIL parse_count must refuse empty, letters, a sign and a huge value")
        fails += 1
    if fails > 0:
        raise Error("boot_result: " + String(fails) + " check(s) failed")
    print("boot_result: PASS")
