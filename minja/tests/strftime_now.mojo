# Audit finding "strftime_now prints a 1970 date" (docs/state-machines.md): without MINJA_NOW the template clock is the
# wall clock. Run with MINJA_NOW unset: mojo build minja/tests/strftime_now.mojo -I .
from minja import render_chat


def main() raises:
    var out = render_chat("{{ strftime_now('%Y') }}", "{}", "", "", "")
    var year = Int(atol(out.strip()))
    if year < 2026:
        raise Error("FAIL strftime_now: printed year " + out + " (the uptime clock), expected the wall-clock year")
    print("strftime_now: PASS year " + out)
