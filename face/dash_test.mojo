"""Tests dash.mojo with no engine, no window: number format, done-line ingestion, the 60-sample ring, the fail word of a
502 reply, a reply with no timings, and the vram reader.  usage: mojo build -I . -I face face/dash_test.mojo"""
from dash import Dash, fmt1, vram_card, HIST
from json import parse
from net import Reply


def check(ok: Bool, what: String) raises:
    if not ok:
        raise Error("FAIL dash_test: " + what)


def main() raises:
    check(fmt1(142.34) == "142.3" and fmt1(0.05) == "0.1" and fmt1(9.96) == "10.0" and fmt1(-1.0) == "0.0", "fmt1")
    var d = Dash()
    d.on_done(parse('{"tok_s_gen":142.34,"path":"mega","fail":0}'))
    check(d.runs == 1 and d.path == "mega" and d.fail == 0 and abs(d.tok_s - 142.34) < 1e-9, "tok_s_gen done line")
    d.on_done(parse('{"tok_s":88.5}'))  # the engine's own spelling, no path word
    check(d.runs == 2 and d.path == "" and abs(d.tok_s - 88.5) < 1e-9, "tok_s spelling")
    d.on_done(parse('{"prefill_s":0.1}'))  # no tok/s at all: not a run
    check(d.runs == 2, "a line without tok/s counted")
    for i in range(HIST + 10):
        d.on_done(parse('{"tok_s_gen":' + String(i + 1) + '.0,"fail":2}'))
    check(len(d.hist) == HIST and d.hist[HIST - 1] == Float64(HIST + 10) and d.fail == 2, "ring keeps the newest " + String(HIST))
    var e = Dash()
    e.on_reply(Reply(200, '{"timings":{"tok_s_gen":131.0,"path":"spec","fail":0}}', String("")))
    check(e.runs == 1 and e.path == "spec", "200 reply")
    e.on_reply(Reply(502, '{"error":{"message":"engine: the run produced invalid tokens (fail word 3); the output","type":"engine_invalid_tokens"}}', String("")))
    check(e.fail == 3 and e.runs == 1, "502 fail word")
    e.on_reply(Reply(0, String(""), String("timeout")))
    check(e.runs == 1, "no reply")
    var v = vram_card()
    check(v[1] == -1 or (v[0] >= 0 and v[0] <= v[1]), "vram used within total")
    print("PASS dash_test: format, done lines, ring, fail word, vram " + String(v[0]) + "/" + String(v[1]))
