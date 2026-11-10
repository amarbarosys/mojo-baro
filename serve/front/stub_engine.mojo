"""A stand-in for `serve/engine.mojo` (BARO_SERVE=1) that speaks the line protocol of serve/PROTOCOL.md on stdin
and stdout, for the tests of engine_client.mojo. It proves the client module, not the engine.

  stub_engine MODE        MODE: normal | log-before-ready | ready-missing-field | never-ready | dies-idle |
                          dies-mid-stream | dies-by-signal | drops-done | slow

A request of n tokens prints tokens id*1000+i; the stub looks for a cancel line between tokens (stdin is a
socket under posix.spawn_pair, so recv does not block when asked not to). Exit code 0 at end of stdin.
"""
from json import Json, parse
from posix import recv, send_all, kill, getpid, SIGKILL
from std.sys import argv
from std.sys import exit
from std.time import sleep

comptime READY = '{"ready":true,"tmax":16384,"mrows":8,"kmax":8,"spec_k":2,"kv":"int8","pack":"stub"}\n'


def say(s: String):
    _ = send_all(1, s.as_bytes())


def lines(mut buf: List[UInt8], block: Bool) -> List[String]:
    """Complete lines now in the buffer after one recv; exits the stub at end of stdin."""
    var out = List[String]()
    if recv(0, buf, block=block) == 0:
        exit(0)
    var start = 0
    for i in range(len(buf)):
        if buf[i] == 10:
            out.append(String(unsafe_from_utf8=Span(buf)[start:i]))
            start = i + 1
    var rest = List[UInt8](Span(buf)[start:])
    buf = rest^
    return out^


def serve(mode: String, text: String, mut buf: List[UInt8], mut held: List[String]) raises:
    var j = parse(text)
    if j.has("cancel"):
        return
    var id = j.get("id").int()
    var n = j.get("n").int()
    var cancelled = False
    var sent = 0
    for i in range(n):
        if mode == "slow":
            sleep(0.1)
        say(String('{"id":', id, ',"tok":', id * 1000 + i, "}\n"))
        sent += 1
        if mode == "dies-mid-stream" and i == 1:
            exit(3)
        if mode == "dies-by-signal" and i == 1:
            _ = kill(getpid(), SIGKILL)
        for l in lines(buf, False):
            if l.startswith('{"cancel":'):
                cancelled = True
            else:
                held.append(l)
        if cancelled:
            break
    if mode == "drops-done":
        return
    var finish = "cancelled" if cancelled else "length"
    say(String('{"id":', id, ',"done":true,"n":', sent, ',"prefill_s":0.001,"decode_s":0.01,"tok_s":100.0,"finish":"', finish, '"}\n'))


def main() raises:
    var mode = String("normal") if len(argv()) < 2 else String(argv()[1])
    var buf = List[UInt8]()
    if mode == "log-before-ready":
        say("BARO_TMAX: 16384\n")
        say("GTT: used 0\n")
        say('{"progress":0.5}\n')  # JSON without a known key is a log line too
        say("pack stub loaded\n")
    if mode == "ready-missing-field":
        say('{"ready":true,"tmax":16384,"mrows":8,"kmax":8,"pack":"stub"}\n')  # no spec_k
        while True:
            _ = lines(buf, True)
    if mode == "never-ready":
        while True:
            sleep(0.05)
            _ = lines(buf, False)
    say(READY)
    if mode == "dies-idle":
        sleep(0.2)
        exit(7)
    var held = List[String]()
    while True:
        var got = lines(buf, True) if len(held) == 0 else held.copy()
        held = List[String]()
        for l in got:
            serve(mode, l, buf, held)
