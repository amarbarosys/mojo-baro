# Schedule model of the vero replay (bench/vero-replay.sh) on a one-GPU engine, from one arm's
# req.tsv: PAR=2 workers take conversations in numeric order, each runs its turns in order; a
# request is P seconds of prefill (rows = prompt - cached) then G tokens at d = decode_s / G each.
# Policies:
#   serial  one request at a time to completion (what baro does today; must reproduce the arm)
#   b2      both workers decoding -> one 2-row step costing c * (da + db) / 2; a prefill runs whole
#   b2s     as b2, prefill in 1024-row chunks with one step of the other worker's decode between
#   b2m     as b2s, the other worker's decode row rides inside each prefill chunk at mix * d
# s scales every prefill. Prints wall and GPU-busy seconds per policy, c and s.
#   tools/vero-sched-sim REQ_TSV [PAR] [C,...]   (C: 2-row step cost factors, default 1.0,1.2,1.4,1.6,2.0)
from std.sys import argv
from std.math import min, max


@fieldwise_init
struct Req(Copyable, Movable):
    var conv: Int
    var turn: Int
    var rows: Int
    var pf: Float64
    var gen: Int
    var d: Float64


@fieldwise_init
struct Worker(Copyable, Movable):
    var conv_i: Int
    var req_i: Int
    var pf_left: Float64
    var pf_chunk: Float64
    var gen_left: Int
    var d: Float64
    var active: Bool


def load(path: String) raises -> List[Req]:
    var out = List[Req]()
    var text = open(path, "r").read()
    var first = True
    for line in text.split("\n"):
        if first:
            first = False
            continue
        if line.byte_length() == 0:
            continue
        var f = line.split("\t")
        var gen = atol(f[6])
        var dec = atof(f[7])
        var cached = 0 if f[4].byte_length() == 0 else atol(f[4])
        out.append(Req(conv=atol(f[0]), turn=atol(f[1]), rows=atol(f[3]) - cached, pf=atof(f[5]),
                       gen=gen, d=dec / Float64(max(gen, 1))))
    return out^


def by_conv(reqs: List[Req]) -> List[List[Req]]:
    var maxc = 0
    for ref r in reqs:
        maxc = max(maxc, r.conv)
    var out = List[List[Req]]()
    for c in range(1, maxc + 1):
        var turns = List[Req]()
        for t in range(1, 1000):
            var found = False
            for ref r in reqs:
                if r.conv == c and r.turn == t:
                    turns.append(r.copy())
                    found = True
            if not found:
                break
        if len(turns) > 0:
            out.append(turns^)
    return out^


def start(mut w: Worker, convs: List[List[Req]], s: Float64):
    ref r = convs[w.conv_i][w.req_i]
    w.pf_left = r.pf * s
    w.pf_chunk = w.pf_left * 1024.0 / Float64(max(r.rows, 1)) if r.rows > 1024 else w.pf_left
    w.gen_left = r.gen
    w.d = r.d
    w.active = True


def advance(mut w: Worker, convs: List[List[Req]], mut next_conv: Int, s: Float64):
    w.req_i += 1
    if w.req_i < len(convs[w.conv_i]):
        start(w, convs, s)
        return
    if next_conv < len(convs):
        w.conv_i = next_conv
        w.req_i = 0
        next_conv += 1
        start(w, convs, s)
        return
    w.active = False


def finish_if_done(mut w: Worker, convs: List[List[Req]], mut next_conv: Int, s: Float64):
    if w.active and w.pf_left <= 0 and w.gen_left <= 0:
        advance(w, convs, next_conv, s)


def simulate(convs: List[List[Req]], par: Int, policy: String, c: Float64, s: Float64, mix: Float64) -> Float64:
    var ws = List[Worker]()
    var next_conv = 0
    for _ in range(par):
        var w = Worker(conv_i=0, req_i=0, pf_left=0, pf_chunk=0, gen_left=0, d=0, active=False)
        if next_conv < len(convs):
            w.conv_i = next_conv
            next_conv += 1
            start(w, convs, s)
        ws.append(w^)
    var t = 0.0
    var turn = 0
    while True:
        var live = List[Int]()
        for i in range(len(ws)):
            if ws[i].active:
                live.append(i)
        if len(live) == 0:
            break
        if policy == "serial" or len(live) == 1:
            var i = live[0] if policy != "serial" else live[turn % len(live)]
            if policy == "serial":
                t += ws[i].pf_left + Float64(ws[i].gen_left) * ws[i].d
                ws[i].pf_left = 0
                ws[i].gen_left = 0
                finish_if_done(ws[i], convs, next_conv, s)
                turn += 1
                continue
            if ws[i].pf_left > 0:
                t += ws[i].pf_left
                ws[i].pf_left = 0
            else:
                t += ws[i].d
                ws[i].gen_left -= 1
            finish_if_done(ws[i], convs, next_conv, s)
            continue
        var a = live[0]
        var b = live[1]
        var pa = ws[a].pf_left > 0
        var pb = ws[b].pf_left > 0
        if not pa and not pb:
            t += c * (ws[a].d + ws[b].d) / 2
            ws[a].gen_left -= 1
            ws[b].gen_left -= 1
        elif pa and pb:
            var i = a if turn % 2 == 0 else b
            turn += 1
            var q = ws[i].pf_left if policy == "b2" else min(ws[i].pf_left, ws[i].pf_chunk)
            t += q
            ws[i].pf_left -= q
        else:
            var p = a if pa else b
            var o = b if pa else a
            if policy == "b2":
                t += ws[p].pf_left
                ws[p].pf_left = 0
            else:
                var q = min(ws[p].pf_left, ws[p].pf_chunk)
                t += q
                ws[p].pf_left -= q
                if policy == "b2m":
                    t += mix * ws[o].d
                else:
                    t += ws[o].d
                ws[o].gen_left -= 1
        finish_if_done(ws[a], convs, next_conv, s)
        finish_if_done(ws[b], convs, next_conv, s)
    return t


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: vero-sched-sim REQ_TSV [PAR] [C,...]")
    var reqs = load(String(args[1]))
    var par = atol(args[2]) if len(args) > 2 else 2
    var convs = by_conv(reqs)
    var pf = 0.0
    var dec = 0.0
    for ref r in reqs:
        pf += r.pf
        dec += r.d * Float64(r.gen)
    print("requests", len(reqs), "conversations", len(convs), "prefill_s", pf, "decode_s", dec)
    var ss: List[Float64] = [1.0, 0.7, 0.45]
    var cs: List[Float64] = [1.0, 1.2, 1.4, 1.6, 2.0]
    if len(args) > 3:
        cs = List[Float64]()
        for part in String(args[3]).split(","):
            cs.append(atof(part))
    for s in ss:
        print("s", s, "serial", simulate(convs, par, "serial", 1.0, s, 0.0))
        for c in cs:
            print("  c", c, "b2", simulate(convs, par, "b2", c, s, 0.0), "b2s", simulate(convs, par, "b2s", c, s, 0.0),
                  "b2m(0.15)", simulate(convs, par, "b2m", c, s, 0.15))
