"""baro face, engine panel (F2): the right pane. One `Dash` holds what the pane shows; `dash_pane` refreshes it (at most
once a second) and draws it.

  engine state tag, queue depth     /health of the front (engine_state, queue)
  model card                        /v1/models id + /health pack, limits (tmax, seqs, spec_k, tier)
  tok/s, spark line, path, fail     the `timings` object of a completion reply (the done line): tok_s_gen, path, fail.
                                    F1's chat feeds it with `Dash.on_done`; FACE_DASH_PROBE=N makes the first refresh send
                                    N short completions itself (prompt from FACE_DASH_PROMPT, a JSON value) (the F2 gate does; a window with no chat would show "no run yet").
                                    A 502 engine_invalid_tokens reply carries the fail word in its message: shown too.
  VRAM bar                          /sys/class/drm/card*/device/mem_info_vram_{used,total}: the card with the most VRAM
                                    (the same counters rocm-smi prints)

Each refresh costs up to three short HTTP reads on the window thread (300 ms cap each), as in F0's status line.
"""
from std.ffi import external_call
from std.os import getenv
from std.time import perf_counter_ns
from imkit.ui import Ui
from json import Json, parse
from net import Reply, http_get, read_reply, tcp_connect

comptime HIST = 60  # tok/s samples kept for the spark line
comptime GIB = Float64(1 << 30)


def fmt1(x: Float64) -> String:
    """One decimal, rounded half up: 142.3. (The gate reads this back by OCR.)"""
    var t = Int(x * 10.0 + 0.5) if x > 0.0 else 0
    return String(t // 10) + "." + String(t % 10)


def read_int(path: String) -> Int:
    """A sysfs counter, or -1 when it is not there."""
    try:
        with open(path, "r") as f:
            return Int(f.read().strip())
    except:
        return -1


def vram_card() -> Tuple[Int, Int]:
    """(used, total) bytes of the card with the most VRAM, (-1, -1) when none reports."""
    var best_used = -1
    var best_total = -1
    for c in range(4):
        var base = "/sys/class/drm/card" + String(c) + "/device/mem_info_vram_"
        var total = read_int(base + "total")
        if total > best_total:
            best_total = total
            best_used = read_int(base + "used")
    return (best_used, best_total)


def post_json(host: String, port: Int, path: String, body: String, timeout_ms: Int) -> Reply:
    try:
        var fd = tcp_connect(host, port)
        if fd < 0:
            return Reply(0, String(""), String("connection refused"))
        var req = "POST " + path + " HTTP/1.1\r\nHost: " + host + "\r\nContent-Type: application/json\r\nContent-Length: " + String(body.byte_length()) + "\r\nConnection: close\r\n\r\n" + body
        var n = req.byte_length()
        if Int(external_call["send", Int](fd, req.unsafe_ptr(), UInt(n), Int32(0x4000))) != n:
            _ = external_call["close", Int32](fd)
            return Reply(0, String(""), String("send failed"))
        return read_reply(fd, timeout_ms)
    except e:
        return Reply(0, String(""), String(e))


struct Dash(Movable):
    var state: String  # engine_state word, "" before the first answer
    var up: Bool  # the front answered /health
    var queue: Int
    var model: String
    var pack: String
    var limits: String  # "tmax 16384 seqs 1 spec_k 2 tier 0"
    var runs: Int  # done lines seen
    var tok_s: Float64  # last done line
    var path: String
    var fail: Int
    var hist: List[Float64]
    var vram_used: Int
    var vram_total: Int
    var last_ns: Int
    var probed: Bool

    def __init__(out self):
        self.state = ""
        self.up = False
        self.queue = 0
        self.model = ""
        self.pack = ""
        self.limits = ""
        self.runs = 0
        self.tok_s = 0.0
        self.path = ""
        self.fail = 0
        self.hist = List[Float64]()
        self.vram_used = -1
        self.vram_total = -1
        self.last_ns = 0
        self.probed = False

    def on_done(mut self, timings: Json):
        """One done line (the `timings` object of a reply, or the engine's own done line: tok_s_gen or tok_s)."""
        var t = timings.get("tok_s_gen") if timings.has("tok_s_gen") else timings.get("tok_s")
        if not t.is_num():
            return
        self.tok_s = t.num()
        self.runs += 1
        if len(self.hist) >= HIST:
            _ = self.hist.pop(0)
        self.hist.append(self.tok_s)
        self.path = timings.get("path").str() if timings.has("path") else String("")
        self.fail = timings.get("fail").int() if timings.has("fail") and timings.get("fail").is_num() else 0

    def on_reply(mut self, r: Reply):
        """A completion reply: 200 carries `timings`; a 502 invalid-tokens reply carries the fail word in its message."""
        try:
            var j = parse(r.body)
            if r.status == 200 and j.has("timings"):
                self.on_done(j.get("timings"))
                print("dash done tok_s_gen=" + fmt1(self.tok_s) + " raw=" + String(self.tok_s) + " path=" + self.path + " fail=" + String(self.fail), flush=True)
            elif j.has("error"):
                var m = j.get("error").get("message").str()
                var k = m.find("fail word ")
                if k >= 0:
                    self.fail = Int(String(m[byte = k + 10 :]).split(")")[0])
                print("dash reply HTTP " + String(r.status) + " fail=" + String(self.fail), flush=True)
        except e:
            print("dash reply unreadable: " + String(e), flush=True)

    def probe(mut self, host: String, port: Int, n: Int, tokens: Int):
        # FACE_DASH_PROMPT: a JSON prompt (a string, or token ids when the front has no tokenizer)
        var prompt = getenv("FACE_DASH_PROMPT", '"The capital of France is"')
        for _ in range(n):
            var body = '{"model":"face","prompt":' + prompt + ',"max_tokens":' + String(tokens) + ',"temperature":0}'
            self.on_reply(post_json(host, port, "/v1/completions", body, 60000))

    def refresh(mut self, host: String, port: Int) raises:
        try:
            var r = http_get(host, port, "/health")
            self.up = r.status != 0
            if r.status != 0:
                var j = parse(r.body)
                self.state = j.get("engine_state").str() if j.has("engine_state") else String("?")
                self.queue = j.get("queue").int() if j.has("queue") else 0
                var pk = j.get("pack").str()  # empty for null or a non-string
                if pk:
                    self.pack = pk
                if j.has("limits"):
                    var l = j.get("limits")
                    var s = String("")
                    for k in ["tmax", "seqs", "spec_k", "tier"]:
                        if l.has(k) and l.get(k).is_num():
                            s += (" " if s else "") + String(k) + " " + String(l.get(k).int())
                    self.limits = s
            else:
                self.state = ""
        except:
            self.up = False
        if self.up and not self.model:
            try:
                var m = http_get(host, port, "/v1/models")
                if m.status == 200:
                    var j = parse(m.body)
                    self.model = j.get("data").at(0).get("id").str()
            except:
                pass
        var v = vram_card()
        self.vram_used, self.vram_total = v[0], v[1]
        if self.up and not self.probed:
            self.probed = True
            var n = getenv("FACE_DASH_PROBE", "")
            if n:
                self.probe(host, port, Int(n), Int(getenv("FACE_DASH_TOKENS", "64")))
                self.refresh_vram()

    def refresh_vram(mut self):
        var v = vram_card()
        self.vram_used, self.vram_total = v[0], v[1]


def short(s: String, n: Int) -> String:
    """The tail of a path or id: model ids are file paths."""
    var b = s.byte_length()
    if b <= n:
        return s
    return String("~") + String(s[byte = b - n + 1 :])


def dash_pane(mut ui: Ui, mut d: Dash, host: String, port: Int, x0: Float32, y0: Float32, x1: Float32, y1: Float32):
    """Refresh (at most once a second, the first call at once) and draw the engine panel into [x0,x1] x [y0,y1]."""
    var now = Int(perf_counter_ns())
    if d.last_ns == 0 or now - d.last_ns > 1_000_000_000:
        try:
            d.refresh(host, port)
        except:
            d.up = False
        d.last_ns = Int(perf_counter_ns())
    var t = ui.theme.copy()
    ui.panel(x0, y0, x1, y1, "Engine")
    var x = x0 + 20
    var w = x1 - x0 - 40
    # state tag + queue
    var tag = t.muted
    if d.up:
        tag = t.good if d.state == "idle" else t.accent
    else:
        tag = t.bad
    ui.rrect(x, y0 + 52, x + 120, y0 + 80, 14, tag)
    _ = ui.text(x + 14, y0 + 56, d.state if d.state else String("down"), 1, t.on_accent)
    _ = ui.text(x + 136, y0 + 56, "queue " + String(d.queue), 1, t.text)
    # model card
    _ = ui.text(x, y0 + 98, "Model card", 1, t.muted)
    _ = ui.text(x, y0 + 122, "model " + short(d.model if d.model else String("-"), 30), 0, t.text)
    _ = ui.text(x, y0 + 142, "pack " + short(d.pack if d.pack else String("-"), 31), 0, t.text)
    _ = ui.text(x, y0 + 162, d.limits if d.limits else String("limits -"), 0, t.text)
    # tok/s and the done line's words
    _ = ui.text(x, y0 + 196, "Last run", 1, t.muted)
    if d.runs > 0:
        _ = ui.text(x, y0 + 218, "tok/s " + fmt1(d.tok_s), 2, t.text)
        _ = ui.text(x, y0 + 252, "path " + (d.path if d.path else String("-")) + "  fail " + String(d.fail), 1, t.bad if d.fail != 0 else t.text)
    else:
        _ = ui.text(x, y0 + 218, "no run yet", 1, t.muted)
        _ = ui.text(x, y0 + 252, "fail " + String(d.fail), 1, t.bad if d.fail != 0 else t.muted)
    # spark line of the last HIST done lines, scaled to its own max
    var sy0 = y0 + 290
    var sy1 = y0 + 370
    ui.rrect(x, sy0, x + w, sy1, 8, t.surface)
    var n = len(d.hist)
    if n > 1:
        var hi = d.hist[0]
        for i in range(n):
            if d.hist[i] > hi:
                hi = d.hist[i]
        var dx = (w - 16) / Float32(HIST - 1)
        for i in range(n - 1):
            var ya = sy1 - 8 - Float32(d.hist[i] / hi) * (sy1 - sy0 - 16)
            var yb = sy1 - 8 - Float32(d.hist[i + 1] / hi) * (sy1 - sy0 - 16)
            ui.line(x + 8 + Float32(i) * dx, ya, x + 8 + Float32(i + 1) * dx, yb, 2, t.accent)
    _ = ui.text(x, sy1 + 6, "tok/s, last " + String(HIST) + " runs", 0, t.muted)
    # VRAM bar
    _ = ui.text(x, y0 + 408, "VRAM", 1, t.muted)
    if d.vram_total > 0 and d.vram_used >= 0:
        _ = ui.text(x, y0 + 430, fmt1(Float64(d.vram_used) / GIB) + " / " + fmt1(Float64(d.vram_total) / GIB) + " GiB", 2, t.text)
        var frac = Float32(Float64(d.vram_used) / Float64(d.vram_total))
        ui.rrect(x, y0 + 468, x + w, y0 + 488, 10, t.surface)
        if frac > 0:
            ui.rrect(x, y0 + 468, x + max(Float32(20), w * frac), y0 + 488, 10, t.warn if frac > 0.9 else t.good)
    else:
        _ = ui.text(x, y0 + 430, "no card reports VRAM", 1, t.muted)
