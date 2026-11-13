"""baro face, chat pane (F1): message list, input field, tokens streamed over SSE from the front, stop, scroll, copy.

  var chat = Chat("127.0.0.1", 8080)
  chat.draw(ui, inp, x0, y0, x1, y1)      once a frame, inside Ui.begin / Ui.finish

No thread: one non-blocking socket, polled for what has arrived at the top of every draw (the frame loop never waits on
the front). The request is HTTP/1.0 so the SSE body is not chunk-framed. Closing the socket (Stop, a finished answer,
or the Chat going out of scope when the window closes) is the cancel: the front sees the client gone and cancels the run
(its log line "engine: client dropped request N; cancelling").
"""
from std.ffi import OwnedDLHandle, external_call
from imkit.ui import Ui, cp_floor
from imkit.window import Input, readable
from json import parse
from net import tcp_connect

comptime SC_RETURN = 40  # SDL scancode
comptime LH = Float32(20)  # line height at font size 0
comptime ID_FIELD = 9100
comptime ID_SEND = 9101
comptime ID_STOP = 9102
comptime ID_COPY = 9103
comptime MAX_TOKENS = 16000  # the repo floor for any run (bench/min16k.sh)


# ---- SSE: pure functions, tested offline by face/chat_test.mojo --------------------------------------------------


def json_escape(s: String) -> String:
    var out = List[UInt8]()
    for b in s.as_bytes():
        if b == 34 or b == 92:
            out.append(92)
            out.append(b)
        elif b == 10:
            out.append(92)
            out.append(110)
        elif b == 13:
            out.append(92)
            out.append(114)
        elif b == 9:
            out.append(92)
            out.append(116)
        elif b < 32:
            out.append(32)
        else:
            out.append(b)
    return String(unsafe_from_utf8=Span(out))


def sse_split(mut pending: List[UInt8]) -> List[String]:
    """Complete events (text up to a blank line) out of `pending`; the incomplete tail stays in `pending`."""
    var out = List[String]()
    var start = 0
    var i = 0
    var n = len(pending)
    while i + 1 < n:
        if pending[i] == 10 and pending[i + 1] == 10:
            out.append(String(unsafe_from_utf8=Span(pending)[start:i]))
            start = i + 2
            i += 2
        else:
            i += 1
    if start > 0:
        var rest = List[UInt8]()
        rest.extend(Span(pending)[start:])
        pending = rest^
    return out^


struct Piece(Copyable, Movable):
    var kind: Int  # 0 nothing (keep-alive, empty delta), 1 text, 2 done, 3 error
    var text: String

    def __init__(out self, kind: Int, var text: String):
        self.kind, self.text = kind, text^


def sse_decode(ev: String) -> Piece:
    """One event: the `data:` payload as the text delta, the end marker, or the front's error object."""
    var payload = String("")
    var found = False
    for line in ev.split("\n"):
        if line.startswith("data:"):
            payload = String(String(line[byte=5:]).strip())
            found = True
    if not found:
        return Piece(0, String(""))
    if payload == "[DONE]":
        return Piece(2, String(""))
    try:
        var j = parse(payload)
        if j.has("error"):
            var m = j.get("error").text("message", "front error")
            return Piece(3, m^)
        var ch = j.get("choices")
        if ch.len() > 0:
            var d = ch.at(0).get("delta")
            return Piece(1, d.text("content"))
    except e:
        return Piece(3, "bad event: " + String(e))
    return Piece(0, String(""))


def wrap(mut ui: Ui, text: String, w: Float32) -> List[String]:
    """`text` cut into lines no wider than w at font size 0: at newlines, then at the last space that fits, else mid-word."""
    var out = List[String]()
    for para in text.split("\n"):
        var rest = String(para)
        if rest.byte_length() == 0:
            out.append(String(""))
            continue
        while rest.byte_length() > 0:
            var n = rest.byte_length()
            if ui.tw(rest, 0) > w:
                var lo = 1  # widest prefix that fits, by bisection (a linear scan cost O(n^2) glyph widths per line)
                var hi = min(n, 400)
                while lo < hi:
                    var mid = (lo + hi + 1) // 2
                    var k = cp_floor(rest.as_bytes(), mid)
                    if k > 0 and ui.tw(StringSlice(unsafe_from_utf8=rest.as_bytes()[:k]), 0) <= w:
                        lo = mid
                    else:
                        hi = mid - 1
                n = max(cp_floor(rest.as_bytes(), lo), 1)
                var sp = -1
                for i in range(n):
                    if rest.as_bytes()[i] == 32:
                        sp = i
                if sp > 0 and n < rest.byte_length():
                    n = sp
            out.append(String(StringSlice(unsafe_from_utf8=rest.as_bytes()[:n])))
            var tail = String(StringSlice(unsafe_from_utf8=rest.as_bytes()[n:]))
            rest = String(tail.lstrip(" ")) if n < rest.byte_length() else String("")
    return out^


# ---- clipboard (SDL3, through its own handle on the library the window already loaded) ---------------------------


def clipboard_set(text: String) -> Bool:
    try:
        var sdl = OwnedDLHandle("libSDL3.so.0")
        var t = text
        return sdl.get_function[Bool]("SDL_SetClipboardText")(Int(t.unsafe_ptr()))
    except:
        return False


# ---- the pane -----------------------------------------------------------------------------------------------------


struct Msg(Copyable, Movable):
    var user: Bool
    var text: String
    var lines: List[String]
    var wrapped: Int  # byte length of `text` that `lines` was built from
    var wrap_w: Float32

    def __init__(out self, user: Bool, var text: String):
        self.user, self.wrapped, self.wrap_w = user, -1, 0
        self.text = text^
        self.lines = List[String]()


struct Chat(Movable):
    var host: String
    var port: Int
    var msgs: List[Msg]
    var input: String
    var off: Float32
    var follow: Bool
    var fd: Int32  # the open stream, -1 when idle
    var raw: List[UInt8]  # response bytes until the header is complete
    var header_done: Bool
    var pending: List[UInt8]  # SSE bytes not yet a full event
    var note: String  # status or error line under the input
    var streamed: Int  # bytes of the current answer so far
    var stop_at: SIMD[DType.float32, 2]  # where the Stop / Send buttons were drawn last frame (scripted gates click them)
    var send_at: SIMD[DType.float32, 2]
    var copy_at: SIMD[DType.float32, 2]
    var copied: String  # the text last handed to the clipboard

    def __init__(out self, host: String, port: Int):
        self.host, self.port = host.copy(), port
        self.msgs = List[Msg]()
        self.input = String("")
        self.off, self.follow, self.fd, self.header_done, self.streamed = 0, True, -1, False, 0
        self.raw = List[UInt8]()
        self.pending = List[UInt8]()
        self.note = String("")
        self.copied = String("")
        self.stop_at = SIMD[DType.float32, 2](0)
        self.send_at = SIMD[DType.float32, 2](0)
        self.copy_at = SIMD[DType.float32, 2](0)

    def __deinit__(deinit self):
        if self.fd >= 0:
            _ = external_call["close", Int32](self.fd)  # window closed mid-answer: the front cancels the run

    def busy(self) -> Bool:
        return self.fd >= 0

    def last_answer(self) -> String:
        var k = len(self.msgs) - 1
        while k >= 0:
            if not self.msgs[k].user:
                return self.msgs[k].text.copy()
            k -= 1
        return String("")

    def _close(mut self):
        if self.fd >= 0:
            _ = external_call["close", Int32](self.fd)
            self.fd = -1

    def stop(mut self):
        """Cancel the answer in flight: close the stream; the text so far stays in the list."""
        if self.fd >= 0:
            self._close()
            self.note = "stopped"

    def submit(mut self):
        """Send `input` as the next user turn and open the stream. A refused connection is a note, never a crash."""
        if self.fd >= 0 or self.input.byte_length() == 0:
            return
        self.msgs.append(Msg(True, self.input^))
        self.input = String("")
        var body = String('{"stream":true,"max_tokens":') + String(MAX_TOKENS) + String(',"messages":[')
        for i in range(len(self.msgs)):
            if i > 0:
                body += ","
            body += '{"role":"' + ("user" if self.msgs[i].user else "assistant") + '","content":"' + json_escape(self.msgs[i].text) + '"}'
        body += "]}"
        self.msgs.append(Msg(False, String("")))
        self.follow = True
        self.raw = List[UInt8]()
        self.pending = List[UInt8]()
        self.header_done, self.streamed = False, 0
        try:
            var fd = tcp_connect(self.host, self.port)
            if fd < 0:
                self.note = "front unreachable (connection refused)"
                return
            var req = "POST /v1/chat/completions HTTP/1.0\r\nHost: " + self.host + "\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nContent-Length: " + String(body.byte_length()) + "\r\n\r\n" + body
            var n = req.byte_length()
            if Int(external_call["send", Int](fd, req.unsafe_ptr(), UInt(n), Int32(0x4000))) != n:
                _ = external_call["close", Int32](fd)
                self.note = "send to the front failed"
                return
            self.fd = fd
            self.note = "waiting for the first token"
        except e:
            self.note = "front: " + String(e)

    def _feed(mut self, chunk: Span[UInt8, _]):
        self.pending.extend(chunk)
        for ev in sse_split(self.pending):
            var p = sse_decode(ev)
            if p.kind == 1 and p.text.byte_length() > 0:
                self.msgs[len(self.msgs) - 1].text += p.text
                self.streamed += p.text.byte_length()
                self.note = "streaming, " + String(self.streamed) + " bytes"
            elif p.kind == 2:
                self._close()
                self.note = "done, " + String(self.streamed) + " bytes"
                return
            elif p.kind == 3:
                self._close()
                self.note = "front: " + p.text
                return

    def pump(mut self):
        """Read what has arrived (up to 64 KB), without waiting."""
        var buf = List[UInt8](length=8192, fill=0)
        var budget = 8
        while self.fd >= 0 and budget > 0 and readable(Int(self.fd), 0):
            budget -= 1
            var n = Int(external_call["recv", Int](self.fd, buf.unsafe_ptr(), UInt(8192), Int32(0)))
            if n <= 0:
                self._close()
                if self.note.startswith("streaming") or self.note.startswith("waiting"):
                    self.note = "front closed the stream without [DONE]"
                return
            if self.header_done:
                self._feed(Span(buf)[:n])
                continue
            self.raw.extend(Span(buf)[:n])
            var s = String(unsafe_from_utf8=Span(self.raw))
            var sep = s.find("\r\n\r\n")
            if sep < 0:
                continue
            var head = String(s[byte=0:sep])
            var status = 0
            try:
                status = Int(String(head[byte=9:12]))
            except:
                status = 0
            if status != 200:
                var rest = String(s[byte = sep + 4 :])
                self._close()
                self.note = "front: HTTP " + String(status) + " " + String(rest[byte=0 : min(rest.byte_length(), 160)])
                return
            if head.lower().find("transfer-encoding: chunked") >= 0:
                self._close()
                self.note = "front sent a chunked body (HTTP/1.0 asked for none)"
                return
            self.header_done = True
            var body = List[UInt8]()
            body.extend(Span(self.raw)[sep + 4 :])
            self.raw = List[UInt8]()
            self._feed(Span(body))

    def draw(mut self, mut ui: Ui, inp: Input, x0: Float32, y0: Float32, x1: Float32, y1: Float32):
        self.pump()
        var t = ui.theme.copy()
        ui.panel(x0, y0, x1, y1, "Chat")
        var vx, vy = x0 + 16, y0 + 40
        var vw, vh = x1 - x0 - 32, y1 - 76 - vy
        var tw = vw - 28  # text width inside the viewport (room for the bar)
        # rows: per message one header row, its text lines, one gap row
        var total = 0
        for i in range(len(self.msgs)):
            ref m = self.msgs[i]
            if m.wrapped != m.text.byte_length() or m.wrap_w != tw:
                m.lines = wrap(ui, m.text, tw)
                m.wrapped, m.wrap_w = m.text.byte_length(), tw
            total += len(m.lines) + 2
        ui.x, ui.y = vx, vy
        var hot = ui.hovered(vx, vy, vx + vw, vy + vh)
        var maxoff = max(Float32(total) * LH - vh, Float32(0))
        if hot and inp.wheel > 0:
            self.follow = False  # the wheel up (imkit: positive moves toward the start) leaves the tail
        if self.follow:
            self.off = maxoff
        var rng = ui.scroll_begin(9200, vw, vh, total, LH, self.off)
        if self.off >= maxoff - 1:
            self.follow = True
        if len(self.msgs) == 0:
            _ = ui.text(vx + 12, vy + 12, "Type a message below and press Enter.", 0, t.muted)
        var k = 0
        for i in range(len(self.msgs)):
            ref m = self.msgs[i]
            var n = len(m.lines) + 2
            if k + n > rng[0] and k < rng[1]:
                if rng[0] <= k < rng[1]:
                    _ = ui.text(vx + 12, ui.scroll_y(k), "You" if m.user else "Assistant", 0, t.muted if m.user else t.accent)
                for j in range(len(m.lines)):
                    var r = k + 1 + j
                    if rng[0] <= r < rng[1]:
                        _ = ui.text(vx + 12, ui.scroll_y(r), m.lines[j], 0, t.text)
            k += n
        ui.scroll_end()
        # input row: field, Send, Stop, Copy
        var fy = y1 - 68
        ui.x, ui.y = vx, fy
        _ = ui.field(ID_FIELD, self.input, 2000)
        if ui.field_focused and inp.key(SC_RETURN):
            self.submit()
        var bx = vx + 360 + 8
        ui.x, ui.y = bx, fy
        self.send_at = SIMD[DType.float32, 2](ui.x + 30, ui.y + 15)
        if ui.button(ID_SEND, "Send", 60, 30, on=not self.busy()):
            self.submit()
        self.stop_at = SIMD[DType.float32, 2](ui.x + 30, ui.y + 15)
        if ui.button(ID_STOP, "Stop", 60, 30, on=self.busy()):
            self.stop()
        self.copy_at = SIMD[DType.float32, 2](ui.x + 30, ui.y + 15)
        if ui.button(ID_COPY, "Copy", 60, 30):
            var a = self.last_answer()
            if a.byte_length() > 0:
                self.copied = a^
                self.note = "copied, " + String(self.copied.byte_length()) + " bytes" if clipboard_set(self.copied) else "copy: no window clipboard here"
        _ = ui.text(vx, y1 - 26, self.note if self.note else String("idle"), 0, t.muted)
