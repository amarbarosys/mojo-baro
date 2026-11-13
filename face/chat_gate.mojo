"""F1 gate driver: the chat pane, headless (no window, no display), driven the way a user drives it. Typed text goes in as
Input.text, Enter as the Return key, Stop as a mouse press and release on the button the pane drew. Frames go to PPM
files (face/chat-gate.sh converts and OCR-reads them).

  chat_gate --front HOST:PORT --out DIR --prompt TEXT [--prompt TEXT ...] [--timeout S]
      send each prompt in turn, wait for its answer, write DIR/chat-<n>-mid.ppm (first bytes streamed) and DIR/chat-<n>.ppm
      (answer complete), print "ANSWER <n> <note> | <text>"
  chat_gate ... --stop-after N      send the one prompt, click Stop once N bytes streamed, write DIR/chat-stop.ppm, print "STOP <note>"
  chat_gate ... --close-after N     send the one prompt, drop the Chat (what closing the window does) at N bytes, print "CLOSED"
"""
from std.os import getenv
from std.sys import argv
from std.time import sleep
from max.gpu.host import DeviceContext
from imkit.draw import UH, UW
from imkit.font import Font
from imkit.layer import Layer
from imkit.theme import amarbaro
from imkit.ui import Ui
from imkit.window import Input
from app import BG, SPLIT, write_ppm
from chat import Chat


def frame(mut ui: Ui, mut layer: Layer, mut chat: Chat, inp: Input) raises:
    ui.begin(inp)
    chat.draw(ui, inp, 20, 16, SPLIT, Float32(UH - 56))
    ui.finish()
    layer.render(ui)
    if ui.d.dropped:
        raise Error("FAIL chat_gate: " + String(ui.d.dropped) + " draw commands dropped")


def idle() -> Input:
    return Input(0, 0, 0, 0, 0, 0, 0, 0, SIMD[DType.uint64, 2](0), False)


def click(mut ui: Ui, mut layer: Layer, mut chat: Chat, at: SIMD[DType.float32, 2]) raises:
    var keys = SIMD[DType.uint64, 2](0)
    frame(ui, layer, chat, Input(at[0], at[1], 0, 0, 0, 1, 1, 0, keys, False))
    frame(ui, layer, chat, Input(at[0], at[1], 0, 0, 0, 0, 0, 1, keys, False))


def send(mut ui: Ui, mut layer: Layer, mut chat: Chat, prompt: String) raises:
    var keys = SIMD[DType.uint64, 2](0)
    frame(ui, layer, chat, Input(0, 0, 0, 0, 0, 0, 0, 0, keys, False, prompt))  # typed into the focused field
    if chat.input != prompt:
        raise Error("FAIL chat_gate: the field holds '" + chat.input + "', not the typed prompt")
    frame(ui, layer, chat, Input(0, 0, 0, 0, 0, 0, 0, 0, SIMD[DType.uint64, 2](UInt64(1) << 40, 0), False))  # Return
    if not chat.busy() and len(chat.msgs) == 0:
        raise Error("FAIL chat_gate: Return did not send: " + chat.note)


def main() raises:
    var host = String("127.0.0.1")
    var port = 8080
    var out = String("")
    var prompts = List[String]()
    var timeout_s = 180
    var stop_after = -1
    var close_after = -1
    var a = argv()
    var i = 1
    while i < len(a):
        var k = String(a[i])
        if k == "--front" and i + 1 < len(a):
            var hp = String(a[i + 1]).split(":")
            host = String(hp[0])
            port = Int(hp[1])
            i += 2
        elif k == "--out" and i + 1 < len(a):
            out = String(a[i + 1])
            i += 2
        elif k == "--prompt" and i + 1 < len(a):
            prompts.append(String(a[i + 1]))
            i += 2
        elif k == "--timeout" and i + 1 < len(a):
            timeout_s = Int(String(a[i + 1]))
            i += 2
        elif k == "--stop-after" and i + 1 < len(a):
            stop_after = Int(String(a[i + 1]))
            i += 2
        elif k == "--close-after" and i + 1 < len(a):
            close_after = Int(String(a[i + 1]))
            i += 2
        else:
            raise Error("FAIL usage: chat_gate --front H:P --out DIR --prompt TEXT [--prompt ...] [--timeout S] [--stop-after N | --close-after N]")
    if not out or len(prompts) == 0:
        raise Error("FAIL usage: --out DIR and at least one --prompt")
    var ui = Ui(Font(getenv("IMKIT_FONT", "/usr/share/fonts/TTF/Roboto-Medium.ttf"), [13, 16, 22]))
    ui.set_theme(amarbaro())
    var layer = Layer(DeviceContext(), ui)
    var bg = List[UInt32](length=UW * UH, fill=BG)
    var chat = Chat(host, port)
    for n in range(len(prompts)):
        send(ui, layer, chat, prompts[n])
        var mid = False
        var waited = 0
        while chat.busy():
            frame(ui, layer, chat, idle())
            if not mid and chat.streamed >= 20:
                mid = True
                write_ppm(out + "/chat-" + String(n + 1) + "-mid.ppm", layer, bg)
            if stop_after >= 0 and chat.streamed >= stop_after:
                click(ui, layer, chat, chat.stop_at)
                frame(ui, layer, chat, idle())
                write_ppm(out + "/chat-stop.ppm", layer, bg)
                print("STOP", chat.note, "| bytes", chat.streamed, "busy", chat.busy())
                return
            if close_after >= 0 and chat.streamed >= close_after:
                print("CLOSED at bytes", chat.streamed)
                return  # chat goes out of scope: the socket closes, as when the window closes
            sleep(0.02)
            waited += 20
            if waited > timeout_s * 1000:
                raise Error("FAIL chat_gate: no end of answer " + String(n + 1) + " after " + String(timeout_s) + " s, note: " + chat.note)
        for _ in range(3):
            frame(ui, layer, chat, idle())
        write_ppm(out + "/chat-" + String(n + 1) + ".ppm", layer, bg)
        print("ANSWER", n + 1, chat.note, "|", chat.last_answer())
    if stop_after >= 0 or close_after >= 0:
        raise Error("FAIL chat_gate: the answer ended before the stop point")
