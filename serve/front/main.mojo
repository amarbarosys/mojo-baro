"""baro-front: the Mojo HTTP front. Joins text.mojo, engine_client.mojo and http.mojo.

    baro-front ENGINE PACK [--host H] [--port N] [--workers N] [--template FILE]

ENGINE is the engine binary, PACK its pack directory (with tokenizer-meta.json). The engine child
gets BARO_SERVE=1 and BARO_PACK=PACK and inherits every other BARO_* knob of this process.
UNVERIFIED against the Rust front until docs/lanes/2026-11-09-mojo-front.md "Verification" passes.
"""
from std.os import setenv
from std.sys import argv
from api import TextLike, DetokLike, EngineLike, GenRequest, Ev
from bento.server import Server
from engine_client import EngineClient, Request, Event
from http import Front
from json import Json, parse
from posix import block_signals
from text import Text, Detok


struct DetokA(DetokLike):
    var d: Detok

    def __init__(out self, var d: Detok):
        self.d = d^

    def push(mut self, id: Int32) -> String:
        return self.d.push(id)

    def flush(mut self) -> String:
        return self.d.flush()


struct TextA(TextLike):
    comptime D = DetokA
    var t: Text

    def __init__(out self, var t: Text):
        self.t = t^

    def encode(self, s: String, add_special: Bool) raises -> List[Int32]:
        return self.t.encode(s, add_special)

    def decode(self, ids: Span[Int32, _]) -> String:
        return self.t.decode(ids)

    def apply_chat_template(self, messages: Json, tools: Json, kwargs: Json) raises -> String:
        return self.t.apply_chat_template(messages, tools, kwargs)

    def tool_xml(self) -> Bool:
        return self.t.tool_xml()

    def stop_ids(self) -> List[Int32]:
        return self.t.stop_ids()

    def detok(self) -> DetokA:
        return DetokA(self.t.detok())


def _f32(v: Optional[Float64]) -> Optional[Float32]:
    if v:
        return Optional[Float32](Float32(v.value()))
    return Optional[Float32]()


struct EngineA(EngineLike):
    var c: EngineClient
    var evs: List[Event]

    def __init__(out self, var c: EngineClient):
        self.c = c^
        self.evs = List[Event]()

    def submit(mut self, req: GenRequest) raises:
        var r = Request(req.id, req.prompt.copy(), req.n, req.spec)
        r.stop = req.stop.copy()
        r.ckpt = req.ckpt.copy()
        r.temperature = _f32(req.temperature)
        r.top_p = _f32(req.top_p)
        r.top_k = req.top_k
        r.min_p = _f32(req.min_p)
        r.seed = req.seed
        r.presence_penalty = _f32(req.presence_penalty)
        r.frequency_penalty = _f32(req.frequency_penalty)
        if req.schema:
            r.schema = parse(req.schema.value())
        r.reasoning = req.reasoning
        self.c.submit(r)

    def cancel(mut self, id: Int) -> Bool:
        return self.c.cancel(id)

    def poll(mut self, mut out: List[Ev], timeout_ms: Int) raises -> Int:
        self.evs.clear()
        var n = self.c.poll(self.evs, timeout_ms)
        for e in self.evs:
            out.append(Ev(e.kind, e.id, e.tok, e.text))
        return n

    def state(self) -> Int:
        return self.c.state()

    def state_name(self) -> String:
        return self.c.state_name()

    def dead_reason(self) -> String:
        return self.c.dead_reason()

    def shutdown(mut self, grace_ms: Int):
        self.c.shutdown(grace_ms)

    def ready_info(self) -> Json:
        return self.c.ready_info()


def _pack_name(pack: String) -> String:
    var parts = pack.split("/")
    var i = len(parts) - 1
    while i > 0 and parts[i].byte_length() == 0:
        i -= 1
    return String(parts[i])


def main() raises:
    block_signals()  # before any thread: SIGTERM must reach the server's signal fd, not kill a worker
    var a = argv()
    if len(a) < 3:
        raise Error("FAIL usage: baro-front ENGINE PACK [--host H] [--port N] [--workers N] [--template FILE]")
    var engine = String(a[1])
    var pack = String(a[2])
    var host = String("127.0.0.1")
    var port = 8080
    var workers = 8
    var template = String("")
    var i = 3
    while i + 1 < len(a):
        var k = String(a[i])
        var v = String(a[i + 1])
        if k == "--host":
            host = v
        elif k == "--port":
            port = Int(v)
        elif k == "--workers":
            workers = Int(v)
        elif k == "--template":
            template = v
        else:
            raise Error("FAIL usage: unknown option " + k)
        i += 2
    if i < len(a):
        raise Error("FAIL usage: option without a value: " + String(a[i]))
    # Same engine defaults as the Rust front (serve/src/main.rs `defaults`), only where unset.
    for kv in [("BARO_MOE_DOT", "1"), ("BARO_TIER_ZC", "1"), ("BARO_TIER_PINNED", "1"), ("BARO_PREFILL", "1"), ("BARO_SPEC", "1"), ("BARO_TMAX", "32768")]:
        _ = setenv(kv[0], kv[1], overwrite=False)
    _ = setenv("BARO_SERVE", "1")
    _ = setenv("BARO_PACK", pack)
    var text = TextA(Text(pack + "/tokenizer.json", template))
    var argv_child = List[String]()
    argv_child.append(engine)
    var front = Front[TextA, EngineA](text^, EngineA(EngineClient(argv_child, List[String]())), _pack_name(pack))
    var srv = Server[Front[TextA, EngineA]](front^, host, port)
    print("listening on http://" + host + ":" + String(srv.port()), "engine", engine, "pack", pack, "workers", workers, flush=True)
    srv.serve(workers, True, 2000)
