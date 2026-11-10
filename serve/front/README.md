# serve/front: the Mojo HTTP front

**State: in build, UNVERIFIED. `baro-serve` (Rust, `serve/src/`) is the default front.** Nothing
here is linked into it. This directory replaces the Rust front on the core path once the checks
in `docs/lanes/2026-11-09-mojo-front.md` (section Verification) pass.

Libraries are vendored at repo root, drift-checked by `tools/ci-checks.sh`: `bento/` (HTTP/1.1
server, SSE), `aio/` (epoll), `json/`, `posix/` (spawn, signals, sockets), `systemd/`. In-repo:
`serve/tokenizer.mojo`, `minja/`. Build: `./.venv/bin/mojo build FILE -I . -o .work/OUT` (Mojo 1.2.0).

## Frozen contract

Three modules, three owners. Each lane builds and tests against this text, not against another
lane's branch. A wrong contract is a QUESTION to the boss pane, never a silent change.

```
serve/front/text.mojo                                            lane front-text
  struct Text(Movable):
      __init__(out self, tokenizer_json: String, template_file: String = "") raises
      encode(self, s: String, add_special: Bool) raises -> List[Int32]
      decode(self, ids: Span[Int32, _]) -> String
      apply_chat_template(self, messages: Json, tools: Json, kwargs: Json) raises -> String
      tool_xml(self) -> Bool                    # template writes Qwen3.5 XML tool calls
      stop_ids(self) -> List[Int32]             # eos and template stops, as Rust compute_stop defaults
      detok(self) -> Detok                      # a fresh streaming detokenizer
      A first argument ending in .gguf is loaded directly; else the engine tokenizer is built
      from `source_gguf` of the sibling tokenizer-meta.json (named error when it is missing).
  struct Detok(Movable):                        # streaming detokenizer, one per request
      push(mut self, id: Int32) -> String       # text printable now; holds a partial UTF-8 tail
      flush(mut self) -> String
  role_boundaries(...), common_prefix(a, b): same results as serve/src/text.rs
  Missing value = Json null. Results equal the Rust `Text` byte for byte.

serve/front/engine_client.mojo                                   lane front-engine
  comptime ENG_STARTING = 0; ENG_IDLE = 1; ENG_BUSY = 2; ENG_DEAD = 3; ENG_STOPPING = 4
  comptime EV_TOKEN = 0; EV_DONE = 1; EV_ERROR = 2; EV_READY = 3; EV_DEAD = 4; EV_OTHER = 5
  struct Request(Copyable, Movable): the fields of serve/src/protocol.rs `Request`, same names
      line(self) -> String                      # the bytes Rust writes, pinned by a test
  struct Event(Copyable, Movable): kind: Int; id: Int; tok: Int32; text: String  # text = error or raw JSON
  struct EngineClient(Movable):
      __init__(out self, argv: List[String], env: List[String]) raises   # spawns, does not wait
          # child stdin+stdout = one AF_UNIX socketpair end (posix.spawn_pair, added upstream
          # in mojo-posix); the parent end is fd(); shutdown = shutdown_write(fd) then wait
      fd(self) -> Int32                         # child stdout, non-blocking, for an aio loop
      state(self) -> Int;  state_name(self) -> String;  dead_reason(self) -> String
      ready_info(self) -> Json                  # the ready line (tmax, mrows, kmax, spec_k, kv, pack)
      submit(mut self, req: Request) raises     # named error unless IDLE or BUSY
      cancel(mut self, id: Int) -> Bool
      poll(mut self, mut out: List[Event], timeout_ms: Int) raises -> Int   # reads lines, sees child exit
      shutdown(mut self, grace_ms: Int)         # closes stdin first, then waits, then kills
  One writer for the state: a private `_set(new, why)` that logs `engine: OLD -> NEW (why)`.
  Single-threaded: one owner thread calls every method.

serve/front/http.mojo + api.mojo                                 lane front-http
  trait DetokLike:  push / flush, signatures as Detok
  trait TextLike:   encode / decode / apply_chat_template / tool_xml / stop_ids, signatures as Text;
                    `comptime D: DetokLike` and `detok(self) -> Self.D`
  trait EngineLike: submit / cancel / poll / state / state_name / ready_info, signatures as EngineClient
  struct Front[T: TextLike, E: EngineLike] is a bento.server.Handler.
  Handlers run on many worker threads; the engine has ONE owner, a dispatcher that holds the
  EngineLike and gives each request its own event queue (bento/sync.mojo). No handler touches
  the engine.
  Routes: GET  /health /v1/models /models /props
          POST /tokenize /detokenize /v1/cancel
          POST /v1/completions /completions /v1/chat/completions /chat/completions (JSON and SSE)
```

## Six Rust defects the Mojo front does not copy

`exchange/fsm-audit/report-A.md` found state the Rust front keeps in loose flags. Each row is a
DELIBERATE DIFFERENCE: tested, and listed in the lane report. Everything else is parsed-value
parity with the Rust front.

| Rust today (report-A id) | Mojo front |
|---|---|
| M1: engine state inferred from 5 carriers; a death while idle is unseen; `/health` says ok with a dead engine | one `state()` tag; `poll` sees the child exit; `/health` returns `engine_state` and the exit reason |
| M1: stop waits 30 s because stdin is never closed | `shutdown` closes stdin first |
| wire: a ready line with a missing field waits 600 s | named error `ready line: missing <field>` |
| M2: a client disconnect does not cancel | disconnect mid-stream sends cancel; `/v1/cancel` answers `state: queued, running or unknown` |
| M3: a stream that ends with no terminal event returns a partial answer as success | `502 engine stream ended without a terminal event`; SSE sends an error chunk before `[DONE]` |
| M4: the length guard is repeated by hand in 4 places | one check at submit |

Not shown yet: that any of this holds against the real engine. The tests run against a stub.
