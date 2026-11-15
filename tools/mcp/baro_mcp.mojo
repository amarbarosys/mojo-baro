"""baro-mcp: MCP server on stdio (newline-delimited JSON-RPC 2.0) for operating mojo-baro. Tools read the repo's
docs/settings.toml and docs/operate-capabilities.toml at each call and proxy the running front's GET endpoints.
Template: mojo-neuron/src/mcp.mojo. Errors of a call come back as isError results, not protocol errors.
Build: mojo build tools/mcp/baro_mcp.mojo -I . -I face -I tools/mcp -o .work/baro-mcp
Env: BARO_MCP_URL (default http://127.0.0.1:8080), BARO_MCP_REPO (default: two levels above the binary)."""
from std.os import getenv
from std.os.path import realpath
from std.ffi import external_call
from json import Json, parse as json_parse
from toml import parse as toml_parse, TomlValue
from net import Reply, http_get
from std.os.path import exists
from std.subprocess import run
from http import http_post
from stdin import read_line

comptime SUPPORTED = ["2024-11-05", "2025-03-26", "2025-06-18"]  # tools/list and tools/call only: identical in all three
comptime S_KEY = '"key":{"type":"string","description":"settings key, e.g. BARO_TMAX or sampling.temperature"}'
comptime S_NAME = '"name":{"type":"string","description":"settings key, e.g. BARO_TMAX or sampling.temperature"}'
comptime S_CAP = '"id":{"type":"string","description":"capability id, e.g. kvq, spec, tmax"}'


def tool(name: String, desc: String, props: String, req: String) raises -> Json:
    return json_parse('{"name":"' + name + '","description":"' + desc + '","inputSchema":{"type":"object","properties":{' + props + '},"required":[' + req + ']}}')


def tools() raises -> Json:
    var a = Json.array()
    a.push(tool("status", "Health, props, models and the first 40 lines of /metrics of the running front. Reads limits.tmax (BARO_TMAX), pack (BARO_PACK) and pool (BARO_POOL) from /health. A non-200 health is a normal result; an unreachable front is isError.", "", ""))
    a.push(tool("models", "The data array of GET /v1/models: the chat model and the embedding model the front serves.", "", ""))
    a.push(tool("settings", "Settings rows from docs/settings.toml, filtered by group, level and kind; each row names its key (for example BARO_TMAX, BARO_KVQ, sampling.temperature). running_env lists the BARO_* and MODULAR_* variables of the engine process from .work/gui-front.pid.",
        '"group":{"type":"string","enum":["sampling","engine","model","front"]},"level":{"type":"string","enum":["basic","advanced"]},"kind":{"type":"string"}', ""))
    a.push(tool("capabilities", "Rows of docs/operate-capabilities.toml: enable, default, best, status, gui, check, source and the settings keys each touches. One row when id is given.",
        S_CAP, ""))
    a.push(tool("explain_setting", "Full row of one setting from docs/settings.toml, plus the capability ids whose settings list names it (for example BARO_TMAX lists tmax). An unknown key is isError with similar keys.",
        S_NAME, '"name"'))
    a.push(tool("chat", "One chat completion from the running front (POST /v1/chat/completions, stream false). Give exactly one of prompt or messages. Sends neurons false unless given (the front may default BARO_NEURONS=1, which adds a brain_ask loop). Sends sampling.max_tokens (default 16000), sampling.temperature, sampling.top_p, sampling.top_k and seed when given. Returns content, finish_reason, usage, timings and health before and after. A non-200 answer (for example HTTP 502 engine_invalid_tokens) is isError.",
        '"prompt":{"type":"string"},"messages":{"type":"array"},"max_tokens":{"type":"integer"},"temperature":{"type":"number"},"top_p":{"type":"number"},"top_k":{"type":"integer"},"seed":{"type":"integer"},"neurons":{"type":"boolean"}', ""))
    a.push(tool("embed", "Embeddings from POST /v1/embeddings (embedding model, see the models tool). input is a string or an array of strings. Returns count, dims, first4 per input and health before and after; full true returns the whole vectors. An unknown model name (HTTP 404) is isError.",
        '"input":{"type":"string"},"full":{"type":"boolean"}', '"input"'))
    a.push(tool("checkpoint_list", "The checkpoints of the front (GET /v1/checkpoints) with health before and after.", "", ""))
    a.push(tool("checkpoint_fork", "Fork checkpoint id into branches (POST /v1/checkpoints/{id}/fork, 1 to 8 branches, identity checked by the server). Returns the fork answer, checkpoint ids before and after, and health before and after. id must match [A-Za-z0-9_-]{1,64}.",
        '"id":{"type":"string"},"branches":{"type":"integer"},"identity":{"type":"string"}', '"id"'))
    a.push(tool("checkpoint_export", "Export checkpoint id as a .baro file. Not supported: the MCP HTTP client reads text, and the binary body would be corrupted. Always isError binary download not supported.",
        '"id":{"type":"string"}', '"id"'))
    a.push(tool("checkpoint_import", "Import a .baro file from <repo>/.work/mcp-export/ by name (NAME.baro, name matches [A-Za-z0-9_-]{1,64}). Not supported: the server takes a multipart upload (meta and file) that this client cannot send safely, so the call returns an error saying so.",
        '"name":{"type":"string"}', '"name"'))
    a.push(tool("set_and_restart", "Set engine settings and restart the owned engine (tools/baro serve started by the GUI). Only env and server-env keys (for example BARO_TMAX, BARO_SPEC, BARO_KVQ is refused: build flag). Validates every key and value first, stops the owned engine with face/stop-engine.sh, starts the new one, waits up to 180 s for /health, and reports took_effect per key from running_env. Keys: the settings keys in docs/settings.toml.",
        '"settings":{"type":"object"},"model":{"type":"string"},"port":{"type":"integer"}', '"settings"'))
    a.push(tool("preset", "A [[preset]] of docs/settings.toml by name (Fast, Balanced, Quality, Chat, Code, Reasoning, Low VRAM, Long context). Without apply: lists restart_env (BARO_* keys it sets), request_fields (sampling.* and chat_template_kwargs.* for the client) and refused (build-flag keys, for example BARO_KVQ). apply true restarts the owned engine with the restart_env keys.",
        '"name":{"type":"string"},"apply":{"type":"boolean"}', '"name"'))
    return a^


def reply(id: Json, result: Json):
    var o = Json.object()
    o.set("jsonrpc", Json.of(String("2.0")))
    o.set("id", id)
    o.set("result", result)
    print(o.dumps())


def fail(id: Json, code: Int, msg: String):
    var e = Json.object()
    e.set("code", Json.of(code))
    e.set("message", Json.of(msg))
    var o = Json.object()
    o.set("jsonrpc", Json.of(String("2.0")))
    o.set("id", id)
    o.set("error", e)
    print(o.dumps())


def _parent(p: String) -> String:
    var i = p.rfind("/")
    return String(p[byte=0:i]) if i > 0 else String("/")


def repo_root() raises -> String:
    var r = getenv("BARO_MCP_REPO")
    if r.byte_length() > 0:
        return r
    # the binary is <repo>/.work/baro-mcp: its directory is .work, one level above it is the repo
    return _parent(_parent(realpath("/proc/self/exe")))


def endpoint() raises -> Tuple[String, Int]:
    var s = getenv("BARO_MCP_URL", "http://127.0.0.1:8080")
    if s.startswith("http://"):
        var rest = String(s[byte=7:])
        s = rest^
    var c = s.find(":")
    if c < 0:
        return (String(s), 80)
    return (String(s[byte=0:c]), Int(String(s[byte=c + 1:])))


def read_file(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def tj(v: TomlValue) raises -> Json:
    if v.is_string():
        return Json.of(v.as_string())
    if v.is_bool():
        return Json.of(v.as_bool())
    if v.is_int():
        return Json.of(v.as_int())
    if v.is_float():
        return Json.of(v.as_float())
    if v.is_array():
        var a = Json.array()
        for x in v.as_array():
            a.push(tj(x))
        return a^
    if v.is_table():
        return tbl(v.as_table())
    return Json()


def tbl(t: Dict[String, TomlValue]) raises -> Json:
    var o = Json.object()
    for k in t.keys():
        o.set(String(k), tj(t[String(k)]))
    return o^


def rows(section: String, file: String) raises -> List[Dict[String, TomlValue]]:
    var root = toml_parse(read_file(repo_root() + "/docs/" + file))
    var out = List[Dict[String, TomlValue]]()
    if section in root:
        for e in root[section].as_array():
            out.append(e.as_table())
    return out^


def str_of(t: Dict[String, TomlValue], k: String) raises -> String:
    return t[k].as_string() if k in t and t[k].is_string() else String("")


def body_json(body: String) -> Json:
    try:
        return json_parse(body)
    except:
        return Json.of(body)


def must_reach(r: Reply) raises:
    if r.status == 0:
        raise Error("unreachable: " + r.why)


def status_tool() raises -> Json:
    var ep = endpoint()
    var h = http_get(ep[0], ep[1], "/health", 800)
    must_reach(h)
    var p = http_get(ep[0], ep[1], "/props", 800)
    must_reach(p)
    var m = http_get(ep[0], ep[1], "/v1/models", 800)
    must_reach(m)
    var mt = http_get(ep[0], ep[1], "/metrics", 800)
    var o = Json.object()
    o.set("health_code", Json.of(h.status))
    o.set("health", body_json(h.body))
    o.set("props", body_json(p.body))
    o.set("models", body_json(m.body))
    if mt.status == 200:
        var head = String("")
        var n = 0
        for ln in mt.body.split("\n"):
            if n >= 40:
                break
            head += String(ln) + "\n"
            n += 1
        o.set("metrics_head", Json.of(head))
    return o^


def models_tool() raises -> Json:
    var ep = endpoint()
    var m = http_get(ep[0], ep[1], "/v1/models", 800)
    must_reach(m)
    return json_parse(m.body).get("data")


def running_env() raises -> Json:
    var o = Json.object()
    var pidpath = repo_root() + "/.work/gui-front.pid"
    try:
        var pid = String(read_file(pidpath).strip())
        var env = read_file("/proc/" + pid + "/environ")
        for part in env.split("\0"):
            var s = String(part)
            var eq = s.find("=")
            if eq <= 0:
                continue
            var name = String(s[byte=0:eq])
            if name.startswith("BARO_") or name.startswith("MODULAR_"):
                o.set(name, Json.of(String(s[byte=eq + 1:])))
    except:
        pass  # no pidfile, or the process is gone: empty object
    return o^


def settings_tool(args: Json) raises -> Json:
    var group = args.text("group")
    var level = args.text("level")
    var kind = args.text("kind")
    var a = Json.array()
    for t in rows("setting", "settings.toml"):
        if group != "" and str_of(t, "group") != group:
            continue
        if level != "" and str_of(t, "level") != level:
            continue
        if kind != "" and str_of(t, "kind") != kind:
            continue
        var r = Json.object()
        for k in ["key", "kind", "restart", "status", "level", "summary", "default"]:
            if k in t:
                r.set(k, tj(t[k]))
        a.push(r^)
    var o = Json.object()
    o.set("settings", a^)
    o.set("running_env", running_env())
    return o^


def capabilities_tool(args: Json) raises -> Json:
    var want = args.text("id")
    var a = Json.array()
    var found = False
    for c in rows("capability", "operate-capabilities.toml"):
        if want != "" and str_of(c, "id") != want:
            continue
        found = found or want != ""
        a.push(tbl(c))
    if want != "" and not found:
        raise Error("unknown capability: " + want)
    return a^


def explain_tool(args: Json) raises -> Json:
    var name = args.text("name")
    if name == "":
        raise Error("explain_setting needs name")
    var hits = List[String]()
    for t in rows("setting", "settings.toml"):
        var key = str_of(t, "key")
        if key == name:
            var o = tbl(t)
            var caps = Json.array()
            for c in rows("capability", "operate-capabilities.toml"):
                if "settings" in c:
                    for x in c["settings"].as_array():
                        if x.as_string() == name:
                            caps.push(Json.of(str_of(c, "id")))
            o.set("capabilities", caps^)
            return o^
        if key.find(name) >= 0 and len(hits) < 5:
            hits.append(key)
    var msg = "unknown setting: " + name
    if len(hits) > 0:
        msg += "; similar: " + ", ".join(hits)
    raise Error(msg)


def valid_name(s: String, suffix: String) raises -> Bool:
    """id: [A-Za-z0-9_-]{1,64}. With suffix (".baro") the name must end in it and the part before must be an id."""
    var body = s.copy()
    if suffix.byte_length() > 0:
        if not s.endswith(suffix):
            return False
        body = String(s[byte=0:s.byte_length() - suffix.byte_length()])
    var n = body.byte_length()
    if n < 1 or n > 64:
        return False
    var b = body.as_bytes()
    for i in range(n):
        var c = Int(b[i])
        var ok = (c >= 48 and c <= 57) or (c >= 65 and c <= 90) or (c >= 97 and c <= 122) or c == 95 or c == 45
        if not ok:
            return False
    return True


def health_state() raises -> Json:
    var ep = endpoint()
    var h = http_get(ep[0], ep[1], "/health", 800)
    must_reach(h)
    var j = body_json(h.body)
    var o = Json.object()
    o.set("health_code", Json.of(h.status))
    o.set("engine_state", Json.of(j.text("engine_state")))
    o.set("queue", j.get("queue"))
    return o^


def ids_of(body: Json) raises -> Json:
    var a = Json.array()
    var arr = body.get("checkpoints") if body.has("checkpoints") else body.copy()
    for c in arr.items():
        a.push(Json.of(c.text("id")))
    return a^


def list_ids() raises -> Json:
    var ep = endpoint()
    var r = http_get(ep[0], ep[1], "/v1/checkpoints", 800)
    must_reach(r)
    if r.status != 200:
        raise Error("HTTP " + String(r.status) + ": " + r.body)
    return ids_of(body_json(r.body))


def post_ok(path: String, body: String) raises -> Json:
    var ep = endpoint()
    var r = http_post(ep[0], ep[1], path, body, 60000)
    must_reach(r)
    if r.status != 200:
        raise Error("HTTP " + String(r.status) + ": " + r.body)
    return json_parse(r.body)


def chat_tool(args: Json) raises -> Json:
    var has_p = args.has("prompt")
    var has_m = args.has("messages")
    if has_p == has_m:
        raise Error("give exactly one of prompt or messages")
    var msgs = Json.array()
    if has_p:
        var one = Json.object()
        one.set("role", Json.of(String("user")))
        one.set("content", args.get("prompt"))
        msgs.push(one^)
    else:
        msgs = args.get("messages")
    var body = Json.object()
    body.set("messages", msgs^)
    body.set("stream", Json.of(False))
    body.set("neurons", args.get("neurons") if args.has("neurons") else Json.of(False))  # tools/baro serve defaults BARO_NEURONS=1: no brain_ask loop unless asked
    body.set("max_tokens", args.get("max_tokens") if args.has("max_tokens") else Json.of(16000))
    for k in ["temperature", "top_p", "top_k", "seed"]:
        if args.has(k):
            body.set(k, args.get(k))
    var before = health_state()
    try:
        var j = post_ok("/v1/chat/completions", body.dumps())
        var c = j.get("choices").items()[0].copy()
        var o = Json.object()
        o.set("content", Json.of(c.get("message").text("content")))
        o.set("finish_reason", Json.of(c.text("finish_reason")))
        o.set("usage", j.get("usage"))
        o.set("timings", j.get("timings"))
        o.set("before", before.copy())
        o.set("after", health_state())
        return o^
    except e:
        raise Error(String(e) + " before=" + before.dumps())


def embed_tool(args: Json) raises -> Json:
    var body = Json.object()
    body.set("input", args.get("input"))
    var before = health_state()
    try:
        var j = post_ok("/v1/embeddings", body.dumps())
        var data = j.get("data").items()
        var o = Json.object()
        o.set("count", Json.of(len(data)))
        o.set("dims", Json.of(data[0].get("embedding").len() if len(data) > 0 else 0))
        var first = Json.array()
        var full = Json.array()
        for d in data:
            var e4 = Json.array()
            var emb = d.get("embedding").items()
            for i in range(min(4, len(emb))):
                e4.push(emb[i].copy())
            first.push(e4^)
            if args.has("full") and args.get("full").dumps() == "true":
                full.push(d.get("embedding"))
        o.set("first4", first^)
        if args.has("full"):
            o.set("vectors", full^)
        o.set("before", before.copy())
        o.set("after", health_state())
        return o^
    except e:
        raise Error(String(e) + " before=" + before.dumps())


def checkpoint_list_tool() raises -> Json:
    var before = health_state()
    var ep = endpoint()
    var r = http_get(ep[0], ep[1], "/v1/checkpoints", 800)
    must_reach(r)
    if r.status != 200:
        raise Error("HTTP " + String(r.status) + ": " + r.body + " before=" + before.dumps())
    var o = Json.object()
    o.set("checkpoints", body_json(r.body))
    o.set("before", before.copy())
    o.set("after", health_state())
    return o^


def checkpoint_fork_tool(args: Json) raises -> Json:
    var id = args.text("id")
    if not valid_name(id, ""):
        raise Error("bad id")
    var n = 1
    if args.has("branches"):
        n = args.get("branches").int()
    if n < 1 or n > 8:
        raise Error("branches must be 1 to 8")
    var body = Json.object()
    var br = Json.array()
    for _ in range(n):
        br.push(Json.object())
    body.set("branches", br^)
    if args.has("identity"):
        body.set("identity", args.get("identity"))
    var before = health_state()
    try:
        var ids_before = list_ids()
        var j = post_ok("/v1/checkpoints/" + id + "/fork", body.dumps())
        var o = Json.object()
        o.set("fork", j^)
        o.set("ids_before", ids_before^)
        o.set("ids_after", list_ids())
        o.set("before", before.copy())
        o.set("after", health_state())
        return o^
    except e:
        raise Error(String(e) + " before=" + before.dumps())


def checkpoint_export_tool(args: Json) raises -> Json:
    if not valid_name(args.text("id"), ""):
        raise Error("bad id")
    # Not supported: read_reply (face/net.mojo) builds the body with String(unsafe_from_utf8=...), so bytes at or
    # above 0x80 of the .baro file would be replaced. Refuse instead of writing a corrupted file.
    raise Error("binary download not supported")


def checkpoint_import_tool(args: Json) raises -> Json:
    if not valid_name(args.text("name"), ".baro"):
        raise Error("bad name")
    # POST /v1/checkpoints/import takes multipart parts meta and file (serve/src/turn.rs:173); the String-based client
    # cannot send the binary file part safely.
    raise Error("binary upload not supported: /v1/checkpoints/import takes multipart meta and file (serve/src/turn.rs:173)")


def object_keys(text: String) raises -> List[String]:
    """Keys at depth 1 of a compact JSON object text (the dumps of an args object)."""
    var keys = List[String]()
    var b = text.as_bytes()
    var n = text.byte_length()
    var depth = 0
    var i = 0
    while i < n:
        var c = Int(b[i])
        if c == 123 or c == 91:
            depth += 1
        elif c == 125 or c == 93:
            depth -= 1
        elif c == 34:
            var start = i + 1
            i += 1
            while i < n and Int(b[i]) != 34:
                if Int(b[i]) == 92:
                    i += 1
                i += 1
            if depth == 1 and i + 1 < n and Int(b[i + 1]) == 58:
                keys.append(String(text[byte=start:i]))
        i += 1
    return keys^


def find_key(rs: List[Dict[String, TomlValue]], key: String) raises -> Int:
    for i in range(len(rs)):
        if str_of(rs[i], "key") == key:
            return i
    return -1


def num_of(v: TomlValue) raises -> Float64:
    return Float64(v.as_int()) if v.is_int() else v.as_float()


def toml_text(v: TomlValue) raises -> String:
    if v.is_bool():
        return String("1") if v.as_bool() else String("0")
    if v.is_string():
        return v.as_string()
    if v.is_int():
        return String(v.as_int())
    return String(v.as_float())


def norm_value(row: Dict[String, TomlValue], key: String, raw: String) raises -> String:
    """Validation of one setting value, in the order of brief S7: kind, charset, type, range, floor."""
    var kind = str_of(row, "kind")
    if kind == "build":
        raise Error("build flag: " + key + " is set by a profile [build] section (tools/baro -D flags); an environment variable does not change it. Use the profile route.")
    if kind == "shell" or kind == "external":
        raise Error("shell or external setting: " + key + " is not changed by an environment variable")
    if kind == "request":
        raise Error(key + " is a request field: send it in the chat request")
    if kind != "env" and kind != "server-env":
        raise Error("unsupported kind for " + key + ": " + kind)
    var n = raw.byte_length()
    if n < 1 or n > 200:
        raise Error("bad value length for " + key)
    var b = raw.as_bytes()
    for i in range(n):
        var c = Int(b[i])
        var ok = (c >= 48 and c <= 57) or (c >= 65 and c <= 90) or (c >= 97 and c <= 122) or c == 95 or c == 46 or c == 47 or c == 58 or c == 43 or c == 45
        if not ok:
            raise Error("bad characters in value for " + key)
    var t = str_of(row, "type")
    if t == "bool":
        if raw == "true" or raw == "1":
            return String("1")
        if raw == "false" or raw == "0":
            return String("0")
        raise Error("bad bool value for " + key)
    if t == "int" or t == "float":
        var x: Float64
        try:
            x = Float64(raw)
        except:
            raise Error("not a number for " + key + ": " + raw)
        if t == "int" and raw.find(".") >= 0:
            raise Error("not an integer for " + key + ": " + raw)
        if key == "BARO_TMAX" and x < 16384.0:
            raise Error("BARO_TMAX below 16384 is refused (repo floor, bench/min16k.sh)")
        if "min" in row and x < num_of(row["min"]):
            raise Error("below min for " + key)
        if "max" in row and x > num_of(row["max"]):
            raise Error("above max for " + key)
    if t == "enum":
        var ok = False
        for c in row["choices"].as_array():
            if c.as_string() == raw:
                ok = True
        if not ok:
            raise Error("not one of the choices for " + key)
    return raw


def collect(args_settings: Json) raises -> List[Tuple[String, String]]:
    var rs = rows("setting", "settings.toml")
    var out = List[Tuple[String, String]]()
    for key in object_keys(args_settings.dumps()):
        var idx = find_key(rs, key)
        if idx < 0:
            raise Error("unknown setting: " + key)
        var raw = args_settings.get(key).dumps()
        if raw.startswith('"'):
            var inner = String(raw[byte=1:raw.byte_length() - 1])
            raw = inner^
        out.append((key, norm_value(rs[idx], key, raw)))
    return out^


def owned_pid() -> String:
    try:
        return String(read_file(repo_root() + "/.work/gui-front.pid").strip())
    except:
        return String("")


def owned_model(pid: String) raises -> String:
    """The argument after `serve` in the owned engine's cmdline (NUL separated)."""
    var parts = read_file("/proc/" + pid + "/cmdline").split("\0")
    for i in range(len(parts) - 1):
        if String(parts[i]) == "serve":
            return String(parts[i + 1])
    return String("")


def restart_with(pairs: List[Tuple[String, String]], model_arg: String, port_arg: Int) raises -> Json:
    var repo = repo_root()
    var pid = owned_pid()
    var model = model_arg.copy()
    if model == "":
        if pid == "":
            raise Error("no owned engine and no model given")
        model = owned_model(pid)
    if model == "":
        raise Error("no owned engine and no model given")
    for i in range(model.byte_length()):
        var c = Int(model.as_bytes()[i])
        var ok = (c >= 48 and c <= 57) or (c >= 65 and c <= 90) or (c >= 97 and c <= 122) or c == 95 or c == 46 or c == 47 or c == 58 or c == 43 or c == 45
        if not ok:
            raise Error("bad characters in model path")
    if not exists(model):
        raise Error("model path does not exist: " + model)
    var ep = endpoint()
    var port = port_arg if port_arg > 0 else ep[1]
    var before = Json.object()
    try:
        before.set("health", health_state())
    except:
        before.set("health", Json.of(String("no engine answering")))  # a cold start has nothing to snapshot
    before.set("running_env", running_env())
    if pid != "":
        var stop = run("cd '" + repo + "' && bash face/stop-engine.sh; echo rc=$?")
        if not String(stop).strip().endswith("rc=0"):
            raise Error("stop failed: " + String(stop).strip())
    var env = String("")
    for p in pairs:
        env += p[0] + "='" + p[1] + "' "
    # setsid may fork, so $! can be a dead parent: the shell writes its own $$, then execs the engine (same pid)
    _ = run("cd '" + repo + "' && (setsid sh -c 'echo $$ > .work/gui-front.pid; exec env " + env + "tools/baro serve '\\''"
            + model + "'\\'' --port " + String(port) + "' > .work/gui-front.log 2>&1 < /dev/null &)")
    var up = False
    for tick in range(180):
        var r = http_get(ep[0], ep[1], "/health", 800)
        if r.status == 200:
            up = True
            break
        var live = owned_pid()
        # the shell writes the pid file a moment after launch: no pid yet is a death only after 5 ticks
        if (live == "" and tick >= 5) or (live != "" and not exists("/proc/" + live)):
            var tail = run("tail -n 3 '" + repo + "/.work/gui-front.log' 2>&1")
            raise Error("engine exited before /health: " + String(tail).strip())
        _ = run("sleep 1")
    if not up:
        raise Error("engine did not answer /health within 180 s")
    var after = Json.object()
    after.set("health", health_state())
    var new_env = running_env()
    after.set("running_env", new_env.copy())
    var took = Json.object()
    var note = String("")
    for p in pairs:
        var got = new_env.text(p[0])
        took.set(p[0], Json.of(got == p[1]))
        if got != p[1]:
            note += p[0] + " not in running_env of the new engine; "
    var o = Json.object()
    o.set("before", before^)
    o.set("after", after^)
    o.set("took_effect", took^)
    o.set("restarted", Json.of(True))
    if note != "":
        o.set("note", Json.of(note))
    return o^


def set_and_restart_tool(args: Json) raises -> Json:
    if not args.has("settings"):
        raise Error("settings is required")
    var pairs = collect(args.get("settings"))
    return restart_with(pairs, args.text("model"), Int(args.get("port").int()) if args.has("port") else 0)


def preset_tool(args: Json) raises -> Json:
    var name = args.text("name")
    var ps = rows("preset", "settings.toml")
    var names = List[String]()
    var idx = -1
    for i in range(len(ps)):
        names.append(str_of(ps[i], "name"))
        if names[i] == name:
            idx = i
    if idx < 0:
        raise Error("unknown preset: " + name + "; names: " + ", ".join(names))
    var rs = rows("setting", "settings.toml")
    var vals = ps[idx]["values"].as_table()
    var restart_env = Json.object()
    var req = Json.object()
    var refused = Json.array()
    var pairs = List[Tuple[String, String]]()
    for k in vals.keys():
        var key = String(k)
        var v = vals[key].copy()
        var ri = find_key(rs, key)
        if ri < 0:
            var r0 = Json.object()
            r0.set("key", Json.of(key))
            r0.set("reason", Json.of(String("not a setting in docs/settings.toml")))
            refused.push(r0^)
            continue
        var kind = str_of(rs[ri], "kind")
        if kind == "env" or kind == "server-env":
            var text = norm_value(rs[ri], key, toml_text(v))
            restart_env.set(key, Json.of(text))
            pairs.append((key, text))
        elif kind == "request":
            req.set(key, tj(v))
        else:
            var r1 = Json.object()
            r1.set("key", Json.of(key))
            r1.set("reason", Json.of(String("kind " + kind + ": not an environment variable; a build flag needs the profile route")))
            refused.push(r1^)
    var o = Json.object()
    o.set("name", Json.of(name))
    o.set("summary", Json.of(str_of(ps[idx], "summary")))
    o.set("restart_env", restart_env^)
    o.set("request_fields", req^)
    o.set("refused", refused^)
    if args.has("apply") and args.get("apply").dumps() == "true":
        o.set("result", restart_with(pairs, String(""), 0))
    return o^


def call(name: String, args: Json) raises -> Json:
    if name == "status":
        return status_tool()
    if name == "models":
        return models_tool()
    if name == "settings":
        return settings_tool(args)
    if name == "capabilities":
        return capabilities_tool(args)
    if name == "explain_setting":
        return explain_tool(args)
    if name == "chat":
        return chat_tool(args)
    if name == "embed":
        return embed_tool(args)
    if name == "checkpoint_list":
        return checkpoint_list_tool()
    if name == "checkpoint_fork":
        return checkpoint_fork_tool(args)
    if name == "checkpoint_export":
        return checkpoint_export_tool(args)
    if name == "checkpoint_import":
        return checkpoint_import_tool(args)
    if name == "set_and_restart":
        return set_and_restart_tool(args)
    if name == "preset":
        return preset_tool(args)
    raise Error("unknown tool: " + name)


def handle(line: String) raises:
    var m = json_parse(line)
    var method = m.text("method")
    var id = m.get("id")
    if not m.has("id"):  # a notification: no answer
        return
    if method == "initialize":
        var r = Json.object()
        var want = m.get("params").text("protocolVersion", "")
        var ver = String("2025-06-18")  # newest we speak; a client asking for anything else is told this one
        var sup = materialize[SUPPORTED]()
        for i in range(len(sup)):
            if want == sup[i]:
                ver = want
        r.set("protocolVersion", Json.of(ver))
        var caps = Json.object()
        caps.set("tools", Json.object())
        r.set("capabilities", caps)
        var info = Json.object()
        info.set("name", Json.of(String("baro-mcp")))
        info.set("version", Json.of(String("0.1.0")))
        r.set("serverInfo", info)
        reply(id, r)
    elif method == "ping":
        reply(id, Json.object())
    elif method == "tools/list":
        var r = Json.object()
        r.set("tools", tools())
        reply(id, r)
    elif method == "tools/call":
        var p = m.get("params")
        var r = Json.object()
        var content = Json.array()
        var item = Json.object()
        item.set("type", Json.of(String("text")))
        try:
            var args = p.get("arguments")
            item.set("text", Json.of(call(p.text("name"), args).dumps()))
            r.set("isError", Json.of(False))
        except e:
            item.set("text", Json.of(String(e)))
            r.set("isError", Json.of(True))
        content.push(item)
        r.set("content", content)
        reply(id, r)
    else:
        fail(id, -32601, "method not found: " + method)


def main() raises:
    var eof = False
    while not eof:
        var line = read_line(eof)
        if line.byte_length() == 0:
            continue
        try:
            handle(line)
        except e:
            fail(Json(), -32700, "parse error: " + String(e))
