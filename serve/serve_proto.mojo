"""Generic line protocol helpers, shared between every engine binary that
speaks serve/PROTOCOL.md over stdin/stdout (BARO_SERVE=1): the byte-scanner
JSON reader, the request line parser, and the sampler-field struct it fills.

Extracted verbatim out of serve/engine.mojo (M4, briefs/2026-09-15-wiring-lane.md):
this file carried none of qwen35's model logic, so moving it here and having
engine.mojo import it instead of defining it locally is a refactor, not a
rewrite -- qwen35/qwen35moe behavior is unchanged. serve/spark.mojo (the
dense-family engine) imports the same functions rather than re-implementing
the wire format, per serve/PROTOCOL.md's own instruction to reuse the
existing protocol.
"""
from std.ffi import c_ssize_t, external_call


def read_line(fd: Int) raises -> Optional[String]:
    # One line from fd, newline stripped; None at EOF with nothing buffered.
    var buf = List[UInt8]()
    var b = List[UInt8](unsafe_uninit_length=1)
    while True:
        var n = external_call["__read", c_ssize_t](fd, b.unsafe_ptr(), 1)
        if n <= 0:
            if len(buf) == 0:
                return None
            break
        if b[0] == 10:
            break
        buf.append(b[0])
    return String(from_utf8=Span[UInt8](buf))


def cancel_pending(fd: Int, req_id: Int, mut pending: List[String]) raises -> Bool:
    # Non-blocking check for a "{"cancel":ID}" line on fd, once per
    # step_window call. poll(fd, POLLIN, 0) never blocks; a hit means the
    # writer's single write() of a short line is already in the pipe, so the
    # blocking read_line below will not stall on a torn line.
    var pfd = List[UInt8](unsafe_uninit_length=8)
    var p = pfd.unsafe_ptr()
    p.unsafe_bitcast[Int32]().unsafe_offset(0)[] = Int32(fd)
    p.unsafe_bitcast[Int16]().unsafe_offset(2)[] = Int16(1)  # events = POLLIN
    p.unsafe_bitcast[Int16]().unsafe_offset(3)[] = Int16(0)  # revents
    var r = external_call["poll", Int32](p, UInt64(1), Int32(0))
    if r <= 0 or (Int(p.unsafe_bitcast[Int16]().unsafe_offset(3)[]) & 1) == 0:
        return False
    var line_in = read_line(fd)
    if not line_in:
        return False
    var line = line_in.value()
    var cancel = is_cancel_line(line)
    var cid = cancel[1]
    if not cancel[0]:
        # Not a cancel: this is the NEXT REQUEST, already in the pipe because
        # the client queued it while this one was still decoding. Hand it back
        # to the request loop. Dropping it here silently ate 19 of 20 requests
        # the first time bench/force-ab-serve.sh was ever run (2026-09-12).
        pending.append(line)
        return False
    return cid == req_id


def is_cancel_line(line: String) -> Tuple[Bool, Int]:
    # The one definition of a {"cancel":ID} line, for every engine (B-M12): the key with an integer value
    # and no request body. A request line never carries "cancel" as a key of its own.
    var i = json_key(line, "cancel")
    var cid = 0
    if i < 0 or json_key(line, "prompt") >= 0 or not json_int(line, i, cid):
        return (False, 0)
    return (True, cid)


def idle_cancel_note(cid: Int) -> String:
    # One defined behaviour for a cancel that arrives while nothing decodes: drop it, say so (a log line, not JSON).
    return "cancel: id " + String(cid) + " is not decoding, dropped"


def json_key(line: String, key: String) -> Int:
    # Index of the first byte of the value for "key": ..., or -1.
    var i = line.find(String("\"") + key + "\"")
    if i < 0:
        return -1
    var b = line.as_bytes()
    i += key.byte_length() + 2
    while i < len(b) and (b[i] == 32 or b[i] == 9):
        i += 1
    if i >= len(b) or b[i] != 58:
        return -1
    i += 1
    while i < len(b) and (b[i] == 32 or b[i] == 9):
        i += 1
    return i


def json_str(line: String, mut i: Int, mut v: String) -> Bool:
    # Plain JSON string at i: no escapes other than \\ and \" (paths never
    # need more). False when i is not at an opening quote or unterminated.
    var b = line.as_bytes()
    if i >= len(b) or b[i] != 34:
        return False
    i += 1
    var out = List[UInt8]()
    while i < len(b) and b[i] != 34:
        if b[i] == 92 and i + 1 < len(b):
            i += 1
        out.append(b[i])
        i += 1
    if i >= len(b):
        return False
    v = String(unsafe_from_utf8=Span(out))
    return True


def json_int(line: String, mut i: Int, mut v: Int) -> Bool:
    var b = line.as_bytes()
    var neg = False
    if i < len(b) and b[i] == 45:
        neg = True
        i += 1
    var have = False
    v = 0
    while i < len(b) and b[i] >= 48 and b[i] <= 57:
        v = v * 10 + Int(b[i] - 48)
        i += 1
        have = True
    if neg:
        v = -v
    return have


def json_float(line: String, mut i: Int, mut v: Float64) -> Bool:
    # Plain decimal (sign, digits, optional '.', digits); no exponent form --
    # none of the sampler fields need one.
    var b = line.as_bytes()
    var neg = False
    if i < len(b) and b[i] == 45:
        neg = True
        i += 1
    var have = False
    var ip: Float64 = 0
    while i < len(b) and b[i] >= 48 and b[i] <= 57:
        ip = ip * 10 + Float64(Int(b[i] - 48))
        i += 1
        have = True
    var frac: Float64 = 0
    if i < len(b) and b[i] == 46:
        i += 1
        var scale: Float64 = 1
        while i < len(b) and b[i] >= 48 and b[i] <= 57:
            scale /= 10
            frac += Float64(Int(b[i] - 48)) * scale
            i += 1
            have = True
    v = ip + frac
    if neg:
        v = -v
    return have


@fieldwise_init
struct SampleParams(Copyable, Movable):
    # C3 (bench/chat-protocol.md): parsed from the wire, not yet acted on --
    # the live decode loop still always takes the greedy/MTP path. Ready for
    # serve/sample_ref.mojo (host reference) or kernels/sample.mojo (device,
    # lane-KSAMP) to read once either is wired in. temperature <= 0 means
    # "off" throughout, matching both references' own convention.
    var temperature: Float64
    var top_p: Float64
    var top_k: Int
    var min_p: Float64
    var seed: UInt64
    var presence_penalty: Float64
    var frequency_penalty: Float64
    # Item 4, briefs/2026-09-16-sampling-all-models-lane.md: 0 means "off",
    # matching every other field's convention here. Bounded to
    # kernels/sample.mojo's amar_topn_probs CAP (<= 20) by the caller.
    var top_logprobs: Int
    # HTTP latent stream extensions: raw post-final-norm hidden rows and
    # pre-penalty logits, emitted after each generated token.
    var hidden: Int
    var logits_topk: Int
    # P0a-e: 1 asks for the last prompt token's post-final-norm hidden row,
    # L2-normalized, as one {"id":ID,"embed":[...]} line before the first
    # token line. Not a sampling knob; it rides here so parse_request keeps
    # its signature across the three harnesses that call it.
    var embed: Int


def default_sample_params() -> SampleParams:
    return SampleParams(temperature=0, top_p=1.0, top_k=0, min_p=0, seed=0, presence_penalty=0, frequency_penalty=0, top_logprobs=0, embed=0, hidden=0, logits_topk=0)


def parse_request(
    line: String, mut id: Int, mut prompt: List[Int], mut n: Int, mut spec: Bool, mut has_spec: Bool, mut stop: List[List[Int]], mut ckpt: List[Int], mut sample: SampleParams,
    mut state_save: String, mut state_load: String, mut state_prefix: Bool,
    mut semantic_manifest_path: String, mut semantic_values_path: String, mut semantic_positions: Int, mut semantic_now_s: Int64,
) -> String:
    # {"id":INT,"prompt":[INT,...],"n":INT,"spec":BOOL,"stop":[[INT,...],...],
    #  "ckpt":[INT,...],"temperature":FLOAT,"top_p":FLOAT,"top_k":INT,
    #  "min_p":FLOAT,"seed":INT,"presence_penalty":FLOAT,
    #  "frequency_penalty":FLOAT,"top_logprobs":INT}; everything past prompt/n
    #  optional. Returns
    # "" on success, else the error text (id is set when it parsed).
    id = 0
    var i = json_key(line, "id")
    if i < 0 or not json_int(line, i, id):
        return "missing or non-integer id"
    i = json_key(line, "n")
    if i < 0 or not json_int(line, i, n):
        return "missing or non-integer n"
    i = json_key(line, "prompt")
    var b = line.as_bytes()
    if i < 0 or i >= len(b) or b[i] != 91:
        return "missing prompt array"
    i += 1
    while True:
        while i < len(b) and (b[i] == 32 or b[i] == 44):
            i += 1
        if i >= len(b):
            return "unterminated prompt array"
        if b[i] == 93:
            break
        var v = 0
        if not json_int(line, i, v) or v < 0:
            return "prompt must hold non-negative integers"
        prompt.append(v)
    has_spec = False
    i = json_key(line, "spec")
    if i >= 0:
        if line.as_bytes()[i] == 116:
            spec = True
            has_spec = True
        elif line.as_bytes()[i] == 102:
            spec = False
            has_spec = True
        else:
            return "spec must be true or false"
    var si = json_key(line, "stop")
    if si >= 0:
        if si >= len(b) or b[si] != 91:
            return "stop must be an array of arrays"
        si += 1
        while True:
            while si < len(b) and (b[si] == 32 or b[si] == 44):
                si += 1
            if si >= len(b):
                return "unterminated stop array"
            if b[si] == 93:
                break
            if b[si] != 91:
                return "stop entries must be arrays of token ids"
            si += 1
            var seq = List[Int]()
            while True:
                while si < len(b) and (b[si] == 32 or b[si] == 44):
                    si += 1
                if si >= len(b):
                    return "unterminated stop sequence"
                if b[si] == 93:
                    si += 1
                    break
                var v2 = 0
                if not json_int(line, si, v2) or v2 < 0:
                    return "stop sequence must hold non-negative integers"
                seq.append(v2)
            stop.append(seq^)
    var ci = json_key(line, "ckpt")
    if ci >= 0:
        if ci >= len(b) or b[ci] != 91:
            return "ckpt must be an array of integers"
        ci += 1
        while True:
            while ci < len(b) and (b[ci] == 32 or b[ci] == 44):
                ci += 1
            if ci >= len(b):
                return "unterminated ckpt array"
            if b[ci] == 93:
                break
            var v3 = 0
            if not json_int(line, ci, v3) or v3 < 0:
                return "ckpt must hold non-negative integers"
            ckpt.append(v3)
    # Checkpoint API (LatentOS plan 10 sec 8): per-request state file paths,
    # the same BAROST01 files BARO_STATE_SAVE / BARO_STATE_LOAD write and read.
    state_save = ""
    state_load = ""
    var ssi = json_key(line, "state_save")
    if ssi >= 0 and not json_str(line, ssi, state_save):
        return "state_save must be a string"
    var sli = json_key(line, "state_load")
    if sli >= 0 and not json_str(line, sli, state_load):
        return "state_load must be a string"
    state_prefix = False
    var spi = json_key(line, "state_prefix")
    if spi >= 0:
        if line.as_bytes()[spi] == 116:
            state_prefix = True
        elif line.as_bytes()[spi] != 102:
            return "state_prefix must be true or false"
        if state_prefix and state_load == "":
            return "state_prefix needs state_load"
    # Semantic latent (LatentOS plan 12 BX): per-request binding manifest and values
    semantic_manifest_path = ""
    semantic_values_path = ""
    semantic_positions = 0
    semantic_now_s = Int64(0)
    var smi = json_key(line, "semantic_manifest")
    if smi >= 0 and not json_str(line, smi, semantic_manifest_path):
        return "semantic_manifest must be a string"
    var svi = json_key(line, "semantic_values")
    if svi >= 0 and not json_str(line, svi, semantic_values_path):
        return "semantic_values must be a string"
    var spi2 = json_key(line, "semantic_positions")
    if spi2 >= 0:
        var ip = 0
        if not json_int(line, spi2, ip) or ip < 1:
            return "semantic_positions must be a positive integer"
        semantic_positions = ip
    var sni = json_key(line, "semantic_now_s")
    if sni >= 0:
        var iv = 0
        if not json_int(line, sni, iv) or iv <= 0:
            return "semantic_now_s must be a positive integer (epoch seconds)"
        semantic_now_s = Int64(iv)
    var fi = json_key(line, "temperature")
    if fi >= 0:
        var fv: Float64 = 0
        if not json_float(line, fi, fv):
            return "temperature must be a number"
        sample.temperature = fv
    fi = json_key(line, "top_p")
    if fi >= 0:
        var fv2: Float64 = 0
        if not json_float(line, fi, fv2):
            return "top_p must be a number"
        sample.top_p = fv2
    fi = json_key(line, "top_k")
    if fi >= 0:
        var iv = 0
        if not json_int(line, fi, iv):
            return "top_k must be an integer"
        sample.top_k = iv
    fi = json_key(line, "min_p")
    if fi >= 0:
        var fv3: Float64 = 0
        if not json_float(line, fi, fv3):
            return "min_p must be a number"
        sample.min_p = fv3
    fi = json_key(line, "seed")
    if fi >= 0:
        var iv2 = 0
        if not json_int(line, fi, iv2) or iv2 < 0:
            return "seed must be a non-negative integer"
        sample.seed = UInt64(iv2)
    fi = json_key(line, "presence_penalty")
    if fi >= 0:
        var fv4: Float64 = 0
        if not json_float(line, fi, fv4):
            return "presence_penalty must be a number"
        sample.presence_penalty = fv4
    fi = json_key(line, "frequency_penalty")
    if fi >= 0:
        var fv5: Float64 = 0
        if not json_float(line, fi, fv5):
            return "frequency_penalty must be a number"
        sample.frequency_penalty = fv5
    fi = json_key(line, "top_logprobs")
    if fi >= 0:
        var iv3 = 0
        if not json_int(line, fi, iv3) or iv3 < 0:
            return "top_logprobs must be a non-negative integer"
        sample.top_logprobs = iv3
    fi = json_key(line, "embed")
    if fi >= 0:
        if line.as_bytes()[fi] == 116:
            sample.embed = 1
        elif line.as_bytes()[fi] != 102:
            return "embed must be true or false"
    fi = json_key(line, "hidden")
    if fi >= 0:
        if line.as_bytes()[fi] == 116:
            sample.hidden = 1
        elif line.as_bytes()[fi] != 102:
            return "hidden must be true or false"
    fi = json_key(line, "logits_topk")
    if fi >= 0:
        var iv4 = 0
        if not json_int(line, fi, iv4) or iv4 < 0 or iv4 > 20:
            return "logits_topk must be an integer from 0 to 20"
        sample.logits_topk = iv4
    return ""


def json_value_span(line: String, start: Int) -> Int:
    # `start` points at the first byte of a JSON value. Returns the index
    # one past the value's last byte, or -1 if malformed/truncated. Handles
    # nested objects/arrays and quoted strings (with escapes) so a whole
    # schema object can be sliced out verbatim without a real JSON parser on
    # this side (grammar.json_value.parse_json_bytes does the real parse,
    # once the sliced text reaches it).
    var b = line.as_bytes()
    var n = len(b)
    if start >= n:
        return -1
    var c = b[start]
    if c == 34:  # string
        var i = start + 1
        while i < n:
            if b[i] == 92:
                i += 2
                continue
            if b[i] == 34:
                return i + 1
            i += 1
        return -1
    if c == 123 or c == 91:  # object / array
        var close: UInt8 = 125 if c == 123 else 93
        var depth = 0
        var i = start
        var in_str = False
        while i < n:
            var ch = b[i]
            if in_str:
                if ch == 92:
                    i += 2
                    continue
                if ch == 34:
                    in_str = False
                i += 1
                continue
            if ch == 34:
                in_str = True
            elif ch == c:
                depth += 1
            elif ch == close:
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return -1
    # number / true / false / null: scan to the next structural byte.
    var i = start
    while i < n and b[i] != 44 and b[i] != 125 and b[i] != 93 and b[i] != 32 and b[i] != 9:
        i += 1
    return i


def parse_schema_field(line: String) -> String:
    # A5/JSON-enforcement item 1: {"schema": {...}} carries
    # response_format.json_schema.schema verbatim from main.rs. Returns the
    # raw JSON text of the object, or "" when the field is absent or
    # malformed (absent is the normal case: only a response_format request
    # sends it). A separate function, not a parse_request field, so every
    # existing call site (engine.mojo, spark.mojo) is unaffected until each
    # opts in.
    var i = json_key(line, "schema")
    if i < 0:
        return ""
    var b = line.as_bytes()
    if i >= len(b) or b[i] != 123:
        return ""
    var end = json_value_span(line, i)
    if end < 0:
        return ""
    return String(line[byte=i:end])


def parse_str_field(line: String, key: String) -> String:
    # A top-level string field ("" when absent or not a string). json_str
    # decodes only \\ and \" -- main.rs keeps control bytes out of the
    # tool regexes for that reason.
    var i = json_key(line, key)
    var v = String("")
    if i < 0 or not json_str(line, i, v):
        return ""
    return v


def parse_reasoning_field(line: String) -> Bool:
    # JSON-enforcement item 2: {"reasoning": BOOL}, main.rs's own read of
    # chat_template_kwargs.enable_thinking (default true, matching Qwythos's
    # template: reasoning is on unless the caller explicitly turns it off).
    # Only meaningful alongside "schema" -- the reasoning-boundary scan is
    # skipped entirely when there is no grammar to gate.
    var i = json_key(line, "reasoning")
    if i < 0:
        return True
    var b = line.as_bytes()
    if i < len(b) and b[i] == 102:
        return False
    return True
