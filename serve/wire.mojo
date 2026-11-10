"""Words the engine writes on its stdout wire (serve/PROTOCOL.md), as pure functions.

No GPU, no MAX: serve/test_fsm_wire.mojo builds this module on the CPU. One word per
terminal state (docs/state-machines.md, decision 2); the engines call these, they do
not spell the JSON by hand.
"""
from grammar.matcher import Matcher
from serve_proto import json_key, json_int


def json_escape(s: String) -> String:
    # Quote, backslash and control bytes; everything else (UTF-8 included) passes through.
    var out = List[UInt8]()
    var hex = "0123456789abcdef".as_bytes()
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
            for c in "\\u00".as_bytes():
                out.append(c)
            out.append(hex[Int(b >> 4)])
            out.append(hex[Int(b & 15)])
        else:
            out.append(b)
    return String(unsafe_from_utf8=Span[UInt8](out))


def err_line(id: Int, msg: String) -> String:
    return String("{\"id\":") + String(id) + ",\"error\":\"" + json_escape(msg) + "\"}"


def window_exceeded(n_prompt: Int, n_gen: Int, tmax: Int) -> Bool:
    return n_prompt + n_gen > tmax


def window_error(n_prompt: Int, n_gen: Int, tmax: Int) -> String:
    return "prompt+n exceeds TMAX " + String(tmax) + " (prompt " + String(n_prompt) + ", n " + String(n_gen) + ", after prefix extension)"


def line_id(line: String) -> Int:
    # The request id of a control line; 0 when absent, like the ipc ops.
    var i = json_key(line, "id")
    var out = 0
    if i < 0 or not json_int(line, i, out):
        return 0
    return out


def latent_fork_error() -> String:
    return "latent_fork: no checkpoint in the chain"


def latent_resume_error(cid: String) -> String:
    return "latent_resume: unknown latent_id " + cid


# B-M7: one lifecycle tag per sequence of the pair / seqs scheduler, one transition (SeqTag.end), one finish word.
comptime SEQ_QUEUED = 0
comptime SEQ_PREFILL = 1
comptime SEQ_DECODE = 2
comptime SEQ_DONE = 3
comptime FIN_LENGTH = 0
comptime FIN_STOP = 1
comptime FIN_CANCELLED = 2
comptime FIN_ERROR = 3


def finish_word(reason: Int) -> String:
    if reason == FIN_STOP:
        return "stop"
    if reason == FIN_CANCELLED:
        return "cancelled"
    if reason == FIN_ERROR:
        return "error"
    return "length"


@fieldwise_init
struct SeqTag(Copyable, Movable):
    var phase: Int
    var reason: Int  # FIN_*, meaningful once phase == SEQ_DONE (SEQ_DONE with FIN_LENGTH = ran to n)

    def __init__(out self):
        self.phase = SEQ_PREFILL
        self.reason = FIN_LENGTH

    def decoding(mut self):
        if self.phase == SEQ_PREFILL:
            self.phase = SEQ_DECODE

    def end(mut self, reason: Int):
        # The first end wins: a cancel that arrives after a stop, or a stop after an error, changes nothing.
        if self.phase != SEQ_DONE:
            self.phase = SEQ_DONE
            self.reason = reason

    def ended(self) -> Bool:
        return self.phase == SEQ_DONE


# B-M3: the decode path a request ran on. "prefill" and "replay" of the seam describe the prompt phase, which
# pf_rows (> 0 = batched prefill, else replay) and prefill_rows already carry on the same line: not repeated here.
comptime PATH_NONE = -1
comptime PATH_LAUNCH = 0
comptime PATH_MEGA = 1
comptime PATH_MEGA_WIN = 2
comptime PATH_SPEC = 3
comptime PATH_PAIR = 4
comptime PATH_SEQS = 5


def path_word(p: Int) -> String:
    if p == PATH_MEGA:
        return "mega"
    if p == PATH_MEGA_WIN:
        return "mega_win"
    if p == PATH_SPEC:
        return "spec"
    if p == PATH_PAIR:
        return "pair"
    if p == PATH_SEQS:
        return "seqs"
    return "launch"


def grammar_word(has_grammar: Bool, pending_think: Bool, stopped: Bool) -> String:
    # B-M6: none (no grammar), wait (until </think>), match (constraining), done (terminated).
    if not has_grammar:
        return "none"
    if pending_think:
        return "wait"
    if stopped:
        return "done"
    return "match"


def slens_base_ok(base: List[List[Float32]], c: Int, k: Int) -> Bool:
    # B-M15: k > 0 needs the base row stored by an earlier k == 0 request for the same context; slens_step
    # raised "slens: no base for context" inside the decode loop and the engine died. A stored base is never empty.
    if k == 0:
        return True
    return c >= 0 and c < len(base) and len(base[c]) > 0


def slens_base_error(c: Int) -> String:
    return "slens: no base for context " + String(c) + " (send slens_k 0 for it first)"


def reset_request_knobs(
    mut imp_on: Bool, mut gain_l: List[Int], mut gain_n: List[Int], mut gain_k: List[Float32], mut steer_on: Bool,
    mut grammar: Optional[Matcher], mut pending_think: Bool, mut think_buf: List[UInt8], mut grammar_stop: Bool,
    mut masked_draws: Int, mut accepted: Int, steer_boot: Bool, imp_loaded: Bool,
    mut imp_scale: Float32, mut imp_layer: Int, mut imp_klim: Int, mut imp_lo: Int,
    mut spl_layer: Int, mut spl_zero: Bool, mut spl_n: Int,
):
    # B-M2 / B-M14: every per-request semantic input lives one request; boot input is restored. Called at the top
    # of the request loop, before the first `continue`, so a3b / a3c pairs and control lines never run with the
    # previous request's steer, gain, implants or grammar. The parse block still sets the request's own values.
    imp_on = imp_loaded
    gain_l = List[Int]()
    gain_n = List[Int]()
    gain_k = List[Float32]()
    steer_on = steer_boot
    grammar = None
    pending_think = False
    think_buf = List[UInt8]()
    grammar_stop = False
    masked_draws = 0
    accepted = 0
    imp_scale = 1.0  # neuron2b sweep knobs (implants_scale / _layer / _k / _lo) and the splice
    imp_layer = -1
    imp_klim = -1
    imp_lo = 0
    spl_layer = -1
    spl_zero = False
    spl_n = 0


struct PathPlan(Copyable, Movable):
    var tag: Int
    var mega: Bool  # cfg.mega: the single-token persistent kernel
    var mega_win: Bool  # cfg.mega_win: the persistent window kernel
    var why: String  # the first exclusion that turned the megakernel off, "" when none

    def __init__(out self, tag: Int, mega: Bool, mega_win: Bool, var why: String):
        self.tag = tag
        self.mega = mega
        self.mega_win = mega_win
        self.why = why^


def decide_path(
    mega_on: Bool, mega_win_on: Bool, spec: Bool, cap_on: Bool, knobs_on: Bool, want_extra: Bool,
    want_grammar: Bool, has_img: Bool, sampling: Bool,
) -> PathPlan:
    # The one exclusion table (B-M3). knobs_on = a per-request device knob is set (gain, implants, and
    # whatever a later knob adds): the megakernels do not read it. An image excludes BOTH kernels; the
    # window kernel used to skip that test (mega_win_req had no has_img).
    var why = String("")
    if cap_on:
        why = "capture"
    elif knobs_on:
        why = "per-request knob"
    elif want_extra:
        why = "penalties/logprobs/hidden/slens"
    elif want_grammar:
        why = "grammar"
    elif has_img:
        why = "image"
    var open = why == ""
    var mega = mega_on and open
    var mega_win = mega_win_on and open and not sampling  # sampling needs the target's probability rows, not its argmax
    var tag = PATH_LAUNCH
    if spec and mega_win:
        tag = PATH_MEGA_WIN
    elif spec:
        tag = PATH_SPEC
    elif mega:
        tag = PATH_MEGA
    return PathPlan(tag, mega, mega_win, why^)


def tier_word(active: Bool, cpu_on: Bool, zc: Bool, pinned: Bool, hot: Bool, mmap: Bool) -> String:
    # B-M9: the expert tier's mode as one word (the log line prints it with its numbers; the wire gets the word).
    if not active:
        return "off"
    var w = String("page-cache")
    if cpu_on:
        w = "pinned+cpu"
    elif zc:
        w = "pinned+zc"
    elif pinned:
        w = "pinned"
    elif hot:
        w = "hot"
    if mmap:
        w += "+mmap"
    return w


def ready_words(tier: String, seqs: Int) -> String:
    return ",\"tier\":\"" + tier + "\",\"seqs\":" + String(seqs)


struct DoneWords:
    """Extra fields of the done line, one word per terminal state (docs/lanes/2026-11-09-fsm-fixes.md,
    Wire seam). The engine fills the ones it knows and appends json() once, at both done-line sites."""
    var fail: Int  # mega fail word, 0 = ok, non-zero = tokens after a timed-out grid barrier are invalid
    var path: Int  # PATH_*, PATH_NONE = not stated
    var grammar: String  # grammar_word(), "" = not stated
    var gdraws: Int  # masked draws; accepted and rejected are matcher.accept() results; -1 = no grammar
    var gaccepted: Int
    var grejected: Int
    var captured: Int  # positions appended to the BARO_CAPTURE files by this request, -1 = capture off
    var load_tier: String  # where the request's state came from: hot | file | ipc; "" = nothing loaded (the front used to guess from load_s)

    def __init__(out self):
        self.captured = -1
        self.load_tier = ""
        self.fail = 0
        self.path = PATH_NONE
        self.grammar = ""
        self.gdraws = -1
        self.gaccepted = -1
        self.grejected = -1

    def json(self) -> String:
        var out = ",\"fail\":" + String(self.fail)
        if self.path != PATH_NONE:
            out += ",\"path\":\"" + path_word(self.path) + "\""
        if self.grammar != "":
            out += ",\"grammar\":\"" + self.grammar + "\""
        if self.gdraws >= 0:
            out += ",\"gdraws\":" + String(self.gdraws) + ",\"gaccepted\":" + String(self.gaccepted) + ",\"grejected\":" + String(self.grejected)
        if self.captured >= 0:
            out += ",\"captured\":" + String(self.captured)
        if self.load_tier != "":
            out += ",\"load_tier\":\"" + self.load_tier + "\""
        return out
