"""Pure-function tests for the fix-engine lane (docs/lanes/2026-11-09-fsm-fixes.md, rows B-M*).
No accelerator, no pack. One test per row id; each reproduced its symptom on the lane's base commit.

Build: ./.venv/bin/mojo build serve/test_fsm_wire.mojo -I . -I serve -o .work/test_fsm_wire
"""
from std.testing import assert_equal
from max.gpu.host import DeviceContext
from prefix import Checkpoint
from serve_proto import is_cancel_line, idle_cancel_note
from grammar.matcher import Matcher
from wire import reset_request_knobs, slens_base_ok, slens_base_error
from wire import grammar_word, tier_word, ready_words
from wire import PATH_NONE, PATH_LAUNCH, PATH_MEGA, PATH_MEGA_WIN, PATH_SPEC, PATH_PAIR, PATH_SEQS, path_word, decide_path
from wire import SeqTag, SEQ_PREFILL, SEQ_DECODE, SEQ_DONE, FIN_LENGTH, FIN_STOP, FIN_CANCELLED, FIN_ERROR, finish_word
from wire import err_line, json_escape, DoneWords, window_exceeded, window_error, line_id, latent_fork_error, latent_resume_error


def test_errline_escapes() raises:
    # B-M1 (err_line): a message holding a quote or a backslash broke the line.
    assert_equal(err_line(7, String('bad "x" and \\ path')), String('{"id":7,"error":"bad \\"x\\" and \\\\ path"}'))
    assert_equal(err_line(0, String("plain")), String('{"id":0,"error":"plain"}'))
    assert_equal(json_escape(String("a\nb\tc\x01")), String("a\\nb\\tc\\u0001"))
    assert_equal(json_escape(String("naïve")), String("naïve"))


def test_done_fail() raises:
    # B-M1: the done line carried no fail word; an invalid-token run read finish "length" with nothing else.
    var dw = DoneWords()
    assert_equal(dw.json(), String(',"fail":0'))
    dw.fail = 2  # injected mega fail word (the live NOT-RESIDENT case cannot be forced, see the plan)
    assert_equal(dw.json(), String(',"fail":2'))


def test_window_after_extension() raises:
    # B-M1: state_prefix / ipc_ingest grow the prompt after the first check; the second check must name the id, not exit(2).
    assert_equal(window_exceeded(100, 28, 128), False)
    assert_equal(window_exceeded(100, 29, 128), True)
    assert_equal(window_error(100, 29, 128), String("prompt+n exceeds TMAX 128 (prompt 100, n 29, after prefix extension)"))
    assert_equal(err_line(5, window_error(100, 29, 128)).startswith('{"id":5,"error":"prompt+n exceeds TMAX 128'), True)


def test_latent_guards() raises:
    # B-M5: latent_resume with an unknown id raised a Dict key error out of main (engine dead, no wire line).
    assert_equal(line_id(String('{"id":9,"latent_resume":"lat_1"}')), 9)
    assert_equal(line_id(String('{"latent_resume":"lat_1"}')), 0)
    assert_equal(err_line(9, latent_resume_error(String("lat_1"))), String('{"id":9,"error":"latent_resume: unknown latent_id lat_1"}'))
    assert_equal(latent_fork_error(), String("latent_fork: no checkpoint in the chain"))


def test_cancel_line() raises:
    # B-M12: three engines, three behaviours (a missing-id error, silence, a prefix match): one definition now.
    var c = is_cancel_line(String('{"cancel":7}'))
    assert_equal(c[0], True)
    assert_equal(c[1], 7)
    var spaced = is_cancel_line(String('{ "cancel" : 12 }'))  # q4x's startswith() only matched the first, unspaced key
    assert_equal(spaced[0], True)
    assert_equal(spaced[1], 12)
    assert_equal(is_cancel_line(String('{"id":3,"prompt":[1,2],"n":4}'))[0], False)
    assert_equal(is_cancel_line(String('{"id":3,"cancel":3,"prompt":[1]}'))[0], False)
    assert_equal(is_cancel_line(String('{"cancel":"x"}'))[0], False)
    assert_equal(idle_cancel_note(7), String("cancel: id 7 is not decoding, dropped"))


def test_seq_tag() raises:
    # B-M7: KV exhaustion set `cancelled`, so the client read an error line and then finish "cancelled"
    # for the same id, identical to a real cancel; `cancelled` and `stopped` could both be true.
    var t = SeqTag()
    assert_equal(t.phase, SEQ_PREFILL)
    assert_equal(t.ended(), False)
    t.decoding()
    assert_equal(t.phase, SEQ_DECODE)
    t.end(FIN_ERROR)
    assert_equal(t.phase, SEQ_DONE)
    assert_equal(finish_word(t.reason), String("error"))
    t.end(FIN_CANCELLED)  # a later cancel must not rewrite the error
    t.end(FIN_STOP)
    assert_equal(finish_word(t.reason), String("error"))
    t.decoding()  # DONE never returns to DECODE
    assert_equal(t.phase, SEQ_DONE)
    var c = SeqTag()
    c.end(FIN_CANCELLED)
    assert_equal(finish_word(c.reason), String("cancelled"))
    assert_equal(finish_word(SeqTag().reason), String("length"))
    var s = SeqTag()
    s.end(FIN_STOP)
    assert_equal(finish_word(s.reason), String("stop"))


def test_checkpoint_install() raises:
    # B-M4: four files wrote valid/pending by hand; (valid, pending) = (True, True) must not exist.
    var ctx = DeviceContext()
    var c = Checkpoint(pos=0, hash=List[UInt8](), gen=0, valid=False, pending=True, pinned=False, boundary=False, conv_h=ctx.enqueue_create_host_buffer[DType.float32](1), ssm_h=ctx.enqueue_create_host_buffer[DType.float32](1), kv_h=ctx.enqueue_create_host_buffer[DType.float32](0), pages=List[Int32](), hits=0)
    c.set_valid()  # pending -> valid clears pending (commit path)
    assert_equal(c.valid, True)
    assert_equal(c.pending, False)
    var h = List[UInt8]()
    h.append(7)
    c.valid = False
    c.pending = True  # a stale pending flag at install time
    c.install(40, h^, 9, True, True)
    assert_equal(c.pos, 40)
    assert_equal(c.gen, 9)
    assert_equal(len(c.hash), 1)
    assert_equal(c.valid, True)
    assert_equal(c.pending, False)  # install never leaves (True, True)
    assert_equal(c.pinned and c.boundary, True)


def test_decide_path() raises:
    # B-M3: the chosen path was not stated, and mega_win_req skipped the image test that mega_req had.
    # args: mega_on, mega_win_on, spec, cap_on, knobs_on, want_extra, want_grammar, has_img, sampling
    var plain = decide_path(True, False, False, False, False, False, False, False, False)
    assert_equal(plain.tag, PATH_MEGA)
    assert_equal(plain.mega, True)
    assert_equal(plain.why, String(""))
    assert_equal(decide_path(False, False, False, False, False, False, False, False, False).tag, PATH_LAUNCH)
    assert_equal(decide_path(True, False, True, False, False, False, False, False, False).tag, PATH_SPEC)
    var win = decide_path(True, True, True, False, False, False, False, False, False)
    assert_equal(win.tag, PATH_MEGA_WIN)
    assert_equal(win.mega_win, True)
    var img = decide_path(True, True, True, False, False, False, False, True, False)  # image on a MEGA_WIN engine
    assert_equal(img.mega, False)
    assert_equal(img.mega_win, False)  # was True before
    assert_equal(img.tag, PATH_SPEC)
    assert_equal(img.why, String("image"))
    var samp = decide_path(True, True, True, False, False, False, False, False, True)
    assert_equal(samp.mega_win, False)  # sampling keeps the window on the launch path
    assert_equal(samp.mega, True)
    assert_equal(decide_path(True, True, False, True, False, False, False, False, False).why, String("capture"))
    assert_equal(decide_path(True, False, False, False, True, False, False, False, False).why, String("per-request knob"))
    assert_equal(decide_path(True, False, False, False, False, False, True, False, False).why, String("grammar"))
    assert_equal(path_word(PATH_PAIR), String("pair"))
    assert_equal(path_word(PATH_SEQS), String("seqs"))
    assert_equal(path_word(PATH_MEGA_WIN), String("mega_win"))
    var dw = DoneWords()
    assert_equal(dw.json(), String(',"fail":0'))  # not stated: no path field
    dw.path = PATH_SPEC
    assert_equal(dw.json(), String(',"fail":0,"path":"spec"'))
    assert_equal(PATH_NONE < 0, True)


def test_grammar_words() raises:
    # B-M6: draws != accepted was a log line only; C6: window.mojo counted the attempt, dropped accept()'s result.
    assert_equal(grammar_word(False, False, False), String("none"))
    assert_equal(grammar_word(True, True, False), String("wait"))
    assert_equal(grammar_word(True, False, False), String("match"))
    assert_equal(grammar_word(True, False, True), String("done"))
    var dw = DoneWords()
    dw.grammar = grammar_word(True, False, True)
    dw.gdraws = 12
    dw.gaccepted = 11
    dw.grejected = 1  # the desync a reader can now see: one draw the matcher refused
    assert_equal(dw.json(), String(',"fail":0,"grammar":"done","gdraws":12,"gaccepted":11,"grejected":1'))
    var plain = DoneWords()
    plain.grammar = grammar_word(False, False, False)
    assert_equal(plain.json(), String(',"fail":0,"grammar":"none"'))


def test_ready_and_capture_words() raises:
    # B-M9: a slow tier mode had no name on the wire (a log string only). B-M13: no way to check capture files.
    assert_equal(tier_word(False, False, False, False, False, False), String("off"))
    assert_equal(tier_word(True, False, False, False, False, False), String("page-cache"))
    assert_equal(tier_word(True, False, True, True, False, False), String("pinned+zc"))
    assert_equal(tier_word(True, True, True, True, False, False), String("pinned+cpu"))
    assert_equal(tier_word(True, False, False, True, False, True), String("pinned+mmap"))
    assert_equal(tier_word(True, False, False, False, True, False), String("hot"))
    assert_equal(ready_words(String("pinned+zc"), 2), String(',"tier":"pinned+zc","seqs":2'))
    var dw = DoneWords()
    assert_equal(dw.json(), String(',"fail":0'))  # capture off: no field
    dw.captured = 37
    assert_equal(dw.json(), String(',"fail":0,"captured":37'))
    dw.load_tier = "hot"  # A-M15: the engine says where state came from; the front stops guessing from load_s < 0.002
    assert_equal(dw.json(), String(',"fail":0,"captured":37,"load_tier":"hot"'))


def test_reset_request_knobs() raises:
    # B-M2: bufs.steer_on / gain / imp_on and the grammar fields persisted from a request into the control lines
    # and a3b / a3c pairs that `continue` before the parse block resets them.
    var imp_on = True
    var gain_l: List[Int] = [3]
    var gain_n: List[Int] = [5]
    var gain_k: List[Float32] = [2.0]
    var steer_on = True
    var grammar = Optional[Matcher](None)
    var pending_think = True
    var think_buf = List[UInt8]()
    think_buf.append(60)
    var grammar_stop = True
    var masked = 9
    var accepted = 8
    var imp_scale: Float32 = 0.5
    var imp_layer = 4
    var imp_klim = 7
    var imp_lo = 2
    var spl_layer = 3
    var spl_zero = True
    var spl_n = 100
    reset_request_knobs(imp_on, gain_l, gain_n, gain_k, steer_on, grammar, pending_think, think_buf, grammar_stop, masked, accepted, False, False, imp_scale, imp_layer, imp_klim, imp_lo, spl_layer, spl_zero, spl_n)
    assert_equal(imp_on, False)  # boot value, not the previous request's
    assert_equal(len(gain_l) + len(gain_n) + len(gain_k), 0)
    assert_equal(steer_on, False)
    assert_equal(grammar.__bool__(), False)
    assert_equal(pending_think, False)
    assert_equal(len(think_buf), 0)
    assert_equal(grammar_stop, False)
    assert_equal(masked + accepted, 0)
    assert_equal(imp_scale, Float32(1.0))  # the sweep knobs and the splice are back to "off"
    assert_equal(imp_layer + imp_klim, -2)
    assert_equal(imp_lo, 0)
    assert_equal(spl_layer, -1)
    assert_equal(spl_zero, False)
    assert_equal(spl_n, 0)
    reset_request_knobs(imp_on, gain_l, gain_n, gain_k, steer_on, grammar, pending_think, think_buf, grammar_stop, masked, accepted, True, True, imp_scale, imp_layer, imp_klim, imp_lo, spl_layer, spl_zero, spl_n)
    assert_equal(imp_on, True)  # boot implants and boot steer come back
    assert_equal(steer_on, True)


def test_slens_base() raises:
    # B-M15: k > 0 before any k == 0 for that context raised inside the decode loop; the engine died.
    var base = List[List[Float32]]()
    assert_equal(slens_base_ok(base, 2, 0), True)  # k == 0 stores the base
    assert_equal(slens_base_ok(base, 2, 1), False)  # no base yet: refuse
    base.append(List[Float32]())
    base.append(List[Float32]())
    base.append(List[Float32](length=4, fill=0.0))
    assert_equal(slens_base_ok(base, 2, 1), True)
    assert_equal(slens_base_ok(base, 1, 1), False)  # a hole below the stored context
    assert_equal(slens_base_ok(base, -1, 1), False)
    assert_equal(slens_base_error(2), String("slens: no base for context 2 (send slens_k 0 for it first)"))


def main() raises:
    test_errline_escapes()
    test_done_fail()
    test_window_after_extension()
    test_latent_guards()
    test_cancel_line()
    test_seq_tag()
    test_checkpoint_install()
    test_decide_path()
    test_grammar_words()
    test_ready_and_capture_words()
    test_reset_request_knobs()
    test_slens_base()
    print("PASS fsm wire: err_line escaping, done fail word, window after extension, latent guards, cancel line, seq tag, checkpoint install, decide_path, grammar words, ready and capture words, reset request knobs, slens base")
