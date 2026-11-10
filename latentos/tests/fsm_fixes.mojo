# Audit rows C7, C9, C10, C11, C12, C13, C14 (docs/state-machines.md): each check is the symptom that was reported,
# so it fails on the code before the fix. Prints one line per check, exits non-zero when any failed.
from std.ffi import external_call
from std.memory import Pointer

import latentos.agent as agent
import latentos.contract as contract
import latentos.client as client
import latentos.crypto as crypto
import latentos.ipc as ipc
import latentos.proto as proto
import latentos.router as router
import latentos.store as store
import latentos.sys as sys
import latentos.transport as transport


def check(id: String, ok: Bool, why: String, mut fails: Int):
    if ok:
        print("ok   " + id + ": " + why)
    else:
        print("FAIL " + id + ": " + why)
        fails += 1


def raised(id: String, want: String, mut fails: Int, fired: Bool, got: String):
    check(id, fired and want in got, "raises " + want + (" (got: " + got + ")" if fired else " (nothing raised)"), fails)


def hdr(length: Int) -> proto.LatentHeader:
    var out = proto.LatentHeader()
    out.kind = proto.KIND_KV_PAGES
    out.dtype = proto.DTYPE_F32
    out.payload_len = UInt64(length)
    out.weights_uuid = Array[UInt8, 16](fill=1)
    out.runtime = Array[UInt8, 32](fill=2)
    out.tokenizer_sha = Array[UInt8, 32](fill=3)
    return out^


def main() raises:
    var fails = 0

    # C13: the crypto shim initialises libsodium itself. sodium_init() returns 1 when it already ran, 0 on its first
    # run: this is the first thing the process does, so 1 means the shim did it.
    var nonce = List[UInt8](length=24, fill=0)
    external_call["los_randombytes", NoneType](nonce.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), UInt64(24))
    var rc = external_call["sodium_init", Int32]()
    check("C13", rc == 1, "sodium_init after the first shim call returns " + String(Int(rc)) + " (1 = the shim initialised it)", fails)

    # C14: no oracle (ROUTE_NONE) is not a wrong answer, and has no regret of about -1e30.
    var ri = router.RouteInput()
    ri.payload_valid = True
    ri.recipe_valid = True
    ri.text_valid = True
    ri.terms.payload_transfer_s = 1.0
    ri.terms.recipe_prefill_s = 9.0
    ri.terms.text_prefill_s = 9.0
    var d = router.choose_route(ri)
    check("C14", not d.wrong and d.regret_s > -1.0, "no oracle: wrong=" + String(d.wrong) + " regret_s=" + String(d.regret_s), fails)
    ri.local_valid = True
    var d2 = router.choose_route(ri)
    check("C14", not d2.wrong and d2.regret_s > -1.0, "no oracle, local route: wrong=" + String(d2.wrong), fails)

    # C9: a put of an expired entry does not revive it (acquire refuses the same entry), and get names the reason.
    var s = ipc.LatentStore()
    var h = hdr(16)
    h.ttl_s = 0
    var m = ipc.mint_memfd_rw("fsm-c9", 16)
    _ = sys.sys_munmap(m[1], 16)
    var addr = s.put(h, m[0])
    var fired = False
    var got = String("")
    try:
        _ = s.put(h, m[0])
    except e:
        fired = True
        got = String(e)
    raised("C9 put", "STATE:expired", fails, fired, got)
    fired = False
    got = ""
    try:
        _ = s.get(addr)
    except e:
        fired = True
        got = String(e)
    raised("C9 get expired", "STATE:expired", fails, fired, got)
    fired = False
    got = ""
    try:
        _ = s.get("no-such-address")
    except e:
        fired = True
        got = String(e)
    raised("C9 get unknown", "STATE:stale_handle", fails, fired, got)

    # C10: release is False for "still referenced" only; a second release of the same handle is named.
    var id = contract.EngineIdentity()
    id.weights_uuid = Array[UInt8, 16](fill=1)
    id.runtime = Array[UInt8, 32](fill=2)
    id.sigma_id = Array[UInt8, 32](fill=0)
    id.tokenizer_sha = Array[UInt8, 32](fill=3)
    var limits = store.StoreLimits()
    limits.max_handles = 16
    limits.max_bytes = 4096
    limits.default_ttl_s = 50
    var api = client.LatentOSClient(id, limits)
    var grant = contract.Authority()
    grant.tenant = "tenant"
    grant.room = "room"
    grant.session = "session"
    grant.sender = "sender"
    grant.recipient = "recipient"
    grant.expires_at_s = 1000
    grant.token[0] = 1
    var payload = List[UInt8]()
    payload.append(1)
    var grand = api.create(hdr(1), payload, grant, "grand", contract.TIER_DISK, 1)
    _ = api.fork(grand, "child", 2)
    var first = api.release(grand, 3)
    check("C10 first", not first, "release of a referenced parent returns False", fails)
    fired = False
    got = ""
    try:
        _ = api.release(grand, 4)
    except e:
        fired = True
        got = String(e)
    raised("C10 second", "STATE:already_released", fails, fired, got)

    # C11: a session with no manifest says so (the relay's word), not "identity"; a completed session keeps its result.
    var session = transport.TransferSession(1024)
    fired = False
    got = ""
    try:
        _ = session.accept_chunk(crypto.EncryptedChunk("w", 0, 0, 1, Array[UInt8, 24](fill=0), Array[UInt8, 16](fill=0), List[UInt8](length=1, fill=0), List[UInt8]()), 5)
    except e:
        fired = True
        got = String(e)
    raised("C11 chunk", "TRANSFER:manifest_required", fails, fired, got)
    var key = crypto.CryptoKey("k", List[UInt8](length=32, fill=7))
    fired = False
    got = ""
    try:
        _ = session.finalize(key, 5)
    except e:
        fired = True
        got = String(e)
    raised("C11 finalize", "TRANSFER:manifest_required", fails, fired, got)
    session.wire_id = "w"
    session.completed = True
    session.result = List[UInt8](length=3, fill=9)
    session.manifest.expires_at_s = 10
    fired = False
    var out = List[UInt8]()
    try:
        out = session.finalize(key, 100)
    except e:
        fired = True
        got = String(e)
    check("C11 retry", not fired and len(out) == 3, "a completed session returns its result after the manifest expired" + (" (raised: " + got + ")" if fired else ""), fails)

    # C12: a writable memfd is neither mapped as immutable nor sent as one.
    var mw = ipc.mint_memfd_rw("fsm-c12", 4096)
    _ = sys.sys_munmap(mw[1], 4096)
    fired = False
    got = ""
    try:
        _ = ipc.map_readonly(mw[0], 4096)
    except e:
        fired = True
        got = String(e)
    raised("C12 map", "STATE:unsealed", fails, fired, got)
    var sv = List[Int32](length=2, fill=0)
    _ = external_call["socketpair", Int32](Int32(1), Int32(1), Int32(0), sv.unsafe_ptr())
    var sent = ipc.send_handle(sv[0], hdr(4096), mw[0])
    check("C12 send unsealed", not sent, "send_handle of an unsealed memfd returned " + String(sent), fails)
    var ms = ipc.mint_memfd_rw("fsm-c12-sealed", 4096)
    var sealed = ipc.seal_and_finalize(ms[0], ms[1], 4096)
    var sent2 = ipc.send_handle(sv[0], hdr(4096), ms[0])
    check("C12 send sealed", sealed and sent2, "a sealed memfd still goes out (sealed=" + String(sealed) + " sent=" + String(sent2) + ")", fails)

    # C7: the agent says which stage it reached, sends READY only after a written manifest.
    var cfg = agent.AgentConfig()
    cfg.manifest_out = "/proc/fsm-fixes-no-such-dir/manifest.json"
    cfg.manifest_fallback = "/proc/fsm-fixes-no-such-dir/fallback.json"
    var a = agent.LatentAgent(cfg)
    a.run_check()
    check("C7 stage", a.last_stage == "L6", "last_stage=" + a.last_stage, fails)
    check("C7 ready", (not a.manifest_written) and (not a.ready_sent), "manifest_written=" + String(a.manifest_written) + " ready_sent=" + String(a.ready_sent) + " (no writable path)", fails)

    if fails > 0:
        raise Error("fsm_fixes: " + String(fails) + " check(s) failed")
    print("fsm_fixes: PASS")
