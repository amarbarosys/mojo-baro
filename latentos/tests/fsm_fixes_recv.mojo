# Audit row C12: recv_handle_why says why no handle arrived (the five failures used to look alike).
from std.ffi import external_call

import latentos.ipc as ipc
import latentos.proto as proto
import latentos.sys as sys


def hdr() -> proto.LatentHeader:
    var out = proto.LatentHeader()
    out.kind = proto.KIND_KV_PAGES
    out.dtype = proto.DTYPE_F32
    out.payload_len = 4096
    out.weights_uuid = Array[UInt8, 16](fill=1)
    out.runtime = Array[UInt8, 32](fill=2)
    out.tokenizer_sha = Array[UInt8, 32](fill=3)
    return out^


def pair() -> List[Int32]:
    var sv = List[Int32](length=2, fill=0)
    _ = external_call["socketpair", Int32](Int32(1), Int32(1), Int32(0), sv.unsafe_ptr())
    return sv^


def main() raises:
    var fails = 0
    var a = pair()
    var junk = List[UInt8](length=10, fill=7)
    _ = external_call["send", Int64](a[0], junk.unsafe_ptr(), 10, Int32(0))
    var r1 = ipc.recv_handle_why(a[1])
    if r1[2] != "short_read" or r1[1] >= 0:
        print("FAIL short_read: got '" + r1[2] + "'")
        fails += 1
    var b = pair()
    var raw = List[UInt8](length=proto.LATENT_HEADER_SIZE, fill=0)
    hdr().serialize(raw.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    _ = external_call["send", Int64](b[0], raw.unsafe_ptr(), proto.LATENT_HEADER_SIZE, Int32(0))
    var r2 = ipc.recv_handle_why(b[1])
    if r2[2] != "no_rights" or r2[1] >= 0:
        print("FAIL no_rights: got '" + r2[2] + "'")
        fails += 1
    var c = pair()
    _ = sys.sys_close(c[0])
    var r3 = ipc.recv_handle_why(c[1])
    if r3[2] != "peer_closed":
        print("FAIL peer_closed: got '" + r3[2] + "'")
        fails += 1
    var d = pair()
    var m = ipc.mint_memfd_rw("fsm-recv", 4096)
    _ = ipc.seal_and_finalize(m[0], m[1], 4096)
    if not ipc.send_handle(d[0], hdr(), m[0]):
        raise Error("send_handle of a sealed memfd failed")
    var r4 = ipc.recv_handle_why(d[1])
    if r4[2] != "" or r4[1] < 0:
        print("FAIL ok path: reason '" + r4[2] + "' fd " + String(Int(r4[1])))
        fails += 1
    if fails > 0:
        raise Error("fsm_fixes_recv: " + String(fails) + " failed")
    print("fsm_fixes_recv: PASS (short_read, no_rights, peer_closed, success)")
