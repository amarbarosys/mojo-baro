from std.memory import Pointer

import latentos.ipc as ipc
import latentos.proto as proto
import latentos.sys as sys


comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]


def main() raises:
    var pair = ipc.create_ipc_pair()
    var payload = List[UInt8]()
    for i in range(32):
        payload.append(UInt8(i))
    var header = proto.LatentHeader()
    header.kind = proto.KIND_SSM_CKPT
    header.dtype = proto.DTYPE_F32
    header.payload_len = UInt64(len(payload))
    var sent_hash = ipc.send_tcp_latent(pair[0], header, payload.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), len(payload))
    var received = List[UInt8](unsafe_uninit_length=len(payload))
    var result = ipc.recv_tcp_latent(pair[1], received.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), len(received))
    if result[0].kind != proto.KIND_SSM_CKPT or result[1] != len(payload) or not result[3] or result[2] != sent_hash:
        raise Error("legacy TCP round trip")
    for i in range(len(payload)):
        if received[i] != payload[i]:
            raise Error("legacy TCP bytes")
    var mismatch = header.copy()
    mismatch.payload_len = UInt64(len(payload) + 1)
    var rejected = False
    try:
        _ = ipc.send_tcp_latent(pair[0], mismatch, payload.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), len(payload))
    except _:
        rejected = True
    if not rejected:
        raise Error("legacy TCP length mismatch accepted")
    var store = ipc.LatentStore()
    var store_header = proto.LatentHeader()
    store_header.kind = proto.KIND_SSM_CKPT
    store_header.dtype = proto.DTYPE_F32
    store_header.payload_len = 16
    var minted = ipc.mint_memfd_rw("legacy-store-test", 16)
    var address = store.put(store_header, minted[0])
    _ = sys.sys_munmap(minted[1], 16)
    var duplicate = store.put(store_header, minted[0])
    if duplicate != address:
        raise Error("legacy duplicate address")
    var collision = store_header.copy()
    collision.runtime[0] = 9
    var collision_rejected = False
    try:
        _ = store.put(collision, minted[0])
    except _:
        collision_rejected = True
    if not collision_rejected:
        raise Error("legacy collision accepted")
    if store.release(address):
        raise Error("legacy first release removed live lease")
    if not store.release(address):
        raise Error("legacy final release did not remove entry")
    if store.count() != 0 or store.bytes_retained() != 0:
        raise Error("legacy retained bytes")
    _ = sys.sys_close(pair[0])
    _ = sys.sys_close(pair[1])
    print("IPC_SMALL_PASS")
