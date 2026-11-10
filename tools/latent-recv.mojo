"""L1 gate receiver (bench/chat-protocol.md). Listens on a unix socket, takes
N checkpoint handles from the engine, ingests each one and writes the ingested
conv+ssm payload to disk so the gate can compare it byte for byte against what
the engine minted.

This is NOT latentos-agent: it speaks the same ipc/proto wire, and nothing
more. Interop with the production daemon is unproven by this gate.

    tools/latent-recv.mojo SOCKPATH N OUTDIR
"""
from std.sys import argv, exit
from max.gpu.host import DeviceContext

import latentos.ipc as ipc
import latentos.sys as sys
from latent import ingest_checkpoint_latent
from prefix import Checkpoint, CONV_SLOT, SSM_SLOT, f32
from std.collections import Span


def main() raises:
    var a = argv()
    if len(a) != 4:
        print("usage: latent-recv SOCKPATH N OUTDIR")
        exit(2)
    var path = String(a[1])
    var want = Int(String(a[2]))
    var outdir = String(a[3])

    var lfd = ipc.create_unix_listener(path)
    if lfd < 0:
        print("FAIL: cannot listen on", path)
        exit(1)
    print("recv: listening on", path, "for", want, "handles")

    var ctx = DeviceContext()
    var conv = ctx.enqueue_create_host_buffer[f32](CONV_SLOT)
    var ssm = ctx.enqueue_create_host_buffer[f32](SSM_SLOT)
    ctx.synchronize()
    var ckpt = Checkpoint(pos=0, hash=List[UInt8](), gen=0, valid=False,
                          pending=False, pinned=False, boundary=False,
                          conv_h=conv^, ssm_h=ssm^,
                          kv_h=ctx.enqueue_create_host_buffer[f32](1),
                          pages=List[Int32](), hits=0)

    var sock = sys.sys_accept(lfd)
    if sock < 0:
        print("FAIL: accept")
        exit(1)

    var got = 0
    while got < want:
        var res = ipc.recv_handle(sock)
        var header = res[0].copy()
        var fd = res[1]
        if fd < 0:
            print("FAIL: recv_handle returned fd", fd, "after", got, "handles")
            exit(1)
        if not header.is_valid():
            print("FAIL: invalid LatentHeader on handle", got)
            exit(1)
        ingest_checkpoint_latent(header, fd, ckpt)
        with open(outdir + "/recv-" + String(got) + ".bin", "w") as f:
            f.write_bytes(Span[UInt8](unsafe_ptr=ckpt.conv_h.unsafe_ptr().unsafe_bitcast[UInt8](), length=CONV_SLOT * 4))
            f.write_bytes(Span[UInt8](unsafe_ptr=ckpt.ssm_h.unsafe_ptr().unsafe_bitcast[UInt8](), length=SSM_SLOT * 4))
        print("recv: handle", got, "pos", ckpt.pos, "payload_len", header.payload_len)
        got += 1

    print("PASS: received", got, "handles")
    _ = sys.sys_close(sock)
    _ = sys.sys_close(lfd)
