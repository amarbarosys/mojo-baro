"""Mint/ingest round trip for the LatentOS sidecar (serve/latent.mojo).

Exists because the sidecar shipped broken: Checkpoint.hash became a 32-byte
sha256 at b3b4244 and serve/latent.mojo, merged at 506e91d from a branch that
predated it, assigned that List[UInt8] into the wire header's UInt64. Nothing
built the file, so it stayed broken until 2e7b5d3.

Building tools/latent-recv.mojo does NOT catch that class: Mojo type-checks
lazily per reached symbol, and the receiver imports only the ingest side, so a
break inside mint_checkpoint_latent compiles clean. This test names BOTH
directions, so the whole path is type-checked, and then checks the bytes.

    ./.venv/bin/mojo build kernels/test_latent.mojo -I kernels -I serve \
        -o .work/test_latent && ./.work/test_latent
"""
from std.sys import exit
from max.gpu.host import DeviceContext

from latent import mint_checkpoint_latent, ingest_checkpoint_latent
from prefix import Checkpoint, CONV_SLOT, SSM_SLOT, f32


def main() raises:
    var ctx = DeviceContext()
    var fails = 0

    var conv = ctx.enqueue_create_host_buffer[f32](CONV_SLOT)
    var ssm = ctx.enqueue_create_host_buffer[f32](SSM_SLOT)
    var kv = ctx.enqueue_create_host_buffer[f32](0)
    ctx.synchronize()

    # A pattern that is not all-zero and not constant, so a memcpy that copies
    # the wrong region or the wrong length cannot pass by accident.
    for i in range(CONV_SLOT):
        conv[i] = Float32(i % 977) * 0.5 - 3.0
    for i in range(SSM_SLOT):
        ssm[i] = Float32((i * 7) % 1319) * -0.25 + 1.0

    var src = Checkpoint(pos=41, hash=List[UInt8](), gen=1, valid=True,
                         pending=False, pinned=False, boundary=True,
                         conv_h=conv^, ssm_h=ssm^, kv_h=kv^, pages=List[Int32](), hits=0)
    for b in range(32):
        src.hash.append(UInt8((b * 31 + 7) % 256))

    var res = mint_checkpoint_latent(src)
    var header = res[0].copy()
    var fd = res[1]
    if fd < 0:
        print("FAIL: mint returned fd", fd)
        exit(1)
    if not header.is_valid():
        print("FAIL: minted header is not valid")
        exit(1)

    var dconv = ctx.enqueue_create_host_buffer[f32](CONV_SLOT)
    var dssm = ctx.enqueue_create_host_buffer[f32](SSM_SLOT)
    var dkv = ctx.enqueue_create_host_buffer[f32](0)
    ctx.synchronize()
    var dst = Checkpoint(pos=0, hash=List[UInt8](), gen=0, valid=False,
                         pending=False, pinned=False, boundary=False,
                         conv_h=dconv^, ssm_h=dssm^, kv_h=dkv^, pages=List[Int32](), hits=0)
    ingest_checkpoint_latent(header, fd, dst)

    if dst.pos != src.pos:
        print("FAIL: pos", dst.pos, "!=", src.pos)
        fails += 1
    var bad = 0
    for i in range(CONV_SLOT):
        if dst.conv_h[i] != src.conv_h[i]:
            bad += 1
    if bad > 0:
        print("FAIL: conv differs in", bad, "of", CONV_SLOT)
        fails += 1
    bad = 0
    for i in range(SSM_SLOT):
        if dst.ssm_h[i] != src.ssm_h[i]:
            bad += 1
    if bad > 0:
        print("FAIL: ssm differs in", bad, "of", SSM_SLOT)
        fails += 1

    # Documented, deliberate: 64 bits on the wire cannot carry the 256-bit
    # prefix key, so ingest leaves it empty instead of fabricating one. If this
    # ever starts matching, the wire format changed and Chain.lookup's
    # behaviour changed with it.
    if len(dst.hash) != 0:
        print("FAIL: ingested hash should be empty, got", len(dst.hash), "bytes")
        fails += 1

    if fails > 0:
        print("latent round trip FAILED with", fails, "problems")
        exit(1)
    print("PASS: latent mint/ingest round trip, conv", CONV_SLOT, "ssm", SSM_SLOT, "floats exact")
