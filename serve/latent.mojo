# latent.mojo — LatentOS IPC engine sidecar for mojo-baro (03 §1, §2).
# Implements mint_latent and ingest_latent for byte-exact SSM checkpoints and KV pages.

from std.os import getenv, setenv
from std.memory import Pointer, unsafe_memset, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

import latentos
import latentos.sys as sys
import latentos.proto as proto
import latentos.ipc as ipc

from registry import CONV_SLOT, SSM_SLOT, N_ATT, NKVH, KVPAGE, KVHSTR, KVQ, KVT
from prefix import Checkpoint, Chain, CKPT_BYTES, f32

comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]


def hash64(h: List[UInt8]) -> UInt64:
    """First 8 bytes of a sha256 prefix key, little-endian, as the wire
    header's 64-bit cache key. A shortened key, not the key itself."""
    var v: UInt64 = 0
    for i in range(min(8, len(h))):
        v |= UInt64(h[i]) << UInt64(8 * i)
    return v


# KV cache geometry constants (03 §2)
# 8 layers × 4 heads × (128 tokens * 256 dim) = 1,048,576 floats per page
comptime PGSTR = N_ATT * NKVH * KVHSTR
comptime PAGE_K_BYTES = PGSTR * 4  # 4,194,304 bytes (4 MiB)
comptime PAGE_V_BYTES = PGSTR * 4  # 4,194,304 bytes (4 MiB)
comptime PAGE_KV_BYTES = 2 * PAGE_K_BYTES  # 8,388,608 bytes (8 MiB / 128-token page = 64 KiB / token)

# ==============================================================================
# SSM Checkpoint Mint & Ingest (50.25 MiB)
# ==============================================================================


def mint_checkpoint_latent(
    ckpt: Checkpoint,
    weights_uuid: Array[UInt8, 16] = Array[UInt8, 16](fill=0),
    role_sha: Array[UInt8, 32] = Array[UInt8, 32](fill=0),
    runtime_sha: Array[UInt8, 32] = Array[UInt8, 32](fill=0),
    tokenizer_sha: Array[UInt8, 32] = Array[UInt8, 32](fill=0),
) raises -> Tuple[proto.LatentHeader, Int32]:
    """Mints an immutable, sealed memfd from an in-memory Checkpoint and returns (LatentHeader, fd).
    Mint order (03 §4): write -> munmap -> fcntl(F_ADD_SEALS) -> export.
    """
    if not ckpt.valid:
        raise Error("Cannot mint invalid checkpoint")

    var minted = ipc.mint_memfd_rw("latent-ssm-ckpt", CKPT_BYTES)
    var mfd = minted[0]
    var dst = minted[1]

    # Copy CONV_SLOT (2,359,296 bytes) and SSM_SLOT (50,331,648 bytes)
    var conv_bytes = CONV_SLOT * 4
    var ssm_bytes = SSM_SLOT * 4

    var src_conv = ckpt.conv_h.unsafe_ptr().unsafe_bitcast[UInt8]()
    var src_ssm = ckpt.ssm_h.unsafe_ptr().unsafe_bitcast[UInt8]()

    unsafe_memcpy(dest=dst, src=src_conv, count=conv_bytes)
    unsafe_memcpy(
        dest=dst.unsafe_offset(conv_bytes), src=src_ssm, count=ssm_bytes
    )

    # Seal the memfd before export
    var sealed = ipc.seal_and_finalize(mfd, dst, CKPT_BYTES)
    if not sealed:
        _ = sys.sys_close(mfd)
        raise Error("Failed to seal minted checkpoint memfd")

    # Build 256-byte header
    var header = proto.LatentHeader()
    header.kind = UInt8(proto.KIND_SSM_CKPT)
    header.dtype = UInt8(proto.DTYPE_F32)
    header.pos_hi = UInt32(ckpt.pos)
    header.prefix_hash = hash64(ckpt.hash)
    header.payload_len = UInt64(CKPT_BYTES)
    header.weights_uuid = weights_uuid.copy()
    header.role_sha = role_sha.copy()
    header.runtime = runtime_sha.copy()
    header.tokenizer_sha = tokenizer_sha.copy()

    return (header^, mfd)


def ingest_checkpoint_latent(
    header: proto.LatentHeader,
    fd: Int32,
    mut ckpt: Checkpoint,
) raises:
    """Ingests a sealed memfd into an engine Checkpoint buffer for immediate zero-copy Chain restore.
    """
    if not header.is_valid():
        _ = sys.sys_close(fd)
        raise Error("Invalid LatentHeader in ingest_checkpoint_latent")

    if header.kind != UInt8(proto.KIND_SSM_CKPT):
        _ = sys.sys_close(fd)
        raise Error(
            "Header kind mismatch: expected SSM_CKPT, got " + header.kind_name()
        )

    if header.payload_len != UInt64(CKPT_BYTES):
        _ = sys.sys_close(fd)
        raise Error(
            "Header payload length mismatch: expected "
            + String(CKPT_BYTES)
            + ", got "
            + String(header.payload_len)
        )

    # Zero-copy map of sealed payload
    var src = ipc.map_readonly(fd, CKPT_BYTES)

    var conv_bytes = CONV_SLOT * 4
    var ssm_bytes = SSM_SLOT * 4

    var dst_conv = ckpt.conv_h.unsafe_ptr().unsafe_bitcast[UInt8]()
    var dst_ssm = ckpt.ssm_h.unsafe_ptr().unsafe_bitcast[UInt8]()

    unsafe_memcpy(dest=dst_conv, src=src, count=conv_bytes)
    unsafe_memcpy(
        dest=dst_ssm, src=src.unsafe_offset(conv_bytes), count=ssm_bytes
    )

    # Release mapping and descriptor
    _ = sys.sys_munmap(src, CKPT_BYTES)
    _ = sys.sys_close(fd)

    # Update checkpoint metadata
    ckpt.pos = Int(header.pos_hi)
    # The wire header carries a 64-bit prefix_hash; Checkpoint.hash is the full
    # 32-byte sha256 the prefix cache matches on (b3b4244). 64 bits cannot
    # reconstruct it, so the hash is left EMPTY rather than filled with a
    # truncation that would compare equal to nothing and look like a real key.
    # Consequence, recorded rather than hidden: an ingested checkpoint restores
    # its payload exactly but is NOT reachable by Chain.lookup, whose bytes_eq
    # never matches an empty hash. Carrying the full sha needs a wire change.
    ckpt.hash = List[UInt8]()
    ckpt.set_valid()


def mint_chain_slot(
    chain: Chain,
    slot_idx: Int,
    weights_uuid: Array[UInt8, 16] = Array[UInt8, 16](fill=0),
    role_sha: Array[UInt8, 32] = Array[UInt8, 32](fill=0),
    runtime_sha: Array[UInt8, 32] = Array[UInt8, 32](fill=0),
    tokenizer_sha: Array[UInt8, 32] = Array[UInt8, 32](fill=0),
) raises -> Tuple[proto.LatentHeader, Int32]:
    """Mints a checkpoint from an active slot in Chain."""
    if slot_idx < 0 or slot_idx >= len(chain.items):
        raise Error("Invalid slot_idx in mint_chain_slot: " + String(slot_idx))
    return mint_checkpoint_latent(
        chain.items[slot_idx],
        weights_uuid,
        role_sha,
        runtime_sha,
        tokenizer_sha,
    )


def ingest_into_chain(
    mut chain: Chain,
    header: proto.LatentHeader,
    fd: Int32,
) raises -> Int:
    """Ingests a sealed checkpoint memfd directly into an engine Chain slot, returning the slot index.
    """
    if chain.cap == 0 or len(chain.items) == 0:
        _ = sys.sys_close(fd)
        raise Error("Chain has zero capacity")

    # Match existing pos or find first non-valid slot
    var idx = -1
    for i in range(len(chain.items)):
        if chain.items[i].valid and chain.items[i].pos == Int(header.pos_hi):
            idx = i
            break
    if idx < 0:
        for i in range(len(chain.items)):
            if not chain.items[i].valid and not chain.items[i].pending:
                idx = i
                break
    if idx < 0:
        # Evict oldest (smallest pos)
        idx = 0
        for i in range(1, len(chain.items)):
            if chain.items[i].pos < chain.items[idx].pos:
                idx = i

    chain.gen += 1
    chain.items[idx].gen = chain.gen
    ingest_checkpoint_latent(header, fd, chain.items[idx])
    return idx


# ==============================================================================
# KV Cache Pool Mint & Ingest (64 KiB / token)
# ==============================================================================


def mint_kv_page_host(
    src_k: BytePtr,
    src_v: BytePtr,
    page_idx: Int,
    num_pages: Int = 1,
    prefix_hash: UInt64 = 0,
    weights_uuid: Array[UInt8, 16] = Array[UInt8, 16](fill=0),
    role_sha: Array[UInt8, 32] = Array[UInt8, 32](fill=0),
    runtime_sha: Array[UInt8, 32] = Array[UInt8, 32](fill=0),
    tokenizer_sha: Array[UInt8, 32] = Array[UInt8, 32](fill=0),
) raises -> Tuple[proto.LatentHeader, Int32]:
    """Mints sealed memfd from host-accessible K and V page buffers."""
    var total_bytes = num_pages * PAGE_KV_BYTES
    var minted = ipc.mint_memfd_rw("latent-kv-pages", total_bytes)
    var mfd = minted[0]
    var dst = minted[1]

    var k_bytes = num_pages * PAGE_K_BYTES
    var v_bytes = num_pages * PAGE_V_BYTES

    unsafe_memcpy(dest=dst, src=src_k, count=k_bytes)
    unsafe_memcpy(dest=dst.unsafe_offset(k_bytes), src=src_v, count=v_bytes)

    var sealed = ipc.seal_and_finalize(mfd, dst, total_bytes)
    if not sealed:
        _ = sys.sys_close(mfd)
        raise Error("Failed to seal KV page memfd")

    var header = proto.LatentHeader()
    header.kind = UInt8(proto.KIND_KV_PAGES)
    header.dtype = UInt8(proto.DTYPE_F32)
    header.pos_lo = UInt32(page_idx * KVPAGE)
    header.pos_hi = UInt32((page_idx + num_pages) * KVPAGE)
    header.prefix_hash = prefix_hash
    header.payload_len = UInt64(total_bytes)
    header.weights_uuid = weights_uuid.copy()
    header.role_sha = role_sha.copy()
    header.runtime = runtime_sha.copy()
    header.tokenizer_sha = tokenizer_sha.copy()

    return (header^, mfd)


def ingest_kv_page_host(
    header: proto.LatentHeader,
    fd: Int32,
    dst_k: BytePtr,
    dst_v: BytePtr,
) raises:
    """Ingests a sealed KV page memfd into host-accessible destination buffers.
    """
    if not header.is_valid():
        _ = sys.sys_close(fd)
        raise Error("Invalid LatentHeader in ingest_kv_page_host")

    if header.kind != UInt8(proto.KIND_KV_PAGES):
        _ = sys.sys_close(fd)
        raise Error(
            "Header kind mismatch: expected KV_PAGES, got " + header.kind_name()
        )

    var total_bytes = Int(header.payload_len)
    var num_pages = (Int(header.pos_hi) - Int(header.pos_lo)) // KVPAGE
    if num_pages < 1 or total_bytes != num_pages * PAGE_KV_BYTES:
        _ = sys.sys_close(fd)
        raise Error(
            "Payload length does not match page range: " + String(total_bytes)
        )

    var src = ipc.map_readonly(fd, total_bytes)
    var k_bytes = num_pages * PAGE_K_BYTES
    var v_bytes = num_pages * PAGE_V_BYTES

    unsafe_memcpy(dest=dst_k, src=src, count=k_bytes)
    unsafe_memcpy(dest=dst_v, src=src.unsafe_offset(k_bytes), count=v_bytes)

    _ = sys.sys_munmap(src, total_bytes)
    _ = sys.sys_close(fd)


struct KvLatentStage(Movable):
    """Persistent pinned host stage for KV page mint/ingest (S2: removes per-call hipHostMalloc/Free churn)."""
    var host_k: HostBuffer[KVT]
    var host_v: HostBuffer[KVT]
    var cap_pages: Int

    def __init__(out self, ctx: DeviceContext, cap_pages: Int = 8) raises:
        self.cap_pages = max(1, cap_pages)
        self.host_k = ctx.enqueue_create_host_buffer[KVT](self.cap_pages * PGSTR)
        self.host_v = ctx.enqueue_create_host_buffer[KVT](self.cap_pages * PGSTR)

    def ensure(mut self, ctx: DeviceContext, num_pages: Int) raises:
        if num_pages > self.cap_pages:
            self.cap_pages = num_pages
            self.host_k = ctx.enqueue_create_host_buffer[KVT](self.cap_pages * PGSTR)
            self.host_v = ctx.enqueue_create_host_buffer[KVT](self.cap_pages * PGSTR)


def _get_persistent_kv_stage(
    ctx: DeviceContext, num_pages: Int
) raises -> Pointer[KvLatentStage, MutUntrackedOrigin]:
    var env_s = getenv("_BARO_KV_STAGE_PTR", "")
    if env_s != "":
        var addr = Int(atol(env_s))
        if addr != 0:
            var p = Pointer[KvLatentStage, MutUntrackedOrigin](unsafe_from_address=addr)
            p[].ensure(ctx, num_pages)
            return p
    var p = unsafe_alloc[KvLatentStage](1)
    p.unsafe_write(KvLatentStage(ctx, max(8, num_pages)))
    _ = setenv("_BARO_KV_STAGE_PTR", String(Int(p)), True)
    return p


def mint_kv_latent(
    ctx: DeviceContext,
    kc_d: DeviceBuffer[KVT],
    vc_d: DeviceBuffer[KVT],
    page_idx: Int,
    num_pages: Int = 1,
    prefix_hash: UInt64 = 0,
    weights_uuid: Array[UInt8, 16] = Array[UInt8, 16](fill=0),
    role_sha: Array[UInt8, 32] = Array[UInt8, 32](fill=0),
    runtime_sha: Array[UInt8, 32] = Array[UInt8, 32](fill=0),
    tokenizer_sha: Array[UInt8, 32] = Array[UInt8, 32](fill=0),
) raises -> Tuple[proto.LatentHeader, Int32]:
    """Extracts KV cache pages from device memory into a sealed LatentOS memfd."""
    comptime if KVT != DType.float32:
        raise Error(
            "LatentOS KV pages carry f32 KV; this engine has BARO_KVQ=" + KVQ
        )
    var total_floats = num_pages * PGSTR
    var stage = _get_persistent_kv_stage(ctx, num_pages)
    var host_k = stage[].host_k.create_sub_buffer[KVT](0, total_floats)
    var host_v = stage[].host_v.create_sub_buffer[KVT](0, total_floats)

    var kc_slice = DeviceBuffer[KVT](
        ctx,
        kc_d.unsafe_ptr().unsafe_offset(page_idx * PGSTR),
        total_floats,
        owning=False,
    )
    var vc_slice = DeviceBuffer[KVT](
        ctx,
        vc_d.unsafe_ptr().unsafe_offset(page_idx * PGSTR),
        total_floats,
        owning=False,
    )

    ctx.enqueue_copy(dst_buf=host_k, src_buf=kc_slice)
    ctx.enqueue_copy(dst_buf=host_v, src_buf=vc_slice)
    ctx.synchronize()

    var src_k = host_k.unsafe_ptr().unsafe_bitcast[UInt8]()
    var src_v = host_v.unsafe_ptr().unsafe_bitcast[UInt8]()

    return mint_kv_page_host(
        src_k,
        src_v,
        page_idx,
        num_pages,
        prefix_hash,
        weights_uuid,
        role_sha,
        runtime_sha,
        tokenizer_sha,
    )


def ingest_kv_latent(
    ctx: DeviceContext,
    mut kc_d: DeviceBuffer[KVT],
    mut vc_d: DeviceBuffer[KVT],
    header: proto.LatentHeader,
    fd: Int32,
) raises:
    """Ingests a sealed LatentOS KV page memfd directly into GPU attention KV cache."""
    comptime if KVT != DType.float32:
        raise Error(
            "LatentOS KV pages carry f32 KV; this engine has BARO_KVQ=" + KVQ
        )
    if not header.is_valid() or header.kind != UInt8(proto.KIND_KV_PAGES):
        if fd >= 0:
            _ = sys.sys_close(fd)
        raise Error("Invalid LatentHeader in ingest_kv_latent")

    var page_idx = Int(header.pos_lo) // KVPAGE
    var num_pages = (Int(header.pos_hi) - Int(header.pos_lo)) // KVPAGE
    var total_floats = num_pages * PGSTR

    var stage = _get_persistent_kv_stage(ctx, num_pages)
    var host_k = stage[].host_k.create_sub_buffer[KVT](0, total_floats)
    var host_v = stage[].host_v.create_sub_buffer[KVT](0, total_floats)

    var dst_k = host_k.unsafe_ptr().unsafe_bitcast[UInt8]()
    var dst_v = host_v.unsafe_ptr().unsafe_bitcast[UInt8]()

    ingest_kv_page_host(header, fd, dst_k, dst_v)

    var kc_slice = DeviceBuffer[KVT](
        ctx,
        kc_d.unsafe_ptr().unsafe_offset(page_idx * PGSTR),
        total_floats,
        owning=False,
    )
    var vc_slice = DeviceBuffer[KVT](
        ctx,
        vc_d.unsafe_ptr().unsafe_offset(page_idx * PGSTR),
        total_floats,
        owning=False,
    )

    ctx.enqueue_copy(dst_buf=kc_slice, src_buf=host_k)
    ctx.enqueue_copy(dst_buf=vc_slice, src_buf=host_v)
    ctx.synchronize()


# ==============================================================================
# Engine IPC Client Daemon Connection
# ==============================================================================


struct EngineLatentClient:
    """IPC client in the engine process talking to latentos-agent."""

    var agent_sock: Int32

    def __init__(out self, agent_sock: Int32):
        self.agent_sock = agent_sock

    def export_chain_slot(self, chain: Chain, slot_idx: Int) raises -> Bool:
        """Mints a checkpoint from Chain and exports it to latentos-agent."""
        var res = mint_chain_slot(chain, slot_idx)
        var header = res[0].copy()
        var fd = res[1]
        var ok = ipc.send_handle(self.agent_sock, header, fd)
        _ = sys.sys_close(fd)
        return ok
