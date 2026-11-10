"""KV page and checkpoint handoff between engine processes over HIP IPC (Teleport KV T3a, T3b).

The memfd path (latent.mojo mint_kv_latent / ingest_kv_latent) crosses PCIe twice: device to
pinned host, memfd, pinned host to device. Here process A copies the pages into a raw hipMalloc
stage (an IPC handle names a whole allocation, and MAX may sub-allocate small buffers from its
pool), exports the 64-byte handle, and keeps the stage alive until B acks. Process B opens the
handle through libamarbaro_shim (Mojo cannot pass the 64-byte handle by value) and copies with a
kernel: a kernel reads the mapped region at local VRAM speed (430 GB/s), hipMemcpy from it ran
42 GB/s (exchange/receipts/teleport-kv/).

T3b (AMDHQ 06-experiments `## T3b`): a checkpoint travels as one stage laid out
`[K pages 0..n-1][V pages 0..n-1][conv f32][ssm f32]` (logical page order) plus a 320-byte message,
the 256-byte LatentHeader (kind KV_PAGES, pos, prefix_hash, role_sha = the pack salt) and the
handle. B checks the header against its own pack salt and the prefix tokens before any HIP call.

The caller owns the ordering: A must not refill or free the stage until B has acked its ingest.

HIP and the shim are reached through dlopen, so the engine's link line is unchanged:
libamdhip64 from /opt/rocm/lib, the shim from BARO_SHIM_LIB (default: the soname, found when the
binary links it, as bench/kv_ipc_bench.mojo does).
"""
from std.ffi import OwnedDLHandle
from std.os import getenv
from std.math import ceildiv
from std.memory import unsafe_memcpy
from std.memory.alloc import unsafe_alloc

from max.gpu import global_idx, block_dim, grid_dim
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

import latentos.proto as proto
from registry import KVT, KVPAGE, KVHSTR, HD, N_ATT, NKVH, CONV_SLOT, SSM_SLOT
from prefix import Chain, prefix_hash, f32, _hex_to_bytes
from latent import PGSTR, hash64
from sha256 import sha256

comptime HANDLE_BYTES = 64
comptime MSG_BYTES = proto.LATENT_HEADER_SIZE + HANDLE_BYTES
comptime STATE_BYTES = (CONV_SLOT + SSM_SLOT) * 4
comptime hipIpcMemLazyEnablePeerAccess: UInt32 = 1
comptime HIP_LIB = "/opt/rocm/lib/libamdhip64.so"


def _hip(rc: Int32, what: String) raises:
    if rc != 0:
        raise Error(what + " failed, hipError " + String(rc))


def _at[dt: DType](addr: Int) -> UnsafePointer[Scalar[dt], MutAnyOrigin]:
    return UnsafePointer[Scalar[dt], MutAnyOrigin](unsafe_from_address=addr)


def stage_alloc(nbytes: Int, handle: Pointer[UInt8, MutUntrackedOrigin]) raises -> Int:
    """hipMalloc `nbytes` and write its 64-byte IPC handle into `handle`; returns the device address."""
    var hip = OwnedDLHandle(HIP_LIB)
    var addr = Int(0)
    _hip(hip.get_function[Int32]("hipMalloc")(MutPointer(to=addr), UInt64(nbytes)), "hipMalloc stage")
    var rc = hip.get_function[Int32]("hipIpcGetMemHandle")(handle, addr)
    if rc != 0:
        _ = hip.get_function[Int32]("hipFree")(addr)
        _hip(rc, "hipIpcGetMemHandle")
    return addr


def stage_new(nbytes: Int) raises -> Tuple[Int, List[UInt8]]:
    """A reusable export stage: (device address, 64-byte IPC handle)."""
    var handle = List[UInt8](length=HANDLE_BYTES, fill=0)
    var addr = stage_alloc(nbytes, Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(handle.unsafe_ptr())))
    return (addr, handle^)


def host_mapped(nbytes: Int) raises -> Tuple[Int, Int]:
    """Pinned host memory kernels read in place (hipHostMalloc, mapped): (host address, device
    address). Never freed: it holds pack tensors for the life of the process."""
    var hip = OwnedDLHandle(HIP_LIB)
    var h = Int(0)
    var d = Int(0)
    _hip(hip.get_function[Int32]("hipHostMalloc")(MutPointer(to=h), UInt64(nbytes), UInt32(2)), "hipHostMalloc")
    _hip(hip.get_function[Int32]("hipHostGetDevicePointer")(MutPointer(to=d), h, UInt32(0)), "hipHostGetDevicePointer")
    return (h, d)


def stage_free(addr: Int) raises:
    var hip = OwnedDLHandle(HIP_LIB)
    _hip(hip.get_function[Int32]("hipFree")(addr), "hipFree stage")


def _ipc_open(handle: Pointer[UInt8, MutUntrackedOrigin]) raises -> Int:
    var shim = OwnedDLHandle(getenv("BARO_SHIM_LIB", "libamarbaro_shim.so"))
    var src = Int(0)
    _hip(shim.get_function[Int32]("amarbaro_ipc_open")(MutPointer(to=src), handle, hipIpcMemLazyEnablePeerAccess), "hipIpcOpenMemHandle")
    return src


def _ipc_close(src: Int) raises:
    var hip = OwnedDLHandle(HIP_LIB)
    _hip(hip.get_function[Int32]("hipIpcCloseMemHandle")(src), "hipIpcCloseMemHandle")


struct KvIpcStage(Movable):
    """A's side: one raw device allocation holding K then V for `num_pages` pages, and its handle."""
    var ptr: Int
    var floats: Int
    var handle: Pointer[UInt8, MutUntrackedOrigin]

    def __init__(out self, num_pages: Int) raises:
        self.floats = num_pages * PGSTR
        self.handle = unsafe_alloc[UInt8](HANDLE_BYTES)
        self.ptr = stage_alloc(2 * self.floats * 4, self.handle)

    def __deinit__(deinit self):
        try:
            stage_free(self.ptr)
        except:
            pass
        self.handle.free()

    def fill(self, ctx: DeviceContext, kc_d: DeviceBuffer[KVT], vc_d: DeviceBuffer[KVT], page_idx: Int) raises:
        comptime if KVT != DType.float32:
            raise Error("KV IPC carries f32 KV pages")
        var k_src = Int(kc_d.unsafe_ptr().unsafe_offset(page_idx * PGSTR))
        var v_src = Int(vc_d.unsafe_ptr().unsafe_offset(page_idx * PGSTR))
        ctx.enqueue_function[_copy_f32](Int64(k_src), Int64(self.ptr), Int64(self.floats // 4), grid_dim=1920, block_dim=256)
        ctx.enqueue_function[_copy_f32](Int64(v_src), Int64(self.ptr + self.floats * 4), Int64(self.floats // 4), grid_dim=1920, block_dim=256)
        ctx.synchronize()


def _copy_f32(src: Int64, dst: Int64, n4: Int64):
    var s = MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(src))
    var d = MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(dst))
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < Int(n4):
        d.unsafe_store[width=4](i * 4, s.unsafe_load[width=4](i * 4))
        i += stride


def ingest_kv_ipc(
    ctx: DeviceContext, kc_d: DeviceBuffer[KVT], vc_d: DeviceBuffer[KVT],
    page_idx: Int, num_pages: Int, handle: Pointer[UInt8, MutUntrackedOrigin],
) raises:
    """B's side: map A's stage, copy K and V into this engine's pages with a kernel, unmap."""
    comptime assert PGSTR % 4 == 0, "_copy_f32 moves vec4s"
    comptime if KVT != DType.float32:
        raise Error("KV IPC carries f32 KV pages")
    var floats = num_pages * PGSTR
    var src = _ipc_open(handle)
    var k_dst = Int(kc_d.unsafe_ptr().unsafe_offset(page_idx * PGSTR))
    var v_dst = Int(vc_d.unsafe_ptr().unsafe_offset(page_idx * PGSTR))
    try:
        ctx.enqueue_function[_copy_f32](Int64(src), Int64(k_dst), Int64(floats // 4), grid_dim=1920, block_dim=256)
        ctx.enqueue_function[_copy_f32](Int64(src + floats * 4), Int64(v_dst), Int64(floats // 4), grid_dim=1920, block_dim=256)
        ctx.synchronize()
    except e:
        _ipc_close(src)
        raise e^
    _ipc_close(src)


# ---- T3b: a whole checkpoint (KV pages + conv/SSM slots) --------------------


def ipc_payload_bytes(pos: Int) -> Int:
    return 2 * ceildiv(pos, KVPAGE) * PGSTR * 4 + STATE_BYTES


def hex_encode(b: Span[UInt8, _]) -> String:
    comptime digits = "0123456789abcdef"
    var s = String(capacity_bytes=2 * len(b))
    for x in b:
        s += digits[byte=Int(x >> 4)]
        s += digits[byte=Int(x & 0x0F)]
    return s^


def tokens_from_hex(s: String) -> List[Int]:
    """Little-endian u32 ids, 8 hex digits each (the front's `ipc_prefix`)."""
    var raw = _hex_to_bytes(s)
    var toks = List[Int](capacity=len(raw) // 4)
    for t in range(len(raw) // 4):
        toks.append(Int(raw[4 * t]) | (Int(raw[4 * t + 1]) << 8) | (Int(raw[4 * t + 2]) << 16) | (Int(raw[4 * t + 3]) << 24))
    return toks^


def ipc_header(chain: Chain, idx: Int) -> proto.LatentHeader:
    var h = proto.LatentHeader()
    h.kind = UInt8(proto.KIND_KV_PAGES)
    h.dtype = UInt8(proto.DTYPE_F32)
    h.pos_hi = UInt32(chain.items[idx].pos)
    h.prefix_hash = hash64(chain.items[idx].hash)
    h.payload_len = UInt64(ipc_payload_bytes(chain.items[idx].pos))
    for i in range(min(32, len(chain.salt))):
        h.role_sha[i] = chain.salt[i]
    return h^


def ipc_refusal(h: proto.LatentHeader, salt: List[UInt8], tokens: List[Int], tmax: Int) -> String:
    """Empty when B may open the handle; otherwise the refusal (front: 409). Touches no HIP."""
    if not h.is_valid() or h.kind != UInt8(proto.KIND_KV_PAGES) or h.dtype != UInt8(proto.DTYPE_F32):
        return "IPC refused: not an f32 KV_PAGES LatentHeader"
    if len(salt) < 32:
        return "IPC refused: saved from a different pack"
    for i in range(32):
        if h.role_sha[i] != salt[i]:
            return "IPC refused: saved from a different pack"
    var pos = Int(h.pos_hi)
    if pos < 1 or pos >= tmax or pos != len(tokens):
        return "IPC refused: pos " + String(pos) + " vs " + String(len(tokens)) + " prefix tokens, TMAX " + String(tmax)
    if h.payload_len != UInt64(ipc_payload_bytes(pos)):
        return "IPC refused: payload_len " + String(h.payload_len) + " != " + String(ipc_payload_bytes(pos))
    if h.prefix_hash != hash64(prefix_hash(salt, tokens, pos)):
        return "IPC refused: prefix_hash mismatch"
    return ""


comptime WEIGHTS_MAGIC = "BAROWT01"
comptime WEIGHTS_RECORD_BYTES = 8 + 8 + 32 + HANDLE_BYTES


def weights_record(total: Int, salt: List[UInt8], handle: List[UInt8]) -> List[UInt8]:
    """T3b step 2, the file the weight exporter writes: magic | int64 LE pack bytes | pack salt | handle."""
    var rec = List[UInt8](capacity=WEIGHTS_RECORD_BYTES)
    var magic = StringSlice(WEIGHTS_MAGIC).as_bytes()
    for i in range(8):
        rec.append(magic[i])
    for i in range(8):
        rec.append(UInt8((total >> (8 * i)) & 0xFF))
    for i in range(32):
        rec.append(salt[i] if i < len(salt) else 0)
    for i in range(HANDLE_BYTES):
        rec.append(handle[i])
    return rec^


def weights_refusal(rec: List[UInt8], salt: List[UInt8], total: Int) -> String:
    """Empty when this engine may map the weights in `rec`; otherwise why not. Touches no HIP."""
    if len(rec) != WEIGHTS_RECORD_BYTES:
        return "weights import refused: record is " + String(len(rec)) + " bytes, not " + String(WEIGHTS_RECORD_BYTES)
    var magic = StringSlice(WEIGHTS_MAGIC).as_bytes()
    for i in range(8):
        if rec[i] != magic[i]:
            return "weights import refused: not a BAROWT01 record"
    var n = 0
    for i in range(8):
        n |= Int(rec[8 + i]) << (8 * i)
    if n != total:
        return "weights import refused: exporter holds " + String(n) + " bytes, this pack is " + String(total)
    if len(salt) < 32:
        return "weights import refused: exported from a different pack"
    for i in range(32):
        if rec[16 + i] != salt[i]:
            return "weights import refused: exported from a different pack"
    return ""


def weights_export(path: String, total: Int, salt: List[UInt8], handle: List[UInt8]) raises:
    var rec = weights_record(total, salt, handle)
    with open(path, "w") as f:
        f.write_bytes(Span(rec))


def weights_import(path: String, salt: List[UInt8], total: Int) raises -> Int:
    """Map the exporter's weight buffer; raises before any HIP call on a record for another pack."""
    var rec: List[UInt8]
    with open(path, "r") as f:
        rec = f.read_bytes()
    var why = weights_refusal(rec, salt, total)
    if why != "":
        raise Error(why + " (" + path + ")")
    return _ipc_open(Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(rec.unsafe_ptr()) + 16 + 32))


def export_ckpt_ipc(
    ctx: DeviceContext, chain: Chain, idx: Int, kc_d: DeviceBuffer[KVT], vc_d: DeviceBuffer[KVT],
    kvtab_h: HostBuffer[DType.int32], addr: Int, handle: List[UInt8],
) raises -> List[UInt8]:
    """A's side: copy chain item `idx` (its pages through the block table, then conv and SSM) into
    the stage at `addr` (at least ipc_payload_bytes(pos)) and return the 320-byte message.
    Stages are reused, never freed: on ROCm an allocation exported with hipIpcGetMemHandle is not
    reclaimed by hipFree (bench/t3b-ipc-free-probe.c: hipMalloc fails after 44 x 0.58 GB)."""
    comptime if KVT != DType.float32:
        raise Error("KV IPC carries f32 KV pages")
    var pos = chain.items[idx].pos
    var npages = ceildiv(pos, KVPAGE)
    var h = ipc_header(chain, idx)
    var msg = List[UInt8](length=MSG_BYTES, fill=0)
    h.serialize(Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(msg.unsafe_ptr())))
    for i in range(HANDLE_BYTES):
        msg[proto.LATENT_HEADER_SIZE + i] = handle[i]
    var v0 = addr + npages * PGSTR * 4
    var s0 = addr + 2 * npages * PGSTR * 4
    for p in range(npages):
        var pp = Int(chain.page_of(idx, p)) if chain.npages > 0 else Int(kvtab_h[p])
        ctx.enqueue_copy(dst_buf=DeviceBuffer[KVT](ctx, _at[KVT](addr + p * PGSTR * 4), PGSTR, owning=False), src_buf=DeviceBuffer[KVT](ctx, kc_d.unsafe_ptr().unsafe_offset(pp * PGSTR), PGSTR, owning=False))
        ctx.enqueue_copy(dst_buf=DeviceBuffer[KVT](ctx, _at[KVT](v0 + p * PGSTR * 4), PGSTR, owning=False), src_buf=DeviceBuffer[KVT](ctx, vc_d.unsafe_ptr().unsafe_offset(pp * PGSTR), PGSTR, owning=False))
    ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, _at[f32](s0), CONV_SLOT, owning=False), src_buf=chain.items[idx].conv_h)
    ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, _at[f32](s0 + CONV_SLOT * 4), SSM_SLOT, owning=False), src_buf=chain.items[idx].ssm_h)
    ctx.synchronize()
    return msg^


def ingest_ckpt_ipc(
    ctx: DeviceContext, mut chain: Chain, kc_d: DeviceBuffer[KVT], vc_d: DeviceBuffer[KVT],
    kvtab_h: HostBuffer[DType.int32], mut kvtab_d: DeviceBuffer[DType.int32], tpages: Int,
    msg: List[UInt8], tokens: List[Int],
) raises -> Int:
    if chain.npages > 0:
        chain.share_reset(ctx, kvtab_h, kvtab_d, tpages)
    """B's side, after ipc_refusal returned empty: open A's stage, kernel-copy K and V into pool
    pages 0..n-1 (identity block table, as load_state lands a file), copy conv and SSM into a free
    chain item that then carries this prefix's hash, close. Returns the item index."""
    comptime assert PGSTR % 4 == 0, "_copy_f32 moves vec4s"
    comptime if KVT != DType.float32:
        raise Error("KV IPC carries f32 KV pages")
    if chain.cap == 0:
        raise Error("IPC ingest needs a checkpoint slot")
    var h = proto.LatentHeader.deserialize(Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(msg.unsafe_ptr())))
    var pos = Int(h.pos_hi)
    var floats = ceildiv(pos, KVPAGE) * PGSTR
    var idx = -1
    for i in range(len(chain.items)):
        if chain.items[i].valid and chain.items[i].pos == pos:
            idx = i
    if idx < 0:
        idx = 0
        for i in range(len(chain.items)):
            if not chain.items[i].valid:
                idx = i
                break
    var src = _ipc_open(Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(msg.unsafe_ptr()) + proto.LATENT_HEADER_SIZE))
    try:
        ctx.enqueue_function[_copy_f32](Int64(src), Int64(Int(kc_d.unsafe_ptr())), Int64(floats // 4), grid_dim=1920, block_dim=256)
        ctx.enqueue_function[_copy_f32](Int64(src + floats * 4), Int64(Int(vc_d.unsafe_ptr())), Int64(floats // 4), grid_dim=1920, block_dim=256)
        var s0 = src + 2 * floats * 4
        ctx.enqueue_copy(dst_buf=chain.items[idx].conv_h, src_buf=DeviceBuffer[f32](ctx, _at[f32](s0), CONV_SLOT, owning=False))
        ctx.enqueue_copy(dst_buf=chain.items[idx].ssm_h, src_buf=DeviceBuffer[f32](ctx, _at[f32](s0 + CONV_SLOT * 4), SSM_SLOT, owning=False))
        ctx.synchronize()
    except e:
        _ipc_close(src)
        raise e^
    _ipc_close(src)
    chain.install(idx, pos, prefix_hash(chain.salt, tokens, pos), True, True)
    for i in range(tpages):
        kvtab_h[i] = Int32(i)
    ctx.enqueue_copy(dst_buf=kvtab_d, src_buf=kvtab_h)
    ctx.synchronize()
    return idx


def ckpt_region_sha(
    ctx: DeviceContext, chain: Chain, idx: Int, kc_d: DeviceBuffer[KVT], vc_d: DeviceBuffer[KVT],
    kvtab_h: HostBuffer[DType.int32],
) raises -> String:
    """sha256 over conv, SSM, then K and V of tokens [0, pos) per (page, layer, head) row block:
    exactly the bytes a T3b handoff moves that a later fork does not overwrite (it writes >= pos)."""
    comptime if KVT != DType.float32:
        raise Error("KV IPC carries f32 KV pages")
    var pos = chain.items[idx].pos
    var kvn = pos * N_ATT * NKVH * HD
    var buf = ctx.enqueue_create_host_buffer[KVT](CONV_SLOT + SSM_SLOT + 2 * kvn)
    ctx.synchronize()
    var b8 = buf.unsafe_ptr().unsafe_bitcast[UInt8]()
    unsafe_memcpy(dest=b8, src=chain.items[idx].conv_h.unsafe_ptr().unsafe_bitcast[UInt8](), count=CONV_SLOT * 4)
    unsafe_memcpy(dest=b8.unsafe_offset(CONV_SLOT * 4), src=chain.items[idx].ssm_h.unsafe_ptr().unsafe_bitcast[UInt8](), count=SSM_SLOT * 4)
    var off = CONV_SLOT + SSM_SLOT
    for which in range(2):
        var base = kc_d.unsafe_ptr() if which == 0 else vc_d.unsafe_ptr()
        for p in range(ceildiv(pos, KVPAGE)):
            var rows = min(KVPAGE, pos - p * KVPAGE)
            var pp = Int(chain.page_of(idx, p)) if chain.npages > 0 else Int(kvtab_h[p])
            for a in range(N_ATT):
                for kh in range(NKVH):
                    var e0 = ((pp * N_ATT + a) * NKVH + kh) * KVHSTR
                    ctx.enqueue_copy(dst_buf=buf.create_sub_buffer[KVT](off, rows * HD), src_buf=DeviceBuffer[KVT](ctx, base.unsafe_offset(e0), rows * HD, owning=False))
                    off += rows * HD
    ctx.synchronize()
    var d = sha256(Span[UInt8](unsafe_ptr=b8, length=off * 4))
    _ = buf.unsafe_ptr()
    return hex_encode(Span[UInt8](unsafe_ptr=d.unsafe_ptr(), length=32))
