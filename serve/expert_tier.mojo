"""B4 stage 2b: routed experts in host RAM, an LRU of resident experts in VRAM.

Preregistered in `bench/moe-tier-protocol.md`. The trunk stays in VRAM; the
120 routed expert tensors (40 layers x gate/up/down) live in a split pack's
`experts.bin` (`tools/pack-split-experts.py`) and are fetched into a VRAM
cache on demand.

Why this needs no kernel change, which is the whole design:

  `moe_gate_up_q4k_pack` reads expert `e` at `e * FFN * row_bytes` from a base
  pointer it is handed, and its gate-to-up distance is a RUNTIME argument
  (`up_offset`). `amar_moe_down_q4k` reads `e * N * row_bytes` from its own
  base. Neither knows how many experts exist. So a cache holding `cap`
  experts in the same inner layout, a base pointer into it, an `up_offset` of
  `cap * per_expert_bytes`, and an index array holding SLOTS instead of
  expert ids runs both kernels unchanged.

Cache layout per layer, one contiguous block:

  [ gate: cap x eb ][ up: cap x eb ][ down: cap x ebd ]

`eb` is 589,824 bytes for this pack's gate and up; `ebd` is 589,824 (q4_k) or
860,160 (q6_k: layers 34, 38, 39). At cap 64 that is 113 MB per layer and
4.53 GB over 40 layers, against 18.1 GB for all 256.

The host store is PINNED by default since 2026-09-17 (the maintainer's decision after
stage 3, `exchange/lane-MOE3-report.md`): with the page cache, every miss
does a blocking pread into the staging buffer before its copy, 55% of the
tier's host time, and the measured cost was 48.57 against 67.12 tok/s pinned,
identical output on all 20 prompts. The DMA rate itself is the same either
way (stage 1: 28.78 vs 28.60 GB/s). `BARO_TIER_PINNED=0` reads through the
page cache instead when the 18 GB of locked RAM is not affordable, and the
tier prints which one it is using, because that is an arm-defining
parameter (P1).

A miss costs a host round trip: the top-8 is only known after the router
runs, so the ids come back to the host, the LRU decides, and the fetches are
enqueued before the gate/up launch. That is inherent to demand fetching and is
what stage 3's prefetch exists to remove.
"""
from std.collections import Dict, List
from std.atomic import Ordering, fence
from std.ffi import c_ssize_t, external_call, OwnedDLHandle
from std.memory import UnsafePointer, alloc, bitcast
from std.os import getenv
from std.time import perf_counter_ns

from max.algorithm import parallelize
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer, DeviceEvent

from moe_pack import (
    TensorInfo, parse_moe_index, tensor_byte_size, resolve_plain, N_EXP, TOPK, QK_K, Q4_K_BYTES,
)
from expert_cpu import expert_ffn_cpu
from cpu_worker import CpuJob, cpu_worker_start, cpu_worker_submit, cpu_worker_done
from doorbell import tier_bell_out, tier_bell_in, tier_wait_add
import latentos.sys as sys
import model_qwen35moe as P

# One staging slot per piece a single layer can fetch: 8 experts x gate, up,
# down. With a slot per piece nothing in flight is overwritten, so the copies
# need no synchronize between them; the next layer's prepare() already syncs to
# read the router's ids back, which is what makes the previous layer's fetches
# safe to overwrite. Syncing per piece instead cost 720 syncs per token and
# 13.6 tok/s on the first live run.
comptime STAGE_SLOTS = 24
comptime SLOT_BYTES = 1 << 20  # 1 MiB, above this pack's 860,160-byte q6_k expert
comptime STAGE_BYTES = STAGE_SLOTS * SLOT_BYTES
# BARO_TIER_CPU shapes (qwen35moe); the tier checks them against the pack's
# expert bytes at startup.
comptime CPU_H = P.H
comptime CPU_FFN = P.FFN


def hip_register_file(path: String, nbytes: Int) raises -> Int:
    """mmap `path` read-only (MAP_SHARED) and hipHostRegister it ReadOnly | Mapped:
    the kernels and the DMA engine read the page-cache pages themselves, so the
    expert store is the only RAM copy of experts.bin (kernels/test_hostregister.mojo:
    same GB/s as a pinned HostBuffer). The mapping lives until the process exits."""
    var lib = OwnedDLHandle("/opt/rocm/lib/libamdhip64.so")
    var addr = Int(0)
    with open(path, "r") as f:
        var fd = Int32(f._get_raw_fd())
        var size = Int(f.seek(0, 2))
        if size < nbytes:
            raise Error("expert tier: " + path + " is " + String(size) + " bytes, the index needs " + String(nbytes))
        addr = Int(sys.sys_mmap(nbytes, Int32(sys.PROT_READ), Int32(sys.MAP_SHARED), fd))
    if addr == -1:
        raise Error("expert tier: mmap failed for " + path)
    var rc = lib.get_function[Int32]("hipHostRegister")(addr, nbytes, UInt32(0x0A))
    if rc != 0:
        raise Error("expert tier: hipHostRegister(ReadOnly|Mapped) rc " + String(rc) + " for " + path + "; BARO_TIER_MMAP=0 reads it into a pinned buffer instead")
    return addr


def hip_coherent(nbytes: Int) raises -> Int:
    """Coherent, mapped pinned memory from libamdhip64 (hipHostMalloc flags
    Coherent | Mapped). MAX's host buffers are non-coherent: a GPU<->host
    handoff through them took 3.9 ms against 1.95 us (kernels/test_doorbell.mojo)."""
    var lib = OwnedDLHandle("/opt/rocm/lib/libamdhip64.so")
    var addr = Int(0)
    var rc = lib.get_function[Int32]("hipHostMalloc")(UnsafePointer(to=addr), nbytes, UInt32(0x40000002))
    if rc != 0 or addr == 0:
        raise Error("hipHostMalloc(coherent) failed rc " + String(rc))
    var z = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=addr)
    for i in range(nbytes):
        z[i] = 0
    return addr


def ptr_i32(addr: Int) -> UnsafePointer[Int32, MutAnyOrigin]:
    return UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=addr)


def ptr_f32(addr: Int) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=addr)


def ptr_bf16(addr: Int) -> UnsafePointer[BFloat16, MutAnyOrigin]:
    return UnsafePointer[BFloat16, MutAnyOrigin](unsafe_from_address=addr)


struct TierCpu(Copyable, Movable):
    """BARO_TIER_CPU (bench/moe-tier-cpu-protocol.md): a miss is computed on the
    CPU from the pinned store while the GPU runs the hits. Its router weight
    is zeroed on the device so the GPU's pass over the (refilling) slot adds
    nothing; the slot is refilled on a side stream so the LRU keeps adapting."""

    var on: Bool
    var max: Int
    var n: Int
    var misses: Int
    var ns: Int
    var stream_id: Int
    var fill_ev: List[DeviceEvent]
    var x_bf: HostBuffer[DType.bfloat16]
    var x_f: HostBuffer[DType.float32]
    var wt_h: HostBuffer[DType.float32]
    var wt_d: DeviceBuffer[DType.float32]
    var ids: HostBuffer[DType.int32]
    var w: HostBuffer[DType.float32]
    var hbuf: HostBuffer[DType.float32]
    var xq: HostBuffer[DType.int8]
    var aux: HostBuffer[DType.float32]
    var part_h: HostBuffer[DType.float32]
    var part_d: DeviceBuffer[DType.float32]
    var fill_src: List[Int]
    var fill_dst: List[Int]
    var fill_len: List[Int]
    var job: Int
    var wseq: Int
    var wlayer: Int
    var wait_ns: Int
    var t_sub: Int
    var t_done: Int
    var t_plan: Int
    var sub_done_ns: Int
    var done_bell_ns: Int
    var cycle_ns: Int
    var cycle_n: Int
    var jobs: Int

    def __init__(out self, ctx: DeviceContext, on: Bool, n_layers: Int) raises:
        self.on = on
        self.max = Int(getenv("BARO_TIER_CPU_MAX", "8"))
        self.n = 0
        self.misses = 0
        self.ns = 0
        self.stream_id = 0
        self.fill_ev = List[DeviceEvent]()
        for _ in range(n_layers if on else 0):
            self.fill_ev.append(ctx.create_event())
        var one = not on
        self.x_bf = ctx.enqueue_create_host_buffer[DType.bfloat16](1 if one else CPU_H)
        self.x_f = ctx.enqueue_create_host_buffer[DType.float32](1 if one else CPU_H)
        self.wt_h = ctx.enqueue_create_host_buffer[DType.float32](1 if one else TOPK)
        self.wt_d = ctx.enqueue_create_buffer[DType.float32](1)
        self.ids = ctx.enqueue_create_host_buffer[DType.int32](1 if one else TOPK)
        self.w = ctx.enqueue_create_host_buffer[DType.float32](1 if one else TOPK)
        self.hbuf = ctx.enqueue_create_host_buffer[DType.float32](1 if one else TOPK * CPU_FFN)
        self.xq = ctx.enqueue_create_host_buffer[DType.int8](1 if one else CPU_H + TOPK * CPU_FFN)
        self.aux = ctx.enqueue_create_host_buffer[DType.float32](1 if one else 2 * (CPU_H + TOPK * CPU_FFN) // 32)
        self.part_h = ctx.enqueue_create_host_buffer[DType.float32](1 if one else CPU_H)
        self.part_d = ctx.enqueue_create_buffer[DType.float32](1 if one else CPU_H)
        self.fill_src = List[Int]()
        self.fill_dst = List[Int]()
        self.fill_len = List[Int]()
        self.job = 0
        self.wseq = 0
        self.wlayer = 0
        self.wait_ns = 0
        self.t_sub = 0
        self.t_done = 0
        self.t_plan = 0
        self.sub_done_ns = 0
        self.done_bell_ns = 0
        self.cycle_ns = 0
        self.cycle_n = 0
        self.jobs = 0
        if on:
            _ = ctx.create_stream()
            self.stream_id = ctx.num_streams() - 1


struct LayerGeom(Copyable, Movable):
    """Byte geometry of one layer's three routed expert tensors."""

    var gate_off: Int
    var up_off: Int
    var down_off: Int
    var eb: Int
    var ebd: Int

    def __init__(out self, tensors: Dict[String, TensorInfo], layer: Int) raises:
        var g = resolve_plain(tensors, "blk." + String(layer) + ".ffn_gate_exps.weight")
        var u = resolve_plain(tensors, "blk." + String(layer) + ".ffn_up_exps.weight")
        var d = resolve_plain(tensors, "blk." + String(layer) + ".ffn_down_exps.weight")
        self.gate_off = g.offset
        self.up_off = u.offset
        self.down_off = d.offset
        self.eb = tensor_byte_size(g.dtype, g.n_elem) // N_EXP
        self.ebd = tensor_byte_size(d.dtype, d.n_elem) // N_EXP
        var ub = tensor_byte_size(u.dtype, u.n_elem) // N_EXP
        if ub != self.eb:
            raise Error(
                "layer " + String(layer) + ": up expert bytes " + String(ub)
                + " differ from gate's " + String(self.eb)
                + ", so one up_offset cannot address both"
            )


struct LayerLru(Copyable, Movable):
    """Expert id to cache slot, least-recently-used eviction.

    Deliberately the same policy the offline replay used
    (`bench/moe-locality.py`): capacity per layer, reset per request, ties
    broken by recency of use. Gate 1 replays this code over the same trace and
    must reproduce the replay's hit rate, because a tier whose residency
    decisions differ from the measurement's is not the thing that was
    measured.
    """

    var cap: Int
    var slot_of: Dict[Int, Int]
    var expert_at: List[Int]
    var used_at: List[Int]
    var pinned_experts: Dict[Int, Bool]
    var is_pinned_slot: List[Bool]
    var pinned_slots: Int
    var max_pin: Int
    var clock: Int

    def __init__(out self, cap: Int):
        self.cap = cap
        self.slot_of = Dict[Int, Int]()
        self.expert_at = List[Int](unsafe_uninit_length=cap)
        self.used_at = List[Int](unsafe_uninit_length=cap)
        self.pinned_experts = Dict[Int, Bool]()
        self.is_pinned_slot = List[Bool](capacity=cap)
        for i in range(cap):
            self.expert_at[i] = -1
            self.used_at[i] = -1
            self.is_pinned_slot.append(False)
        self.pinned_slots = 0
        self.max_pin = cap // 2
        self.clock = 0

    def add_pinned_expert(mut self, expert: Int):
        self.pinned_experts[expert] = True

    def reset(mut self):
        if len(self.pinned_experts) == 0:
            self.slot_of = Dict[Int, Int]()
            for i in range(self.cap):
                self.expert_at[i] = -1
                self.used_at[i] = -1
                self.is_pinned_slot[i] = False
            self.pinned_slots = 0
            self.clock = 0
            return
        self.slot_of = Dict[Int, Int]()
        self.pinned_slots = 0
        for i in range(self.cap):
            if self.is_pinned_slot[i] and self.expert_at[i] >= 0:
                self.slot_of[self.expert_at[i]] = i
                self.used_at[i] = 0
                self.pinned_slots += 1
            else:
                self.expert_at[i] = -1
                self.used_at[i] = -1
                self.is_pinned_slot[i] = False
        self.clock = 0

    def touch(mut self, expert: Int) raises -> Tuple[Int, Bool]:
        """-> (slot, hit). On a miss the caller must fill the slot."""
        self.clock += 1
        var boost = self.cap if (expert in self.pinned_experts) else 0
        if expert in self.slot_of:
            var s = self.slot_of[expert]
            self.used_at[s] = self.clock + boost
            return (s, True)
        var victim = 0
        var oldest = self.used_at[0]
        for i in range(self.cap):
            if self.expert_at[i] < 0:
                victim = i
                break
            if self.used_at[i] < oldest:
                oldest = self.used_at[i]
                victim = i
        var evicted = self.expert_at[victim]
        if evicted >= 0:
            _ = self.slot_of.pop(evicted)
        if self.is_pinned_slot[victim]:
            self.is_pinned_slot[victim] = False
            self.pinned_slots -= 1
        self.expert_at[victim] = expert
        self.used_at[victim] = self.clock + boost
        self.slot_of[expert] = victim
        if (expert in self.pinned_experts) and self.pinned_slots < self.max_pin:
            self.is_pinned_slot[victim] = True
            self.pinned_slots += 1
        return (victim, False)


struct ExpertTier(Copyable, Movable):
    var active: Bool
    var cap: Int
    var pinned: Bool
    var n_layers: Int
    var geom: List[LayerGeom]
    var lru: List[LayerLru]
    var cache: DeviceBuffer[DType.uint8]
    var layer_bytes: Int
    var stage: HostBuffer[DType.uint8]
    var store: HostBuffer[DType.uint8]
    var store_path: String
    var store_mmap: Bool
    var slots_h: HostBuffer[DType.int32]
    var slot_next: Int
    var refs: Int
    var hits: Int
    var prefetch_reads: Int
    var pinned_count: Int
    var draft_cache_only: Bool
    var draft_skipped: Int
    var bytes_fetched: Int
    var fetch_ns: Int
    var b_open: List[Int]
    var b_readback: List[Int]
    var b_lru: List[Int]
    var b_pread: List[Int]
    var b_copy: List[Int]
    var b_writeback: List[Int]
    var hot_active: Bool
    var hot_cap: Int
    var hot_layer_base: List[Int]
    var hot_store: HostBuffer[DType.uint8]
    var hot_slot_of: List[Dict[Int, Int]]
    var hot_next: List[Int]
    var zc: Bool
    var hoste_h: HostBuffer[DType.int32]
    var hoste_d: DeviceBuffer[DType.int32]
    var bytes_zc: Int
    var pf_stage: DeviceBuffer[DType.uint8]
    var pf_stage_bytes: Int
    var pf_overlap: Bool
    var pf_stream_id: Int
    var pf_slot_layer: List[Int]
    var pf_copy_ev: List[DeviceEvent]
    var pf_read_ev: List[DeviceEvent]
    var cpu: TierCpu
    var caps: List[Int]
    var lbases: List[Int]
    var bell_on: Bool
    var seq: Int
    var pending: List[Int]
    var lag: Int
    var bell: Int
    var bell_x: Int
    var plan: Int
    var flag: Int
    var part: Int
    var flag2: Int
    var bell_stream: Int

    def __init__(
        out self, ctx: DeviceContext, packdir: String, cap: Int, n_layers: Int
    ) raises:
        """Sizes everything from the pack's own index, never from constants."""
        var tensors = parse_moe_index(packdir + "/index.txt")
        self.active = True
        self.cap = cap
        self.n_layers = n_layers
        self.pinned = getenv("BARO_TIER_PINNED", "1") == "1"
        var cpu_on = self.pinned and getenv("BARO_TIER_CPU", "0") == "1"
        self.zc = self.pinned and getenv("BARO_TIER_ZC", "1") == "1"
        self.cpu = TierCpu(ctx, cpu_on, n_layers)
        self.bytes_zc = 0
        self.pf_stage = ctx.enqueue_create_buffer[DType.uint8](1)
        self.pf_stage_bytes = 0
        self.pf_overlap = False
        self.pf_stream_id = 0
        self.pf_slot_layer = [-1, -1]
        self.pf_copy_ev = [ctx.create_event(), ctx.create_event()]
        self.pf_read_ev = [ctx.create_event(), ctx.create_event()]
        self.hoste_h = ctx.enqueue_create_host_buffer[DType.int32](TOPK)
        self.hoste_d = ctx.enqueue_create_buffer[DType.int32](TOPK)
        self.hot_active = (not self.pinned) and getenv("BARO_TIER_HOT", "0") == "1"
        self.hot_cap = Int(getenv("BARO_TIER_HOTCAP", "128"))
        self.geom = List[LayerGeom]()
        self.lru = List[LayerLru]()
        self.hot_layer_base = List[Int]()
        self.hot_slot_of = List[Dict[Int, Int]]()
        self.hot_next = List[Int]()
        var per_layer = 0
        var store_bytes = 0
        var hot_bytes = 0
        var gs = List[LayerGeom]()
        for l in range(n_layers):
            var g = LayerGeom(tensors, l)
            var lb = cap * (2 * g.eb + g.ebd)
            if lb > per_layer:
                per_layer = lb
            gs.append(g^)
        # Slots are sized per layer in bytes (Strata finding 8): the budget is
        # cap x the largest layer's experts, as before, and a layer of smaller
        # experts (q4_k down) holds more of them. BARO_TIER_BYTES=0: cap everywhere.
        var by_bytes = getenv("BARO_TIER_BYTES", "1") == "1"
        self.caps = List[Int]()
        self.lbases = List[Int]()
        var cache_bytes = 0
        for l in range(n_layers):
            var c = per_layer // (2 * gs[l].eb + gs[l].ebd) if by_bytes else cap
            self.caps.append(min(c, N_EXP))
            self.lbases.append(cache_bytes)
            cache_bytes += self.caps[l] * (2 * gs[l].eb + gs[l].ebd)
        for l in range(n_layers):
            var g = gs[l].copy()
            store_bytes += N_EXP * (2 * g.eb + g.ebd)
            self.hot_layer_base.append(hot_bytes)
            hot_bytes += self.hot_cap * (2 * g.eb + g.ebd)
            self.hot_slot_of.append(Dict[Int, Int]())
            self.hot_next.append(0)
            self.geom.append(g^)
            self.lru.append(LayerLru(self.caps[l]))
        self.layer_bytes = per_layer
        self.hot_store = ctx.enqueue_create_host_buffer[DType.uint8](
            hot_bytes if self.hot_active else 1
        )
        self.cache = ctx.enqueue_create_buffer[DType.uint8](cache_bytes)
        self.stage = ctx.enqueue_create_host_buffer[DType.uint8](STAGE_BYTES)
        self.slots_h = ctx.enqueue_create_host_buffer[DType.int32](TOPK)
        self.store_path = packdir + "/experts.bin"
        var t_alloc = perf_counter_ns()
        var use_mmap = self.pinned and not cpu_on and getenv("BARO_TIER_MMAP", "1") == "1"
        self.store_mmap = use_mmap
        if use_mmap:
            var maddr = hip_register_file(self.store_path, store_bytes)
            self.store = HostBuffer[DType.uint8](
                ctx, UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=maddr), store_bytes, owning=False
            )
        else:
            self.store = ctx.enqueue_create_host_buffer[DType.uint8](
                store_bytes if self.pinned else 1
            )
        ctx.synchronize()
        var alloc_s = Float64(perf_counter_ns() - t_alloc) / 1e9
        if use_mmap:
            print("expert tier: store mmap+hipHostRegister (one RAM copy) s", alloc_s)
        self.slot_next = 0
        self.refs = 0
        self.hits = 0
        self.prefetch_reads = 0
        self.pinned_count = 0
        self.draft_cache_only = getenv("BARO_DRAFT_CACHE_ONLY", "0") == "1"
        self.draft_skipped = 0
        self.bytes_fetched = 0
        self.fetch_ns = 0
        var pin_file = getenv("BARO_TIER_PIN_FILE", "")
        if pin_file != "":
            with open(pin_file, "r") as pf:
                var pbytes = pf.read_bytes()
                if len(pbytes) < 8 or pbytes[0] != UInt8(0x50) or pbytes[1] != UInt8(0x49) or pbytes[2] != UInt8(0x4E) or pbytes[3] != UInt8(0x31):
                    raise Error("expert tier: invalid PIN1 magic in " + pin_file)
                var pcnt = Int(pbytes[4]) | (Int(pbytes[5]) << 8) | (Int(pbytes[6]) << 16) | (Int(pbytes[7]) << 24)
                if len(pbytes) < 8 + pcnt * 4:
                    raise Error("expert tier: truncated PIN1 file " + pin_file)
                for pi in range(pcnt):
                    var off = 8 + pi * 4
                    var ply = Int(pbytes[off]) | (Int(pbytes[off + 1]) << 8)
                    var pex = Int(pbytes[off + 2]) | (Int(pbytes[off + 3]) << 8)
                    if ply >= 0 and ply < n_layers and pex >= 0 and pex < N_EXP:
                        self.lru[ply].add_pinned_expert(pex)
                        self.pinned_count += 1
            print("expert tier: loaded", self.pinned_count, "pinned (layer, expert) slots from", pin_file)
        self.b_open = List[Int]()
        self.b_readback = List[Int]()
        self.b_lru = List[Int]()
        self.b_pread = List[Int]()
        self.b_copy = List[Int]()
        self.b_writeback = List[Int]()
        for _ in range(n_layers):
            self.b_open.append(0)
            self.b_readback.append(0)
            self.b_lru.append(0)
            self.b_pread.append(0)
            self.b_copy.append(0)
            self.b_writeback.append(0)
        ctx.synchronize()
        if self.pinned and not use_mmap:
            var t_read = perf_counter_ns()
            with open(self.store_path, "r") as f:
                var fd = f._get_raw_fd()
                var sptr = self.store.unsafe_ptr()
                comptime RSPLIT = 8
                var rerr = List[Int64](unsafe_uninit_length=RSPLIT)
                var rerr_ptr = rerr.unsafe_ptr()
                def rpart(t: Int) {imm fd, imm store_bytes, imm sptr, imm rerr_ptr}:
                    var lo = store_bytes * t // RSPLIT
                    var hi = store_bytes * (t + 1) // RSPLIT
                    var got = lo
                    while got < hi:
                        var n = external_call["pread", c_ssize_t](
                            fd, sptr.unsafe_offset(got), hi - got, Int64(got)
                        )
                        if n <= 0:
                            rerr_ptr[unsafe_offset=t] = 1
                            return
                        got += Int(n)
                    rerr_ptr[unsafe_offset=t] = 0
                parallelize(rpart, RSPLIT, RSPLIT)
                for t in range(RSPLIT):
                    if rerr[t] != 0:
                        raise Error("expert tier: short read of experts.bin")
            print("expert tier: store load alloc_s", alloc_s, "read_s", Float64(perf_counter_ns() - t_read) / 1e9)
        if self.cpu.on:
            if self.geom[0].eb != CPU_FFN * (CPU_H // QK_K) * Q4_K_BYTES:
                raise Error("BARO_TIER_CPU: expert bytes " + String(self.geom[0].eb) + " do not match the CPU kernel's H/FFN")
            ctx.enqueue_memset(self.cache, 0)
            ctx.synchronize()
        # Doorbell decode (Strata port P2): the GPU hands the router's top-k to
        # the host through coherent pinned memory and spins until the plan is
        # back, so no layer pays a synchronize. Needs the ZC path (misses fill
        # their slots in-kernel). BARO_TIER_BELL=0 keeps prepare().
        self.bell_on = self.zc and getenv("BARO_TIER_BELL", "1") == "1"
        self.seq = 0
        self.pending = List[Int]()
        # BARO_PROFILE syncs inside the layer loop: serve every layer at once there.
        self.lag = Int(getenv("BARO_TIER_BELL_LAG", "4")) if getenv("BARO_PROFILE", "0") == "0" else 0
        self.bell = 0
        self.bell_x = 0
        self.plan = 0
        self.flag = 0
        self.part = 0
        self.flag2 = 0
        self.bell_stream = 0
        if self.bell_on:
            self.bell = hip_coherent(4 * (1 + 2 * TOPK))
            self.bell_x = hip_coherent(2 * CPU_H)
            self.plan = hip_coherent(4 * 3 * TOPK)
            self.flag = hip_coherent(64)
            self.part = hip_coherent(4 * CPU_H)
            self.flag2 = hip_coherent(64)
            _ = ctx.create_stream()
            self.bell_stream = ctx.num_streams() - 1
        var mode = String("page-cache")
        if self.cpu.on:
            mode = String("pinned+cpu") + " cpu_max " + String(self.cpu.max)
        elif self.zc:
            mode = String("pinned+zc")
        elif self.pinned:
            mode = String("pinned")
        elif self.hot_active:
            mode = String("hot ") + String(self.hot_cap)
        if use_mmap:
            mode += "+mmap"
        print(
            "expert tier: cap", cap, " cache", Float64(per_layer * n_layers) / 1e9,
            "GB  host store", Float64(store_bytes) / 1e9,
            "GB  mode", mode, " bell", self.bell_on, " slots_by_bytes", by_bytes, " slots", self.caps[0], "..", self.caps[n_layers - 1],
            " hot_bytes", Float64(hot_bytes) / 1e9 if self.hot_active else 0.0,
        )

    def __init__(out self, ctx: DeviceContext) raises:
        """BARO_TIER unset: everything stays in the pack and moe_ffn takes its
        old path. One-byte buffers rather than zero-length ones, because a
        zero-length device buffer is not worth the special case."""
        self.active = False
        self.cap = 0
        self.pinned = False
        self.zc = False
        self.cpu = TierCpu(ctx, False, 0)
        self.bytes_zc = 0
        self.pf_stage = ctx.enqueue_create_buffer[DType.uint8](1)
        self.pf_stage_bytes = 0
        self.pf_overlap = False
        self.pf_stream_id = 0
        self.pf_slot_layer = [-1, -1]
        self.pf_copy_ev = [ctx.create_event(), ctx.create_event()]
        self.pf_read_ev = [ctx.create_event(), ctx.create_event()]
        self.hoste_h = ctx.enqueue_create_host_buffer[DType.int32](1)
        self.hoste_d = ctx.enqueue_create_buffer[DType.int32](1)
        self.n_layers = 0
        self.geom = List[LayerGeom]()
        self.lru = List[LayerLru]()
        self.layer_bytes = 0
        self.cache = ctx.enqueue_create_buffer[DType.uint8](1)
        self.stage = ctx.enqueue_create_host_buffer[DType.uint8](1)
        self.store = ctx.enqueue_create_host_buffer[DType.uint8](1)
        self.slots_h = ctx.enqueue_create_host_buffer[DType.int32](1)
        self.store_path = ""
        self.store_mmap = False
        self.slot_next = 0
        self.refs = 0
        self.hits = 0
        self.prefetch_reads = 0
        self.pinned_count = 0
        self.draft_cache_only = False
        self.draft_skipped = 0
        self.bytes_fetched = 0
        self.fetch_ns = 0
        self.b_open = List[Int]()
        self.b_readback = List[Int]()
        self.b_lru = List[Int]()
        self.b_pread = List[Int]()
        self.b_copy = List[Int]()
        self.b_writeback = List[Int]()
        self.hot_active = False
        self.hot_cap = 0
        self.hot_layer_base = List[Int]()
        self.hot_store = ctx.enqueue_create_host_buffer[DType.uint8](1)
        self.hot_slot_of = List[Dict[Int, Int]]()
        self.hot_next = List[Int]()
        self.caps = List[Int]()
        self.lbases = List[Int]()
        self.bell_on = False
        self.seq = 0
        self.pending = List[Int]()
        self.lag = 0
        self.bell = 0
        self.bell_x = 0
        self.plan = 0
        self.flag = 0
        self.part = 0
        self.flag2 = 0
        self.bell_stream = 0

    def _stage_copy(mut self, ctx: DeviceContext, layer: Int, slot: Int) raises:
        var eb = self.geom[layer].eb
        var ebd = self.geom[layer].ebd
        var base = slot * self.pf_stage_bytes
        ctx.enqueue_copy(
            dst_buf=DeviceBuffer[DType.uint8](ctx, self.pf_stage.unsafe_ptr().unsafe_offset(base), N_EXP * eb, owning=False),
            src_buf=self.store.create_sub_buffer[DType.uint8](self.geom[layer].gate_off, N_EXP * eb),
        )
        ctx.enqueue_copy(
            dst_buf=DeviceBuffer[DType.uint8](ctx, self.pf_stage.unsafe_ptr().unsafe_offset(base + N_EXP * eb), N_EXP * eb, owning=False),
            src_buf=self.store.create_sub_buffer[DType.uint8](self.geom[layer].up_off, N_EXP * eb),
        )
        ctx.enqueue_copy(
            dst_buf=DeviceBuffer[DType.uint8](ctx, self.pf_stage.unsafe_ptr().unsafe_offset(base + 2 * N_EXP * eb), N_EXP * ebd, owning=False),
            src_buf=self.store.create_sub_buffer[DType.uint8](self.geom[layer].down_off, N_EXP * ebd),
        )
        self.pf_slot_layer[slot] = layer

    def stage_layer(mut self, ctx: DeviceContext, layer: Int) raises -> Int:
        """Batched prefill (bench/moe-prefill-protocol.md, design item 5 and
        round 1b item 3): all N_EXP experts of one layer, pinned store to a
        VRAM staging slot, gate | up | down; returns the slot's byte base.
        A chunk touches nearly every expert, so the slot cache and its per-row
        prepare are bypassed and left untouched.

        Two slots and a second stream: while layer L's kernels read slot
        L % 2 on the compute stream, layer L+1 copies into the other slot on
        the copy stream. That copy first waits for the event recorded after
        layer L-1's kernels (the slot's last reader), and layer L+1's kernels
        wait for the copy's event. BARO_PF_OVERLAP=0 keeps one stream: the
        copy is then ordered with the kernels by the queue itself."""
        if not self.pinned:
            raise Error("expert tier: batched prefill needs BARO_TIER_PINNED=1")
        if self.pf_stage_bytes == 0:
            var need = 0
            for l in range(self.n_layers):
                var lb = N_EXP * (2 * self.geom[l].eb + self.geom[l].ebd)
                if lb > need:
                    need = lb
            self.pf_overlap = getenv("BARO_PF_OVERLAP", "1") == "1"
            self.pf_stage = ctx.enqueue_create_buffer[DType.uint8](need * (2 if self.pf_overlap else 1))
            self.pf_stage_bytes = need
            if self.pf_overlap:
                _ = ctx.create_stream()
                self.pf_stream_id = ctx.num_streams() - 1
            print("BARO_TIER prefill stage:", Float64(need) / 1e9, "GB x", 2 if self.pf_overlap else 1, " overlap stream:", self.pf_stream_id)
        if not self.pf_overlap:
            self._stage_copy(ctx, layer, 0)
            return 0
        var slot = layer % 2
        var cctx = ctx.select_stream(self.pf_stream_id)
        if self.pf_slot_layer[slot] != layer:
            # Nothing pre-staged this layer (first layer of the first chunk).
            ctx.synchronize()
            self._stage_copy(cctx, layer, slot)
            cctx.stream().synchronize()
        else:
            ctx.stream().enqueue_wait_for(self.pf_copy_ev[slot])
        return slot * self.pf_stage_bytes

    def stage_next(mut self, ctx: DeviceContext, layer: Int) raises:
        """Call after layer `layer`'s expert kernels are enqueued: starts the
        next layer's copy into the slot layer - 1 used."""
        if not self.pf_overlap:
            return
        var nxt = (layer + 1) % self.n_layers
        var slot = nxt % 2
        var cctx = ctx.select_stream(self.pf_stream_id)
        # Slot `slot` was last read by layer - 1's kernels; their event was
        # recorded before this layer's kernels were enqueued, so the copy
        # runs while this layer computes.
        cctx.stream().enqueue_wait_for(self.pf_read_ev[slot])
        self._stage_copy(cctx, nxt, slot)
        var ev = ctx.create_event()
        cctx.stream().record_event(ev)
        var rd = ctx.create_event()
        ctx.stream().record_event(rd)
        self.pf_read_ev[layer % 2] = rd^
        self.pf_copy_ev[slot] = ev^

    def layer_base(self, layer: Int) -> Int:
        return self.lbases[layer]

    def gate_base(self, layer: Int) -> Int:
        return self.layer_base(layer)

    def up_offset(self, layer: Int) -> Int:
        """What the gate/up kernel needs as its runtime gate-to-up distance."""
        return self.caps[layer] * self.geom[layer].eb

    def down_base(self, layer: Int) -> Int:
        return self.layer_base(layer) + 2 * self.caps[layer] * self.geom[layer].eb

    def reset(mut self):
        """Per request: a new conversation starts with a cold tier, which is
        also what the offline replay assumed when it reset per prompt. The
        hot store (this request's touched-expert reuse cache, item 2) is
        request-scoped too: a later request's routing has no bearing on this
        one's, so a stale hot_slot_of entry would only ever be a wrong guess,
        never a correctness risk (it is never consulted across resets)."""
        for i in range(self.n_layers):
            self.lru[i].reset()
        for i in range(self.n_layers):
            self.hot_slot_of[i] = Dict[Int, Int]()
            self.hot_next[i] = 0

    def _copy_hot_piece(
        mut self, ctx: DeviceContext, hot_off: Int, dev_off: Int, nbytes: Int, layer: Int
    ) raises:
        """A piece already resident in this request's hot store: direct
        device copy, no pread."""
        var tc0 = perf_counter_ns()
        ctx.enqueue_copy(
            dst_buf=DeviceBuffer[DType.uint8](
                ctx, self.cache.unsafe_ptr().unsafe_offset(dev_off), nbytes, owning=False
            ),
            src_buf=self.hot_store.create_sub_buffer[DType.uint8](hot_off, nbytes),
        )
        self.b_copy[layer] = self.b_copy[layer] + Int(perf_counter_ns() - tc0)
        self.bytes_fetched += nbytes

    def _fetch_piece(
        mut self, ctx: DeviceContext, fd: Int, host_off: Int, dev_off: Int, nbytes: Int,
        layer: Int, hot_dst: Int,
    ) raises:
        """One expert projection, host to device, through the staging buffer
        unless the whole store is pinned (then the copy is direct), or
        hot_dst >= 0: this is this request's first touch of an expert with
        room in the hot store, so the pread lands there instead of the
        rotating stage ring, persisting for a same-request re-reference
        (item 2, _copy_hot_piece) instead of paying another pread."""
        if self.pinned:
            var tc0 = perf_counter_ns()
            if self.store_mmap:
                if nbytes > SLOT_BYTES:
                    raise Error("expert tier: staging slot too small for " + String(nbytes))
                var sbase = (self.slot_next % STAGE_SLOTS) * SLOT_BYTES
                self.slot_next += 1
                _ = external_call["memcpy", Int](
                    Int(self.stage.unsafe_ptr().unsafe_offset(sbase)), Int(self.store.unsafe_ptr().unsafe_offset(host_off)), nbytes
                )
                ctx.enqueue_copy(
                    dst_buf=DeviceBuffer[DType.uint8](
                        ctx, self.cache.unsafe_ptr().unsafe_offset(dev_off), nbytes, owning=False
                    ),
                    src_buf=self.stage.create_sub_buffer[DType.uint8](sbase, nbytes),
                )
                self.b_copy[layer] = self.b_copy[layer] + Int(perf_counter_ns() - tc0)
                self.bytes_fetched += nbytes
                return
            ctx.enqueue_copy(
                dst_buf=DeviceBuffer[DType.uint8](
                    ctx, self.cache.unsafe_ptr().unsafe_offset(dev_off), nbytes, owning=False
                ),
                src_buf=self.store.create_sub_buffer[DType.uint8](host_off, nbytes),
            )
            self.b_copy[layer] = self.b_copy[layer] + Int(perf_counter_ns() - tc0)
            self.bytes_fetched += nbytes
            return
        if nbytes > SLOT_BYTES:
            raise Error("expert tier: staging slot too small for " + String(nbytes))
        var use_hot = hot_dst >= 0
        var base = hot_dst
        if not use_hot:
            base = (self.slot_next % STAGE_SLOTS) * SLOT_BYTES
            self.slot_next += 1
        var got = 0
        var tp0 = perf_counter_ns()
        if use_hot:
            while got < nbytes:
                var n = external_call["pread", c_ssize_t](
                    fd, self.hot_store.unsafe_ptr().unsafe_offset(base + got),
                    nbytes - got, Int64(host_off + got),
                )
                if n <= 0:
                    raise Error("expert tier: short read at " + String(host_off))
                got += Int(n)
        else:
            while got < nbytes:
                var n = external_call["pread", c_ssize_t](
                    fd, self.stage.unsafe_ptr().unsafe_offset(base + got),
                    nbytes - got, Int64(host_off + got),
                )
                if n <= 0:
                    raise Error("expert tier: short read at " + String(host_off))
                got += Int(n)
        self.b_pread[layer] = self.b_pread[layer] + Int(perf_counter_ns() - tp0)
        var tc0 = perf_counter_ns()
        if use_hot:
            ctx.enqueue_copy(
                dst_buf=DeviceBuffer[DType.uint8](
                    ctx, self.cache.unsafe_ptr().unsafe_offset(dev_off), nbytes, owning=False
                ),
                src_buf=self.hot_store.create_sub_buffer[DType.uint8](base, nbytes),
            )
        else:
            ctx.enqueue_copy(
                dst_buf=DeviceBuffer[DType.uint8](
                    ctx, self.cache.unsafe_ptr().unsafe_offset(dev_off), nbytes, owning=False
                ),
                src_buf=self.stage.create_sub_buffer[DType.uint8](base, nbytes),
            )
        self.b_copy[layer] = self.b_copy[layer] + Int(perf_counter_ns() - tc0)
        self.bytes_fetched += nbytes

    def cpu_readback(
        mut self, ctx: DeviceContext, x_d: DeviceBuffer[DType.bfloat16], wt_d: DeviceBuffer[DType.float32]
    ) raises:
        """Enqueue x (row 0) and the router weights to the host; prepare()'s
        sync completes them, so the readback adds no round trip."""
        ctx.enqueue_copy(dst_buf=self.cpu.x_bf, src_buf=DeviceBuffer[DType.bfloat16](ctx, x_d.unsafe_ptr(), CPU_H, owning=False))
        ctx.enqueue_copy(dst_buf=self.cpu.wt_h, src_buf=DeviceBuffer[DType.float32](ctx, wt_d.unsafe_ptr(), TOPK, owning=False))
        self.cpu.wt_d = DeviceBuffer[DType.float32](ctx, wt_d.unsafe_ptr(), TOPK, owning=False)

    def cpu_compute(mut self, ctx: DeviceContext, layer: Int) raises:
        """This layer's misses on the CPU, then the partial up to part_d.
        Called after the hit kernels are enqueued, so the two overlap."""
        var t0 = perf_counter_ns()
        for i in range(CPU_H):
            self.cpu.x_f[i] = self.cpu.x_bf[i].cast[DType.float32]()
        var base = self.store.unsafe_ptr()
        expert_ffn_cpu[CPU_H, CPU_FFN](
            base.unsafe_offset(self.geom[layer].gate_off), base.unsafe_offset(self.geom[layer].up_off), base.unsafe_offset(self.geom[layer].down_off),
            self.cpu.ids.unsafe_ptr(), self.cpu.w.unsafe_ptr(), self.cpu.n,
            self.cpu.x_f.unsafe_ptr(), self.cpu.hbuf.unsafe_ptr(), self.cpu.xq.unsafe_ptr(),
            self.cpu.aux.unsafe_ptr(), self.cpu.part_h.unsafe_ptr(),
        )
        ctx.enqueue_copy(dst_buf=self.cpu.part_d, src_buf=self.cpu.part_h)
        self.cpu.ns += Int(perf_counter_ns() - t0)

    def cpu_refill(mut self, ctx: DeviceContext, layer: Int) raises:
        """Refill this layer's CPU-miss slots, after everything already on the
        main stream (the layer's expert kernels, which skip those slots) and
        after cpu_compute, so the copies never race a reader and never compete
        with the CPU's own reads of the same experts."""
        if len(self.cpu.fill_src) == 0:
            return
        var cctx = ctx.select_stream(self.cpu.stream_id)
        var done = ctx.create_event()
        ctx.stream().record_event(done)
        cctx.stream().enqueue_wait_for(done)
        var tc0 = perf_counter_ns()
        for i in range(len(self.cpu.fill_src)):
            cctx.enqueue_copy(
                dst_buf=DeviceBuffer[DType.uint8](
                    ctx, self.cache.unsafe_ptr().unsafe_offset(self.cpu.fill_dst[i]), self.cpu.fill_len[i], owning=False
                ),
                src_buf=self.store.create_sub_buffer[DType.uint8](self.cpu.fill_src[i], self.cpu.fill_len[i]),
            )
            self.bytes_fetched += self.cpu.fill_len[i]
        cctx.stream().record_event(self.cpu.fill_ev[layer])
        self.b_copy[layer] = self.b_copy[layer] + Int(perf_counter_ns() - tc0)
        self.cpu.fill_src.clear()
        self.cpu.fill_dst.clear()
        self.cpu.fill_len.clear()

    def _plan(mut self, ctx: DeviceContext, layer: Int, fd: Int, bell: Bool) raises:
        """The LRU decision for one layer: slots_h holds expert ids on entry and
        cache slots on exit, hoste_h the ZC miss marks, cpu.ids/w the CPU misses.
        A miss that needs a DMA (q6_k down under ZC) is enqueued on `ctx`."""
        # Copied out, not held as a reference: _fetch_piece takes `mut self`
        # (it counts bytes), which invalidates an interior reference into
        # self.geom while the fetch loop still needs the geometry.
        var eb = self.geom[layer].eb
        var ebd = self.geom[layer].ebd
        var lbase = self.layer_base(layer)
        var lcap = self.caps[layer]
        var gate_store = self.geom[layer].gate_off
        var up_store = self.geom[layer].up_off
        var down_store = self.geom[layer].down_off
        for j in range(TOPK):
            self.hoste_h[j] = Int32(-1)
        self.cpu.n = 0
        var cpu_layer = self.cpu.on and ebd == eb
        var cctx = ctx.select_stream(self.cpu.stream_id)
        if cpu_layer and not bell:
            # Last token's slot refills for this layer must land before any of
            # them is read as a hit. Expected to be long done; not measured.
            ctx.stream().enqueue_wait_for(self.cpu.fill_ev[layer])
        for j in range(TOPK):
            var e = Int(self.slots_h[j])
            if e < 0 or e >= N_EXP:
                raise Error("expert tier: router returned expert id " + String(e))
            if self.draft_cache_only and (e not in self.lru[layer].slot_of) and len(self.lru[layer].slot_of) > 0:
                self.refs += 1
                self.draft_skipped += 1
                self.slots_h[j] = Int32(0)
                continue
            var tl0 = perf_counter_ns()
            var r = self.lru[layer].touch(e)
            self.b_lru[layer] = self.b_lru[layer] + Int(perf_counter_ns() - tl0)
            var slot = r[0]
            self.refs += 1
            if r[1]:
                self.hits += 1
            else:
                var hot_off = -1
                var hot_ready = False
                if self.hot_active:
                    if e in self.hot_slot_of[layer]:
                        hot_off = self.hot_layer_base[layer] + self.hot_slot_of[layer][e] * (2 * eb + ebd)
                        hot_ready = True
                    elif self.hot_next[layer] < self.hot_cap:
                        var new_slot = self.hot_next[layer]
                        self.hot_next[layer] = new_slot + 1
                        self.hot_slot_of[layer][e] = new_slot
                        hot_off = self.hot_layer_base[layer] + new_slot * (2 * eb + ebd)
                if cpu_layer and self.cpu.n < self.cpu.max:
                    self.cpu.ids[self.cpu.n] = Int32(e)
                    self.cpu.w[self.cpu.n] = self.cpu.wt_h[j]
                    self.cpu.wt_h[j] = 0
                    self.cpu.n += 1
                    self.cpu.misses += 1
                    self.hoste_h[j] = Int32(-2)
                    self.cpu.fill_src.append(gate_store + e * eb)
                    self.cpu.fill_dst.append(lbase + slot * eb)
                    self.cpu.fill_len.append(eb)
                    self.cpu.fill_src.append(up_store + e * eb)
                    self.cpu.fill_dst.append(lbase + lcap * eb + slot * eb)
                    self.cpu.fill_len.append(eb)
                    self.cpu.fill_src.append(down_store + e * ebd)
                    self.cpu.fill_dst.append(lbase + 2 * lcap * eb + slot * ebd)
                    self.cpu.fill_len.append(ebd)
                elif hot_ready:
                    self._copy_hot_piece(ctx, hot_off, lbase + slot * eb, eb, layer)
                    self._copy_hot_piece(ctx, hot_off + eb, lbase + lcap * eb + slot * eb, eb, layer)
                    self._copy_hot_piece(ctx, hot_off + 2 * eb, lbase + 2 * lcap * eb + slot * ebd, ebd, layer)
                else:
                    var hot_up = -1
                    var hot_down = -1
                    if hot_off >= 0:
                        hot_up = hot_off + eb
                        hot_down = hot_off + 2 * eb
                    if self.zc:
                        # Zero-copy miss: the gate/up kernel reads this expert
                        # from the pinned store and fills the slot itself. The
                        # down piece too when it is q4_k (ebd == eb); the q6_k
                        # layers keep the DMA for down, their kernel is unchanged.
                        self.hoste_h[j] = Int32(e)
                        self.bytes_zc += 2 * eb
                        self.bytes_fetched += 2 * eb
                        if ebd == eb:
                            self.bytes_zc += ebd
                            self.bytes_fetched += ebd
                        else:
                            self._fetch_piece(
                                ctx, fd, down_store + e * ebd,
                                lbase + 2 * lcap * eb + slot * ebd, ebd, layer, hot_down,
                            )
                    else:
                        self._fetch_piece(
                            ctx, fd, gate_store + e * eb, lbase + slot * eb, eb, layer, hot_off,
                        )
                        self._fetch_piece(
                            ctx, fd, up_store + e * eb, lbase + lcap * eb + slot * eb, eb, layer, hot_up,
                        )
                        self._fetch_piece(
                            ctx, fd, down_store + e * ebd,
                            lbase + 2 * lcap * eb + slot * ebd, ebd, layer, hot_down,
                        )
            self.slots_h[j] = Int32(slot)

    def bell_enqueue(
        mut self, ctx: DeviceContext, layer: Int, idx_d: DeviceBuffer[DType.int32],
        wt_d: DeviceBuffer[DType.float32], x_d: DeviceBuffer[DType.bfloat16],
    ) raises:
        """Enqueue this layer's doorbell (ids, weights, x to the host) and the
        GPU's wait for the plan; serve() answers it. Serves the oldest pending
        layer once more than `lag` are queued, so the stream never holds a whole
        token behind a spinning kernel."""
        self.seq += 1
        var seq = Int32(self.seq)
        ctx.enqueue_function[tier_bell_out](
            idx_d.unsafe_ptr(), wt_d.unsafe_ptr(), x_d.unsafe_ptr(),
            ptr_i32(self.bell), ptr_bf16(self.bell_x), seq, Int32(CPU_H),
            grid_dim=1, block_dim=256,
        )
        ctx.enqueue_function[tier_bell_in](
            ptr_i32(self.plan), ptr_i32(self.flag), seq,
            idx_d.unsafe_ptr(), self.hoste_d.unsafe_ptr(), wt_d.unsafe_ptr(),
            grid_dim=1, block_dim=32,
        )

    def bell_wait_add(mut self, ctx: DeviceContext, layer: Int, xres_d: DeviceBuffer[DType.float32], xoff: Int) raises:
        """CPU layers: the GPU adds the CPU's partial once the host raises flag2."""
        ctx.enqueue_function[tier_wait_add](
            ptr_f32(self.part), ptr_i32(self.flag2), Int32(self.seq),
            xres_d.unsafe_ptr().unsafe_offset(xoff), Int32(CPU_H),
            grid_dim=(CPU_H + 255) // 256, block_dim=256,
        )

    def bell_queued(mut self, ctx: DeviceContext, layer: Int) raises:
        self.pending.append(layer)
        while len(self.pending) > self.lag:
            self.serve(ctx)

    def drain(mut self, ctx: DeviceContext) raises:
        """Before any host sync: answer every queued doorbell, then collect the CPU worker's last job."""
        while len(self.pending) > 0:
            self.serve(ctx)
        _ = self._reap(ctx, True)

    def _cpu_worker_init(mut self) raises:
        """Lazy, on the first CPU layer: the job's fixed buffers, then the pthread."""
        var job = alloc[CpuJob](1)
        job[].ids = self.cpu.ids.unsafe_ptr()
        job[].wts = self.cpu.w.unsafe_ptr()
        job[].x_bf = UnsafePointer[BFloat16, MutUntrackedOrigin](unsafe_from_address=self.bell_x)
        job[].x_f = self.cpu.x_f.unsafe_ptr()
        job[].hbuf = self.cpu.hbuf.unsafe_ptr()
        job[].xq = self.cpu.xq.unsafe_ptr()
        job[].aux = self.cpu.aux.unsafe_ptr()
        job[].dst = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=self.part)
        job[].flag2 = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=self.flag2)
        _ = cpu_worker_start[CPU_H, CPU_FFN](job)
        self.cpu.job = Int(job)

    def _reap(mut self, ctx: DeviceContext, block: Bool) raises -> Bool:
        """The in-flight CPU job, if any: once the worker is done, count its time and enqueue the
        refills of its CPU-miss slots on the doorbell stream (this layer's kernels skip them, HE = -2,
        and the next token's plan for the layer waits on fill_ev first). Host state stays single-owner:
        only this thread touches the fill lists, and the next _plan runs after the reap."""
        if self.cpu.wseq == 0:
            return True
        var job = UnsafePointer[CpuJob, MutUntrackedOrigin](unsafe_from_address=self.cpu.job)
        var seq = Int32(self.cpu.wseq)
        if not cpu_worker_done(job, seq):
            if not block:
                return False
            var tw = perf_counter_ns()
            var spins = 0
            while not cpu_worker_done(job, seq):
                spins += 1
                if spins & 0xFFFF == 0 and perf_counter_ns() - tw > 5_000_000_000:
                    self._release()
                    raise Error("expert tier: cpu worker never finished layer " + String(self.cpu.wlayer) + " (5 s)")
            self.cpu.wait_ns += Int(perf_counter_ns() - tw)
        self.cpu.ns += job[].ns
        self.cpu.wseq = 0
        self.cpu.t_done = perf_counter_ns()
        self.cpu.sub_done_ns += self.cpu.t_done - self.cpu.t_sub
        self.cpu.jobs += 1
        var cctx = ctx.select_stream(self.bell_stream)
        for i in range(len(self.cpu.fill_src)):
            cctx.enqueue_copy(
                dst_buf=DeviceBuffer[DType.uint8](ctx, self.cache.unsafe_ptr().unsafe_offset(self.cpu.fill_dst[i]), self.cpu.fill_len[i], owning=False),
                src_buf=self.store.create_sub_buffer[DType.uint8](self.cpu.fill_src[i], self.cpu.fill_len[i]),
            )
            self.bytes_fetched += self.cpu.fill_len[i]
        cctx.stream().record_event(self.cpu.fill_ev[self.cpu.wlayer])
        self.cpu.fill_src.clear()
        self.cpu.fill_dst.clear()
        self.cpu.fill_len.clear()
        return True

    def serve(mut self, ctx: DeviceContext) raises:
        var layer = self.pending.pop(0)
        var seq = self.seq - len(self.pending)
        var t0 = perf_counter_ns()
        var bell = ptr_i32(self.bell)
        var spins = 0
        while bell.unsafe_load[volatile=True]() < Int32(seq):
            spins += 1
            if self.cpu.wseq != 0:
                _ = self._reap(ctx, False)
            if spins & 0xFFFF == 0 and perf_counter_ns() - t0 > 5_000_000_000:
                self._release()
                raise Error("expert tier: doorbell for layer " + String(layer) + " never rang (5 s)")
        var t_bell = perf_counter_ns()
        self.b_readback[layer] = self.b_readback[layer] + Int(t_bell - t0)
        var tp0 = t_bell
        if self.cpu.t_plan != 0:
            self.cpu.cycle_ns += t_bell - self.cpu.t_plan
            self.cpu.cycle_n += 1
        if self.cpu.wseq == 0 and self.cpu.t_done > self.cpu.t_plan:
            self.cpu.done_bell_ns += t_bell - self.cpu.t_done
        _ = self._reap(ctx, True)
        for j in range(TOPK):
            self.slots_h[j] = bell[1 + j]
        var cpu_layer = self.cpu.on and self.geom[layer].ebd == self.geom[layer].eb
        if cpu_layer:
            for j in range(TOPK):
                self.cpu.wt_h[j] = bitcast[DType.float32](bell[1 + TOPK + j])
        var cctx = ctx.select_stream(self.bell_stream)
        var before = self.bytes_fetched - self.bytes_zc
        self._plan(cctx, layer, -1, True)
        if self.bytes_fetched - self.bytes_zc != before:
            cctx.stream().synchronize()
        if cpu_layer:
            # The worker computes the misses and raises flag2; this thread only plans. Handed off as soon
            # as the CPU set is known, so the job overlaps the rest of the plan.
            if self.cpu.job == 0:
                self._cpu_worker_init()
            var job = UnsafePointer[CpuJob, MutUntrackedOrigin](unsafe_from_address=self.cpu.job)
            var base = self.store.unsafe_ptr()
            job[].gate = base.unsafe_offset(self.geom[layer].gate_off)
            job[].up = base.unsafe_offset(self.geom[layer].up_off)
            job[].down = base.unsafe_offset(self.geom[layer].down_off)
            job[].n = self.cpu.n
            job[].flag2_val = Int32(seq)
            self.cpu.wseq = seq
            self.cpu.wlayer = layer
            self.cpu.t_sub = perf_counter_ns()
            cpu_worker_submit(job, Int32(seq))
        if cpu_layer:
            # This layer's refills from the previous token must have landed
            # before the plan can name those slots as hits.
            self.cpu.fill_ev[layer].synchronize()
        var plan = ptr_i32(self.plan)
        for j in range(TOPK):
            plan[j] = self.slots_h[j]
            plan[TOPK + j] = self.hoste_h[j]
            plan[2 * TOPK + j] = bitcast[DType.int32](self.cpu.wt_h[j]) if cpu_layer else bell[1 + TOPK + j]
        fence[Ordering.RELEASE]()
        ptr_i32(self.flag).unsafe_store[volatile=True](Int32(seq))
        self.cpu.t_plan = perf_counter_ns()
        self.b_lru[layer] = self.b_lru[layer] + Int(self.cpu.t_plan - tp0)
        self.fetch_ns += Int(self.cpu.t_plan - t0)

    def _release(mut self):
        """Unblock every spinning kernel (hang guard)."""
        print("expert tier: HANG GUARD released the spinning kernels (a doorbell or cpu-worker wait passed 5 s); the run is void")
        ptr_i32(self.flag).unsafe_store[volatile=True](Int32(1 << 30))
        ptr_i32(self.flag2).unsafe_store[volatile=True](Int32(1 << 30))

    def prepare(
        mut self, ctx: DeviceContext, layer: Int, mut idx_d: DeviceBuffer[DType.int32]
    ) raises:
        """Read the router's top-8 back, fill misses, write SLOTS in place.

        The device index buffer holds expert ids on entry and cache slots on
        exit, so the two expert kernels see slots and need no change. The read
        back is a real host round trip: the top-8 does not exist until the
        router has run.
        """
        var t0 = perf_counter_ns()
        var to0 = perf_counter_ns()
        var fh = open(self.store_path if not self.pinned else "/dev/null", "r")
        var fd = fh._get_raw_fd()
        self.slot_next = 0
        self.b_open[layer] = self.b_open[layer] + Int(perf_counter_ns() - to0)
        var trb0 = perf_counter_ns()
        ctx.enqueue_copy(
            dst_buf=self.slots_h,
            src_buf=DeviceBuffer[DType.int32](ctx, idx_d.unsafe_ptr(), TOPK, owning=False),
        )
        ctx.synchronize()
        self.b_readback[layer] = self.b_readback[layer] + Int(perf_counter_ns() - trb0)
        self._plan(ctx, layer, fd, False)
        var cpu_layer = self.cpu.on and self.geom[layer].ebd == self.geom[layer].eb
        var tw0 = perf_counter_ns()
        ctx.enqueue_copy(
            dst_buf=DeviceBuffer[DType.int32](ctx, idx_d.unsafe_ptr(), TOPK, owning=False),
            src_buf=self.slots_h,
        )
        if self.zc:
            ctx.enqueue_copy(dst_buf=self.hoste_d, src_buf=self.hoste_h)
        if cpu_layer and self.cpu.n > 0:
            ctx.enqueue_copy(dst_buf=self.cpu.wt_d, src_buf=self.cpu.wt_h)
        self.b_writeback[layer] = self.b_writeback[layer] + Int(perf_counter_ns() - tw0)
        var tc1 = perf_counter_ns()
        fh.close()
        self.b_open[layer] = self.b_open[layer] + Int(perf_counter_ns() - tc1)
        self.fetch_ns += Int(perf_counter_ns() - t0)

    def report(self, tokens: Int):
        var rate = Float64(self.hits) / Float64(self.refs) if self.refs > 0 else 0.0
        var eff_rate = Float64(self.hits - self.prefetch_reads) / Float64(self.refs) if self.refs > 0 else 0.0
        print(
            "expert tier: refs", self.refs, " hits", self.hits,
            " prefetch_reads", self.prefetch_reads,
            " hit_rate", rate,
            " effective_hit_rate", eff_rate,
            " pinned_experts", self.pinned_count,
            " draft_skipped", self.draft_skipped,
            " bytes_fetched", self.bytes_fetched,
            " bytes_zc", self.bytes_zc,
            " cpu_misses", self.cpu.misses,
            " cpu_s", Float64(self.cpu.ns) / 1e9,
            " cpu_wait_s", Float64(self.cpu.wait_ns) / 1e9,
            " steps", self.refs // (self.n_layers * TOPK) if self.n_layers > 0 else 0,
            " bytes_per_step", Float64(self.bytes_fetched) / Float64(self.refs // (self.n_layers * TOPK)) if self.refs > 0 else 0.0,
            " fetch_s", Float64(self.fetch_ns) / 1e9,
        )
        if getenv("BARO_TIER_STAMP", "0") == "1":
            print(
                "tier stamp: cycle_us (plan raised to next bell seen)",
                Float64(self.cpu.cycle_ns) / Float64(max(self.cpu.cycle_n, 1)) / 1e3,
                " cpu jobs", self.cpu.jobs,
                " submit_to_done_seen_us", Float64(self.cpu.sub_done_ns) / Float64(max(self.cpu.jobs, 1)) / 1e3,
                " worker_compute_us", Float64(self.cpu.ns) / Float64(max(self.cpu.jobs, 1)) / 1e3,
                " done_seen_to_bell_us", Float64(self.cpu.done_bell_ns) / Float64(max(self.cpu.jobs, 1)) / 1e3,
            )
            print(
                "tier stamp: layer  open_us  readback_us  lru_us  pread_us  copy_us",
                " writeback_us  row_us",
            )
            var sum_ns = 0
            for l in range(self.n_layers):
                var row_ns = (
                    self.b_open[l] + self.b_readback[l] + self.b_lru[l]
                    + self.b_pread[l] + self.b_copy[l] + self.b_writeback[l]
                )
                sum_ns += row_ns
                print(
                    "tier stamp:", l, Float64(self.b_open[l]) / 1e3,
                    Float64(self.b_readback[l]) / 1e3, Float64(self.b_lru[l]) / 1e3,
                    Float64(self.b_pread[l]) / 1e3, Float64(self.b_copy[l]) / 1e3,
                    Float64(self.b_writeback[l]) / 1e3, Float64(row_ns) / 1e3,
                )
            print(
                "tier stamp: sum_s", Float64(sum_ns) / 1e9,
                " fetch_s", Float64(self.fetch_ns) / 1e9,
                " sum_per_step_us",
                Float64(sum_ns) / Float64(self.refs // (self.n_layers * TOPK)) / 1e3 if self.refs > 0 else 0.0,
            )
