"""baro engine: full-model greedy decode for qwen35 (Qwythos-9B), milestone 4.

Loads .work/engine-pack/ (fixed tensor order, 2D bf16 weights pre-transposed
to B-layout), reads prompt token ids from .work/engine-pack/prompt-tokens.txt,
runs the 32-block hybrid stack (24 gated-delta-net + 8 gated full-attention,
MTP block skipped) over a window of up to MROWS tokens at a time, and prints
greedy token ids.

Parity target: byte-identical token ids vs llama.cpp on the same GGUF.
"""
from std.math import ceildiv, sqrt, exp, log
from std.memory import bitcast, unsafe_memcpy
from std.os import getenv, setenv, makedirs
from kvpage import PageTable
from std.sys import exit, has_accelerator
from std.time import perf_counter_ns
from ngram import ngram_propose

from max.algorithm import parallelize
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major
from registry import *


from window import *
from rows import step_rows
from harness import *
from prefix import *
from moe_pack import parse_moe_index, resolve_plain
import latentos.proto as proto
from latent import EngineLatentClient, mint_checkpoint_latent, ingest_checkpoint_latent, hash64
from latent_ipc import MSG_BYTES, ipc_payload_bytes, export_ckpt_ipc, ingest_ckpt_ipc, ipc_refusal, ckpt_region_sha, stage_new, hex_encode, tokens_from_hex
from prefix import _hex_to_bytes
from latentos.ipc import connect_unix_socket
from state import amar_kv4_unpack, amar_kv6_decode, KV4_TAIL, KV5_TAIL, KV4_PACKED, KV4_ROWS, KV6_TAB, KV6_STREAMS
from serve_proto import (
    read_line, cancel_pending, is_cancel_line, idle_cancel_note, json_key, json_int, json_float, json_str,
    SampleParams, default_sample_params, parse_request,
    parse_schema_field, parse_reasoning_field, parse_str_field,
)
from grammar_rt import GrammarRuntime, compile_schema_matcher, compile_tool_matcher, tool_open_observe
from wire import reset_request_knobs, slens_base_ok, slens_base_error, tier_word, ready_words, grammar_word, PATH_PAIR, PATH_SEQS, decide_path, path_word, SEQ_PREFILL, SeqTag, FIN_STOP, FIN_CANCELLED, FIN_ERROR, finish_word, DoneWords, window_exceeded, window_error, line_id, latent_fork_error, latent_resume_error
from grammar.automaton import Bitset
from latent_mlp import LatentMLPAdapter




# --- engine state file (LatentOS use 2: a saved prefix survives the process) --
# BAROST01 | int64 pos, conv_n, ssm_n, kv_n | pack salt (32 B) | int32 tokens[pos]
# | f32 conv[conv_n] | f32 ssm[ssm_n] | f32 K[kv_n] | f32 V[kv_n]
# BAROST02 | same header | f32 conv[conv_n] | f32 ssm[ssm_n]
# | f32 Kscale[kv_n/KVHSTR] | int8 K[kv_n] | f32 Vscale[kv_n/KVHSTR] | int8 V[kv_n]
# K/V are the first ceil(pos/KVPAGE) pages of the page-major pool, so every
# position below pos is included; the checkpoint is the one saved at pos.
# BAROST02 quantizes one KVHSTR block (one page, one attention layer, one kv
# head; KVPAD included but always 0 today) per scale, matching the memory
# layout's own addressing unit exactly, so dequant needs no reshaping.
def _put_i64(mut out: List[UInt8], v: Int):
    for b in range(8):
        out.append(UInt8((v >> (8 * b)) & 0xFF))


def _get_i64_ptr(data: Pointer[UInt8, MutUntrackedOrigin], off: Int) -> Int:
    var v = 0
    for b in range(8):
        v |= Int(data[off + b]) << (8 * b)
    return v


def _get_i64(data: List[UInt8], off: Int) -> Int:
    var v = 0
    for b in range(8):
        v |= Int(data[off + b]) << (8 * b)
    return v


def _flush_kv7_run(mut hi_rle: List[UInt8], sym: Int, run_len: Int):
    if run_len <= 0:
        return
    if sym == 0x7E or sym == 0x7F or run_len >= 4:
        hi_rle.append(UInt8(0x7F))
        hi_rle.append(UInt8(run_len - 1))
        hi_rle.append(UInt8(sym))
    else:
        for _ in range(run_len):
            hi_rle.append(UInt8(sym))


def _encode_kv7_slots(
    conv_ptr: Pointer[Scalar[f32], MutUntrackedOrigin],
    ssm_ptr: Pointer[Scalar[f32], MutUntrackedOrigin],
    mut hi_rle: List[UInt8],
    mut lo_raw: List[UInt8],
):
    var total = CONV_SLOT + SSM_SLOT
    lo_raw = List[UInt8](capacity=total)
    hi_rle = List[UInt8](capacity=total // 2)
    var prev_sym = -1
    var run_len = 0
    for i in range(total):
        var fval = conv_ptr[unsafe_offset=i] if i < CONV_SLOT else ssm_ptr[unsafe_offset=i - CONV_SLOT]
        var bits = Int(bitcast[DType.uint16, 1](SIMD[DType.float16, 1](fval))[0])
        var sym = 0x7E
        if bits != 0:
            sym = (bits >> 8) & 0xFF
            lo_raw.append(UInt8(bits & 0xFF))
        if sym == prev_sym and run_len < 256:
            run_len += 1
        else:
            _flush_kv7_run(hi_rle, prev_sym, run_len)
            prev_sym = sym
            run_len = 1
    _flush_kv7_run(hi_rle, prev_sym, run_len)


def _decode_kv7_slots(
    hi_rle: Pointer[UInt8, MutUntrackedOrigin],
    hi_comp_len: Int,
    lo_raw: Pointer[UInt8, MutUntrackedOrigin],
    conv_dst: Pointer[Scalar[f32], MutUntrackedOrigin],
    ssm_dst: Pointer[Scalar[f32], MutUntrackedOrigin],
) raises:
    var total = CONV_SLOT + SSM_SLOT
    var out_idx = 0
    var lo_idx = 0
    var r = 0
    while r < hi_comp_len:
        var b0 = Int(hi_rle[unsafe_offset=r])
        var count = 1
        var sym = b0
        if b0 == 0x7F:
            count = Int(hi_rle[unsafe_offset=r + 1]) + 1
            sym = Int(hi_rle[unsafe_offset=r + 2])
            r += 3
        else:
            r += 1
        for _ in range(count):
            if out_idx >= total:
                raise Error("BARO_STATE_LOAD: BAROKV7 high-plane overflow")
            var val = Scalar[f32](0.0)
            if sym != 0x7E:
                var lo = Int(lo_raw[unsafe_offset=lo_idx])
                lo_idx += 1
                var bits = UInt16((sym << 8) | lo)
                val = Scalar[f32](bitcast[DType.float16, 1](SIMD[DType.uint16, 1](bits))[0])
            if out_idx < CONV_SLOT:
                conv_dst[unsafe_offset=out_idx] = val
            else:
                ssm_dst[unsafe_offset=out_idx - CONV_SLOT] = val
            out_idx += 1
    if out_idx != total:
        raise Error("BARO_STATE_LOAD: BAROKV7 decoded " + String(out_idx) + " slots, expected " + String(total))


def save_state(
    ctx: DeviceContext, chain: Chain, kc_d: DeviceBuffer[KVT], vc_d: DeviceBuffer[KVT], kvtab_h: HostBuffer[DType.int32],
    path: String, prompt: List[Int], pos: Int,
) raises:
    comptime if KVT != DType.float32 and not KVQ8:
        raise Error("BARO_STATE_SAVE: state files store f32 or int8 (BARO_KVQ=int8) KV; this engine has BARO_KVQ=" + KVQ)
    else:
        # The checkpoint must be THIS prompt's. Matching on pos alone wrote another conversation's
        # conv and SSM state into the file whenever two resident checkpoints shared a position
        # (bench/fork-bytes-check.sh: p05's export carried p02's recurrent state, both at pos 14).
        var want = prefix_hash(chain.salt, prompt, pos)
        var idx = -1
        for i in range(len(chain.items)):
            if chain.items[i].valid and chain.items[i].pos == pos and bytes_eq(chain.items[i].hash, want):
                idx = i
        if idx < 0:
            raise Error("BARO_STATE_SAVE: no committed checkpoint for this prompt at pos " + String(pos))
        var kvn = ceildiv(pos, KVPAGE) * N_ATT * NKVH * KVHSTR
        var kh = ctx.enqueue_create_host_buffer[KVT](kvn)
        var vh = ctx.enqueue_create_host_buffer[KVT](kvn)
        # logical page order through the block table, so the file never depends on the mapping
        comptime PAGE_ELEMS = N_ATT * NKVH * KVHSTR
        for p in range(ceildiv(pos, KVPAGE)):
            var pp = Int(chain.page_of(idx, p)) if chain.npages > 0 else Int(kvtab_h[p])
            ctx.enqueue_copy(dst_buf=kh.create_sub_buffer[KVT](p * PAGE_ELEMS, PAGE_ELEMS), src_buf=DeviceBuffer[KVT](ctx, kc_d.unsafe_ptr().unsafe_offset(pp * PAGE_ELEMS), PAGE_ELEMS, owning=False))
            ctx.enqueue_copy(dst_buf=vh.create_sub_buffer[KVT](p * PAGE_ELEMS, PAGE_ELEMS), src_buf=DeviceBuffer[KVT](ctx, vc_d.unsafe_ptr().unsafe_offset(pp * PAGE_ELEMS), PAGE_ELEMS, owning=False))
        ctx.synchronize()
        # BAROST02 is the state-layer default. Set BARO_STATE_INT8=0 for the
        # legacy f32 BAROST01 format, or BARO_STATE_KV7=1 for plane-split BAROKV7.
        # Exact checkpoint continuation is the default API contract. Int8 KV
        # state remains available as an explicit opt-in for size-oriented use.
        var int8 = getenv("BARO_STATE_INT8", "0") != "0"
        var kv7 = getenv("BARO_STATE_KV7", "0") == "1"
        var head = List[UInt8]()
        var magic = String("BAROST02" if int8 else "BAROST01")
        comptime if KVQ8:
            # BAROST03 (P7): the int8 pool's pages verbatim, rows + page-tail
            # scales; no host quantisation, and load is a plain H2D.
            magic = String("BAROST03")
        if kv7:
            for i in range(7):
                head.append(String("BAROKV7").as_bytes()[i])
            head.append(UInt8(0))
            magic = String("BAROKV7")
        else:
            for i in range(8):
                head.append(magic.as_bytes()[i])
        _put_i64(head, pos)
        _put_i64(head, CONV_SLOT)
        _put_i64(head, SSM_SLOT)
        _put_i64(head, kvn)
        for b in chain.salt:
            head.append(b)
        for t in range(pos):
            for b in range(4):
                head.append(UInt8((prompt[t] >> (8 * b)) & 0xFF))
        with open(path, "w") as f:
            f.write_bytes(Span(head))
            if kv7:
                var hi_rle = List[UInt8]()
                var lo_raw = List[UInt8]()
                _encode_kv7_slots(chain.items[idx].conv_h.unsafe_ptr(), chain.items[idx].ssm_h.unsafe_ptr(), hi_rle, lo_raw)
                var len_buf = List[UInt8]()
                _put_i64(len_buf, len(hi_rle))
                _put_i64(len_buf, len(lo_raw))
                f.write_bytes(Span(len_buf))
                f.write_bytes(Span(hi_rle))
                f.write_bytes(Span(lo_raw))
            else:
                f.write_bytes(Span[UInt8](unsafe_ptr=chain.items[idx].conv_h.unsafe_ptr().unsafe_bitcast[UInt8](), length=CONV_SLOT * 4))
                f.write_bytes(Span[UInt8](unsafe_ptr=chain.items[idx].ssm_h.unsafe_ptr().unsafe_bitcast[UInt8](), length=SSM_SLOT * 4))
            comptime if KVQ8:
                f.write_bytes(Span[UInt8](unsafe_ptr=kh.unsafe_ptr().unsafe_bitcast[UInt8](), length=kvn))
                f.write_bytes(Span[UInt8](unsafe_ptr=vh.unsafe_ptr().unsafe_bitcast[UInt8](), length=kvn))
            else:
                if int8 or kv7:
                    var ngroups = kvn // KVHSTR
                    var kscale = List[Scalar[f32]](unsafe_uninit_length=ngroups)
                    var kq = List[Scalar[i8]](unsafe_uninit_length=kvn)
                    _quantize_kv_int8(kh, kvn, kscale, kq)
                    var vscale = List[Scalar[f32]](unsafe_uninit_length=ngroups)
                    var vq = List[Scalar[i8]](unsafe_uninit_length=kvn)
                    _quantize_kv_int8(vh, kvn, vscale, vq)
                    f.write_bytes(Span[UInt8](unsafe_ptr=kscale.unsafe_ptr().unsafe_bitcast[UInt8](), length=ngroups * 4))
                    f.write_bytes(Span[UInt8](unsafe_ptr=kq.unsafe_ptr().unsafe_bitcast[UInt8](), length=kvn))
                    f.write_bytes(Span[UInt8](unsafe_ptr=vscale.unsafe_ptr().unsafe_bitcast[UInt8](), length=ngroups * 4))
                    f.write_bytes(Span[UInt8](unsafe_ptr=vq.unsafe_ptr().unsafe_bitcast[UInt8](), length=kvn))
                else:
                    f.write_bytes(Span[UInt8](unsafe_ptr=kh.unsafe_ptr().unsafe_bitcast[UInt8](), length=kvn * 4))
                    f.write_bytes(Span[UInt8](unsafe_ptr=vh.unsafe_ptr().unsafe_bitcast[UInt8](), length=kvn * 4))
        print("state saved:", path, " pos", pos, " kv pages", ceildiv(pos, KVPAGE), " format", magic)


def _quantize_kv_int8[T: DType](hb: HostBuffer[T], kvn: Int, mut scales: List[Scalar[f32]], mut qdata: List[Scalar[i8]]):
    var ngroups = kvn // KVHSTR
    for g in range(ngroups):
        var base = g * KVHSTR
        var amax = Scalar[f32](0)
        for i in range(KVHSTR):
            var v = abs(hb[base + i].cast[f32]())
            if v > amax:
                amax = v
        var scale = amax / 127 if amax > 0 else Scalar[f32](1)
        var inv = Scalar[f32](127) / amax if amax > 0 else Scalar[f32](0)
        scales[g] = scale
        for i in range(KVHSTR):
            var qf = round(hb[base + i].cast[f32]() * inv)
            qf = min(max(qf, Scalar[f32](-127)), Scalar[f32](127))
            qdata[base + i] = qf.cast[i8]()


def _dequantize_kv_int8[T: DType](mut hb: HostBuffer[T], data: List[UInt8], scale_off: Int, q_off: Int, kvn: Int):
    var scales = data.unsafe_ptr().unsafe_offset(scale_off).unsafe_bitcast[Scalar[f32]]()
    var qdata = data.unsafe_ptr().unsafe_offset(q_off).unsafe_bitcast[Scalar[i8]]()
    var ngroups = kvn // KVHSTR
    for g in range(ngroups):
        var scale = scales[g]
        var base = g * KVHSTR
        for i in range(KVHSTR):
            hb[base + i] = (qdata[base + i].cast[f32]() * scale).cast[T]()


struct LoadStage(Movable):
    # One pinned host stage per engine, allocated once at startup (P1 stage B,
    # AMDHQ runs/latent-os/P1-stageA-split-2026-09-24.md: the List read of a
    # 2.15 GB f32 file cost 2.19 s, hipHostMalloc per load 0.000 s). buf holds
    # the header, tokens, conv/SSM slots and, for BAROST02, the int8 payload;
    # kh/vh are the f32 KV staging buffers the H2D copies read from. An f32
    # file's K and V land in kh/vh straight from the read.
    var buf: HostBuffer[DType.uint8]
    var kh: HostBuffer[KVT]
    var vh: HostBuffer[KVT]
    var kv4_d: DeviceBuffer[KVT]
    var buf_cap: Int
    var kvn_cap: Int
    var kv4_cap: Int

    def __init__(out self, ctx: DeviceContext, tmax: Int) raises:
        self.kvn_cap = ceildiv(tmax, KVPAGE) * N_ATT * NKVH * KVHSTR
        var ngroups = self.kvn_cap // KVHSTR
        self.buf_cap = 72 + 4 * tmax + (CONV_SLOT + SSM_SLOT) * 4
        self.kv4_cap = 0
        comptime if not KVQ8:
            # BAROST02 payload is staged before the dequant; the int8 pool's
            # BAROST03 goes straight into the (int8) staging pair.
            self.buf_cap += 2 * (ngroups * 4 + self.kvn_cap)
        else:
            # BAROKV4/5 (18-own-codec): per pool block 512 (256 f16) tail bytes +
            # 16384 packed nibbles. K's part is read into kh and V's into vh
            # (each is at most kvn_cap bytes, no extra pinned slab), then
            # unpacked from a device scratch.
            self.kv4_cap = 2 * (self.kvn_cap // KVHSTR) * (KV4_TAIL + KV4_PACKED)
        self.kv4_d = ctx.enqueue_create_buffer[KVT](max(self.kv4_cap, 1))
        self.buf = ctx.enqueue_create_host_buffer[DType.uint8](self.buf_cap)
        self.kh = ctx.enqueue_create_host_buffer[KVT](self.kvn_cap)
        self.vh = ctx.enqueue_create_host_buffer[KVT](self.kvn_cap)
        ctx.synchronize()


def _read_exact(mut f: FileHandle, ptr: Pointer[UInt8, MutUntrackedOrigin], n: Int) raises -> Bool:
    var done = 0
    while done < n:
        var got = f.read(Span[UInt8](unsafe_ptr=ptr.unsafe_offset(done), length=n - done))
        if got <= 0:
            return False
        done += got
    return True


def _dequantize_kv_int8_par[T: DType](mut hb: HostBuffer[T], src: Pointer[UInt8, MutUntrackedOrigin], scale_off: Int, q_off: Int, kvn: Int):
    # Same per-element arithmetic as _dequantize_kv_int8 (bit-exact), split over
    # whole KVHSTR groups so no two workers touch one group.
    var scales = src.unsafe_offset(scale_off).unsafe_bitcast[Scalar[f32]]()
    var qdata = src.unsafe_offset(q_off).unsafe_bitcast[Scalar[i8]]()
    var dst = hb.unsafe_ptr()
    var ngroups = kvn // KVHSTR
    var chunks = 64
    var per = ceildiv(ngroups, chunks)

    def work(c: Int) {per, ngroups, scales, qdata, dst}:
        var g0 = c * per
        var g1 = min(ngroups, g0 + per)
        for g in range(g0, g1):
            var scale = scales[g]
            var base = g * KVHSTR
            for i in range(KVHSTR):
                dst[base + i] = (qdata[base + i].cast[f32]() * scale).cast[T]()

    parallelize(work, chunks)


def load_state(
    ctx: DeviceContext, mut chain: Chain, kc_d: DeviceBuffer[KVT], vc_d: DeviceBuffer[KVT],
    kvtab_h: HostBuffer[DType.int32], mut kvtab_d: DeviceBuffer[DType.int32], tpages: Int,
    path: String, tmax: Int, mut split: List[Float64], mut stage: LoadStage,
) raises -> Int:
    if chain.npages > 0:
        chain.share_reset(ctx, kvtab_h, kvtab_d, tpages)
    comptime if KVT != DType.float32 and not KVQ8:
        raise Error("BARO_STATE_LOAD: state files store f32 or int8 (BARO_KVQ=int8) KV; this engine has BARO_KVQ=" + KVQ)
    else:
        var t0 = perf_counter_ns()
        var data = stage.buf.unsafe_ptr()
        var m1 = String("BAROST01")
        var m2 = String("BAROST02")
        var m3 = String("BAROST03")
        var m4 = String("BAROKV4")
        var m5 = String("BAROKV5")
        var m6 = String("BAROKV6")
        var m7 = String("BAROKV7")
        var pos = 0
        var kvn = 0
        var is_v2 = True
        var is_v4 = False
        var is_v7 = False
        var hi_comp_len = 0
        var t16 = False
        var kv4_bytes = 0
        var kv6 = False
        var kv6_k = 0
        var kv6_v = 0
        var t1 = t0
        var t2 = t0
        var t2b = t0
        var t3 = t0
        with open(path, "r") as f:
            if not _read_exact(f, data, 72):
                raise Error("BARO_STATE_LOAD: file too short")
            var is_v3 = True
            is_v4 = data[7] == 0
            var is_v5 = data[7] == 0
            var is_v6 = data[7] == 0
            is_v7 = data[7] == 0
            for i in range(8):
                if data[i] != m2.as_bytes()[i]:
                    is_v2 = False
                if data[i] != m3.as_bytes()[i]:
                    is_v3 = False
                if i < 7 and data[i] != m4.as_bytes()[i]:
                    is_v4 = False
                if i < 7 and data[i] != m5.as_bytes()[i]:
                    is_v5 = False
                if i < 7 and data[i] != m6.as_bytes()[i]:
                    is_v6 = False
                if i < 7 and data[i] != m7.as_bytes()[i]:
                    is_v7 = False
            t16 = is_v5 or is_v6
            kv6 = is_v6
            is_v4 = is_v4 or is_v5 or is_v6
            comptime if KVQ8:
                if is_v7:
                    is_v3 = True
                if not is_v3 and not is_v4:
                    raise Error("BARO_STATE_LOAD: this engine (BARO_KVQ=int8) reads BAROST03 pool files and BAROKV4/5/6/7 payloads only")
                is_v2 = False
            else:
                if is_v7:
                    is_v2 = True
                if is_v4:
                    raise Error("BARO_STATE_LOAD: BAROKV4 is an int8-pool payload; this engine has f32 KV")
                if is_v3:
                    raise Error("BARO_STATE_LOAD: BAROST03 is an int8-pool file; this engine has f32 KV")
                if not is_v2:
                    for i in range(8):
                        if data[i] != m1.as_bytes()[i]:
                            raise Error("BARO_STATE_LOAD: not a BAROST01/BAROST02/BAROKV7 state file")
            pos = Int(_get_i64_ptr(data, 8))
            kvn = Int(_get_i64_ptr(data, 32))
            if _get_i64_ptr(data, 16) != CONV_SLOT or _get_i64_ptr(data, 24) != SSM_SLOT:
                raise Error("BARO_STATE_LOAD: slot sizes differ from this engine build")
            for i in range(32):
                if data[40 + i] != chain.salt[i]:
                    raise Error("BARO_STATE_LOAD: saved from a different pack")
            if pos < 1 or pos >= tmax or kvn != ceildiv(pos, KVPAGE) * N_ATT * NKVH * KVHSTR or kvn > stage.kvn_cap:
                raise Error("BARO_STATE_LOAD: bad pos/kv size for TMAX " + String(tmax))
            var fixed = 4 * pos + (CONV_SLOT + SSM_SLOT) * (2 if t16 else 4)
            if is_v7:
                if not _read_exact(f, data.unsafe_offset(72), 4 * pos + 16):
                    raise Error("BARO_STATE_LOAD: payload length mismatch")
                hi_comp_len = Int(_get_i64_ptr(data, 72 + 4 * pos))
                var lo_comp_len = Int(_get_i64_ptr(data, 72 + 4 * pos + 8))
                if hi_comp_len < 0 or hi_comp_len > (CONV_SLOT + SSM_SLOT) * 2 or lo_comp_len < 0 or lo_comp_len > (CONV_SLOT + SSM_SLOT):
                    raise Error("BARO_STATE_LOAD: bad BAROKV7 plane lengths")
                var kv7_rem = hi_comp_len + lo_comp_len
                if not _read_exact(f, data.unsafe_offset(72 + 4 * pos + 16), kv7_rem):
                    raise Error("BARO_STATE_LOAD: payload length mismatch")
                fixed = 4 * pos + 16 + kv7_rem
            else:
                if not _read_exact(f, data.unsafe_offset(72), fixed):
                    raise Error("BARO_STATE_LOAD: payload length mismatch")
            var ngroups = kvn // KVHSTR
            var off = 72 + fixed
            comptime KV_ELEM = 1 if KVQ8 else 4
            var nblocks = kvn // KVHSTR
            kv4_bytes = nblocks * ((KV5_TAIL if t16 else KV4_TAIL) + KV4_PACKED)
            if kv6:
                var head6 = nblocks * KV5_TAIL + KV6_TAB + (nblocks * KV6_STREAMS + 1) * 4
                var kp8 = stage.kh.unsafe_ptr().unsafe_bitcast[UInt8]()
                var vp8 = stage.vh.unsafe_ptr().unsafe_bitcast[UInt8]()
                if not _read_exact(f, kp8, head6):
                    raise Error("BARO_STATE_LOAD: payload length mismatch")
                var ks = Int(kp8.unsafe_offset(head6 - 4).unsafe_bitcast[UInt32]()[0])
                if head6 + ks > stage.kvn_cap or not _read_exact(f, kp8.unsafe_offset(head6), ks):
                    raise Error("BARO_STATE_LOAD: payload length mismatch")
                if not _read_exact(f, vp8, head6):
                    raise Error("BARO_STATE_LOAD: payload length mismatch")
                var vs = Int(vp8.unsafe_offset(head6 - 4).unsafe_bitcast[UInt32]()[0])
                if head6 + vs > stage.kvn_cap or not _read_exact(f, vp8.unsafe_offset(head6), vs):
                    raise Error("BARO_STATE_LOAD: payload length mismatch")
                kv6_k = head6 + ks
                kv6_v = head6 + vs
                if kv6_k + kv6_v > stage.kv4_cap:
                    raise Error("BARO_STATE_LOAD: BAROKV6 payload larger than the device scratch")
            elif is_v4:
                if not _read_exact(f, stage.kh.unsafe_ptr().unsafe_bitcast[UInt8](), kv4_bytes):
                    raise Error("BARO_STATE_LOAD: payload length mismatch")
                if not _read_exact(f, stage.vh.unsafe_ptr().unsafe_bitcast[UInt8](), kv4_bytes):
                    raise Error("BARO_STATE_LOAD: payload length mismatch")
            elif is_v2:
                if not _read_exact(f, data.unsafe_offset(off), 2 * (ngroups * 4 + kvn)):
                    raise Error("BARO_STATE_LOAD: payload length mismatch")
            else:
                if not _read_exact(f, stage.kh.unsafe_ptr().unsafe_bitcast[UInt8](), kvn * KV_ELEM):
                    raise Error("BARO_STATE_LOAD: payload length mismatch")
                if not _read_exact(f, stage.vh.unsafe_ptr().unsafe_bitcast[UInt8](), kvn * KV_ELEM):
                    raise Error("BARO_STATE_LOAD: payload length mismatch")
            var probe = List[UInt8](capacity=1)
            probe.append(0)
            if f.read(Span[UInt8](unsafe_ptr=probe.unsafe_ptr(), length=1)) > 0:
                raise Error("BARO_STATE_LOAD: payload length mismatch")
            t1 = perf_counter_ns()
            if chain.cap == 0:
                raise Error("BARO_STATE_LOAD: needs a checkpoint slot")
            var tokens = List[Int](capacity=pos)
            for t in range(pos):
                var v = 0
                for b in range(4):
                    v |= Int(data[72 + 4 * t + b]) << (8 * b)
                tokens.append(v)
            var idx = 0
            for i in range(len(chain.items)):
                if not chain.items[i].valid:
                    idx = i
                    break
            chain.install(idx, pos, prefix_hash(chain.salt, tokens, pos), True, True)
            var coff = 72 + 4 * pos
            if t16:
                var h16 = data.unsafe_offset(coff).unsafe_bitcast[Float16]()
                var cdst = chain.items[idx].conv_h.unsafe_ptr()
                var sdst = chain.items[idx].ssm_h.unsafe_ptr()
                for i in range(0, CONV_SLOT, 16):
                    cdst.unsafe_offset(i).store(h16.unsafe_offset(i).load[width=16]().cast[DType.float32]())
                for i in range(0, SSM_SLOT, 16):
                    sdst.unsafe_offset(i).store(h16.unsafe_offset(CONV_SLOT + i).load[width=16]().cast[DType.float32]())
            elif is_v7:
                _decode_kv7_slots(
                    data.unsafe_offset(coff + 16),
                    hi_comp_len,
                    data.unsafe_offset(coff + 16 + hi_comp_len),
                    chain.items[idx].conv_h.unsafe_ptr(),
                    chain.items[idx].ssm_h.unsafe_ptr(),
                )
            else:
                unsafe_memcpy(dest=chain.items[idx].conv_h.unsafe_ptr().unsafe_bitcast[UInt8](), src=data.unsafe_offset(coff), count=CONV_SLOT * 4)
                unsafe_memcpy(dest=chain.items[idx].ssm_h.unsafe_ptr().unsafe_bitcast[UInt8](), src=data.unsafe_offset(coff + CONV_SLOT * 4), count=SSM_SLOT * 4)
            t2 = perf_counter_ns()
            t2b = t2
            if is_v2:
                var k_scale_off = off
                var k_q_off = off + ngroups * 4
                var v_scale_off = k_q_off + kvn
                var v_q_off = v_scale_off + ngroups * 4
                _dequantize_kv_int8_par(stage.kh, data, k_scale_off, k_q_off, kvn)
                _dequantize_kv_int8_par(stage.vh, data, v_scale_off, v_q_off, kvn)
            t3 = perf_counter_ns()
        comptime if KVQ8:
            if kv6:
                var nb = kvn // KVHSTR
                ctx.enqueue_copy(dst_buf=stage.kv4_d.create_sub_buffer[KVT](0, kv6_k), src_buf=stage.kh.create_sub_buffer[KVT](0, kv6_k))
                ctx.enqueue_copy(dst_buf=stage.kv4_d.create_sub_buffer[KVT](kv6_k, kv6_v), src_buf=stage.vh.create_sub_buffer[KVT](0, kv6_v))
                var d = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(stage.kv4_d.unsafe_ptr()))
                var kp = Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=Int(kc_d.unsafe_ptr()))
                var vp = Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=Int(vc_d.unsafe_ptr()))
                ctx.enqueue_function[amar_kv6_decode](d, kp, Int32(nb), grid_dim=nb, block_dim=KV6_STREAMS)
                ctx.enqueue_function[amar_kv6_decode](d.unsafe_offset(kv6_k), vp, Int32(nb), grid_dim=nb, block_dim=KV6_STREAMS)
            elif is_v4:
                var nb = kvn // KVHSTR
                var kb = kv4_bytes
                ctx.enqueue_copy(dst_buf=stage.kv4_d.create_sub_buffer[KVT](0, kb), src_buf=stage.kh.create_sub_buffer[KVT](0, kb))
                ctx.enqueue_copy(dst_buf=stage.kv4_d.create_sub_buffer[KVT](kb, kb), src_buf=stage.vh.create_sub_buffer[KVT](0, kb))
                var d = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(stage.kv4_d.unsafe_ptr()))
                var kp = Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=Int(kc_d.unsafe_ptr()))
                var vp = Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=Int(vc_d.unsafe_ptr()))
                if t16:
                    ctx.enqueue_function[amar_kv4_unpack[True]](d, kp, Int32(nb), grid_dim=nb, block_dim=256)
                    ctx.enqueue_function[amar_kv4_unpack[True]](d.unsafe_offset(kb), vp, Int32(nb), grid_dim=nb, block_dim=256)
                else:
                    ctx.enqueue_function[amar_kv4_unpack[False]](d, kp, Int32(nb), grid_dim=nb, block_dim=256)
                    ctx.enqueue_function[amar_kv4_unpack[False]](d.unsafe_offset(kb), vp, Int32(nb), grid_dim=nb, block_dim=256)
            else:
                ctx.enqueue_copy(dst_buf=DeviceBuffer[KVT](ctx, kc_d.unsafe_ptr(), kvn, owning=False), src_buf=stage.kh.create_sub_buffer[KVT](0, kvn))
                ctx.enqueue_copy(dst_buf=DeviceBuffer[KVT](ctx, vc_d.unsafe_ptr(), kvn, owning=False), src_buf=stage.vh.create_sub_buffer[KVT](0, kvn))
        else:
            ctx.enqueue_copy(dst_buf=DeviceBuffer[KVT](ctx, kc_d.unsafe_ptr(), kvn, owning=False), src_buf=stage.kh.create_sub_buffer[KVT](0, kvn))
            ctx.enqueue_copy(dst_buf=DeviceBuffer[KVT](ctx, vc_d.unsafe_ptr(), kvn, owning=False), src_buf=stage.vh.create_sub_buffer[KVT](0, kvn))
        # the file holds logical pages 0..n-1 contiguously, so the table is the identity after a load
        for i in range(tpages):
            kvtab_h[i] = Int32(i)
        ctx.enqueue_copy(dst_buf=kvtab_d, src_buf=kvtab_h)
        ctx.synchronize()
        var t4 = perf_counter_ns()
        split.clear()
        split.append(Float64(t1 - t0) / 1e9)
        split.append(Float64(t2 - t1) / 1e9)
        split.append(Float64(t2b - t2) / 1e9)
        split.append(Float64(t3 - t2b) / 1e9)
        split.append(Float64(t4 - t3) / 1e9)
        return pos


def state_tokens(path: String) raises -> List[Int]:
    # The token list a state file was saved with (header pos, then 4 bytes per
    # id), for a request that names the file and sends only its suffix.
    var hdr: List[UInt8]
    with open(path, "r") as f:
        hdr = f.read_bytes(72)
        if len(hdr) < 72:
            raise Error("BARO_STATE_LOAD: file too short")
        var pos = _get_i64(hdr, 8)
        if pos < 1 or pos > TCAP:
            raise Error("BARO_STATE_LOAD: bad pos in state file header")
        var raw = f.read_bytes(4 * pos)
        if len(raw) != 4 * pos:
            raise Error("BARO_STATE_LOAD: payload length mismatch")
        var toks = List[Int](capacity=pos)
        for t in range(pos):
            var v = 0
            for b in range(4):
                v |= Int(raw[4 * t + b]) << (8 * b)
            toks.append(v)
        return toks^


def state_hot(chain: Chain, path: String, prompt: List[Int]) raises -> Int:
    # P2a hot fork by reference: the chain already holds this file's prefix
    # (pos and hash from its 72-byte header match a valid item whose KV is
    # still in the pool), so the file is not read. Returns the item index or
    # -1. A header that names another pack or a bad file is not a refusal
    # here: the full load path raises the same errors a moment later.
    var hdr: List[UInt8]
    with open(path, "r") as f:
        hdr = f.read_bytes(72)
    if len(hdr) < 72:
        return -1
    var m1 = String("BAROST01")
    var m2 = String("BAROST02")
    var m3 = String("BAROST03")
    var ok = True
    for i in range(8):
        if hdr[i] != m1.as_bytes()[i] and hdr[i] != m2.as_bytes()[i] and hdr[i] != m3.as_bytes()[i] and not (i < 7 and (hdr[i] == String("BAROKV4").as_bytes()[i] or hdr[i] == String("BAROKV5").as_bytes()[i] or hdr[i] == String("BAROKV6").as_bytes()[i] or hdr[i] == String("BAROKV7").as_bytes()[i]) or i == 7 and hdr[i] == 0):
            ok = False
    if not ok:
        return -1
    for i in range(32):
        if hdr[40 + i] != chain.salt[i]:
            return -1
    var pos = _get_i64(hdr, 8)
    if pos < 1 or pos > len(prompt) - 1:
        return -1
    var h = prefix_hash(chain.salt, prompt, pos)
    for i in range(len(chain.items)):
        if chain.items[i].valid and chain.items[i].pos == pos and bytes_eq(h, chain.items[i].hash):
            return i
    return -1


def slens_step(row: HostBuffer[f32], mut base: List[List[Float32]], mut acc: List[Float32], mut cnt: Int, mut kl: Float64, mut dnll: Float64, mut dld: Float64, mut nll: Float64, c: Int, k: Int, flags: Int, id: Int, tgt: Int = -1, foil: Int = -1) raises:
    """Steer lens, context c. k == 0: keep log_softmax(row) as the base of c. k > 0: add the log-prob change against
    that base into acc (flags bit0: start a new sum); flags bit1: print the k tokens whose mean change rose most (up)
    and fell most (dn) over the contexts added since the last start."""
    var p = row.unsafe_ptr()
    var m = p[0]
    for j in range(VOCAB):
        m = max(m, p[j])
    var z: Float64 = 0
    for j in range(VOCAB):
        z += Float64(exp(p[j] - m))
    var lse = m + Float32(log(z))
    while len(base) <= c:
        base.append(List[Float32]())
    if k == 0:
        base[c] = List[Float32](length=VOCAB, fill=0)
        for j in range(VOCAB):
            base[c][j] = p[j] - lse
        return
    if len(base[c]) != VOCAB:
        raise Error("slens: no base for context " + String(c))
    if flags & 1 != 0 or len(acc) != VOCAB:
        acc = List[Float32](length=VOCAB, fill=0)
        cnt = 0
        kl = 0
        dnll = 0
        dld = 0
        nll = 0
    var kc: Float64 = 0
    for j in range(VOCAB):
        var dl = p[j] - lse - base[c][j]
        acc[j] += dl
        kc -= Float64(exp(base[c][j])) * Float64(dl)  # KL(base || steered) = sum p_base (lsm_base - lsm_steered)
    kl += kc
    if tgt >= 0 and tgt < VOCAB:
        dnll += Float64(base[c][tgt] - (p[tgt] - lse))  # NLL change of the target token: nll_new - nll_base
        nll += Float64(lse - p[tgt])  # absolute NLL of the target token
        if foil >= 0 and foil < VOCAB:
            dld += Float64((p[tgt] - lse - (p[foil] - lse)) - (base[c][tgt] - base[c][foil]))  # change of logit(target) - logit(foil)
    cnt += 1
    if flags & 2 == 0:
        return
    var uv = List[Float32](length=k, fill=-1e30)
    var ui = List[Int](length=k, fill=-1)
    var dv = List[Float32](length=k, fill=-1e30)
    var di = List[Int](length=k, fill=-1)
    var inv = Float32(1.0) / Float32(cnt)
    for j in range(VOCAB):
        var d = acc[j] * inv
        for sgn in range(2):
            var x = d if sgn == 0 else -d
            if x <= (uv[k - 1] if sgn == 0 else dv[k - 1]):
                continue
            var q = k - 1
            while q > 0 and (uv[q - 1] if sgn == 0 else dv[q - 1]) < x:
                if sgn == 0:
                    uv[q] = uv[q - 1]
                    ui[q] = ui[q - 1]
                else:
                    dv[q] = dv[q - 1]
                    di[q] = di[q - 1]
                q -= 1
            if sgn == 0:
                uv[q] = x
                ui[q] = j
            else:
                dv[q] = x
                di[q] = j
    var out = String("{\"id\":") + String(id) + ",\"slens\":{\"up\":["
    for q in range(k):
        out += ("," if q > 0 else "") + "[" + String(ui[q]) + "," + String(uv[q]) + "]"
    out += "],\"dn\":["
    for q in range(k):
        out += ("," if q > 0 else "") + "[" + String(di[q]) + "," + String(-dv[q]) + "]"
    print(out + "],\"kl\":" + String(kl / Float64(cnt)) + ("" if tgt < 0 else ",\"dnll\":" + String(dnll / Float64(cnt)) + ",\"nll\":" + String(nll / Float64(cnt)) + ("" if foil < 0 else ",\"dld\":" + String(dld / Float64(cnt)))) + "}}")


def a3b_state() -> WindowState:
    return WindowState(pos=0, pos_prev=0, ring=0, n_drafted=0, n_accepted=0, n_spec_windows=0, n_dumped=0, tp=0, tq=0, pf_att=0, pf_ssm=0, pf_ffn=0, pf_head=0, pf_proc=0, pf_draft=0, fc=[0, 0, 0, 0, 0, 0], pc=[0, 0, 0, 0, 0, 0, 0, 0], ac=[0, 0, 0, 0, 0, 0], p3=[0, 0, 0, 0], pfx=[0, 0, 0, 0], grammar=None, grammar_mask=Bitset(1), grammar_pending_think=False, grammar_think_buf=List[UInt8](), grammar_stop=False, grammar_masked_draws=0, grammar_accepted=0, tool_re="", tool_buf=List[UInt8](), grammar_tool=False)


def parse_a3b_request(line: String, tmax: Int, mut req_id: Int, mut prompt: List[Int], mut gen_n: Int, mut sample: SampleParams) raises -> String:
    var spec = False
    var has_spec = False
    var stop = List[List[Int]]()
    var ckpt = List[Int]()
    var state_save = String("")
    var state_load = String("")
    var state_prefix = False
    var semantic_manifest_path = String("")
    var semantic_values_path = String("")
    var semantic_positions = 0
    var semantic_now_s = Int64(0)
    var err = parse_request(line, req_id, prompt, gen_n, spec, has_spec, stop, ckpt, sample, state_save, state_load, state_prefix, semantic_manifest_path, semantic_values_path, semantic_positions, semantic_now_s)
    if err != "":
        return err
    if sample.embed == 1:
        return "embed is not wired on the multi-sequence path (BARO_SEQS > 1); send it to a single-sequence engine"
    if len(prompt) < 1:
        return "empty prompt"
    if gen_n < 1:
        return "n must be >= 1"
    if len(prompt) + gen_n > tmax:
        return "prompt+n exceeds TMAX " + String(tmax)
    if spec:
        return "A3B requires spec=false"
    if len(stop) > 0 or len(ckpt) > 0 or state_save != "" or state_load != "" or state_prefix:
        return "A3B accepts only prompt, n, and temperature=0"
    if json_key(line, "force") >= 0 or json_key(line, "schema") >= 0 or sample.temperature != 0:
        return "A3B accepts only greedy requests"
    return ""


def run_a3b_pair(
    ctx: DeviceContext, mut bufs: WindowBufs, line_a: String, line_b: String,
    pack_q4: Bool, draft_q4: Bool, q4_off: Int, e: Int, kcfg: Int, spec_dbg: Bool,
    expert_trace: Bool, att_split: Int, dot3: Bool, pf_chunk: Int, pf_on: Bool,
    mega: Bool, mega_win: Bool, fr_k: Int, fr_off: Int, fr_ids_off: Int, tmax: Int,
    kvtab_mode: String,
) raises:
    var id_a = 0
    var id_b = 0
    var prompt_a = List[Int]()
    var prompt_b = List[Int]()
    var n_a = 0
    var n_b = 0
    var sample_a = default_sample_params()
    var sample_b = default_sample_params()
    var err = parse_a3b_request(line_a, tmax, id_a, prompt_a, n_a, sample_a)
    if err != "":
        print(err_line(id_a, err))
        return
    err = parse_a3b_request(line_b, tmax, id_b, prompt_b, n_b, sample_b)
    if err != "":
        print(err_line(id_b, err))
        return
    if id_a == id_b:
        print(err_line(id_b, "A3B request ids must be distinct"))
        return
    var page_alloc = PageTable(SEQ_CAP * bufs.tpages)
    var pages_a = page_alloc.alloc(bufs.tpages)
    var pages_b = page_alloc.alloc(bufs.tpages)
    var view_a = sequence_bufs(ctx, bufs, 0, tmax)
    var view_b = sequence_bufs(ctx, bufs, 1, tmax)
    for p in range(bufs.tpages):
        var pa = pages_a[p]
        var pb = pages_b[p]
        if kvtab_mode == "reverse":
            pa = pages_a[bufs.tpages - 1 - p]
            pb = pages_b[bufs.tpages - 1 - p]
        bufs.kvtab_h[p] = Int32(pa)
        bufs.kvtab_h[bufs.tpages + p] = Int32(pb)
    ctx.enqueue_memset(view_a.convstate_d, 0)
    ctx.enqueue_memset(view_a.sstate_d, 0)
    ctx.enqueue_memset(view_a.kc_d, 0)
    ctx.enqueue_memset(view_a.vc_d, 0)
    ctx.enqueue_memset(view_a.toks_d, 0)
    ctx.enqueue_memset(view_b.convstate_d, 0)
    ctx.enqueue_memset(view_b.sstate_d, 0)
    ctx.enqueue_memset(view_b.kc_d, 0)
    ctx.enqueue_memset(view_b.vc_d, 0)
    ctx.enqueue_memset(view_b.toks_d, 0)
    ctx.enqueue_copy(dst_buf=view_a.kvtab_d, src_buf=bufs.kvtab_h.create_sub_buffer[DType.int32](0, bufs.tpages))
    ctx.enqueue_copy(dst_buf=view_b.kvtab_d, src_buf=bufs.kvtab_h.create_sub_buffer[DType.int32](bufs.tpages, bufs.tpages))
    var toks_a = ctx.enqueue_create_host_buffer[DType.int32](tmax)
    var toks_b = ctx.enqueue_create_host_buffer[DType.int32](tmax)
    for i in range(len(prompt_a)):
        toks_a[i] = Int32(prompt_a[i])
    for i in range(len(prompt_b)):
        toks_b[i] = Int32(prompt_b[i])
    ctx.enqueue_copy(dst_buf=view_a.toks_d, src_buf=toks_a)
    ctx.enqueue_copy(dst_buf=view_b.toks_d, src_buf=toks_b)
    ctx.synchronize()
    print("A3B slots: id", id_a, "slot 0 pages", pages_a[0], "..", pages_a[len(pages_a) - 1], "id", id_b, "slot 1 pages", pages_b[0], "..", pages_b[len(pages_b) - 1], "mapping", kvtab_mode)
    var pf_rows_a = len(prompt_a) - 1 if pf_on and len(prompt_a) - 1 >= PF_MIN else 0
    var pf_rows_b = len(prompt_b) - 1 if pf_on and len(prompt_b) - 1 >= PF_MIN else 0
    var pf_tail_a = pf_rows_a % MROWS if pf_rows_a > 0 else 0
    var pf_tail_b = pf_rows_b % MROWS if pf_rows_b > 0 else 0
    if pf_tail_a == 0 and pf_rows_a > 0:
        pf_tail_a = MROWS
    if pf_tail_b == 0 and pf_rows_b > 0:
        pf_tail_b = MROWS
    var cfg_a = WindowCfg(pack_q4=pack_q4, draft_q4=draft_q4, q4_off=q4_off, fr_k=fr_k, fr_off=fr_off, fr_ids_off=fr_ids_off, e=e, kcfg=kcfg, spec=False, spec_dbg=spec_dbg, expert_trace=expert_trace, serve=True, req_id=id_a, prof=False, pf2=False, pf3=False, pf4=False, dump=False, dump4=False, dump_layer=0, mega=mega, att_split=att_split, mega_win=mega_win, dot3=dot3, pf_chunk=pf_chunk, pf_rows=pf_rows_a, pf_tail=pf_tail_a, n_total=len(prompt_a) + n_a, replay_m=MROWS, n_prompt=len(prompt_a), sample=sample_a.copy(), dump_pen=False, mrope=False, inject=False)
    var cfg_b = cfg_a.copy()
    cfg_b.req_id = id_b
    cfg_b.pf_rows = pf_rows_b
    cfg_b.pf_tail = pf_tail_b
    cfg_b.n_total = len(prompt_b) + n_b
    cfg_b.n_prompt = len(prompt_b)
    cfg_b.sample = sample_b.copy()
    var st_a = a3b_state()
    var st_b = a3b_state()
    var alive_a = True
    var alive_b = True
    while alive_a or alive_b:
        if alive_a:
            step_window(ctx, view_a, cfg_a, st_a)
            ctx.synchronize()
            print("A3B boundary: resident ids", id_a, id_b, "slots 0,1")
            if st_a.pos >= cfg_a.n_total - 1:
                alive_a = False
        if alive_b:
            step_window(ctx, view_b, cfg_b, st_b)
            ctx.synchronize()
            print("A3B boundary: resident ids", id_a, id_b, "slots 0,1")
            if st_b.pos >= cfg_b.n_total - 1:
                alive_b = False
    ctx.enqueue_copy(dst_buf=toks_a, src_buf=view_a.toks_d)
    ctx.enqueue_copy(dst_buf=toks_b, src_buf=view_b.toks_d)
    ctx.synchronize()
    var generated_a = st_a.pos + 1 - len(prompt_a)
    var generated_b = st_b.pos + 1 - len(prompt_b)
    print("A3B done: id", id_a, "n", generated_a)
    print("{\"id\":" + String(id_a) + ",\"done\":true,\"n\":" + String(generated_a) + ",\"prefill_s\":0.0,\"decode_s\":0.0,\"tok_s\":0.0,\"cached\":0,\"prefill_rows\":0,\"restore_s\":0.0,\"checkpoints\":0,\"finish\":\"length\"}")
    print("A3B done: id", id_b, "n", generated_b)
    print("{\"id\":" + String(id_b) + ",\"done\":true,\"n\":" + String(generated_b) + ",\"prefill_s\":0.0,\"decode_s\":0.0,\"tok_s\":0.0,\"cached\":0,\"prefill_rows\":0,\"restore_s\":0.0,\"checkpoints\":0,\"finish\":\"length\"}")


def run_a3c_serial_row(
    ctx: DeviceContext, mut bufs: WindowBufs, mut page_alloc: PageTable,
    line: String, seq: Int, pack_q4: Bool, draft_q4: Bool, q4_off: Int, e: Int,
    kcfg: Int, spec_dbg: Bool, expert_trace: Bool, att_split: Int, dot3: Bool,
    pf_chunk: Int, pf_on: Bool, mega: Bool, mega_win: Bool, fr_k: Int,
    fr_off: Int, fr_ids_off: Int, tmax: Int, kvtab_mode: String,
) raises:
    var req_id = 0
    var prompt = List[Int]()
    var gen_n = 0
    var sample = default_sample_params()
    var err = parse_a3b_request(line, tmax, req_id, prompt, gen_n, sample)
    if err != "":
        print(err_line(req_id, err))
        return
    var pages = page_alloc.alloc(bufs.tpages)
    var view = sequence_bufs(ctx, bufs, seq, tmax)
    for p in range(bufs.tpages):
        var phys = pages[p]
        if kvtab_mode == "reverse":
            phys = pages[bufs.tpages - 1 - p]
        bufs.kvtab_h[seq * bufs.tpages + p] = Int32(phys)
    ctx.enqueue_memset(view.convstate_d, 0)
    ctx.enqueue_memset(view.sstate_d, 0)
    ctx.enqueue_memset(view.kc_d, 0)
    ctx.enqueue_memset(view.vc_d, 0)
    ctx.enqueue_memset(view.toks_d, 0)
    ctx.enqueue_copy(
        dst_buf=view.kvtab_d,
        src_buf=bufs.kvtab_h.create_sub_buffer[DType.int32](seq * bufs.tpages, bufs.tpages),
    )
    var toks = ctx.enqueue_create_host_buffer[DType.int32](tmax)
    for i in range(len(prompt)):
        toks[i] = Int32(prompt[i])
    ctx.enqueue_copy(dst_buf=view.toks_d, src_buf=toks)
    ctx.synchronize()
    print("A3C serial row: id", req_id, "slot", seq, "pages", pages[0], "..", pages[len(pages) - 1], "mapping", kvtab_mode)
    var pf_rows = len(prompt) - 1 if pf_on and len(prompt) - 1 >= PF_MIN else 0
    var pf_tail = pf_rows % MROWS if pf_rows > 0 else 0
    if pf_tail == 0 and pf_rows > 0:
        pf_tail = MROWS
    var cfg = WindowCfg(pack_q4=pack_q4, draft_q4=draft_q4, q4_off=q4_off, fr_k=fr_k, fr_off=fr_off, fr_ids_off=fr_ids_off, e=e, kcfg=kcfg, spec=False, spec_dbg=spec_dbg, expert_trace=expert_trace, serve=True, req_id=req_id, prof=False, pf2=False, pf3=False, pf4=False, dump=False, dump4=False, dump_layer=0, mega=mega, att_split=att_split, mega_win=mega_win, dot3=dot3, pf_chunk=pf_chunk, pf_rows=pf_rows, pf_tail=pf_tail, n_total=len(prompt) + gen_n, replay_m=MROWS, n_prompt=len(prompt), sample=sample.copy(), dump_pen=False, mrope=False, inject=False)
    var st = a3b_state()
    var t0 = perf_counter_ns()
    while st.pos < cfg.n_total - 1:
        step_window(ctx, view, cfg, st)
        ctx.synchronize()
    var elapsed = Float64(perf_counter_ns() - t0) / 1e9
    var generated = st.pos + 1 - len(prompt)
    print("A3C serial done: id", req_id, "n", generated, "wall_s", elapsed)
    print("{\"id\":" + String(req_id) + ",\"done\":true,\"n\":" + String(generated) + ",\"a3c_mode\":\"serial\",\"wall_s\":" + String(elapsed) + ",\"finish\":\"length\"}")


def run_a3c_serial(
    ctx: DeviceContext, mut bufs: WindowBufs, line0: String, line1: String,
    line2: String, line3: String, pack_q4: Bool, draft_q4: Bool, q4_off: Int,
    e: Int, kcfg: Int, spec_dbg: Bool, expert_trace: Bool, att_split: Int,
    dot3: Bool, pf_chunk: Int, pf_on: Bool, mega: Bool, mega_win: Bool,
    fr_k: Int, fr_off: Int, fr_ids_off: Int, tmax: Int, kvtab_mode: String,
) raises:
    var page_alloc = PageTable(SEQ_CAP * bufs.tpages)
    print("A3C batch: mode serial rows 4")
    run_a3c_serial_row(ctx, bufs, page_alloc, line0, 0, pack_q4, draft_q4, q4_off, e, kcfg, spec_dbg, expert_trace, att_split, dot3, pf_chunk, pf_on, mega, mega_win, fr_k, fr_off, fr_ids_off, tmax, kvtab_mode)
    run_a3c_serial_row(ctx, bufs, page_alloc, line1, 1, pack_q4, draft_q4, q4_off, e, kcfg, spec_dbg, expert_trace, att_split, dot3, pf_chunk, pf_on, mega, mega_win, fr_k, fr_off, fr_ids_off, tmax, kvtab_mode)
    run_a3c_serial_row(ctx, bufs, page_alloc, line2, 2, pack_q4, draft_q4, q4_off, e, kcfg, spec_dbg, expert_trace, att_split, dot3, pf_chunk, pf_on, mega, mega_win, fr_k, fr_off, fr_ids_off, tmax, kvtab_mode)
    run_a3c_serial_row(ctx, bufs, page_alloc, line3, 3, pack_q4, draft_q4, q4_off, e, kcfg, spec_dbg, expert_trace, att_split, dot3, pf_chunk, pf_on, mega, mega_win, fr_k, fr_off, fr_ids_off, tmax, kvtab_mode)
    print("A3C batch done: mode serial rows 4")


def run_a3c_serial_pair(
    ctx: DeviceContext, mut bufs: WindowBufs, line0: String, line1: String,
    pack_q4: Bool, draft_q4: Bool, q4_off: Int, e: Int, kcfg: Int,
    spec_dbg: Bool, expert_trace: Bool, att_split: Int, dot3: Bool,
    pf_chunk: Int, pf_on: Bool, mega: Bool, mega_win: Bool, fr_k: Int,
    fr_off: Int, fr_ids_off: Int, tmax: Int, kvtab_mode: String,
) raises:
    var page_alloc = PageTable(SEQ_CAP * bufs.tpages)
    print("A3C batch: mode serial rows 2")
    var id0 = 0
    var id1 = 0
    var probe0 = List[Int]()
    var probe1 = List[Int]()
    var n0 = 0
    var n1 = 0
    var sp0 = default_sample_params()
    var sp1 = default_sample_params()
    var err0 = parse_a3b_request(line0, tmax, id0, probe0, n0, sp0)
    var err1 = parse_a3b_request(line1, tmax, id1, probe1, n1, sp1)
    if err0 != "":
        print(err_line(id0, err0))
        return
    if err1 != "":
        print(err_line(id1, err1))
        return
    if id0 == id1:
        print(err_line(id1, "A3C request ids must be distinct"))
        return
    print("A3C step: mode serial ids", id0, id1, "slots 0,1")
    run_a3c_serial_row(ctx, bufs, page_alloc, line0, 0, pack_q4, draft_q4, q4_off, e, kcfg, spec_dbg, expert_trace, att_split, dot3, pf_chunk, pf_on, mega, mega_win, fr_k, fr_off, fr_ids_off, tmax, kvtab_mode)
    run_a3c_serial_row(ctx, bufs, page_alloc, line1, 1, pack_q4, draft_q4, q4_off, e, kcfg, spec_dbg, expert_trace, att_split, dot3, pf_chunk, pf_on, mega, mega_win, fr_k, fr_off, fr_ids_off, tmax, kvtab_mode)
    print("A3C batch done: mode serial rows 2")



# --- Vision (docs/vision.md): image embeddings from the mtmd-embed sidecar -----
# A request may carry "embeds":["/abs/path.bemb", ...], one file per run of
# <|image_pad|> ids in the prompt, in order. The file (tools/mtmd-embed.c)
# holds the projected token embeddings and their mrope decoder positions.
comptime IMAGE_PAD = 248056


def parse_embeds(line: String, mut embeds: List[String]) -> String:
    var i = json_key(line, "embeds")
    if i < 0:
        return ""
    var b = line.as_bytes()
    if i >= len(b) or b[i] != 91:
        return "embeds must be an array of file paths"
    i += 1
    while True:
        while i < len(b) and (b[i] == 32 or b[i] == 44):
            i += 1
        if i >= len(b):
            return "unterminated embeds array"
        if b[i] == 93:
            break
        var path = String("")
        if not json_str(line, i, path):
            return "embeds must be an array of file paths"
        i += 1
        embeds.append(path)
    return ""


@always_inline
def le32(b: List[UInt8], o: Int) -> Int:
    return Int(b[o]) | (Int(b[o + 1]) << 8) | (Int(b[o + 2]) << 16) | (Int(b[o + 3]) << 24)


def stage_vision(ctx: DeviceContext, mut bufs: WindowBufs, prompt: List[Int], embeds: List[String], tmax: Int) raises -> String:
    """Fills bufs.pos3_d (3-axis rope positions per token), bufs.imgmap_d
    (token -> image row or -1) and bufs.img_d (image rows) for one request.
    Text tokens get (p, p, p); an image run gets (p + t, p + y, p + x) from
    its file and advances p by the file's n_pos, as llama.cpp's mtmd does."""
    for i in range(tmax):
        bufs.imgmap_h[i] = -1
    var rows = 0
    var p = 0
    var i = 0
    var k = 0
    while i < len(prompt):
        if prompt[i] != IMAGE_PAD:
            bufs.pos3_h[3 * i] = Int32(p)
            bufs.pos3_h[3 * i + 1] = Int32(p)
            bufs.pos3_h[3 * i + 2] = Int32(p)
            p += 1
            i += 1
            continue
        var run = 0
        while i + run < len(prompt) and prompt[i + run] == IMAGE_PAD:
            run += 1
        if k >= len(embeds):
            return "prompt has more image runs than embeds (" + String(len(embeds)) + ")"
        var buf: List[UInt8]
        with open(embeds[k], "r") as f:
            buf = f.read_bytes()
        if len(buf) < 20 or buf[0] != 66 or buf[1] != 69 or buf[2] != 77 or buf[3] != 66:
            return "not a BEMB file: " + embeds[k]
        var n_tok = le32(buf, 8)
        var n_embd = le32(buf, 12)
        var n_pos = le32(buf, 16)
        if n_embd != H:
            return "embeds width " + String(n_embd) + " != H " + String(H) + ": " + embeds[k]
        if n_tok != run:
            return "image " + String(k) + " has " + String(n_tok) + " tokens but the prompt run has " + String(run)
        if rows + n_tok > IMG_CAP:
            return "image tokens exceed IMG_CAP " + String(IMG_CAP)
        var hdr = 20 + 12 * n_tok
        if len(buf) != hdr + 4 * n_tok * H:
            return "embeds file size mismatch: " + embeds[k]
        var fp = UnsafePointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=Int(buf.unsafe_ptr())).unsafe_offset(hdr).unsafe_bitcast[Float32]()
        for j in range(n_tok):
            var t = le32(buf, 20 + 12 * j)
            var x = le32(buf, 24 + 12 * j)
            var y = le32(buf, 28 + 12 * j)
            bufs.pos3_h[3 * (i + j)] = Int32(p + t)
            bufs.pos3_h[3 * (i + j) + 1] = Int32(p + y)
            bufs.pos3_h[3 * (i + j) + 2] = Int32(p + x)
            bufs.imgmap_h[i + j] = Int32(rows + j)
            for c in range(H):
                bufs.img_h[(rows + j) * H + c] = fp[unsafe_offset=j * H + c]
        rows += n_tok
        p += n_pos
        i += run
        k += 1
    if k != len(embeds):
        return "embeds has " + String(len(embeds)) + " files but the prompt has " + String(k) + " image runs"
    var adapter_dir = getenv("BARO_SEMANTIC_ADAPTER_DIR", "")
    if adapter_dir != "" and rows > 0:
        var adapter = LatentMLPAdapter(adapter_dir, H, 256, H)
        var scratch_rank = List[Float32](unsafe_uninit_length=256)
        for r in range(rows):
            adapter.forward_ptr_in_place(bufs.img_h.unsafe_ptr(), scratch_rank, r * H)
        print("vision: applied semantic latent MLP adapter (rank 256) in-place on", rows, "image rows")
    for j in range(len(prompt), tmax):
        bufs.pos3_h[3 * j] = Int32(p)
        bufs.pos3_h[3 * j + 1] = Int32(p)
        bufs.pos3_h[3 * j + 2] = Int32(p)
        p += 1
    ctx.enqueue_copy(dst_buf=DeviceBuffer[DType.int32](ctx, bufs.pos3_d.unsafe_ptr(), 3 * tmax, owning=False), src_buf=bufs.pos3_h)
    ctx.enqueue_copy(dst_buf=DeviceBuffer[DType.int32](ctx, bufs.imgmap_d.unsafe_ptr(), tmax, owning=False), src_buf=bufs.imgmap_h)
    ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, bufs.img_d.unsafe_ptr(), rows * H, owning=False), src_buf=bufs.img_h.create_sub_buffer[f32](0, rows * H))
    ctx.synchronize()
    print("vision: images", len(embeds), " image tokens", rows, " text positions end at", p - (tmax - len(prompt)), " mrope on, megakernel off, prefix checkpoints off")
    return ""


@fieldwise_init
struct PairSeq(Copyable, Movable):
    var id: Int
    var prompt: List[Int]
    var stop_seqs: List[List[Int]]
    var ckpt_hints: List[Int]
    var force: List[Int]
    var predicted: List[Int]
    var cfg: WindowCfg
    var st: WindowState
    var view: WindowBufs
    var slot: Int
    var cached: Int
    var pf_rows: Int
    var restore_s: Float64
    var t0: Int
    var t_prefill_end: Int
    var tag: SeqTag  # phase + finish reason (wire.mojo, B-M7): one writer, SeqTag.end
    var pair_steps: Int


def parse_force_field(req: String, mut force: List[Int]):
    var fi2 = json_key(req, "force")
    if fi2 < 0:
        return
    force = List[Int]()
    var fb = req.as_bytes()
    if fi2 < len(fb) and fb[fi2] == 91:
        fi2 += 1
        while True:
            while fi2 < len(fb) and (fb[fi2] == 32 or fb[fi2] == 44):
                fi2 += 1
            if fi2 >= len(fb) or fb[fi2] == 93:
                break
            var fv = 0
            if not json_int(req, fi2, fv):
                break
            force.append(fv)


def drain_stdin(mut pending: List[String]) raises:
    # Every line already in the pipe, without blocking: request lines in arrival order plus
    # any cancel; the caller matches cancels against the two in-flight ids.
    while True:
        var pfd = List[UInt8](unsafe_uninit_length=8)
        var p = pfd.unsafe_ptr()
        p.unsafe_bitcast[Int32]().unsafe_offset(0)[] = Int32(0)
        p.unsafe_bitcast[Int16]().unsafe_offset(2)[] = Int16(1)
        p.unsafe_bitcast[Int16]().unsafe_offset(3)[] = Int16(0)
        var r = external_call["poll", Int32](p, UInt64(1), Int32(0))
        if r <= 0 or (Int(p.unsafe_bitcast[Int16]().unsafe_offset(3)[]) & 1) == 0:
            return
        var line_in = read_line(0)
        if not line_in:
            return
        pending.append(line_in.value())


def pair_admit(
    ctx: DeviceContext, mut bufs: WindowBufs, mut chain: Chain, line: String, slot: Int, tmax: Int,
    pf_on: Bool, pf_min: Int, proto_cfg: WindowCfg, force_env: List[Int], mut deferred: Bool, lazy: Bool = False,
    defer_full: Bool = False,
) raises -> Optional[PairSeq]:
    var req_id = 0
    var prompt = List[Int]()
    var req_n = 0
    var req_spec = False
    var req_has_spec = False
    var stop_seqs = List[List[Int]]()
    var ckpt_hints = List[Int]()
    var sample = default_sample_params()
    var s_save = String("")
    var s_load = String("")
    var s_prefix = False
    var sm = String("")
    var sv = String("")
    var sp = 0
    var sn = Int64(0)
    var perr = parse_request(line, req_id, prompt, req_n, req_spec, req_has_spec, stop_seqs, ckpt_hints, sample, s_save, s_load, s_prefix, sm, sv, sp, sn)
    if perr == "" and len(prompt) < 1:
        perr = "empty prompt"
    if perr == "" and req_n < 1:
        perr = "n must be >= 1"
    if perr == "" and len(prompt) + req_n > tmax:
        perr = "prompt+n exceeds TMAX " + String(tmax)
    if perr == "" and (s_save != "" or s_load != "" or json_key(line, "schema") >= 0 or json_key(line, "embeds") >= 0 or sample.hidden != 0 or sample.embed != 0 or sample.logits_topk > 0 or sample.top_logprobs > 0 or sample.presence_penalty != 0 or sample.frequency_penalty != 0):
        perr = "BARO_SEQS=2 serves prompt/n/stop/ckpt/sampling requests only"
    if perr != "":
        print(err_line(req_id, perr))
        return None
    var force = force_env.copy()
    parse_force_field(line, force)
    var t0 = perf_counter_ns()
    var view = sequence_bufs(ctx, bufs, slot, tmax, share=True)
    var ckpt_idx = chain.lookup(prompt, len(prompt))
    var cached = chain.pos_of(ckpt_idx)
    chain.touch(ckpt_idx)
    # lazy (BARO_SEQS > 2): pages for the prompt plus BARO_SEQS_RESERVE decode tokens now (a burst that cannot
    # all fit is refused here, HTTP 429, instead of failing mid-decode), the rest one page at a time.
    var reserve = min(req_n, atol(getenv("BARO_SEQS_RESERVE", "512")))
    var need = min(bufs.tpages, ceildiv(len(prompt) + (reserve if lazy else req_n) + KMAX + 1, KVPAGE))
    if defer_full and chain.free_pages() < need - (cached // KVPAGE if ckpt_idx >= 0 else 0):
        # Waiting costs nothing; evicting checkpoints (the shared prefix among them) to squeeze in would.
        deferred = True
        return None
    try:
        chain.map_pages(ctx, ckpt_idx, cached, need, bufs.kc_d, bufs.vc_d, bufs.kvtab_h, bufs.kvtab_d, bufs.tpages, slot)
    except e:
        chain.lives[slot] = List[Int32]()
        if defer_full:
            # Another request holds pages it frees at done: the caller keeps this line queued instead of refusing
            # a request that fits a moment later.
            deferred = True
            return None
        print(err_line(req_id, "kv pool full: " + String(e)))
        return None
    var t_restore = perf_counter_ns()
    if ckpt_idx >= 0:
        chain.restore(ctx, view.convstate_d, view.sstate_d, 0, ckpt_idx)
    else:
        ctx.enqueue_memset(view.convstate_d, 0)
        ctx.enqueue_memset(view.sstate_d, 0)
    var toks_h = ctx.enqueue_create_host_buffer[DType.int32](tmax)
    ctx.synchronize()
    for i in range(tmax):
        toks_h[i] = 0
    for i in range(len(prompt)):
        toks_h[i] = Int32(prompt[i])
    ctx.enqueue_copy(dst_buf=view.toks_d, src_buf=toks_h)
    ctx.synchronize()
    var restore_s = Float64(perf_counter_ns() - t_restore) / 1e9
    var pf_rows = 0
    var pf_tail = 0
    if pf_on and len(prompt) - 1 - cached >= pf_min:
        pf_rows = len(prompt) - 1
        pf_tail = (pf_rows - cached) % MROWS
        if pf_tail == 0:
            pf_tail = MROWS
    var cfg = proto_cfg.copy()
    cfg.req_id = req_id
    cfg.pf_rows = pf_rows
    cfg.pf_tail = pf_tail
    cfg.n_total = len(prompt) + req_n
    cfg.n_prompt = len(prompt)
    cfg.sample = sample.copy()
    var st = a3b_state()
    st.reset(t0)
    st.pos = cached
    st.pos_prev = cached
    print("prompt tokens:", len(prompt), " n:", req_n, " spec: False  temperature:", sample.temperature, " slot:", slot, " cached:", cached, " replay rows:", len(prompt) - 1 - cached)
    return PairSeq(id=req_id, prompt=prompt^, stop_seqs=stop_seqs^, ckpt_hints=ckpt_hints^, force=force^, predicted=List[Int](), cfg=cfg^, st=st^, view=view^, slot=slot, cached=cached, pf_rows=pf_rows, restore_s=restore_s, t0=t0, t_prefill_end=t0, tag=SeqTag(), pair_steps=0)


def pair_after_step(ctx: DeviceContext, mut q: PairSeq, mut chain: Chain, pos_before: Int, ckpt_cap: Int, toks_h: HostBuffer[DType.int32], tmax: Int, save_end: Bool = True) raises:
    var np = len(q.prompt)
    if len(q.force) > 0 and pos_before >= np - 1:
        if q.st.pos != pos_before + 1:
            raise Error("BARO_FORCE: step advanced by more than one position -- void arm")
        var fi = pos_before - (np - 1)
        if fi < len(q.force):
            var one = ctx.enqueue_create_host_buffer[DType.int32](1)
            ctx.enqueue_copy(dst_buf=one, src_buf=DeviceBuffer[DType.int32](ctx, q.view.toks_d.unsafe_ptr().unsafe_offset(q.st.pos), 1, owning=False))
            ctx.synchronize()
            q.predicted.append(Int(one[0]))
            one[0] = Int32(q.force[fi])
            ctx.enqueue_copy(dst_buf=DeviceBuffer[DType.int32](ctx, q.view.toks_d.unsafe_ptr().unsafe_offset(q.st.pos), 1, owning=False), src_buf=one)
            ctx.synchronize()
    if ckpt_cap > 0 and q.st.pos > q.cached and q.st.pos < np:
        if (q.st.pos == np - 1 and save_end) or q.st.pos % CKPT_PERIOD == 0:
            chain.save(ctx, q.view.convstate_d, q.view.sstate_d, q.st.ring, q.st.pos, q.prompt, False, False, q.slot)
        else:
            var hi = hint_index(q.ckpt_hints, q.st.pos)
            if hi >= 0:
                chain.save(ctx, q.view.convstate_d, q.view.sstate_d, q.st.ring, q.st.pos, q.prompt, hi == 0 and chain.npages == 0, True, q.slot)
    if q.tag.phase == SEQ_PREFILL and q.st.pos >= np:
        ctx.synchronize()
        q.t_prefill_end = perf_counter_ns()
        q.tag.decoding()
    if len(q.stop_seqs) > 0 and q.st.pos >= np:
        ctx.synchronize()
        ctx.enqueue_copy(dst_buf=toks_h, src_buf=DeviceBuffer[DType.int32](ctx, q.view.toks_d.unsafe_ptr(), tmax, owning=False))
        ctx.synchronize()
        var gen_len = q.st.pos + 1 - np
        for si in range(len(q.stop_seqs)):
            var L = len(q.stop_seqs[si])
            if L == 0 or L > gen_len:
                continue
            var ok = True
            for k in range(L):
                if Int(toks_h[q.st.pos - L + 1 + k]) != q.stop_seqs[si][k]:
                    ok = False
                    break
            if ok:
                q.tag.end(FIN_STOP)
                break


def tool_arm(mut st: WindowState, mut grt: Optional[GrammarRuntime], packdir: String, line: String) raises -> String:
    # Tool-name constraint (briefs/2026-11-07-ailab-tools-and-cast.md item 1). main.rs sends
    # tool_re (the text after `<tool_call>` through the name) and, for tool_choice "required",
    # tool_re0 (the name alone: the prompt already ends inside the opening). Returns an error
    # line text, "" when armed or when the request carries no tools.
    st.tool_re = parse_str_field(line, "tool_re")
    st.tool_buf = List[UInt8]()
    st.grammar_tool = False
    var re0 = parse_str_field(line, "tool_re0")
    if st.tool_re == "" and re0 == "":
        return ""
    if not grt:
        grt = Optional(GrammarRuntime(packdir))
    if grt.value().vocab_size != VOCAB:
        return "tools: grammar vocab_size " + String(grt.value().vocab_size) + " != engine VOCAB " + String(VOCAB)
    _ = compile_tool_matcher(grt.value(), st.tool_re if st.tool_re != "" else re0)  # a bad pattern fails here, not mid-decode
    if re0 != "":
        tool_install(st, grt.value(), re0)
    return ""


def tool_install(mut st: WindowState, rt: GrammarRuntime, pattern: String) raises:
    st.grammar = Optional(compile_tool_matcher(rt, pattern))
    st.grammar_tool = True
    st.grammar_pending_think = False
    st.grammar_stop = False
    if len(st.grammar_mask.words) * 64 < VOCAB:
        st.grammar_mask = Bitset(VOCAB)


def tool_after_step(ctx: DeviceContext, mut st: WindowState, b: WindowBufs, one_h: HostBuffer[DType.int32], grt: Optional[GrammarRuntime]) raises:
    # After every generated token of a request with tools: release a name matcher that
    # terminated (window.mojo set grammar_stop on it), else scan the new token for the
    # `<tool_call>` opening and arm the name matcher for the next draw.
    if not grt or (st.tool_re == "" and not st.grammar_tool):
        return
    if st.grammar_tool:
        if st.grammar_stop:
            st.grammar = None
            st.grammar_tool = False
            st.grammar_stop = False
            st.tool_buf = List[UInt8]()
        return
    if st.grammar:
        return
    ctx.enqueue_copy(dst_buf=one_h, src_buf=DeviceBuffer[DType.int32](ctx, b.toks_d.unsafe_ptr().unsafe_offset(st.pos), 1, owning=False))
    ctx.synchronize()
    var tok = Int(one_h[0])
    if tok >= 0 and tok < VOCAB and tool_open_observe(st.tool_buf, grt.value().vocab[].token_bytes[tok]):
        var pat = st.tool_re
        tool_install(st, grt.value(), pat)


def pair_finish(ctx: DeviceContext, q: PairSeq, mut chain: Chain, bufs: WindowBufs, path: Int) raises:
    ctx.synchronize()
    chain.commit()
    chain.lives[q.slot] = List[Int32]()
    var tpe = q.t_prefill_end if q.tag.phase != SEQ_PREFILL else perf_counter_ns()
    var dt = Float64(perf_counter_ns() - q.t0) / 1e9
    var prefill_s = Float64(tpe - q.t0) / 1e9
    var decode_s = dt - prefill_s
    var n_gen = q.st.pos + 1 - len(q.prompt)
    if len(q.force) > 0:
        var ps = String("")
        var agree = 0
        var checked = min(len(q.predicted), len(q.force))
        for i in range(len(q.predicted)):
            ps += String(q.predicted[i]) + " "
            if i < len(q.force) and q.predicted[i] == q.force[i]:
                agree += 1
        print("predicted:", ps)
        print("forced agreement:", agree, "/", checked)
    var flw = ctx.enqueue_create_host_buffer[DType.uint32](3)
    ctx.enqueue_copy(dst_buf=flw, src_buf=bufs.ctr_d)
    ctx.synchronize()
    print("mega fail word:", flw[2], "" if flw[2] == 0 else " NOT-RESIDENT: a grid barrier timed out, tokens after it are invalid")
    print("mega barrier gen:", flw[1], " pair steps:", q.pair_steps, " id", q.id)
    var finish = finish_word(q.tag.reason)
    var done_line = String("{\"id\":") + String(q.id) + ",\"done\":true,\"n\":" + String(n_gen)
    done_line += ",\"prefill_s\":" + String(prefill_s) + ",\"decode_s\":" + String(decode_s)
    done_line += ",\"tok_s\":" + String(Float64(n_gen - 1) / decode_s if n_gen > 1 else 0.0)
    done_line += ",\"cached\":" + String(q.cached) + ",\"prefill_rows\":" + String(len(q.prompt) - 1 - q.cached) + ",\"pf_rows\":" + String(q.pf_rows) + ",\"restore_s\":" + String(q.restore_s) + ",\"checkpoints\":" + String(chain.count_valid())
    done_line += ",\"load_s\":0.0,\"load_read_s\":0.0,\"load_parse_s\":0.0,\"load_pin_s\":0.0,\"load_host_s\":0.0"
    done_line += ",\"pair_steps\":" + String(q.pair_steps) + ",\"slot\":" + String(q.slot)
    done_line += ",\"finish\":\"" + finish + "\""
    var dw = DoneWords()
    dw.fail = Int(flw[2])
    dw.path = path
    dw.grammar = grammar_word(False, False, False)  # pair and seqs serve no response_format
    print(done_line + dw.json() + "}")


def serve_pair(
    ctx: DeviceContext, mut bufs: WindowBufs, mut chain: Chain, mut pending: List[String], tmax: Int,
    pf_on: Bool, pf_min: Int, proto_cfg: WindowCfg, force_env: List[Int], ckpt_cap: Int,
) raises:
    var trace = getenv("BARO_PAIR_TRACE", "0") == "1"
    # BARO_SEQS=2: two requests in flight over one engine. Prefill is exclusive (the other
    # sequence pauses); two decoding sequences step together in one R = 2 megakernel launch
    # (window.mojo step_pair); one decodes alone through step_window exactly as before.
    var active = List[PairSeq]()
    var toks_h = ctx.enqueue_create_host_buffer[DType.int32](tmax)
    var zbuf = ctx.enqueue_create_buffer[f32](2 * NH_V * SSTATE)
    var rbuf = ctx.enqueue_create_buffer[bf16](2 * NH_V * SSTATE)
    ctx.synchronize()
    var eof = False
    while not eof or len(active) > 0:
        if len(active) == 0 and len(pending) == 0 and not eof:
            var line_in = read_line(0)
            if not line_in:
                eof = True
            else:
                pending.append(line_in.value())
        elif not eof:
            drain_stdin(pending)
        var keep = List[String]()
        for li in range(len(pending)):
            var line = pending[li].copy()
            var cancel = is_cancel_line(line)
            var cid = cancel[1]
            if cancel[0]:
                for ai in range(len(active)):
                    if active[ai].id == cid:
                        active[ai].tag.end(FIN_CANCELLED)
                continue
            if json_key(line, "latent_fork") >= 0 or json_key(line, "latent_resume") >= 0 or json_key(line, "ipc_export") >= 0 or json_key(line, "ipc_release") >= 0 or json_key(line, "ipc_sha") >= 0 or json_key(line, "ipc_ingest") >= 0 or json_key(line, "a3b") >= 0 or json_key(line, "a3c") >= 0 or json_key(line, "a3c2") >= 0:
                print(err_line(0, "BARO_SEQS=2: control line not served in pair mode"))
                continue
            if len(active) < 2:
                var slot = 0
                if len(active) == 1 and active[0].slot == 0:
                    slot = 1
                ctx.enqueue_memset(bufs.ctr_d, 0)
                var no_defer = False
                var adm = pair_admit(ctx, bufs, chain, line, slot, tmax, pf_on, pf_min, proto_cfg, force_env, no_defer)
                if adm:
                    active.append(adm.take())
                    print("pair: admitted id", active[len(active) - 1].id, "slot", slot, "in flight", len(active))
            else:
                keep.append(line^)
        pending = keep^
        if len(active) == 0:
            continue
        var pf = -1
        for ai in range(len(active)):
            if active[ai].st.pos + 1 < active[ai].cfg.n_prompt and pf < 0:
                pf = ai
        if pf >= 0:
            var pos_before = active[pf].st.pos
            if len(active[pf].ckpt_hints) > 0 and active[pf].st.pos < active[pf].pf_rows:
                var step_cfg = active[pf].cfg.copy()
                step_cfg.pf_chunk = min(step_cfg.pf_chunk, next_ckpt_stop(active[pf].ckpt_hints, active[pf].st.pos, active[pf].pf_rows) - active[pf].st.pos)
                step_window(ctx, active[pf].view, step_cfg, active[pf].st)
            else:
                step_window(ctx, active[pf].view, active[pf].cfg, active[pf].st)
            if trace:
                ctx.synchronize()
                print("pair trace: pf step id", active[pf].id, " slot", active[pf].slot, " pos", pos_before, "->", active[pf].st.pos, " ring", active[pf].st.ring, " pf_rows", active[pf].pf_rows, " in flight", len(active))
            pair_after_step(ctx, active[pf], chain, pos_before, ckpt_cap, toks_h, tmax)
        elif len(active) == 2:
            # Row r of the kernel is slot r (row 1's state and KV table are slot 0's pointer plus
            # one slot stride), so the rows are ordered by slot, not by admission: after slot 0
            # finishes and is refilled while slot 1 decodes, active[0] is the slot-1 sequence.
            var a = 0 if active[0].slot == 0 else 1
            var b = 1 - a
            var pb0 = active[a].st.pos
            var pb1 = active[b].st.pos
            var toks1 = active[b].view.toks_d.copy()
            var cfg0 = active[a].cfg.copy()
            var cfg1 = active[b].cfg.copy()
            step_pair(ctx, active[a].view, toks1, cfg0, cfg1, active[a].st, pb1, active[b].st.ring, zbuf, rbuf)
            if trace and (active[a].pair_steps < 3 or active[b].pair_steps < 3):
                ctx.synchronize()
                print("pair trace: pair step ids", active[a].id, active[b].id, " pos", pb0, pb1, " rings", active[a].st.ring, active[b].st.ring, " steps", active[a].pair_steps, active[b].pair_steps)
            active[b].st.ring = (active[b].st.ring + 1) % SLOTS
            active[b].st.pos_prev = pb1
            active[b].st.pos = pb1 + 1
            active[a].pair_steps += 1
            active[b].pair_steps += 1
            pair_after_step(ctx, active[a], chain, pb0, ckpt_cap, toks_h, tmax)
            pair_after_step(ctx, active[b], chain, pb1, ckpt_cap, toks_h, tmax)
        else:
            var pos_before = active[0].st.pos
            step_window(ctx, active[0].view, active[0].cfg, active[0].st)
            pair_after_step(ctx, active[0], chain, pos_before, ckpt_cap, toks_h, tmax)
        var ai = len(active) - 1
        while ai >= 0:
            if active[ai].tag.ended() or active[ai].st.pos >= active[ai].cfg.n_total - 1:
                pair_finish(ctx, active[ai], chain, bufs, PATH_PAIR)
                _ = active.pop(ai)
            ai -= 1


def serve_seqs(
    ctx: DeviceContext, mut bufs: WindowBufs, mut chain: Chain, mut pending: List[String], tmax: Int,
    pf_on: Bool, pf_min: Int, proto_cfg: WindowCfg, force_env: List[Int], ckpt_cap: Int, nseq: Int, packdir: String,
) raises:
    # BARO_SEQS=N > 2 on a dense pack (docs/design/multi-session-serve.md): N requests in flight,
    # one slot each (own conv/SSM ring, own page list; prefix pages shared read-only through the
    # checkpoint chain). Each loop: admit into free slots, one prefill step for the oldest
    # prefilling request, then one decode step for every decoding request. A request's ids do
    # not depend on N: every step is that slot's own m = 1 step_window.
    var trace = getenv("BARO_PAIR_TRACE", "0") == "1"
    var rows_on = getenv("BARO_SEQS_ROWS", "1") == "1"
    print("seqs: rows step", rows_on, " group", MROWS)
    var timing = getenv("BARO_SEQS_TIME", "0") == "1"
    # BARO_CKPT_END=0: no prompt-end checkpoint (a 50 MB copy per request that only an identical prompt reuses;
    # every request then restores at the shared prefix). BARO_SEQS_PF=one: one prefill step per loop instead of
    # draining every prefilling request before the decode step (lockstep bursts want full decode batches).
    var save_end = getenv("BARO_CKPT_END", "1") == "1"
    var pf_all = getenv("BARO_SEQS_PF", "all") == "all"
    print("seqs: ckpt_end", save_end, " prefill", "all" if pf_all else "one")
    var hist_ms = List[Int](length=nseq + 1, fill=0)
    var hist_n = List[Int](length=nseq + 1, fill=0)
    var t_pf = 0
    var t_dec = 0
    var n_pf = 0
    var n_dec = 0
    var n_rows = 0
    var loops = 0
    var active = List[PairSeq]()
    var n_deferred = 0
    var toks_h = ctx.enqueue_create_host_buffer[DType.int32](tmax)
    var one_h = ctx.enqueue_create_host_buffer[DType.int32](1)
    var grt: Optional[GrammarRuntime] = None
    ctx.synchronize()
    var eof = False
    while not eof or len(active) > 0:
        if len(active) == 0 and len(pending) == 0 and not eof:
            var line_in = read_line(0)
            if not line_in:
                eof = True
            else:
                pending.append(line_in.value())
        elif not eof:
            drain_stdin(pending)
        var keep = List[String]()
        var blocked = False
        for li in range(len(pending)):
            var line = pending[li].copy()
            var cancel = is_cancel_line(line)
            var cid = cancel[1]
            if cancel[0]:
                for ai in range(len(active)):
                    if active[ai].id == cid:
                        active[ai].tag.end(FIN_CANCELLED)
                continue
            if json_key(line, "latent_fork") >= 0 or json_key(line, "latent_resume") >= 0 or json_key(line, "ipc_export") >= 0 or json_key(line, "ipc_release") >= 0 or json_key(line, "ipc_sha") >= 0 or json_key(line, "ipc_ingest") >= 0 or json_key(line, "a3b") >= 0 or json_key(line, "a3c") >= 0 or json_key(line, "a3c2") >= 0:
                print(err_line(0, "BARO_SEQS>2: control line not served in multi-sequence mode"))
                continue
            if len(active) < nseq and not blocked:
                var used = List[Bool](length=nseq, fill=False)
                for ai in range(len(active)):
                    used[active[ai].slot] = True
                var slot = 0
                while used[slot]:
                    slot += 1
                var deferred = False
                var adm = pair_admit(ctx, bufs, chain, line, slot, tmax, pf_on, pf_min, proto_cfg, force_env, deferred, lazy=True, defer_full=len(active) > 0)
                if adm:
                    var q = adm.take()
                    var terr = String("")
                    try:
                        terr = tool_arm(q.st, grt, packdir, line)
                    except e:
                        terr = "tools: " + String(e)
                    if terr != "":
                        print(err_line(q.id, terr))
                        ctx.synchronize()
                        chain.commit()
                        chain.lives[q.slot] = List[Int32]()
                        continue
                    active.append(q^)
                    print("seqs: admitted id", active[len(active) - 1].id, "slot", slot, "in flight", len(active))
                elif deferred:
                    n_deferred += 1
                    blocked = True
                    keep.append(line^)
            else:
                keep.append(line^)
        pending = keep^
        if len(active) == 0:
            continue
        var pf = -1
        for ai in range(len(active)):
            if active[ai].st.pos + 1 < active[ai].cfg.n_prompt and pf < 0:
                pf = ai
        if timing:
            ctx.synchronize()
        var t_a = perf_counter_ns()
        while pf >= 0:
            n_pf += 1
            var pos_before = active[pf].st.pos
            if len(active[pf].ckpt_hints) > 0 and active[pf].st.pos < active[pf].pf_rows:
                var step_cfg = active[pf].cfg.copy()
                step_cfg.pf_chunk = min(step_cfg.pf_chunk, next_ckpt_stop(active[pf].ckpt_hints, active[pf].st.pos, active[pf].pf_rows) - active[pf].st.pos)
                step_window(ctx, active[pf].view, step_cfg, active[pf].st)
            else:
                step_window(ctx, active[pf].view, active[pf].cfg, active[pf].st)
            if trace:
                ctx.synchronize()
                print("seqs trace: pf step id", active[pf].id, " slot", active[pf].slot, " pos", pos_before, "->", active[pf].st.pos, " in flight", len(active))
            pair_after_step(ctx, active[pf], chain, pos_before, ckpt_cap, toks_h, tmax, save_end)
            var was = pf
            pf = -1
            if pf_all:
                for ai in range(len(active)):
                    if active[ai].st.pos + 1 < active[ai].cfg.n_prompt and pf < 0:
                        pf = ai
            if pf < 0:
                pf = was
                break
        if timing:
            ctx.synchronize()
        var t_b = perf_counter_ns()
        if pf >= 0:
            t_pf += Int(t_b - t_a)
        var dec = List[Int]()
        for ai in range(len(active)):
            if ai == pf or active[ai].st.pos + 1 < active[ai].cfg.n_prompt:
                continue
            if active[ai].tag.ended() or active[ai].st.pos >= active[ai].cfg.n_total - 1:
                continue
            try:
                chain.grow(ctx, active[ai].slot, min(bufs.tpages, ceildiv(active[ai].st.pos + 2, KVPAGE)), bufs.kvtab_h, bufs.kvtab_d, bufs.tpages)
            except e:
                print(err_line(active[ai].id, "kv pool exhausted at pos " + String(active[ai].st.pos) + ": " + String(e)))
                active[ai].tag.end(FIN_ERROR)
                continue
            dec.append(ai)
        # A slot drawing a tool name (grammar set) takes its own m = 1 step_window: the masked
        # draw lives there, the rows step has none. A name is a few tokens per call.
        var dec_rows = List[Int]()
        for di in range(len(dec)):
            var ai = dec[di]
            if rows_on and not active[ai].st.grammar:
                dec_rows.append(ai)
                continue
            var pos_before = active[ai].st.pos
            step_window(ctx, active[ai].view, active[ai].cfg, active[ai].st)
            active[ai].pair_steps += 1
            pair_after_step(ctx, active[ai], chain, pos_before, ckpt_cap, toks_h, tmax, save_end)
            tool_after_step(ctx, active[ai].st, active[ai].view, one_h, grt)
        var n_step = len(dec)
        dec = dec_rows^
        var g0 = 0
        while rows_on and g0 < len(dec):
            var g1 = min(len(dec), g0 + MROWS)
            var views = List[WindowBufs]()
            var poss = List[Int]()
            var rings = List[Int]()
            var ids = List[Int]()
            var samples = List[SampleParams]()
            for di in range(g0, g1):
                samples.append(active[dec[di]].cfg.sample.copy())
                views.append(active[dec[di]].view.copy())
                poss.append(active[dec[di]].st.pos)
                rings.append(active[dec[di]].st.ring)
                ids.append(active[dec[di]].id)
            step_rows(ctx, bufs, views, poss, rings, ids, proto_cfg.att_split, samples)
            for di in range(g0, g1):
                var ai = dec[di]
                var pos_before = active[ai].st.pos
                active[ai].st.ring = (active[ai].st.ring + 1) % SLOTS
                active[ai].st.pos_prev = pos_before
                active[ai].st.pos = pos_before + 1
                active[ai].pair_steps += 1
                pair_after_step(ctx, active[ai], chain, pos_before, ckpt_cap, toks_h, tmax, save_end)
                tool_after_step(ctx, active[ai].st, active[ai].view, one_h, grt)
            g0 = g1
        if timing and n_step > 0:
            ctx.synchronize()
            t_dec += Int(perf_counter_ns() - t_b)
            n_dec += 1
            n_rows += n_step
            hist_ms[n_step] += Int(perf_counter_ns() - t_b)
            hist_n[n_step] += 1
        loops += 1
        if timing and loops % 64 == 0:
            var hs = String("")
            for r in range(1, nseq + 1):
                if hist_n[r] > 0:
                    hs += " r" + String(r) + ":" + String(hist_n[r]) + "x" + String(Float64(Int(Float64(hist_ms[r]) / 1e4 / Float64(hist_n[r]))) / 100.0) + "ms"
            print("seqs step ms by rows:", hs)
            print("seqs: admissions deferred for pages", n_deferred)
            print("seqs time: loops", loops, " prefill steps", n_pf, " ms", Float64(t_pf) / 1e6, " decode steps", n_dec, " ms", Float64(t_dec) / 1e6, " rows", n_rows, " ms/step", Float64(t_dec) / 1e6 / Float64(max(1, n_dec)), " rows/step", Float64(n_rows) / Float64(max(1, n_dec)))
        var ai = len(active) - 1
        while ai >= 0:
            if active[ai].tag.ended() or active[ai].st.pos >= active[ai].cfg.n_total - 1:
                pair_finish(ctx, active[ai], chain, bufs, PATH_SEQS)
                _ = active.pop(ai)
            ai -= 1


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var vram_pool_pct = getenv("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT", "")
    if vram_pool_pct == "":
        vram_pool_pct = "2"
        setenv("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT", vram_pool_pct)
        print("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT: unset (defaulting to 2%)")
    else:
        print("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT:", vram_pool_pct)
    var gtt_base = gtt_used()
    print("GTT: card", find_drm_dev(), "gtt_used", gtt_base)
    print_vram_log("before_device_context")
    var ctx = DeviceContext()
    print_vram_log("after_device_context")
    var mega = getenv("BARO_MEGA", "1" if MEGA_ALLOWED and MEGA_FITS else "0") == "1"
    if mega and MEGA_ALLOWED and not MEGA_FITS:
        raise Error("BARO_MEGA=1: the dense persistent kernel needs ATT <= INNER")
    if mega and not MEGA_ALLOWED and getenv("BARO_TIER", "") != "":
        raise Error("BARO_MEGA=1 (the MoE persistent token kernel) needs the routed experts in VRAM; unset BARO_TIER")
    if mega and not MEGA_ALLOWED and getenv("BARO_EXPERTS", "") != "":
        raise Error("BARO_MEGA=1 (the MoE persistent token kernel) does not write the expert trace; unset BARO_EXPERTS")
    var packdir = getenv("BARO_PACK", ".work/engine-pack-q4")
    var pack = load_pack(ctx, packdir)
    print_vram_log("after_pack_load")
    gtt_base = gtt_check("after_pack_load", gtt_base)
    if mega and IS_MOE and pack.down_q6 != (1 << 34) | (1 << 38) | (1 << 39):
        raise Error("BARO_MEGA=1: the MoE persistent kernel hard-codes q6_k down layers 34/38/39 (kernels/mega_moe.mojo); this pack's differ")
    var wbuf = pack.wbuf
    var off = pack.off.copy()
    var q4_off = pack.q4_off
    var have_q4_draft = pack.have_q4_draft
    var pack_q4 = pack.pack_q4
    var e = pack.e
    var draft_q4 = getenv("BARO_DRAFT_Q4", "0") == "1" and have_q4_draft
    print("BARO_DRAFT_Q4:", draft_q4)
    var fr_k = pack.fr_k if getenv("BARO_FR", "0") == "1" else 0
    var fr_off = pack.fr_off
    var fr_ids_off = pack.fr_ids_off
    print("BARO_FR:", fr_k > 0, "k", fr_k)
    var dot3 = getenv("BARO_DOT", "0") == "1" and not pack_q4
    print("BARO_DOT:", dot3)
    print("pack q4 trunk:", pack_q4)
    # engine-pack --kq keeps the file's Q4_K / Q6_K rows: the launch path (window.gemm_kq) and the m=1 megakernel read them.
    var pack_kq = pack.pack_kq and not IS_MOE
    print("pack kq trunk:", pack_kq)
    if pack_kq and pack.kq_mixed and mega:
        # window.mojo picks amar_mega_token[KQ=True] from off[2]'s kind; it would decode a q8 trunk weight as Q4_K.
        print("kq pack carries q8 trunk weights: BARO_MEGA forced off (the kq megakernel reads Q4_K / Q6_K only)")
        mega = False
    if pack_kq:
        if getenv("BARO_SPEC", "1") == "1":
            raise Error("kq pack: the MTP draft reads q4/q8 only; set BARO_SPEC=0")
        if getenv("BARO_MEGA_WIN", "0") == "1":
            raise Error("kq pack: the window megakernel reads q4/q8 only; unset BARO_MEGA_WIN")
    var cus = cu_count()
    var mega_grid = MEGA_G if MEGA_ALLOWED else MOE_G
    print("compute units:", cus, " megakernel grid:", mega_grid)
    if mega and 0 < cus < mega_grid:
        if getenv("BARO_MEGA", "") == "1":
            print("WARNING BARO_MEGA=1 with", cus, "CUs <", mega_grid, "blocks: the grid barrier may not be co-resident; read the fail word")
        else:
            print("BARO_MEGA forced off: card has", cus, "CUs, megakernel grid needs", mega_grid)
            mega = False
    print("BARO_MEGA:", mega)
    print("BARO_TURBO:", TURBO)
    var kvtab_mode = getenv("BARO_KVTAB", "identity")
    var mega_win = getenv("BARO_MEGA_WIN", "0") == "1"
    print("BARO_MEGA_WIN:", mega_win)
    var pf5 = getenv("BARO_PROFILE", "0") == "5"
    var dump_path = getenv("BARO_DUMP", "")
    var dump = dump_path != ""
    var dump4 = getenv("BARO_DUMP4", "0") == "1"
    var dump_layer = atol(getenv("BARO_DUMP_LAYER", "0"))
    # Item 4 verification (coordinator review, 2026-09-16): per-step raw-row
    # + history dump for an independent host comparison, read once here (the
    # harness), never in window.mojo. Off by default.
    var dump_pen_dir = getenv("BARO_DUMP_LOGITS_DIR", "")

    var serve = getenv("BARO_SERVE", "0") == "1"
    print("BARO_SERVE:", serve)
    var kcfg = 2
    try:
        with open(packdir + "/spec-k.txt", "r") as f:
            var kd = f.read_bytes()
            var kv_ = 0
            var kh = False
            for i in range(len(kd)):
                var b = Int(kd[i])
                if b >= 48 and b <= 57:
                    kv_ = kv_ * 10 + (b - 48)
                    kh = True
                elif kh:
                    break
            if kh:
                kcfg = kv_
    except:
        kcfg = 2
    var kenv = getenv("BARO_SPEC_K", "")
    if kenv != "":
        kcfg = atol(kenv)
    if kcfg < 1:
        kcfg = 1
    var ngram_min = atol(getenv("BARO_NGRAM_MIN", "1"))
    print("spec k:", kcfg, " ngram min:", ngram_min)
    if getenv("BARO_SPEC", "1") == "1" and kcfg + 2 > SLOTS:
        raise Error("BARO_SLOTS=" + String(SLOTS) + " too small: need kcfg+2=" + String(kcfg + 2))
    # Prefill plan (bench/prefill-protocol.md): prompt rows 0 .. L-2 go through
    # prefill_forward in chunks of pf_chunk; the last prompt token still runs
    # the megakernel. The MTP draft's "process" window is kept identical to
    # the decode window path: pos_prev is set so it covers the last
    # (L-1) mod 8 rows (8 when 0), whose post-final-norm hidden rows the last
    # chunk writes into hn_d. pf_rows/pf_tail are per request (below).
    var tmax = atol(getenv("BARO_TMAX", String(TMAX)))
    if tmax < CP + PF_MIN:
        tmax = CP + PF_MIN
    var att_split = atol(getenv("BARO_ATT_SPLIT_T", String(TMAX)))
    if getenv("BARO_ATT_SPLIT", "0") == "1":
        att_split = 0
    print("att split:", att_split)
    var pf_chunk = min(atol(getenv("BARO_PREFILL_C", String(CP))), CP)
    if pf_chunk < PF_MIN:
        pf_chunk = PF_MIN
    # Batched prefill is the default on every profile since 2026-09-24
    # (the maintainer's call after the MoE gates: identity 23/23 in both modes, needle
    # at 75k; bench/moe-prefill-protocol.md). BARO_PREFILL=0 replays the
    # prompt rows through the m=1 path. A tier without the pinned store
    # cannot stage a layer per chunk, and an expert trace wants every prompt
    # row's ids, so both fall back to replay.
    var cap_dir = getenv("BARO_CAPTURE", "")  # docs/format-capture.md: per-token ffn in/out + residual of BARO_CAPTURE_LAYERS to f16 files
    var cap_on = cap_dir != ""
    comptime assert PF_CAP_ROWS >= CP, "capture staging must hold a whole prefill chunk (PF_CAP_ROWS >= CP)"
    if cap_on and pf_chunk > PF_CAP_ROWS:
        raise Error("BARO_CAPTURE: prefill chunk " + String(pf_chunk) + " is over PF_CAP_ROWS " + String(PF_CAP_ROWS))
    var cap_fields = getenv("BARO_CAPTURE_FIELDS", "xyr")
    comptime if IS_MOE:
        if cap_on:
            raise Error("BARO_CAPTURE: dense qwen35 only (MoE router capture is item A5)")
    if cap_on:
        for ci in range(cap_fields.byte_length()):
            if cap_fields.as_bytes()[ci] != UInt8(ord("x")) and cap_fields.as_bytes()[ci] != UInt8(ord("y")) and cap_fields.as_bytes()[ci] != UInt8(ord("r")):
                raise Error("BARO_CAPTURE_FIELDS: letters x y r only")
    var steer_serve = getenv("BARO_STEER_SERVE", "0") == "1"
    var imp_loaded = getenv("BARO_IMPLANTS", "") != ""
    var pf_on = getenv("BARO_PREFILL", "1") == "1" and getenv("BARO_STEER", "") == "" and not steer_serve and not imp_loaded  # steering has no batched-prefill hook here
    if IS_MOE and pf_on and getenv("BARO_TIER", "") != "" and getenv("BARO_TIER_PINNED", "1") != "1":
        pf_on = False
    if IS_MOE and pf_on and getenv("BARO_EXPERTS", "") != "":
        pf_on = False
    # Tier mode stages all 256 experts of every layer from host per batched
    # chunk (ExpertTier.stage_layer, ~18 GB, ~1.2 s) whatever its row count,
    # while one-token replay reads the VRAM expert cache at ~12.7 ms a row
    # (regescore A/B run 3: 58 rows cost 1.29 s batched). Short replays, such
    # as a task after a restored persona checkpoint, go one token at a time.
    var pf_min = PF_MIN
    var replay_m = 1 if cap_on else atol(getenv("BARO_REPLAY_M", "4"))  # capture records one position per step
    if replay_m < 1 or replay_m > MROWS:
        raise Error("BARO_REPLAY_M must be 1.." + String(MROWS))
    print("BARO_REPLAY_M:", replay_m, " (replay window rows cap, MROWS", MROWS, ")")
    if IS_MOE and getenv("BARO_TIER", "") != "":
        pf_min = max(PF_MIN, atol(getenv("BARO_PF_MIN_TIER", "96")))
    print("BARO_PREFILL:", pf_on, " chunk", pf_chunk, " min rows", pf_min)
    print("BARO_PF_GEMM:", "wmma" if pf_gemm_wmma() else "exact")
    # Default ON since 2026-09-12: draft decode measured 1.1042x on the frozen
    # 20-prompt median. Every BARO_FORCE caller pins BARO_SPEC=0 explicitly
    # (force-ab.sh, force-ab-serve.sh, dense-run.sh, ornith-run.sh,
    # llama-handoff.sh) and the engine raises if the two are combined, so an
    # unpinned identity gate fails loudly rather than measuring the wrong arm.
    # Still gated on MEGA_ALLOWED: the MoE profile has no draft head.
    var spec_env = ((MEGA_ALLOWED and getenv("BARO_SPEC", "1") == "1") or (IS_MOE and getenv("BARO_NGRAM", "0") == "1")) and getenv("BARO_STEER", "") == "" and getenv("BARO_STEER_SERVE", "0") != "1" and not cap_on and not imp_loaded
    print("BARO_SPEC:", spec_env)
    if not MEGA_ALLOWED and getenv("BARO_SPEC", "") == "1":
        print("BARO_SPEC=1 ignored: this profile has no draft head")
    var spec_dbg = getenv("BARO_SPEC_DBG", "0") == "1"
    # B4 stage 2 (bench/moe-locality-protocol.md): BARO_EXPERTS=<path> writes
    # the router's top-8 per layer per decoded token after each request. A
    # traced run is an instrumentation run: it adds a device-to-device copy per
    # layer, so its tok/s is not a receipt for anything.
    var expert_trace_path = getenv("BARO_EXPERTS", "")
    var expert_trace = expert_trace_path != ""
    print("BARO_EXPERTS:", expert_trace_path if expert_trace else "off")
    # Teacher-forced agreement (identity gate, CLAUDE.md: never greedy equality
    # past ~256 ids). Off by default (empty path -> empty list, decode
    # unchanged): the model's own no-spec argmax is recorded as "predicted"
    # and the reference id is force-fed into the next step's context
    # regardless of what the model chose, same file format and semantics as
    # serve/spark.mojo:260-261.
    var force_env = List[Int]()
    var force_path = getenv("BARO_FORCE", "")
    if force_path != "":
        with open(force_path, "r") as ff:
            var fdata = ff.read_bytes()
            var fval = 0
            var fhave = False
            for i in range(len(fdata)):
                var fb = Int(fdata[i])
                if fb >= 48 and fb <= 57:
                    fval = fval * 10 + (fb - 48)
                    fhave = True
                else:
                    if fhave:
                        force_env.append(fval)
                    fval = 0
                    fhave = False
            if fhave:
                force_env.append(fval)
        print("BARO_FORCE:", force_path, " (", len(force_env), "ids )")
    var bufs = alloc_bufs(ctx, pack, tmax)
    if cap_on:
        capture_meta(cap_dir, getenv("BARO_CAPTURE_MODEL", pack.packdir), cap_fields, bufs)
    var steer_boot = bufs.steer_on  # BARO_STEER from the environment: every request is steered unless it says otherwise
    var slens_base = List[List[Float32]]()  # steer lens state (docs/format.md steer.bin): base log-softmax per context, running sum, KL
    var slens_acc = List[Float32]()
    var slens_cnt = 0
    var slens_dnll: Float64 = 0
    var slens_dld: Float64 = 0
    var slens_nll: Float64 = 0
    var slens_kl: Float64 = 0
    print_vram_log("after_alloc_bufs")
    gtt_base = gtt_check("after_alloc_bufs", gtt_base)
    print("BARO_MOE_DOT:", bufs.moe_dot)
    var kvtab = PageTable(bufs.tpages)
    if kvtab_mode == "reverse":
        kvtab.reverse()
    elif kvtab_mode != "identity":
        raise Error("BARO_KVTAB must be identity or reverse")
    kvtab.upload(ctx, bufs.kvtab_h, bufs.kvtab_d)
    print("BARO_KVTAB:", kvtab_mode, " kv pages:", bufs.tpages)
    comptime if IS_MOE and KVT != DType.float32 and not KVQ8:
        raise Error("BARO_KVQ=" + KVQ + " is supported only as f32 or int8 for MoE")
    comptime KVB = 1 if KVT == DType.int8 else (2 if KVT == DType.bfloat16 else 4)
    print("BARO_KVQ:", KVQ, " kv dtype:", String(KVT), " kv pool bytes:", 2 * bufs.kvpool * KVB, " kv bytes/token:", 2 * bufs.kvpool * KVB // (bufs.tpages * KVPAGE), " seq_cap:", SEQ_CAP)
    # The dense path consumes Pack.off's historical order.  The MoE pack is
    # lexical by tensor name, so build a per-block semantic order by name;
    # the offsets remain byte offsets into the same Pack.wbuf blob.
    comptime if not MEGA_ALLOWED:
        var moe_tensors = parse_moe_index(packdir + "/index.txt")
        var moe_logical = List[Int]()
        moe_logical.append(resolve_plain(moe_tensors, "token_embd.weight").offset)
        for layer in range(N_LAYERS):
            var pfx = "blk." + String(layer) + "."
            if is_attn(layer):
                for name in [
                    "attn_norm.weight", "attn_q.weight", "attn_k.weight",
                    "attn_v.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                    "attn_output.weight",
                ]:
                    moe_logical.append(resolve_plain(moe_tensors, pfx + name).offset)
                moe_logical.append(resolve_plain(moe_tensors, pfx + "post_attention_norm.weight").offset)
                for name in [
                    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                    "ffn_gate_inp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                    "ffn_down_shexp.weight", "ffn_gate_inp_shexp.weight",
                ]:
                    moe_logical.append(resolve_plain(moe_tensors, pfx + name).offset)
            else:
                for name in [
                    "attn_norm.weight", "attn_qkv.weight", "attn_gate.weight",
                    "ssm_alpha.weight", "ssm_beta.weight", "ssm_conv1d.weight",
                    "ssm_a", "ssm_dt.bias", "ssm_norm.weight", "ssm_out.weight",
                ]:
                    moe_logical.append(resolve_plain(moe_tensors, pfx + name).offset)
                moe_logical.append(resolve_plain(moe_tensors, pfx + "post_attention_norm.weight").offset)
                for name in [
                    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                    "ffn_gate_inp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                    "ffn_down_shexp.weight", "ffn_gate_inp_shexp.weight",
                ]:
                    moe_logical.append(resolve_plain(moe_tensors, pfx + name).offset)
        moe_logical.append(resolve_plain(moe_tensors, "output_norm.weight").offset)
        moe_logical.append(resolve_plain(moe_tensors, "output.weight").offset)
        off = moe_logical.copy()
        bufs.off = moe_logical.copy()
        if len(moe_logical) > OFF_CAP:
            raise Error("MoE offset table exceeds OFF_CAP")
        var moff_h = ctx.enqueue_create_host_buffer[DType.int64](OFF_CAP)
        ctx.synchronize()
        for i in range(OFF_CAP):
            moff_h[i] = Int64(moe_logical[i]) if i < len(moe_logical) else 0
        ctx.enqueue_copy(dst_buf=bufs.off_d, src_buf=moff_h)
        ctx.synchronize()
    var toks_d = bufs.toks_d

    var wst = WindowState(pos=0, pos_prev=0, ring=0, n_drafted=0, n_accepted=0, n_spec_windows=0, n_dumped=0, tp=0, tq=0, pf_att=0, pf_ssm=0, pf_ffn=0, pf_head=0, pf_proc=0, pf_draft=0, fc=[0, 0, 0, 0, 0, 0], pc=[0, 0, 0, 0, 0, 0, 0, 0], ac=[0, 0, 0, 0, 0, 0], p3=[0, 0, 0, 0], pfx=[0, 0, 0, 0], grammar=None, grammar_mask=Bitset(1), grammar_pending_think=False, grammar_think_buf=List[UInt8](), grammar_stop=False, grammar_masked_draws=0, grammar_accepted=0, tool_re="", tool_buf=List[UInt8](), grammar_tool=False)
    var ckpt_cap = atol(getenv("BARO_CKPT", "8")) if serve else 0
    if ckpt_cap < 0:
        ckpt_cap = 0
    var state_load = getenv("BARO_STATE_LOAD", "")
    var state_save = getenv("BARO_STATE_SAVE", "")
    if ckpt_cap == 0 and (state_load != "" or state_save != ""):
        ckpt_cap = 1
    var chain = Chain(ctx, ckpt_cap, packdir)
    if atol(getenv("BARO_KV_SHARE", "0")) > 0:
        if SEQ_CAP > 2 and atol(getenv("BARO_SEQS", "1")) <= 2:
            raise Error("BARO_KV_SHARE pages slots 0-1 over the whole pool: build with -D BARO_SEQ_CAP=1 or 2 (or serve BARO_SEQS>2)")
        chain.npages = kv_pool_seqs() * bufs.tpages
        chain.share_reset(ctx, bufs.kvtab_h, bufs.kvtab_d, bufs.tpages)
    print("BARO_KV_SHARE: pages", chain.npages)
    var load_stage = LoadStage(ctx, tmax)
    print("state load stage: pinned", Float64(load_stage.buf_cap + 2 * load_stage.kvn_cap * (1 if KVQ8 else 4)) / 1e9, "GB (buf + 2 x KV staging) for TMAX", tmax)
    print("checkpoints: cap", ckpt_cap, ", bytes", Float64(CKPT_BYTES) / 1e6, "MB each, period", CKPT_PERIOD)
    # L1 (bench/chat-protocol.md): hand prefix checkpoints to an out-of-process
    # latentos-agent as sealed memfds. Unset, nothing here runs. Set and
    # unreachable, this RAISES: a handoff that silently does not happen is the
    # failure this sidecar exists to make visible.
    var latent_sock = getenv("BARO_LATENT_SOCK", "")
    var latent_fd = Int32(-1)
    var latent_gen = 0
    if latent_sock != "":
        latent_fd = connect_unix_socket(latent_sock)
        if latent_fd < 0:
            raise Error("BARO_LATENT_SOCK set but cannot connect: " + latent_sock)
        print("latent: exporting checkpoints to", latent_sock)
    if state_load != "":
        var t_ld = perf_counter_ns()
        var split0 = List[Float64]()
        var lpos = load_state(ctx, chain, bufs.kc_d, bufs.vc_d, bufs.kvtab_h, bufs.kvtab_d, bufs.tpages, state_load, tmax, split0, load_stage)
        print("state loaded:", state_load, " pos", lpos, " in", Float64(perf_counter_ns() - t_ld) / 1e9, "s", " read", split0[0], " parse", split0[1], " pin", split0[2], " host", split0[3], " h2d", split0[4])
    var req_id = 0
    # Requests read off fd 0 by the cancel probe mid-generation, in arrival
    # order; the request loop drains these before touching fd 0 again.
    var pending = List[String]()
    # JSON-enforcement item 1 (briefs/2026-09-16-json-enforcement-lane.md):
    # the grammar vocab/trie, built once on the first response_format
    # request (not at startup -- every existing run that never sets it pays
    # nothing, not even the trie build).
    var grt: Optional[GrammarRuntime] = None
    var one_h = ctx.enqueue_create_host_buffer[DType.int32](1)
    if serve:
        print("{\"ready\":true,\"tmax\":" + String(tmax) + ",\"mrows\":" + String(MROWS) + ",\"kmax\":" + String(KMAX) + ",\"spec_k\":" + String(0 if pack_kq else kcfg) + ",\"kv\":\"" + String(KVT) + "\",\"pack\":\"" + packdir + "\"" + ready_words(tier_word(bufs.tier.active, bufs.tier.cpu.on, bufs.tier.zc, bufs.tier.pinned, bufs.tier.hot_active, bufs.tier.store_mmap), max(1, atol(getenv("BARO_SEQS", "1")))) + "}")

    var seqs_mode = atol(getenv("BARO_SEQS", "1"))
    print("BARO_SEQS:", seqs_mode)
    if serve and seqs_mode == 2:
        if SEQ_CAP != 2 or chain.npages == 0 or not IS_MOE or not mega:
            raise Error("BARO_SEQS=2 needs -D BARO_SEQ_CAP=2, BARO_KV_SHARE>0, the MoE profile and BARO_MEGA=1")
        var proto_cfg = WindowCfg(pack_q4=pack_q4, draft_q4=draft_q4, q4_off=q4_off, e=e, kcfg=kcfg, spec=False, spec_dbg=spec_dbg, expert_trace=False, serve=True, req_id=0, prof=False, pf2=False, pf3=False, pf4=False, dump=False, dump4=False, dump_layer=0, mega=mega, att_split=att_split, mega_win=False, dot3=dot3, pf_chunk=pf_chunk, pf_rows=0, pf_tail=0, n_total=0, fr_k=fr_k, fr_off=fr_off, fr_ids_off=fr_ids_off, replay_m=replay_m, n_prompt=0, sample=default_sample_params(), dump_pen=False, mrope=False, inject=False)
        serve_pair(ctx, bufs, chain, pending, tmax, pf_on, pf_min, proto_cfg, force_env, ckpt_cap)
        return
    if serve and seqs_mode > 2:
        if seqs_mode > SEQ_CAP or chain.npages == 0 or IS_MOE:
            raise Error("BARO_SEQS=N>2 needs a dense pack, -D BARO_SEQ_CAP>=N and BARO_KV_SHARE>0 (pool: BARO_KV_POOL)")
        var proto_cfg = WindowCfg(pack_q4=pack_q4, draft_q4=draft_q4, q4_off=q4_off, e=e, kcfg=kcfg, spec=False, spec_dbg=spec_dbg, expert_trace=False, serve=True, req_id=0, prof=False, pf2=False, pf3=False, pf4=False, dump=False, dump4=False, dump_layer=0, mega=False, att_split=att_split, mega_win=False, dot3=dot3, pf_chunk=pf_chunk, pf_rows=0, pf_tail=0, n_total=0, fr_k=fr_k, fr_off=fr_off, fr_ids_off=fr_ids_off, replay_m=replay_m, n_prompt=0, sample=default_sample_params(), dump_pen=False, mrope=False, inject=False)
        print("seqs: N", seqs_mode, " slots", SEQ_CAP, " kv pages", chain.npages, " mega False spec False")
        serve_seqs(ctx, bufs, chain, pending, tmax, pf_on, pf_min, proto_cfg, force_env, ckpt_cap, seqs_mode, packdir)
        return

    # --- request loop: one prompt from the file (BARO_SERVE=0, then exit) or
    # JSON lines from stdin until EOF (BARO_SERVE=1, serve/PROTOCOL.md) ------
    var latent_fd_table = Dict[String, Int32]()
    var latent_header_table = Dict[String, proto.LatentHeader]()
    var latent_tokens_table = Dict[String, Int]()
    # T3b (AMDHQ 06-experiments `## T3b`): export stages, each sized for TMAX and reused (an
    # IPC-exported allocation is never reclaimed by hipFree on ROCm); ipc_busy holds the export id
    # a stage serves until its release, 0 when free. ipc_opens counts handles this engine opened
    # (a refused header must leave it unchanged).
    var ipc_ptrs = List[Int]()
    var ipc_handles = List[List[UInt8]]()
    var ipc_busy = List[Int]()
    var ipc_next = 0
    var ipc_opens = 0
    while True:
        gtt_base = gtt_check("request", gtt_base)
        var prompt = List[Int]()
        var gen_n = GEN_N
        var spec = spec_env
        var ckpt_idx = -1
        var cached = 0
        var slens_c = -1
        var slens_k = 0
        var slens_flags = 3
        var cap_req = cap_dir  # capture target of this request: BARO_CAPTURE, or BARO_CAPTURE/NAME
        var slens_foil = -1  # request slens_foil: with slens_tgt, the line adds "dld", the change of logit(target) - logit(foil)
        var slens_tgt = -1  # request slens_tgt: token whose NLL change slens adds as "dnll"
        var restore_s = 0.0
        var load_s = 0.0
        var load_tier = String("")  # where a state_load / ipc_ingest came from: hot (chain), file, ipc; "" = nothing loaded
        var load_split = List[Float64]()
        for _ in range(5):
            load_split.append(0.0)
        var t_req = perf_counter_ns()
        var force = force_env.copy()
        var stop_seqs = List[List[Int]]()
        var ckpt_hints = List[Int]()
        var sample = default_sample_params()
        var req_state_save = String("")
        var req_state_load = String("")
        var req_state_prefix = False
        var req_semantic_manifest_path = String("")
        var req_semantic_values_path = String("")
        var req_semantic_positions = 0
        var req_semantic_now_s = Int64(0)
        var embeds = List[String]()
        var has_img = False
        if serve:
            var line_in = Optional[String](None)
            if len(pending) > 0:
                line_in = Optional(pending.pop(0))
            else:
                line_in = read_line(0)
            if not line_in:
                break
            reset_request_knobs(bufs.imp_on, bufs.gain_l, bufs.gain_n, bufs.gain_k, bufs.steer_on, wst.grammar, wst.grammar_pending_think, wst.grammar_think_buf, wst.grammar_stop, wst.grammar_masked_draws, wst.grammar_accepted, steer_boot, imp_loaded, bufs.imp_scale, bufs.imp_layer, bufs.imp_klim, bufs.imp_lo, bufs.spl_layer, bufs.spl_zero, bufs.spl_n)
            var idle_cancel = is_cancel_line(line_in.value())
            if idle_cancel[0]:
                print(idle_cancel_note(idle_cancel[1]))
                continue
            if json_key(line_in.value(), "latent_fork") >= 0:
                if chain.count_valid() < 1:
                    print(err_line(line_id(line_in.value()), latent_fork_error()))
                    continue
                var cid = "lat_" + String(perf_counter_ns())
                var idx = chain.count_valid() - 1
                var active_ckpt_pos = chain.items[idx].pos
                var minted = mint_checkpoint_latent(chain.items[idx])
                latent_header_table[cid] = minted[0].copy()
                latent_fd_table[cid] = minted[1]
                latent_tokens_table[cid] = active_ckpt_pos
                print("{\"latent_id\": \"" + cid + "\", \"tokens\": " + String(active_ckpt_pos) + ", \"bytes\": " + String(minted[0].payload_len) + "}")
                continue
            if json_key(line_in.value(), "latent_resume") >= 0:
                var resume_pos = json_key(line_in.value(), "latent_resume")
                var cid = String("")
                _ = json_str(line_in.value(), resume_pos, cid)
                if cid not in latent_fd_table or cid not in latent_header_table or cid not in latent_tokens_table:
                    print(err_line(line_id(line_in.value()), latent_resume_error(cid)))
                    continue
                var fd = latent_fd_table[cid]
                var header = latent_header_table[cid].copy()
                var tokens = latent_tokens_table[cid]
                
                ingest_checkpoint_latent(header, fd, chain.items[0])
                chain.items[0].pos = tokens
                chain.mark_valid(0)
                print("{\"latent_id\": \"" + cid + "\"}")
                continue
            if json_key(line_in.value(), "ipc_export") >= 0 or json_key(line_in.value(), "ipc_release") >= 0 or json_key(line_in.value(), "ipc_sha") >= 0 or json_key(line_in.value(), "ipc_stats") >= 0:
                var op_line = line_in.value().copy()
                var oid = 0
                var idk = json_key(op_line, "id")
                _ = json_int(op_line, idk, oid)
                try:
                    var t_op = perf_counter_ns()
                    var ek = json_key(op_line, "ipc_export")
                    var rk = json_key(op_line, "ipc_release")
                    if json_key(op_line, "ipc_stats") >= 0:
                        var live = 0
                        for b in ipc_busy:
                            live += 1 if b != 0 else 0
                        print("{\"id\":" + String(oid) + ",\"ipc_op\":\"stats\",\"opens\":" + String(ipc_opens) + ",\"live\":" + String(live) + ",\"stages\":" + String(len(ipc_busy)) + "}")
                    elif ek >= 0:
                        var path = String("")
                        _ = json_str(op_line, ek, path)
                        var toks = state_tokens(path)
                        var want = prefix_hash(chain.salt, toks, len(toks))
                        var idx = -1
                        for attempt in range(2):
                            for i in range(len(chain.items)):
                                if chain.items[i].valid and chain.items[i].pos == len(toks) and bytes_eq(chain.items[i].hash, want):
                                    idx = i
                            if idx >= 0 or attempt == 1:
                                break
                            var ex_split = List[Float64]()
                            _ = load_state(ctx, chain, bufs.kc_d, bufs.vc_d, bufs.kvtab_h, bufs.kvtab_d, bufs.tpages, path, tmax, ex_split, load_stage)
                        if idx < 0:
                            raise Error("ipc_export: no chain item for " + path)
                        var si = -1
                        for i in range(len(ipc_busy)):
                            if ipc_busy[i] == 0:
                                si = i
                        if si < 0:
                            var st = stage_new(ipc_payload_bytes(tmax))
                            ipc_ptrs.append(st[0])
                            ipc_handles.append(st[1].copy())
                            ipc_busy.append(0)
                            si = len(ipc_busy) - 1
                        var msg = export_ckpt_ipc(ctx, chain, idx, bufs.kc_d, bufs.vc_d, bufs.kvtab_h, ipc_ptrs[si], ipc_handles[si])
                        ipc_next += 1
                        ipc_busy[si] = ipc_next
                        var live = 0
                        for b in ipc_busy:
                            live += 1 if b != 0 else 0
                        print("{\"id\":" + String(oid) + ",\"ipc_op\":\"export\",\"ipc\":" + String(ipc_next) + ",\"msg\":\"" + hex_encode(Span(msg)) + "\",\"pos\":" + String(len(toks)) + ",\"prefix_hash\":\"" + proto.u64_to_hex(hash64(want)) + "\",\"bytes\":" + String(ipc_payload_bytes(len(toks))) + ",\"live\":" + String(live) + ",\"stages\":" + String(len(ipc_busy)) + ",\"export_s\":" + String(Float64(perf_counter_ns() - t_op) / 1e9) + "}")
                    elif rk >= 0:
                        var sid = 0
                        _ = json_int(op_line, rk, sid)
                        var at = -1
                        for i in range(len(ipc_busy)):
                            if ipc_busy[i] == sid:
                                at = i
                        if at < 0:
                            raise Error("ipc_release: no live stage " + String(sid))
                        ipc_busy[at] = 0
                        var live = 0
                        for b in ipc_busy:
                            live += 1 if b != 0 else 0
                        print("{\"id\":" + String(oid) + ",\"ipc_op\":\"release\",\"ipc\":" + String(sid) + ",\"live\":" + String(live) + ",\"stages\":" + String(len(ipc_busy)) + "}")
                    else:
                        var spos = 0
                        var sk = json_key(op_line, "ipc_sha")
                        _ = json_int(op_line, sk, spos)
                        var ph = String("")
                        var phk = json_key(op_line, "ipc_ph")
                        if phk >= 0:
                            _ = json_str(op_line, phk, ph)
                        var idx = -1
                        var hits = 0
                        for i in range(len(chain.items)):
                            if chain.items[i].valid and chain.items[i].pos == spos and (ph == "" or proto.u64_to_hex(hash64(chain.items[i].hash)) == ph):
                                idx = i
                                hits += 1
                        if hits > 1:
                            raise Error("ipc_sha: " + String(hits) + " chain items at pos " + String(spos) + "; name the prefix_hash")
                        if idx < 0:
                            raise Error("ipc_sha: no chain item at pos " + String(spos) + " with prefix_hash " + ph)
                        var sha = ckpt_region_sha(ctx, chain, idx, bufs.kc_d, bufs.vc_d, bufs.kvtab_h)
                        print("{\"id\":" + String(oid) + ",\"ipc_op\":\"sha\",\"sha\":\"" + sha + "\",\"pos\":" + String(spos) + ",\"opens\":" + String(ipc_opens) + ",\"sha_s\":" + String(Float64(perf_counter_ns() - t_op) / 1e9) + "}")
                except e:
                    print(err_line(oid, String(e)))
                continue
            if json_key(line_in.value(), "a3c2") >= 0:
                var line1 = read_line(0)
                if not line1:
                    print(err_line(0, "A3C two-row batch ended before its second row"))
                    continue
                run_a3c_serial_pair(ctx, bufs, line_in.value(), line1.value(), pack_q4, draft_q4, q4_off, e, kcfg, spec_dbg, expert_trace, att_split, dot3, pf_chunk, pf_on, mega, mega_win, fr_k, fr_off, fr_ids_off, tmax, kvtab_mode)
                continue
            if json_key(line_in.value(), "a3c") >= 0:
                var line1 = read_line(0)
                var line2 = read_line(0)
                var line3 = read_line(0)
                if not line1 or not line2 or not line3:
                    print(err_line(0, "A3C batch ended before four rows"))
                    continue
                run_a3c_serial(ctx, bufs, line_in.value(), line1.value(), line2.value(), line3.value(), pack_q4, draft_q4, q4_off, e, kcfg, spec_dbg, expert_trace, att_split, dot3, pf_chunk, pf_on, mega, mega_win, fr_k, fr_off, fr_ids_off, tmax, kvtab_mode)
                continue
            if json_key(line_in.value(), "a3b") >= 0:
                var line_b = Optional[String](None)
                if len(pending) > 0:
                    line_b = Optional(pending.pop(0))
                else:
                    line_b = read_line(0)
                if not line_b:
                    print(err_line(0, "A3B pair ended before its second request"))
                    continue
                run_a3b_pair(ctx, bufs, line_in.value(), line_b.value(), pack_q4, draft_q4, q4_off, e, kcfg, spec_dbg, expert_trace, att_split, dot3, pf_chunk, pf_on, mega, mega_win, fr_k, fr_off, fr_ids_off, tmax, kvtab_mode)
                continue
            t_req = perf_counter_ns()
            var req_n = 0
            var req_spec = False
            var req_has_spec = False
            var perr = parse_request(line_in.value(), req_id, prompt, req_n, req_spec, req_has_spec, stop_seqs, ckpt_hints, sample, req_state_save, req_state_load, req_state_prefix, req_semantic_manifest_path, req_semantic_values_path, req_semantic_positions, req_semantic_now_s)
            if perr == "":
                perr = parse_embeds(line_in.value(), embeds)
                has_img = len(embeds) > 0
                if has_img:
                    spec = False
            if perr == "" and len(prompt) < 1:
                perr = "empty prompt"
            if perr == "" and req_n < 1:
                perr = "n must be >= 1"
            if perr == "" and len(prompt) + req_n > tmax:
                perr = "prompt+n exceeds TMAX " + String(tmax)
            # per-request steering (BARO_STEER_SERVE=1): steer_file (f32[N_LAYERS * H], or f32[H] with steer_layer) or
            # steer_bank + steer_layer + steer_row + steer_scale; a request without them runs unsteered.
            if perr == "":
                var sline = line_in.value()
                var lk = json_key(sline, "slens_ctx")
                if lk >= 0:
                    if not json_int(sline, lk, slens_c):
                        perr = "slens_ctx must be an integer"
                    var lk2 = json_key(sline, "slens_k")
                    if lk2 >= 0 and not json_int(sline, lk2, slens_k):
                        perr = "slens_k must be an integer"
                    var lk5 = json_key(sline, "slens_foil")
                    if lk5 >= 0 and not json_int(sline, lk5, slens_foil):
                        perr = "slens_foil must be an integer"
                    var lk4 = json_key(sline, "slens_tgt")
                    if lk4 >= 0 and not json_int(sline, lk4, slens_tgt):
                        perr = "slens_tgt must be an integer"
                    var lk3 = json_key(sline, "slens_flags")
                    if lk3 >= 0 and not json_int(sline, lk3, slens_flags):
                        perr = "slens_flags must be an integer"
                    if perr == "" and slens_foil >= 0 and slens_tgt < 0:
                        perr = "slens_foil needs slens_tgt (the logit difference is target minus foil)"
                    if perr == "" and (slens_tgt >= VOCAB or slens_foil >= VOCAB):
                        perr = "slens_tgt and slens_foil must be below the vocabulary size " + String(VOCAB)
                    if sample.top_logprobs < 1:
                        sample.top_logprobs = 1  # the staged logits row exists only on the top-logprobs path (launch-path head)
            # request field "capture": NAME records into BARO_CAPTURE/NAME (engine started with BARO_CAPTURE)
            var cname = String("")
            var ckey = json_key(line_in.value(), "capture")
            if perr == "" and ckey >= 0 and json_str(line_in.value(), ckey, cname):
                if not cap_on:
                    perr = "capture needs the engine started with BARO_CAPTURE=DIR and BARO_CAPTURE_LAYERS"
                elif cname.byte_length() == 0 or cname.find("/") >= 0 or cname.find("..") >= 0:
                    perr = "capture name: no slash or .. allowed"
                else:
                    cap_req = cap_dir + "/" + cname
                    try:
                        makedirs(cap_req, exist_ok=True)
                        capture_meta(cap_req, getenv("BARO_CAPTURE_MODEL", pack.packdir), cap_fields, bufs)
                    except e:
                        perr = String(e)
            bufs.imp_on = imp_loaded  # request field "implants": 0 switches the loaded set off for this request
            var ikey = json_key(line_in.value(), "implants")
            var ival = 1
            if perr == "" and ikey >= 0 and imp_loaded and json_int(line_in.value(), ikey, ival):
                bufs.imp_on = ival != 0
            bufs.imp_scale = 1.0
            bufs.imp_layer = -1
            bufs.imp_klim = -1
            bufs.imp_lo = 0
            var ilo = json_key(line_in.value(), "implants_lo")
            var losel = 0
            if perr == "" and imp_loaded and ilo >= 0 and json_int(line_in.value(), ilo, losel):
                bufs.imp_lo = losel
            var ikk = json_key(line_in.value(), "implants_k")
            var ksel = -1
            if perr == "" and imp_loaded and ikk >= 0 and json_int(line_in.value(), ikk, ksel):
                bufs.imp_klim = ksel
            var iks = json_key(line_in.value(), "implants_scale")
            var ikl = json_key(line_in.value(), "implants_layer")
            var fsc = 1.0
            var lsel = -1
            if perr == "" and imp_loaded and iks >= 0 and json_float(line_in.value(), iks, fsc):
                bufs.imp_scale = Float32(fsc)
            if perr == "" and imp_loaded and ikl >= 0 and json_int(line_in.value(), ikl, lsel):
                bufs.imp_layer = lsel
            if perr == "" and (ilo >= 0 or ikk >= 0 or iks >= 0 or ikl >= 0) and not imp_loaded:
                perr = "implants_lo / implants_k / implants_scale / implants_layer need the engine started with BARO_IMPLANTS=FILE"
            if perr == "" and bufs.imp_klim >= 0 and bufs.imp_lo > bufs.imp_klim:
                perr = "implants_lo " + String(bufs.imp_lo) + " is above implants_k " + String(bufs.imp_klim) + ": empty row range"
            bufs.spl_layer = -1  # request fields splice_layer + (splice FILE | splice_zero:1), docs/prereg-splice.md
            bufs.spl_zero = False
            bufs.spl_n = 0
            var spk = json_key(line_in.value(), "splice_layer")
            if perr == "" and spk >= 0:
                var spl = 0
                var spz = 0
                var spf = String("")
                var spkf = json_key(line_in.value(), "splice")
                var spkz = json_key(line_in.value(), "splice_zero")
                if not json_int(line_in.value(), spk, spl) or spl < 0 or spl >= N_LAYERS:
                    perr = "splice_layer must be an integer layer"
                elif not steer_serve:
                    perr = "splice needs BARO_STEER_SERVE=1 (layer loop, no batched prefill)"
                elif spkz >= 0 and json_int(line_in.value(), spkz, spz) and spz != 0:
                    bufs.spl_layer = spl
                    bufs.spl_zero = True
                elif spkf >= 0 and json_str(line_in.value(), spkf, spf):
                    try:
                        var sb = open(spf, "r").read_bytes()
                        if len(sb) % (H * 4) != 0 or len(sb) // (H * 4) > SPLICE_MAX or len(sb) == 0:
                            raise Error("splice file: f32[n][" + String(H) + "] with n <= " + String(SPLICE_MAX))
                        if len(prompt) + req_n > len(sb) // (H * 4):  # the layer loop reads row pos for every position the request visits: refuse a file with fewer rows (also rows left by an earlier longer file)
                            raise Error("splice file has " + String(len(sb) // (H * 4)) + " rows, this request visits " + String(len(prompt) + req_n) + " positions (prompt + n)")
                        bufs.spl_n = len(sb) // (H * 4)
                        var sh = ctx.enqueue_create_host_buffer[f32](len(sb) // 4)
                        for sj in range(len(sb)):
                            sh.unsafe_ptr().unsafe_bitcast[UInt8]()[sj] = sb[sj]
                        ctx.enqueue_copy(dst_buf=DeviceBuffer[f32](ctx, bufs.spl_d.unsafe_ptr(), len(sb) // 4, owning=False), src_buf=sh)
                        ctx.synchronize()
                        bufs.spl_layer = spl
                    except e:
                        perr = String(e)
                else:
                    perr = "splice_layer needs splice FILE or splice_zero:1"
            bufs.gain_l = List[Int]()
            bufs.gain_n = List[Int]()
            bufs.gain_k = List[Float32]()
            var gkey = json_key(line_in.value(), "gain")  # "L:N:k,...": FFN activation N of layer L times k (docs/format-capture.md neighbours: ablation, surgery gate)
            var gstr = String("")
            if perr == "" and gkey >= 0 and json_str(line_in.value(), gkey, gstr):
                if not steer_serve:
                    perr = "gain needs BARO_STEER_SERVE=1 (batched prefill and speculation off)"
                try:
                    for gu in gstr.split(","):
                        var gq = gu.split(":")
                        if len(gq) != 3 or Int(gq[0]) < 0 or Int(gq[0]) >= N_LAYERS or Int(gq[1]) < 0 or Int(gq[1]) >= FFN:
                            raise Error("gain wants L:N:k with L < " + String(N_LAYERS) + ", N < " + String(FFN))
                        bufs.gain_l.append(Int(gq[0]))
                        bufs.gain_n.append(Int(gq[1]))
                        bufs.gain_k.append(Float32(Float64(gq[2])))
                except e:
                    perr = String(e)
            bufs.steer_on = steer_boot
            if perr == "" and steer_serve:
                var sfp = String("")
                var sbk = String("")
                var line = line_in.value()
                var ki = json_key(line, "steer_bank")
                var kf = json_key(line, "steer_file")
                var kl = json_key(line, "steer_layer")
                var sl = -1
                if kl >= 0 and not json_int(line, kl, sl):
                    perr = "steer_layer must be an integer"
                try:
                    if perr == "" and ki >= 0 and json_str(line, ki, sbk):
                        var sr = -1
                        var ks = json_key(line, "steer_row")
                        var kc = json_key(line, "steer_scale")
                        var sc: Float64 = 1.0
                        if kl < 0 or ks < 0 or not json_int(line, ks, sr):
                            raise Error("steer_bank needs integer steer_layer and steer_row")
                        if kc >= 0 and not json_float(line, kc, sc):
                            raise Error("steer_scale must be a number")
                        load_steer_row(ctx, bufs.sv_d, sbk, sl, sr, Float32(sc))
                        bufs.steer_on = True
                    elif perr == "" and kf >= 0 and json_str(line, kf, sfp):
                        load_steer_vec(ctx, bufs.sv_d, sfp, sl)
                        bufs.steer_on = True
                except e:
                    perr = String(e)
            # C3 fix round (bench/chat-protocol.md, exchange/lane-WIRING-report.md
            # C3 section): serve/sample_ref.mojo's ceil'd top-p mass target was
            # the root cause of the M5 gate-2 finding above; fixed in 054bd22
            # and gate 2 now passes clean on all four truncated shapes
            # (top_p<1 and/or top_k>0, with or without min_p), so the refusal
            # for those shapes is lifted.
            #
            # C3 tail round (bench/chat-protocol.md): the untruncated shape
            # (top_p=1, top_k=0, min_p<=0) is a separate, still-open defect —
            if perr != "":
                print(err_line(req_id, perr))
                continue
            gen_n = req_n
            if req_has_spec:
                spec = req_spec
            if spec and pack_kq:
                # The kq trunk has no MTP draft path (startup refuses BARO_SPEC=1); a request that asks for
                # speculation decodes without it instead of faulting on the draft weights.
                print("kq pack: spec requested, served without speculation")
                spec = False
            # A1 (bench/spec-sample-protocol.md): temperature > 0 used to force
            # spec off, because accepting a draft on argmax equality under
            # sampling silently changes the output distribution. The window now
            # runs the real rule instead (accept with min(1, p/q) on the
            # truncated distributions, residual draw on the first rejection,
            # bonus token from p), so sampling and speculation compose.
            # JSON-enforcement item 1/2: reset every request, schema or not
            # -- wst is reused across requests, so a prior request's matcher
            # must never leak into one that never asked for response_format.
            wst.grammar = None
            wst.grammar_pending_think = False
            wst.grammar_think_buf = List[UInt8]()
            wst.grammar_stop = False
            wst.grammar_masked_draws = 0
            wst.grammar_accepted = 0
            var schema_raw = parse_schema_field(line_in.value())
            if schema_raw != "":
                # The masked sampler applies the grammar mask before top_k /
                # top_p / min_p (0a6328b), so the request's own truncation is
                # honoured; only speculation stays off for grammar requests.
                spec = False
                if not grt:
                    try:
                        grt = Optional(GrammarRuntime(packdir))
                    except e:
                        print(err_line(req_id, "response_format: grammar runtime failed to load: " + String(e)))
                        continue
                if grt.value().vocab_size != VOCAB:
                    print(err_line(req_id, "response_format: grammar vocab_size " + String(grt.value().vocab_size) + " != engine VOCAB " + String(VOCAB) + " (stale pack?)"))
                    continue
                try:
                    wst.grammar = Optional(compile_schema_matcher(grt.value(), schema_raw))
                except e:
                    print(err_line(req_id, "response_format schema: " + String(e)))
                    continue
                wst.grammar_pending_think = parse_reasoning_field(line_in.value())
                wst.grammar_mask = Bitset(VOCAB)
                print("response_format: schema compiled, reasoning wait:", wst.grammar_pending_think)
            wst.tool_re = ""
            wst.tool_buf = List[UInt8]()
            wst.grammar_tool = False
            if not wst.grammar:
                # Tool names (response_format, when set, governs the whole reply instead).
                var terr = String("")
                try:
                    terr = tool_arm(wst, grt, packdir, line_in.value())
                except e:
                    terr = "tools: " + String(e)
                if terr != "":
                    print(err_line(req_id, terr))
                    continue
                if wst.tool_re != "" or wst.grammar_tool:
                    # spec windows draw m > 1 tokens unmasked; the name draw is m = 1 only
                    spec = False
            if spec and len(prompt) < 2:
                print("prompt has 1 token: speculation off for this request (the draft head needs prefill rows)")
                spec = False
            print("prompt tokens:", len(prompt), " n:", gen_n, " spec:", spec, " temperature:", sample.temperature)
            # Per-request teacher forcing. BARO_FORCE is read once at startup,
            # so a forced run used to need one process per prompt: 20 pack
            # loads for a 20-prompt gate, 1.4 s each on the MoE pack. A
            # "force":[ids] field lets one resident process serve the whole
            # gate. Absent, the request inherits BARO_FORCE.
            parse_force_field(line_in.value(), force)
            # M1a prefix checkpoint: restore the SSM slot for the longest
            # hashed prefix and keep the KV pool (position addressed, [0, cached)
            # still in place); the draft KV is never prefilled, so it is zeroed
            # exactly as on the cold path. A miss is today's path.
            # spec mode replays at least one prompt row so the draft head's
            # hidden rows (hn_d) are fresh: the checkpoint at len-1 is skipped.
            var ipc_key = json_key(line_in.value(), "ipc_ingest")
            if ipc_key >= 0:
                # T3b B side: the 320-byte message from engine A plus the prefix ids (hex u32 LE);
                # the header is checked against this pack and these ids before any HIP call.
                var t_ipc = perf_counter_ns()
                var ipc_hex = String("")
                var pre_hex = String("")
                _ = json_str(line_in.value(), ipc_key, ipc_hex)
                var pk = json_key(line_in.value(), "ipc_prefix")
                if pk >= 0:
                    _ = json_str(line_in.value(), pk, pre_hex)
                var pre = tokens_from_hex(pre_hex)
                var msg = _hex_to_bytes(ipc_hex)
                var refusal = String("IPC refused: message is not 320 bytes")
                if len(msg) == MSG_BYTES:
                    refusal = ipc_refusal(proto.LatentHeader.deserialize(Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(msg.unsafe_ptr()))), chain.salt, pre, tmax)
                if refusal != "":
                    print(err_line(req_id, refusal))
                    continue
                ipc_opens += 1
                var ipc_err = String("")
                try:
                    _ = ingest_ckpt_ipc(ctx, chain, bufs.kc_d, bufs.vc_d, bufs.kvtab_h, bufs.kvtab_d, bufs.tpages, msg, pre)
                except e:
                    ipc_err = String(e)
                if ipc_err != "":
                    print(err_line(req_id, ipc_err))
                    continue
                var ipc_pos = len(pre)
                for t in prompt:
                    pre.append(t)
                prompt = pre^
                load_s = Float64(perf_counter_ns() - t_ipc) / 1e9
                load_tier = String("ipc")
                print("ipc ingest: pos", ipc_pos, " bytes", ipc_payload_bytes(ipc_pos), " in", load_s, "s")
            if req_state_load != "":
                # Checkpoint API: bring the named state file into the chain
                # before the lookup; the lookup then finds it by prefix hash.
                # A refusal (different pack, bad file) is this request's
                # error, never the engine's death.
                var load_err = String("")
                var t_hot = perf_counter_ns()
                if req_state_prefix:
                    # P2b: the line carried the suffix only; the prefix is the
                    # state file's own token list (header, 4 bytes per id).
                    try:
                        var pre = state_tokens(req_state_load)
                        for t in prompt:
                            pre.append(t)
                        prompt = pre^
                    except e:
                        load_err = String(e)
                    if load_err != "":
                        print(err_line(req_id, load_err))
                        continue
                var hot_idx = state_hot(chain, req_state_load, prompt)
                if hot_idx >= 0:
                    load_s = Float64(perf_counter_ns() - t_hot) / 1e9
                    load_tier = String("hot")
                    print("state hot:", req_state_load, " pos", chain.items[hot_idx].pos, " chain hit, file skipped, in", load_s, "s")
                else:
                    try:
                        var t_ld = perf_counter_ns()
                        var lpos = load_state(ctx, chain, bufs.kc_d, bufs.vc_d, bufs.kvtab_h, bufs.kvtab_d, bufs.tpages, req_state_load, tmax, load_split, load_stage)
                        load_s = Float64(perf_counter_ns() - t_ld) / 1e9
                        load_tier = String("file")
                        print("state loaded:", req_state_load, " pos", lpos, " in", load_s, "s", " read", load_split[0], " parse", load_split[1], " pin", load_split[2], " host", load_split[3], " h2d", load_split[4])
                    except e:
                        load_err = String(e)
                if load_err != "":
                    print(err_line(req_id, load_err))
                    continue
            if window_exceeded(len(prompt), gen_n, tmax):
                # state_prefix / ipc_ingest grew the prompt after the first window check: an error line, never exit(2) (B-M1)
                print(err_line(req_id, window_error(len(prompt), gen_n, tmax)))
                continue
            ckpt_idx = -1 if (has_img or bufs.steer_on or cap_on) else chain.lookup(prompt, len(prompt) - 1 if spec else len(prompt))  # a steered prompt must not reuse (or leave) unsteered state
            cached = chain.pos_of(ckpt_idx)
            chain.touch(ckpt_idx)
            if chain.npages > 0:
                var need = min(bufs.tpages, ceildiv(len(prompt) + gen_n + KMAX + 1, KVPAGE))
                chain.map_pages(ctx, ckpt_idx, cached, need, bufs.kc_d, bufs.vc_d, bufs.kvtab_h, bufs.kvtab_d, bufs.tpages)
            else:
                chain.invalidate_above(cached)
            var t_restore = perf_counter_ns()
            if ckpt_idx >= 0:
                chain.restore(ctx, bufs.convstate_d, bufs.sstate_d, 0, ckpt_idx)
            else:
                ctx.enqueue_memset(bufs.convstate_d, 0)
                ctx.enqueue_memset(bufs.sstate_d, 0)
                if chain.npages == 0:
                    ctx.enqueue_memset(bufs.kc_d, 0)
                    ctx.enqueue_memset(bufs.vc_d, 0)
            # The draft KV pool (SEQ_CAP x kvpool1 per K and V) is only read by
            # the speculative window; a spec-off request leaves it alone
            # (QS batch item 5: 2.4 ms of a hot fork's restore_s were these).
            if spec:
                ctx.enqueue_memset(bufs.kc32_d, 0)
                ctx.enqueue_memset(bufs.vc32_d, 0)
            ctx.enqueue_memset(bufs.ctr_d, 0)
            ctx.enqueue_memset(bufs.prof_d, 0)
            ctx.synchronize()
            restore_s = Float64(perf_counter_ns() - t_restore) / 1e9
        else:
            var prompt_path = getenv("BARO_PROMPT", packdir + "/prompt-tokens.txt")
            print("prompt file:", prompt_path)
            with open(prompt_path, "r") as f:
                var data = f.read_bytes()
                var val = 0
                var have = False
                for i in range(len(data)):
                    var b = Int(data[i])
                    if b >= 48 and b <= 57:
                        val = val * 10 + (b - 48)
                        have = True
                    else:
                        if have:
                            prompt.append(val)
                        val = 0
                        have = False
                if have:
                    prompt.append(val)
            print("prompt tokens:", len(prompt))
            if chain.cap > 0:
                ckpt_idx = -1 if (has_img or bufs.steer_on or cap_on) else chain.lookup(prompt, len(prompt) - 1 if spec else len(prompt))  # a steered prompt must not reuse (or leave) unsteered state
                cached = chain.pos_of(ckpt_idx)
                if ckpt_idx >= 0:
                    var t_rs = perf_counter_ns()
                    chain.restore(ctx, bufs.convstate_d, bufs.sstate_d, 0, ckpt_idx)
                    ctx.synchronize()
                    restore_s = Float64(perf_counter_ns() - t_rs) / 1e9

        var pf_rows = 0
        var pf_tail = 0
        if pf_on and len(prompt) - 1 - cached >= pf_min:
            pf_rows = len(prompt) - 1
            pf_tail = (pf_rows - cached) % MROWS
            if pf_tail == 0:
                pf_tail = MROWS
        var prefill_rows = len(prompt) - 1 - cached
        print("TMAX:", tmax, " kv dtype:", String(KVT), " prefill chunk:", pf_chunk, " prefill rows:", pf_rows, " cached:", cached, " replay rows:", prefill_rows)
        # BARO_MARGIN=1 (with BARO_FORCE): after each forced step, recompute the
        # head from that step's final residual (final rmsnorm, q4 head GEMM,
        # reduce) and print the top-2 logits and their gap next to the argmax
        # the engine produced.
        var margin = getenv("BARO_MARGIN", "0") == "1" and len(force) > 0
        print("BARO_MARGIN:", margin)

        # --- decode loop ---------------------------------------------------------


        var n_total = len(prompt) + gen_n  # window already checked after prefix extension (B-M1)
        var toks_h = ctx.enqueue_create_host_buffer[DType.int32](tmax)
        ctx.synchronize()
        for i in range(tmax):
            toks_h[i] = 0
        for i in range(len(prompt)):
            toks_h[i] = Int32(prompt[i])
        ctx.enqueue_copy(dst_buf=DeviceBuffer[DType.int32](ctx, toks_d.unsafe_ptr(), tmax, owning=False), src_buf=toks_h)
        ctx.synchronize()
        if has_img:
            var verr = stage_vision(ctx, bufs, prompt, embeds, tmax)
            if verr != "":
                print(err_line(req_id, verr))
                continue
        var t0 = perf_counter_ns()
        var t_prefill_end = t0
        # BARO_PROFILE=1: synchronize at sub-block boundaries and attribute GPU
        # time to attn / ssm / ffn / head. Off by default; the timed path is
        # untouched.
        var prof = getenv("BARO_PROFILE", "0") != "0"
        var pf2 = getenv("BARO_PROFILE", "0") == "2"
        # BARO_PROFILE=3: split the draft path (blk32_forward's process + k-1
        # draft-step calls, plus the accept block) into layer body / head gemm+
        # reduce / argmax / accept-sync buckets; "other" is the leftover
        # tokcp/bookkeeping dispatch cost inside pf_proc+pf_draft.
        var pf3 = getenv("BARO_PROFILE", "0") == "3"
        # BARO_PROFILE=4: same idea as 2, on the ffn sub-block -- rmsnorm / gate
        # gemm / up gemm / swiglu / down gemm / residual add. Serialized, so the
        # stage sum exceeds the unsynchronized sub-block time; compare stages
        # within an arm and the same stage across m, never add these into a budget.
        var pf4 = getenv("BARO_PROFILE", "0") == "4"
        # Items 3-4 (briefs/2026-09-16-sampling-all-models-lane.md), fix for
        # the coordinator's P1 inert-parameter finding: penalties/top_logprobs
        # are only applied on window.mojo's plain (non-spec) single-token
        # decode step, so a default request (spec ON by default, temperature
        # possibly 0) could silently never reach it. Force both the
        # megakernel and speculation off for the rest of THIS request instead
        # of letting the parameter fall through unused.
        var want_extra = sample.presence_penalty != 0 or sample.frequency_penalty != 0 or sample.top_logprobs > 0 or sample.hidden == 1 or sample.logits_topk > 0 or slens_c >= 0
        var want_grammar = wst.grammar.__bool__() or wst.tool_re != ""
        if want_extra or want_grammar:
            spec = False
        comptime if IS_MOE:
            # Prompt lookup currently verifies plain greedy tokens only.
            # A multi-id stop sequence could complete inside a verified window; single-id stops
            # (the EOS ids baro-serve sends on every chat request) are never drafted, so a
            # window can only end on one and the end-only stop check below stays exact.
            var multi_stop = False
            for si in range(len(stop_seqs)):
                if len(stop_seqs[si]) > 1:
                    multi_stop = True
            if sample.temperature > 0 or sample.embed != 0 or dump or expert_trace or multi_stop:
                spec = False
        # A6 (bench/chat-protocol.md A6): every m == 1 decode step runs the
        # megakernel. window.mojo folds the head (argmax inside the launch)
        # only for a plain greedy request; sampling, penalties, top_logprobs
        # and grammar requests run the layers with fold_head 0 (the shape
        # the window already uses) and the launch-path head plus the
        # sampler after it, so their logits edits and mask sit between the
        # head and the draw as before (amar_sample_row is argmax-equivalent
        # at temperature <= 0, P-K2).
        var plan = decide_path(mega, mega_win, spec, cap_on, len(bufs.gain_l) != 0 or bufs.imp_on or bufs.spl_layer >= 0, want_extra, want_grammar, has_img, sample.temperature > 0)
        var mega_req = plan.mega
        # A1: the megakernel WINDOW writes the window's tokens itself, so it
        # cannot host the speculative sampling rule (which needs the target's
        # full probability rows, not its argmax). Sampling therefore stays on
        # the launch path for the window too, exactly as it already does for
        # the single-token megakernel above.
        var mega_win_req = plan.mega_win
        print("path:", path_word(plan.tag), "" if plan.why == "" else " (megakernel off: " + plan.why + ")")
        var dump_pen = want_extra and (dump_pen_dir != "" or slens_c >= 0)
        var cfg = WindowCfg(pack_q4=pack_q4, draft_q4=draft_q4, q4_off=q4_off, e=e, kcfg=kcfg, spec=spec, spec_dbg=spec_dbg, expert_trace=expert_trace, serve=serve, req_id=req_id, prof=prof, pf2=pf2, pf3=pf3, pf4=pf4, dump=dump, dump4=dump4, dump_layer=dump_layer, mega=mega_req, att_split=att_split, mega_win=mega_win_req, dot3=dot3, pf_chunk=pf_chunk, pf_rows=pf_rows, pf_tail=pf_tail, n_total=n_total, fr_k=fr_k, fr_off=fr_off, fr_ids_off=fr_ids_off, replay_m=replay_m, n_prompt=len(prompt), sample=sample.copy(), dump_pen=dump_pen, mrope=has_img, inject=has_img)
        if len(force) > 0 and cfg.spec:
            raise Error("BARO_FORCE requires BARO_SPEC=0 (teacher forcing is a no-spec identity gate)")
        wst.reset(t0)
        wst.pos = cached
        wst.pos_prev = cached
        var prefill_done = False
        var cancelled = False
        var stopped = False
        var errored = False  # a request-level failure that ends the request with finish "error" (B-M15)
        var predicted = List[Int]()
        var force_tok_h = ctx.enqueue_create_host_buffer[DType.int32](1)
        var lg_h = ctx.enqueue_create_host_buffer[f32](VOCAB if margin else 1)
        var hidden_h = ctx.enqueue_create_host_buffer[f32](H)
        var ngram_ns = 0
        var verify_ns = 0
        var verify_rows = 0
        var history = List[Int]()
        var hist_end = 0
        # The stopwatch stays here, in the harness that is never embedded in a
        # gguf: step_window cannot reach t0, t_prefill_end or dt (P-A, 2026-09-08).
        while wst.pos < n_total - 1:
            var pos_before = wst.pos
            var ngram_n = 0
            comptime if IS_MOE:
                if cfg.spec and wst.pos >= len(prompt) and wst.pos + 2 < n_total:
                    var draft_start = perf_counter_ns()
                    # Copy only tokens committed since the last proposal; the
                    # full 4096-token re-copy per step cost ~20% of decode.
                    if len(history) == 0:
                        hist_end = wst.pos + 1 - min(wst.pos + 1, 4096)
                    var hn = min(wst.pos + 1 - hist_end, 4096)
                    if hn > 0:
                        ctx.enqueue_copy(dst_buf=bufs.pen_hist_h.create_sub_buffer[DType.int32](0, hn), src_buf=DeviceBuffer[DType.int32](ctx, bufs.toks_d.unsafe_ptr().unsafe_offset(wst.pos + 1 - hn), hn, owning=False))
                        ctx.synchronize()
                        for hi in range(hn):
                            history.append(Int(bufs.pen_hist_h[hi]))
                        hist_end = wst.pos + 1
                    var proposed = ngram_propose(history, min(min(kcfg, MROWS - 1), n_total - 2 - wst.pos), ngram_min)
                    ngram_n = len(proposed)
                    for ni in range(len(proposed)):
                        for si in range(len(stop_seqs)):
                            if ni < ngram_n and stop_seqs[si][0] == proposed[ni]:
                                ngram_n = ni
                    if ngram_n > 0:
                        for ni in range(ngram_n):
                            bufs.win_h[ni] = Int32(proposed[ni])
                        ctx.enqueue_copy(dst_buf=DeviceBuffer[DType.int32](ctx, bufs.toks_d.unsafe_ptr().unsafe_offset(wst.pos + 1), ngram_n, owning=False), src_buf=bufs.win_h.create_sub_buffer[DType.int32](0, ngram_n))
                        ctx.synchronize()
                    ngram_ns += Int(perf_counter_ns() - draft_start)
            var verify_start = perf_counter_ns()
            if len(ckpt_hints) > 0 and wst.pos < pf_rows:
                # A hint may fall inside what would otherwise be one big
                # prefill chunk; cap this call's chunk so wst.pos actually
                # stops there (checkpoints are only taken between calls).
                var step_cfg = cfg.copy()
                step_cfg.pf_chunk = min(cfg.pf_chunk, next_ckpt_stop(ckpt_hints, wst.pos, pf_rows) - wst.pos)
                step_window(ctx, bufs, step_cfg, wst)
            else:
                step_window(ctx, bufs, cfg, wst, ngram_n)
            if cap_on:
                var adv = wst.pos - pos_before
                if adv == 1:
                    capture_write(ctx, bufs, cap_req, cap_fields, pos_before, req_id)
                elif adv > 1 and pos_before < pf_rows and adv <= PF_CAP_ROWS:  # a batched prefill chunk (BARO_STEER_SERVE unset keeps it on)
                    capture_write_chunk(ctx, bufs, cap_req, cap_fields, pos_before, adv, req_id)
                else:
                    raise Error("BARO_CAPTURE: a step advanced " + String(adv) + " positions (speculation or the megakernel is on, or a prefill chunk over " + String(PF_CAP_ROWS) + " rows)")
            if pos_before == len(prompt) - 1 and wst.pos == pos_before + 1 and getenv("BARO_VRAM_LOG", "0") == "1":
                ctx.synchronize()
                print_vram_log("after_first_token")
            if ngram_n > 0:
                ctx.synchronize()
                verify_ns += Int(perf_counter_ns() - verify_start)
                verify_rows += ngram_n + 1
            if sample.hidden == 1 and pos_before >= len(prompt) - 1:
                ctx.enqueue_copy(dst_buf=hidden_h, src_buf=DeviceBuffer[f32](ctx, bufs.hn_d.unsafe_ptr(), H, owning=False))
            if sample.hidden == 1 or sample.logits_topk > 0:
                ctx.synchronize()
            if sample.hidden == 1 and pos_before >= len(prompt) - 1:
                var hl = String("{\"id\":") + String(req_id) + ",\"hidden\":["
                for hi in range(H):
                    if hi > 0:
                        hl += ","
                    hl += String(hidden_h[hi])
                print(hl + "]}")
            if sample.logits_topk > 0 and pos_before >= len(prompt) - 1:
                var tk = min(sample.logits_topk, 20)
                var ids = List[Int]()
                var vals = List[Float64]()
                for _ in range(tk):
                    ids.append(-1)
                    vals.append(-1e30)
                for vi in range(VOCAB):
                    var lv = Float64(bufs.dump_row_h[vi])
                    var at = 0
                    while at < tk and lv <= vals[at]:
                        at += 1
                    if at < tk:
                        var j = tk - 1
                        while j > at:
                            ids[j] = ids[j - 1]
                            vals[j] = vals[j - 1]
                            j -= 1
                        ids[at] = vi
                        vals[at] = lv
                var tl = String("{\"id\":") + String(req_id) + ",\"logits_topk\":["
                for i in range(tk):
                    if i > 0:
                        tl += ","
                    tl += "{\"id\":" + String(ids[i]) + ",\"logit\":" + String(vals[i]) + "}"
                print(tl + "]}")
            # Item 4 verification (coordinator review, 2026-09-16): the sync
            # and file write live here, in the harness, never in
            # window.mojo. bufs.dump_row_h and bufs.pen_hist_h were staged
            # by window.mojo's own enqueue_copy calls (cfg.dump_pen /
            # penalties-or-logprobs), synchronized here for the first time.
            if sample.embed == 1 and pos_before == len(prompt) - 1:
                var emb_h = ctx.enqueue_create_host_buffer[f32](H)
                ctx.enqueue_copy(dst_buf=emb_h, src_buf=DeviceBuffer[f32](ctx, bufs.hn_d.unsafe_ptr(), H, owning=False))
                ctx.synchronize()
                var ss: Float64 = 0
                for ei in range(H):
                    ss += Float64(emb_h[ei]) * Float64(emb_h[ei])
                var inv = 1.0 / sqrt(ss) if ss > 0 else 0.0
                var el = String("{\"id\":") + String(req_id) + ",\"embed\":["
                for ei in range(H):
                    if ei > 0:
                        el += ","
                    el += String(Float32(Float64(emb_h[ei]) * inv))
                print(el + "]}")
            if slens_c >= 0 and cfg.dump_pen and wst.pos == pos_before + 1 and pos_before + 1 == len(prompt) and not slens_base_ok(slens_base, slens_c, slens_k):
                print(err_line(req_id, slens_base_error(slens_c)))
                errored = True
                break
            if slens_c >= 0 and cfg.dump_pen and wst.pos == pos_before + 1 and pos_before + 1 == len(prompt):
                ctx.synchronize()
                slens_step(bufs.dump_row_h, slens_base, slens_acc, slens_cnt, slens_kl, slens_dnll, slens_dld, slens_nll, slens_c, slens_k, slens_flags, req_id, slens_tgt, slens_foil)
            if cfg.dump_pen and dump_pen_dir != "" and wst.pos == pos_before + 1 and pos_before + 1 >= len(prompt):
                ctx.synchronize()
                var step = pos_before + 1 - len(prompt)
                with open(dump_pen_dir + "/row-" + String(step) + ".bin", "w") as f:
                    var p = bufs.dump_row_h.unsafe_ptr().unsafe_bitcast[UInt8]()
                    f.write_bytes(Span[UInt8](unsafe_ptr=p, length=VOCAB * 4))
                with open(dump_pen_dir + "/row-" + String(step) + ".json", "w") as jf:
                    var js = String("{\"temperature\":") + String(sample.temperature) + ",\"top_k\":" + String(sample.top_k)
                    js += ",\"top_p\":" + String(sample.top_p) + ",\"min_p\":" + String(sample.min_p)
                    js += ",\"presence_penalty\":" + String(sample.presence_penalty) + ",\"frequency_penalty\":" + String(sample.frequency_penalty)
                    js += ",\"top_logprobs\":" + String(sample.top_logprobs) + ",\"history\":["
                    for i in range(step):
                        if i > 0:
                            js += ","
                        js += String(Int(bufs.pen_hist_h[i]))
                    js += "]}"
                    jf.write_bytes(js.as_bytes())
            if len(force) > 0 and pos_before >= len(prompt) - 1:
                if wst.pos != pos_before + 1:
                    raise Error("BARO_FORCE: step advanced by more than one position (spec/prefill batching) -- void arm")
                var fi = pos_before - (len(prompt) - 1)
                if fi < len(force):
                    ctx.enqueue_copy(dst_buf=force_tok_h, src_buf=DeviceBuffer[DType.int32](ctx, toks_d.unsafe_ptr().unsafe_offset(wst.pos), 1, owning=False))
                    ctx.synchronize()
                    var got = Int(force_tok_h[0])
                    predicted.append(got)
                    if margin:
                        var wf = 1 + N_SSM * 10 + N_ATT * 7 + N_LAYERS * 4
                        var Xm = TileTensor(bufs.x_d, xm_layout)
                        var CurBm = TileTensor(bufs.curb_d, xm_layout)
                        ctx.enqueue_function[rmsc_k](Xm, tens_f32(ctx, bufs.wbuf, bufs.off[wf], H, h_layout), CurBm, Int32(H), Float32(1e-6), grid_dim=1, block_dim=256)
                        var Pv = TileTensor(bufs.p_v_d, p_v)
                        gemm_w[VOCAB, H](ctx, CurBm, bufs.wbuf, bufs.off[wf + 1], bufs.kind[wf + 1], Pv, 1)
                        ctx.enqueue_function[r_head](Pv, TileTensor(bufs.logits_d, vm_layout), Int32(1), Int32(VOCAB), grid_dim=ceildiv(VOCAB, 256), block_dim=256)
                        ctx.enqueue_copy(dst_buf=lg_h, src_buf=DeviceBuffer[f32](ctx, bufs.logits_d.unsafe_ptr(), VOCAB, owning=False))
                        ctx.synchronize()
                        var i1 = 0
                        var i2 = -1
                        for i in range(1, VOCAB):
                            if lg_h[i] > lg_h[i1]:
                                i2 = i1
                                i1 = i
                            elif i2 < 0 or lg_h[i] > lg_h[i2]:
                                i2 = i
                        var refv = force[fi]
                        print("MARGIN pos", fi, " engine_argmax", got, " ref", refv, " top1", i1, lg_h[i1], " top2", i2, lg_h[i2], " gap", lg_h[i1] - lg_h[i2], " ref_logit", lg_h[refv])
                    force_tok_h[0] = Int32(force[fi])
                    ctx.enqueue_copy(dst_buf=DeviceBuffer[DType.int32](ctx, toks_d.unsafe_ptr().unsafe_offset(wst.pos), 1, owning=False), src_buf=force_tok_h)
                    ctx.synchronize()
            if ckpt_cap > 0 and not has_img and not bufs.steer_on and wst.pos > cached and wst.pos < len(prompt):
                if wst.pos == len(prompt) - 1 or wst.pos % CKPT_PERIOD == 0:
                    chain.save(ctx, bufs.convstate_d, bufs.sstate_d, wst.ring, wst.pos, prompt, False, False)
                else:
                    var hi = hint_index(ckpt_hints, wst.pos)
                    if hi >= 0:
                        chain.save(ctx, bufs.convstate_d, bufs.sstate_d, wst.ring, wst.pos, prompt, hi == 0 and chain.npages == 0, True)
            if not prefill_done and wst.pos >= len(prompt):
                ctx.synchronize()
                t_prefill_end = perf_counter_ns()
                prefill_done = True
            if serve and cancel_pending(0, req_id, pending):
                cancelled = True
                break
            if wst.pos >= len(prompt):
                tool_after_step(ctx, wst, bufs, one_h, grt)
            if wst.grammar_stop:
                # JSON-enforcement item 1: the matcher reached an accepting
                # state with no further legal byte -- same "stop" finish as
                # a stop sequence, no new wire value.
                stopped = True
                break
            if len(stop_seqs) > 0 and wst.pos >= len(prompt):
                ctx.synchronize()
                ctx.enqueue_copy(dst_buf=toks_h, src_buf=DeviceBuffer[DType.int32](ctx, toks_d.unsafe_ptr(), tmax, owning=False))
                ctx.synchronize()
                var gen_len = wst.pos + 1 - len(prompt)
                for si in range(len(stop_seqs)):
                    var seq = stop_seqs[si].copy()
                    var L = len(seq)
                    if L == 0 or L > gen_len:
                        continue
                    var ok = True
                    for k in range(L):
                        if Int(toks_h[wst.pos - L + 1 + k]) != seq[k]:
                            ok = False
                            break
                    if ok:
                        stopped = True
                        break
                if stopped:
                    break
        if not prefill_done:
            ctx.synchronize()
            t_prefill_end = perf_counter_ns()

        var t_host = Float64(perf_counter_ns() - t0) / 1e9
        ctx.synchronize()
        chain.commit()
        if latent_fd >= 0:
            var client = EngineLatentClient(latent_fd)
            var exported = 0
            for i in range(len(chain.items)):
                if chain.items[i].valid and chain.items[i].gen > latent_gen:
                    if not client.export_chain_slot(chain, i):
                        raise Error("latent export failed at slot " + String(i))
                    exported += 1
            latent_gen = chain.gen
            print("latent: exported", exported, "checkpoints, chain gen", latent_gen)
        if state_save != "":
            save_state(ctx, chain, bufs.kc_d, bufs.vc_d, bufs.kvtab_h, state_save, prompt, len(prompt) - 1)
        if req_state_save != "":
            var t_sv = perf_counter_ns()
            save_state(ctx, chain, bufs.kc_d, bufs.vc_d, bufs.kvtab_h, req_state_save, prompt, len(prompt) - 1)
            print("state saved:", req_state_save, " pos", len(prompt) - 1, " in", Float64(perf_counter_ns() - t_sv) / 1e9, "s")
        var dt = Float64(perf_counter_ns() - t0) / 1e9
        print("host_enqueue_s:", t_host, " gpu_total_s:", dt)
        var flw = ctx.enqueue_create_host_buffer[DType.uint32](3)
        ctx.enqueue_copy(dst_buf=flw, src_buf=bufs.ctr_d)
        ctx.synchronize()
        print("mega fail word:", flw[2], "" if flw[2] == 0 else " NOT-RESIDENT: a grid barrier timed out, tokens after it are invalid")
        if wst.grammar.__bool__():
            # Gate 4 (bench/grammar-protocol.md): masked_draws must equal
            # accepted every time -- read on every grammar-constrained run,
            # not just when something looks wrong.
            print("grammar masked draws:", wst.grammar_masked_draws, " accepted:", wst.grammar_accepted, " terminated:", wst.grammar_stop, "" if wst.grammar_masked_draws == wst.grammar_accepted else " MISMATCH: masked draw and accept counts desynced")
        # P1 receipt from the device, not from the env echo: the grid-barrier
        # generation counter is written only by a persistent kernel's barriers
        # (the launch path never touches ctr_d), so gen = 0 means no persistent
        # kernel ran this process, and on the MoE profile gen / MOE_BARRIERS is
        # the number of tokens the MoE kernel produced.
        comptime if not MEGA_ALLOWED:
            var nbar = MOE_BARRIERS_DOT if bufs.moe_dot else MOE_BARRIERS
            print("mega barrier gen:", flw[1], " MoE kernel tokens:", Int(flw[1]) // nbar, "(", nbar, "barriers per token, grid", MOE_G, ")")
        else:
            print("mega barrier gen:", flw[1])
        if dump:
            ctx.synchronize()
            with open(dump_path, "w") as f:
                var dpp = bufs.dump_h.unsafe_ptr().unsafe_bitcast[UInt8]()
                f.write_bytes(Span[UInt8](unsafe_ptr=dpp, length=wst.n_dumped * 2 * N_LAYERS * H * 4))
            print("dumped", wst.n_dumped, "tokens x", N_LAYERS, "layers to", dump_path)
        if dump and getenv("BARO_DUMP_DIR", "") != "":
            # Same container the packer writes and tools/tdiff.mojo reads:
            # data.bin plus one "name f32 byte_offset n_elem" line per tensor.
            # Names match llama.cpp's own (attn_residual-N, attn_post_norm-N,
            # l_out-N, linear_attn_out-N) so a dump joins against
            # llama-eval-callback's LLAMA_DUMP_DIR output by name, with no
            # mapping table to drift.
            var ddir = getenv("BARO_DUMP_DIR", "")
            ctx.synchronize()
            var dpp = bufs.dump_h.unsafe_ptr()
            var names = List[String]()
            var slots = List[Int]()
            var lens = List[Int]()
            comptime if not MEGA_ALLOWED:
                if dump4 and (dump_layer + 1) % 4 == 0:
                    # attention layer: the SSM slots never fire, and slots 4-6
                    # carry the attention captures instead.
                    names.append(String("attn_residual-") + String(dump_layer)); slots.append(0); lens.append(H)
                    names.append(String("attn_post_norm-") + String(dump_layer)); slots.append(1); lens.append(H)
                    names.append(String("l_out-") + String(dump_layer)); slots.append(2); lens.append(H)
                    # Names are llama's, exactly. The first pass called our
                    # pre-gate Ao "attn_output" and the o_proj result
                    # "attn_oproj"; llama calls those attn_pregate and
                    # attn_output, so tdiff was joining unrelated tensors of
                    # different lengths and reported relL2 22.69 on a layer
                    # whose downstream was provably at floor.
                    names.append(String("attn_pregate-") + String(dump_layer)); slots.append(4); lens.append(ATT)
                    names.append(String("attn_output-") + String(dump_layer)); slots.append(6); lens.append(H)
                    names.append(String("Qcur_full-") + String(dump_layer)); slots.append(8); lens.append(QF)
                    names.append(String("Qcur_normed-") + String(dump_layer)); slots.append(12); lens.append(ATT)
                    names.append(String("Qcur-") + String(dump_layer)); slots.append(14); lens.append(ATT)
                    names.append(String("Kcur_normed-") + String(dump_layer)); slots.append(16); lens.append(KV)
                    names.append(String("Kcur-") + String(dump_layer)); slots.append(17); lens.append(KV)
                elif dump4:
                    # SSM layer sub-block captures, slot order set in window.mojo
                    names.append(String("attn_residual-") + String(dump_layer)); slots.append(0)
                    names.append(String("attn_post_norm-") + String(dump_layer)); slots.append(1); lens.append(H)
                    names.append(String("l_out-") + String(dump_layer)); slots.append(2); lens.append(H)
                    names.append(String("final_norm-") + String(dump_layer)); slots.append(3); lens.append(H)
                    names.append(String("ssm_gates-") + String(dump_layer)); slots.append(5); lens.append(H)
                    names.append(String("ssm_state_out-") + String(dump_layer)); slots.append(6); lens.append(H)
                    names.append(String("linear_attn_out-") + String(dump_layer)); slots.append(7); lens.append(H)
                    names.append(String("conv_out-") + String(dump_layer)); slots.append(8); lens.append(H)
                else:
                    for layer in range(N_LAYERS):
                        names.append(String("attn_residual-") + String(layer))
                        slots.append(2 * layer)
                        lens.append(H)
            var off = 0
            var idx = String("")
            with open(ddir + "/data.bin", "w") as fb:
                for i in range(len(names)):
                    var base = slots[i] * H
                    var ln = lens[i]
                    fb.write_bytes(
                        Span[UInt8](
                            unsafe_ptr=dpp.unsafe_offset(base).unsafe_bitcast[UInt8](),
                            length=ln * 4,
                        )
                    )
                    idx += names[i] + " f32 " + String(off) + " " + String(ln) + "\n"
                    off += ln * 4
            with open(ddir + "/index.txt", "w") as fi:
                fi.write(idx)
            print("dump dir:", ddir, len(names), "tensors")
        if pf5:
            var ph = ctx.enqueue_create_host_buffer[DType.int64](16 * N_LAYERS + 4)
            ctx.enqueue_copy(dst_buf=ph, src_buf=bufs.prof_d)
            var fl = ctx.enqueue_create_host_buffer[DType.uint32](3)
            ctx.enqueue_copy(dst_buf=fl, src_buf=bufs.ctr_d)
            ctx.synchronize()
            var sub_ssm = 0.0
            var sub_att = 0.0
            var ffn_us = 0.0
            for layer in range(N_LAYERS):
                var a = Float64(ph[16 * layer + 7] - ph[16 * layer]) / 100.0
                var b = Float64(ph[16 * layer + 11] - ph[16 * layer + 7]) / 100.0
                if is_attn(layer):
                    sub_att += a
                else:
                    sub_ssm += a
                ffn_us += b
            var head_us = Float64(ph[16 * N_LAYERS + 3] - ph[16 * N_LAYERS]) / 100.0
            print("mega profile (last token, us): ssm sub-blocks", sub_ssm, " attn sub-blocks", sub_att, " ffn", ffn_us, " head", head_us, " total", Float64(ph[16 * N_LAYERS + 3] - ph[0]) / 100.0, " fail", fl[2])
            # BARO_PROFILE=5 raw: the prof buffer verbatim, 16 slots per layer
            # plus 4 tail slots, s_sendmsg.rtn(131) REALTIME ticks at 100 MHz.
            # Printed as raw ticks AND as us relative to slot 0 of layer 0, so
            # a reader can re-derive every aggregate above instead of taking
            # it on trust.
            print("PROFRAW slots", 16 * N_LAYERS + 4, " hz 100000000  base", ph[0])
            for layer in range(N_LAYERS):
                var line = String("PROFRAW L") + String(layer) + (" attn" if is_attn(layer) else " ssm ")
                for i in range(16):
                    line += " " + String(ph[16 * layer + i])
                print(line)
            var tail = String("PROFRAW TAIL")
            for i in range(4):
                tail += " " + String(ph[16 * N_LAYERS + i])
            print(tail)
            # per-phase sums over the layers of each kind, in stamp order (us)
            var ssm_seq: List[Int] = [0, 12, 1, 2, 3, 4, 5, 6, 13, 14, 7]
            var att_seq: List[Int] = [0, 1, 2, 3, 4, 5, 7]
            var ffn_seq: List[Int] = [7, 8, 9, 10, 15, 11]
            var ssm_ph = Array[Float64, 10](fill=0.0)
            var att_ph = Array[Float64, 6](fill=0.0)
            var ffn_ph = Array[Float64, 5](fill=0.0)
            for layer in range(N_LAYERS):
                var b = 16 * layer
                if is_attn(layer):
                    for i in range(6):
                        att_ph[i] += Float64(ph[b + att_seq[i + 1]] - ph[b + att_seq[i]]) / 100.0
                else:
                    for i in range(10):
                        ssm_ph[i] += Float64(ph[b + ssm_seq[i + 1]] - ph[b + ssm_seq[i]]) / 100.0
                for i in range(5):
                    ffn_ph[i] += Float64(ph[b + ffn_seq[i + 1]] - ph[b + ffn_seq[i]]) / 100.0
            var ssm_s = String("mega phases ssm (24 layers, us, stamps 0>12>1>2>3>4>5>6>13>14>7):")
            for i in range(10):
                ssm_s += " " + String(Int(ssm_ph[i]))
            print(ssm_s)
            var att_s = String("mega phases attn (8 layers, us, stamps 0>1>2>3>4>5>7):")
            for i in range(6):
                att_s += " " + String(Int(att_ph[i]))
            print(att_s)
            var ffn_s = String("mega phases ffn (32 layers, us, stamps 7>8>9>10>15>11 = rms|gate_up|silu|stage|down):")
            for i in range(5):
                ffn_s += " " + String(Int(ffn_ph[i]))
            print(ffn_s)
            var head_s = String("mega phases head (us, stamps 0>1>2>3):")
            for i in range(3):
                head_s += " " + String(Int(Float64(ph[16 * N_LAYERS + i + 1] - ph[16 * N_LAYERS + i]) / 100.0))
            print(head_s)
            comptime if not MEGA_ALLOWED:
                # kernels/mega_moe.mojo stamps 0..12 per layer (attention layers
                # skip 6) and NPROF * N_LAYERS at the end; per-phase sums over
                # the layers of each kind, last token, us.
                var mssm_seq: List[Int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                var matt_seq: List[Int] = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
                var mssm = Array[Float64, 12](fill=0.0)
                var matt = Array[Float64, 11](fill=0.0)
                var mtot = 0.0
                for layer in range(N_LAYERS):
                    var b = 16 * layer
                    if is_attn(layer):
                        for i in range(11):
                            matt[i] += Float64(ph[b + matt_seq[i + 1]] - ph[b + matt_seq[i]]) / 100.0
                    else:
                        for i in range(12):
                            mssm[i] += Float64(ph[b + mssm_seq[i + 1]] - ph[b + mssm_seq[i]]) / 100.0
                    mtot += Float64(ph[b + 12] - ph[b]) / 100.0
                var ms = String("moe phases ssm (30 layers, us, stamps 0>1>2>3>4>5>6>7>8>9>10>11>12 = rms|proj|gates+conv|l2|delta|gated|ssm_out|rms|router|top8|gate_up|down):")
                for i in range(12):
                    ms += " " + String(Int(mssm[i]))
                print(ms)
                var ma = String("moe phases attn (10 layers, us, stamps 0>1>2>3>4>5>7>8>9>10>11>12 = rms|proj|heads|attn|gmul|out|rms|router|top8|gate_up|down):")
                for i in range(11):
                    ma += " " + String(Int(matt[i]))
                print(ma)
                print("moe kernel layers total us:", Int(mtot), " kernel span us:", Int(Float64(ph[16 * N_LAYERS] - ph[0]) / 100.0))
        if prof:
            var tot = Float64(wst.pf_att + wst.pf_ssm + wst.pf_ffn + wst.pf_head)
            print("profile: attn", Float64(wst.pf_att) / 1e9, Float64(wst.pf_att) / tot)
            print("profile: ssm", Float64(wst.pf_ssm) / 1e9, Float64(wst.pf_ssm) / tot)
            print("profile: ffn", Float64(wst.pf_ffn) / 1e9, Float64(wst.pf_ffn) / tot)
            print("profile: head", Float64(wst.pf_head) / 1e9, Float64(wst.pf_head) / tot)
            print("profile: mtp_proc", Float64(wst.pf_proc) / 1e9, " mtp_draft", Float64(wst.pf_draft) / 1e9)
        if pf3:
            var other = wst.pf_proc + wst.pf_draft - wst.p3[0] - wst.p3[1] - wst.p3[2]
            print(
                "profile3: layer", Float64(wst.p3[0]) / 1e9, " head", Float64(wst.p3[1]) / 1e9,
                " argmax", Float64(wst.p3[2]) / 1e9, " accept", Float64(wst.p3[3]) / 1e9,
                " other", Float64(other) / 1e9,
            )
            if wst.n_spec_windows > 0:
                var nw = Float64(wst.n_spec_windows)
                print(
                    "profile3_per_window_ms: layer", Float64(wst.p3[0]) / 1e6 / nw, " head", Float64(wst.p3[1]) / 1e6 / nw,
                    " argmax", Float64(wst.p3[2]) / 1e6 / nw, " accept", Float64(wst.p3[3]) / 1e6 / nw,
                    " other", Float64(other) / 1e6 / nw,
                )
        if pf4:
            var fnames = ["rmsnorm", "gemm_gate", "gemm_up", "swiglu", "gemm_down", "r_add"]
            var ft = Float64(wst.fc[0] + wst.fc[1] + wst.fc[2] + wst.fc[3] + wst.fc[4] + wst.fc[5])
            for i in range(6):
                print("ffn-kernel:", fnames[i], Float64(wst.fc[i]) / 1e9, Float64(wst.fc[i]) / ft)
        if pf2:
            var names = ["gemm4+reduce2", "rgates", "conv", "l2", "delta", "gated", "out_gemm+add", "-"]
            var st = Float64(wst.pc[0] + wst.pc[1] + wst.pc[2] + wst.pc[3] + wst.pc[4] + wst.pc[5] + wst.pc[6])
            for i in range(7):
                print("ssm-kernel:", names[i], Float64(wst.pc[i]) / 1e9, Float64(wst.pc[i]) / st)
            var anames = ["norm+q_gemm", "kv_gemm", "split+norm+rope+append", "attention", "gmul+out_gemm+add"]
            for i in range(5):
                print("att-kernel:", anames[i], Float64(wst.ac[i]) / 1e9)
        ctx.enqueue_copy(dst_buf=toks_h, src_buf=DeviceBuffer[DType.int32](ctx, toks_d.unsafe_ptr(), tmax, owning=False))
        ctx.synchronize()
        var generated = List[Int]()
        for i in range(len(prompt), wst.pos + 1):
            generated.append(Int(toks_h[i]))
        var prefill_s = Float64(t_prefill_end - t0) / 1e9
        var decode_s = dt - prefill_s
        var n_gen = len(generated)
        # tok/s_gen is the only number comparable to llama.cpp: it divides
        # n_gen - 1 by decode time alone, matching timings.predicted_per_second
        # (n_gen, not the requested gen_n, so an early stop/cancel reports the
        # rate over what actually ran). tok/s_total includes prefill and is
        # reported for completeness only -- it is not the engine's throughput
        # against any external baseline.
        print("tokens:", n_gen, " prefill_s:", prefill_s, " decode_s:", decode_s)
        print("tok/s_total:", Float64(n_gen) / dt, " tok/s_gen:", Float64(n_gen - 1) / decode_s if n_gen > 1 else 0.0)
        gtt_base = gtt_check("after_request", gtt_base)
        var line = String("")
        for i in range(len(generated)):
            line += String(generated[i]) + " "
        print("GENERATED:", line)
        # B4 stage 2b gate 3: the tier's own counters, per request. Printed
        # here rather than in window.mojo so the wiring patch stays small.
        if bufs.tier.active:
            bufs.tier.report(n_gen)
        # B4 stage 2: one copy out, after the request, not per layer. The file
        # is appended so a 20-prompt sweep is one trace; each block names the
        # prompt's own token count so the replay can refuse a truncated trace.
        if expert_trace:
            ctx.enqueue_copy(dst_buf=bufs.etrace_h, src_buf=bufs.etrace_d)
            ctx.synchronize()
            var tl = String("# trace n_gen=") + String(n_gen) + " layers=" + String(N_LAYERS) + " topk=" + String(TOPK_TRACE) + " gates=1 prompt_tokens=" + String(len(prompt)) + "\n"
            var rows = min(n_gen, TRACE_TOK)
            for tk in range(rows):
                for ly in range(N_LAYERS):
                    var base = (tk * N_LAYERS + ly) * 2 * TOPK_TRACE
                    var any_set = False
                    for e in range(TOPK_TRACE):
                        if bufs.etrace_h[base + e] >= 0:
                            any_set = True
                    if not any_set:
                        continue
                    tl += String(tk) + " " + String(ly)
                    for e in range(TOPK_TRACE):
                        tl += " " + String(bufs.etrace_h[base + e])
                    for e in range(TOPK_TRACE):  # routing weights of the same top-k, f32 stored as int32 bits
                        tl += " " + String(bitcast[DType.float32](bufs.etrace_h[base + TOPK_TRACE + e]))
                    tl += "\n"
            with open(expert_trace_path, "a") as tf:
                tf.write(tl)
            print("expert trace appended:", expert_trace_path, " rows", rows, "x", N_LAYERS)
        if len(force) > 0:
            var ps = String("")
            var agree = 0
            var checked = min(len(predicted), len(force))
            for i in range(len(predicted)):
                ps += String(predicted[i]) + " "
                if i < len(force) and predicted[i] == force[i]:
                    agree += 1
            print("predicted:", ps)
            print("forced agreement:", agree, "/", checked)
        if spec:
            print("mtp: drafted", wst.n_drafted, " accepted", wst.n_accepted, " k", kcfg)
        if serve:
            var finish = String("length")
            if errored:
                finish = String("error")
            elif cancelled:
                finish = String("cancelled")
            elif stopped:
                finish = String("stop")
            var done_line = String("{\"id\":") + String(req_id) + ",\"done\":true,\"n\":" + String(n_gen)
            done_line += ",\"prefill_s\":" + String(prefill_s) + ",\"decode_s\":" + String(decode_s)
            done_line += ",\"tok_s\":" + String(Float64(n_gen - 1) / decode_s if n_gen > 1 else 0.0)
            done_line += ",\"cached\":" + String(cached) + ",\"prefill_rows\":" + String(prefill_rows) + ",\"pf_rows\":" + String(pf_rows) + ",\"restore_s\":" + String(restore_s) + ",\"checkpoints\":" + String(chain.count_valid())
            done_line += ",\"load_s\":" + String(load_s) + ",\"load_read_s\":" + String(load_split[0]) + ",\"load_parse_s\":" + String(load_split[1]) + ",\"load_pin_s\":" + String(load_split[2]) + ",\"load_host_s\":" + String(load_split[3]) + ",\"load_h2d_s\":" + String(load_split[4]) + ",\"wall_s\":" + String(Float64(perf_counter_ns() - t_req) / 1e9)
            done_line += ",\"finish\":\"" + finish + "\""
            var dw = DoneWords()
            dw.fail = Int(flw[2])
            dw.path = plan.tag
            dw.load_tier = load_tier
            if cap_on:
                dw.captured = wst.pos  # capture appends every advanced position (replay_m is 1 under capture)
            dw.grammar = grammar_word(wst.grammar.__bool__(), wst.grammar_pending_think, wst.grammar_stop)
            if wst.grammar.__bool__():
                dw.gdraws = wst.grammar_masked_draws
                dw.gaccepted = wst.grammar_accepted
                dw.grejected = wst.grammar.value().rejected
            done_line += dw.json()
            if spec:
                done_line += ",\"drafted\":" + String(wst.n_drafted) + ",\"accepted\":" + String(wst.n_accepted) + ",\"k\":" + String(kcfg)
                comptime if IS_MOE:
                    done_line += ",\"draft_kind\":\"ngram\",\"windows\":" + String(wst.n_spec_windows) + ",\"verify_rows\":" + String(verify_rows)
                    done_line += ",\"draft_s\":" + String(Float64(ngram_ns) / 1e9) + ",\"verify_s\":" + String(Float64(verify_ns) / 1e9)
            print(done_line + "}")
            continue

        # briefs/2026-09-16-sampling-all-models-lane.md item 1: the draft-head
        # receipt below is dense-only (MEGA_ALLOWED gates it off for
        # qwen35moe), so it never captures a real logits row for the MoE
        # sampler gates. This dumps the last decode step's target row
        # (b.logits_d row 0, the same buffer argmax_k/sample_row_k read from
        # in window.mojo) on either profile. Off by default, unconditional
        # on the profile, no effect on any existing path.
        var dump_logits_path = getenv("BARO_DUMP_LOGITS", "")
        if dump_logits_path != "":
            var lg_h = ctx.enqueue_create_host_buffer[f32](VOCAB)
            ctx.enqueue_copy(
                dst_buf=lg_h,
                src_buf=DeviceBuffer[f32](ctx, bufs.logits_d.unsafe_ptr(), VOCAB, owning=False),
            )
            ctx.synchronize()
            with open(dump_logits_path, "w") as f:
                var p = lg_h.unsafe_ptr().unsafe_bitcast[UInt8]()
                f.write_bytes(Span[UInt8](unsafe_ptr=p, length=VOCAB * 4))
            print("dumped final logits row to", dump_logits_path)

        if not MEGA_ALLOWED or pack_kq:
            return

        # MTP (NextN) draft head receipt, blk.32: last generated token paired with
        # the last trunk hidden row, attended at position 0 (arm A: empty draft
        # KV, identical to the 2026-09-01 validation dump; arm B: draft KV holds
        # the run, so only arm A's DRAFT line is the receipt).
        var hn_last = DeviceBuffer[f32](ctx, bufs.hn_d.unsafe_ptr(), H, owning=False)
        blk32_forward(ctx, wbuf, off, e, 1, 0, n_total - 1, True, hn_last,
            bufs.x_d, bufs.curb_d, bufs.qf_d, bufs.q_d, bufs.k_d, bufs.v_d, bufs.gate_d, bufs.ao_d, bufs.resb_d, bufs.fgb_d, bufs.p_qf_d, bufs.p_kv_d, bufs.p_h_d,
            bufs.p_ffn_d, bufs.p_ffn2_d, bufs.p_v_d, bufs.logits_d, bufs.cc_d, bufs.de_d, bufs.hd_d, bufs.kc32_d, bufs.vc32_d, bufs.kvtab_d, toks_d, bufs.dtok_d,
            False, wst.p3, False, 0, pack_q4)
        var last_tok = generated[len(generated) - 1]
        ctx.synchronize()
        var draft_logits_h = ctx.enqueue_create_host_buffer[f32](VOCAB)
        ctx.enqueue_copy(
            dst_buf=draft_logits_h,
            src_buf=DeviceBuffer[f32](ctx, bufs.logits_d.unsafe_ptr(), VOCAB, owning=False),
        )
        # h_nextn is dumped too: tools/draft-ref.py's numpy reference needs the
        # EXACT f32 hidden state the GPU consumed, or a mismatch says nothing about
        # correctness -- it would just mean the two sides were fed different inputs.
        var hn_h = ctx.enqueue_create_host_buffer[f32](H)
        ctx.enqueue_copy(
            dst_buf=hn_h,
            src_buf=DeviceBuffer[f32](ctx, bufs.hn_d.unsafe_ptr(), H, owning=False),
        )
        var dtok1_h = ctx.enqueue_create_host_buffer[DType.int32](1)
        ctx.enqueue_copy(
            dst_buf=dtok1_h,
            src_buf=DeviceBuffer[DType.int32](ctx, bufs.dtok_d.unsafe_ptr(), 1, owning=False),
        )
        ctx.synchronize()

        with open(".work/draft-logits.bin", "w") as df:
            var dp = draft_logits_h.unsafe_ptr().unsafe_bitcast[UInt8]()
            var dspan = Span[UInt8](unsafe_ptr=dp, length=VOCAB * 4)
            df.write_bytes(dspan)

        with open(".work/draft-hn.bin", "w") as hf:
            var hp = hn_h.unsafe_ptr().unsafe_bitcast[UInt8]()
            var hspan = Span[UInt8](unsafe_ptr=hp, length=H * 4)
            hf.write_bytes(hspan)

        print("DRAFT: from_token", last_tok, "draft_argmax", Int(dtok1_h[0]))
        break
