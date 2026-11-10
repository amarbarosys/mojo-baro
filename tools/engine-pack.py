#!/usr/bin/env python3
"""Build the engine weight pack from a GGUF: one flat binary + text index.

2D bf16 weights are stored TRANSPOSED (B-layout [in, out]) for the skinny
GEMM; f32 tensors (norms, conv, ssm scalars) as-is. Emits, per line:
  name dtype offset_bytes n_elem
in a fixed, engine-known order. The blk.32 NextN draft head is appended
after output.weight, so every trunk offset is unchanged by its presence.

Usage: tools/engine-pack.py MODEL.gguf OUTDIR [--q8|--q4|--q2b3|--tq1|--tq2] [--q4-draft]
       tools/engine-pack.py --identity OUTDIR

--identity: writes/refreshes OUTDIR/identity.json (pack_sha256 over pack.bin,
tokenizer_sha256 over OUTDIR/tokenizer.json when present, the identity block
P1's cross-node salt uses; P1-STATE-API.md CONTRACT 2). Run this AFTER
tokenizer.json lands in OUTDIR, not as part of packing itself: packing writes
pack.bin before the tokenizer export step adds tokenizer.json, so an
identity.json written inline at pack time would have no tokenizer to hash.
Cached by pack.bin's size and mtime; a later --identity call is a no-op
until pack.bin actually changes.

A source GGUF may hold its 2D weights as K-quants (Q4_K, Q6_K, ...) instead of
bf16: each such tensor is dequantised to f32 with gguf-py
(`gguf.quants.dequantize`), rounded to bf16 (round-to-nearest-even), then fed
through the same bf16 paths below (quantize/transpose) as a native-bf16
source. `token_embd.weight` stays row-major, untransposed, unquantized either
way.

--q8: every 2D bf16 weight except token_embd is stored int8 in weight-native
[out, in] layout followed by fp16 block scales [out, in/32], ggml q8_0
rounding (d = amax/127 in fp32, q = roundf(x * (1/d)), d stored as fp16).
Index dtype is "q8"; n_elem counts weights, the scales follow at
offset + n_elem bytes.

--q4-draft: after the existing order (trunk untouched, keeps whatever the
--q8 flag gave it), append output.weight a second time as ggml-exact Q4_0
(int4 nibbles [N, K/2], element order per quantize_row_q4_0_ref: qs[j] low
nibble = x[j], high nibble = x[j+16], j in 0..15 per 32-block; fp16 block-32
scale d = max/-8 where max is the SIGNED value at the largest-|x| position)
under index name "output.weight.q4draft", dtype "q4". This is the MTP draft
head's weight only; the draft path opts in via BARO_DRAFT_Q4, the trunk
lookup (name "output.weight") is unaffected since it matches the first,
unchanged, entry.

--q2b3 / --tq1 / --tq2: same tensor selection as --q8, ternary block codecs
instead. Each stores the block payload bytes verbatim in weight-native
[out, in/BLOCK * PAYLOAD] order followed by fp16 block scales [out, in/BLOCK]
(scale = fp16(amax), weights are exactly {-d, 0, +d}). Index dtypes and
geometry:
  q2b3  Q2_B3 (B3S fork, ggml type 43): 128 weights, 26 payload bytes,
        base-3 five trits per byte in the v2 chunk-aligned layout
  tq1   TQ1_0 (ggml type 34): 256 weights, 52 payload bytes (qs[48] + qh[4])
  tq2   TQ2_0 (ggml type 35): 256 weights, 64 payload bytes, 2 bits each
Quantizers are quantize_row_*_ref from ggml, vectorized; tools/b3s-check.py
proves them bit-equal to the C in tools/ternary-ref.c.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.quants import dequantize as gguf_dequantize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("ge", Path(__file__).parent / "gguf-extract.py")
ge = module_from_spec(spec)
spec.loader.exec_module(ge)

def is_attn(i):
    return (i + 1) % 4 == 0


def mtp_names(n_layers):
    base = f"blk.{n_layers}."
    core = ["attn_norm.weight", "attn_q.weight", "attn_k.weight",
            "attn_v.weight", "attn_q_norm.weight", "attn_k_norm.weight",
            "attn_output.weight", "post_attention_norm.weight",
            "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
            "nextn.eh_proj.weight", "nextn.enorm.weight",
            "nextn.hnorm.weight", "nextn.shared_head_norm.weight"]
    return [base + n for n in core]


def layer_names(i):
    base = f"blk.{i}."
    if is_attn(i):
        core = ["attn_norm.weight", "attn_q.weight", "attn_k.weight",
                "attn_v.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                "attn_output.weight"]
    else:
        core = ["attn_norm.weight", "attn_qkv.weight", "attn_gate.weight",
                "ssm_alpha.weight", "ssm_beta.weight", "ssm_conv1d.weight",
                "ssm_a", "ssm_dt.bias", "ssm_norm.weight", "ssm_out.weight"]
    ffn = ["post_attention_norm.weight", "ffn_gate.weight", "ffn_up.weight",
           "ffn_down.weight"]
    return [base + n for n in core + ffn]


ROWS = 8192


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x32):
    """Round-to-nearest-even, matching ggml's ggml_fp32_to_bf16."""
    u = x32.astype(np.float32).view(np.uint32)
    bias = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + bias) >> 16).astype(np.uint16)


def dequantize_kquant(f, data_start, toff, ttype, shape, r0=0):
    """A Q4_K/Q6_K (or any other gguf-py-known non-bf16/f32/f16) tensor,
    dequantised to bf16 raw bytes via gguf-py so it can pass through the
    same bf16 quantize/store paths below. With r0, shape is a row block
    starting at row r0 of a 2-D tensor."""
    qtype = GGMLQuantizationType(ttype)
    block, tsize = GGML_QUANT_SIZES[qtype]
    n_elem = int(np.prod(shape))
    assert n_elem % block == 0, (shape, block)
    f.seek(data_start + toff + r0 * (shape[-1] // block) * tsize)
    raw_q = np.frombuffer(f.read(n_elem // block * tsize), dtype=np.uint8)
    w32 = gguf_dequantize(raw_q, qtype).reshape(shape).astype(np.float32)
    return f32_to_bf16(w32).tobytes()


def quantize_q8_0(w16):
    x = bf16_to_f32(w16).reshape(w16.shape[0], -1, 32)
    amax = np.abs(x).max(axis=2)
    d = (amax / 127.0).astype(np.float32)
    idv = np.where(d != 0, np.float32(1.0) / d, np.float32(0)).astype(np.float32)
    x0 = (x * idv[:, :, None]).astype(np.float32)
    q = np.sign(x0) * np.floor(np.abs(x0) + np.float32(0.5))
    q = np.clip(q, -128, 127).astype(np.int8)
    return q.reshape(w16.shape[0], -1), d.astype(np.float16)


def quantize_q4_0(w16):
    """ggml quantize_row_q4_0_ref, vectorized. w16: [N, K] bf16, K % 32 == 0."""
    n_out, k = w16.shape
    x = bf16_to_f32(w16).reshape(n_out, k // 32, 32)
    amax_idx = np.abs(x).argmax(axis=2)
    signed_max = np.take_along_axis(x, amax_idx[:, :, None], axis=2)[:, :, 0]
    d = (signed_max / np.float32(-8.0)).astype(np.float32)
    idv = np.where(d != 0, np.float32(1.0) / d, np.float32(0)).astype(np.float32)
    x0 = x[:, :, 0:16] * idv[:, :, None]
    x1 = x[:, :, 16:32] * idv[:, :, None]
    xi0 = np.clip(np.trunc(x0 + np.float32(8.5)), 0, 15).astype(np.uint8)
    xi1 = np.clip(np.trunc(x1 + np.float32(8.5)), 0, 15).astype(np.uint8)
    qs = (xi0 | (xi1 << np.uint8(4))).astype(np.uint8)
    return qs.reshape(n_out, -1), d.astype(np.float16)


def _ternary_codes(w16, block):
    """ggml ternary rounding shared by q2_b3 / tq1_0 / tq2_0: per block of
    `block` weights, d = amax, id = 1/amax (fp32), code = roundf(x*id) + 1 in
    {0,1,2} (half away from zero, as roundf/lroundf). Returns codes
    [N, nb, block] uint8 and scales [N, nb] fp16."""
    n_out, k = w16.shape
    assert k % block == 0, (k, block)
    x = bf16_to_f32(w16).reshape(n_out, k // block, block)
    amax = np.abs(x).max(axis=2).astype(np.float32)
    with np.errstate(divide="ignore"):
        idv = np.where(amax > 0, np.float32(1.0) / amax, np.float32(0)).astype(np.float32)
    p = (x * idv[:, :, None]).astype(np.float32)
    q = np.rint(p)
    tie = np.abs(p) == np.float32(0.5)
    q[tie] = np.sign(p[tie])
    q = np.clip(q.astype(np.int32) + 1, 0, 2).astype(np.uint8)
    return q, amax.astype(np.float16)


def _q2b3_pack_matrix():
    P = np.zeros((128, 26), dtype=np.int32)
    pw3 = [1, 3, 9, 27, 81]
    for j in range(128):
        c, t = j >> 5, j & 31
        byte = 6 * c + t // 5 if t < 30 else 24 + (c >> 1)
        digit = t % 5 if t < 30 else 2 * (c & 1) + (t - 30)
        P[j, byte] = pw3[digit]
    return P


def quantize_q2_b3(w16):
    """B3S quantize_row_q2_b3_ref, vectorized. w16: [N, K] bf16, K % 128 == 0.
    Returns payload [N, K/128 * 26] uint8 and scales [N, K/128] fp16."""
    q, d = _ternary_codes(w16, 128)
    qs = (q.astype(np.int32) @ _q2b3_pack_matrix()).astype(np.uint8)
    return qs.reshape(w16.shape[0], -1), d


def _tq1_pack_matrix():
    P = np.zeros((256, 52), dtype=np.int32)
    for m in range(32):
        for n in range(5):
            P[m + n * 32, m] = 3 ** (4 - n)
    for m in range(16):
        for n in range(5):
            P[160 + m + n * 16, 32 + m] = 3 ** (4 - n)
    for j in range(4):
        for m in range(4):
            P[240 + j + m * 4, 48 + j] = 3 ** (4 - m)
    return P


def quantize_tq1_0(w16):
    """ggml quantize_row_tq1_0_ref, vectorized. w16: [N, K] bf16, K % 256 == 0.
    Returns payload [N, K/256 * 52] uint8 (qs[48] then qh[4] per block, the
    ceiling-scaled base-3 bytes) and scales [N, K/256] fp16."""
    q, d = _ternary_codes(w16, 256)
    v = q.astype(np.int32) @ _tq1_pack_matrix()
    qs = ((v * 256 + 242) // 243).astype(np.uint8)
    return qs.reshape(w16.shape[0], -1), d


def _tq2_pack_matrix():
    P = np.zeros((256, 64), dtype=np.int32)
    for j in (0, 32):
        for l in range(4):
            for m in range(32):
                P[j * 4 + l * 32 + m, j + m] = 1 << (2 * l)
    return P


def quantize_tq2_0(w16):
    """ggml quantize_row_tq2_0_ref, vectorized. w16: [N, K] bf16, K % 256 == 0.
    Returns payload [N, K/256 * 64] uint8 (2 bits per weight) and scales
    [N, K/256] fp16."""
    q, d = _ternary_codes(w16, 256)
    qs = (q.astype(np.int32) @ _tq2_pack_matrix()).astype(np.uint8)
    return qs.reshape(w16.shape[0], -1), d


TERNARY = {"q2b3": quantize_q2_b3, "tq1": quantize_tq1_0, "tq2": quantize_tq2_0}


def _read_bf16(f, data_start, infos, name):
    """Any 2D weight tensor as raw bf16 bytes + its (out, in) shape, regardless
    of source dtype (bf16/f16/f32 direct, or a K-quant dequantised via gguf-py)."""
    dims, ttype, toff = infos[name]
    shape = list(reversed(dims))
    if ttype not in ge.GGML_BYTES:
        return dequantize_kquant(f, data_start, toff, ttype, shape), shape
    tname, esize = ge.GGML_BYTES[ttype]
    n_elem = int(np.prod(dims))
    f.seek(data_start + toff)
    raw = f.read(n_elem * esize)
    if tname == "bf16":
        return raw, shape
    if tname == "f16":
        w32 = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
        return f32_to_bf16(w32).tobytes(), shape
    if tname == "f32":
        w32 = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        return f32_to_bf16(w32).tobytes(), shape
    raise SystemExit(f"{name}: unsupported dtype {tname}")


def _read_f32(f, data_start, infos, name):
    """Any tensor as raw f32 bytes at full available precision (a K-quant
    source goes through gguf-py's dequantizer directly, not a bf16 round trip,
    so token_embd keeps native precision the way tools/spark-pack.py does)."""
    dims, ttype, toff = infos[name]
    shape = list(reversed(dims))
    n_elem = int(np.prod(dims))
    if ttype in ge.GGML_BYTES:
        tname, esize = ge.GGML_BYTES[ttype]
        f.seek(data_start + toff)
        raw = f.read(n_elem * esize)
        if tname == "f32":
            return raw
        if tname == "bf16":
            return bf16_to_f32(np.frombuffer(raw, dtype=np.uint16)).astype(np.float32).tobytes()
        if tname == "f16":
            return np.frombuffer(raw, dtype=np.float16).astype(np.float32).tobytes()
        raise SystemExit(f"{name}: unsupported dtype {tname}")
    qtype = GGMLQuantizationType(ttype)
    block, tsize = GGML_QUANT_SIZES[qtype]
    assert n_elem % block == 0, (name, shape, block)
    f.seek(data_start + toff)
    raw_q = np.frombuffer(f.read(n_elem // block * tsize), dtype=np.uint8)
    return gguf_dequantize(raw_q, qtype).reshape(shape).astype(np.float32).tobytes()


KQ_NATIVE = {GGMLQuantizationType.Q4_K: "q4k", GGMLQuantizationType.Q6_K: "q6k"}


def _read_kq(f, data_start, infos, names):
    """Q4_K / Q6_K parts as their own gguf blocks, concatenated along the output axis (serve/spark.mojo's
    amar_gemv_kq reads them natively, no bf16 round, no q8 requantise). Returns (raw, dt, n_elem), or None
    when a part is another type or the part types are not one A-rows-then-B-rows split; dt is "q4k", "q6k"
    or "A/B/<elements of the A rows>"."""
    types = [GGMLQuantizationType(infos[n][1]) for n in names]
    if any(t not in KQ_NATIVE for t in types) or sum(a != b for a, b in zip(types, types[1:])) > 1:
        return None
    raws, elems = [], []
    for n, t in zip(names, types):
        dims, _, toff = infos[n]
        block, tsize = GGML_QUANT_SIZES[t]
        n_elem = int(np.prod(dims))
        f.seek(data_start + toff)
        raws.append(f.read(n_elem // block * tsize))
        elems.append(n_elem)
    dt = KQ_NATIVE[types[0]]
    if types[0] != types[-1]:
        dt += f"/{KQ_NATIVE[types[-1]]}/{sum(e for t, e in zip(types, elems) if t == types[0])}"
    return b"".join(raws), dt, sum(elems)


def pack_sha256(pack_bin):
    h = hashlib.sha256()
    with open(pack_bin, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def write_identity(outdir, model=None):
    """Writes <outdir>/identity.json: pack_sha256 (pack.bin, full-file hash)
    plus tokenizer_sha256 (tokenizer.json file bytes, the SAME algorithm
    serve/src/checkpoints.rs already uses for its tokenizer_sha field) so
    the Mojo engine and the Rust front agree on one salt without either side
    re-deriving a hash the other computed differently (P1-STATE-API.md
    CONTRACT 2, coordinator amendment 2026-09-17, 0c68cb4).

    Called two ways that see different files: at pack time (this function,
    right after pack.bin is written, `model` given) tokenizer.json does not
    exist yet in the real pipeline (the tokenizer export step runs after packing) so tokenizer_sha256 is null; `--identity
    PACKDIR` (`ensure_identity`, `model=None`) runs later once it does. A
    prior identity.json's `source`/`general_uuid`/`tokenizer_sha256` are
    kept when this call cannot recompute them, so either order converges on
    a complete file.
    """
    pack_bin = outdir / "pack.bin"
    stat = pack_bin.stat()
    tok_path = outdir / "tokenizer.json"
    tokenizer_sha256 = hashlib.sha256(tok_path.read_bytes()).hexdigest() if tok_path.exists() else None
    id_path = outdir / "identity.json"
    prior = {}
    if id_path.exists():
        try:
            prior = json.loads(id_path.read_text())
        except (OSError, json.JSONDecodeError):
            prior = {}
    identity = {
        "pack_sha256": pack_sha256(pack_bin),
        "pack_bytes": stat.st_size,
        "pack_mtime_ns": stat.st_mtime_ns,
        "tokenizer_sha256": tokenizer_sha256 if tokenizer_sha256 is not None else prior.get("tokenizer_sha256"),
        "source": str(model) if model is not None else prior.get("source"),
        "general_uuid": prior.get("general_uuid"),
    }
    id_path.write_text(json.dumps(identity, indent=2) + "\n")
    tok_head = identity["tokenizer_sha256"][:16] if identity["tokenizer_sha256"] else None
    print(f"identity: pack_sha256={identity['pack_sha256'][:16]}... tokenizer_sha256={tok_head}")
    return identity


def ensure_identity(outdir):
    """`write_identity`, skipped when `outdir/identity.json` already matches
    the current `pack.bin` size and mtime AND already has a tokenizer_sha256
    if `tokenizer.json` exists (coordinator: cache by pack_bytes and
    pack_mtime_ns, a full pack_sha256 re-hash is not free)."""
    pack_bin = outdir / "pack.bin"
    if not pack_bin.exists():
        raise SystemExit(f"engine-pack --identity: no pack.bin in {outdir}")
    id_path = outdir / "identity.json"
    if id_path.exists():
        stat = pack_bin.stat()
        try:
            existing = json.loads(id_path.read_text())
        except (OSError, json.JSONDecodeError):
            existing = {}
        tok_path = outdir / "tokenizer.json"
        tok_covered = existing.get("tokenizer_sha256") is not None or not tok_path.exists()
        if existing.get("pack_bytes") == stat.st_size and existing.get("pack_mtime_ns") == stat.st_mtime_ns and tok_covered:
            print(f"identity: cached (pack unchanged), pack_sha256={existing['pack_sha256'][:16]}...")
            return existing
    return write_identity(outdir)


def pack_dense(model, outdir):
    """Pack a plain dense transformer (llama/qwen2/granite arch: separate Q/K/V
    weights, no per-head output gate) into the pack format serve/spark.mojo
    reads for a QKV_BIAS/HAS_GATE=False profile: attn_norm f32, attn_qkv q8
    (Q/K/V weights concatenated along the output axis and quantised together,
    ggml-style so the fused GEMM's row order is [Q rows][K rows][V rows]),
    optional attn_qkv.bias f32 (concatenated the same way), attn_output q8,
    ffn_norm f32, ffn_gate/ffn_up/ffn_down q8. token_embd is exact f32 (no
    bf16 rounding, matching tools/spark-pack.py); output.weight is the tied
    token_embd re-quantised to q8 when the GGUF has no separate output tensor
    (tools/gen-profile.mojo's TIE_EMBED), else that tensor quantised directly.
    Q4_K / Q6_K matrices (a Q4_K_M file) are written as their own gguf blocks, dt q4k / q6k, or
    "q4k/q6k/<A elements>" for a fused QKV whose V rows are Q6_K (_read_kq); other source types
    take the bf16 round and q8 requantise.

    Spark2_5's per-head attention gate is emitted as q8 between attn_qkv and
    attn_output when present, matching the order tools/spark-pack.py writes.
    """
    f, infos, data_start, kv = ge.parse(model)
    arch = kv["general.architecture"]
    n_layers = kv[f"{arch}.block_count"]
    has_bias = "blk.0.attn_q.bias" in infos
    has_gate = "blk.0.attn_gate.weight" in infos
    tied = "output.weight" not in infos

    outdir.mkdir(parents=True, exist_ok=True)
    idx_lines = []
    off = 0
    with open(outdir / "pack.bin", "wb") as out:
        def emit(name, raw, dt, n_elem):
            nonlocal off
            out.write(raw)
            idx_lines.append(f"{name} {dt} {off} {n_elem}")
            off += len(raw)

        emit("token_embd.weight", _read_f32(f, data_start, infos, "token_embd.weight"), "f32",
             int(np.prod(infos["token_embd.weight"][0])))

        def emit_mat(name, parts):
            kq = _read_kq(f, data_start, infos, parts)
            if kq is not None:
                emit(name, *kq)
                return
            ws = [_read_bf16(f, data_start, infos, p) for p in parts]
            w = np.concatenate([np.frombuffer(raw, dtype=np.uint16).reshape(sh) for raw, sh in ws], axis=0)
            q, d = quantize_q8_0(w)
            emit(name, q.tobytes() + d.tobytes(), "q8", w.size)

        for i in range(n_layers):
            b = f"blk.{i}."
            emit(b + "attn_norm.weight", _read_f32(f, data_start, infos, b + "attn_norm.weight"),
                 "f32", int(np.prod(infos[b + "attn_norm.weight"][0])))

            if b + "attn_qkv.weight" in infos:
                emit_mat(b + "attn_qkv.weight", [b + "attn_qkv.weight"])
            else:
                emit_mat(b + "attn_qkv.weight", [b + "attn_q.weight", b + "attn_k.weight", b + "attn_v.weight"])

            if has_bias:
                qb = np.frombuffer(_read_f32(f, data_start, infos, b + "attn_q.bias"), dtype=np.float32)
                kb = np.frombuffer(_read_f32(f, data_start, infos, b + "attn_k.bias"), dtype=np.float32)
                vb = np.frombuffer(_read_f32(f, data_start, infos, b + "attn_v.bias"), dtype=np.float32)
                bias = np.concatenate([qb, kb, vb])
                emit(b + "attn_qkv.bias", bias.astype(np.float32).tobytes(), "f32", bias.size)

            if has_gate:
                emit_mat(b + "attn_gate.weight", [b + "attn_gate.weight"])
            emit_mat(b + "attn_output.weight", [b + "attn_output.weight"])

            emit(b + "ffn_norm.weight", _read_f32(f, data_start, infos, b + "ffn_norm.weight"),
                 "f32", int(np.prod(infos[b + "ffn_norm.weight"][0])))

            for part in ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"):
                emit_mat(b + part, [b + part])

        emit("output_norm.weight", _read_f32(f, data_start, infos, "output_norm.weight"),
             "f32", int(np.prod(infos["output_norm.weight"][0])))

        emit_mat("output.weight", ["token_embd.weight" if tied else "output.weight"])

    (outdir / "index.txt").write_text("\n".join(idx_lines) + "\n")
    print(f"packed {arch}, {n_layers} layers, bias={has_bias}, gate={has_gate}, tied={tied}, {off/2**30:.2f} GiB")
    write_identity(outdir, model=model)


MOE_DENSE_Q8 = {"attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv", "attn_gate", "ssm_out"}


def pack_moe(model, outdir):
    """Pack qwen35moe in fixed lexical tensor order without expanding experts."""
    f, infos, data_start, kv = ge.parse(model)
    arch = kv["general.architecture"]
    if arch != "qwen35moe":
        raise SystemExit(f"--arch qwen35moe requires qwen35moe GGUF, got {arch}")
    outdir.mkdir(parents=True, exist_ok=True)
    names = sorted(infos)
    idx_lines = []
    off = 0
    with open(outdir / "pack.bin", "wb") as out:
        for name in names:
            dims, ttype, toff = infos[name]
            n_elem = int(np.prod(dims))
            if ttype in ge.GGML_BYTES:
                tname, esize = ge.GGML_BYTES[ttype]
                f.seek(data_start + toff)
                raw = f.read(n_elem * esize)
                dt = tname
            else:
                qtype = GGMLQuantizationType(ttype)
                block, tsize = GGML_QUANT_SIZES[qtype]
                f.seek(data_start + toff)
                raw = f.read((n_elem // block) * tsize)
                dt = qtype.name.lower()
            if name == "output.weight":
                shape = list(reversed(dims))
                raw_bf16 = dequantize_kquant(f, data_start, toff, ttype, shape)
                q, d = quantize_q8_0(np.frombuffer(raw_bf16, dtype=np.uint16).reshape(shape))
                raw = q.tobytes() + d.tobytes()
                dt = "q8"
            elif dt == "q8_0" and name.split(".")[-2] in MOE_DENSE_Q8:
                # R6a (bench/moe-persist-protocol.md): projections in the dense q8
                # layout the megakernel phases read, a byte split of the q8_0
                # blocks (32 int8 + f16 scale each), every value bit-equal.
                blk = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
                raw = blk[:, 2:].tobytes() + blk[:, :2].tobytes()
                dt = "q8"
            out.write(raw)
            idx_lines.append(f"{name} {dt} {off} {n_elem}")
            off += len(raw)
    (outdir / "index.txt").write_text("\n".join(idx_lines) + "\n")
    counts = {}
    for line in idx_lines:
        dt = line.split()[1]
        counts[dt] = counts.get(dt, 0) + 1
    print(f"packed {len(names)} tensors, {off/2**30:.2f} GiB, dtypes={counts}")
    write_identity(outdir, model=model)


def main():
    if "--identity" in sys.argv:
        i = sys.argv.index("--identity")
        outdir = Path(sys.argv[i + 1])
        del sys.argv[i:i + 2]
        ensure_identity(outdir)
        return
    if "--arch" in sys.argv:
        i = sys.argv.index("--arch")
        arch = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        if arch == "qwen35moe":
            pack_moe(Path(sys.argv[1]), Path(sys.argv[2]))
            return
        raise SystemExit(f"unsupported architecture: {arch}")
    if "--dense" in sys.argv:
        sys.argv.remove("--dense")
        pack_dense(Path(sys.argv[1]), Path(sys.argv[2]))
        return
    q8 = "--q8" in sys.argv
    if q8:
        sys.argv.remove("--q8")
    q4 = "--q4" in sys.argv
    if q4:
        sys.argv.remove("--q4")
    assert not (q8 and q4), "--q8 and --q4 are exclusive"
    ternary = None
    for flag in TERNARY:
        if "--" + flag in sys.argv:
            sys.argv.remove("--" + flag)
            ternary = flag
    assert not ((q8 or q4) and ternary), "--q8/--q4 and a ternary flag are exclusive"
    q4_draft = "--q4-draft" in sys.argv
    if q4_draft:
        sys.argv.remove("--q4-draft")
    model, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    f, infos, data_start, kv = ge.parse(model)

    # Trunk depth from the GGUF: block_count minus the MTP draft block(s).
    n_layers = kv["qwen35.block_count"] - kv.get("qwen35.nextn_predict_layers", 0)
    order = ["token_embd.weight"]
    for i in range(n_layers):
        order += layer_names(i)
    order += ["output_norm.weight", "output.weight"]
    order += mtp_names(n_layers)

    idx_lines = []
    off = 0
    with open(outdir / "pack.bin", "wb") as out:
        for name in order:
            # Tied embeddings (Qwen3.5-4B and smaller): no output.weight, the head is token_embd.
            tied = name == "output.weight" and name not in infos
            dims, ttype, toff = infos["token_embd.weight" if tied else name]
            n_elem = int(np.prod(dims))
            shape = list(reversed(dims))
            bf16_in = ttype not in ge.GGML_BYTES or ge.GGML_BYTES[ttype][0] == "bf16"
            if len(shape) == 2 and bf16_in and (q8 or q4 or name == "token_embd.weight"):
                # Row blocks: a whole 248k x 4096 table through f32 peaked at 20.7 GB.
                tname = "bf16" if name == "token_embd.weight" else "q8" if q8 else "q4"
                ds, n_out = [], 0
                for r0 in range(0, shape[0], ROWS):
                    rs = [min(ROWS, shape[0] - r0), shape[1]]
                    if ttype in ge.GGML_BYTES:
                        f.seek(data_start + toff + r0 * shape[1] * 2)
                        w = np.frombuffer(f.read(rs[0] * shape[1] * 2), dtype=np.uint16).reshape(rs)
                    else:
                        w = np.frombuffer(dequantize_kquant(f, data_start, toff, ttype, rs, r0), dtype=np.uint16).reshape(rs)
                    if tname == "bf16":
                        out.write(w.tobytes())
                        n_out += w.nbytes
                        continue
                    q, d = quantize_q8_0(w) if q8 else quantize_q4_0(w)
                    out.write(q.tobytes())
                    n_out += q.nbytes
                    ds.append(d)
                for d in ds:
                    out.write(d.tobytes())
                    n_out += d.nbytes
                idx_lines.append(f"{name} {tname} {off} {n_elem}")
                off += n_out
                continue
            if ttype in ge.GGML_BYTES:
                tname, esize = ge.GGML_BYTES[ttype]
                f.seek(data_start + toff)
                raw = f.read(n_elem * esize)
            else:
                tname = "bf16"
                raw = dequantize_kquant(f, data_start, toff, ttype, shape)
            if tname == "bf16" and len(shape) == 2 and name != "token_embd.weight":
                w = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
                if q8:
                    q, d = quantize_q8_0(w)
                    raw = q.tobytes() + d.tobytes()
                    tname = "q8"
                elif q4:
                    q, d = quantize_q4_0(w)
                    raw = q.tobytes() + d.tobytes()
                    tname = "q4"
                elif ternary:
                    q, d = TERNARY[ternary](w)
                    raw = q.tobytes() + d.tobytes()
                    tname = ternary
                else:
                    raw = np.ascontiguousarray(w.T).tobytes()
            out.write(raw)
            idx_lines.append(f"{name} {tname} {off} {n_elem}")
            off += len(raw)
        if q4_draft:
            dims, ttype, toff = infos["output.weight"]
            shape = list(reversed(dims))
            n_elem = int(np.prod(dims))
            if ttype in ge.GGML_BYTES:
                tname, esize = ge.GGML_BYTES[ttype]
                assert tname == "bf16", tname
                f.seek(data_start + toff)
                w = np.frombuffer(f.read(n_elem * esize), dtype=np.uint16).reshape(shape)
            else:
                w = np.frombuffer(
                    dequantize_kquant(f, data_start, toff, ttype, shape), dtype=np.uint16
                ).reshape(shape)
            q, d = quantize_q4_0(w)
            raw = q.tobytes() + d.tobytes()
            out.write(raw)
            idx_lines.append(f"output.weight.q4draft q4 {off} {n_elem}")
            off += len(raw)
    (outdir / "index.txt").write_text("\n".join(idx_lines) + "\n")
    print(f"packed {len(order)} tensors, {off/2**30:.2f} GiB")
    write_identity(outdir, model=model)


if __name__ == "__main__":
    main()
