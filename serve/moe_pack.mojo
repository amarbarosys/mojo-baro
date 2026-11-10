"""qwen35moe pack index: every tensor resolved BY NAME to its dtype, byte
offset and element count (never by its position in index.txt), plus the
semantic layer over that map -- given a layer, projection and expert id,
the byte offset and raw-quant superblock geometry of a row range. Read-only:
opens `.work/moe-w1/pack` built by the packing lane, never writes a pack.

Row-major convention for an expert-stacked tensor (ffn_*_exps.weight):
expert is the slowest-varying axis, so expert e occupies a contiguous
[out_dim, in_dim] block starting at e * out_dim * in_dim elements
(tools/moe-ref.py: "ffn_gate_exps [2048, 512, 256] arrives as (expert, out,
in)"). Verified against the real pack: resolved byte sizes for all 733
tensors sum exactly to the real pack.bin size, and every recorded offset
equals the running sum of the preceding tensors' resolved sizes
(bench/moe-loader-protocol.md).

Also sizes the profile-dependent host state (KV pool, SSM state slots) from
serve/model_qwen35moe.mojo, the qwen35moe model profile -- never a second
copy of its numbers. KVPAGE/CONV/NH_V/SSTATE below are fixed kernel-format
geometry (kernels/attn.mojo, kernels/ssm.mojo), identical for every profile;
they are restated here rather than imported so this module stays GPU-free
(kernels/attn.mojo and kernels/ssm.mojo pull in the GPU kernel toolchain).
"""
from std.collections import Dict
from std.math import ceildiv

import model_qwen35moe as P

comptime F32_BYTES = 4
comptime Q8_0_BLOCK = 32
comptime Q8_0_BYTES = 34
comptime QK_K = 256
comptime Q4_K_BYTES = 144
comptime Q6_K_BYTES = 210

# Routing facts, not host-buffer dimensions: not part of the model profile
# (serve/model_qwen35moe.mojo), stated once here (brief / plan doc, real
# pack: 256 experts of ffn_*_exps.weight per layer, 8 selected per token).
comptime N_EXP = P.N_EXP
comptime TOPK = P.TOPK

comptime KVPAGE = 128
comptime KVPAD = 0
comptime KVHSTR = KVPAGE * P.HD + KVPAD
comptime CONV = 8192
comptime NH_V = 32
comptime SSTATE = 128
comptime TMAX = 1088
comptime SM = 8


@fieldwise_init
struct TensorInfo(Copyable, Movable):
    var dtype: String
    var offset: Int
    var n_elem: Int


def tensor_byte_size(dtype: String, n_elem: Int) raises -> Int:
    if dtype == "f32":
        return n_elem * F32_BYTES
    elif dtype == "q8_0":
        return (n_elem // Q8_0_BLOCK) * Q8_0_BYTES
    elif dtype == "q4_k":
        return (n_elem // QK_K) * Q4_K_BYTES
    elif dtype == "q6_k":
        return (n_elem // QK_K) * Q6_K_BYTES
    elif dtype == "q8":
        return n_elem + (n_elem // 32) * 2
    else:
        raise Error("unknown moe pack dtype " + dtype)


def block_geometry(dtype: String) raises -> Tuple[Int, Int]:
    """(elements per superblock, bytes per superblock)."""
    if dtype == "f32":
        return (1, F32_BYTES)
    elif dtype == "q8_0":
        return (Q8_0_BLOCK, Q8_0_BYTES)
    elif dtype == "q4_k":
        return (QK_K, Q4_K_BYTES)
    elif dtype == "q6_k":
        return (QK_K, Q6_K_BYTES)
    else:
        raise Error("no row geometry for dtype " + dtype)


def parse_moe_index(path: String) raises -> Dict[String, TensorInfo]:
    var out = Dict[String, TensorInfo]()
    with open(path, "r") as f:
        for line in f.read().splitlines():
            var parts = line.split(" ")
            if len(parts) < 4:
                continue
            var name = String(parts[0])
            var dtype = String(parts[1])
            var offset = Int(parts[2])
            var n_elem = Int(parts[3])
            out[name] = TensorInfo(dtype, offset, n_elem)
    return out^


@fieldwise_init
struct ExpertLoc(Copyable, Movable):
    var byte_offset: Int
    var block_elems: Int
    var block_bytes: Int
    var row_blocks: Int
    var row_bytes: Int
    var n_rows: Int


def resolve_expert(
    tensors: Dict[String, TensorInfo], layer: Int, proj: String,
    is_shared: Bool, expert: Int, row_start: Int, row_count: Int,
    out_dim: Int, in_dim: Int,
) raises -> ExpertLoc:
    """proj is "gate", "up" or "down". Looks the tensor up BY NAME; the
    caller's layer/expert/projection/row-range never comes from position."""
    var suffix = "_shexp.weight" if is_shared else "_exps.weight"
    var name = "blk." + String(layer) + ".ffn_" + proj + suffix
    if name not in tensors:
        raise Error("unresolved tensor " + name)
    var info = tensors[name].copy()
    var geom = block_geometry(info.dtype)
    var block_elems = geom[0]
    var block_bytes = geom[1]
    if in_dim % block_elems != 0:
        raise Error(
            "row width " + String(in_dim) + " not a multiple of block "
            + String(block_elems) + " for " + name
        )
    var row_blocks = in_dim // block_elems
    var row_bytes = row_blocks * block_bytes
    var per_expert_bytes = out_dim * row_bytes
    var expert_base = info.offset + (0 if is_shared else expert * per_expert_bytes)
    var byte_offset = expert_base + row_start * row_bytes
    return ExpertLoc(byte_offset, block_elems, block_bytes, row_blocks, row_bytes, row_count)


def resolve_plain(tensors: Dict[String, TensorInfo], name: String) raises -> TensorInfo:
    """Router / norm / non-expert tensor: whole-tensor offset, by name."""
    if name not in tensors:
        raise Error("unresolved tensor " + name)
    return tensors[name].copy()


@fieldwise_init
struct ProfileHostState(Copyable, Movable):
    var kvpool: Int
    var kvpool1: Int
    var conv_slot: Int
    var ssm_slot: Int
    var slots: Int


def size_host_state(tmax: Int = TMAX) -> ProfileHostState:
    """KV pool and SSM state slots for the qwen35moe profile, read from
    serve/model_qwen35moe.mojo (H 2048, QF 8192, KV 512, 40 layers, 30 SSM,
    10 attention, HD 256, NKVH 2 -- P.N_ATT, P.NKVH, P.N_SSM below)."""
    var tpages = ceildiv(tmax, KVPAGE)
    var kvpool = tpages * P.N_ATT * P.NKVH * KVHSTR
    var kvpool1 = tpages * P.NKVH * KVHSTR
    var conv_slot = P.N_SSM * 3 * CONV
    var ssm_slot = P.N_SSM * NH_V * SSTATE * SSTATE
    var slots = SM + 1
    return ProfileHostState(kvpool, kvpool1, conv_slot, ssm_slot, slots)
