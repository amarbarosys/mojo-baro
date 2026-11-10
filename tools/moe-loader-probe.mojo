"""Standalone probe: resolves every tensor in the real qwen35moe pack
(.work/moe-w1/pack, built by the packing lane, read-only here) by name
against serve/moe_pack.mojo, sums the resolved byte sizes against the pack's
real byte size, and prints blk.0's router, shared-expert and expert-0
offsets plus the qwen35moe profile-sized host state. Every number is read
back from the pack or the profile, never assumed. bench/moe-loader-protocol.md.

usage: .work/moe-loader-probe [PACKDIR]   (default .work/moe-w1/pack)
"""
from std.sys import argv, exit

from moe_pack import (
    parse_moe_index, tensor_byte_size, resolve_expert, resolve_plain,
    size_host_state, N_EXP,
)
import model_qwen35moe as P


def file_size(path: String) raises -> Int:
    with open(path, "r") as f:
        return Int(f.seek(0, 2))


def main() raises:
    var args = argv()
    var packdir = String(args[1]) if len(args) > 1 else String(".work/moe-w1/pack")
    var index_path = packdir + "/index.txt"
    var pack_path = packdir + "/pack.bin"

    var tensors = parse_moe_index(index_path)
    print("tensors resolved:", len(tensors))

    var resolved_total = 0
    for entry in tensors.items():
        resolved_total += tensor_byte_size(entry.value.dtype, entry.value.n_elem)
    var real_total = file_size(pack_path)
    print("resolved bytes:", resolved_total, "pack bytes:", real_total)

    var fail = False
    if len(tensors) != 733:
        print("FAIL: tensor count", len(tensors), "!= 733")
        fail = True
    if resolved_total != real_total:
        print("FAIL: resolved size does not match pack.bin")
        fail = True

    # --- blk.0: router, shared expert, expert 0 of each projection --------
    var router = resolve_plain(tensors, "blk.0.ffn_gate_inp.weight")
    print("blk.0 router offset:", router.offset)

    for proj in ["gate", "up", "down"]:
        var p = String(proj)
        # routed and shared experts share the same FFN width (P.FFN = 512)
        var out_dim = P.H if p == "down" else P.FFN
        var in_dim = P.FFN if p == "down" else P.H

        var shared = resolve_expert(tensors, 0, p, True, 0, 0, out_dim, out_dim, in_dim)
        print("blk.0 shared", p, "offset:", shared.byte_offset)

        var e0 = resolve_expert(tensors, 0, p, False, 0, 0, out_dim, out_dim, in_dim)
        print("blk.0 expert0", p, "offset:", e0.byte_offset)

        var e_last = resolve_expert(tensors, 0, p, False, N_EXP - 1, 0, out_dim, out_dim, in_dim)
        var tname = "blk.0.ffn_" + p + "_exps.weight"
        var tinfo = resolve_plain(tensors, tname)
        var per_expert_bytes = out_dim * e_last.row_bytes
        var tensor_total = tensor_byte_size(tinfo.dtype, tinfo.n_elem)
        if e0.byte_offset != tinfo.offset:
            print("FAIL: expert0", p, "offset does not match tensor base")
            fail = True
        if e_last.byte_offset + per_expert_bytes != tinfo.offset + tensor_total:
            print("FAIL: expert", N_EXP - 1, p, "does not reach the tensor's end")
            fail = True

    # --- Q6_K down_exps layers (34, 38, 39 in this GGUF's UD quant mix) ----
    for layer in [34, 38, 39]:
        var name = "blk." + String(layer) + ".ffn_down_exps.weight"
        var info = resolve_plain(tensors, name)
        if info.dtype != "q6_k":
            print("FAIL:", name, "expected q6_k, got", info.dtype)
            fail = True
        var e0 = resolve_expert(tensors, layer, "down", False, 0, 0, P.H, P.H, P.FFN)
        if e0.byte_offset != info.offset:
            print("FAIL:", name, "expert0 offset mismatch")
            fail = True
        print("blk.", layer, "down_exps dtype:", info.dtype, "expert0 offset:", e0.byte_offset)

    # --- profile-sized host state, read from serve/model_qwen35moe.mojo ---
    var st = size_host_state()
    print(
        "profile H", P.H, "QF", P.QF, "KV", P.KV, "N_LAYERS", P.N_LAYERS,
        "N_SSM", P.N_SSM, "N_ATT", P.N_ATT, "HD", P.HD, "NKVH", P.NKVH,
    )
    print(
        "host state kvpool", st.kvpool, "kvpool1", st.kvpool1,
        "conv_slot", st.conv_slot, "ssm_slot", st.ssm_slot, "slots", st.slots,
    )

    if fail:
        print("MOE LOADER PROBE: FAIL")
        exit(1)
    print("MOE LOADER PROBE: PASS")
