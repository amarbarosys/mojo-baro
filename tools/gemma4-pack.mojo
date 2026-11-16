"""Build the gemma4 (Forge-1 E2B) engine pack from an F16 GGUF, with no copy of the weights.

The pack is a directory:
  pack.bin     symlink to the GGUF (the engine reads tensor bytes in place: 0 bytes on disk, was 9.26 GB as a copy)
  index.txt    one line per tensor: name dt file_offset nbytes (dt f16 or f32), every GGUF tensor exactly once
  profile.txt  one line per layer: L il kind hd has_kv kv_src slot ffn (kind S = sliding window, F = full)
  profile.mojo comptime shapes for serve/gemma4.mojo

Per-layer facts (spec: llama.cpp gemma4 hparams, docs/provenance/gemma4.md): layers past n_layer - shared_kv_layers
have no K/V and read the KV of layer n_kv_start - 2 (sliding) or - 1 (full); a slot counts the layers with own KV
inside the pool of that kind (S pool HD 256, F pool HD 512). Every tensor name, type and shape is checked against
the profile; the count must equal the file's tensor count.

Usage: .work/gemma4-pack /abs/MODEL.gguf OUTDIR [--tmax N]
Build: ./.venv/bin/mojo build tools/gemma4-pack.mojo -I serve -I tools -o .work/gemma4-pack
Check: tools/gemma4-pack-check.sh MODEL.gguf OUTDIR (gguf-py oracle: count, bytes, sha per tensor)
"""
from std.ffi import external_call
from std.sys import argv, exit, stderr
from gguf_reader import GGUFModel

comptime T_F32 = 0
comptime T_F16 = 1


def want(m: GGUFModel, name: String, gt: Int, d0: Int, d1: Int, mut seen: Int, mut nbytes: Int, mut out: String) raises:
    if name not in m.tensor_type:
        raise Error("missing tensor " + name)
    var dims = m.tensor_dims[name].copy()
    var t = m.tensor_type[name]
    var ok = t == gt and dims[0] == d0 and ((d1 == 0 and len(dims) == 1) or (d1 > 0 and len(dims) == 2 and dims[1] == d1))
    if not ok:
        raise Error(name + ": type " + String(t) + " dims " + String(dims[0]) + "x" + String(dims[1] if len(dims) > 1 else 0)
            + ", profile wants type " + String(gt) + " " + String(d0) + "x" + String(d1))
    var n = d0 * max(d1, 1) * (4 if gt == T_F32 else 2)
    out += name + " " + ("f32" if gt == T_F32 else "f16") + " " + String(m.data_start + m.tensor_off[name]) + " " + String(n) + "\n"
    seen += 1
    nbytes += n


def main() raises:
    var av = argv()
    if len(av) < 3 or not String(av[1]).startswith("/"):
        print("usage: gemma4-pack /abs/MODEL.gguf OUTDIR [--tmax N]", file=stderr)
        exit(2)
    var src = String(av[1])
    var outdir = String(av[2])
    var tmax = 16384
    if len(av) == 5 and String(av[3]) == "--tmax":
        tmax = Int(String(av[4]))
    var m = GGUFModel(src)
    if m.arch != "gemma4":
        raise Error("arch " + m.arch + " is not gemma4")
    var nl = m.akv_int("block_count", 0)
    var h = m.akv_int("embedding_length", 0)
    var nqh = m.akv_int("attention.head_count", 0)
    var hd_f = m.akv_int("attention.key_length", 0)
    var hd_s = m.akv_int("attention.key_length_swa", 0)
    var ple = m.akv_int("embedding_length_per_layer_input", 0)
    var win = m.akv_int("attention.sliding_window", 0)
    var pat = m.int_arrays["gemma4.attention.sliding_window_pattern"].copy()
    var ffn = m.int_arrays["gemma4.feed_forward_length"].copy()
    var vocab = m.tensor_dims["token_embd.weight"][1]
    if m.akv_int("attention.head_count_kv", 0) != 1 or m.akv_int("attention.value_length", 0) != hd_f or m.akv_int("attention.value_length_swa", 0) != hd_s:
        raise Error("profile: expects one KV head and K == V head dims")
    if m.akv_int("rope.dimension_count", 0) != hd_f or m.akv_int("rope.dimension_count_swa", 0) != hd_s:
        raise Error("profile: expects full rotary dims (n_rot == head dim)")
    if len(pat) != nl or len(ffn) != nl:
        raise Error("profile: per-layer arrays do not match block_count")
    var nkv = nl - m.akv_int("attention.shared_kv_layers", 0)
    if nkv < 2:
        raise Error("profile: n_kv_start < 2")

    var idx = String()
    var prof = String()
    var seen = 0
    var nbytes = 0
    var ns = 0
    var nf = 0
    var slot = List[Int]()
    var ffn_max = 0
    for il in range(nl):
        var swa = pat[il] == 1
        var hd = hd_s if swa else hd_f
        var kv = il < nkv
        var srcl = il if kv else nkv - (2 if swa else 1)
        if pat[srcl] != pat[il]:
            raise Error("profile: layer " + String(il) + " KV source " + String(srcl) + " has another kind")
        var s: Int
        if kv:
            s = ns if swa else nf
            if swa:
                ns += 1
            else:
                nf += 1
        else:
            s = slot[srcl]
        slot.append(s)
        ffn_max = max(ffn_max, ffn[il])
        prof += "L " + String(il) + " " + ("S" if swa else "F") + " " + String(hd) + " " + String(Int(kv)) + " " + String(srcl) + " " + String(s) + " " + String(ffn[il]) + "\n"
        var b = "blk." + String(il) + "."
        want(m, b + "attn_norm.weight", T_F32, h, 0, seen, nbytes, idx)
        want(m, b + "attn_q.weight", T_F16, h, nqh * hd, seen, nbytes, idx)
        want(m, b + "attn_q_norm.weight", T_F32, hd, 0, seen, nbytes, idx)
        if kv:
            want(m, b + "attn_k.weight", T_F16, h, hd, seen, nbytes, idx)
            want(m, b + "attn_v.weight", T_F16, h, hd, seen, nbytes, idx)
            want(m, b + "attn_k_norm.weight", T_F32, hd, 0, seen, nbytes, idx)
        want(m, b + "attn_output.weight", T_F16, nqh * hd, h, seen, nbytes, idx)
        want(m, b + "post_attention_norm.weight", T_F32, h, 0, seen, nbytes, idx)
        want(m, b + "ffn_norm.weight", T_F32, h, 0, seen, nbytes, idx)
        want(m, b + "ffn_gate.weight", T_F16, h, ffn[il], seen, nbytes, idx)
        want(m, b + "ffn_up.weight", T_F16, h, ffn[il], seen, nbytes, idx)
        want(m, b + "ffn_down.weight", T_F16, ffn[il], h, seen, nbytes, idx)
        want(m, b + "post_ffw_norm.weight", T_F32, h, 0, seen, nbytes, idx)
        want(m, b + "inp_gate.weight", T_F16, h, ple, seen, nbytes, idx)
        want(m, b + "proj.weight", T_F16, ple, h, seen, nbytes, idx)
        want(m, b + "post_norm.weight", T_F32, h, 0, seen, nbytes, idx)
        want(m, b + "layer_output_scale.weight", T_F32, 1, 0, seen, nbytes, idx)
    want(m, "token_embd.weight", T_F16, h, vocab, seen, nbytes, idx)
    want(m, "per_layer_token_embd.weight", T_F16, ple * nl, vocab, seen, nbytes, idx)
    want(m, "per_layer_model_proj.weight", T_F16, h, ple * nl, seen, nbytes, idx)
    want(m, "per_layer_proj_norm.weight", T_F32, ple, 0, seen, nbytes, idx)
    want(m, "output_norm.weight", T_F32, h, 0, seen, nbytes, idx)
    want(m, "rope_freqs.weight", T_F32, hd_f // 2, 0, seen, nbytes, idx)
    if seen != len(m.tensor_names):
        raise Error("accounted for " + String(seen) + " of " + String(len(m.tensor_names)) + " tensors")

    var link = outdir + "/pack.bin"
    if external_call["symlink", Int32](src.as_c_string_slice().unsafe_ptr(), link.as_c_string_slice().unsafe_ptr()) != 0:
        raise Error("symlink " + link + " failed (exists already?)")
    with open(outdir + "/index.txt", "w") as f:
        f.write(idx)
    with open(outdir + "/profile.txt", "w") as f:
        f.write(prof)
    var p = String("# generated by tools/gemma4-pack.mojo from ") + src + " -- do not hand-edit\n"
    p += "comptime H = " + String(h) + "\ncomptime VOCAB = " + String(vocab) + "\ncomptime NL = " + String(nl) + "\n"
    p += "comptime NQH = " + String(nqh) + "\ncomptime HD_S = " + String(hd_s) + "\ncomptime HD_F = " + String(hd_f) + "\n"
    p += "comptime WIN = " + String(win) + "\ncomptime PLE = " + String(ple) + "\ncomptime FFN_MAX = " + String(ffn_max) + "\n"
    p += "comptime NKV = " + String(nkv) + "\ncomptime NS_SLOTS = " + String(ns) + "\ncomptime NF_SLOTS = " + String(nf) + "\n"
    p += "comptime EPS = Float32(" + String(m.akv_float("attention.layer_norm_rms_epsilon", 1e-6)) + ")\n"
    p += "comptime BASE_F = Float32(" + String(m.akv_float("rope.freq_base", 0)) + ")\n"
    p += "comptime BASE_S = Float32(" + String(m.akv_float("rope.freq_base_swa", 0)) + ")\n"
    p += "comptime SOFTCAP = Float32(" + String(m.akv_float("final_logit_softcapping", 0)) + ")\n"
    p += "comptime TMAX = " + String(tmax) + "\n"
    with open(outdir + "/profile.mojo", "w") as f:
        f.write(p)
    print("gemma4-pack:", seen, "tensors,", nbytes, "bytes, layers", nl, "S slots", ns, "F slots", nf, "-> " + outdir)
