"""Emit a comptime Mojo profile module from a GGUF's own metadata.

Every dense-family GGUF (llama, qwen2, granite, spark2_5) stores its recipe
under uniform `<arch>.*` keys (llama.cpp convention), so the dims and norm
eps are read directly; VOCAB is read off `token_embd.weight`'s own shape
(the tensor bytes are authority, not a metadata count that can drift).
Rope type (norm/neox) and activation are not stored in the GGUF -- llama.cpp
picks them per architecture in `llm_arch_rope_type` / the model's graph
builder, so they are looked up from a small per-arch table below, same as
any other recipe fact read out of llama.cpp source as spec.

Usage: tools/gen-profile.mojo MODEL.gguf OUT.mojo [--tmax N]
Build:  ./.venv/bin/mojo build tools/gen-profile.mojo -I tools -o .work/gen-profile
(tools/gguf_reader.mojo is a symlink to ~/iTools/lib/mojo/gguf-reader.mojo)
"""
from std.math import sqrt
from std.collections import Dict
from std.sys import argv
from gguf_reader import GGUFModel


def is_rope_neox_arch(arch: String) -> Bool:
    # llama_model_rope_type (llama-model.cpp): LLM_ARCH_QWEN2's group, "pairs
    # offset by n_rot/2". spark2_5 is NEOX by construction (kernels/spark_kernels.mojo
    # predates this table), not a llama.cpp fact.
    return arch == "spark2_5" or arch == "qwen2"


def is_rope_norm_arch(arch: String) -> Bool:
    # llama_model_rope_type (llama-model.cpp): LLM_ARCH_GRANITE is grouped with
    # LLM_ARCH_LLAMA under "normal RoPE, pairs of consecutive head values" --
    # NOT with LLM_ARCH_QWEN2's NeoX-style half-offset group. Measured:
    # granite-4.2-3b scored ~40/64 forced agreement vs llama.cpp with NEOX
    # assumed, 98.4-100% after moving it here (bench/dense-protocol.md).
    # Full table + citations: ~/Brain/mojo/mojo-baro/2026-09-11-llama-arch-recipe-facts.md
    return arch == "llama" or arch == "granite"


def is_gelu_arch(arch: String) -> Bool:
    # spark2_5 predates this table (kernels/spark_kernels.mojo's GELU tanh-approx epilogue
    # was hand-written, not read from llama.cpp -- it isn't in llama.cpp at all).
    return arch == "spark2_5"


def is_silu_arch(arch: String) -> Bool:
    # build_layer_ffn in llama.cpp's src/models/{llama,qwen2,granite}.cpp: build_ffn(...,
    # LLM_FFN_SILU, LLM_FFN_PAR, ...). ~/Brain/mojo/mojo-baro/2026-09-11-llama-arch-recipe-facts.md
    return arch == "llama" or arch == "qwen2" or arch == "granite"


def find_swa_period(pattern: List[Int]) raises -> Tuple[Int, Int]:
    """pattern[i] == 1 means layer i is windowed, 0 means full attention.
    Returns (period, full_phase) with swa(i) == (i % period != full_phase),
    or raises if no period in [1, 8] reproduces the array exactly."""
    var n = len(pattern)
    for period in range(1, 9):
        for full_phase in range(period):
            var ok = True
            for i in range(n):
                if (pattern[i] == 1) != ((i % period) != full_phase):
                    ok = False
                    break
            if ok:
                return (period, full_phase)
    raise Error("sliding_window_pattern is not periodic in [1,8]; gen-profile only supports a period/phase SWA rule")


comptime SUPPORTED_OPS = "GET_ROWS RMS_NORM MUL MUL_MAT ADD RESHAPE VIEW PERMUTE CONT TRANSPOSE CPY SET_ROWS ROPE FLASH_ATTN_EXT SOFT_MAX SWIGLU GEGLU SILU GELU SCALE"
comptime UNSUPPORTED_NAMES = "post_norm q_norm k_norm Qcur_normed Kcur_normed"


def read_recipe(path: String) raises -> Tuple[Dict[String, String], Bool]:
    """Ops of a llama-eval-callback log: {op: first tensor name}, and False after printing FAIL coverage lines."""
    var ops = Dict[String, String]()
    var names = Dict[String, String]()
    var text: String
    with open(path, "r") as f:
        text = f.read()
    for line in text.splitlines():
        var l = String(line)
        if "common_debug_cb_eval:" not in l or " = (" not in l:
            continue
        var head = l.split(" = (", 1)
        var hp = String(head[0]).split(":")
        var name = String(hp[len(hp) - 1].strip())
        var rest = String(head[1]).split(")", 1)
        if len(rest) < 2:
            continue
        var op = String(String(String(rest[1]).strip()).split("(")[0])
        if op not in ops:
            ops[op] = name
        var stem = name
        var dash = name.rfind("-")
        if dash > 0:
            try:
                _ = atol(String(name[byte=dash + 1 :]))
                stem = String(name[byte=:dash])
            except:
                pass
        names[stem] = op
    var ok = True
    for e in ops.items():
        if (" " + e.key + " ") not in String(" " + SUPPORTED_OPS + " "):
            print("FAIL coverage: op " + e.key + " (" + e.value + ") has no spark template")
            ok = False
    for n in names.keys():
        for u in UNSUPPORTED_NAMES.split(" "):
            if u in n:
                print("FAIL coverage: tensor " + n + " (" + names[n] + ") needs a " + u + " template")
                ok = False
                break
    return (ops^, ok)


def mojo_bool(b: Bool) -> String:
    return "True" if b else "False"


def main() raises:
    var args = argv()
    if len(args) < 3:
        print("usage: gen-profile MODEL.gguf OUT.mojo [--tmax N] [--recipe EVALCB.log]")
        return
    var model_path = String(args[1])
    var out_path = String(args[2])
    var tmax = 4096
    var recipe = String("")
    for i in range(3, len(args)):
        if String(args[i]) == "--tmax" and i + 1 < len(args):
            tmax = atol(String(args[i + 1]))
        if String(args[i]) == "--recipe" and i + 1 < len(args):
            recipe = String(args[i + 1])

    var m = GGUFModel(model_path)
    var arch = m.arch
    var rops = Dict[String, String]()
    if recipe != "":
        var r = read_recipe(recipe)
        if "SCALE" in r[0] and arch != "granite":
            print("FAIL coverage: op SCALE (" + r[0]["SCALE"] + ") outside granite's metadata multipliers")
        if not r[1]:
            raise Error("coverage: " + arch + " needs ops spark does not implement (FAIL lines above)")
        rops = r[0].copy()
        if "SCALE" in rops and arch != "granite":
            raise Error("coverage: " + arch + " needs ops spark does not implement (FAIL lines above)")

    var h = m.akv_int("embedding_length", 0)
    var ffn = m.akv_int("feed_forward_length", 0)
    var n_layers = m.akv_int("block_count", 0)
    var nqh = m.akv_int("attention.head_count", 0)
    var nkvh = m.akv_int("attention.head_count_kv", 0)
    var hd = m.akv_int("attention.key_length", h // nqh)
    var norm_eps = m.akv_float("attention.layer_norm_rms_epsilon", 1e-6)
    var nrot_full = m.akv_int("rope.dimension_count", hd)
    var base_full = m.akv_float("rope.freq_base", 10000.0)

    var embd_dims = m.tensor_dims.get("token_embd.weight", List[Int]())
    if len(embd_dims) != 2:
        raise Error("token_embd.weight: expected 2 dims, got " + String(len(embd_dims)))
    var vocab = embd_dims[1]
    var tie_embed = not m.tensor_present("output.weight")
    var rope_freqs = m.tensor_present("rope_freqs.weight")
    var qkv_bias = m.tensor_present("blk.0.attn_q.bias")
    var has_gate = m.tensor_present("blk.0.attn_gate.weight")

    var swa_key = arch + ".attention.sliding_window_pattern"
    var swa_win: Int
    var nrot_swa: Int
    var base_swa: Float64
    var swa_period: Int
    var swa_full_phase: Int
    if len(m.int_arrays.get(swa_key, List[Int]())) > 0:
        var pattern = m.int_arrays[swa_key].copy()
        swa_win = m.akv_int("attention.sliding_window", 0)
        nrot_swa = m.akv_int("rope.dimension_count_swa", nrot_full)
        base_swa = m.akv_float("rope.freq_base_swa", base_full)
        var pf = find_swa_period(pattern)
        swa_period = pf[0]
        swa_full_phase = pf[1]
    else:
        swa_win = 0
        nrot_swa = nrot_full
        base_swa = base_full
        swa_period = 1
        swa_full_phase = 0

    var rope_neox: Bool
    if is_rope_neox_arch(arch):
        rope_neox = True
    elif is_rope_norm_arch(arch):
        rope_neox = False
    else:
        raise Error("unknown architecture " + arch + ": add it to is_rope_neox_arch or is_rope_norm_arch in tools/gen-profile.mojo after reading llama.cpp's llm_arch_rope_type for it")

    var activation_gelu: Bool
    var table_act = is_gelu_arch(arch) or is_silu_arch(arch)
    if recipe != "":
        var gelu = "GEGLU" in rops or "GELU" in rops
        var silu = "SWIGLU" in rops or "SILU" in rops
        if gelu == silu:
            raise Error("recipe: activation unknown (gelu " + String(gelu) + ", silu " + String(silu) + ")")
        activation_gelu = gelu
        if table_act and activation_gelu != is_gelu_arch(arch):
            print("WARN recipe != table: activation_gelu", activation_gelu)
        print("recipe: activation from ops (" + ("gelu" if gelu else "silu") + "), rope from table, coverage PASS")
    elif is_gelu_arch(arch):
        activation_gelu = True
    elif is_silu_arch(arch):
        activation_gelu = False
    else:
        raise Error("unknown architecture " + arch + ": add it to is_gelu_arch or is_silu_arch in tools/gen-profile.mojo after reading the model's graph builder in llama.cpp")

    var granite_mult = arch == "granite"
    var default_attn_scale = 1.0 / sqrt(Float64(hd))
    var attn_scale = m.akv_float("attention.scale", default_attn_scale) if granite_mult else default_attn_scale
    var embed_scale = m.akv_float("embedding_scale", 1.0)
    var residual_scale = m.akv_float("residual_scale", 1.0)
    var logit_scale = m.akv_float("logit_scale", 1.0)

    var ctx_len = m.akv_int("context_length", tmax)
    if ctx_len < tmax:
        print("gen-profile: warning: TMAX", tmax, "exceeds", arch + ".context_length", ctx_len)

    var out = String("# generated by tools/gen-profile.mojo from ") + model_path + " (arch=" + arch + ") -- do not hand-edit\n"
    out += "comptime H = " + String(h) + "\n"
    out += "comptime FFN = " + String(ffn) + "\n"
    out += "comptime VOCAB = " + String(vocab) + "\n"
    out += "comptime N_LAYERS = " + String(n_layers) + "\n"
    out += "comptime NQH = " + String(nqh) + "\n"
    out += "comptime NKVH = " + String(nkvh) + "\n"
    out += "comptime HD = " + String(hd) + "\n"
    out += "comptime NORM_EPS = Float32(" + String(norm_eps) + ")\n"
    out += "comptime NROT_FULL = " + String(nrot_full) + "\n"
    out += "comptime BASE_FULL = Float32(" + String(base_full) + ")\n"
    out += "comptime NROT_SWA = " + String(nrot_swa) + "\n"
    out += "comptime BASE_SWA = Float32(" + String(base_swa) + ")\n"
    out += "comptime SWA_WIN = " + String(swa_win) + "\n"
    out += "comptime SWA_PERIOD = " + String(swa_period) + "\n"
    out += "comptime SWA_FULL_PHASE = " + String(swa_full_phase) + "\n"
    out += "comptime ROPE_NEOX = " + mojo_bool(rope_neox) + "\n"
    out += "comptime QKV_BIAS = " + mojo_bool(qkv_bias) + "\n"
    out += "comptime HAS_GATE = " + mojo_bool(has_gate) + "\n"
    out += "comptime TIE_EMBED = " + mojo_bool(tie_embed) + "\n"
    out += "comptime ROPE_FREQS = " + mojo_bool(rope_freqs) + "  # rope_freqs.weight: last pack entry, divides full-rope theta\n"
    out += "comptime ACTIVATION_GELU = " + mojo_bool(activation_gelu) + "\n"
    out += "comptime GRANITE_MULT = " + mojo_bool(granite_mult) + "\n"
    out += "comptime ATTN_SCALE = Float32(" + String(attn_scale) + ")\n"
    out += "comptime EMBED_SCALE = Float32(" + String(embed_scale) + ")\n"
    out += "comptime RESIDUAL_SCALE = Float32(" + String(residual_scale) + ")\n"
    out += "comptime LOGIT_SCALE = Float32(" + String(logit_scale) + ")\n"
    out += "comptime TMAX = " + String(tmax) + "\n"
    with open(out_path, "w") as f:
        f.write(out)
    print("wrote", out_path, "(arch=" + arch, "H=" + String(h), "FFN=" + String(ffn),
          "N_LAYERS=" + String(n_layers), "NQH=" + String(nqh), "NKVH=" + String(nkvh),
          "HD=" + String(hd), "VOCAB=" + String(vocab) + ")")
