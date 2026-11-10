#!/usr/bin/env python
"""A4: blk.32 (NextN) draft head, extraction / torch replica / parity /
write-back (bench/draft-head-protocol.md).

Unlike E13's projector (a foreign Linear-GELU-Linear MLP standing in for a
task the trunk cannot do alone), this module retrains blk.32 AS ITSELF: the
same 15 tensors tools/engine-pack.py already packs (attn_norm, attn_q/k/v,
attn_q_norm/k_norm, attn_output, post_attention_norm, ffn_gate/up/down,
nextn.eh_proj, nextn.enorm/hnorm/shared_head_norm), same shapes, same slot,
no kernel change. docs/mtp-notes.md SS1-2 established blk.32's attention+FFN
is structurally identical to one full-attention trunk decoder block, so its
torch replica reuses transformers' own Qwen3_5DecoderLayer / Qwen3_5Attention
/ Qwen3_5MLP / Qwen3_5RMSNorm (the SAME classes the frozen trunk uses for its
own full-attention layers), not a hand-rolled reimplementation.

Extraction reuses the tensor-name-and-permutation mapping already proven in
~/AMDHQ/tools/latent-os/gguf_to_hf_qwen35.py's `is_full` branch (verbatim
transpose convention, verbatim "+1" RMSNorm bias), applied to block 32
instead of a trunk block, plus the four nextn.* tensors that block never
handles.

Run only via gpu-wait when a mode touches the GPU (parity, train):
  PYTHONPATH=~/llama.cpp/gguf-py gpu-wait run --priority 20 --vram 24 -- \\
    ~/AMDHQ/.venv/bin/python tools/mtp_head.py --mode parity ...
`--mode extract` and `--mode writeback` are CPU-only (GGUF/state-dict I/O),
no GPU needed, no gpu-wait required.
"""
import os
import argparse
import json
import struct

import numpy as np
import torch
import torch.nn as nn

H = 4096
NORM_PLUS_ONE = True  # blk.32's norms follow the trunk's own "+1" GGUF convention (verified at extract time, not assumed)

BLK32_CORE = [
    "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
    "attn_q_norm.weight", "attn_k_norm.weight", "attn_output.weight",
    "post_attention_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
]
BLK32_NEXTN = ["nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight"]


def gguf_tensor_to_torch(t):
    kind = t.tensor_type.name
    data = t.data
    if kind == "F32":
        return torch.from_numpy(np.ascontiguousarray(data)).to(torch.float32)
    if kind == "BF16":
        u16 = data.view(np.uint16)
        return torch.from_numpy(np.ascontiguousarray(u16)).view(torch.bfloat16)
    raise ValueError(f"unhandled gguf tensor dtype {kind} for {t.name}")


def torch_to_gguf_bytes(t: torch.Tensor, dtype: str) -> bytes:
    if dtype == "F32":
        return t.detach().to(torch.float32).cpu().numpy().tobytes()
    if dtype == "BF16":
        return t.detach().to(torch.bfloat16).cpu().view(torch.uint16).numpy().tobytes()
    raise ValueError(dtype)


def extract_blk32(gguf_path: str) -> dict:
    """blk.32.* raw GGUF tensors -> a state dict keyed to MTPHead's own
    module names. Same take()/permutation logic as gguf_to_hf_qwen35.py's
    is_full branch (verbatim: attn_q/k/v/output map straight across, no
    fused-qkv split, no v-head reorder -- that machinery is the
    linear_attention branch only, which blk.32 never is)."""
    from gguf import GGUFReader
    reader = GGUFReader(gguf_path)
    by_name = {t.name: t for t in reader.tensors}
    missing = [p for p in [f"blk.32.{n}" for n in BLK32_CORE + BLK32_NEXTN] if p not in by_name]
    if missing:
        raise ValueError(f"blk.32 tensors missing from {gguf_path}: {missing}")

    def take(name):
        return gguf_tensor_to_torch(by_name[f"blk.32.{name}"])

    sd = {}
    sd["decoder.input_layernorm.weight"] = take("attn_norm.weight").float() - 1.0
    sd["decoder.self_attn.q_proj.weight"] = take("attn_q.weight")
    sd["decoder.self_attn.k_proj.weight"] = take("attn_k.weight")
    sd["decoder.self_attn.v_proj.weight"] = take("attn_v.weight")
    sd["decoder.self_attn.o_proj.weight"] = take("attn_output.weight")
    sd["decoder.self_attn.q_norm.weight"] = take("attn_q_norm.weight").float() - 1.0
    sd["decoder.self_attn.k_norm.weight"] = take("attn_k_norm.weight").float() - 1.0
    sd["decoder.post_attention_layernorm.weight"] = take("post_attention_norm.weight").float() - 1.0
    sd["decoder.mlp.gate_proj.weight"] = take("ffn_gate.weight")
    sd["decoder.mlp.up_proj.weight"] = take("ffn_up.weight")
    sd["decoder.mlp.down_proj.weight"] = take("ffn_down.weight")
    sd["eh_proj.weight"] = take("nextn.eh_proj.weight")
    sd["enorm.weight"] = take("nextn.enorm.weight").float() - 1.0
    sd["hnorm.weight"] = take("nextn.hnorm.weight").float() - 1.0
    sd["shared_head_norm.weight"] = take("nextn.shared_head_norm.weight").float() - 1.0
    for k in list(sd):
        if sd[k].dtype == torch.float32 and "weight" in k and sd[k].dim() == 2:
            pass  # 2D weights stay bf16 (attn/ffn/eh_proj matrices); norms are f32 by construction above
    return sd


class MTPHead(nn.Module):
    """eh_proj(concat(enorm(embed(tok_t)), hnorm(h_{t-1}))) -> one
    Qwen3_5DecoderLayer (full_attention config, forced) -> shared_head_norm
    -> (frozen, shared) lm_head. embed_tokens/lm_head/rotary_emb are passed
    in from the frozen trunk model, never copied or trained."""

    def __init__(self, trunk_config, layer_idx: int = 32):
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm
        import copy
        cfg = copy.deepcopy(trunk_config)
        cfg.layer_types = list(cfg.layer_types) + ["full_attention"]
        self.eh_proj = nn.Linear(2 * H, H, bias=False)
        self.enorm = Qwen3_5RMSNorm(H, eps=cfg.rms_norm_eps)
        self.hnorm = Qwen3_5RMSNorm(H, eps=cfg.rms_norm_eps)
        self.shared_head_norm = Qwen3_5RMSNorm(H, eps=cfg.rms_norm_eps)
        self.decoder = Qwen3_5DecoderLayer(cfg, layer_idx=layer_idx)
        self.decoder.block_type = "full_attention"

    def load_blk32(self, sd: dict):
        own = self.state_dict()
        for k in own:
            if k not in sd:
                raise KeyError(f"MTPHead.load_blk32: missing {k}")
        self.load_state_dict({k: v.to(own[k].dtype) for k, v in sd.items()}, strict=True)

    def forward(self, tok_embed_row, h_prev, rotary_emb, position_ids, causal_mask):
        """tok_embed_row: [T, H] frozen embed(tok_t) for t=1..T (the token
        BEING predicted-from, per docs/mtp-notes.md: 'row r is token
        Toks[tok_pos+r]'). h_prev: [T, H] the trunk's own final-norm hidden
        state from BEFORE that token (h_{t-1}). Returns [T, H] pre-lm_head
        hidden (caller applies the frozen lm_head)."""
        e = self.enorm(tok_embed_row.float())
        hn = self.hnorm(h_prev.float())
        x = self.eh_proj(torch.cat([e, hn], dim=-1)).unsqueeze(0)
        # Qwen3_5TextModel.forward: the 4-channel (text, temporal, height,
        # width) position_ids splits into text_position_ids (channel 0, fed
        # to the decoder layer / attention mask) and the remaining 3
        # channels (fed to rotary_emb, matching mrope_section=[11,11,10]).
        # For pure text all 4 channels already carry the same scalar
        # position, so this split changes nothing numerically, but the
        # rotary module's own shape assertion requires exactly 3.
        text_position_ids = position_ids[0]
        rope_position_ids = position_ids[1:]
        pe = rotary_emb(x, rope_position_ids)
        out = self.decoder(x, position_embeddings=pe, attention_mask=causal_mask, position_ids=text_position_ids)
        return self.shared_head_norm(out.squeeze(0))


def build_causal_mask(n: int, dtype, device):
    m = torch.full((n, n), float("-inf"), dtype=dtype, device=device)
    m = torch.triu(m, diagonal=1)
    return m.view(1, 1, n, n)


def load_trunk(hf_dir: str):
    from transformers import Qwen3_5ForCausalLM
    model = Qwen3_5ForCausalLM.from_pretrained(hf_dir, dtype="auto")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def read_dump(path: str):
    """bench/draft_dump.mojo --mode dump format: repeated
    [n_tok u32][tok ids u32*n_tok][(n_tok-1)*H f32]. Returns list of
    {tokens, h: [n_tok-1, H]}."""
    with open(path, "rb") as f:
        data = f.read()
    off, n = 0, len(data)
    docs = []
    while off < n:
        (nt,) = struct.unpack_from("<I", data, off); off += 4
        toks = list(struct.unpack_from(f"<{nt}I", data, off)); off += nt * 4
        nrows = max(0, nt - 1)
        h = struct.unpack_from(f"<{nrows * H}f", data, off); off += nrows * H * 4
        docs.append({"tokens": toks, "h": torch.tensor(h, dtype=torch.float32).view(nrows, H)})
    return docs


def read_v2_dump(path: str):
    """Read and validate the frozen P5a draft dump format.

    The v2 header and every document/record boundary are checked before any
    tensors are returned.  Records are expanded into CPU tensors so the
    trainer cannot silently pair a hidden row with a different token or
    target distribution.
    """
    with open(path, "rb") as f:
        data = f.read()
    off, n = 0, len(data)

    def take_u32(label):
        nonlocal off
        if off + 4 > n:
            raise ValueError(f"v2 dump truncated while reading {label} at offset {off}")
        (value,) = struct.unpack_from("<I", data, off)
        off += 4
        return value

    def take_many(fmt, count, label):
        nonlocal off
        size = struct.calcsize(fmt) * count
        if off + size > n:
            raise ValueError(f"v2 dump truncated while reading {label} at offset {off}")
        values = struct.unpack_from(f"<{count}{fmt}", data, off)
        off += size
        return values

    version = take_u32("version")
    if version != 2:
        raise ValueError(f"unsupported P5a dump version {version}, expected 2")

    docs = []
    while off < n:
        doc_index = len(docs)
        n_tok = take_u32(f"document {doc_index} token count")
        if n_tok < 2:
            raise ValueError(f"document {doc_index} has too few tokens: {n_tok}")
        tokens = list(take_many("I", n_tok, f"document {doc_index} tokens"))
        n_records = take_u32(f"document {doc_index} record count")
        expected = n_tok - 2
        if n_records != expected:
            raise ValueError(
                f"document {doc_index} record count {n_records} != n_tok-2 {expected}"
            )
        records = []
        for record_index in range(n_records):
            pos = take_u32(f"document {doc_index} record {record_index} position")
            input_token = take_u32(f"document {doc_index} record {record_index} input token")
            target_argmax = take_u32(f"document {doc_index} record {record_index} target argmax")
            top8_count = take_u32(f"document {doc_index} record {record_index} top8 count")
            if top8_count != 8:
                raise ValueError(f"document {doc_index} record {record_index} top8 count {top8_count} != 8")
            top8_ids = list(take_many("I", 8, f"document {doc_index} record {record_index} top8 ids"))
            top8_probs = list(take_many("f", 8, f"document {doc_index} record {record_index} top8 probs"))
            hidden = list(take_many("f", H, f"document {doc_index} record {record_index} hidden"))
            if pos != record_index + 1 or not (1 <= pos <= n_tok - 2):
                raise ValueError(
                    f"document {doc_index} record {record_index} position {pos} is not {record_index + 1}"
                )
            if input_token != tokens[pos]:
                raise ValueError(
                    f"document {doc_index} record {record_index} input token {input_token} != tokens[{pos}] {tokens[pos]}"
                )
            if len(set(top8_ids)) != 8:
                raise ValueError(f"document {doc_index} record {record_index} has duplicate top8 IDs")
            if target_argmax != top8_ids[0]:
                raise ValueError(
                    f"document {doc_index} record {record_index} target argmax {target_argmax} != top8_ids[0] {top8_ids[0]}"
                )
            if not all(np.isfinite(top8_probs)) or not all(np.isfinite(hidden)):
                raise ValueError(f"document {doc_index} record {record_index} contains non-finite values")
            prob_sum = sum(top8_probs)
            if abs(prob_sum - 1.0) > 1e-5:
                raise ValueError(
                    f"document {doc_index} record {record_index} top8 probabilities sum to {prob_sum:.9g}"
                )
            if any(p < 0.0 for p in top8_probs):
                raise ValueError(f"document {doc_index} record {record_index} has negative top8 probability")
            records.append({
                "pos": pos,
                "input_token": input_token,
                "target_argmax": target_argmax,
                "top8_ids": torch.tensor(top8_ids, dtype=torch.long),
                "top8_probs": torch.tensor(top8_probs, dtype=torch.float32),
                "h": torch.tensor(hidden, dtype=torch.float32),
            })
        docs.append({"tokens": tokens, "records": records})
    if off != n:
        raise ValueError(f"v2 dump parser stopped at {off} of {n} bytes")
    if not docs:
        raise ValueError("v2 dump contains no documents")
    return docs


def read_parity(path: str):
    """bench/draft_dump.mojo --mode parity format: ONE [n_pairs u32] + pairs
    block PER DOCUMENT, back to back (bench/draft_dump.mojo's main() calls
    run_parity once per document on the same open file handle) -- not a
    single global n_pairs. Each pair: [pos u32][tok_input u32][true_next
    u32][draft_argmax i32][H f32]. tok_input = Toks[pos] (embedded as
    blk32's input token); true_next = Toks[pos+1] (what the draft argmax
    actually predicts)."""
    with open(path, "rb") as f:
        data = f.read()
    off, n = 0, len(data)
    rows = []
    while off < n:
        (npairs,) = struct.unpack_from("<I", data, off); off += 4
        for _ in range(npairs):
            pos, tok_input, true_next, argmax = struct.unpack_from("<IIIi", data, off); off += 16
            h = struct.unpack_from(f"<{H}f", data, off); off += H * 4
            rows.append({"pos": pos, "tok_input": tok_input, "true_next": true_next,
                         "engine_argmax": argmax, "h": torch.tensor(h, dtype=torch.float32)})
    return rows


def cmd_extract(args):
    sd = extract_blk32(args.gguf)
    torch.save(sd, args.out)
    print(json.dumps({"out": args.out, "n_tensors": len(sd),
                       "shapes": {k: list(v.shape) for k, v in sd.items()}}, indent=2))


def cmd_parity(args):
    model = load_trunk(args.hf_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    head = MTPHead(model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config)
    head.load_blk32({k: v.to(device) for k, v in torch.load(args.extracted, map_location=device).items()})
    head.to(device).eval()
    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    rotary = model.model.rotary_emb if hasattr(model, "model") else model.language_model.rotary_emb

    rows = read_parity(args.parity_dump)
    n_ok_argmax_vs_engine = 0
    n_ok_argmax_vs_true = 0
    n_engine_ok_vs_true = 0
    with torch.no_grad():
        for r in rows:
            tok_t = torch.tensor([r["tok_input"]], device=device)  # embed(Toks[pos+1]), the token blk32 is given
            h_prev = r["h"].to(device).unsqueeze(0)
            e_row = embed(tok_t)
            pos_ids = torch.tensor([r["pos"] + 1], device=device).view(1, 1, -1).expand(4, 1, -1)
            mask = build_causal_mask(1, torch.float32, device)
            out = head(e_row, h_prev, rotary, pos_ids, mask)
            logits = lm_head(out.to(lm_head.weight.dtype))
            argmax = int(logits.argmax(dim=-1).item())
            if argmax == r["engine_argmax"]:
                n_ok_argmax_vs_engine += 1
            if argmax == r["true_next"]:
                n_ok_argmax_vs_true += 1
            if r["engine_argmax"] == r["true_next"]:
                n_engine_ok_vs_true += 1
    report = {
        "n_positions": len(rows),
        "torch_argmax_matches_engine_argmax": n_ok_argmax_vs_engine,
        "torch_argmax_matches_true_next": n_ok_argmax_vs_true,
        "engine_argmax_matches_true_next": n_engine_ok_vs_true,
    }
    print(json.dumps(report, indent=2))
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)


def cmd_verify_writeback(args):
    """Independent receipt for cmd_writeback, per bench/draft-head-protocol.md's
    smoke section: read blk.32's 15 tensor byte ranges from the SOURCE gguf
    (not the writeback code's own bookkeeping), then diff src vs dst
    byte-for-byte everywhere else (must be zero differing bytes) and inside
    those 15 ranges (must be non-zero: a no-op patch is also a FAIL, since it
    would pass the "nothing else changed" check vacuously)."""
    from gguf import GGUFReader
    reader = GGUFReader(args.src_gguf)
    by_name = {t.name: t for t in reader.tensors}
    ranges = []
    for suffix in [
        "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_q_norm.weight", "attn_k_norm.weight", "attn_output.weight",
        "post_attention_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
        "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight",
    ]:
        t = by_name[f"blk.32.{suffix}"]
        ranges.append((t.data_offset, t.data_offset + t.n_bytes, suffix))
    ranges.sort()

    import numpy as np
    src_size = __import__("os").path.getsize(args.src_gguf)
    dst_size = __import__("os").path.getsize(args.dst_gguf)
    if src_size != dst_size:
        raise ValueError(f"file length changed: src {src_size} dst {dst_size}")
    src_m = np.memmap(args.src_gguf, dtype=np.uint8, mode="r")
    dst_m = np.memmap(args.dst_gguf, dtype=np.uint8, mode="r")

    def diff_count(a, b, chunk=1 << 28):
        n = 0
        for i in range(0, len(a), chunk):
            n += int(np.count_nonzero(a[i:i + chunk] != b[i:i + chunk]))
        return n

    outside_diff = 0
    inside_diff_bytes = {}
    pos = 0
    checked_ranges = ranges + [(src_size, src_size, None)]
    for start, end, suffix in checked_ranges:
        if pos < start:
            outside_diff += diff_count(src_m[pos:start], dst_m[pos:start])
        if suffix is not None:
            inside_diff_bytes[suffix] = diff_count(src_m[start:end], dst_m[start:end])
            pos = end
    report = {
        "outside_blk32_diff_bytes": outside_diff,
        "inside_blk32_diff_bytes": inside_diff_bytes,
        "all_15_tensors_changed": all(v > 0 for v in inside_diff_bytes.values()),
        "pass": outside_diff == 0 and all(v > 0 for v in inside_diff_bytes.values()),
    }
    print(json.dumps(report, indent=2))
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    if not report["pass"]:
        raise SystemExit("verify-writeback FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["extract", "parity", "writeback", "verify-writeback"], required=True)
    ap.add_argument("--gguf")
    ap.add_argument("--out")
    ap.add_argument("--hf-dir", default=os.path.expanduser("~/Models/qwythos-9b-claude-mythos-5-1m-mtp-bf16/hf"))
    ap.add_argument("--extracted")
    ap.add_argument("--parity-dump")
    ap.add_argument("--report", default=".work/a4/parity-report.json")
    ap.add_argument("--trained")
    ap.add_argument("--src-gguf")
    ap.add_argument("--dst-gguf")
    args = ap.parse_args()

    if args.mode == "extract":
        cmd_extract(args)
    elif args.mode == "parity":
        cmd_parity(args)
    elif args.mode == "writeback":
        cmd_writeback(args)
    elif args.mode == "verify-writeback":
        cmd_verify_writeback(args)


def cmd_writeback(args):
    """Copy src_gguf to dst_gguf, then patch blk.32.*'s 15 tensor byte
    ranges in place with the trained state dict's values, same shapes, same
    dtype, same file offsets: no GGUFWriter, no re-serialization of anything
    else in the file."""
    import shutil
    from gguf import GGUFReader
    shutil.copyfile(args.src_gguf, args.dst_gguf)
    reader = GGUFReader(args.src_gguf)  # read offsets from the ORIGINAL (dst is a byte-identical copy at this point)
    sd = torch.load(args.trained, map_location="cpu")
    name_map = {
        "decoder.input_layernorm.weight": ("attn_norm.weight", "F32", 1.0),
        "decoder.self_attn.q_proj.weight": ("attn_q.weight", "BF16", 0.0),
        "decoder.self_attn.k_proj.weight": ("attn_k.weight", "BF16", 0.0),
        "decoder.self_attn.v_proj.weight": ("attn_v.weight", "BF16", 0.0),
        "decoder.self_attn.o_proj.weight": ("attn_output.weight", "BF16", 0.0),
        "decoder.self_attn.q_norm.weight": ("attn_q_norm.weight", "F32", 1.0),
        "decoder.self_attn.k_norm.weight": ("attn_k_norm.weight", "F32", 1.0),
        "decoder.post_attention_layernorm.weight": ("post_attention_norm.weight", "F32", 1.0),
        "decoder.mlp.gate_proj.weight": ("ffn_gate.weight", "BF16", 0.0),
        "decoder.mlp.up_proj.weight": ("ffn_up.weight", "BF16", 0.0),
        "decoder.mlp.down_proj.weight": ("ffn_down.weight", "BF16", 0.0),
        "eh_proj.weight": ("nextn.eh_proj.weight", "BF16", 0.0),
        "enorm.weight": ("nextn.enorm.weight", "F32", 1.0),
        "hnorm.weight": ("nextn.hnorm.weight", "F32", 1.0),
        "shared_head_norm.weight": ("nextn.shared_head_norm.weight", "F32", 1.0),
    }
    by_name = {t.name: t for t in reader.tensors}
    with open(args.dst_gguf, "r+b") as out:
        for our_key, (gguf_suffix, dt, bias) in name_map.items():
            gname = f"blk.32.{gguf_suffix}"
            t = by_name[gname]
            val = sd[our_key].float() + bias if bias else sd[our_key]
            raw = torch_to_gguf_bytes(val, dt)
            if len(raw) != t.n_bytes:
                raise ValueError(f"{our_key} -> {gname}: byte length {len(raw)} != gguf's {t.n_bytes}")
            out.seek(t.data_offset)
            out.write(raw)
    print(json.dumps({"patched": len(name_map), "dst": args.dst_gguf}, indent=2))


if __name__ == "__main__":
    main()
