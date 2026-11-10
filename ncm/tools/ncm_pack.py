"""ncm_pack.py CKPT_DIR OUT_DIR [DATA_DIR [ORACLE_DIR]]: NCM sudoku .pt -> OUT_DIR/ncm-sudoku-easy.gguf (GGUF v3, arch
"ncm", f32) plus OUT_DIR/weights.f32 + manifest.tsv (the same tensors, raw).
Reads the torch zip directly (raw storages + a pickle index restricted to tensors), so neither torch nor arbitrary
pickle code runs. Linear weights are stored transposed ([in, out]), the layout ncm.mojo's GEMM reads; every tensor
starts on a 64-float boundary. manifest.tsv: name, offset (floats), shape as stored. Meta lines: levels, phi (quant_resi index
per level, PhiNonShared ticks rule). The GGUF carries ncm.levels / ncm.phi and, given ORACLE_DIR (tools/ncm_dump.py),
i32 tensors ncm.run.puzzles [64, 91] + ncm.run.ref.tokens [64, 89] (the oracle's tokens): the file checks itself. DATA_DIR (puzzles.pickle, solutions.pickle, plain lists): -> puzzles.i32 [N, 91],
solutions.i32 [N, 89]."""
import collections, pickle, struct, sys, zipfile
import numpy as np


def load(path):
    z = zipfile.ZipFile(path)
    root = [n for n in z.namelist() if n.endswith("data.pkl")][0].rsplit("/", 1)[0]

    def rebuild(storage, offset, size, stride, *_):
        dt = {"FloatStorage": np.float32, "LongStorage": np.int64}[storage[1]]
        flat = np.frombuffer(z.read(f"{root}/data/{storage[2]}"), dtype=dt)
        it = np.dtype(dt).itemsize
        return np.lib.stride_tricks.as_strided(flat[offset:], size, [s * it for s in stride]).copy()

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if (mod, name) == ("torch._utils", "_rebuild_tensor_v2"): return rebuild
            if (mod, name) == ("collections", "OrderedDict"): return collections.OrderedDict
            if mod == "torch" and name.endswith("Storage"): return name
            raise pickle.UnpicklingError(f"refused {mod}.{name}")

        def persistent_load(self, pid): return pid

    d = U(z.open(f"{root}/data.pkl")).load()
    return d.get("model", d)


def main():
    ck, out = sys.argv[1], sys.argv[2]
    gen, tok = load(f"{ck}/generator.pt"), load(f"{ck}/tokenizer.pt")
    tensors = {"g." + k: v for k, v in gen.items()} | {"t." + k: v for k, v in tok.items()}
    keep = {k: v for k, v in tensors.items()
            if v.dtype == np.float32 and not any(s in k for s in ("ema_", "codebook_hit", "attn_mask", "encoder.blocks",
                                                                  "learned_down", "encoder.wpe", "encoder.ln"))}
    levels = [1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89]  # tokenizer manual_levels (config/eval/sudoku-easy.yaml)
    K = len(levels)
    ticks = np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)
    phi = [int(np.argmin(np.abs(ticks - k / (K - 1)))) for k in range(K)]
    rows, blob, off, gt = [], [], 0, []
    for name, v in keep.items():
        if v.ndim == 2 and (".c_attn.weight" in name or ".c_proj.weight" in name or ".c_fc.weight" in name or
                            "lm_heads" in name or name.endswith("wte.weight") and name.startswith("g.") or
                            "output_layer" in name):
            v = v.T  # Linear [out, in] -> [in, out]
        shape = v.shape  # as stored (Linear weights transposed)
        gt.append((name, np.ascontiguousarray(v, dtype=np.float32)))
        v = np.ascontiguousarray(v, dtype=np.float32).ravel()
        pad = (-off) % 64
        blob.append(np.zeros(pad, np.float32)); off += pad
        rows.append(f"{name}\t{off}\t{'x'.join(map(str, shape))}")
        blob.append(v); off += v.size
    np.concatenate(blob).tofile(f"{out}/weights.f32")
    with open(f"{out}/manifest.tsv", "w") as f:
        f.write(f"#levels\t{','.join(map(str, levels))}\n#phi\t{','.join(map(str, phi))}\n")
        f.write("\n".join(rows) + "\n")
    print(f"{len(rows)} tensors, {off * 4 / 1e6:.1f} MB, phi {phi}")
    kv = {"general.architecture": "ncm", "general.name": "ncm-sudoku-easy", "ncm.levels": levels, "ncm.phi": phi}
    if len(sys.argv) > 4:  # self-check: 64 puzzles + the oracle's tokens for them
        gt.append(("ncm.run.puzzles", np.array(pickle.load(open(f"{sys.argv[3]}/puzzles.pickle", "rb"))[:64], np.int32)))
        gt.append(("ncm.run.ref.tokens", np.fromfile(f"{sys.argv[4]}/tokens.i32", np.int32)[:64 * 89].reshape(64, 89)))
    gguf(f"{out}/ncm-sudoku-easy.gguf", gt, kv)
    print("gguf", f"{out}/ncm-sudoku-easy.gguf", len(gt), "tensors (incl. ncm.run.puzzles, ncm.run.ref.tokens as i32)")
    if len(sys.argv) > 3:
        for n in ("puzzles", "solutions"):  # plain int lists; no globals in these pickles
            a = np.array(pickle.load(open(f"{sys.argv[3]}/{n}.pickle", "rb")), dtype=np.int32)
            a.tofile(f"{out}/{n}.i32"); print(n, a.shape)


def gguf(path, tensors, kv):
    """Minimal GGUF v3 writer: string / int32-array KVs, f32 or i32 tensors (dims in ggml order), 32-byte aligned data."""
    s8 = lambda t: struct.pack("<Q", len(t.encode())) + t.encode()
    out = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(kv)))
    for k, v in kv.items():
        out += s8(k) + (struct.pack("<I", 8) + s8(v) if isinstance(v, str) else
                        struct.pack("<IIQ", 9, 5, len(v)) + np.asarray(v, np.int32).tobytes())
    off, data = 0, []
    for name, a in tensors:
        off += (-off) % 32
        out += s8(name) + struct.pack("<I", a.ndim) + b"".join(struct.pack("<Q", d) for d in reversed(a.shape))
        out += struct.pack("<IQ", 26 if a.dtype == np.int32 else 0, off)  # GGML_TYPE_I32 / F32
        data.append((off, a)); off += a.nbytes
    out += bytes((-len(out)) % 32)
    with open(path, "wb") as f:
        f.write(out); base = f.tell()
        for o, a in data:
            f.seek(base + o); f.write(np.ascontiguousarray(a).tobytes())


if __name__ == "__main__":
    main()
