#!/usr/bin/env bash
# M1 gate for tools/gemma4-pack.mojo against gguf-py (the oracle, llama.cpp's own reader):
#   count and bytes of index.txt == the file's tensor table, every entry's offset/size/type == gguf-py's,
#   sha256 of the pack bytes == sha256 of gguf-py's tensor bytes, f16 -> f32 round trip equal,
#   profile.txt == the kv_source rule recomputed from gguf-py's kv,
#   and a reverse arm: the same check on an index with one offset moved by 2 bytes must FAIL.
# Usage: tools/gemma4-pack-check.sh MODEL.gguf PACKDIR
set -euo pipefail
GGUF=$1; PACK=$2
GGUFPY=${GGUFPY:-$HOME/Projects/imports/llama.cpp/gguf-py}
check() {
python3 -I - "$GGUF" "$1" "$PACK/profile.txt" "$GGUFPY" <<'PY'
import sys, hashlib, numpy as np
sys.path.insert(0, sys.argv[4])
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
ref = {t.name: t for t in r.tensors}
lines = [l.split() for l in open(sys.argv[2]).read().splitlines()]
bad = []
if len(lines) != len(r.tensors) or len({l[0] for l in lines}) != len(lines):
    bad.append(f"count {len(lines)} entries vs {len(r.tensors)} tensors")
tot = 0
pack = np.memmap(sys.argv[1], dtype=np.uint8, mode="r")
for name, dt, off, n in lines:
    off, n = int(off), int(n); tot += n
    t = ref.get(name)
    if t is None: bad.append(f"{name}: not in file"); continue
    want_dt = {"F16": "f16", "F32": "f32"}.get(t.tensor_type.name)
    if (off, n, dt) != (int(t.data_offset), int(t.n_bytes), want_dt):
        bad.append(f"{name}: index ({off},{n},{dt}) vs gguf-py ({t.data_offset},{t.n_bytes},{want_dt})"); continue
    a, b = pack[off:off + n], t.data.reshape(-1).view(np.uint8)  # both views of the mmap; streamed in 64 MB chunks
    ha, hb, rt = hashlib.sha256(), hashlib.sha256(), True
    for c in range(0, n, 1 << 26):
        x, y = a[c:c + (1 << 26)], b[c:c + (1 << 26)]
        ha.update(x); hb.update(y)
        if dt == "f16":
            rt &= np.array_equal(x.view(np.float16).astype(np.float32), y.view(np.float16).astype(np.float32), equal_nan=True)
    if ha.digest() != hb.digest():
        bad.append(f"{name}: sha differs"); continue
    if not rt:
        bad.append(f"{name}: f16 -> f32 round trip differs")
if tot != sum(int(t.n_bytes) for t in r.tensors):
    bad.append(f"bytes {tot} vs {sum(int(t.n_bytes) for t in r.tensors)}")
kv = {k: f.contents() for k, f in r.fields.items() if k.startswith("gemma4.")}
nl = kv["gemma4.block_count"]; pat = kv["gemma4.attention.sliding_window_pattern"]
nkv = nl - kv["gemma4.attention.shared_kv_layers"]; ffn = kv["gemma4.feed_forward_length"]
slots, cnt, want = [], {True: 0, False: 0}, []
for il in range(nl):
    s = bool(pat[il]); src = il if il < nkv else nkv - (2 if s else 1)
    if il < nkv: slots.append(cnt[s]); cnt[s] += 1
    else: slots.append(slots[src])
    hd = kv["gemma4.attention.key_length_swa"] if s else kv["gemma4.attention.key_length"]
    want.append(f"L {il} {'S' if s else 'F'} {hd} {int(il < nkv)} {src} {slots[il]} {ffn[il]}")
got = open(sys.argv[3]).read().splitlines()
if got != want:
    bad += [f"profile: '{g}' vs '{w}'" for g, w in zip(got, want) if g != w] or ["profile: line count"]
print("\n".join(f"FAIL {b}" for b in bad[:10]))
print(f"entries {len(lines)} tensors {len(r.tensors)} bytes {tot} problems {len(bad)}")
sys.exit(1 if bad else 0)
PY
}
echo "== arm: pack =="
check "$PACK/index.txt" || { echo "FAIL gemma4-pack-check: pack"; exit 1; }
echo "== reverse arm: one offset moved by 2 bytes must fail =="
mut=$(mktemp -p "$PACK" mutant.XXXX)
awk 'NR == 2 { $3 = $3 + 2 } { print }' "$PACK/index.txt" > "$mut"
if check "$mut"; then rm -f "$mut"; echo "FAIL gemma4-pack-check: mutant passed"; exit 1; fi
rm -f "$mut"
echo "PASS gemma4-pack-check: every tensor accounted for, bytes equal, mutant caught"
