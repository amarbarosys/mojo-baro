#!/usr/bin/env bash
# usage: bench/kvc-bytes.sh INT8_ENGINE PACK RECON_BAROST03 KV4_PAYLOAD P_TOKENS OUT
# C20 stage B (AMDHQ 06 "## C20:"): the engine's BAROKV4 load path is byte-exact against the
# host reconstruction. Two requests on one BARO_KVQ=int8 engine, prompt = the stored 32k
# tokens, n=1, state_load then state_save:
#   A  load RECON (BAROST03, dequantised on the host)  -> save A.baro
#   B  load PAYLOAD (BAROKV4, unpacked on the device)  -> save B.baro
# Gate: A.baro == B.baro byte for byte (cmp), both loaded at the file's pos, and A's pools
# below pos (row pos itself is recomputed by the restore) equal RECON's (the save path itself is not the thing under test).
# One gpu-wait job. GATE_DRYRUN=1: stub engine copies the loaded file, every post-GPU step runs.
set -euo pipefail
cd "$(dirname "$0")/.."
eng=$1; pack=$2; recon=$3; pay=$4; ptok=$5; out=$6
fail() { echo "FAIL $1: $2"; exit 1; }
mkdir -p "$out"
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ] && [ "${GATE_DRYRUN:-0}" != 1 ]; then
    {
        echo "engine=$eng sha=$(sha256sum "$eng" | cut -c1-16) commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | tr '\n' ' ' | head -c 80)'"
        echo "pack=$pack recon=$recon $(sha256sum "$recon" | cut -c1-16) payload=$pay $(sha256sum "$pay" | cut -c1-16) P=$(sha256sum "$ptok" | cut -c1-16)"
        echo "n=1 tmax=32768 spec=0"
    } > "$out/arm.txt"
    cat "$out/arm.txt"
    exec gpu-wait run --priority 20 --vram 22 --timeout 600 -- "$0" "$@"
fi
if [ "${GATE_DRYRUN:-0}" = 1 ]; then
    stub="$out/stub-engine.py"
    cat > "$stub" <<'PY'
import json, shutil, struct, sys
print('{"ready":true}')
for line in sys.stdin:
    r = json.loads(line)
    pos = struct.unpack("<q", open(r["state_load"], "rb").read(16)[8:16])[0]
    print(f"state loaded: {r['state_load']}  pos {pos}")
    print(json.dumps({"id": r["id"], "tok": 7}))
    shutil.copy(sys.argv[1], r["state_save"])
    print(f"state saved: {r['state_save']}  pos {pos}  in 0.0 s")
PY
    eng="python3 $stub $recon"
fi
echo "${GPU_WAITING_ROOM_JOB:-dryrun}" > "$out/gpu-wait-job.txt"
trap 'pkill -TERM -P $$ 2>/dev/null || :' EXIT
# one engine per arm: a second load of the same prefix in one process hits the checkpoint
# chain ("state hot: ... chain hit, file skipped") and never reads the file under test
for arm in A B; do
    st=$recon; [ "$arm" = B ] && st=$pay
    python3 - "$ptok" "$st" "$out" "$arm" <<'PY'
import json, sys
P = [int(x) for x in open(sys.argv[1]).read().split()]
open(f"{sys.argv[3]}/requests-{sys.argv[4]}.jsonl", "w").write(json.dumps({"id": 1, "prompt": P, "n": 1, "spec": False, "state_load": sys.argv[2], "state_save": f"{sys.argv[3]}/{sys.argv[4]}.baro"}) + "\n")
PY
    env BARO_SERVE=1 BARO_SPEC=0 BARO_PACK="$pack" BARO_TMAX=32768 $eng < "$out/requests-$arm.jsonl" > "$out/out-$arm.txt" 2> "$out/err-$arm.txt" || fail "engine $arm" "$out/err-$arm.txt"
    grep -q '"ready":true' "$out/out-$arm.txt" || fail "engine $arm" "no ready line in $out/out-$arm.txt"
    grep -q "^state loaded: $st  pos 31999" "$out/out-$arm.txt" || fail "load $arm" "no 'state loaded: $st  pos 31999' line in $out/out-$arm.txt (a chain hit or a wrong pos)"
    grep -q "^state saved: $out/$arm.baro  pos 31999" "$out/out-$arm.txt" || fail "save $arm" "no 'state saved' line in $out/out-$arm.txt"
    grep '^state loaded:' "$out/out-$arm.txt" | sed 's/^/'"$arm"': /'
done
python3 - "$recon" "$out/A.baro" "$out/B.baro" "$out" <<'PY'
import numpy as np, sys
recon, A, B, out = sys.argv[1:]
KVHSTR = 128 * 256 + 512
def parts(path):
    raw = np.fromfile(path, dtype=np.uint8)
    pos, conv_n, ssm_n, kv_n = (int(x) for x in raw[8:40].view(np.int64))
    fixed = 72 + 4 * pos + 4 * (conv_n + ssm_n)
    return raw, pos, fixed, kv_n
ra, pa, fa, ka = parts(A); rb, pb, fb, kb = parts(B); rr, pr, fr, kr = parts(recon)
if (pa, fa, ka) != (pb, fb, kb) or ra.size != rb.size:
    sys.exit(f"FAIL shape: A {(pa, fa, ka, ra.size)} B {(pb, fb, kb, rb.size)}")
same_ab = np.array_equal(ra, rb)
same_hdr = ra[:8].tobytes() == b"BAROST03" and rb[:8].tobytes() == b"BAROST03"
ndiff = int((ra != rb).sum())
# rows below pos of A's pools vs RECON's: the restore recomputes row pos from the last prompt
# token (it differs in ~240 of 256 bytes per head), rows past it are whatever the engine holds
def rows(raw, fixed, kv_n, pos):
    pages = kv_n // (8 * 4 * KVHSTR)
    o = []
    for kv in range(2):
        pool = raw[fixed + kv * kv_n: fixed + (kv + 1) * kv_n].reshape(pages, 8, 4, KVHSTR)
        r = pool[..., :128 * 256].reshape(pages, 8, 4, 128, 256).transpose(1, 2, 0, 3, 4).reshape(8, 4, pages * 128, 256)[:, :, :pos, :]
        t = pool[..., 128 * 256:].reshape(pages, 8, 4, 128, 4).transpose(1, 2, 0, 3, 4).reshape(8, 4, pages * 128, 4)[:, :, :pos, :]
        o += [r, t]
    return o
ar = rows(ra, fa, ka, pa); rrw = rows(rr, fr, kr, pr)
same_ar = all(np.array_equal(x, y) for x, y in zip(ar, rrw))
verdict = "PASS" if same_ab and same_hdr and same_ar else "FAIL"
with open(f"{out}/summary.md", "w") as f:
    f.write("# kvc-bytes (C20 stage B)\n\n")
    f.write(f"- A (RECON load -> save) vs B (BAROKV4 load -> save): {'identical' if same_ab else f'{ndiff} bytes differ'} over {ra.size} B, pos {pa}\n")
    f.write(f"- A's K/V rows and tails below pos (row pos is rewritten by the restore) vs RECON: {'identical' if same_ar else 'DIFFER'}\n")
    f.write(f"- verdict {verdict}: the device unpack {'is' if verdict == 'PASS' else 'is NOT'} byte-exact against the host reconstruction\n")
print(open(f"{out}/summary.md").read())
sys.exit(0 if verdict == "PASS" else 1)
PY
