#!/usr/bin/env bash
# Offline checks of tools/baro-lib.sh and the receipt contract (audit rows C1, C2, C3, C4). No GPU, no build.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
. "$here/tools/baro-lib.sh"
t=$here/.work/fix-front-test/baro-fsm; rm -rf "$t"; mkdir -p "$t"
fail() { echo "FAIL $1: $2" >&2; exit 1; }

# C3: the receipt line, in the format tools/baro prints, carries the paths; the prose lines around it do not decide.
log=$t/cache.log
cat > "$log" <<LOG
engine: built in 312s -> /c/engine-k9c19529
engine: miss /c/engine-k9c19529
pack: built in 5s -> /c/packs/abc (291 tensors)
pack: miss /c/packs/abc (291 tensors)
receipt: id=q35 engine=miss pack=miss elapsed_s=330 engine_path=/c/engine-k9c19529 pack_path=/c/packs/abc tokenizer=built tune_rows=3
LOG
r=$(grep '^receipt: ' "$log" | tail -1)
[ "$(receipt_get "$r" engine_path)" = /c/engine-k9c19529 ] || fail C3 "engine_path"
[ "$(receipt_get "$r" pack_path)" = /c/packs/abc ] || fail C3 "pack_path"
[ "$(receipt_get "$r" engine)" = miss ] || fail C3 "engine= must stay hit|miss, got $(receipt_get "$r" engine)"
[ "$(receipt_get "$r" tokenizer)" = built ] && [ "$(receipt_get "$r" tune_rows)" = 3 ] || fail C2 "tokenizer / tune_rows"

# the same keys are what tools/baro prints (a static check: the receipt line is built in one place)
grep -q 'receipt: id=.* engine_path=.* pack_path=.* tokenizer=\$tok_state tune_rows=' "$here/tools/baro" || fail C2 "tools/baro receipt line lacks the keys"
grep -q 'receipt_get "\$receipt" engine_path' "$here/tools/baro-multi-serve" || fail C3 "multi-serve does not read engine_path"

# C1: a work dir remembers its model; another file with the same basename is refused by name.
mkdir -p "$t/work"
work_identity_check "$t/work" aaaaaaaaaaaaaaaaaaaaaaaa || fail C1 "a new dir must pass"
echo aaaaaaaaaaaaaaaaaaaaaaaa > "$t/work/model.sha"
work_identity_check "$t/work" aaaaaaaaaaaaaaaaaaaaaaaa || fail C1 "the same model must pass"
if msg=$(work_identity_check "$t/work" bbbbbbbbbbbbbbbbbbbbbbbb 2>&1); then fail C1 "another model passed"; fi
case $msg in *"another file"*"--work"*) ;; *) fail C1 "message does not name the cause: $msg";; esac

# C1: model_sha hashes once per (path,size,mtime) and answers from the cache after.
printf 'one' > "$t/m.gguf"
export BARO_CACHE=$t/cache
s1=$(model_sha "$t/m.gguf"); s2=$(model_sha "$t/m.gguf")
[ "$s1" = "$(sha256sum "$t/m.gguf" | cut -d' ' -f1)" ] && [ "$s1" = "$s2" ] || fail C1 "model_sha"
[ "$(wc -l < "$t/cache/sha-cache.tsv")" = 1 ] || fail C1 "the second call re-hashed"
printf 'two!' > "$t/m.gguf"
[ "$(model_sha "$t/m.gguf")" != "$s1" ] || fail C1 "a changed file kept its old sha"

# C4: the receipt object says whether the harness was checked.
grep -q -- '--argjson harness_checked' "$here/tools/baro" && grep -q 'harness_checked:\$harness_checked' "$here/tools/baro" || fail C4 "receipt lacks harness_checked"
echo "baro-fsm: PASS (C1 C2 C3 C4)"
