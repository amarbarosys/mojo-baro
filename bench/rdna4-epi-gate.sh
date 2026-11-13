#!/usr/bin/env bash
# RDNA4 accumulator-epilogue gate (docs/lanes/2026-11-12-rdna4-epilogue-brief.md). No gfx12 card here: the
# gfx12 row decode (kernels/arch.mojo acc_row) runs on gfx1100 under -D BARO_EMU_GFX12_ROWS=true, where
# arch.mma hands the accumulator back in gfx12 row order. Reuses tools/build-many.sh (cached builds),
# tools/isa-receipt.py + iTool isa-diff (ISA identity) and iTool mutant-check (planted row formula).
#
# usage: bench/rdna4-epi-gate.sh build ROOT TAG ACCEL [mojo args...]  build every target (ONLY=regex narrows) from tree ROOT
#                                                                     into .work/rdna4-epi/bin/TAG; CPU only
#        bench/rdna4-epi-gate.sh isa TAG_A TAG_B     per target: embedded gfx code objects byte-equal AND isa-diff clean; CPU only
#        bench/rdna4-epi-gate.sh run TAG             parity tests of TAG; queues itself under gpu-wait (--shared --vram 1)
#        (test_prefill reads .work/engine-pack-q8 relative to its cwd and this box has no q8 pack: it runs in .work/rdna4-epi/fx,
#        where that pack is blk.0.ffn_gate.weight of the q4 pack requantised to int8 = 15*(nibble-8), scale/15)
#        bench/rdna4-epi-gate.sh engine REF EMU [MUT]  dense engine, 20 prompts of ~2K ids, teacher-forced agreement (BARO_FORCE,
#                                                    TMAX 16384, fixed 64 forced ids): REF greedy ids forced on REF (control),
#                                                    EMU (must be 100%) and MUT (must not be); tags name .work/rdna4-epi/bin/TAG/engine
#        bench/rdna4-epi-gate.sh srccopy SRC DST     every *.mojo of tree SRC into DST (mutant roots; SRC env for `mutants`)
#        bench/rdna4-epi-gate.sh mutants             per kernel family: emulation build with ONE site back at the gfx11
#                                                    formula, its test must FAIL with a parity failure (mutant-check)
# env: BARO_KQ_PACK (kq pack for test_kq_prefill_gemm; missing pack = SKIP, never PASS), JOBS (parallel builds, default 3)
# Lines: `PASS ...` / `FAIL <step>: <reason>` / `SKIP <name>: <reason>` / `PARITYFAIL ...`; exit 1 on any FAIL.
# Receipts: .work/rdna4-epi/ (arm-TAG.txt, run-TAG-NAME.log, isa/, mutants.txt).
set -euo pipefail
cd "$(dirname "$0")/.."
W=.work/rdna4-epi
KQPACK=${BARO_KQ_PACK:-$HOME/Projects/mojo/mojo-baro/.work/ornith-stint/pack-kq}
GW=$HOME/.local/bin/gpu-wait
mkdir -p "$W"

targets() {
  cat <<'EOF'
test_wmma_rdna34|kernels/test_wmma_rdna34.mojo|kernels
test_prefill|kernels/test_prefill.mojo|kernels serve
test_mmq|kernels/test_mmq.mojo|kernels
test_moe_wmma|kernels/test_moe_wmma.mojo|kernels
test_kq_prefill_gemm|kernels/test_kq_prefill_gemm.mojo|kernels tools
bench_fp16|bench/bench_fp16.mojo|kernels
bench_fp16_pipe|bench/bench_fp16_pipe.mojo|kernels
engine|serve/engine.mojo|. kernels
EOF
}

cmds() {
  local root=$1 tag=$2 accel=$3 only=$4 n src incs d inc
  shift 4
  mkdir -p "$W/bin/$tag"
  while IFS='|' read -r n src incs; do
    [[ $n =~ $only ]] || continue
    inc=""
    for d in $incs; do inc="$inc -I $root/$d"; done
    echo "./.venv/bin/mojo build $root/$src$inc $* -o $W/bin/$tag/$n  --target-accelerator $accel"
  done < <(targets)
}

build() {
  local root=$1 tag=$2 accel=$3
  shift 3
  cmds "$root" "$tag" "$accel" "${ONLY:-.}" "$@" | JOBS=${JOBS:-3} tools/build-many.sh "rdna4-epi-$tag" \
    || { echo "FAIL build $tag: see .work/build-many/rdna4-epi-$tag"; exit 1; }
}

isa() {
  local a=$1 b=$2 n t ka kb nco
  while IFS='|' read -r n _ _; do
    [[ $n =~ ${ONLY:-.} ]] || continue
    [ -x "$W/bin/$a/$n" ] && [ -x "$W/bin/$b/$n" ] || { echo "FAIL isa $n: binary missing in $a or $b"; exit 1; }
    for t in "$a" "$b"; do
      rm -rf "$W/isa/$t/$n"; mkdir -p "$W/isa/$t/$n"
      ./.venv/bin/python tools/isa-receipt.py "$W/bin/$t/$n" --hist 0 --dump "$W/isa/$t/$n" > "$W/isa/$t/$n.receipt" \
        || { echo "FAIL isa $n: isa-receipt on $t"; exit 1; }
    done
    nco=$(find "$W/isa/$a/$n" -name '*.co' | wc -l)
    [ "$nco" -gt 0 ] || { echo "FAIL isa $n: no gfx code object in the binary"; exit 1; }
    ka=$(cd "$W/isa/$a/$n" && sha256sum -- *.co | sha256sum | cut -c1-16)
    kb=$(cd "$W/isa/$b/$n" && sha256sum -- *.co | sha256sum | cut -c1-16)
    "$HOME/iTools/bin/isa-diff" "$W/isa/$a/$n.receipt" "$W/isa/$a/$n" "$W/isa/$b/$n.receipt" "$W/isa/$b/$n" > "$W/isa/$n.diff" 2>&1 \
      || { echo "FAIL isa $n: isa-diff verdict, see $W/isa/$n.diff"; exit 1; }
    [ "$ka" = "$kb" ] || { echo "FAIL isa $n: code objects differ ($ka vs $kb, $nco objects)"; exit 1; }
    echo "PASS isa $n: $nco code objects byte-equal ($ka), isa-diff clean"
  done < <(targets)
}

fixture() {
  local fx=$W/fx q4
  [ -f "$fx/.work/engine-pack-q8/index.txt" ] && return 0
  q4=$(readlink -f .work/engine-pack-q4)
  [ -f "$q4/pack.bin" ] || { echo "FAIL fixture: no q4 pack at .work/engine-pack-q4"; return 1; }
  mkdir -p "$fx/.work/engine-pack-q8"
  ln -sfn "$q4" "$fx/.work/engine-pack-q4"
  ./.venv/bin/python - "$q4" "$fx/.work/engine-pack-q8" <<'PY' || { echo "FAIL fixture: q8 requantise"; return 1; }
import sys, numpy as np
src, dst = sys.argv[1:]
N, K = 12288, 4096
off = next(int(l.split()[2]) for l in open(src + "/index.txt") if l.split()[:2] == ["blk.0.ffn_gate.weight", "q4"])
raw = np.fromfile(src + "/pack.bin", dtype=np.uint8, count=N * K // 2 + N * (K // 32) * 2, offset=off)
nib = raw[: N * K // 2].reshape(N, K // 32, 16)
codes = np.concatenate([nib & 0xF, nib >> 4], axis=2).astype(np.int16) - 8
scale = raw[N * K // 2 :].view(np.float16).astype(np.float32) / 15
with open(dst + "/pack.bin", "wb") as f:
    (codes * 15).astype(np.int8).reshape(N, K).tofile(f)
    scale.astype(np.float16).tofile(f)
open(dst + "/index.txt", "w").write(f"blk.0.ffn_gate.weight q8 0 {N * K + N * (K // 32) * 2}\n")
PY
}

run_one() {
  local bin=$PWD/$W/bin/$1/$2 log=$PWD/$W/run-$1-$2.log rc=0
  [ -x "$bin" ] || { echo "no binary $bin" > "$log"; return 2; }
  case $2 in
    bench_fp16|bench_fp16_pipe)
      "$bin" > "$log" 2>&1 || return 2
      grep -q '"correct": ' "$log" || return 2
      grep -q '"correct": true' "$log" || return 1 ;;
    test_kq_prefill_gemm)
      [ -f "$KQPACK/pack.bin" ] || { echo "no kq pack at $KQPACK" > "$log"; return 3; }
      mkdir -p "$W/kqpf-$1"
      "$bin" "$KQPACK" "$W/kqpf-$1" > "$log" 2>&1 || rc=$?
      [ "$rc" = 0 ] || return $((rc == 1 ? 1 : 2))
      ./.venv/bin/python tools/kq-prefill-ref.py "$W/kqpf-$1" >> "$log" 2>&1 || return 1 ;;
    test_prefill)
      fixture > "$log" 2>&1 || return 2
      (cd "$W/fx" && "$bin") > "$log" 2>&1 || rc=$?
      [ "$rc" = 0 ] || return $((rc == 1 ? 1 : 2)) ;;
    *)
      "$bin" > "$log" 2>&1 || rc=$?
      [ "$rc" = 0 ] || return $((rc == 1 ? 1 : 2)) ;;
  esac
}

verdict() {
  local tag=$1 n=$2 rc=$3 log=$W/run-$1-$2.log
  case $rc in
    0) echo "PASS $tag $n: $(grep -aE 'PASS|max_err' "$log" | tail -1 | cut -c1-140)" ;;
    1) echo "PARITYFAIL $tag $n: $(grep -aE 'FAIL|rror:|failure|differing|max_err' "$log" | tail -1 | cut -c1-170)" ;;
    3) echo "SKIP $tag $n: $(head -1 "$log")" ;;
    *) echo "FAIL run $tag $n: did not run to a verdict (rc $rc), see $log" ;;
  esac
}

arm() {
  local tag=$1 n
  { echo "tag=$tag commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | tr '\n' ' ')'"
    echo "kq_pack=$KQPACK present=$([ -f "$KQPACK/pack.bin" ] && echo yes || echo no)"
    { grep -m1 -oE 'gfx[0-9]+' /sys/class/drm/card*/device/product_name 2>/dev/null || /opt/rocm/bin/rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9]+'; } | head -1 | sed 's/^/device=/' || true
    for n in "$W/bin/$tag"/*; do echo "bin $(basename "$n") sha256=$(sha256sum "$n" | cut -c1-16)"; done
  } > "$W/arm-$tag.txt"
}

run() {
  local tag=$1 n rc fails=0 skips=0
  if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
    exec "$GW" run --shared --vram 1 --timeout 900 -- env BARO_KQ_PACK="$KQPACK" ONLY="${ONLY:-.}" "$PWD/bench/rdna4-epi-gate.sh" run "$tag"
  fi
  arm "$tag"
  while IFS='|' read -r n _ _; do
    [ "$n" = engine ] && continue
    [[ $n =~ ${ONLY:-.} ]] || continue
    rc=0; run_one "$tag" "$n" || rc=$?
    verdict "$tag" "$n" "$rc"
    case $rc in 0) ;; 3) skips=$((skips + 1)) ;; *) fails=$((fails + 1)) ;; esac
  done < <(targets)
  echo "run $tag: fails=$fails skips=$skips"
  [ "$fails" = 0 ] || exit 1
}

muts() {
  cat <<'EOF'
rdna34_store|test_wmma_rdna34.mojo|s/acc_row(0, i, h)/2 * i + h/|test_wmma_rdna34
prefill_q4|matmul_prefill.mojo|1,120s/acc_row(\(.*\), i, half)/\1 + 2 * i + half/|test_prefill
prefill_q8|matmul_prefill.mojo|121,$s/acc_row(\(.*\), i, half)/\1 + 2 * i + half/|test_prefill
prefill_lds|matmul_prefill_lds.mojo|s/acc_row(\(.*\), i, half)/\1 + 2 * i + half/|test_prefill
prefill_kq|matmul_prefill_kq.mojo|s/acc_row(\(.*\), i, half)/\1 + 2 * i + half/|test_kq_prefill_gemm
mmq_rows|matmul_mmq.mojo|s/acc_row(\(.*\), i, half)/\1 + 2 * i + half/|test_mmq
mmq_scale|matmul_mmq.mojo|s/s_row = block_row + mmq_scale_slot(tid % BM)/s_row = s_row/|test_mmq
moe_wmma|moe_wmma.mojo|s/acc_row(\(.*\), i, half)/\1 + 2 * i + half/|test_moe_wmma
wmma_lds|matmul_wmma_lds.mojo|s/acc_row(\(.*\), i, half)/\1 + 2 * i + half/|bench_fp16
wmma_pipe|matmul_wmma_pipe.mojo|s/acc_row(\(.*\), i, half)/\1 + 2 * i + half/|bench_fp16_pipe
attn_mask|attn.mojo|s/acc_row(t0, i, half) <= lim/t0 + 2 * i + half <= lim/|test_prefill
attn_repack|attn.mojo|s/bf\[acc_row(0, i, 0)\]/bf[2 * i]/;s/bf\[acc_row(0, i, 1)\]/bf[2 * i + 1]/|test_prefill
attn_out|attn.mojo|s/acc_row(dt \* 16, i, half)\]/dt * 16 + 2 * i + half]/|test_prefill
EOF
}

srccopy() {
  mkdir -p "$2"
  rsync -aL --exclude='/.work/' --exclude='/.venv' --exclude='/.git' --exclude='target/' --exclude='/exchange/' \
    --include='*/' --include='*.mojo' --exclude='*' --prune-empty-dirs "$1/" "$2/"
}

mutant_gate() {
  local m=$1 t=$2 root=$W/mut/$1 rc=0
  cmds "$root" "mut-$m" gfx1100 "^$t\$" -D BARO_EMU_GFX12_ROWS=true | JOBS=${JOBS:-3} tools/build-many.sh "rdna4-epi-mut-$m" > "$W/mut-$m.build" 2>&1 \
    || { echo "FAIL build mutant $m: see $W/mut-$m.build"; return 0; }
  "$GW" run --shared --vram 1 --timeout 600 -- env BARO_KQ_PACK="$KQPACK" "$PWD/bench/rdna4-epi-gate.sh" run-one "mut-$m" "$t" || rc=$?
  return "$rc"
}

mutants() {
  local m f ex t root bad=0
  : > "$W/mutants.txt"
  while IFS='|' read -r m f ex t; do
    [[ $m =~ ${ONLY:-.} ]] || continue
    if [ "$t" = test_kq_prefill_gemm ] && [ ! -f "$KQPACK/pack.bin" ]; then
      echo "SKIP mutant $m: no kq pack at $KQPACK" | tee -a "$W/mutants.txt"; continue
    fi
    root=$W/mut/$m
    rm -rf "$root"; mkdir -p "$root"
    srccopy "${SRC:-.}" "$root"
    { MUTANT_EXPECT='^PARITYFAIL' "$HOME/iTools/bin/mutant-check" "$root/kernels/$f" "$ex" -- "$PWD/bench/rdna4-epi-gate.sh" mutant-gate "$m" "$t" 2>&1 \
        | sed "s/^/mutant $m: /" >> "$W/mutants.txt"; } &
    while [ "$(jobs -rp | wc -l)" -ge "${JOBS:-3}" ]; do wait -n || true; done
  done < <(muts)
  wait
  cat "$W/mutants.txt"
  while IFS='|' read -r m _ _ t; do
    [[ $m =~ ${ONLY:-.} ]] || continue
    grep -qE "^(mutant $m: PASS|SKIP mutant $m:)" "$W/mutants.txt" || { echo "FAIL mutant $m: not killed by $t"; bad=$((bad + 1)); }
  done < <(muts)
  echo "mutants: alive_or_broken=$bad skipped=$(grep -c '^SKIP' "$W/mutants.txt" || true)"
  [ "$bad" = 0 ] || exit 1
}

engine() {
  local ref=$PWD/$W/bin/$1/engine emu=$PWD/$W/bin/$2/engine mut="" out=$W/engine pack tmax=16384 doc=1920 E tf p arm bin a n rows rd void=0 bad=0 mutfull=0 np=0
  [ -n "${3:-}" ] && mut=$PWD/$W/bin/$3/engine
  if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
    exec "$GW" run --shared --vram 12 --timeout 1800 -- "$PWD/bench/rdna4-epi-gate.sh" engine "$@"
  fi
  . bench/min16k.sh; min16k "$tmax"
  pack=$(readlink -f .work/engine-pack-q4)
  for bin in "$ref" "$emu" ${mut:+"$mut"}; do [ -x "$bin" ] || { echo "FAIL engine: no binary $bin"; exit 1; }; done
  [ "$(sha256sum "$ref" | cut -c1-16)" != "$(sha256sum "$emu" | cut -c1-16)" ] || { echo "FAIL engine: ref and emu are the same binary"; exit 1; }
  rm -rf "$out"; mkdir -p "$out/prompts"
  awk -v n="$doc" '{for (i = 1; i <= NF; i++) {c++; if (c <= n) printf "%s ", $i; if (c >= n) exit}}' bench/prefill-prompts/p32768.tokens > "$out/doc.ids"
  [ "$(wc -w < "$out/doc.ids")" = "$doc" ] || { echo "FAIL engine: doc prefix is short"; exit 1; }
  for tf in bench/mtp-prompts/p*.tokens; do { cat "$out/doc.ids"; tr -s ' \n' ' ' < "$tf"; echo; } > "$out/prompts/$(basename "$tf")"; done
  E="BARO_PACK=$pack BARO_TMAX=$tmax BARO_SPEC=0"
  { echo "commit=$(git rev-parse --short HEAD) dirty='$(git status --porcelain | tr '\n' ' ')'"
    echo "ref=$1 sha=$(sha256sum "$ref" | cut -c1-16) emu=$2 sha=$(sha256sum "$emu" | cut -c1-16) mut=${3:-none} sha=$([ -n "$mut" ] && sha256sum "$mut" | cut -c1-16)"
    echo "pack=$pack index_sha=$(sha256sum "$pack/index.txt" | cut -c1-16) bytes=$(stat -c %s "$pack/pack.bin")"
    echo "env='$E' prompts=$(ls "$out/prompts" | wc -l) doc_ids=$doc (first ids of bench/prefill-prompts/p32768.tokens + each bench/mtp-prompts question)"
    echo "bar: ctl (ref forced on its own ids) 100% on every prompt, else VOID; emu 100% on every prompt; prefill rows >= 16 on every run;"
    echo "     TMAX read back = $tmax; mega fail word 0; mut (one site at the gfx11 formula under emulation) below 100% on every prompt"
  } | tee "$out/arm.txt"
  echo "prompt rows ctl emu mut" > "$out/results.txt"
  for tf in "$out"/prompts/p*.tokens; do
    p=$(basename "$tf" .tokens); np=$((np + 1))
    env $E BARO_PROMPT="$tf" "$ref" > "$out/$p.ref.log" 2>&1 || { echo "FAIL engine: ref run $p, see $out/$p.ref.log"; exit 1; }
    grep -a '^GENERATED:' "$out/$p.ref.log" | sed 's/^GENERATED://' > "$out/$p.ids"
    [ -s "$out/$p.ids" ] || { echo "FAIL engine: ref printed no GENERATED for $p"; exit 1; }
    rows=$(grep -aoE 'prefill rows: [0-9]+' "$out/$p.ref.log" | grep -oE '[0-9]+$' | head -1)
    rd=""
    for arm in ctl emu mut; do
      case $arm in ctl) bin=$ref ;; emu) bin=$emu ;; mut) bin=$mut ;; esac
      [ -n "$bin" ] || { rd="$rd -"; continue; }
      env $E BARO_PROMPT="$tf" BARO_FORCE="$out/$p.ids" "$bin" > "$out/$p.$arm.log" 2>&1 || true
      set -- $(grep -aoE 'forced agreement: [0-9]+ / [0-9]+' "$out/$p.$arm.log" | grep -oE '[0-9]+' | tr '\n' ' ')
      a=${1:-}; n=${2:-0}
      if [ -z "$a" ] || [ "$n" = 0 ]; then rd="$rd VOID"; [ "$arm" = mut ] || void=$((void + 1)); continue; fi
      rd="$rd $a/$n"
      grep -aq "TMAX: $tmax " "$out/$p.$arm.log" || { echo "FAIL engine: $p $arm TMAX read-back is not $tmax"; exit 1; }
      [ "$(grep -aoE 'prefill rows: [0-9]+' "$out/$p.$arm.log" | grep -oE '[0-9]+$' | head -1)" = "$rows" ] || { echo "FAIL engine: $p $arm prefill rows differ from ref"; exit 1; }
      case $arm in
        mut) [ "$a" != "$n" ] || mutfull=$((mutfull + 1)) ;;
        *) grep -aq 'mega fail word: 0' "$out/$p.$arm.log" || { echo "FAIL engine: $p $arm fail word not 0"; exit 1; }
           [ "$a" = "$n" ] || { [ "$arm" = ctl ] && void=$((void + 1)) || bad=$((bad + 1)); } ;;
      esac
    done
    [ "${rows:-0}" -ge 16 ] || { echo "FAIL engine: $p prefill rows ${rows:-none} < 16, the prefill path did not run"; exit 1; }
    echo "$p $rows$rd" >> "$out/results.txt"
  done
  column -t "$out/results.txt"
  [ "$np" = 20 ] || { echo "FAIL engine: $np prompts, want 20"; exit 1; }
  [ "$void" = 0 ] || { echo "FAIL engine: VOID, $void control or unreadable runs (the instrument, not the candidate)"; exit 1; }
  [ "$bad" = 0 ] || { echo "FAIL engine: emulation build below 100% on $bad of $np prompts"; exit 1; }
  [ -z "$mut" ] || [ "$mutfull" = 0 ] || { echo "FAIL engine: mutant reached 100% on $mutfull prompts, the gate cannot see a wrong row"; exit 1; }
  echo "PASS engine: emu 100% on $np/$np prompts, ctl 100%, mutant below 100% on every prompt (mut=${mut:+yes})"
}

case ${1:-} in
  build) shift; build "$@" ;;
  isa) isa "$2" "$3" ;;
  run) run "$2" ;;
  run-one) rc=0; run_one "$2" "$3" || rc=$?; verdict "$2" "$3" "$rc"; exit "$rc" ;;
  mutant-gate) mutant_gate "$2" "$3" ;;
  mutants) mutants ;;
  engine) shift; engine "$@" ;;
  srccopy) srccopy "$2" "$3" ;;
  *) echo "usage: $0 build ROOT TAG ACCEL [mojo args] | isa TAG_A TAG_B | run TAG | mutants | engine REF EMU [MUT] | srccopy SRC DST" >&2; exit 2 ;;
esac
