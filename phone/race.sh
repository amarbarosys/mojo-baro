#!/usr/bin/env bash
# race.sh OUT [ROUNDS] [ARMS]: Q4_0 GEMV race on the Xiaomi (Adreno 840) at Qwen3-8B decode shapes. wavewait's wwk
# (OpenCL event gpu_us; empty counter list: the Xiaomi's counters are closed and ww-phone stalls on the full list),
# driven through ph.sh (both phone locks). Arms interleaved per round; one row per run: round arm shape us/kernel.
# Fill is wwk's fixed pattern (chk): timing only; correctness is the engine's fp64 oracle selftest.
# Arms: ours_* = phone/cl/q4_0_ours_*.cl (or wavewait examples), flat layout; ns / ns_v = llama.cpp gemv_noshuffle.
set -euo pipefail
out=${1:?usage: race.sh OUT [ROUNDS] [ARMS] [SHAPES]}; rounds=${2:-3}; arms=${3:-ours_r4 ns ns_v}
shapes=${4:-4096x4096 1024x4096 12288x4096 4096x12288 151936x4096}
here=$(cd "$(dirname "$0")" && pwd); ww=~/Projects/mojo/wavewait
L=~/Projects/imports/llama.cpp/ggml/src/ggml-opencl/kernels
NDK=/opt/android-sdk/ndk/29.0.13113456/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang
[ "$ww/build/wwk" -nt "$ww/tools/wwk.c" ] || (cd "$ww" && "$NDK" -O2 -Wall -I assets/adreno tools/wwk.c -o build/wwk -ldl) || { echo "FAIL build: wwk"; exit 1; }
mkdir -p "$out"; res=$out/race.tsv; : > "$res"; : > "$out/empty.list"
ks=""; for a in $arms; do case $a in ours_*) k=$here/cl/q4_0_$a.cl; [ -f "$k" ] || k=$ww/examples/phone/q4_0_$a.cl; ks="$ks $k" ;; esac; done
"$here/ph.sh" push 30 "$ww/build/wwk" "$out/empty.list" $ks "$L/gemv_noshuffle_q4_0_f32.cl" > /dev/null
mb() { awk -v b="$1" 'BEGIN{printf "%.6f", b / 1048576}'; }
for r in $(seq 1 "$rounds"); do for s in $shapes; do for a in $arms; do
  M=${s%x*} K=${s#*x}; q=$(mb $((M * K / 2))); d=$(mb $((M * K / 16))); x=$(mb $((K * 4))); y=$(mb $((M * 4)))
  n=$(( 40 * 4096 * 4096 / (M * K) )); [ $n -lt 4 ] && n=4
  qa=buf:$q; case $a in *img*) qa=img:$q:rgba32ui ;; esac  # ours_*img*: q as an rgba32ui image1d_buffer
  case $a in
    ours_*) k=q4_0_$a.cl
       args="$qa bufh:$d buf:$x buf:$y i32:$K local:1024 k=gemv g=$((M / ${ROWS:-4} * 64)) l=64 o=-cl-std=CL3.0,-cl-mad-enable" ;;
    ns|ns_v) k=gemv_noshuffle_q4_0_f32.cl; v=; [ $a = ns_v ] && v=,-DVECTOR_SUB_GROUP_BROADCAST
       args="img:$q:r32ui buf:$d img:$x:rgba32f u64:0 buf:$y u64:0 i32:$K i32:$M i32:1 i32:$K i32:1 i32:$M i32:1 i32:1 i32:1 k=kernel_gemv_noshuffle_q4_0_f32 g=$(( (M / 2 + 63) / 64 * 64 )),4 l=64,4 o=-cl-std=CL3.0,-cl-mad-enable,-DSIMDGROUP_WIDTH=64$v" ;;
    *) echo "FAIL arm: $a" >&2; exit 2 ;;
  esac
  [ -n "${RACE_ARGS_OVERRIDE:-}" ] && args=$RACE_ARGS_OVERRIDE
  o=$("$here/ph.sh" run 120 "chmod +x wwk; timeout 90 ./wwk run $k $n 1 k.bin empty.list $args > /dev/null 2> r.txt; echo wrc=\$?; grep -E 'gpu_us|FAIL' r.txt; grep -A3 -m1 error r.txt")
  us=$(echo "$o" | awk -v n=$n '$1 == "gpu_us" {printf "%.1f", $2 / n}')
  [ -n "$us" ] || { echo "FAIL run: $a $s: $(echo "$o" | tr '\n' ' ' | cut -c1-400)" >&2; exit 1; }
  printf '%s\t%s\t%s\t%s\n' "$r" "$a" "$s" "$us" | tee -a "$res"
done; done; done
awk -F'\t' '{k=$2"\t"$3; if (!(k in b) || $4 < b[k]) b[k]=$4} END {for (k in b) print "BEST\t" k "\t" b[k]}' "$res" | sort -k3,3 -k2,2 | tee "$out/best.tsv"
