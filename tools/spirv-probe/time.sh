#!/usr/bin/env bash
# Lane G1 timing round on the lab host's Radeon R5 M330: the lowered rmsnorm, vocab softmax row and q4 GEMV
# (skinny_q4rowb + reduce) against llama.cpp's Vulkan backend on the same card, same shapes.
# Protocol and frozen predictions: exchange/lane-G1-timing-protocol.md.
# usage: time.sh verify   every arm at every shape, parity only, no clocks read (P2: before the freeze)
#        time.sh run      the timed round; log lands in .work/spirv-probe/timing/run-<utc>.log
set -euo pipefail
MODE="${1:?usage: time.sh verify|run}"
case "$MODE" in verify|run) ;; *) echo "FAIL usage: time.sh verify|run"; exit 2 ;; esac
R="$(cd "$(dirname "$0")/../.." && pwd)"
T="$R/tools/spirv-probe"
W="$R/.work/spirv-probe"
LAB="${LAB:-root@lab-host}"
MODEL=qwen2.5-0.5b-instruct-q4_0-pure.gguf
# N x K (rows x columns). The lowered GEMV needs K % 1024 == 0 and N % 8 == 0.
QWEN3_06B="2048x1024 1024x1024 1024x2048 3072x1024 1024x3072 151936x1024"
QWEN25_05B_PADDED="896x1024 128x1024 4864x1024 896x5120"
QWEN3_17B="2048x2048 1024x2048 6144x2048 2048x6144 151936x2048"
OURS="1024x4096 $QWEN3_06B $QWEN25_05B_PADDED $QWEN3_17B"
QWEN25_05B_TRUE="896x896 128x896 4864x896 896x4864 151936x896"
OURS="$(printf '%s\n' $OURS | awk '!seen[$0]++' | tr '\n' ' ')"
REF="$OURS $QWEN25_05B_TRUE"
die() { echo "FAIL $1: $2"; exit 1; }

mkdir -p "$W/timing"
BUILD_ONLY=1 TAG=t-ew "$T/run.sh" elementwise > "$W/timing/build-ew.log" 2>&1 || die build-ew "$W/timing/build-ew.log"
BUILD_ONLY=1 TAG=t-smax DEFS="-D V=151936" "$T/run.sh" elementwise > "$W/timing/build-smax.log" 2>&1 || die build-smax "$W/timing/build-smax.log"
for s in $OURS; do
    BUILD_ONLY=1 TAG="t-gemv-$s" DEFS="-D N=${s%x*} -D K=${s#*x}" "$T/run.sh" gemv > "$W/timing/build-gemv-$s.log" 2>&1 || die "build-gemv-$s" "$W/timing/build-gemv-$s.log"
done
echo "built: t-ew t-smax $(echo $OURS | wc -w) gemv shapes"

ssh -o BatchMode=yes "$LAB" 'rm -rf /root/g0/timing && mkdir -p /root/g0/timing' || die lab-ssh "$LAB unreachable"
for tag in t-ew t-smax $(for s in $OURS; do echo "t-gemv-$s"; done); do
    ssh -o BatchMode=yes "$LAB" "mkdir -p /root/g0/timing/$tag" && scp -q "$W/$tag"/out/*.spv "$LAB:/root/g0/timing/$tag/" || die lab-scp "$tag"
done
scp -q "$T/host.c" "$T/llama-perf-shapes.patch" "$LAB:/root/g0/" || die lab-scp host.c

cat > "$W/timing/lab.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
MODE=$MODE; OURS="$OURS"; REF="$REF"; MODEL=/root/g0/$MODEL
EOF
cat >> "$W/timing/lab.sh" <<'EOF'
cd /root/g0
PM=/sys/kernel/debug/dri/1/amdgpu_pm_info
fails=""
# step <name> <sed filter> <cmd...>: run, keep the full log, show it through the filter, record a failure
step() { local name="$1" filt="$2"; shift 2; local rc=0; "$@" > "timing/$name.log" 2>&1 || rc=$?; sed -E 's/\x1b\[[0-9;]*m//g' "timing/$name.log" | sed -E "$filt"; if [ "$rc" -ne 0 ]; then echo "FAIL $name: exit $rc, /root/g0/timing/$name.log"; fails="$fails $name"; fi; }
OURF='/Rusticl warning/d'; REFF='/^\s*$/d; /not supported/d'; SYNCF='/G0 |Backend|Device desc/!d' 
gcc -O2 -Wall host.c -o host -lOpenCL -lm 2> gcc.log || { echo "FAIL host-build: $(head -5 gcc.log)"; exit 1; }
if pgrep -x ninja > /dev/null; then echo "FAIL llama-build: a ninja is still running on the lab box, timing next to a build is void"; exit 1; fi
if ! grep -q G0_SHAPES llama.cpp/tests/test-backend-ops.cpp; then (cd llama.cpp && patch -p1 < ../llama-perf-shapes.patch) > patch.log 2>&1 || { echo "FAIL llama-patch: /root/g0/patch.log"; exit 1; }; fi
ninja -C llama.cpp/build llama-bench test-backend-ops > llama-rebuild.log 2>&1 || { echo "FAIL llama-rebuild: /root/g0/llama-rebuild.log"; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL model: $MODEL missing"; exit 1; }

echo "== receipt: host"
date -u +%FT%TZ; uname -r; pacman -Q mesa opencl-mesa vulkan-radeon | tr '\n' ' '; echo
echo "llama.cpp source $(cat llama.cpp/SOURCE_COMMIT) + llama-perf-shapes.patch; build flags: $(grep -E 'GGML_VULKAN:|CMAKE_BUILD_TYPE:' llama.cpp/build/CMakeCache.txt | tr '\n' ' ')"
echo "cpu governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor), loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
echo "gpu idle: $(grep 'power level' $PM)"
sha256sum host llama.cpp/build/bin/test-backend-ops llama.cpp/build/bin/llama-bench "$MODEL" | sed 's/^\(.\{16\}\)[0-9a-f]*/\1/'
VKIDX="$(llama.cpp/build/bin/llama-bench --list-devices 2>/dev/null | sed -nE 's/^ *Vulkan([0-9]+): .*Radeon.*/\1/p' | head -1)"
[ -n "$VKIDX" ] || { echo "FAIL vk-device: no Radeon in llama-bench --list-devices"; exit 1; }
echo "vulkan index of the Radeon: $VKIDX"

( while :; do grep 'power level' $PM | sed "s/^/$(date +%s) /"; sleep 0.5; done ) >> timing/pm-samples.log 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT
TF=""; [ "$MODE" = run ] && TF="--time"
export RUSTICL_ENABLE=radeonsi

echo "== ours: rmsnorm H=4096, lowered vs host-reduction"
for r in 1 2 4 8 16 32 64; do step "ours-rmsnorm-R$r" "$OURF" ./host timing/t-ew $TF R=$r amar_rmsnorm amar_rmsnorm_hostreduce; done
echo "== ours: softmax, one vocab row"
step ours-softmax "$OURF" ./host timing/t-smax $TF R=1 V=151936 amar_softmax_rows
echo "== ours: q4 GEMV + reduce"
for s in $OURS; do step "ours-gemv-$s" "$OURF" ./host "timing/t-gemv-$s" $TF GN=${s%x*} GK=${s#*x} amar_skinny_reduce; done
echo "pm levels seen during ours: $(cut -d' ' -f2- timing/pm-samples.log | sort | uniq -c | tr '\n' ';')"; : > timing/pm-samples.log

echo "== reference: llama.cpp Vulkan, op level (test-backend-ops perf, backend Vulkan$VKIDX)"
REFSHAPES="$(echo $REF | tr ' ' ',')"; [ "$MODE" = verify ] && REFSHAPES="8x1024"
step ref-ops-batched "$REFF" env G0_SHAPES="$REFSHAPES" llama.cpp/build/bin/test-backend-ops perf -b "Vulkan$VKIDX"
if [ "$MODE" = run ]; then
    echo "== reference: same, one op per graph (submit + sync per op)"
    step ref-ops-sync "$SYNCF" env G0_NRUNS=1 G0_SHAPES="$REFSHAPES" llama.cpp/build/bin/test-backend-ops perf -b "Vulkan$VKIDX"
fi
echo "pm levels seen during reference ops: $(cut -d' ' -f2- timing/pm-samples.log | sort | uniq -c | tr '\n' ';')"; : > timing/pm-samples.log

echo "== reference: llama-bench end to end, $MODEL"
NGEN=64; REPS=5; [ "$MODE" = verify ] && { NGEN=4; REPS=1; }
step bench-radeon "$REFF" env GGML_VK_VISIBLE_DEVICES=$VKIDX llama.cpp/build/bin/llama-bench -m "$MODEL" -ngl 99 -p 0 -n $NGEN -r $REPS
echo "pm levels seen during llama-bench on the Radeon: $(cut -d' ' -f2- timing/pm-samples.log | sort | uniq -c | tr '\n' ';')"
if [ "$MODE" = run ]; then
    # -dev none, not -ngl 0: on a Vulkan build -ngl 0 measured 38 tok/s against 54 here (2026-09-17, cause not established)
    : > timing/pm-samples.log
    echo "-- context arm: same model on the CPU (i5-6200U), every GPU device off"
    step bench-cpu "$REFF" llama.cpp/build/bin/llama-bench -m "$MODEL" -dev none -p 0 -n $NGEN -r $REPS
    echo "pm levels seen on the Radeon during the CPU arm: $(cut -d' ' -f2- timing/pm-samples.log | sort | uniq -c | tr '\n' ';')"
fi
echo "gpu after: $(grep 'power level' $PM)"
[ -z "$fails" ] || { echo "FAIL $(echo $fails | wc -w) steps:$fails"; exit 1; }
echo "PASS timing-$MODE"
EOF
scp -q "$W/timing/lab.sh" "$LAB:/root/g0/timing-lab.sh" || die lab-scp lab.sh
LOG="$W/timing/$MODE-$(date -u +%Y%m%dT%H%M%SZ).log"
ssh -o BatchMode=yes "$LAB" 'bash /root/g0/timing-lab.sh' 2>&1 | tee "$LOG"
echo "log: $LOG"
