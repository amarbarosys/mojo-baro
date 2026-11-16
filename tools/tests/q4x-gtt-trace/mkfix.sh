#!/usr/bin/env bash
# mkfix: a fake /sys + /proc tree for tools/q4x-gtt-trace.{py,mojo} (GTT_TRACE_ROOT) and a fake engine. usage: mkfix.sh DIR
set -euo pipefail
d=${1:?DIR}; rm -rf "$d"; mkdir -p "$d/root/sys/class/drm/card0/device" "$d/root/sys/class/drm/card0-DP-1" "$d/root/sys/class/drm/card1/device" \
  "$d/root/proc/pressure" "$d/root/proc/4242/fdinfo" "$d/root/proc/4343/fdinfo" "$d/root/proc/4444/fdinfo" "$d/root/proc/self" "$d/nocard/sys/class/drm/card0/device"
dev="$d/root/sys/class/drm/card0/device"
echo 25753026560 > "$dev/mem_info_vram_total"; echo 1234567890 > "$dev/mem_info_gtt_used"; echo 9876543210 > "$dev/mem_info_vram_used"
echo 1000000000 > "$d/root/sys/class/drm/card1/device/mem_info_vram_total"      # a small card: skipped by the 20 GB rule
echo 1000000000 > "$d/nocard/sys/class/drm/card0/device/mem_info_vram_total"
printf 'MemTotal:       65000000 kB\nMemAvailable:   40000000 kB\nCached:          9000000 kB\nMapped:           800000 kB\nActive(file):    4000000 kB\nInactive(file):  5000000 kB\nDirty:               120 kB\nAnonPages:       7000000 kB\nMlocked:           64000 kB\nHugePages_Total:       0\n' > "$d/root/proc/meminfo"
printf 'nr_free_pages 12345\npgmajfault 77\npgfault 123456\npgscan_kswapd 5\npgscan_direct 6\npgsteal_kswapd 7\npgsteal_direct 8\npgmigrate_success 9\npgmigrate_fail 10\ncompact_stall 11\ncompact_migrate_scanned 12\nnuma_pages_migrated 13\nworkingset_refault_file 14\nallocstall_normal 15\nthp_fault_alloc 16\n' > "$d/root/proc/vmstat"
printf 'some avg10=0.00 avg60=0.00 avg300=0.00 total=111222\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=3344\n' > "$d/root/proc/pressure/memory"
printf '   8       0 sda 1 0 2 0 0 0 0 0 0 0 0\n 259       0 nvme0n1 100 0 5555555 0 0 0 0 0 0 0 0\n 259       1 nvme0n1p1 1 0 2 0 0 0 0 0 0 0 0\n' > "$d/root/proc/diskstats"
printf 'fake-engine\n' > "$d/root/proc/4242/comm"; printf 'Name:\tfake\nVmRSS:\t  500000 kB\nVmLck:\t  100000 kB\nVmSwap:\t0 kB\n' > "$d/root/proc/4242/status"
printf 'rchar: 1\nread_bytes: 4096000\nwrite_bytes: 0\n' > "$d/root/proc/4242/io"
echo '4242 (fake engine) S 1 4242 4242 0 -1 4194560 1000 0 321 0 5 3 0 0 20 0 1 0 100 1000000 500 18446744073709551615' > "$d/root/proc/4242/stat"
printf 'pos: 0\ndrm-driver:\tamdgpu\ndrm-client-id:\t7\ndrm-memory-gtt:\t2048 KiB\ndrm-memory-vram:\t4096 KiB\n' > "$d/root/proc/4242/fdinfo/3"
printf 'pos: 0\ndrm-driver:\tamdgpu\ndrm-client-id:\t7\ndrm-memory-gtt:\t2048 KiB\ndrm-memory-vram:\t4096 KiB\n' > "$d/root/proc/4242/fdinfo/4"
printf 'pos: 0\ndrm-driver:\tamdgpu\ndrm-client-id:\t9\ndrm-memory-gtt:\t10 KiB\ndrm-memory-vram:\t20 KiB\n' > "$d/root/proc/4242/fdinfo/5"
printf 'pos: 0\nflags: 02\n' > "$d/root/proc/4242/fdinfo/6"
printf 'desktop-shell\n' > "$d/root/proc/4343/comm"
printf 'pos: 0\ndrm-driver:\tamdgpu\ndrm-client-id:\t1\ndrm-memory-gtt:\t70000 KiB\ndrm-memory-vram:\t999 KiB\n' > "$d/root/proc/4343/fdinfo/8"
printf 'tiny\n' > "$d/root/proc/4444/comm"
printf 'pos: 0\ndrm-driver:\tamdgpu\ndrm-client-id:\t2\ndrm-memory-gtt:\t100 KiB\ndrm-memory-vram:\t1 KiB\n' > "$d/root/proc/4444/fdinfo/9"
cat > "$d/fake-engine.sh" <<'ENG'
#!/usr/bin/env bash
# the arguments are echoed back; one stderr line sits between two stdout lines; exit code from FAKE_ENGINE_RC
echo "args: $*"
echo "stderr line \"quoted\" back\\slash	tab, caf$(printf '\303\251')" >&2
echo "tok 1 2 3"
exit "${FAKE_ENGINE_RC:-0}"
ENG
chmod +x "$d/fake-engine.sh"; echo '[1, 2, 3]' > "$d/prompt.json"; echo "fixture in $d"
