#!/usr/bin/env bash
# usage: tools/build-many.sh LABEL < commands      (one `mojo build ... -o X ...` command per line)
# Runs the builds JOBS at a time (default 4), each capped at nproc/JOBS threads (a Mojo build takes every
# core by default: 6 uncapped jobs took 221 s); cold preflight 4x4 128.7 s vs serial 170-310 s. Each has its own
# -o path and log under .work/build-many/LABEL/. A build is skipped when its cache key matches a
# previous success: key = the command without its -o + the Mojo version + the venv's MAX lib dir (the
# binary's RUNPATH: a venv on another Python has another path) + shim/ + the sha256 of every
# *.mojo in the build's import closure (src, then every `import`/`from` root resolved against the
# source's dir and each -I dir; a root that is a package dir pulls in the whole dir, so the closure is a
# superset). An edit rebuilds only the builds that can see it. A hit copies the cached binary to -o
# (kept under .work/build-cache/bin, newest BUILD_CACHE_KEEP=256), so a lane reuses the binary a
# preflight built with the same flags. BUILD_CACHE=0 disables it. Ends `FAIL k/N <names>` non-zero.
set -euo pipefail
cd "$(dirname "$0")/.."
label=${1:?label}; jobs=${JOBS:-4}; dir=.work/build-many/$label; cache=.work/build-cache
rm -rf "$dir"; mkdir -p "$dir" "$cache"
mojo=./.venv/bin/mojo; [ -x "$mojo" ] || mojo=mojo
mkdir -p "$cache/bin"
tree=$( { "$mojo" --version 2>&1; readlink -f .venv/lib/python3*/site-packages/modular/lib 2>/dev/null || true; git ls-files -co --exclude-standard 'shim/*' | sort | tr '\n' '\0' | xargs -0 -r sha256sum --; } | sha256sum | cut -c1-32)
closure() {  # $1 = build command; prints the sorted *.mojo files the build can import
  local cmd=$1 src incs f r d q=() seen=""
  src=$(grep -oE '[^ ]+\.mojo' <<< "$cmd" | head -1)
  incs=$(grep -oE '(^| )-I +[^ ]+' <<< "$cmd" | awk '{print $2}')
  q=("$src")
  while ((${#q[@]})); do
    f=${q[0]}; q=("${q[@]:1}")
    case " $seen " in *" $f "*) continue;; esac
    seen="$seen $f"
    for r in $(grep -hoE '^[[:space:]]*(from|import)[[:space:]]+\.*[A-Za-z_][A-Za-z0-9_]*' "$f" | awk '{print $2}' | sort -u); do
      if [[ $r == .* ]]; then d=$(dirname "$f"); while IFS= read -r g; do q+=("$g"); done < <(find "$d" -name '*.mojo'); continue; fi
      for d in "$(dirname "$f")" $incs; do
        [ -f "$d/$r.mojo" ] && q+=("$d/$r.mojo")
        [ -d "$d/$r" ] && while IFS= read -r g; do q+=("$g"); done < <(find "$d/$r" -name '*.mojo')
      done
    done
  done
  tr ' ' '\n' <<< "$seen" | sed '/^$/d' | sort -u
}
threads=$(( $(nproc) / jobs )); [ "$threads" -ge 1 ] || threads=1
i=0
# The engine builds are the longest (about 74 s each), so they start first.
while IFS= read -r cmd; do
  [ -n "$cmd" ] || continue
  i=$((i + 1)); printf '%s\n' "$cmd" > "$dir/$i.cmd"
done < <(sed '/^$/d' | awk '/serve\/engine\.mojo/ {print; next} {rest[++n] = $0} END {for (k = 1; k <= n; k++) print rest[k]}')
n=$i; [ "$n" -gt 0 ] || { echo "FAIL build-many $label: no commands"; exit 1; }
one() {  # $1 index
  local i=$1 cmd key
  cmd=$(cat "$dir/$i.cmd")
  local o; o=$(grep -oE '(^| )-o +[^ ]+' <<< "$cmd" | awk '{print $2}' | tail -1); o=$(bash -c "echo $o")
  key=$( { printf '%s\n' "$tree" "$(sed -E 's#(^| )-o +[^ ]+##' <<< "$cmd")"; closure "$cmd" | tr '\n' '\0' | xargs -0 sha256sum --; } | sha256sum | cut -c1-32)
  echo "$key" > "$dir/$i.key"
  if [ "${BUILD_CACHE:-1}" = 1 ] && [ -f "$cache/bin/$key" ]; then
    [ -z "$o" ] || { mkdir -p "$(dirname "$o")"; cp --reflink=auto "$cache/bin/$key" "$o"; }
    touch "$cache/bin/$key"; echo cached > "$dir/$i.st"; return 0
  fi
  cmd=$(printf '%s --num-threads %s' "$cmd" "$threads" | sed -E "s#(^| )-o [^ ]+#\1-o $dir/bin-$i#")
  if bash -c "$cmd" > "$dir/$i.log" 2>&1; then
    echo ok > "$dir/$i.st"; cp --reflink=auto "$dir/bin-$i" "$cache/bin/$key"
    [ -z "$o" ] || { mkdir -p "$(dirname "$o")"; cp --reflink=auto "$dir/bin-$i" "$o"; }
  else echo fail > "$dir/$i.st"; fi
}
export -f one closure; export dir tree cache threads
seq 1 "$n" | xargs -P "$jobs" -I{} bash -c 'one {}'
bad=(); cached=0
for i in $(seq 1 "$n"); do
  st=$(cat "$dir/$i.st" 2>/dev/null || echo missing)
  [ "$st" = cached ] && cached=$((cached + 1))
  if [ "$st" != ok ] && [ "$st" != cached ]; then
    name=$(grep -oE '[^ ]+\.mojo' "$dir/$i.cmd" | head -1)
    bad+=("$name"); echo "  FAIL $name: $(grep -m1 'error:' "$dir/$i.log" 2>/dev/null | cut -c1-140) (log $dir/$i.log)"
  fi
done
rm -f "$dir"/bin-*
ls -t "$cache/bin" | tail -n +$(( ${BUILD_CACHE_KEEP:-256} + 1 )) | sed "s#^#$cache/bin/#" | xargs -r rm -f
if [ "${#bad[@]}" -gt 0 ]; then echo "FAIL ${#bad[@]}/$n ${bad[*]}"; exit 1; fi
echo "OK build-many $label: $n builds, $cached cached, jobs $jobs x $threads threads"
