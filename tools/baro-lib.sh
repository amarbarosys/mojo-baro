# Shared shell helpers of tools/baro and tools/baro-multi-serve (sourced, never run).

# receipt_get LINE KEY: the value of ` KEY=VALUE` on a `receipt:` line (no spaces inside a value).
receipt_get() { printf '%s\n' "$1" | sed -n "s/.* $2=\([^ ]*\).*/\1/p" | head -1; }

# model_sha FILE: sha256 of the file, cached by (path, size, mtime) in $BARO_CACHE/sha-cache.tsv so a warm run never
# re-hashes a multi-GB file it already hashed on this box.
model_sha() {
  local m=$1 f=${BARO_CACHE:-$HOME/.cache/baro}/sha-cache.tsv k s
  mkdir -p "$(dirname "$f")"; touch "$f"
  k=$(stat -c '%s %Y' "$m")
  s=$(awk -v m="$m" -v k="$k" '$1==m && ($2 FS $3)==k {print $4}' "$f" | tail -1)
  if [ -z "$s" ]; then
    s=$(sha256sum "$m" | cut -d' ' -f1)
    echo "$m $k $s" >> "$f"
  fi
  printf '%s\n' "$s"
}

# work_identity_check WORK SHA: a work dir holds the build of ONE model file. 0 when it is new or stamped with SHA;
# 1 with the reason on stderr when it was built from another file (two gguf files with one basename share a dir).
work_identity_check() {
  local w=$1 sha=$2 f=$1/model.sha
  if [ -s "$f" ] && [ "$(cat "$f")" != "$sha" ]; then
    echo "work dir $w holds a build of another file (sha $(cut -c1-16 "$f")), this file is $(printf %s "$sha" | cut -c1-16); pass --work or remove it" >&2
    return 1
  fi
  return 0
}
