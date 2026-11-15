#!/usr/bin/env bash
# operate-audit-check: the committed gate for docs/OPERATE-AUDIT.md. Builds tools/operate-audit.mojo, requires --check to PASS on the tree,
# then plants five defects in a copy and requires each to FAIL: a capability naming a setting that does not exist, an unknown status word,
# a flag naming a missing setting, a stale audit file and a stale skill reference. A gate that cannot fail is not a gate.
set -euo pipefail
cd "$(dirname "$0")/.."
MOJO=${MOJO:-./.venv/bin/mojo}
mkdir -p .work
rm -f .work/operate-audit
timeout 300 "$MOJO" build tools/operate-audit.mojo -I . -o .work/operate-audit > .work/operate-audit-build.log 2>&1 || { echo "FAIL build: see .work/operate-audit-build.log"; exit 1; }
[ -x .work/operate-audit ] || { echo "FAIL build: no binary"; exit 1; }
.work/operate-audit --check || { echo "FAIL check: the tree is stale or inconsistent"; exit 1; }
sources_tracked() { # every file named in a `source` or `best` field of $1 is tracked by git (a public doc must not cite a private file)
  local f=$1 bad=0 p
  for p in $(grep -E '^(source|best|gui|check) = ' "$f" | grep -oE '[A-Za-z0-9_./-]+\.(md|mojo|rs|sh|toml|py|json)' | sort -u); do
    git ls-files --error-unmatch -- "$p" >/dev/null 2>&1 || { echo "untracked or missing: $p"; bad=1; }
  done
  return $bad
}
sources_tracked docs/operate-capabilities.toml || { echo "FAIL sources: a cited file is not tracked"; exit 1; }
T=.work/operate-audit-copy
rm -rf "$T"; mkdir -p "$T/docs" "$T/skills/mojo-baro-operate/references"
for d in serve kernels tools face embed cpu; do ln -s "$PWD/$d" "$T/$d"; done
cp docs/settings.toml docs/operate-capabilities.toml docs/OPERATE-AUDIT.md "$T/docs/"
cp skills/mojo-baro-operate/references/capabilities.md "$T/skills/mojo-baro-operate/references/"
plant() { # name, sed expr, file, expected text
  cp docs/settings.toml docs/operate-capabilities.toml docs/OPERATE-AUDIT.md "$T/docs/"
  sed -i "$2" "$T/$3"
  out=$(.work/operate-audit --root "$T" --check 2>&1 || true)
  grep -q "$4" <<<"$out" || { echo "FAIL falsifier $1: not caught (got: $out)"; exit 1; }
  echo "ok falsifier: $1"
}
plant "unknown setting in a capability" '0,/"BARO_SPEC_K"\]/s//"BARO_SPEC_K", "BARO_ZZ_NOPE"]/' docs/operate-capabilities.toml 'names BARO_ZZ_NOPE'
plant "unknown status word" '0,/status = "WORKS"/s//status = "GREAT"/' docs/operate-capabilities.toml 'status GREAT'
plant "flag names a missing setting" '0,/setting = "BARO_PACK"/s//setting = "BARO_ZZ_NOPE"/' docs/operate-capabilities.toml 'names setting BARO_ZZ_NOPE'
plant "stale audit file" '$a extra line' docs/OPERATE-AUDIT.md 'FAIL stale'
plant "stale skill reference" '$a extra line' skills/mojo-baro-operate/references/capabilities.md 'FAIL stale'
cp docs/operate-capabilities.toml "$T/cap.toml"; sed -i '0,/^source = "/s//source = "CLAUDE.md; /' "$T/cap.toml"
if sources_tracked "$T/cap.toml" > /dev/null 2>&1; then echo "FAIL falsifier citing an untracked file: not caught"; exit 1; fi
echo "ok falsifier: a citation of an untracked file"
rm -rf "$T"
echo "PASS operate-audit-check"
