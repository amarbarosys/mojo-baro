#!/usr/bin/env bash
# usage: tools/template-latent-check.sh TEMPLATE.jinja MODEL.gguf [CASEDIR]   (CPU only)
# Latent split invariant: render prefix (latent_part=prefix) and full; full must start with prefix, prefix must end in a newline,
# the remainder must start with a non-space character, and ids(prefix) + ids(rest) must equal ids(full) token for token.
set -euo pipefail
cd "$(dirname "$0")/.."
tpl=${1:?usage: tools/template-latent-check.sh TEMPLATE MODEL.gguf [CASEDIR]}; gguf=${2:?}; dir=${3:-tests/template-latent-cases}
mj=.work/minja-cli; bt=./baro-tokenize; w=.work/template-latent; mkdir -p $w; bad=0; n=0
[ -x $mj ] && [ -x $bt ] || { echo "FAIL setup: need $mj and $bt"; exit 1; }
for c in "$dir"/*.json; do
  b=$(basename "$c" .json); n=$((n+1)); why=""
  jq -c '. + {latent_part:"prefix"}' "$c" > $w/$b.prefix.json
  $mj "$tpl" $w/$b.prefix.json > $w/$b.prefix.txt
  $mj "$tpl" "$c" > $w/$b.full.txt
  pl=$(stat -c %s $w/$b.prefix.txt)
  if ! cmp -s -n $pl $w/$b.prefix.txt $w/$b.full.txt; then why="full does not start with prefix"; fi
  tail -c +$((pl+1)) $w/$b.full.txt > $w/$b.rest.txt
  [ "$(tail -c 1 $w/$b.prefix.txt | od -An -tx1 | tr -d ' \n')" = "0a" ] || why="$why; prefix does not end in newline"
  first=$(head -c 1 $w/$b.rest.txt | od -An -tx1 | tr -d ' \n'); case "$first" in 20|0a|09|0d|"") why="$why; rest starts with space or is empty";; esac
  for p in prefix rest full; do $bt encode $w/$b.$p.txt "$gguf" > $w/$b.$p.ids; done
  cat $w/$b.prefix.ids $w/$b.rest.ids > $w/$b.joined.ids
  cmp -s $w/$b.joined.ids $w/$b.full.ids || why="$why; token ids differ (prefix $(wc -l < $w/$b.prefix.ids) + rest $(wc -l < $w/$b.rest.ids) vs full $(wc -l < $w/$b.full.ids))"
  if [ -z "$why" ]; then echo "ok   $b  prefix $(wc -l < $w/$b.prefix.ids) rest $(wc -l < $w/$b.rest.ids) tokens"; else echo "FAIL $b:$why"; bad=$((bad+1)); fi
done
echo "cases $n, failing $bad"; [ $bad -eq 0 ] || { echo "FAIL template-latent-check: $bad case(s)"; exit 1; }
