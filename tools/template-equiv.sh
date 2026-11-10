#!/usr/bin/env bash
# usage: tools/template-equiv.sh OLD.jinja NEW.jinja [CASEDIR]   (CPU only; the production renderer is minja)
# Renders every case with both templates and compares the bytes. Exit 0 only if all cases are identical.
set -euo pipefail
cd "$(dirname "$0")/.."
old=${1:?usage: tools/template-equiv.sh OLD NEW [CASEDIR]}; new=${2:?}; dir=${3:-tests/template-cases}
bin=.work/minja-cli
if [ ! -x "$bin" ]; then ./.venv/bin/mojo build tools/minja-cli.mojo -I . -o "$bin" > .work/minja-cli.build.log 2>&1 || { echo "FAIL build: minja-cli (.work/minja-cli.build.log)"; exit 1; }; fi
mkdir -p .work/template-equiv; bad=0; n=0
for c in "$dir"/*.json; do
  b=$(basename "$c" .json); n=$((n+1))
  "$bin" "$old" "$c" > .work/template-equiv/$b.old 2> .work/template-equiv/$b.old.err || true
  "$bin" "$new" "$c" > .work/template-equiv/$b.new 2> .work/template-equiv/$b.new.err || true
  if cmp -s .work/template-equiv/$b.old .work/template-equiv/$b.new && cmp -s .work/template-equiv/$b.old.err .work/template-equiv/$b.new.err; then echo "same $b"; else echo "DIFF $b"; bad=$((bad+1)); fi
done
echo "cases $n, differing $bad"; [ $bad -eq 0 ] || { echo "FAIL template-equiv: $bad case(s) differ (see .work/template-equiv)"; exit 1; }
