#!/usr/bin/env bash
# The vero persona (RegesCore personas/persona-vero.jinja) @includes live files from newera and
# Brain, so it changes whenever they do (2026-10-17 08:25: +2.8k tokens). This rebuilds the one
# every L5X arm ran (sha256 e2dedbb5c45410cd, 2026-10-16) from git, the way personas/build.sh
# builds it, and refuses any other result. Default revs: newera ce2d037, Brain 87788089.
#   bench/vero-persona-pin.sh OUT.jinja [NEWERA_REV BRAIN_REV EXPECT_SHA16]
# USE=vero-lite builds prompts/vero-lite.md at the same revs (L5X-2 phase 4); EXPECT_SHA16 "-" skips
# the check (prints the sha to pin next time).
# then: PERSONA=OUT.jinja bench/vero-ab.sh ...
# FIND=SHA16 SINCE=date UNTIL=date bench/vero-persona-pin.sh OUT.jinja: search every (newera, Brain)
# commit pair in that window for the one whose rebuild has sha SHA16 (a persona sha seen in an old
# arm.txt), print it and leave that build in OUT.jinja.
set -euo pipefail
out=${1:?usage: vero-persona-pin.sh OUT.jinja [NEWERA_REV BRAIN_REV EXPECT_SHA16]}
use=${USE:-vero} nr=${2:-ce2d037} br=${3:-87788089} want=${4:-e2dedbb5c45410cd}
P=$HOME/Models/RegesCore-1.0-35/personas
src() {
	local f=${1/#\~/$HOME}
	case $f in
	$HOME/Projects/newera/*) git -C "$HOME/Projects/newera" show "$nr:${f#"$HOME"/Projects/newera/}" ;;
	$HOME/Brain/*) git -C "$HOME/Brain" show "$br:${f#"$HOME"/Brain/}" ;;
	/*) cat "$f" ;;
	*) cat "$P/prompts/$f" ;;
	esac
}
expand() {
	while IFS= read -r line || [[ -n "$line" ]]; do
		if [[ "$line" == "@include "* ]]; then src "${line#@include }" | expand; else printf '%s\n' "$line"; fi
	done
}
if [[ -n ${FIND:-} ]]; then
	for n in $(git -C "$HOME/Projects/newera" log --since="${SINCE:?}" --until="${UNTIL:?}" --format=%h); do
		for b in $(git -C "$HOME/Brain" log --since="$SINCE" --until="$UNTIL" --format=%h); do
			USE=$use FIND= "$0" "$out" "$n" "$b" "$FIND" > /dev/null 2>&1 && { echo "PASS find: newera $n Brain $b -> $FIND"; exit 0; }
		done
	done
	echo "FAIL find: no commit pair in $SINCE..$UNTIL rebuilds $FIND"; exit 1
fi
text="$(expand < "$P/prompts/$use.md")"
lit="$(printf '%s' "$text" | jq -Rs .)"
{ printf '{#- persona-%s: built by personas/build.sh from prompts/%s.md, do not edit #}\n' "$use" "$use"
  printf '{%%- set _regescore_default = %s %%}\n' "$lit"
  sed "s/{%- set _sys_msg = none %}/{%- set _sys_msg = {'role': 'system', 'content': _regescore_default} %}/" "$P/../chat-templates/davidio-chat-template.jinja"
} > "$out"
got=$(sha256sum "$out" | cut -c1-16)
[[ $want == - || $got == "$want" ]] || { echo "FAIL pin: rebuilt persona is $got, want $want (newera $nr, Brain $br)"; exit 1; }
echo "PASS pin: $out sha256 $got"
