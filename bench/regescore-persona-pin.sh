#!/usr/bin/env bash
# Rebuild the persona-video.jinja the regescore Blender A/B ran on (runs 1-5, sha256
# 91e834ef4d0d5a09...) into OUT_DIR. ~/Models/RegesCore-1.0-35/personas/build.sh bakes the
# Blender skill in from ~/Brain at build time, and the file was rebuilt with a newer skill
# at 2026-09-24 03:28 (Brain 4806f0fe); this rebuilds it from the skill one commit earlier.
#   bench/regescore-persona-pin.sh .work/regescore-ab/persona-pin
set -euo pipefail
out=$(realpath -m "${1:?usage: regescore-persona-pin.sh OUT_DIR}")
P=$HOME/Models/RegesCore-1.0-35/personas
rev=${SKILL_REV:-4806f0fe~1}
want=${PERSONA_SHA:-91e834ef4d0d5a09}
rm -rf "$out"; mkdir -p "$out/prompts" "$out/skill/references"
git -C "$HOME/Brain" show "$rev:Skills/own/blender-python-api/SKILL.md" > "$out/skill/SKILL.md"
git -C "$HOME/Brain" show "$rev:Skills/own/blender-python-api/references/recipes.md" > "$out/skill/references/recipes.md"
cp "$P/prompts/core.md" "$out/prompts/"
sed "s#~/Brain/Skills/own/blender-python-api/#$out/skill/#" "$P/prompts/video.md" > "$out/prompts/video.md"
sed "s#base=\"\$here/../chat-templates/davidio-chat-template.jinja\"#base=\"$P/../chat-templates/davidio-chat-template.jinja\"#" \
  "$P/build.sh" > "$out/build.sh"
bash "$out/build.sh" > "$out/build.log"
got=$(sha256sum "$out/persona-video.jinja" | cut -c1-${#want})
[[ $got == "$want" ]] || { echo "FAIL pin: persona-video.jinja is $got, want $want (prompts/*.md or the base template changed too)"; exit 1; }
echo "PASS pin: $out/persona-video.jinja sha256 $got"
