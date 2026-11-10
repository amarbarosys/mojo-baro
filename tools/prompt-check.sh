#!/usr/bin/env bash
# usage: tools/prompt-check.sh MODEL.gguf [BUDGET_TOKENS]   (CPU only)
# The OrcaSAQ system prompt and tool list must (1) be valid tool JSON, (2) render through serve/templates/qwen35.jinja,
# (3) fit the token budget as a stable prefix (default 1800 = under 60% of the 3072 default window).
set -euo pipefail
cd "$(dirname "$0")/.."
gguf=${1:?usage: tools/prompt-check.sh MODEL.gguf [BUDGET]}; budget=${2:-1800}
sys=serve/prompts/orcasaq-system.md; tools=serve/prompts/orcasaq-tools.json; tpl=serve/templates/qwen35.jinja; w=.work/prompt-check; mkdir -p $w
fail() { echo "FAIL $1: $2"; exit 1; }
jq -e 'all(.[]; .type=="function" and (.function.name|test("^[a-z_]+$")) and (.function.description|length>0) and ((.function.parameters.required - (.function.parameters.properties|keys)) == []))' $tools > /dev/null || fail tools "invalid tool JSON"
[ "$(jq -r '.[].function.name' $tools | sort | uniq -d | wc -l)" = 0 ] || fail tools "duplicate names"
jq -nc --rawfile s $sys --slurpfile t $tools '{messages:[{role:"system",content:$s},{role:"user",content:"x"}],tools:$t[0],enable_thinking:false,add_generation_prompt:true}' > $w/case.json
.work/minja-cli $tpl $w/case.json > $w/render.txt || fail render "minja-cli"
grep -q '<tools>' $w/render.txt && grep -q 'latent_handoff' $w/render.txt || fail render "tools block missing"
./.venv/bin/python - $tpl $w/case.json <<'PY' || fail render "jinja2 cannot render the case"
import json, sys
from pathlib import Path
from jinja2.sandbox import ImmutableSandboxedEnvironment
e = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
def rx(m): raise ValueError(m)
e.globals["raise_exception"] = rx
e.from_string(Path(sys.argv[1]).read_text()).render(**json.loads(Path(sys.argv[2]).read_text()))
PY
n=$(./baro-tokenize encode $w/render.txt "$gguf" | wc -l)
echo "rendered prefix: $n tokens (budget $budget), tools $(jq length $tools), system $(wc -w < $sys) words"
[ "$n" -le "$budget" ] || fail budget "$n tokens over $budget"
echo "PASS prompt-check"
