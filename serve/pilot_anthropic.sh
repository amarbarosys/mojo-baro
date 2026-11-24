#!/usr/bin/env bash
# Real Claude Code pilot for the Anthropic Messages endpoint.
# Direct invocation self-submits this GPU workload via gpu-wait.
# If an outer gpu-wait wrapper is used, set BARO_ANTHROPIC_PILOT_ADMITTED=1.
# Run: ./serve/pilot_anthropic.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

if [ "${BARO_ANTHROPIC_PILOT_ADMITTED:-0}" != "1" ]; then
  command -v gpu-wait >/dev/null || { echo "FAIL gpu-wait: command not found"; exit 1; }
  exec gpu-wait run --vram 16 --mem 12 --timeout 900 --cwd "${REPO_DIR}" \
    --label anthropic-qwythos-four-tool -- \
    env BARO_ANTHROPIC_PILOT_ADMITTED=1 "${SCRIPT_DIR}/pilot_anthropic.sh" "$@"
fi

PORT="${PILOT_PORT:-18765}"
BASE="http://127.0.0.1:${PORT}"
MODEL_GGUF="${MODEL_GGUF:-$HOME/Models/qwythos-9b-v2-mtp-q6_k/Qwythos-9B-v2-MTP-Q4_0-BARO-eeb082e.gguf}"
MODEL_PACK="${MODEL_PACK:-$HOME/.cache/baro/qwen35-c19706672fb25b4a/packs/5657eb0f4ce2e32245090603d4e45e6eb18098f353c902bd61a22a9a56adece9}"
MODEL_ENGINE="${MODEL_ENGINE:-$HOME/.cache/baro/qwen35-c19706672fb25b4a/engine-df9114edc69b}"
MODEL="Qwythos-9B-v2"
QUANT="Q4_0"
BAKE="eeb082e"
TMAX=65536
FRONT="${REPO_DIR}/.work/baro-front"
WORK="${REPO_DIR}/.work"
SERVER_LOG="${WORK}/anthropic-pilot-server.log"
METRICS="${WORK}/anthropic-pilot-metrics.txt"
AGGREGATE="${WORK}/anthropic-full-65k-aggregate.txt"
SCRATCH="${WORK}/anthropic-pilot-scratch.txt"
SERVER_PID=0

fail() { echo "FAIL $1: $2" >&2; exit 1; }
mkdir -p "${WORK}"

[ -f "${MODEL_GGUF}" ] || fail model "missing GGUF: ${MODEL_GGUF}"
if [ ! -x "${MODEL_ENGINE}" ] || [ ! -d "${MODEL_PACK}" ]; then
  BUILD_LOG="${WORK}/anthropic-pilot-engine-build.log"
  echo "Building or packing pilot engine with profiles/anthropic-pilot.toml..."
  tools/baro serve "${MODEL_GGUF}" --profile anthropic-pilot --no-serve > "${BUILD_LOG}" 2>&1 \
    || { tail -40 "${BUILD_LOG}"; fail engine-build "see ${BUILD_LOG}"; }
  BUILT_ENGINE=$(awk '/^engine: built in .* -> / { sub(/^.* -> /, ""); e=$0 } /^engine: (hit|miss) / { e=$3 } END { print e }' "${BUILD_LOG}")
  [ -x "${BUILT_ENGINE}" ] || fail engine-build "could not locate executable in ${BUILD_LOG}"
  MODEL_ENGINE="${BUILT_ENGINE}"
  BUILT_PACK=$(sed -n 's/^pack: \(.*\) ([0-9][0-9]* tensors).*/\1/p' "${BUILD_LOG}" | tail -1)
  [ -d "${MODEL_PACK}" ] || MODEL_PACK="${BUILT_PACK}"
  [ -d "${MODEL_PACK}" ] || fail pack-build "could not locate pack in ${BUILD_LOG}"
fi
[ -x "${MODEL_ENGINE}" ] || fail engine "not executable: ${MODEL_ENGINE}"
[ -d "${MODEL_PACK}" ] || fail pack "not found: ${MODEL_PACK}"

if [ ! -x "${FRONT}" ]; then
  echo "Building baro-front..."
  ./.venv/bin/mojo build serve/front/main.mojo -I . -I serve -I serve/front -o "${FRONT}" \
    > "${WORK}/anthropic-pilot-front-build.log" 2>&1 \
    || { tail -40 "${WORK}/anthropic-pilot-front-build.log"; fail front-build "failed"; }
fi

cleanup() {
  if [ "${SERVER_PID}" -gt 0 ]; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=0
  fi
}
trap cleanup EXIT

: > "${SERVER_LOG}"
: > "${METRICS}"
printf 'pilot_status = "old"\n' > "${SCRATCH}"
echo "PILOT_MODEL=${MODEL} quant=${QUANT} bake=${BAKE} YaRN=x4 KV=int8 TMAX=${TMAX}"
BARO_NEURONS=0 BARO_SERVE=1 BARO_SPEC=1 BARO_TMAX="${TMAX}" BARO_LOG_PROMPTS=1 \
  BARO_DEFAULT_TOP_P=0.95 BARO_DEFAULT_TOP_K=20 BARO_PACK="${MODEL_PACK}" \
  "${FRONT}" "${MODEL_ENGINE}" "${MODEL_PACK}" --host 127.0.0.1 --port "${PORT}" --workers 4 \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

READY=0
for _ in $(seq 1 90); do
  if curl -fsS "${BASE}/health" -o "${WORK}/anthropic-pilot-health.json"; then READY=1; break; fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then cat "${SERVER_LOG}"; fail server "process exited before health"; fi
  sleep 0.2
done
[ "${READY}" = 1 ] || { cat "${SERVER_LOG}"; fail health "server did not become ready"; }
python3 - <<'PY'
import json
from pathlib import Path
d=json.loads(Path('.work/anthropic-pilot-health.json').read_text())
assert d['limits']['tmax'] == 65536, d.get('limits')
print('HEALTH_PASS tmax=65536')
PY

COUNT_JSON=$(curl -fsS "${BASE}/v1/messages/count_tokens" -H 'content-type: application/json' \
  -H 'x-api-key: local-only-pilot' \
  --data '{"messages":[{"role":"user","content":"Hello pilot test"}]}')
echo "COUNT_TOKENS ${COUNT_JSON}"
/usr/bin/time -f '%e' -o "${WORK}/anthropic-pilot-smoke.seconds" \
  curl -fsS --no-buffer "${BASE}/v1/messages" -H 'content-type: application/json' \
  -H 'x-api-key: local-only-pilot' \
  --data '{"model":"qwythos-9b-v2","max_tokens":32,"stream":true,"messages":[{"role":"user","content":"Reply exactly READY."}]}' \
  -o "${WORK}/anthropic-pilot-smoke.sse"
python3 - <<'PY'
from pathlib import Path
frames=Path('.work/anthropic-pilot-smoke.sse').read_text().strip().split('\n\n')
events=[next((x[7:] for x in frame.splitlines() if x.startswith('event: ')), '') for frame in frames]
assert events[0]=='message_start' and events[-2:]==['message_delta','message_stop'] and 'content_block_delta' in events,events
print('SMOKE_PASS SSE event lifecycle')
PY

SESSION_ID=$(python3 -c 'import uuid; print(uuid.uuid4())')
export ANTHROPIC_BASE_URL="${BASE}" ANTHROPIC_API_KEY=local-only-pilot ANTHROPIC_MODEL=claude-sonnet-4-5
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_AUTOUPDATER=1 CBM_GATE_OFF=1
BARE_COMMON=(--bare --output-format stream-json --verbose --permission-mode acceptEdits)
READ_ARGS=("${BARE_COMMON[@]}" --tools Read --allowedTools=Read)
EDIT_ARGS=("${BARE_COMMON[@]}" --tools Edit --allowedTools=Edit)
BASH_ARGS=("${BARE_COMMON[@]}" --tools Bash --allowedTools=Bash)
SKILL_ARGS=(--output-format stream-json --verbose --permission-mode acceptEdits --tools Skill --allowedTools=Skill)

run_turn() {
  local name=$1
  shift
  local before after
  before=$(grep -c '^anthropic request input_tokens=' "${SERVER_LOG}" || true)
  "$@"
  after=$(grep -c '^anthropic request input_tokens=' "${SERVER_LOG}" || true)
  printf '\nTURN %s\n' "${name}" >> "${METRICS}"
  awk -v lo="$((before + 1))" -v hi="${after}" \
    '/^anthropic request input_tokens=/ { n++; active=(n >= lo && n <= hi) } \
     active && (/^anthropic request input_tokens=/ || /tok\/s_gen/) { print }' \
    "${SERVER_LOG}" >> "${METRICS}"
}

echo "CLAUDE_SESSION=${SESSION_ID} mode=bare tools=Read,Edit,Bash; Skill resumes same ID in normal mode"
run_turn Read env CBM_GATE_OFF=1 claude --session-id "${SESSION_ID}" -p \
  "Use Read to inspect ${SCRATCH}. Return its exact current line, then stop." \
  "${READ_ARGS[@]}" > "${WORK}/anthropic-pilot-read.jsonl"
run_turn Edit env CBM_GATE_OFF=1 claude --resume "${SESSION_ID}" -p \
  "Use Edit to change pilot_status from old to done in ${SCRATCH}. Report completion, then stop." \
  "${EDIT_ARGS[@]}" > "${WORK}/anthropic-pilot-edit.jsonl"
run_turn Bash env CBM_GATE_OFF=1 claude --resume "${SESSION_ID}" -p \
  "Use Bash to run cat ${SCRATCH}. Confirm the exact output, then finish." \
  "${BASH_ARGS[@]}" > "${WORK}/anthropic-pilot-bash.jsonl"
run_turn Skill env CBM_GATE_OFF=1 claude --resume "${SESSION_ID}" -p \
  'Invoke the Skill tool exactly once for mojo-lean. Read that skill, then give one concise rule from it and finish.' \
  "${SKILL_ARGS[@]}" > "${WORK}/anthropic-pilot-skill.jsonl"

python3 - <<'PY'
import json,re,statistics
from pathlib import Path

files={'Read':'.work/anthropic-pilot-read.jsonl','Edit':'.work/anthropic-pilot-edit.jsonl',
       'Bash':'.work/anthropic-pilot-bash.jsonl','Skill':'.work/anthropic-pilot-skill.jsonl'}
sessions=set(); actions={}
def walk(v,uses,results):
 if isinstance(v,dict):
  if v.get('type')=='tool_use': uses.append(v)
  elif v.get('type')=='tool_result': results.append(v)
  for child in v.values(): walk(child,uses,results)
 elif isinstance(v,list):
  for child in v: walk(child,uses,results)
def flatten(v):
 if isinstance(v,str): return v
 if isinstance(v,list): return '\n'.join(flatten(x) for x in v)
 if isinstance(v,dict): return v.get('text',json.dumps(v,sort_keys=True))
 return str(v)
for name,filename in files.items():
 uses=[];results=[]
 for line in Path(filename).read_text().splitlines():
  event=json.loads(line)
  if event.get('session_id'): sessions.add(event['session_id'])
  walk(event,uses,results)
 uses=[u for u in uses if u.get('name')==name]
 assert len(uses)==1,(name,[(u.get('name'),u.get('input')) for u in uses])
 assert len(results)==1,(name,len(results))
 actions[name]=(uses[0].get('input',{}),flatten(results[0].get('content','')))
assert len(sessions)==1,sessions
assert 'pilot_status = "old"' in actions['Read'][1],actions['Read']
assert 'updated successfully' in actions['Edit'][1].lower(),actions['Edit']
assert actions['Bash'][1].strip()=='pilot_status = "done"',actions['Bash']
assert 'mojo-lean' in str(actions['Skill'][0]).lower(),actions['Skill']
assert 'Launching skill: mojo-lean' in actions['Skill'][1],actions['Skill']
scratch=Path('.work/anthropic-pilot-scratch.txt').read_text()
assert scratch=='pilot_status = "done"\n',repr(scratch)

log=Path('.work/anthropic-pilot-server.log').read_text()
assert 'tmax: 65536' in log.lower() and 'kv dtype: int8' in log.lower()
metric_text=Path('.work/anthropic-pilot-metrics.txt').read_text()
turns={}; current=None
for line in metric_text.splitlines():
 if line.startswith('TURN '): current=line[5:];turns[current]=[]
 elif current: turns[current].append(line)
all_rates=[]; maximum=0
lines=['PILOT PASS','model=Qwythos-9B-v2 quant=Q4_0 bake=eeb082e YaRN=x4 KV=int8 TMAX=65536',
       'session_id='+next(iter(sessions))]
outcomes={'Read':'returned pilot_status = "old"',
          'Edit':'changed pilot_status from old to done',
          'Bash':'printed pilot_status = "done"',
          'Skill':'invoked mojo-lean; Launching skill: mojo-lean'}
for name in ('Read','Edit','Bash','Skill'):
 section=turns[name]
 prompts=[int(m.group(1)) for line in section if (m:=re.search(r'anthropic request input_tokens=\s*(\d+)',line))]
 rates=[float(m.group(1)) for line in section if (m:=re.search(r'tok/s_gen:\s*([0-9.]+)',line))]
 assert prompts and rates,(name,section)
 all_rates.extend(rates);maximum=max(maximum,max(prompts))
 lines.append(f'{name}: {outcomes[name]}; prompt_tokens={prompts}; tok/s_gen={[round(x,2) for x in rates]}; median_tok_s={statistics.median(rates):.2f}')
lines.append(f'max_prompt_tokens={maximum} headroom={65536-maximum} median_tok_s_all={statistics.median(all_rates):.2f} generations={len(all_rates)}')
receipt='\n'.join(lines)+'\n'
Path('.work/anthropic-full-65k-aggregate.txt').write_text(receipt)
print(receipt,end='')
PY

echo "PILOT_PASS model=${MODEL} quant=${QUANT} TMAX=${TMAX} aggregate=${AGGREGATE}"
