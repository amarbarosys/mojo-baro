#!/usr/bin/env bash
# usage: N=20 bench/crew-handoff.sh OUTDIR   (re-execs under gpu-wait; 2 engines, one weight copy; about 6 GPU-minutes)
# Short test of the latent handoff primitive (bench/crew-prereg.md amendment 2). The worker note is the generator's exact worked
# solution, so no worker text is generated. Speaker D answers in 12 tokens. Arms: T0 no note, T note as text (T2, T3 reruns = floor),
# Tn noise note, L note as state of worker W over /v1/ipc/fork, Lx state of another item.
set -euo pipefail
cd "$(dirname "$0")/.."; export LC_ALL=C
. bench/min16k.sh; min16k 16384 16000
out=$(realpath -m "${1:?usage: bench/crew-handoff.sh OUTDIR}"); mkdir -p "$out"
if [ -z "${GPU_WAITING_ROOM_JOB:-}" ]; then
  exec gpu-wait run --priority ${PRIO:-90} --vram 23 --timeout ${TIMEOUT:-900} -- env GPU_WAITING_ROOM_JOB=1 N="${N:-20}" L="${L:-6}" SEED="${SEED:-11}" PATH="$PATH" "$PWD/bench/crew-handoff.sh" "$out"
fi
N=${N:-20}; L=${L:-6}; port=${PORT:-8290}
eng=$(realpath .work/bake27/engine-lean); pack=$(realpath .work/bake27/verify3/pack); tok=$(realpath .work/crew/tok/tokenizer.json)
shim=$(realpath .work/shim-build/libamarbaro_shim.so); serve=serve/target/release/baro-serve; rec=$out/weights.rec
fail() { echo "FAIL $1: $2"; exit 1; }
used=$(( $(cat /sys/class/drm/card1/device/mem_info_vram_used) >> 20 )); [ "$used" -lt 2560 ] || fail preflight "vram_used ${used} MiB"
pids=(); stop_all() { for p in "${pids[@]}"; do kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }; trap stop_all EXIT
for i in 0 1; do
  w=(BARO_WEIGHTS_IMPORT="$rec"); [ "$i" = 0 ] && w=(BARO_WEIGHTS_EXPORT="$rec")
  setsid env "${w[@]}" BARO_EMBD_HOST=1 BARO_PACK="$pack" BARO_SPEC=1 BARO_MEGA=1 BARO_TMAX=16384 BARO_SHIM_LIB="$shim" BARO_VRAM_LOG=1 \
    "$serve" --engine "$eng" --pack "$pack" --tokenizer "$tok" --port $((port + i)) > "$out/serve-$i.out" 2> "$out/serve-$i.err" &
  pids+=($!)
  for _ in $(seq 180); do curl -sf "http://127.0.0.1:$((port + i))/health" >/dev/null 2>&1 && break; sleep 1; done
  curl -sf "http://127.0.0.1:$((port + i))/health" >/dev/null || fail "health $i" "$out/serve-$i.err"
done
grep -q "weights exported over IPC" "$out/serve-0.err" || fail readback "engine 0 exported no weights"
grep -q "weights imported over IPC" "$out/serve-1.err" || fail readback "engine 1 imported no weights"
echo "N=$N L=$L seed=${SEED:-11} engine_sha=$(sha256sum "$eng" | cut -c1-16) pack_sha=$(cut -c1-60 "$pack/identity.json" | tr -d "\n" | sha256sum | cut -c1-16) engines=2 spec=1 tmax=2560 vram_mib=$(( $(cat /sys/class/drm/card1/device/mem_info_vram_used) >> 20 )) commit=$(git rev-parse --short HEAD)" > "$out/arm.txt"
check_fw() { if grep -q "fail word: [1-9]" "$out"/serve-*.err; then fail fail-word "NOT-RESIDENT, tokens invalid"; fi; }
post() { local b c; b=$(curl -s -w "\n%{http_code}" -X POST "http://127.0.0.1:$1$2" -H 'content-type: application/json' -d "$3"); c=${b##*$'\n'}; b=${b%$'\n'*}; if [ "${c:-0}" -ge 400 ] || [ "${c:-0}" = 0 ]; then echo "FAIL http $c $2: ${b:0:300}" >&2; exit 1; fi; printf '%s' "$b"; }
tok_ids() { post $port /tokenize "$(jq -n --arg c "$1" '{content:$c,add_special:false}')" | jq -c .tokens; }
gen() { post $port /v1/completions "{\"prompt\":$1,\"max_tokens\":16000,\"temperature\":0,\"spec\":true}"; }
txt() { jq -r '.choices[0].text // .text // empty'; }
ans() { { printf 'ANSWER:'; cat; } | grep -o 'ANSWER:[ ]*-\?[0-9]\+' | head -1 | grep -o -- '-\?[0-9]\+$' || true; }
res() { jq -nc --arg a "$1" --arg i "$2" --arg ans "$3" --arg e "$4" --arg w "$5" '{arm:$a,item:($i|tonumber),ans:$ans,ok:($ans==$e),wall_s:($w|tonumber)}' >> "$out/results.jsonl"; }
agree() { jq -s --arg a "$1" --arg b "$2" '[group_by(.item)[]|select((map(select(.arm==$a))|length)>0 and (map(select(.arm==$b))|length)>0)|(map(select(.arm==$a))[0].ans)==(map(select(.arm==$b))[0].ans)]|map(select(.))|length' "$out/results.jsonl"; }
HEAD_='<|im_start|>user\n'; TAIL_='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
.work/crew_tasks "$L" "$N" ${SEED:-11} mod1000 > "$out/tasks.json"; : > "$out/results.jsonl"
for i in $(seq 0 $((N-1))); do
  task=$(jq -r ".[$i].prompt" "$out/tasks.json"); note=$(jq -r ".[$i].note" "$out/tasks.json"); printf '%s\n' "$note" > "$out/note-$i.txt"
  printf -v pre "$HEAD_%s\n\nNotes from colleague 1:\n%s" "$task" "$note"
  tok_ids "$pre" > "$out/ip-$i.json"
done
for i in $(seq 0 $((N-1))); do
  e=$(jq -r ".[$i].expected" "$out/tasks.json"); task=$(jq -r ".[$i].prompt" "$out/tasks.json"); note=$(jq -r ".[$i].note" "$out/tasks.json")
  printf -v suf "\n\nGive the final answer.${TAIL_}ANSWER:"; ip=$(< "$out/ip-$i.json"); is=$(tok_ids "$suf")
  all=$(jq -nc --argjson a "$ip" --argjson b "$is" '$a+$b')
  printf -v p0 "$HEAD_%s${TAIL_}ANSWER:" "$task"; r=$(gen "$(tok_ids "$p0")"); res T0 $i "$(echo "$r" | txt | ans)" $e 0
  for arm in T T2 T3; do t0=$(date +%s.%N); r=$(gen "$all"); t1=$(date +%s.%N); res $arm $i "$(echo "$r" | txt | ans)" $e $(awk "BEGIN{print $t1 - $t0}"); done
  nz=$(printf '%s' "$note" | tr -s ' \n' '\n\n' | shuf --random-source=<(yes) | tr '\n' ' ')
  printf -v nzp "$HEAD_%s\n\nNotes from colleague 1:\n%s$suf" "$task" "$nz"; r=$(gen "$(tok_ids "$nzp")"); res Tn $i "$(echo "$r" | txt | ans)" $e 0
  for arm in L Lx; do
    j=$i; [ $arm = Lx ] && j=$(( (i + N - 1) % N ))
    tc0=$(date +%s.%N)
    cid=$(post $((port + 1)) /v1/checkpoints "$(jq -nc --argjson p "$(cat "$out/ip-$j.json")" '{prompt:$p,ttl_s:600,max_tokens:1,spec:true}')" | jq -r .checkpoint.id)
    tc1=$(date +%s.%N); res CK$arm $i "" "" $(awk "BEGIN{print $tc1 - $tc0}")
    t0=$(date +%s.%N)
    r=$(post $((port + 1)) /v1/ipc/fork "$(jq -nc --arg c "$cid" --arg t "127.0.0.1:$port" --argjson s "$is" '{checkpoint:$c,target:$t,suffix:$s,max_tokens:16000,temperature:0,spec:true}')")
    t1=$(date +%s.%N); res $arm $i "$(echo "$r" | txt | ans)" $e $(awk "BEGIN{print $t1 - $t0}")
    echo "$arm item $i $(echo "$r" | jq -c '{pos:.pos,tier:.tier,ipc:{bytes:.ipc.bytes,prefix_hash:.ipc.prefix_hash,live_after_release:.ipc.live_after_release},cached_tokens:.cached_tokens}')" >> "$out/fingerprint.txt"
    curl -s -X DELETE "http://127.0.0.1:$((port + 1))/v1/checkpoints/$cid" >/dev/null
  done
  check_fw; echo "item $i" >> "$out/progress.txt"
done
for a in T0 T T2 T3 Tn L Lx; do echo "$a correct $(jq -s "[.[]|select(.arm==\"$a\" and .ok)]|length" "$out/results.jsonl") / $N  median_wall_s $(jq -s "[.[]|select(.arm==\"$a\")|.wall_s]|sort|.[length/2|floor]" "$out/results.jsonl")" | tee -a "$out/arm.txt"; done
echo "answer agreement: L~T2 $(agree L T2)/$N  floor T3~T2 $(agree T3 T2)/$N  Lx~T2 $(agree Lx T2)/$N  T~T2 $(agree T T2)/$N" | tee -a "$out/arm.txt"
mcnemar() { jq -r --arg a "$1" --arg b "$2" -s '[group_by(.item)[]|{a:(map(select(.arm==$a))[0].ok),b:(map(select(.arm==$b))[0].ok)}]|"\([.[]|select(.a and (.b|not))]|length) \([.[]|select((.a|not) and .b)]|length)"' "$out/results.jsonl" | awk -v A="$1" -v B="$2" '{b=$1;c=$2;n=b+c;m=(b<c)?b:c;p=0;for(k=0;k<=m;k++){x=1;for(j=1;j<=k;j++)x=x*(n-j+1)/j;p+=x}p=p*2/(2^n);if(n==0||p>1)p=1;printf "McNemar %s vs %s: only-%s %d, only-%s %d, exact p %.2e\n",A,B,A,b,B,c,p}'; }
{ mcnemar L Lx; mcnemar L Tn; mcnemar L T0; mcnemar L T3; } | tee -a "$out/arm.txt"
ck=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$((port + 1))/v1/checkpoints" -H 'content-type: application/json' -d "$(jq -nc --argjson p "$(cat "$out/ip-0.json")" '{prompt:$p,ttl_s:300,max_tokens:1,spec:true}')")
cid=$(curl -s -X POST "http://127.0.0.1:$((port + 1))/v1/checkpoints" -H 'content-type: application/json' -d "$(jq -nc --argjson p "$(cat "$out/ip-0.json")" '{prompt:$p,ttl_s:300,max_tokens:1,spec:true}')" | jq -r .checkpoint.id)
for what in pack prefix_hash; do c=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$((port + 1))/v1/ipc/fork" -H 'content-type: application/json' -d "$(jq -nc --arg c "$cid" --arg t "127.0.0.1:$port" --arg w "$what" --argjson s "$(tok_ids "$(printf "\n\nGive the final answer.${TAIL_}ANSWER:")")" '{checkpoint:$c,target:$t,suffix:$s,max_tokens:16000,temperature:0,spec:true,tamper:$w}')"); echo "tamper $what -> HTTP $c" | tee -a "$out/arm.txt"; done
curl -s -X DELETE "http://127.0.0.1:$((port + 1))/v1/checkpoints/$cid" >/dev/null
