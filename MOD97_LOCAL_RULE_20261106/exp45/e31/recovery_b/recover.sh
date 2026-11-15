#!/bin/bash
# E31 worker B, stage 2: recovery after a container reset. Idempotent; safe to run at any time.
#
#   bash recover.sh check   report only: environment, work folder, which lives are pushed, running or missing
#   bash recover.sh go      repair what is missing, set aside partial files of killed lives, restart the queue with the lives not yet pushed
#
# Lives cannot resume mid-way (E26's life loop writes no checkpoints, and resume is off), so a killed life is rerun from its start.
# The source of truth for "done" is the branch: a commit "E31 worker B, stage 2: life <arm> <seed> (exit 0)" on
# origin/claude/charming-clarke-u9g850. Nothing here edits a source file or a pushed result.
# Override paths with S=<scratchpad> REPO=<main clone> if the session's scratchpad differs.
set -u
MODE=${1:-check}
S=${S:-/tmp/claude-0/-home-user-digitalbaby/029f5b11-1f0a-5a9b-9105-3bb53d7ced9b/scratchpad}
REPO=${REPO:-/home/user/digitalbaby}
WT=$REPO/.worktrees/codex
M=MOD97_LOCAL_RULE_20261106
C=$WT/$M
BR=claude/charming-clarke-u9g850
PY=/tmp/venv313/bin/python
WK=$S/e31/work
L=$S/e31/logs2
JOBS="P2a 73129
P2a 93683
P2a 116089
P2c 73129
P2c 93683
P2c 116089"
EXP16_BLOB=774813cf5c0e06010087ca74a36468c13a9f9be2      # exp16_integrated.py as at the E26 births (f4559971); the current one fails E26 admission
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
say() { echo "[recover $MODE] $*"; }
fail=0

# 1. the clone and worktree ------------------------------------------------------------------------------------------
if [ ! -d "$C" ]; then
  say "worktree $WT missing"
  if [ "$MODE" = go ]; then
    git -C "$REPO" fetch -q origin "refs/heads/$BR:refs/remotes/origin/$BR" && git -C "$REPO" worktree add -q -B e31-worker-b "$WT" "origin/$BR" && say "worktree rebuilt from origin/$BR" || { say "FAILED to rebuild worktree"; exit 2; }
  else fail=1; fi
fi

# 2. the Python environment (host of the brief: Python 3.13.18, torch 2.15.0+cpu, numpy 2.4.6) --------------------------
if ! "$PY" -c 'import sys,torch,numpy; assert sys.version.split()[0]=="3.13.18" and torch.__version__=="2.15.0+cpu" and numpy.__version__=="2.4.6"' 2>/dev/null; then
  say "venv /tmp/venv313 missing or wrong"
  if [ "$MODE" = go ]; then
    /usr/bin/python3.13 -m venv /tmp/venv313 && /tmp/venv313/bin/pip install -q torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu \
      && /tmp/venv313/bin/pip install -q numpy==2.4.6 && say "venv rebuilt" || { say "FAILED to rebuild venv"; exit 2; }
  else fail=1; fi
fi

# 3. the work folder: the clone's sources and E26 births, with exp16 at the births' version ---------------------------
need_work=0
[ -f "$WK/exp31_switch.py" ] && [ -f "$WK/exp45/e26/e26_dev_S_116089.pt" ] || need_work=1
if [ $need_work = 0 ] && [ "$(git -C "$WT" hash-object "$WK/exp16_integrated.py")" != $EXP16_BLOB ]; then need_work=1; fi
if [ $need_work = 1 ]; then
  say "work folder $WK incomplete"
  if [ "$MODE" = go ]; then
    mkdir -p "$WK/exp45/e26" "$WK/exp45/e31"
    cp "$C"/*.py "$WK"/ && cp "$C"/exp45/BIRTH_MANIFEST.json "$WK"/exp45/ && cp "$C"/exp45/e26/e26_dev_* "$WK"/exp45/e26/ \
      && git -C "$WT" cat-file blob $EXP16_BLOB > "$WK/exp16_integrated.py" && say "work folder rebuilt" || { say "FAILED to rebuild work folder"; exit 2; }
  else fail=1; fi
fi

# 5. which lives are done, running, or to be (re)run ------------------------------------------------------------------
git -C "$WT" fetch -q origin "refs/heads/$BR:refs/remotes/origin/$BR" 2>/dev/null || say "(fetch failed; using local refs)"
LOG=$(git -C "$WT" log --format=%s "origin/$BR" 2>/dev/null; git -C "$WT" log --format=%s HEAD 2>/dev/null)
todo=""
while read -r arm seed; do
  if echo "$LOG" | grep -qx "E31 worker B, stage 2: life $arm $seed (exit 0)"; then say "$arm $seed: pushed"
  elif pgrep -f "exp31_switch.py life $arm $seed\$" >/dev/null; then say "$arm $seed: RUNNING (pid $(pgrep -f "exp31_switch.py life $arm $seed\$" | tr '\n' ' '))"
  else say "$arm $seed: not done, not running"; todo="$todo$arm $seed
"; fi
done <<< "$JOBS"
sched=$(pgrep -f "bash $L/sched.sh|bash sched.sh" | tr '\n' ' ')
say "scheduler: ${sched:-not running}; PAUSE: $([ -f $L/PAUSE ] && echo present || echo absent); disk free: $(df -h --output=avail / | tail -1 | tr -d ' ')"

if [ "$MODE" != go ]; then [ -z "$todo" ] && say "nothing to restart" || say "would restart: $(echo $todo)"; exit $fail; fi
if [ -n "$sched" ]; then say "scheduler already running: not starting a second one"; exit 0; fi
if pgrep -f "exp31_switch.py life" >/dev/null; then say "lives running without a scheduler: not restarting; inspect by hand"; exit 3; fi
[ -z "$todo" ] && { say "all six lives pushed; nothing to restart"; exit 0; }

# 6a. the queue scripts, regenerated only now that nothing of the queue runs (bash reads a script as it executes it) -------------------------------------------
mkdir -p "$L"
cat > "$L/life.sh" <<EOF
#!/bin/bash
# usage: life.sh <arm> <seed>   (full dose, in the work folder; then copy into the clone and push, one push at a time)
arm=\$1; seed=\$2
cd $WK || exit 9
echo "\${arm}_\$seed start \$(date -u +%FT%TZ)" >> $L/wall.txt
$PY exp31_switch.py life \$arm \$seed > $L/e31_life_\${arm}_\${seed}.log 2>&1 < /dev/null
rc=\$?
echo "\${arm}_\$seed exit \$rc \$(date -u +%FT%TZ)" >> $L/wall.txt
(
  flock 9
  mkdir -p $C/exp45/e31/logs_b
  cp $WK/exp45/e31/e31_life_\${arm}_\${seed}.json $C/exp45/e31/ 2>/dev/null
  cp $WK/exp45/e31/e31_life_\${arm}_\${seed}.pt $WK/exp45/e31/e31_life_\${arm}_\${seed}_Bhh.pt $C/exp45/e31/ 2>/dev/null
  cp $L/e31_life_\${arm}_\${seed}.log $L/wall.txt $C/exp45/e31/logs_b/
  cd $WT && git add $M/exp45/e31 && git add -f $M/exp45/e31/e31_life_\${arm}_\${seed}*.pt 2>/dev/null
  git -c user.useConfigOnly=false commit -q -m "E31 worker B, stage 2: life \$arm \$seed (exit \$rc)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012L5f7WRCtxBkKWVj4mCZJ7" && for t in 2 4 8 16; do git push -q origin HEAD:refs/heads/$BR && break; sleep \$t; done
  echo "\${arm}_\$seed pushed \$(git log -1 --format=%h) \$(date -u +%FT%TZ)" >> $L/wall.txt
) 9>$L/push.lock
EOF
cat > "$L/sched.sh" <<EOF
#!/bin/bash
while read -r arm seed; do
  while [ -f $L/PAUSE ] || [ "\$(pgrep -fc 'venv313/bin/python (exp31_switch.py life|exp32_births.py dev|exp28_everybyte.py dev)')" -ge 3 ]; do sleep 20; done
  bash $L/life.sh \$arm \$seed &
  sleep 5
done < $L/jobs.txt
wait
echo "stage 2 queue done \$(date -u +%FT%TZ)" >> $L/wall.txt
EOF

# 6. set partial files aside, then restart the queue with the remaining lives -----------------------------------------
while read -r arm seed; do
  [ -z "$arm" ] && continue
  for f in "$WK/exp45/e31/e31_life_${arm}_${seed}.json" "$WK/exp45/e31/e31_life_${arm}_${seed}.pt" "$WK/exp45/e31/e31_life_${arm}_${seed}_Bhh.pt" "$L/e31_life_${arm}_${seed}.log"; do
    [ -f "$f" ] && mv "$f" "$f.restart_killed_$STAMP" && say "set aside $(basename $f)"
  done
done <<< "$todo"
printf '%s' "$todo" > "$L/jobs.txt"
echo "recover.sh $STAMP: restarting $(echo $todo | wc -w | awk '{print $1/2}') lives from their start: $(echo $todo)" >> "$L/wall.txt"
rm -f "$L/PAUSE"
cd "$L" && setsid nohup bash "$L/sched.sh" > "$L/sched.out" 2>&1 < /dev/null &
sleep 2
say "queue restarted: $(cat $L/jobs.txt | tr '\n' ';')"
