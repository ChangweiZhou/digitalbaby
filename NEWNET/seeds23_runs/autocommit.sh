# commit and push seeds23_runs logs/results every 20 min until the queue is done
cd /home/user/digitalbaby/NEWNET
while :; do
  d=0; [ -f seeds23_runs/done.flag ] && d=1
  git add seeds23_runs >/dev/null 2>&1
  git commit -qm "NEWNET seeds23_runs: logs/results so far ($(cat seeds23_runs/res_t2*.jsonl 2>/dev/null | wc -l) runs finished)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017dSEpWqUVkyw5SyyeE6M13" >/dev/null 2>&1 && \
  for i in 1 2 3 4; do git push -q origin claude/newnet-seeds23-sei232 && break; sleep $((i*10)); done
  [ $d = 1 ] && break
  sleep 1200
done
