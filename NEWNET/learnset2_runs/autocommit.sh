#!/bin/bash
# commit and push the learnset2 logs every 20 minutes while lives run
cd /home/user/digitalbaby
while true; do
  sleep 1200
  git add NEWNET/learnset2_runs/ && git commit -q -m "NEWNET learnset2: logs in progress

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017dSEpWqUVkyw5SyyeE6M13" && for i in 1 2 3 4; do git push -q -u origin claude/newnet-learnset2-sei232 && break; sleep $((2**i)); done
  pgrep -f t29_learnset2.py > /dev/null || break
done
