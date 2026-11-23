#!/bin/bash
# commit and push rcseeds_runs/ logs every 20 minutes
cd /home/user/digitalbaby/NEWNET
while true; do sleep 1200
  git add rcseeds_runs net12_rcrep.py t23_life_idea12_rc.py RCSEEDS_REPORT.md 2>/dev/null
  git commit -q -m "NEWNET rcseeds: logs so far

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017dSEpWqUVkyw5SyyeE6M13" && for i in 1 2 3 4; do git push -q -u origin claude/newnet-rcseeds-sei232 && break; sleep $((i*5)); done
done
