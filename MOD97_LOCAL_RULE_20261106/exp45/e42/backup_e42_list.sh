#!/bin/sh
# Backup loop for list mode: every 15 min commit receipts, logs and the .pt weights of COMPLETED runs only, push to this host's branch
# (/root/.e42_branch). Commit trailer lines (attribution) from /root/.e42_trailer. Ends after a final pass once launch_list_<list>.log says "list end".
cd "$(dirname "$0")/../.." || exit 1
N=$(cat /root/.e42_list); B=$(cat /root/.e42_branch)
while :; do
  fin=0; grep -q 'list end' exp45/e42/logs/launch_list_$N.log 2>/dev/null && fin=1
  for lg in exp45/e42/logs/e42_*_93683.log exp45/e42/logs/e42_*_116089.log; do
    [ -f "$lg" ] && grep -q '^e42 complete' "$lg" || continue
    b=$(basename "$lg" .log); for p in exp45/e42/$b.pt exp45/e42/${b}_recurrent.pt; do [ -f "$p" ] && git add -f "$p"; done
  done
  git add exp45/e42 exp45/e42_qual
  if ! git diff --cached --quiet; then
    git commit -q -m "E42 WIP backup ($N): receipts, logs and weights of completed runs ($(date -u +%FT%TZ))

$(cat /root/.e42_trailer 2>/dev/null)"
    for i in 1 2 3 4; do git push -q -u origin $B && break; sleep $((2**i)); done
    echo "$(date -u +%FT%TZ) backup $(git log -1 --format=%h)"
  fi
  [ $fin = 1 ] && { echo "$(date -u +%FT%TZ) backup loop end"; exit 0; }
  sleep 900
done
