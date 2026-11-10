#!/bin/zsh
# Machine lock for the owner's Mac (10 Nov audit: the earlier launchers overwrote and removed any lock that contained a fixed word).
#   source machine_lock.sh; lock_take "<what is running>" || exit 1; ...; lock_release
# The lock is created exclusively (never overwritten) and carries a token unique to this launcher; release removes it only if the token matches.
MINIFLY_LOCK=/tmp/MINIFLY_MACHINE_LOCK
lock_take() {
  LOCK_TOKEN="$(date +%Y%m%dT%H%M%S)-$$-$RANDOM"
  if ( set -o noclobber; print -r -- "$LOCK_TOKEN | Claude | $1 | $(date)" > $MINIFLY_LOCK ) 2>/dev/null; then return 0; fi
  echo "lock held: $(cat $MINIFLY_LOCK 2>/dev/null)"; return 1
}
lock_release() {
  if [ -e $MINIFLY_LOCK ] && grep -q "^$LOCK_TOKEN |" $MINIFLY_LOCK; then rm -f $MINIFLY_LOCK; echo "lock released $(date)"; else echo "lock not ours; left in place: $(cat $MINIFLY_LOCK 2>/dev/null)"; fi
}
# run_jobs <receipt file> then one line per job on stdin: "<log path>|<command>". Waits for each child and records its own exit code.
run_jobs() {
  local receipt=$1; local -a pids logs; : > $receipt
  while IFS='|' read -r log cmd; do [ -z "$cmd" ] && continue; eval "nohup $cmd > $log 2>&1 < /dev/null &"; pids+=($!); logs+=($log); done
  local bad=0
  for i in {1..${#pids}}; do wait ${pids[$i]}; rc=$?; echo "$rc ${logs[$i]}" >> $receipt; [ $rc -ne 0 ] && bad=1; done
  return $bad
}
