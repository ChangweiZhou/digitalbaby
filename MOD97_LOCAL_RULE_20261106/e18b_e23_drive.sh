#!/bin/bash
# E18b / E18c / E23 driver (11 Nov 2026): runs the waves in the plan's order, skips any wave already filed in exp45, re-runs an
# interrupted wave from the start (E18B_RERUN=1, Mr. Xu's instruction of 11 Nov), and commits and pushes each finished wave.
# After a container restart: run it again. Stops at the first wave that exits nonzero.
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
BR=claude/funny-mendel-34puq7
declare -A FIRST=([b2p]=e17_dev_bank2_nbd_lh_bT_tall_e18b [b2all]=e17_dev_bank2_nbd_lh_bT_tall_lossall [b3p]=e17_dev_bank2_nbd_lh_bT_e18b
  [b3all]=e17_dev_bank2_nbd_lh_bT_lossall [r2]=e17_dev_bank2_nbd_rh_bT_tall [lives_b2]=e17_life_T_bank2_nbd_lh_tall_bT_raw65536_e18c
  [lives_r2]=e17_life_T_bank2_nbd_rh_tall_bT_raw65536_e18c [h0]=e17_dev_bank2_nbd_lh_bT_tall_h1all [n1]=e17_dev_bank2_nbd_lh_bT_tall_h1all_fh1
  [f4]=e17_dev_bank2_nbd_lh_bT_tall_h1all_fh4 [f4s]=e17_dev_bank2_nbd_lh_bT_tall_h1all_fh4s)
for w in b2p b2all b3p b3all r2 lives_b2 lives_r2 h0 n1 f4 f4s; do
  done_=1; for s in 73129 93683 116089; do [ -e "exp45/${FIRST[$w]}_${s}.json" ] && [ -e "exp45/${FIRST[$w]}_${s}.pt" ] || done_=0; done
  [ $done_ = 1 ] && { echo "$w filed, skipping"; continue; }
  E18B_RERUN=1 ./e18b_e23_wave.sh $w; rc=$?
  git -C .. add MOD97_LOCAL_RULE_20261106/exp45/logs/e18b_e23_* 
  for s in 73129 93683 116089; do for e in json pt; do f="exp45/${FIRST[$w]}_${s}.$e"; [ -e "$f" ] && git -C .. add -f "MOD97_LOCAL_RULE_20261106/$f"; done; done
  git -C .. commit -qm "E18b/E18c/E23 wave $w filed (exit status $rc): receipts, weights, logs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017dSEpVADZGP7oBW3FTCHng"
  for t in 2 4 8 16 0; do git -C .. push -q origin $BR && break; [ $t = 0 ] && echo "PUSH FAILED after retries"; sleep $t; done
  [ $rc -ne 0 ] && { echo "wave $w exited $rc; stopping"; exit $rc; }
done
echo "all waves filed"
