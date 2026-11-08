#!/bin/zsh
cd "$(dirname "$0")/../.."
export E11_TRACE=scan
for s in 73129 93683 116089; do
  E11_WD=0.25 E11_SCHEDULE=interleaved python3 exp11_cell.py MT $s > exp45/logs/e15A_int025_$s.log 2>&1 &
  E11_WD=0.25 E11_REPLAY=fresh python3 exp11_cell.py NT $s > exp45/logs/e15B_NT025fr_$s.log 2>&1 &
  ( E11_GATES=split python3 exp11_cell.py dev M $s > exp45/logs/e15C_dev_g2_$s.log 2>&1 && E11_GATES=split E11_WD=0.25 E11_REPLAY=fresh python3 exp11_cell.py MT $s > exp45/logs/e15C_MT_g2_025fr_$s.log 2>&1 ) &
done
( for s in 73129 93683 116089; do for a in ring_sign ring_conf_sign; do E10_DEV_DOSE=32768 E10_WD=0.25 python3 exp10_ring.py $a $s > exp45/logs/e15D_${a}_wd025_$s.log 2>&1; done; done ) &
wait
echo E15_DONE
