#!/bin/zsh
cd "$(dirname "$0")/../.."
export E11_TRACE=scan
for s in 73129 93683 116089; do
  E11_WD=0.25 python3 exp11_cell.py MT $s > exp45/logs/e11d_wd025_$s.log 2>&1 &
  E11_REPLAY=fresh python3 exp11_cell.py MT $s > exp45/logs/e11d_fr_$s.log 2>&1 &
  E11_SCHEDULE=interleaved python3 exp11_cell.py MT $s > exp45/logs/e11d_int_$s.log 2>&1 &
  E11_WD=0.25 E11_REPLAY=fresh python3 exp11_cell.py MT $s > exp45/logs/e11d_wd025fr_$s.log 2>&1 &
done
wait
echo E11D_DONE
