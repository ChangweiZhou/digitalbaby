#!/bin/bash
# E60, the four lives of `sweep-after-naive`: after a container restart (or at any time), make the host ready and continue the four lives
# where they stand. Idempotent: creates the virtual environment if it is missing (Python 3.13, torch 2.15.0+cpu, numpy 2.5.3), and starts
# the job driver and the saver only if they are not running. The driver (the main worker's l2_driver.py, unchanged) continues every
# unfinished life from its latest state on disk with `life --resume`.
#   bash l2_sweep_restart.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SW="$HERE/../exp45/e44/L2/sweep_after_naive"
PY=/home/user/venv/bin/python
if [ ! -x "$PY" ]; then
  uv venv --python python3.13 /home/user/venv
  uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cpu 'torch==2.15.0+cpu'
  uv pip install --python "$PY" 'numpy==2.5.3'
fi
"$PY" -c "import sys, torch, numpy; print(sys.version.split()[0], torch.__version__, numpy.__version__)"
mkdir -p "$SW/logs"
cd "$HERE"
if ! pgrep -f "l2_driver.py ../exp45/e44/L2/sweep_after_naive/san_jobs.json" > /dev/null; then
  if pgrep -f "sw_run.py" > /dev/null; then echo "lives are running without a driver; not starting a second driver"; else
    (setsid nohup "$PY" l2_driver.py ../exp45/e44/L2/sweep_after_naive/san_jobs.json --n 3 >> "$SW/logs/san_driver.log" 2>&1 &)
    echo "driver started"
  fi
else echo "driver is running"; fi
if ! pgrep -f "l2_san.py saveloop" > /dev/null; then
  (setsid nohup "$PY" l2_san.py saveloop 25 >> "$SW/logs/saver.log" 2>&1 &)
  echo "saver started"
else echo "saver is running"; fi
