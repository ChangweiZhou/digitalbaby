#!/bin/bash
# E44, job L2: after a container restart (or at any time), make the host ready and continue the roster where it stands.
# Idempotent: creates the virtual environment if it is missing (Python 3.13, torch 2.15.0+cpu, numpy 2.5.3), and starts the job driver
# and the saver only if they are not running. The driver continues every unfinished life from its latest state on disk (l2_driver.py).
#   bash l2_restart.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
L2="$HERE/../exp45/e44/L2"
PY=/home/user/venv313/bin/python
if [ ! -x "$PY" ]; then
  uv venv --python python3.13 /home/user/venv313
  uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cpu 'torch==2.15.0+cpu'
  uv pip install --python "$PY" 'numpy==2.5.3'
fi
"$PY" -c "import sys, torch, numpy; print(sys.version.split()[0], torch.__version__, numpy.__version__)"
mkdir -p "$L2/logs"
cd "$HERE"
if ! pgrep -f "l2_driver.py ../exp45/e44/L2/roster_jobs.json" > /dev/null; then
  if pgrep -f "sw_run.py" > /dev/null; then echo "lives are running without a driver; not starting a second driver"; else
    (setsid nohup "$PY" l2_driver.py ../exp45/e44/L2/roster_jobs.json --n 3 >> "$L2/logs/roster_driver.log" 2>&1 &)
    echo "driver started"
  fi
else echo "driver is running"; fi
if ! pgrep -f "l2_lives.py saveloop" > /dev/null; then
  (setsid nohup "$PY" l2_lives.py saveloop 30 >> "$L2/logs/saver.log" 2>&1 &)
  echo "saver started"
else echo "saver is running"; fi
