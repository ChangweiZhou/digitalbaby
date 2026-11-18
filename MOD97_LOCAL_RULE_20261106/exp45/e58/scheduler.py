"""E58: keeps at most three runs going, one thread each, from a list that may grow (exp45/e58/queues/all.txt, one
"arm rule fraction seed" a line, taken in order). A run whose receipt is complete is skipped. A run with an incomplete receipt and no
process (an interrupted run) has its receipt and files moved to exp45/e58/interrupted/ and is started again from its beginning.
    python3 exp45/e58/scheduler.py        (from MOD97_LOCAL_RULE_20261106; ends when the file exp45/e58/queues/STOP exists and nothing runs)
"""
import json, os, subprocess, time
from pathlib import Path

OUT = Path("exp45/e58"); Q = OUT / "queues" / "all.txt"; PY = "/home/user/e58venv/bin/python"
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
running = {}
(OUT / "logs").mkdir(exist_ok=True)


def others():
    """Runs started by something else (the first list was started with run_queue.sh)."""
    o = subprocess.run(["pgrep", "-f", f"^{PY} exp58_new_family.py run"], capture_output=True, text=True).stdout.split()
    mine = {str(p.pid) for p in running.values()}
    return [p for p in o if p not in mine]


def cmdlines():
    o = subprocess.run(["pgrep", "-af", f"^{PY} exp58_new_family.py run"], capture_output=True, text=True).stdout
    return o


while True:
    for k in [k for k, p in running.items() if p.poll() is not None]:
        del running[k]
    lines = [l.split() for l in Q.read_text().splitlines() if l.strip()] if Q.exists() else []
    for arm, rule, f, seed in lines:
        if len(running) + len(others()) >= 3:
            break
        name = f"e58_{arm}_{rule}_f{f}_{seed}"
        if name in running or f"run {arm} {rule} {f} {seed}" in cmdlines():
            continue
        rec = OUT / f"{name}.json"
        if rec.exists():
            if json.loads(rec.read_text()).get("complete"):
                continue
            d = OUT / "interrupted"; d.mkdir(exist_ok=True)
            tag = time.strftime("%Y%m%dT%H%M%S")
            for p in list(OUT.glob(f"{name}*")) + list((OUT / "logs").glob(f"{name}.log")):
                p.rename(d / f"{p.name}.{tag}")
        else:
            for p in OUT.glob(f"{name}_row*"):        # a file written before the first receipt cannot exist; guard all the same
                d = OUT / "interrupted"; d.mkdir(exist_ok=True); p.rename(d / f"{p.name}.{time.strftime('%Y%m%dT%H%M%S')}")
        log = open(OUT / "logs" / f"{name}.log", "w")
        running[name] = subprocess.Popen([PY, "exp58_new_family.py", "run", arm, rule, f, seed], stdout=log, stderr=subprocess.STDOUT, env=env)
    if (OUT / "queues" / "STOP").exists() and not running:
        break
    time.sleep(5)
