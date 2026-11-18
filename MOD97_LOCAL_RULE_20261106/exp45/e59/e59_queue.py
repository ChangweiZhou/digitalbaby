"""E59's queue: runs the given jobs with at most three processes at once, one thread each, detached from any wrapper.

    python3 e59_queue.py <seed> [--gate]      (run from MOD97_LOCAL_RULE_20261106/exp45/e59/, or with its path)

The jobs of a seed are the 20 runs of the brief, in this order: born n2048 S, born n2048 B, fresh n2048 S (check 4's three), then
the others. With --gate, no job after the first three starts until both born runs of 2,048 pairs have passed check 4 at row 64 (their
receipts hold check4["64"] with every entry true). A job whose receipt exists already is skipped (it was finished, or it is to be moved
to interrupted/ by hand before it is started again). Start and end times and each run's exit code go to logs/queue.log.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent            # exp45/e59
M = HERE.parents[1]                               # MOD97_LOCAL_RULE_20261106
PY = sys.executable
LOG = HERE / "logs" / "queue.log"
STARTS, NS, RULES = ("fresh", "born", "lived"), (2048, 512, 256), ("S", "B")


def jobs(seed):
    first = [("born", 2048, "S"), ("born", 2048, "B"), ("fresh", 2048, "S")]
    rest = [(s, n, r) for n in NS for s in STARTS for r in RULES if (s, n, r) not in first] + [("born-shuffled", 2048, r) for r in RULES]
    return first + rest


def log(msg):
    with LOG.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")


def gate_open(seed):
    for r in RULES:
        p = HERE / f"e59_born_n2048_{r}_{seed}.json"
        if not p.exists():
            return False
        c = json.loads(p.read_text()).get("check4", {}).get("64")
        if not c or not all(c.values()):
            return False
    return True


def main():
    seed = int(sys.argv[1]); gate = "--gate" in sys.argv
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    todo = [j for j in jobs(seed) if not (HERE / f"e59_{j[0]}_n{j[1]}_{j[2]}_{seed}.json").exists()]
    running = {}
    log(f"queue start seed {seed}: {len(todo)} jobs, gate {gate}")
    while todo or running:
        for j, (p, f) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                f.close(); del running[j]; log(f"end {j[0]} n{j[1]} {j[2]} {seed} exit={rc}")
        if (HERE / "STOP").exists():
            log(f"STOP file present: {(HERE / 'STOP').read_text().strip()}; no further job starts"); todo = []
        started_three = len(jobs(seed)) - len(todo) >= 3
        can = (not gate) or (not started_three) or gate_open(seed)
        while todo and len(running) < 3 and can:
            j = todo.pop(0)
            lf = open(HERE / "logs" / f"e59_{j[0]}_n{j[1]}_{j[2]}_{seed}.log", "a")
            p = subprocess.Popen([PY, "exp59_unrelated_starts.py", "run", j[0], str(j[1]), j[2], str(seed)], cwd=M, env=env, stdout=lf,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            running[j] = (p, lf); log(f"start {j[0]} n{j[1]} {j[2]} {seed} pid {p.pid}")
            started_three = len(jobs(seed)) - len(todo) >= 3
            can = (not gate) or (not started_three) or gate_open(seed)
        time.sleep(10)
    log(f"queue end seed {seed}")


if __name__ == "__main__":
    main()
