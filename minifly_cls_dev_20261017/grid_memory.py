"""G1/G2 memory-gate grid (4 parallel workers, resumable)."""
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import data

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "memory"
FULL = dict(front="perm", homeo=1, fresh_adapt=1, teach="delta")
ARMS = {"FULL": {}, "FE0": dict(front="fe0"), "NO_HOMEO": dict(homeo=0), "NO_FRESH_ADAPT": dict(fresh_adapt=0),
        "SIGNED": dict(teach="signed"), "LOGGED": dict(teach="logged")}


def jobs():
    for arm, w in itertools.product(ARMS, data.WORLDS):
        yield "addition", arm, w
    for arm in ("FULL", "FE0"):
        yield "order", arm, data.WORLDS[0]


def one(job):
    task, arm, w = job
    cfg = dict(FULL, **ARMS[arm])
    path = OUT / f"{task}_{arm}_{w}.json"
    if path.exists():
        return f"skip {path.name}"
    tmp = path.with_suffix(".tmp")
    cmd = [sys.executable, str(HERE / "gate_memory.py"), "--world", str(w), "--task", task, "--save", str(tmp),
           "--front", cfg["front"], "--homeo", str(cfg["homeo"]), "--fresh-adapt", str(cfg["fresh_adapt"]),
           "--teach", cfg["teach"]]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        return f"FAIL {path.name}: {p.stderr[-400:]}"
    tmp.rename(path)
    return f"done {path.name} {p.stdout.strip()}"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        for msg in ex.map(one, list(jobs())):
            print(msg, flush=True)
