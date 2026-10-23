"""G3 slow-core grid (4 parallel single-thread workers, resumable)."""
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import data

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "slow"
SEEDS = (0, 1)
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")


def jobs():
    for rule, wd, ticks in itertools.product(("bptt", "eprop"), (0.0, 1.0), (0, 2)):
        for w, s in itertools.product(data.WORLDS, SEEDS):
            yield dict(rule=rule, wd=wd, ticks=ticks, world=w, seed=s)


def one(j):
    path = OUT / f"{j['rule']}_wd{j['wd']:g}_t{j['ticks']}_{j['world']}_s{j['seed']}.json"
    if path.exists():
        return f"skip {path.name}"
    tmp = path.with_suffix(".tmp")
    cmd = [sys.executable, str(HERE / "gate_slow.py"), "--world", str(j["world"]), "--seed", str(j["seed"]),
           "--rule", j["rule"], "--wd", str(j["wd"]), "--ticks", str(j["ticks"]), "--save", str(tmp)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
    if p.returncode:
        return f"FAIL {path.name}: {p.stderr[-400:]}"
    tmp.rename(path)
    return f"done {path.name} {p.stdout.strip()}"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    order = sorted(jobs(), key=lambda j: j["rule"] == "bptt")   # slower e-prop jobs first
    with ThreadPoolExecutor(4) as ex:
        for msg in ex.map(one, order):
            print(msg, flush=True)
