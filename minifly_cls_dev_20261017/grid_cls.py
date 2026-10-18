"""G4 combined-fly grid (4 parallel single-thread workers, resumable). Slow core: the best G3 cell (bptt, wd 1, 2 ticks)."""
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import data

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "cls"
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")


def one(job):
    replay, w, mem_replay = job
    tag = "" if mem_replay == 0 else f"_memreplay{mem_replay}"
    path = OUT / f"{replay}{tag}_bptt_wd1_t2_{w}_s0.json"
    if path.exists():
        return f"skip {path.name}"
    tmp = path.with_suffix(".tmp")
    cmd = [sys.executable, str(HERE / "gate_cls.py"), "--world", str(w), "--replay", replay, "--rule", "bptt",
           "--wd", "1.0", "--ticks", "2", "--mem-replay", str(mem_replay), "--save", str(tmp)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
    if p.returncode:
        return f"FAIL {path.name}: {p.stderr[-400:]}"
    tmp.rename(path)
    return f"done {path.name} {p.stdout.strip()}"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        jobs = [(r, w, 0) for r, w in itertools.product(("reservoir", "recent"), data.WORLDS)]
        jobs += [("reservoir", w, 40) for w in data.WORLDS]   # sleep replays into both systems
        for msg in ex.map(one, jobs):
            print(msg, flush=True)
