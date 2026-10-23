"""Run the stage-4 transfer grid (4 parallel single-thread workers). Resumable: an existing result file is skipped."""
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "transfer"
WORLDS = (202610143901, 202610143902, 202610143903)
SEEDS = (0, 1, 2)
ARMS = ("FRESH", "MEM", "WD", "GEN", "SHUF")


def one(job):
    arm, w, s = job
    path = OUT / f"{arm}_{w}_s{s}.json"
    if path.exists():
        return f"skip {path.name}"
    tmp = path.with_suffix(".tmp")
    p = subprocess.run([sys.executable, str(HERE / "transfer.py"), "--world", str(w), "--seed", str(s),
                        "--arm", arm, "--save", str(tmp)], capture_output=True, text=True)
    if p.returncode:
        return f"FAIL {path.name}: {p.stderr[-300:]}"
    tmp.rename(path)
    return f"done {path.name} {p.stdout.strip()}"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(a, w, s) for w, s in itertools.product(WORLDS, SEEDS) for a in ARMS]
    with ThreadPoolExecutor(4) as ex:
        for msg in ex.map(one, jobs):
            print(msg, flush=True)
