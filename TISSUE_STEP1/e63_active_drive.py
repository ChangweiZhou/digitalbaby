"""E63, decision 66: launches the runs of e63_active_runs.py, at most three processes, one thread each. A run whose receipt exists is
skipped; a run with a checkpoint is resumed. a, b, c and abcm wait for the original's state at byte 1,024 of their tissue. Child processes
are started with the environment variables beginning E11_, E16_ and EXP45_ removed.

    setsid nohup <python> e63_active_drive.py > runs_active/drive.out 2>&1 &
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs_active"
PY = sys.executable
MAXP = 3
FIRSTS = ("first", "first_withdrawbyuse", "first_f11written")
JOBS = ([(t, "orig") for t in FIRSTS + ("coincidence", "fixed")] + [(t, "abc") for t in FIRSTS + ("coincidence", "fixed")]
        + [(t, v) for t in FIRSTS + ("coincidence", "fixed") for v in ("a", "b", "c")] + [("first_withdrawbyuse", "abcm")])


def note(s):
    with open(OUT / "LAUNCHES.txt", "a") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + s + "\n")


def live():
    r = subprocess.run(["pgrep", "-af", "e63_active_runs.py run"], capture_output=True, text=True).stdout.splitlines()
    return [l for l in r if "pgrep" not in l]


def ready(t, v):
    return v in ("orig", "abc") or (OUT / t / "orig" / "state1024.npz").exists()


def main():
    OUT.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E11_", "E16_", "EXP45_"))}
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    note(f"drive begin; {len(live())} runs alive")
    todo = [j for j in JOBS if not (OUT / j[0] / j[1] / "receipt.json").exists()]
    procs = []
    while todo:
        if len(live()) >= MAXP:
            time.sleep(10); continue
        j = next((j for j in todo if ready(*j) and not any(f" run {j[0]} {j[1]}" in l for l in live())), None)
        if j is None:
            time.sleep(10); continue
        todo.remove(j)
        t, v = j
        d = OUT / t / v; d.mkdir(parents=True, exist_ok=True)
        res = (d / "checkpoint.npz").exists()
        e = dict(env, E63A_BESIDE=str(len(live())))
        p = subprocess.Popen([PY, str(HERE / "e63_active_runs.py"), "run", t, v] + (["--resume"] if res else []), cwd=HERE, env=e,
                             stdout=open(d / "log.txt", "a"), stderr=subprocess.STDOUT)
        procs.append((p, t, v))
        note(f"start {t} {v}{' (resume)' if res else ''} pid {p.pid} beside {len(live()) - 1}")
        time.sleep(2)
    for p, t, v in procs:
        note(f"exit {p.wait()} {t} {v}")
    note("drive end")


if __name__ == "__main__":
    main()
