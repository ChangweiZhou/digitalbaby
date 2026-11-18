"""E52: launches the runs of exp52_rate.py, at most three processes at once, one thread each, in the brief's order of rates.

    python3 exp52_drive.py main            # the eight at 0.001, check 1b (stop on failure), the ten at 0.003, the ten at 0.0003
    python3 exp52_drive.py further <rule> <H> <fp> [...]   # runs at 0.01 for the named cases (S 256 25 B 512 50 ...)

A run whose complete receipt exists is skipped. A partial receipt (and its log) left by an interrupted run is moved to exp45/e52/interrupted/
before the run is started again from its beginning. The driver refuses to start while any run of exp52_rate.py is alive.
Launch detached: setsid nohup <python> exp52_drive.py main > exp45/e52/logs/driver.out 2>&1 &
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp45" / "e52"
LOGS = OUT / "logs"
INTR = OUT / "interrupted"
SEED = 93683
PY = sys.executable
LONG = {256: 136704, 512: 40960}
DOSE = {256: 40960, 512: 12288}
MAXP = 3


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def note(s):
    with open(LOGS / "LAUNCHES.txt", "a") as f:
        f.write(f"{stamp()} {s}\n")
    print(stamp(), s, flush=True)


def name(rule, H, fp, rate, product):
    return f"e52_{rule}_h{H}_f{fp}_lr{rate:g}{'_product' if product else ''}_{SEED}"


def cost(job):
    rule, H, fp, rate, product = job
    return (LONG[H] if rate == 0.0003 else DOSE[H]) * (0.11 if H == 512 else 0.042)


def prepare(job):
    """True if the run is to be started; moves what an interrupted run left."""
    n = name(*job); rp = OUT / f"{n}.json"
    if rp.exists():
        try:
            if json.loads(rp.read_text()).get("complete"):
                return False
        except Exception:      # noqa: BLE001  a receipt cut off in writing is a partial receipt
            pass
    left = [p for p in (rp, OUT / f"{n}.pt", OUT / f"{n}.pt.tmp", LOGS / f"{n}.log") if p.exists()]
    if left:
        INTR.mkdir(parents=True, exist_ok=True); t = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        for p in left:
            p.rename(INTR / f"{p.name}.{t}")
        note(f"moved to interrupted/: {', '.join(p.name for p in left)}")
    return True


def stage(jobs):
    jobs = sorted([j for j in jobs if prepare(j)], key=cost, reverse=True)
    live = []; codes = {}
    while jobs or live:
        for p, j in list(live):
            c = p.poll()
            if c is not None:
                live.remove((p, j)); codes[name(*j)] = c; note(f"exit {c} {name(*j)}")
        while jobs and len(live) < MAXP:
            j = jobs.pop(0); rule, H, fp, rate, product = j; n = name(*j)
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", E52_BESIDE=f"{len(live)} at start (max {MAXP - 1})")
            cmd = [PY, "exp52_rate.py", "run", rule, str(H), str(fp), str(SEED), f"{rate:g}"] + (["--product"] if product else [])
            lf = open(LOGS / f"{n}.log", "w")
            p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT)
            live.append((p, j)); note(f"start {n} pid {p.pid} (beside: {len(live) - 1})")
            time.sleep(2)
        time.sleep(15)
    return codes


def main_jobs(rate):
    jobs = [(rule, H, fp, rate, False) for H in (256, 512) for fp in (50, 25) for rule in "BS"]
    if rate != 0.001:
        jobs += [(rule, 256, 50, rate, True) for rule in "BS"]
    return jobs


def main(a):
    LOGS.mkdir(parents=True, exist_ok=True)
    alive = subprocess.run(["pgrep", "-f", "exp52_rate.py run"], capture_output=True, text=True).stdout.split()
    if alive:
        raise SystemExit(f"runs are alive ({alive}): not starting")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    note(f"driver begin {' '.join(a)}; HEAD {head}; {os.cpu_count()} cores")
    if a[0] == "main":
        c = stage(main_jobs(0.001)); note(f"stage 0.001 done: {c}")
        q = HERE / "exp45" / "e52_qual" / "check1b.json"
        if not q.exists():
            r = subprocess.run([PY, "exp52_rate.py", "check", "1b"], cwd=HERE, env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                               capture_output=True, text=True)
            note(f"check 1b: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]}")
        if not json.loads(q.read_text())["pass"]:
            note("check 1b FAILED: no run at another rate is started"); return
        c = stage(main_jobs(0.003)); note(f"stage 0.003 done: {c}")
        c = stage(main_jobs(0.0003)); note(f"stage 0.0003 done: {c}")
    elif a[0] == "further":
        jobs = [(a[i], int(a[i + 1]), int(a[i + 2]), 0.01, False) for i in range(1, len(a), 3)]
        c = stage(jobs); note(f"stage 0.01 done: {c}")
    note("driver end")


if __name__ == "__main__":
    main(sys.argv[1:])
