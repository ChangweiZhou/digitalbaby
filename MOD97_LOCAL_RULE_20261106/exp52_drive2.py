"""E52: the driver after the worker restart of 18 November (replaces exp52_drive.py, which is kept as it ran).

    python3 exp52_drive2.py main                      # what remains of 0.003 (exp52_rate.py), check 4, then 0.0003 (exp52_resume.py if check 4 passed)
    python3 exp52_drive2.py further <rule> <H> <fp> [...]   # runs at 0.01 (exp52_resume.py if check 4 passed)

At most three processes: every live `exp52_rate.py run`, `exp52_resume.py run` or `exp52_resume.py check` on the host counts, including runs
left by a driver that died. A run with a complete receipt is skipped; a run that is alive is not started again. A run of exp52_resume.py with a
checkpoint (<receipt>.ckpt) and an incomplete receipt is resumed (--resume); any other partial receipt is moved with its log to
exp45/e52/interrupted/ and the run is started again from its beginning.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp45" / "e52"
QUAL = HERE / "exp45" / "e52_qual"
LOGS = OUT / "logs"
INTR = OUT / "interrupted"
SEED = 93683
PY = sys.executable
LONG = {256: 136704, 512: 40960}
DOSE = {256: 40960, 512: 12288}
MAXP = 3
PAT = r"exp52_(rate|resume)\.py (run|check)"


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def note(s):
    with open(LOGS / "LAUNCHES.txt", "a") as f:
        f.write(f"{stamp()} {s}\n")
    print(stamp(), s, flush=True)


def live_cmds():
    r = subprocess.run(["pgrep", "-af", PAT], capture_output=True, text=True).stdout.splitlines()
    return [l for l in r if "pgrep" not in l]


def name(rule, H, fp, rate, product):
    return f"e52_{rule}_h{H}_f{fp}_lr{rate:g}{'_product' if product else ''}_{SEED}"


def argv(job):
    rule, H, fp, rate, product = job
    return [rule, str(H), str(fp), str(SEED), f"{rate:g}"] + (["--product"] if product else [])


def is_alive(job):
    a = " ".join(argv(job)) + ("" if job[4] else "")
    return any((" run " + a) in l and (job[4] or "--product" not in l) for l in live_cmds())


def cost(job):
    rule, H, fp, rate, product = job
    return (LONG[H] if rate == 0.0003 else DOSE[H]) * (0.11 if H == 512 else 0.042)


def prepare(job, runner):
    """'skip', 'fresh' or 'resume'."""
    if is_alive(job):
        return "skip"
    n = name(*job); rp = OUT / f"{n}.json"; cp = OUT / f"{n}.ckpt"
    if rp.exists():
        try:
            if json.loads(rp.read_text()).get("complete"):
                return "skip"
        except Exception:      # noqa: BLE001
            pass
    if runner == "exp52_resume.py" and cp.exists() and rp.exists() and not (OUT / f"{n}.pt").exists():
        return "resume"
    left = [p for p in (rp, cp, OUT / f"{n}.pt", OUT / f"{n}.pt.tmp", OUT / f"{n}.ckpt.tmp", LOGS / f"{n}.log") if p.exists()]
    if left:
        INTR.mkdir(parents=True, exist_ok=True); t = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        for p in left:
            p.rename(INTR / f"{p.name}.{t}")
        note(f"moved to interrupted/: {', '.join(p.name for p in left)}")
    return "fresh"


def wait_slot():
    while len(live_cmds()) >= MAXP:
        time.sleep(15)


def launch(job, runner, mode):
    n = name(*job)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", E52_BESIDE=f"{len(live_cmds())} at start (max {MAXP - 1})")
    cmd = [PY, runner, "run"] + argv(job) + (["--resume"] if mode == "resume" else [])
    lf = open(LOGS / f"{n}.log", "a" if mode == "resume" else "w")
    p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT)
    note(f"start{' (resume)' if mode == 'resume' else ''} {n} with {runner} pid {p.pid} (beside: {len(live_cmds()) - 1})")
    return p


def stage(jobs, runner, extra=None):
    """extra: (label, argv) of a check to run in the first free slot."""
    todo = []
    for j in sorted(jobs, key=cost, reverse=True):
        m = prepare(j, runner)
        if m != "skip":
            todo.append((j, m))
    procs = []
    if extra:
        wait_slot()
        lf = open(QUAL / "logs" / f"{extra[0]}.log", "w")
        procs.append((subprocess.Popen(extra[1], cwd=HERE, env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                                       stdout=lf, stderr=subprocess.STDOUT), extra[0]))
        note(f"start {extra[0]}"); time.sleep(2)
    for j, m in todo:
        wait_slot(); procs.append((launch(j, runner, m), name(*j))); time.sleep(2)
    for p, n in procs:
        c = p.wait(); note(f"exit {c} {n}")
    while any(" run " in l for l in live_cmds()):    # runs left by an earlier driver
        time.sleep(15)


def main_jobs(rate):
    jobs = [(rule, H, fp, rate, False) for H in (256, 512) for fp in (50, 25) for rule in "BS"]
    if rate != 0.001:
        jobs += [(rule, 256, 50, rate, True) for rule in "BS"]
    return jobs


def check4_passed():
    q = QUAL / "check4.json"
    return q.exists() and json.loads(q.read_text())["pass"]


def main(a):
    LOGS.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    note(f"driver2 begin {' '.join(a)}; HEAD {head}; live: {len(live_cmds())}")
    if a[0] == "main":
        assert json.loads((QUAL / "check1b.json").read_text())["pass"]
        extra = None if (QUAL / "check4.json").exists() else ("check4", [PY, "exp52_resume.py", "check", "4"])
        stage(main_jobs(0.003), "exp52_rate.py", extra); note("stage 0.003 done")
        runner = "exp52_resume.py" if check4_passed() else "exp52_rate.py"
        note(f"check 4 {'passed' if runner == 'exp52_resume.py' else 'NOT passed'}: the runs at 0.0003 use {runner}")
        stage(main_jobs(0.0003), runner); note("stage 0.0003 done")
    elif a[0] == "further":
        jobs = [(a[i], int(a[i + 1]), int(a[i + 2]), 0.01, False) for i in range(1, len(a), 3)]
        stage(jobs, "exp52_resume.py" if check4_passed() else "exp52_rate.py"); note("stage 0.01 done")
    note("driver2 end")


if __name__ == "__main__":
    main(sys.argv[1:])
