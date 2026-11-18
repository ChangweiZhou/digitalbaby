"""E52, the other two seeds (coordinator's decision 19 of 18 November 2026): the runs at rate 0.003 of both rules at half width with
f = 0.50 and 0.25 and at present width with f = 0.25, on seeds 73129 and 116089, at E41's dose, with exp52_resume.py (checkpoint every
4,096 updates). Receipts go to exp45/e52_seeds/, apart from the first seed's (exp52_tables.py reads exp45/e52/ without a seed key).
The runs at 0.001 they are read beside are E41's receipts of the same seeds (exp45/e41/), which exp52_rate.py reproduced exactly on
seed 93683 (check 1b).

    python3 exp52_drive3.py          # launch detached: setsid nohup <python> exp52_drive3.py > exp45/e52_seeds/logs/driver3.out 2>&1 &

At most three processes, counting every live run of exp52_rate.py or exp52_resume.py on the host. A run with a complete receipt is
skipped; a run with a checkpoint and an incomplete receipt is resumed (--resume); any other partial receipt is moved with its log to
exp45/e52_seeds/interrupted/ and the run is started again from its beginning.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp45" / "e52_seeds"
LOGS = OUT / "logs"
INTR = OUT / "interrupted"
PY = sys.executable
SEEDS = (73129, 116089)
RATE = 0.003
CASES = ((256, 50), (256, 25), (512, 25))
MAXP = 3
PAT = r"exp52_(rate|resume)\.py (run|check)"


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def note(s):
    with open(LOGS / "LAUNCHES.txt", "a") as f:
        f.write(f"{stamp()} {s}\n")
    print(stamp(), s, flush=True)


def live():
    r = subprocess.run(["pgrep", "-af", PAT], capture_output=True, text=True).stdout.splitlines()
    return [l for l in r if "pgrep" not in l]


def name(rule, H, fp, seed):
    return f"e52_{rule}_h{H}_f{fp}_lr{RATE:g}_{seed}"


def args(rule, H, fp, seed):
    return [rule, str(H), str(fp), str(seed), f"{RATE:g}", "--out", str(OUT)]


def alive(job):
    a = " run " + " ".join(args(*job)[:5]) + " "
    return any(a in l + " " for l in live())


def prepare(job):
    if alive(job):
        return "skip"
    n = name(*job); rp = OUT / f"{n}.json"; cp = OUT / f"{n}.ckpt"
    if rp.exists():
        try:
            if json.loads(rp.read_text()).get("complete"):
                return "skip"
        except Exception:      # noqa: BLE001
            pass
    if cp.exists() and rp.exists() and not (OUT / f"{n}.pt").exists():
        return "resume"
    left = [p for p in (rp, cp, OUT / f"{n}.pt", OUT / f"{n}.pt.tmp", OUT / f"{n}.ckpt.tmp", LOGS / f"{n}.log") if p.exists()]
    if left:
        INTR.mkdir(parents=True, exist_ok=True); t = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        for p in left:
            p.rename(INTR / f"{p.name}.{t}")
        note(f"moved to interrupted/: {', '.join(p.name for p in left)}")
    return "fresh"


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    note(f"driver3 begin; HEAD {head}; live: {len(live())}")
    jobs = [(rule, H, fp, seed) for seed in SEEDS for H, fp in CASES for rule in "SB"]
    todo = [(j, m) for j in jobs for m in [prepare(j)] if m != "skip"]
    procs = []
    for j, m in todo:
        while len(live()) >= MAXP:
            time.sleep(15)
        n = name(*j)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", E52_BESIDE=f"{len(live())} at start (max {MAXP - 1})")
        lf = open(LOGS / f"{n}.log", "a" if m == "resume" else "w")
        p = subprocess.Popen([PY, "exp52_resume.py", "run"] + args(*j) + (["--resume"] if m == "resume" else []),
                             cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT)
        note(f"start{' (resume)' if m == 'resume' else ''} {n} pid {p.pid} (beside: {len(live()) - 1})")
        procs.append((p, n)); time.sleep(2)
    for p, n in procs:
        note(f"exit {p.wait()} {n}")
    while any(" run " in l for l in live()):
        time.sleep(15)
    note("driver3 end")


if __name__ == "__main__":
    main()
