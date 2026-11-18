"""E62: the launcher of the eighteen runs (brief revision 2). It trains nothing itself: it starts each run's own command,
`python3 exp62_connections.py run <N> all <S|B> 73129 --resume`, as a separate process, at most three at a time, in the brief's order
(N = 8, 32, 255; then 2, 128, 10; then 4, 16, 64; the sweep before the exact gradient at each N), with output appended to the run's own log.

Started again after a stop of the host, it skips complete runs; a run with a saved state continues from it (`--resume`); a run that was
stopped before its first saved state has its files moved to exp45/e62/interrupted/ and is started again. Every child gets the
environment without E11_, E16_ and EXP45_ variables (decision 37) and E62_BESIDE = the number of other runs in progress at its start.

    python3 e62_driver.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp45" / "e62"
LOGS = OUT / "logs"
SEED = 73129
ORDER = [8, 32, 255, 2, 128, 10, 4, 16, 64]
JOBS = [(N, rule) for N in ORDER for rule in "SB"]


def stem(N, rule):
    return f"e62_N{N}_all_{rule}_{SEED}"


def say(*a):
    line = time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + " ".join(str(x) for x in a)
    with open(LOGS / "driver.log", "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def complete(N, rule):
    p = OUT / f"{stem(N, rule)}.json"
    return p.exists() and json.loads(p.read_text()).get("complete") is True


def env(beside):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("E11_", "E16_", "EXP45_"))}
    e.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", E62_BESIDE=str(beside),
             PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    return e


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    running = {}; failed = []
    say("driver start")
    while True:
        for key, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                del running[key]
                say("ended", stem(*key), "exit", rc)
                if rc != 0:
                    failed.append(key)
        for N, rule in JOBS:
            if len(running) >= 3:
                break
            key = (N, rule)
            if key in running or key in failed or complete(N, rule):
                continue
            s = stem(N, rule)
            if not (OUT / f"{s}.ckpt").exists():
                left = [q for q in OUT.glob(f"{s}*") if q.is_file()]
                if left:                                     # stopped before its first saved state: move its files, start again
                    dest = OUT / "interrupted" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                    dest.mkdir(parents=True, exist_ok=True)
                    for q in left:
                        q.rename(dest / q.name)
                    say("moved the files of", s, "to", dest.relative_to(HERE), "(stopped before its first saved state); started again")
            log = open(LOGS / f"{s}.log", "a")
            cmd = ["python3", "exp62_connections.py", "run", str(N), "all", rule, str(SEED), "--resume"]
            running[key] = subprocess.Popen(cmd, cwd=HERE, env=env(len(running)), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            say("launched", s, " ".join(cmd), "beside", len(running) - 1)
        if not running:
            say("driver done; not complete:", [stem(*k) for k in JOBS if not complete(*k)], "failed:", [stem(*k) for k in failed])
            return 0
        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
