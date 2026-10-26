"""Run one attempt of the headroom check and log it.

python attempt.py --label collect-main -- /tmp/venv_n/bin/python collect.py --tag main

Refuses to start when the logged CPU already reaches the cap (AUTHORIZATION.json). The child is killed if its own CPU
time would take the total past the cap (RLIMIT_CPU). Appends one row to runlog/ATTEMPTS.jsonl with the measured child CPU.
"""
import argparse
import datetime
import json
import resource
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "runlog" / "ATTEMPTS.jsonl"


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    cap_s = json.loads((HERE / "AUTHORIZATION.json").read_text())["granted"]["cap_cpu_h"] * 3600
    rows = [json.loads(line) for line in LOG.read_text().splitlines()] if LOG.exists() else []
    used = sum(r["child_cpu_s"] for r in rows)
    if used >= cap_s:
        sys.exit(f"cap reached: {used:.0f} s of {cap_s:.0f} s")
    left = int(cap_s - used)

    def limit():
        resource.setrlimit(resource.RLIMIT_CPU, (left, left + 5))

    start = utc()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    code = subprocess.run(cmd, cwd=HERE, preexec_fn=limit).returncode
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    row = {"n": len(rows) + 1, "label": a.label, "argv": cmd, "utc_start": start, "utc_end": utc(), "exit_code": code,
           "child_cpu_s": round(cpu, 1), "cumulative_cpu_s": round(used + cpu, 1), "cap_cpu_s": cap_s,
           "killed_by_cap": code in (-24, -9)}
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))
    sys.exit(code)


if __name__ == "__main__":
    main()
