"""E44, job L2 (roster note OPUS_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md, revision 2): a job driver for sw_run.py.

    python3 l2_driver.py <jobs.json> [--n 3]

jobs.json: a list of jobs, each {"id": name, "out": directory, "args": [sw_run.py arguments without --out], "stem": the life's file stem
(optional), "alone": true if the job must run with no other job (cost measurements), "after": [ids that must be done first],
"needs": [files that must exist first]}. Jobs are started in the list's order, the first that may start taking the next free place.
At most n jobs at once (default 3), each a process of its own with OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 (sw_run.py sets torch to one
thread), in a session of its own, with no time limit. A job is done when <out>/<id>.done exists (written at exit code 0) and is not
started again; a job that fails gets <out>/<id>.failed and is not retried. Logs: <out>/<id>.log (appended).

Restarts. If a job has a stem, no receipt <out>/<stem>.json exists and a state <out>/<stem>_state_v<n>.pt does, the job is continued with
`life --resume <the state with the largest n>` (the life's settings are the state's); leftover *.tmp files of that stem are removed first.
Every start is recorded in <out>/<id>.starts.jsonl with its command, so that the report can say which lives were resumed.
The driver trains nothing itself.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = [sys.executable, str(HERE / "sw_run.py")]


def latest_state(out, stem):
    best = None
    for p in Path(out).glob(f"{stem}_state_v*.pt"):
        m = re.fullmatch(re.escape(stem) + r"_state_v(\d+)\.pt", p.name)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), p)
    return best


def command(job):
    out, stem = Path(job["out"]), job.get("stem")
    if stem and not (out / f"{stem}.json").exists():
        st = latest_state(out, stem)
        if st is not None and st[0] > 0:
            for t in out.glob(f"{stem}*.tmp"):
                t.unlink()
            return ["life", "--resume", str(st[1]), "--out", str(out)], st[0]
    return list(job["args"]) + ["--out", str(out)], None


def main(argv):
    jobs = json.loads(Path(argv[0]).read_text())
    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 3
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    done = lambda j: (Path(j["out"]) / f"{j['id']}.done").exists()
    failed = lambda j: (Path(j["out"]) / f"{j['id']}.failed").exists()
    by_id = {j["id"]: j for j in jobs}
    running = []
    while True:
        for r in list(running):
            rc = r[0].poll()
            if rc is not None:
                running.remove(r)
                j = r[1]
                (Path(j["out"]) / f"{j['id']}.{'done' if rc == 0 else 'failed'}").write_text(
                    json.dumps(dict(exit=rc, wall_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))) + "\n")
                print(time.strftime("%H:%M:%S"), "done" if rc == 0 else f"FAILED ({rc})", j["id"], flush=True)
        todo = [j for j in jobs if not done(j) and not failed(j) and j["id"] not in [r[1]["id"] for r in running]]
        if not todo and not running:
            break
        for j in todo:
            if len(running) >= n or any(r[1].get("alone") for r in running):
                break
            if j.get("alone") and running:
                break
            if any(failed(by_id[a]) for a in j.get("after", [])):
                (Path(j["out"]) / f"{j['id']}.failed").write_text(json.dumps(dict(exit=None, reason="a job it follows failed")) + "\n")
                continue
            if not all(done(by_id[a]) for a in j.get("after", [])):
                continue
            if not all(Path(f).exists() for f in j.get("needs", [])):
                continue
            Path(j["out"]).mkdir(parents=True, exist_ok=True)
            args, resumed = command(j)
            with open(Path(j["out"]) / f"{j['id']}.starts.jsonl", "a") as f:
                f.write(json.dumps(dict(wall_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), resumed_from_visits=resumed,
                                        command=" ".join(["python3", "sw_run.py", *args]))) + "\n")
            log = open(Path(j["out"]) / f"{j['id']}.log", "a")
            running.append((subprocess.Popen(RUN + args, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(HERE), start_new_session=True), j))
            print(time.strftime("%H:%M:%S"), "resumed" if resumed else "started", j["id"], flush=True)
        time.sleep(2)
    print(time.strftime("%H:%M:%S"), "all jobs finished", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
