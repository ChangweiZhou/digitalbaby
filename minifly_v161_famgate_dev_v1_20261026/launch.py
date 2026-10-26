"""V161 launch wrapper (V155 wrapper, unchanged rules): emulates the research-loop kit's launch rules for this legacy-scope delivery.

python launch.py run --stage {testbed|technical|dev|rehearsal|official} -- <cmd ...>
python launch.py close --kind {testbed|technical|dev|rehearsal} --decision {continue|revise|stop|escalate} --reason "..."

Rules enforced at every run (the child is also killed by RLIMIT_CPU if it would pass the CPU cap): the stage is granted in AUTHORIZATION.json; LOCK.json exists and every locked hash matches;
no attempt of another kind while a cycle is open; test cycles (technical, dev, rehearsal) within max_cycles; cumulative
child CPU and learner records within the caps. Every attempt appends one row to runlog/ATTEMPTS.jsonl (append-only)
with its argv, exit code, measured child CPU, learner records, lock sha and the result files it wrote.
"""
import argparse
import datetime
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNLOG = HERE / "runlog"
COUNTED = ("technical", "dev", "rehearsal")
GRANT = {"testbed": "technical", "technical": "technical", "dev": "dev_cycles", "rehearsal": "dev_cycles"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rows(name):
    p = RUNLOG / name
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def append(name, row):
    RUNLOG.mkdir(exist_ok=True)
    with open(RUNLOG / name, "a") as f:
        f.write(json.dumps(row) + "\n")


def open_cycle(attempts, cycles):
    """Kind and ids of attempts not covered by any close (an open cycle), or (None, [])."""
    covered = {i for c in cycles for i in c["attempts"]}
    left = [a for a in attempts if a["attempt_id"] not in covered]
    return (left[0]["stage"], [a["attempt_id"] for a in left]) if left else (None, [])


def check_lock():
    lock = json.loads((HERE / "LOCK.json").read_text())
    bad = [rel for rel, d in lock["closure_sha256"].items() if sha(ROOT / rel) != d]
    if bad:
        raise SystemExit(f"launch: LOCK mismatch, rebuild and commit the lock first: {bad}")
    return sha(HERE / "LOCK.json")


def run(stage, cmd):
    auth = json.loads((HERE / "AUTHORIZATION.json").read_text())
    granted = auth["stages"]["official"]["granted"] if stage == "official" else auth["stages"][GRANT[stage]]
    if not granted:
        raise SystemExit(f"launch: stage {stage} is not granted in AUTHORIZATION.json")
    lock_sha = check_lock()
    attempts, cycles = rows("ATTEMPTS.jsonl"), rows("CYCLES.jsonl")
    kind, _ = open_cycle(attempts, cycles)
    if kind is not None and kind != stage:
        raise SystemExit(f"launch: a {kind} cycle is open; close it before a {stage} attempt")
    if stage == "official" and kind is not None:
        raise SystemExit("launch: every test attempt must be in a closed cycle before official")
    if stage in COUNTED and kind != stage:
        used = sum(c["kind"] in COUNTED for c in cycles)
        if used >= auth["caps"]["max_cycles"]:
            raise SystemExit(f"launch: test-cycle cap {auth['caps']['max_cycles']} reached")
    cpu = sum(a["child_cpu_s"] for a in attempts) / 3600
    recs = sum(a.get("learner_records", 0) for a in attempts)
    if cpu >= auth["caps"]["max_cpu_h"] or recs >= auth["caps"]["max_records"]:
        raise SystemExit(f"launch: cap reached (CPU {cpu:.2f} h, records {recs})")
    attempt_id = f"{stage}-{len(attempts) + 1:03d}"
    env = dict(__import__("os").environ, MINIFLY_STAGE=stage, MINIFLY_ATTEMPT_ID=attempt_id)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start, t0 = utc(), time.time()
    left = max(1, int(auth["caps"]["max_cpu_h"] * 3600 - cpu * 3600))

    def limit():
        resource.setrlimit(resource.RLIMIT_CPU, (left, left + 5))
    rc = subprocess.call(cmd, cwd=HERE, env=env, preexec_fn=limit)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    child = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    res_dir = HERE / "results" / stage
    outputs = sorted(str(p.relative_to(HERE)) for p in res_dir.rglob("*") if p.is_file() and p.stat().st_mtime >= t0) \
        if res_dir.exists() else []
    recs_new = 0
    for o in outputs:
        if o.endswith(".json") and not Path(o).name.startswith("TESTBED"):
            recs_new += json.loads((HERE / o).read_text()).get("learner_records", 0) if o.endswith(".json") else 0
    row = {"attempt_id": attempt_id, "stage": stage, "argv": cmd, "utc_start": start, "utc_end": utc(),
           "exit_code": rc, "child_cpu_s": round(child, 1), "learner_records": recs_new, "lock_sha256": lock_sha,
           "outputs": outputs, "killed_by_cap": rc in (-24, -9)}
    append("ATTEMPTS.jsonl", row)
    print(json.dumps({k: row[k] for k in ("attempt_id", "exit_code", "child_cpu_s", "learner_records")}))
    return rc


def close(kind, decision, reason):
    attempts, cycles = rows("ATTEMPTS.jsonl"), rows("CYCLES.jsonl")
    open_kind, ids = open_cycle(attempts, cycles)
    if open_kind != kind:
        raise SystemExit(f"close: no open {kind} cycle (open: {open_kind})")
    locks = {a["lock_sha256"] for a in attempts if a["exit_code"] == 0 and a["outputs"]}
    row = {"kind": kind, "decision": decision, "reason": reason, "utc": utc(), "attempts": ids,
           "k_test_cycles": sum(c["kind"] in COUNTED for c in cycles) + (kind in COUNTED),
           "cumulative_cpu_h": round(sum(a["child_cpu_s"] for a in attempts) / 3600, 3),
           "cumulative_learner_records": sum(a.get("learner_records", 0) for a in attempts),
           "designs_tried_k": len(locks)}
    append("CYCLES.jsonl", row)
    print(json.dumps(row))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--stage", required=True, choices=["testbed", "technical", "dev", "rehearsal", "official"])
    r.add_argument("rest", nargs=argparse.REMAINDER)
    c = sub.add_parser("close")
    c.add_argument("--kind", required=True, choices=["testbed", "technical", "dev", "rehearsal"])
    c.add_argument("--decision", required=True, choices=["continue", "revise", "stop", "escalate"])
    c.add_argument("--reason", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
        sys.exit(run(a.stage, cmd))
    close(a.kind, a.decision, a.reason)


if __name__ == "__main__":
    main()
