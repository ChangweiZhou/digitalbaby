"""Package A V3-CLAUDE science launcher: 13 arms x 64 worlds, write-once receipts, enforced budget, durable ledger.

Contract (SPEC_LOCK.md "Execution plan", RESOURCE_BUDGET.json):
* Starts only if SOURCE_LOCK.json verifies, the budget carries an explicit approval, and the lock/budget commit is
  acknowledged by the remote. Every existing receipt is validated (schema, arm/world/kind, lock digest, executed-closure
  provenance, independent audit incl. yoked dependencies) before it counts as committed; an invalid receipt is an
  integrity stop, never overwritten or replaced.
* RUN_LEDGER.json persists cumulative active wall seconds, worker-seconds (core-hours are enforced as worker-hours:
  the sum over world-arm jobs of dispatch-to-exit wall seconds, killed jobs included), and the failure record.
  Restart rules: an interruption (no failure recorded) resumes only uncommitted world-arms; an infrastructure
  (persistence) failure resumes after persistence is re-verified; an integrity or per-job budget stop requires a
  recorded authorization in CLEAR_AUTHORIZATION.json naming its failure id; cumulative budget exhaustion is final.
* Live checks every poll: per-job wall deadline and resident memory (job killed on breach), cumulative active wall,
  worker-hours and results disk. A killed job leaves no receipt (receipts are linked into place only when complete).
* Receipts are committed and pushed every PERSIST_EVERY accepted receipts (and at stop); a stage/commit/push failure,
  or a remote head that does not equal the local head, stops dispatch and is recorded durably.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

ROOT = paths.ROOT
REPO = ROOT.parent
ARMS = ("R1", "R0", "R0_signed", "R3", "R3_randtarget", "Z2", "Z0_resource", "Z2_rand", "R1_rand",
        "P0", "P1", "P2", "P4")
DEPS = {"R1_rand": "R1", "Z2_rand": "Z2"}
WORLDS = tuple(range(190001, 190065))
PERSIST_EVERY = 4
BRANCH = "claude/charming-clarke-u9g850"
TRAILER = ("\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_012L5f7WRCtxBkKWVj4mCZJ7")
FINAL_FAILURES = ("budget_exhausted",)
AUTH_FAILURES = ("integrity", "budget_job")


class Env:
    """Everything the driver touches outside itself; tests replace these."""

    def __init__(self, sci_dir: Path = ROOT / "results" / "science", jobs_dir: Path = ROOT / "scratch" / "jobs"):
        self.sci = sci_dir
        self.jobs = jobs_dir
        self.clock = time.monotonic
        self.sleep = time.sleep
        self.poll_s = 20.0

    # -- hooks ----------------------------------------------------------------------------------
    def lock(self) -> dict:
        import lock
        return lock.verify_lock()

    def budget(self) -> dict:
        return json.loads((ROOT / "RESOURCE_BUDGET.json").read_text())

    def job_target(self):
        return _job_entry

    def validate(self, arm, world, raw: bytes, lock: dict, deps: dict) -> None:
        validate_receipt(arm, world, raw, lock, deps)

    def persist(self, message: str) -> None:
        git_persist(message, [self.sci.relative_to(REPO).as_posix()])

    def verify_remote(self) -> None:
        git_verify_remote()

    def rss(self, pid: int) -> int:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except OSError:
            return 0
        return 0

    def disk(self) -> int:
        return sum(p.stat().st_size for p in self.sci.rglob("*") if p.is_file())


# ----------------------------------------------------------------------------- receipts
def receipt_path(sci: Path, arm: str, world: int) -> Path:
    return sci / arm / f"{world}.json.gz"


def validate_receipt(arm, world, raw: bytes, lock: dict, deps: dict) -> None:
    import audit_a
    import runner
    r = json.loads(gzip.decompress(raw))
    if (r.get("schema") != "MINIFLY-A3-CLAUDE-RECEIPT-v1" or r.get("arm") != arm or r.get("world") != world or
            r.get("kind") != "science" or r.get("lock_digest") != lock["lock_digest"]):
        raise AssertionError(f"receipt identity/lock mismatch {arm}/{world}")
    if r.get("source_sha256") != runner.source_hashes():
        raise AssertionError(f"receipt source provenance differs from the locked closure {arm}/{world}")
    cal = json.loads((ROOT / "results" / "calibration" / "R_CALIBRATION.json").read_text())
    kw = {"theta": cal["theta"], "scales": (cal["scales"]["shared"], cal["scales"]["private"])}
    if arm == "R1_rand":
        kw["r1_receipt"] = json.loads(gzip.decompress(deps["R1"]))
    if arm == "Z2_rand":
        kw["z2_receipt"] = json.loads(gzip.decompress(deps["Z2"]))
    audit_a.audit_receipt(r, **kw)


def _job_entry(arm: str, world: int, result: str) -> None:
    out = {"arm": arm, "world": world}
    try:
        import runner
        out.update(ok=True, **runner.run(arm, world, kind="science"))
    except BaseException as exc:  # noqa: BLE001
        out.update(ok=False, error=repr(exc), traceback=traceback.format_exc()[-4000:])
    tmp = Path(result + ".tmp")
    tmp.write_text(json.dumps(out))
    os.replace(tmp, result)


# ----------------------------------------------------------------------------- git persistence
def _git(*args, check=True):
    p = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.strip()[-300:]}")
    return p


def git_verify_remote() -> None:
    head = _git("rev-parse", "HEAD").stdout.strip()
    remote = _git("ls-remote", "origin", f"refs/heads/{BRANCH}").stdout.split()
    if not remote or remote[0] != head:
        raise RuntimeError(f"remote head {remote[:1]} != local head {head}")


def git_persist(message: str, rel_paths: list[str]) -> None:
    _git("add", "--", *rel_paths)
    if _git("diff", "--cached", "--quiet", check=False).returncode != 0:
        _git("commit", "-q", "-m", message + TRAILER)
    last = None
    for delay in (0, 2, 4, 8, 16):
        time.sleep(delay)
        p = _git("push", "-q", "-u", "origin", BRANCH, check=False)
        if p.returncode == 0:
            git_verify_remote()
            return
        last = p.stderr.strip()[-300:]
    raise RuntimeError(f"git push failed after retries: {last}")


# ----------------------------------------------------------------------------- driver
class Driver:
    def __init__(self, env: Env):
        self.env = env
        self.sci = env.sci
        self.ledger_path = self.sci / "RUN_LEDGER.json"
        self.status_path = self.sci / "RUN_STATUS.json"
        self.auth_path = self.sci / "CLEAR_AUTHORIZATION.json"

    def _write(self, path: Path, doc: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, path)

    def save(self) -> None:
        self._write(self.ledger_path, self.ledger)

    def fail(self, kind: str, detail: dict) -> None:
        if self.ledger["failure"] is None:
            self.ledger["failure"] = {"type": kind, "id": uuid.uuid4().hex, "detail": detail,
                                      "at_active_wall_s": self.ledger["active_wall_s"]}
            self.save()
            print(json.dumps({"STOP": self.ledger["failure"]}, default=str), flush=True)

    # startup ------------------------------------------------------------------------------------
    def load_ledger(self, lock: dict) -> None:
        if self.ledger_path.exists():
            self.ledger = json.loads(self.ledger_path.read_text())
            if self.ledger.get("lock_digest") != lock["lock_digest"]:
                raise SystemExit("RUN_LEDGER.json belongs to a different lock; refusing to start")
        else:
            self.ledger = {"schema": "MINIFLY-A3-CLAUDE-RUN-LEDGER-v1", "lock_digest": lock["lock_digest"],
                           "active_wall_s": 0.0, "worker_s": 0.0, "failure": None, "cleared": [],
                           "sessions": [], "persisted_receipts": 0}
        f = self.ledger["failure"]
        if f is None:
            return
        if f["type"] in FINAL_FAILURES:
            raise SystemExit(f"cumulative budget exhausted ({f['id']}); final, cannot resume")
        if f["type"] in AUTH_FAILURES:
            auth = json.loads(self.auth_path.read_text()) if self.auth_path.exists() else {}
            if auth.get("failure_id") != f["id"] or not auth.get("authorized_by") or not auth.get("reason"):
                raise SystemExit(f"{f['type']} stop {f['id']} requires CLEAR_AUTHORIZATION.json naming it")
            self.ledger["cleared"].append({"failure": f, "authorization": auth})
        elif f["type"] == "infrastructure":
            self.env.verify_remote()            # persistence must work again before resuming
            self.ledger["cleared"].append({"failure": f, "authorization": "persistence re-verified"})
        else:
            raise SystemExit(f"unknown failure type {f['type']}")
        self.ledger["failure"] = None
        self.save()

    def validate_existing(self, lock: dict) -> dict:
        for p in self.sci.rglob("*.json.gz.*.tmp"):
            p.unlink()                           # an interrupted job's never-linked temporary file
        done = {}
        for arm in sorted(ARMS, key=lambda a: a in DEPS):          # dependencies first
            for w in WORLDS:
                p = receipt_path(self.sci, arm, w)
                if not p.exists():
                    continue
                raw = p.read_bytes()
                deps = {}
                if arm in DEPS:
                    if (DEPS[arm], w) not in done:
                        raise AssertionError(f"{arm}/{w} exists without a valid {DEPS[arm]} dependency")
                    deps[DEPS[arm]] = receipt_path(self.sci, DEPS[arm], w).read_bytes()
                self.env.validate(arm, w, raw, lock, deps)
                done[(arm, w)] = hashlib.sha256(raw).hexdigest()
        return done

    # main ---------------------------------------------------------------------------------------
    def run(self, workers: int | None = None) -> dict:
        lock = self.env.lock()
        budget = self.env.budget()
        if budget.get("approval", {}).get("approved") is not True:
            raise SystemExit("RESOURCE_BUDGET.json is not approved; science may not start")
        hard = budget["hard_budget"]
        workers = workers or int(hard["workers"])
        self.load_ledger(lock)
        try:
            done = self.validate_existing(lock)
        except Exception as exc:  # noqa: BLE001
            self.fail("integrity", {"stage": "validate_existing", "error": repr(exc)})
            return self.finish(lock, {}, persist=False)
        try:
            self.env.verify_remote()             # lock/budget commit (and any receipts) acknowledged remotely
        except Exception as exc:  # noqa: BLE001
            self.fail("infrastructure", {"stage": "startup_verify_remote", "error": repr(exc)})
            return self.finish(lock, done, persist=False)
        self.ledger["sessions"].append({"started_wall": time.time(), "valid_receipts_at_start": len(done)})
        self.save()
        self.env.jobs.mkdir(parents=True, exist_ok=True)
        queue = [(a, w) for w in WORLDS for a in ARMS if (a, w) not in done]
        running = {}                             # item -> (process, t_start, result_path)
        since_persist = 0
        last = self.env.clock()
        ctx = mp.get_context("fork")
        while True:
            now = self.env.clock()
            dt = now - last
            last = now
            self.ledger["active_wall_s"] += dt
            self.ledger["worker_s"] += dt * len(running)
            # live budget checks ------------------------------------------------------------------
            for item, (proc, t0, _) in list(running.items()):
                over_t = now - t0 > hard["per_world_arm_life_s_max"]
                over_m = self.env.rss(proc.pid) > hard["peak_rss_bytes_per_worker"]
                if over_t or over_m:
                    proc.kill()
                    proc.join()
                    running.pop(item)
                    self.fail("budget_job", {"arm": item[0], "world": item[1],
                                             "reason": "deadline" if over_t else "rss", "killed": True})
            if (self.ledger["active_wall_s"] > hard["active_wall_hours"] * 3600 or
                    self.ledger["worker_s"] > hard["core_hours"] * 3600 or
                    self.env.disk() > hard["results_disk_bytes"]):
                for item, (proc, _, _) in list(running.items()):
                    proc.kill()
                    proc.join()
                    running.pop(item)
                self.fail("budget_exhausted", {"active_wall_s": self.ledger["active_wall_s"],
                                               "worker_s": self.ledger["worker_s"], "disk": self.env.disk()})
            # collect finished jobs ---------------------------------------------------------------
            for item, (proc, t0, res) in list(running.items()):
                if proc.is_alive():
                    continue
                proc.join()
                running.pop(item)
                arm, w = item
                out = json.loads(Path(res).read_text()) if Path(res).exists() else {
                    "ok": False, "error": f"worker exited {proc.exitcode} without a result"}
                if not out.get("ok"):
                    self.fail("integrity", {"arm": arm, "world": w, "error": out.get("error"),
                                            "traceback": out.get("traceback")})
                    continue
                try:
                    raw = receipt_path(self.sci, arm, w).read_bytes()
                    deps = {DEPS[arm]: receipt_path(self.sci, DEPS[arm], w).read_bytes()} if arm in DEPS else {}
                    self.env.validate(arm, w, raw, lock, deps)
                except Exception as exc:  # noqa: BLE001
                    self.fail("integrity", {"arm": arm, "world": w, "stage": "validate_new", "error": repr(exc)})
                    continue
                done[item] = hashlib.sha256(raw).hexdigest()
                since_persist += 1
                print(json.dumps({k: out.get(k) for k in ("arm", "world", "life_s", "peak_rss_bytes",
                                                          "receipt_bytes")}), flush=True)
            # persistence ------------------------------------------------------------------------
            if since_persist >= PERSIST_EVERY and self.ledger["failure"] is None:
                if self.persist(lock, done):
                    since_persist = 0
            # dispatch ---------------------------------------------------------------------------
            if self.ledger["failure"] is None:
                for item in list(queue):
                    if len(running) >= workers:
                        break
                    arm, w = item
                    if arm in DEPS and (DEPS[arm], w) not in done:
                        continue
                    queue.remove(item)
                    res = str(self.env.jobs / f"{arm}_{w}.json")
                    if Path(res).exists():
                        Path(res).unlink()
                    proc = ctx.Process(target=self.env.job_target(), args=(arm, w, res), daemon=True)
                    proc.start()
                    running[item] = (proc, self.env.clock(), res)
            self.save()
            self.status(lock, done, running)
            if not running and (self.ledger["failure"] is not None or not queue):
                break
            self.env.sleep(self.env.poll_s)
        return self.finish(lock, done)

    def persist(self, lock, done) -> bool:
        self.status(lock, done, {})
        try:
            self.env.persist(f"minifly_a_v3: science receipts ({len(done)}/{len(ARMS) * len(WORLDS)} validated)")
            self.ledger["persisted_receipts"] = len(done)
            self.save()
            return True
        except Exception as exc:  # noqa: BLE001
            self.fail("infrastructure", {"stage": "persist", "error": repr(exc)})
            return False

    def status(self, lock, done, running) -> None:
        self._write(self.status_path, {"lock_digest": lock["lock_digest"], "validated_receipts": len(done),
                                       "roster": len(ARMS) * len(WORLDS), "running": sorted(map(list, running)),
                                       "failure": self.ledger["failure"],
                                       "active_wall_s": self.ledger["active_wall_s"],
                                       "worker_s": self.ledger["worker_s"]})

    def finish(self, lock, done, persist=True) -> dict:
        n = len(done)
        state = ("stopped_on_failure" if self.ledger["failure"] else ("complete" if n == len(ARMS) * len(WORLDS) else "incomplete"))
        self.status(lock, done, {})
        self._write(self.status_path, {**json.loads(self.status_path.read_text()), "state": state})
        if persist:
            try:
                self.env.persist(f"minifly_a_v3: science receipts ({n}/{len(ARMS) * len(WORLDS)} validated, {state})")
            except Exception as exc:  # noqa: BLE001
                self.fail("infrastructure", {"stage": "final_persist", "error": repr(exc)})
        return {"state": state, "validated": n, "failure": self.ledger["failure"]}


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(json.dumps(Driver(Env()).run(workers), default=str), flush=True)


if __name__ == "__main__":
    main()
