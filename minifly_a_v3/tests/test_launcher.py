"""Synthetic acceptance tests for src/drive_science.py (launch review 2026-09-30).

The real Driver, real forked worker processes and real write-once file handling are used; the learner, lock,
receipt validator, git, clock, memory and disk readings are replaced by fakes. No science trajectory runs.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import drive_science as ds  # noqa: E402

LOCK = {"lock_digest": "L" * 64}
ARMS = ("R1", "R0", "R1_rand")
WORLDS = (1, 2)
HARD = {"workers": 2, "active_wall_hours": 10.0, "core_hours": 20.0, "per_world_arm_life_s_max": 100.0,
        "peak_rss_bytes_per_worker": 1000, "results_disk_bytes": 10 ** 9}


def fake_receipt(arm, world, lock=LOCK["lock_digest"]):
    return gzip.compress(json.dumps({"arm": arm, "world": world, "lock": lock}).encode())


def _write_receipt(sci, arm, world, raw):
    p = sci / arm / f"{world}.json.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_bytes(raw)
    os.link(tmp, p)
    tmp.unlink()


def make_job(kind):
    def job(arm, world, result):
        sci = Path(os.environ["A3_TEST_SCI"])
        with open(sci.parent / "joblog.txt", "a") as f:
            f.write(f"{arm},{world}\n")
        if kind == "hang":
            while True:
                time.sleep(0.05)
        if kind == "fail" and arm == "R0":
            out = {"ok": False, "arm": arm, "world": world, "error": "synthetic scientific failure"}
        else:
            _write_receipt(sci, arm, world, fake_receipt(arm, world))
            out = {"ok": True, "arm": arm, "world": world, "life_s": 1.0, "peak_rss_bytes": 1, "receipt_bytes": 1}
        Path(result).write_text(json.dumps(out))
    return job


class FakeEnv(ds.Env):
    def __init__(self, base: Path, *, job="ok", hard=None, approved=True, push_ok=True, remote_ok=True,
                 rss=0, disk=0):
        super().__init__(base / "science", base / "jobs")
        os.environ["A3_TEST_SCI"] = str(self.sci)
        self._job, self._hard, self._approved = job, dict(HARD, **(hard or {})), approved
        self._push_ok, self._remote_ok, self._rss, self._disk = push_ok, remote_ok, rss, disk
        self.t = 0.0
        self.clock = lambda: self.t
        self.poll_s = 20.0
        self.persists = 0
        self.sleep = self._virtual_sleep      # the base Env binds time.sleep as an instance attribute

    def _virtual_sleep(self, s):
        self.t += s
        time.sleep(0.02)

    def lock(self):
        return LOCK

    def budget(self):
        return {"approval": {"approved": self._approved}, "hard_budget": self._hard}

    def job_target(self):
        return make_job(self._job)

    def validate(self, arm, world, raw, lock, deps):
        r = json.loads(gzip.decompress(raw))
        if r["arm"] != arm or r["world"] != world or r["lock"] != lock["lock_digest"]:
            raise AssertionError("fake receipt identity/lock mismatch")
        if arm in ds.DEPS and ds.DEPS[arm] not in deps:
            raise AssertionError("missing dependency")

    def persist(self, message):
        self.persists += 1
        if not self._push_ok:
            raise RuntimeError("synthetic push failure")

    def verify_remote(self):
        if not self._remote_ok:
            raise RuntimeError("synthetic remote mismatch")

    def rss(self, pid):
        return self._rss

    def disk(self):
        return self._disk


def setup():
    ds.ARMS, ds.WORLDS, ds.DEPS = ARMS, WORLDS, {"R1_rand": "R1"}
    ds.PERSIST_EVERY = 2
    return Path(tempfile.mkdtemp(prefix="a3launch-"))


def joblog(base):
    p = base / "joblog.txt"
    return [tuple(x.split(",")) for x in p.read_text().split()] if p.exists() else []


def expect_exit(fn, text):
    try:
        fn()
    except SystemExit as exc:
        assert text in str(exc), exc
        return
    raise AssertionError(f"expected SystemExit containing {text!r}")


def test_complete_run_and_yoked_order():
    base = setup()
    out = ds.Driver(FakeEnv(base)).run()
    assert out["state"] == "complete" and out["validated"] == 6, out
    log = joblog(base)
    for w in ("1", "2"):
        assert log.index(("R1", w)) < log.index(("R1_rand", w))
    shutil.rmtree(base)


def test_unapproved_budget_refused():
    base = setup()
    expect_exit(lambda: ds.Driver(FakeEnv(base, approved=False)).run(), "not approved")
    assert joblog(base) == []
    shutil.rmtree(base)


def test_cumulative_budget_exhaustion_is_final_across_restart():
    base = setup()
    env = FakeEnv(base, hard={"active_wall_hours": 30 / 3600})
    out = ds.Driver(env).run()
    assert out["failure"]["type"] == "budget_exhausted"
    expect_exit(lambda: ds.Driver(FakeEnv(base)).run(), "final")
    shutil.rmtree(base)


def test_core_hour_and_disk_caps_enforced():
    base = setup()
    out = ds.Driver(FakeEnv(base, job="hang", hard={"core_hours": 50 / 3600})).run()
    assert out["failure"]["type"] == "budget_exhausted" and out["failure"]["detail"]["worker_s"] > 50
    shutil.rmtree(base)
    base = setup()
    out = ds.Driver(FakeEnv(base, disk=10 ** 10)).run()
    assert out["failure"]["type"] == "budget_exhausted" and out["validated"] == 0
    shutil.rmtree(base)


def test_hung_worker_hits_virtual_deadline_and_restart_needs_authorization():
    base = setup()
    out = ds.Driver(FakeEnv(base, job="hang")).run()
    f = out["failure"]
    assert f["type"] == "budget_job" and f["detail"]["reason"] == "deadline" and f["detail"]["killed"]
    assert not list((base / "science").rglob("*.json.gz")), "a killed job left a receipt"
    expect_exit(lambda: ds.Driver(FakeEnv(base)).run(), "requires CLEAR_AUTHORIZATION")
    (base / "science" / "CLEAR_AUTHORIZATION.json").write_text(json.dumps(
        {"failure_id": f["id"], "authorized_by": "test", "reason": "synthetic"}))
    out = ds.Driver(FakeEnv(base)).run()
    assert out["state"] == "complete"
    led = json.loads((base / "science" / "RUN_LEDGER.json").read_text())
    assert led["cleared"] and led["cleared"][0]["failure"]["id"] == f["id"]
    shutil.rmtree(base)


def test_rss_breach_kills_job():
    base = setup()
    out = ds.Driver(FakeEnv(base, job="hang", rss=10 ** 6)).run()
    assert out["failure"]["type"] == "budget_job" and out["failure"]["detail"]["reason"] == "rss"
    shutil.rmtree(base)


def test_scientific_failure_blocks_restart_and_wrong_authorization():
    base = setup()
    out = ds.Driver(FakeEnv(base, job="fail")).run()
    assert out["failure"]["type"] == "integrity"
    n = len(joblog(base))
    expect_exit(lambda: ds.Driver(FakeEnv(base)).run(), "requires CLEAR_AUTHORIZATION")
    (base / "science" / "CLEAR_AUTHORIZATION.json").write_text(json.dumps(
        {"failure_id": "wrong", "authorized_by": "x", "reason": "y"}))
    expect_exit(lambda: ds.Driver(FakeEnv(base)).run(), "requires CLEAR_AUTHORIZATION")
    assert len(joblog(base)) == n, "a job ran after an uncleared integrity stop"
    shutil.rmtree(base)


def test_wrong_lock_and_corrupt_receipts_rejected():
    for raw in (fake_receipt("R0", 1, lock="X" * 64), b"not gzip"):
        base = setup()
        _write_receipt(base / "science", "R0", 1, raw)
        before = hashlib.sha256((base / "science" / "R0" / "1.json.gz").read_bytes()).hexdigest()
        out = ds.Driver(FakeEnv(base)).run()
        assert out["failure"]["type"] == "integrity" and joblog(base) == []
        assert hashlib.sha256((base / "science" / "R0" / "1.json.gz").read_bytes()).hexdigest() == before
        shutil.rmtree(base)


def test_yoked_arm_without_valid_dependency_rejected():
    base = setup()
    _write_receipt(base / "science", "R1_rand", 1, fake_receipt("R1_rand", 1))
    out = ds.Driver(FakeEnv(base)).run()
    assert out["failure"]["type"] == "integrity" and "dependency" in out["failure"]["detail"]["error"]
    shutil.rmtree(base)


def test_failed_push_stops_dispatch_then_resumes_after_reverification():
    base = setup()
    out = ds.Driver(FakeEnv(base, push_ok=False)).run()
    assert out["failure"]["type"] == "infrastructure" and out["validated"] < 6
    n = len(joblog(base))
    assert n <= ds.PERSIST_EVERY + HARD["workers"], "dispatch continued after a failed push"
    try:
        ds.Driver(FakeEnv(base, remote_ok=False)).run()
        raise AssertionError("resumed an infrastructure stop without re-verified persistence")
    except RuntimeError as exc:
        assert "remote" in str(exc)
    assert len(joblog(base)) == n
    out3 = ds.Driver(FakeEnv(base)).run()
    assert out3["state"] == "complete"
    shutil.rmtree(base)


def test_interruption_resumes_only_uncommitted_and_receipts_immutable():
    base = setup()
    for arm, w in (("R1", 1), ("R0", 1), ("R1", 2)):
        _write_receipt(base / "science", arm, w, fake_receipt(arm, w))
    (base / "science" / "R0" / "2.json.gz.999.tmp").write_bytes(b"partial")   # an interrupted job's temp file
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (base / "science").rglob("*.json.gz")}
    out = ds.Driver(FakeEnv(base)).run()
    assert out["state"] == "complete"
    assert sorted(joblog(base)) == sorted([("R1_rand", "1"), ("R0", "2"), ("R1_rand", "2")])
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in before.items())
    assert not list((base / "science").rglob("*.tmp"))
    shutil.rmtree(base)


if __name__ == "__main__":
    import inspect
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f()
        print("PASS", f.__name__, flush=True)
    print(f"{len(fns)} launcher tests passed")
