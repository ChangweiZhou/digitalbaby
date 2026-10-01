"""Transport-only recovery adapter for an unchanged, source-locked experiment.

The frozen driver owns dispatch, validation, budgets, and persistence frequency.
This adapter replaces only its Git push transport with a fail-closed request/ACK
exchange handled by the authorized GitHub connector. No learner code is patched.
Run from the repository with the exact SOURCE_LOCK environment.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import drive_science  # noqa: E402


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, timeout=10)
    if proc.returncode:
        raise RuntimeError(f"git {args[0]} failed: {proc.stderr[-500:]}")
    return proc.stdout.strip()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def process_age_s(birth: float) -> float:
    """Conservative /proc starttime comparison, without retiming the Driver.

    Linux /proc process birth uses time since boot. BOOTTIME includes suspension;
    using it here can stop a worker earlier than the frozen monotonic deadline,
    never grant extra time. The Driver's clock and ledger remain monotonic.
    """
    now = time.monotonic()
    if hasattr(time, "CLOCK_BOOTTIME"):
        try:
            now = max(now, time.clock_gettime(time.CLOCK_BOOTTIME))
        except OSError:
            pass
    return now - birth


class ConnectorEnv(drive_science.Env):
    """Retain the frozen driver's scientific and resource hooks unchanged."""

    def __init__(self, *, repo=None, branch=None, spool=None, timeout_s=600.0,
                 science_path="minifly_a_v3/results/science"):
        super().__init__()
        self.repo = Path(repo) if repo is not None else ROOT.parent
        self.branch = branch or drive_science.BRANCH
        self.spool = Path(spool) if spool is not None else ROOT / "scratch" / "connector"
        self.timeout_s = timeout_s
        self.science_path = science_path
        self.ack_poll_s = 1.0
        self.driver = None
        self._driver_clock = self.clock
        self.last_driver_clock = None
        self.clock = self._tracked_clock

    def _tracked_clock(self):
        # Return the original clock value unchanged. At persistence the most
        # recent Driver clock read is its loop poll, before receipt validation.
        value = self._driver_clock()
        self.last_driver_clock = value
        return value

    def watchdog(self, children, started, baseline, hard) -> None:
        """Enforce the frozen limits while the driver waits for persistence.

        The driver will account the full wait on its next poll. Do not add to its
        counters here, which would double-charge time. Failures use its existing
        stop/authorization categories, never clear or relax a limit.
        """
        if self.driver is None:
            return
        elapsed = time.monotonic() - started
        for proc in children:
            if not proc.is_alive():
                continue
            try:
                stat = Path(f"/proc/{proc.pid}/stat").read_text()
                # Fields after the final ')' begin with field 3; starttime is 22.
                birth = int(stat.rsplit(")", 1)[1].split()[19]) / os.sysconf("SC_CLK_TCK")
            except FileNotFoundError:
                continue
            age = process_age_s(birth)
            reason = ("deadline" if age > hard["per_world_arm_life_s_max"] else
                      "rss" if self.rss(proc.pid) > hard["peak_rss_bytes_per_worker"] else None)
            if reason:
                proc.kill()
                proc.join()
                self.driver.fail("budget_job", {"pid": proc.pid, "reason": reason, "killed": True,
                                                 "stage": "connector_persistence_watchdog"})
                raise RuntimeError(f"persistence watchdog stopped worker {proc.pid}: {reason}")
        if (baseline["active_wall_s"] + elapsed > hard["active_wall_hours"] * 3600 or
                baseline["worker_s"] + elapsed * len(children) > hard["core_hours"] * 3600 or
                self.disk() > hard["results_disk_bytes"]):
            for proc in children:
                if proc.is_alive():
                    proc.kill()
                    proc.join()
            self.driver.fail("budget_exhausted", {"stage": "connector_persistence_watchdog",
                                                   "elapsed_persistence_s": elapsed})
            raise RuntimeError("persistence watchdog: cumulative budget exhausted")

    def verify_remote(self, *, git_call=None) -> None:
        run_git = git_call if git_call is not None else lambda *args: git(self.repo, *args)
        head = run_git("rev-parse", "HEAD")
        remote = run_git("ls-remote", "origin", f"refs/heads/{self.branch}").split()
        if not remote or remote[0] != head:
            raise RuntimeError(f"remote head {remote[:1]} != local head {head}")

    def persist(self, message: str) -> None:
        children = mp.active_children()
        started = self.last_driver_clock if self.last_driver_clock is not None else time.monotonic()
        baseline = dict(self.driver.ledger) if self.driver is not None else {}
        hard = self.budget()["hard_budget"] if self.driver is not None else {}

        def guarded_git(*args):
            self.watchdog(children, started, baseline, hard)
            try:
                return git(self.repo, *args)
            finally:
                # Check even when Git fails or times out: a budget failure must
                # retain its stricter category before infrastructure handling.
                self.watchdog(children, started, baseline, hard)

        self.verify_remote(git_call=guarded_git)
        parent = guarded_git("rev-parse", "HEAD")
        pre_staged = guarded_git("diff", "--cached", "--name-only").splitlines()
        if any(not p.startswith(self.science_path + "/") for p in pre_staged):
            raise RuntimeError("refusing to include unrelated staged changes")
        guarded_git("add", "--", self.science_path)
        changed = guarded_git("diff", "--cached", "--name-status").splitlines()
        if not changed:
            return
        entries = []
        for line in changed:
            status, path = line.split("\t", 1)
            if status not in ("A", "M") or not path.startswith(self.science_path + "/"):
                raise RuntimeError(f"unexpected persistence change: {line}")
            if path.endswith(".json.gz") and status != "A":
                raise RuntimeError(f"refusing to replace an existing receipt: {path}")
            mode, sha, stage = guarded_git("ls-files", "-s", "--", path).split("\t")[0].split()
            if stage != "0" or mode != "100644":
                raise RuntimeError(f"unexpected staged object: {path}")
            entries.append({"path": path, "sha": sha, "mode": mode, "type": "blob"})
        tree = guarded_git("write-tree")
        request_id = uuid.uuid4().hex
        request = {"schema": "MINIFLY-A3-CONNECTOR-PERSISTENCE-v1", "request_id": request_id,
                   "repository": "ChangweiZhou/digitalbaby", "branch": self.branch,
                   "parent_sha": parent, "tree_sha": tree, "entries": entries,
                   "message": message + "\n\nResume the unchanged source-locked A V3 experiment."}
        pending = self.spool / f"{request_id}.request.json"
        ack_file = self.spool / f"{request_id}.ack.json"
        atomic_json(pending, request)
        print(json.dumps({"persistence_request": str(pending), "entries": len(entries)}), flush=True)
        deadline = time.monotonic() + self.timeout_s
        while not ack_file.exists():
            self.watchdog(children, started, baseline, hard)
            if time.monotonic() > deadline:
                raise RuntimeError(f"connector persistence ACK timed out: {request_id}")
            time.sleep(self.ack_poll_s)
        ack = json.loads(ack_file.read_text())
        self.watchdog(children, started, baseline, hard)
        if ack.get("request_id") != request_id or ack.get("ok") is not True:
            raise RuntimeError(f"connector persistence failed: {ack}")
        commit = ack.get("commit_sha", "")
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise RuntimeError("invalid connector commit SHA")
        guarded_git("fetch", "--quiet", "origin", f"refs/heads/{self.branch}")
        fetched = guarded_git("rev-parse", "FETCH_HEAD")
        if fetched != commit:
            raise RuntimeError(f"remote changed before ACK verification: {fetched} != {commit}")
        if guarded_git("show", "-s", "--format=%P", commit) != parent:
            raise RuntimeError("connector commit has unexpected parent(s)")
        if guarded_git("rev-parse", f"{commit}^{{tree}}") != tree:
            raise RuntimeError("connector commit tree differs from staged snapshot")
        if guarded_git("write-tree") != tree:
            raise RuntimeError("local index changed during connector persistence")
        # Compare-and-swap only the local branch; the connector already performed
        # the non-force remote update. Keep the working files/index untouched.
        guarded_git("update-ref", "HEAD", commit, parent)
        self.verify_remote(git_call=guarded_git)
        atomic_json(self.spool / f"{request_id}.verified.json",
                    {"request_id": request_id, "commit_sha": commit, "tree_sha": tree, "ok": True})


if __name__ == "__main__":
    env = ConnectorEnv()
    driver = drive_science.Driver(env)
    env.driver = driver
    print(json.dumps(driver.run()), flush=True)
