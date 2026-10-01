"""Transport-only recovery adapter for an unchanged, source-locked experiment.

The frozen driver owns dispatch, validation, budgets, and persistence frequency.
This adapter replaces only its Git push transport with a fail-closed request/ACK
exchange handled by the authorized GitHub connector. No learner code is patched.
Run from the repository with the exact SOURCE_LOCK environment.
"""
from __future__ import annotations

import json
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
    proc = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, timeout=15)
    if proc.returncode:
        raise RuntimeError(f"git {args[0]} failed: {proc.stderr[-500:]}")
    return proc.stdout.strip()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class ConnectorEnv(drive_science.Env):
    """Retain the frozen driver's scientific and resource hooks unchanged."""

    def __init__(self, *, repo=None, branch=None, spool=None, timeout_s=60.0,
                 science_path="minifly_a_v3/results/science"):
        super().__init__()
        self.repo = Path(repo) if repo is not None else ROOT.parent
        self.branch = branch or drive_science.BRANCH
        self.spool = Path(spool) if spool is not None else ROOT / "scratch" / "connector"
        self.timeout_s = timeout_s
        self.science_path = science_path
        self.ack_poll_s = 1.0

    def verify_remote(self) -> None:
        head = git(self.repo, "rev-parse", "HEAD")
        remote = git(self.repo, "ls-remote", "origin", f"refs/heads/{self.branch}").split()
        if not remote or remote[0] != head:
            raise RuntimeError(f"remote head {remote[:1]} != local head {head}")

    def persist(self, message: str) -> None:
        self.verify_remote()
        parent = git(self.repo, "rev-parse", "HEAD")
        pre_staged = git(self.repo, "diff", "--cached", "--name-only").splitlines()
        if any(not p.startswith(self.science_path + "/") for p in pre_staged):
            raise RuntimeError("refusing to include unrelated staged changes")
        git(self.repo, "add", "--", self.science_path)
        changed = git(self.repo, "diff", "--cached", "--name-status").splitlines()
        if not changed:
            return
        entries = []
        for line in changed:
            status, path = line.split("\t", 1)
            if status not in ("A", "M") or not path.startswith(self.science_path + "/"):
                raise RuntimeError(f"unexpected persistence change: {line}")
            if path.endswith(".json.gz") and status != "A":
                raise RuntimeError(f"refusing to replace an existing receipt: {path}")
            mode, sha, stage = git(self.repo, "ls-files", "-s", "--", path).split("\t")[0].split()
            if stage != "0" or mode != "100644":
                raise RuntimeError(f"unexpected staged object: {path}")
            entries.append({"path": path, "sha": sha, "mode": mode, "type": "blob"})
        tree = git(self.repo, "write-tree")
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
            if time.monotonic() > deadline:
                raise RuntimeError(f"connector persistence ACK timed out: {request_id}")
            time.sleep(self.ack_poll_s)
        ack = json.loads(ack_file.read_text())
        if ack.get("request_id") != request_id or ack.get("ok") is not True:
            raise RuntimeError(f"connector persistence failed: {ack}")
        commit = ack.get("commit_sha", "")
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise RuntimeError("invalid connector commit SHA")
        git(self.repo, "fetch", "--quiet", "origin", f"refs/heads/{self.branch}")
        fetched = git(self.repo, "rev-parse", "FETCH_HEAD")
        if fetched != commit:
            raise RuntimeError(f"remote changed before ACK verification: {fetched} != {commit}")
        if git(self.repo, "show", "-s", "--format=%P", commit) != parent:
            raise RuntimeError("connector commit has unexpected parent(s)")
        if git(self.repo, "rev-parse", f"{commit}^{{tree}}") != tree:
            raise RuntimeError("connector commit tree differs from staged snapshot")
        if git(self.repo, "write-tree") != tree:
            raise RuntimeError("local index changed during connector persistence")
        # Compare-and-swap only the local branch; the connector already performed
        # the non-force remote update. Keep the working files/index untouched.
        git(self.repo, "update-ref", "HEAD", commit, parent)
        self.verify_remote()
        atomic_json(self.spool / f"{request_id}.verified.json",
                    {"request_id": request_id, "commit_sha": commit, "tree_sha": tree, "ok": True})


if __name__ == "__main__":
    print(json.dumps(drive_science.Driver(ConnectorEnv()).run()), flush=True)
