"""Local-remote acceptance tests for the recovery-only Git transport."""
from __future__ import annotations

import concurrent.futures
import json
import multiprocessing as mp
import subprocess
import time
from pathlib import Path

import pytest

from connector_persistence import ConnectorEnv, atomic_json, git


@pytest.fixture
def setup(tmp_path):
    remote, repo = tmp_path / "remote.git", tmp_path / "repo"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    repo.mkdir()
    git(repo, "init", "-q", "-b", "test")
    git(repo, "config", "user.name", "Recovery transport test")
    git(repo, "config", "user.email", "test@example.invalid")
    science = repo / "minifly_a_v3/results/science"
    science.mkdir(parents=True)
    (science / "RUN_STATUS.json").write_text('{}\n')
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Initial")
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "-q", "origin", "test")
    env = ConnectorEnv(repo=repo, branch="test", spool=tmp_path / "spool", timeout_s=5)
    env.ack_poll_s = 0.01
    (science / "RUN_STATUS.json").write_text('{"validated":1}\n')
    (science / "R0").mkdir()
    (science / "R0/190001.json.gz").write_bytes(b"test bytes")
    return env, science


def service(env, mode="success"):
    deadline = time.monotonic() + 5
    while not list(env.spool.glob("*.request.json")):
        if time.monotonic() > deadline:
            raise AssertionError("no persistence request")
        time.sleep(0.01)
    request = json.loads(next(env.spool.glob("*.request.json")).read_text())
    rid = request["request_id"]
    if mode == "failure":
        atomic_json(env.spool / f"{rid}.ack.json", {"request_id": rid, "ok": False, "error": "simulated"})
        return
    tree = request["tree_sha"]
    parent = request["parent_sha"]
    if mode == "wrong_tree":
        tree = git(env.repo, "rev-parse", f"{parent}^{{tree}}")
    if mode == "wrong_parent":
        parent = git(env.repo, "commit-tree", tree, "-p", parent, "-m", "Unexpected intermediate commit")
    commit = git(env.repo, "commit-tree", tree, "-p", parent, "-m", request["message"])
    git(env.repo, "push", "-q", "origin", f"{commit}:refs/heads/test")
    if mode == "remote_advanced":
        later = git(env.repo, "commit-tree", tree, "-p", commit, "-m", "Concurrent publication")
        git(env.repo, "push", "-q", "origin", f"{later}:refs/heads/test")
    if mode == "index_mutated":
        (env.repo / "unexpected.txt").write_text("concurrent index modification")
        git(env.repo, "add", "unexpected.txt")
    if mode == "late_ack":
        time.sleep(0.1)
    atomic_json(env.spool / f"{rid}.ack.json", {"request_id": rid, "ok": True, "commit_sha": commit})


def test_exact_snapshot_roundtrip(setup):
    env, _ = setup
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(service, env)
        env.persist("one receipt")
        future.result()
    env.verify_remote()
    assert not git(env.repo, "status", "--porcelain")
    assert len(list(env.spool.glob("*.verified.json"))) == 1


@pytest.mark.parametrize("mode,match", [("failure", "persistence failed"),
                                        ("wrong_tree", "tree differs"),
                                        ("remote_advanced", "remote changed"),
                                        ("wrong_parent", "unexpected parent"),
                                        ("index_mutated", "index changed")])
def test_fail_closed(setup, mode, match):
    env, _ = setup
    initial = git(env.repo, "rev-parse", "HEAD")
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(service, env, mode)
        with pytest.raises(RuntimeError, match=match):
            env.persist("one receipt")
        future.result()
    assert git(env.repo, "rev-parse", "HEAD") == initial
    assert not list(env.spool.glob("*.verified.json"))


def test_timeout_fails_closed(setup):
    env, _ = setup
    env.timeout_s = 0.01
    with pytest.raises(RuntimeError, match="timed out"):
        env.persist("one receipt")


def test_late_ack_cannot_silently_advance_local_head(setup):
    env, _ = setup
    initial = git(env.repo, "rev-parse", "HEAD")
    env.timeout_s = 0.01
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(service, env, "late_ack")
        with pytest.raises(RuntimeError, match="timed out"):
            env.persist("one receipt")
        future.result()
    assert git(env.repo, "rev-parse", "HEAD") == initial
    assert not list(env.spool.glob("*.verified.json"))


def test_existing_receipt_cannot_be_replaced(setup):
    env, science = setup
    git(env.repo, "add", ".")
    git(env.repo, "commit", "-qm", "Existing receipt")
    git(env.repo, "push", "-q", "origin", "test")
    (science / "R0/190001.json.gz").write_bytes(b"replacement")
    with pytest.raises(RuntimeError, match="replace an existing receipt"):
        env.persist("must fail")


def test_unrelated_staged_change_rejected(setup):
    env, _ = setup
    (env.repo / "other.txt").write_text("unrelated")
    git(env.repo, "add", "other.txt")
    with pytest.raises(RuntimeError, match="unrelated staged"):
        env.persist("must fail")


@pytest.mark.parametrize("limit,category", [("rss", "budget_job"), ("deadline", "budget_job"),
                                           ("wall", "budget_exhausted"), ("worker", "budget_exhausted"),
                                           ("disk", "budget_exhausted")])
def test_watchdog_enforces_original_failure_categories(setup, limit, category):
    env, _ = setup
    class Driver:
        def __init__(self):
            self.failures = []
        def fail(self, kind, detail):
            self.failures.append((kind, detail))
    env.driver = Driver()
    hard = {"per_world_arm_life_s_max": 2400, "peak_rss_bytes_per_worker": 800000000,
            "active_wall_hours": 90, "core_hours": 360, "results_disk_bytes": 1000000000}
    baseline = {"active_wall_s": 0.0, "worker_s": 0.0}
    if limit == "rss":
        hard["peak_rss_bytes_per_worker"] = 1
    elif limit == "deadline":
        hard["per_world_arm_life_s_max"] = -1
    elif limit == "wall":
        hard["active_wall_hours"] = 0
    elif limit == "worker":
        hard["core_hours"] = 0
    else:
        hard["results_disk_bytes"] = -1
    proc = mp.get_context("fork").Process(target=time.sleep, args=(30,))
    proc.start()
    try:
        with pytest.raises(RuntimeError, match="watchdog"):
            env.watchdog([proc], time.monotonic() - 1, baseline, hard)
        assert not proc.is_alive()
        assert env.driver.failures[0][0] == category
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join()
