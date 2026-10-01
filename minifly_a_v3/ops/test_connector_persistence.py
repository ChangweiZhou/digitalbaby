"""Local-remote acceptance tests for the recovery-only Git transport."""
from __future__ import annotations

import concurrent.futures
import json
import multiprocessing as mp
import subprocess
import time
from pathlib import Path

import pytest

import connector_persistence as transport
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


def test_driver_clock_returns_original_values_unchanged(setup):
    env, _ = setup
    values = iter((123.25, 124.75))
    env._driver_clock = lambda: next(values)
    assert env.last_driver_clock is None
    assert env.clock() == env.last_driver_clock == 123.25
    assert env.clock() == env.last_driver_clock == 124.75


def test_every_persistence_git_command_is_guarded_and_snapshot_preserved(setup, monkeypatch):
    env, _ = setup
    events = []
    children = [object(), object()]
    monkeypatch.setattr(transport.mp, "active_children", lambda: children)
    env._driver_clock = lambda: 123.25
    assert env.clock() == 123.25
    def watch(snapshot, started, baseline, hard):
        assert snapshot is children and len(snapshot) == 2
        assert started == 123.25
        events.append("watch")
    def tracked_git(repo, *args):
        events.append(("git", args))
        return git(repo, *args)
    monkeypatch.setattr(env, "watchdog", watch)
    monkeypatch.setattr(transport, "git", tracked_git)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(service, env)
        env.persist("guarded roundtrip")
        future.result()
    git_events = [event for event in events if isinstance(event, tuple)]
    assert len(git_events) >= 15
    assert sum(event[1][0] == "ls-remote" for event in git_events) == 2
    for i, event in enumerate(events):
        if isinstance(event, tuple):
            assert events[i - 1] == events[i + 1] == "watch"


def test_watchdog_runs_after_a_failed_git_command(setup, monkeypatch):
    env, _ = setup
    events = []
    monkeypatch.setattr(env, "watchdog", lambda *args: events.append("watch"))
    def broken_git(*args):
        events.append("git")
        raise RuntimeError("synthetic Git failure")
    monkeypatch.setattr(transport, "git", broken_git)
    with pytest.raises(RuntimeError, match="synthetic Git failure"):
        env.persist("must stop")
    assert events == ["watch", "git", "watch"]


def test_pre_persistence_validation_gap_is_included_in_budget(setup, monkeypatch):
    env, _ = setup
    failures = []
    class Driver:
        ledger = {"active_wall_s": 0.0, "worker_s": 0.0}
        def fail(self, kind, detail):
            failures.append((kind, detail))
    env.driver = Driver()
    env._driver_clock = lambda: 100.0
    assert env.clock() == 100.0
    monkeypatch.setattr(transport.time, "monotonic", lambda: 110.0)
    monkeypatch.setattr(transport.mp, "active_children", lambda: [])
    monkeypatch.setattr(env, "disk", lambda: 0)
    monkeypatch.setattr(env, "budget", lambda: {"hard_budget": {
        "active_wall_hours": 5 / 3600, "core_hours": 360, "results_disk_bytes": 10 ** 9}})
    git_called = []
    monkeypatch.setattr(transport, "git", lambda *args: git_called.append(args))
    with pytest.raises(RuntimeError, match="cumulative budget exhausted"):
        env.persist("must fail before Git")
    assert not git_called
    assert failures[0][0] == "budget_exhausted"
    assert failures[0][1]["elapsed_persistence_s"] == 10.0
    assert env.driver.ledger == {"active_wall_s": 0.0, "worker_s": 0.0}


@pytest.mark.parametrize("boot_now,expected_age", [(120.0, 40.0), (90.0, 20.0)])
def test_boottime_guard_is_conservative_without_retiming_driver(monkeypatch, boot_now, expected_age):
    monkeypatch.setattr(transport.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(transport.time, "CLOCK_BOOTTIME", 7, raising=False)
    monkeypatch.setattr(transport.time, "clock_gettime", lambda clock: boot_now)
    env = ConnectorEnv()
    assert transport.process_age_s(80.0) == expected_age
    assert env.clock() == env.last_driver_clock == 100.0


def test_boottime_unavailable_falls_back_to_original_clock(monkeypatch):
    monkeypatch.setattr(transport.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(transport.time, "CLOCK_BOOTTIME", 7, raising=False)
    def unavailable(clock):
        raise OSError("unsupported")
    monkeypatch.setattr(transport.time, "clock_gettime", unavailable)
    assert transport.process_age_s(80.0) == 20.0
