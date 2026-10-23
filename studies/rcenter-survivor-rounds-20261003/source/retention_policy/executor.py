# SPDX-License-Identifier: GPL-3.0-or-later
"""Minimal serialized local duplicate retirement. No remote actions or CLI.

Production invocation requires independent hash-bound acceptance and a genuine
same-controller freshly read-back checkpoint. Tests operate only in temporary roots.
"""
import fcntl
import os
from pathlib import Path
import stat
import sys
import time
import uuid

import retention as r

JOURNAL = "operations/private_official_policy/retention_journal"
MAX_INTENT_BYTES = 8 * 1024
MAX_COMPLETION_BYTES = 2 * 1024


def code_hashes():
    return {n: r.file_snapshot(Path(__file__).parent / n)["sha256"]
            for n in ("retention.py", "executor.py")}


def require_host(root):
    root = Path(root).resolve()
    state, _ = r.read_json(r.safe_path(root, "operations/HOST_WALL.json"))
    token = os.environ.get("SURVIVOR_HOST_TOKEN")
    r.check(token and state.get("active") is True and state.get("host_token") == token,
            "active original tool host required")
    r.check(state.get("boot_id") == Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "host boot changed")
    r.check(0 <= time.monotonic() - state["updated_monotonic_s"] <= 5 and
            0 <= state["effective_s"] <= 18 * 3600, "stale host or host wall cap")
    with r.opened_regular(r.safe_path(root, "operations/HOST_COORDINATOR.lock")) as keeper:
        try:
            fcntl.flock(keeper, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(keeper, fcntl.LOCK_UN)
            raise r.PolicyError("original host keeper lock is not held")
    return {k: state[k] for k in ("host_token", "boot_id")}


def require_acceptance(root, acceptance_path, launch_sha256):
    p = Path(acceptance_path)
    p = r.safe_path(root, str(p.relative_to(root)) if p.is_absolute() else str(p))
    value, sha = r.read_json(p)
    r.check(value.get("schema") == "RC-LOCAL-RETENTION-EXECUTION-ACCEPTANCE-v1" and
            value.get("accepted") is True and value.get("independent_review") is True and
            value.get("scope") == "verified_local_duplicate_retirement", "retention acceptance missing or wrong scope")
    r.check(value.get("retention_hashes") == code_hashes() and value.get("policy_sha256") == r.digest(r.POLICY) and
            value.get("launch_protection_sha256") == launch_sha256, "retention acceptance hash binding mismatch")
    return {"path": str(p.relative_to(root)), "sha256": sha, **value}


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def mkdirs(root, rel):
    current = Path(root)
    for part in Path(rel).parts:
        current /= part
        r.check(not current.is_symlink(), "receipt directory symlink")
        if not current.exists():
            current.mkdir()
            fsync_dir(current.parent)
        r.check(current.is_dir(), "receipt path is not a directory")


def atomic_receipt(directory, name, value):
    """Publish a complete create-only receipt by link, with crash-safe fsyncs."""
    directory = Path(directory)
    temp = directory / (".receipt-" + uuid.uuid4().hex + ".tmp")
    target = directory / name
    with temp.open("xb") as out:
        out.write(r.canonical(value) + b"\n")
        out.flush()
        os.fsync(out.fileno())
    try:
        os.link(temp, target, follow_symlinks=False)
        fsync_dir(directory)
    finally:
        # Only our newly generated metadata staging copy, never an archive.
        temp.unlink()
        fsync_dir(directory)
    return r.file_snapshot(target)["sha256"]


def append_event(root, event):
    mkdirs(root, JOURNAL)
    rows, _ = r.journal(root, JOURNAL)
    value = {"sequence": len(rows), "previous_sha256": rows[-1][1] if rows else None, **event}
    bound = MAX_INTENT_BYTES if event.get("event") == "retirement_intent" else MAX_COMPLETION_BYTES
    r.check(len(r.canonical(value)) + 1 <= bound, "retirement receipt size bound exceeded")
    name = f"{len(rows):06d}.json"
    sha = atomic_receipt(Path(root) / JOURNAL, name, value)
    return {"path": JOURNAL + "/" + name, "sha256": sha}


def compact_candidate(row):
    """Persist coverage commitments once, not an ever-growing repeated payload map."""
    coverage = row["coverage"]
    r.check(r.digest(coverage["required_payload"]) == coverage["required_payload_sha256"] and
            len(coverage["required_payload"]) == coverage["payload_count"], "coverage commitment changed")
    return {**row, "coverage": {k: v for k, v in coverage.items() if k != "required_payload"}}


def require_settled_journal(root):
    path = r.safe_path(root, JOURNAL)
    if not path.exists():
        return
    rows, _ = r.journal(root, JOURNAL)
    pending = set()
    for _, _, row in rows:
        if row.get("event") == "retirement_intent":
            r.check(row["retirement_id"] not in pending, "duplicate pending retirement")
            pending.add(row["retirement_id"])
        elif row.get("event") == "retirement_complete":
            r.check(row["retirement_id"] in pending, "retirement completion without intent")
            pending.remove(row["retirement_id"])
    r.check(not pending, "unfinished retirement intent requires read-only reconciliation; no automatic retry")


def verify_fresh_readback(root, checkpoint):
    state, state_sha = r.read_json(r.safe_path(root, "operations/CHECKPOINT_STATE.json"))
    r.check(checkpoint == state and state.get("status") == "private_backup_readback_verified", "same-session checkpoint acknowledgement changed")
    path = r.safe_path(root, state["readback_path"])
    r.check(path.parent == Path(root) / "backups/readback" or path.parent == Path(root) / "backups/reconcile-readback",
            "readback must be an explicit local materialization")
    snapshot = r.file_snapshot(path)
    r.check(snapshot["sha256"] == state["sha256"], "current readback bytes changed")
    # Rehash and verify every payload member in the actual just-read-back file.
    entry = {**state, "local_path": state["readback_path"], "bytes": snapshot["bytes"]}
    r.archive_payload(root, entry)
    return {"checkpoint_state_sha256": state_sha, "path": state["readback_path"], **snapshot}


def execute_current(root, *, checkpoint, acceptance_path, launch_manifest,
                    launch_sha256, expected_plan_sha256=None, extra_live_references=()):
    """Retire only a newly planned, accepted local duplicate under the original lock.

    Caller must invoke after genuine private_ack AND completed private transport
    phase in the same serialized controller. All checkpoint build/ack/restore calls
    in that controller must also use recovery.exclusive_lock; do not nest this call
    while already holding that lock. No extra remote lookup is needed.
    """
    root = Path(root).resolve()
    manifest_path = Path(launch_manifest)
    launch_manifest = r.safe_path(root, str(manifest_path.relative_to(root)) if manifest_path.is_absolute() else str(manifest_path))
    # Import only the frozen pure recovery/locking module, never a learner.
    runtime = Path(__file__).resolve().parents[1] / "runtime"
    sys.path.insert(0, str(runtime))
    import recovery
    with recovery.exclusive_lock(root) as lock:
        lock.require(root)
        host = require_host(root)
        accepted = require_acceptance(root, acceptance_path, launch_sha256)
        require_settled_journal(root)
        readback = verify_fresh_readback(root, checkpoint)
        proposal = r.plan(root, launch_manifest, launch_sha256, extra_live_references=extra_live_references)
        r.check(not proposal["blockers"], "retention plan blocked")
        if expected_plan_sha256 is not None:
            r.check(proposal["plan_sha256"] == expected_plan_sha256, "expected exact plan changed")
        # This exact immutable plan is bound into every durable intent. Independent
        # acceptance grants the bounded category; it does not authorize a stale plan.
        planned = [row for row in proposal["archives"] if row["decision"] == "proposed_local_duplicate_retirement"]
        result = {"plan_sha256": proposal["plan_sha256"], "removed": [], "removed_bytes": 0,
                  "retained_launch_originals": len(r.read_json(launch_manifest)[0]["files"])}
        for row in planned:
            lock.require(root)
            r.check(require_host(root) == host, "host changed")
            r.check(require_acceptance(root, acceptance_path, launch_sha256) == accepted, "acceptance changed")
            r.check(verify_fresh_readback(root, checkpoint) == readback, "current readback changed")
            # A fresh complete decision accounts for any newly discovered live input.
            fresh = r.plan(root, launch_manifest, launch_sha256, extra_live_references=extra_live_references)
            candidate = next((a for a in fresh["archives"] if a["path"] == row["path"]), None)
            r.check(candidate == row and not fresh["blockers"], "planned candidate changed")
            for rel, sha in proposal["evidence_input_sha256"].items():
                r.check(r.read_json(r.safe_path(root, rel))[1] == sha, "plan evidence changed")
            path = r.safe_path(root, row["path"])
            r.check(path.parent == root / "backups" and r.NAME.fullmatch(path.name), "candidate escaped exact scope")
            retirement_id = uuid.uuid4().hex
            intent = append_event(root, {"event": "retirement_intent", "retirement_id": retirement_id,
                                        "plan_sha256": proposal["plan_sha256"], "candidate": compact_candidate(row),
                                        "policy_sha256": proposal["policy_sha256"], "retention_hashes": accepted["retention_hashes"],
                                        "launch_protection_sha256": launch_sha256, "acceptance": accepted,
                                        "host": host, "fresh_readback": readback})
            # Final exact-byte/inode/type/link check occurs AFTER the intent is durable
            # and immediately before the unlink, while both writer and host guards hold.
            lock.require(root)
            r.check(require_host(root) == host, "host changed before unlink")
            snapshot = r.file_snapshot(path)
            r.check(all(snapshot[k] == row[k] for k in ("sha256", "bytes", "stat_identity")), "candidate changed after intent")
            r.check(snapshot["stat_identity"][-1] == 1 and stat.S_ISREG(path.lstat().st_mode), "unsafe candidate after intent")
            path.unlink()
            fsync_dir(path.parent)
            complete = append_event(root, {"event": "retirement_complete", "retirement_id": retirement_id,
                                          "intent": intent, "plan_sha256": proposal["plan_sha256"],
                                          "path": row["path"], "sha256": row["sha256"], "bytes": row["bytes"],
                                          "host": host, "directory_fsynced": True})
            result["removed"].append({"path": row["path"], "sha256": row["sha256"], "intent": intent, "completion": complete})
            result["removed_bytes"] += row["bytes"]
        return result
