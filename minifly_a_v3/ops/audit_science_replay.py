"""Run the predeclared science replay audit without changing the locked experiment.

The sample is fixed by SPEC_LOCK.md: all 13 arms, worlds 190001--190004.
Only the frozen W-branch replay functions are used; no science receipt is created,
replaced, or rerun, and no probe accuracy is calculated. Results are write-once,
input/source/runtime-keyed evidence under results/recovery/science_replay.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import resource
import sys
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARMS = ("R0", "R1", "R1_rand", "R0_signed", "R3", "R3_randtarget",
        "Z0_resource", "Z2", "Z2_rand", "P0", "P1", "P2", "P4")
WORLDS = (190001, 190002, 190003, 190004)
DEPENDENCIES = {"R1_rand": "R1", "Z2_rand": "Z2"}
SIGNED = ("R0_signed", "R3", "R3_randtarget")
SCHEMA = "MINIFLY-A3-SCIENCE-REPLAY-RECOVERY-v1"
OUTPUT = ROOT / "results" / "recovery" / "science_replay"
SCOPE = {"specification": "SPEC_LOCK.md:148-152", "branch": "W", "records": 600,
         "Z_and_P_store": 0, "R_novelty": "private pre-answer codes",
         "signed_R_shared_values": "all four shared stores, all 600 records",
         "P_concentration_checkpoints": [336, 600], "probe_recalculation": False}


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context() -> dict:
    import lock
    locked = lock.verify_lock()
    # Match runner.source_hashes without importing the learner implementation.
    source = lock.code_files()
    for name in ("package/MANIFEST.json", "SPEC_LOCK.md", "ARM_ROSTER.json",
                 "results/calibration/R_CALIBRATION.json"):
        source[name] = locked["files"][name]
    return {"lock_digest": locked["lock_digest"], "locked_environment": locked["environment"],
            "receipt_source_sha256": source, "wrapper_sha256": file_hash(Path(__file__)),
            "runtime": {"python_build": sys.version, "platform": platform.platform()},
            "scope": SCOPE}


def receipt_path(arm: str, world: int) -> Path:
    return ROOT / "results" / "science" / arm / f"{world}.json.gz"


def load(arm: str, world: int, ctx: dict) -> tuple[dict, str]:
    raw = receipt_path(arm, world).read_bytes()
    receipt = json.loads(gzip.decompress(raw))
    expected = {"schema": "MINIFLY-A3-CLAUDE-RECEIPT-v1", "arm": arm,
                "world": world, "kind": "science", "lock_digest": ctx["lock_digest"],
                "source_sha256": ctx["receipt_source_sha256"]}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise AssertionError(f"receipt identity or source mismatch: {arm}/{world}")
    return receipt, hashlib.sha256(raw).hexdigest()


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, sort_keys=True, indent=1, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checked_cache(path: Path, identity: dict, key: str) -> dict | None:
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    expected_keys = {"schema", "key", "identity", "pass", "checks", "resources", "error"}
    if (set(report) != expected_keys or report["schema"] != SCHEMA or report["key"] != key
            or report["identity"] != identity or type(report["pass"]) is not bool):
        raise AssertionError(f"invalid cached audit record: {path}")
    if report["pass"] and (report["error"] is not None or not report["checks"]):
        raise AssertionError(f"incomplete cached audit record: {path}")
    if report["pass"]:
        checks = report["checks"]
        arm, world = identity["arm"], identity["world"]
        log = checks["independent_log_audit"]
        reconstructed = checks["independent_replay"]
        if (log.get("pass") is not True or log.get("arm") != arm or log.get("world") != world
                or reconstructed.get("arm") != arm):
            raise AssertionError(f"cached audit identity/scope mismatch: {path}")
        if arm.startswith("P"):
            complete = reconstructed.get("p_updates_checked") == 600 and reconstructed.get("pd_checkpoints_checked") == 2
        elif arm.startswith("Z"):
            complete = (reconstructed.get("branch") == "W" and reconstructed.get("store") == 0
                        and isinstance(reconstructed.get("z_events_checked"), int)
                        and reconstructed["z_events_checked"] > 0)
        else:
            complete = reconstructed.get("novelty_checked") == 600
            if arm in SIGNED:
                complete = complete and reconstructed.get("shared_values", {}).get("records_checked") == 600
        if not complete:
            raise AssertionError(f"incomplete cached audit scope: {path}")
    return report


def replay(receipt: dict, dependency: dict | None, cal: dict) -> dict:
    import audit_a
    import audit_replay
    arm = receipt["arm"]
    kwargs = {"theta": cal["theta"], "scales": (cal["scales"]["shared"], cal["scales"]["private"])}
    if dependency is not None:
        audit_a.audit_receipt(dependency, **kwargs)
    if arm == "R1_rand":
        kwargs["r1_receipt"] = dependency
    elif arm == "Z2_rand":
        kwargs["z2_receipt"] = dependency
    log_audit = audit_a.audit_receipt(receipt, **kwargs)
    if arm in audit_a.Z_ARMS:
        result = audit_replay.replay_z(receipt, dependency if arm == "Z2_rand" else None)
        expected = sum(row[0] == "Z" and row[2] == 0 for row in receipt["mechanism_events"]["W"])
        assert result["z_events_checked"] == expected, "incomplete Z event replay"
    elif arm in audit_a.P_ARMS:
        result = audit_replay.replay_p(receipt, require_pd=True)
        assert result["p_updates_checked"] == 600, "incomplete P update replay"
        assert result["pd_checkpoints_checked"] == 2, "incomplete P concentration replay"
    else:
        result = audit_replay.replay_r_novelty(receipt)
        assert result["novelty_checked"] == 600, "incomplete R novelty replay"
        if arm in SIGNED:
            result["shared_values"] = audit_replay.replay_r_shared(receipt, cal["scales"]["shared"], limit=600)
            assert result["shared_values"]["records_checked"] == 600, "incomplete R shared-value replay"
    return {"independent_log_audit": log_audit, "independent_replay": result}


def audit_one(item: tuple[str, int], ctx: dict) -> dict:
    arm, world = item
    started = time.monotonic()
    # Input failures stop the wrapper. They never produce a passing cache entry.
    receipt, sha = load(arm, world, ctx)
    dependency = None
    hashes = {f"{arm}/{world}": sha}
    if arm in DEPENDENCIES:
        dependency, dep_sha = load(DEPENDENCIES[arm], world, ctx)
        hashes[f"{DEPENDENCIES[arm]}/{world}"] = dep_sha
        declared = receipt.get("params", {}).get(f"paired_{DEPENDENCIES[arm]}_receipt_sha256")
        if declared != dep_sha:
            raise AssertionError(f"declared yoke hash differs from its paired receipt: {arm}/{world}")
    identity = {**ctx, "arm": arm, "world": world, "receipt_sha256": hashes}
    key = digest(identity)
    path = OUTPUT / arm / f"{world}.{key}.json"
    cached = checked_cache(path, identity, key)
    if cached is not None:
        return {"arm": arm, "world": world, "key": key, "pass": cached["pass"],
                "cached": True, "record": str(path.relative_to(ROOT)), "record_sha256": file_hash(path)}
    cal = json.loads((ROOT / "results" / "calibration" / "R_CALIBRATION.json").read_text())
    result, error = None, None
    try:
        result = replay(receipt, dependency, cal)
        # Detect unexpected concurrent input changes before accepting evidence.
        assert context() == ctx, "audit source/runtime changed during replay"
        for name, expected in hashes.items():
            dependency_arm, dependency_world = name.split("/")
            assert file_hash(receipt_path(dependency_arm, int(dependency_world))) == expected, "receipt changed during replay"
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    report = {"schema": SCHEMA, "key": key, "identity": identity, "pass": error is None,
              "checks": result, "error": error,
              "resources": {"elapsed_s": time.monotonic() - started,
                            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                            "finished_unix_s": time.time()}}
    write_once(path, report)
    return {"arm": arm, "world": world, "key": key, "pass": report["pass"],
            "cached": False, "record": str(path.relative_to(ROOT)), "record_sha256": file_hash(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=1,
                        help="audit-only concurrency; default 1, run separately from science workers")
    args = parser.parse_args()
    ctx = context()
    sample = [(arm, world) for world in WORLDS for arm in ARMS]
    records = []
    if args.workers == 1:
        for item in sample:
            record = audit_one(item, ctx)
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            if not record["pass"]:
                return 1
    else:
        # Bounded batches: at most four independent audits resident, no eager
        # queue of all receipts; a failure prevents dispatch of the next batch.
        with ProcessPoolExecutor(args.workers) as executor:
            for start in range(0, len(sample), args.workers):
                futures = [executor.submit(audit_one, item, ctx) for item in sample[start:start + args.workers]]
                batch = [future.result() for future in futures]
                records.extend(batch)
                for record in batch:
                    print(json.dumps(record, sort_keys=True), flush=True)
                if not all(record["pass"] for record in batch):
                    return 1
    assert len(records) == 52 and context() == ctx
    for record in records:
        assert file_hash(ROOT / record["record"]) == record["record_sha256"], "audit record changed"
        saved = json.loads((ROOT / record["record"]).read_text())
        for name, expected in saved["identity"]["receipt_sha256"].items():
            arm, world = name.split("/")
            assert file_hash(receipt_path(arm, int(world))) == expected, "receipt changed before summary"
    summary = {"schema": SCHEMA + "-SUMMARY", "context": ctx, "pass": True,
               "sample_arms": list(ARMS), "sample_worlds": list(WORLDS), "records": records,
               "scope_limit": "W-branch mechanism reconstruction only; not all-store/all-branch/probe replay"}
    # Cache-hit flags are operational only. Use a new write-once summary per run.
    summary_path = OUTPUT / f"SUMMARY.{uuid.uuid4().hex}.json"
    write_once(summary_path, summary)
    print(json.dumps({"pass": True, "audited": len(records),
                      "summary": str(summary_path.relative_to(ROOT)), "summary_sha256": file_hash(summary_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
