# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only, fail-closed local checkpoint-copy retirement planner.

There is deliberately no unlink, delete, remote mutation, or execution entry point.
This module does not import the scientific runtime or the frozen transport layer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

CHUNK = 1024 * 1024
NAME = re.compile(r"checkpoint-([0-9a-f]{64})\.zip\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
POLICY = {
    "schema": "RC-LOCAL-CHECKPOINT-RETENTION-v1",
    "mode": "read_only_planner_with_separately_gated_executor",
    "candidate": "backups/checkpoint-<64-lowercase-hex-content-identity>.zip",
    "retain_latest_distinct_verified_versions": 4,
    "minimum_newer_verified_versions": 2,
    "protect_all_launch_originals": True,
    "require_exact_readback_and_library_metadata": True,
    "require_current_newer_scientific_payload_coverage": True,
    "require_no_live_archive_references": True,
    "backup_cap_bytes": 2 * 1024**3,
    "archive_cap_bytes": 48 * 1024**2,
    "future_execution_requires": [
        "separate_hash_bound_policy_approval_and_exact_execution_plan_binding",
        "live_host_token_boot_heartbeat_and_keeper_lock",
        "shared_exclusive_backup_writer_lock",
        "fresh_revalidation_under_lock",
        "fsynced_intent_receipt_before_unlink",
        "fsynced_backups_directory_after_unlink",
        "fsynced_completion_receipt_and_receipt_directory",
    ],
}
SCIENTIFIC_PREFIXES = ("source/", "tests/", "receipts/", "history/", "protocol/", "audits/", "results/")
LIVE_STATE_NAMES = {
    "CHECKPOINT_STATE.json", "PRIVATE_PERSISTENCE_STATE.json", "PERSISTENCE_STATE.json",
    "TRANSPORT_PENDING.json", "RESERVATION_ACK.json", "ACTIVE_JOB.json", "RESTORE_PROOF.json",
}


class PolicyError(ValueError):
    pass


def check(ok, message):
    if not ok:
        raise PolicyError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _stamp(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)


def safe_path(root, rel):
    root = Path(root).resolve()
    p = PurePosixPath(rel)
    check(isinstance(rel, str) and str(p) == rel and not p.is_absolute() and
          rel != "." and ".." not in p.parts, "unsafe relative path")
    current = root
    for part in p.parts:
        current /= part
        check(not current.is_symlink(), "symlink path rejected: " + rel)
    return current


def opened_regular(path):
    """Open a nonsymlink regular file, checking all existing ancestor components."""
    path = Path(path)
    check(not any(p.is_symlink() for p in (path, *path.parents)), "symlink rejected")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    f = os.fdopen(fd, "rb")
    if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
        f.close()
        raise PolicyError("not a regular file")
    return f


def hash_stream(f):
    h = hashlib.sha256()
    for block in iter(lambda: f.read(CHUNK), b""):
        h.update(block)
    return h.hexdigest()


def file_snapshot(path):
    with opened_regular(path) as f:
        before = os.fstat(f.fileno())
        h = hash_stream(f)
        after = os.fstat(f.fileno())
        check(_stamp(before) == _stamp(after) == _stamp(Path(path).lstat()), "file changed during read")
    return {"sha256": h, "bytes": before.st_size, "stat_identity": list(_stamp(before))}


def read_json(path):
    with opened_regular(path) as f:
        before = os.fstat(f.fileno())
        check(before.st_size <= 64 * 1024**2, "metadata size limit")
        raw = f.read()
        check(_stamp(before) == _stamp(os.fstat(f.fileno())) == _stamp(Path(path).lstat()),
              "metadata changed during read")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def create_json_once(path, value):
    """Create-only auditable metadata, never an archive; fsync file and directory."""
    path = Path(path)
    check(path.suffix == ".json", "only JSON receipt writes permitted")
    check(not any(p.is_symlink() for p in (path.parent, *path.parent.parents)), "symlink parent")
    raw = canonical(value) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return hashlib.sha256(raw).hexdigest()


def launch_protection(root):
    """Snapshot every currently present direct original ZIP, never readback copies.

    Must be captured before launch. External acceptance pins the returned manifest's
    file hash; create_json_once never overwrites it. Recheck under writer lock at launch.
    """
    root = Path(root).resolve()
    base = safe_path(root, "backups")
    names = sorted(p.name for p in base.iterdir() if p.suffix == ".zip")
    entries = {"backups/" + name: file_snapshot(base / name) for name in names}
    check(names == sorted(p.name for p in base.iterdir() if p.suffix == ".zip"), "launch inventory changed")
    for rel, entry in entries.items():
        check(file_snapshot(root / rel) == entry, "launch archive changed")
    return {"schema": "RC-IMMUTABLE-LAUNCH-PROTECTION-v1", "files": entries,
            "original_count": len(entries), "original_bytes": sum(e["bytes"] for e in entries.values()),
            "all_existing_direct_backup_zips_protected": True}


def journal(root, rel):
    path = safe_path(root, rel)
    check(path.is_dir(), "missing evidence journal: " + rel)
    rows, inputs, previous = [], {}, None
    names = sorted(path.glob("*.json"))
    for sequence, p in enumerate(names):
        check(p.name == f"{sequence:06d}.json", "noncontiguous journal")
        row, sha = read_json(p)
        check(row.get("sequence") == sequence and row.get("previous_sha256") == previous,
              "journal chain mismatch: " + rel)
        name = str(p.relative_to(root))
        inputs[name] = sha
        rows.append((name, sha, row))
        previous = sha
    check(names == sorted(path.glob("*.json")), "journal changed during scan")
    return rows, inputs


def verified_evidence(root):
    """Join a genuine readback acknowledgement to exact successful Library metadata.

    Upload success alone never counts. Historical records are proofs of readback at
    that time; execution still needs a fresh current remote check under approved host.
    """
    rows, inputs = journal(root, "operations/transport_journal")
    for rel in ("operations/private_policy/transport_journal", "operations/private_official_policy/transport_journal"):
        private = safe_path(root, rel)
        if private.exists():
            extra, hashes = journal(root, rel)
            rows += extra
            inputs.update(hashes)
    evidence = {}
    seen_versions = {}
    for source, sha, ack in rows:
        if ack.get("event") != "private_readback_verified":
            continue
        local = ack.get("local_path", "")
        match = NAME.fullmatch(Path(local).name)
        check(match is not None and local == "backups/" + Path(local).name, "bad acknowledgement path")
        check(ack.get("status") == "private_backup_readback_verified" and
              ack.get("content_identity") == match.group(1) and HEX.fullmatch(ack.get("sha256", "")) and
              type(ack.get("version")) is int and ack["version"] >= 1 and
              isinstance(ack.get("library_file_id"), str) and ack["library_file_id"].startswith("libfile_") and
              isinstance(ack.get("file_id"), str) and ack["file_id"].startswith("file_") and
              ack.get("file_name") == Path(local).name, "invalid readback acknowledgement")
        key = (ack["library_file_id"], ack["version"])
        identity = (ack["file_id"], ack["sha256"], ack["content_identity"], local)
        check(key not in seen_versions or seen_versions[key] == identity, "conflicting Library version")
        seen_versions[key] = identity
        matches = []
        for metadata_path, metadata_sha, row in rows:
            lib = row.get("library", {})
            if (row.get("phase") == "private_readback" and row.get("sha256") == ack["sha256"] and
                    row.get("contentIdentity") == ack["content_identity"] and lib.get("status") == "succeeded" and
                    lib.get("library_file_id") == ack["library_file_id"] and lib.get("file_id") == ack["file_id"] and
                    lib.get("current_version_number") == ack["version"] and lib.get("file_name") == ack["file_name"] and
                    type(lib.get("file_size_bytes")) is int and lib["file_size_bytes"] > 0):
                matches.append({"path": metadata_path, "sha256": metadata_sha, "bytes": lib["file_size_bytes"]})
        if not matches:
            continue
        check(len({r["bytes"] for r in matches}) == 1, "conflicting Library byte counts")
        entry = {k: ack[k] for k in ("local_path", "sha256", "content_identity", "library_file_id", "file_id", "version")}
        entry.update(bytes=matches[0]["bytes"], readback_evidence={"path": source, "sha256": sha}, library_metadata=matches)
        check(local not in evidence or all(evidence[local][k] == entry[k] for k in
              ("sha256", "content_identity", "library_file_id", "file_id", "version", "bytes")), "conflicting local archive evidence")
        evidence[local] = entry
    return evidence, inputs


def archive_payload(root, entry):
    path = safe_path(root, entry["local_path"])
    with opened_regular(path) as f:
        before = os.fstat(f.fileno())
        check(before.st_size == entry["bytes"] <= POLICY["archive_cap_bytes"], "archive byte limit/mismatch")
        check(hash_stream(f) == entry["sha256"], "archive hash changed")
        f.seek(0)
        with zipfile.ZipFile(f) as z:
            names = z.namelist()
            check(len(names) == len(set(names)), "duplicate ZIP member")
            check(sum(i.file_size for i in z.infolist()) <= POLICY["backup_cap_bytes"], "uncompressed archive bound")
            for i in z.infolist():
                p = PurePosixPath(i.filename)
                mode = (i.external_attr >> 16) & 0o170000
                check(not p.is_absolute() and ".." not in p.parts and str(p) == i.filename and
                      not i.is_dir() and mode in (0, stat.S_IFREG) and not i.flag_bits & 1, "unsafe ZIP member")
            check(z.getinfo("CHECKPOINT_MANIFEST.json").file_size <= 64 * 1024**2, "manifest too large")
            manifest = json.loads(z.read("CHECKPOINT_MANIFEST.json"))
            payload = manifest["all_payload_sha256"]
            check(manifest.get("schema") == "RC-SURVIVOR-CHECKPOINT-v2" and
                  manifest.get("content_identity") == entry["content_identity"] and
                  set(names) == set(payload) | {"CHECKPOINT_MANIFEST.json"}, "checkpoint manifest mismatch")
            check(digest(manifest["logical_identity"]) == entry["content_identity"] and
                  manifest["scientific_files"] == manifest["logical_identity"]["scientific"], "logical identity mismatch")
            check(all(payload.get(p) == h for p, h in manifest["scientific_files"].items()), "scientific map mismatch")
            for name, expected in payload.items():
                check(isinstance(expected, str) and HEX.fullmatch(expected), "invalid member hash")
                with z.open(name) as member:
                    check(hash_stream(member) == expected, "member hash mismatch: " + name)
        check(_stamp(before) == _stamp(os.fstat(f.fileno())) == _stamp(path.lstat()), "archive changed during verification")
    return payload


def collect_references(value, strings, versions, library_id):
    if isinstance(value, str):
        strings.add(value)
    elif isinstance(value, list):
        for item in value:
            collect_references(item, strings, versions, library_id)
    elif isinstance(value, dict):
        if value.get("library_file_id", library_id) == library_id:
            for key in ("version", "private_version"):
                if type(value.get(key)) is int:
                    versions.add(value[key])
        for item in value.values():
            collect_references(item, strings, versions, library_id)


def plan(root, launch_manifest, expected_launch_sha256, *, extra_live_references=()):
    """Return an exact hash-bound proposal; never perform retirement.

    The hash argument must be pinned in separately reviewed launch acceptance. Extra
    active writer/readback/restore references supplement automatic state discovery.
    """
    root = Path(root).resolve()
    launch, launch_sha = read_json(launch_manifest)
    check(launch_sha == expected_launch_sha256, "launch protection manifest hash changed")
    check(launch.get("schema") == "RC-IMMUTABLE-LAUNCH-PROTECTION-v1" and
          launch.get("all_existing_direct_backup_zips_protected") is True, "launch manifest invalid")
    protected = launch["files"]
    check(launch["original_count"] == len(protected) and
          launch["original_bytes"] == sum(v["bytes"] for v in protected.values()), "launch manifest totals")
    for rel in protected:
        check(rel.startswith("backups/") and len(PurePosixPath(rel).parts) == 2 and rel.endswith(".zip"), "launch path scope")
    evidence, inputs = verified_evidence(root)
    current, current_sha = read_json(safe_path(root, "operations/CHECKPOINT_STATE.json"))
    inputs["operations/CHECKPOINT_STATE.json"] = current_sha
    current_entry = evidence.get(current.get("local_path"))
    check(current_entry is not None and current.get("status") == "private_backup_readback_verified" and
          all(current.get(k) == current_entry[k] for k in ("sha256", "version", "content_identity", "library_file_id", "file_id")),
          "current checkpoint lacks matching durable readback evidence")
    lib = current_entry["library_file_id"]
    versions = sorted({e["version"] for e in evidence.values() if e["library_file_id"] == lib}, reverse=True)
    check(current_entry["version"] == versions[0], "current checkpoint is not latest verified version")
    latest_four = versions[:4]
    reference_strings, reference_versions = set(extra_live_references), set()
    live_files = sorted(p for p in safe_path(root, "operations").rglob("*.json")
                        if p.name in LIVE_STATE_NAMES and not any(part in p.parts for part in ("attempts", "quarantine")))
    blockers = []
    # A remote acknowledgement cannot substitute for the promised recent local
    # recovery window. Missing or modified retained copies stop every proposal.
    for version in latest_four:
        recent = [e for e in evidence.values() if e["library_file_id"] == lib and e["version"] == version]
        available = False
        for e in recent:
            try:
                snapshot = file_snapshot(safe_path(root, e["local_path"]))
                available |= (snapshot["sha256"], snapshot["bytes"]) == (e["sha256"], e["bytes"])
            except (OSError, PolicyError):
                pass
        if not available:
            blockers.append("recent_local_version_missing_or_changed:" + str(version))
    for p in live_files:
        value, h = read_json(p)
        inputs[str(p.relative_to(root))] = h
        collect_references(value, reference_strings, reference_versions, lib)
        if p.name == "TRANSPORT_PENDING.json" and value is not None:
            if not isinstance(value, dict) or not str(value.get("phase", "")).startswith("public_"):
                blockers.append("unresolved_private_transport:" + str(p.relative_to(root)))
    for rel, entry in protected.items():
        try:
            snapshot = file_snapshot(safe_path(root, rel))
            if (snapshot["sha256"], snapshot["bytes"]) != (entry["sha256"], entry["bytes"]):
                blockers.append("protected_original_changed:" + rel)
        except (OSError, PolicyError):
            blockers.append("protected_original_missing_or_unsafe:" + rel)
    base = safe_path(root, "backups")
    names = sorted(p.name for p in base.iterdir())
    rows, proposed, ignored, covering_payload = [], [], [], None
    for name in names:
        rel = "backups/" + name
        match = NAME.fullmatch(name)
        if not match:
            ignored.append(rel)
            continue
        row = {"path": rel, "decision": "retain", "reasons": []}
        reasons = row["reasons"]
        if rel in protected:
            reasons.append("protected_prelaunch_original")
        try:
            snapshot = file_snapshot(safe_path(root, rel))
            row.update(snapshot)
            if snapshot["stat_identity"][-1] != 1:
                reasons.append("multiply_linked_archive")
        except (OSError, PolicyError):
            reasons.append("not_safe_regular_nonsymlink_file")
            rows.append(row)
            continue
        e = evidence.get(rel)
        if e is None:
            reasons.append("unverified_or_missing_exact_library_metadata")
        else:
            row["evidence"] = e
            if e["library_file_id"] != lib:
                reasons.append("different_library_identity")
            if e["sha256"] != snapshot["sha256"] or e["bytes"] != snapshot["bytes"]:
                reasons.append("archive_hash_or_size_changed")
            if e["version"] in latest_four:
                reasons.append("latest_four_distinct_verified_versions")
            if sum(v > e["version"] for v in versions) < 2:
                reasons.append("fewer_than_two_newer_verified_versions")
            if rel == current_entry["local_path"]:
                reasons.append("current_checkpoint")
            if (e["version"] in reference_versions or reference_strings.intersection(
                    {rel, name, str(root / rel), e["sha256"], e["content_identity"], e["file_id"]})):
                reasons.append("live_reference_needed")
        if blockers:
            reasons.append("global_safety_blocker")
        if not reasons:
            try:
                if covering_payload is None:
                    covering_payload = archive_payload(root, current_entry)
                payload = archive_payload(root, e)
                required = {p: h for p, h in payload.items() if p.startswith(SCIENTIFIC_PREFIXES) or p == "README.md"}
                check(required and any(p.startswith("source/") for p in required), "missing scientific payload")
                missing = [p for p, h in required.items() if covering_payload.get(p) != h]
                if missing:
                    reasons.append("unique_scientific_payload_not_currently_covered")
                    row["uncovered_paths"] = missing
                else:
                    row["coverage"] = {"current_archive": current_entry, "required_payload": required,
                                       "required_payload_sha256": digest(required), "payload_count": len(required)}
            except (OSError, PolicyError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
                reasons.append("archive_or_coverage_verification_failed")
                row["verification_error"] = str(exc)
        if not reasons:
            row["decision"] = "proposed_local_duplicate_retirement"
            proposed.append(row)
        rows.append(row)
    check(names == sorted(p.name for p in base.iterdir()), "backup directory changed during plan")
    for rel, h in inputs.items():
        check(read_json(safe_path(root, rel))[1] == h, "evidence changed during plan")
    result = {"schema": "RC-LOCAL-RETENTION-PLAN-v1", "dry_run": True, "separate_executor_required": True,
              "policy": POLICY, "policy_sha256": digest(POLICY), "planner_sha256": file_snapshot(__file__)["sha256"],
              "launch_protection_sha256": launch_sha, "evidence_input_sha256": inputs,
              "current_checkpoint": current_entry, "retained_recent_versions": latest_four,
              "live_reference_versions": sorted(reference_versions), "extra_live_references": sorted(extra_live_references),
              "blockers": blockers, "archives": rows, "noncandidate_paths_preserved": ignored,
              "proposed_count": len(proposed), "proposed_bytes": sum(r["bytes"] for r in proposed),
              "execution_status": "disabled_requires_separately_approved_executor_and_live_serialization"}
    result["plan_sha256"] = digest(result)
    return result


def check_plan_approval(proposal, approval):
    """Pure approval binding checker, NOT an executor and NOT proof of user consent.

    The independently reviewed external coordinator owns approval authenticity and
    live host/write lock checks. This function cannot authorize or perform deletion.
    """
    body = {k: v for k, v in proposal.items() if k != "plan_sha256"}
    check(proposal.get("plan_sha256") == digest(body), "plan hash mismatch")
    check(approval.get("schema") == "RC-LOCAL-RETENTION-APPROVAL-v1" and
          approval.get("accepted") is True and approval.get("independent_review") is True and
          approval.get("scope") == "exact_planned_local_duplicate_archives_only", "approval missing or wrong scope")
    for k in ("plan_sha256", "policy_sha256", "planner_sha256", "launch_protection_sha256"):
        check(approval.get(k) == proposal[k], "approval hash mismatch: " + k)
    check(not proposal["blockers"], "plan has safety blockers")
    return {"approval_hash_binding_valid": True, "execution_authorized_by_module": False,
            "deletion_implemented": False}


def forecast(protected_bytes, base_archive_bytes, compressed_world_bytes, raw_world_bytes, worlds=64):
    check(all(type(n) is int and n >= 0 for n in
              (protected_bytes, base_archive_bytes, compressed_world_bytes, raw_world_bytes, worlds)), "invalid forecast inputs")
    final = base_archive_bytes + worlds * compressed_world_bytes
    all_future = sum(2 * base_archive_bytes + (2 * w - 1) * compressed_world_bytes for w in range(1, worlds + 1))
    return {"model": "measured-linear-growth-estimate-not-runtime-cap-waiver", "worlds": worlds,
            "protected_original_bytes": protected_bytes, "base_archive_bytes": base_archive_bytes,
            "compressed_bytes_per_world": compressed_world_bytes, "raw_bytes_per_world": raw_world_bytes,
            "future_prepost_zip_count": 2 * worlds, "future_prepost_zip_bytes_all_retained": all_future,
            "all_original_zip_bytes": protected_bytes + all_future, "final_archive_estimate_bytes": final,
            "protected_plus_four_recent_estimate_bytes": protected_bytes + 4 * final,
            "bounded_peak_with_three_transient_copies_and_128MiB_staging": protected_bytes + 7 * final + 128 * 1024**2,
            "raw_world_receipts_bytes": worlds * raw_world_bytes,
            "archive_cap_bytes": POLICY["archive_cap_bytes"], "backup_cap_bytes": POLICY["backup_cap_bytes"],
            "unmodelled": ["metadata growth", "terminal analysis", "restore extraction", "unverified or pinned archives"],
            "required_runtime_guards": "retain existing exact byte, output, archive, staging and free-disk caps; stop if exceeded"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("launch_manifest", type=Path)
    parser.add_argument("--launch-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(plan(args.root, args.launch_manifest, args.launch_sha256), sort_keys=True, indent=2))
