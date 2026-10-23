# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure standard-library tests. Every test root and ZIP is a generated fixture."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import retention as r


class Fixture:
    def __init__(self, root, *, unverified=(), omitted_current=(), metadata_in_p2=False):
        self.root = Path(root)
        self.lib = "libfile_synthetic_only"
        (self.root / "backups").mkdir()
        (self.root / "operations/transport_journal").mkdir(parents=True)
        self.entries = {}
        self.previous = None
        self.sequence = 0
        self.p2_previous = None
        self.p2_sequence = 0
        for v in range(1, 7):
            data = {"source/frozen.py": b"synthetic immutable source\n",
                    "receipts/science/1/manifest.json": b'{"synthetic":true}\n',
                    "operations/RUN_LEDGER.json": r.canonical({"fixture_version": v})}
            if v == 6:
                for name in omitted_current:
                    data.pop(name)
            scientific = {p: hashlib.sha256(b).hexdigest() for p, b in data.items() if p.startswith(("source/", "receipts/"))}
            logical = {"scientific": scientific, "accounting": {"fixture_version": v}}
            identity = r.digest(logical)
            name = "checkpoint-" + identity + ".zip"
            path = self.root / "backups" / name
            manifest = {"schema": "RC-SURVIVOR-CHECKPOINT-v2", "content_identity": identity,
                        "logical_identity": logical, "scientific_files": scientific,
                        "all_payload_sha256": {p: hashlib.sha256(b).hexdigest() for p, b in data.items()}}
            with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as z:
                for p, b in data.items():
                    z.writestr(p, b)
                z.writestr("CHECKPOINT_MANIFEST.json", r.canonical(manifest))
            sha = r.file_snapshot(path)["sha256"]
            ack = {"status": "private_backup_readback_verified", "local_path": "backups/" + name,
                   "sha256": sha, "content_identity": identity, "library_file_id": self.lib,
                   "file_id": "file_synthetic_" + str(v), "version": v, "file_name": name,
                   "readback_path": "backups/readback/" + name}
            self.entries[v] = ack
            if v not in unverified:
                writer = self.append_p2 if metadata_in_p2 else self.append
                writer({"phase": "private_readback", "sha256": sha, "contentIdentity": identity,
                             "library": {"status": "succeeded", "library_file_id": self.lib,
                                         "file_id": ack["file_id"], "current_version_number": v,
                                         "file_name": name, "file_size_bytes": path.stat().st_size}})
                self.append({"event": "private_readback_verified", **ack})
            if v == 1:
                self.launch = self.root / "launch.json"
                self.launch_sha = r.create_json_once(self.launch, r.launch_protection(self.root))
        self.write("operations/CHECKPOINT_STATE.json", self.entries[6])
        self.write("operations/TRANSPORT_PENDING.json", None)
        (self.root / "backups/notes.txt").write_text("not an archive")

    def write(self, rel, value):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(r.canonical(value) + b"\n")

    def append(self, value):
        row = {"sequence": self.sequence, "previous_sha256": self.previous, **value}
        rel = f"operations/transport_journal/{self.sequence:06d}.json"
        self.write(rel, row)
        self.previous = r.file_snapshot(self.root / rel)["sha256"]
        self.sequence += 1

    def plan(self, **kwargs):
        return r.plan(self.root, self.launch, self.launch_sha, **kwargs)

    def append_p2(self, value):
        row = {"sequence": self.p2_sequence, "previous_sha256": self.p2_previous, **value}
        rel = f"operations/private_official_policy/transport_journal/{self.p2_sequence:06d}.json"
        self.write(rel, row)
        self.p2_previous = r.file_snapshot(self.root / rel)["sha256"]
        self.p2_sequence += 1

    def row(self, plan, version):
        return next(a for a in plan["archives"] if a["path"] == self.entries[version]["local_path"])


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="retention-synthetic-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_eligible_duplicate_plan_and_exact_approval_never_delete(self):
        f = Fixture(self.root)
        before = {p.name: p.read_bytes() for p in (self.root / "backups").iterdir()}
        proposal = f.plan()
        self.assertEqual(proposal["proposed_count"], 1)
        row = f.row(proposal, 2)
        self.assertEqual(row["decision"], "proposed_local_duplicate_retirement")
        self.assertEqual(row["coverage"]["payload_count"], 2)
        self.assertEqual(row["coverage"]["current_archive"]["version"], 6)
        approval = {"schema": "RC-LOCAL-RETENTION-APPROVAL-v1", "accepted": True,
                    "independent_review": True, "scope": "exact_planned_local_duplicate_archives_only"}
        approval.update({k: proposal[k] for k in ("plan_sha256", "policy_sha256", "planner_sha256", "launch_protection_sha256")})
        self.assertFalse(r.check_plan_approval(proposal, approval)["execution_authorized_by_module"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.root / "backups").iterdir()})

    def test_protected_original_current_and_four_recent(self):
        f = Fixture(self.root)
        p = f.plan()
        self.assertIn("protected_prelaunch_original", f.row(p, 1)["reasons"])
        self.assertIn("current_checkpoint", f.row(p, 6)["reasons"])
        self.assertEqual(p["retained_recent_versions"], [6, 5, 4, 3])
        for v in (3, 4, 5, 6):
            self.assertIn("latest_four_distinct_verified_versions", f.row(p, v)["reasons"])

    def test_official_metadata_journal_joins_legacy_readback_ack(self):
        f = Fixture(self.root, metadata_in_p2=True)
        p = f.plan()
        self.assertEqual(p["proposed_count"], 1)
        self.assertTrue(f.row(p, 2)["evidence"]["library_metadata"][0]["path"].startswith(
            "operations/private_official_policy/transport_journal/"))

    def test_unverified_or_pending_upload_is_never_eligible(self):
        f = Fixture(self.root, unverified=(2,))
        self.assertIn("unverified_or_missing_exact_library_metadata", f.row(f.plan(), 2)["reasons"])
        f.write("operations/private_policy/TRANSPORT_PENDING.json", {"phase": "private_upload", "status": "running"})
        p = f.plan()
        self.assertEqual(p["proposed_count"], 0)
        self.assertTrue(p["blockers"])

    def test_exact_hash_changed(self):
        f = Fixture(self.root)
        with (self.root / f.entries[2]["local_path"]).open("ab") as out:
            out.write(b"changed")
        self.assertIn("archive_hash_or_size_changed", f.row(f.plan(), 2)["reasons"])

    def test_missing_recent_local_version_blocks_every_proposal(self):
        f = Fixture(self.root)
        (self.root / f.entries[3]["local_path"]).unlink()  # Generated fixture only.
        p = f.plan()
        self.assertIn("recent_local_version_missing_or_changed:3", p["blockers"])
        self.assertEqual(p["proposed_count"], 0)

    def test_completed_job_archive_reference_requires_reack(self):
        f = Fixture(self.root)
        f.write("operations/private_policy/PRIVATE_PERSISTENCE_STATE.json", {
            "jobs": {"science/1": {"checkpoint": f.entries[2], "private_readback_verified": True}}})
        self.assertIn("live_reference_needed", f.row(f.plan(), 2)["reasons"])

    def test_bare_per_world_version_reference_is_protected(self):
        f = Fixture(self.root)
        f.write("operations/PERSISTENCE_STATE.json", {"worlds": {"1": {"private_version": 2}}})
        self.assertIn("live_reference_needed", f.row(f.plan(), 2)["reasons"])

    def test_explicit_active_restore_input_is_protected(self):
        f = Fixture(self.root)
        p = f.plan(extra_live_references=[f.entries[2]["local_path"]])
        self.assertIn("live_reference_needed", f.row(p, 2)["reasons"])

    def test_unique_receipt_without_current_coverage_is_protected(self):
        f = Fixture(self.root, omitted_current=("receipts/science/1/manifest.json",))
        row = f.row(f.plan(), 2)
        self.assertIn("unique_scientific_payload_not_currently_covered", row["reasons"])
        self.assertEqual(row["uncovered_paths"], ["receipts/science/1/manifest.json"])

    def test_unique_source_without_current_coverage_is_protected(self):
        f = Fixture(self.root, omitted_current=("source/frozen.py",))
        self.assertIn("unique_scientific_payload_not_currently_covered", f.row(f.plan(), 2)["reasons"])

    def test_symlink_is_rejected_and_noncandidate_is_preserved(self):
        f = Fixture(self.root)
        target = self.root / f.entries[2]["local_path"]
        target.unlink()  # Generated synthetic fixture, never a live archive.
        target.symlink_to(self.root / f.entries[6]["local_path"])
        p = f.plan()
        self.assertIn("not_safe_regular_nonsymlink_file", f.row(p, 2)["reasons"])
        self.assertIn("backups/notes.txt", p["noncandidate_paths_preserved"])

    def test_protection_manifest_is_create_only_and_hash_pinned(self):
        f = Fixture(self.root)
        with self.assertRaises(FileExistsError):
            r.create_json_once(f.launch, {})
        with self.assertRaisesRegex(r.PolicyError, "manifest hash changed"):
            r.plan(self.root, f.launch, "0" * 64)

    def test_protected_original_change_blocks_entire_plan(self):
        f = Fixture(self.root)
        with (self.root / f.entries[1]["local_path"]).open("ab") as out:
            out.write(b"changed protected copy")
        p = f.plan()
        self.assertTrue(any(b.startswith("protected_original_changed") for b in p["blockers"]))
        self.assertEqual(p["proposed_count"], 0)

    def test_journal_tamper_fails_closed(self):
        f = Fixture(self.root)
        p = self.root / "operations/transport_journal/000000.json"
        value = json.loads(p.read_text())
        value["library"]["file_id"] = "file_changed"
        p.write_bytes(r.canonical(value))
        with self.assertRaisesRegex(r.PolicyError, "journal chain mismatch"):
            f.plan()

    def test_upload_metadata_alone_cannot_prove_readback(self):
        f = Fixture(self.root, unverified=(2,))
        ack = f.entries[2]
        f.append({"phase": "private_readback", "sha256": ack["sha256"], "contentIdentity": ack["content_identity"],
                  "library": {"status": "succeeded", "library_file_id": f.lib, "file_id": ack["file_id"],
                              "current_version_number": 2, "file_name": ack["file_name"],
                              "file_size_bytes": (self.root / ack["local_path"]).stat().st_size}})
        self.assertIn("unverified_or_missing_exact_library_metadata", f.row(f.plan(), 2)["reasons"])

    def test_approval_rejects_missing_and_wrong_bindings(self):
        f = Fixture(self.root)
        proposal = f.plan()
        with self.assertRaises(r.PolicyError):
            r.check_plan_approval(proposal, {})
        a = {"schema": "RC-LOCAL-RETENTION-APPROVAL-v1", "accepted": True, "independent_review": True,
             "scope": "exact_planned_local_duplicate_archives_only",
             **{k: proposal[k] for k in ("plan_sha256", "policy_sha256", "planner_sha256", "launch_protection_sha256")}}
        a["policy_sha256"] = "0" * 64
        with self.assertRaisesRegex(r.PolicyError, "approval hash mismatch"):
            r.check_plan_approval(proposal, a)
        proposal["proposed_count"] = 100
        with self.assertRaisesRegex(r.PolicyError, "plan hash mismatch"):
            r.check_plan_approval(proposal, a)

    def test_distinct_versions_not_acknowledgement_count(self):
        f = Fixture(self.root)
        f.append({"event": "private_readback_verified", **f.entries[6]})
        self.assertEqual(f.plan()["retained_recent_versions"], [6, 5, 4, 3])

    def test_forecast_preserves_caps_and_has_no_native_imports(self):
        x = r.forecast(72013372, 7383162, 540961, 3132834)
        self.assertEqual(x["future_prepost_zip_bytes_all_retained"], 3160820992)
        self.assertEqual(x["protected_plus_four_recent_estimate_bytes"], 240032036)
        self.assertEqual(x["bounded_peak_with_three_transient_copies_and_128MiB_staging"], 500263762)
        self.assertGreater(x["all_original_zip_bytes"], x["backup_cap_bytes"])
        self.assertNotIn("engine", sys.modules)
        self.assertNotIn("survivor_frozen_learner", sys.modules)
        self.assertNotIn("numpy", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
