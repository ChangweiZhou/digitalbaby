# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic execution only. Never points an executor at the real study root."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest import mock

import executor as e
import retention as r
from test_retention import Fixture


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="retention-execution-synthetic-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.f = Fixture(self.root, metadata_in_p2=True)
        self.ack = self.f.entries[6]
        readback = self.root / self.ack["readback_path"]
        readback.parent.mkdir()
        shutil.copyfile(self.root / self.ack["local_path"], readback)
        self.keeper = (self.root / "operations/HOST_COORDINATOR.lock").open("a+")
        fcntl.flock(self.keeper, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.addCleanup(self.keeper.close)
        self.env = mock.patch.dict(os.environ, {"SURVIVOR_HOST_TOKEN": "synthetic-host-only"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.host()
        self.acceptance = {"schema": "RC-LOCAL-RETENTION-EXECUTION-ACCEPTANCE-v1", "accepted": True,
                           "independent_review": True, "scope": "verified_local_duplicate_retirement",
                           "retention_hashes": e.code_hashes(), "policy_sha256": r.digest(r.POLICY),
                           "launch_protection_sha256": self.f.launch_sha}
        self.f.write("operations/retention_policy/ACCEPTED.json", self.acceptance)

    def host(self, **changes):
        self.f.write("operations/HOST_WALL.json", {"active": True, "host_token": "synthetic-host-only",
                     "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                     "updated_monotonic_s": time.monotonic(), "effective_s": 0, **changes})

    def execute(self, **kwargs):
        return e.execute_current(self.root, checkpoint=self.ack,
                                 acceptance_path="operations/retention_policy/ACCEPTED.json",
                                 launch_manifest="launch.json", launch_sha256=self.f.launch_sha, **kwargs)

    def test_actual_synthetic_retirement_has_fsynced_hash_chain_and_preserves_protected(self):
        p = self.f.plan()
        before = {v: r.file_snapshot(self.root / a["local_path"])["sha256"] for v, a in self.f.entries.items()}
        result = self.execute(expected_plan_sha256=p["plan_sha256"])
        self.assertEqual(len(result["removed"]), 1)
        self.assertEqual(result["removed"][0]["path"], self.f.entries[2]["local_path"])
        for v in (1, 3, 4, 5, 6):
            self.assertEqual(r.file_snapshot(self.root / self.f.entries[v]["local_path"])["sha256"], before[v])
        rows, _ = r.journal(self.root, e.JOURNAL)
        self.assertEqual([v[2]["event"] for v in rows], ["retirement_intent", "retirement_complete"])
        self.assertEqual(rows[1][2]["intent"]["sha256"], rows[0][1])
        self.assertTrue(rows[1][2]["directory_fsynced"])
        self.assertNotIn("required_payload", rows[0][2]["candidate"]["coverage"])
        self.assertEqual(rows[0][2]["candidate"]["coverage"]["required_payload_sha256"],
                         self.f.row(p, 2)["coverage"]["required_payload_sha256"])
        self.assertLessEqual(len(r.canonical(rows[0][2])) + 1, e.MAX_INTENT_BYTES)
        self.assertLessEqual(len(r.canonical(rows[1][2])) + 1, e.MAX_COMPLETION_BYTES)
        self.assertEqual(self.execute()["removed"], [])

    def test_missing_host_keeper_fails_without_mutation(self):
        fcntl.flock(self.keeper, fcntl.LOCK_UN)
        with self.assertRaisesRegex(r.PolicyError, "keeper lock"):
            self.execute()
        self.assertTrue((self.root / self.f.entries[2]["local_path"]).exists())

    def test_stale_host_fails_without_mutation(self):
        self.host(updated_monotonic_s=time.monotonic() - 30)
        with self.assertRaisesRegex(r.PolicyError, "stale host"):
            self.execute()

    def test_unaccepted_code_hash_fails(self):
        self.acceptance["retention_hashes"]["executor.py"] = "0" * 64
        self.f.write("operations/retention_policy/ACCEPTED.json", self.acceptance)
        with self.assertRaisesRegex(r.PolicyError, "acceptance hash"):
            self.execute()

    def test_changed_readback_fails(self):
        with (self.root / self.ack["readback_path"]).open("ab") as out:
            out.write(b"changed fixture")
        with self.assertRaisesRegex(r.PolicyError, "readback bytes changed"):
            self.execute()

    def test_exact_plan_binding_rejects_stale_plan(self):
        with self.assertRaisesRegex(r.PolicyError, "exact plan changed"):
            self.execute(expected_plan_sha256="0" * 64)
        self.assertFalse((self.root / e.JOURNAL).exists())

    def test_original_supervisor_writer_lock_is_exclusive(self):
        # The exact lock used by frozen recovery.exclusive_lock, not a new mutex.
        with (self.root / "operations/SUPERVISOR.lock").open("a+") as other:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.execute()

    def test_change_after_durable_intent_stops_and_leaves_reconciliation_block(self):
        append = e.append_event
        path = self.root / self.f.entries[2]["local_path"]

        def change(root, event):
            result = append(root, event)
            if event["event"] == "retirement_intent":
                with path.open("ab") as out:
                    out.write(b"changed after durable intent")
            return result

        with mock.patch.object(e, "append_event", side_effect=change):
            with self.assertRaisesRegex(r.PolicyError, "candidate changed after intent"):
                self.execute()
        self.assertTrue(path.exists())
        with self.assertRaisesRegex(r.PolicyError, "unfinished retirement intent"):
            self.execute()

    def test_crash_after_unlink_stops_next_execution_for_reconciliation(self):
        append = e.append_event

        def crash(root, event):
            if event["event"] == "retirement_complete":
                raise OSError("synthetic crash after directory fsync")
            return append(root, event)

        with mock.patch.object(e, "append_event", side_effect=crash):
            with self.assertRaisesRegex(OSError, "synthetic crash"):
                self.execute()
        self.assertFalse((self.root / self.f.entries[2]["local_path"]).exists())
        with self.assertRaisesRegex(r.PolicyError, "unfinished retirement intent"):
            self.execute()

    def test_live_reference_blocks_synthetic_retirement(self):
        result = self.execute(extra_live_references=[self.f.entries[2]["local_path"]])
        self.assertEqual(result["removed"], [])

    def test_large_coverage_map_does_not_expand_durable_intent(self):
        row = self.f.row(self.f.plan(), 2)
        compact_before = e.compact_candidate(row)
        payload = {"receipts/synthetic/" + str(n): r.digest({"n": n}) for n in range(4000)}
        row["coverage"].update(required_payload=payload, required_payload_sha256=r.digest(payload), payload_count=len(payload))
        compact = e.compact_candidate(row)
        self.assertNotIn("required_payload", compact["coverage"])
        self.assertLess(abs(len(r.canonical(compact)) - len(r.canonical(compact_before))), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
