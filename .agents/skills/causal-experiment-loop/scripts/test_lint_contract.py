#!/usr/bin/env python3
"""No-dependency unit tests: run python3 -m unittest discover -s scripts."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from lint_contract import validate_contract, validate_receipts

EXAMPLE = Path(__file__).resolve().parent.parent / "assets" / "experiment_contract.example.json"


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_example_is_structurally_valid(self):
        errors, warnings = validate_contract(self.data)
        self.assertEqual(errors, [])
        self.assertTrue(any("placeholder" in x for x in warnings))

    def test_unplanned_change_fails(self):
        d = copy.deepcopy(self.data)
        d["arms"][1]["config"]["decay"] = 0.3
        errs, _ = validate_contract(d)
        self.assertTrue(any("intended_changes" in e for e in errs))

    def test_fake_matched_stream_fails(self):
        d = copy.deepcopy(self.data)
        d["arms"][1]["stream_id"] = "different"
        errs, _ = validate_contract(d)
        self.assertTrue(any("claims matched" in e for e in errs))

    def test_duplicate_seed_fails(self):
        d = copy.deepcopy(self.data)
        d["seeds"] = [101, 101]
        errs, _ = validate_contract(d)
        self.assertTrue(any("duplicates" in e for e in errs))

    def test_prelock_receipt_fails(self):
        r = [{"arm": "local", "seed": 101, "run_id": "r1", "started_at": "2026-11-09T17:30:00-05:00", "dataset_hash": self.data["dataset_hash"], "panel_hash": self.data["panel_hash"], "status": "COMPLETE"}]
        errs, _ = validate_receipts(self.data, r)
        self.assertTrue(any("before prediction lock" in e for e in errs))

    def test_discovery_confirmation_overlap_fails(self):
        d = copy.deepcopy(self.data)
        d["discovery_seeds"] = [101]
        d["confirmation_seeds"] = [101, 202, 303]
        errs, _ = validate_contract(d)
        self.assertTrue(any("discovery/confirmation seed overlap" in e for e in errs))

    def test_single_factor_claim_with_two_changes_fails(self):
        d = copy.deepcopy(self.data)
        d["arms"][1]["config"]["decay"] = 0.3
        d["arms"][1]["intended_changes"] = ["update", "decay"]
        errs, _ = validate_contract(d)
        self.assertTrue(any("claims single_factor" in e for e in errs))

    def test_zero_gradient_with_active_decay_fails(self):
        r = [{"arm": "local", "seed": 101, "run_id": "r1", "started_at": "2026-11-09T18:10:00-05:00", "dataset_hash": self.data["dataset_hash"], "panel_hash": self.data["panel_hash"], "status": "COMPLETE", "gradient_probe": [{"parameter": "book.operator_plus", "grad_norm": 0, "decay_rate": 0.25, "frozen": False}]}]
        errs, _ = validate_receipts(self.data, r)
        self.assertTrue(any("zero-gradient but decayed" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
