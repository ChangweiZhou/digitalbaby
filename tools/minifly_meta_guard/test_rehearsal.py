import unittest

from rehearsal import REQUIRED, validate

class RehearsalTest(unittest.TestCase):
    def setUp(self):
        self.launch = {"source_sha256": {"run.py": "a"*64}, "entrypoint": "run.py",
                       "required_coverage": ["sensor_geometry"]}
        self.receipt = {**self.launch, "status": "COMPLETED", "fixture_only": True,
                        "coverage": {k: "PASS" for k in REQUIRED | {"sensor_geometry"}}}

    def test_complete_current_rehearsal(self):
        self.assertEqual(validate(self.launch, self.receipt), [])

    def test_component_checks_miss_production_geometry(self):
        del self.receipt["coverage"]["sensor_geometry"]
        self.assertIn("MISSING_PASS:sensor_geometry", validate(self.launch, self.receipt))

    def test_rehearsal_from_old_source_is_rejected(self):
        self.receipt["source_sha256"] = {"run.py": "b"*64}
        self.assertIn("SOURCE_LOCK_MISMATCH", validate(self.launch, self.receipt))

    def test_early_exit_cannot_claim_completion(self):
        self.receipt["status"] = "FAILED"
        self.assertIn("REHEARSAL_NOT_COMPLETED", validate(self.launch, self.receipt))
