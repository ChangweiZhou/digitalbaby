"""LAUNCH_QUALIFICATION.json: qualifies launcher/lock/tooling changes after the final technical lives.

* Learner executed closure unchanged vs the final technical receipts (so the 13 lives remain the qualification).
* Launcher synthetic fault/resume acceptance tests (tests/test_launcher.py) and unit tests pass.
* The real receipt validator accepts a real final technical receipt relabelled as science under a test lock digest,
  and rejects it under a wrong lock digest or a wrong arm/world (no trajectory is run).
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
import lock  # noqa: E402
import runner  # noqa: E402
import drive_science  # noqa: E402

TF = paths.ROOT / "results" / "technical_final"
LEARNER = ("src/paths.py", "src/stores.py", "src/systems.py", "src/runner.py", "src/calibrate_r.py",
           "SPEC_LOCK.md", "ARM_ROSTER.json", "results/calibration/R_CALIBRATION.json", "package/MANIFEST.json")


def sh(cmd):
    p = subprocess.run(cmd, cwd=paths.ROOT, capture_output=True, text=True)
    return {"cmd": " ".join(cmd), "returncode": p.returncode,
            "tail": [x for x in p.stdout.strip().splitlines() if not x.startswith("{")][-3:]}


def validator_checks() -> dict:
    out = {}
    fake_lock = {"lock_digest": "T" * 64}
    raws = {a: (TF / a / "190000.json.gz").read_bytes() for a in ("R1", "R1_rand", "Z2", "Z2_rand", "P2")}

    def relabel(arm):
        r = json.loads(gzip.decompress(raws[arm]))
        r["kind"], r["lock_digest"] = "science", fake_lock["lock_digest"]
        r["source_sha256"] = runner.source_hashes()
        return r

    def enc(r):
        return gzip.compress(json.dumps(r).encode())

    deps = {"R1_rand": {"R1": enc(relabel("R1"))}, "Z2_rand": {"Z2": enc(relabel("Z2"))}}
    for arm in ("R1", "R1_rand", "Z2_rand", "P2"):
        r = relabel(arm)
        drive_science.validate_receipt(arm, 190000, enc(r), fake_lock, deps.get(arm, {}))
        out[f"accept_{arm}"] = True
        for name, mut in (("wrong_lock", lambda d: d.__setitem__("lock_digest", "X" * 64)),
                          ("wrong_world", lambda d: d.__setitem__("world", 190001)),
                          ("wrong_kind", lambda d: d.__setitem__("kind", "technical_final")),
                          ("wrong_source", lambda d: d["source_sha256"].__setitem__("src/stores.py", "0" * 64))):
            bad = copy.deepcopy(r)
            mut(bad)
            try:
                drive_science.validate_receipt(arm, 190000, enc(bad), fake_lock, deps.get(arm, {}))
                out[f"reject_{arm}_{name}"] = False
            except AssertionError:
                out[f"reject_{arm}_{name}"] = True
    try:
        zr = relabel("Z2_rand")
        z = next(x for x in zr["mechanism_events"]["W"] if x[0] == "Z" and x[5] > 0)
        k = max(z[7], key=lambda n: z[7][n][2])
        z[7][k][2] *= 0.9
        z[5] = sum(v[2] for v in z[7].values())
        z[6] = z[5]
        drive_science.validate_receipt("Z2_rand", 190000, enc(zr), fake_lock, deps["Z2_rand"])
        out["reject_Z2_rand_dose_vs_paired_Z2"] = False
    except AssertionError:
        out["reject_Z2_rand_dose_vs_paired_Z2"] = True
    return out


def main() -> None:
    audit = json.loads((TF / "TECHNICAL_AUDIT.json").read_text())
    rec = json.loads(gzip.decompress((TF / "R0" / "190000.json.gz").read_bytes()))["source_sha256"]
    now = runner.source_hashes()
    learner_equal = {f: rec.get(f) == now.get(f) for f in LEARNER}
    changed = sorted(f for f in set(rec) | set(now) if rec.get(f) != now.get(f))
    launcher = sh([sys.executable, "tests/test_launcher.py"])
    units = sh([sys.executable, "tests/test_units.py"])
    val = validator_checks()
    doc = {"schema": "MINIFLY-A3-CLAUDE-LAUNCH-QUALIFICATION-v1",
           "basis": "launch review CLAUDE_A_V3_LAUNCH_REVIEW_20260930 (commit reviewed d164568)",
           "final_technical_audit_pass": audit["pass"],
           "learner_executed_closure_unchanged": learner_equal,
           "files_changed_since_final_technical_lives": changed,
           "changed_files_role": "launch/lock/audit tooling and tests; none is executed inside a learner life",
           "launcher_acceptance_tests": launcher, "unit_tests": units, "real_validator_checks": val,
           "launch_contract_note": ("SPEC_LOCK.md is part of the learner closure and is unchanged; its 'batches of 16' "
                                    "and restart wording are superseded by LAUNCH_CONTRACT.md and src/drive_science.py "
                                    "(persist every 4 validated receipts; durable ledger; authorization rules)."),
           "code_sha256": lock.code_files(), "package_files_match_manifest": bool(lock.package_files())}
    doc["pass"] = (audit["pass"] and all(learner_equal.values()) and launcher["returncode"] == 0 and
                   units["returncode"] == 0 and all(val.values()))
    (TF / "LAUNCH_QUALIFICATION.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: doc[k] for k in ("pass", "files_changed_since_final_technical_lives", "real_validator_checks")},
                     indent=1))


if __name__ == "__main__":
    main()
