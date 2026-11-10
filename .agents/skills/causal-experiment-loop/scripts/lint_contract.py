#!/usr/bin/env python3
"""Static lint for a preregistered causal experiment contract.

Usage:
  python3 scripts/lint_contract.py assets/experiment_contract.example.json
  python3 scripts/lint_contract.py contract.json --receipts receipts.json

This checks integrity only. It DOES NOT establish causal identification, calculate
scientific endpoints, inspect model weights, or authorize running experiments.
No third-party dependencies.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REQUIRED = [
    "schema_version", "experiment_id", "status", "question",
    "primary_hypothesis", "competing_hypothesis", "scientific_constraints",
    "locked_at", "lock_commit", "reference_commit", "dataset_hash",
    "panel_hash", "birth_checkpoint_hash", "seeds", "arms",
    "primary_metric", "hard_subgroups", "predictions", "quality_gates",
    "halt_if", "run_authorization"
]


def parse_time(value):
    try:
        t = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return t if t.tzinfo else None
    except (TypeError, AttributeError, ValueError):
        return None


def validate_contract(obj):
    errors, warnings = [], []
    if not isinstance(obj, dict):
        return ["Contract root must be a JSON object"], warnings
    for field in REQUIRED:
        if field not in obj or obj[field] in (None, "", [], {}):
            errors.append(f"Missing/empty required field: {field}")
    if errors:
        # Continue only if basic containers exist; remaining messages are secondary.
        if not isinstance(obj.get("arms"), list) or not isinstance(obj.get("predictions"), list):
            return errors, warnings
    if obj.get("status") not in {"DRAFT", "PREREGISTERED", "RUNNING", "FILED", "INVALIDATED"}:
        errors.append("status must be DRAFT/PREREGISTERED/RUNNING/FILED/INVALIDATED")
    if parse_time(obj.get("locked_at")) is None:
        errors.append("locked_at must be timezone-aware ISO-8601")
    for f in ("lock_commit", "reference_commit"):
        value = obj.get(f, "")
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", value):
            errors.append(f"{f} must be an exact 40-hex git SHA")
        elif len(set(value.lower())) == 1:
            warnings.append(f"{f} resembles a placeholder SHA; replace before running")
    for f in ("dataset_hash", "panel_hash", "birth_checkpoint_hash"):
        value = obj.get(f, "")
        if not isinstance(value, str) or not value.startswith("sha256:"):
            errors.append(f"{f} must include sha256: digest provenance")
        elif "REPLACE" in value.upper():
            warnings.append(f"{f} contains placeholder digest; replace before running")
    seeds = obj.get("seeds", [])
    if not isinstance(seeds, list) or len(seeds) < 1:
        errors.append("seeds must be a nonempty array")
        seeds = []
    elif len(seeds) != len({str(s) for s in seeds}):
        errors.append("seeds contain duplicates: repeated runs are not new seeds")
    arms = obj.get("arms", [])
    if not isinstance(arms, list) or len(arms) == 0:
        errors.append("arms must be a nonempty array")
        arms = []
    by_id = {}
    for arm in arms:
        if not isinstance(arm, dict):
            errors.append("each arm must be an object")
            continue
        aid = arm.get("id")
        if not aid or aid in by_id:
            errors.append(f"missing or duplicate arm id: {aid!r}")
        by_id[aid] = arm
        for k in ("role", "comparator_type", "stream_id", "config"):
            if k not in arm:
                errors.append(f"arm {aid} missing {k}")
        if arm.get("comparator_type") not in {"matched", "ceiling", "diagnostic"}:
            errors.append(f"arm {aid} has invalid comparator_type")
        if not isinstance(arm.get("config"), dict):
            errors.append(f"arm {aid} config must be an object")
        if arm.get("role") == "ceiling" and arm.get("comparator_type") != "ceiling":
            errors.append(f"arm {aid}: ceiling role must be labeled ceiling comparator")
    for aid, arm in by_id.items():
        base_id = arm.get("baseline")
        if not base_id:
            continue
        base = by_id.get(base_id)
        if base is None:
            errors.append(f"arm {aid} names missing baseline {base_id}")
            continue
        if not isinstance(arm.get("config"), dict) or not isinstance(base.get("config"), dict):
            continue
        all_keys = set(base["config"]) | set(arm["config"])
        diff = sorted(k for k in all_keys if base["config"].get(k) != arm["config"].get(k))
        expected = arm.get("intended_changes")
        if not isinstance(expected, list):
            errors.append(f"arm {aid} needs explicit intended_changes compared with {base_id}")
        elif sorted(expected) != diff:
            errors.append(f"arm {aid} changed config keys {diff}, but intended_changes is {sorted(expected)}")
        if arm.get("comparator_type") == "matched" and arm.get("stream_id") != base.get("stream_id"):
            errors.append(f"arm {aid} claims matched comparison but stream_id differs from {base_id}")
        if arm.get("comparator_type") == "ceiling" and arm.get("stream_id") != base.get("stream_id"):
            warnings.append(f"arm {aid} uses a richer/different stream: ceiling only, not causal matched-control")
        if any(k in (expected or []) for k in ("architecture", "loss", "replay", "decay", "learning_rate")):
            warnings.append(f"arm {aid}: changed more than update rule; carefully qualify causal conclusion")
    preds = obj.get("predictions", [])
    seen_pred = set()
    if not isinstance(preds, list) or not preds:
        errors.append("predictions must be a nonempty array")
        preds = []
    for p in preds:
        if not isinstance(p, dict):
            errors.append("each prediction must be an object")
            continue
        for field in ("id", "arm", "metric", "operator", "threshold", "population", "pass_policy"):
            if field not in p or p[field] in (None, ""):
                errors.append(f"prediction {p.get('id', '?')} missing {field}")
        pid = p.get("id")
        if pid in seen_pred:
            errors.append(f"duplicate prediction id {pid}")
        seen_pred.add(pid)
        if p.get("arm") not in by_id:
            errors.append(f"prediction {pid} references unknown arm {p.get('arm')}")
        if p.get("operator") not in {">", ">=", "<", "<=", "=="}:
            errors.append(f"prediction {pid} must use explicit operator")
        if not isinstance(p.get("threshold"), (int, float)) or isinstance(p.get("threshold"), bool):
            errors.append(f"prediction {pid}: threshold must be numeric")
        if p.get("pass_policy") not in {"all_seeds", "at_least_n_seeds", "mean"}:
            errors.append(f"prediction {pid}: unknown pass_policy")
        if p.get("pass_policy") == "at_least_n_seeds" and not isinstance(p.get("min_seeds"), int):
            errors.append(f"prediction {pid}: at_least_n_seeds needs integer min_seeds")
    if obj.get("primary_metric") and not any(p.get("metric") == obj["primary_metric"] for p in preds if isinstance(p, dict)):
        warnings.append("No preregistered prediction explicitly uses primary_metric")
    if not obj.get("hard_subgroups"):
        warnings.append("No hard/shortcut subgroups identified")
    if "gradients_reachable" not in obj.get("quality_gates", []):
        warnings.append("No gradients_reachable gate: consider per-token/per-layer audit")
    if "gradient_reachability_invariant_fails" not in obj.get("halt_if", []):
        warnings.append("No halt on gradient-reachability failure")
    if obj.get("run_authorization", "").upper().startswith("NOT_AUTHORIZED"):
        warnings.append("Experiment has not been authorized to launch")
    return errors, warnings


def validate_receipts(obj, receipts):
    errors, warnings = [], []
    if isinstance(receipts, dict):
        receipts = receipts.get("receipts")
    if not isinstance(receipts, list):
        return ["Receipts root must be a JSON list or {receipts:[...]}"], warnings
    arm_ids = {a.get("id") for a in obj.get("arms", []) if isinstance(a, dict)}
    seed_ids = {str(s) for s in obj.get("seeds", [])}
    locked = parse_time(obj.get("locked_at"))
    identities = set()
    observed = set()
    for i, r in enumerate(receipts):
        if not isinstance(r, dict):
            errors.append(f"receipt {i} not an object")
            continue
        arm, seed = r.get("arm"), str(r.get("seed"))
        if arm not in arm_ids or seed not in seed_ids:
            errors.append(f"receipt {i} unknown arm/seed: {arm}/{seed}")
        run_id = r.get("run_id")
        key = (arm, seed, run_id)
        if key in identities:
            errors.append(f"duplicate receipt identity {key}")
        identities.add(key)
        observed.add((arm, seed))
        started = parse_time(r.get("started_at"))
        if not started:
            errors.append(f"receipt {i} has missing/naive started_at")
        elif locked and started < locked:
            errors.append(f"receipt {i} started before prediction lock")
        if r.get("dataset_hash") != obj.get("dataset_hash"):
            errors.append(f"receipt {i} dataset_hash mismatch")
        if r.get("panel_hash") != obj.get("panel_hash"):
            errors.append(f"receipt {i} panel_hash mismatch")
        if r.get("status") not in {"COMPLETE", "PARTIAL", "INVALIDATED", "STOPPED"}:
            errors.append(f"receipt {i} invalid/missing status")
    for arm in sorted(arm_ids):
        missing = sorted(seed_ids - {s for a, s in observed if a == arm})
        if missing:
            warnings.append(f"arm {arm} lacks receipts for seeds {missing}; not fully filed")
    return errors, warnings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("contract", type=Path)
    p.add_argument("--receipts", type=Path)
    args = p.parse_args()
    try:
        data = json.loads(args.contract.read_text(encoding="utf-8"))
        errs, warns = validate_contract(data)
        if args.receipts:
            r = json.loads(args.receipts.read_text(encoding="utf-8"))
            more_errs, more_warns = validate_receipts(data, r)
            errs.extend(more_errs)
            warns.extend(more_warns)
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: failed to read input: {e}")
        return 2
    for msg in errs:
        print(f"ERROR: {msg}")
    for msg in warns:
        print(f"WARN:  {msg}")
    print(f"Summary: {len(errs)} error(s), {len(warns)} warning(s). Lint != science verification.")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
