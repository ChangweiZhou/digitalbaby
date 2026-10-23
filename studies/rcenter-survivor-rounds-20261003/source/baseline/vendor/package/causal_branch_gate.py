#!/usr/bin/env python3
"""Independent causal-branch gate for the frozen 600-record fixture.

Uses literal expected counts, never the runner's branch-allowance logic.
It is a technical gate, not a science trajectory or an E3 analysis.
"""
from __future__ import annotations

import argparse
import ast
import copy
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "REFERENCE_SOURCE/MINIFLY_THREE_MECHANISM_ROUND_20260928/common_platform.py"
DOMAINS = ("old_fact", "old_relation", "new_fact", "new_relation")
EXPECTED = {
    "W": (768, 288, 768, 144),
    "N_old_fact": (0, 288, 768, 144),
    "N_old_rel": (768, 0, 768, 144),
    "N_new_fact": (768, 288, 0, 144),
    "N_new_rel": (768, 288, 768, 0),
}


def check_function(func) -> None:
    """Compare all 20 literal decisions against the independent fixture matrix."""
    for branch, counts in EXPECTED.items():
        for name, count in zip(DOMAINS, counts, strict=True):
            stage, domain = name.split("_", 1)
            got = func(branch, stage, domain)
            if type(got) is not bool or got != (count != 0):
                raise AssertionError(f"branch decision wrong: {branch}/{name}={got}")
    for args in (("N_old_relation", "old", "relation"),
                 ("W", "old", "rel"), ("W", "future", "fact")):
        try:
            func(*args)
        except ValueError:
            continue
        raise AssertionError(f"invalid branch/stage/domain accepted: {args}")


def check_source(path: Path = SOURCE) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "branch_allows"]
    if len(funcs) != 1:
        raise AssertionError("exactly one branch_allows definition required")
    # Compile only the function: no Full151 or third-party imports are needed.
    node = funcs[0]
    env = {"BRANCHES": tuple(EXPECTED)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(path), "exec"), env)
    check_function(env["branch_allows"])


def check_receipt(receipt: dict) -> None:
    writes = receipt.get("writes")
    if not isinstance(writes, dict) or set(writes) != set(EXPECTED):
        raise AssertionError("receipt branch roster differs from five-branch contract")
    for branch, counts in EXPECTED.items():
        actual = writes[branch]
        if not isinstance(actual, dict) or set(actual) != set(DOMAINS):
            raise AssertionError(f"write-domain roster wrong in {branch}")
        for domain, expected in zip(DOMAINS, counts, strict=True):
            if type(actual[domain]) is not int or actual[domain] != expected:
                raise AssertionError(
                    f"write ledger wrong: {branch}/{domain}={actual[domain]}, expected {expected}")

    first = receipt.get("first")
    if not isinstance(first, list) or len(first) != 600 * len(EXPECTED):
        raise AssertionError("missing first-response records")
    seen: dict[int, dict[str, float]] = {}
    for row in first:
        i, branch, t = row["record"], row["branch"], row["teacher_at"]
        if type(i) is not int or i not in range(600) or branch not in EXPECTED:
            raise AssertionError("unexpected record or branch in first-response roster")
        by_branch = seen.setdefault(i, {})
        if branch in by_branch:
            raise AssertionError("duplicate first-response record")
        by_branch[branch] = t
    if set(seen) != set(range(600)):
        raise AssertionError("first-response record IDs incomplete")
    for i, by_branch in seen.items():
        if set(by_branch) != set(EXPECTED) or len(set(by_branch.values())) != 1:
            raise AssertionError(f"branch exposure or teacher time differs at record {i}")


def self_test() -> dict:
    check_source()
    # The source defect in V2 must be rejected by the independent oracle.
    def old_rule(branch, stage, domain):
        if branch not in EXPECTED or stage not in ("old", "new") or domain not in ("fact", "relation"):
            raise ValueError("invalid")
        return branch != f"N_{stage}_{domain}"
    try:
        check_function(old_rule)
    except AssertionError:
        pass
    else:
        raise AssertionError("V2 relation-branch defect escaped source gate")

    fixture = {
        "writes": {b: dict(zip(DOMAINS, counts, strict=True)) for b, counts in EXPECTED.items()},
        "first": [{"record": i, "branch": b, "teacher_at": float(i)}
                  for i in range(600) for b in EXPECTED],
    }
    check_receipt(fixture)
    rejected = 0
    for b in EXPECTED:
        for domain in DOMAINS:
            damaged = copy.deepcopy(fixture)
            damaged["writes"][b][domain] += 1
            try:
                check_receipt(damaged)
            except AssertionError:
                rejected += 1
            else:
                raise AssertionError(f"tampered {b}/{domain} ledger escaped gate")
    for alteration in ("missing_record", "changed_time"):
        damaged = copy.deepcopy(fixture)
        if alteration == "missing_record":
            damaged["first"].pop()
        else:
            damaged["first"][1]["teacher_at"] += 1.0
        try:
            check_receipt(damaged)
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError(f"tampered {alteration} escaped gate")
    return {"source_decisions": 20, "synthetic_receipt": "passed",
            "v2_defect": "rejected", "tamper_cases_rejected": rejected,
            "science_trajectories_run": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--receipt", type=Path,
                        help="full-life technical receipt JSON or JSON.gz to validate")
    inputs.add_argument("--receipts-dir", type=Path,
                        help="recursively audit every JSON.gz science receipt under this directory")
    args = parser.parse_args()
    result = self_test()
    paths = [args.receipt] if args.receipt is not None else []
    if args.receipts_dir is not None:
        paths = sorted(args.receipts_dir.rglob("*.json.gz"))
        if not paths:
            raise AssertionError("no JSON.gz receipts found")
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            try:
                check_receipt(json.load(stream))
            except AssertionError as exc:
                raise AssertionError(f"{path}: {exc}") from exc
    if paths:
        result["receipts_checked"] = len(paths)
        result["receipt_gate"] = "passed"
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
