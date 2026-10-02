"""Operational, no-cache receipt storage for the unchanged Package A V3 analysis.

Preparation/import does not run analysis. The only analyzer AST edit replaces the
local ``loaded = {}`` initializer. This file is deliberately outside SOURCE_LOCK.
See BOUNDED_ANALYSIS_ADAPTER.md for the qualification and acceptance boundary.
"""
from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import gzip
import hashlib
import json
import os
import sys
import types
from collections.abc import Mapping
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/analyze_a.py"
EXPECTED_SOURCE_SHA256 = "809d9f3ddf6efdd0df691fdf37501c8a1f5ffcc581f1e2448ac1bcbebcc833e2"
EXPECTED_LOCK_SHA256 = "f321f6c00e43a7f476728522357d50514444e7e5d7cfe2ca3996e2c8b8976572"
EXPECTED_LOCK_DIGEST = "2d8692585aee41cd9e9a70be53b0f656df10447f9b0d3b04b3163e96173d5310"
FACTORY_NAME = "_a3_receipt_backed_mapping_factory"


class StorageIntegrityError(AssertionError):
    """A receipt, pinned source, or qualification changed or is incomplete."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check(condition, message):
    if not condition:
        raise StorageIntegrityError(message)


class ReceiptBackedMapping(Mapping):
    """Write-once registration, with fresh hash-verified JSON on every retrieval.

    No decoded receipts or compressed contents are retained. The original main
    decodes each receipt at registration; the value is discarded only after its
    on-disk compressed bytes match the digest just recorded by original main.
    Every read hashes a single bytes snapshot and decodes that same snapshot.
    """
    __slots__ = ("_base", "_declared_sha", "_pins", "_sealed", "registrations",
                 "accesses", "rechecks")

    def __init__(self, base: Path, declared_sha: dict):
        self._base = Path(base)
        self._declared_sha = declared_sha
        self._pins = {}
        self._sealed = False
        self.registrations = self.accesses = self.rechecks = 0

    @staticmethod
    def _name(key):
        check(isinstance(key, tuple) and len(key) == 2 and
              isinstance(key[0], str) and type(key[1]) is int, "invalid receipt key")
        arm, world = key
        check(arm and Path(arm).name == arm and arm not in (".", ".."), "invalid arm path")
        return f"{arm}/{world}"

    def __setitem__(self, key, value):
        name = self._name(key)
        check(not self._sealed and key not in self._pins, f"receipt registration is write-once: {name}")
        check(isinstance(value, dict), f"decoded receipt is not a dictionary: {name}")
        expected = self._declared_sha.get(name)
        check(isinstance(expected, str) and len(expected) == 64, f"missing original digest: {name}")
        path = self._base / key[0] / f"{key[1]}.json.gz"
        raw = path.read_bytes()
        check(sha256(raw) == expected, f"receipt changed during registration: {name}")
        self._pins[key] = (path, expected)
        self.registrations += 1
        # Do not save value: original main supplied json.loads(gzip.decompress(raw)).

    def _read_verified(self, key):
        path, expected = self._pins[key]
        name = self._name(key)
        check(self._declared_sha.get(name) == expected, f"original digest changed: {name}")
        raw = path.read_bytes()
        check(sha256(raw) == expected, f"receipt changed after registration: {name}")
        return raw

    def __getitem__(self, key):
        raw = self._read_verified(key)
        self.accesses += 1
        return json.loads(gzip.decompress(raw))

    def __iter__(self):
        return iter(self._pins)

    def __len__(self):
        return len(self._pins)

    def recheck_all(self):
        check(set(self._declared_sha) == {self._name(k) for k in self._pins},
              "digest/registered roster mismatch")
        for key in self._pins:
            self._read_verified(key)
        self.rechecks += 1
        self._sealed = True

    def evidence(self):
        return {"registered": self.registrations, "accesses": self.accesses,
                "complete_rechecks": self.rechecks, "decoded_receipts_cached": 0,
                "receipt_sha256": {self._name(k): h for k, (_, h) in self._pins.items()}}


def _initializer(tree):
    mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    check(len(mains) == 1, "expected exactly one top-level main")
    candidates = [n for n in ast.walk(mains[0]) if isinstance(n, ast.Assign) and
                  any(isinstance(t, ast.Name) and t.id == "loaded" for t in n.targets)]
    check(len(candidates) == 1, "expected exactly one loaded initializer")
    node = candidates[0]
    check(node in mains[0].body and len(node.targets) == 1 and
          isinstance(node.targets[0], ast.Name) and node.targets[0].id == "loaded",
          "loaded initializer is not the expected direct assignment")
    return node


def assert_only_storage_edit(original: ast.Module, adapted: ast.Module):
    """Fail unless replacing the one Call with the original Dict restores all AST bytes.

    Locations are included: no executable node, literal, statement order, or
    source position outside this single expression may differ.
    """
    before = _initializer(original)
    after = _initializer(adapted)
    check(isinstance(before.value, ast.Dict) and not before.value.keys and not before.value.values,
          "original loaded initializer is not an empty dictionary")
    expected = ast.Call(func=ast.Name(id=FACTORY_NAME, ctx=ast.Load()),
                        args=[ast.Name(id="RES", ctx=ast.Load()),
                              ast.Name(id="receipts_sha", ctx=ast.Load())], keywords=[])
    check(ast.dump(after.value, include_attributes=False) == ast.dump(expected, include_attributes=False),
          "unexpected storage replacement expression")
    restored = copy.deepcopy(adapted)
    _initializer(restored).value = copy.deepcopy(before.value)
    check(ast.dump(original, include_attributes=True) == ast.dump(restored, include_attributes=True),
          "analyzer AST changed beyond loaded initializer")


def transform_source(source: bytes):
    check(sha256(source) == EXPECTED_SOURCE_SHA256, "frozen analyzer source hash mismatch")
    original = ast.parse(source, filename=str(SOURCE))
    check(not any(isinstance(n, ast.Name) and n.id == FACTORY_NAME for n in ast.walk(original)),
          "storage factory name collides with analyzer")
    adapted = copy.deepcopy(original)
    node = _initializer(adapted)
    node.value = ast.copy_location(ast.Call(func=ast.Name(id=FACTORY_NAME, ctx=ast.Load()),
        args=[ast.Name(id="RES", ctx=ast.Load()), ast.Name(id="receipts_sha", ctx=ast.Load())],
        keywords=[]), node.value)
    ast.fix_missing_locations(adapted)
    assert_only_storage_edit(original, adapted)
    return original, adapted


def verify_frozen_environment():
    check(sha256((ROOT / "SOURCE_LOCK.json").read_bytes()) == EXPECTED_LOCK_SHA256,
          "SOURCE_LOCK file hash mismatch")
    check(sha256(SOURCE.read_bytes()) == EXPECTED_SOURCE_SHA256, "frozen analyzer source hash mismatch")
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import lock
    check(Path(lock.__file__).resolve() == ROOT / "src/lock.py", "wrong lock module imported")
    locked = lock.verify_lock()  # Original complete file and environment checks, not a replacement.
    check(locked["lock_digest"] == EXPECTED_LOCK_DIGEST, "source lock digest mismatch")
    return {"lock_digest": locked["lock_digest"], "source_lock_sha256": EXPECTED_LOCK_SHA256,
            "analyzer_sha256": EXPECTED_SOURCE_SHA256, "environment": lock._env(),
            "executable": sys.executable, "executable_sha256": sha256(Path(sys.executable).read_bytes())}


def load_analyzer(*, adapted=True):
    source = SOURCE.read_bytes()
    original, changed = transform_source(source)
    registry = []

    def factory(base, declared_sha):
        check(not registry, "multiple loaded mappings instantiated")
        mapping = ReceiptBackedMapping(base, declared_sha)
        registry.append(mapping)
        return mapping

    module = types.ModuleType("_a3_bounded_analyzer" if adapted else "_a3_frozen_analyzer_reference")
    module.__file__ = str(SOURCE)
    module.__dict__[FACTORY_NAME] = factory
    exec(compile(changed if adapted else original, str(SOURCE), "exec"), module.__dict__)
    return module, registry


def run_full(qualification_path: Path):
    """Hold one process-wide admission lock through analysis and publication.

    The lock file is persistent and must never be unlinked: removing it could
    let another invocation lock a different inode while this one still runs.
    """
    lock_dir = ROOT / "scratch" / "bounded_analysis_runs"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / "FULL_ANALYSIS.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StorageIntegrityError("another full analysis invocation holds the admission lock") from exc
        return _run_full_exclusive(qualification_path)


def _run_full_exclusive(qualification_path: Path):
    """Explicit admission only; never called by qualification tests.

    Original main writes the normal FINAL_METRICS.json unchanged. Acceptance is
    conditional on the postflight receipt/source recheck and verification marker.
    Any output from a failed invocation is quarantined, never accepted as final.
    """
    runtime = verify_frozen_environment()
    qualification_bytes = qualification_path.read_bytes()
    q = json.loads(qualification_bytes)
    check(q.get("schema") == "MINIFLY-A3-BOUNDED-ANALYSIS-QUALIFICATION-v1" and
          all(q.get(k) is True for k in ("pass", "test_only", "byte_exact_final_metrics",
              "byte_exact_stdout", "identical_scientific_call_order_arguments_and_returns",
              "original_audits_enabled", "source_immutable", "source_lock_immutable",
              "qualification_code_identity_unchanged")),
          "passing exact-equivalence test qualification required")
    check(q.get("numerical_tolerance_used") is None, "tolerance-based qualification is not accepted")
    adapter_sha = sha256(Path(__file__).read_bytes())
    test_sha = sha256((ROOT / "ops/test_bounded_analysis.py").read_bytes())
    qualification_sha = sha256(qualification_bytes)
    check(q.get("adapter_sha256") == adapter_sha, "qualified adapter hash mismatch")
    check(q.get("test_sha256") == test_sha,
          "qualified test harness hash mismatch")
    check(q.get("runtime") == runtime, "qualified runtime/source identity mismatch")
    status = json.loads((ROOT / "results/science/RUN_STATUS.json").read_text())
    check(status.get("running") == [], "science workers still running according to RUN_STATUS")
    check(status.get("validated_receipts") == 832, "full validated roster required")
    module, registry = load_analyzer()
    output = ROOT / "results/FINAL_METRICS.json"
    marker = ROOT / "results/FINAL_ANALYSIS_STORAGE_VERIFICATION.json"
    check(not output.exists() and not marker.exists(), "refusing to overwrite existing final analysis")
    run_dir = ROOT / "scratch" / "bounded_analysis_runs" / os.urandom(12).hex()
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        module.main()  # All frozen scientific calls and output operations, same order.
        check(len(registry) == 1 and len(registry[0]) == len(module.ROSTER) * len(module.WORLDS) == 832,
              "full registered roster mismatch")
        registry[0].recheck_all()
        check(verify_frozen_environment() == runtime, "source/runtime changed during analysis")
        check(sha256(Path(__file__).read_bytes()) == adapter_sha and
              sha256((ROOT / "ops/test_bounded_analysis.py").read_bytes()) == test_sha and
              sha256(qualification_path.read_bytes()) == qualification_sha,
              "adapter/test/qualification changed during analysis")
        evidence = {"schema": "MINIFLY-A3-BOUNDED-ANALYSIS-ACCEPTANCE-v1", "pass": True,
                    "runtime": runtime, "adapter_sha256": adapter_sha,
                    "qualification_sha256": qualification_sha,
                    "final_metrics_sha256": sha256(output.read_bytes()),
                    "storage": registry[0].evidence()}
        pending_marker = run_dir / "VERIFIED_ACCEPTANCE.json"
        pending_marker.write_text(json.dumps(evidence, indent=1, sort_keys=True) + "\n")
        os.link(pending_marker, marker)  # Atomic, complete marker, refuses any overwrite.
        return evidence
    except BaseException:
        if output.exists():
            output.replace(run_dir / "UNACCEPTED_FINAL_METRICS.json")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-full-qualified", action="store_true",
                        help="Explicitly execute all 832 receipts only after independent review")
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    if not args.run_full_qualified or not args.qualification:
        parser.error("no default analysis: explicit --run-full-qualified and --qualification are required")
    run_full(args.qualification)


if __name__ == "__main__":
    main()
