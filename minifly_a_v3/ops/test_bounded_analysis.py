"""Isolated, bounded qualification. No simulation and no production final output.

One original-reference child followed by one adapter child. Each uses exactly
26 real receipts: all 13 frozen arms, worlds 190001 and 190002, selected in code
before inspecting endpoint values. Original audits remain enabled and delegated.
The only test-only scientific-global restriction is WORLDS (same both runs).
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import gc
import gzip
import hashlib
import io
import json
import math
import os
import pickle
import random
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

sys.dont_write_bytecode = True
import bounded_analysis as ba

ROOT = ba.ROOT
SAMPLE_WORLDS = (190001, 190002)


def code_identity():
    return {"adapter_sha256": ba.sha256(Path(ba.__file__).read_bytes()),
            "test_sha256": ba.sha256(Path(__file__).read_bytes())}


IMPORTED_CODE_IDENTITY = code_identity()


def verify_code_identity(expected):
    ba.check(code_identity() == expected, "adapter/test code changed during qualification")


def code_identity_tests():
    expected = code_identity()
    outcomes = {}
    for key in expected:
        changed = {**expected, key: "0" * 64}
        outcomes[key + "_change_rejected"] = expect_error(
            lambda: verify_code_identity(changed), fragment="code changed")
    return outcomes


def digest(value):
    return ba.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def expect_error(action, exception=(AssertionError, FileNotFoundError, gzip.BadGzipFile, EOFError), fragment=None):
    try:
        action()
    except exception as exc:
        if fragment is not None:
            assert fragment in str(exc), (fragment, type(exc).__name__)
        return type(exc).__name__
    raise AssertionError("expected integrity failure did not occur")


def storage_tests(base):
    base.mkdir()
    path = base / "R0/190001.json.gz"
    path.parent.mkdir()
    key, name = ("R0", 190001), "R0/190001"
    value = {"arm": "R0", "world": 190001, "rows": [1, 2, 3], "nan": float("nan")}
    raw = gzip.compress(json.dumps(value).encode(), mtime=0)
    h = ba.sha256(raw)
    outcomes = {}

    def fresh():
        path.write_bytes(raw)
        declared = {name: h}
        mapping = ba.ReceiptBackedMapping(base, declared)
        mapping[key] = json.loads(gzip.decompress(raw))
        return mapping, declared

    mapping, declared = fresh()
    a, b = mapping[key], mapping[key]
    assert a is not b and a["rows"] == b["rows"] and math.isnan(a["nan"]) and math.isnan(b["nan"])
    a["rows"][0] = 99
    assert mapping[key]["rows"] == [1, 2, 3]
    assert list(mapping) == [key] and len(mapping) == 1
    assert all(isinstance(p, Path) and isinstance(s, str) for p, s in mapping._pins.values())
    mapping.recheck_all()
    outcomes["fresh_reads_nan_and_no_decoded_cache"] = True
    outcomes["duplicate_registration_rejected"] = expect_error(lambda: mapping.__setitem__(key, value))
    outcomes["deletion_unsupported"] = expect_error(lambda: mapping.__delitem__(key), (AttributeError,))
    mapping, declared = fresh()
    path.unlink()
    outcomes["missing_on_access"] = expect_error(lambda: mapping[key])
    outcomes["missing_on_final_recheck"] = expect_error(mapping.recheck_all)
    outcomes["missing_on_registration"] = expect_error(
        lambda: ba.ReceiptBackedMapping(base, declared).__setitem__(key, value))
    mapping, declared = fresh()
    path.write_bytes(b"corrupt gzip")
    outcomes["corrupt_after_registration"] = expect_error(lambda: mapping[key], fragment="changed after")
    outcomes["corrupt_on_final_recheck"] = expect_error(mapping.recheck_all)
    outcomes["corrupt_before_registration"] = expect_error(
        lambda: ba.ReceiptBackedMapping(base, declared).__setitem__(key, value), fragment="during registration")
    mapping, declared = fresh()
    # Valid gzip with identical decompressed bytes, but a different header, must also fail.
    path.write_bytes(gzip.compress(json.dumps(value).encode(), mtime=1))
    outcomes["compressed_bytes_mutated_same_json"] = expect_error(lambda: mapping[key])
    outcomes["valid_gzip_mutated_final_recheck"] = expect_error(mapping.recheck_all)
    mapping, declared = fresh()
    declared[name] = "0" * 64
    outcomes["digest_reference_mutated"] = expect_error(lambda: mapping[key], fragment="original digest changed")
    outcomes["digest_reference_mutated_final_recheck"] = expect_error(mapping.recheck_all)
    mapping, declared = fresh()
    declared["R1/190001"] = "0" * 64
    outcomes["extra_digest_roster_entry"] = expect_error(mapping.recheck_all, fragment="roster mismatch")
    outcomes["unregistered_access"] = expect_error(lambda: mapping[("R0", 9)], (KeyError,))
    mapping, declared = fresh()
    del declared[name]
    outcomes["missing_digest_roster_entry"] = expect_error(mapping.recheck_all)
    corrupt = gzip.compress(b"not valid JSON", mtime=0)
    path.write_bytes(corrupt)
    outcomes["invalid_json_original_load"] = expect_error(
        lambda: json.loads(gzip.decompress(path.read_bytes())), (json.JSONDecodeError,))
    path.write_bytes(b"invalid gzip")
    outcomes["corrupt_gzip_original_load"] = expect_error(lambda: json.loads(gzip.decompress(path.read_bytes())))
    return outcomes


def ast_tests():
    source = ba.SOURCE.read_bytes()
    original, adapted = ba.transform_source(source)
    outcomes = {"exact_single_initializer_change": True}
    changed = copy.deepcopy(adapted)
    alpha = next(n for n in changed.body if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == "ALPHA" for t in n.targets))
    alpha.value.value = 0.04
    outcomes["different_statistical_literal_rejected"] = expect_error(
        lambda: ba.assert_only_storage_edit(original, changed), fragment="beyond loaded")
    changed = copy.deepcopy(adapted)
    ba._initializer(changed).value.args.reverse()
    outcomes["different_factory_arguments_rejected"] = expect_error(
        lambda: ba.assert_only_storage_edit(original, changed), fragment="replacement expression")
    changed = copy.deepcopy(adapted)
    main = next(n for n in changed.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    main.body.append(copy.deepcopy(ba._initializer(changed)))
    outcomes["extra_initializer_rejected"] = expect_error(lambda: ba.assert_only_storage_edit(original, changed))
    outcomes["source_byte_change_rejected"] = expect_error(lambda: ba.transform_source(source + b"\n"),
                                                          fragment="source hash mismatch")
    # AST/location round trip catches an altered output operation too.
    changed = copy.deepcopy(adapted)
    constant = next(n for n in ast.walk(changed) if isinstance(n, ast.Constant) and n.value == "FINAL_METRICS.json")
    constant.value = "OTHER.json"
    outcomes["output_operation_change_rejected"] = expect_error(lambda: ba.assert_only_storage_edit(original, changed))
    return outcomes


def acceptance_path_tests(base):
    """Synthetic operational tests only; never call scientific main or touch results.

    Stub only analyzer/environment dependencies inside this temporary fixture.
    Production entry point, guards, marker publication and quarantine are real.
    """
    base.mkdir()
    saved = (ba.ROOT, ba.__file__, ba.verify_frozen_environment, ba.load_analyzer)
    outcomes = {}
    runtime = {"synthetic_fixture": True}
    try:
        for case in ("accept", "concurrent_run", "running", "incomplete", "existing", "existing_marker", "bad_equivalence",
                     "adapter_mismatch", "receipt_recheck_failure", "runtime_changed",
                     "qualification_changed", "adapter_changed", "test_changed"):
            root = base / case
            (root / "ops").mkdir(parents=True)
            (root / "results/science").mkdir(parents=True)
            adapter = root / "ops/bounded_analysis.py"
            adapter.write_bytes(Path(saved[1]).read_bytes())
            harness = root / "ops/test_bounded_analysis.py"
            harness.write_bytes(Path(__file__).read_bytes())
            ba.ROOT, ba.__file__ = root, str(adapter)
            status = {"running": [], "validated_receipts": 832}
            if case == "running":
                status["running"] = [["R0", 190001]]
            if case == "incomplete":
                status["validated_receipts"] = 831
            (root / "results/science/RUN_STATUS.json").write_text(json.dumps(status))
            q = {"schema": "MINIFLY-A3-BOUNDED-ANALYSIS-QUALIFICATION-v1",
                 **{k: True for k in ("pass", "test_only", "byte_exact_final_metrics",
                    "byte_exact_stdout", "identical_scientific_call_order_arguments_and_returns",
                    "original_audits_enabled", "source_immutable", "source_lock_immutable",
                    "qualification_code_identity_unchanged")},
                 "numerical_tolerance_used": None, "runtime": runtime,
                 "adapter_sha256": ba.sha256(adapter.read_bytes()),
                 "test_sha256": ba.sha256(harness.read_bytes())}
            if case == "bad_equivalence":
                q["byte_exact_final_metrics"] = False
            if case == "adapter_mismatch":
                q["adapter_sha256"] = "0" * 64
            qual = root / "qualification.json"
            qual.write_text(json.dumps(q))
            output = root / "results/FINAL_METRICS.json"
            marker = root / "results/FINAL_ANALYSIS_STORAGE_VERIFICATION.json"
            if case == "existing":
                output.write_bytes(b"prior output must be untouched")
            if case == "existing_marker":
                marker.write_bytes(b"prior marker must be untouched")
            verification_calls = []
            def verify():
                verification_calls.append(True)
                return {"changed": True} if case == "runtime_changed" and len(verification_calls) > 1 else runtime
            ba.verify_frozen_environment = verify
            calls = []
            def fake_main():
                calls.append(True)
                if case == "concurrent_run":
                    # A second invocation must fail before it reaches main or
                    # can quarantine/overwrite this invocation's output.
                    expect_error(lambda: ba.run_full(qual), fragment="admission lock")
                    probe = (
                        "import fcntl,sys\n"
                        "with open(sys.argv[1], 'a+b') as handle:\n"
                        " try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
                        " except BlockingIOError: pass\n"
                        " else: raise AssertionError('concurrent process acquired admission lock')\n"
                    )
                    subprocess.run([sys.executable, "-B", "-c", probe,
                                    str(root / "scratch/bounded_analysis_runs/FULL_ANALYSIS.lock")], check=True)
                output.write_bytes(b'{"synthetic_operational_test": true}\n')
                if case == "qualification_changed":
                    qual.write_bytes(qual.read_bytes() + b"\n")
                if case == "adapter_changed":
                    adapter.write_bytes(adapter.read_bytes() + b"\n")
                if case == "test_changed":
                    harness.write_bytes(harness.read_bytes() + b"\n")
            class FakeRegistry:
                def __len__(self):
                    return 832
                def recheck_all(self):
                    if case == "receipt_recheck_failure":
                        raise ba.StorageIntegrityError("synthetic changed receipt")
                def evidence(self):
                    return {"synthetic": True}
            ba.load_analyzer = lambda: (types.SimpleNamespace(main=fake_main, ROSTER=range(13), WORLDS=range(64)),
                                        [FakeRegistry()])
            if case in ("accept", "concurrent_run"):
                result = ba.run_full(qual)
                assert result["pass"] is True and marker.exists() and output.exists()
                assert json.loads(marker.read_text())["final_metrics_sha256"] == ba.sha256(output.read_bytes())
                assert len(calls) == 1
                # The lock is released, but a completed result remains protected.
                expect_error(lambda: ba.run_full(qual), fragment="refusing to overwrite")
                outcomes[case] = True
            else:
                outcomes[case] = expect_error(lambda: ba.run_full(qual))
                if case == "existing_marker":
                    assert marker.read_bytes() == b"prior marker must be untouched"
                else:
                    assert not marker.exists()
                if case == "existing":
                    assert output.read_bytes() == b"prior output must be untouched" and not calls
                else:
                    assert not output.exists()
                if case in ("running", "incomplete", "existing", "existing_marker", "bad_equivalence", "adapter_mismatch"):
                    assert not calls
                else:
                    assert len(calls) == 1
                    assert len(list((root / "scratch/bounded_analysis_runs").rglob("UNACCEPTED_FINAL_METRICS.json"))) == 1
    finally:
        ba.ROOT, ba.__file__, ba.verify_frozen_environment, ba.load_analyzer = saved
    return outcomes


def rss_bytes():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise AssertionError("RSS unavailable")


def child_case(case, directory, expected_code_identity):
    # No broad global paths mutation: only this analyzer module receives a ROOT proxy.
    # lock/auditor/fixture still import the original locked paths module.
    assert case in ("original", "adapted")
    assert directory.is_relative_to(ROOT / "scratch")
    ba.check(sys.flags.optimize == 0, "qualification requires Python assertions enabled")
    verify_code_identity(expected_code_identity)
    ba.check(IMPORTED_CODE_IDENTITY == expected_code_identity, "child imported unqualified adapter/test code")
    directory.mkdir(parents=True, exist_ok=False)
    runtime = ba.verify_frozen_environment()
    analyzer, registry = ba.load_analyzer(adapted=(case == "adapted"))
    assert tuple(analyzer.ROSTER) == tuple(json.loads((ROOT / "SOURCE_LOCK.json").read_text())["arms"])
    assert len(analyzer.ROSTER) * len(SAMPLE_WORLDS) == 26
    sample_manifest = {}
    for arm in analyzer.ROSTER:
        for world in SAMPLE_WORLDS:
            p = analyzer.RES / arm / f"{world}.json.gz"
            sample_manifest[f"{arm}/{world}"] = ba.sha256(p.read_bytes())
    (directory / "results/calibration").mkdir(parents=True)
    cal = ROOT / "results/calibration/R_CALIBRATION.json"
    shutil.copyfile(cal, directory / "results/calibration/R_CALIBRATION.json")
    assert ba.sha256(cal.read_bytes()) == ba.sha256((directory / "results/calibration/R_CALIBRATION.json").read_bytes())
    analyzer.paths = types.SimpleNamespace(ROOT=directory)
    analyzer.WORLDS = SAMPLE_WORLDS  # Explicit test-only restriction, no change to ROSTER or statistical constants.
    assert analyzer.RES == ROOT / "results/science"
    trace = hashlib.sha256()
    counts = {}
    audit_order, mutation_checks = [], []

    def event(name, args):
        counts[name] = counts.get(name, 0) + 1
        trace.update(json.dumps([name, args], sort_keys=True, separators=(",", ":")).encode() + b"\n")

    original_audit = analyzer.audit_a.audit_receipt

    def audited(r, **kwargs):
        # Establish that the real frozen auditor leaves receipt/yoke objects untouched.
        before = digest([r, kwargs])
        event("audit", {"args_sha256": digest([r]), "kwargs_sha256": digest(kwargs)})
        answer = original_audit(r, **kwargs)
        event("audit_return", answer)
        assert digest([r, kwargs]) == before, "frozen audit mutated receipt or yoke"
        mutation_checks.append(True)
        audit_order.append([r["arm"], r["world"]])
        return answer

    analyzer.audit_a.audit_receipt = audited
    for name in ("metrics", "acc", "interval", "describe"):
        original = getattr(analyzer, name)
        def wrapped(*args, _name=name, _original=original, **kwargs):
            event(_name, {"args_sha256": digest(args), "kwargs_sha256": digest(kwargs)})
            answer = _original(*args, **kwargs)
            event(_name + "_return", answer)
            return answer
        setattr(analyzer, name, wrapped)
    original_t = analyzer.student_t
    def ppf(*args, **kwargs):
        event("student_t_ppf", [args, kwargs])
        answer = original_t.ppf(*args, **kwargs)
        event("student_t_ppf_return", float(answer))
        return answer
    analyzer.student_t = types.SimpleNamespace(ppf=ppf)
    import numpy as np
    random.seed(170812)  # Identical harness starting states, does not change any frozen seeded fixture.
    np.random.seed(170812)
    rng_before = ba.sha256(pickle.dumps([random.getstate(), np.random.get_state()]))
    gc.collect()
    baseline = rss_bytes()
    started = time.monotonic()
    console = io.StringIO()
    with contextlib.redirect_stdout(console):
        analyzer.main()
    if case == "adapted":
        assert len(registry) == 1 and len(registry[0]) == 26
        registry[0].recheck_all()
        assert registry[0].accesses == 30  # 26 primary reads plus two yokes per world.
    else:
        assert registry == []
    assert ba.verify_frozen_environment() == runtime
    verify_code_identity(expected_code_identity)
    rng_after = ba.sha256(pickle.dumps([random.getstate(), np.random.get_state()]))
    assert rng_after == rng_before, "analysis consumed global Python/NumPy RNG state"
    assert audit_order == [[a, w] for a in analyzer.ROSTER for w in SAMPLE_WORLDS]
    assert len(mutation_checks) == 26 and all(mutation_checks)
    for name, expected in sample_manifest.items():
        arm, world = name.split("/")
        assert ba.sha256((ROOT / "results/science" / arm / f"{world}.json.gz").read_bytes()) == expected
    output = (directory / "results/FINAL_METRICS.json").read_bytes()
    assert json.loads(output)["receipt_sha256"] == sample_manifest
    assert json.loads(output)["family_size_m"] == 26
    assert json.loads(output)["package_contrasts_available"] == 12
    assert list(json.loads(output)["package_contrasts_unavailable"]) == ["P3"]
    result = {"case": case, "test_only": True, "runtime": runtime,
              "code_identity": expected_code_identity, "code_identity_after_unchanged": True,
              "worlds": list(SAMPLE_WORLDS),
              "arms": list(analyzer.ROSTER), "sample_receipts": 26, "sample_sha256": sample_manifest,
              "output_sha256": ba.sha256(output), "stdout_sha256": ba.sha256(console.getvalue().encode()),
              "scientific_call_trace_sha256": trace.hexdigest(), "scientific_call_counts": counts,
              "audit_calls": len(audit_order), "auditor_receipt_and_yoke_nonmutation_checks": len(mutation_checks),
              "global_rng_state_unchanged": True, "runtime_identity_after_unchanged": True,
              "baseline_rss_bytes": baseline, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "elapsed_seconds": time.monotonic() - started,
              "storage": registry[0].evidence() if registry else {"type": "original in-memory dict"}}
    (directory / "CASE_EVIDENCE.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    # Endpoint values are deliberately not printed by the qualification child.
    print(json.dumps({"case": case, "pass": True, "audit_calls": len(audit_order)}))


def qualify(destination):
    ba.check(sys.flags.optimize == 0, "qualification requires Python assertions enabled")
    expected_code_identity = code_identity()
    ba.check(IMPORTED_CODE_IDENTITY == expected_code_identity, "imported adapter/test code changed before qualification")
    assert not destination.resolve().is_relative_to(ROOT / "results"), "qualification must stay outside real results"
    assert not destination.exists(), "qualification output exists; select a new filename"
    run_dir = ROOT / "scratch/bounded_analysis_qualification" / os.urandom(10).hex()
    run_dir.mkdir(parents=True, exist_ok=False)
    production_output = ROOT / "results/FINAL_METRICS.json"
    production_before = ba.sha256(production_output.read_bytes()) if production_output.exists() else None
    unit = {"storage": storage_tests(run_dir / "storage_unit_tests"), "ast": ast_tests(),
            "qualification_code_identity": code_identity_tests(),
            "synthetic_admission_acceptance_quarantine": acceptance_path_tests(run_dir / "acceptance_unit_tests")}
    verify_code_identity(expected_code_identity)
    source_before = ba.sha256(ba.SOURCE.read_bytes())
    lock_before = ba.sha256((ROOT / "SOURCE_LOCK.json").read_bytes())
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    cases = {}
    for case in ("original", "adapted"):
        verify_code_identity(expected_code_identity)
        outdir = run_dir / case
        subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--child", case, "--directory", str(outdir),
                        "--expected-code-identity", json.dumps(expected_code_identity, sort_keys=True)],
                       env=env, check=True)
        verify_code_identity(expected_code_identity)
        cases[case] = json.loads((outdir / "CASE_EVIDENCE.json").read_text())
    a, b = cases["original"], cases["adapted"]
    exact_fields = ("runtime", "code_identity", "worlds", "arms", "sample_receipts", "sample_sha256", "output_sha256",
                    "stdout_sha256", "scientific_call_trace_sha256", "scientific_call_counts", "audit_calls")
    assert all(a[k] == b[k] for k in exact_fields), "original/adapter qualification mismatch"
    assert (run_dir / "original/results/FINAL_METRICS.json").read_bytes() == (
        run_dir / "adapted/results/FINAL_METRICS.json").read_bytes(), "final JSON bytes differ"
    assert ba.sha256(ba.SOURCE.read_bytes()) == source_before
    assert ba.sha256((ROOT / "SOURCE_LOCK.json").read_bytes()) == lock_before
    production_after = ba.sha256(production_output.read_bytes()) if production_output.exists() else None
    assert production_before == production_after, "production final output changed"
    verify_code_identity(expected_code_identity)
    result = {"schema": "MINIFLY-A3-BOUNDED-ANALYSIS-QUALIFICATION-v1", "pass": True,
              "test_only": True, "final_analysis_executed": False, "scientific_roster_restricted_only_in_tests": True,
              "sample_declaration": {"worlds": list(SAMPLE_WORLDS), "all_frozen_arms": True,
                                     "receipts_per_case": 26, "selection": "fixed in harness before endpoint inspection"},
              "ast_edit": "main.loaded empty Dict initializer only; all other nodes/locations identical",
              "source_immutable": True, "source_lock_immutable": True, "production_output_unchanged": True,
              "original_audits_enabled": True, "byte_exact_final_metrics": True,
              "byte_exact_stdout": True, "identical_scientific_call_order_arguments_and_returns": True,
              "numerical_tolerance_used": None, "unit_tests": unit, "cases": cases,
              "runtime": a["runtime"], **expected_code_identity,
              "qualification_code_identity_unchanged": True,
              "evidence_directory": str(run_dir),
              "limitations": ["26-receipt qualification is not a full-roster scientific result",
                              "Production admission/marker/quarantine tested on synthetic fixtures only, never on full science results",
                              "User approved operational amendment; independent parent review still required before production use",
                              "Extra decompression and disk reads trade time for bounded receipt memory",
                              "Exact compressed-byte hash rechecks do not lock out hostile concurrent filesystem writers"]}
    destination.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"pass": True, "test_only": True, "qualification": str(destination),
                      "byte_exact_final_metrics": True, "audits_per_case": 26,
                      "original_peak_rss_bytes": a["peak_rss_bytes"], "adapter_peak_rss_bytes": b["peak_rss_bytes"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=("original", "adapted"))
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--expected-code-identity")
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    if args.child:
        assert args.directory is not None
        assert args.expected_code_identity is not None
        child_case(args.child, args.directory, json.loads(args.expected_code_identity))
    else:
        assert args.qualification is not None
        qualify(args.qualification)


if __name__ == "__main__":
    main()
