"""Read existing technical evidence, lock sources and package. Never import a learner."""
import ast
import datetime
import hashlib
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
DESIGN = ROOT / "native_context_binding_design_v1_20261016"
RESULTS = HERE / "results"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    cycles = HERE / "cycles"
    totals = {}
    ledger_details = []
    for path in sorted(cycles.glob("cycle*/attempt*/**/EXECUTION_LEDGER.json")):
        ledger = read(path)
        assert all(row["status"] == "COMPLETED" for row in ledger["rows"])
        count = sum(row["records_executed"] for row in ledger["rows"])
        assert count == ledger["full_record_applications"]
        assert all(row["records_executed"] == row["records_planned"] for row in ledger["rows"])
        cycle = path.relative_to(cycles).parts[0]
        totals[cycle] = totals.get(cycle, 0) + count
        ledger_details.append(dict(path=str(path.relative_to(ROOT)), records=count, sha256=sha(path)))
    assert totals == {"cycle1": 20, "cycle2": 72, "cycle3": 164}, totals
    assert sum(totals.values()) == 256

    snapshot_checks = []
    for path in sorted(cycles.rglob("RESULT.json")) + [cycles / "cycle3/attempt1/pipeline_smoke/RECEIPT.json"]:
        result = read(path)
        if "source_identity" not in result:
            continue
        if "passed" in result:
            assert result["passed"] is True
        snapshot = path.parent / "source"
        # Four-worker job results share their dispatcher's immutable source snapshot.
        if not snapshot.is_dir():
            snapshot = cycles / "cycle3/attempt1/source"
        for name, digest in result["source_identity"].items():
            assert sha(snapshot / name) == digest, (path, name)
        snapshot_checks.append(dict(receipt=str(path.relative_to(ROOT)),
                                   snapshot=str(snapshot.relative_to(ROOT)),
                                   files=len(result["source_identity"])))

    final_receipt = read(cycles / "cycle3/attempt1/pipeline_smoke/RECEIPT.json")
    assert final_receipt["stage"] == "SMOKE"
    assert final_receipt["schema"] == "NATIVE_BIND_TECHNICAL_SMOKE_V1"
    for name, digest in final_receipt["source_identity"].items():
        assert sha(HERE / name) == digest, ("changed qualified source", name)
    independent = read(cycles / "cycle3/attempt1/pipeline_smoke/INDEPENDENT_AUDIT.json")
    assert independent["accepted"] and independent["science"] is False
    assert len(independent["tamper_cases_rejected"]) == 9
    assert independent["unauthorized_science_rejected"]
    assert independent["additional_training_records"] == 0
    science_receipts = []
    for path in cycles.rglob("RECEIPT.json"):
        if read(path).get("schema") == "NATIVE_BIND_SCIENCE_V1":
            science_receipts.append(str(path))
    assert science_receipts == []
    assert not (RESULTS / "DEV").exists() and not (RESULTS / "CONFIRM").exists()

    design = read(DESIGN / "DESIGN.json")
    assert design["technical"]["training_record_application_cap"] == 256
    assert design["science_jobs_executed"] == 0
    assert design["launch_authorized"] is False
    assert len(set(design["confirm"]["worlds"])) == 16
    assert [w for s in design["confirm"]["sessions"] for w in s["worlds"]] == design["confirm"]["worlds"]
    assert sum(s["full_jobs"] for s in design["confirm"]["sessions"]) == 64
    assert sum(s["record_applications"] for s in design["confirm"]["sessions"]) == 262144
    assert design["maximum_science"]["record_applications"] == 49152 + 262144
    for path in HERE.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))

    parent_lock = read(ROOT / "native_interface_transfer_dev_v1_20261015/SOURCE_LOCK.json")
    parent_files = {}
    for name, digest in parent_lock["files"].items():
        if "/.venv/" in name or "/scratch/" in name:
            continue
        assert sha(ROOT / name) == digest, ("frozen parent changed", name)
        # Include source ancestry, not an old run's authority or old reports.
        if Path(name).suffix == ".py" or name.endswith("OUTPUT_CODEBOOK.json") or name == "r_center_core_v1/SOURCE_LOCK.json":
            parent_files[name] = digest
    vendor = ROOT / "r_center_core_v1/vendor"
    for path in vendor.rglob("*"):
        if path.is_file() and path.suffix in (".py", ".npz", ".json", ".txt") and "__pycache__" not in path.parts:
            parent_files[str(path.relative_to(ROOT))] = sha(path)
    operative = dict(parent_files)
    for path in list(HERE.glob("*.py")) + [HERE / "OUTPUT_CODEBOOK.json", DESIGN / "DESIGN.json", DESIGN / "EXPERIMENT.md"]:
        operative[str(path.relative_to(ROOT))] = sha(path)
    lock = dict(schema="NATIVE_BIND_SOURCE_LOCK_V1", utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                operative_sources=dict(sorted(operative.items())), environment=parent_lock["environment"],
                executable="/Users/pencilbard/Downloads/second_try_copy/V88/.venv-rce-v87-followup/bin/python",
                runtime_overlay="arithmetic_kernel_no_attention_dev_v3_20261013/.venv/lib/python3.11/site-packages",
                runtime_binaries_embedded=False, historical_parent_sources_are_not_new_arms=True,
                science_authorized=False, campaign_operations_qualified=False)
    write(HERE / "SOURCE_LOCK.json", lock)

    # Process cleanup is separate from read-only evidence inspection.
    # The restricted filesystem process cannot inspect native process state.
    assert read(HERE / "ops/AWAKE.json")["technical_work_complete"]

    final_audit = dict(schema="NATIVE_BIND_TECHNICAL_FINAL_AUDIT_V1", accepted=True,
        accepted_scope="Three-cycle kernel and single-job technical evidence only",
        verdict="KERNEL_AND_SINGLE_JOB_TECHNICALLY_QUALIFIED", cycle_records=totals,
        technical_records=256, science_jobs=0, remaining_training_budget=0,
        ledger_details=ledger_details, source_snapshots=snapshot_checks,
        qualified_sources_unchanged=True, frozen_parent_files_unchanged=True,
        receipt_tamper_cases_rejected=9, extra_training_in_finalization=0,
        four_worker_state_parity=read(cycles / "cycle3/attempt1/resource/RESULT.json")["concurrency_state_parity"],
        formal_campaign_ready=False, behavioral_efficacy_tested=False,
        launch_authorized=False, upload_authorized=False,
        source_lock_sha256=sha(HERE / "SOURCE_LOCK.json"),
        historical_failure="Cycle1 import-name collision resolved before any record; preserved",
        source_snapshot_disclosure="Cycle1 exact source reconstructed to recorded hashes; all snapshots verified")
    write(RESULTS / "FINAL_AUDIT.json", final_audit)

    files = set()
    bundle = RESULTS / "RESULT_BUNDLE.zip"
    excluded = {bundle, RESULTS / "PACKAGE.json", RESULTS / "BUNDLE_INVENTORY.json"}
    for folder in (HERE, DESIGN):
        for path in folder.rglob("*"):
            if not path.is_file() or path in excluded:
                continue
            if any(part in ("scratch", "__pycache__", ".git", ".venv") for part in path.parts):
                continue
            if path.suffix in (".pyc", ".tmp", ".zip"):
                continue
            files.add(path)
    files.update(ROOT / name for name in parent_files)
    # The local current-pointer is not copied: it includes unrelated historical runs.
    inventory = {str(path.relative_to(ROOT)): sha(path) for path in sorted(files)}
    write(RESULTS / "BUNDLE_INVENTORY.json", dict(schema="NATIVE_BIND_PACKAGE_INVENTORY_V1", files=inventory))
    files.add(RESULTS / "BUNDLE_INVENTORY.json")
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            archive.write(path, str(path.relative_to(ROOT)))
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        for name, digest in inventory.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    package = dict(schema="NATIVE_BIND_TECHNICAL_PACKAGE_V1", complete=True,
        bundle=str(bundle), sha256=sha(bundle), bytes=bundle.stat().st_size,
        files=len(files), technical_records=256, science_jobs=0,
        qualification_scope="Kernel and single-job, not full campaign operations or behavioral efficacy",
        runtime_binaries_embedded=False, github_uploaded=False)
    write(RESULTS / "PACKAGE.json", package)
    print(json.dumps(dict(audit_accepted=True, technical_records=256, science_jobs=0,
                          source_files_locked=len(operative), package_bytes=package["bytes"],
                          package_files=package["files"], campaign_ready=False)))


if __name__ == "__main__":
    main()
