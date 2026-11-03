"""Check the coverage and source identity declared by a rehearsal receipt."""
import argparse
import json
from pathlib import Path
import re

REQUIRED = frozenset({
    "input_preflight", "evaluation", "full_batch", "residual_batch",
    "checkpoint_roundtrip", "branch_restore", "reduction", "receipt_audit",
    "package_readback",
})

def validate(launch, receipt):
    """Return all failed checks. This validates evidence shape, not its authorship."""
    errors = []
    sources = launch.get("source_sha256")
    if (not isinstance(sources, dict) or not sources
            or any(not isinstance(v, str) or not re.fullmatch("[0-9a-f]{64}", v)
                   for v in sources.values())
            or sources != receipt.get("source_sha256")):
        errors.append("SOURCE_LOCK_MISMATCH")
    if not launch.get("entrypoint") or launch["entrypoint"] != receipt.get("entrypoint"):
        errors.append("ENTRYPOINT_MISMATCH")
    if receipt.get("status") != "COMPLETED" or receipt.get("fixture_only") is not True:
        errors.append("REHEARSAL_NOT_COMPLETED")
    coverage = receipt.get("coverage")
    if not isinstance(coverage, dict):
        coverage = {}
    required = REQUIRED | set(launch.get("required_coverage", []))
    errors.extend(f"MISSING_PASS:{name}" for name in sorted(required) if coverage.get(name) != "PASS")
    return errors

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch")
    parser.add_argument("receipt")
    args = parser.parse_args()
    try:
        launch, receipt = (json.loads(Path(p).read_text()) for p in (args.launch, args.receipt))
        errors = validate(launch, receipt)
    except (OSError, ValueError, TypeError, AttributeError) as error:
        errors = [f"INVALID_INPUT:{error}"]
    print(json.dumps({"status": "FAIL" if errors else "PASS_DECLARED_REHEARSAL_ONLY",
                      "errors": errors}))
    return int(bool(errors))

if __name__ == "__main__":
    raise SystemExit(main())
