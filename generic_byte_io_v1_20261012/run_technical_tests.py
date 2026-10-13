"""Run the bounded DEV API test suite, not formal scientific trajectories."""

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time
import unittest

import numpy as np

from generic_byte_io import ByteCore


def main():
    root = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    output = io.StringIO()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    elapsed = time.perf_counter() - started
    text = output.getvalue()
    files = [root / "generic_byte_io" / "__init__.py", root / "generic_byte_io" / "core.py", root / "tests" / "test_core.py", Path(__file__)]
    report = {
        "schema": "GENERIC_BYTE_IO_TECHNICAL_TESTS_V1",
        "classification": "E0 interface tests plus bounded E1 fixtures; not formal science",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "elapsed_seconds": elapsed, "python": sys.version,
        "numpy": np.__version__, "platform": platform.platform(),
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "storage": ByteCore().storage_accounting(),
        "formal_science_jobs": 0,
    }
    destination = root / "technical"
    destination.mkdir(exist_ok=True)
    (destination / "UNIT_TESTS.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "unit_tests.log").write_text(text)
    print(text, end="")
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
