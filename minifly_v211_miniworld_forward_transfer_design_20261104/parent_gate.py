"""Fail closed when executable provenance or replay parity is absent."""
import hashlib
import json
from pathlib import Path
import tempfile

def check(source, receipt):
    if not source.is_file() or not receipt.is_file():
        return "MISSING_ROUND15_PARENT"
    record=json.loads(receipt.read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest()!=record.get("source_sha256"):
        return "SOURCE_MISMATCH"
    # A hash-matched source does not constitute behavioral parity.
    return "NEEDS_INDEPENDENT_REPLAY"

if __name__=="__main__":
    status=check(Path("parent/round15.py"),Path("parent/REPLAY.json"))
    assert status=="MISSING_ROUND15_PARENT"
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)
        (p/"fixture.py").write_text("# synthetic fixture\n")
        (p/"fixture.json").write_text('{"source_sha256":"wrong"}')
        assert check(p/"fixture.py",p/"fixture.json")=="SOURCE_MISMATCH"
    Path("CYCLE1_TEST.json").write_text(json.dumps(dict(status=status,technical_check="PASS",
                                                       training_questions=0))+"\n")
