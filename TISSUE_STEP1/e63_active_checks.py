"""E63, decision 66: E63's checks 2 to 7 (step1_checks.py, not edited) run on the copy tissue_step1_active.py instead of tissue_step1.py.
The copy is put in the place of the module tissue_step1 before step1_checks is imported; the receipts go to qual_active/ and record the
copy's SHA-256.

    python3 e63_active_checks.py <2..7>
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tissue_step1_active as C     # noqa: E402

sys.modules["tissue_step1"] = C
import step1_checks as K            # noqa: E402

K.QUAL = HERE / "qual_active"


def write(n, obj):
    K.QUAL.mkdir(exist_ok=True)
    obj["checked_program"] = "tissue_step1_active.py"
    obj["program_sha256"] = C.sha_file(HERE / "tissue_step1_active.py")
    obj["original_sha256"] = C.sha_file(HERE / "tissue_step1.py")
    obj["checks_sha256"] = C.sha_file(HERE / "step1_checks.py")
    obj["versions"] = C.versions()
    obj["git_head"] = C.git_head()
    (K.QUAL / f"check{n}.json").write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


K.write = write

if __name__ == "__main__":
    n = int(sys.argv[1])
    assert 2 <= n <= 7
    K.run_check(n, None)
    print(f"check {n}:", json.loads((K.QUAL / f"check{n}.json").read_text())["pass"], flush=True)
