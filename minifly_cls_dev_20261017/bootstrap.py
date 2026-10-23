"""Path setup: the native Full151 fly is imported read-only from minifly_a_v3 (its src/ and extracted package/)."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")
SRC = ROOT / "minifly_a_v3" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import stores  # noqa: E402,F401  (minifly_a_v3 native stores: canonical newborn Full151, logged/signed teach)
