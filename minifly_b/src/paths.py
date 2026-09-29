"""Package-local import paths; frozen REFERENCE_SOURCE is never modified."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "package"
ROUND = PKG / "REFERENCE_SOURCE" / "MINIFLY_THREE_MECHANISM_ROUND_20260928"
for p in (str(ROUND), str(PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)
