"""Package-local import paths; the V3 scaffold under package/ is never modified."""
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "scratch" / "numba_cache"))
PKG = ROOT / "package"
ROUND = PKG / "REFERENCE_SOURCE" / "MINIFLY_THREE_MECHANISM_ROUND_20260928"
CONTENT = PKG / "REFERENCE_SOURCE" / "FULL151_VISIBLE_CONTEXT_BRIDGE_20260927"
BYTECORE9 = PKG / "REFERENCE_SOURCE" / "BYTE_CORE_V9"
for p in (str(ROUND), str(PKG), str(CONTENT), str(BYTECORE9)):
    if p not in sys.path:
        sys.path.insert(0, p)
