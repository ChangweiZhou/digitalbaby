"""Package-local import paths; frozen REFERENCE_SOURCE is never modified."""
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
# numba's on-disk JIT cache must never land inside the verified package tree
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "scratch" / "numba_cache"))
PKG = ROOT / "package"
ROUND = PKG / "REFERENCE_SOURCE" / "MINIFLY_THREE_MECHANISM_ROUND_20260928"
for p in (str(ROUND), str(PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)
