"""Exact archived native-package location; no historical checkout is changed."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'upstream'
PKG = ROOT / 'native_package'
ROUND = PKG / 'REFERENCE_SOURCE/MINIFLY_THREE_MECHANISM_ROUND_20260928'
CONTENT = PKG / 'REFERENCE_SOURCE/FULL151_VISIBLE_CONTEXT_BRIDGE_20260927'
BYTECORE9 = PKG / 'REFERENCE_SOURCE/BYTE_CORE_V9'
for path in (ROUND, PKG, CONTENT, BYTECORE9):
    sys.path.insert(0, str(path))
