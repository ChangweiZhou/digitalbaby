"""Operations layer around immutable, already-qualified LINK science."""
import sys
from pathlib import Path

OPS = Path(__file__).resolve().parent
ROOT = OPS.parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import runtime  # noqa: E402,F401
sys.path.insert(0, str(OPS))

