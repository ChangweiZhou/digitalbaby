"""Read-only parent imports, with all cache writes confined to this new exercise."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / 'r_center_core_v1'
sys.dont_write_bytecode = True
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / 'scratch' / 'numba'))
for path in (PARENT / 'vendor' / 'src', PARENT / 'vendor' / 'package'):
    sys.path.insert(0, str(path))
import paths  # noqa: F401,E402
