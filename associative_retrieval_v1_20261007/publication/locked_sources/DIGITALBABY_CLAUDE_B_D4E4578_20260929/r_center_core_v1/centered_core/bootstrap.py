"""Package-local, frozen native dependencies; no scientific runner is imported."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / 'scratch' / 'numba'))
sys.path.insert(0, str(ROOT / 'vendor' / 'src'))
import paths  # noqa: E402,F401
