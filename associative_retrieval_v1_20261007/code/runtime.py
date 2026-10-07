"""Import immutable parent runtime. This release has no science dispatcher."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent / 'next_core_mechanism_v1_20261006'
sys.dont_write_bytecode = True
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'technical/numba')
sys.path.insert(0, str(PARENT))
import bootstrap  # noqa: E402,F401
sys.path.insert(0, str(Path(__file__).resolve().parent))

