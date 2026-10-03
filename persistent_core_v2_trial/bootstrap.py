import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / 'r_center_core_v1'
COMPACT = ROOT.parent / 'r_center_compact_ablation_v1'
sys.dont_write_bytecode = True
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'scratch/numba')
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(COMPACT))
import centered_core  # Load the immutable vendor paths once, before local names.
sys.path.insert(0, str(ROOT))
