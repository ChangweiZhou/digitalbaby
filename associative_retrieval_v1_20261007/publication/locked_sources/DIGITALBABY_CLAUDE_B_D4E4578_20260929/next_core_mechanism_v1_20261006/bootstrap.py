"""Isolated qualification runtime; immutable parents are imported, not changed."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / 'r_center_core_v1'
COMPACT = ROOT.parent / 'r_center_compact_ablation_v1'
V2 = ROOT.parent / 'persistent_core_v2_trial'
LATIN = ROOT.parents[1] / 'DIGITALBABY_GITHUB_UPDATES_20261005/studies/rcenter-survivor-rounds-20261003/source/runtime/survivor_fixture.py'
sys.dont_write_bytecode = True
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'scratch/numba')
sys.path.insert(0,str(PARENT))
import centered_core
sys.path.insert(0,str(COMPACT))
sys.path.insert(0,str(ROOT))
