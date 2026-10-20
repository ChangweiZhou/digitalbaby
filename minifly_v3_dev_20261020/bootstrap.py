"""Read-only native assets; this module contains no task data or answer logic."""
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(HERE / 'scratch/numba')
import paths
sys.path.insert(0, str(ROOT / 'minifly_a_v3/src'))
import stores


def upstream_fastmem():
    name = 'v3_pinned_fastmem'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / 'upstream/fastmem.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
