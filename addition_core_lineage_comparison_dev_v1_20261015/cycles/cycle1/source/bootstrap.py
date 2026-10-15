"""Source-only imports and explicit existing compatible dependency overlay."""
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(HERE / 'scratch' / 'numba')
TORCH_SITE = ROOT / 'arithmetic_kernel_no_attention_dev_v3_20261013/.venv/lib/python3.11/site-packages'
NATIVE_SITE = ROOT.parent / 'V88/.venv-rce-v87-followup/lib/python3.11/site-packages'
for p in (NATIVE_SITE, TORCH_SITE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
PARENT = ROOT / 'r_center_core_v1'
for p in (PARENT / 'vendor/package', PARENT / 'vendor/src'):
    sys.path.insert(0, str(p))
import paths

def source_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
