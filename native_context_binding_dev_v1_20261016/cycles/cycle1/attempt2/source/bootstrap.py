"""Retained native sources, read-only; task data never imported here."""
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["NUMBA_CACHE_DIR"] = str(HERE / "scratch" / "numba")
for path in (
    ROOT.parent / "V88/.venv-rce-v87-followup/lib/python3.11/site-packages",
    ROOT / "arithmetic_kernel_no_attention_dev_v3_20261013/.venv/lib/python3.11/site-packages",
    ROOT / "r_center_core_v1/vendor/package",
    ROOT / "r_center_core_v1/vendor/src",
    ROOT / "r_center_core_v1",
):
    if path.is_dir():
        sys.path.insert(0, str(path))
import paths  # noqa: E402,F401


def source_module(name, relative):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
