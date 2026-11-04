"""Load the frozen V195-R1 source with its relative parent path intact."""
import importlib.util
import os
from pathlib import Path

PARENT = Path(__file__).resolve().parent.parent / "minifly_v195_selective_revision_design_20261103" / "r1" / "run.py"
spec = importlib.util.spec_from_file_location("v195_parent", PARENT)
parent = importlib.util.module_from_spec(spec)
cwd = Path.cwd()
try:
    os.chdir(PARENT.parent)
    spec.loader.exec_module(parent)
    parent.PARENT = parent.PARENT.resolve()
finally:
    os.chdir(cwd)
