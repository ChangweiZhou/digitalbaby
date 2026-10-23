# SPDX-License-Identifier: GPL-3.0-or-later
"""Use exact archived Learner class with explicit new-study governance adapter."""
import importlib.util,sys,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'baseline'
# Bind new programme governance before importing the byte-identical learner.
import integrity
assert Path(integrity.__file__).resolve()==HERE/'integrity.py'
sys.path.insert(0,str(BASE/'src'))
os.environ.setdefault('NUMBA_CACHE_DIR',str(HERE.parents[1]/'operations'/'numba'))
_spec=importlib.util.spec_from_file_location('survivor_frozen_learner',BASE/'src/learner.py')
_mod=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_mod)
Learner=_mod.Learner
ALPHABET=_mod.ALPHABET
SCALES=(1.4911274663291492,1.3452365735750882)
