import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent/'autonomous_observation_v1_20261007'
sys.dont_write_bytecode = True
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'scratch/numba'))
sys.path.insert(0,str(PARENT))
import core as native_core
import spec as native_spec
import streams as native_streams
import checkpoint as native_checkpoint
