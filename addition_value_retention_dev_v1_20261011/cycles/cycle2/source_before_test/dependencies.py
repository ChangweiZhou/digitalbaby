"""Read-only import of the qualified native dependency; caches stay here."""
import os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'scratch/numba'))
PARENT=ROOT.parent/'autonomous_observation_v1_20261007'
sys.path.insert(0,str(PARENT))
import core as inherited
import checkpoint as inherited_checkpoint


# Restore the new project's module search after immutable parent imports.
sys.path.insert(0,str(ROOT))
