"""Import immutable, already-qualified ERROR learning and fixtures."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'next_core_mechanism_v1_20261006'
sys.dont_write_bytecode=True
sys.path.insert(0,str(PARENT))
import bootstrap
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'scratch/numba')
from core import Core
from assays import make_world,DT,RECORD_SECONDS
from auditor import instrument,check_record
from worker import mutable_bytes
import checkpoint
from io_utils import atomic_json,write_receipt,read_receipt,digest
