"""Unmodified scientific implementation; new orchestration lives separately."""
import os,sys,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FROZEN=ROOT.parent/'next_core_mechanism_v1_20261006'
DOCUMENTS=ROOT.parents[1]/'PROJECT_DOCUMENTS/MINIFLY_NEXT_CORE_BUDGET_V2_20261006'
sys.dont_write_bytecode=True
sys.path.insert(0,str(FROZEN))
import bootstrap
from core import Core,ChoiceOrgan,policies
from assays import make_world,permitted,DT,RECORD_SECONDS
from io_utils import atomic_json,atomic_bytes,canonical,digest,read_receipt,write_receipt
import worker as frozen_worker
import checkpoint as frozen_checkpoint
import auditor as frozen_auditor
import integrity as frozen_integrity
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'scratch/numba')
import numba
numba.config.CACHE_DIR=os.environ['NUMBA_CACHE_DIR']
sys.path.insert(0,str(ROOT))
SCIENTIFIC_ID='0701272ef469e110a0a5286fe7022cc2103f6cac4e2a357fce47b4c6d9a98fe2'
DEVELOPMENT_IDS=tuple(range(61005001,61005007))
def load_plan():return json.loads((ROOT/'PLAN.json').read_text())
def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_files():
    files={str(p):file_sha(p) for p in ROOT.glob('*.py')}
    for n in ('PLAN.json','SPEC_LOCK.md','requirements.txt'):files[str(ROOT/n)]=file_sha(ROOT/n)
    files.update(frozen_integrity.source_map())
    return files
def execution_identity():return digest(source_files())
def require_sources(expected=None):
    if frozen_integrity.identity()!=SCIENTIFIC_ID:raise ValueError('frozen scientific source changed')
    if expected is not None and execution_identity()!=expected:raise ValueError('execution source changed')
def require_environment():
    import numpy,scipy,numba,pandas
    got=[sys.version.split()[0],numpy.__version__,scipy.__version__,numba.__version__,pandas.__version__]
    if got!=['3.11.5','2.2.6','1.14.1','0.61.2','2.2.3']:raise ValueError('pinned environment mismatch')
    return got
