import os,sys,json,hashlib,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
# Immutable v3 native engine is the audited parent, not a separately scored learner.
sys.path.insert(0,str(ROOT/'minifly_v3_dev_20261020'))
import bootstrap
os.environ['NUMBA_CACHE_DIR']=str(HERE/'scratch/numba')
import numpy as np
from native import NativeBank,event_batch,query_batch
from readout import predict_drive

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p=HERE/p;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.pending');q.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');q.replace(p)
def spec():return json.loads((HERE/'SPEC.json').read_text())
def lock():
 files=list(HERE.glob('*.py'))+[HERE/'SPEC.json',HERE/'design/EXPERIMENT.md']
 files+=list((ROOT/'minifly_v3_dev_20261020').glob('*.py'))
 files+=list((ROOT/'minifly_a_v3/src').glob('*.py'))
 return {str(p.relative_to(ROOT)):sha(p) for p in files}
