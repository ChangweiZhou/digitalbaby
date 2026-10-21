"""Operations and pinned native imports; no task data in the learner dependency path."""
import os, sys, json, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(ROOT/'minifly_v3_dev_20261020'))
import bootstrap
# Parent bootstrap supplies paths only. All new compilation artifacts stay here.
os.environ['NUMBA_CACHE_DIR'] = str(HERE/'scratch/numba')
import numba
numba.config.CACHE_DIR = str(HERE/'scratch/numba')
import numpy as np
from native import NativeBank, event_batch, query_batch
from readout import predict_drive

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    path = HERE/path; path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.pending')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)
def source_lock():
    files=list(HERE.glob('*.py'))+[HERE/'PARAMETER_LOCK.json',HERE/'AUTHORIZATION.json']
    files+=list((HERE/'design').glob('*'))
    files += [ROOT/'minifly_v3_dev_20261020'/f for f in ['native.py','bootstrap.py','paths.py']]
    files += [ROOT/'minifly_a_v3/src/stores.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in files if p.is_file()}

def fresh_bank(n, seed):
    """One immutable canonical anatomy; nine independent stacked mutable states.

    Does not import the old fastmem C16 codebook or independent answer predictor.
    """
    from types import SimpleNamespace
    model, birth = bootstrap.stores.birth('native')
    m=model.fly.m
    parent=SimpleNamespace(bank=[SimpleNamespace(fly=model.fly,birth=birth)]*9,
         side_n={s:int(np.ceil(m.active_fraction*np.sum(m.kc_side==s))) for s in (0,1)})
    b=NativeBank(parent);old=len(b.slow[0]);rng=np.random.default_rng(seed)
    rows=np.arange(old) if n==old else rng.integers(0,old,n)
    for k in ('Q','T','reader_Q','kc_side'):
        a=np.array(getattr(b,k)[rows],copy=True)
        if k in ('Q','reader_Q'): a*=old/n
        setattr(b,k,a)
    b.fast=np.zeros((9,n,2));b.slow=np.zeros((9,n));b.adapt=np.ones((9,n))
    b.side_n={s:int(np.ceil(b.active_fraction*np.sum(b.kc_side==s))) for s in (0,1)}
    return b
