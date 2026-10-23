from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'iteration6/src'))
from retention_model import RetentionLearner
from prepare_interface import patterns,encode
sys.path.insert(0,str(BASE/'iteration8/src'))
from readout import predict_drive,analytic_baseline
import numpy as np
import pandas as pd
CONFIG=json.loads((ROOT/'config.json').read_text())
DATA=ROOT/'data';OUT=ROOT/'results';MODELS=ROOT/'models'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def state_error(a,b):
    return max(float(np.max(abs(np.asarray(getattr(a,k),float)-np.asarray(getattr(b,k),float)))) for k in ('fast','slow','adapt','elapsed','event_count','presentation_count'))
def events(A,B,punished_role=0,bouts=6):
    for _ in range(bouts):
        for role,p in enumerate((A,B)):
            yield 30.,p,float(role==punished_role)
            yield 135.,None,0.
def train(m,A,B,role=0,bouts=6):
    for dt,p,s in events(A,B,role,bouts):m.step(dt,p,s)

def source_inputs():
    with np.load(BASE/'iteration9/data/input_banks.npz') as f:return np.vstack([f[k] for k in f.files])
