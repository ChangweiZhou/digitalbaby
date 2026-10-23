"""V76 imports immutable parents; current experiment paths are local."""
from pathlib import Path
import sys,os,json,time,copy
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'iteration20/src'))
from common75 import np,pd,sha,atomic_json,FrozenReader,READER,MODEL,Fly,probe,train,schedule,observed_activity,state_error
from model73 import Fly as NativeBase
from bridge75 import Bridge as Bridge75
sys.path.insert(0,str(ROOT/'src'))
CFG=json.loads((ROOT/'config.json').read_text())

def bank(seed):
    with np.load(ROOT/'data/inputs.npz') as z:return z[f'F_{seed}'].copy(),z[f'roles_{seed}'].copy()

def confidence(x):
    x=np.asarray(x,float);assert len(x) and np.isfinite(x).all()
    rng=np.random.default_rng(CFG['bootstrap_seed'])
    a=x[rng.integers(len(x),size=(CFG['bootstrap_draws'],len(x)))].mean(1)
    return dict(mean=float(x.mean()),low=float(np.quantile(a,.025)),high=float(np.quantile(a,.975)),n=len(x))

def sign_p(x):
    x=np.asarray(x,float);n=len(x);assert n<=16
    s=2*((np.arange(2**n)[:,None]>>np.arange(n))&1)-1
    return float(np.mean(abs(s@x/n)>=abs(x.mean())-1e-14))

def holm(ps):
    out=np.empty(len(ps));last=0.
    for j,i in enumerate(np.argsort(ps)):last=max(last,min(1.,(len(ps)-j)*ps[i]));out[i]=last
    return out

def unsigned_observe(m,codes,reader,h=0):
    """Evaluator returns actual and reader separately; online caller uses reader only."""
    return probe(m,codes,np.ones(len(codes)//2,dtype=int),reader,h)

def state_arrays(m):return np.concatenate([m.fast.ravel(),m.slow,m.adapt])
