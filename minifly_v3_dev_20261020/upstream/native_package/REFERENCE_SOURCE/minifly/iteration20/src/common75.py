from pathlib import Path
import os,sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'iteration19/src'))
from common74 import np,pd,atomic_json,sha,FrozenReader,READER,MODEL,events,train,schedule,metrics
from model74 import Fly,probe
sys.path.insert(0,str(ROOT/'src'))
CFG=json.loads((ROOT/'config.json').read_text())

def bank(seed):
    with np.load(ROOT/'data/inputs.npz') as z:return z[f'F_{seed}'].copy(),z[f'roles_{seed}'].copy()

def confidence(x):
    x=np.asarray(x,float);assert len(x) and np.isfinite(x).all()
    rng=np.random.default_rng(CFG['bootstrap_seed']);b=x[rng.integers(len(x),size=(CFG['bootstrap_draws'],len(x)))].mean(1)
    return dict(mean=float(x.mean()),low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)),n=len(x))

def sign_p(x):
    x=np.asarray(x,float);n=len(x)
    if n<=16:
        signs=2*((np.arange(2**n)[:,None]>>np.arange(n))&1)-1
        return float(np.mean(abs(signs@x/n)>=abs(x.mean())-1e-14))
    rng=np.random.default_rng(CFG['bootstrap_seed']);b=(rng.integers(0,2,(50000,n))*2-1)@x/n
    return float((1+np.sum(abs(b)>=abs(x.mean())-1e-14))/(len(b)+1))

def holm(ps):
    order=np.argsort(ps);out=np.empty(len(ps));last=0.
    for j,i in enumerate(order):last=max(last,min(1.,(len(ps)-j)*ps[i]));out[i]=last
    return out

def state_error(a,b):
    return max(float(np.max(abs(getattr(a,k)-getattr(b,k)))) for k in ['fast','slow','adapt'])

def observed_activity(m,codes):
    end=1-(1-m.adapt*np.exp(-.25*codes))*np.exp(-5/m.ta)
    return .5*(m.adapt+end)*codes

def clip_fraction(m,codes,h=0):
    n=m.clone()
    if h:n.mode='no_learning';n.step(h)
    dx=observed_activity(n,codes)
    gamma=np.clip((dx*(n.k0[3]+n.fast[:,0]))@n.Q[:,:2]+35.2,0,71.66)-35.2
    raw=(dx*(n.k0[5]+n.fast[:,1]+n.slow))@n.Q[:,2:]+gamma@n.MM
    return float(((raw<=-11.2)|(raw>=19.96)).mean())
