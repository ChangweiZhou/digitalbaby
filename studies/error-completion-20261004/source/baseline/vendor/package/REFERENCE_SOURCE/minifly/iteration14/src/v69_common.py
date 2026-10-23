from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'iteration13/src'))
from v68_common import np,pd,PortFly,SmallFly,FrozenReader,events,rest,state_error,fingerprint,trained,measure,sha
from v66_common import analytic_baseline
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
DATA=ROOT/'data';OUT=ROOT/'results';MODEL=BASE/'iteration10/models/reference.npz';READER=BASE/'iteration8/models/current_RBF64.npz'
CFG=json.loads((ROOT/'config.json').read_text());SEEDS=CFG['development_seeds'];ARMS=CFG['arms']
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def ci(v):
    a=np.asarray(v,float);a=a[np.isfinite(a)]
    if len(a)==0:return dict(mean=None,low=None,high=None,n=0)
    rng=np.random.default_rng(6909001);b=a[rng.integers(0,len(a),(2000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)),n=len(a))
def locked():
    lk=json.loads((ROOT/'checks/lock.json').read_text())
    assert sha(ROOT/'config.json')==lk['config_sha'] and sha(ROOT/'PROTOCOL.md')==lk['protocol_sha']
    assert all(sha(ROOT/'src'/n)==h for n,h in lk['code'].items())
    return lk
def batch_probe(m,codes,roles,reader):
    """Independent neutral five-second probes, vectorized over cues; no state mutation."""
    end=1-(1-m.adapt*np.exp(-.25*codes))*np.exp(-5/m.ta);dx=.5*(m.adapt+end)*codes
    drive=dx@m.Q;gamma=np.clip((dx*(m.k0[3]+m.fast[:,0]))@m.Q[:,:2]+35.2,0,71.66)-35.2
    alpha=np.clip((dx*(m.k0[5]+m.fast[:,1]+m.slow))@m.Q[:,2:]+gamma@m.MM+11.2,0,31.16)-11.2
    zero=analytic_baseline(drive,m.k0,m.MM);obs=reader.predict(dx)
    mem=(alpha-zero).mean(1);read=(alpha-obs).mean(1);direction=2*np.asarray(roles)-1
    actual=direction*(mem[::2]-mem[1::2]);r=direction*(read[::2]-read[1::2])
    return actual,r,(actual>=1)&(r>=reader.pair_threshold)
