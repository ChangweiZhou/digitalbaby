"""V74 support. No Codex/network dependency; unchanged parents remain external."""
from pathlib import Path
import sys, json, hashlib, os, time, tempfile
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
sys.path.insert(0,str(BASE/'iteration18/src'))
from v73_common import np,pd,MODEL,READER,FrozenReader,events
from model73 import Fly as ParentFly
sys.path.insert(0,str(ROOT/'src'))
CFG=json.loads((ROOT/'config.json').read_text()) if (ROOT/'config.json').exists() else {}

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(obj,indent=2,allow_nan=False)+'\n'
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def ci(x):
    x=np.asarray(x,float);assert x.ndim==1 and len(x) and np.isfinite(x).all()
    rng=np.random.default_rng(CFG.get('bootstrap_seed',7409001))
    y=x[rng.integers(len(x),size=(5000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),low=float(np.quantile(y,.025)),high=float(np.quantile(y,.975)),n=len(x))

def bank(seed,fresh=False):
    path=ROOT/'data/fresh_inputs.npz' if fresh else BASE/'iteration18/data/input_banks.npz'
    with np.load(path) as z:return z[f'F_{seed}'].copy(),z[f'roles_{seed}'].copy()

def load_config(name):return CFG['audit_configs'][name].copy()

def finite(m):
    if not all(np.isfinite(getattr(m,k)).all() for k in ['fast','slow','adapt']):raise FloatingPointError('Nonfinite learner state')

def metrics(m,codes,roles,reader,target,oldmask,no=None,initial=None,h=0):
    from model74 import probe
    a,r,b=probe(m,codes,roles,reader,h)
    out=dict(target_actual=float(a[target]),target_reader=float(r[target]),target_choice=float(b[target]),
             old_accuracy=float(b[oldmask].mean()),target_sign=float(a[target]>0))
    for th in [.5,1.,2.]:out[f'target_joint_at_{th:g}']=float(a[target]>=th and r[target]>=max(th,reader.pair_threshold))
    if no is not None:
        na=probe(no,codes,roles,reader,h)[0];d=np.maximum(na[oldmask]-a[oldmask],0)
        out.update(old_damage=float(d.mean()),old_worst=float(d.max()))
    if initial is not None:
        im=initial[oldmask[:len(initial)]];now=b[:len(initial)][oldmask[:len(initial)]]
        out['old_loss']=float((im&~now).mean())
    return out

def schedule(m,A,B,role,bouts):
    for dt,p,u in events(A,B,int(role),bouts):
        yield (dt*m.settings.get('gap',1.) if not np.any(p) else dt),p,u

def train(m,sched,callback=None):
    for dt,p,u in sched:
        r=m.step(dt,p,u)
        if callback is not None:callback(dt,r)
    finite(m)

def prior_refs():
    lock=json.loads((BASE/'iteration18/checks/lock.json').read_text())
    refs=lock['parent_assets'].copy()
    for n in ['iteration18/config.json','iteration18/models/anatomy.npz','iteration18/data/input_banks.npz',
              'iteration18/checks/lock.json','iteration18/REPORT.md','iteration18/results/decision.json']:
        refs[n]=sha(BASE/n)
    for p in (BASE/'iteration18/src').glob('*.py'):refs[str(p.relative_to(BASE))]=sha(p)
    for seed in CFG['surgery_seeds']:
        for load in CFG['loads']:
            p=BASE/f'iteration18/data/runs/{seed}/REF_{load}.npz';refs[str(p.relative_to(BASE))]=sha(p)
    return refs
