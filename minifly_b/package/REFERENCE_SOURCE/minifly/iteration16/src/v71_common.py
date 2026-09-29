"""Immutable parent imports with V71-specific paths and configuration."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'iteration15/src'))
from v70_common import np,pd,PortFly,FrozenReader,events,rest,fingerprint,state_error,batch_probe,sha,context
from v70_actions import waterfill
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
DATA=ROOT/'data';OUT=ROOT/'results';MODEL=BASE/'iteration10/models/reference.npz';READER=BASE/'iteration8/models/current_RBF64.npz'
CFG=json.loads((ROOT/'config.json').read_text());SEEDS=CFG['seeds']
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def locked():
    r=json.loads((ROOT/'checks/lock.json').read_text())
    assert sha(ROOT/'PROTOCOL.md')==r['protocol_sha'] and sha(ROOT/'config.json')==r['config_sha']
    assert all(sha(ROOT/'src'/p)==h for p,h in r['code'].items())
    return r
def ci(x):
    a=np.asarray(x,float);assert len(a) and np.isfinite(a).all();rng=np.random.default_rng(CFG['bootstrap_seed'])
    b=a[rng.integers(len(a),size=(CFG['bootstrap_draws'],len(a)))].mean(1)
    return dict(mean=float(a.mean()),low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)),n=len(a))
def metrics(result,target,no,parent):
    a,r,b=result;old=np.arange(len(a))!=target;d=np.maximum(no[0][old]-a[old],0)
    return dict(target_actual=float(a[target]),target_reader=float(r[target]),target_both=float(b[target]),old_D=float(d.mean()),old_worst=float(d.max()),old_p95=float(np.quantile(d,.95)),
      old_both=float(b[old].mean()),old_joint_loss=float((parent[2][old]&~b[old]).mean()),old_n=int(old.sum()))
def probe(m,h,pc,pr,reader,n):return tuple(z[:n] for z in batch_probe(rest(m,h),pc,pr,reader))
