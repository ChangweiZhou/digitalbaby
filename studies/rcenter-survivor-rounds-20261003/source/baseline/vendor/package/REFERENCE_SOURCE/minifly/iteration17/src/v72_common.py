"""Version-specific orchestration around immutable, previously verified parent functions."""
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'iteration16/src'))
from v71_common import np,pd,PortFly,FrozenReader,events,rest,fingerprint,state_error,batch_probe,sha,context
from receivers import HistoryFly,transport_plan,allocate as previous_allocate,intervene,verify_transport
from v70_actions import waterfill
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
DATA=ROOT/'data';OUT=ROOT/'results';MODEL=BASE/'iteration10/models/reference.npz';READER=BASE/'iteration8/models/current_RBF64.npz'
CFG=json.loads((ROOT/'config.json').read_text());SEEDS=CFG['seeds']
sys.path.insert(0,str(ROOT/'src'))
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def locked():
    r=json.loads((ROOT/'checks/lock.json').read_text())
    assert sha(ROOT/'config.json')==r['config_sha'] and sha(ROOT/'PROTOCOL.md')==r['protocol_sha']
    assert all(sha(ROOT/'src'/n)==h for n,h in r['code'].items())
    assert sha(DATA/'anatomy.npz')==r['anatomy_sha']
    return r
def ci(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    if not len(x):return dict(mean=None,low=None,high=None,n=0)
    rng=np.random.default_rng(CFG['bootstrap_seed']);b=x[rng.integers(len(x),size=(CFG['bootstrap_draws'],len(x)))].mean(1)
    return dict(mean=float(x.mean()),low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)),n=len(x))
def metrics(result,target,no,parent):
    a,r,b=result;old=np.arange(len(a))!=target;d=np.maximum(no[0][old]-a[old],0);k=max(1,int(np.ceil(.1*old.sum())))
    return dict(target_actual=float(a[target]),target_reader=float(r[target]),target_both=float(b[target]),
      old_D=float(d.mean()),old_worst=float(d.max()),old_p95=float(np.quantile(d,.95)),old_CVaR10=float(np.sort(d)[-k:].mean()),
      old_both=float(b[old].mean()),old_joint_loss=float((parent[2][old]&~b[old]).mean()),old_n=int(old.sum()))
class AuditFly(HistoryFly):
    def __init__(self,path):
        super().__init__(path);self.ACT=np.zeros_like(self.slow);self.COUNT=np.zeros_like(self.slow)
    def clone(self,reset_plastic=False):
        m=super().clone(reset_plastic);m.ACT=self.ACT.copy();m.COUNT=self.COUNT.copy()
        if reset_plastic:m.ACT[:]=0;m.COUNT[:]=0
        return m
    def step(self,dt,pn_activity=None,punishment=0.,space=False):
        r=super().step(dt,pn_activity,punishment,space);e=np.exp(-dt/self.kernel['slow_tau'])
        self.ACT=(self.ACT+r['sensory_activity'])*e
        self.COUNT=(self.COUNT+(r['raw_increment'][:,2]!=0))*e
        return r
def save_snapshot(path,m,**kwargs):
    np.savez_compressed(path,fast=m.fast,slow=m.slow,adapt=m.adapt,H=m.H,ACT=m.ACT,COUNT=m.COUNT,elapsed=m.elapsed,**kwargs)
