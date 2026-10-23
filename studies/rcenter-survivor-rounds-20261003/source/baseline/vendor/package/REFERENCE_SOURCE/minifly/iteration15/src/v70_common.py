"""V70 imports immutable V69/parent helpers; no parent code is copied or changed."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'iteration14/src'))
from v69_common import np,pd,PortFly,FrozenReader,events,rest,fingerprint,state_error,batch_probe,sha
from actions import ActionFly
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
    a=np.asarray(x,float);assert len(a)>0 and np.isfinite(a).all()
    r=np.random.default_rng(CFG['bootstrap_seed']);b=a[r.integers(len(a),size=(CFG['bootstrap_draws'],len(a)))].mean(1)
    return dict(mean=float(a.mean()),low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)),n=len(a))
def context(seed,load,case):
    b=np.load(BASE/'iteration12/data/input_banks.npz');f=np.load(BASE/'iteration14/data/reserved_inputs.npz')
    old=b[f'F_{seed}'];roles=b[f'roles_{seed}'];new=f[f'F_{seed}'];nr=f[f'roles_{seed}'];li=[12,24,48,96].index(load)
    cp=PortFly.restore(BASE/f'iteration14/data/runs/{seed}/checkpoint_{load}.npz',MODEL)
    focal=int((((SEEDS.index(seed)+li)%4)+.5)*load/4)
    A,B=new[2*li:2*li+2] if case=='NEW' else old[2*focal:2*focal+2]
    role=int(nr[li]) if case=='NEW' else 1-int(roles[focal]);target=load if case=='NEW' else focal
    pc=np.array([cp.encode(p) for p in np.vstack((old[:2*load],new[2*li:2*li+2]))]);pr=np.r_[roles[:load],nr[li]];pr[target]=role
    return cp,A,B,role,pc,pr,target
def metrics(a,r,b,target,no,pa):
    old=np.arange(len(a))!=target;d=np.maximum(no[0][old]-a[old],0)
    return dict(target_actual=float(a[target]),target_reader=float(r[target]),target_both=float(b[target]),target_gain=float(a[target]-no[0][target]),
      old_n=int(old.sum()),old_D=float(d.mean()),old_D_sum=float(d.sum()),old_worst_D=float(d.max()),old_p95_D=float(np.quantile(d,.95)),
      old_both=float(b[old].mean()),old_joint_loss=int((pa[2][old]&~b[old]).sum()),old_joint_loss_rate=float((pa[2][old]&~b[old]).mean()))
