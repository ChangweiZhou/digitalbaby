"""V68 helpers; immutable parent imports use distinct module names."""
from pathlib import Path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'iteration12/src'))
from v67_common import np,pd,PortFly,SmallFly,FrozenReader,events,state_error,state,rest,fingerprint,trained,measure,sha
from ports import port_step
from feasibility_check import probe_parts
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
DATA=ROOT/'data';OUT=ROOT/'results';MODEL=BASE/'iteration10/models/reference.npz';READER=BASE/'iteration8/models/current_RBF64.npz'
SEEDS=list(range(6701001,6701017))

def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def ci(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    if not len(a):return dict(mean=None,low=None,high=None,n=0)
    rng=np.random.default_rng(6809001);b=a[rng.integers(0,len(a),(2000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)),n=len(a))
def parents():
    r=json.loads((BASE/'iteration12/checks/lock.json').read_text())['parent_assets'].copy()
    names=['iteration12/src/ports.py','iteration12/src/v67_common.py','iteration12/src/campaign.py','iteration12/src/analyze.py',
           'iteration12/checks/lock.json','iteration12/config.json','iteration12/PROTOCOL.md','iteration12/data/input_banks.npz',
           'iteration12/models/local_logistic.npz','iteration12/results/L_classifier_summary.csv','iteration12/REPORT.md',
           'v68_review/REPORT.md','v68_review/anatomy_audit/checks.json','iteration11/checks/lock.json']
    for n in names:r[n]=sha(BASE/n)
    for seed in SEEDS:
        folder=BASE/'iteration12/data/runs'/str(seed)
        for n in ('checkpoint.npz','local_features.npz','local_outcomes.csv','parent_capacity.csv','receipt.json'):
            r[str((folder/n).relative_to(BASE))]=sha(folder/n)
    return r
def locked():
    lock=json.loads((ROOT/'checks/lock.json').read_text())
    assert sha(ROOT/'config.json')==lock['config_sha'] and sha(ROOT/'PROTOCOL.md')==lock['protocol_sha']
    assert all(sha(ROOT/'src'/n)==h for n,h in lock['code'].items())
    return lock
