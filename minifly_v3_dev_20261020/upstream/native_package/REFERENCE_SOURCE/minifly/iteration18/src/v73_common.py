from pathlib import Path
import sys,json,time,hashlib,copy
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'iteration17/src'))
from v72_common import np,pd,PortFly,FrozenReader,events,sha,batch_probe as legacy_probe,MODEL,READER,state_error
from smallfly import SmallFly
from hybrid import njit
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;DATA=ROOT/'data';OUT=ROOT/'results'
CFG=json.loads((ROOT/'config.json').read_text());SEEDS=CFG['seeds'];sys.path.insert(0,str(ROOT/'src'))
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def ci(values):
    x=np.asarray(values,float);assert np.isfinite(x).all()
    rng=np.random.default_rng(CFG['bootstrap_seed']);b=x[rng.integers(len(x),size=(CFG['bootstrap_draws'],len(x)))].mean(1)
    return dict(mean=float(x.mean()),low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)),n=len(x))
def locked():
    r=json.loads((ROOT/'checks/lock.json').read_text())
    assert sha(ROOT/'config.json')==r['config_sha'] and sha(ROOT/'PROTOCOL.md')==r['protocol_sha']
    assert all(sha(ROOT/'src'/n)==h for n,h in r['code'].items())
    assert sha(DATA/'input_banks.npz')==r['input_sha'] and sha(ROOT/'models/anatomy.npz')==r['anatomy_sha']
    return r
