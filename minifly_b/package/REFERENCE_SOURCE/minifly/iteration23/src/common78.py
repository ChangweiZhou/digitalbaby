from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'iteration22/src'))
from common77 import np,pd,json,time,copy,sha,atomic_json,Fly,FrozenReader,READER,observed_activity,state_arrays,state_error,sign_p,holm,DEN,geometry,pair_events,step_encoded
from model77 import Learner as ParentLearner
from assays76 import measured,labels_for
sys.path.insert(0,str(ROOT/'src'))
CFG=json.loads((ROOT/'config.json').read_text())
def bank(seed):
    with np.load(ROOT/'data/inputs.npz') as z:return z[f'F_{seed}'].copy(),z[f'roles_{seed}'].copy()
def confidence(x):
    x=np.asarray(x,float);assert len(x) and np.isfinite(x).all()
    rng=np.random.default_rng(CFG['bootstrap_seed']);a=x[rng.integers(len(x),size=(CFG['bootstrap_draws'],len(x)))].mean(1)
    return dict(mean=float(x.mean()),low=float(np.quantile(a,.025)),high=float(np.quantile(a,.975)),n=len(x))
